#include "web_debugger.hpp"

#include "logger.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string_view>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace
{
constexpr std::chrono::milliseconds kPollInterval(200);
constexpr std::chrono::milliseconds kStreamPollInterval(25);
constexpr size_t kMaxRequestBytes = 8192;

int64_t steady_now_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

bool send_all(int fd, const void * data, size_t size)
{
  const auto * cursor = static_cast<const char *>(data);
  size_t remaining = size;
  while (remaining > 0) {
    const ssize_t sent = ::send(fd, cursor, remaining, 0);
    if (sent <= 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += sent;
    remaining -= static_cast<size_t>(sent);
  }
  return true;
}

bool send_response(
  int fd, std::string_view status, std::string_view content_type,
  const std::vector<unsigned char> & body)
{
  const std::string headers =
    "HTTP/1.1 " + std::string(status) + "\r\n"
    "Content-Type: " + std::string(content_type) + "\r\n"
    "Content-Length: " + std::to_string(body.size()) + "\r\n"
    "Cache-Control: no-store\r\n"
    "Connection: close\r\n\r\n";
  return send_all(fd, headers.data(), headers.size()) &&
         send_all(fd, body.data(), body.size());
}

bool send_response(
  int fd, std::string_view status, std::string_view content_type,
  std::string_view body)
{
  const std::string headers =
    "HTTP/1.1 " + std::string(status) + "\r\n"
    "Content-Type: " + std::string(content_type) + "\r\n"
    "Content-Length: " + std::to_string(body.size()) + "\r\n"
    "Cache-Control: no-store\r\n"
    "Connection: close\r\n\r\n";
  return send_all(fd, headers.data(), headers.size()) &&
         send_all(fd, body.data(), body.size());
}

bool send_empty_response(int fd, std::string_view status)
{
  const std::string headers =
    "HTTP/1.1 " + std::string(status) + "\r\n"
    "Content-Length: 0\r\n"
    "Cache-Control: no-store\r\n"
    "Connection: close\r\n\r\n";
  return send_all(fd, headers.data(), headers.size());
}

bool send_stream_headers(int fd, std::string_view boundary)
{
  const std::string headers =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=" + std::string(boundary) + "\r\n"
    "Cache-Control: no-store\r\n"
    "Connection: close\r\n\r\n";
  return send_all(fd, headers.data(), headers.size());
}

bool send_mjpeg_frame(int fd, std::string_view boundary, const std::vector<uchar> & jpeg)
{
  const std::string part_headers =
    "--" + std::string(boundary) + "\r\n"
    "Content-Type: image/jpeg\r\n"
    "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
  static const std::string part_tail = "\r\n";
  return send_all(fd, part_headers.data(), part_headers.size()) &&
         send_all(fd, jpeg.data(), jpeg.size()) &&
         send_all(fd, part_tail.data(), part_tail.size());
}

std::string sanitize_bind_host(const std::string & host)
{
  if (host.empty() || host == "*") return "0.0.0.0";
  if (host == "localhost") return "127.0.0.1";
  return host;
}
}  // namespace

namespace tools
{
WebDebugger::WebDebugger(const std::string & host, uint16_t port)
: host_(sanitize_bind_host(host)), port_(port)
{
  server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    tools::logger()->warn("WebDebugger socket() failed: {}", std::strerror(errno));
    return;
  }

  const int enable = 1;
  ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

  sockaddr_in address {};
  address.sin_family = AF_INET;
  address.sin_port = ::htons(port_);
  if (::inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1) {
    tools::logger()->warn("WebDebugger invalid bind host: {}", host_);
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  if (::bind(server_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
    tools::logger()->warn(
      "WebDebugger bind({}:{}) failed: {}", host_, port_, std::strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  if (::listen(server_fd_, 8) < 0) {
    tools::logger()->warn("WebDebugger listen() failed: {}", std::strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  server_thread_ = std::thread(&WebDebugger::server_loop, this);
}

WebDebugger::~WebDebugger()
{
  stop_ = true;
  if (server_fd_ >= 0) {
    ::shutdown(server_fd_, SHUT_RDWR);
    ::close(server_fd_);
    server_fd_ = -1;
  }
  if (server_thread_.joinable()) server_thread_.join();
  std::lock_guard<std::mutex> lock(client_threads_mutex_);
  for (auto & thread : client_threads_) {
    if (thread.joinable()) thread.join();
  }
}

bool WebDebugger::good() const { return server_fd_ >= 0; }

std::string WebDebugger::url() const
{
  const std::string display_host = host_ == "0.0.0.0" ? "127.0.0.1" : host_;
  return "http://" + display_host + ":" + std::to_string(port_) + "/";
}

void WebDebugger::update_state(const nlohmann::json & state)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  state_json_ = state.dump();
}

void WebDebugger::update_main_frame(const cv::Mat & frame, int jpeg_quality)
{
  if (frame.empty()) return;

  std::vector<int> params = {
    cv::IMWRITE_JPEG_QUALITY,
    std::max(30, std::min(95, jpeg_quality)),
  };
  std::vector<uchar> encoded;
  if (!cv::imencode(".jpg", frame, encoded, params)) return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  main_jpeg_.swap(encoded);
  ++main_frame_seq_;
}

void WebDebugger::update_ballistic_frame(const cv::Mat & frame, int jpeg_quality)
{
  if (frame.empty()) return;

  std::vector<int> params = {
    cv::IMWRITE_JPEG_QUALITY,
    std::max(30, std::min(95, jpeg_quality)),
  };
  std::vector<uchar> encoded;
  if (!cv::imencode(".jpg", frame, encoded, params)) return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  ballistic_jpeg_.swap(encoded);
  ++ballistic_frame_seq_;
}

bool WebDebugger::has_active_client(std::chrono::milliseconds ttl) const
{
  return steady_now_ms() - last_client_touch_ms_.load() <= ttl.count();
}

void WebDebugger::touch_client() const
{
  last_client_touch_ms_.store(steady_now_ms());
}

void WebDebugger::server_loop()
{
  while (!stop_) {
    pollfd poll_fd {};
    poll_fd.fd = server_fd_;
    poll_fd.events = POLLIN;
    const int poll_ret = ::poll(&poll_fd, 1, static_cast<int>(kPollInterval.count()));
    if (poll_ret <= 0) continue;

    sockaddr_in client_addr {};
    socklen_t client_len = sizeof(client_addr);
    const int client_fd =
      ::accept(server_fd_, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      if (!stop_) {
        tools::logger()->warn("WebDebugger accept() failed: {}", std::strerror(errno));
      }
      continue;
    }

    std::lock_guard<std::mutex> lock(client_threads_mutex_);
    client_threads_.emplace_back([this, client_fd]() {
      handle_client(client_fd);
      ::close(client_fd);
    });
  }
}

void WebDebugger::handle_client(int client_fd)
{
  touch_client();

  timeval timeout {};
  timeout.tv_sec = 1;
  timeout.tv_usec = 0;
  ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

  std::string request;
  request.reserve(1024);
  char buffer[1024];
  while (request.find("\r\n\r\n") == std::string::npos && request.size() < kMaxRequestBytes) {
    const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
    if (received <= 0) break;
    request.append(buffer, static_cast<size_t>(received));
  }

  std::istringstream stream(request);
  std::string method;
  std::string path;
  std::string version;
  stream >> method >> path >> version;

  if (method != "GET") {
    send_response(client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8", "GET only");
    return;
  }

  const auto query_pos = path.find('?');
  if (query_pos != std::string::npos) path = path.substr(0, query_pos);

  if (path == "/" || path == "/index.html") {
    const std::string html = index_html();
    send_response(client_fd, "200 OK", "text/html; charset=utf-8", html);
    return;
  }

  if (path == "/api/state") {
    std::string payload;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      payload = state_json_;
    }
    send_response(client_fd, "200 OK", "application/json; charset=utf-8", payload);
    return;
  }

  if (path == "/stream/main.mjpg") {
    stream_jpeg(client_fd, false);
    return;
  }

  if (path == "/stream/ballistic.mjpg") {
    stream_jpeg(client_fd, true);
    return;
  }

  if (path == "/api/frames/main.jpg" || path == "/api/frames/ballistic.jpg") {
    std::vector<uchar> payload;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      payload = path == "/api/frames/main.jpg" ? main_jpeg_ : ballistic_jpeg_;
    }

    if (payload.empty()) {
      send_empty_response(client_fd, "204 No Content");
      return;
    }

    send_response(client_fd, "200 OK", "image/jpeg", payload);
    return;
  }

  if (path == "/healthz") {
    send_response(client_fd, "200 OK", "text/plain; charset=utf-8", "ok");
    return;
  }

  send_response(client_fd, "404 Not Found", "text/plain; charset=utf-8", "not found");
}

void WebDebugger::stream_jpeg(int client_fd, bool ballistic)
{
  constexpr std::string_view kBoundary = "frame";
  if (!send_stream_headers(client_fd, kBoundary)) return;

  uint64_t last_seq = 0;
  while (!stop_) {
    std::vector<uchar> jpeg;
    uint64_t seq = 0;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (ballistic) {
        jpeg = ballistic_jpeg_;
        seq = ballistic_frame_seq_;
      } else {
        jpeg = main_jpeg_;
        seq = main_frame_seq_;
      }
    }

    if (!jpeg.empty() && seq != 0 && seq != last_seq) {
      touch_client();
      if (!send_mjpeg_frame(client_fd, kBoundary, jpeg)) break;
      last_seq = seq;
      continue;
    }

    touch_client();
    std::this_thread::sleep_for(kStreamPollInterval);
  }
}

std::string WebDebugger::index_html()
{
  return R"HTML(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Artisans战队自瞄网页调试器</title>
  <style>
    :root {
      --bg0: #07131a;
      --bg1: #0d2029;
      --panel: rgba(7, 18, 24, 0.84);
      --panel-border: rgba(112, 164, 182, 0.22);
      --text: #eef7f9;
      --muted: #8eabb3;
      --accent: #35d0ba;
      --accent-2: #f9a03f;
      --danger: #ff5a5f;
      --ok: #8de969;
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 20% 15%, rgba(53, 208, 186, 0.18), transparent 30%),
        radial-gradient(circle at 85% 8%, rgba(249, 160, 63, 0.2), transparent 22%),
        linear-gradient(160deg, var(--bg0), var(--bg1) 56%, #08161c);
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
      background-size: 28px 28px;
      pointer-events: none;
      opacity: 0.6;
    }

    .shell {
      width: min(1760px, calc(100vw - 16px));
      margin: 8px auto 16px;
      display: grid;
      gap: 12px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      overflow: hidden;
    }

    .hero {
      padding: 16px 20px;
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(0, 1.7fr) minmax(260px, 0.8fr);
      align-items: end;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(28px, 3.8vw, 44px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      font-weight: 700;
    }

    .hero p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
      max-width: 820px;
    }

    .status-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .status-chip, .card, .chart-card {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 18px;
    }

    .status-chip {
      padding: 12px 14px;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .value {
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
      font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    }

    .value.fire {
      color: var(--danger);
      text-shadow: 0 0 18px rgba(255, 90, 95, 0.35);
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0% { opacity: 0.72; }
      50% { opacity: 1; }
      100% { opacity: 0.72; }
    }

    .switcher {
      display: flex;
      gap: 8px;
      padding: 0 2px;
      flex-wrap: wrap;
    }

    .switch-btn {
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: var(--muted);
      border-radius: 999px;
      padding: 10px 16px;
      font-size: 13px;
      font-family: "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif;
      cursor: pointer;
      transition: 0.2s ease;
    }

    .switch-btn.active {
      color: var(--text);
      border-color: rgba(53, 208, 186, 0.35);
      background: linear-gradient(135deg, rgba(53, 208, 186, 0.18), rgba(249, 160, 63, 0.10));
      box-shadow: inset 0 0 0 1px rgba(53, 208, 186, 0.14);
    }

    .view {
      display: none;
    }

    .view.active {
      display: grid;
    }

    .visual-grid {
      gap: 12px;
      grid-template-columns: minmax(0, 1.58fr) minmax(320px, 0.82fr);
      align-items: start;
    }

    .analysis-grid {
      gap: 12px;
      grid-template-columns: minmax(0, 1.06fr) minmax(0, 0.94fr);
      align-items: start;
    }

    .charts-panel {
      padding: 12px;
      position: sticky;
      top: 8px;
    }

    .media-panel {
      display: grid;
      gap: 9px;
      padding: 12px;
    }

    .panel-head, .chart-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .panel-head h2, .chart-head h3 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0.02em;
    }

    .meta {
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    }

    .frame-box {
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background:
        linear-gradient(135deg, rgba(53, 208, 186, 0.12), transparent 42%),
        rgba(1, 9, 12, 0.78);
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .frame-box.square {
      aspect-ratio: 16 / 9.6;
    }

    .frame-box.analysis-main {
      aspect-ratio: 16 / 10;
    }

    .frame-box img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      image-rendering: auto;
    }

    .placeholder {
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .cards {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      padding: 0 12px 12px;
    }

    .card {
      padding: 12px;
      min-height: 126px;
    }

    .card h3 {
      margin: 0 0 12px;
      font-size: 14px;
      font-weight: 600;
    }

    .kv {
      display: grid;
      gap: 8px;
    }

    .row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      color: var(--muted);
    }

    .row strong {
      color: var(--text);
      font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
      font-weight: 600;
      text-align: right;
    }

    .charts-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 10px;
    }

    .chart-card {
      padding: 12px;
      min-height: 0;
    }

    .chart-card .chart-head {
      align-items: flex-start;
      flex-direction: column;
      gap: 8px;
    }

    .legend {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }

    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: -1px;
      background: currentColor;
    }

    canvas {
      width: 100%;
      height: 138px;
      display: block;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent),
        rgba(0, 0, 0, 0.24);
    }

    @media (max-width: 1480px) {
      .analysis-grid {
        grid-template-columns: minmax(0, 1fr);
      }

      .charts-panel {
        position: static;
      }
    }

    @media (max-width: 1280px) {
      .visual-grid, .cards {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 960px) {
      .charts-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 720px) {
      .shell {
        width: min(100vw - 10px, 1840px);
        margin: 6px auto 12px;
      }

      .hero {
        padding: 16px;
        grid-template-columns: 1fr;
      }

      .status-grid, .cards {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel hero">
      <div>
        <h1>Artisans战队<br />自瞄网页调试器</h1>
      </div>
      <div class="status-grid">
        <div class="status-chip">
          <div class="label">Link</div>
          <div class="value" id="status-link">Waiting</div>
        </div>
        <div class="status-chip">
          <div class="label">Fire Gate</div>
          <div class="value" id="status-fire">SAFE</div>
        </div>
        <div class="status-chip">
          <div class="label">Target</div>
          <div class="value" id="status-target">none</div>
        </div>
        <div class="status-chip">
          <div class="label">Latency</div>
          <div class="value" id="status-latency">-- ms</div>
        </div>
      </div>
    </section>

    <section class="panel media-panel">
      <div class="panel-head">
        <h2>调试视图</h2>
        <div class="switcher">
          <button class="switch-btn active" id="tab-visual" type="button">作战画面</button>
          <button class="switch-btn" id="tab-analysis" type="button">曲线联调</button>
        </div>
      </div>
    </section>

    <section class="view visual-grid active" id="view-visual">
      <section class="panel media-panel">
        <div class="panel-head">
          <h2>主图像</h2>
          <div class="meta" id="main-meta-visual">awaiting stream</div>
        </div>
        <div class="frame-box">
          <img id="main-frame-visual" alt="Main overlay visual" />
          <div class="placeholder" id="main-placeholder-visual">Waiting for MJPEG Stream</div>
        </div>
      </section>

      <section class="panel media-panel">
        <div class="panel-head">
          <h2>弹道面板</h2>
          <div class="meta" id="ballistic-meta">awaiting stream</div>
        </div>
        <div class="frame-box square">
          <img id="ballistic-frame" alt="Ballistic panel" />
          <div class="placeholder" id="ballistic-placeholder">Waiting for MJPEG Stream</div>
        </div>
      </section>
    </section>

    <section class="view analysis-grid" id="view-analysis">
      <section class="panel media-panel">
        <div class="panel-head">
          <h2>主图像</h2>
          <div class="meta" id="main-meta-analysis">awaiting stream</div>
        </div>
        <div class="frame-box analysis-main">
          <img id="main-frame-analysis" alt="Main overlay analysis" />
          <div class="placeholder" id="main-placeholder-analysis">Waiting for MJPEG Stream</div>
        </div>
      </section>

      <section class="panel charts-panel">
        <div class="panel-head">
          <h2>曲线调节</h2>
          <div class="meta" id="curve-meta">angles in rad</div>
        </div>
        <div class="charts-grid">
          <article class="chart-card">
            <div class="chart-head">
              <h3>Yaw</h3>
              <div class="legend">
                <span style="color:#35d0ba">gimbal</span>
                <span style="color:#f9a03f">preview</span>
                <span style="color:#ff5a5f">sent</span>
              </div>
            </div>
            <canvas id="chart-yaw"></canvas>
          </article>
          <article class="chart-card">
            <div class="chart-head">
              <h3>Pitch</h3>
              <div class="legend">
                <span style="color:#35d0ba">gimbal</span>
                <span style="color:#f9a03f">preview</span>
                <span style="color:#ff5a5f">sent</span>
              </div>
            </div>
            <canvas id="chart-pitch"></canvas>
          </article>
          <article class="chart-card">
            <div class="chart-head">
              <h3>弹道误差</h3>
              <div class="legend">
                <span style="color:#35d0ba">lateral</span>
                <span style="color:#f9a03f">vertical</span>
                <span style="color:#ff5a5f">total</span>
              </div>
            </div>
            <canvas id="chart-error"></canvas>
          </article>
          <article class="chart-card">
            <div class="chart-head">
              <h3>规划延迟 / 自旋</h3>
              <div class="legend">
                <span style="color:#35d0ba">delay ms</span>
                <span style="color:#f9a03f">w rad/s</span>
              </div>
            </div>
            <canvas id="chart-planner"></canvas>
          </article>
        </div>
      </section>
    </section>

    <section class="panel cards">
      <article class="card">
        <h3>Preview</h3>
        <div class="kv" id="card-preview"></div>
      </article>
      <article class="card">
        <h3>Command</h3>
        <div class="kv" id="card-command"></div>
      </article>
      <article class="card">
        <h3>Planner</h3>
        <div class="kv" id="card-planner"></div>
      </article>
      <article class="card">
        <h3>Ballistic</h3>
        <div class="kv" id="card-ballistic"></div>
      </article>
    </section>
  </div>

  <script>
    const statePollMs = 100;
    const historySeconds = 18;
    const histories = {
      yaw: [],
      pitch: [],
      lateral: [],
      vertical: [],
      total: [],
      delay: [],
      spin: [],
    };
    histories.yawPreview = [];
    histories.yawSent = [];
    histories.pitchPreview = [];
    histories.pitchSent = [];

    let latestState = null;
    let stateLastOkAt = 0;
    let activeView = "visual";

    const text = (value, digits = 2, suffix = "") => {
      if (value === null || value === undefined || Number.isNaN(value)) return "--";
      return `${Number(value).toFixed(digits)}${suffix}`;
    };

    const boolText = (value, yes = "ON", no = "OFF") => value ? yes : no;
    const cardRow = (name, value) => `<div class="row"><span>${name}</span><strong>${value}</strong></div>`;

    const get = (obj, path, fallback = null) => {
      return path.split(".").reduce((acc, key) => {
        if (acc === null || acc === undefined) return undefined;
        return acc[key];
      }, obj) ?? fallback;
    };

    const trimHistory = (series, nowSec) => {
      while (series.length && nowSec - series[0].t > historySeconds) {
        series.shift();
      }
    };

    const pushPoint = (name, value, nowSec) => {
      if (value === null || value === undefined || Number.isNaN(value)) return;
      histories[name].push({ t: nowSec, v: Number(value) });
      trimHistory(histories[name], nowSec);
    };

    const updateStatus = (state) => {
      const ageMs = Date.now() - get(state, "server.unix_ms", Date.now());
      const linkEl = document.getElementById("status-link");
      linkEl.textContent = ageMs < 600 ? "Live" : "Stale";
      linkEl.style.color = ageMs < 600 ? "var(--ok)" : "var(--danger)";

      const fire = !!get(state, "preview.fire", false);
      const fireEl = document.getElementById("status-fire");
      fireEl.textContent = fire ? "FIRE" : "SAFE";
      fireEl.className = fire ? "value fire" : "value";

      const targetText = get(state, "preview.has_target", false)
        ? `${get(state, "preview.target_name", "target")} / ${get(state, "preview.armor_type", "--")}`
        : "none";
      document.getElementById("status-target").textContent = targetText;
      document.getElementById("status-latency").textContent = `${text(get(state, "frame.latency_ms"), 1, " ms")}`;
    };

    const updateCards = (state) => {
      document.getElementById("card-preview").innerHTML = [
        cardRow("target yaw", `${text(get(state, "preview.target_yaw_deg"), 2, " deg")}`),
        cardRow("target pitch", `${text(get(state, "preview.target_pitch_deg"), 2, " deg")}`),
        cardRow("plan yaw", `${text(get(state, "preview.plan_yaw_deg"), 2, " deg")}`),
        cardRow("plan pitch", `${text(get(state, "preview.plan_pitch_deg"), 2, " deg")}`),
        cardRow("target xyz", `${text(get(state, "preview.target_x_m"), 2)}, ${text(get(state, "preview.target_y_m"), 2)}, ${text(get(state, "preview.target_z_m"), 2)}`),
      ].join("");

      document.getElementById("card-command").innerHTML = [
        cardRow("unit source", `${get(state, "command.gimbal_source_unit", "--")}`),
        cardRow("gimbal yaw", `${text(get(state, "command.gimbal_yaw_deg"), 2, " deg")}`),
        cardRow("gimbal pitch", `${text(get(state, "command.gimbal_pitch_deg"), 2, " deg")}`),
        cardRow("sent yaw", `${text(get(state, "command.plan_yaw_deg"), 2, " deg")}`),
        cardRow("sent pitch", `${text(get(state, "command.plan_pitch_deg"), 2, " deg")}`),
        cardRow("fired", boolText(get(state, "command.fired", false), "yes", "no")),
      ].join("");

      document.getElementById("card-planner").innerHTML = [
        cardRow("armor id", `${get(state, "planner.selected_armor", "--")}`),
        cardRow("spin gate", boolText(get(state, "planner.spin_gate", false), "on", "off")),
        cardRow("delay", `${text(get(state, "planner.delay_ms"), 1, " ms")}`),
        cardRow("w", `${text(get(state, "planner.w_rad_s"), 2, " rad/s")}`),
        cardRow("h", `${text(get(state, "planner.h_m"), 3, " m")}`),
      ].join("");

      document.getElementById("card-ballistic").innerHTML = [
        cardRow("verdict", `${get(state, "ballistic.hit", false) ? "HIT" : "MISS"}`),
        cardRow("target d", `${text(get(state, "ballistic.target_dist_3d_m"), 2, " m")}`),
        cardRow("yaw residual", `${text(get(state, "ballistic.yaw_residual_deg"), 3, " deg")}`),
        cardRow("pitch residual", `${text(get(state, "ballistic.pitch_residual_deg"), 3, " deg")}`),
        cardRow("total miss", `${text(get(state, "ballistic.total_error_mm"), 1, " mm")}`),
      ].join("");
    };

    const resizeCanvas = (canvas) => {
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return ratio;
    };

    const drawSeriesChart = (canvasId, lines) => {
      const canvas = document.getElementById(canvasId);
      const ratio = resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgba(255,255,255,0.02)";
      ctx.fillRect(0, 0, w, h);

      const pad = 24 * ratio;
      const innerW = w - pad * 2;
      const innerH = h - pad * 2;
      if (innerW <= 0 || innerH <= 0) return;

      ctx.strokeStyle = "rgba(141,233,105,0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i += 1) {
        const y = pad + innerH * (i / 4);
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(pad + innerW, y);
        ctx.stroke();
      }

      const allPoints = lines.flatMap(line => line.series);
      if (!allPoints.length) {
        ctx.fillStyle = "rgba(142,171,179,0.8)";
        ctx.font = `${12 * ratio}px IBM Plex Mono`;
        ctx.fillText("waiting for data", pad, h / 2);
        return;
      }

      const nowSec = allPoints[allPoints.length - 1].t;
      const minT = Math.max(0, nowSec - historySeconds);
      let minV = Math.min(...allPoints.map(p => p.v));
      let maxV = Math.max(...allPoints.map(p => p.v));
      if (Math.abs(maxV - minV) < 1e-4) {
        minV -= 1;
        maxV += 1;
      }
      const padV = (maxV - minV) * 0.12;
      minV -= padV;
      maxV += padV;

      const mapX = (t) => pad + ((t - minT) / historySeconds) * innerW;
      const mapY = (v) => pad + innerH - ((v - minV) / (maxV - minV)) * innerH;

      lines.forEach((line) => {
        if (line.series.length < 1) return;
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 2.2 * ratio;
        ctx.beginPath();
        line.series.forEach((point, idx) => {
          const x = mapX(point.t);
          const y = mapY(point.v);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      ctx.fillStyle = "rgba(238,247,249,0.78)";
      ctx.font = `${11 * ratio}px IBM Plex Mono`;
      ctx.fillText(maxV.toFixed(2), pad, pad - 6 * ratio);
      ctx.fillText(minV.toFixed(2), pad, h - 8 * ratio);
    };

    const renderCharts = () => {
      drawSeriesChart("chart-yaw", [
        { color: "#35d0ba", series: histories.yaw },
        { color: "#f9a03f", series: histories.yawPreview },
        { color: "#ff5a5f", series: histories.yawSent },
      ]);
      drawSeriesChart("chart-pitch", [
        { color: "#35d0ba", series: histories.pitch },
        { color: "#f9a03f", series: histories.pitchPreview },
        { color: "#ff5a5f", series: histories.pitchSent },
      ]);
      drawSeriesChart("chart-error", [
        { color: "#35d0ba", series: histories.lateral },
        { color: "#f9a03f", series: histories.vertical },
        { color: "#ff5a5f", series: histories.total },
      ]);
      drawSeriesChart("chart-planner", [
        { color: "#35d0ba", series: histories.delay },
        { color: "#f9a03f", series: histories.spin },
      ]);
    };

    const updateHistory = (state) => {
      const nowSec = get(state, "server.unix_ms", Date.now()) / 1000.0;
      pushPoint("yaw", get(state, "command.gimbal_yaw_rad"), nowSec);
      pushPoint("yawPreview", get(state, "preview.plan_yaw_rad"), nowSec);
      pushPoint("yawSent", get(state, "command.plan_yaw_rad"), nowSec);

      pushPoint("pitch", get(state, "command.gimbal_pitch_rad"), nowSec);
      pushPoint("pitchPreview", get(state, "preview.plan_pitch_rad"), nowSec);
      pushPoint("pitchSent", get(state, "command.plan_pitch_rad"), nowSec);

      pushPoint("lateral", get(state, "ballistic.lateral_error_mm"), nowSec);
      pushPoint("vertical", get(state, "ballistic.vertical_error_mm"), nowSec);
      pushPoint("total", get(state, "ballistic.total_error_mm"), nowSec);
      pushPoint("delay", get(state, "planner.delay_ms"), nowSec);
      pushPoint("spin", get(state, "planner.w_rad_s"), nowSec);
    };

    const applyState = (state) => {
      latestState = state;
      stateLastOkAt = Date.now();
      updateStatus(state);
      updateCards(state);
      updateHistory(state);
      renderCharts();
    };

    const refreshMeta = () => {
      const age = stateLastOkAt ? Date.now() - stateLastOkAt : null;
      const stale = age === null || age > 1200;
      const mainMetaText = stale ? "state stale" : `state age ${age} ms`;
      document.getElementById("main-meta-visual").textContent = mainMetaText;
      document.getElementById("main-meta-analysis").textContent = mainMetaText;
      document.getElementById("ballistic-meta").textContent = stale
        ? "waiting ballistic panel"
        : `ballistic ${text(get(latestState, "ballistic.total_error_mm"), 1, " mm")}`;
      document.getElementById("curve-meta").textContent = stale
        ? "state reconnecting"
        : `angles in rad / gimbal auto:${get(latestState, "command.gimbal_source_unit", "--")} / history ${historySeconds}s`;
    };

    const stateLoop = async () => {
      while (true) {
        try {
          const res = await fetch(`/api/state?ts=${Date.now()}`, { cache: "no-store" });
          if (res.ok) {
            const state = await res.json();
            applyState(state);
          }
        } catch (err) {
        }
        refreshMeta();
        await new Promise(resolve => setTimeout(resolve, statePollMs));
      }
    };

    const createStreamController = (imgId, placeholderId, path) => {
      const img = document.getElementById(imgId);
      const placeholder = document.getElementById(placeholderId);
      let enabled = false;

      const start = () => {
        if (enabled) return;
        enabled = true;
        img.src = `${path}?ts=${Date.now()}`;
      };

      const stop = () => {
        enabled = false;
        img.removeAttribute("src");
        placeholder.style.display = "block";
      };

      img.onload = () => {
        placeholder.style.display = "none";
      };

      img.onerror = () => {
        placeholder.style.display = "block";
        if (!enabled) return;
        setTimeout(() => {
          if (!enabled) return;
          img.src = `${path}?ts=${Date.now()}`;
        }, 500);
      };

      return { start, stop };
    };

    const mainVisualStream = createStreamController("main-frame-visual", "main-placeholder-visual", "/stream/main.mjpg");
    const mainAnalysisStream = createStreamController("main-frame-analysis", "main-placeholder-analysis", "/stream/main.mjpg");
    const ballisticStream = createStreamController("ballistic-frame", "ballistic-placeholder", "/stream/ballistic.mjpg");

    const setView = (view) => {
      activeView = view;
      const visual = view === "visual";
      document.getElementById("view-visual").classList.toggle("active", visual);
      document.getElementById("view-analysis").classList.toggle("active", !visual);
      document.getElementById("tab-visual").classList.toggle("active", visual);
      document.getElementById("tab-analysis").classList.toggle("active", !visual);

      if (visual) {
        mainAnalysisStream.stop();
        mainVisualStream.start();
        ballisticStream.start();
      } else {
        mainVisualStream.stop();
        ballisticStream.stop();
        mainAnalysisStream.start();
      }

      renderCharts();
    };

    document.getElementById("tab-visual").addEventListener("click", () => setView("visual"));
    document.getElementById("tab-analysis").addEventListener("click", () => setView("analysis"));

    window.addEventListener("resize", renderCharts);
    stateLoop();
    setView(activeView);
  </script>
</body>
</html>
  )HTML";
}

}  // namespace tools
