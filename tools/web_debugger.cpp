#include "tools/web_debugger.hpp"

#include "tools/logger.hpp"
#include "tools/path.hpp"
#include "tools/runtime_params.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include <opencv2/imgcodecs.hpp>

namespace
{
constexpr std::chrono::milliseconds kPollInterval(200);
constexpr std::chrono::milliseconds kStreamPollInterval(25);
constexpr size_t kMaxRequestBytes = 8192;
constexpr int kMinWebMode = 1;
constexpr int kMaxWebMode = 3;

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

bool is_scalar_history_value(const nlohmann::json & value)
{
  return value.is_number() || value.is_boolean() || value.is_null();
}

void trim_json_array(nlohmann::json & array, size_t max_points)
{
  if (!array.is_array()) return;
  while (array.size() > max_points) {
    array.erase(array.begin());
  }
}

nlohmann::json default_overlay_config()
{
  return nlohmann::json::object({
    {"stabilize", true},
    {"state_layers", true},
    {"armors", true},
    {"labels", true},
    {"target_motion", true},
    {"aim", true},
    {"decision_hud", true},
    {"decision_track", true},
    {"footer", true},
  });
}

nlohmann::json merge_overlay_config(
  const nlohmann::json & incoming, const nlohmann::json & fallback)
{
  nlohmann::json merged = fallback.is_object() ? fallback : default_overlay_config();
  const auto defaults = default_overlay_config();
  for (auto it = defaults.begin(); it != defaults.end(); ++it) {
    const auto & key = it.key();
    if (incoming.contains(key) && incoming.at(key).is_boolean()) {
      merged[key] = incoming.at(key).get<bool>();
    } else if (!merged.contains(key) || !merged.at(key).is_boolean()) {
      merged[key] = it.value();
    }
  }
  return merged;
}

std::string to_lower_copy(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char ch) {return static_cast<char>(std::tolower(ch));});
  return value;
}

int clamp_web_mode(int mode)
{
  return std::clamp(mode, kMinWebMode, kMaxWebMode);
}

std::string web_mode_key(int mode)
{
  switch (clamp_web_mode(mode)) {
    case 1:
      return "auto_aim";
    case 2:
      return "small_buff";
    case 3:
      return "big_buff";
    default:
      return "auto_aim";
  }
}

std::string web_mode_label(int mode)
{
  switch (clamp_web_mode(mode)) {
    case 1:
      return "自瞄";
    case 2:
      return "小符";
    case 3:
      return "大符";
    default:
      return "自瞄";
  }
}

nlohmann::json web_mode_payload(int mode)
{
  const int clamped_mode = clamp_web_mode(mode);
  return nlohmann::json::object({
    {"mode", clamped_mode},
    {"mode_key", web_mode_key(clamped_mode)},
    {"mode_label", web_mode_label(clamped_mode)},
    {"source", "web"},
    {"choices", nlohmann::json::array({
      nlohmann::json::object({
        {"mode", 1},
        {"key", "auto_aim"},
        {"label", "自瞄"},
      }),
      nlohmann::json::object({
        {"mode", 2},
        {"key", "small_buff"},
        {"label", "小符"},
      }),
      nlohmann::json::object({
        {"mode", 3},
        {"key", "big_buff"},
        {"label", "大符"},
      }),
    })},
  });
}

size_t parse_content_length(const std::string & header_block)
{
  std::istringstream header_stream(header_block);
  std::string line;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) continue;
    const std::string key = to_lower_copy(line.substr(0, colon_pos));
    if (key != "content-length") continue;
    try {
      return static_cast<size_t>(std::stoul(line.substr(colon_pos + 1)));
    } catch (...) {
      return 0;
    }
  }
  return 0;
}
}  // namespace

namespace tools
{
WebDebugger::WebDebugger(const std::string & host, uint16_t port)
: host_(sanitize_bind_host(host)), port_(port), overlay_config_(default_overlay_config())
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

bool WebDebugger::good() const
{
  return server_fd_ >= 0;
}

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

void WebDebugger::update_log(const nlohmann::json & log)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  log_json_ = log.dump();
}

void WebDebugger::update_plot_sample(const nlohmann::json & sample)
{
  if (!sample.is_object()) return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  auto append_series = [&](const std::string & key, const nlohmann::json & value) {
    if (!is_scalar_history_value(value)) return;
    auto & series = plot_history_[key];
    if (!series.is_array()) series = nlohmann::json::array();
    if (value.is_boolean()) {
      series.push_back(value.get<bool>() ? 1 : 0);
    } else {
      series.push_back(value);
    }
    trim_json_array(series, max_plot_points_);
  };

  if (sample.contains("time")) {
    append_series("time", sample.at("time"));
  } else if (sample.contains("t")) {
    append_series("time", sample.at("t"));
  }

  for (auto it = sample.begin(); it != sample.end(); ++it) {
    if (it.key() == "time" || it.key() == "t") continue;
    append_series(it.key(), it.value());
  }

  plot_json_ = plot_history_.dump();
}

void WebDebugger::set_plot_history_limit(size_t max_points)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  max_plot_points_ = std::max<size_t>(10, max_points);
  for (auto it = plot_history_.begin(); it != plot_history_.end(); ++it) {
    trim_json_array(it.value(), max_plot_points_);
  }
  plot_json_ = plot_history_.dump();
}

void WebDebugger::update_overlay_config(const nlohmann::json & config)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  overlay_config_ = merge_overlay_config(config, overlay_config_);
}

nlohmann::json WebDebugger::overlay_config() const
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  return overlay_config_;
}

void WebDebugger::set_runtime_config_path(const std::string & config_path)
{
  runtime_config_path_ = tools::resolve_config_path_string(config_path);
  tools::runtime_params::register_config(runtime_config_path_);
}

void WebDebugger::set_selected_mode(int mode)
{
  selected_mode_.store(clamp_web_mode(mode));
}

int WebDebugger::selected_mode() const
{
  return clamp_web_mode(selected_mode_.load());
}

void WebDebugger::update_main_frame(const cv::Mat & frame, int jpeg_quality)
{
  if (frame.empty()) return;

  std::vector<int> params = {
    cv::IMWRITE_JPEG_QUALITY,
    std::clamp(jpeg_quality, 30, 95),
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
    std::clamp(jpeg_quality, 30, 95),
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

std::string WebDebugger::load_static_asset(const std::string & relative_path)
{
  const auto asset_path = tools::resolve_runtime_path(
    std::filesystem::path("assets/web_debugger") / relative_path);

  std::ifstream file(asset_path, std::ios::binary);
  if (!file.is_open()) return {};

  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::string WebDebugger::content_type_for_asset(const std::string & relative_path)
{
  const auto extension = std::filesystem::path(relative_path).extension().string();
  if (extension == ".html") return "text/html; charset=utf-8";
  if (extension == ".css") return "text/css; charset=utf-8";
  if (extension == ".js") return "application/javascript; charset=utf-8";
  if (extension == ".json") return "application/json; charset=utf-8";
  if (extension == ".png") return "image/png";
  if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
  return "text/plain; charset=utf-8";
}

void WebDebugger::server_loop()
{
  while (!stop_) {
    pollfd poll_fd {};
    poll_fd.fd = server_fd_;
    poll_fd.events = POLLIN;
    const int poll_ret =
      ::poll(&poll_fd, 1, static_cast<int>(kPollInterval.count()));
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
  while (
    request.find("\r\n\r\n") == std::string::npos &&
    request.size() < kMaxRequestBytes)
  {
    const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
    if (received <= 0) break;
    request.append(buffer, static_cast<size_t>(received));
  }

  const auto header_end = request.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    send_response(
      client_fd, "400 Bad Request", "text/plain; charset=utf-8",
      "invalid request");
    return;
  }

  const std::string header_block = request.substr(0, header_end);
  const size_t content_length = parse_content_length(header_block);
  const size_t body_start = header_end + 4;
  while (
    request.size() < body_start + content_length &&
    request.size() < kMaxRequestBytes)
  {
    const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer), 0);
    if (received <= 0) break;
    request.append(buffer, static_cast<size_t>(received));
  }

  const std::string body = request.substr(
    body_start, std::min(content_length, request.size() > body_start ? request.size() - body_start : 0UL));

  std::istringstream stream(header_block);
  std::string method;
  std::string path;
  std::string version;
  stream >> method >> path >> version;

  if (method != "GET" && method != "POST") {
    send_response(
      client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
      "GET/POST only");
    return;
  }

  const auto query_pos = path.find('?');
  if (query_pos != std::string::npos) path = path.substr(0, query_pos);

  if (path == "/" || path == "/index.html") {
    auto html = load_static_asset("index.html");
    if (html.empty()) {
      html = "<!DOCTYPE html><html><body><h1>debug ui missing</h1></body></html>";
    }
    send_response(client_fd, "200 OK", "text/html; charset=utf-8", html);
    return;
  }

  if (path.rfind("/static/", 0) == 0) {
    const std::string relative_path = path.substr(1);
    if (relative_path.find("..") != std::string::npos) {
      send_response(
        client_fd, "400 Bad Request", "text/plain; charset=utf-8",
        "invalid path");
      return;
    }

    const auto asset = load_static_asset(relative_path);
    if (asset.empty()) {
      send_response(
        client_fd, "404 Not Found", "text/plain; charset=utf-8", "not found");
      return;
    }

    send_response(
      client_fd, "200 OK", content_type_for_asset(relative_path), asset);
    return;
  }

  if (path == "/api/state") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
    std::string payload;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      payload = state_json_;
    }
    send_response(
      client_fd, "200 OK", "application/json; charset=utf-8", payload);
    return;
  }

  if (path == "/api/mode") {
    if (method == "GET") {
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        web_mode_payload(selected_mode()).dump());
      return;
    }

    try {
      const auto incoming = body.empty() ? nlohmann::json::object() : nlohmann::json::parse(body);
      if (!incoming.contains("mode") || !incoming.at("mode").is_number_integer()) {
        throw std::runtime_error("mode must be an integer");
      }
      set_selected_mode(incoming.at("mode").get<int>());
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        web_mode_payload(selected_mode()).dump());
    } catch (const std::exception & e) {
      send_response(
        client_fd, "400 Bad Request", "application/json; charset=utf-8",
        nlohmann::json({{"error", e.what()}}).dump());
    }
    return;
  }

  if (path == "/data") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
    std::string payload;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      payload = plot_json_;
    }
    send_response(
      client_fd, "200 OK", "application/json; charset=utf-8", payload);
    return;
  }

  if (path == "/log") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
    std::string payload;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      payload = log_json_;
    }
    send_response(
      client_fd, "200 OK", "application/json; charset=utf-8", payload);
    return;
  }

  if (path == "/api/overlay") {
    if (method == "GET") {
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        overlay_config().dump());
      return;
    }

    try {
      const auto incoming = body.empty() ? nlohmann::json::object() : nlohmann::json::parse(body);
      update_overlay_config(incoming);
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        overlay_config().dump());
    } catch (const std::exception &) {
      send_response(
        client_fd, "400 Bad Request", "application/json; charset=utf-8",
        R"({"error":"invalid overlay config"})");
    }
    return;
  }

  if (path == "/api/params") {
    if (runtime_config_path_.empty()) {
      send_response(
        client_fd, "503 Service Unavailable", "application/json; charset=utf-8",
        nlohmann::json({
          {"enabled", false},
          {"error", "runtime parameter session not configured"},
        }).dump());
      return;
    }

    if (method == "GET") {
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        tools::runtime_params::describe(runtime_config_path_).dump());
      return;
    }

    try {
      const auto incoming = body.empty() ? nlohmann::json::object() : nlohmann::json::parse(body);
      const auto payload = tools::runtime_params::apply(runtime_config_path_, incoming);
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        payload.dump());
    } catch (const std::exception & e) {
      send_response(
        client_fd, "400 Bad Request", "application/json; charset=utf-8",
        nlohmann::json({{"error", e.what()}}).dump());
    }
    return;
  }

  if (path == "/api/params/reset") {
    if (runtime_config_path_.empty()) {
      send_response(
        client_fd, "503 Service Unavailable", "application/json; charset=utf-8",
        nlohmann::json({
          {"enabled", false},
          {"error", "runtime parameter session not configured"},
        }).dump());
      return;
    }

    if (method != "POST") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "POST only");
      return;
    }

    try {
      const auto incoming = body.empty() ? nlohmann::json::object() : nlohmann::json::parse(body);
      std::vector<std::string> keys;
      if (incoming.contains("keys")) {
        if (!incoming.at("keys").is_array()) {
          throw std::runtime_error("keys must be an array");
        }
        for (const auto & key : incoming.at("keys")) {
          if (!key.is_string()) {
            throw std::runtime_error("keys must contain strings");
          }
          keys.push_back(key.get<std::string>());
        }
      }

      const auto payload = tools::runtime_params::reset(runtime_config_path_, keys);
      send_response(
        client_fd, "200 OK", "application/json; charset=utf-8",
        payload.dump());
    } catch (const std::exception & e) {
      send_response(
        client_fd, "400 Bad Request", "application/json; charset=utf-8",
        nlohmann::json({{"error", e.what()}}).dump());
    }
    return;
  }

  if (path == "/stream/main.mjpg") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
    stream_jpeg(client_fd, false);
    return;
  }

  if (path == "/stream/ballistic.mjpg") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
    stream_jpeg(client_fd, true);
    return;
  }

  if (path == "/api/frames/main.jpg" || path == "/api/frames/ballistic.jpg") {
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
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
    if (method != "GET") {
      send_response(
        client_fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
        "GET only");
      return;
    }
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

}  // namespace tools
