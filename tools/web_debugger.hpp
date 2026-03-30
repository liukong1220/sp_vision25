#ifndef TOOLS__WEB_DEBUGGER_HPP
#define TOOLS__WEB_DEBUGGER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace tools
{
class WebDebugger
{
public:
  WebDebugger(const std::string & host = "0.0.0.0", uint16_t port = 8090);
  ~WebDebugger();

  WebDebugger(const WebDebugger &) = delete;
  WebDebugger & operator=(const WebDebugger &) = delete;

  bool good() const;
  std::string url() const;

  void update_state(const nlohmann::json & state);
  void update_main_frame(const cv::Mat & frame, int jpeg_quality = 70);
  void update_ballistic_frame(const cv::Mat & frame, int jpeg_quality = 70);

  bool has_active_client(
    std::chrono::milliseconds ttl = std::chrono::milliseconds(2000)) const;

private:
  void server_loop();
  void handle_client(int client_fd);
  void stream_jpeg(int client_fd, bool ballistic);
  void touch_client() const;

  static std::string index_html();

  std::string host_;
  uint16_t port_ = 0;
  int server_fd_ = -1;
  std::atomic<bool> stop_{false};
  std::thread server_thread_;
  std::mutex client_threads_mutex_;
  std::vector<std::thread> client_threads_;

  mutable std::atomic<int64_t> last_client_touch_ms_{0};
  mutable std::mutex data_mutex_;
  std::string state_json_ = "{}";
  std::vector<uchar> main_jpeg_;
  std::vector<uchar> ballistic_jpeg_;
  uint64_t main_frame_seq_ = 0;
  uint64_t ballistic_frame_seq_ = 0;
};

}  // namespace tools

#endif  // TOOLS__WEB_DEBUGGER_HPP
