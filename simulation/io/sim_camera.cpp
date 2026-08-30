#include "sim_camera.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace sim_io
{
const char * to_string(ReadStatus status)
{
  switch (status) {
    case ReadStatus::Ok:
      return "Ok";
    case ReadStatus::Timeout:
      return "Timeout";
    case ReadStatus::Stale:
      return "Stale";
    case ReadStatus::Rejected:
      return "Rejected";
    case ReadStatus::Disconnected:
      return "Disconnected";
  }
  return "Unknown";
}

namespace
{
double percentile(std::vector<double> & sorted, double q)
{
  if (sorted.empty()) return 0.0;
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double w = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

constexpr std::size_t MAX_AGE_SAMPLES = 200000;
}  // namespace

SimCamera::SimCamera(SimCameraConfig config) : config_(config), client_(config_.shm) {}

bool SimCamera::open(std::string * error)
{
  if (!client_.open(error)) return false;

  // 先建立时钟映射，再读第一帧，否则第一帧的帧龄没有意义。
  clock_.resample();

  // 时钟纪元自检：simulator 用 SystemTime/UNIX_EPOCH，本地用 system_clock。
  // Linux/libstdc++ 上两者同纪元。若不同纪元，心跳算出来的年龄会是几十年，
  // 这里立刻抓住而不是让帧龄统计变成天文数字。
  const std::uint64_t hb = client_.heartbeat_ns();
  if (hb != 0) {
    const double age_ms =
      std::chrono::duration<double, std::milli>(clock_.age_now(hb)).count();
    if (std::abs(age_ms) > 60000.0) {
      if (error) {
        *error = "心跳时间戳与本地时钟相差 " + std::to_string(age_ms / 1000.0) +
                 " s, 时钟纪元不一致, 拒绝启动";
      }
      client_.close();
      return false;
    }
  }
  return true;
}

void SimCamera::close() { client_.close(); }

bool SimCamera::heartbeat_alive() const
{
  const std::uint64_t hb = client_.heartbeat_ns();
  if (hb == 0) return false;
  return heartbeat_age_ms() <= config_.heartbeat_timeout_ms;
}

double SimCamera::heartbeat_age_ms() const
{
  const std::uint64_t hb = client_.heartbeat_ns();
  if (hb == 0) return std::numeric_limits<double>::infinity();
  return std::chrono::duration<double, std::milli>(clock_.age_now(hb)).count();
}

ReadStatus SimCamera::try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  img.release();

  if (!client_.connected()) return ReadStatus::Disconnected;

  // wall clock 会被 NTP 调整。周期性重采样并在跳变时清掉帧龄基准。
  if (clock_.resample_if_due(
        std::chrono::nanoseconds(static_cast<std::int64_t>(config_.clock_resample_ms * 1e6)))) {
    // 跳变本身不丢帧，但跨越跳变的帧龄不可信，交由上层按 clock_jumps() 降级。
  }

  FrameBundle bundle;
  const ConsumeStatus status = client_.consume_frame(&bundle);

  switch (status) {
    case ConsumeStatus::NotConnected:
      return ReadStatus::Disconnected;
    case ConsumeStatus::NoFrame:
      return heartbeat_alive() ? ReadStatus::Timeout : ReadStatus::Disconnected;
    case ConsumeStatus::Corrupted:
      ++rejected_frames_;
      return ReadStatus::Disconnected;
    case ConsumeStatus::ImageInvalid:
    case ConsumeStatus::PoseMissing:
    case ConsumeStatus::PoseSeqMismatch:
    case ConsumeStatus::SeqRegressed:
      ++rejected_frames_;
      return ReadStatus::Rejected;
    case ConsumeStatus::Ok:
      break;
  }

  const auto now = std::chrono::steady_clock::now();
  const auto steady_ts = clock_.to_steady(bundle.timestamp_ns);
  const double age_ms = std::chrono::duration<double, std::milli>(now - steady_ts).count();

  // 源时间戳落在未来：时钟不同步或跳变，帧龄不可信。计数但不当成过期帧丢弃。
  if (age_ms < 0.0) ++future_frames_;

  if (age_ms > config_.max_frame_age_ms) {
    ++stale_frames_;
    return ReadStatus::Stale;
  }

  // 拷进自有 Mat。src 只是共享内存上的非拥有头，cvtColor/copyTo 之后
  // img 与共享内存再无关系。这一步必须在本函数内完成：
  // 一旦返回，下一次 consume_frame 就可能让 bundle.pixels 失效。
  const cv::Mat src(
    static_cast<int>(bundle.height), static_cast<int>(bundle.width), CV_8UC3,
    const_cast<std::uint8_t *>(bundle.pixels));

  if (config_.convert_rgb_to_bgr) {
    cv::cvtColor(src, img, cv::COLOR_RGB2BGR);
  } else {
    src.copyTo(img);
  }

  if (img.data == bundle.pixels) {
    // 理论上不可能：cvtColor/copyTo 都会分配新缓冲。留作零拷贝改造时的护栏。
    img = img.clone();
  }

  bundle_ = bundle;
  bundle_.pixels = nullptr;  // 不让调用方拿到已经过期的指针
  bundle_.pixel_bytes = 0;
  has_bundle_ = true;

  timestamp = steady_ts;

  if (frame_age_ms_.size() < MAX_AGE_SAMPLES) frame_age_ms_.push_back(age_ms);
  ++ok_frames_;

  if (has_last_ok_) {
    const double dt_s = std::chrono::duration<double>(steady_ts - last_ok_steady_).count();
    if (dt_s > 1e-6) {
      const double inst = 1.0 / dt_s;
      fps_ = fps_ > 0.0 ? fps_ * 0.9 + inst * 0.1 : inst;
    }
  }
  last_ok_steady_ = steady_ts;
  has_last_ok_ = true;

  return ReadStatus::Ok;
}

ReadStatus SimCamera::read_blocking(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::nanoseconds(
                          static_cast<std::int64_t>(config_.read_timeout_ms * 1e6));

  for (;;) {
    const ReadStatus status = try_read(img, timestamp);
    if (status != ReadStatus::Timeout) return status;
    if (std::chrono::steady_clock::now() >= deadline) return ReadStatus::Timeout;
    std::this_thread::sleep_for(
      std::chrono::nanoseconds(static_cast<std::int64_t>(config_.poll_interval_us * 1e3)));
  }
}

void SimCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  const ReadStatus status = read_blocking(img, timestamp);
  if (status != ReadStatus::Ok) img.release();
}

double SimCamera::camera_fps() const { return fps_; }

FrameAgeStats SimCamera::frame_age_stats() const
{
  FrameAgeStats stats;
  if (frame_age_ms_.empty()) return stats;

  std::vector<double> sorted = frame_age_ms_;
  std::sort(sorted.begin(), sorted.end());

  double sum = 0.0;
  for (double v : sorted) sum += v;

  stats.count = sorted.size();
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.mean_ms = sum / static_cast<double>(sorted.size());
  stats.p50_ms = percentile(sorted, 0.50);
  stats.p95_ms = percentile(sorted, 0.95);
  stats.p99_ms = percentile(sorted, 0.99);
  return stats;
}

void SimCamera::reset_stats()
{
  frame_age_ms_.clear();
  ok_frames_ = 0;
  stale_frames_ = 0;
  rejected_frames_ = 0;
  future_frames_ = 0;
  has_last_ok_ = false;
  fps_ = 0.0;
}

}  // namespace sim_io
