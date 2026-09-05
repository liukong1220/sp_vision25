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
    case ReadStatus::ClockJump:
      return "ClockJump";
    case ReadStatus::Reconnected:
      return "Reconnected";
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
  // 看门狗必须比帧龄门限更早触发，否则断流后存在一段"帧已过期但 FAULT_NO_NEW_FRAME
  // 还没置起来"的窗口。这里在 open() 时把它收紧到 max_frame_age_ms 以内，
  // read_timeout_ms 也不允许超过它（阻塞读返回得比看门狗还晚就没有意义）。
  if (
    config_.no_new_frame_timeout_ms <= 0.0 ||
    config_.no_new_frame_timeout_ms > config_.max_frame_age_ms) {
    config_.no_new_frame_timeout_ms = config_.max_frame_age_ms;
  }
  if (config_.read_timeout_ms > config_.no_new_frame_timeout_ms) {
    config_.read_timeout_ms = config_.no_new_frame_timeout_ms;
  }

  if (!client_.open(error)) return false;

  last_remap_check_ = std::chrono::steady_clock::now();

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

void SimCamera::invalidate_after_epoch_change()
{
  // 换代之后旧发布端的一切都不再可信：同帧位姿、帧龄基准、fps 滑动平均。
  // 统计计数器（ok/stale/rejected/...）保留，报告需要看到跨换代的累计值。
  has_bundle_ = false;
  bundle_ = FrameBundle{};
  has_last_ok_ = false;
  has_arrival_ = false;
  fps_ = 0.0;
  jump_pending_ = false;
  ++reconnects_;
}

bool SimCamera::try_recover_publisher(ReadStatus * out)
{
  const auto now = std::chrono::steady_clock::now();
  const double since_check =
    std::chrono::duration<double, std::milli>(now - last_remap_check_).count();
  if (since_check < config_.remap_check_ms) return false;
  last_remap_check_ = now;

  std::string err;

  if (!client_.connected()) {
    // 手里一张映射都没有：只能直接尝试建立。此时 paths_changed() 没有比较基准
    // （它以 connected() 为前提，未连接时恒为 false），必须走 open()。
    if (!client_.open(&err)) {
      *out = ReadStatus::Disconnected;
      return true;
    }
    invalidate_after_epoch_change();
    *out = ReadStatus::Reconnected;
    return true;
  }

  // 仍有旧映射：只有文件身份真的变了才换，否则旧映射依然是唯一真相来源。
  if (!client_.paths_changed()) return false;

  if (client_.remap(&err)) {
    invalidate_after_epoch_change();
    *out = ReadStatus::Reconnected;
    return true;
  }

  // 重映射失败（文件刚被重建、长度还没设好）。remap() 是事务化的，旧映射仍在，
  // 下个检查周期继续重试。
  *out = ReadStatus::Disconnected;
  return true;
}

double SimCamera::frame_gap_ms() const
{
  if (!has_arrival_) return std::numeric_limits<double>::infinity();
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now() - last_ok_arrival_)
    .count();
}

bool SimCamera::no_new_frame() const
{
  return frame_gap_ms() > config_.no_new_frame_timeout_ms;
}

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

  if (!client_.connected()) {
    ReadStatus recovered = ReadStatus::Disconnected;
    if (try_recover_publisher(&recovered)) return recovered;
    return ReadStatus::Disconnected;
  }

  // wall clock 会被 NTP 调整。跳变一旦发生，帧龄基准就不可信：本函数下面用
  // clock_.to_steady() 把源端 wall clock 映射到本地 steady clock，偏移刚被改过
  // 就意味着"跳变前发布、跳变后消费"的那一帧算出来的帧龄含有整个跳变量。
  // 所以跳变必须作为 ReadStatus 上报，由上层置 FAULT_CLOCK_JUMP 并在该帧禁止
  // 控制与开火，而不是只记一个计数器。
  if (clock_.resample_if_due(
        std::chrono::nanoseconds(static_cast<std::int64_t>(config_.clock_resample_ms * 1e6)))) {
    jump_pending_ = true;
  }
  if (jump_pending_) {
    // 跳变后的第一次读取直接丢弃：这一帧的帧龄跨越了跳变点。清掉帧龄基准和
    // fps 滑动平均，避免把跳变量算进统计。
    jump_pending_ = false;
    ++clock_jump_frames_;
    has_last_ok_ = false;
    fps_ = 0.0;
    return ReadStatus::ClockJump;
  }

  FrameBundle bundle;
  const ConsumeStatus status = client_.consume_frame(&bundle);

  switch (status) {
    case ConsumeStatus::NotConnected:
      return ReadStatus::Disconnected;
    case ConsumeStatus::NoFrame: {
      // 没有新帧：可能是发布端**正常退出**后又被拉起——它会 unlink 掉共享内存
      // 文件，重建时是新 inode，本进程手里的旧映射既看不到新帧也看不到新的
      // created_ns（心跳同样停在旧值上）。按周期复查文件身份。
      //
      // 这一步必须排在心跳判断**之前**：重启可以发生在心跳超时很久之后，把
      // `!heartbeat_alive()` 的早退放在前面，等于心跳一超时就再也不检查新
      // inode，慢重启永远恢复不了。
      ReadStatus recovered = ReadStatus::Disconnected;
      if (try_recover_publisher(&recovered)) return recovered;
      if (!heartbeat_alive()) return ReadStatus::Disconnected;
      return ReadStatus::Timeout;
    }
    case ConsumeStatus::Corrupted:
      ++rejected_frames_;
      return ReadStatus::Disconnected;
    case ConsumeStatus::EpochChanged:
      // 同一个 inode 被新发布端复用（SIGKILL 后重启）。丢弃本帧并复位。
      ++rejected_frames_;
      invalidate_after_epoch_change();
      return ReadStatus::Reconnected;
    case ConsumeStatus::Remapped:
      ++rejected_frames_;
      invalidate_after_epoch_change();
      return ReadStatus::Reconnected;
    case ConsumeStatus::ImageInvalid:
    case ConsumeStatus::PoseMissing:
    case ConsumeStatus::PoseSeqMismatch:
    case ConsumeStatus::PoseTimestampMismatch:
    case ConsumeStatus::SeqRegressed:
      ++rejected_frames_;
      return ReadStatus::Rejected;
    case ConsumeStatus::Ok:
      break;
  }

  const auto now = std::chrono::steady_clock::now();
  const auto steady_ts = clock_.to_steady(bundle.timestamp_ns);
  const double age_ms = std::chrono::duration<double, std::milli>(now - steady_ts).count();

  // 源时间戳允许有限的调度/采样抖动；超出容差说明时钟不同步或跳变，不能进入算法。
  if (age_ms < 0.0) {
    ++future_frames_;
    if (-age_ms > config_.max_future_frame_ms) {
      ++rejected_frames_;
      return ReadStatus::Rejected;
    }
  }

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
  last_ok_arrival_ = now;
  has_arrival_ = true;

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
    // 断流看门狗在阻塞读内部就要生效，不能等 read_timeout_ms 才报。
    if (no_new_frame()) return ReadStatus::Timeout;
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
  clock_jump_frames_ = 0;
  has_last_ok_ = false;
  has_arrival_ = false;
  fps_ = 0.0;
}

}  // namespace sim_io
