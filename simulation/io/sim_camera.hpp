#ifndef SIMULATION_IO__SIM_CAMERA_HPP
#define SIMULATION_IO__SIM_CAMERA_HPP

// 把仿真共享内存的图像通道包装成 io::CameraBase，使 YOLO/Solver/Tracker 这条链路
// 完全不需要知道图像来自渲染器还是真实相机。
//
// 三条硬性约束：
//   1. 时间戳保留源端 wall clock，同时映射到本地 steady_clock 供 Tracker 使用。
//   2. 图像必须拷进自有 cv::Mat；共享内存槽位指针的有效期只到下一次消费。
//   3. 过期帧（帧龄超阈值）、倒退帧、pose 不同帧的帧一律丢弃，不进算法。

#include <chrono>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "clock_bridge.hpp"
#include "io/camera.hpp"
#include "shared_memory_client.hpp"

namespace sim_io
{
enum class ReadStatus
{
  Ok,
  Timeout,       // 超时内没有新帧
  Stale,         // 帧龄超过阈值，已丢弃
  Rejected,      // 帧号倒退 / pose 不同帧 / 图像不合约，已丢弃
  Disconnected,  // 心跳超时或共享内存不可用
  ClockJump,     // 本帧跨越了 realtime<->steady 偏移跳变，帧龄不可信，已丢弃
  Reconnected,   // 发布端换代（换 inode 或 created_ns 变化），已重连，上层必须复位
};

const char * to_string(ReadStatus status);

struct SimCameraConfig
{
  double max_frame_age_ms = 100.0;      // 超过即视为过期帧
  double max_future_frame_ms = 5.0;      // 允许的未来时间戳抖动；超出即拒绝
  double heartbeat_timeout_ms = 500.0;  // 心跳停滞超过此值视为仿真端离线
  double read_timeout_ms = 1000.0;      // 阻塞 read 的等待上限
  double poll_interval_us = 200.0;      // 轮询间隔
  double clock_resample_ms = 1000.0;    // realtime<->steady 偏移重采样周期
  bool convert_rgb_to_bgr = true;       // 仿真端发布 RGB8，OpenCV 默认 BGR

  // "多久没有新帧就算断流"。必须**不大于** max_frame_age_ms，否则看门狗比帧龄
  // 门限还慢：原来 read_timeout_ms=1000ms 而 max_frame_age_ms/state_timeout_ms
  // 都是 450ms，断流后有 550ms 的窗口里 FAULT_NO_NEW_FRAME 还没置起来。
  // open() 会按 max_frame_age_ms 收紧这两个值并在 effective_* 里报告实际生效值。
  double no_new_frame_timeout_ms = 250.0;

  // 共享内存文件身份（inode）复查周期。只在没有新帧时才复查，正常出图时不做 stat。
  double remap_check_ms = 200.0;

  // 共享内存位置。默认与 simulator 一致；测试可指向临时目录以避免与真实仿真进程互相干扰。
  SharedMemoryClient::Options shm{};
};

struct FrameAgeStats
{
  std::size_t count = 0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
};

class SimCamera : public io::CameraBase
{
public:
  explicit SimCamera(SimCameraConfig config = {});

  bool open(std::string * error);
  void close();
  bool connected() const { return client_.connected(); }

  // io::CameraBase 接口。超时或被拒时 img 置空，调用方需检查 img.empty()。
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  double camera_fps() const override;

  // 明确区分“没有帧”和“帧被丢弃”，仿真入口用这个。
  ReadStatus try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  ReadStatus read_blocking(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  // 与最近一次 Ok 帧严格同帧的 pose 束。sim_gimbal 只能用这个，不能用“最新姿态”。
  const FrameBundle & last_bundle() const { return bundle_; }
  bool has_bundle() const { return has_bundle_; }
  std::uint64_t last_frame_seq() const { return bundle_.frame_seq; }
  std::uint64_t last_timestamp_ns() const { return bundle_.timestamp_ns; }

  // 最近一次 Ok 帧的源端采样时刻与本地接收时刻。两者都必须交给 SimGimbal，
  // 由它分别用于"世界观测年龄"与"本地状态保有时长"这两个语义不同的判据。
  FrameStamps last_stamps() const { return FrameStamps{last_ok_steady_, last_ok_arrival_}; }
  bool has_stamps() const { return has_last_ok_ && has_arrival_; }

  const CameraInfo * camera_info() const { return client_.camera_info(); }
  SharedMemoryClient & client() { return client_; }
  const SharedMemoryClient & client() const { return client_; }
  ClockBridge & clock() { return clock_; }
  const ClockBridge & clock() const { return clock_; }
  // 测试用：注入时钟跳变（见 ClockBridge::debug_shift_offset_ns）。
  ClockBridge & clock_for_test() { return clock_; }

  bool heartbeat_alive() const;
  double heartbeat_age_ms() const;

  FrameAgeStats frame_age_stats() const;
  std::uint64_t stale_frames() const { return stale_frames_; }
  std::uint64_t rejected_frames() const { return rejected_frames_; }
  std::uint64_t ok_frames() const { return ok_frames_; }
  std::uint64_t future_frames() const { return future_frames_; }
  std::uint64_t clock_jumps() const { return clock_.jump_count(); }
  // 因跨越时钟跳变而被丢弃的帧数。与 clock_jumps() 不同：一次跳变最多丢一帧。
  std::uint64_t clock_jump_frames() const { return clock_jump_frames_; }
  // 观测到的发布端换代次数（inode 变化 + created_ns 变化，两路合计）。
  std::uint64_t reconnects() const { return reconnects_; }

  // 距最近一次 Ok 帧的毫秒数；从未取到过帧时返回 +inf。看门狗用这个，不要等
  // read_blocking 超时。
  double frame_gap_ms() const;
  // 断流判据：frame_gap_ms() 超过 no_new_frame_timeout_ms 的生效值。
  bool no_new_frame() const;
  double effective_no_new_frame_timeout_ms() const { return config_.no_new_frame_timeout_ms; }
  double effective_read_timeout_ms() const { return config_.read_timeout_ms; }

  void reset_stats();

private:
  SimCameraConfig config_;
  SharedMemoryClient client_;
  ClockBridge clock_;

  FrameBundle bundle_{};
  bool has_bundle_ = false;

  std::vector<double> frame_age_ms_;
  std::uint64_t ok_frames_ = 0;
  std::uint64_t stale_frames_ = 0;
  std::uint64_t rejected_frames_ = 0;
  std::uint64_t future_frames_ = 0;
  std::uint64_t clock_jump_frames_ = 0;
  std::uint64_t reconnects_ = 0;

  std::chrono::steady_clock::time_point last_ok_steady_{};
  bool has_last_ok_ = false;
  double fps_ = 0.0;

  // 最近一次 Ok 帧到手的**本地 steady 时刻**。last_ok_steady_ 是帧的时间戳映射值，
  // 会被时钟跳变污染，不能用来做看门狗。
  std::chrono::steady_clock::time_point last_ok_arrival_{};
  bool has_arrival_ = false;

  // resample 报了跳变、但还没有据此丢掉一帧。
  bool jump_pending_ = false;
  std::chrono::steady_clock::time_point last_remap_check_{};

  // 换代/重连后清掉与旧发布端相关的本地状态（同帧位姿、帧龄基准、fps）。
  void invalidate_after_epoch_change();

  // 尝试恢复与发布端的连接，按 remap_check_ms 节流。
  //
  // 返回 true 表示这一拍已有结论，*out 有效（Reconnected 或 Disconnected）；
  // 返回 false 表示没到检查周期、或文件身份没变，调用方继续原有判断。
  //
  // 必须能从**心跳已超时**和**完全未连接**两种状态进入：发布端正常退出会 unlink
  // 掉文件，隔很久才重新拉起（人工重启、脚本串行跑两段），这段时间心跳早就超时了。
  // 原来 NoFrame 分支里 `if (!heartbeat_alive()) return Disconnected;` 挡在文件身份
  // 复查之前，而 try_read 的第一行又在未连接时直接返回，于是"慢重启"永远恢复不了。
  bool try_recover_publisher(ReadStatus * out);
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_CAMERA_HPP
