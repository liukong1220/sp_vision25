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
};

const char * to_string(ReadStatus status);

struct SimCameraConfig
{
  double max_frame_age_ms = 100.0;      // 超过即视为过期帧
  double heartbeat_timeout_ms = 500.0;  // 心跳停滞超过此值视为仿真端离线
  double read_timeout_ms = 1000.0;      // 阻塞 read 的等待上限
  double poll_interval_us = 200.0;      // 轮询间隔
  double clock_resample_ms = 1000.0;    // realtime<->steady 偏移重采样周期
  bool convert_rgb_to_bgr = true;       // 仿真端发布 RGB8，OpenCV 默认 BGR

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

  const CameraInfo * camera_info() const { return client_.camera_info(); }
  SharedMemoryClient & client() { return client_; }
  const SharedMemoryClient & client() const { return client_; }
  ClockBridge & clock() { return clock_; }
  const ClockBridge & clock() const { return clock_; }

  bool heartbeat_alive() const;
  double heartbeat_age_ms() const;

  FrameAgeStats frame_age_stats() const;
  std::uint64_t stale_frames() const { return stale_frames_; }
  std::uint64_t rejected_frames() const { return rejected_frames_; }
  std::uint64_t ok_frames() const { return ok_frames_; }
  std::uint64_t future_frames() const { return future_frames_; }
  std::uint64_t clock_jumps() const { return clock_.jump_count(); }
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

  std::chrono::steady_clock::time_point last_ok_steady_{};
  bool has_last_ok_ = false;
  double fps_ = 0.0;
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_CAMERA_HPP
