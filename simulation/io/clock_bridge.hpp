#ifndef SIMULATION_IO__CLOCK_BRIDGE_HPP
#define SIMULATION_IO__CLOCK_BRIDGE_HPP

// simulator 的 timestamp_ns 来自 Rust `SystemTime::now().duration_since(UNIX_EPOCH)`，
// 即 wall clock（Linux 上等价于 CLOCK_REALTIME / std::chrono::system_clock）。
// sp_vision25 内部的时长、帧龄和 Tracker 时间轴全部使用 steady_clock
// （CLOCK_MONOTONIC）。两者不能互相解释。
//
// 这里不做“第一帧对齐”式的隐式假设，而是直接在同一台机器上把两个时钟一起读出来，
// 得到 offset = steady - realtime。由于两次读取之间只隔几十纳秒，这个 offset 是
// 真实偏移而不是估计值，所以由它算出的帧龄是绝对帧龄，而不是相对第一帧的差值。
//
// wall clock 会被 NTP/手工调整跳变。offset 变化超过阈值时记为一次时钟跳变，
// 交由上层在跳变期间禁止开火并丢弃跨越跳变的帧。原始 timestamp_ns 始终保留。

#include <chrono>
#include <cstdint>

namespace sim_io
{
class ClockBridge
{
public:
  // jump_threshold_ns：offset 变化超过该值即判定为 wall clock 跳变。
  // 默认 2 ms，远大于同机两次时钟读取的抖动，又足以捕捉有意义的时间调整。
  explicit ClockBridge(std::int64_t jump_threshold_ns = 2'000'000);

  // 立即采样一次 realtime<->steady 偏移。首次调用建立基准。
  // 返回 true 表示相对上一次采样检测到跳变。
  bool resample();

  // 距离上次采样超过 interval 时才重新采样。返回是否检测到跳变。
  bool resample_if_due(std::chrono::nanoseconds interval);

  // 把源端 wall clock 纳秒映射到本地 steady_clock。
  std::chrono::steady_clock::time_point to_steady(std::uint64_t source_realtime_ns) const;

  // 源时间戳相对 now 的帧龄。源时间戳在未来时为负值，调用方需要显式处理。
  std::chrono::nanoseconds age(
    std::uint64_t source_realtime_ns, std::chrono::steady_clock::time_point now) const;

  std::chrono::nanoseconds age_now(std::uint64_t source_realtime_ns) const;

  std::int64_t offset_ns() const { return offset_ns_; }
  int jump_count() const { return jump_count_; }
  std::int64_t last_jump_ns() const { return last_jump_ns_; }
  bool initialized() const { return initialized_; }

  // 单次紧密采样：steady, realtime, steady 三次读取，取 steady 中点，
  // 并在多次尝试中挑选读取间隔最小的一组，把采样自身误差压到最低。
  static std::int64_t sample_offset_ns(int attempts = 8);

private:
  std::int64_t jump_threshold_ns_;
  std::int64_t offset_ns_ = 0;
  std::int64_t last_jump_ns_ = 0;
  int jump_count_ = 0;
  bool initialized_ = false;
  std::chrono::steady_clock::time_point last_sample_{};
};

}  // namespace sim_io

#endif  // SIMULATION_IO__CLOCK_BRIDGE_HPP
