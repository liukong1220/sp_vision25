#include "clock_bridge.hpp"

#include <limits>

namespace sim_io
{
namespace
{
std::int64_t steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

std::int64_t realtime_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}
}  // namespace

ClockBridge::ClockBridge(std::int64_t jump_threshold_ns) : jump_threshold_ns_(jump_threshold_ns) {}

std::int64_t ClockBridge::sample_offset_ns(int attempts)
{
  if (attempts < 1) attempts = 1;

  std::int64_t best_gap = std::numeric_limits<std::int64_t>::max();
  std::int64_t best_offset = 0;

  for (int i = 0; i < attempts; ++i) {
    const std::int64_t s0 = steady_now_ns();
    const std::int64_t r = realtime_now_ns();
    const std::int64_t s1 = steady_now_ns();

    const std::int64_t gap = s1 - s0;
    if (gap < best_gap) {
      best_gap = gap;
      best_offset = s0 + gap / 2 - r;
    }
  }

  return best_offset;
}

bool ClockBridge::resample()
{
  const std::int64_t offset = sample_offset_ns();
  last_sample_ = std::chrono::steady_clock::now();

  if (!initialized_) {
    offset_ns_ = offset;
    initialized_ = true;
    return false;
  }

  const std::int64_t delta = offset - offset_ns_;
  const std::int64_t magnitude = delta < 0 ? -delta : delta;
  offset_ns_ = offset;

  if (magnitude > jump_threshold_ns_) {
    ++jump_count_;
    last_jump_ns_ = delta;
    return true;
  }

  return false;
}

bool ClockBridge::resample_if_due(std::chrono::nanoseconds interval)
{
  if (initialized_ && std::chrono::steady_clock::now() - last_sample_ < interval) return false;
  return resample();
}

std::chrono::steady_clock::time_point ClockBridge::to_steady(
  std::uint64_t source_realtime_ns) const
{
  const std::int64_t steady_ns = static_cast<std::int64_t>(source_realtime_ns) + offset_ns_;
  return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(steady_ns));
}

std::chrono::nanoseconds ClockBridge::age(
  std::uint64_t source_realtime_ns, std::chrono::steady_clock::time_point now) const
{
  return now - to_steady(source_realtime_ns);
}

std::chrono::nanoseconds ClockBridge::age_now(std::uint64_t source_realtime_ns) const
{
  return age(source_realtime_ns, std::chrono::steady_clock::now());
}

}  // namespace sim_io
