#ifndef SIMULATION_IO__DYNAMIC_BUDGET_HPP
#define SIMULATION_IO__DYNAMIC_BUDGET_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sim_io
{
struct DynamicMotionInput
{
  double age_s = 0.0;
  // Worst-case command publication -> simulator consumption delay. This is part of the
  // source-to-actuation bound and must not be omitted just because a later frame can measure it.
  double command_to_consume_delay_s = 0.0;
  double chassis_yaw_rate_radps = 0.0;
  double chassis_translation_speed_mps = 0.0;
  double target_rotation_radps = 0.0;
  double target_translation_speed_mps = 0.0;
  double target_range_m = 1.0;
  double target_radius_m = 0.2;
};

struct DynamicMotionBound
{
  double angle_error_deg = std::numeric_limits<double>::infinity();
  double position_error_m = std::numeric_limits<double>::infinity();
  bool finite = false;
};

inline DynamicMotionBound conservative_dynamic_bound(const DynamicMotionInput & input)
{
  if (!std::isfinite(input.age_s) || input.age_s < 0.0 ||
      !std::isfinite(input.chassis_yaw_rate_radps) ||
      !std::isfinite(input.chassis_translation_speed_mps) ||
      !std::isfinite(input.target_rotation_radps) ||
      !std::isfinite(input.target_translation_speed_mps) ||
      !std::isfinite(input.target_range_m) || input.target_range_m <= 1e-6 ||
      !std::isfinite(input.target_radius_m) || input.target_radius_m < 0.0)
    return {};

  if (!std::isfinite(input.command_to_consume_delay_s) || input.command_to_consume_delay_s < 0.0)
    return {};

  const double effective_age_s = input.age_s + input.command_to_consume_delay_s;
  const double translation =
    input.chassis_translation_speed_mps + input.target_translation_speed_mps;
  DynamicMotionBound result;
  result.angle_error_deg =
    effective_age_s * (std::abs(input.chassis_yaw_rate_radps) +
                   std::abs(input.target_rotation_radps) + translation / input.target_range_m) *
    180.0 / M_PI;
  result.position_error_m =
    effective_age_s * (translation + std::abs(input.target_rotation_radps) * input.target_radius_m);
  result.finite = std::isfinite(result.angle_error_deg) && std::isfinite(result.position_error_m);
  return result;
}

inline bool observed_target_translation_within_assumed(
  double observed_mps, double assumed_max_mps)
{
  return std::isfinite(observed_mps) && std::isfinite(assumed_max_mps) &&
    observed_mps <= assumed_max_mps;
}

inline double resolve_command_to_consume_delay_s(
  double config_s, const std::vector<double> & send_to_consume_ms)
{
  const double config = (std::isfinite(config_s) && config_s >= 0.0) ? config_s : 0.0;
  if (send_to_consume_ms.empty()) return config;
  std::vector<double> sorted = send_to_consume_ms;
  std::sort(sorted.begin(), sorted.end());
  const double pos = 0.99 * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  const double p99_ms = (lo == hi)
    ? sorted[lo]
    : sorted[lo] * (1.0 - (pos - static_cast<double>(lo))) +
      sorted[hi] * (pos - static_cast<double>(lo));
  return std::max(config, p99_ms / 1000.0);
}
}  // namespace sim_io

#endif  // SIMULATION_IO__DYNAMIC_BUDGET_HPP
