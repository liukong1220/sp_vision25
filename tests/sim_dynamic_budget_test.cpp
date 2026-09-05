#include <cmath>
#include <cstdio>

#include "simulation/io/dynamic_budget.hpp"

namespace
{
bool assumed_target_translation_holds(double observed_mps, double assumed_max_mps)
{
  return sim_io::observed_target_translation_within_assumed(observed_mps, assumed_max_mps);
}
}  // namespace

int main()
{
  sim_io::DynamicMotionInput full_input;
  full_input.age_s = 0.2;
  full_input.command_to_consume_delay_s = 0.1;
  full_input.chassis_yaw_rate_radps = 1.0;
  full_input.chassis_translation_speed_mps = 0.5;
  full_input.target_rotation_radps = 2.0;
  full_input.target_translation_speed_mps = 0.7;
  full_input.target_range_m = 2.0;
  full_input.target_radius_m = 0.3;
  const auto full = sim_io::conservative_dynamic_bound(full_input);
  auto no_consume_delay_input = full_input;
  no_consume_delay_input.command_to_consume_delay_s = 0.0;
  const auto no_consume_delay = sim_io::conservative_dynamic_bound(no_consume_delay_input);
  sim_io::DynamicMotionInput yaw_only_input = full_input;
  yaw_only_input.chassis_translation_speed_mps = 0.0;
  yaw_only_input.target_rotation_radps = 0.0;
  yaw_only_input.target_translation_speed_mps = 0.0;
  const auto yaw_only = sim_io::conservative_dynamic_bound(yaw_only_input);
  sim_io::DynamicMotionInput no_target_translation_input = full_input;
  no_target_translation_input.target_translation_speed_mps = 0.0;
  const auto no_target_translation =
    sim_io::conservative_dynamic_bound(no_target_translation_input);
  sim_io::DynamicMotionInput future_input = full_input;
  future_input.age_s = -0.01;
  const auto future = sim_io::conservative_dynamic_bound(future_input);

  // Positional `{age, yaw, ...}` would map yaw onto command_to_consume_delay_s
  // and ignore a real delay. Named fields must make a larger delay increase the bound.
  const bool delay_increases_bound =
    full.finite && full.angle_error_deg > no_consume_delay.angle_error_deg &&
    full.position_error_m > no_consume_delay.position_error_m;

  const double config_delay_s = 0.02;
  const std::vector<double> consume_ms{10.0, 12.0, 40.0, 80.0};
  const double resolved_delay =
    sim_io::resolve_command_to_consume_delay_s(config_delay_s, consume_ms);
  const bool delay_uses_p99 = resolved_delay > config_delay_s && resolved_delay >= 0.07;
  const double config_only =
    sim_io::resolve_command_to_consume_delay_s(config_delay_s, {});
  const bool empty_samples_keep_config = config_only == config_delay_s;

  const bool translation_ok = assumed_target_translation_holds(0.4, 0.5);
  const bool translation_exceeds = !assumed_target_translation_holds(0.7, 0.5);

  const bool ok = delay_increases_bound &&
    full.angle_error_deg > no_target_translation.angle_error_deg &&
    full.position_error_m > no_target_translation.position_error_m &&
    no_target_translation.position_error_m > yaw_only.position_error_m && !future.finite &&
    delay_uses_p99 && empty_samples_keep_config && translation_ok && translation_exceeds;
  std::printf(
    "sim_dynamic_budget_test: %s full=(%.4fdeg,%.4fm) yaw_only=(%.4fdeg,%.4fm) delay=%.4f\n",
    ok ? "PASS" : "FAIL", full.angle_error_deg, full.position_error_m,
    yaw_only.angle_error_deg, yaw_only.position_error_m, resolved_delay);
  return ok ? 0 : 1;
}
