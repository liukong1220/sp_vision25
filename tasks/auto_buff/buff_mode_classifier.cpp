#include "buff_mode_classifier.hpp"

#include <algorithm>
#include <cmath>

namespace auto_buff
{
void RuneModeClassifier::reset()
{
  mode_ = PowerRuneMode::Unknown;
  has_angle_ = false;
  last_angle_ = 0.0;
  unwrapped_angle_ = 0.0;
  last_timestamp_ = {};
  speeds_.clear();
}

PowerRuneMode RuneModeClassifier::observe(
  PowerRune & rune, std::chrono::steady_clock::time_point timestamp)
{
  if (mode_ != PowerRuneMode::Unknown) {
    rune.set_mode(mode_);
    return mode_;
  }
  if (rune.is_unsolve() || !std::isfinite(rune.ypr_in_world[2])) return mode_;

  const double angle = rune.ypr_in_world[2];
  if (!has_angle_) {
    has_angle_ = true;
    last_angle_ = angle;
    unwrapped_angle_ = angle;
    last_timestamp_ = timestamp;
    return mode_;
  }

  const double dt = std::chrono::duration<double>(timestamp - last_timestamp_).count();
  if (!(dt > 1e-4) || dt > config_.max_sample_gap_s) {
    reset();
    has_angle_ = true;
    last_angle_ = angle;
    unwrapped_angle_ = angle;
    last_timestamp_ = timestamp;
    return mode_;
  }

  // The selected blade can jump by 2*pi/5. Unwrap in that periodic space before differentiating.
  constexpr double blade_period = 2.0 * CV_PI / 5.0;
  double delta = angle - last_angle_;
  while (delta > blade_period / 2.0) delta -= blade_period;
  while (delta < -blade_period / 2.0) delta += blade_period;
  unwrapped_angle_ += delta;
  last_angle_ = angle;
  last_timestamp_ = timestamp;

  const double speed = std::abs(delta / dt);
  if (!std::isfinite(speed) || speed > 4.0) return mode_;
  speeds_.push_back({timestamp, speed});
  while (!speeds_.empty() &&
         std::chrono::duration<double>(timestamp - speeds_.front().timestamp).count() >
           config_.small_confirmation_s)
    speeds_.pop_front();

  if (speeds_.size() < config_.min_speed_samples) return mode_;
  std::size_t big_evidence = 0;
  double max_deviation = 0.0;
  for (const auto & sample : speeds_) {
    const double deviation = std::abs(sample.abs_speed_radps - config_.small_speed_radps);
    max_deviation = std::max(max_deviation, deviation);
    if (deviation >= config_.big_deviation_radps) ++big_evidence;
  }
  if (big_evidence >= config_.big_deviation_samples) {
    mode_ = PowerRuneMode::Big;
  } else {
    const double span =
      std::chrono::duration<double>(speeds_.back().timestamp - speeds_.front().timestamp).count();
    if (span >= config_.small_confirmation_s * 0.9 &&
        max_deviation <= config_.small_max_deviation_radps)
      mode_ = PowerRuneMode::Small;
  }
  rune.set_mode(mode_);
  return mode_;
}
}  // namespace auto_buff
