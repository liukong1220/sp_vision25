#ifndef AUTO_BUFF__MODE_CLASSIFIER_HPP
#define AUTO_BUFF__MODE_CLASSIFIER_HPP

#include <chrono>
#include <cstddef>
#include <deque>
#include <optional>

#include "buff_type.hpp"

namespace auto_buff
{
struct RuneModeClassifierConfig
{
  std::size_t min_speed_samples = 8;
  double small_confirmation_s = 2.5;
  double small_speed_radps = CV_PI / 3.0;
  double small_max_deviation_radps = 0.12;
  double big_deviation_radps = 0.18;
  std::size_t big_deviation_samples = 2;
  double max_sample_gap_s = 0.75;
};

// `all` mode is selected only from solved rune motion. Ground truth is intentionally absent
// from this API. Big may be identified early from varying speed; Small is confirmed only after
// a full observation window, so an initially-small-looking Big rune remains Unknown.
class RuneModeClassifier
{
public:
  explicit RuneModeClassifier(RuneModeClassifierConfig config = {}) : config_(config) {}

  PowerRuneMode observe(PowerRune & rune, std::chrono::steady_clock::time_point timestamp);
  void reset();
  PowerRuneMode mode() const { return mode_; }
  std::size_t speed_samples() const { return speeds_.size(); }

private:
  struct SpeedSample
  {
    std::chrono::steady_clock::time_point timestamp;
    double abs_speed_radps = 0.0;
  };

  RuneModeClassifierConfig config_;
  PowerRuneMode mode_ = PowerRuneMode::Unknown;
  bool has_angle_ = false;
  double last_angle_ = 0.0;
  double unwrapped_angle_ = 0.0;
  std::chrono::steady_clock::time_point last_timestamp_{};
  std::deque<SpeedSample> speeds_;
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__MODE_CLASSIFIER_HPP
