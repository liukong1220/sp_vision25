#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_mode_classifier.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/path.hpp"

namespace
{
int failures = 0;

void check(bool condition, const char * what)
{
  std::printf("%-72s %s\n", what, condition ? "ok" : "FAIL");
  if (!condition) ++failures;
}

class FakeTarget final : public auto_buff::Target
{
public:
  explicit FakeTarget(Eigen::Vector3d point) : point_(std::move(point))
  {
    unsolvable_ = false;
    ekf_.x = Eigen::VectorXd::Zero(7);
  }

  void set_point(Eigen::Vector3d point) { point_ = std::move(point); }
  void set_blade_angle(double angle_rad) { ekf_.x[5] = angle_rad; }

  void get_target(
    const std::optional<auto_buff::PowerRune> &,
    std::chrono::steady_clock::time_point &) override
  {
  }

  void predict(double) override {}

  std::unique_ptr<auto_buff::Target> clone() const override
  {
    return std::make_unique<FakeTarget>(*this);
  }

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d &) const override { return point_; }

private:
  Eigen::Vector3d point_;

  void init(double, const auto_buff::PowerRune &) override {}
  void update(double, const auto_buff::PowerRune &) override {}
};

auto_buff::PowerRune solved_rune(double angle)
{
  auto_buff::PowerRune rune;
  rune.ypr_in_world = {0.0, 0.0, angle};
  return rune;
}

bool yaw_opposite(double commanded, double truth)
{
  const double delta = std::remainder(commanded - truth, 2.0 * CV_PI);
  return std::abs(delta) > CV_PI / 2.0;
}

// Mirrors sim_auto_buff activate_mode: mode change MUST replace the estimator instance.
struct ModeRouter
{
  std::unique_ptr<auto_buff::Target> target;
  auto_buff::RuneModeClassifier classifier;
  auto_buff::PowerRuneMode active = auto_buff::PowerRuneMode::Unknown;
  std::uint64_t small_instances = 0;
  std::uint64_t big_instances = 0;

  void activate(auto_buff::PowerRuneMode selected)
  {
    const bool replacing = static_cast<bool>(target);
    target.reset();
    classifier.reset();
    active = selected;
    if (selected == auto_buff::PowerRuneMode::Small) {
      target = std::make_unique<auto_buff::SmallTarget>();
      ++small_instances;
    } else if (selected == auto_buff::PowerRuneMode::Big) {
      target = std::make_unique<auto_buff::BigTarget>();
      ++big_instances;
    }
    if (replacing) {
      check(target != nullptr, "mode change must construct a fresh estimator");
    }
  }
};
}  // namespace

int main()
{
  const std::string config = tools::resolve_config_path_string("configs/simulation.yaml");
  auto timestamp = std::chrono::steady_clock::now();
  const auto t0 = std::chrono::steady_clock::time_point{};

  {
    auto_buff::RuneModeClassifier classifier;
    double angle = 0.0;
    auto_buff::PowerRuneMode mode = auto_buff::PowerRuneMode::Unknown;
    for (int i = 0; i < 160; ++i) {
      angle += (CV_PI / 3.0) * 0.02;
      auto rune = solved_rune(angle);
      mode = classifier.observe(rune, t0 + std::chrono::milliseconds(20 * i));
    }
    check(mode == auto_buff::PowerRuneMode::Small, "constant-speed window classifies as small");

    double big_angle = angle;
    constexpr double amplitude = 0.90;
    constexpr double omega = 1.94;
    for (int i = 0; i < 8; ++i) {
      const double t = i * 0.02;
      const double speed = amplitude * std::sin(omega * t) + 2.09 - amplitude;
      big_angle += speed * 0.02;
      auto rune = solved_rune(big_angle);
      mode = classifier.observe(rune, t0 + std::chrono::milliseconds(20 * (160 + i)));
    }
    check(
      mode == auto_buff::PowerRuneMode::Small,
      "locked classifier keeps small without reset (router must reset on switch)");

    classifier.reset();
    check(classifier.mode() == auto_buff::PowerRuneMode::Unknown, "reset clears locked mode");
    check(classifier.speed_samples() == 0, "reset clears speed history");

    auto first = solved_rune(big_angle);
    const auto after_reset =
      classifier.observe(first, t0 + std::chrono::milliseconds(20 * 200));
    check(
      after_reset == auto_buff::PowerRuneMode::Unknown,
      "after reset, classifier does not keep the old small label");
  }

  {
    ModeRouter router;
    router.activate(auto_buff::PowerRuneMode::Small);
    check(dynamic_cast<auto_buff::SmallTarget *>(router.target.get()) != nullptr,
          "small mode constructs SmallTarget");
    check(router.target && router.target->is_unsolve(), "fresh SmallTarget starts unsolvable");
    check(router.small_instances == 1 && router.big_instances == 0, "one small estimator so far");
    check(router.classifier.mode() == auto_buff::PowerRuneMode::Unknown,
          "activate_mode resets classifier");

    router.activate(auto_buff::PowerRuneMode::Big);
    check(dynamic_cast<auto_buff::BigTarget *>(router.target.get()) != nullptr,
          "big mode constructs BigTarget, not reused SmallTarget");
    check(dynamic_cast<auto_buff::SmallTarget *>(router.target.get()) == nullptr,
          "old SmallTarget is not retained across mode change");
    check(router.target && router.target->is_unsolve(), "fresh BigTarget starts unsolvable");
    check(router.small_instances == 1 && router.big_instances == 1,
          "mode switch allocates a new big estimator");
    check(router.classifier.mode() == auto_buff::PowerRuneMode::Unknown,
          "mode switch resets classifier so old small lock cannot stick");

    router.activate(auto_buff::PowerRuneMode::Small);
    check(dynamic_cast<auto_buff::SmallTarget *>(router.target.get()) != nullptr,
          "switch back allocates a new SmallTarget");
    check(router.small_instances == 2 && router.big_instances == 1,
          "each mode change creates a new estimator, never reuse");
  }

  {
    auto_buff::Aimer aimer(config);
    FakeTarget target({5.0, 0.0, 0.0});
    target.set_blade_angle(0.0);
    auto ts = timestamp;
    const auto first = aimer.aim(target, ts, 24.0, false);
    check(!first.shoot, "first valid aim confirms and does not fire");

    const auto second = aimer.aim(target, ts, 24.0, false);
    check(second.control && !second.shoot, "stable track may control but not fire before gap");

    target.set_blade_angle(2.4);
    target.set_point({5.0, 2.2, 0.0});
    const auto outlier = aimer.aim(target, ts, 24.0, false);
    check(!outlier.shoot, "noisy/outlier blade angles must not fire");
    if (outlier.shoot) {
      check(false, "outlier angles still producing fire");
    }
  }

  {
    auto_buff::Aimer aimer(config);
    const Eigen::Vector3d truth{5.0, 0.0, 0.0};
    const Eigen::Vector3d opposite{-5.0, 0.0, 0.0};
    const double truth_yaw = std::atan2(truth.y(), truth.x());

    FakeTarget target(truth);
    auto ts = timestamp;
    (void)aimer.aim(target, ts, 24.0, false);
    (void)aimer.aim(target, ts, 24.0, false);

    target.set_point(opposite);
    target.set_blade_angle(CV_PI);
    const auto command = aimer.aim(target, ts, 24.0, false);
    const bool opposite_cmd = yaw_opposite(command.yaw, truth_yaw);
    check(!(opposite_cmd && command.shoot),
          "commanded direction opposite truth must not fire");
    if (command.shoot && opposite_cmd) {
      check(false, "opposite-to-truth command still producing fire");
    }
  }

  {
    auto_buff::Aimer aimer(config);
    const Eigen::Vector3d truth{5.0, 0.0, 0.0};
    const Eigen::Vector3d opposite{-5.0, 0.0, 0.0};
    const double truth_yaw = std::atan2(truth.y(), truth.x());
    FakeTarget target(opposite);
    auto ts = timestamp;
    const auto first = aimer.aim(target, ts, 24.0, false);
    const auto second = aimer.aim(target, ts, 24.0, false);
    const bool first_opposite = yaw_opposite(first.yaw, truth_yaw);
    const bool second_opposite = yaw_opposite(second.yaw, truth_yaw);
    check(!(first_opposite && first.shoot), "first opposite-to-truth aim must not fire");
    check(!(second_opposite && second.shoot), "stable opposite-to-truth aim must not fire");
  }

  std::printf(
    "buff_counterexample_test: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
