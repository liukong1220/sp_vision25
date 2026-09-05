#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_mode_classifier.hpp"
#include "tools/path.hpp"

namespace
{
int failures = 0;

void check(bool condition, const char * what)
{
  std::printf("%-64s %s\n", what, condition ? "ok" : "FAIL");
  if (!condition) ++failures;
}

class FakeTarget final : public auto_buff::Target
{
public:
  enum class Motion { Fixed, SecondFar, SecondUnsolvable };

  FakeTarget(double distance_m, Motion motion = Motion::Fixed)
  : distance_m_(distance_m), motion_(motion)
  {
    unsolvable_ = false;
    ekf_.x = Eigen::VectorXd::Zero(7);
  }

  void get_target(
    const std::optional<auto_buff::PowerRune> &,
    std::chrono::steady_clock::time_point &) override
  {
  }

  void predict(double) override
  {
    ++predict_calls_;
    if (predict_calls_ == 2 && motion_ == Motion::SecondFar) distance_m_ = 25.0;
    if (predict_calls_ == 2 && motion_ == Motion::SecondUnsolvable) distance_m_ = 200.0;
  }

  std::unique_ptr<auto_buff::Target> clone() const override
  {
    return std::make_unique<FakeTarget>(*this);
  }

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d &) const override
  {
    return {distance_m_, 0.0, 0.0};
  }

private:
  double distance_m_;
  Motion motion_;
  int predict_calls_ = 0;

  void init(double, const auto_buff::PowerRune &) override {}
  void update(double, const auto_buff::PowerRune &) override {}
};

auto_buff::PowerRune solved_rune(double angle)
{
  auto_buff::PowerRune rune;
  rune.ypr_in_world = {0.0, 0.0, angle};
  return rune;
}

auto_buff::PowerRune make_big_rune(double angle)
{
  auto_buff::PowerRune rune;
  rune.set_mode(auto_buff::PowerRuneMode::Big);
  rune.xyz_in_world = Eigen::Vector3d(5.0, 0.0, 0.0);
  rune.ypr_in_world = Eigen::Vector3d(0.0, 0.0, angle);
  rune.ypd_in_world = Eigen::Vector3d(0.0, 0.0, 5.0);
  rune.blade_xyz_in_world = Eigen::Vector3d(5.0, 0.0, 0.7);
  rune.blade_ypd_in_world = Eigen::Vector3d(0.0, 0.14, 5.05);
  return rune;
}

class InspectableBigTarget final : public auto_buff::BigTarget
{
public:
  void inject_diverge()
  {
    if (ekf_.x.size() > 8) {
      ekf_.x[7] = 10.0;
      ekf_.x[8] = 10.0;
    }
  }
};
}  // namespace

int main()
{
  const std::string config = tools::resolve_config_path_string("configs/simulation.yaml");
  const auto timestamp = std::chrono::steady_clock::now();

  {
    auto_buff::Aimer aimer(config);
    FakeTarget target(200.0);
    auto ts = timestamp;
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(
      aimer.last_status() == auto_buff::Aimer::SolveStatus::Trajectory0Unsolvable,
      "trajectory0 不可解有显式状态");
    check(!command.control && !command.shoot, "trajectory0 不可解返回 no-control/no-fire");
  }

  {
    auto_buff::Aimer aimer(config);
    FakeTarget target(10.0, FakeTarget::Motion::SecondUnsolvable);
    auto ts = timestamp;
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(
      aimer.last_status() == auto_buff::Aimer::SolveStatus::Trajectory1Unsolvable,
      "trajectory1 不可解有显式状态");
    check(!command.control && !command.shoot, "trajectory1 不可解返回 no-control/no-fire");
  }

  {
    auto_buff::Aimer aimer(config);
    FakeTarget target(10.0, FakeTarget::Motion::SecondFar);
    auto ts = timestamp;
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(aimer.last_status() == auto_buff::Aimer::SolveStatus::TimeError, "大时间误差有显式状态");
    check(!command.control && !command.shoot, "大时间误差返回 no-control/no-fire");
  }

  {
    auto_buff::Aimer aimer(config);
    FakeTarget first(5.0);
    auto ts = timestamp;
    const auto command1 = aimer.aim(first, ts, 24.0, false);
    check(aimer.last_status() == auto_buff::Aimer::SolveStatus::Ok, "首次有效目标可解");
    check(!command1.control && !command1.shoot, "首次有效目标先确认且不开火");

    FakeTarget second(5.0);
    const auto command2 = aimer.aim(second, ts, 24.0, false);
    check(command2.control && !command2.shoot, "连续有效目标恢复控制但不提前开火");
  }

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
    check(mode == auto_buff::PowerRuneMode::Small, "恒速观测窗后分类为 small");
  }

  {
    auto_buff::RuneModeClassifier classifier;
    double angle = 0.0;
    bool ever_small = false;
    auto_buff::PowerRuneMode mode = auto_buff::PowerRuneMode::Unknown;
    constexpr double amplitude = 0.90;
    constexpr double omega = 1.94;
    for (int i = 0; i < 160; ++i) {
      const double t = i * 0.02;
      const double speed = amplitude * std::sin(omega * t) + 2.09 - amplitude;
      angle += speed * 0.02;
      auto rune = solved_rune(angle);
      mode = classifier.observe(rune, t0 + std::chrono::milliseconds(20 * i));
      ever_small = ever_small || mode == auto_buff::PowerRuneMode::Small;
    }
    check(!ever_small, "大符正弦运动在任意前缀都不误判为 small");
    check(mode == auto_buff::PowerRuneMode::Big, "大符正弦运动分类为 big");
  }

  {
    auto_buff::Aimer aimer(config);
    auto_buff::BigTarget target;
    auto ts = t0;
    auto rune = make_big_rune(0.0);
    target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    check(target.is_unsolve(), "big first fit unsolvable");
    check(target.fitter_sample_count() == 0, "big first fit fitter empty");
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(!command.control && !command.shoot, "big first fit no control/fire");
    check(aimer.last_status() == auto_buff::Aimer::SolveStatus::NoTarget, "big first fit Aimer NoTarget");
  }

  {
    auto_buff::Aimer aimer(config);
    auto_buff::BigTarget target;
    for (int i = 0; i < 3; ++i) {
      auto ts = t0 + std::chrono::milliseconds(20 * i);
      auto rune = make_big_rune(0.04 * i);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    check(target.is_unsolve(), "big 1-2 samples unsolvable");
    check(target.fitter_sample_count() < 3, "big 1-2 samples fitter count");
    auto ts = t0 + std::chrono::milliseconds(40);
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(!command.control && !command.shoot, "big 1-2 samples no control/fire");
  }

  {
    auto_buff::Aimer aimer(config);
    InspectableBigTarget target;
    for (int i = 0; i < 4; ++i) {
      auto ts = t0 + std::chrono::milliseconds(20 * i);
      auto rune = make_big_rune(0.04 * i);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    target.inject_diverge();
    auto ts = t0 + std::chrono::milliseconds(80);
    auto rune = make_big_rune(0.16);
    target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    check(target.is_unsolve(), "big diverge unsolvable");
    check(target.fitter_sample_count() == 0, "big diverge fitter sample_count==0");
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(!command.control && !command.shoot, "big diverge no control/fire");
  }

  {
    auto_buff::Aimer aimer(config);
    auto_buff::BigTarget target;
    for (int i = 0; i < 4; ++i) {
      auto ts = t0 + std::chrono::milliseconds(20 * i);
      auto rune = make_big_rune(0.04 * i);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    for (int i = 4; i < 11; ++i) {
      auto ts = t0 + std::chrono::milliseconds(20 * i);
      target.get_target(std::nullopt, ts);
    }
    {
      auto ts = t0 + std::chrono::milliseconds(220);
      auto rune = make_big_rune(0.44);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    check(target.is_unsolve(), "big loss>6 cleanup unsolvable");
    check(target.fitter_sample_count() == 0, "big loss>6 fitter cleared");
    {
      auto ts = t0 + std::chrono::milliseconds(240);
      auto rune = make_big_rune(0.48);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    check(target.is_unsolve(), "big loss recovery re-init unsolvable");
    check(target.fitter_sample_count() == 0, "big loss recovery init no samples");
    {
      auto ts = t0 + std::chrono::milliseconds(260);
      auto rune = make_big_rune(0.52);
      target.get_target(std::optional<auto_buff::PowerRune>(rune), ts);
    }
    check(target.is_unsolve(), "big loss recovery unsolvable until enough samples");
    auto ts = t0 + std::chrono::milliseconds(260);
    const auto command = aimer.aim(target, ts, 24.0, false);
    check(!command.control && !command.shoot, "big loss recovery no control/fire");
  }

  std::printf("buff_safety_test: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
