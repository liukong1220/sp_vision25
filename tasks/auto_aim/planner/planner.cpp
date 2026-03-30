#include "planner.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
constexpr double kNormalVisibleAngle = 60.0 / 57.3;
constexpr double kSpinSpeedThreshold = 2.0;

AimSelection choose_aim_selection(
  const Target & target, double coming_angle, double leaving_angle, int & lock_id)
{
  AimSelection selection;
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return selection;

  const Eigen::VectorXd ekf_x = target.ekf_x();
  selection.center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  selection.delta_angle_list.reserve(armor_xyza_list.size());

  for (const auto & xyza : armor_xyza_list) {
    selection.delta_angle_list.push_back(
      tools::limit_rad(xyza[3] - selection.center_yaw));
  }

  auto fill_selection = [&](int armor_id, bool used_spin_gate) {
    selection.valid = true;
    selection.armor_id = armor_id;
    selection.used_spin_gate = used_spin_gate;
    selection.xyza = armor_xyza_list[armor_id];
  };

  auto fallback_to_closest = [&]() {
    int best_id = 0;
    double best_score = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(selection.delta_angle_list.size()); ++i) {
      const double score = std::abs(selection.delta_angle_list[i]);
      if (score < best_score) {
        best_score = score;
        best_id = i;
      }
    }
    lock_id = -1;
    fill_selection(best_id, false);
  };

  // 当目标还没有发生过真实跳板时，EKF 只对当前板的观测最可信。
  // 这时强行按照小陀螺策略切板，往往会把初始相位带偏，导致第一枪就明显超前。
  if (!target.jumped) {
    lock_id = 0;
    fill_selection(0, false);
    return selection;
  }

  const double target_w = ekf_x[7];
  const bool use_spin_gate =
    std::abs(target_w) > kSpinSpeedThreshold || target.name == ArmorName::outpost;

  if (!use_spin_gate) {
    std::vector<int> visible_id_list;
    for (int i = 0; i < static_cast<int>(selection.delta_angle_list.size()); ++i) {
      if (std::abs(selection.delta_angle_list[i]) <= kNormalVisibleAngle) {
        visible_id_list.push_back(i);
      }
    }

    // 非小陀螺时，仍然保留“锁板”策略，避免两块侧前板在 40~50 度附近来回抖动。
    if (visible_id_list.empty()) {
      fallback_to_closest();
      return selection;
    }

    if (visible_id_list.size() == 1) {
      lock_id = -1;
      fill_selection(visible_id_list.front(), false);
      return selection;
    }

    if (
      std::find(visible_id_list.begin(), visible_id_list.end(), lock_id) ==
      visible_id_list.end())
    {
      lock_id = visible_id_list.front();
      double best_score = std::abs(selection.delta_angle_list[lock_id]);
      for (const int id : visible_id_list) {
        const double score = std::abs(selection.delta_angle_list[id]);
        if (score < best_score) {
          best_score = score;
          lock_id = id;
        }
      }
    }

    fill_selection(lock_id, false);
    return selection;
  }

  // 小陀螺时，不再盲目选“最近板”，而是优先选正在进入可射击窗口的那块板。
  // 这样可以显著减少逆时针/顺时针时总感觉“点位靠前或靠后”的现象。
  int best_id = -1;
  double best_score = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(selection.delta_angle_list.size()); ++i) {
    const double delta_angle = selection.delta_angle_list[i];
    if (std::abs(delta_angle) > coming_angle) continue;

    bool entering_window = false;
    if (target_w > 0) entering_window = delta_angle < leaving_angle;
    if (target_w < 0) entering_window = delta_angle > -leaving_angle;
    if (!entering_window) continue;

    const double score = std::abs(delta_angle);
    if (score < best_score) {
      best_score = score;
      best_id = i;
    }
  }

  if (best_id == -1) {
    fallback_to_closest();
    return selection;
  }

  lock_id = -1;
  fill_selection(best_id, true);
  return selection;
}
}  // namespace

Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  coming_angle_ = tools::read<double>(yaml, "comming_angle") / 57.3;
  leaving_angle_ = tools::read<double>(yaml, "leaving_angle") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  
  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. 先基于“当前真正准备击打的装甲板”估计弹丸飞行时间。
  // 之前这里使用的是最近装甲板，小陀螺时很容易把预测时间算到另一块板上，
  // 结果就是切高低板时 pitch 偏高，或者逆时针慢转时整体打点偏一侧。
  const auto fly_time_selection = preview_aim_selection(target);
  if (!fly_time_selection.valid) return {false};

  const double min_dist = fly_time_selection.xyza.head<2>().norm();
  auto bullet_traj =
    tools::Trajectory(bullet_speed, min_dist, fly_time_selection.xyza.z());
  if (bullet_traj.unsolvable) return {false};
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  tools::logger()->debug(
    "yaw: {:.4f}, yaw_vel: {:.4f}, yaw_acc: {:.4f}, pitch: {:.4f}, pitch_vel: {:.4f}, pitch_acc: {:.4f}",
    plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel, plan.pitch_acc);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  debug_delay_time = 0.0;
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
  debug_delay_time = delay_time;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

AimSelection Planner::preview_aim_selection(const Target & target) const
{
  int lock_id = -1;
  const double coming_angle =
    (target.name == ArmorName::outpost) ? 70.0 / 57.3 : coming_angle_;
  const double leaving_angle =
    (target.name == ArmorName::outpost) ? 30.0 / 57.3 : leaving_angle_;
  return choose_aim_selection(target, coming_angle, leaving_angle, lock_id);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  const double coming_angle =
    (target.name == ArmorName::outpost) ? 70.0 / 57.3 : coming_angle_;
  const double leaving_angle =
    (target.name == ArmorName::outpost) ? 30.0 / 57.3 : leaving_angle_;
  const auto selection =
    choose_aim_selection(target, coming_angle, leaving_angle, lock_id_);
  if (!selection.valid) throw std::runtime_error("No valid armor selected!");

  debug_xyza = selection.xyza;
  debug_armor_id = selection.armor_id;
  debug_used_spin_gate = selection.used_spin_gate;
  debug_center_yaw = selection.center_yaw;
  debug_delta_angle_list = selection.delta_angle_list;

  const Eigen::Vector3d xyz = selection.xyza.head<3>();
  const double dist_xy = xyz.head<2>().norm();
  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, dist_xy, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);

  // 这里不能用 static 保存上一帧的 yaw_vel。
  // 因为切板瞬间会把“上一帧目标的速度”带进“当前帧参考轨迹”，
  // 最直观的表现就是小陀螺切板后，MPC 参考轨迹突然被拉歪，点位像是被推前了一截。
  double last_yaw_vel =
    tools::limit_rad(yaw_pitch(0) - yaw_pitch_last(0)) / DT;

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    const double yaw_acc = (yaw_vel - last_yaw_vel) / DT;

    // 切板时参考角速度会出现突变。这里做一个轻量抑制，
    // 只削掉尖峰，不改掉整体旋向，避免 MPC 被一帧异常“拽着走”。
    if (std::abs(yaw_acc) > 10.0) {
      yaw_vel *= 0.7;
    }
    last_yaw_vel = yaw_vel;

    const auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim
