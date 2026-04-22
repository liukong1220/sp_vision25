#include "planner.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"
#include "tools/runtime_params.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
constexpr double kNormalVisibleAngle = 60.0 / 57.3;
constexpr double kSpinSpeedThreshold = 2.0;
constexpr int kHitTimeIterMax = 6;
constexpr double kHitTimeTol = 1e-3;
constexpr double kMinOutpostFireAngle = 4.0 / 57.3;
constexpr double kMaxOutpostFireAngle = 12.0 / 57.3;

double resolve_outpost_fire_phase_angle(double leaving_angle)
{
  if (leaving_angle <= 1e-6) return 8.0 / 57.3;

  const double fire_angle =
    std::clamp(leaving_angle * 0.5, kMinOutpostFireAngle, kMaxOutpostFireAngle);
  return std::min(fire_angle, leaving_angle);
}

double resolve_selected_delta_angle(const AimSelection & selection)
{
  if (
    selection.armor_id < 0 ||
    selection.armor_id >= static_cast<int>(selection.delta_angle_list.size()))
  {
    return 0.0;
  }
  return selection.delta_angle_list[selection.armor_id];
}

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
    selection.selected_delta_angle = resolve_selected_delta_angle(selection);
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
    const int armor_count = static_cast<int>(armor_xyza_list.size());
    const int observed_id = (target.last_id % armor_count + armor_count) % armor_count;
    lock_id = observed_id;
    fill_selection(observed_id, false);
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

struct HitTargetSolution
{
  bool valid = false;
  Target target_at_hit;
  AimSelection selection;
  double fly_time = 0.0;
  int iter_count = 0;
  bool converged = false;
};

HitTargetSolution solve_hit_target(
  const Target & base_target, double bullet_speed, double coming_angle, double leaving_angle,
  int initial_lock_id,
  const std::function<Eigen::Vector3d(const Target &, const AimSelection &)> & resolve_aim_xyz)
{
  HitTargetSolution solution;
  int working_lock_id = initial_lock_id;
  double fly_time = 0.0;
  int previous_armor_id = -1;

  for (int iter = 0; iter < kHitTimeIterMax; ++iter) {
    Target iter_target = base_target;
    if (fly_time > 0.0) {
      iter_target.predict(fly_time);
    }

    auto selection =
      choose_aim_selection(iter_target, coming_angle, leaving_angle, working_lock_id);
    if (!selection.valid) return solution;

    const Eigen::Vector3d xyz = resolve_aim_xyz(iter_target, selection);
    const double dist_xy = xyz.head<2>().norm();
    auto bullet_traj = tools::Trajectory(bullet_speed, dist_xy, xyz.z());
    if (bullet_traj.unsolvable) return solution;

    Target hit_target = base_target;
    hit_target.predict(bullet_traj.fly_time);

    if (
      iter > 0 && std::abs(bullet_traj.fly_time - fly_time) < kHitTimeTol &&
      selection.armor_id == previous_armor_id)
    {
      solution.valid = true;
      solution.target_at_hit = hit_target;
      solution.selection = selection;
      solution.fly_time = bullet_traj.fly_time;
      solution.iter_count = iter + 1;
      solution.converged = true;
      return solution;
    }

    fly_time = bullet_traj.fly_time;
    previous_armor_id = selection.armor_id;
    solution.valid = true;
    solution.target_at_hit = hit_target;
    solution.selection = selection;
    solution.fly_time = bullet_traj.fly_time;
    solution.iter_count = iter + 1;
  }

  return solution;
}
}  // namespace

Planner::Planner(const std::string & config_path)
: config_path_(tools::resolve_config_path_string(config_path)), yaw_solver_(nullptr), pitch_solver_(nullptr)
{
  auto yaml = tools::load(config_path_);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  coming_angle_ = tools::read<double>(yaml, "comming_angle") / 57.3;
  leaving_angle_ = tools::read<double>(yaml, "leaving_angle") / 57.3;
  outpost_coming_angle_ = tools::read_or<double>(yaml, "outpost_comming_angle", 70.0) / 57.3;
  outpost_leaving_angle_ = tools::read_or<double>(yaml, "outpost_leaving_angle", 30.0) / 57.3;
  outpost_delay_time_ = tools::read_or<double>(yaml, "outpost_delay_time", 0.0);
  outpost_fire_z_compensation_ =
    tools::read_or<std::vector<double>>(yaml, "outpost_fire_z_compensation", {0.0, 0.0, 0.0});
  if (outpost_fire_z_compensation_.size() != 3) {
    tools::logger()->warn(
      "[Planner] outpost_fire_z_compensation size {} invalid, fallback to zeros",
      outpost_fire_z_compensation_.size());
    outpost_fire_z_compensation_ = {0.0, 0.0, 0.0};
  }
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver();
  setup_pitch_solver();
  runtime_params_version_ = tools::runtime_params::version(config_path_);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  refresh_runtime_params_if_needed();
  debug_hit_fly_time = 0.0;
  debug_hit_iter_count = 0;
  debug_hit_converged = false;
  debug_fire_tracking_error = 0.0;
  debug_fire_phase_limit = 0.0;
  debug_fire_track_ready = false;
  debug_fire_phase_ready = false;

  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. 对命中时刻做固定点迭代：
  //    目标先按控制延迟预测到“当前决策时刻”，再在 planner 内部迭代
  //    `选板 -> 算飞行时间 -> 预测到命中时刻 -> 再选板`。
  //    这一步对 outpost 尤其关键，否则会出现 tracker 跟的是眼前板，
  //    planner 却拿另一块板的粗略飞行时间去做整段 MPC 参考。
  const auto [coming_angle, leaving_angle] = resolve_angle_window(target);
  const auto hit_solution =
    solve_hit_target(
      target, bullet_speed, coming_angle, leaving_angle, lock_id_,
      [&](const Target & iter_target, const AimSelection & selection) {
        return resolve_aim_xyz(iter_target, selection);
      });
  if (!hit_solution.valid) return {false};
  debug_hit_fly_time = hit_solution.fly_time;
  debug_hit_iter_count = hit_solution.iter_count;
  debug_hit_converged = hit_solution.converged;
  target = hit_solution.target_at_hit;

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  AimSelection selection;
  try {
    int planning_lock_id = hit_solution.selection.armor_id >= 0 ? hit_solution.selection.armor_id : lock_id_;
    const auto yaw_pitch =
      solve_aim_command(target, bullet_speed, planning_lock_id, &selection);
    update_debug_selection(target, selection);
    lock_id_ = planning_lock_id;
    yaw0 = yaw_pitch(0);
    traj = get_trajectory(target, yaw0, bullet_speed, planning_lock_id);
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
  const double tracking_error =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_));
  debug_fire_tracking_error = tracking_error;
  debug_fire_track_ready = tracking_error < fire_thresh_;
  debug_fire_phase_ready = true;

  bool fire_ready = debug_fire_track_ready;
  if (target.name == ArmorName::outpost) {
    // Keep using the nearest board for trajectory continuity in the inter-board gap,
    // but only allow fire once the selected plate is back in a tight hit phase.
    const bool spin_gate_ready = !target.jumped || selection.used_spin_gate;
    debug_fire_phase_limit = resolve_outpost_fire_phase_angle(leaving_angle);
    debug_fire_phase_ready =
      spin_gate_ready && std::abs(selection.selected_delta_angle) <= debug_fire_phase_limit;
    fire_ready = fire_ready && hit_solution.converged && debug_fire_phase_ready;
  }

  plan.fire = fire_ready;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  refresh_runtime_params_if_needed();
  debug_delay_time = 0.0;
  if (!target.has_value()) {
    lock_id_ = -1;
    return {false};
  }

  const double delay_time = resolve_delay_time(*target);
  debug_delay_time = delay_time;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

AimSelection Planner::preview_aim_selection(const Target & target) const
{
  int lock_id = target.last_id;
  const auto [coming_angle, leaving_angle] = resolve_angle_window(target);
  return choose_aim_selection(target, coming_angle, leaving_angle, lock_id);
}

void Planner::refresh_runtime_params_if_needed()
{
  const auto current_version = tools::runtime_params::version(config_path_);
  if (current_version == 0 || current_version == runtime_params_version_) return;

  yaw_offset_ = tools::runtime_params::get_double(config_path_, "yaw_offset") / 57.3;
  pitch_offset_ = tools::runtime_params::get_double(config_path_, "pitch_offset") / 57.3;
  coming_angle_ = tools::runtime_params::get_double(config_path_, "comming_angle") / 57.3;
  leaving_angle_ = tools::runtime_params::get_double(config_path_, "leaving_angle") / 57.3;
  outpost_coming_angle_ =
    tools::runtime_params::get_double(config_path_, "outpost_comming_angle") / 57.3;
  outpost_leaving_angle_ =
    tools::runtime_params::get_double(config_path_, "outpost_leaving_angle") / 57.3;
  outpost_delay_time_ = tools::runtime_params::get_double(config_path_, "outpost_delay_time");
  outpost_fire_z_compensation_ =
    tools::runtime_params::get_number_array(config_path_, "outpost_fire_z_compensation");
  if (outpost_fire_z_compensation_.size() != 3) {
    tools::logger()->warn(
      "[Planner] outpost_fire_z_compensation size {} invalid, fallback to zeros",
      outpost_fire_z_compensation_.size());
    outpost_fire_z_compensation_ = {0.0, 0.0, 0.0};
  }
  fire_thresh_ = tools::runtime_params::get_double(config_path_, "fire_thresh");
  decision_speed_ = tools::runtime_params::get_double(config_path_, "decision_speed");
  high_speed_delay_time_ = tools::runtime_params::get_double(config_path_, "high_speed_delay_time");
  low_speed_delay_time_ = tools::runtime_params::get_double(config_path_, "low_speed_delay_time");

  setup_yaw_solver();
  setup_pitch_solver();
  runtime_params_version_ = current_version;
  tools::logger()->info("[Planner] runtime params updated to v{}", current_version);
}

void Planner::setup_yaw_solver()
{
  double max_yaw_acc = 0.0;
  std::vector<double> Q_yaw;
  std::vector<double> R_yaw;

  if (tools::runtime_params::is_registered(config_path_)) {
    max_yaw_acc = tools::runtime_params::get_double(config_path_, "max_yaw_acc");
    Q_yaw = tools::runtime_params::get_number_array(config_path_, "Q_yaw");
    R_yaw = tools::runtime_params::get_number_array(config_path_, "R_yaw");
  } else {
    auto yaml = tools::load(config_path_);
    max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
    Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
    R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");
  }

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  constexpr double rho = 1.0;
  if (!yaw_solver_) {
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), rho, 2, 1, HORIZON, 0);
  } else {
    yaw_solver_->work->Q = Q + Eigen::Matrix<double, 2, 1>::Constant(rho);
    yaw_solver_->work->R = R + Eigen::Matrix<double, 1, 1>::Constant(rho);
    yaw_solver_->work->Adyn = A;
    yaw_solver_->work->Bdyn = B;
    yaw_solver_->work->fdyn = f;
    tiny_precompute_and_set_cache(
      yaw_solver_->cache, A, B, f, yaw_solver_->work->Q.asDiagonal(),
      yaw_solver_->work->R.asDiagonal(), 2, 1, rho, 0);
  }

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver()
{
  double max_pitch_acc = 0.0;
  std::vector<double> Q_pitch;
  std::vector<double> R_pitch;

  if (tools::runtime_params::is_registered(config_path_)) {
    max_pitch_acc = tools::runtime_params::get_double(config_path_, "max_pitch_acc");
    Q_pitch = tools::runtime_params::get_number_array(config_path_, "Q_pitch");
    R_pitch = tools::runtime_params::get_number_array(config_path_, "R_pitch");
  } else {
    auto yaml = tools::load(config_path_);
    max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
    Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
    R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");
  }

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  constexpr double rho = 1.0;
  if (!pitch_solver_) {
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), rho, 2, 1, HORIZON, 0);
  } else {
    pitch_solver_->work->Q = Q + Eigen::Matrix<double, 2, 1>::Constant(rho);
    pitch_solver_->work->R = R + Eigen::Matrix<double, 1, 1>::Constant(rho);
    pitch_solver_->work->Adyn = A;
    pitch_solver_->work->Bdyn = B;
    pitch_solver_->work->fdyn = f;
    tiny_precompute_and_set_cache(
      pitch_solver_->cache, A, B, f, pitch_solver_->work->Q.asDiagonal(),
      pitch_solver_->work->R.asDiagonal(), 2, 1, rho, 0);
  }

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::solve_aim_command(
  const Target & target, double bullet_speed, int & lock_id, AimSelection * selection,
  Eigen::Vector3d * aim_xyz) const
{
  const auto [coming_angle, leaving_angle] = resolve_angle_window(target);
  const auto resolved_selection =
    choose_aim_selection(target, coming_angle, leaving_angle, lock_id);
  if (!resolved_selection.valid) throw std::runtime_error("No valid armor selected!");
  if (selection != nullptr) *selection = resolved_selection;

  const Eigen::Vector3d xyz = resolve_aim_xyz(target, resolved_selection);
  if (aim_xyz != nullptr) *aim_xyz = xyz;
  const double dist_xy = xyz.head<2>().norm();
  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, dist_xy, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

void Planner::update_debug_selection(const Target & target, const AimSelection & selection)
{
  debug_xyza = selection.xyza;
  debug_armor_id = selection.armor_id;
  debug_physical_armor_id = target.physical_armor_id(selection.armor_id);
  debug_used_spin_gate = selection.used_spin_gate;
  debug_center_yaw = selection.center_yaw;
  debug_selected_z_offset = target.armor_z_offset(selection.armor_id);
  debug_selected_aim_z_compensation = resolve_aim_z_compensation(target, selection.armor_id);
  debug_selected_delta_angle = selection.selected_delta_angle;
  debug_xyza[2] += debug_selected_aim_z_compensation;
  debug_fixed_center_rotation_model = target.fixed_center_rotation_model();
  debug_delta_angle_list = selection.delta_angle_list;
}

std::pair<double, double> Planner::resolve_angle_window(const Target & target) const
{
  if (target.name == ArmorName::outpost) {
    const double coming_angle = outpost_coming_angle_ > 0.0 ? outpost_coming_angle_ : coming_angle_;
    const double leaving_angle =
      outpost_leaving_angle_ > 0.0 ? outpost_leaving_angle_ : leaving_angle_;
    return {coming_angle, leaving_angle};
  }
  return {coming_angle_, leaving_angle_};
}

double Planner::resolve_delay_time(const Target & target) const
{
  if (target.name == ArmorName::outpost && outpost_delay_time_ > 0.0) {
    return outpost_delay_time_;
  }

  return
    std::abs(target.ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
}

double Planner::resolve_aim_z_compensation(const Target & target, int armor_id) const
{
  if (target.name != ArmorName::outpost || armor_id < 0) return 0.0;
  if (outpost_fire_z_compensation_.size() != 3) return 0.0;

  const int physical_id = target.physical_armor_id(armor_id);
  if (physical_id < 0 || physical_id >= static_cast<int>(outpost_fire_z_compensation_.size())) {
    return 0.0;
  }
  return outpost_fire_z_compensation_[physical_id];
}

Eigen::Vector3d Planner::resolve_aim_xyz(
  const Target & target, const AimSelection & selection) const
{
  Eigen::Vector3d xyz = selection.xyza.head<3>();
  xyz.z() += resolve_aim_z_compensation(target, selection.armor_id);
  return xyz;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  AimSelection selection;
  const auto yaw_pitch = solve_aim_command(target, bullet_speed, lock_id_, &selection);
  update_debug_selection(target, selection);
  return yaw_pitch;
}

Trajectory Planner::get_trajectory(
  Target & target, double yaw0, double bullet_speed, int initial_lock_id)
{
  Trajectory traj;
  int trajectory_lock_id = initial_lock_id;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = solve_aim_command(target, bullet_speed, trajectory_lock_id);

  target.predict(DT);
  auto yaw_pitch = solve_aim_command(target, bullet_speed, trajectory_lock_id);

  // 这里不能用 static 保存上一帧的 yaw_vel。
  // 因为切板瞬间会把“上一帧目标的速度”带进“当前帧参考轨迹”，
  // 最直观的表现就是小陀螺切板后，MPC 参考轨迹突然被拉歪，点位像是被推前了一截。
  double last_yaw_vel =
    tools::limit_rad(yaw_pitch(0) - yaw_pitch_last(0)) / DT;

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = solve_aim_command(target, bullet_speed, trajectory_lock_id);

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
