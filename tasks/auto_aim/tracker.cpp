#include "tracker.hpp"

#include <cmath>
#include <limits>
#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"
#include "tools/runtime_params.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{
namespace
{
struct OutpostMatchResult
{
  std::list<Armor>::iterator armor_it;
  int id = -1;
  int offset = 0;
  int physical_id = -1;
  double score = std::numeric_limits<double>::infinity();
  double reprojection_error = std::numeric_limits<double>::infinity();
  double xy_error = std::numeric_limits<double>::infinity();
  double z_error = std::numeric_limits<double>::infinity();
  bool valid = false;
};

double average_point_error(
  const std::vector<cv::Point2f> & measured, const std::vector<cv::Point2f> & predicted)
{
  if (measured.size() != predicted.size() || measured.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double total_error = 0.0;
  for (std::size_t i = 0; i < measured.size(); ++i) {
    total_error += cv::norm(measured[i] - predicted[i]);
  }
  return total_error / static_cast<double>(measured.size());
}

int cyclic_id_distance(int lhs, int rhs, int armor_num)
{
  const int diff = std::abs(lhs - rhs);
  return std::min(diff, armor_num - diff);
}

OutpostMatchResult select_best_outpost_match(
  const Target & target, Solver & solver, std::list<Armor> & armors)
{
  OutpostMatchResult best_match;
  const int armor_num = static_cast<int>(target.armor_xyza_list().size());
  if (armor_num <= 0) return best_match;

  constexpr double kYawGate = 6.0 / 57.3;
  constexpr double kPitchGate = 5.0 / 57.3;
  constexpr double kArmorYawGate = 14.0 / 57.3;
  constexpr double kDistanceGate = 0.16;
  constexpr double kXYGate = 0.12;
  constexpr double kZGate = 0.05;
  constexpr double kReprojectionGate = 18.0;

  for (auto it = armors.begin(); it != armors.end(); ++it) {
    auto & armor = *it;
    if (armor.name != target.name || armor.type != target.armor_type) continue;

    solver.solve(armor);
    const Eigen::Vector3d observed_xyz = armor.xyz_in_world;
    const Eigen::Vector3d observed_ypd = armor.ypd_in_world;

    for (int id = 0; id < armor_num; ++id) {
      for (int offset = 0; offset < armor_num; ++offset) {
        Target mapped_target = target;
        mapped_target.set_armor_id_offset(offset, target.last_id);
        const auto predicted_armors = mapped_target.armor_xyza_list();
        const auto & predicted_xyza = predicted_armors[id];
        const Eigen::Vector3d predicted_xyz = predicted_xyza.head<3>();
        const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(predicted_xyz);
        const auto predicted_points =
          solver.reproject_armor(predicted_xyz, predicted_xyza[3], armor.type, armor.name);

        const double reprojection_error = average_point_error(armor.points, predicted_points);
        const double los_yaw_error =
          std::abs(tools::limit_rad(observed_ypd[0] - predicted_ypd[0]));
        const double pitch_error = std::abs(observed_ypd[1] - predicted_ypd[1]);
        const double distance_error = std::abs(observed_ypd[2] - predicted_ypd[2]);
        const double armor_yaw_error =
          std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_xyza[3]));
        const double xy_error = (observed_xyz.head<2>() - predicted_xyz.head<2>()).norm();
        const double z_error = std::abs(observed_xyz.z() - predicted_xyz.z());

        double continuity_penalty = 0.0;
        if (id != target.last_id) {
          const int step = cyclic_id_distance(id, target.last_id, armor_num);
          continuity_penalty += target.jumped ? 0.8 * step : 0.35 * step;
        }
        if (offset != target.armor_id_offset()) {
          const int step = cyclic_id_distance(offset, target.armor_id_offset(), armor_num);
          continuity_penalty += target.jumped ? 0.22 * step : 0.08 * step;
        }

        const double score =
          std::pow(reprojection_error / kReprojectionGate, 2) +
          std::pow(los_yaw_error / kYawGate, 2) +
          std::pow(pitch_error / kPitchGate, 2) +
          std::pow(distance_error / kDistanceGate, 2) +
          std::pow(armor_yaw_error / kArmorYawGate, 2) +
          std::pow(xy_error / kXYGate, 2) +
          std::pow(z_error / kZGate, 2) + continuity_penalty;

        if (score >= best_match.score) continue;

        best_match.armor_it = it;
        best_match.id = id;
        best_match.offset = offset;
        best_match.physical_id = mapped_target.physical_armor_id(id);
        best_match.score = score;
        best_match.reprojection_error = reprojection_error;
        best_match.xy_error = xy_error;
        best_match.z_error = z_error;
        best_match.valid = true;
      }
    }
  }

  return best_match;
}

bool accept_outpost_match(const OutpostMatchResult & match)
{
  return
    match.valid && std::isfinite(match.score) && std::isfinite(match.reprojection_error) &&
    match.reprojection_error < 90.0 && match.xy_error < 0.40 && match.z_error < 0.20 &&
    match.score < 36.0;
}
}  // namespace

Tracker::Tracker(const std::string & config_path, Solver & solver)
: config_path_(tools::resolve_config_path_string(config_path)),
  solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = tools::load(config_path_);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
  outpost_radius_ = tools::read_or<double>(yaml, "outpost_radius", 0.2765);
  outpost_spin_speed_lock_ = tools::read_or<double>(yaml, "outpost_spin_speed_lock", 2.51);
  outpost_fixed_center_rotation_model_ =
    tools::read_or<bool>(yaml, "outpost_fixed_center_rotation_model", true);
  outpost_armor_z_offsets_ =
    tools::read_or<std::vector<double>>(yaml, "outpost_armor_z_offsets", {0.0, -0.102, 0.102});
  if (outpost_armor_z_offsets_.size() != 3) {
    tools::logger()->warn(
      "[Tracker] outpost_armor_z_offsets size {} invalid, fallback to default 3-board model",
      outpost_armor_z_offsets_.size());
    outpost_armor_z_offsets_ = {0.0, -0.102, 0.102};
  }
  runtime_params_version_ = tools::runtime_params::version(config_path_);
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  refresh_runtime_params_if_needed();

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->info("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  refresh_runtime_params_if_needed();

  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

void Tracker::refresh_runtime_params_if_needed()
{
  const auto current_version = tools::runtime_params::version(config_path_);
  if (current_version == 0 || current_version == runtime_params_version_) return;

  const auto old_enemy_color = enemy_color_;
  const auto old_outpost_radius = outpost_radius_;
  const auto old_outpost_spin_speed_lock = outpost_spin_speed_lock_;
  const auto old_outpost_fixed_center_rotation_model = outpost_fixed_center_rotation_model_;
  const auto old_outpost_armor_z_offsets = outpost_armor_z_offsets_;

  enemy_color_ =
    (tools::runtime_params::get_string(config_path_, "enemy_color") == "red") ?
    Color::red : Color::blue;
  min_detect_count_ = tools::runtime_params::get_int(config_path_, "min_detect_count");
  max_temp_lost_count_ = tools::runtime_params::get_int(config_path_, "max_temp_lost_count");
  outpost_max_temp_lost_count_ =
    tools::runtime_params::get_int(config_path_, "outpost_max_temp_lost_count");
  normal_temp_lost_count_ = max_temp_lost_count_;
  outpost_radius_ = tools::runtime_params::get_double(config_path_, "outpost_radius");
  outpost_spin_speed_lock_ =
    tools::runtime_params::get_double(config_path_, "outpost_spin_speed_lock");
  outpost_fixed_center_rotation_model_ =
    tools::runtime_params::get_bool(config_path_, "outpost_fixed_center_rotation_model");
  outpost_armor_z_offsets_ =
    tools::runtime_params::get_number_array(config_path_, "outpost_armor_z_offsets");

  if (outpost_armor_z_offsets_.size() != 3) {
    tools::logger()->warn(
      "[Tracker] outpost_armor_z_offsets size {} invalid, fallback to default 3-board model",
      outpost_armor_z_offsets_.size());
    outpost_armor_z_offsets_ = {0.0, -0.102, 0.102};
  }

  const bool reset_for_consistency =
    old_enemy_color != enemy_color_ ||
    old_outpost_radius != outpost_radius_ ||
    old_outpost_spin_speed_lock != outpost_spin_speed_lock_ ||
    old_outpost_fixed_center_rotation_model != outpost_fixed_center_rotation_model_ ||
    old_outpost_armor_z_offsets != outpost_armor_z_offsets_;
  if (reset_for_consistency && state_ != "lost") {
    state_ = "lost";
    pre_state_ = "lost";
    detect_count_ = 0;
    temp_lost_count_ = 0;
    tools::logger()->info("[Tracker] runtime params changed, tracker reset for consistency");
  }

  runtime_params_version_ = current_version;
  tools::logger()->info("[Tracker] runtime params updated to v{}", current_version);
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(
      armor, t, outpost_radius_, 3, P0_dig, outpost_armor_z_offsets_,
      outpost_fixed_center_rotation_model_, outpost_spin_speed_lock_);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  int found_count = 0;
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
  }

  if (target_.name == ArmorName::outpost) {
    target_.tracker_debug_candidate_count = found_count;
    target_.tracker_debug_match_valid = false;
    target_.tracker_debug_match_id = -1;
    target_.tracker_debug_match_score = -1.0;
    target_.tracker_debug_reprojection_px = -1.0;
    target_.tracker_debug_xy_error_m = -1.0;
    target_.tracker_debug_z_error_m = -1.0;
  }

  if (found_count == 0) return false;

  if (target_.name == ArmorName::outpost) {
    const auto best_match = select_best_outpost_match(target_, solver_, armors);
    if (best_match.valid) {
      target_.tracker_debug_match_id = best_match.id;
      target_.tracker_debug_match_score = best_match.score;
      target_.tracker_debug_reprojection_px = best_match.reprojection_error;
      target_.tracker_debug_xy_error_m = best_match.xy_error;
      target_.tracker_debug_z_error_m = best_match.z_error;
    }

    const bool accept = accept_outpost_match(best_match);
    target_.tracker_debug_match_valid = accept;
    if (!accept) {
      if (best_match.valid) {
        tools::logger()->debug(
          "[Tracker] reject outpost match: id={} phys={} offset={} score={:.2f}, reproj={:.2f}, xy={:.3f}, z={:.3f}",
          best_match.id, best_match.physical_id, best_match.offset, best_match.score,
          best_match.reprojection_error, best_match.xy_error, best_match.z_error);
      }
      return false;
    }

    target_.set_armor_id_offset(best_match.offset, target_.last_id);
    target_.update(*best_match.armor_it, best_match.id);
    return true;
  }

  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x
    )
      continue;

    solver_.solve(armor);

    target_.update(armor);
  }

  return true;
}

}  // namespace auto_aim
