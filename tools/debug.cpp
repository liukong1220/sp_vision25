#include "tools/debug.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace
{
constexpr double kGravity = 9.7833;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;
constexpr double kGimbalRadiansSanityThreshold = 4.0 * M_PI;

double bullet_height(double horizontal_dist, double bullet_speed, double launch_pitch)
{
  const double cos_pitch = std::cos(launch_pitch);
  if (std::abs(cos_pitch) < 1e-5 || bullet_speed <= 1e-5) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return horizontal_dist * std::tan(launch_pitch) -
         kGravity * horizontal_dist * horizontal_dist /
           (2.0 * bullet_speed * bullet_speed * cos_pitch * cos_pitch);
}

void draw_outlined_text(
  cv::Mat & img, const std::string & text, const cv::Point & org, double scale,
  const cv::Scalar & color, int thickness = 1)
{
  cv::putText(
    img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0),
    thickness + 2);
  cv::putText(
    img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
}

tools::debug::NormalizedAngle normalize_angle_value(
  double raw, bool source_is_degree)
{
  tools::debug::NormalizedAngle value;
  value.raw = raw;
  if (source_is_degree) {
    value.deg = raw;
    value.rad = tools::debug::deg2rad(raw);
  } else {
    value.rad = raw;
    value.deg = tools::debug::rad2deg(raw);
  }
  return value;
}

tools::debug::BallisticDiagnostic build_ballistic_diagnostic_impl(
  bool control, bool fire, double command_yaw, double command_pitch,
  const Eigen::Vector4d & aim_xyza, auto_aim::ArmorType armor_type, double bullet_speed,
  double yaw_offset, double pitch_offset)
{
  tools::debug::BallisticDiagnostic diag;
  if (!control) return diag;

  diag.raw_bullet_speed = bullet_speed;
  diag.bullet_speed_fallback = bullet_speed < 10.0 || bullet_speed > 25.0;
  if (diag.bullet_speed_fallback) bullet_speed = 22.0;

  diag.valid = true;
  diag.fire = fire;
  diag.armor_type = armor_type;
  diag.bullet_speed = bullet_speed;
  diag.yaw_offset = yaw_offset;
  diag.pitch_offset = pitch_offset;
  diag.target_xyz = aim_xyza.head<3>();
  diag.target_dist_xy = std::hypot(diag.target_xyz.x(), diag.target_xyz.y());
  diag.target_dist_3d = diag.target_xyz.norm();
  diag.target_geo_yaw = std::atan2(diag.target_xyz.y(), diag.target_xyz.x());

  const auto pure_traj =
    tools::Trajectory(bullet_speed, diag.target_dist_xy, diag.target_xyz.z());
  diag.unsolvable = pure_traj.unsolvable;
  if (diag.unsolvable) return diag;

  diag.target_geo_pitch = -pure_traj.pitch;
  diag.required_cmd_yaw = tools::limit_rad(diag.target_geo_yaw + yaw_offset);
  diag.required_cmd_pitch = -(pure_traj.pitch + pitch_offset);
  diag.command_yaw = command_yaw;
  diag.command_pitch = command_pitch;
  diag.yaw_residual = tools::limit_rad(command_yaw - diag.required_cmd_yaw);
  diag.pitch_residual = command_pitch - diag.required_cmd_pitch;

  const Eigen::Vector2d target_xy(diag.target_xyz.x(), diag.target_xyz.y());
  const Eigen::Vector2d shot_dir(std::cos(command_yaw), std::sin(command_yaw));
  const double along = target_xy.dot(shot_dir);
  const double lateral = target_xy.x() * shot_dir.y() - target_xy.y() * shot_dir.x();
  diag.lateral_error = along >= 0.0 ? lateral : target_xy.norm();

  const double bullet_z =
    bullet_height(diag.target_dist_xy, bullet_speed, -command_pitch);
  diag.vertical_error =
    std::isfinite(bullet_z) ? (bullet_z - diag.target_xyz.z()) : 1e9;
  diag.total_error = std::hypot(diag.lateral_error, diag.vertical_error);

  const double half_width =
    (armor_type == auto_aim::ArmorType::big ? kBigArmorWidth : kSmallArmorWidth) /
    2.0;
  const double half_height = kLightbarLength / 2.0;
  diag.hit =
    along >= 0.0 && std::abs(diag.lateral_error) <= half_width &&
    std::abs(diag.vertical_error) <= half_height;
  return diag;
}
}  // namespace

namespace tools::debug
{

double rad2deg(double rad)
{
  return rad * 180.0 / M_PI;
}

double deg2rad(double deg)
{
  return deg * M_PI / 180.0;
}

GimbalStateUnitMode parse_gimbal_state_unit_mode(const std::string & unit)
{
  if (unit == "deg" || unit == "degree" || unit == "degrees") {
    return GimbalStateUnitMode::deg;
  }
  if (unit == "rad" || unit == "radian" || unit == "radians") {
    return GimbalStateUnitMode::rad;
  }
  return GimbalStateUnitMode::auto_detect;
}

NormalizedGimbalState normalize_gimbal_state(
  const io::GimbalState & gs, GimbalStateUnitMode unit_mode)
{
  bool source_is_degree = false;
  switch (unit_mode) {
    case GimbalStateUnitMode::deg:
      source_is_degree = true;
      break;
    case GimbalStateUnitMode::rad:
      source_is_degree = false;
      break;
    case GimbalStateUnitMode::auto_detect:
    default:
      source_is_degree =
        std::abs(gs.yaw) > kGimbalRadiansSanityThreshold ||
        std::abs(gs.pitch) > kGimbalRadiansSanityThreshold ||
        std::abs(gs.yaw_vel) > kGimbalRadiansSanityThreshold ||
        std::abs(gs.pitch_vel) > kGimbalRadiansSanityThreshold;
      break;
  }

  return {
    source_is_degree,
    normalize_angle_value(gs.yaw, source_is_degree),
    normalize_angle_value(gs.yaw_vel, source_is_degree),
    normalize_angle_value(gs.pitch, source_is_degree),
    normalize_angle_value(gs.pitch_vel, source_is_degree),
  };
}

bool has_cli_option(int argc, char * argv[], const std::string & long_option)
{
  const std::string exact = "--" + long_option;
  const std::string prefix = exact + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == exact || arg.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

int64_t unix_time_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

std::string armor_type_to_string(auto_aim::ArmorType armor_type)
{
  const auto index = static_cast<size_t>(armor_type);
  if (index < auto_aim::ARMOR_TYPES.size()) return auto_aim::ARMOR_TYPES[index];
  return "unknown";
}

std::string armor_name_to_string(auto_aim::ArmorName armor_name)
{
  const auto index = static_cast<size_t>(armor_name);
  if (index < auto_aim::ARMOR_NAMES.size()) return auto_aim::ARMOR_NAMES[index];
  return "unknown";
}

std::string spin_direction_to_string(double yaw_rate_rad_s, double threshold)
{
  if (yaw_rate_rad_s > threshold) return "CCW(+)";
  if (yaw_rate_rad_s < -threshold) return "CW(-)";
  return "STEADY";
}

int spin_direction_sign(double yaw_rate_rad_s, double threshold)
{
  if (yaw_rate_rad_s > threshold) return 1;
  if (yaw_rate_rad_s < -threshold) return -1;
  return 0;
}

BallisticDiagnostic build_ballistic_diagnostic(
  const auto_aim::Plan & plan, const Eigen::Vector4d & aim_xyza,
  auto_aim::ArmorType armor_type, double bullet_speed, double yaw_offset,
  double pitch_offset)
{
  return build_ballistic_diagnostic_impl(
    plan.control, plan.fire, plan.yaw, plan.pitch, aim_xyza, armor_type,
    bullet_speed, yaw_offset, pitch_offset);
}

BallisticDiagnostic build_ballistic_diagnostic(
  const io::Command & command, const Eigen::Vector4d & aim_xyza,
  auto_aim::ArmorType armor_type, double bullet_speed, double yaw_offset,
  double pitch_offset)
{
  return build_ballistic_diagnostic_impl(
    command.control, command.shoot, command.yaw, command.pitch, aim_xyza,
    armor_type, bullet_speed, yaw_offset, pitch_offset);
}

nlohmann::json ballistic_to_json(const BallisticDiagnostic & diag)
{
  nlohmann::json data;
  data["valid"] = diag.valid;
  data["unsolvable"] = diag.unsolvable;
  data["hit"] = diag.hit;
  data["fire"] = diag.fire;
  data["bullet_speed_fallback"] = diag.bullet_speed_fallback;
  data["armor_type"] = armor_type_to_string(diag.armor_type);
  data["bullet_speed_raw_mps"] = diag.raw_bullet_speed;
  data["bullet_speed_effective_mps"] = diag.bullet_speed;
  data["bullet_speed_mps"] = diag.bullet_speed;
  data["yaw_offset_deg"] = rad2deg(diag.yaw_offset);
  data["pitch_offset_deg"] = rad2deg(diag.pitch_offset);
  data["target_x_m"] = diag.target_xyz.x();
  data["target_y_m"] = diag.target_xyz.y();
  data["target_z_m"] = diag.target_xyz.z();
  data["target_dist_xy_m"] = diag.target_dist_xy;
  data["target_dist_3d_m"] = diag.target_dist_3d;
  data["target_geo_yaw_deg"] = rad2deg(diag.target_geo_yaw);
  data["target_geo_pitch_deg"] = rad2deg(diag.target_geo_pitch);
  data["required_cmd_yaw_deg"] = rad2deg(diag.required_cmd_yaw);
  data["required_cmd_pitch_deg"] = rad2deg(diag.required_cmd_pitch);
  data["command_yaw_deg"] = rad2deg(diag.command_yaw);
  data["command_pitch_deg"] = rad2deg(diag.command_pitch);
  data["yaw_residual_deg"] = rad2deg(diag.yaw_residual);
  data["pitch_residual_deg"] = rad2deg(diag.pitch_residual);
  data["lateral_error_mm"] = diag.lateral_error * 1000.0;
  data["vertical_error_mm"] = diag.vertical_error * 1000.0;
  data["total_error_mm"] = diag.total_error * 1000.0;
  return data;
}

nlohmann::json estimator_to_json(const auto_aim::Target * target)
{
  nlohmann::json data;
  if (target == nullptr) {
    data["nis"] = nullptr;
    data["nis_gate"] = nullptr;
    data["update_accepted"] = nullptr;
    data["recent_reject_rate"] = nullptr;
    data["accepted_updates"] = nullptr;
    data["rejected_updates"] = nullptr;
    data["car_yaw_deg"] = nullptr;
    data["car_pitch_deg"] = nullptr;
    data["car_roll_deg"] = nullptr;
    data["radius_1_m"] = nullptr;
    data["radius_2_m"] = nullptr;
    return data;
  }

  const auto & diagnostics = target->ekf().data;
  const auto read = [&](const char * key, double fallback = 0.0) {
    const auto it = diagnostics.find(key);
    return it == diagnostics.end() ? fallback : it->second;
  };
  data["nis"] = read("nis");
  data["nis_gate"] = target->nis_gate();
  data["update_accepted"] = read("update_accepted", 1.0) > 0.5;
  data["recent_reject_rate"] = read("recent_nis_failures");
  data["accepted_updates"] = read("accepted_updates");
  data["rejected_updates"] = read("rejected_updates");
  data["measurement_dim"] = read("measurement_dim");
  data["uvl_left_angle_rad"] = read("uvl_left_angle");
  data["uvl_left_center_u_px"] = read("uvl_left_center_u");
  data["uvl_left_center_v_px"] = read("uvl_left_center_v");
  data["uvl_left_length_px"] = read("uvl_left_length");
  data["uvl_right_angle_rad"] = read("uvl_right_angle");
  data["uvl_right_center_u_px"] = read("uvl_right_center_u");
  data["uvl_right_center_v_px"] = read("uvl_right_center_v");
  data["uvl_right_length_px"] = read("uvl_right_length");
  const Eigen::Vector3d car_rpy = target->car_rpy();
  data["car_yaw_deg"] = rad2deg(car_rpy[0]);
  data["car_pitch_deg"] = rad2deg(car_rpy[1]);
  data["car_roll_deg"] = rad2deg(car_rpy[2]);
  data["radius_1_m"] = target->radius(0);
  data["radius_2_m"] = target->radius(1);
  return data;
}

nlohmann::json mpc_to_json(const auto_aim::Planner & planner)
{
  return {
    {"yaw_solver_status", planner.debug_yaw_solver_status},
    {"pitch_solver_status", planner.debug_pitch_solver_status},
    {"yaw_solver_iterations", planner.debug_yaw_solver_iterations},
    {"pitch_solver_iterations", planner.debug_pitch_solver_iterations},
    {"mpc_converged",
     planner.debug_yaw_solver_status == 0 && planner.debug_pitch_solver_status == 0},
  };
}

void draw_ballistic_panel(cv::Mat & panel, const BallisticDiagnostic & diag)
{
  panel = cv::Scalar(24, 28, 34);
  const cv::Rect side_rect(35, 40, 360, 230);
  const cv::Rect top_rect(445, 40, 360, 230);
  const cv::Rect text_rect(25, 295, 790, 155);

  cv::rectangle(panel, side_rect, cv::Scalar(70, 75, 85), 1);
  cv::rectangle(panel, top_rect, cv::Scalar(70, 75, 85), 1);
  cv::rectangle(panel, text_rect, cv::Scalar(70, 75, 85), 1);
  draw_outlined_text(
    panel, "Ballistic Debug", {30, 24}, 0.75, cv::Scalar(255, 255, 255), 2);

  if (!diag.valid) {
    draw_outlined_text(
      panel, "No valid target / plan", {250, 210}, 0.9,
      cv::Scalar(120, 220, 255), 2);
    return;
  }

  if (diag.unsolvable) {
    draw_outlined_text(
      panel, "Trajectory Unsolvable", {215, 185}, 0.9,
      cv::Scalar(0, 80, 255), 2);
    draw_outlined_text(
      panel,
      fmt::format(
        "speed raw/use: {:.2f} / {:.2f} m/s  d/z: {:.2f} / {:.2f} m",
        diag.raw_bullet_speed, diag.bullet_speed, diag.target_dist_xy, diag.target_xyz.z()),
      {140, 225}, 0.6, cv::Scalar(220, 220, 220), 1);
    draw_outlined_text(
      panel,
      fmt::format(
        "offset yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.yaw_offset),
        rad2deg(diag.pitch_offset)),
      {190, 260}, 0.55, cv::Scalar(220, 220, 220), 1);
    return;
  }

  auto map_to_rect = [](double x, double y, const cv::Rect & rect, double min_x,
                         double max_x, double min_y, double max_y) {
    const double nx =
      max_x - min_x > 1e-6 ? (x - min_x) / (max_x - min_x) : 0.0;
    const double ny =
      max_y - min_y > 1e-6 ? (y - min_y) / (max_y - min_y) : 0.0;
    const int px =
      rect.x + static_cast<int>(std::clamp(nx, 0.0, 1.0) * rect.width);
    const int py = rect.y + rect.height -
      static_cast<int>(std::clamp(ny, 0.0, 1.0) * rect.height);
    return cv::Point(px, py);
  };

  draw_outlined_text(
    panel, "Side View (d-z)", {side_rect.x + 10, side_rect.y - 10}, 0.5,
    cv::Scalar(220, 220, 220));
  draw_outlined_text(
    panel, "Top View (x-y)", {top_rect.x + 10, top_rect.y - 10}, 0.5,
    cv::Scalar(220, 220, 220));

  const double max_dist = std::max(1.0, diag.target_dist_xy * 1.2);
  const double current_z_at_target =
    bullet_height(diag.target_dist_xy, diag.bullet_speed, -diag.command_pitch);
  const double min_z = std::min(
    {-0.15, diag.target_xyz.z() - 0.1,
     std::isfinite(current_z_at_target) ? current_z_at_target - 0.1 : -0.15});
  const double max_z = std::max(
    {0.25, diag.target_xyz.z() + 0.15,
     std::isfinite(current_z_at_target) ? current_z_at_target + 0.1 : 0.25});

  const int sample_num = 100;
  std::vector<cv::Point> ideal_curve;
  std::vector<cv::Point> cmd_curve;
  ideal_curve.reserve(sample_num);
  cmd_curve.reserve(sample_num);
  for (int i = 0; i < sample_num; ++i) {
    const double d =
      max_dist * static_cast<double>(i) / static_cast<double>(sample_num - 1);
    const double ideal_z =
      bullet_height(d, diag.bullet_speed, -diag.target_geo_pitch);
    const double cmd_z =
      bullet_height(d, diag.bullet_speed, -diag.command_pitch);
    if (std::isfinite(ideal_z)) {
      ideal_curve.push_back(
        map_to_rect(d, ideal_z, side_rect, 0.0, max_dist, min_z, max_z));
    }
    if (std::isfinite(cmd_z)) {
      cmd_curve.push_back(
        map_to_rect(d, cmd_z, side_rect, 0.0, max_dist, min_z, max_z));
    }
  }
  for (size_t i = 1; i < ideal_curve.size(); ++i) {
    cv::line(
      panel, ideal_curve[i - 1], ideal_curve[i], cv::Scalar(60, 200, 120), 2,
      cv::LINE_AA);
  }
  for (size_t i = 1; i < cmd_curve.size(); ++i) {
    cv::line(
      panel, cmd_curve[i - 1], cmd_curve[i], cv::Scalar(0, 220, 255), 2,
      cv::LINE_AA);
  }

  const auto target_side_pt = map_to_rect(
    diag.target_dist_xy, diag.target_xyz.z(), side_rect, 0.0, max_dist, min_z,
    max_z);
  cv::circle(panel, target_side_pt, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  if (std::isfinite(current_z_at_target)) {
    const auto cmd_hit_pt = map_to_rect(
      diag.target_dist_xy, current_z_at_target, side_rect, 0.0, max_dist,
      min_z, max_z);
    cv::circle(
      panel, cmd_hit_pt, 5, cv::Scalar(0, 220, 255), -1, cv::LINE_AA);
  }

  const double max_xy = std::max(
    {1.0, std::abs(diag.target_xyz.x()) * 1.25,
     std::abs(diag.target_xyz.y()) * 1.25});
  const auto origin_top_pt =
    map_to_rect(0.0, 0.0, top_rect, -max_xy, max_xy, -max_xy, max_xy);
  const auto target_top_pt = map_to_rect(
    diag.target_xyz.x(), diag.target_xyz.y(), top_rect, -max_xy, max_xy,
    -max_xy, max_xy);
  cv::arrowedLine(
    panel, origin_top_pt, target_top_pt, cv::Scalar(60, 200, 120), 2,
    cv::LINE_AA, 0, 0.06);

  const double ray_len = std::max(1.0, diag.target_dist_xy * 1.1);
  const cv::Point cmd_ray_pt = map_to_rect(
    ray_len * std::cos(diag.command_yaw), ray_len * std::sin(diag.command_yaw),
    top_rect, -max_xy, max_xy, -max_xy, max_xy);
  cv::arrowedLine(
    panel, origin_top_pt, cmd_ray_pt, cv::Scalar(0, 220, 255), 2, cv::LINE_AA,
    0, 0.06);
  cv::circle(panel, target_top_pt, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

  const cv::Scalar verdict_color =
    diag.hit ? cv::Scalar(60, 220, 120) : cv::Scalar(0, 80, 255);
  draw_outlined_text(
    panel, diag.hit ? "Verdict: HIT" : "Verdict: MISS",
    {text_rect.x + 15, text_rect.y + 28}, 0.75, verdict_color, 2);
  draw_outlined_text(
    panel,
    fmt::format(
      "plan.fire: {}  speed raw/use: {:.2f} / {:.2f} m/s{}",
      diag.fire ? "true" : "false", diag.raw_bullet_speed, diag.bullet_speed,
      diag.bullet_speed_fallback ? "  [fallback]" : ""),
    {text_rect.x + 15, text_rect.y + 58}, 0.52, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel,
    fmt::format(
      "offset yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.yaw_offset),
      rad2deg(diag.pitch_offset)),
    {text_rect.x + 15, text_rect.y + 84}, 0.52, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel,
    fmt::format(
      "target xyz: ({:.2f}, {:.2f}, {:.2f})  d_xy: {:.2f}  d_3d: {:.2f}",
      diag.target_xyz.x(), diag.target_xyz.y(), diag.target_xyz.z(),
      diag.target_dist_xy, diag.target_dist_3d),
    {text_rect.x + 15, text_rect.y + 110}, 0.50,
    cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel,
    fmt::format(
      "geo yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.target_geo_yaw),
      rad2deg(diag.target_geo_pitch)),
    {text_rect.x + 15, text_rect.y + 136}, 0.50, cv::Scalar(60, 200, 120));
  draw_outlined_text(
    panel,
    fmt::format(
      "cmd-ref yaw/pitch: {:.2f} / {:.2f} deg",
      rad2deg(diag.required_cmd_yaw), rad2deg(diag.required_cmd_pitch)),
    {text_rect.x + 15, text_rect.y + 162}, 0.50, cv::Scalar(100, 180, 255));
  draw_outlined_text(
    panel,
    fmt::format(
      "plan yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.command_yaw),
      rad2deg(diag.command_pitch)),
    {text_rect.x + 15, text_rect.y + 188}, 0.50, cv::Scalar(0, 220, 255));
  draw_outlined_text(
    panel,
    fmt::format(
      "yaw/pitch residual: {:.3f} / {:.3f} deg",
      rad2deg(diag.yaw_residual), rad2deg(diag.pitch_residual)),
    {text_rect.x + 390, text_rect.y + 58}, 0.50,
    cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("lateral miss: {:.1f} mm", diag.lateral_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 84}, 0.50,
    cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel,
    fmt::format("vertical miss: {:.1f} mm", diag.vertical_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 110}, 0.50,
    cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("total miss: {:.1f} mm", diag.total_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 136}, 0.50,
    cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel,
    fmt::format(
      "armor size: {} x {:.0f}mm",
      diag.armor_type == auto_aim::ArmorType::big ? "230" : "135",
      kLightbarLength * 1000.0),
    {text_rect.x + 390, text_rect.y + 162}, 0.50,
    cv::Scalar(230, 230, 230));
}

}  // namespace tools::debug
