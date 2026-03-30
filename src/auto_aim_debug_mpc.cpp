#include <fmt/core.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include "tasks/auto_aim/target.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/trajectory.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/web_debugger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数yaml配置文件路径 }"
  "{show-local     | false                  | 保留本地OpenCV调试窗口(显式传参时覆盖yaml) }"
  "{disable-web    | false                  | 禁用内置网页调试器(显式传参时覆盖yaml) }"
  "{web-host       | 0.0.0.0                | 网页调试器绑定地址(显式传参时覆盖yaml) }"
  "{web-port       | 8090                   | 网页调试器端口(显式传参时覆盖yaml) }"
  "{web-fps        | 8.0                    | 网页图像刷新帧率(显式传参时覆盖yaml) }"
  "{web-scale      | 0.7                    | 网页图像缩放系数(显式传参时覆盖yaml) }"
  "{web-jpeg-quality | 70                   | 网页JPEG质量(30-95, 显式传参时覆盖yaml) }"
  "{web-client-ttl-ms | 2000                | 最近访问多久内继续渲染网页帧(显式传参时覆盖yaml) }";

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

double deg2rad(double deg) {
  return deg * M_PI / 180.0;
}

namespace
{
constexpr double kGravity = 9.7833;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;
constexpr double kGimbalRadiansSanityThreshold = 4.0 * M_PI;

struct BallisticDiagnostic
{
  bool valid = false;
  bool unsolvable = false;
  bool hit = false;
  bool fire = false;
  auto_aim::ArmorType armor_type = auto_aim::ArmorType::small;
  double bullet_speed = 0.0;
  double yaw_offset = 0.0;
  double pitch_offset = 0.0;
  Eigen::Vector3d target_xyz = Eigen::Vector3d::Zero();
  double target_dist_xy = 0.0;
  double target_dist_3d = 0.0;
  double target_geo_yaw = 0.0;
  double target_geo_pitch = 0.0;
  double required_cmd_yaw = 0.0;
  double required_cmd_pitch = 0.0;
  double command_yaw = 0.0;
  double command_pitch = 0.0;
  double yaw_residual = 0.0;
  double pitch_residual = 0.0;
  double lateral_error = 0.0;
  double vertical_error = 0.0;
  double total_error = 0.0;
};

struct NormalizedAngle
{
  double raw = 0.0;
  double rad = 0.0;
  double deg = 0.0;
};

enum class GimbalStateUnitMode
{
  auto_detect,
  rad,
  deg,
};

struct NormalizedGimbalState
{
  bool source_is_degree = false;
  NormalizedAngle yaw;
  NormalizedAngle yaw_vel;
  NormalizedAngle pitch;
  NormalizedAngle pitch_vel;
};

NormalizedAngle normalize_angle_value(double raw, bool source_is_degree)
{
  NormalizedAngle value;
  value.raw = raw;
  if (source_is_degree) {
    value.deg = raw;
    value.rad = deg2rad(raw);
  } else {
    value.rad = raw;
    value.deg = rad2deg(raw);
  }
  return value;
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

double bullet_height(double horizontal_dist, double bullet_speed, double launch_pitch)
{
  const double cos_pitch = std::cos(launch_pitch);
  if (std::abs(cos_pitch) < 1e-5 || bullet_speed <= 1e-5) return std::numeric_limits<double>::quiet_NaN();

  return horizontal_dist * std::tan(launch_pitch) -
         kGravity * horizontal_dist * horizontal_dist /
           (2.0 * bullet_speed * bullet_speed * cos_pitch * cos_pitch);
}

void draw_outlined_text(
  cv::Mat & img, const std::string & text, const cv::Point & org, double scale,
  const cv::Scalar & color, int thickness = 1)
{
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thickness + 2);
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
}

BallisticDiagnostic build_ballistic_diagnostic(
  const auto_aim::Plan & plan, const Eigen::Vector4d & aim_xyza, auto_aim::ArmorType armor_type,
  double bullet_speed, double yaw_offset, double pitch_offset)
{
  BallisticDiagnostic diag;
  if (!plan.control) return diag;

  if (bullet_speed < 10.0 || bullet_speed > 25.0) bullet_speed = 22.0;

  diag.valid = true;
  diag.fire = plan.fire;
  diag.armor_type = armor_type;
  diag.bullet_speed = bullet_speed;
  diag.yaw_offset = yaw_offset;
  diag.pitch_offset = pitch_offset;
  diag.target_xyz = aim_xyza.head<3>();
  diag.target_dist_xy = std::hypot(diag.target_xyz.x(), diag.target_xyz.y());
  diag.target_dist_3d = diag.target_xyz.norm();
  diag.target_geo_yaw = std::atan2(diag.target_xyz.y(), diag.target_xyz.x());

  const auto pure_traj = tools::Trajectory(bullet_speed, diag.target_dist_xy, diag.target_xyz.z());
  diag.unsolvable = pure_traj.unsolvable;
  if (diag.unsolvable) return diag;

  diag.target_geo_pitch = -pure_traj.pitch;
  diag.required_cmd_yaw = tools::limit_rad(diag.target_geo_yaw + yaw_offset);
  diag.required_cmd_pitch = -(pure_traj.pitch + pitch_offset);
  diag.command_yaw = plan.yaw;
  diag.command_pitch = plan.pitch;
  diag.yaw_residual = tools::limit_rad(plan.yaw - diag.required_cmd_yaw);
  diag.pitch_residual = plan.pitch - diag.required_cmd_pitch;

  const Eigen::Vector2d target_xy(diag.target_xyz.x(), diag.target_xyz.y());
  const Eigen::Vector2d shot_dir(std::cos(plan.yaw), std::sin(plan.yaw));
  const double along = target_xy.dot(shot_dir);
  const double lateral = target_xy.x() * shot_dir.y() - target_xy.y() * shot_dir.x();
  diag.lateral_error = (along >= 0.0) ? lateral : target_xy.norm();

  const double bullet_z = bullet_height(diag.target_dist_xy, bullet_speed, -plan.pitch);
  diag.vertical_error = std::isfinite(bullet_z) ? (bullet_z - diag.target_xyz.z()) : 1e9;
  diag.total_error = std::hypot(diag.lateral_error, diag.vertical_error);

  const double half_width =
    (armor_type == auto_aim::ArmorType::big ? kBigArmorWidth : kSmallArmorWidth) / 2.0;
  const double half_height = kLightbarLength / 2.0;
  diag.hit =
    along >= 0.0 && std::abs(diag.lateral_error) <= half_width &&
    std::abs(diag.vertical_error) <= half_height;
  return diag;
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
  draw_outlined_text(panel, "Ballistic Debug", {30, 24}, 0.75, cv::Scalar(255, 255, 255), 2);

  if (!diag.valid) {
    draw_outlined_text(panel, "No valid target / plan", {250, 210}, 0.9, cv::Scalar(120, 220, 255), 2);
    return;
  }

  if (diag.unsolvable) {
    draw_outlined_text(panel, "Trajectory Unsolvable", {215, 185}, 0.9, cv::Scalar(0, 80, 255), 2);
    draw_outlined_text(
      panel, fmt::format("speed: {:.2f} m/s  target d/z: {:.2f} / {:.2f} m",
                         diag.bullet_speed, diag.target_dist_xy, diag.target_xyz.z()),
      {140, 225}, 0.6, cv::Scalar(220, 220, 220), 1);
    draw_outlined_text(
      panel, fmt::format("offset yaw/pitch: {:.2f} / {:.2f} deg",
                         rad2deg(diag.yaw_offset), rad2deg(diag.pitch_offset)),
      {190, 260}, 0.55, cv::Scalar(220, 220, 220), 1);
    return;
  }

  auto map_to_rect =
    [](double x, double y, const cv::Rect & rect, double min_x, double max_x, double min_y, double max_y) {
      const double nx = (max_x - min_x > 1e-6) ? (x - min_x) / (max_x - min_x) : 0.0;
      const double ny = (max_y - min_y > 1e-6) ? (y - min_y) / (max_y - min_y) : 0.0;
      const int px = rect.x + static_cast<int>(std::clamp(nx, 0.0, 1.0) * rect.width);
      const int py = rect.y + rect.height - static_cast<int>(std::clamp(ny, 0.0, 1.0) * rect.height);
      return cv::Point(px, py);
    };

  draw_outlined_text(panel, "Side View (d-z)", {side_rect.x + 10, side_rect.y - 10}, 0.5, cv::Scalar(220, 220, 220));
  draw_outlined_text(panel, "Top View (x-y)", {top_rect.x + 10, top_rect.y - 10}, 0.5, cv::Scalar(220, 220, 220));

  const double max_dist = std::max(1.0, diag.target_dist_xy * 1.2);
  const double current_z_at_target = bullet_height(diag.target_dist_xy, diag.bullet_speed, -diag.command_pitch);
  const double min_z =
    std::min({-0.15, diag.target_xyz.z() - 0.1, std::isfinite(current_z_at_target) ? current_z_at_target - 0.1 : -0.15});
  const double max_z =
    std::max({0.25, diag.target_xyz.z() + 0.15, std::isfinite(current_z_at_target) ? current_z_at_target + 0.1 : 0.25});

  const int sample_num = 100;
  std::vector<cv::Point> ideal_curve;
  std::vector<cv::Point> cmd_curve;
  ideal_curve.reserve(sample_num);
  cmd_curve.reserve(sample_num);
  for (int i = 0; i < sample_num; ++i) {
    const double d = max_dist * static_cast<double>(i) / static_cast<double>(sample_num - 1);
    const double ideal_z = bullet_height(d, diag.bullet_speed, -diag.target_geo_pitch);
    const double cmd_z = bullet_height(d, diag.bullet_speed, -diag.command_pitch);
    if (std::isfinite(ideal_z))
      ideal_curve.push_back(map_to_rect(d, ideal_z, side_rect, 0.0, max_dist, min_z, max_z));
    if (std::isfinite(cmd_z))
      cmd_curve.push_back(map_to_rect(d, cmd_z, side_rect, 0.0, max_dist, min_z, max_z));
  }
  for (size_t i = 1; i < ideal_curve.size(); ++i) {
    cv::line(panel, ideal_curve[i - 1], ideal_curve[i], cv::Scalar(60, 200, 120), 2, cv::LINE_AA);
  }
  for (size_t i = 1; i < cmd_curve.size(); ++i) {
    cv::line(panel, cmd_curve[i - 1], cmd_curve[i], cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
  }

  const auto target_side_pt =
    map_to_rect(diag.target_dist_xy, diag.target_xyz.z(), side_rect, 0.0, max_dist, min_z, max_z);
  cv::circle(panel, target_side_pt, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  if (std::isfinite(current_z_at_target)) {
    const auto cmd_hit_pt =
      map_to_rect(diag.target_dist_xy, current_z_at_target, side_rect, 0.0, max_dist, min_z, max_z);
    cv::circle(panel, cmd_hit_pt, 5, cv::Scalar(0, 220, 255), -1, cv::LINE_AA);
  }

  const double max_xy = std::max(
    {1.0, std::abs(diag.target_xyz.x()) * 1.25, std::abs(diag.target_xyz.y()) * 1.25});
  const auto origin_top_pt = map_to_rect(0.0, 0.0, top_rect, -max_xy, max_xy, -max_xy, max_xy);
  const auto target_top_pt = map_to_rect(
    diag.target_xyz.x(), diag.target_xyz.y(), top_rect, -max_xy, max_xy, -max_xy, max_xy);
  cv::arrowedLine(panel, origin_top_pt, target_top_pt, cv::Scalar(60, 200, 120), 2, cv::LINE_AA, 0, 0.06);

  const double ray_len = std::max(1.0, diag.target_dist_xy * 1.1);
  const cv::Point cmd_ray_pt = map_to_rect(
    ray_len * std::cos(diag.command_yaw), ray_len * std::sin(diag.command_yaw), top_rect, -max_xy,
    max_xy, -max_xy, max_xy);
  cv::arrowedLine(panel, origin_top_pt, cmd_ray_pt, cv::Scalar(0, 220, 255), 2, cv::LINE_AA, 0, 0.06);
  cv::circle(panel, target_top_pt, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

  const cv::Scalar verdict_color = diag.hit ? cv::Scalar(60, 220, 120) : cv::Scalar(0, 80, 255);
  draw_outlined_text(
    panel, diag.hit ? "Verdict: HIT" : "Verdict: MISS", {text_rect.x + 15, text_rect.y + 28}, 0.75,
    verdict_color, 2);
  draw_outlined_text(
    panel, fmt::format("plan.fire: {}  speed: {:.2f} m/s", diag.fire ? "true" : "false", diag.bullet_speed),
    {text_rect.x + 15, text_rect.y + 58}, 0.52, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("offset yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.yaw_offset), rad2deg(diag.pitch_offset)),
    {text_rect.x + 15, text_rect.y + 84}, 0.52, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("target xyz: ({:.2f}, {:.2f}, {:.2f})  d_xy: {:.2f}  d_3d: {:.2f}",
                       diag.target_xyz.x(), diag.target_xyz.y(), diag.target_xyz.z(),
                       diag.target_dist_xy, diag.target_dist_3d),
    {text_rect.x + 15, text_rect.y + 110}, 0.50, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("geo yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.target_geo_yaw), rad2deg(diag.target_geo_pitch)),
    {text_rect.x + 15, text_rect.y + 136}, 0.50, cv::Scalar(60, 200, 120));
  draw_outlined_text(
    panel, fmt::format("cmd-ref yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.required_cmd_yaw), rad2deg(diag.required_cmd_pitch)),
    {text_rect.x + 15, text_rect.y + 162}, 0.50, cv::Scalar(100, 180, 255));
  draw_outlined_text(
    panel, fmt::format("plan yaw/pitch: {:.2f} / {:.2f} deg", rad2deg(diag.command_yaw), rad2deg(diag.command_pitch)),
    {text_rect.x + 15, text_rect.y + 188}, 0.50, cv::Scalar(0, 220, 255));
  draw_outlined_text(
    panel, fmt::format("yaw/pitch residual: {:.3f} / {:.3f} deg", rad2deg(diag.yaw_residual), rad2deg(diag.pitch_residual)),
    {text_rect.x + 390, text_rect.y + 58}, 0.50, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("lateral miss: {:.1f} mm", diag.lateral_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 84}, 0.50, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("vertical miss: {:.1f} mm", diag.vertical_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 110}, 0.50, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("total miss: {:.1f} mm", diag.total_error * 1000.0),
    {text_rect.x + 390, text_rect.y + 136}, 0.50, cv::Scalar(230, 230, 230));
  draw_outlined_text(
    panel, fmt::format("armor size: {} x {:.0f}mm", diag.armor_type == auto_aim::ArmorType::big ? "230" : "135",
                       kLightbarLength * 1000.0),
    {text_rect.x + 390, text_rect.y + 162}, 0.50, cv::Scalar(230, 230, 230));
}

std::string armor_type_to_string(auto_aim::ArmorType armor_type)
{
  switch (armor_type) {
    case auto_aim::ArmorType::big:
      return "big";
    case auto_aim::ArmorType::small:
      return "small";
    default:
      return "unknown";
  }
}

std::string armor_name_to_string(auto_aim::ArmorName armor_name)
{
  const auto index = static_cast<size_t>(armor_name);
  if (index < auto_aim::ARMOR_NAMES.size()) return auto_aim::ARMOR_NAMES[index];
  return "unknown";
}

int64_t unix_time_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

bool has_cli_option(int argc, char * argv[], const std::string & long_option)
{
  const std::string exact = "--" + long_option;
  const std::string prefix = exact + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == exact) return true;
    if (arg.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

nlohmann::json ballistic_to_json(const BallisticDiagnostic & diag)
{
  nlohmann::json data;
  data["valid"] = diag.valid;
  data["unsolvable"] = diag.unsolvable;
  data["hit"] = diag.hit;
  data["fire"] = diag.fire;
  data["armor_type"] = armor_type_to_string(diag.armor_type);
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
}  // namespace

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const auto yaml = tools::load(config_path);
  const double yaw_offset = tools::read<double>(yaml, "yaw_offset") / 57.3;
  const double pitch_offset = tools::read<double>(yaml, "pitch_offset") / 57.3;
  const auto gimbal_state_unit_mode = parse_gimbal_state_unit_mode(
    tools::read_or<std::string>(yaml, "gimbal_state_unit", "auto"));
  const bool show_local = has_cli_option(argc, argv, "show-local") ?
    cli.get<bool>("show-local") : tools::read_or<bool>(yaml, "show_local", false);
  const bool disable_web = has_cli_option(argc, argv, "disable-web") ?
    cli.get<bool>("disable-web") : tools::read_or<bool>(yaml, "disable_web", false);
  const std::string web_host = has_cli_option(argc, argv, "web-host") ?
    cli.get<std::string>("web-host") : tools::read_or<std::string>(yaml, "web_host", "0.0.0.0");
  const uint16_t web_port = static_cast<uint16_t>(std::clamp(
    has_cli_option(argc, argv, "web-port") ?
      cli.get<int>("web-port") : tools::read_or<int>(yaml, "web_port", 8090),
    1, 65535));
  const double web_fps = std::clamp(
    has_cli_option(argc, argv, "web-fps") ?
      cli.get<double>("web-fps") : tools::read_or<double>(yaml, "web_fps", 8.0),
    1.0, 60.0);
  const double display_scale = std::clamp(
    has_cli_option(argc, argv, "web-scale") ?
      cli.get<double>("web-scale") : tools::read_or<double>(yaml, "web_scale", 0.7),
    0.25, 1.0);
  const int web_jpeg_quality = std::clamp(
    has_cli_option(argc, argv, "web-jpeg-quality") ?
      cli.get<int>("web-jpeg-quality") : tools::read_or<int>(yaml, "web_jpeg_quality", 70),
    30, 95);
  const auto web_client_ttl = std::chrono::milliseconds(std::max(
    250,
    has_cli_option(argc, argv, "web-client-ttl-ms") ?
      cli.get<int>("web-client-ttl-ms") : tools::read_or<int>(yaml, "web_client_ttl_ms", 2000)));
  const auto web_frame_interval =
    std::chrono::milliseconds(static_cast<int>(1000.0 / web_fps));
  const auto web_state_interval = 80ms;

  std::unique_ptr<tools::WebDebugger> web_debugger;
  if (!disable_web) {
    web_debugger = std::make_unique<tools::WebDebugger>(web_host, web_port);
    if (web_debugger->good()) {
      tools::logger()->info(
        "Web debugger listening on {}:{} (open {})", web_host, web_port, web_debugger->url());
      tools::logger()->info(
        "Web debugger config: fps={} scale={} jpeg={} ttl={}ms", web_fps, display_scale,
        web_jpeg_quality, web_client_ttl.count());
    } else {
      tools::logger()->warn("Web debugger disabled because the server failed to start.");
      web_debugger.reset();
    }
  }

  if (show_local) {
    tools::logger()->info("Local OpenCV debug windows enabled.");
  } else if (!web_debugger) {
    tools::logger()->warn("Both local window and web debugger are disabled.");
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  auto_aim::Planner debug_planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::mutex command_state_mutex;
  nlohmann::json command_state = nlohmann::json::object();

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      const auto target = target_queue.front();
      const auto gs = gimbal.state();
      const auto normalized_gimbal = normalize_gimbal_state(gs, gimbal_state_unit_mode);
      const auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      const bool fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = normalized_gimbal.yaw.deg;
      data["gimbal_yaw_vel"] = normalized_gimbal.yaw_vel.deg;
      data["gimbal_pitch"] = normalized_gimbal.pitch.deg;
      data["gimbal_pitch_vel"] = normalized_gimbal.pitch_vel.deg;
      data["target_yaw"] = rad2deg(plan.target_yaw);
      data["target_pitch"] = rad2deg(plan.target_pitch);
      data["plan_yaw"] = rad2deg(plan.yaw);
      data["plan_yaw_vel"] = rad2deg(plan.yaw_vel);
      data["plan_yaw_acc"] = rad2deg(plan.yaw_acc);
      data["plan_pitch"] = rad2deg(plan.pitch);
      data["plan_pitch_vel"] = rad2deg(plan.pitch_vel);
      data["plan_pitch_acc"] = rad2deg(plan.pitch_acc);
      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

      nlohmann::json snapshot;
      snapshot["has_target"] = target.has_value();
      snapshot["fire"] = plan.fire;
      snapshot["fired"] = fired;
      snapshot["gimbal_source_unit"] = normalized_gimbal.source_is_degree ? "deg" : "rad";
      snapshot["gimbal_yaw_raw"] = normalized_gimbal.yaw.raw;
      snapshot["gimbal_yaw_deg"] = normalized_gimbal.yaw.deg;
      snapshot["gimbal_yaw_rad"] = normalized_gimbal.yaw.rad;
      snapshot["gimbal_pitch_raw"] = normalized_gimbal.pitch.raw;
      snapshot["gimbal_pitch_deg"] = normalized_gimbal.pitch.deg;
      snapshot["gimbal_pitch_rad"] = normalized_gimbal.pitch.rad;
      snapshot["gimbal_yaw_vel_raw"] = normalized_gimbal.yaw_vel.raw;
      snapshot["gimbal_yaw_vel_deg"] = normalized_gimbal.yaw_vel.deg;
      snapshot["gimbal_yaw_vel_rad"] = normalized_gimbal.yaw_vel.rad;
      snapshot["gimbal_pitch_vel_raw"] = normalized_gimbal.pitch_vel.raw;
      snapshot["gimbal_pitch_vel_deg"] = normalized_gimbal.pitch_vel.deg;
      snapshot["gimbal_pitch_vel_rad"] = normalized_gimbal.pitch_vel.rad;
      snapshot["target_yaw_deg"] = rad2deg(plan.target_yaw);
      snapshot["target_yaw_rad"] = plan.target_yaw;
      snapshot["target_pitch_deg"] = rad2deg(plan.target_pitch);
      snapshot["target_pitch_rad"] = plan.target_pitch;
      snapshot["plan_yaw_deg"] = rad2deg(plan.yaw);
      snapshot["plan_yaw_rad"] = plan.yaw;
      snapshot["plan_pitch_deg"] = rad2deg(plan.pitch);
      snapshot["plan_pitch_rad"] = plan.pitch;
      snapshot["plan_yaw_vel_deg"] = rad2deg(plan.yaw_vel);
      snapshot["plan_yaw_vel_rad"] = plan.yaw_vel;
      snapshot["plan_pitch_vel_deg"] = rad2deg(plan.pitch_vel);
      snapshot["plan_pitch_vel_rad"] = plan.pitch_vel;
      snapshot["plan_yaw_acc_deg"] = rad2deg(plan.yaw_acc);
      snapshot["plan_yaw_acc_rad"] = plan.yaw_acc;
      snapshot["plan_pitch_acc_deg"] = rad2deg(plan.pitch_acc);
      snapshot["plan_pitch_acc_rad"] = plan.pitch_acc;
      snapshot["bullet_speed_mps"] = gs.bullet_speed;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];
        data["target_vz"] = target->ekf_x()[5];
        data["target_h"] = target->ekf_x()[10];
        data["w"] = target->ekf_x()[7];

        snapshot["target_z_m"] = target->ekf_x()[4];
        snapshot["target_vz_mps"] = target->ekf_x()[5];
        snapshot["target_h_m"] = target->ekf_x()[10];
        snapshot["target_w_rad_s"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
        snapshot["target_z_m"] = nullptr;
        snapshot["target_vz_mps"] = nullptr;
        snapshot["target_h_m"] = nullptr;
        snapshot["target_w_rad_s"] = nullptr;
      }

      data["planner_selected_armor"] = planner.debug_armor_id;
      data["planner_delay_ms"] = planner.debug_delay_time * 1000.0;
      data["planner_spin_gate"] = planner.debug_used_spin_gate ? 1 : 0;

      snapshot["selected_armor"] = planner.debug_armor_id;
      snapshot["delay_ms"] = planner.debug_delay_time * 1000.0;
      snapshot["spin_gate"] = planner.debug_used_spin_gate;

      {
        std::lock_guard<std::mutex> lock(command_state_mutex);
        command_state = std::move(snapshot);
      }

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);
  auto last_web_frame_time = std::chrono::steady_clock::now() - web_frame_interval;
  auto last_web_state_time = std::chrono::steady_clock::now() - web_state_interval;

  while (!exiter.exit()) {
    camera.read(img, t);
    const auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    auto targets = tracker.track(armors, t);
    const auto gs = gimbal.state();

    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) current_target = targets.front();

    const auto current_plan = debug_planner.plan(current_target, gs.bullet_speed);
    BallisticDiagnostic ballistic_diag;
    if (current_target.has_value() && current_plan.control) {
      ballistic_diag = build_ballistic_diagnostic(
        current_plan, debug_planner.debug_xyza, current_target->armor_type, gs.bullet_speed,
        yaw_offset, pitch_offset);
    }

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    const double latency_ms =
      tools::delta_time(std::chrono::steady_clock::now(), t) * 1000.0;
    const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
    const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;

    nlohmann::json latest_command_state;
    {
      std::lock_guard<std::mutex> lock(command_state_mutex);
      latest_command_state = command_state;
    }

    const auto now = std::chrono::steady_clock::now();
    const bool need_web_frame =
      web_debugger && web_debugger->has_active_client(web_client_ttl) &&
      (now - last_web_frame_time >= web_frame_interval);
    const bool need_visual_output = show_local || need_web_frame;

    if (web_debugger && now - last_web_state_time >= web_state_interval) {
      nlohmann::json web_state;
      web_state["server"]["unix_ms"] = unix_time_ms();
      web_state["frame"]["latency_ms"] = latency_ms;
      web_state["frame"]["image_width"] = img.cols;
      web_state["frame"]["image_height"] = img.rows;
      web_state["preview"]["has_target"] = current_target.has_value();
      web_state["preview"]["fire"] = current_plan.fire;
      web_state["preview"]["target_name"] =
        current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
      web_state["preview"]["armor_type"] =
        current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
      web_state["preview"]["target_yaw_deg"] = rad2deg(current_plan.target_yaw);
      web_state["preview"]["target_yaw_rad"] = current_plan.target_yaw;
      web_state["preview"]["target_pitch_deg"] = rad2deg(current_plan.target_pitch);
      web_state["preview"]["target_pitch_rad"] = current_plan.target_pitch;
      web_state["preview"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
      web_state["preview"]["plan_yaw_rad"] = current_plan.yaw;
      web_state["preview"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
      web_state["preview"]["plan_pitch_rad"] = current_plan.pitch;
      if (current_target.has_value()) {
        web_state["preview"]["target_x_m"] = debug_planner.debug_xyza[0];
        web_state["preview"]["target_y_m"] = debug_planner.debug_xyza[1];
        web_state["preview"]["target_z_m"] = debug_planner.debug_xyza[2];
      } else {
        web_state["preview"]["target_x_m"] = nullptr;
        web_state["preview"]["target_y_m"] = nullptr;
        web_state["preview"]["target_z_m"] = nullptr;
      }
      web_state["planner"]["selected_armor"] = debug_planner.debug_armor_id;
      web_state["planner"]["spin_gate"] = debug_planner.debug_used_spin_gate;
      web_state["planner"]["delay_ms"] = debug_planner.debug_delay_time * 1000.0;
      web_state["planner"]["w_rad_s"] = current_w;
      web_state["planner"]["h_m"] = current_h;
      web_state["ballistic"] = ballistic_to_json(ballistic_diag);
      web_state["command"] = latest_command_state;
      web_debugger->update_state(web_state);
      last_web_state_time = now;
    }

    if (need_visual_output) {
      cv::Mat annotated_img = img.clone();

      if (current_target.has_value()) {
        const auto & target = *current_target;

        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        int armor_idx = 0;
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(annotated_img, image_points, {0, 255, 0});
          if (image_points.empty()) {
            ++armor_idx;
            continue;
          }

          float min_x = image_points.front().x;
          float max_y = image_points.front().y;
          for (const auto & pt : image_points) {
            min_x = std::min(min_x, pt.x);
            max_y = std::max(max_y, pt.y);
          }

          int text_x = static_cast<int>(min_x);
          int text_y = static_cast<int>(max_y) + 22;
          text_x = std::max(0, std::min(text_x, annotated_img.cols - 220));
          text_y = std::max(30, std::min(text_y, annotated_img.rows - 130));

          const std::vector<std::string> armor_lines = {
            fmt::format("armor:{}", armor_idx),
            fmt::format("yaw: {:.1f}", rad2deg(xyza[3])),
            fmt::format("x: {:.2f}", xyza[1]),
            fmt::format("y: {:.2f}", xyza[0]),
            fmt::format("z: {:.2f}", xyza[2]),
          };
          const double font_scale_local = 0.50;
          const int line_gap = 22;
          for (size_t line_i = 0; line_i < armor_lines.size(); ++line_i) {
            cv::Point org(text_x, text_y + static_cast<int>(line_i) * line_gap);
            cv::putText(
              annotated_img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX,
              font_scale_local, cv::Scalar(0, 0, 0), 3);
            cv::putText(
              annotated_img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX,
              font_scale_local, cv::Scalar(255, 0, 255), 2);
          }
          ++armor_idx;
        }

        auto target_future = target;
        constexpr int kRawTrajSteps = 18;
        constexpr double kRawTrajDt = 0.03;
        constexpr double kArrowMinPixelStep = 8.0;
        std::vector<cv::Point> raw_traj_centers;
        std::vector<int> raw_traj_ids;
        raw_traj_centers.reserve(kRawTrajSteps);
        raw_traj_ids.reserve(kRawTrajSteps);

        for (int step = 0; step < kRawTrajSteps; ++step) {
          const auto future_selection = debug_planner.preview_aim_selection(target_future);
          if (!future_selection.valid) break;

          auto pred_points = solver.reproject_armor(
            future_selection.xyza.head(3), future_selection.xyza[3], target.armor_type,
            target.name);
          if (!pred_points.empty()) {
            cv::Point2f center(0.0f, 0.0f);
            for (const auto & pt : pred_points) {
              center.x += pt.x;
              center.y += pt.y;
            }
            center.x /= static_cast<float>(pred_points.size());
            center.y /= static_cast<float>(pred_points.size());

            raw_traj_centers.emplace_back(
              static_cast<int>(center.x), static_cast<int>(center.y));
            raw_traj_ids.push_back(future_selection.armor_id);
          }

          target_future.predict(kRawTrajDt);
        }

        std::vector<cv::Point> traj_centers;
        std::vector<int> traj_ids;
        if (!raw_traj_centers.empty()) {
          traj_centers.push_back(raw_traj_centers.front());
          traj_ids.push_back(raw_traj_ids.front());

          for (size_t i = 1; i < raw_traj_centers.size(); ++i) {
            const bool switched = raw_traj_ids[i] != traj_ids.back();
            const double pixel_step = cv::norm(raw_traj_centers[i] - traj_centers.back());
            const bool is_last = i + 1 == raw_traj_centers.size();
            if (switched || pixel_step >= kArrowMinPixelStep || is_last) {
              traj_centers.push_back(raw_traj_centers[i]);
              traj_ids.push_back(raw_traj_ids[i]);
            }
          }
        }

        for (size_t i = 0; i < traj_centers.size(); ++i) {
          cv::circle(annotated_img, traj_centers[i], 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
          cv::circle(
            annotated_img, traj_centers[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
        }

        for (size_t i = 1; i < traj_centers.size(); ++i) {
          const bool switched = traj_ids[i] != traj_ids[i - 1];
          const cv::Scalar traj_color =
            switched ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 0);

          cv::arrowedLine(
            annotated_img, traj_centers[i - 1], traj_centers[i], cv::Scalar(0, 0, 0), 5,
            cv::LINE_AA, 0, 0.32);
          cv::arrowedLine(
            annotated_img, traj_centers[i - 1], traj_centers[i], traj_color, 2, cv::LINE_AA, 0,
            0.32);

          if (switched) {
            cv::putText(
              annotated_img, "switch", traj_centers[i] + cv::Point(6, -6),
              cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 165, 255), 1);
          }
        }

        if (!traj_centers.empty()) {
          cv::putText(
            annotated_img, "pred traj", traj_centers.front() + cv::Point(8, -10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        }

        if (current_plan.control) {
          const Eigen::Vector4d aim_xyza = debug_planner.debug_xyza;
          auto image_points = solver.reproject_armor(
            aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(annotated_img, image_points, {0, 0, 255});
        }
      }

      cv::Mat display_img;
      cv::resize(annotated_img, display_img, {}, display_scale, display_scale);

      const int font_face = cv::FONT_HERSHEY_SIMPLEX;
      const double font_scale = 0.4;
      const cv::Scalar color(0, 255, 255);
      const int thickness = 1;

      if (current_plan.fire) {
        const std::string fire_text = "fire!";
        int fire_baseline = 0;
        const cv::Size fire_size =
          cv::getTextSize(fire_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &fire_baseline);
        const cv::Point fire_org(
          (display_img.cols - fire_size.width) / 2, fire_size.height + 10);
        cv::putText(
          display_img, fire_text, fire_org, cv::FONT_HERSHEY_SIMPLEX, 1.0,
          cv::Scalar(0, 0, 255), 2);
      }

      const std::string latency_info = fmt::format("latency: {:.2f} ms", latency_ms);
      int latency_baseline = 0;
      const cv::Size latency_size =
        cv::getTextSize(latency_info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &latency_baseline);
      const cv::Point latency_org(
        display_img.cols - latency_size.width - 10, latency_size.height + 10);
      cv::putText(
        display_img, latency_info, latency_org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 255, 255), 1);

      const int info_x = 12;
      const int info_y = 20;
      const int info_line_gap = 18;
      const std::vector<std::string> planner_lines = {
        fmt::format("armor_id: {}", debug_planner.debug_armor_id),
        fmt::format("spin_gate: {}", debug_planner.debug_used_spin_gate ? "on" : "off"),
        fmt::format("delay: {:.1f} ms", debug_planner.debug_delay_time * 1000.0),
        fmt::format("w: {:.2f} rad/s", current_w),
        fmt::format("h: {:.3f} m", current_h),
      };
      for (size_t i = 0; i < planner_lines.size(); ++i) {
        const cv::Point org(info_x, info_y + static_cast<int>(i) * info_line_gap);
        cv::putText(
          display_img, planner_lines[i], org, font_face, font_scale, cv::Scalar(0, 0, 0),
          thickness + 2);
        cv::putText(display_img, planner_lines[i], org, font_face, font_scale, color, thickness);
      }

      draw_ballistic_panel(ballistic_panel, ballistic_diag);

      if (need_web_frame && web_debugger) {
        web_debugger->update_main_frame(display_img, web_jpeg_quality);
        web_debugger->update_ballistic_frame(ballistic_panel, web_jpeg_quality);
        last_web_frame_time = now;
      }

      if (show_local) {
        cv::imshow("Auto Aim Debug", display_img);
        cv::imshow("Ballistic Debug", ballistic_panel);
        const auto key = cv::waitKey(1);
        if (key == 'q') break;
      }
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  if (show_local) cv::destroyAllWindows();

  return 0;
}
