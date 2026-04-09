#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/trajectory.hpp"
#include "tools/web_debugger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr double kGravity = 9.7833;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/standard3.yaml | yaml配置文件的路径}"
  "{timestamp-path |                   | 显式指定txt时间戳文件路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{bullet-speed   | 23.0              | 离线回放使用的弹速(m/s)}"
  "{playback-speed | 1.0               | 离线回放速度倍率  }"
  "{show-local     | false             | 保留本地OpenCV调试窗口(显式传参时覆盖yaml) }"
  "{disable-web    | false             | 禁用内置网页调试器(显式传参时覆盖yaml) }"
  "{web-host       | 0.0.0.0           | 网页调试器绑定地址(显式传参时覆盖yaml) }"
  "{web-port       | 8090              | 网页调试器端口(显式传参时覆盖yaml) }"
  "{web-fps        | 30.0               | 网页图像刷新帧率(显式传参时覆盖yaml) }"
  "{web-scale      | 0.7               | 网页图像缩放系数(显式传参时覆盖yaml) }"
  "{web-jpeg-quality | 70              | 网页JPEG质量(30-95, 显式传参时覆盖yaml) }"
  "{web-client-ttl-ms | 2000           | 最近访问多久内继续渲染网页帧(显式传参时覆盖yaml) }"
  "{@input-path    | assets/demo/test.mp4  | avi和txt文件的路径}";

double rad2deg(double rad)
{
  return rad * 180.0 / M_PI;
}

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
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thickness + 2);
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
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

bool is_video_extension(const std::string & ext)
{
  return ext == ".avi" || ext == ".mp4" || ext == ".mov" || ext == ".mkv";
}

std::string resolve_video_path(const std::string & input_path)
{
  const std::filesystem::path input(input_path);
  if (input.has_extension() && is_video_extension(input.extension().string())) {
    return input.string();
  }

  for (const char * ext : {".avi", ".mp4", ".mov", ".mkv"}) {
    const auto candidate = input_path + ext;
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return input_path + ".avi";
}

std::string resolve_text_path(
  const std::string & input_path, const std::string & cli_timestamp_path)
{
  if (!cli_timestamp_path.empty()) return cli_timestamp_path;

  std::filesystem::path input(input_path);
  if (input.has_extension() && is_video_extension(input.extension().string())) {
    input.replace_extension(".txt");
    return input.string();
  }
  return input_path + ".txt";
}

int64_t unix_time_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

BallisticDiagnostic build_ballistic_diagnostic(
  const io::Command & command, const Eigen::Vector4d & aim_xyza, auto_aim::ArmorType armor_type,
  double bullet_speed, double yaw_offset, double pitch_offset)
{
  BallisticDiagnostic diag;
  if (!command.control) return diag;

  diag.valid = true;
  diag.fire = command.shoot;
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
  diag.command_yaw = command.yaw;
  diag.command_pitch = command.pitch;
  diag.yaw_residual = tools::limit_rad(command.yaw - diag.required_cmd_yaw);
  diag.pitch_residual = command.pitch - diag.required_cmd_pitch;

  const Eigen::Vector2d target_xy(diag.target_xyz.x(), diag.target_xyz.y());
  const Eigen::Vector2d shot_dir(std::cos(command.yaw), std::sin(command.yaw));
  const double along = target_xy.dot(shot_dir);
  const double lateral = target_xy.x() * shot_dir.y() - target_xy.y() * shot_dir.x();
  diag.lateral_error = (along >= 0.0) ? lateral : target_xy.norm();

  const double bullet_z = bullet_height(diag.target_dist_xy, bullet_speed, -command.pitch);
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
    draw_outlined_text(panel, "No valid target / command", {220, 210}, 0.9, cv::Scalar(120, 220, 255), 2);
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
    const double d = max_dist * static_cast<double>(i) / static_cast<double>(sample_num - 1);
    const double ideal_z = bullet_height(d, diag.bullet_speed, -diag.target_geo_pitch);
    const double cmd_z = bullet_height(d, diag.bullet_speed, -diag.command_pitch);
    if (std::isfinite(ideal_z)) {
      ideal_curve.push_back(map_to_rect(d, ideal_z, side_rect, 0.0, max_dist, min_z, max_z));
    }
    if (std::isfinite(cmd_z)) {
      cmd_curve.push_back(map_to_rect(d, cmd_z, side_rect, 0.0, max_dist, min_z, max_z));
    }
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
    ray_len * std::cos(diag.command_yaw), ray_len * std::sin(diag.command_yaw), top_rect,
    -max_xy, max_xy, -max_xy, max_xy);
  cv::arrowedLine(panel, origin_top_pt, cmd_ray_pt, cv::Scalar(0, 220, 255), 2, cv::LINE_AA, 0, 0.06);
  cv::circle(panel, target_top_pt, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

  const cv::Scalar verdict_color = diag.hit ? cv::Scalar(60, 220, 120) : cv::Scalar(0, 80, 255);
  draw_outlined_text(
    panel, diag.hit ? "Verdict: HIT" : "Verdict: MISS", {text_rect.x + 15, text_rect.y + 28}, 0.75,
    verdict_color, 2);
  draw_outlined_text(
    panel, fmt::format("fire: {}  speed: {:.2f} m/s", diag.fire ? "true" : "false", diag.bullet_speed),
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
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto input_path = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto cli_timestamp_path = cli.get<std::string>("timestamp-path");
  const auto start_index = cli.get<int>("start-index");
  const auto end_index = cli.get<int>("end-index");
  const double bullet_speed = std::max(1.0, cli.get<double>("bullet-speed"));
  const double playback_speed = std::max(0.1, cli.get<double>("playback-speed"));
  const auto yaml = tools::load(config_path);
  const double yaw_offset = tools::read<double>(yaml, "yaw_offset") / 57.3;
  const double pitch_offset = tools::read<double>(yaml, "pitch_offset") / 57.3;
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

  tools::Plotter plotter;
  tools::Exiter exiter;

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

  const auto video_path = resolve_video_path(input_path);
  const auto text_path = resolve_text_path(input_path, cli_timestamp_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  if (!video.isOpened()) {
    tools::logger()->error("Failed to open video: {}", video_path);
    return 1;
  }
  tools::logger()->info("Using video file: {}", video_path);
  const bool has_timestamp_file = text.is_open();
  const double video_fps_raw = video.get(cv::CAP_PROP_FPS);
  const double fallback_video_fps = video_fps_raw > 1.0 ? video_fps_raw : 30.0;
  if (has_timestamp_file) {
    tools::logger()->info("Using timestamp file: {}", text_path);
  } else {
    tools::logger()->warn(
      "Timestamp file not found: {}. Falling back to video FPS {:.2f} and identity quaternion.",
      text_path, fallback_video_fps);
  }

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);
  auto playback_start = std::chrono::steady_clock::now();
  io::Command last_command;
  auto last_web_frame_time = std::chrono::steady_clock::now() - web_frame_interval;
  auto last_web_state_time = std::chrono::steady_clock::now() - web_state_interval;
  double first_t = -1.0;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  if (has_timestamp_file) {
    for (int i = 0; i < start_index; i++) {
      double t, w, x, y, z;
      text >> t >> w >> x >> y >> z;
    }
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    double t = 0.0;
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (has_timestamp_file) {
      text >> t >> w >> x >> y >> z;
      if (!text.good()) break;
    } else {
      t = static_cast<double>(frame_count - start_index) / fallback_video_fps;
    }

    if (first_t < 0.0) first_t = t;
    const double relative_t = std::max(0.0, t - first_t);
    const auto target_wall_time = playback_start +
      std::chrono::microseconds(static_cast<int64_t>(relative_t * 1e6 / playback_speed));
    std::this_thread::sleep_until(target_wall_time);

    video.read(img);
    if (img.empty()) break;

    const auto frame_start = std::chrono::steady_clock::now();
    const auto timestamp =
      playback_start + std::chrono::microseconds(static_cast<int64_t>(relative_t * 1e6));

    solver.set_R_gimbal2world(Eigen::Quaterniond(w, x, y, z));

    const auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    const auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    const auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, bullet_speed, false);
    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
    {
      command.shoot = true;
    }
    if (command.control) last_command = command;

    const auto finish = std::chrono::steady_clock::now();
    const double processing_ms = tools::delta_time(finish, frame_start) * 1000.0;
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    Eigen::Quaterniond gimbal_q(w, x, y, z);
    const Eigen::Vector3d gimbal_ypr =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img, fmt::format("gimbal yaw {:.2f}", gimbal_ypr[0] * 57.3), {10, 90}, {255, 255, 255});

    nlohmann::json data;
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    data["gimbal_yaw"] = gimbal_ypr[0] * 57.3;
    data["gimbal_pitch"] = gimbal_ypr[1] * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["shoot"] = command.shoot;
    data["bullet_speed"] = bullet_speed;
    data["t"] = relative_t;
    data["t"] = relative_t;

    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) {
      current_target = targets.front();
      const auto & target = *current_target;

      for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      const auto aim_point = aimer.debug_aim_point;
      const auto image_points = solver.reproject_armor(
        aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      const Eigen::VectorXd x_state = target.ekf_x();
      data["x"] = x_state[0];
      data["vx"] = x_state[1];
      data["y"] = x_state[2];
      data["vy"] = x_state[3];
      data["z"] = x_state[4];
      data["vz"] = x_state[5];
      data["a"] = x_state[6] * 57.3;
      data["w"] = x_state[7];
      data["r"] = x_state[8];
      data["l"] = x_state[9];
      data["h"] = x_state[10];
      data["last_id"] = target.last_id;
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    } else {
      data["w"] = 0.0;
      data["h"] = 0.0;
    }

    plotter.plot(data);

    BallisticDiagnostic ballistic_diag;
    if (current_target.has_value() && aimer.debug_aim_point.valid && command.control) {
      ballistic_diag = build_ballistic_diagnostic(
        command, aimer.debug_aim_point.xyza, current_target->armor_type, bullet_speed,
        yaw_offset, pitch_offset);
    }

    const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
    const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
    const bool spin_gate = std::abs(current_w) > 2.0;
    const auto now = std::chrono::steady_clock::now();
    const bool need_web_frame =
      web_debugger && web_debugger->has_active_client(web_client_ttl) &&
      (now - last_web_frame_time >= web_frame_interval);
    const bool need_visual_output = show_local || need_web_frame;

    if (web_debugger && now - last_web_state_time >= web_state_interval) {
      nlohmann::json web_state;
      web_state["server"]["unix_ms"] = unix_time_ms();
      web_state["frame"]["latency_ms"] = processing_ms;
      web_state["frame"]["image_width"] = img.cols;
      web_state["frame"]["image_height"] = img.rows;
      web_state["frame"]["playback_t_s"] = relative_t;
      web_state["frame"]["raw_t_s"] = t;
      web_state["frame"]["frame_index"] = frame_count;
      web_state["preview"]["has_target"] = current_target.has_value();
      web_state["preview"]["fire"] = command.shoot;
      web_state["preview"]["target_name"] =
        current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
      web_state["preview"]["armor_type"] =
        current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
      if (ballistic_diag.valid) {
        web_state["preview"]["target_yaw_deg"] = rad2deg(ballistic_diag.target_geo_yaw);
        web_state["preview"]["target_yaw_rad"] = ballistic_diag.target_geo_yaw;
        web_state["preview"]["target_pitch_deg"] = rad2deg(ballistic_diag.target_geo_pitch);
        web_state["preview"]["target_pitch_rad"] = ballistic_diag.target_geo_pitch;
      } else {
        web_state["preview"]["target_yaw_deg"] = nullptr;
        web_state["preview"]["target_yaw_rad"] = nullptr;
        web_state["preview"]["target_pitch_deg"] = nullptr;
        web_state["preview"]["target_pitch_rad"] = nullptr;
      }
      if (command.control) {
        web_state["preview"]["plan_yaw_deg"] = rad2deg(command.yaw);
        web_state["preview"]["plan_yaw_rad"] = command.yaw;
        web_state["preview"]["plan_pitch_deg"] = rad2deg(command.pitch);
        web_state["preview"]["plan_pitch_rad"] = command.pitch;
      } else {
        web_state["preview"]["plan_yaw_deg"] = nullptr;
        web_state["preview"]["plan_yaw_rad"] = nullptr;
        web_state["preview"]["plan_pitch_deg"] = nullptr;
        web_state["preview"]["plan_pitch_rad"] = nullptr;
      }
      if (current_target.has_value() && aimer.debug_aim_point.valid) {
        web_state["preview"]["target_x_m"] = aimer.debug_aim_point.xyza[0];
        web_state["preview"]["target_y_m"] = aimer.debug_aim_point.xyza[1];
        web_state["preview"]["target_z_m"] = aimer.debug_aim_point.xyza[2];
      } else {
        web_state["preview"]["target_x_m"] = nullptr;
        web_state["preview"]["target_y_m"] = nullptr;
        web_state["preview"]["target_z_m"] = nullptr;
      }
      web_state["planner"]["selected_armor"] = current_target.has_value() ? current_target->last_id : -1;
      web_state["planner"]["spin_gate"] = spin_gate;
      web_state["planner"]["delay_ms"] = 0.0;
      web_state["planner"]["w_rad_s"] = current_w;
      web_state["planner"]["h_m"] = current_h;
      web_state["ballistic"] = ballistic_to_json(ballistic_diag);
      web_state["command"]["has_target"] = current_target.has_value();
      web_state["command"]["fired"] = command.shoot;
      web_state["command"]["gimbal_source_unit"] = "rad";
      web_state["command"]["gimbal_yaw_raw"] = gimbal_ypr[0];
      web_state["command"]["gimbal_yaw_deg"] = rad2deg(gimbal_ypr[0]);
      web_state["command"]["gimbal_yaw_rad"] = gimbal_ypr[0];
      web_state["command"]["gimbal_pitch_raw"] = gimbal_ypr[1];
      web_state["command"]["gimbal_pitch_deg"] = rad2deg(gimbal_ypr[1]);
      web_state["command"]["gimbal_pitch_rad"] = gimbal_ypr[1];
      if (command.control) {
        web_state["command"]["plan_yaw_deg"] = rad2deg(command.yaw);
        web_state["command"]["plan_yaw_rad"] = command.yaw;
        web_state["command"]["plan_pitch_deg"] = rad2deg(command.pitch);
        web_state["command"]["plan_pitch_rad"] = command.pitch;
      } else {
        web_state["command"]["plan_yaw_deg"] = nullptr;
        web_state["command"]["plan_yaw_rad"] = nullptr;
        web_state["command"]["plan_pitch_deg"] = nullptr;
        web_state["command"]["plan_pitch_rad"] = nullptr;
      }
      web_state["command"]["bullet_speed_mps"] = bullet_speed;
      web_debugger->update_state(web_state);
      last_web_state_time = now;
    }

    if (need_visual_output) {
      cv::Mat display_img;
      cv::resize(img, display_img, {}, display_scale, display_scale);

      if (command.shoot) {
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

      const std::vector<std::string> debug_lines = {
        fmt::format("frame: {}", frame_count),
        fmt::format("playback: {:.2f}s x{:.2f}", relative_t, playback_speed),
        fmt::format("latency: {:.2f} ms", processing_ms),
        fmt::format("bullet_speed: {:.1f} m/s", bullet_speed),
        fmt::format("w: {:.2f} rad/s", current_w),
        fmt::format("h: {:.3f} m", current_h),
      };
      for (size_t i = 0; i < debug_lines.size(); ++i) {
        const cv::Point org(12, 20 + static_cast<int>(i) * 18);
        cv::putText(
          display_img, debug_lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.45,
          cv::Scalar(0, 0, 0), 3);
        cv::putText(
          display_img, debug_lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.45,
          cv::Scalar(0, 255, 255), 1);
      }

      draw_ballistic_panel(ballistic_panel, ballistic_diag);

      if (need_web_frame && web_debugger) {
        web_debugger->update_main_frame(display_img, web_jpeg_quality);
        web_debugger->update_ballistic_frame(ballistic_panel, web_jpeg_quality);
        last_web_frame_time = now;
      }

      if (show_local) {
        cv::imshow("Auto Aim Test Web", display_img);
        cv::imshow("Ballistic Debug", ballistic_panel);
        const auto key = cv::waitKey(1);
        if (key == 'q') break;
      }
    }
  }

  if (show_local) cv::destroyAllWindows();
  return 0;
}
