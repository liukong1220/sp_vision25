#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/debug.hpp"
#include "tools/debug_visualization.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"
#include "tools/plotter.hpp"
#include "tools/web_debugger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;
using tools::debug::BallisticDiagnostic;
using tools::debug::armor_name_to_string;
using tools::debug::armor_type_to_string;
using tools::debug::ballistic_to_json;
using tools::debug::build_ballistic_diagnostic;
using tools::debug::draw_ballistic_panel;
using tools::debug::has_cli_option;
using tools::debug::rad2deg;
using tools::debug::unix_time_ms;

namespace
{
enum class DebugMode : int
{
  AutoAim = 1,
  SmallBuff = 2,
  BigBuff = 3,
};

struct BuffModeResult
{
  std::optional<auto_buff::PowerRune> power_rune;
  auto_aim::Plan plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  bool target_solved = false;
  std::string overlay_stage = "buff_idle";
  std::vector<cv::Point2f> current_projection;
  std::vector<cv::Point2f> predicted_projection;
  double rune_yaw_deg = 0.0;
  double rune_pitch_deg = 0.0;
  double rune_dist_m = 0.0;
  double blade_yaw_deg = 0.0;
  double blade_pitch_deg = 0.0;
  double blade_dist_m = 0.0;
  double buff_yaw_deg = 0.0;
  double buff_pitch_deg = 0.0;
  double buff_roll_deg = 0.0;
  double target_angle_deg = 0.0;
  double target_spd_deg_s = 0.0;
  double target_spd_rad_s = 0.0;
  double fit_a_deg_s = 0.0;
  double fit_w_rad_s = 0.0;
  double fit_fi_deg = 0.0;
  double aim_x_m = 0.0;
  double aim_y_m = 0.0;
  double aim_z_m = 0.0;
  double predicted_aim_x_m = 0.0;
  double predicted_aim_y_m = 0.0;
  double predicted_aim_z_m = 0.0;
};

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{timestamp-path |                   | 显式指定txt时间戳文件路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{bullet-speed   | 23.0              | 离线回放使用的弹速(m/s)}"
  "{playback-speed | 1.0               | 离线回放速度倍率  }"
  "{show-local     | false             | 保留本地OpenCV调试窗口(显式传参时覆盖yaml) }"
  "{disable-web    | false             | 禁用内置网页调试器(显式传参时覆盖yaml) }"
  "{web-host       | 0.0.0.0           | 网页调试器绑定地址(显式传参时覆盖yaml) }"
  "{web-port       | 8090              | 网页调试器端口(显式传参时覆盖yaml) }"
  "{web-fps        | 30.0              | 网页图像刷新帧率(显式传参时覆盖yaml) }"
  "{web-scale      | 0.7               | 网页图像缩放系数(显式传参时覆盖yaml) }"
  "{web-jpeg-quality | 70              | 网页JPEG质量(30-95, 显式传参时覆盖yaml) }"
  "{web-client-ttl-ms | 2000           | 最近访问多久内继续渲染网页帧(显式传参时覆盖yaml) }"
  "{@input-path    | assets/demo/demo.avi  | avi和txt文件的路径}";

bool is_video_extension(const std::string & ext)
{
  return ext == ".avi" || ext == ".mp4" || ext == ".mov" || ext == ".mkv";
}

std::string resolve_video_path(const std::string & input_path)
{
  const std::filesystem::path input(input_path);
  if (input.has_extension() && is_video_extension(input.extension().string())) {
    return tools::resolve_runtime_path_string(input_path);
  }

  for (const char * ext : {".avi", ".mp4", ".mov", ".mkv"}) {
    const auto candidate = tools::resolve_runtime_path(input_path + ext);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return tools::resolve_runtime_path_string(input_path + ".avi");
}

std::string resolve_text_path(
  const std::string & input_path, const std::string & cli_timestamp_path)
{
  if (!cli_timestamp_path.empty()) {
    return tools::resolve_runtime_path_string(cli_timestamp_path);
  }

  std::filesystem::path input(input_path);
  if (input.has_extension() && is_video_extension(input.extension().string())) {
    input.replace_extension(".txt");
    return tools::resolve_runtime_path_string(input.string());
  }
  return tools::resolve_runtime_path_string(input_path + ".txt");
}

DebugMode clamp_debug_mode(int mode)
{
  if (mode <= static_cast<int>(DebugMode::AutoAim)) return DebugMode::AutoAim;
  if (mode >= static_cast<int>(DebugMode::BigBuff)) return DebugMode::BigBuff;
  return static_cast<DebugMode>(mode);
}

const char * debug_mode_key(DebugMode mode)
{
  switch (mode) {
    case DebugMode::AutoAim:
      return "auto_aim";
    case DebugMode::SmallBuff:
      return "small_buff";
    case DebugMode::BigBuff:
      return "big_buff";
    default:
      return "auto_aim";
  }
}

const char * debug_mode_label(DebugMode mode)
{
  switch (mode) {
    case DebugMode::AutoAim:
      return "自瞄";
    case DebugMode::SmallBuff:
      return "小符";
    case DebugMode::BigBuff:
      return "大符";
    default:
      return "自瞄";
  }
}

double effective_bullet_speed(DebugMode mode, double raw_speed)
{
  if (mode == DebugMode::AutoAim) {
    return (raw_speed < 10.0 || raw_speed > 25.0) ? 22.0 : raw_speed;
  }
  return raw_speed < 10.0 ? 24.0 : raw_speed;
}

bool bullet_speed_fallback(DebugMode mode, double raw_speed)
{
  if (mode == DebugMode::AutoAim) {
    return raw_speed < 10.0 || raw_speed > 25.0;
  }
  return raw_speed < 10.0;
}

bool overlay_flag(const nlohmann::json & overlay_config, const char * key, bool fallback = true)
{
  if (!overlay_config.is_object()) return fallback;
  if (!overlay_config.contains(key) || !overlay_config.at(key).is_boolean()) return fallback;
  return overlay_config.at(key).get<bool>();
}

void draw_panel_line(
  cv::Mat & panel, int row, const std::string & label, const std::string & value,
  const cv::Scalar & value_color = {220, 240, 255})
{
  const int y = 92 + row * 34;
  cv::putText(
    panel, label, {24, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58, {126, 180, 205}, 1, cv::LINE_AA);
  cv::putText(
    panel, value, {256, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58, value_color, 1, cv::LINE_AA);
}

void draw_buff_panel(
  cv::Mat & panel, DebugMode mode, const BuffModeResult & buff_result,
  const Eigen::Vector3d & gimbal_ypr, double processing_ms, double raw_bullet_speed,
  double effective_speed, bool speed_fallback)
{
  panel.setTo(cv::Scalar(8, 18, 28));
  cv::rectangle(panel, {0, 0, panel.cols, 58}, cv::Scalar(22, 48, 76), cv::FILLED);
  cv::putText(
    panel, fmt::format("{} Offline Diagnostic", debug_mode_label(mode)), {24, 38},
    cv::FONT_HERSHEY_SIMPLEX, 0.84, {235, 246, 255}, 2, cv::LINE_AA);

  draw_panel_line(panel, 0, "Latency", fmt::format("{:.1f} ms", processing_ms));
  draw_panel_line(
    panel, 1, "Gimbal",
    fmt::format(
      "yaw {:+.2f} deg   pitch {:+.2f} deg",
      rad2deg(gimbal_ypr[0]), rad2deg(gimbal_ypr[1])));
  draw_panel_line(
    panel, 2, "Bullet",
    fmt::format(
      "{:.2f} -> {:.2f} m/s{}",
      raw_bullet_speed, effective_speed, speed_fallback ? "  [fallback]" : ""));
  draw_panel_line(
    panel, 3, "Rune",
    buff_result.power_rune.has_value() ?
      fmt::format(
        "yaw {:+.2f} deg  pitch {:+.2f} deg  dis {:.3f} m",
        buff_result.rune_yaw_deg, buff_result.rune_pitch_deg, buff_result.rune_dist_m) :
      "waiting");
  draw_panel_line(
    panel, 4, "Blade",
    buff_result.power_rune.has_value() ?
      fmt::format(
        "yaw {:+.2f} deg  pitch {:+.2f} deg  dis {:.3f} m",
        buff_result.blade_yaw_deg, buff_result.blade_pitch_deg, buff_result.blade_dist_m) :
      "waiting");
  draw_panel_line(
    panel, 5, "State",
    buff_result.target_solved ?
      fmt::format(
        "angle {:+.2f} deg  spd {:+.2f} deg/s",
        buff_result.target_angle_deg, buff_result.target_spd_deg_s) :
      "unsolved");

  if (mode == DebugMode::BigBuff) {
    draw_panel_line(
      panel, 6, "Model",
      buff_result.target_solved ?
        fmt::format(
          "a {:+.2f} deg/s  w {:.3f} rad/s  fi {:+.2f} deg",
          buff_result.fit_a_deg_s, buff_result.fit_w_rad_s, buff_result.fit_fi_deg) :
        "waiting");
  } else {
    draw_panel_line(
      panel, 6, "Pose",
      buff_result.power_rune.has_value() ?
        fmt::format(
          "yaw {:+.2f}  pitch {:+.2f}  roll {:+.2f} deg",
          buff_result.buff_yaw_deg, buff_result.buff_pitch_deg, buff_result.buff_roll_deg) :
        "waiting");
  }

  draw_panel_line(
    panel, 7, "Plan",
    fmt::format(
      "yaw {:+.2f} deg  pitch {:+.2f} deg",
      rad2deg(buff_result.plan.yaw), rad2deg(buff_result.plan.pitch)),
    buff_result.plan.control ? cv::Scalar(162, 255, 210) : cv::Scalar(155, 172, 186));
  draw_panel_line(
    panel, 8, "PlanRate",
    fmt::format(
      "yaw {:+.2f} deg/s  pitch {:+.2f} deg/s",
      rad2deg(buff_result.plan.yaw_vel), rad2deg(buff_result.plan.pitch_vel)));
  draw_panel_line(
    panel, 9, "Trigger",
    fmt::format(
      "control {}  fire {}  fired {}",
      buff_result.plan.control ? "ON" : "OFF",
      buff_result.plan.fire ? "YES" : "NO",
      buff_result.plan.fire ? "YES" : "NO"),
    buff_result.plan.fire ? cv::Scalar(120, 180, 255) : cv::Scalar(220, 240, 255));
}

cv::Mat render_buff_debug_frame(
  const cv::Mat & source, const BuffModeResult & buff_result, const nlohmann::json & overlay_config,
  DebugMode mode, double display_scale, double processing_ms, double raw_bullet_speed)
{
  cv::Mat display = source.clone();
  if (display.empty()) return display;

  const bool show_armors = overlay_flag(overlay_config, "armors", true);
  const bool show_labels = overlay_flag(overlay_config, "labels", true);
  const bool show_aim = overlay_flag(overlay_config, "aim", true);
  const bool show_footer = overlay_flag(overlay_config, "footer", true);
  const bool show_hud = overlay_flag(overlay_config, "decision_hud", true);

  if (show_armors && buff_result.power_rune.has_value() && !buff_result.power_rune->fanblades.empty()) {
    const auto & target = buff_result.power_rune->fanblades.front();
    if (target.points.size() >= 4) {
      std::vector<cv::Point2f> target_outline(target.points.begin(), target.points.begin() + 4);
      tools::draw_points(display, target_outline, {0, 255, 255}, 2);
    }
    cv::circle(display, target.center, 4, {0, 0, 255}, cv::FILLED);
    cv::circle(display, buff_result.power_rune->r_center, 4, {255, 0, 255}, cv::FILLED);
  }

  if (show_armors && buff_result.current_projection.size() >= 4) {
    tools::draw_points(
      display,
      std::vector<cv::Point2f>(
        buff_result.current_projection.begin(), buff_result.current_projection.begin() + 4),
      {0, 255, 0}, 2);
    if (buff_result.current_projection.size() > 4) {
      tools::draw_points(
        display,
        std::vector<cv::Point2f>(
          buff_result.current_projection.begin() + 4, buff_result.current_projection.end()),
        {0, 255, 0}, 2);
    }
  }

  if (show_aim && buff_result.predicted_projection.size() >= 4) {
    tools::draw_points(
      display,
      std::vector<cv::Point2f>(
        buff_result.predicted_projection.begin(), buff_result.predicted_projection.begin() + 4),
      {255, 120, 0}, 2);
    if (buff_result.predicted_projection.size() > 4) {
      tools::draw_points(
        display,
        std::vector<cv::Point2f>(
          buff_result.predicted_projection.begin() + 4, buff_result.predicted_projection.end()),
        {255, 120, 0}, 2);
    }
  }

  if (show_hud) {
    cv::rectangle(display, {18, 18, 450, 112}, {9, 21, 31}, cv::FILLED);
    cv::rectangle(display, {18, 18, 450, 112}, {92, 145, 180}, 1);
    cv::putText(
      display, fmt::format("{} · {}", debug_mode_label(mode), buff_result.overlay_stage),
      {34, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.88, {232, 245, 255}, 2, cv::LINE_AA);
    cv::putText(
      display,
      fmt::format(
        "proc {:.1f} ms   bullet {:.2f} m/s   control {}",
        processing_ms, raw_bullet_speed, buff_result.plan.control ? "ON" : "OFF"),
      {34, 78}, cv::FONT_HERSHEY_SIMPLEX, 0.56, {154, 194, 215}, 1, cv::LINE_AA);
    cv::putText(
      display,
      buff_result.target_solved ?
        fmt::format(
          "blade {:+.1f} / {:+.1f} deg   fire {}",
          buff_result.blade_yaw_deg, buff_result.blade_pitch_deg,
          buff_result.plan.fire ? "YES" : "NO") :
        "target unsolved",
      {34, 104}, cv::FONT_HERSHEY_SIMPLEX, 0.56,
      buff_result.plan.fire ? cv::Scalar(120, 180, 255) : cv::Scalar(154, 194, 215),
      1, cv::LINE_AA);
  }

  if (show_labels && buff_result.power_rune.has_value()) {
    cv::putText(
      display,
      fmt::format(
        "R {:+.1f} / {:+.1f} deg",
        buff_result.rune_yaw_deg, buff_result.rune_pitch_deg),
      {28, display.rows - 58}, cv::FONT_HERSHEY_SIMPLEX, 0.56, {0, 220, 255}, 1, cv::LINE_AA);
    cv::putText(
      display,
      fmt::format(
        "B {:+.1f} / {:+.1f} deg",
        buff_result.blade_yaw_deg, buff_result.blade_pitch_deg),
      {28, display.rows - 32}, cv::FONT_HERSHEY_SIMPLEX, 0.56, {120, 255, 175}, 1, cv::LINE_AA);
  }

  if (show_footer) {
    cv::rectangle(
      display, {0, display.rows - 24, display.cols, 24}, {6, 13, 21}, cv::FILLED);
    cv::putText(
      display,
      fmt::format(
        "plan yaw {:+.2f} deg  pitch {:+.2f} deg  yaw_vel {:+.2f} deg/s",
        rad2deg(buff_result.plan.yaw), rad2deg(buff_result.plan.pitch),
        rad2deg(buff_result.plan.yaw_vel)),
      {18, display.rows - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.48, {224, 240, 250}, 1, cv::LINE_AA);
  }

  if (display_scale != 1.0) {
    cv::resize(display, display, {}, display_scale, display_scale);
  }
  return display;
}

nlohmann::json build_offline_mode_state(DebugMode mode)
{
  nlohmann::json data;
  data["mode"] = static_cast<int>(mode);
  data["mode_key"] = debug_mode_key(mode);
  data["mode_label"] = debug_mode_label(mode);
  data["source"] = "web";
  data["serial_mode_raw"] = -1;
  data["serial_mode_key"] = "offline";
  data["serial_mode_label"] = "offline playback";
  return data;
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
  const auto config_path = tools::resolve_config_path_string(cli.get<std::string>("config-path"));
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
      web_debugger->set_runtime_config_path(config_path);
      web_debugger->set_plot_history_limit(600);
      web_debugger->set_selected_mode(static_cast<int>(DebugMode::AutoAim));
      tools::logger()->info(
        "Web debugger listening on {}:{} (open {})", web_host, web_port, web_debugger->url());
      tools::logger()->info(
        "Web debugger config: fps={} scale={} jpeg={} ttl={}ms", web_fps, display_scale,
        web_jpeg_quality, web_client_ttl.count());
      tools::logger()->info(
        "Web runtime params bound to config: {}", tools::resolve_config_path_string(config_path));
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
  auto_aim::Solver auto_aim_solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, auto_aim_solver);
  auto debug_planner = std::make_unique<auto_aim::Planner>(config_path);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto small_target = std::make_unique<auto_buff::SmallTarget>();
  auto big_target = std::make_unique<auto_buff::BigTarget>();
  auto buff_aimer = std::make_unique<auto_buff::Aimer>(config_path);

  DebugMode current_mode = web_debugger ? clamp_debug_mode(web_debugger->selected_mode()) :
    DebugMode::AutoAim;
  DebugMode last_mode = current_mode;

  cv::Mat img;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);
  auto playback_start = std::chrono::steady_clock::now();
  auto last_web_frame_time = std::chrono::steady_clock::now() - web_frame_interval;
  auto last_web_state_time = std::chrono::steady_clock::now() - web_state_interval;
  double first_t = -1.0;
  constexpr int kFpsWindowSize = 30;
  std::array<double, kFpsWindowSize> fps_history {};
  int fps_history_idx = 0;
  int fps_sample_count = 0;
  auto last_frame_time = std::chrono::steady_clock::now();

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
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    if (has_timestamp_file) {
      text >> t >> qw >> qx >> qy >> qz;
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
    const double frame_dt = tools::delta_time(frame_start, last_frame_time);
    last_frame_time = frame_start;
    if (frame_dt > 0.0) {
      fps_history[fps_history_idx] = 1.0 / frame_dt;
      fps_history_idx = (fps_history_idx + 1) % kFpsWindowSize;
      if (fps_sample_count < kFpsWindowSize) ++fps_sample_count;
    }
    double smoothed_fps = 0.0;
    if (fps_sample_count > 0) {
      double sum = 0.0;
      for (int i = 0; i < fps_sample_count; ++i) sum += fps_history[i];
      smoothed_fps = sum / fps_sample_count;
    }
    auto timestamp =
      playback_start + std::chrono::microseconds(static_cast<int64_t>(relative_t * 1e6));
    const Eigen::Quaterniond gimbal_q(qw, qx, qy, qz);
    const Eigen::Vector3d gimbal_ypr =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

    current_mode = web_debugger ? clamp_debug_mode(web_debugger->selected_mode()) :
      DebugMode::AutoAim;
    if (current_mode != last_mode) {
      tools::logger()->info(
        "Switch offline web mode: {} -> {}",
        debug_mode_label(last_mode), debug_mode_label(current_mode));
      tracker = std::make_unique<auto_aim::Tracker>(config_path, auto_aim_solver);
      debug_planner = std::make_unique<auto_aim::Planner>(config_path);
      small_target = std::make_unique<auto_buff::SmallTarget>();
      big_target = std::make_unique<auto_buff::BigTarget>();
      buff_aimer = std::make_unique<auto_buff::Aimer>(config_path);
      last_mode = current_mode;
    }

    nlohmann::json data;
    data["mode"] = static_cast<int>(current_mode);
    data["gimbal_yaw"] = gimbal_ypr[0] * 57.3;
    data["gimbal_pitch"] = gimbal_ypr[1] * 57.3;
    data["bullet_speed"] = bullet_speed;
    data["t"] = relative_t;

    auto_aim::Plan current_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    std::optional<auto_aim::Target> current_target;
    BallisticDiagnostic ballistic_diag;
    BuffModeResult buff_result;

    auto run_buff_pipeline = [&](auto & target_state) -> BuffModeResult {
      BuffModeResult result;
      buff_solver.set_R_gimbal2world(gimbal_q);
      result.power_rune = buff_detector.detect(img);
      buff_solver.solve(result.power_rune);
      target_state.get_target(result.power_rune, timestamp);
      auto target_copy = target_state;

      io::GimbalState offline_state{};
      offline_state.yaw = static_cast<float>(gimbal_ypr[0]);
      offline_state.pitch = static_cast<float>(gimbal_ypr[1]);
      offline_state.yaw_vel = 0.0f;
      offline_state.pitch_vel = 0.0f;
      offline_state.bullet_speed = static_cast<float>(bullet_speed);
      offline_state.bullet_count = 0;
      result.plan = buff_aimer->mpc_aim(target_copy, timestamp, offline_state, false);

      result.target_solved = !target_state.is_unsolve();
      result.overlay_stage =
        result.target_solved ? (result.plan.control ? "buff_control" : "buff_track") :
        (result.power_rune.has_value() ? "buff_detect" : "buff_idle");

      if (result.power_rune.has_value()) {
        const auto & p = result.power_rune.value();
        result.rune_yaw_deg = rad2deg(p.ypd_in_world[0]);
        result.rune_pitch_deg = rad2deg(p.ypd_in_world[1]);
        result.rune_dist_m = p.ypd_in_world[2];
        result.blade_yaw_deg = rad2deg(p.blade_ypd_in_world[0]);
        result.blade_pitch_deg = rad2deg(p.blade_ypd_in_world[1]);
        result.blade_dist_m = p.blade_ypd_in_world[2];
        result.buff_yaw_deg = rad2deg(p.ypr_in_world[0]);
        result.buff_pitch_deg = rad2deg(p.ypr_in_world[1]);
        result.buff_roll_deg = rad2deg(p.ypr_in_world[2]);
        result.plan.target_yaw = static_cast<float>(p.blade_ypd_in_world[0]);
        result.plan.target_pitch = static_cast<float>(-p.blade_ypd_in_world[1]);
      }

      if (result.target_solved) {
        const auto x_state = target_state.ekf_x();
        const auto aim_xyz = target_state.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
        const auto predicted_aim_xyz =
          target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
        const auto current_origin = target_state.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
        const auto predicted_origin = target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));

        result.aim_x_m = aim_xyz.x();
        result.aim_y_m = aim_xyz.y();
        result.aim_z_m = aim_xyz.z();
        result.predicted_aim_x_m = predicted_aim_xyz.x();
        result.predicted_aim_y_m = predicted_aim_xyz.y();
        result.predicted_aim_z_m = predicted_aim_xyz.z();
        result.target_angle_deg = rad2deg(x_state[5]);
        result.target_spd_deg_s = rad2deg(x_state[6]);
        result.target_spd_rad_s = x_state[6];
        result.current_projection = buff_solver.reproject_buff(current_origin, x_state[4], x_state[5]);
        const auto predicted_x = target_copy.ekf_x();
        result.predicted_projection =
          buff_solver.reproject_buff(predicted_origin, predicted_x[4], predicted_x[5]);

        if (x_state.size() >= 10) {
          result.fit_a_deg_s = rad2deg(x_state[7]);
          result.fit_w_rad_s = x_state[8];
          result.fit_fi_deg = rad2deg(x_state[9]);
        }
      }

      return result;
    };

    if (current_mode == DebugMode::AutoAim) {
      auto_aim_solver.set_R_gimbal2world(gimbal_q);

      const auto yolo_start = std::chrono::steady_clock::now();
      auto armors = yolo.detect(img, frame_count);

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

      const auto tracker_start = std::chrono::steady_clock::now();
      auto targets = tracker->track(armors, timestamp);
      if (!targets.empty()) current_target = targets.front();

      const auto planner_start = std::chrono::steady_clock::now();
      current_plan = debug_planner->plan(current_target, bullet_speed);
      const auto finish = std::chrono::steady_clock::now();
      const double processing_ms = tools::delta_time(finish, frame_start) * 1000.0;
      const double inference_ms = tools::delta_time(tracker_start, yolo_start) * 1e3;
      const double tracker_ms = tools::delta_time(planner_start, tracker_start) * 1e3;
      const double planner_ms = tools::delta_time(finish, planner_start) * 1e3;
      tools::logger()->info(
        "[{}][{}] yolo: {:.1f}ms, tracker: {:.1f}ms, planner: {:.1f}ms",
        frame_count, debug_mode_label(current_mode), inference_ms, tracker_ms, planner_ms);

      data["pipeline_pending"] = 0;
      data["pipeline_inflight"] = 0;
      data["pipeline_dropped"] = 0;
      data["pipeline_inference_latency_ms"] = inference_ms;

      data["target_yaw"] = rad2deg(current_plan.target_yaw);
      data["target_pitch"] = rad2deg(current_plan.target_pitch);
      data["plan_yaw"] = rad2deg(current_plan.yaw);
      data["plan_yaw_vel"] = rad2deg(current_plan.yaw_vel);
      data["plan_yaw_acc"] = rad2deg(current_plan.yaw_acc);
      data["plan_pitch"] = rad2deg(current_plan.pitch);
      data["plan_pitch_vel"] = rad2deg(current_plan.pitch_vel);
      data["plan_pitch_acc"] = rad2deg(current_plan.pitch_acc);
      data["cmd_yaw"] = rad2deg(current_plan.yaw);
      data["cmd_pitch"] = rad2deg(current_plan.pitch);
      data["shoot"] = current_plan.fire ? 1 : 0;
      data["fire"] = current_plan.fire ? 1 : 0;

      if (current_target.has_value()) {
        const auto & target = *current_target;
        const Eigen::VectorXd x_state = target.ekf_x();
        data["x"] = x_state[0];
        data["vx"] = x_state[1];
        data["y"] = x_state[2];
        data["vy"] = x_state[3];
        data["z"] = x_state[4];
        data["vz"] = x_state[5];
        const Eigen::Vector3d car_rpy = target.car_rpy();
        data["a"] = car_rpy[0] * 57.3;
        data["car_pitch"] = car_rpy[1] * 57.3;
        data["car_roll"] = car_rpy[2] * 57.3;
        data["w"] = x_state[7];
        data["r"] = target.radius(0);
        data["l"] = target.radius(1) - target.radius(0);
        data["h"] = x_state[10];
        data["last_id"] = target.last_id;
        data["residual_yaw"] = target.ekf().data.at("residual_yaw");
        data["residual_pitch"] = target.ekf().data.at("residual_pitch");
        data["residual_distance"] = target.ekf().data.at("residual_distance");
        data["residual_angle"] = target.ekf().data.at("residual_angle");
        data["nis"] = target.ekf().data.at("nis");
        data["nis_gate"] = target.nis_gate();
        data["update_accepted"] = target.ekf().data.at("update_accepted");
        data["nees"] = target.ekf().data.at("nees");
        data["nis_fail"] = target.ekf().data.at("nis_fail");
        data["nees_fail"] = target.ekf().data.at("nees_fail");
        data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
        data["uvl_left_center_u"] = target.ekf().data.at("uvl_left_center_u");
        data["uvl_left_center_v"] = target.ekf().data.at("uvl_left_center_v");
        data["uvl_right_center_u"] = target.ekf().data.at("uvl_right_center_u");
        data["uvl_right_center_v"] = target.ekf().data.at("uvl_right_center_v");
        data["tracker_match_valid"] = target.tracker_debug_match_valid ? 1 : 0;
        data["tracker_match_id"] = target.tracker_debug_match_id;
        data["tracker_match_score"] = target.tracker_debug_match_score;
        data["tracker_reprojection_px"] = target.tracker_debug_reprojection_px;
      } else {
        data["w"] = 0.0;
        data["h"] = 0.0;
        data["tracker_match_valid"] = 0;
        data["tracker_match_id"] = -1;
        data["tracker_match_score"] = -1.0;
        data["tracker_reprojection_px"] = -1.0;
      }

      data["planner_selected_armor"] = debug_planner->debug_armor_id;
      data["planner_delay_ms"] = debug_planner->debug_delay_time * 1000.0;
      data["planner_hit_fly_time_ms"] = debug_planner->debug_hit_fly_time * 1000.0;
      data["planner_hit_iters"] = debug_planner->debug_hit_iter_count;
      data["planner_hit_converged"] = debug_planner->debug_hit_converged ? 1 : 0;
      data["planner_yaw_solver_iterations"] = debug_planner->debug_yaw_solver_iterations;
      data["planner_pitch_solver_iterations"] = debug_planner->debug_pitch_solver_iterations;
      data["planner_spin_gate"] = debug_planner->debug_used_spin_gate ? 1 : 0;
      data["planner_center_yaw"] = rad2deg(debug_planner->debug_center_yaw);
      data["planner_turn_sign"] =
        tools::debug::spin_direction_sign(current_target.has_value() ? current_target->ekf_x()[7] : 0.0);
      data["planner_selected_z_offset"] = debug_planner->debug_selected_z_offset;
      data["planner_selected_aim_z_compensation"] =
        debug_planner->debug_selected_aim_z_compensation;
      data["planner_fixed_model"] = debug_planner->debug_fixed_center_rotation_model ? 1 : 0;

      if (current_target.has_value() && current_plan.control) {
        ballistic_diag = build_ballistic_diagnostic(
          current_plan, debug_planner->debug_xyza, current_target->armor_type, bullet_speed,
          yaw_offset, pitch_offset);
      }

      const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
      const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
      const nlohmann::json overlay_config =
        web_debugger ? web_debugger->overlay_config() : nlohmann::json::object();
      const auto now = std::chrono::steady_clock::now();
      const bool need_web_frame =
        web_debugger && web_debugger->has_active_client(web_client_ttl) &&
        (now - last_web_frame_time >= web_frame_interval);
      const bool need_visual_output = show_local || need_web_frame;

      plotter.plot(data);
      if (web_debugger) web_debugger->update_plot_sample(data);

      if (web_debugger && now - last_web_state_time >= web_state_interval) {
        nlohmann::json web_state;
        web_state["server"]["unix_ms"] = unix_time_ms();
        web_state["mode"] = build_offline_mode_state(current_mode);
        web_state["frame"]["latency_ms"] = processing_ms;
        web_state["frame"]["image_width"] = img.cols;
        web_state["frame"]["image_height"] = img.rows;
        web_state["frame"]["playback_t_s"] = relative_t;
        web_state["frame"]["raw_t_s"] = t;
        web_state["frame"]["frame_index"] = frame_count;
        web_state["frame"]["bullet_speed_mps"] = bullet_speed;
        web_state["frame"]["bullet_speed_effective_mps"] = effective_bullet_speed(current_mode, bullet_speed);
        web_state["frame"]["bullet_speed_fallback"] = bullet_speed_fallback(current_mode, bullet_speed);
        web_state["frame"]["bullet_speed_source"] = "offline-cli";
        web_state["pipeline"]["pending"] = 0;
        web_state["pipeline"]["inflight"] = 0;
        web_state["pipeline"]["max_inflight"] = 1;
        web_state["pipeline"]["submitted"] = frame_count;
        web_state["pipeline"]["dropped"] = 0;
        web_state["pipeline"]["inference_latency_ms"] = inference_ms;
        web_state["pipeline"]["tracker_latency_ms"] = tracker_ms;
        web_state["pipeline"]["planner_latency_ms"] = planner_ms;
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
          web_state["preview"]["target_x_m"] = debug_planner->debug_xyza[0];
          web_state["preview"]["target_y_m"] = debug_planner->debug_xyza[1];
          web_state["preview"]["target_z_m"] = debug_planner->debug_xyza[2];
        } else {
          web_state["preview"]["target_x_m"] = nullptr;
          web_state["preview"]["target_y_m"] = nullptr;
          web_state["preview"]["target_z_m"] = nullptr;
        }
        web_state["planner"]["selected_armor"] = debug_planner->debug_armor_id;
        web_state["planner"]["physical_armor"] = debug_planner->debug_physical_armor_id;
        web_state["planner"]["spin_gate"] = debug_planner->debug_used_spin_gate;
        web_state["planner"]["delay_ms"] = debug_planner->debug_delay_time * 1000.0;
        web_state["planner"]["hit_fly_time_ms"] = debug_planner->debug_hit_fly_time * 1000.0;
        web_state["planner"]["hit_iter_count"] = debug_planner->debug_hit_iter_count;
        web_state["planner"]["hit_converged"] = debug_planner->debug_hit_converged;
        web_state["planner"]["center_yaw_deg"] = rad2deg(debug_planner->debug_center_yaw);
        web_state["planner"]["turn_direction"] =
          tools::debug::spin_direction_to_string(current_w);
        web_state["planner"]["turn_sign"] =
          tools::debug::spin_direction_sign(current_w);
        web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
        for (const double delta_angle : debug_planner->debug_delta_angle_list) {
          web_state["planner"]["delta_angle_deg_list"].push_back(rad2deg(delta_angle));
        }
        web_state["planner"]["w_rad_s"] = current_w;
        web_state["planner"]["h_m"] = current_h;
        web_state["planner"]["selected_z_offset_m"] = debug_planner->debug_selected_z_offset;
        web_state["planner"]["selected_aim_z_compensation_m"] =
          debug_planner->debug_selected_aim_z_compensation;
        web_state["planner"]["fixed_center_rotation_model"] =
          debug_planner->debug_fixed_center_rotation_model;
        web_state["planner"].update(tools::debug::mpc_to_json(*debug_planner));
        web_state["estimator"] =
          tools::debug::estimator_to_json(current_target ? &*current_target : nullptr);
        if (current_target.has_value()) {
          web_state["tracker"]["candidate_count"] = current_target->tracker_debug_candidate_count;
          web_state["tracker"]["match_valid"] = current_target->tracker_debug_match_valid;
          web_state["tracker"]["match_id"] = current_target->tracker_debug_match_id;
          web_state["tracker"]["match_score"] = current_target->tracker_debug_match_score;
          web_state["tracker"]["reprojection_px"] = current_target->tracker_debug_reprojection_px;
          web_state["tracker"]["xy_error_m"] = current_target->tracker_debug_xy_error_m;
          web_state["tracker"]["z_error_m"] = current_target->tracker_debug_z_error_m;
        } else {
          web_state["tracker"]["candidate_count"] = 0;
          web_state["tracker"]["match_valid"] = false;
          web_state["tracker"]["match_id"] = -1;
          web_state["tracker"]["match_score"] = nullptr;
          web_state["tracker"]["reprojection_px"] = nullptr;
          web_state["tracker"]["xy_error_m"] = nullptr;
          web_state["tracker"]["z_error_m"] = nullptr;
        }
        web_state["overlay"]["stage"] =
          tools::debug_visualization::live_overlay_stage_to_string(
            tools::debug_visualization::resolve_live_overlay_stage(
              current_target.has_value(), current_plan));
        web_state["overlay"]["controls"] =
          overlay_config.is_object() ? overlay_config : nlohmann::json::object();
        web_state["ballistic"] = ballistic_to_json(ballistic_diag);
        web_state["command"]["has_target"] = current_target.has_value();
        web_state["command"]["fire"] = current_plan.fire;
        web_state["command"]["fired"] = current_plan.fire;
        web_state["command"]["gimbal_source_unit"] = "rad";
        web_state["command"]["gimbal_yaw_raw"] = gimbal_ypr[0];
        web_state["command"]["gimbal_yaw_deg"] = rad2deg(gimbal_ypr[0]);
        web_state["command"]["gimbal_yaw_rad"] = gimbal_ypr[0];
        web_state["command"]["gimbal_yaw_vel_raw"] = nullptr;
        web_state["command"]["gimbal_yaw_vel_deg"] = nullptr;
        web_state["command"]["gimbal_yaw_vel_rad"] = nullptr;
        web_state["command"]["gimbal_pitch_raw"] = gimbal_ypr[1];
        web_state["command"]["gimbal_pitch_deg"] = rad2deg(gimbal_ypr[1]);
        web_state["command"]["gimbal_pitch_rad"] = gimbal_ypr[1];
        web_state["command"]["gimbal_pitch_vel_raw"] = nullptr;
        web_state["command"]["gimbal_pitch_vel_deg"] = nullptr;
        web_state["command"]["gimbal_pitch_vel_rad"] = nullptr;
        web_state["command"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
        web_state["command"]["plan_yaw_rad"] = current_plan.yaw;
        web_state["command"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
        web_state["command"]["plan_pitch_rad"] = current_plan.pitch;
        web_state["command"]["plan_yaw_vel_deg"] = rad2deg(current_plan.yaw_vel);
        web_state["command"]["plan_yaw_vel_rad"] = current_plan.yaw_vel;
        web_state["command"]["plan_pitch_vel_deg"] = rad2deg(current_plan.pitch_vel);
        web_state["command"]["plan_pitch_vel_rad"] = current_plan.pitch_vel;
        web_state["command"]["plan_yaw_acc_deg"] = rad2deg(current_plan.yaw_acc);
        web_state["command"]["plan_yaw_acc_rad"] = current_plan.yaw_acc;
        web_state["command"]["plan_pitch_acc_deg"] = rad2deg(current_plan.pitch_acc);
        web_state["command"]["plan_pitch_acc_rad"] = current_plan.pitch_acc;
        web_state["command"]["target_yaw_deg"] = rad2deg(current_plan.target_yaw);
        web_state["command"]["target_yaw_rad"] = current_plan.target_yaw;
        web_state["command"]["target_pitch_deg"] = rad2deg(current_plan.target_pitch);
        web_state["command"]["target_pitch_rad"] = current_plan.target_pitch;
        web_state["command"]["bullet_speed_mps"] = bullet_speed;
        web_state["command"]["bullet_speed_effective_mps"] =
          effective_bullet_speed(current_mode, bullet_speed);
        web_state["command"]["bullet_speed_fallback"] =
          bullet_speed_fallback(current_mode, bullet_speed);
        web_state["command"]["bullet_speed_source"] = "offline-cli";
        web_debugger->update_state(web_state);
        web_debugger->update_log(web_state);
        last_web_state_time = now;
      }

      if (need_visual_output) {
        tools::debug_visualization::LiveOverlayOptions visual_options;
        visual_options.display_scale = display_scale;
        visual_options.latency_ms = processing_ms;
        visual_options.fps = smoothed_fps;
        visual_options.target_name =
          current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
        visual_options.armor_type =
          current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
        visual_options.planner_armor_id = debug_planner->debug_armor_id;
        visual_options.planner_physical_armor_id = debug_planner->debug_physical_armor_id;
        visual_options.planner_spin_gate = debug_planner->debug_used_spin_gate;
        visual_options.planner_delay_ms = debug_planner->debug_delay_time * 1000.0;
        visual_options.planner_center_yaw_deg = rad2deg(debug_planner->debug_center_yaw);
        visual_options.planner_hit_fly_time_ms = debug_planner->debug_hit_fly_time * 1000.0;
        visual_options.planner_hit_iter_count = debug_planner->debug_hit_iter_count;
        visual_options.planner_hit_converged = debug_planner->debug_hit_converged;
        for (const double delta_angle : debug_planner->debug_delta_angle_list) {
          visual_options.planner_delta_angles_deg.push_back(rad2deg(delta_angle));
        }
        visual_options.planner_turn_direction =
          tools::debug::spin_direction_to_string(current_w);
        visual_options.planner_turn_sign =
          tools::debug::spin_direction_sign(current_w);
        visual_options.tracker_candidate_count =
          current_target.has_value() ? current_target->tracker_debug_candidate_count : 0;
        visual_options.tracker_match_valid =
          current_target.has_value() && current_target->tracker_debug_match_valid;
        visual_options.tracker_match_id =
          current_target.has_value() ? current_target->tracker_debug_match_id : -1;
        visual_options.tracker_match_score =
          current_target.has_value() ? current_target->tracker_debug_match_score : -1.0;
        visual_options.tracker_reprojection_px =
          current_target.has_value() ? current_target->tracker_debug_reprojection_px : -1.0;
        visual_options.tracker_xy_error_m =
          current_target.has_value() ? current_target->tracker_debug_xy_error_m : -1.0;
        visual_options.tracker_z_error_m =
          current_target.has_value() ? current_target->tracker_debug_z_error_m : -1.0;
        visual_options.current_w = current_w;
        visual_options.current_h = current_h;
        visual_options.current_selected_z_offset = debug_planner->debug_selected_z_offset;
        visual_options.current_selected_aim_z_compensation =
          debug_planner->debug_selected_aim_z_compensation;
        visual_options.current_fixed_model = debug_planner->debug_fixed_center_rotation_model;
        visual_options.target_jumped =
          current_target.has_value() && current_target->jumped;
        visual_options.is_outpost =
          current_target.has_value() &&
          current_target->name == auto_aim::ArmorName::outpost;
        visual_options.stabilize_annotations = overlay_flag(overlay_config, "stabilize", true);
        visual_options.enable_state_layers = overlay_flag(overlay_config, "state_layers", true);
        visual_options.show_armors = overlay_flag(overlay_config, "armors", true);
        visual_options.show_armor_labels = overlay_flag(overlay_config, "labels", true);
        visual_options.show_target_motion = overlay_flag(overlay_config, "target_motion", true);
        visual_options.show_aim = overlay_flag(overlay_config, "aim", true);
        visual_options.show_decision_hud = overlay_flag(overlay_config, "decision_hud", true);
        visual_options.show_decision_track = overlay_flag(overlay_config, "decision_track", true);
        visual_options.show_footer = overlay_flag(overlay_config, "footer", true);

        const auto display_img = tools::debug_visualization::render_live_debug_frame(
          img, auto_aim_solver, current_target, current_plan, *debug_planner, visual_options);
        draw_ballistic_panel(ballistic_panel, ballistic_diag);

        if (need_web_frame && web_debugger) {
          web_debugger->update_main_frame(display_img, web_jpeg_quality);
          web_debugger->update_ballistic_frame(ballistic_panel, web_jpeg_quality);
          last_web_frame_time = now;
        }

        if (show_local) {
          cv::imshow("Offline Test Web", display_img);
          cv::imshow("Offline Test Panel", ballistic_panel);
          const auto key = cv::waitKey(1);
          if (key == 'q') break;
        }
      }
    } else {
      buff_result =
        current_mode == DebugMode::SmallBuff ? run_buff_pipeline(*small_target) :
        run_buff_pipeline(*big_target);
      current_plan = buff_result.plan;
      const auto finish = std::chrono::steady_clock::now();
      const double processing_ms = tools::delta_time(finish, frame_start) * 1000.0;

      data["target_yaw"] = buff_result.blade_yaw_deg;
      data["target_pitch"] = -buff_result.blade_pitch_deg;
      data["plan_yaw"] = rad2deg(current_plan.yaw);
      data["plan_pitch"] = rad2deg(current_plan.pitch);
      data["plan_yaw_vel"] = rad2deg(current_plan.yaw_vel);
      data["plan_pitch_vel"] = rad2deg(current_plan.pitch_vel);
      data["plan_yaw_acc"] = rad2deg(current_plan.yaw_acc);
      data["plan_pitch_acc"] = rad2deg(current_plan.pitch_acc);
      data["cmd_yaw"] = rad2deg(current_plan.yaw);
      data["cmd_pitch"] = rad2deg(current_plan.pitch);
      data["shoot"] = current_plan.fire ? 1 : 0;
      data["fire"] = current_plan.fire ? 1 : 0;
      data["R_yaw"] = buff_result.rune_yaw_deg;
      data["R_pitch"] = buff_result.rune_pitch_deg;
      data["R_dis"] = buff_result.rune_dist_m;
      data["blade_yaw"] = buff_result.blade_yaw_deg;
      data["blade_pitch"] = buff_result.blade_pitch_deg;
      data["blade_dis"] = buff_result.blade_dist_m;
      data["buff_yaw"] = buff_result.buff_yaw_deg;
      data["buff_pitch"] = buff_result.buff_pitch_deg;
      data["buff_roll"] = buff_result.buff_roll_deg;
      data["angle"] = buff_result.target_angle_deg;
      data["spd"] = buff_result.target_spd_deg_s;
      data["w"] = buff_result.target_spd_rad_s;
      data["a"] = buff_result.fit_a_deg_s;
      data["fi"] = buff_result.fit_fi_deg;
      data["target_z"] = buff_result.aim_z_m;
      data["planner_turn_sign"] =
        tools::debug::spin_direction_sign(buff_result.target_spd_rad_s);

      tools::logger()->info(
        "[{}][{}] detect+solve+aim: {:.1f}ms",
        frame_count, debug_mode_label(current_mode), processing_ms);

      plotter.plot(data);
      if (web_debugger) web_debugger->update_plot_sample(data);

      const nlohmann::json overlay_config =
        web_debugger ? web_debugger->overlay_config() : nlohmann::json::object();
      const auto now = std::chrono::steady_clock::now();
      const bool need_web_frame =
        web_debugger && web_debugger->has_active_client(web_client_ttl) &&
        (now - last_web_frame_time >= web_frame_interval);
      const bool need_visual_output = show_local || need_web_frame;

      if (web_debugger && now - last_web_state_time >= web_state_interval) {
        nlohmann::json web_state;
        web_state["server"]["unix_ms"] = unix_time_ms();
        web_state["mode"] = build_offline_mode_state(current_mode);
        web_state["frame"]["latency_ms"] = processing_ms;
        web_state["frame"]["image_width"] = img.cols;
        web_state["frame"]["image_height"] = img.rows;
        web_state["frame"]["playback_t_s"] = relative_t;
        web_state["frame"]["raw_t_s"] = t;
        web_state["frame"]["frame_index"] = frame_count;
        web_state["frame"]["bullet_speed_mps"] = bullet_speed;
        web_state["frame"]["bullet_speed_effective_mps"] = effective_bullet_speed(current_mode, bullet_speed);
        web_state["frame"]["bullet_speed_fallback"] = bullet_speed_fallback(current_mode, bullet_speed);
        web_state["frame"]["bullet_speed_source"] = "offline-cli";
        web_state["overlay"]["stage"] = buff_result.overlay_stage;
        web_state["overlay"]["controls"] =
          overlay_config.is_object() ? overlay_config : nlohmann::json::object();

        web_state["preview"]["has_target"] = buff_result.target_solved;
        web_state["preview"]["fire"] = current_plan.fire;
        web_state["preview"]["target_name"] = debug_mode_label(current_mode);
        web_state["preview"]["armor_type"] = "buff";
        web_state["preview"]["target_yaw_deg"] = buff_result.blade_yaw_deg;
        web_state["preview"]["target_yaw_rad"] =
          buff_result.power_rune.has_value() ? buff_result.power_rune->blade_ypd_in_world[0] : current_plan.target_yaw;
        web_state["preview"]["target_pitch_deg"] = -buff_result.blade_pitch_deg;
        web_state["preview"]["target_pitch_rad"] =
          buff_result.power_rune.has_value() ? -buff_result.power_rune->blade_ypd_in_world[1] : current_plan.target_pitch;
        web_state["preview"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
        web_state["preview"]["plan_yaw_rad"] = current_plan.yaw;
        web_state["preview"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
        web_state["preview"]["plan_pitch_rad"] = current_plan.pitch;
        web_state["preview"]["target_x_m"] = buff_result.aim_x_m;
        web_state["preview"]["target_y_m"] = buff_result.aim_y_m;
        web_state["preview"]["target_z_m"] = buff_result.aim_z_m;

        web_state["planner"]["selected_armor"] = -1;
        web_state["planner"]["physical_armor"] = -1;
        web_state["planner"]["spin_gate"] = false;
        web_state["planner"]["delay_ms"] = 0.0;
        web_state["planner"]["center_yaw_deg"] = 0.0;
        web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
        web_state["planner"]["turn_direction"] =
          tools::debug::spin_direction_to_string(buff_result.target_spd_rad_s);
        web_state["planner"]["turn_sign"] =
          tools::debug::spin_direction_sign(buff_result.target_spd_rad_s);
        web_state["planner"]["w_rad_s"] = buff_result.target_spd_rad_s;
        web_state["planner"]["h_m"] = buff_result.aim_z_m;
        web_state["planner"]["selected_z_offset_m"] = 0.0;
        web_state["planner"]["selected_aim_z_compensation_m"] = 0.0;
        web_state["planner"]["fixed_center_rotation_model"] = false;

        web_state["tracker"]["candidate_count"] = buff_result.power_rune.has_value() ? 1 : 0;
        web_state["tracker"]["match_valid"] = buff_result.target_solved;
        web_state["tracker"]["match_id"] = buff_result.target_solved ? 0 : -1;
        if (buff_result.power_rune.has_value()) {
          web_state["tracker"]["match_score"] = 1.0;
        } else {
          web_state["tracker"]["match_score"] = nullptr;
        }
        web_state["tracker"]["reprojection_px"] = nullptr;
        web_state["tracker"]["xy_error_m"] = nullptr;
        web_state["tracker"]["z_error_m"] = nullptr;

        web_state["ballistic"]["valid"] = current_plan.control;
        web_state["ballistic"]["unsolvable"] = !current_plan.control;
        web_state["ballistic"]["hit"] = current_plan.fire;
        web_state["ballistic"]["bullet_speed_raw_mps"] = bullet_speed;
        web_state["ballistic"]["bullet_speed_effective_mps"] =
          effective_bullet_speed(current_mode, bullet_speed);
        web_state["ballistic"]["bullet_speed_mps"] =
          effective_bullet_speed(current_mode, bullet_speed);
        web_state["ballistic"]["bullet_speed_fallback"] =
          bullet_speed_fallback(current_mode, bullet_speed);
        web_state["ballistic"]["target_dist_xy_m"] =
          std::sqrt(buff_result.aim_x_m * buff_result.aim_x_m + buff_result.aim_y_m * buff_result.aim_y_m);
        web_state["ballistic"]["target_dist_3d_m"] =
          std::sqrt(
            buff_result.aim_x_m * buff_result.aim_x_m +
            buff_result.aim_y_m * buff_result.aim_y_m +
            buff_result.aim_z_m * buff_result.aim_z_m);
        web_state["ballistic"]["target_height_m"] = buff_result.aim_z_m;

        web_state["buff"]["has_detection"] = buff_result.power_rune.has_value();
        web_state["buff"]["target_solved"] = buff_result.target_solved;
        web_state["buff"]["rune_yaw_deg"] = buff_result.rune_yaw_deg;
        web_state["buff"]["rune_pitch_deg"] = buff_result.rune_pitch_deg;
        web_state["buff"]["rune_dist_m"] = buff_result.rune_dist_m;
        web_state["buff"]["blade_yaw_deg"] = buff_result.blade_yaw_deg;
        web_state["buff"]["blade_pitch_deg"] = buff_result.blade_pitch_deg;
        web_state["buff"]["blade_dist_m"] = buff_result.blade_dist_m;
        web_state["buff"]["buff_yaw_deg"] = buff_result.buff_yaw_deg;
        web_state["buff"]["buff_pitch_deg"] = buff_result.buff_pitch_deg;
        web_state["buff"]["buff_roll_deg"] = buff_result.buff_roll_deg;
        web_state["buff"]["angle_deg"] = buff_result.target_angle_deg;
        web_state["buff"]["spd_deg_s"] = buff_result.target_spd_deg_s;
        web_state["buff"]["spd_rad_s"] = buff_result.target_spd_rad_s;
        web_state["buff"]["fit_a_deg_s"] = buff_result.fit_a_deg_s;
        web_state["buff"]["fit_w_rad_s"] = buff_result.fit_w_rad_s;
        web_state["buff"]["fit_fi_deg"] = buff_result.fit_fi_deg;
        web_state["buff"]["aim_x_m"] = buff_result.aim_x_m;
        web_state["buff"]["aim_y_m"] = buff_result.aim_y_m;
        web_state["buff"]["aim_z_m"] = buff_result.aim_z_m;
        web_state["buff"]["predicted_aim_x_m"] = buff_result.predicted_aim_x_m;
        web_state["buff"]["predicted_aim_y_m"] = buff_result.predicted_aim_y_m;
        web_state["buff"]["predicted_aim_z_m"] = buff_result.predicted_aim_z_m;

        web_state["command"]["has_target"] = buff_result.target_solved;
        web_state["command"]["fire"] = current_plan.fire;
        web_state["command"]["fired"] = current_plan.fire;
        web_state["command"]["gimbal_source_unit"] = "rad";
        web_state["command"]["gimbal_yaw_raw"] = gimbal_ypr[0];
        web_state["command"]["gimbal_yaw_deg"] = rad2deg(gimbal_ypr[0]);
        web_state["command"]["gimbal_yaw_rad"] = gimbal_ypr[0];
        web_state["command"]["gimbal_yaw_vel_raw"] = 0.0;
        web_state["command"]["gimbal_yaw_vel_deg"] = 0.0;
        web_state["command"]["gimbal_yaw_vel_rad"] = 0.0;
        web_state["command"]["gimbal_pitch_raw"] = gimbal_ypr[1];
        web_state["command"]["gimbal_pitch_deg"] = rad2deg(gimbal_ypr[1]);
        web_state["command"]["gimbal_pitch_rad"] = gimbal_ypr[1];
        web_state["command"]["gimbal_pitch_vel_raw"] = 0.0;
        web_state["command"]["gimbal_pitch_vel_deg"] = 0.0;
        web_state["command"]["gimbal_pitch_vel_rad"] = 0.0;
        web_state["command"]["target_yaw_deg"] = buff_result.blade_yaw_deg;
        web_state["command"]["target_yaw_rad"] =
          buff_result.power_rune.has_value() ? buff_result.power_rune->blade_ypd_in_world[0] : current_plan.target_yaw;
        web_state["command"]["target_pitch_deg"] = -buff_result.blade_pitch_deg;
        web_state["command"]["target_pitch_rad"] =
          buff_result.power_rune.has_value() ? -buff_result.power_rune->blade_ypd_in_world[1] : current_plan.target_pitch;
        web_state["command"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
        web_state["command"]["plan_yaw_rad"] = current_plan.yaw;
        web_state["command"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
        web_state["command"]["plan_pitch_rad"] = current_plan.pitch;
        web_state["command"]["plan_yaw_vel_deg"] = rad2deg(current_plan.yaw_vel);
        web_state["command"]["plan_yaw_vel_rad"] = current_plan.yaw_vel;
        web_state["command"]["plan_pitch_vel_deg"] = rad2deg(current_plan.pitch_vel);
        web_state["command"]["plan_pitch_vel_rad"] = current_plan.pitch_vel;
        web_state["command"]["plan_yaw_acc_deg"] = rad2deg(current_plan.yaw_acc);
        web_state["command"]["plan_yaw_acc_rad"] = current_plan.yaw_acc;
        web_state["command"]["plan_pitch_acc_deg"] = rad2deg(current_plan.pitch_acc);
        web_state["command"]["plan_pitch_acc_rad"] = current_plan.pitch_acc;
        web_state["command"]["bullet_speed_mps"] = bullet_speed;
        web_state["command"]["bullet_speed_effective_mps"] =
          effective_bullet_speed(current_mode, bullet_speed);
        web_state["command"]["bullet_speed_fallback"] =
          bullet_speed_fallback(current_mode, bullet_speed);
        web_state["command"]["bullet_speed_source"] = "offline-cli";

        web_debugger->update_state(web_state);
        web_debugger->update_log(web_state);
        last_web_state_time = now;
      }

      if (need_visual_output) {
        const auto display_img = render_buff_debug_frame(
          img, buff_result, overlay_config, current_mode, display_scale, processing_ms,
          bullet_speed);
        draw_buff_panel(
          ballistic_panel, current_mode, buff_result, gimbal_ypr, processing_ms, bullet_speed,
          effective_bullet_speed(current_mode, bullet_speed),
          bullet_speed_fallback(current_mode, bullet_speed));

        if (need_web_frame && web_debugger) {
          web_debugger->update_main_frame(display_img, web_jpeg_quality);
          web_debugger->update_ballistic_frame(ballistic_panel, web_jpeg_quality);
          last_web_frame_time = now;
        }

        if (show_local) {
          cv::imshow("Offline Test Web", display_img);
          cv::imshow("Offline Test Panel", ballistic_panel);
          const auto key = cv::waitKey(1);
          if (key == 'q') break;
        }
      }
    }
  }

  if (show_local) cv::destroyAllWindows();
  return 0;
}
