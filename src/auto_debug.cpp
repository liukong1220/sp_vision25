#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
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
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/web_debugger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;
using tools::debug::BallisticDiagnostic;
using tools::debug::GimbalStateUnitMode;
using tools::debug::NormalizedGimbalState;
using tools::debug::armor_name_to_string;
using tools::debug::armor_type_to_string;
using tools::debug::ballistic_to_json;
using tools::debug::build_ballistic_diagnostic;
using tools::debug::draw_ballistic_panel;
using tools::debug::has_cli_option;
using tools::debug::normalize_gimbal_state;
using tools::debug::parse_gimbal_state_unit_mode;
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

DebugMode clamp_debug_mode(int mode)
{
  if (mode <= static_cast<int>(DebugMode::AutoAim)) return DebugMode::AutoAim;
  if (mode >= static_cast<int>(DebugMode::BigBuff)) return DebugMode::BigBuff;
  return static_cast<DebugMode>(mode);
}

bool is_buff_mode(DebugMode mode)
{
  return mode == DebugMode::SmallBuff || mode == DebugMode::BigBuff;
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

const char * gimbal_mode_key(io::GimbalMode mode)
{
  switch (mode) {
    case io::GimbalMode::AUTO_AIM:
      return "auto_aim";
    case io::GimbalMode::SMALL_BUFF:
      return "small_buff";
    case io::GimbalMode::BIG_BUFF:
      return "big_buff";
    case io::GimbalMode::IDLE:
    default:
      return "idle";
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
  const NormalizedGimbalState & normalized_gimbal, double latency_ms, double raw_bullet_speed,
  double effective_speed, bool speed_fallback, bool fired)
{
  panel.setTo(cv::Scalar(8, 18, 28));
  cv::rectangle(panel, {0, 0, panel.cols, 58}, cv::Scalar(22, 48, 76), cv::FILLED);
  cv::putText(
    panel, fmt::format("{} Diagnostic", debug_mode_label(mode)), {24, 38},
    cv::FONT_HERSHEY_SIMPLEX, 0.92, {235, 246, 255}, 2, cv::LINE_AA);

  draw_panel_line(panel, 0, "Latency", fmt::format("{:.1f} ms", latency_ms));
  draw_panel_line(
    panel, 1, "Gimbal",
    fmt::format(
      "yaw {:+.2f} deg   pitch {:+.2f} deg",
      normalized_gimbal.yaw.deg, normalized_gimbal.pitch.deg));
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
      fired ? "YES" : "NO"),
    buff_result.plan.fire ? cv::Scalar(120, 180, 255) : cv::Scalar(220, 240, 255));
}

cv::Mat render_buff_debug_frame(
  const cv::Mat & source, const BuffModeResult & buff_result, const nlohmann::json & overlay_config,
  DebugMode mode, double display_scale, double latency_ms, double raw_bullet_speed)
{
  cv::Mat display = source.clone();
  if (display.empty()) return display;

  const bool show_armors = overlay_flag(overlay_config, "armors", true);
  const bool show_labels = overlay_flag(overlay_config, "labels", true);
  const bool show_aim = overlay_flag(overlay_config, "aim", true);
  const bool show_footer = overlay_flag(overlay_config, "footer", true);
  const bool show_hud = overlay_flag(overlay_config, "decision_hud", true);

  if (show_armors && buff_result.power_rune.has_value()) {
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
        "latency {:.1f} ms   bullet {:.2f} m/s   control {}",
        latency_ms, raw_bullet_speed, buff_result.plan.control ? "ON" : "OFF"),
      {34, 78}, cv::FONT_HERSHEY_SIMPLEX, 0.56, {154, 194, 215}, 1, cv::LINE_AA);
    cv::putText(
      display,
      buff_result.target_solved ?
        fmt::format(
          "blade {:+.2f} / {:+.2f} deg   fire {}",
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

nlohmann::json build_mode_state(DebugMode mode, io::GimbalMode serial_mode, const io::Gimbal & gimbal)
{
  nlohmann::json data;
  data["mode"] = static_cast<int>(mode);
  data["mode_key"] = debug_mode_key(mode);
  data["mode_label"] = debug_mode_label(mode);
  data["source"] = "web";
  data["serial_mode_raw"] = static_cast<int>(serial_mode);
  data["serial_mode_key"] = gimbal_mode_key(serial_mode);
  data["serial_mode_label"] = gimbal.str(serial_mode);
  return data;
}
}  // namespace

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
  "{web-client-ttl-ms | 2000                | 最近访问多久内继续渲染网页帧(显式传参时覆盖yaml) }"
  "{record-raw-video | false                | 录制原始相机画面(显式传参时覆盖yaml) }"
  "{record-debug-video | false              | 录制主调试画面(显式传参时覆盖yaml) }"
  "{record-debug-fps | 30.0                 | 调试录制帧率(显式传参时覆盖yaml) }"
  "{record-debug-dir | records              | 调试录制输出目录(显式传参时覆盖yaml) }";

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
  const bool record_raw_video = has_cli_option(argc, argv, "record-raw-video") ?
    cli.get<bool>("record-raw-video") : tools::read_or<bool>(yaml, "record_raw_video", false);
  const bool record_debug_video = has_cli_option(argc, argv, "record-debug-video") ?
    cli.get<bool>("record-debug-video") : tools::read_or<bool>(yaml, "record_debug_video", false);
  const double record_debug_fps = std::clamp(
    has_cli_option(argc, argv, "record-debug-fps") ?
      cli.get<double>("record-debug-fps") : tools::read_or<double>(yaml, "record_debug_fps", 30.0),
    1.0, 120.0);
  const std::string record_debug_dir = has_cli_option(argc, argv, "record-debug-dir") ?
    cli.get<std::string>("record-debug-dir") :
    tools::read_or<std::string>(yaml, "record_debug_dir", "records");
  const auto web_frame_interval =
    std::chrono::milliseconds(static_cast<int>(1000.0 / web_fps));
  const auto web_state_interval = 80ms;

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
        "Web debugger config: fps={} scale={} jpeg={} ttl={}ms",
        web_fps, display_scale, web_jpeg_quality, web_client_ttl.count());
      tools::logger()->info(
        "Web runtime params bound to config: {}",
        tools::resolve_config_path_string(config_path));
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

  std::unique_ptr<tools::Recorder> raw_recorder;
  std::unique_ptr<tools::Recorder> debug_recorder;
  if (record_raw_video) {
    raw_recorder = std::make_unique<tools::Recorder>(
      record_debug_fps, record_debug_dir, "auto_debug_raw");
    tools::logger()->info(
      "Raw recording enabled: fps={} dir={}", record_debug_fps, record_debug_dir);
  }
  if (record_debug_video) {
    debug_recorder = std::make_unique<tools::Recorder>(
      record_debug_fps, record_debug_dir, "auto_debug_debug");
    tools::logger()->info(
      "Debug recording enabled: fps={} dir={}", record_debug_fps, record_debug_dir);
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver auto_aim_solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, auto_aim_solver);
  auto planner = std::make_unique<auto_aim::Planner>(config_path);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto small_target = std::make_unique<auto_buff::SmallTarget>();
  auto big_target = std::make_unique<auto_buff::BigTarget>();
  auto buff_aimer = std::make_unique<auto_buff::Aimer>(config_path);

  DebugMode current_mode = web_debugger ? clamp_debug_mode(web_debugger->selected_mode()) :
    DebugMode::AutoAim;
  DebugMode last_mode = current_mode;

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);
  auto last_web_frame_time = std::chrono::steady_clock::now() - web_frame_interval;
  auto last_web_state_time = std::chrono::steady_clock::now() - web_state_interval;
  const auto t0 = std::chrono::steady_clock::now();
  uint16_t last_bullet_count = 0;

  while (!exiter.exit()) {
    camera.read(img, t);
    const auto q = gimbal.q(t);
    const auto serial_mode = gimbal.mode();
    const auto gs = gimbal.state();
    const auto normalized_gimbal = normalize_gimbal_state(gs, gimbal_state_unit_mode);

    if (raw_recorder) raw_recorder->record(img, q, t);

    current_mode = web_debugger ? clamp_debug_mode(web_debugger->selected_mode()) :
      DebugMode::AutoAim;
    if (current_mode != last_mode) {
      tools::logger()->info(
        "Switch web debug mode: {} -> {}",
        debug_mode_label(last_mode), debug_mode_label(current_mode));
      tracker = std::make_unique<auto_aim::Tracker>(config_path, auto_aim_solver);
      planner = std::make_unique<auto_aim::Planner>(config_path);
      small_target = std::make_unique<auto_buff::SmallTarget>();
      big_target = std::make_unique<auto_buff::BigTarget>();
      buff_aimer = std::make_unique<auto_buff::Aimer>(config_path);
      last_mode = current_mode;
    }

    const auto loop_now = std::chrono::steady_clock::now();
    const double latency_ms = tools::delta_time(loop_now, t) * 1000.0;
    const bool fired = gs.bullet_count > last_bullet_count;
    last_bullet_count = gs.bullet_count;
    const double raw_bullet_speed = gs.bullet_speed;
    const double effective_speed = effective_bullet_speed(current_mode, raw_bullet_speed);
    const bool speed_fallback = bullet_speed_fallback(current_mode, raw_bullet_speed);
    const nlohmann::json overlay_config =
      web_debugger ? web_debugger->overlay_config() : nlohmann::json::object();

    nlohmann::json plot_data;
    plot_data["t"] = tools::delta_time(loop_now, t0);
    plot_data["mode"] = static_cast<int>(current_mode);
    plot_data["gimbal_yaw"] = normalized_gimbal.yaw.deg;
    plot_data["gimbal_yaw_vel"] = normalized_gimbal.yaw_vel.deg;
    plot_data["gimbal_pitch"] = normalized_gimbal.pitch.deg;
    plot_data["gimbal_pitch_vel"] = normalized_gimbal.pitch_vel.deg;
    plot_data["bullet_speed"] = raw_bullet_speed;

    auto_aim::Plan current_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    std::optional<auto_aim::Target> current_target;
    BallisticDiagnostic ballistic_diag;
    BuffModeResult buff_result;

    auto run_buff_pipeline = [&](auto & target_state) -> BuffModeResult {
      BuffModeResult result;
      buff_solver.set_R_gimbal2world(q);
      result.power_rune = buff_detector.detect(img);
      buff_solver.solve(result.power_rune);
      target_state.get_target(result.power_rune, t);
      auto target_copy = target_state;
      result.plan = buff_aimer->mpc_aim(target_copy, t, gs, true);
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
        const auto x = target_state.ekf_x();
        const auto aim_xyz = target_state.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
        const auto predicted_aim_xyz = target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
        const auto current_origin = target_state.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
        const auto predicted_origin = target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));

        result.aim_x_m = aim_xyz.x();
        result.aim_y_m = aim_xyz.y();
        result.aim_z_m = aim_xyz.z();
        result.predicted_aim_x_m = predicted_aim_xyz.x();
        result.predicted_aim_y_m = predicted_aim_xyz.y();
        result.predicted_aim_z_m = predicted_aim_xyz.z();
        result.target_angle_deg = rad2deg(x[5]);
        result.target_spd_deg_s = rad2deg(x[6]);
        result.target_spd_rad_s = x[6];
        result.current_projection = buff_solver.reproject_buff(current_origin, x[4], x[5]);
        const auto predicted_x = target_copy.ekf_x();
        result.predicted_projection =
          buff_solver.reproject_buff(predicted_origin, predicted_x[4], predicted_x[5]);

        if (x.size() >= 10) {
          result.fit_a_deg_s = rad2deg(x[7]);
          result.fit_w_rad_s = x[8];
          result.fit_fi_deg = rad2deg(x[9]);
        }
      }

      return result;
    };

    if (current_mode == DebugMode::AutoAim) {
      auto_aim_solver.set_R_gimbal2world(q);
      auto armors = yolo.detect(img);
      auto targets = tracker->track(armors, t);
      if (!targets.empty()) current_target = targets.front();

      current_plan = planner->plan(current_target, raw_bullet_speed);
      gimbal.send(
        current_plan.control, current_plan.fire, current_plan.yaw, current_plan.yaw_vel,
        current_plan.yaw_acc, current_plan.pitch, current_plan.pitch_vel, current_plan.pitch_acc);

      plot_data["target_yaw"] = rad2deg(current_plan.target_yaw);
      plot_data["target_pitch"] = rad2deg(current_plan.target_pitch);
      plot_data["plan_yaw"] = rad2deg(current_plan.yaw);
      plot_data["plan_yaw_vel"] = rad2deg(current_plan.yaw_vel);
      plot_data["plan_yaw_acc"] = rad2deg(current_plan.yaw_acc);
      plot_data["plan_pitch"] = rad2deg(current_plan.pitch);
      plot_data["plan_pitch_vel"] = rad2deg(current_plan.pitch_vel);
      plot_data["plan_pitch_acc"] = rad2deg(current_plan.pitch_acc);
      plot_data["fire"] = current_plan.fire ? 1 : 0;
      plot_data["fired"] = fired ? 1 : 0;

      if (current_target.has_value()) {
        plot_data["target_z"] = current_target->ekf_x()[4];
        plot_data["target_vz"] = current_target->ekf_x()[5];
        plot_data["target_h"] = current_target->ekf_x()[10];
        plot_data["w"] = current_target->ekf_x()[7];
        plot_data["tracker_match_valid"] = current_target->tracker_debug_match_valid ? 1 : 0;
        plot_data["tracker_match_id"] = current_target->tracker_debug_match_id;
        plot_data["tracker_match_score"] = current_target->tracker_debug_match_score;
        plot_data["tracker_reprojection_px"] = current_target->tracker_debug_reprojection_px;
      } else {
        plot_data["w"] = 0.0;
        plot_data["tracker_match_valid"] = 0;
        plot_data["tracker_match_id"] = -1;
        plot_data["tracker_match_score"] = -1.0;
        plot_data["tracker_reprojection_px"] = -1.0;
      }

      plot_data["planner_selected_armor"] = planner->debug_armor_id;
      plot_data["planner_delay_ms"] = planner->debug_delay_time * 1000.0;
      plot_data["planner_hit_fly_time_ms"] = planner->debug_hit_fly_time * 1000.0;
      plot_data["planner_hit_iters"] = planner->debug_hit_iter_count;
      plot_data["planner_hit_converged"] = planner->debug_hit_converged ? 1 : 0;
      plot_data["planner_spin_gate"] = planner->debug_used_spin_gate ? 1 : 0;
      plot_data["planner_center_yaw"] = rad2deg(planner->debug_center_yaw);
      plot_data["planner_turn_sign"] =
        tools::debug::spin_direction_sign(current_target.has_value() ? current_target->ekf_x()[7] : 0.0);
      plot_data["planner_selected_physical_armor"] = planner->debug_physical_armor_id;
      plot_data["planner_selected_z_offset"] = planner->debug_selected_z_offset;
      plot_data["planner_selected_aim_z_compensation"] =
        planner->debug_selected_aim_z_compensation;
      plot_data["planner_selected_delta_deg"] = rad2deg(planner->debug_selected_delta_angle);
      plot_data["planner_fixed_model"] = planner->debug_fixed_center_rotation_model ? 1 : 0;
      plot_data["planner_fire_tracking_error_deg"] = rad2deg(planner->debug_fire_tracking_error);
      plot_data["planner_fire_phase_limit_deg"] = rad2deg(planner->debug_fire_phase_limit);
      plot_data["planner_fire_track_ready"] = planner->debug_fire_track_ready ? 1 : 0;
      plot_data["planner_fire_phase_ready"] = planner->debug_fire_phase_ready ? 1 : 0;

      if (current_target.has_value() && current_plan.control) {
        ballistic_diag = build_ballistic_diagnostic(
          current_plan, planner->debug_xyza, current_target->armor_type, raw_bullet_speed,
          yaw_offset, pitch_offset);
      }
    } else {
      buff_result =
        current_mode == DebugMode::SmallBuff ? run_buff_pipeline(*small_target) :
        run_buff_pipeline(*big_target);
      current_plan = buff_result.plan;

      gimbal.send(
        current_plan.control, current_plan.fire, current_plan.yaw, current_plan.yaw_vel,
        current_plan.yaw_acc, current_plan.pitch, current_plan.pitch_vel, current_plan.pitch_acc);

      plot_data["target_yaw"] = buff_result.blade_yaw_deg;
      plot_data["target_pitch"] = -buff_result.blade_pitch_deg;
      plot_data["plan_yaw"] = rad2deg(current_plan.yaw);
      plot_data["plan_pitch"] = rad2deg(current_plan.pitch);
      plot_data["plan_yaw_vel"] = rad2deg(current_plan.yaw_vel);
      plot_data["plan_pitch_vel"] = rad2deg(current_plan.pitch_vel);
      plot_data["plan_yaw_acc"] = rad2deg(current_plan.yaw_acc);
      plot_data["plan_pitch_acc"] = rad2deg(current_plan.pitch_acc);
      plot_data["fire"] = current_plan.fire ? 1 : 0;
      plot_data["fired"] = fired ? 1 : 0;
      plot_data["R_yaw"] = buff_result.rune_yaw_deg;
      plot_data["R_pitch"] = buff_result.rune_pitch_deg;
      plot_data["R_dis"] = buff_result.rune_dist_m;
      plot_data["blade_yaw"] = buff_result.blade_yaw_deg;
      plot_data["blade_pitch"] = buff_result.blade_pitch_deg;
      plot_data["blade_dis"] = buff_result.blade_dist_m;
      plot_data["buff_yaw"] = buff_result.buff_yaw_deg;
      plot_data["buff_pitch"] = buff_result.buff_pitch_deg;
      plot_data["buff_roll"] = buff_result.buff_roll_deg;
      plot_data["angle"] = buff_result.target_angle_deg;
      plot_data["spd"] = buff_result.target_spd_deg_s;
      plot_data["w"] = buff_result.target_spd_rad_s;
      plot_data["a"] = buff_result.fit_a_deg_s;
      plot_data["fi"] = buff_result.fit_fi_deg;
      plot_data["target_z"] = buff_result.aim_z_m;
      plot_data["planner_turn_sign"] =
        tools::debug::spin_direction_sign(buff_result.target_spd_rad_s);
    }

    plotter.plot(plot_data);
    if (web_debugger) web_debugger->update_plot_sample(plot_data);

    const auto now = std::chrono::steady_clock::now();
    const bool need_web_frame =
      web_debugger && web_debugger->has_active_client(web_client_ttl) &&
      (now - last_web_frame_time >= web_frame_interval);
    const bool need_visual_output = show_local || need_web_frame || debug_recorder != nullptr;

    if (web_debugger && now - last_web_state_time >= web_state_interval) {
      nlohmann::json web_state;
      web_state["server"]["unix_ms"] = unix_time_ms();
      web_state["mode"] = build_mode_state(current_mode, serial_mode, gimbal);
      web_state["frame"]["latency_ms"] = latency_ms;
      web_state["frame"]["image_width"] = img.cols;
      web_state["frame"]["image_height"] = img.rows;
      web_state["frame"]["bullet_speed_mps"] = raw_bullet_speed;
      web_state["frame"]["bullet_speed_effective_mps"] = effective_speed;
      web_state["frame"]["bullet_speed_fallback"] = speed_fallback;
      web_state["frame"]["bullet_speed_source"] = "serial";
      web_state["overlay"]["controls"] =
        overlay_config.is_object() ? overlay_config : nlohmann::json::object();

      nlohmann::json command_state;
      command_state["has_target"] =
        current_mode == DebugMode::AutoAim ? current_target.has_value() : buff_result.target_solved;
      command_state["fire"] = current_plan.fire;
      command_state["fired"] = fired;
      command_state["gimbal_source_unit"] = normalized_gimbal.source_is_degree ? "deg" : "rad";
      command_state["gimbal_yaw_raw"] = normalized_gimbal.yaw.raw;
      command_state["gimbal_yaw_deg"] = normalized_gimbal.yaw.deg;
      command_state["gimbal_yaw_rad"] = normalized_gimbal.yaw.rad;
      command_state["gimbal_pitch_raw"] = normalized_gimbal.pitch.raw;
      command_state["gimbal_pitch_deg"] = normalized_gimbal.pitch.deg;
      command_state["gimbal_pitch_rad"] = normalized_gimbal.pitch.rad;
      command_state["gimbal_yaw_vel_raw"] = normalized_gimbal.yaw_vel.raw;
      command_state["gimbal_yaw_vel_deg"] = normalized_gimbal.yaw_vel.deg;
      command_state["gimbal_yaw_vel_rad"] = normalized_gimbal.yaw_vel.rad;
      command_state["gimbal_pitch_vel_raw"] = normalized_gimbal.pitch_vel.raw;
      command_state["gimbal_pitch_vel_deg"] = normalized_gimbal.pitch_vel.deg;
      command_state["gimbal_pitch_vel_rad"] = normalized_gimbal.pitch_vel.rad;
      command_state["plan_yaw_deg"] = rad2deg(current_plan.yaw);
      command_state["plan_yaw_rad"] = current_plan.yaw;
      command_state["plan_pitch_deg"] = rad2deg(current_plan.pitch);
      command_state["plan_pitch_rad"] = current_plan.pitch;
      command_state["plan_yaw_vel_deg"] = rad2deg(current_plan.yaw_vel);
      command_state["plan_yaw_vel_rad"] = current_plan.yaw_vel;
      command_state["plan_pitch_vel_deg"] = rad2deg(current_plan.pitch_vel);
      command_state["plan_pitch_vel_rad"] = current_plan.pitch_vel;
      command_state["plan_yaw_acc_deg"] = rad2deg(current_plan.yaw_acc);
      command_state["plan_yaw_acc_rad"] = current_plan.yaw_acc;
      command_state["plan_pitch_acc_deg"] = rad2deg(current_plan.pitch_acc);
      command_state["plan_pitch_acc_rad"] = current_plan.pitch_acc;
      command_state["bullet_speed_mps"] = raw_bullet_speed;
      command_state["bullet_speed_effective_mps"] = effective_speed;
      command_state["bullet_speed_fallback"] = speed_fallback;
      command_state["bullet_speed_source"] = "serial";

      if (current_mode == DebugMode::AutoAim) {
        const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
        const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
        const double current_selected_z_offset =
          current_target.has_value() ? planner->debug_selected_z_offset : 0.0;
        const double current_selected_aim_z_compensation =
          current_target.has_value() ? planner->debug_selected_aim_z_compensation : 0.0;
        const bool current_fixed_model =
          current_target.has_value() && planner->debug_fixed_center_rotation_model;

        command_state["target_yaw_deg"] = rad2deg(current_plan.target_yaw);
        command_state["target_yaw_rad"] = current_plan.target_yaw;
        command_state["target_pitch_deg"] = rad2deg(current_plan.target_pitch);
        command_state["target_pitch_rad"] = current_plan.target_pitch;
        if (current_target.has_value()) {
          command_state["target_z_m"] = current_target->ekf_x()[4];
          command_state["target_vz_mps"] = current_target->ekf_x()[5];
          command_state["target_h_m"] = current_target->ekf_x()[10];
          command_state["target_w_rad_s"] = current_target->ekf_x()[7];
          command_state["tracker_candidate_count"] = current_target->tracker_debug_candidate_count;
          command_state["tracker_match_valid"] = current_target->tracker_debug_match_valid;
          command_state["tracker_match_id"] = current_target->tracker_debug_match_id;
          command_state["tracker_match_score"] = current_target->tracker_debug_match_score;
          command_state["tracker_reprojection_px"] = current_target->tracker_debug_reprojection_px;
          command_state["tracker_xy_error_m"] = current_target->tracker_debug_xy_error_m;
          command_state["tracker_z_error_m"] = current_target->tracker_debug_z_error_m;
        } else {
          command_state["target_z_m"] = nullptr;
          command_state["target_vz_mps"] = nullptr;
          command_state["target_h_m"] = nullptr;
          command_state["target_w_rad_s"] = nullptr;
          command_state["tracker_candidate_count"] = 0;
          command_state["tracker_match_valid"] = false;
          command_state["tracker_match_id"] = -1;
          command_state["tracker_match_score"] = nullptr;
          command_state["tracker_reprojection_px"] = nullptr;
          command_state["tracker_xy_error_m"] = nullptr;
          command_state["tracker_z_error_m"] = nullptr;
        }

        web_state["preview"]["has_target"] = current_target.has_value();
        web_state["preview"]["fire"] = current_plan.fire;
        web_state["preview"]["target_name"] =
          current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
        web_state["preview"]["armor_type"] =
          current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
        web_state["preview"]["target_yaw_deg"] = rad2deg(current_plan.target_yaw);
        web_state["preview"]["target_pitch_deg"] = rad2deg(current_plan.target_pitch);
        web_state["preview"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
        web_state["preview"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
        if (current_target.has_value()) {
          web_state["preview"]["target_x_m"] = planner->debug_xyza[0];
          web_state["preview"]["target_y_m"] = planner->debug_xyza[1];
          web_state["preview"]["target_z_m"] = planner->debug_xyza[2];
        } else {
          web_state["preview"]["target_x_m"] = nullptr;
          web_state["preview"]["target_y_m"] = nullptr;
          web_state["preview"]["target_z_m"] = nullptr;
        }

        web_state["planner"]["selected_armor"] = planner->debug_armor_id;
        web_state["planner"]["physical_armor"] = planner->debug_physical_armor_id;
        web_state["planner"]["spin_gate"] = planner->debug_used_spin_gate;
        web_state["planner"]["delay_ms"] = planner->debug_delay_time * 1000.0;
        web_state["planner"]["hit_fly_time_ms"] = planner->debug_hit_fly_time * 1000.0;
        web_state["planner"]["hit_iter_count"] = planner->debug_hit_iter_count;
        web_state["planner"]["hit_converged"] = planner->debug_hit_converged;
        web_state["planner"]["center_yaw_deg"] = rad2deg(planner->debug_center_yaw);
        web_state["planner"]["turn_direction"] =
          tools::debug::spin_direction_to_string(current_w);
        web_state["planner"]["turn_sign"] =
          tools::debug::spin_direction_sign(current_w);
        web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
        for (const double delta_angle : planner->debug_delta_angle_list) {
          web_state["planner"]["delta_angle_deg_list"].push_back(rad2deg(delta_angle));
        }
        web_state["planner"]["w_rad_s"] = current_w;
        web_state["planner"]["h_m"] = current_h;
        web_state["planner"]["selected_z_offset_m"] = current_selected_z_offset;
        web_state["planner"]["selected_aim_z_compensation_m"] =
          current_selected_aim_z_compensation;
        web_state["planner"]["selected_delta_deg"] = rad2deg(planner->debug_selected_delta_angle);
        web_state["planner"]["fixed_center_rotation_model"] = current_fixed_model;
        web_state["planner"]["fire_tracking_error_deg"] =
          rad2deg(planner->debug_fire_tracking_error);
        web_state["planner"]["fire_phase_limit_deg"] =
          rad2deg(planner->debug_fire_phase_limit);
        web_state["planner"]["fire_track_ready"] = planner->debug_fire_track_ready;
        web_state["planner"]["fire_phase_ready"] = planner->debug_fire_phase_ready;

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
        web_state["ballistic"] = ballistic_to_json(ballistic_diag);
      } else {
        command_state["target_yaw_deg"] = buff_result.blade_yaw_deg;
        command_state["target_yaw_rad"] =
          buff_result.power_rune.has_value() ? buff_result.power_rune->blade_ypd_in_world[0] :
          current_plan.target_yaw;
        command_state["target_pitch_deg"] = -buff_result.blade_pitch_deg;
        command_state["target_pitch_rad"] =
          buff_result.power_rune.has_value() ? -buff_result.power_rune->blade_ypd_in_world[1] :
          current_plan.target_pitch;
        command_state["target_x_m"] = buff_result.aim_x_m;
        command_state["target_y_m"] = buff_result.aim_y_m;
        command_state["target_z_m"] = buff_result.aim_z_m;
        command_state["buff_target_solved"] = buff_result.target_solved;

        web_state["preview"]["has_target"] = buff_result.target_solved;
        web_state["preview"]["fire"] = current_plan.fire;
        web_state["preview"]["target_name"] = debug_mode_label(current_mode);
        web_state["preview"]["armor_type"] = "buff";
        web_state["preview"]["target_yaw_deg"] = buff_result.blade_yaw_deg;
        web_state["preview"]["target_pitch_deg"] = -buff_result.blade_pitch_deg;
        web_state["preview"]["plan_yaw_deg"] = rad2deg(current_plan.yaw);
        web_state["preview"]["plan_pitch_deg"] = rad2deg(current_plan.pitch);
        web_state["preview"]["target_x_m"] = buff_result.aim_x_m;
        web_state["preview"]["target_y_m"] = buff_result.aim_y_m;
        web_state["preview"]["target_z_m"] = buff_result.aim_z_m;

        web_state["planner"]["selected_armor"] = -1;
        web_state["planner"]["physical_armor"] = -1;
        web_state["planner"]["spin_gate"] = false;
        web_state["planner"]["delay_ms"] = 0.0;
        web_state["planner"]["turn_direction"] =
          tools::debug::spin_direction_to_string(buff_result.target_spd_rad_s);
        web_state["planner"]["turn_sign"] =
          tools::debug::spin_direction_sign(buff_result.target_spd_rad_s);
        web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
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

        web_state["overlay"]["stage"] = buff_result.overlay_stage;
        web_state["ballistic"]["valid"] = current_plan.control;
        web_state["ballistic"]["unsolvable"] = !current_plan.control;
        web_state["ballistic"]["hit"] = current_plan.fire;
        web_state["ballistic"]["bullet_speed_raw_mps"] = raw_bullet_speed;
        web_state["ballistic"]["bullet_speed_effective_mps"] = effective_speed;
        web_state["ballistic"]["bullet_speed_mps"] = effective_speed;
        web_state["ballistic"]["bullet_speed_fallback"] = speed_fallback;
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
      }

      web_state["command"] = command_state;
      web_debugger->update_state(web_state);
      web_debugger->update_log(web_state);
      last_web_state_time = now;
    }

    if (need_visual_output) {
      cv::Mat display_img;
      if (current_mode == DebugMode::AutoAim) {
        tools::debug_visualization::LiveOverlayOptions visual_options;
        visual_options.display_scale = display_scale;
        visual_options.latency_ms = latency_ms;
        visual_options.target_name =
          current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
        visual_options.armor_type =
          current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
        visual_options.planner_armor_id = planner->debug_armor_id;
        visual_options.planner_physical_armor_id = planner->debug_physical_armor_id;
        visual_options.planner_spin_gate = planner->debug_used_spin_gate;
        visual_options.planner_delay_ms = planner->debug_delay_time * 1000.0;
        visual_options.planner_center_yaw_deg = rad2deg(planner->debug_center_yaw);
        visual_options.planner_hit_fly_time_ms = planner->debug_hit_fly_time * 1000.0;
        visual_options.planner_hit_iter_count = planner->debug_hit_iter_count;
        visual_options.planner_hit_converged = planner->debug_hit_converged;
        visual_options.planner_delta_angles_deg.clear();
        for (const double delta_angle : planner->debug_delta_angle_list) {
          visual_options.planner_delta_angles_deg.push_back(rad2deg(delta_angle));
        }
        const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
        const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
        visual_options.planner_turn_direction =
          tools::debug::spin_direction_to_string(current_w);
        visual_options.planner_turn_sign = tools::debug::spin_direction_sign(current_w);
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
        visual_options.current_selected_z_offset =
          current_target.has_value() ? planner->debug_selected_z_offset : 0.0;
        visual_options.current_selected_aim_z_compensation =
          current_target.has_value() ? planner->debug_selected_aim_z_compensation : 0.0;
        visual_options.current_fixed_model =
          current_target.has_value() && planner->debug_fixed_center_rotation_model;
        visual_options.target_jumped = current_target.has_value() && current_target->jumped;
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

        display_img = tools::debug_visualization::render_live_debug_frame(
          img, auto_aim_solver, current_target, current_plan, *planner, visual_options);
        draw_ballistic_panel(ballistic_panel, ballistic_diag);
      } else {
        display_img = render_buff_debug_frame(
          img, buff_result, overlay_config, current_mode, display_scale, latency_ms,
          raw_bullet_speed);
        draw_buff_panel(
          ballistic_panel, current_mode, buff_result, normalized_gimbal, latency_ms,
          raw_bullet_speed, effective_speed, speed_fallback, fired);
      }

      if (debug_recorder) debug_recorder->record(display_img, q, t);

      if (need_web_frame && web_debugger) {
        web_debugger->update_main_frame(display_img, web_jpeg_quality);
        web_debugger->update_ballistic_frame(ballistic_panel, web_jpeg_quality);
        last_web_frame_time = now;
      }

      if (show_local) {
        cv::imshow("Auto Debug", display_img);
        cv::imshow("Auto Debug Panel", ballistic_panel);
        const auto key = cv::waitKey(1);
        if (key == 'q') break;
      }
    }
  }

  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  if (show_local) cv::destroyAllWindows();
  return 0;
}
