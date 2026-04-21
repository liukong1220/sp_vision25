#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <memory>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/debug.hpp"
#include "tools/debug_visualization.hpp"
#include "tools/exiter.hpp"
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

  std::unique_ptr<tools::Recorder> raw_recorder;
  std::unique_ptr<tools::Recorder> debug_recorder;
  if (record_raw_video) {
    raw_recorder = std::make_unique<tools::Recorder>(
      record_debug_fps, record_debug_dir, "auto_aim_debug_mpc_raw");
    tools::logger()->info(
      "Raw recording enabled: fps={} dir={}", record_debug_fps, record_debug_dir);
  }
  if (record_debug_video) {
    debug_recorder = std::make_unique<tools::Recorder>(
      record_debug_fps, record_debug_dir, "auto_aim_debug_mpc_debug");
    tools::logger()->info(
      "Debug recording enabled: fps={} dir={}", record_debug_fps, record_debug_dir);
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

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
    if (raw_recorder) {
      raw_recorder->record(img, q, t);
    }

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    auto targets = tracker.track(armors, t);
    const auto gs = gimbal.state();
    const auto normalized_gimbal = normalize_gimbal_state(gs, gimbal_state_unit_mode);

    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) current_target = targets.front();

    const auto current_plan = planner.plan(current_target, gs.bullet_speed);
    gimbal.send(
      current_plan.control, current_plan.fire, current_plan.yaw, current_plan.yaw_vel,
      current_plan.yaw_acc, current_plan.pitch, current_plan.pitch_vel, current_plan.pitch_acc);

    const bool fired = gs.bullet_count > last_bullet_count;
    last_bullet_count = gs.bullet_count;

    BallisticDiagnostic ballistic_diag;
    if (current_target.has_value() && current_plan.control) {
      ballistic_diag = build_ballistic_diagnostic(
        current_plan, planner.debug_xyza, current_target->armor_type, gs.bullet_speed,
        yaw_offset, pitch_offset);
    }

    const double latency_ms =
      tools::delta_time(std::chrono::steady_clock::now(), t) * 1000.0;
    const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
    const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
    const double current_selected_z_offset =
      current_target.has_value() ? planner.debug_selected_z_offset : 0.0;
    const double current_selected_aim_z_compensation =
      current_target.has_value() ? planner.debug_selected_aim_z_compensation : 0.0;
    const bool current_fixed_model =
      current_target.has_value() && planner.debug_fixed_center_rotation_model;

    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
    data["gimbal_yaw"] = normalized_gimbal.yaw.deg;
    data["gimbal_yaw_vel"] = normalized_gimbal.yaw_vel.deg;
    data["gimbal_pitch"] = normalized_gimbal.pitch.deg;
    data["gimbal_pitch_vel"] = normalized_gimbal.pitch_vel.deg;
    data["target_yaw"] = rad2deg(current_plan.target_yaw);
    data["target_pitch"] = rad2deg(current_plan.target_pitch);
    data["plan_yaw"] = rad2deg(current_plan.yaw);
    data["plan_yaw_vel"] = rad2deg(current_plan.yaw_vel);
    data["plan_yaw_acc"] = rad2deg(current_plan.yaw_acc);
    data["plan_pitch"] = rad2deg(current_plan.pitch);
    data["plan_pitch_vel"] = rad2deg(current_plan.pitch_vel);
    data["plan_pitch_acc"] = rad2deg(current_plan.pitch_acc);
    data["fire"] = current_plan.fire ? 1 : 0;
    data["fired"] = fired ? 1 : 0;
    data["bullet_speed"] = gs.bullet_speed;

    if (current_target.has_value()) {
      data["target_z"] = current_target->ekf_x()[4];
      data["target_vz"] = current_target->ekf_x()[5];
      data["target_h"] = current_target->ekf_x()[10];
      data["w"] = current_target->ekf_x()[7];
      data["tracker_match_valid"] = current_target->tracker_debug_match_valid ? 1 : 0;
      data["tracker_match_id"] = current_target->tracker_debug_match_id;
      data["tracker_match_score"] = current_target->tracker_debug_match_score;
      data["tracker_reprojection_px"] = current_target->tracker_debug_reprojection_px;
    } else {
      data["w"] = 0.0;
      data["tracker_match_valid"] = 0;
      data["tracker_match_id"] = -1;
      data["tracker_match_score"] = -1.0;
      data["tracker_reprojection_px"] = -1.0;
    }

    data["planner_selected_armor"] = planner.debug_armor_id;
    data["planner_delay_ms"] = planner.debug_delay_time * 1000.0;
    data["planner_hit_fly_time_ms"] = planner.debug_hit_fly_time * 1000.0;
    data["planner_hit_iters"] = planner.debug_hit_iter_count;
    data["planner_hit_converged"] = planner.debug_hit_converged ? 1 : 0;
    data["planner_spin_gate"] = planner.debug_used_spin_gate ? 1 : 0;
    data["planner_center_yaw"] = rad2deg(planner.debug_center_yaw);
    data["planner_turn_sign"] =
      tools::debug::spin_direction_sign(current_target.has_value() ? current_target->ekf_x()[7] : 0.0);
    data["planner_selected_physical_armor"] = planner.debug_physical_armor_id;
    data["planner_selected_z_offset"] = planner.debug_selected_z_offset;
    data["planner_selected_aim_z_compensation"] = planner.debug_selected_aim_z_compensation;
    data["planner_selected_delta_deg"] = rad2deg(planner.debug_selected_delta_angle);
    data["planner_fixed_model"] = planner.debug_fixed_center_rotation_model ? 1 : 0;
    data["planner_fire_tracking_error_deg"] = rad2deg(planner.debug_fire_tracking_error);
    data["planner_fire_phase_limit_deg"] = rad2deg(planner.debug_fire_phase_limit);
    data["planner_fire_track_ready"] = planner.debug_fire_track_ready ? 1 : 0;
    data["planner_fire_phase_ready"] = planner.debug_fire_phase_ready ? 1 : 0;

    plotter.plot(data);
    if (web_debugger) web_debugger->update_plot_sample(data);

    const nlohmann::json overlay_config =
      web_debugger ? web_debugger->overlay_config() : nlohmann::json::object();
    const auto apply_overlay_config =
      [&](tools::debug_visualization::LiveOverlayOptions & visual_options) {
        if (!overlay_config.is_object()) return;
        auto apply_bool = [&](const char * key, bool & field) {
          if (overlay_config.contains(key) && overlay_config.at(key).is_boolean()) {
            field = overlay_config.at(key).get<bool>();
          }
        };
        apply_bool("stabilize", visual_options.stabilize_annotations);
        apply_bool("state_layers", visual_options.enable_state_layers);
        apply_bool("armors", visual_options.show_armors);
        apply_bool("labels", visual_options.show_armor_labels);
        apply_bool("target_motion", visual_options.show_target_motion);
        apply_bool("aim", visual_options.show_aim);
        apply_bool("decision_hud", visual_options.show_decision_hud);
        apply_bool("decision_track", visual_options.show_decision_track);
        apply_bool("footer", visual_options.show_footer);
      };
    const auto overlay_stage = tools::debug_visualization::resolve_live_overlay_stage(
      current_target.has_value(), current_plan);

    const auto now = std::chrono::steady_clock::now();
    const bool need_web_frame =
      web_debugger && web_debugger->has_active_client(web_client_ttl) &&
      (now - last_web_frame_time >= web_frame_interval);
    const bool need_visual_output = show_local || need_web_frame || debug_recorder != nullptr;

    if (web_debugger && now - last_web_state_time >= web_state_interval) {
      nlohmann::json web_state;
      nlohmann::json command_state;
      web_state["server"]["unix_ms"] = unix_time_ms();
      web_state["frame"]["latency_ms"] = latency_ms;
      web_state["frame"]["image_width"] = img.cols;
      web_state["frame"]["image_height"] = img.rows;
      web_state["frame"]["bullet_speed_mps"] = gs.bullet_speed;
      web_state["frame"]["bullet_speed_source"] = "serial";
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
        web_state["preview"]["target_x_m"] = planner.debug_xyza[0];
        web_state["preview"]["target_y_m"] = planner.debug_xyza[1];
        web_state["preview"]["target_z_m"] = planner.debug_xyza[2];
      } else {
        web_state["preview"]["target_x_m"] = nullptr;
        web_state["preview"]["target_y_m"] = nullptr;
        web_state["preview"]["target_z_m"] = nullptr;
      }
      web_state["planner"]["selected_armor"] = planner.debug_armor_id;
      web_state["planner"]["physical_armor"] = planner.debug_physical_armor_id;
      web_state["planner"]["spin_gate"] = planner.debug_used_spin_gate;
      web_state["planner"]["delay_ms"] = planner.debug_delay_time * 1000.0;
      web_state["planner"]["hit_fly_time_ms"] = planner.debug_hit_fly_time * 1000.0;
      web_state["planner"]["hit_iter_count"] = planner.debug_hit_iter_count;
      web_state["planner"]["hit_converged"] = planner.debug_hit_converged;
      web_state["planner"]["center_yaw_deg"] = rad2deg(planner.debug_center_yaw);
      web_state["planner"]["turn_direction"] =
        tools::debug::spin_direction_to_string(current_w);
      web_state["planner"]["turn_sign"] =
        tools::debug::spin_direction_sign(current_w);
      web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
      for (const double delta_angle : planner.debug_delta_angle_list) {
        web_state["planner"]["delta_angle_deg_list"].push_back(rad2deg(delta_angle));
      }
      web_state["planner"]["w_rad_s"] = current_w;
      web_state["planner"]["h_m"] = current_h;
      web_state["planner"]["selected_z_offset_m"] = current_selected_z_offset;
      web_state["planner"]["selected_aim_z_compensation_m"] =
        current_selected_aim_z_compensation;
      web_state["planner"]["selected_delta_deg"] = rad2deg(planner.debug_selected_delta_angle);
      web_state["planner"]["fixed_center_rotation_model"] = current_fixed_model;
      web_state["planner"]["fire_tracking_error_deg"] =
        rad2deg(planner.debug_fire_tracking_error);
      web_state["planner"]["fire_phase_limit_deg"] =
        rad2deg(planner.debug_fire_phase_limit);
      web_state["planner"]["fire_track_ready"] = planner.debug_fire_track_ready;
      web_state["planner"]["fire_phase_ready"] = planner.debug_fire_phase_ready;
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
        tools::debug_visualization::live_overlay_stage_to_string(overlay_stage);
      web_state["overlay"]["controls"] =
        overlay_config.is_object() ? overlay_config : nlohmann::json::object();
      web_state["ballistic"] = ballistic_to_json(ballistic_diag);
      command_state["has_target"] = current_target.has_value();
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
      command_state["target_yaw_deg"] = rad2deg(current_plan.target_yaw);
      command_state["target_yaw_rad"] = current_plan.target_yaw;
      command_state["target_pitch_deg"] = rad2deg(current_plan.target_pitch);
      command_state["target_pitch_rad"] = current_plan.target_pitch;
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
      command_state["bullet_speed_mps"] = gs.bullet_speed;
      command_state["bullet_speed_source"] = "serial";
      command_state["bullet_speed_effective_mps"] =
        (gs.bullet_speed < 10.0 || gs.bullet_speed > 25.0) ? 22.0 : gs.bullet_speed;
      command_state["bullet_speed_fallback"] =
        gs.bullet_speed < 10.0 || gs.bullet_speed > 25.0;
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
      web_state["command"] = command_state;
      web_debugger->update_state(web_state);
      web_debugger->update_log(web_state);
      last_web_state_time = now;
    }

    if (need_visual_output) {
      tools::debug_visualization::LiveOverlayOptions visual_options;
      visual_options.display_scale = display_scale;
      visual_options.latency_ms = latency_ms;
      visual_options.target_name =
        current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
      visual_options.armor_type =
        current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
      visual_options.planner_armor_id = planner.debug_armor_id;
      visual_options.planner_physical_armor_id = planner.debug_physical_armor_id;
      visual_options.planner_spin_gate = planner.debug_used_spin_gate;
      visual_options.planner_delay_ms = planner.debug_delay_time * 1000.0;
      visual_options.planner_center_yaw_deg = rad2deg(planner.debug_center_yaw);
      visual_options.planner_hit_fly_time_ms = planner.debug_hit_fly_time * 1000.0;
      visual_options.planner_hit_iter_count = planner.debug_hit_iter_count;
      visual_options.planner_hit_converged = planner.debug_hit_converged;
      visual_options.planner_delta_angles_deg.clear();
      for (const double delta_angle : planner.debug_delta_angle_list) {
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
      visual_options.current_selected_z_offset = current_selected_z_offset;
      visual_options.current_selected_aim_z_compensation = current_selected_aim_z_compensation;
      visual_options.current_fixed_model = current_fixed_model;
      visual_options.target_jumped =
        current_target.has_value() && current_target->jumped;
      visual_options.is_outpost =
        current_target.has_value() &&
        current_target->name == auto_aim::ArmorName::outpost;
      apply_overlay_config(visual_options);

      const auto display_img = tools::debug_visualization::render_live_debug_frame(
        img, solver, current_target, current_plan, planner, visual_options);

      draw_ballistic_panel(ballistic_panel, ballistic_diag);

      if (debug_recorder) {
        debug_recorder->record(display_img, q, t);
      }

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

  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  if (show_local) cv::destroyAllWindows();

  return 0;
}
