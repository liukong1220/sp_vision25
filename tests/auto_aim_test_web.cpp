#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/debug.hpp"
#include "tools/debug_visualization.hpp"
#include "tools/exiter.hpp"
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
      web_debugger->set_plot_history_limit(600);
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
  auto_aim::Planner debug_planner(config_path);

  cv::Mat img;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);
  auto playback_start = std::chrono::steady_clock::now();
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

    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) current_target = targets.front();

    const auto planner_start = std::chrono::steady_clock::now();
    const auto current_plan = debug_planner.plan(current_target, bullet_speed);

    const auto finish = std::chrono::steady_clock::now();
    const double processing_ms = tools::delta_time(finish, frame_start) * 1000.0;
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, planner: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(planner_start, tracker_start) * 1e3,
      tools::delta_time(finish, planner_start) * 1e3);

    Eigen::Quaterniond gimbal_q(w, x, y, z);
    const Eigen::Vector3d gimbal_ypr =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

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
    data["bullet_speed"] = bullet_speed;
    data["t"] = relative_t;
 
    if (current_target.has_value()) {
      const auto & target = *current_target;

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

    data["planner_selected_armor"] = debug_planner.debug_armor_id;
    data["planner_delay_ms"] = debug_planner.debug_delay_time * 1000.0;
    data["planner_spin_gate"] = debug_planner.debug_used_spin_gate ? 1 : 0;
    data["planner_center_yaw"] = rad2deg(debug_planner.debug_center_yaw);
    data["planner_turn_sign"] =
      tools::debug::spin_direction_sign(current_target.has_value() ? current_target->ekf_x()[7] : 0.0);
    data["planner_selected_z_offset"] = debug_planner.debug_selected_z_offset;
    data["planner_fixed_model"] = debug_planner.debug_fixed_center_rotation_model ? 1 : 0;

    plotter.plot(data);
    if (web_debugger) web_debugger->update_plot_sample(data);

    BallisticDiagnostic ballistic_diag;
    if (current_target.has_value() && current_plan.control) {
      ballistic_diag = build_ballistic_diagnostic(
        current_plan, debug_planner.debug_xyza, current_target->armor_type, bullet_speed,
        yaw_offset, pitch_offset);
    }

    const double current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
    const double current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
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
      web_state["planner"]["center_yaw_deg"] = rad2deg(debug_planner.debug_center_yaw);
      web_state["planner"]["turn_direction"] =
        tools::debug::spin_direction_to_string(current_w);
      web_state["planner"]["turn_sign"] =
        tools::debug::spin_direction_sign(current_w);
      web_state["planner"]["delta_angle_deg_list"] = nlohmann::json::array();
      for (const double delta_angle : debug_planner.debug_delta_angle_list) {
        web_state["planner"]["delta_angle_deg_list"].push_back(rad2deg(delta_angle));
      }
      web_state["planner"]["w_rad_s"] = current_w;
      web_state["planner"]["h_m"] = current_h;
      web_state["planner"]["selected_z_offset_m"] = debug_planner.debug_selected_z_offset;
      web_state["planner"]["fixed_center_rotation_model"] =
        debug_planner.debug_fixed_center_rotation_model;
      web_state["overlay"]["stage"] =
        tools::debug_visualization::live_overlay_stage_to_string(overlay_stage);
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
      web_state["command"]["bullet_speed_mps"] = bullet_speed;
      web_debugger->update_state(web_state);
      web_debugger->update_log(web_state);
      last_web_state_time = now;
    }

    if (need_visual_output) {
      tools::debug_visualization::LiveOverlayOptions visual_options;
      visual_options.display_scale = display_scale;
      visual_options.latency_ms = processing_ms;
      visual_options.target_name =
        current_target.has_value() ? armor_name_to_string(current_target->name) : "none";
      visual_options.armor_type =
        current_target.has_value() ? armor_type_to_string(current_target->armor_type) : "none";
      visual_options.planner_armor_id = debug_planner.debug_armor_id;
      visual_options.planner_spin_gate = debug_planner.debug_used_spin_gate;
      visual_options.planner_delay_ms = debug_planner.debug_delay_time * 1000.0;
      visual_options.planner_center_yaw_deg = rad2deg(debug_planner.debug_center_yaw);
      for (const double delta_angle : debug_planner.debug_delta_angle_list) {
        visual_options.planner_delta_angles_deg.push_back(rad2deg(delta_angle));
      }
      visual_options.planner_turn_direction =
        tools::debug::spin_direction_to_string(current_w);
      visual_options.planner_turn_sign =
        tools::debug::spin_direction_sign(current_w);
      visual_options.current_w = current_w;
      visual_options.current_h = current_h;
      visual_options.current_selected_z_offset = debug_planner.debug_selected_z_offset;
      visual_options.current_fixed_model = debug_planner.debug_fixed_center_rotation_model;
      visual_options.target_jumped =
        current_target.has_value() && current_target->jumped;
      visual_options.is_outpost =
        current_target.has_value() &&
        current_target->name == auto_aim::ArmorName::outpost;
      apply_overlay_config(visual_options);

      const auto display_img = tools::debug_visualization::render_live_debug_frame(
        img, solver, current_target, current_plan, debug_planner, visual_options);

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
