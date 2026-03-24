#include <fmt/core.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
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
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数yaml配置文件路径 }";

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

namespace
{
constexpr double kGravity = 9.7833;
constexpr double kLightbarLength = 56e-3;
constexpr double kBigArmorWidth = 230e-3;
constexpr double kSmallArmorWidth = 135e-3;

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
}  // namespace

int main(int argc, char * argv[]) {
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const auto yaml = tools::load(config_path);
  const double yaw_offset = tools::read<double>(yaml, "yaw_offset") / 57.3;
  const double pitch_offset = tools::read<double>(yaml, "pitch_offset") / 57.3;

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
  // 添加原子变量来存储开火状态，确保线程安全
  std::atomic<bool> allow_fire = false;
  
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      // 更新开火状态
      allow_fire = plan.fire;
      
      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

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

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
        data["target_h"] = target->ekf_x()[10];  // 高低装甲板高度差估计
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }
      data["planner_selected_armor"] = planner.debug_armor_id;
      data["planner_delay_ms"] = planner.debug_delay_time * 1000.0;
      data["planner_spin_gate"] = planner.debug_used_spin_gate ? 1 : 0;

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  cv::Mat ballistic_panel(460, 840, CV_8UC3);

  while (!exiter.exit()) {
    camera.read(img, t);
    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    auto targets = tracker.track(armors, t);
    auto gs = gimbal.state();
    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) {
      current_target = targets.front();
    }
    auto current_plan = debug_planner.plan(current_target, gs.bullet_speed);
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

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      int armor_idx = 0;
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
        if (image_points.empty()) {
          ++armor_idx;
          continue;
        }

        // 在每个装甲板正下方绘制解算结果: yaw, x, y, z

        float min_x = image_points.front().x;
        float max_y = image_points.front().y;
        for (const auto & pt : image_points) {
          min_x = std::min(min_x, pt.x);
          max_y = std::max(max_y, pt.y);
        }

        int text_x = static_cast<int>(min_x);
        int text_y = static_cast<int>(max_y) + 22;
        text_x = std::max(0, std::min(text_x, img.cols - 220));
        text_y = std::max(30, std::min(text_y, img.rows - 130));

        // 竖排显示每个装甲板的解算值，使用高对比颜色（洋红）并加黑色描边
        std::vector<std::string> armor_lines = {
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
            img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX, font_scale_local,
            cv::Scalar(0, 0, 0), 3);
          cv::putText(
            img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX, font_scale_local,
            cv::Scalar(255, 0, 255), 2);
        }
        ++armor_idx;
      }

      // 预测装甲板转换轨迹：
      // 这里不再用“最近装甲板”做预览，而是复用 Planner 的选板逻辑，
      // 这样画面中的切板箭头才会和真正下发给云台的板号保持一致。
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
          future_selection.xyza.head(3), future_selection.xyza[3], target.armor_type, target.name);
        if (!pred_points.empty()) {
          cv::Point2f center(0.0f, 0.0f);
          for (const auto & pt : pred_points) {
            center.x += pt.x;
            center.y += pt.y;
          }
          center.x /= static_cast<float>(pred_points.size());
          center.y /= static_cast<float>(pred_points.size());

          raw_traj_centers.emplace_back(static_cast<int>(center.x), static_cast<int>(center.y));
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
        cv::circle(img, traj_centers[i], 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        cv::circle(img, traj_centers[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      }

      for (size_t i = 1; i < traj_centers.size(); ++i) {
        const bool switched = traj_ids[i] != traj_ids[i - 1];
        const cv::Scalar traj_color = switched ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 0);

        cv::arrowedLine(
          img, traj_centers[i - 1], traj_centers[i], cv::Scalar(0, 0, 0), 5, cv::LINE_AA, 0, 0.32);
        cv::arrowedLine(
          img, traj_centers[i - 1], traj_centers[i], traj_color, 2, cv::LINE_AA, 0, 0.32);

        if (switched) {
          cv::putText(
            img, "switch", traj_centers[i] + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(0, 165, 255), 1);
        }
      }

      if (!traj_centers.empty()) {
        cv::putText(
          img, "pred traj", traj_centers.front() + cv::Point(8, -10), cv::FONT_HERSHEY_SIMPLEX,
          0.5, cv::Scalar(255, 255, 0), 2);
      }

      if (current_plan.control) {
        Eigen::Vector4d aim_xyza = debug_planner.debug_xyza;
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
      }
    }
    
    // 创建一个新的图像来显示信息
    cv::Mat display_img;
    cv::resize(img, display_img, {}, 0.7, 0.7);  // 显示时缩小图片尺寸

    // 在图像上绘制文本信息
    int baseline = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.4;
    cv::Scalar color = cv::Scalar(0, 255, 255);  // 黄色
    int thickness = 1;
    

    // 画面中心上方显示开火提示（仅可开火时显示）
    if (current_plan.fire) {
      std::string fire_text = "fire!";
      int fire_baseline = 0;
      cv::Size fire_size =
        cv::getTextSize(fire_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &fire_baseline);
      cv::Point fire_org((display_img.cols - fire_size.width) / 2, fire_size.height + 10);
      cv::putText(
        display_img, fire_text, fire_org, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    // 右上角显示当前端到端延迟（从图像时间戳到当前显示）
    double latency_ms = tools::delta_time(std::chrono::steady_clock::now(), t) * 1000.0;
    std::string latency_info = fmt::format("latency: {:.2f} ms", latency_ms);
    int latency_baseline = 0;
    cv::Size latency_size =
      cv::getTextSize(latency_info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &latency_baseline);
    cv::Point latency_org(display_img.cols - latency_size.width - 10, latency_size.height + 10);
    cv::putText(
      display_img, latency_info, latency_org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 255, 255), 1);

    // 左上角直接显示 MPC 当前真正使用的诊断量。
    // 这几项就是现场判断“切错板 / 提前量过大 / 高低板高度差漂移”的最快入口。
    int info_x = 12;
    int info_y = 20;
    const int info_line_gap = 18;
    const auto current_w = current_target.has_value() ? current_target->ekf_x()[7] : 0.0;
    const auto current_h = current_target.has_value() ? current_target->ekf_x()[10] : 0.0;
    const std::vector<std::string> planner_lines = {
      fmt::format("armor_id: {}", debug_planner.debug_armor_id),
      fmt::format("spin_gate: {}", debug_planner.debug_used_spin_gate ? "on" : "off"),
      fmt::format("delay: {:.1f} ms", debug_planner.debug_delay_time * 1000.0),
      fmt::format("w: {:.2f} rad/s", current_w),
      fmt::format("h: {:.3f} m", current_h),
    };
    for (size_t i = 0; i < planner_lines.size(); ++i) {
      cv::Point org(info_x, info_y + static_cast<int>(i) * info_line_gap);
      cv::putText(
        display_img, planner_lines[i], org, font_face, font_scale, cv::Scalar(0, 0, 0),
        thickness + 2);
      cv::putText(
        display_img, planner_lines[i], org, font_face, font_scale, color, thickness);
    }

    draw_ballistic_panel(ballistic_panel, ballistic_diag);

    cv::imshow("Auto Aim Debug", display_img);  // 单一窗口显示所有信息
    cv::imshow("Ballistic Debug", ballistic_panel);
    auto key = cv::waitKey(1); 
    if (key == 'q') break; 
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
