#include "tools/debug_visualization.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "tools/debug.hpp"
#include "tools/img_tools.hpp"

namespace
{
void blend_panel(
  cv::Mat & img, const cv::Rect & rect, const cv::Scalar & fill, double alpha,
  const cv::Scalar & border)
{
  const cv::Rect bounds(0, 0, img.cols, img.rows);
  const cv::Rect clipped = rect & bounds;
  if (clipped.width <= 0 || clipped.height <= 0) return;

  cv::Mat roi = img(clipped);
  cv::Mat overlay(roi.size(), roi.type(), fill);
  cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
  cv::rectangle(img, clipped, border, 1, cv::LINE_AA);
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

void draw_target_overlay(
  cv::Mat & annotated_img, const auto_aim::Solver & solver,
  const auto_aim::Target & target, const auto_aim::Plan & current_plan,
  const auto_aim::Planner & debug_planner,
  const tools::debug_visualization::LiveOverlayOptions & options)
{
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  std::optional<cv::Point> selected_center;
  int armor_idx = 0;
  for (const Eigen::Vector4d & xyza : armor_xyza_list) {
    const bool is_selected = armor_idx == debug_planner.debug_armor_id;
    const cv::Scalar armor_color =
      is_selected ? cv::Scalar(255, 0, 255) : cv::Scalar(120, 255, 170);
    auto image_points = solver.reproject_armor(
      xyza.head(3), xyza[3], target.armor_type, target.name);
    tools::draw_points(
      annotated_img, image_points, armor_color, is_selected ? 3 : 2);
    if (image_points.empty()) {
      ++armor_idx;
      continue;
    }

    float min_x = image_points.front().x;
    float max_y = image_points.front().y;
    cv::Point2f armor_center(0.0f, 0.0f);
    for (const auto & pt : image_points) {
      min_x = std::min(min_x, pt.x);
      max_y = std::max(max_y, pt.y);
      armor_center += pt;
    }
    armor_center *= 1.0f / static_cast<float>(image_points.size());
    if (is_selected) {
      selected_center = cv::Point(
        static_cast<int>(armor_center.x), static_cast<int>(armor_center.y));
    }

    int text_x = static_cast<int>(min_x);
    int text_y = static_cast<int>(max_y) + 22;
    text_x = std::max(0, std::min(text_x, annotated_img.cols - 220));
    text_y = std::max(30, std::min(text_y, annotated_img.rows - 130));

    const std::vector<std::string> armor_lines = {
      fmt::format("{} armor:{}", is_selected ? "[sel]" : "[viz]", armor_idx),
      fmt::format(
        "d_yaw: {:+.1f}",
        armor_idx < static_cast<int>(debug_planner.debug_delta_angle_list.size()) ?
          tools::debug::rad2deg(debug_planner.debug_delta_angle_list[armor_idx]) : 0.0),
      fmt::format("yaw: {:.1f}", tools::debug::rad2deg(xyza[3])),
      fmt::format("x: {:.2f}", xyza[1]),
      fmt::format("y: {:.2f}", xyza[0]),
      fmt::format("z: {:.2f}", xyza[2]),
    };
    const double font_scale = 0.50;
    const int line_gap = 22;
    for (size_t line_i = 0; line_i < armor_lines.size(); ++line_i) {
      const cv::Point org(text_x, text_y + static_cast<int>(line_i) * line_gap);
      cv::putText(
        annotated_img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX,
        font_scale, cv::Scalar(0, 0, 0), 3);
      cv::putText(
        annotated_img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX,
        font_scale, cv::Scalar(255, 0, 255), 2);
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
    const auto future_selection =
      debug_planner.preview_aim_selection(target_future);
    if (!future_selection.valid) break;

    auto pred_points = solver.reproject_armor(
      future_selection.xyza.head(3), future_selection.xyza[3],
      target.armor_type, target.name);
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

  for (const auto & center : traj_centers) {
    cv::circle(annotated_img, center, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    cv::circle(
      annotated_img, center, 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
  }

  for (size_t i = 1; i < traj_centers.size(); ++i) {
    const bool switched = traj_ids[i] != traj_ids[i - 1];
    const cv::Scalar traj_color =
      switched ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 0);

    cv::arrowedLine(
      annotated_img, traj_centers[i - 1], traj_centers[i], cv::Scalar(0, 0, 0),
      5, cv::LINE_AA, 0, 0.32);
    cv::arrowedLine(
      annotated_img, traj_centers[i - 1], traj_centers[i], traj_color, 2,
      cv::LINE_AA, 0, 0.32);

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

  if (selected_center.has_value()) {
    const cv::Scalar turn_color =
      options.planner_turn_sign > 0 ? cv::Scalar(90, 220, 120) :
      options.planner_turn_sign < 0 ? cv::Scalar(0, 186, 255) :
      cv::Scalar(210, 210, 210);
    cv::circle(
      annotated_img, *selected_center, 10, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::circle(annotated_img, *selected_center, 10, turn_color, 2, cv::LINE_AA);

    if (options.planner_turn_sign != 0) {
      const int arrow_len = std::clamp(
        static_cast<int>(std::abs(options.current_w) * 42.0), 18, 72);
      const cv::Point arrow_end =
        *selected_center + cv::Point(0, options.planner_turn_sign > 0 ? arrow_len : -arrow_len);
      cv::arrowedLine(
        annotated_img, *selected_center, arrow_end, cv::Scalar(0, 0, 0), 5,
        cv::LINE_AA, 0, 0.18);
      cv::arrowedLine(
        annotated_img, *selected_center, arrow_end, turn_color, 2,
        cv::LINE_AA, 0, 0.18);
    }

    draw_outlined_text(
      annotated_img, fmt::format("w {:+.2f}", options.current_w),
      *selected_center + cv::Point(16, -18), 0.48, turn_color, 1);
    draw_outlined_text(
      annotated_img, options.planner_turn_direction,
      *selected_center + cv::Point(16, 8), 0.40, turn_color, 1);
  }

  if (current_plan.control) {
    const Eigen::Vector4d aim_xyza = debug_planner.debug_xyza;
    auto image_points = solver.reproject_armor(
      aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
    tools::draw_points(annotated_img, image_points, {0, 0, 255});
    if (!image_points.empty()) {
      cv::Point2f aim_center(0.0f, 0.0f);
      for (const auto & pt : image_points) {
        aim_center += pt;
      }
      aim_center *= 1.0f / static_cast<float>(image_points.size());
      cv::drawMarker(
        annotated_img, aim_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS,
        18, 2, cv::LINE_AA);
      draw_outlined_text(
        annotated_img, "aim", cv::Point(
          static_cast<int>(aim_center.x) + 10,
          static_cast<int>(aim_center.y) - 10),
        0.45, cv::Scalar(0, 0, 255), 1);
    }
  }
}

void draw_fire_banner(cv::Mat & display_img)
{
  const std::string fire_text = "fire!";
  int fire_baseline = 0;
  const cv::Size fire_size = cv::getTextSize(
    fire_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &fire_baseline);
  const cv::Point fire_org(
    (display_img.cols - fire_size.width) / 2, fire_size.height + 10);
  cv::putText(
    display_img, fire_text, fire_org, cv::FONT_HERSHEY_SIMPLEX, 1.0,
    cv::Scalar(0, 0, 255), 2);
}

void draw_center_crosshair(cv::Mat & display_img)
{
  const cv::Point center(display_img.cols / 2, display_img.rows / 2);
  cv::circle(display_img, center, 6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::line(
    display_img, center + cv::Point(-18, 0), center + cv::Point(-8, 0),
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::line(
    display_img, center + cv::Point(8, 0), center + cv::Point(18, 0),
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::line(
    display_img, center + cv::Point(0, -18), center + cv::Point(0, -8),
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::line(
    display_img, center + cv::Point(0, 8), center + cv::Point(0, 18),
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void draw_footer_panel(cv::Mat & display_img, const auto_aim::Plan & current_plan)
{
  const std::string yaw_line = fmt::format(
    "yaw  tgt:{:+.2f} cmd:{:+.2f} vel:{:+.2f} acc:{:+.2f}",
    tools::debug::rad2deg(current_plan.target_yaw),
    tools::debug::rad2deg(current_plan.yaw),
    tools::debug::rad2deg(current_plan.yaw_vel),
    tools::debug::rad2deg(current_plan.yaw_acc));
  const std::string pitch_line = fmt::format(
    "pitch tgt:{:+.2f} cmd:{:+.2f} vel:{:+.2f} acc:{:+.2f}",
    tools::debug::rad2deg(current_plan.target_pitch),
    tools::debug::rad2deg(current_plan.pitch),
    tools::debug::rad2deg(current_plan.pitch_vel),
    tools::debug::rad2deg(current_plan.pitch_acc));

  int baseline = 0;
  const cv::Size yaw_size =
    cv::getTextSize(yaw_line, cv::FONT_HERSHEY_SIMPLEX, 0.52, 2, &baseline);
  const cv::Size pitch_size =
    cv::getTextSize(pitch_line, cv::FONT_HERSHEY_SIMPLEX, 0.52, 2, &baseline);
  const int width = std::max(yaw_size.width, pitch_size.width) + 30;
  const int height = yaw_size.height + pitch_size.height + 34;
  const cv::Rect rect(
    std::max(12, (display_img.cols - width) / 2),
    std::max(12, display_img.rows - height - 16), width, height);

  blend_panel(
    display_img, rect, cv::Scalar(18, 30, 48), 0.68, cv::Scalar(125, 194, 220));
  draw_outlined_text(
    display_img, yaw_line, {rect.x + 14, rect.y + 24}, 0.52,
    cv::Scalar(220, 245, 255), 2);
  draw_outlined_text(
    display_img, pitch_line, {rect.x + 14, rect.y + 48}, 0.52,
    cv::Scalar(220, 245, 255), 2);
}

void draw_decision_track(
  cv::Mat & display_img, const cv::Rect & rect,
  const tools::debug_visualization::LiveOverlayOptions & options)
{
  if (options.planner_delta_angles_deg.empty()) return;

  const int axis_y = rect.y + rect.height - 22;
  const int axis_x0 = rect.x + 20;
  const int axis_x1 = rect.x + rect.width - 20;
  const int axis_mid = (axis_x0 + axis_x1) / 2;

  cv::line(
    display_img, {axis_x0, axis_y}, {axis_x1, axis_y},
    cv::Scalar(110, 180, 210), 1, cv::LINE_AA);
  cv::line(
    display_img, {axis_mid, rect.y + 108}, {axis_mid, axis_y + 6},
    cv::Scalar(90, 140, 180), 1, cv::LINE_AA);

  draw_outlined_text(
    display_img, "-180", {axis_x0 - 8, axis_y + 18}, 0.35,
    cv::Scalar(165, 210, 228), 1);
  draw_outlined_text(
    display_img, "0", {axis_mid - 3, axis_y + 18}, 0.35,
    cv::Scalar(165, 210, 228), 1);
  draw_outlined_text(
    display_img, "+180", {axis_x1 - 18, axis_y + 18}, 0.35,
    cv::Scalar(165, 210, 228), 1);

  for (size_t i = 0; i < options.planner_delta_angles_deg.size(); ++i) {
    const double delta = std::clamp(options.planner_delta_angles_deg[i], -180.0, 180.0);
    const double ratio = (delta + 180.0) / 360.0;
    const int x = axis_x0 + static_cast<int>(ratio * (axis_x1 - axis_x0));
    const int y = rect.y + 120 + static_cast<int>((i % 2) * 16);
    const bool is_selected = static_cast<int>(i) == options.planner_armor_id;
    const cv::Scalar color =
      is_selected ? cv::Scalar(255, 0, 255) : cv::Scalar(110, 255, 170);
    cv::line(display_img, {x, axis_y}, {x, y}, color, is_selected ? 2 : 1, cv::LINE_AA);
    cv::circle(display_img, {x, y}, is_selected ? 5 : 4, color, -1, cv::LINE_AA);
    draw_outlined_text(
      display_img, fmt::format("A{}", i), {x - 8, y - 8}, 0.35, color, 1);
  }
}

void draw_decision_panel(
  cv::Mat & display_img, const tools::debug_visualization::LiveOverlayOptions & options)
{
  const cv::Rect panel_rect(
    std::max(12, display_img.cols - 332), 14, 318,
    std::min(188, display_img.rows - 28));
  blend_panel(
    display_img, panel_rect, cv::Scalar(10, 21, 35), 0.72,
    cv::Scalar(125, 194, 220));

  const cv::Scalar title_color(224, 244, 255);
  const cv::Scalar value_color(118, 236, 255);
  const cv::Scalar accent_color =
    options.planner_turn_sign > 0 ? cv::Scalar(90, 220, 120) :
    options.planner_turn_sign < 0 ? cv::Scalar(0, 186, 255) :
    cv::Scalar(220, 220, 220);

  draw_outlined_text(
    display_img, "Decision HUD", {panel_rect.x + 16, panel_rect.y + 24}, 0.62,
    title_color, 2);
  draw_outlined_text(
    display_img, options.planner_turn_direction, {panel_rect.x + 16, panel_rect.y + 52},
    0.82, accent_color, 2);
  draw_outlined_text(
    display_img,
    fmt::format("{} / {}", options.target_name, options.armor_type),
    {panel_rect.x + 16, panel_rect.y + 78}, 0.48, title_color, 1);

  const std::vector<std::string> info_lines = {
    fmt::format("selected armor: {}", options.planner_armor_id),
    fmt::format("spin gate: {}", options.planner_spin_gate ? "ON" : "OFF"),
    fmt::format("center yaw: {:+.1f} deg", options.planner_center_yaw_deg),
    fmt::format("target w: {:+.2f} rad/s", options.current_w),
    options.is_outpost ?
      fmt::format("z offset: {:+.3f} m", options.current_selected_z_offset) :
      fmt::format("armor h: {:.3f} m", options.current_h),
  };
  for (size_t i = 0; i < info_lines.size(); ++i) {
    draw_outlined_text(
      display_img, info_lines[i], {panel_rect.x + 16, panel_rect.y + 100 + static_cast<int>(i) * 18},
      0.42, value_color, 1);
  }

  draw_decision_track(display_img, panel_rect, options);
}

void draw_live_status(
  cv::Mat & display_img, const auto_aim::Plan & current_plan,
  const tools::debug_visualization::LiveOverlayOptions & options)
{
  if (current_plan.fire) draw_fire_banner(display_img);

  blend_panel(
    display_img, {12, 14, 196, 72}, cv::Scalar(14, 28, 44), 0.68,
    cv::Scalar(125, 194, 220));
  draw_outlined_text(
    display_img, fmt::format("latency: {:.2f} ms", options.latency_ms),
    {24, 38}, 0.48, cv::Scalar(220, 245, 255), 1);
  draw_outlined_text(
    display_img, fmt::format("delay: {:.1f} ms", options.planner_delay_ms),
    {24, 60}, 0.48, cv::Scalar(118, 236, 255), 1);

  draw_center_crosshair(display_img);
  draw_decision_panel(display_img, options);
  draw_footer_panel(display_img, current_plan);
}
}  // namespace

namespace tools::debug_visualization
{

cv::Mat render_live_debug_frame(
  const cv::Mat & source_img, const auto_aim::Solver & solver,
  const std::optional<auto_aim::Target> & current_target,
  const auto_aim::Plan & current_plan, const auto_aim::Planner & debug_planner,
  const LiveOverlayOptions & options)
{
  cv::Mat annotated_img = source_img.clone();
  if (current_target.has_value()) {
    draw_target_overlay(
      annotated_img, solver, *current_target, current_plan, debug_planner, options);
  }

  cv::Mat display_img;
  cv::resize(
    annotated_img, display_img, {}, options.display_scale, options.display_scale);
  draw_live_status(display_img, current_plan, options);
  return display_img;
}

cv::Mat render_offline_debug_frame(
  const cv::Mat & source_img, const auto_aim::Solver & solver,
  const std::optional<auto_aim::Target> & current_target,
  const auto_aim::Aimer & aimer, const io::Command & command,
  const OfflineOverlayOptions & options)
{
  cv::Mat annotated_img = source_img.clone();
  if (current_target.has_value()) {
    const auto & target = *current_target;
    for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
      auto image_points = solver.reproject_armor(
        xyza.head(3), xyza[3], target.armor_type, target.name);
      tools::draw_points(annotated_img, image_points, {0, 255, 0});
    }

    if (aimer.debug_aim_point.valid) {
      const auto image_points = solver.reproject_armor(
        aimer.debug_aim_point.xyza.head(3), aimer.debug_aim_point.xyza[3],
        target.armor_type, target.name);
      tools::draw_points(annotated_img, image_points, {0, 0, 255});
    }
  }

  cv::Mat display_img;
  cv::resize(
    annotated_img, display_img, {}, options.display_scale, options.display_scale);

  if (command.shoot) draw_fire_banner(display_img);

  const std::vector<std::string> debug_lines = {
    fmt::format("command: {}, {:.2f}, {:.2f}", options.command_control,
      options.command_yaw_deg, options.command_pitch_deg),
    fmt::format("gimbal yaw: {:.2f}", options.gimbal_yaw_deg),
    fmt::format("frame: {}", options.frame_index),
    fmt::format("playback: {:.2f}s x{:.2f}", options.playback_t_s, options.playback_speed),
    fmt::format("latency: {:.2f} ms", options.latency_ms),
    fmt::format("bullet_speed: {:.1f} m/s", options.bullet_speed),
    fmt::format("w: {:.2f} rad/s", options.current_w),
    fmt::format("h: {:.3f} m", options.current_h),
  };
  for (size_t i = 0; i < debug_lines.size(); ++i) {
    draw_outlined_text(
      display_img, debug_lines[i], {12, 20 + static_cast<int>(i) * 18}, 0.45,
      cv::Scalar(0, 255, 255));
  }

  return display_img;
}

}  // namespace tools::debug_visualization
