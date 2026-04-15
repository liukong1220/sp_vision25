#include "tools/debug_visualization.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "tools/debug.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace
{
using tools::debug_visualization::LiveOverlayOptions;
using tools::debug_visualization::LiveOverlayStage;

struct OverlayLayerMask
{
  bool show_armors = false;
  bool show_armor_labels = false;
  bool show_target_motion = false;
  bool show_aim = false;
  bool show_decision_panel = false;
  bool show_decision_track = false;
  bool show_footer = false;
  bool show_fire_banner = false;
  bool show_search_hint = false;
};

struct OverlaySmoothingState
{
  bool valid = false;
  int image_width = 0;
  int image_height = 0;
  std::string target_key;
  cv::Point2f center_label_anchor {};
  std::map<int, cv::Point2f> armor_label_anchors;
};

OverlaySmoothingState & overlay_smoothing_state()
{
  static OverlaySmoothingState state;
  return state;
}

void reset_overlay_smoothing_state()
{
  overlay_smoothing_state() = OverlaySmoothingState {};
}

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
    thickness + 2, cv::LINE_AA);
  cv::putText(
    img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
    cv::LINE_AA);
}

cv::Rect text_block_rect(
  const cv::Point & org, const std::vector<std::string> & lines, double scale,
  int thickness, int line_gap)
{
  int baseline = 0;
  int max_width = 0;
  int max_height = 0;
  for (const auto & line : lines) {
    const auto size =
      cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    max_width = std::max(max_width, size.width);
    max_height = std::max(max_height, size.height);
  }

  const int total_height =
    max_height + static_cast<int>(std::max<int>(0, static_cast<int>(lines.size()) - 1)) * line_gap;
  return {org.x - 4, org.y - max_height - 4, max_width + 8, total_height + baseline + 8};
}

cv::Point clamp_text_origin(
  const cv::Mat & img, const cv::Point2f & desired,
  const std::vector<std::string> & lines, double scale, int thickness, int line_gap)
{
  cv::Point org(
    static_cast<int>(std::lround(desired.x)),
    static_cast<int>(std::lround(desired.y)));
  cv::Rect rect = text_block_rect(org, lines, scale, thickness, line_gap);

  if (rect.x < 4) org.x += 4 - rect.x;
  if (rect.y < 4) org.y += 4 - rect.y;
  rect = text_block_rect(org, lines, scale, thickness, line_gap);

  if (rect.x + rect.width > img.cols - 4) {
    org.x -= rect.x + rect.width - (img.cols - 4);
  }
  if (rect.y + rect.height > img.rows - 4) {
    org.y -= rect.y + rect.height - (img.rows - 4);
  }

  return org;
}

cv::Point resolve_non_overlapping_text_origin(
  const cv::Mat & img, const cv::Point2f & desired,
  const std::vector<std::string> & lines, double scale, int thickness, int line_gap,
  const std::optional<cv::Rect> & avoid_rect)
{
  cv::Point best = clamp_text_origin(
    img, desired, lines, scale, thickness, line_gap);
  if (!avoid_rect.has_value()) return best;

  const std::array<cv::Point2f, 4> offsets = {
    cv::Point2f(0.0f, 0.0f),
    cv::Point2f(0.0f, 26.0f),
    cv::Point2f(34.0f, 0.0f),
    cv::Point2f(0.0f, -26.0f),
  };
  for (const auto & offset : offsets) {
    const cv::Point candidate = clamp_text_origin(
      img, desired + offset, lines, scale, thickness, line_gap);
    if ((text_block_rect(candidate, lines, scale, thickness, line_gap) & *avoid_rect).area() <= 0) {
      return candidate;
    }
    best = candidate;
  }
  return best;
}

std::optional<cv::Point2f> project_world_point(
  const auto_aim::Solver & solver, const Eigen::Vector3d & xyz_in_world)
{
  const auto pixels = solver.world2pixel(
    {cv::Point3f(
      static_cast<float>(xyz_in_world.x()),
      static_cast<float>(xyz_in_world.y()),
      static_cast<float>(xyz_in_world.z()))});
  if (pixels.empty()) return std::nullopt;
  return pixels.front();
}

double projected_radius_px(
  const auto_aim::Solver & solver, const Eigen::Vector3d & xyz_in_world,
  double world_radius)
{
  const auto pixels = solver.world2pixel(
    {
      cv::Point3f(
        static_cast<float>(xyz_in_world.x()),
        static_cast<float>(xyz_in_world.y()),
        static_cast<float>(xyz_in_world.z())),
      cv::Point3f(
        static_cast<float>(xyz_in_world.x()),
        static_cast<float>(xyz_in_world.y() + world_radius),
        static_cast<float>(xyz_in_world.z())),
    });
  if (pixels.size() < 2) return 14.0;
  return std::clamp(
    static_cast<double>(cv::norm(pixels[1] - pixels[0])), 8.0, 42.0);
}

OverlaySmoothingState & prepare_overlay_smoothing_state(
  const cv::Mat & annotated_img, const LiveOverlayOptions & options)
{
  auto & state = overlay_smoothing_state();
  const std::string target_key = fmt::format(
    "{}:{}:{}x{}", options.target_name, options.armor_type,
    annotated_img.cols, annotated_img.rows);
  if (
    !state.valid || state.image_width != annotated_img.cols ||
    state.image_height != annotated_img.rows || state.target_key != target_key)
  {
    state = OverlaySmoothingState {};
    state.valid = true;
    state.image_width = annotated_img.cols;
    state.image_height = annotated_img.rows;
    state.target_key = target_key;
  }
  return state;
}

cv::Point2f smooth_anchor(
  cv::Point2f & cached, const cv::Point2f & desired, bool enable_stabilize,
  double slow_alpha, double fast_alpha, double snap_distance)
{
  if (!enable_stabilize) {
    cached = desired;
    return desired;
  }

  if (cached == cv::Point2f()) {
    cached = desired;
    return desired;
  }

  const double distance = cv::norm(desired - cached);
  const double alpha = distance > snap_distance ? fast_alpha : slow_alpha;
  cached = cached + (desired - cached) * static_cast<float>(alpha);
  return cached;
}

cv::Point2f smoothed_armor_label_anchor(
  OverlaySmoothingState & state, int armor_idx, const cv::Point2f & desired,
  bool enable_stabilize, bool is_selected)
{
  auto & cached = state.armor_label_anchors[armor_idx];
  return smooth_anchor(
    cached, desired, enable_stabilize,
    is_selected ? 0.34 : 0.24,
    is_selected ? 0.74 : 0.58,
    is_selected ? 150.0 : 110.0);
}

cv::Scalar stage_color(LiveOverlayStage stage)
{
  switch (stage) {
    case LiveOverlayStage::kSearch:
      return cv::Scalar(165, 210, 228);
    case LiveOverlayStage::kTracking:
      return cv::Scalar(110, 255, 170);
    case LiveOverlayStage::kLocked:
      return cv::Scalar(0, 215, 255);
    case LiveOverlayStage::kFireReady:
      return cv::Scalar(0, 96, 255);
  }
  return cv::Scalar(220, 220, 220);
}

OverlayLayerMask build_layer_mask(
  const LiveOverlayOptions & options, LiveOverlayStage stage,
  const auto_aim::Plan & current_plan)
{
  OverlayLayerMask mask;
  mask.show_fire_banner = options.show_aim && current_plan.fire;
  mask.show_search_hint = stage == LiveOverlayStage::kSearch;

  if (stage == LiveOverlayStage::kSearch) return mask;

  if (!options.enable_state_layers) {
    mask.show_armors = options.show_armors;
    mask.show_armor_labels = options.show_armors && options.show_armor_labels;
    mask.show_target_motion = options.show_target_motion;
    mask.show_aim = options.show_aim && current_plan.control;
    mask.show_decision_panel = options.show_decision_hud;
    mask.show_decision_track = options.show_decision_hud && options.show_decision_track;
    mask.show_footer = options.show_footer && current_plan.control;
    return mask;
  }

  mask.show_armors = options.show_armors;
  mask.show_armor_labels = options.show_armors && options.show_armor_labels;
  mask.show_target_motion = options.show_target_motion;
  mask.show_decision_panel = options.show_decision_hud;

  if (stage == LiveOverlayStage::kTracking) return mask;

  mask.show_aim = options.show_aim && current_plan.control;
  mask.show_footer = options.show_footer;
  if (stage == LiveOverlayStage::kFireReady) {
    mask.show_decision_track =
      options.show_decision_hud && options.show_decision_track;
  }
  return mask;
}

std::optional<cv::Rect> draw_target_center_motion(
  cv::Mat & annotated_img, const auto_aim::Solver & solver,
  const auto_aim::Target & target, const LiveOverlayOptions & options,
  OverlaySmoothingState & smoothing_state)
{
  const Eigen::VectorXd x_state = target.ekf_x();
  if (x_state.size() < 8) return std::nullopt;

  const Eigen::Vector3d target_center(x_state[0], x_state[2], x_state[4]);
  const auto center_pixel = project_world_point(solver, target_center);
  if (!center_pixel.has_value()) return std::nullopt;

  const double vyaw = x_state[7];
  const double dy = std::clamp(vyaw * 52.0, -88.0, 88.0);
  const cv::Point2f start_pt = *center_pixel;
  const cv::Point2f end_pt = start_pt + cv::Point2f(0.0f, static_cast<float>(dy));

  cv::arrowedLine(
    annotated_img, start_pt, end_pt, cv::Scalar(0, 0, 0), 5, cv::LINE_AA, 0, 0.12);
  cv::arrowedLine(
    annotated_img, start_pt, end_pt, cv::Scalar(50, 255, 50), 3, cv::LINE_AA, 0, 0.12);
  cv::circle(
    annotated_img, start_pt, 5, cv::Scalar(50, 255, 50), -1, cv::LINE_AA);

  const std::string text = fmt::format("V_yaw: {:.2f}", vyaw);
  const cv::Point2f desired_anchor = start_pt + cv::Point2f(12.0f, -16.0f);
  const cv::Point2f smooth_anchor_pt = smooth_anchor(
    smoothing_state.center_label_anchor, desired_anchor,
    options.stabilize_annotations, 0.26, 0.64, 120.0);
  const cv::Point text_org = clamp_text_origin(
    annotated_img, smooth_anchor_pt, {text}, 0.58, 2, 18);
  draw_outlined_text(
    annotated_img, text, text_org, 0.58, cv::Scalar(50, 255, 50), 2);
  return text_block_rect(text_org, {text}, 0.58, 2, 18);
}

void draw_aim_overlay(
  cv::Mat & annotated_img, const auto_aim::Solver & solver,
  const auto_aim::Plan & current_plan, const auto_aim::Planner & debug_planner)
{
  if (!current_plan.control) return;

  const Eigen::Vector3d aim_xyz = debug_planner.debug_xyza.head<3>();
  const auto aim_center = project_world_point(solver, aim_xyz);
  if (!aim_center.has_value()) return;

  const int radius = static_cast<int>(projected_radius_px(solver, aim_xyz, 0.02));
  cv::circle(
    annotated_img, *aim_center, radius, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

  if (current_plan.fire) {
    const int cross_size = std::max(18, radius + 10);
    cv::line(
      annotated_img, *aim_center + cv::Point2f(-cross_size, -cross_size),
      *aim_center + cv::Point2f(cross_size, cross_size),
      cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    cv::line(
      annotated_img, *aim_center + cv::Point2f(-cross_size, cross_size),
      *aim_center + cv::Point2f(cross_size, -cross_size),
      cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }

  const double ctrl_scale = 34.0;
  const cv::Point2f end_pt = *aim_center + cv::Point2f(
    static_cast<float>(-current_plan.yaw_vel * ctrl_scale),
    static_cast<float>(current_plan.pitch_vel * ctrl_scale));
  cv::arrowedLine(
    annotated_img, *aim_center, end_pt, cv::Scalar(0, 215, 255), 4,
    cv::LINE_AA, 0, 0.2);
}

void draw_target_overlay(
  cv::Mat & annotated_img, const auto_aim::Solver & solver,
  const auto_aim::Target & target, const auto_aim::Plan & current_plan,
  const auto_aim::Planner & debug_planner, const LiveOverlayOptions & options,
  const OverlayLayerMask & layers)
{
  auto & smoothing_state = prepare_overlay_smoothing_state(annotated_img, options);
  std::optional<cv::Rect> center_motion_rect;
  if (layers.show_target_motion) {
    center_motion_rect = draw_target_center_motion(
      annotated_img, solver, target, options, smoothing_state);
  }

  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  int armor_idx = 0;
  for (const Eigen::Vector4d & xyza : armor_xyza_list) {
    const bool is_selected = armor_idx == debug_planner.debug_armor_id;
    const cv::Scalar armor_color =
      is_selected ? cv::Scalar(255, 0, 255) : cv::Scalar(120, 255, 170);
    auto image_points = solver.reproject_armor(
      xyza.head(3), xyza[3], target.armor_type, target.name);
    if (image_points.empty()) {
      ++armor_idx;
      continue;
    }

    if (layers.show_armors) {
      tools::draw_points(
        annotated_img, image_points, armor_color, is_selected ? 3 : 2);
    }

    if (layers.show_armor_labels) {
      float min_x = image_points.front().x;
      float max_x = image_points.front().x;
      float max_y = image_points.front().y;
      for (const auto & pt : image_points) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
      }

      const std::string armor_label =
        is_selected ? fmt::format("[sel] A{}", armor_idx) : fmt::format("A{}", armor_idx);
      std::vector<std::string> lines = {armor_label};
      if (armor_idx < static_cast<int>(debug_planner.debug_delta_angle_list.size())) {
        lines.push_back(fmt::format(
          "d_yaw {:+.1f}",
          tools::debug::rad2deg(debug_planner.debug_delta_angle_list[armor_idx])));
      }

      const cv::Point2f desired_anchor(
        (min_x + max_x) * 0.5f - (is_selected ? 30.0f : 16.0f),
        max_y + (is_selected ? 26.0f : 20.0f));
      const cv::Point2f smooth_anchor_pt = smoothed_armor_label_anchor(
        smoothing_state, armor_idx, desired_anchor,
        options.stabilize_annotations, is_selected);
      const cv::Point label_org = resolve_non_overlapping_text_origin(
        annotated_img, smooth_anchor_pt, lines, 0.46, is_selected ? 2 : 1, 18,
        is_selected ? center_motion_rect : std::nullopt);
      draw_outlined_text(
        annotated_img, armor_label, label_org, 0.46, armor_color, is_selected ? 2 : 1);
      if (lines.size() > 1) {
        draw_outlined_text(
          annotated_img, lines[1], {label_org.x, label_org.y + 18},
          0.40, armor_color, 1);
      }
    }

    ++armor_idx;
  }

  if (layers.show_aim) {
    draw_aim_overlay(annotated_img, solver, current_plan, debug_planner);
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
    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
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
  cv::Mat & display_img, const cv::Rect & rect, const LiveOverlayOptions & options,
  int track_top)
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
    display_img, {axis_mid, track_top}, {axis_mid, axis_y + 6},
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
    const int y = track_top + static_cast<int>((i % 2) * 16);
    const bool is_selected = static_cast<int>(i) == options.planner_armor_id;
    const cv::Scalar color =
      is_selected ? cv::Scalar(255, 0, 255) : cv::Scalar(110, 255, 170);
    cv::line(display_img, {x, axis_y}, {x, y}, color, is_selected ? 2 : 1, cv::LINE_AA);
    cv::circle(display_img, {x, y}, is_selected ? 5 : 4, color, -1, cv::LINE_AA);
    draw_outlined_text(
      display_img, fmt::format("A{}", i), {x - 8, y - 8}, 0.35, color, 1);
  }
}

std::string selection_reason(const LiveOverlayOptions & options)
{
  if (options.planner_armor_id < 0) return "no valid armor";
  if (!options.target_jumped) return "trust observed armor";
  if (options.planner_spin_gate) {
    return options.is_outpost ? "outpost spin window" : "entering fire window";
  }
  return "stable visible armor";
}

std::string fire_reason(const auto_aim::Plan & current_plan)
{
  if (!current_plan.control) return "planner idle";
  if (current_plan.fire) return "aligned and ready";

  const double yaw_error_deg =
    tools::debug::rad2deg(tools::limit_rad(current_plan.target_yaw - current_plan.yaw));
  const double pitch_error_deg =
    tools::debug::rad2deg(current_plan.target_pitch - current_plan.pitch);

  if (std::abs(yaw_error_deg) >= std::abs(pitch_error_deg)) {
    return fmt::format("hold yaw {:+.2f} deg", yaw_error_deg);
  }
  return fmt::format("hold pitch {:+.2f} deg", pitch_error_deg);
}

std::string model_reason(const LiveOverlayOptions & options)
{
  if (options.is_outpost) {
    return fmt::format(
      "outpost z {:+.3f} aim {:+.3f} m", options.current_selected_z_offset,
      options.current_selected_aim_z_compensation);
  }
  if (options.current_fixed_model) return "fixed center model";
  return fmt::format("follow center h {:.3f} m", options.current_h);
}

void draw_search_hint(cv::Mat & display_img)
{
  const cv::Rect rect(
    std::max(12, (display_img.cols - 220) / 2), 16, 220, 42);
  blend_panel(
    display_img, rect, cv::Scalar(14, 28, 44), 0.66, cv::Scalar(125, 194, 220));
  draw_outlined_text(
    display_img, "SEARCH TARGET", {rect.x + 18, rect.y + 28},
    0.58, cv::Scalar(220, 245, 255), 2);
}

void draw_decision_panel(
  cv::Mat & display_img, const auto_aim::Plan & current_plan,
  const LiveOverlayOptions & options, LiveOverlayStage stage, bool show_track)
{
  std::vector<std::pair<std::string, std::string>> info_lines = {
    {
      "select",
      fmt::format(
        "{}{} | {}",
        options.planner_armor_id >= 0 ? fmt::format("A{}", options.planner_armor_id) : "A-",
        options.planner_physical_armor_id >= 0 ?
        fmt::format("/P{}", options.planner_physical_armor_id) : "",
        selection_reason(options)),
    },
    {"fire", fire_reason(current_plan)},
    {"model", model_reason(options)},
    {
      "turn",
      fmt::format("{} | w {:+.2f}", options.planner_turn_direction, options.current_w),
    },
    {
      "state",
      fmt::format(
        "spin {}  center {:+.1f}  delay {:.1f}ms",
        options.planner_spin_gate ? "ON" : "OFF",
        options.planner_center_yaw_deg, options.planner_delay_ms),
    },
  };

  if (options.is_outpost) {
    const std::string tracker_state =
      options.tracker_candidate_count <= 0 ? "miss" :
      options.tracker_match_valid ? "ok" : "reject";
    const std::string tracker_id =
      options.tracker_match_id >= 0 ? fmt::format("A{}", options.tracker_match_id) : "A-";
    info_lines.push_back({
      "track",
      fmt::format(
        "cand {}  {} {}  s {:.2f}  rp {:.1f}",
        options.tracker_candidate_count, tracker_id, tracker_state,
        options.tracker_match_score, options.tracker_reprojection_px),
    });
    info_lines.push_back({
      "geom",
      fmt::format(
        "xy {:.3f}m  z {:.3f}m  hit {:.1f}ms",
        options.tracker_xy_error_m, options.tracker_z_error_m, options.planner_hit_fly_time_ms),
    });
    info_lines.push_back({
      "iter",
      fmt::format(
        "{} iter  {}",
        options.planner_hit_iter_count,
        options.planner_hit_converged ? "converged" : "max-iter"),
    });
  }

  const int panel_height =
    (show_track ? 148 : 80) + static_cast<int>(info_lines.size()) * 18;
  const cv::Rect panel_rect(
    std::max(12, display_img.cols - 340), 14, 326,
    std::min(panel_height, display_img.rows - 28));
  blend_panel(
    display_img, panel_rect, cv::Scalar(10, 21, 35), 0.72,
    cv::Scalar(125, 194, 220));

  const cv::Scalar title_color(224, 244, 255);
  const cv::Scalar value_color(118, 236, 255);
  const cv::Scalar accent_color = stage_color(stage);

  draw_outlined_text(
    display_img, "Decision HUD", {panel_rect.x + 16, panel_rect.y + 24}, 0.62,
    title_color, 2);
  draw_outlined_text(
    display_img, live_overlay_stage_to_string(stage),
    {panel_rect.x + 16, panel_rect.y + 52}, 0.74, accent_color, 2);
  draw_outlined_text(
    display_img,
    fmt::format("{} / {}", options.target_name, options.armor_type),
    {panel_rect.x + 16, panel_rect.y + 78}, 0.48, title_color, 1);

  for (size_t i = 0; i < info_lines.size(); ++i) {
    draw_outlined_text(
      display_img, fmt::format("{}:", info_lines[i].first),
      {panel_rect.x + 16, panel_rect.y + 102 + static_cast<int>(i) * 18},
      0.41, title_color, 1);
    draw_outlined_text(
      display_img, info_lines[i].second,
      {panel_rect.x + 86, panel_rect.y + 102 + static_cast<int>(i) * 18},
      0.40, value_color, 1);
  }

  if (show_track) {
    const int track_top =
      panel_rect.y + 110 + static_cast<int>(info_lines.size()) * 18;
    draw_decision_track(display_img, panel_rect, options, track_top);
  }
}

void draw_live_status(
  cv::Mat & display_img, const auto_aim::Plan & current_plan,
  const LiveOverlayOptions & options, LiveOverlayStage stage,
  const OverlayLayerMask & layers)
{
  if (layers.show_fire_banner && current_plan.fire) draw_fire_banner(display_img);

  blend_panel(
    display_img, {12, 14, 212, 92}, cv::Scalar(14, 28, 44), 0.68,
    cv::Scalar(125, 194, 220));
  draw_outlined_text(
    display_img, fmt::format("stage: {}", live_overlay_stage_to_string(stage)),
    {24, 36}, 0.50, stage_color(stage), 2);
  draw_outlined_text(
    display_img, fmt::format("latency: {:.2f} ms", options.latency_ms),
    {24, 60}, 0.48, cv::Scalar(220, 245, 255), 1);
  draw_outlined_text(
    display_img, fmt::format("delay: {:.1f} ms", options.planner_delay_ms),
    {24, 82}, 0.48, cv::Scalar(118, 236, 255), 1);

  draw_center_crosshair(display_img);
  if (layers.show_search_hint) draw_search_hint(display_img);
  if (layers.show_decision_panel) {
    draw_decision_panel(
      display_img, current_plan, options, stage, layers.show_decision_track);
  }
  if (layers.show_footer) draw_footer_panel(display_img, current_plan);
}
}  // namespace

namespace tools::debug_visualization
{

const char * live_overlay_stage_to_string(LiveOverlayStage stage)
{
  switch (stage) {
    case LiveOverlayStage::kSearch:
      return "SEARCH";
    case LiveOverlayStage::kTracking:
      return "TRACKING";
    case LiveOverlayStage::kLocked:
      return "LOCKED";
    case LiveOverlayStage::kFireReady:
      return "FIRE READY";
  }
  return "UNKNOWN";
}

LiveOverlayStage resolve_live_overlay_stage(
  bool has_target, const auto_aim::Plan & current_plan)
{
  if (!has_target) return LiveOverlayStage::kSearch;
  if (!current_plan.control) return LiveOverlayStage::kTracking;
  if (current_plan.fire) return LiveOverlayStage::kFireReady;
  return LiveOverlayStage::kLocked;
}

cv::Mat render_live_debug_frame(
  const cv::Mat & source_img, const auto_aim::Solver & solver,
  const std::optional<auto_aim::Target> & current_target,
  const auto_aim::Plan & current_plan, const auto_aim::Planner & debug_planner,
  const LiveOverlayOptions & options)
{
  const LiveOverlayStage stage =
    resolve_live_overlay_stage(current_target.has_value(), current_plan);
  const OverlayLayerMask layers = build_layer_mask(options, stage, current_plan);

  cv::Mat annotated_img = source_img.clone();
  if (current_target.has_value()) {
    draw_target_overlay(
      annotated_img, solver, *current_target, current_plan, debug_planner, options, layers);
  } else {
    reset_overlay_smoothing_state();
  }

  cv::Mat display_img;
  cv::resize(
    annotated_img, display_img, {}, options.display_scale, options.display_scale);
  draw_live_status(display_img, current_plan, options, stage, layers);
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
