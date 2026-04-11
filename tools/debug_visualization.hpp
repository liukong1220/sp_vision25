#ifndef TOOLS__DEBUG_VISUALIZATION_HPP
#define TOOLS__DEBUG_VISUALIZATION_HPP

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"

namespace tools::debug_visualization
{

struct LiveOverlayOptions
{
  double display_scale = 1.0;
  double latency_ms = 0.0;
  std::string target_name = "none";
  std::string armor_type = "none";
  int planner_armor_id = -1;
  bool planner_spin_gate = false;
  double planner_delay_ms = 0.0;
  double planner_center_yaw_deg = 0.0;
  std::vector<double> planner_delta_angles_deg;
  std::string planner_turn_direction = "STEADY";
  int planner_turn_sign = 0;
  double current_w = 0.0;
  double current_h = 0.0;
  double current_selected_z_offset = 0.0;
  bool current_fixed_model = false;
  bool is_outpost = false;
};

struct OfflineOverlayOptions
{
  double display_scale = 1.0;
  int frame_index = 0;
  double playback_t_s = 0.0;
  double playback_speed = 1.0;
  double latency_ms = 0.0;
  double bullet_speed = 0.0;
  double current_w = 0.0;
  double current_h = 0.0;
  bool command_control = false;
  double command_yaw_deg = 0.0;
  double command_pitch_deg = 0.0;
  double gimbal_yaw_deg = 0.0;
};

cv::Mat render_live_debug_frame(
  const cv::Mat & source_img, const auto_aim::Solver & solver,
  const std::optional<auto_aim::Target> & current_target,
  const auto_aim::Plan & current_plan, const auto_aim::Planner & debug_planner,
  const LiveOverlayOptions & options);

cv::Mat render_offline_debug_frame(
  const cv::Mat & source_img, const auto_aim::Solver & solver,
  const std::optional<auto_aim::Target> & current_target,
  const auto_aim::Aimer & aimer, const io::Command & command,
  const OfflineOverlayOptions & options);

}  // namespace tools::debug_visualization

#endif  // TOOLS__DEBUG_VISUALIZATION_HPP
