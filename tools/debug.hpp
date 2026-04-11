#ifndef TOOLS__DEBUG_HPP
#define TOOLS__DEBUG_HPP

#include <cstdint>
#include <string>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "tasks/auto_aim/armor.hpp"

namespace io
{
struct Command;
struct GimbalState;
}

namespace auto_aim
{
struct Plan;
}

namespace tools::debug
{

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

double rad2deg(double rad);
double deg2rad(double deg);

GimbalStateUnitMode parse_gimbal_state_unit_mode(const std::string & unit);
NormalizedGimbalState normalize_gimbal_state(
  const io::GimbalState & gs, GimbalStateUnitMode unit_mode);

bool has_cli_option(int argc, char * argv[], const std::string & long_option);
int64_t unix_time_ms();

std::string armor_type_to_string(auto_aim::ArmorType armor_type);
std::string armor_name_to_string(auto_aim::ArmorName armor_name);
std::string spin_direction_to_string(double yaw_rate_rad_s, double threshold = 0.15);
int spin_direction_sign(double yaw_rate_rad_s, double threshold = 0.15);

BallisticDiagnostic build_ballistic_diagnostic(
  const auto_aim::Plan & plan, const Eigen::Vector4d & aim_xyza, auto_aim::ArmorType armor_type,
  double bullet_speed, double yaw_offset, double pitch_offset);

BallisticDiagnostic build_ballistic_diagnostic(
  const io::Command & command, const Eigen::Vector4d & aim_xyza,
  auto_aim::ArmorType armor_type, double bullet_speed, double yaw_offset,
  double pitch_offset);

nlohmann::json ballistic_to_json(const BallisticDiagnostic & diag);
void draw_ballistic_panel(cv::Mat & panel, const BallisticDiagnostic & diag);

}  // namespace tools::debug

#endif  // TOOLS__DEBUG_HPP
