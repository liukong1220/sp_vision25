#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

struct AimSelection
{
  bool valid = false;
  int armor_id = -1;
  bool used_spin_gate = false;
  double center_yaw = 0.0;
  double selected_delta_angle = 0.0;
  Eigen::Vector4d xyza = Eigen::Vector4d::Zero();
  std::vector<double> delta_angle_list;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  int debug_armor_id = -1;
  int debug_physical_armor_id = -1;
  bool debug_used_spin_gate = false;
  double debug_delay_time = 0.0;
  double debug_center_yaw = 0.0;
  double debug_selected_z_offset = 0.0;
  double debug_selected_aim_z_compensation = 0.0;
  double debug_selected_delta_angle = 0.0;
  bool debug_fixed_center_rotation_model = false;
  double debug_hit_fly_time = 0.0;
  int debug_hit_iter_count = 0;
  bool debug_hit_converged = false;
  double debug_fire_tracking_error = 0.0;
  double debug_fire_phase_limit = 0.0;
  bool debug_fire_track_ready = false;
  bool debug_fire_phase_ready = false;
  int debug_yaw_solver_status = -1;
  int debug_pitch_solver_status = -1;
  int debug_yaw_solver_iterations = 0;
  int debug_pitch_solver_iterations = 0;
  std::vector<double> debug_delta_angle_list;
  Planner(const std::string & config_path);
  ~Planner();

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);
  AimSelection preview_aim_selection(const Target & target) const;

private:
  std::string config_path_;
  uint64_t runtime_params_version_ = 0;
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double bullet_speed_min_;
  double bullet_speed_max_;
  double bullet_speed_fallback_;
  double coming_angle_, leaving_angle_;
  double outpost_coming_angle_, outpost_leaving_angle_;
  double outpost_delay_time_;
  std::vector<double> outpost_fire_z_compensation_;
  int lock_id_ = -1;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void refresh_runtime_params_if_needed();
  void setup_yaw_solver();
  void setup_pitch_solver();

  Eigen::Matrix<double, 2, 1> solve_aim_command(
    const Target & target, double bullet_speed, int & lock_id,
    AimSelection * selection = nullptr, Eigen::Vector3d * aim_xyz = nullptr) const;
  void update_debug_selection(const Target & target, const AimSelection & selection);
  std::pair<double, double> resolve_angle_window(const Target & target) const;
  double resolve_delay_time(const Target & target) const;
  double resolve_aim_z_compensation(const Target & target, int armor_id) const;
  Eigen::Vector3d resolve_aim_xyz(const Target & target, const AimSelection & selection) const;
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed, int initial_lock_id);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
