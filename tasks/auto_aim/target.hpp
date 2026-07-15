#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Solver;

namespace target_state
{
enum Index
{
  CX,
  VCX,
  CY,
  VCY,
  CZ,
  VCZ,
  ROT_Z,
  VYAW,
  LOG_R1,
  LOG_R2,
  H,
  ROT_Y,
  ROT_X,
  SIZE
};
}  // namespace target_state

struct TargetEstimatorParams
{
  double acceleration_variance = 100.0;
  double yaw_acceleration_variance = 400.0;
  double roll_pitch_random_walk = 2e-3;
  double geometry_random_walk = 1e-4;
  double uvl_angle_variance = 2.5e-3;
  double uvl_center_variance = 9.0;
  double uvl_length_variance = 9.0;
  double nis_gate = 20.090;  // chi-square, 8 DoF, 99% confidence
};

struct ArmorPose
{
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
};

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped = false;
  int last_id = 0;  // debug only
  int tracker_debug_candidate_count = 0;
  bool tracker_debug_match_valid = false;
  int tracker_debug_match_id = -1;
  double tracker_debug_match_score = -1.0;
  double tracker_debug_reprojection_px = -1.0;
  double tracker_debug_xy_error_m = -1.0;
  double tracker_debug_z_error_m = -1.0;

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, const std::vector<double> & armor_z_offsets = {},
    bool fixed_center_rotation_model = false, double spin_speed_lock = 2.51,
    const TargetEstimatorParams & estimator_params = {});
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  bool update(const Armor & armor, const Solver & solver);
  bool update(const Armor & armor, int id, const Solver & solver);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  Eigen::Vector3d center_xyz_in_world() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  std::vector<ArmorPose> armor_pose_list() const;
  Eigen::Vector3d car_rpy() const;
  double radius(int id = 0) const;
  double armor_z_offset(int id) const;
  int physical_armor_id(int id) const;
  int armor_id_offset() const;
  void set_armor_id_offset(int offset, int reference_id = 0);
  bool fixed_center_rotation_model() const;
  double nis_gate() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_ = 0;
  int armor_id_offset_ = 0;
  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false, is_converged_ = false;
  bool fixed_center_rotation_model_ = false;
  double spin_speed_lock_ = 2.51;
  TargetEstimatorParams estimator_params_;
  std::vector<double> armor_z_offsets_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  bool update_uvl(const Armor & armor, int id, const Solver & solver);
  int normalize_armor_id(int id) const;

  bool uses_full_rotation() const;
  double armor_pitch() const;
  Eigen::Matrix3d state_rotation(const Eigen::VectorXd & x) const;
  ArmorPose armor_pose(const Eigen::VectorXd & x, int id) const;
  Eigen::VectorXd predict_uvl(
    const Eigen::VectorXd & x, int id, const Solver & solver) const;
  Eigen::MatrixXd uvl_jacobian(
    const Eigen::VectorXd & x, int id, const Solver & solver) const;
  Eigen::VectorXd predict_state(const Eigen::VectorXd & x, double dt) const;
  Eigen::MatrixXd predict_jacobian(const Eigen::VectorXd & x, double dt) const;
  Eigen::VectorXd inject_error(
    const Eigen::VectorXd & nominal, const Eigen::VectorXd & delta) const;
  Eigen::VectorXd box_minus(
    const Eigen::VectorXd & nominal, const Eigen::VectorXd & value) const;
  void clamp_state(Eigen::VectorXd & x) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
