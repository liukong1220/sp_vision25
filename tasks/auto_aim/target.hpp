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
    bool fixed_center_rotation_model = false, double spin_speed_lock = 2.51);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);
  void update(const Armor & armor, int id);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  Eigen::Vector3d center_xyz_in_world() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  double armor_z_offset(int id) const;
  int physical_armor_id(int id) const;
  int armor_id_offset() const;
  void set_armor_id_offset(int offset, int reference_id = 0);
  bool fixed_center_rotation_model() const;

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
  std::vector<double> armor_z_offsets_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  int normalize_armor_id(int id) const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
