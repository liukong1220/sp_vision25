#include "target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "solver.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double kMinRadius = 0.05;
constexpr double kMaxRadius = 1.0;
constexpr double kNumericEpsilon = 1e-5;

Eigen::Vector3d rotation_vector(const Eigen::VectorXd & x)
{
  return {x[target_state::ROT_X], x[target_state::ROT_Y], x[target_state::ROT_Z]};
}

Eigen::Matrix3d rotation_from_state(const Eigen::VectorXd & x, bool full_rotation)
{
  if (!full_rotation) {
    return tools::so3_exp(Eigen::Vector3d(0.0, 0.0, x[target_state::ROT_Z]));
  }
  return tools::so3_exp(rotation_vector(x));
}

void write_rotation(Eigen::VectorXd & x, const Eigen::Matrix3d & rotation)
{
  const Eigen::Vector3d rotvec = tools::so3_log(rotation);
  x[target_state::ROT_X] = rotvec.x();
  x[target_state::ROT_Y] = rotvec.y();
  x[target_state::ROT_Z] = rotvec.z();
}

Eigen::VectorXd inject_state(
  const Eigen::VectorXd & nominal, const Eigen::VectorXd & delta, bool full_rotation)
{
  Eigen::VectorXd value = nominal + delta;
  const Eigen::Vector3d delta_rotation = full_rotation ?
    Eigen::Vector3d(
      delta[target_state::ROT_X], delta[target_state::ROT_Y], delta[target_state::ROT_Z]) :
    Eigen::Vector3d(0.0, 0.0, delta[target_state::ROT_Z]);
  write_rotation(
    value, rotation_from_state(nominal, full_rotation) * tools::so3_exp(delta_rotation));
  if (!full_rotation) {
    value[target_state::ROT_X] = 0.0;
    value[target_state::ROT_Y] = 0.0;
  }
  return value;
}

Eigen::VectorXd subtract_state(
  const Eigen::VectorXd & nominal, const Eigen::VectorXd & value, bool full_rotation)
{
  Eigen::VectorXd delta = value - nominal;
  const Eigen::Vector3d delta_rotation = tools::so3_log(
    rotation_from_state(nominal, full_rotation).transpose() *
    rotation_from_state(value, full_rotation));
  delta[target_state::ROT_X] = full_rotation ? delta_rotation.x() : 0.0;
  delta[target_state::ROT_Y] = full_rotation ? delta_rotation.y() : 0.0;
  delta[target_state::ROT_Z] = delta_rotation.z();
  return delta;
}

Eigen::Vector4d lightbar_observation(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const cv::Point2f delta = top - bottom;
  const cv::Point2f center = (top + bottom) * 0.5F;
  return {
    std::atan2(delta.x, delta.y), center.x, center.y,
    std::hypot(delta.x, delta.y)};
}

Eigen::VectorXd uvl_observation(const std::vector<cv::Point2f> & points)
{
  if (points.size() != 4) return {};
  Eigen::VectorXd observation(8);
  observation.segment<4>(0) = lightbar_observation(points[0], points[3]);
  observation.segment<4>(4) = lightbar_observation(points[1], points[2]);
  return observation;
}

Eigen::VectorXd subtract_uvl(const Eigen::VectorXd & lhs, const Eigen::VectorXd & rhs)
{
  Eigen::VectorXd residual = lhs - rhs;
  if (residual.size() == 8) {
    residual[0] = tools::limit_rad(residual[0]);
    residual[4] = tools::limit_rad(residual[4]);
  }
  return residual;
}
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, const std::vector<double> & armor_z_offsets,
  bool fixed_center_rotation_model, double spin_speed_lock,
  const TargetEstimatorParams & estimator_params)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  fixed_center_rotation_model_(fixed_center_rotation_model),
  spin_speed_lock_(spin_speed_lock),
  estimator_params_(estimator_params),
  switch_count_(0)
{
  priority = armor.priority;
  armor_z_offsets_.assign(armor_num_, 0.0);
  for (int i = 0; i < std::min<int>(armor_num_, armor_z_offsets.size()); ++i) {
    armor_z_offsets_[i] = armor_z_offsets[i];
  }

  const Eigen::Matrix3d armor_rotation = armor.R_armor2world;
  const Eigen::Matrix3d nominal_armor_pitch =
    Eigen::AngleAxisd(armor_pitch(), Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Matrix3d car_rotation = armor_rotation * nominal_armor_pitch.transpose();
  if (!uses_full_rotation()) {
    const double yaw = armor.ypr_in_world[0];
    car_rotation = tools::so3_exp(Eigen::Vector3d(0.0, 0.0, yaw));
  }
  const Eigen::Vector3d center =
    armor.xyz_in_world + car_rotation * Eigen::Vector3d(radius, 0.0, 0.0);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(target_state::SIZE);
  x0[target_state::CX] = center.x();
  x0[target_state::CY] = center.y();
  x0[target_state::CZ] = center.z();
  x0[target_state::LOG_R1] = std::log(radius);
  x0[target_state::LOG_R2] = std::log(radius);
  write_rotation(x0, car_rotation);
  if (!uses_full_rotation()) {
    x0[target_state::ROT_X] = 0.0;
    x0[target_state::ROT_Y] = 0.0;
  }
  Eigen::VectorXd initial_variance = P0_dig;
  if (initial_variance.size() != target_state::SIZE || !initial_variance.allFinite()) {
    tools::logger()->warn(
      "[Target] invalid initial covariance diagonal (size {}), using identity",
      initial_variance.size());
    initial_variance = Eigen::VectorXd::Ones(target_state::SIZE);
  }
  Eigen::MatrixXd P0 = initial_variance.asDiagonal();

  const bool full_rotation = uses_full_rotation();
  auto x_add = [full_rotation](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    return inject_state(a, b, full_rotation);
  };
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4), fixed_center_rotation_model_(false), spin_speed_lock_(2.51)
{
  name = ArmorName::three;
  armor_type = ArmorType::small;
  priority = ArmorPriority::fifth;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(target_state::SIZE);
  x0[target_state::CX] = x;
  x0[target_state::VYAW] = vyaw;
  x0[target_state::LOG_R1] = std::log(radius);
  x0[target_state::LOG_R2] = std::log(radius);
  x0[target_state::H] = h;
  Eigen::VectorXd P0_dig = Eigen::VectorXd::Zero(target_state::SIZE);
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();
  armor_z_offsets_.assign(armor_num_, 0.0);

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    return inject_state(a, b, true);
  };
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (!std::isfinite(dt)) return;
  dt = std::clamp(dt, -0.1, 0.1);
  if (convergened() && name == ArmorName::outpost && std::abs(ekf_.x[target_state::VYAW]) > 2.0) {
    ekf_.x[target_state::VYAW] =
      ekf_.x[target_state::VYAW] > 0.0 ? spin_speed_lock_ : -spin_speed_lock_;
  }

  const Eigen::MatrixXd F = predict_jacobian(ekf_.x, dt);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(target_state::SIZE, target_state::SIZE);
  const double dt_abs = std::max(std::abs(dt), 1e-6);
  const double dt2 = dt_abs * dt_abs;
  const double dt3 = dt2 * dt_abs;
  const double dt4 = dt2 * dt2;
  const double acceleration_variance = name == ArmorName::outpost ?
    std::min(estimator_params_.acceleration_variance, 10.0) :
    estimator_params_.acceleration_variance;
  const double yaw_acceleration_variance = name == ArmorName::outpost ?
    std::min(estimator_params_.yaw_acceleration_variance, 0.1) :
    estimator_params_.yaw_acceleration_variance;
  constexpr std::array<int, 3> positions{
    target_state::CX, target_state::CY, target_state::CZ};
  constexpr std::array<int, 3> velocities{
    target_state::VCX, target_state::VCY, target_state::VCZ};
  for (int axis = 0; axis < 3; ++axis) {
    const int p = positions[axis];
    const int v = velocities[axis];
    Q(p, p) = 0.25 * dt4 * acceleration_variance;
    Q(p, v) = Q(v, p) = 0.5 * dt3 * acceleration_variance;
    Q(v, v) = dt2 * acceleration_variance;
  }
  Q(target_state::ROT_Z, target_state::ROT_Z) +=
    0.25 * dt4 * yaw_acceleration_variance;
  Q(target_state::ROT_Z, target_state::VYAW) +=
    0.5 * dt3 * yaw_acceleration_variance;
  Q(target_state::VYAW, target_state::ROT_Z) +=
    0.5 * dt3 * yaw_acceleration_variance;
  Q(target_state::VYAW, target_state::VYAW) += dt2 * yaw_acceleration_variance;
  if (uses_full_rotation()) {
    Q(target_state::ROT_X, target_state::ROT_X) +=
      estimator_params_.roll_pitch_random_walk * dt_abs;
    Q(target_state::ROT_Y, target_state::ROT_Y) +=
      estimator_params_.roll_pitch_random_walk * dt_abs;
  }
  const double r1 = radius(0);
  const double r2 = armor_num_ == 4 ? radius(1) : r1;
  Q(target_state::LOG_R1, target_state::LOG_R1) =
    estimator_params_.geometry_random_walk * dt_abs / (r1 * r1);
  Q(target_state::LOG_R2, target_state::LOG_R2) =
    estimator_params_.geometry_random_walk * dt_abs / (r2 * r2);
  Q(target_state::H, target_state::H) = estimator_params_.geometry_random_walk * dt_abs;

  const auto f = [&](const Eigen::VectorXd & x) { return predict_state(x, dt); };
  ekf_.predict(F, Q, f);
}

bool Target::update(const Armor & armor, const Solver & solver)
{
  int best_id = 0;
  double best_error = std::numeric_limits<double>::infinity();
  for (int id = 0; id < armor_num_; ++id) {
    const Eigen::VectorXd predicted = predict_uvl(ekf_.x, id, solver);
    const Eigen::VectorXd measured = uvl_observation(armor.points);
    if (predicted.size() != measured.size() || predicted.size() != 8) continue;
    const double error = subtract_uvl(measured, predicted).squaredNorm();
    if (error < best_error) {
      best_error = error;
      best_id = id;
    }
  }
  return update(armor, best_id, solver);
}

bool Target::update(const Armor & armor, int id, const Solver & solver)
{
  if (!update_uvl(armor, id, solver)) return false;

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;
  return true;
}

bool Target::update_uvl(const Armor & armor, int id, const Solver & solver)
{
  const Eigen::VectorXd z = uvl_observation(armor.points);
  if (z.size() != 8) return false;
  const Eigen::VectorXd predicted = predict_uvl(ekf_.x, id, solver);
  if (predicted.size() != 8 || !predicted.allFinite()) return false;

  const Eigen::MatrixXd H = uvl_jacobian(ekf_.x, id, solver);
  if (H.rows() != 8 || H.cols() != target_state::SIZE || !H.allFinite()) return false;
  Eigen::VectorXd variances(8);
  variances << estimator_params_.uvl_angle_variance,
    estimator_params_.uvl_center_variance, estimator_params_.uvl_center_variance,
    estimator_params_.uvl_length_variance, estimator_params_.uvl_angle_variance,
    estimator_params_.uvl_center_variance, estimator_params_.uvl_center_variance,
    estimator_params_.uvl_length_variance;
  const Eigen::MatrixXd R = variances.asDiagonal();
  const auto h = [&](const Eigen::VectorXd & x) { return predict_uvl(x, id, solver); };
  const bool accepted = ekf_.update_gated(
    z, H, R, h, estimator_params_.nis_gate,
    [](const Eigen::VectorXd & lhs, const Eigen::VectorXd & rhs) {
      return subtract_uvl(lhs, rhs);
    });
  if (!accepted) return false;

  clamp_state(ekf_.x);
  return true;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

Eigen::Vector3d Target::center_xyz_in_world() const
{
  if (ekf_.x.size() < 5) {
    return Eigen::Vector3d::Zero();
  }
  return {ekf_.x[0], ekf_.x[2], ekf_.x[4]};
}

int Target::normalize_armor_id(int id) const
{
  if (armor_num_ <= 0) return 0;
  return (id % armor_num_ + armor_num_) % armor_num_;
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> result;
  for (const auto & pose : armor_pose_list()) {
    const double yaw = tools::eulers(pose.rotation, 2, 1, 0)[0];
    result.push_back({pose.xyz.x(), pose.xyz.y(), pose.xyz.z(), yaw});
  }
  return result;
}

std::vector<ArmorPose> Target::armor_pose_list() const
{
  std::vector<ArmorPose> result;
  result.reserve(armor_num_);
  for (int id = 0; id < armor_num_; ++id) result.push_back(armor_pose(ekf_.x, id));
  return result;
}

Eigen::Vector3d Target::car_rpy() const
{
  return tools::eulers(state_rotation(ekf_.x), 2, 1, 0);
}

double Target::radius(int id) const
{
  const bool second_radius = armor_num_ == 4 && (normalize_armor_id(id) & 1);
  const int index = second_radius ? target_state::LOG_R2 : target_state::LOG_R1;
  return std::exp(std::clamp(ekf_.x[index], std::log(kMinRadius), std::log(kMaxRadius)));
}

double Target::armor_z_offset(int id) const
{
  if (armor_z_offsets_.empty() || armor_num_ <= 0) return 0.0;
  return armor_z_offsets_[physical_armor_id(id)];
}

int Target::physical_armor_id(int id) const
{
  return normalize_armor_id(normalize_armor_id(id) + armor_id_offset_);
}

int Target::armor_id_offset() const { return armor_id_offset_; }

void Target::set_armor_id_offset(int offset, int reference_id)
{
  if (armor_num_ <= 0) return;

  const int normalized_offset = normalize_armor_id(offset);
  if (normalized_offset == armor_id_offset_) return;

  const int reference_local_id = normalize_armor_id(reference_id);
  const double old_reference_z_offset = armor_z_offset(reference_local_id);
  armor_id_offset_ = normalized_offset;
  const double new_reference_z_offset = armor_z_offset(reference_local_id);

  // Keep the currently tracked reference board height continuous while remapping
  // local ids to physical outpost boards.
  if (ekf_.x.size() > 4) {
    ekf_.x[4] += old_reference_z_offset - new_reference_z_offset;
  }
}

bool Target::fixed_center_rotation_model() const { return fixed_center_rotation_model_; }

double Target::nis_gate() const { return estimator_params_.nis_gate; }

bool Target::diverged() const
{
  if (ekf_.x.size() != target_state::SIZE || !ekf_.x.allFinite() || !ekf_.P.allFinite()) {
    return true;
  }
  const double r1 = radius(0);
  const double r2 = armor_num_ == 4 ? radius(1) : r1;
  const bool geometry_valid = r1 >= kMinRadius && r1 <= kMaxRadius &&
                              r2 >= kMinRadius && r2 <= kMaxRadius &&
                              std::abs(ekf_.x[target_state::H]) <= 0.5;
  const bool motion_valid = std::abs(ekf_.x[target_state::VYAW]) <= 20.0;
  if (geometry_valid && motion_valid) return false;
  tools::logger()->info(
    "[Target] invalid 13D state r1={:.3f}, r2={:.3f}, h={:.3f}, vyaw={:.3f}",
    r1, r2, ekf_.x[target_state::H], ekf_.x[target_state::VYAW]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

bool Target::uses_full_rotation() const
{
  return name != ArmorName::outpost && name != ArmorName::base &&
         !fixed_center_rotation_model_;
}

double Target::armor_pitch() const
{
  return (name == ArmorName::outpost ? -15.0 : 15.0) * CV_PI / 180.0;
}

Eigen::Matrix3d Target::state_rotation(const Eigen::VectorXd & x) const
{
  return rotation_from_state(x, uses_full_rotation());
}

ArmorPose Target::armor_pose(const Eigen::VectorXd & x, int id) const
{
  id = normalize_armor_id(id);
  const double theta = id * 2.0 * CV_PI / static_cast<double>(armor_num_);
  const bool second_height = armor_num_ == 4 && (id & 1);
  const double r = std::exp(std::clamp(
    x[second_height ? target_state::LOG_R2 : target_state::LOG_R1],
    std::log(kMinRadius), std::log(kMaxRadius)));
  double z_offset = second_height ? x[target_state::H] : 0.0;
  if (name == ArmorName::outpost) z_offset = armor_z_offset(id);

  const Eigen::Matrix3d car_rotation = state_rotation(x);
  const Eigen::Vector3d local_position(-r * std::cos(theta), -r * std::sin(theta), z_offset);
  const Eigen::Matrix3d local_rotation =
    Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    Eigen::AngleAxisd(armor_pitch(), Eigen::Vector3d::UnitY()).toRotationMatrix();
  ArmorPose result;
  result.xyz = Eigen::Vector3d(
    x[target_state::CX], x[target_state::CY], x[target_state::CZ]) +
    car_rotation * local_position;
  result.rotation = car_rotation * local_rotation;
  return result;
}

Eigen::VectorXd Target::predict_uvl(
  const Eigen::VectorXd & x, int id, const Solver & solver) const
{
  const ArmorPose pose = armor_pose(x, id);
  return uvl_observation(solver.reproject_armor(pose.xyz, pose.rotation, armor_type));
}

Eigen::MatrixXd Target::uvl_jacobian(
  const Eigen::VectorXd & x, int id, const Solver & solver) const
{
  Eigen::MatrixXd H(8, target_state::SIZE);
  for (int column = 0; column < target_state::SIZE; ++column) {
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(target_state::SIZE);
    delta[column] = kNumericEpsilon;
    const Eigen::VectorXd plus = predict_uvl(inject_error(x, delta), id, solver);
    delta[column] = -kNumericEpsilon;
    const Eigen::VectorXd minus = predict_uvl(inject_error(x, delta), id, solver);
    if (
      plus.size() != 8 || minus.size() != 8 || !plus.allFinite() || !minus.allFinite()) {
      return Eigen::MatrixXd::Constant(
        8, target_state::SIZE, std::numeric_limits<double>::quiet_NaN());
    }
    H.col(column) = subtract_uvl(plus, minus) / (2.0 * kNumericEpsilon);
  }
  return H;
}

Eigen::VectorXd Target::predict_state(const Eigen::VectorXd & x, double dt) const
{
  Eigen::VectorXd predicted = x;
  if (fixed_center_rotation_model_) {
    predicted[target_state::VCX] = 0.0;
    predicted[target_state::VCY] = 0.0;
    predicted[target_state::VCZ] = 0.0;
  } else {
    predicted[target_state::CX] += x[target_state::VCX] * dt;
    predicted[target_state::CY] += x[target_state::VCY] * dt;
    predicted[target_state::CZ] += x[target_state::VCZ] * dt;
  }
  const Eigen::Matrix3d predicted_rotation =
    state_rotation(x) *
    tools::so3_exp(Eigen::Vector3d(0.0, 0.0, x[target_state::VYAW] * dt));
  write_rotation(predicted, predicted_rotation);
  clamp_state(predicted);
  return predicted;
}

Eigen::MatrixXd Target::predict_jacobian(const Eigen::VectorXd & x, double dt) const
{
  const Eigen::VectorXd nominal_prediction = predict_state(x, dt);
  Eigen::MatrixXd F(target_state::SIZE, target_state::SIZE);
  for (int column = 0; column < target_state::SIZE; ++column) {
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(target_state::SIZE);
    delta[column] = kNumericEpsilon;
    const Eigen::VectorXd perturbed_prediction = predict_state(inject_error(x, delta), dt);
    F.col(column) = box_minus(nominal_prediction, perturbed_prediction) / kNumericEpsilon;
  }
  return F;
}

Eigen::VectorXd Target::inject_error(
  const Eigen::VectorXd & nominal, const Eigen::VectorXd & delta) const
{
  return inject_state(nominal, delta, uses_full_rotation());
}

Eigen::VectorXd Target::box_minus(
  const Eigen::VectorXd & nominal, const Eigen::VectorXd & value) const
{
  return subtract_state(nominal, value, uses_full_rotation());
}

void Target::clamp_state(Eigen::VectorXd & x) const
{
  x[target_state::LOG_R1] = std::clamp(
    x[target_state::LOG_R1], std::log(kMinRadius), std::log(kMaxRadius));
  x[target_state::LOG_R2] = std::clamp(
    x[target_state::LOG_R2], std::log(kMinRadius), std::log(kMaxRadius));
  if (std::abs(x[target_state::H]) > 0.5) x[target_state::H] = 0.0;
  if (std::abs(x[target_state::VYAW]) > 20.0) x[target_state::VYAW] = 0.0;
  if (!uses_full_rotation()) {
    x[target_state::ROT_X] = 0.0;
    x[target_state::ROT_Y] = 0.0;
  }
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
