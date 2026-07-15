#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr double kRadius = 0.2;
constexpr double kArmorPitch = 15.0 * CV_PI / 180.0;

void expect(bool condition, const std::string & message)
{
  if (!condition) throw std::runtime_error(message);
}

bool points_are_finite(const std::vector<cv::Point2f> & points)
{
  if (points.size() != 4) return false;
  for (const auto & point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
  }
  return true;
}

double mean_pixel_distance(
  const std::vector<cv::Point2f> & lhs, const std::vector<cv::Point2f> & rhs)
{
  expect(lhs.size() == rhs.size() && !lhs.empty(), "invalid projected point list");
  double total = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) total += cv::norm(lhs[i] - rhs[i]);
  return total / static_cast<double>(lhs.size());
}

auto_aim::Armor make_synthetic_armor(
  const auto_aim::Solver & solver, const Eigen::Vector3d & center,
  const Eigen::Matrix3d & car_rotation)
{
  const Eigen::Matrix3d local_rotation =
    Eigen::AngleAxisd(kArmorPitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d armor_position =
    center + car_rotation * Eigen::Vector3d(-kRadius, 0.0, 0.0);
  const Eigen::Matrix3d armor_rotation = car_rotation * local_rotation;
  const auto image_points =
    solver.reproject_armor(armor_position, armor_rotation, auto_aim::ArmorType::small);
  expect(points_are_finite(image_points), "synthetic armor projection is invalid");

  auto_aim::Armor armor(9, 1.0F, cv::Rect(), image_points);
  armor.name = auto_aim::ArmorName::three;
  armor.type = auto_aim::ArmorType::small;
  armor.priority = auto_aim::ArmorPriority::fifth;
  armor.xyz_in_world = armor_position;
  armor.R_armor2world = armor_rotation;
  armor.ypr_in_world = tools::eulers(armor_rotation, 2, 1, 0);
  return armor;
}

void test_so3_exp_log_and_right_perturbation()
{
  const Eigen::Vector3d rotation_vector(0.42, -0.31, 0.27);
  const Eigen::Matrix3d rotation = tools::so3_exp(rotation_vector);
  const Eigen::Matrix3d recovered = tools::so3_exp(tools::so3_log(rotation));
  expect(
    (rotation - recovered).cwiseAbs().maxCoeff() < 1e-12,
    "SO(3) Exp/Log round trip changed the rotation");

  const Eigen::Vector3d right_delta(-0.02, 0.03, 0.04);
  const Eigen::Matrix3d perturbed = rotation * tools::so3_exp(right_delta);
  const Eigen::Vector3d recovered_delta = tools::so3_log(rotation.transpose() * perturbed);
  expect(
    (right_delta - recovered_delta).norm() < 1e-12,
    "SO(3) right perturbation cannot be recovered by box-minus");
}

void test_13d_target_uses_full_rpy_and_uvl(const std::string & config_path)
{
  auto_aim::Solver solver(config_path);
  solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

  const Eigen::Vector3d center(4.2, 0.35, 0.55);
  const Eigen::Vector3d nominal_ypr(0.45, -0.18, 0.12);
  const Eigen::Matrix3d car_rotation = tools::rotation_matrix(nominal_ypr);
  const auto_aim::Armor armor = make_synthetic_armor(solver, center, car_rotation);
  const Eigen::VectorXd initial_variance = Eigen::VectorXd::Constant(
    auto_aim::target_state::SIZE, 1.0);
  auto_aim::Target target(
    armor, std::chrono::steady_clock::now(), kRadius, 4, initial_variance);

  expect(
    target.ekf_x().size() == auto_aim::target_state::SIZE,
    "target state is not the required 13-dimensional model");
  const auto poses = target.armor_pose_list();
  expect(poses.size() == 4, "four-armor target did not produce four rigid-body poses");
  expect(
    (poses.front().xyz - armor.xyz_in_world).norm() < 1e-9,
    "initial armor position is inconsistent with rigid-body geometry");
  expect(
    tools::so3_log(poses.front().rotation.transpose() * armor.R_armor2world).norm() < 1e-9,
    "initial armor RPY is not preserved in the target state");

  const Eigen::Vector3d estimated_ypr = target.car_rpy();
  expect(std::abs(estimated_ypr[1]) > 0.05, "target pitch was discarded");
  expect(std::abs(estimated_ypr[2]) > 0.05, "target roll was discarded");

  const bool accepted = target.update(armor, 0, solver);
  expect(accepted, "exact 8D UVL observation was rejected");
  expect(
    target.ekf().data.at("measurement_dim") == 8.0,
    "target update did not use the 8D UVL measurement model");
  expect(target.ekf().P.allFinite(), "UVL update produced a non-finite covariance");

  const Eigen::Matrix3d changed_rotation =
    tools::rotation_matrix(Eigen::Vector3d(0.45, 0.08, -0.14));
  const auto_aim::Armor changed_armor = make_synthetic_armor(solver, center, changed_rotation);
  auto_aim::Target changed_target(
    changed_armor, std::chrono::steady_clock::now(), kRadius, 4, initial_variance);
  const auto changed_poses = changed_target.armor_pose_list();
  const auto baseline_projection =
    solver.reproject_armor(poses.front().xyz, poses.front().rotation, auto_aim::ArmorType::small);
  const auto changed_projection = solver.reproject_armor(
    changed_poses.front().xyz, changed_poses.front().rotation, auto_aim::ArmorType::small);
  expect(
    mean_pixel_distance(baseline_projection, changed_projection) > 3.0,
    "pitch/roll change did not affect the predicted armor projection");
}
}  // namespace

int main(int argc, char * argv[])
{
  try {
    const std::string config_path = argc > 1 ? argv[1] : "configs/demo.yaml";
    test_so3_exp_log_and_right_perturbation();
    test_13d_target_uses_full_rpy_and_uvl(config_path);
  } catch (const std::exception & error) {
    std::cerr << "target_13d_test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "target_13d_test passed\n";
  return 0;
}
