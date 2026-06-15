#include <gtest/gtest.h>

#include <cmath>

#include "tools/math_tools.hpp"

// --- limit_rad ---

TEST(MathTools, LimitRadInRange)
{
  EXPECT_DOUBLE_EQ(tools::limit_rad(0.0), 0.0);
  EXPECT_DOUBLE_EQ(tools::limit_rad(1.0), 1.0);
  EXPECT_DOUBLE_EQ(tools::limit_rad(-1.0), -1.0);
}

TEST(MathTools, LimitRadWrapsPositive)
{
  double result = tools::limit_rad(M_PI + 0.5);
  EXPECT_NEAR(result, -M_PI + 0.5, 1e-12);
}

TEST(MathTools, LimitRadWrapsNegative)
{
  double result = tools::limit_rad(-M_PI - 0.5);
  EXPECT_NEAR(result, M_PI - 0.5, 1e-12);
}

TEST(MathTools, LimitRadExactPi)
{
  EXPECT_DOUBLE_EQ(tools::limit_rad(M_PI), M_PI);
}

TEST(MathTools, LimitRadExactNegPi)
{
  double result = tools::limit_rad(-M_PI);
  EXPECT_NEAR(result, M_PI, 1e-12);
}

TEST(MathTools, LimitRadMultipleWraps)
{
  double result = tools::limit_rad(5 * M_PI);
  EXPECT_NEAR(result, M_PI, 1e-12);
}

// --- square ---

TEST(MathTools, SquareInt)
{
  EXPECT_EQ(tools::square(3), 9);
  EXPECT_EQ(tools::square(-4), 16);
  EXPECT_EQ(tools::square(0), 0);
}

TEST(MathTools, SquareDouble)
{
  EXPECT_DOUBLE_EQ(tools::square(2.5), 6.25);
}

// --- limit_min_max ---

TEST(MathTools, LimitMinMaxInRange)
{
  EXPECT_DOUBLE_EQ(tools::limit_min_max(5.0, 0.0, 10.0), 5.0);
}

TEST(MathTools, LimitMinMaxClampHigh)
{
  EXPECT_DOUBLE_EQ(tools::limit_min_max(15.0, 0.0, 10.0), 10.0);
}

TEST(MathTools, LimitMinMaxClampLow)
{
  EXPECT_DOUBLE_EQ(tools::limit_min_max(-5.0, 0.0, 10.0), 0.0);
}

// --- xyz2ypd / ypd2xyz round-trip ---

TEST(MathTools, Xyz2YpdBasic)
{
  Eigen::Vector3d xyz(1.0, 0.0, 0.0);
  auto ypd = tools::xyz2ypd(xyz);
  EXPECT_NEAR(ypd[0], 0.0, 1e-12);            // yaw
  EXPECT_NEAR(ypd[1], 0.0, 1e-12);            // pitch
  EXPECT_NEAR(ypd[2], 1.0, 1e-12);            // distance
}

TEST(MathTools, Xyz2YpdPositiveYaw)
{
  Eigen::Vector3d xyz(0.0, 1.0, 0.0);
  auto ypd = tools::xyz2ypd(xyz);
  EXPECT_NEAR(ypd[0], M_PI / 2.0, 1e-12);     // yaw = 90°
  EXPECT_NEAR(ypd[1], 0.0, 1e-12);
  EXPECT_NEAR(ypd[2], 1.0, 1e-12);
}

TEST(MathTools, Xyz2YpdPositivePitch)
{
  Eigen::Vector3d xyz(1.0, 0.0, 1.0);
  auto ypd = tools::xyz2ypd(xyz);
  EXPECT_NEAR(ypd[0], 0.0, 1e-12);
  EXPECT_NEAR(ypd[1], M_PI / 4.0, 1e-12);     // pitch = 45°
  EXPECT_NEAR(ypd[2], std::sqrt(2.0), 1e-12);
}

TEST(MathTools, RoundTripXyzYpd)
{
  Eigen::Vector3d xyz(2.0, 3.0, 1.5);
  auto ypd = tools::xyz2ypd(xyz);
  auto xyz_back = tools::ypd2xyz(ypd);
  EXPECT_NEAR(xyz_back[0], xyz[0], 1e-10);
  EXPECT_NEAR(xyz_back[1], xyz[1], 1e-10);
  EXPECT_NEAR(xyz_back[2], xyz[2], 1e-10);
}

TEST(MathTools, RoundTripYpdXyz)
{
  Eigen::Vector3d ypd(0.3, 0.2, 5.0);
  auto xyz = tools::ypd2xyz(ypd);
  auto ypd_back = tools::xyz2ypd(xyz);
  EXPECT_NEAR(ypd_back[0], ypd[0], 1e-10);
  EXPECT_NEAR(ypd_back[1], ypd[1], 1e-10);
  EXPECT_NEAR(ypd_back[2], ypd[2], 1e-10);
}

// --- Jacobian numerical check ---

TEST(MathTools, Xyz2YpdJacobianNumerical)
{
  Eigen::Vector3d xyz(2.0, 1.0, 0.5);
  auto J = tools::xyz2ypd_jacobian(xyz);

  double eps = 1e-6;
  for (int col = 0; col < 3; col++) {
    Eigen::Vector3d xyz_plus = xyz, xyz_minus = xyz;
    xyz_plus[col] += eps;
    xyz_minus[col] -= eps;
    Eigen::Vector3d numerical_col = (tools::xyz2ypd(xyz_plus) - tools::xyz2ypd(xyz_minus)) / (2 * eps);
    for (int row = 0; row < 3; row++) {
      EXPECT_NEAR(J(row, col), numerical_col[row], 1e-5)
        << "Mismatch at J(" << row << "," << col << ")";
    }
  }
}

TEST(MathTools, Ypd2XyzJacobianNumerical)
{
  Eigen::Vector3d ypd(0.3, 0.2, 5.0);
  auto J = tools::ypd2xyz_jacobian(ypd);

  double eps = 1e-6;
  for (int col = 0; col < 3; col++) {
    Eigen::Vector3d ypd_plus = ypd, ypd_minus = ypd;
    ypd_plus[col] += eps;
    ypd_minus[col] -= eps;
    Eigen::Vector3d numerical_col = (tools::ypd2xyz(ypd_plus) - tools::ypd2xyz(ypd_minus)) / (2 * eps);
    for (int row = 0; row < 3; row++) {
      EXPECT_NEAR(J(row, col), numerical_col[row], 1e-5)
        << "Mismatch at J(" << row << "," << col << ")";
    }
  }
}

// --- rotation_matrix / eulers round-trip ---

TEST(MathTools, RotationMatrixIdentity)
{
  Eigen::Vector3d ypr(0.0, 0.0, 0.0);
  auto R = tools::rotation_matrix(ypr);
  EXPECT_TRUE(R.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(MathTools, RotationMatrixYawOnly)
{
  Eigen::Vector3d ypr(M_PI / 4.0, 0.0, 0.0);
  auto R = tools::rotation_matrix(ypr);
  Eigen::Vector3d v(1.0, 0.0, 0.0);
  auto v_rot = R * v;
  EXPECT_NEAR(v_rot[0], std::cos(M_PI / 4.0), 1e-12);
  EXPECT_NEAR(v_rot[1], std::sin(M_PI / 4.0), 1e-12);
  EXPECT_NEAR(v_rot[2], 0.0, 1e-12);
}

TEST(MathTools, EulersFromQuaternionIdentity)
{
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  auto e = tools::eulers(q, 2, 1, 0);
  EXPECT_NEAR(e[0], 0.0, 1e-12);
  EXPECT_NEAR(e[1], 0.0, 1e-12);
  EXPECT_NEAR(e[2], 0.0, 1e-12);
}

TEST(MathTools, EulersFromRotationMatrix)
{
  Eigen::Vector3d ypr(0.3, 0.2, 0.1);
  auto R = tools::rotation_matrix(ypr);
  auto e = tools::eulers(R, 2, 1, 0);
  EXPECT_NEAR(e[0], ypr[0], 1e-10);
  EXPECT_NEAR(e[1], ypr[1], 1e-10);
  EXPECT_NEAR(e[2], ypr[2], 1e-10);
}

// --- delta_time ---

TEST(MathTools, DeltaTime)
{
  auto t1 = std::chrono::steady_clock::now();
  auto t2 = t1 + std::chrono::milliseconds(500);
  double dt = tools::delta_time(t2, t1);
  EXPECT_NEAR(dt, 0.5, 1e-6);
}

TEST(MathTools, DeltaTimeNegative)
{
  auto t1 = std::chrono::steady_clock::now();
  auto t2 = t1 + std::chrono::milliseconds(500);
  double dt = tools::delta_time(t1, t2);
  EXPECT_NEAR(dt, -0.5, 1e-6);
}

// --- get_abs_angle ---

TEST(MathTools, GetAbsAngleParallel)
{
  Eigen::Vector2d v1(1.0, 0.0);
  Eigen::Vector2d v2(2.0, 0.0);
  EXPECT_NEAR(tools::get_abs_angle(v1, v2), 0.0, 1e-12);
}

TEST(MathTools, GetAbsAnglePerpendicular)
{
  Eigen::Vector2d v1(1.0, 0.0);
  Eigen::Vector2d v2(0.0, 1.0);
  EXPECT_NEAR(tools::get_abs_angle(v1, v2), M_PI / 2.0, 1e-12);
}

TEST(MathTools, GetAbsAngleOpposite)
{
  Eigen::Vector2d v1(1.0, 0.0);
  Eigen::Vector2d v2(-1.0, 0.0);
  EXPECT_NEAR(tools::get_abs_angle(v1, v2), M_PI, 1e-12);
}

TEST(MathTools, GetAbsAngleZeroVector)
{
  Eigen::Vector2d v1(0.0, 0.0);
  Eigen::Vector2d v2(1.0, 0.0);
  EXPECT_DOUBLE_EQ(tools::get_abs_angle(v1, v2), 0.0);
}
