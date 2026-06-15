#include <gtest/gtest.h>

#include "tools/extended_kalman_filter.hpp"

// The EKF update() logs residual[0..3], so measurement must be >= 4D.
// Use a 4D state (yaw, pitch, distance, angle) + 4D measurement to match
// the hardcoded logging in the production code.

class EKFTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, 5.0, 0.0;

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);

    ekf = tools::ExtendedKalmanFilter(x0, P0);
  }

  tools::ExtendedKalmanFilter ekf;
};

TEST_F(EKFTest, PredictLinear)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
  F(0, 3) = 0.1;  // yaw += angle * dt

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4) * 0.01;

  auto x_pred = ekf.predict(F, Q);

  EXPECT_NEAR(x_pred[0], 0.0, 1e-10);
  EXPECT_NEAR(x_pred[2], 5.0, 1e-10);
}

TEST_F(EKFTest, PredictWithNonlinearModel)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4) * 0.01;

  auto f = [](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_new = x;
    x_new[0] = x[0] + x[3] * 0.1;
    return x_new;
  };

  auto x_pred = ekf.predict(F, Q, f);
  EXPECT_NEAR(x_pred[0], 0.0, 1e-10);  // angle = 0 so no change
  EXPECT_NEAR(x_pred[2], 5.0, 1e-10);
}

TEST_F(EKFTest, UpdateConvergesToMeasurement)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4) * 0.01;
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4) * 0.01;

  Eigen::VectorXd z_target(4);
  z_target << 1.0, 0.5, 3.0, 0.2;

  for (int i = 0; i < 50; i++) {
    ekf.predict(F, Q);
    ekf.update(z_target, H, R);
  }

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(ekf.x[i], z_target[i], 0.5)
      << "State[" << i << "] did not converge";
  }
}

TEST_F(EKFTest, CovarianceReducesOnUpdate)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4) * 0.1;

  double P_trace_before = ekf.P.trace();

  Eigen::VectorXd z(4);
  z << 0.0, 0.0, 5.0, 0.0;
  ekf.update(z, H, R);

  double P_trace_after = ekf.P.trace();
  EXPECT_LT(P_trace_after, P_trace_before);
}

TEST_F(EKFTest, PredictCovarianceGrows)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4) * 0.5;

  double P_trace_before = ekf.P.trace();
  ekf.predict(F, Q);
  double P_trace_after = ekf.P.trace();

  EXPECT_GT(P_trace_after, P_trace_before);
}

TEST_F(EKFTest, UpdateWithCustomH)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4) * 0.1;

  auto h = [](const Eigen::VectorXd & x) -> Eigen::VectorXd { return x; };

  Eigen::VectorXd z(4);
  z << 2.0, 1.0, 3.0, 0.5;

  auto x_upd = ekf.update(z, H, R, h);
  // State should move toward measurement
  EXPECT_GT(x_upd[0], 0.0);
}

TEST_F(EKFTest, NISDataPopulated)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4) * 0.1;

  Eigen::VectorXd z(4);
  z << 0.5, 0.1, 4.5, 0.1;
  ekf.update(z, H, R);

  EXPECT_TRUE(ekf.data.count("nis") > 0);
  EXPECT_TRUE(ekf.data.count("nees") > 0);
  EXPECT_GE(ekf.last_nis, 0.0);
}
