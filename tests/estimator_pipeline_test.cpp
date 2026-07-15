#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "tools/extended_kalman_filter.hpp"
#include "tools/thread_safe_queue.hpp"

namespace
{
using namespace std::chrono_literals;

void expect(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_ekf_measurement_gating()
{
  const Eigen::Vector2d x0 = Eigen::Vector2d::Zero();
  Eigen::Matrix2d P0;
  P0 << 1.0, 0.2, 0.2, 2.0;

  tools::ExtendedKalmanFilter ekf(x0, P0);
  const Eigen::Matrix2d H = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
  R.diagonal() << 0.1, 0.2;
  const auto h = [&H](const Eigen::VectorXd & x) { return H * x; };
  constexpr double nis_gate = 5.991;  // 95% chi-square gate for two dimensions.

  Eigen::Vector2d normal_measurement;
  normal_measurement << 0.25, -0.1;
  const bool accepted = ekf.update_gated(normal_measurement, H, R, h, nis_gate);

  expect(accepted, "EKF rejected a normal measurement");
  expect(ekf.x.norm() > 0.0, "accepted EKF update did not change the state");
  expect(std::isfinite(ekf.last_nis) && ekf.last_nis < nis_gate, "accepted update has invalid NIS");
  expect(ekf.data.at("update_accepted") == 1.0, "accepted update telemetry is incorrect");
  expect(ekf.data.at("accepted_updates") == 1.0, "accepted update counter is incorrect");
  expect(ekf.P.allFinite(), "posterior covariance contains non-finite values");
  expect(
    (ekf.P - ekf.P.transpose()).cwiseAbs().maxCoeff() < 1e-12,
    "posterior covariance is not symmetric");

  const Eigen::VectorXd state_before_outlier = ekf.x;
  const Eigen::MatrixXd covariance_before_outlier = ekf.P;
  Eigen::Vector2d outlier_measurement;
  outlier_measurement << 100.0, -100.0;
  const bool outlier_accepted =
    ekf.update_gated(outlier_measurement, H, R, h, nis_gate);

  expect(!outlier_accepted, "EKF accepted an outlier measurement");
  expect(ekf.last_nis > nis_gate, "rejected update did not exceed the NIS gate");
  expect(ekf.data.at("update_accepted") == 0.0, "rejected update telemetry is incorrect");
  expect(ekf.data.at("rejected_updates") == 1.0, "rejected update counter is incorrect");
  expect(
    (ekf.x - state_before_outlier).cwiseAbs().maxCoeff() == 0.0,
    "rejected update changed the EKF state");
  expect(
    (ekf.P - covariance_before_outlier).cwiseAbs().maxCoeff() == 0.0,
    "rejected update changed the EKF covariance");
}

void test_bounded_queue_rejects_when_full()
{
  int full_handler_calls = 0;
  tools::ThreadSafeQueue<int> queue(1, [&full_handler_calls] { ++full_handler_calls; });

  expect(queue.try_push(7), "first bounded queue push failed");
  expect(!queue.try_push(8), "bounded queue accepted an item while full");
  expect(full_handler_calls == 1, "full queue handler was not called exactly once");
  expect(queue.size() == 1, "rejected queue push changed the queue size");

  const auto retained = queue.pop_for(0ms);
  expect(retained.has_value() && *retained == 7, "full queue did not retain the original item");
}

void test_pop_when_full_retains_latest()
{
  tools::ThreadSafeQueue<int, true> queue(1);

  expect(queue.try_push(1), "first overwrite queue push failed");
  expect(queue.try_push(2), "overwrite queue rejected the latest item");
  expect(queue.size() == 1, "overwrite queue exceeded its capacity");

  const auto latest = queue.pop_for(0ms);
  expect(latest.has_value() && *latest == 2, "overwrite queue did not retain the latest item");
}

void test_timed_pop()
{
  tools::ThreadSafeQueue<int> queue(2);

  expect(!queue.pop_for(2ms).has_value(), "timed pop returned a value from an empty queue");
  expect(queue.try_push(42), "queue push before timed pop failed");

  const auto value = queue.pop_for(20ms);
  expect(value.has_value() && *value == 42, "timed pop did not return the queued value");
  expect(queue.empty(), "timed pop did not remove the queued value");
}
}  // namespace

int main()
{
  try {
    test_ekf_measurement_gating();
    test_bounded_queue_rejects_when_full();
    test_pop_when_full_retains_latest();
    test_timed_pop();
  } catch (const std::exception & error) {
    std::cerr << "estimator_pipeline_test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "estimator_pipeline_test passed\n";
  return 0;
}
