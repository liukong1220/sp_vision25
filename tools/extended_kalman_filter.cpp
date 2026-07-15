#include "extended_kalman_filter.hpp"

#include <cmath>
#include <limits>
#include <numeric>

namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
  data["update_accepted"] = 1.0;
  data["accepted_updates"] = 0.0;
  data["rejected_updates"] = 0.0;
  data["measurement_dim"] = 0.0;
  data["uvl_left_angle"] = 0.0;
  data["uvl_left_center_u"] = 0.0;
  data["uvl_left_center_v"] = 0.0;
  data["uvl_left_length"] = 0.0;
  data["uvl_right_angle"] = 0.0;
  data["uvl_right_center_u"] = 0.0;
  data["uvl_right_center_v"] = 0.0;
  data["uvl_right_length"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  update_gated(z, H, R, std::move(h), std::numeric_limits<double>::infinity(), z_subtract);
  return x;
}

bool ExtendedKalmanFilter::update_gated(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h, double nis_gate,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  const Eigen::VectorXd innovation = z_subtract(z, h(x));
  const Eigen::MatrixXd S = H * P * H.transpose() + R;
  const Eigen::LDLT<Eigen::MatrixXd> innovation_solver(S);

  double nis = std::numeric_limits<double>::infinity();
  if (innovation_solver.info() == Eigen::Success) {
    nis = innovation.dot(innovation_solver.solve(innovation));
  }
  bool accepted = std::isfinite(nis) && nis >= 0.0 && nis <= nis_gate;

  Eigen::MatrixXd posterior;
  Eigen::VectorXd correction;
  Eigen::VectorXd state_candidate;
  if (accepted) {
    const Eigen::MatrixXd PHt = P * H.transpose();
    const Eigen::MatrixXd K = innovation_solver.solve(PHt.transpose()).transpose();
    correction = K * innovation;
    const Eigen::MatrixXd IKH = I - K * H;
    posterior = IKH * P * IKH.transpose() + K * R * K.transpose();
    posterior = 0.5 * (posterior + posterior.transpose());
    state_candidate = x_add(x, correction);
    accepted =
      K.allFinite() && correction.allFinite() && posterior.allFinite() &&
      state_candidate.allFinite();
  }

  data["nis_fail"] = accepted ? 0.0 : 1.0;
  data["nees_fail"] = 0.0;
  data["update_accepted"] = accepted ? 1.0 : 0.0;
  data["nis"] = nis;
  last_nis = nis;
  total_count_++;
  if (!accepted) {
    nis_count_++;
    data["rejected_updates"] += 1.0;
  } else {
    data["accepted_updates"] += 1.0;
  }

  recent_nis_failures.push_back(accepted ? 0 : 1);
  if (recent_nis_failures.size() > window_size) recent_nis_failures.pop_front();
  const int recent_failures =
    std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  data["recent_nis_failures"] =
    static_cast<double>(recent_failures) / static_cast<double>(recent_nis_failures.size());

  data["measurement_dim"] = static_cast<double>(innovation.size());
  if (innovation.size() == 8) {
    data["uvl_left_angle"] = innovation[0];
    data["uvl_left_center_u"] = innovation[1];
    data["uvl_left_center_v"] = innovation[2];
    data["uvl_left_length"] = innovation[3];
    data["uvl_right_angle"] = innovation[4];
    data["uvl_right_center_u"] = innovation[5];
    data["uvl_right_center_v"] = innovation[6];
    data["uvl_right_length"] = innovation[7];
  } else {
    if (innovation.size() > 0) data["residual_yaw"] = innovation[0];
    if (innovation.size() > 1) data["residual_pitch"] = innovation[1];
    if (innovation.size() > 2) data["residual_distance"] = innovation[2];
    if (innovation.size() > 3) data["residual_angle"] = innovation[3];
  }
  if (!accepted) return false;

  // Joseph form keeps P positive semidefinite under floating-point roundoff.
  P = std::move(posterior);
  x = std::move(state_candidate);

  const Eigen::LDLT<Eigen::MatrixXd> posterior_solver(P);
  data["nees"] = posterior_solver.info() == Eigen::Success ?
    correction.dot(posterior_solver.solve(correction)) :
    std::numeric_limits<double>::infinity();
  return true;
}

}  // namespace tools
