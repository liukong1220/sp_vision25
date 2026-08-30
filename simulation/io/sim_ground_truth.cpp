#include "sim_ground_truth.hpp"

#include <algorithm>
#include <cmath>

namespace sim_io
{
namespace
{
double percentile(std::vector<double> sorted, double q)
{
  if (sorted.empty()) return 0.0;
  std::sort(sorted.begin(), sorted.end());
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double w = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

double mean_of(const std::vector<double> & v)
{
  if (v.empty()) return 0.0;
  double sum = 0.0;
  for (double x : v) sum += x;
  return sum / static_cast<double>(v.size());
}

double normalize_angle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}
}  // namespace

auto_aim::ArmorName armor_label_to_name(std::uint8_t label)
{
  switch (label) {
    case 0:
      return auto_aim::sentry;
    case 1:
      return auto_aim::one;
    case 2:
      return auto_aim::two;
    case 3:
      return auto_aim::three;
    case 4:
      return auto_aim::four;
    case 5:
      return auto_aim::five;
    case 6:
      return auto_aim::outpost;
    case 7:
      return auto_aim::base;
    default:
      return auto_aim::not_armor;
  }
}

std::uint8_t armor_name_to_label(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::one:
      return 1;
    case auto_aim::two:
      return 2;
    case auto_aim::three:
      return 3;
    case auto_aim::four:
      return 4;
    case auto_aim::five:
      return 5;
    case auto_aim::sentry:
      return 0;
    case auto_aim::outpost:
      return 6;
    case auto_aim::base:
      return 7;
    default:
      return 255;
  }
}

bool GroundTruthEvaluator::fetch(std::uint64_t image_frame_seq)
{
  fetched_ = false;
  GroundTruthBatch batch{};
  if (!client_.read_ground_truth(&batch)) return false;

  if (batch.frame_seq != image_frame_seq) {
    ++seq_mismatches_;
    return false;
  }
  if (batch.target_count > GROUND_TRUTH_MAX_TARGETS) return false;

  batch_ = batch;
  fetched_ = true;
  return true;
}

bool GroundTruthEvaluator::fetch_latest_diagnostic_only()
{
  fetched_ = false;
  GroundTruthBatch batch{};
  if (!client_.read_ground_truth(&batch)) return false;
  if (batch.target_count > GROUND_TRUTH_MAX_TARGETS) return false;
  batch_ = batch;
  // 刻意不置 fetched_：这样即使有人误调 evaluate()，也拿不到数据。
  return true;
}

std::optional<GroundTruthTarget> GroundTruthEvaluator::find_by_label(std::uint8_t label) const
{
  if (!fetched_) return std::nullopt;
  for (std::uint32_t i = 0; i < batch_.target_count; ++i) {
    if (batch_.targets[i].armor_label == label) return batch_.targets[i];
  }
  return std::nullopt;
}

std::optional<GroundTruthTarget> GroundTruthEvaluator::find_nearest(
  const Eigen::Vector3d & position, double gate_m) const
{
  if (!fetched_) return std::nullopt;

  std::optional<GroundTruthTarget> best;
  double best_dist = gate_m;
  for (std::uint32_t i = 0; i < batch_.target_count; ++i) {
    const GroundTruthTarget & t = batch_.targets[i];
    const Eigen::Vector3d p(t.position[0], t.position[1], t.position[2]);
    const double d = (p - position).norm();
    if (d < best_dist) {
      best_dist = d;
      best = t;
    }
  }
  return best;
}

GtError GroundTruthEvaluator::evaluate(
  auto_aim::ArmorName name, const Eigen::Vector3d & estimate_in_odom, double yaw, double vyaw,
  double gate_m)
{
  GtError out;
  out.name = name;
  out.est_position = estimate_in_odom;

  std::optional<GroundTruthTarget> gt = find_by_label(armor_name_to_label(name));
  if (!gt.has_value()) gt = find_nearest(estimate_in_odom, gate_m);
  if (!gt.has_value()) return out;

  const Eigen::Vector3d p(gt->position[0], gt->position[1], gt->position[2]);
  const Eigen::Vector3d d = estimate_in_odom - p;

  out.valid = true;
  out.armor_label = gt->armor_label;
  out.gt_position = p;
  out.pos_err_m = d.norm();
  out.xy_err_m = d.head<2>().norm();
  out.z_err_m = std::abs(d.z());
  out.yaw_err_rad = std::abs(normalize_angle(yaw - static_cast<double>(gt->yaw)));
  out.vyaw_err_radps = std::abs(vyaw - static_cast<double>(gt->vyaw));
  return out;
}

void GroundTruthEvaluator::record(const GtError & error)
{
  if (!error.valid) return;
  pos_err_.push_back(error.pos_err_m);
  xy_err_.push_back(error.xy_err_m);
  z_err_.push_back(error.z_err_m);
  yaw_err_.push_back(error.yaw_err_rad);
  vyaw_err_.push_back(error.vyaw_err_radps);
}

GtErrorStats GroundTruthEvaluator::stats() const
{
  GtErrorStats s;
  s.count = pos_err_.size();
  if (pos_err_.empty()) return s;

  s.pos_p50_m = percentile(pos_err_, 0.50);
  s.pos_p95_m = percentile(pos_err_, 0.95);
  s.pos_max_m = *std::max_element(pos_err_.begin(), pos_err_.end());
  s.pos_mean_m = mean_of(pos_err_);
  s.xy_mean_m = mean_of(xy_err_);
  s.z_mean_m = mean_of(z_err_);
  s.yaw_p50_rad = percentile(yaw_err_, 0.50);
  s.yaw_p95_rad = percentile(yaw_err_, 0.95);
  s.vyaw_mean_radps = mean_of(vyaw_err_);
  return s;
}

void GroundTruthEvaluator::reset()
{
  pos_err_.clear();
  xy_err_.clear();
  z_err_.clear();
  yaw_err_.clear();
  vyaw_err_.clear();
  seq_mismatches_ = 0;
  fetched_ = false;
}

}  // namespace sim_io
