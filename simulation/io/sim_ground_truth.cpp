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

bool GroundTruthEvaluator::fetch(
  std::uint64_t image_frame_seq, std::uint64_t image_timestamp_ns)
{
  ++fetch_attempts_;
  fetched_ = false;
  GroundTruthBatch batch{};
  // 只认 consume_frame() 在事务窗口里拷下来的那一份，绝不在这里现读槽位。
  // 本函数在检测/解算之后调用，距 consume_frame() 已有 ~250 ms，那时背压早已放开，
  // 槽位可能已经被后面若干帧覆盖——现读必然读到更新的批次，同帧校验恒不命中。
  if (!client_.frame_ground_truth(&batch)) {
    ++fetch_missing_;
    return false;
  }

  if (batch.frame_seq != image_frame_seq) {
    ++seq_mismatches_;
    // 协议 v4 规定图像、同帧姿态、同帧真值三者 frame_seq/timestamp_ns 严格相等，没有任何允许
    // 的偏移，所以这里任何一次计数都是协议违例，而不是"正常的一两帧延迟"。
    // 保留 skew 的方向与幅度是为了区分违例来源：
    //   d < 0（真值更旧）= 发布端没把真值放进图像那次事务；
    //   d > 0（真值更新）= 消费端取的不是事务窗口里的那一份（背压已放开后现读）。
    const std::int64_t d = static_cast<std::int64_t>(batch.frame_seq) -
                           static_cast<std::int64_t>(image_frame_seq);
    if (seq_skew_samples_ == 0 || d < seq_skew_min_) seq_skew_min_ = d;
    if (seq_skew_samples_ == 0 || d > seq_skew_max_) seq_skew_max_ = d;
    seq_skew_sum_ += d;
    ++seq_skew_samples_;
    return false;
  }
  if (batch.timestamp_ns != image_timestamp_ns) {
    ++timestamp_mismatches_;
    return false;
  }
  for (std::uint32_t i = 0; i < batch.target_count && i < GROUND_TRUTH_MAX_TARGETS; ++i) {
    if (
      batch.targets[i].frame_seq != image_frame_seq ||
      batch.targets[i].timestamp_ns != image_timestamp_ns) {
      ++timestamp_mismatches_;
      return false;
    }
  }
  for (std::uint32_t i = 0; i < batch.rune_count && i < GROUND_TRUTH_MAX_RUNES; ++i) {
    if (
      batch.runes[i].frame_seq != image_frame_seq ||
      batch.runes[i].timestamp_ns != image_timestamp_ns) {
      ++timestamp_mismatches_;
      return false;
    }
  }
  if (batch.target_count > GROUND_TRUTH_MAX_TARGETS) {
    ++fetch_missing_;
    return false;
  }

  batch_ = batch;
  fetched_ = true;
  ++fetch_success_;
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

std::optional<GroundTruthTarget> GroundTruthEvaluator::find_by_label(
  std::uint8_t label, const Eigen::Vector3d & reference, bool * ambiguous) const
{
  if (ambiguous) *ambiguous = false;
  if (!fetched_) return std::nullopt;

  // 先按 (队伍, 标签) 过滤。原来只比 armor_label 并返回第一个命中，
  // 而仿真场景里红蓝三号步兵共用 label=3，谁先被写进 targets[] 就评估谁——
  // 这会让"估计 vs 真值"的误差里混进另一辆车的位置。
  std::optional<GroundTruthTarget> best;
  double best_dist = 0.0;
  std::uint32_t hits = 0;
  for (std::uint32_t i = 0; i < batch_.target_count; ++i) {
    const GroundTruthTarget & t = batch_.targets[i];
    if (t.armor_label != label) continue;
    if (!team_matches(t.team)) continue;
    ++hits;
    const Eigen::Vector3d p(t.position[0], t.position[1], t.position[2]);
    const double d = (p - reference).norm();
    if (!best.has_value() || d < best_dist) {
      best_dist = d;
      best = t;
    }
  }

  // 过滤后仍有多个命中：场景里存在同队同编号的多辆车，真值内容不足以唯一
  // 确定评估对象。这里取最近的一个以便继续出数，但必须把歧义上报。
  if (hits > 1 && ambiguous) *ambiguous = true;
  return best;
}

std::optional<GroundTruthTarget> GroundTruthEvaluator::find_nearest(
  const Eigen::Vector3d & position, double gate_m) const
{
  if (!fetched_) return std::nullopt;

  std::optional<GroundTruthTarget> best;
  double best_dist = gate_m;
  for (std::uint32_t i = 0; i < batch_.target_count; ++i) {
    const GroundTruthTarget & t = batch_.targets[i];
    // 最近邻退化路径同样必须过滤队伍，否则自家车正好更近时就会被选中。
    if (!team_matches(t.team)) continue;
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

  bool ambiguous = false;
  std::optional<GroundTruthTarget> gt =
    find_by_label(armor_name_to_label(name), estimate_in_odom, &ambiguous);
  if (!gt.has_value()) {
    gt = find_nearest(estimate_in_odom, gate_m);
    if (gt.has_value()) {
      out.matched_by_nearest = true;
      ++nearest_matches_;
    }
  }
  if (!gt.has_value()) return out;
  if (ambiguous) ++ambiguous_matches_;

  const Eigen::Vector3d p(gt->position[0], gt->position[1], gt->position[2]);
  const Eigen::Vector3d d = estimate_in_odom - p;

  out.valid = true;
  out.ambiguous = ambiguous;
  out.armor_label = gt->armor_label;
  out.team = gt->team;
  out.identity = gt->identity;
  out.gt_position = p;
  if (gt->armor_position_valid != 0) {
    out.has_armor_position = true;
    out.gt_armor_position =
      Eigen::Vector3d(gt->armor_position[0], gt->armor_position[1], gt->armor_position[2]);
  }
  out.armor_position_degraded = gt->armor_position_degraded != 0;
  if (out.armor_position_degraded) ++degraded_matches_;
  out.pos_err_m = d.norm();
  out.xy_err_m = d.head<2>().norm();
  out.z_err_m = std::abs(d.z());
  out.yaw_err_rad = std::abs(normalize_angle(yaw - static_cast<double>(gt->yaw)));
  out.vyaw_err_radps = std::abs(vyaw - static_cast<double>(gt->vyaw));
  out.gt_vyaw_radps = static_cast<double>(gt->vyaw);
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
  timestamp_mismatches_ = 0;
  fetch_attempts_ = 0;
  fetch_success_ = 0;
  fetch_missing_ = 0;
  seq_skew_samples_ = 0;
  seq_skew_sum_ = 0;
  seq_skew_min_ = 0;
  seq_skew_max_ = 0;
  ambiguous_matches_ = 0;
  nearest_matches_ = 0;
  degraded_matches_ = 0;
  fetched_ = false;
}

}  // namespace sim_io
