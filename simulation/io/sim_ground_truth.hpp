#ifndef SIMULATION_IO__SIM_GROUND_TRUTH_HPP
#define SIMULATION_IO__SIM_GROUND_TRUTH_HPP

// 仿真真值读取与评估。
//
// 真值只能流向评估器，绝不允许进入 YOLO/Solver/Tracker/Planner 的输入，
// 否则整条链路的验证就失去意义。本文件不提供任何把真值写回算法状态的接口。

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>

#include "shared_memory_client.hpp"
#include "tasks/auto_aim/armor.hpp"

namespace sim_io
{
// 仿真端 ArmorLabel 与 sp_vision25 ArmorName 的枚举顺序不同，必须显式换算：
//   Rust  Sentry=0 One=1 Two=2 Three=3 Four=4 Five=5 Outpost=6 Base=7
//   C++   one=0 two=1 three=2 four=3 five=4 sentry=5 outpost=6 base=7
auto_aim::ArmorName armor_label_to_name(std::uint8_t label);
std::uint8_t armor_name_to_label(auto_aim::ArmorName name);

// 真值里的队伍编码，与仿真端 team_to_u8 一致（Red=0, Blue=1）。
constexpr std::uint8_t GT_TEAM_RED = 0;
constexpr std::uint8_t GT_TEAM_BLUE = 1;
constexpr std::uint8_t GT_TEAM_ANY = 255;  // 不过滤（仅诊断用，正式评估必须指定）

struct GtError
{
  bool valid = false;
  std::uint8_t armor_label = 0;
  std::uint8_t team = GT_TEAM_ANY;
  std::uint16_t identity = 0;
  auto_aim::ArmorName name = auto_aim::not_armor;
  double pos_err_m = 0.0;
  double xy_err_m = 0.0;
  double z_err_m = 0.0;
  double yaw_err_rad = 0.0;
  double vyaw_err_radps = 0.0;
  double gt_vyaw_radps = 0.0;
  Eigen::Vector3d gt_position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d est_position{Eigen::Vector3d::Zero()};

  // 被选中装甲板板心真值（若发布端提供）。整车中心与板心不是同一个点，
  // 瞄准误差必须用板心；估计误差（pos_err_m）仍以整车中心为基准，因为
  // Tracker 的 EKF 状态 (CX,CY,CZ) 估的就是车心。
  bool has_armor_position = false;
  bool armor_position_degraded = false;
  Eigen::Vector3d gt_armor_position{Eigen::Vector3d::Zero()};

  // 匹配是否发生了歧义：同一批真值里有多个目标同时满足 (team, label)。
  // 仿真场景里红蓝三号步兵共用 label=3，只按 label 取第一个命中会随机配错车。
  bool ambiguous = false;
  // 退化为最近邻匹配（label 没匹配上）。
  bool matched_by_nearest = false;
};

struct GtErrorStats
{
  std::size_t count = 0;
  double pos_p50_m = 0.0;
  double pos_p95_m = 0.0;
  double pos_max_m = 0.0;
  double pos_mean_m = 0.0;
  double xy_mean_m = 0.0;
  double z_mean_m = 0.0;
  double yaw_p50_rad = 0.0;
  double yaw_p95_rad = 0.0;
  double vyaw_mean_radps = 0.0;
};

class GroundTruthEvaluator
{
public:
  // enemy_team 必须显式给出：真值里红蓝双方的三号步兵都是 label=3（仿真场景
  // setup.rs 里 Infantry::new(Team::Red, INFANTRY_THREE_CONFIG) 与
  // Infantry::new(Team::Blue, INFANTRY_THREE_CONFIG) 共用同一个 armor 配置），
  // 只按 label 匹配会把自家车当成评估对象。GT_TEAM_ANY 仅供诊断。
  GroundTruthEvaluator(SharedMemoryClient & client, std::uint8_t enemy_team)
  : client_(client), enemy_team_(enemy_team)
  {
  }

  std::uint8_t enemy_team() const { return enemy_team_; }
  std::uint64_t ambiguous_matches() const { return ambiguous_matches_; }
  std::uint64_t nearest_matches() const { return nearest_matches_; }
  std::uint64_t degraded_matches() const { return degraded_matches_; }

  // fetch() 的统计覆盖每一次调用，而不是只覆盖有 Tracker 目标的帧。
  std::uint64_t fetch_attempts() const { return fetch_attempts_; }
  std::uint64_t fetch_success() const { return fetch_success_; }
  std::uint64_t fetch_missing() const { return fetch_missing_; }

  // 取本帧真值批次。数据来自 SharedMemoryClient::frame_ground_truth()，即
  // consume_frame() 在发布事务窗口里拷下来的那一份；仅当其 frame_seq 与图像
  // frame_seq/timestamp_ns 同时相等时才认为可用。协议 v4 规定两者严格相等，所以
  // seq_mismatches
  // 的期望值是 0，任何非零都是协议违例。
  bool fetch(std::uint64_t image_frame_seq, std::uint64_t image_timestamp_ns);

  // 只读最新一批真值，不校验帧号。**仅供人看的诊断输出**（例如确认场景里目标
  // 到底在哪个方向），绝不能用于 evaluate()/record()，否则等于拿不同帧的真值
  // 去评估估计值。命名刻意带 diagnostic_ 前缀，避免被误用。
  bool fetch_latest_diagnostic_only();
  bool fetched() const { return fetched_; }
  const GroundTruthBatch & batch() const { return batch_; }
  std::uint64_t frame_seq() const { return batch_.frame_seq; }
  std::uint32_t target_count() const { return fetched_ ? batch_.target_count : 0; }
  std::uint64_t seq_mismatches() const { return seq_mismatches_; }
  std::uint64_t timestamp_mismatches() const { return timestamp_mismatches_; }
  // 真值帧号减图像帧号的统计，仅在违例样本上累计（见 fetch()）。
  std::uint64_t seq_skew_samples() const { return seq_skew_samples_; }
  double seq_skew_mean() const
  {
    return seq_skew_samples_ == 0
             ? 0.0
             : static_cast<double>(seq_skew_sum_) / static_cast<double>(seq_skew_samples_);
  }
  std::int64_t seq_skew_min() const { return seq_skew_min_; }
  std::int64_t seq_skew_max() const { return seq_skew_max_; }

  // 按装甲板标签匹配；标签匹配不到时退化为最近邻匹配（gate_m 以内）。
  // estimate_in_odom 必须已经加上 odom 平移，即与真值同一坐标系。
  GtError evaluate(
    auto_aim::ArmorName name, const Eigen::Vector3d & estimate_in_odom, double yaw,
    double vyaw, double gate_m = 1.5);

  void record(const GtError & error);
  GtErrorStats stats() const;
  void reset();

private:
  SharedMemoryClient & client_;
  GroundTruthBatch batch_{};
  bool fetched_ = false;
  std::uint64_t seq_mismatches_ = 0;
  std::uint64_t timestamp_mismatches_ = 0;
  std::uint64_t fetch_attempts_ = 0;
  std::uint64_t fetch_success_ = 0;
  std::uint64_t fetch_missing_ = 0;
  std::uint64_t seq_skew_samples_ = 0;
  std::int64_t seq_skew_sum_ = 0;
  std::int64_t seq_skew_min_ = 0;
  std::int64_t seq_skew_max_ = 0;
  std::uint8_t enemy_team_ = GT_TEAM_ANY;
  std::uint64_t ambiguous_matches_ = 0;
  std::uint64_t nearest_matches_ = 0;
  std::uint64_t degraded_matches_ = 0;

  std::vector<double> pos_err_;
  std::vector<double> xy_err_;
  std::vector<double> z_err_;
  std::vector<double> yaw_err_;
  std::vector<double> vyaw_err_;

  // 按 (enemy_team, label) 匹配。命中多于一个时置 *ambiguous 并返回距
  // reference 最近的那个——但调用方必须把 ambiguous 记进报告，因为这说明
  // 场景配置或真值内容不足以唯一确定评估对象。
  std::optional<GroundTruthTarget> find_by_label(
    std::uint8_t label, const Eigen::Vector3d & reference, bool * ambiguous) const;
  std::optional<GroundTruthTarget> find_nearest(
    const Eigen::Vector3d & position, double gate_m) const;
  bool team_matches(std::uint8_t team) const
  {
    return enemy_team_ == GT_TEAM_ANY || team == enemy_team_;
  }
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_GROUND_TRUTH_HPP
