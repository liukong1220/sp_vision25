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

struct GtError
{
  bool valid = false;
  std::uint8_t armor_label = 0;
  auto_aim::ArmorName name = auto_aim::not_armor;
  double pos_err_m = 0.0;
  double xy_err_m = 0.0;
  double z_err_m = 0.0;
  double yaw_err_rad = 0.0;
  double vyaw_err_radps = 0.0;
  Eigen::Vector3d gt_position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d est_position{Eigen::Vector3d::Zero()};
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
  explicit GroundTruthEvaluator(SharedMemoryClient & client) : client_(client) {}

  // 拉取当前真值批次。仅当真值 frame_seq 与图像 frame_seq 一致时才认为可用，
  // 避免拿上一帧真值评估这一帧估计。
  bool fetch(std::uint64_t image_frame_seq);

  // 只读最新一批真值，不校验帧号。**仅供人看的诊断输出**（例如确认场景里目标
  // 到底在哪个方向），绝不能用于 evaluate()/record()，否则等于拿不同帧的真值
  // 去评估估计值。命名刻意带 diagnostic_ 前缀，避免被误用。
  bool fetch_latest_diagnostic_only();
  bool fetched() const { return fetched_; }
  const GroundTruthBatch & batch() const { return batch_; }
  std::uint64_t frame_seq() const { return batch_.frame_seq; }
  std::uint32_t target_count() const { return fetched_ ? batch_.target_count : 0; }
  std::uint64_t seq_mismatches() const { return seq_mismatches_; }

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

  std::vector<double> pos_err_;
  std::vector<double> xy_err_;
  std::vector<double> z_err_;
  std::vector<double> yaw_err_;
  std::vector<double> vyaw_err_;

  std::optional<GroundTruthTarget> find_by_label(std::uint8_t label) const;
  std::optional<GroundTruthTarget> find_nearest(
    const Eigen::Vector3d & position, double gate_m) const;
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_GROUND_TRUTH_HPP
