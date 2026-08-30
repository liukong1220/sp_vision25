#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <chrono>
#include <list>
#include <string>
#include <vector>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  std::string config_path_;
  uint64_t runtime_params_version_ = 0;
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  // 判定“相机离线”的帧间隔上限(秒)。超过就把状态打回 lost。
  // 默认 0.1 是按实车流水线速度定的；仿真里 CPU 上跑 YOLO 单帧就要 110ms 以上，
  // dt 恒大于 0.1，tracker 每帧都会被打回 lost，永远确认不了目标。
  // 所以做成可配置(configs/simulation.yaml: tracker_max_dt)，实车默认值不变。
  double max_dt_;
  double outpost_radius_;
  double outpost_spin_speed_lock_;
  bool outpost_fixed_center_rotation_model_;
  std::vector<double> outpost_armor_z_offsets_;
  TargetEstimatorParams estimator_params_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  void refresh_runtime_params_if_needed();

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
