#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <iostream>
#include <list>
#include <unordered_map>

#include "detection.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/ros2/vision_target.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace omniperception
{

// 给 ROS 融合层使用的最小目标摘要。
// 只保留“当前是否有效、目标编号、位置和置信度”这些行为树真正会消费的字段，
// 避免把 Tracker/Armor 的内部细节泄漏到外部接口层。
struct VisionTargetInfo
{
  bool valid = false;
  int target_id = 0;
  double confidence = 0.0;
  Eigen::Vector3d position_gimbal = Eigen::Vector3d::Zero();
};

class Decider
{
public:
  Decider(const std::string & config_path);

  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
    io::USBCamera & usbcam2, io::Camera & back_cammera);

  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera);

  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  bool armor_filter(std::list<auto_aim::Armor> & armors);

  void set_priority(std::list<auto_aim::Armor> & armors);
  //对队列中的每一个DetectionResult进行过滤，同时将DetectionResult排序
  void sort(std::vector<DetectionResult> & detection_queue);

  VisionTargetInfo get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

  // 把自瞄命令和目标摘要统一整理成对外发布的融合消息，
  // 让 sentry / sentry_debug / sentry_multithread 这些入口共用同一套字段语义。
  io::VisionTargetState build_vision_target_state(
    const io::Command & command, const VisionTargetInfo & target_info) const;

  void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

  void get_auto_aim_target(
    std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target);

private:
  int img_width_;
  int img_height_;
  double fov_h_, new_fov_h_;
  double fov_v_, new_fov_v_;
  int mode_;
  int count_;

  // 第一代视觉导航融合参数。
  // 当前先用“目标相对方位 + 距离”给行为树提供建议点索引，
  // 等后面接入更强的地图语义后，再把它升级成真正的区域级映射。
  bool enable_goal_suggestion_ = false;
  int left_goal_index_ = -1;
  int front_goal_index_ = -1;
  int right_goal_index_ = -1;
  int close_range_goal_index_ = -1;
  double front_yaw_abs_threshold_rad_ = 20.0 / 57.3;
  double close_range_threshold_m_ = 4.0;

  auto_aim::Color enemy_color_;
  auto_aim::YOLO detector_;
  std::vector<auto_aim::ArmorName> invincible_armor_;  //无敌状态机器人编号,英雄为1，哨兵为6

  // 定义ArmorName到ArmorPriority的映射类型
  using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

  const PriorityMap mode1 = {
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::forth},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::fifth}};

  const PriorityMap mode2 = {
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};

  // 根据目标相对方位给行为树一个“建议前往哪个点”的索引。
  int suggest_goal_index(const VisionTargetInfo & target_info) const;
};

enum PriorityMode
{
  MODE_ONE = 1,
  MODE_TWO
};

}  // namespace omniperception

#endif
