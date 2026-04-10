#ifndef IO__VISION_TARGET_HPP
#define IO__VISION_TARGET_HPP

#include <Eigen/Dense>

namespace io
{

// 视觉主链对外发布的融合状态。
// 这里刻意把行为树真正关心的字段拆平，方便：
// 1. ROS 话题直接观测；
// 2. 调试时快速确认“是谁在接管导航/云台”；
// 3. 后续继续扩展建议导航点、目标地图坐标等信息。
struct VisionTargetState
{
  bool tracking = false;
  bool nav_hold = false;
  bool fire_permitted = false;
  int target_id = 0;
  int suggested_goal_index = -1;
  double confidence = 0.0;
  double target_distance = 0.0;
  double target_yaw = 0.0;
  double target_pitch = 0.0;
  Eigen::Vector3d target_position_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_position_map = Eigen::Vector3d::Zero();
};

}  // namespace io

#endif  // IO__VISION_TARGET_HPP
