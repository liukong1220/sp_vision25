#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sp_msgs/msg/vision_target_msg.hpp"
#include "vision_target.hpp"

namespace io
{

// 负责把视觉融合状态发布到 ROS2。
// 这里单独做成一个 Node，方便直接 `ros2 topic echo vision/target`
// 查看视觉接管导航时到底发出了什么信息。
class Publish2Nav : public rclcpp::Node
{
public:
  explicit Publish2Nav(const std::string & config_path = "");

  ~Publish2Nav();

  void start();

  void send_data(const VisionTargetState & data);

private:
  void loadVisionFusionConfig(const std::string & config_path);
  void ensureTfListener();
  void publishCalibrationTfIfNeeded();
  builtin_interfaces::msg::Time resolveObservationStamp(const VisionTargetState & data) const;
  bool tryProjectTargetPoint(
    const Eigen::Vector3d & target_position_gimbal, const builtin_interfaces::msg::Time & stamp,
    geometry_msgs::msg::PointStamped & point_map);
  bool isFiniteVector(const Eigen::Vector3d & vector) const;

  rclcpp::Publisher<sp_msgs::msg::VisionTargetMsg>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_point_map_publisher_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::string map_frame_ = "map";
  std::string gimbal_frame_ = "front_vision_calib_gimbal";
  std::string camera_frame_ = "front_industrial_camera_optical_frame";
  std::string mount_frame_ = "gimbal_pitch";
  std::string calib_gimbal_frame_ = "front_vision_calib_gimbal";
  double pose_timeout_s_ = 0.2;
  double transform_timeout_s_ = 0.02;
  double transform_time_backoff_s_ = 0.01;
  double nav_hold_min_confidence_ = 0.6;
  bool publish_target_point_map_ = true;
  bool publish_calib_tf_ = true;
  bool has_camera_to_gimbal_calib_ = false;
  bool calib_tf_published_ = false;
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
};

}  // namespace io

#endif  // Publish2Nav_HPP_
