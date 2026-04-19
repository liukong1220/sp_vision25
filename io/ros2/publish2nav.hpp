#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/buffer.h>
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
  bool tryProjectTargetPoint(
    const Eigen::Vector3d & target_position_gimbal, geometry_msgs::msg::PointStamped & point_map);
  bool isFiniteVector(const Eigen::Vector3d & vector) const;

  rclcpp::Publisher<sp_msgs::msg::VisionTargetMsg>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_point_map_publisher_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string map_frame_ = "map";
  std::string gimbal_frame_ = "gimbal_yaw";
  double pose_timeout_s_ = 0.2;
  double nav_hold_min_confidence_ = 0.6;
  bool publish_target_point_map_ = true;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
