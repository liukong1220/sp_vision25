#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <memory>

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
  Publish2Nav();

  ~Publish2Nav();

  void start();

  void send_data(const VisionTargetState & data);

private:
  rclcpp::Publisher<sp_msgs::msg::VisionTargetMsg>::SharedPtr publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
