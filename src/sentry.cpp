#include <chrono>
#include <limits>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  io::ROS2 ros2(config_path);
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  // 当前实机还没有接入全向感知硬件，这里先不初始化侧向 / 后向相机，
  // 避免程序在启动阶段因为缺少设备而卡住。
  // 后续恢复全向感知时，直接取消下面代码块的注释即可。
#if 0
  io::Camera back_camera(tools::resolve_runtime_path_string("configs/camera.yaml"));
  io::USBCamera usbcam1("video0", config_path);
  io::USBCamera usbcam2("video2", config_path);
#endif

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  omniperception::Decider decider(config_path);
  tools::logger()->info(
    "Omniperception fallback is temporarily disabled in sentry.cpp; tracker lost will stop gimbal control.");

  cv::Mat img;

  std::chrono::steady_clock::time_point timestamp;
  const auto invalid_target_point =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    const auto q = gimbal.q(timestamp);
    const auto gs = gimbal.state();

    solver.set_R_gimbal2world(q);

    auto armors = yolo.detect(img);
    decider.get_invincible_armor(ros2.subscribe_enemy_status());
    decider.armor_filter(armors);
    // decider.get_auto_aim_target(armors, ros2.subscribe_autoaim_target());
    decider.set_priority(armors);

    auto targets = tracker.track(armors, timestamp);
    io::Command command{false, false, 0, 0};
    const auto tracker_state = tracker.state();
    const bool tracker_lost = tracker_state == "lost";

    if (tracker_lost) {
      // 当前版本先关闭“目标丢失后切到全向感知相机继续搜敌”的支路。
      // 这样在没有侧向 / 后向相机的实机上，tracker 丢失后会直接停止控制，
      // 不会因为缺少这些硬件而阻塞主程序。
      //
      // 后续恢复全向感知时，打开下面代码块并恢复上面的相机初始化即可：
      //
      // command = decider.decide(yolo, gimbal_pos, usbcam1, usbcam2, back_camera);
      // gimbal.send(
      //   command.control, command.shoot, static_cast<float>(command.yaw), 0.0F, 0.0F,
      //   static_cast<float>(command.pitch), 0.0F, 0.0F);
      gimbal.send(false, false, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
    } else {
      std::optional<auto_aim::Target> tracked_target = std::nullopt;
      if (!targets.empty()) {
        tracked_target = targets.front();
      }

      const auto plan = planner.plan(tracked_target, gs.bullet_speed);
      command.control = plan.control;
      command.shoot = plan.fire;
      command.yaw = plan.yaw;
      command.pitch = plan.pitch;

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);
    }

    const auto target_info = decider.get_target_info(armors, targets);
    auto vision_target_state = decider.build_vision_target_state(command, target_info);
    if (!target_info.valid) {
      vision_target_state.target_position_gimbal = invalid_target_point;
    }
    ros2.publish(vision_target_state);
  }

  gimbal.send(false, false, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
  return 0;
}
