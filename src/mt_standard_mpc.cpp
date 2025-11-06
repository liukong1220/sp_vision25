#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | show help message}"
  "{@config-path   | configs/standard3.yaml | yaml config path}";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 使用多线程检测器
  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  // 使用MPC规划器
  auto_aim::Planner planner(config_path);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  // 连接主线程和规划线程的目标队列
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};
  std::atomic<bool> quit = false;

  // 检测线程：从相机读取图像并进行检测
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        camera.read(img, t);
        detector.push(img, t);
      } else {
        // 在非自瞄模式下，让出CPU
        std::this_thread::sleep_for(10ms);
      }
    }
  });

  // 规划线程：获取目标并运行MPC进行规划和控制
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        std::this_thread::sleep_for(10ms);
      } else {
        // 在没有目标或非自瞄模式下，降低运行频率
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  // 主线程：模式切换、获取检测结果、跟踪、打符逻辑
  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    // 自瞄模式
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto [img, armors, t] = detector.debug_pop();
      if (img.empty()) {
          std::this_thread::sleep_for(1ms);
          continue;
      }
      auto q = gimbal.q(t);
      
      // recorder.record(img, q, t);

      solver.set_R_gimbal2world(q);
      auto targets = tracker.track(armors, t);

      // 将跟踪到的目标推入队列，供规划线程使用
      if (!targets.empty()) {
        target_queue.push(targets.front());
      } else {
        target_queue.push(std::nullopt);
      }
    }
    // 打符模式
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      cv::Mat img;
      std::chrono::steady_clock::time_point t;
      camera.read(img, t);
      auto q = gimbal.q(t);
      auto gs = gimbal.state();

      // recorder.record(img, q, t);

      buff_solver.set_R_gimbal2world(q);
      auto power_runes = buff_detector.detect(img);
      buff_solver.solve(power_runes);

      auto_aim::Plan buff_plan;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      }
      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

    } else { // 空闲模式
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
      std::this_thread::sleep_for(10ms);
    }
  }

  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}