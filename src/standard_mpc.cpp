#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | yaml配置文件路径 }";

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

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["mode"] = static_cast<int>(mode.load());

      auto gs = gimbal.state();
      data["gimbal_yaw"] = gs.yaw;      
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;
      data["bullet_speed"] = gs.bullet_speed;

      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto plan = planner.plan(target, gs.bullet_speed);

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        auto fired = gs.bullet_count > last_bullet_count;
        last_bullet_count = gs.bullet_count;

        data["plan_control"] = plan.control ? 1 : 0;
        data["plan_fire"] = plan.fire ? 1 : 0;
        data["fired"] = fired ? 1 : 0;
        data["target_yaw"] = plan.target_yaw;
        data["target_pitch"] = plan.target_pitch;
        data["plan_yaw"] = plan.yaw;
        data["plan_yaw_vel"] = plan.yaw_vel;
        data["plan_yaw_acc"] = plan.yaw_acc;
        data["plan_pitch"] = plan.pitch;
        data["plan_pitch_vel"] = plan.pitch_vel;
        data["plan_pitch_acc"] = plan.pitch_acc;

        if (target.has_value()) {
          data["target_x"] = target->ekf_x()[0];
          data["target_y"] = target->ekf_x()[2];
          data["target_z"] = target->ekf_x()[4];
          data["target_w"] = target->ekf_x()[7];
        }

        plotter.plot(data);
        std::this_thread::sleep_for(10ms);
      } else {
        data["plan_control"] = 0;
        data["plan_fire"] = 0;
        data["fired"] = 0;
        plotter.plot(data);
        std::this_thread::sleep_for(50ms);
      }
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    camera.read(img, t);
    auto q = gimbal.q(t);
    auto gs = gimbal.state();
    // recorder.record(img, q, t);  // 需要录像时再开启
    solver.set_R_gimbal2world(q);

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto armors = yolo.detect(img);
      auto targets = tracker.track(armors, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    }

    /// 打符
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
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

    } else
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
