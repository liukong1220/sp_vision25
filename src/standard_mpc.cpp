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
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  // 获取配置文件路径
  auto config_path = cli.get<std::string>("@config-path");
  // 如果用户请求帮助或未提供配置路径，则打印帮助信息并退出
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化退出信号管理器
  tools::Exiter exiter;
  // 初始化绘图工具
  tools::Plotter plotter;
  // 初始化数据记录器
  tools::Recorder recorder;

  // 初始化云台控制接口
  io::Gimbal gimbal(config_path);
  // 初始化摄像头读取接口
  io::Camera camera(config_path);

  // 初始化YOLO目标检测器（自瞄用）
  auto_aim::YOLO yolo(config_path, true);
  // 初始化求解器（用于解算目标位置）
  auto_aim::Solver solver(config_path);
  // 初始化追踪器（用于追踪装甲板目标）
  auto_aim::Tracker tracker(config_path, solver);
  // 初始化路径规划器（用于预测弹道）
  auto_aim::Planner planner(config_path);

  // 创建一个线程安全的目标队列，容量为1，用于传递目标信息
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  // 初始时队列中放入一个空目标（表示无目标）
  target_queue.push(std::nullopt);

  // 初始化打符检测器
  auto_buff::Buff_Detector buff_detector(config_path);
  // 初始化打符求解器
  auto_buff::Solver buff_solver(config_path);
  // 小符目标处理对象
  auto_buff::SmallTarget buff_small_target;
  // 大符目标处理对象
  auto_buff::BigTarget buff_big_target;
  // 打符瞄准器
  auto_buff::Aimer buff_aimer(config_path);

  // 图像帧
  cv::Mat img;
  // 云台姿态四元数
  Eigen::Quaterniond q;
  // 时间戳
  std::chrono::steady_clock::time_point t;

  // 退出标志，用于控制线程
  std::atomic<bool> quit = false;

  // 当前云台模式（使用原子类型确保线程安全）
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  // 上一次云台模式，用于检测模式变化
  auto last_mode{io::GimbalMode::IDLE};

  // 启动规划线程，用于处理自瞄目标并发送控制指令
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      // 如果有目标且当前为自瞄模式
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        // 取出目标
        auto target = target_queue.front();
        // 获取云台状态
        auto gs = gimbal.state();
        // 规划弹道
        auto plan = planner.plan(target, gs.bullet_speed);

        // 发送控制指令到云台
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        // 短暂休眠
        std::this_thread::sleep_for(10ms);
      } else
        // 如果没有目标或非自瞄模式，较长休眠
        std::this_thread::sleep_for(200ms);
    }
  });

  // 主循环：不断读取图像和处理任务
  while (!exiter.exit()) {
    // 获取当前云台模式
    mode = gimbal.mode();

    // 检测是否模式发生变化
    if (last_mode != mode) {
      // 记录模式切换日志
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      // 更新上一次模式
      last_mode = mode.load();
    }

    // 从摄像头读取图像和时间戳
    camera.read(img, t);
    // 获取当前云台姿态（四元数）
    auto q = gimbal.q(t);
    // 获取云台状态（如子弹速度等）
    auto gs = gimbal.state();
    // 记录图像和姿态信息
    recorder.record(img, q, t);
    // 设置求解器的世界坐标系到云台坐标系的旋转矩阵
    solver.set_R_gimbal2world(q);

    /// 自瞄处理
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      // 使用YOLO检测装甲板
      auto armors = yolo.detect(img);
      // 使用追踪器追踪目标
      auto targets = tracker.track(armors, t);
      // 如果有追踪到的目标，放入队列
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        // 否则放入空目标
        target_queue.push(std::nullopt);
    }

    /// 打符处理
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      // 设置打符求解器的旋转矩阵
      buff_solver.set_R_gimbal2world(q);

      // 检测能量机关（打符）
      auto power_runes = buff_detector.detect(img);

      // 求解打符目标
      buff_solver.solve(power_runes);

      // 打符规划结果
      auto_aim::Plan buff_plan;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        // 获取小符目标
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        // 使用MPC算法进行瞄准
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        // 获取大符目标
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        // 使用MPC算法进行瞄准
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      }
      // 发送打符控制指令
      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

    } else
      // 其他模式下发送空控制指令
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  }

  // 设置退出标志
  quit = true;
  // 等待规划线程结束
  if (plan_thread.joinable()) plan_thread.join();
  // 发送最终的空控制指令
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}