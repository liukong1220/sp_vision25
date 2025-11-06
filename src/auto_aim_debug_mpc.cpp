#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
 
  // 初始化退出信号处理器
  tools::Exiter exiter;
  // 初始化绘图工具，用于数据可视化
  tools::Plotter plotter;

  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  // 获取配置文件路径
  auto config_path = cli.get<std::string>(0);
  // 如果有 help 参数或配置文件路径为空，则打印帮助信息并退出
 
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
 
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

 
  // 初始化云台通信模块
  io::Gimbal gimbal(config_path);
  // 初始化相机模块
  io::Camera camera(config_path);

  // 初始化YOLO检测器
  auto_aim::YOLO yolo(config_path, true);
  // 初始化姿态解算器
  auto_aim::Solver solver(config_path);
  // 初始化目标跟踪器
  auto_aim::Tracker tracker(config_path, solver);
  // 初始化运动规划器 (MPC)
  auto_aim::Planner planner(config_path);

  // 创建一个线程安全队列，用于在检测线程和规划线程之间传递目标
  // 容量为1，表示只关心最新的目标
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  // 初始化时推入一个空目标
  target_queue.push(std::nullopt);

  // 创建一个原子布尔值，用于安全地终止规划线程
  std::atomic<bool> quit = false;
  // 创建并启动规划线程
 
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
 
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

 
    // 循环直到接收到退出信号
    while (!quit) {
      // 从队列中获取最新的目标
      auto target = target_queue.front();
      // 获取云台的当前状态
      auto gs = gimbal.state();
      // 根据目标和弹速进行运动规划，得到控制指令
      auto plan = planner.plan(target, gs.bullet_speed);

      // 将规划好的控制指令发送给云台
 
    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

 
      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

 
      // 检测是否发射了子弹
      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      // 准备用于绘图的数据
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      // 记录云台状态数据
 
      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

 
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

 
      // 记录规划器的目标姿态
      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      // 记录规划出的详细运动指令
 
      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

 
      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

 
      // 记录开火指令和状态
      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

      // 如果存在目标，记录其EKF状态
 
      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

 
      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
      }

      if (target.has_value()) {
 
        data["w"] = target->ekf_x()[7]; //角速度
 
        data["w"] = target->ekf_x()[7];
 
      } else {
        data["w"] = 0.0;
      }

 
      // 发送数据到绘图工具
      plotter.plot(data);

      // 线程休眠10ms，控制规划和发送频率为100Hz
 
      plotter.plot(data);

 
      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

 
  auto last_fps_time = std::chrono::steady_clock::now();
  int frame_count = 0;

  // 主循环，处理图像检测和跟踪
  while (!exiter.exit()) {

    frame_count++;
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time).count() >= 1) {
      // 输出到终端
      std::cout << "FPS: " << frame_count << std::endl;
      frame_count = 0;
      last_fps_time = now;
    }

    // 读取相机图像和时间戳
    camera.read(img, t);
    // 获取对应时间戳的云台姿态
    auto q = gimbal.q(t);

    // 更新解算器的云台姿态
    solver.set_R_gimbal2world(q);
    // 使用YOLO检测图像中的装甲板
    auto armors = yolo.detect(img);
    // 跟踪检测到的装甲板，形成稳定目标
    auto targets = tracker.track(armors, t);
    // 将最优先的目标推送到队列中
 
  while (!exiter.exit()) {
    camera.read(img, t);
    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);
    auto targets = tracker.track(armors, t);
 
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

 
    // 如果有目标，则进行调试信息的可视化
    if (!targets.empty()) {
      auto target = targets.front();

      // 将目标的装甲板重投影到图像上并用绿色绘制
 
    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
 
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

 
      // 将规划器的调试用瞄准点重投影到图像上并用红色绘制
  
      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

 
    // 调整图像大小并显示
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    // 等待按键，如果按下 'q' 则退出循环
 
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
 
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

 
  // 通知规划线程退出
  quit = true;
  // 等待规划线程执行完毕
  if (plan_thread.joinable()) plan_thread.join();
  // 发送停止指令给云台
 
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
 
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}