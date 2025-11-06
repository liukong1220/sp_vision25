#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
 
  "{@config-path   | configs/sentry.yaml | 位置参数yaml配置文件路径 }";
 
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";
 

using namespace std::chrono;

int main(int argc, char * argv[])
{
 
  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  // 获取配置文件路径
  auto config_path = cli.get<std::string>(0);
  // 如果有 help 参数或配置文件路径为空，则打印帮助信息并退出
 
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
 
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

 
  // 初始化退出信号处理器
  tools::Exiter exiter;
  // 初始化绘图工具
  tools::Plotter plotter;
  // 初始化视频录制工具，缓冲区大小为100帧
  tools::Recorder recorder(100);  //根据实际帧率调整

  // 初始化主控板通信模块
  io::CBoard cboard(config_path);
  // 初始化相机模块
  io::Camera camera(config_path);

  // 初始化多线程装甲板检测器
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  // 初始化姿态解算器
  auto_aim::Solver solver(config_path);
  // 初始化目标跟踪器
  auto_aim::Tracker tracker(config_path, solver);
  // 初始化目标瞄准器
  auto_aim::Aimer aimer(config_path);
  // 初始化射击控制器
  auto_aim::Shooter shooter(config_path);
  // 初始化多线程指令生成器
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter, true);

  // 创建并启动一个独立的检测线程
 
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(100);  //根据实际帧率调整

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter, true);

 
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

 
    // 循环直到接收到退出信号
    while (!exiter.exit()) {
      // 从相机读取一帧图像和时间戳
      camera.read(img, t);
      // 将图像和时间戳推送到检测器进行异步处理
 
    while (!exiter.exit()) {
      camera.read(img, t);
 
      detector.push(img, t);
    }
  });

 
  // 初始化模式变量
  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  // 主循环，直到接收到退出信号
  while (!exiter.exit()) {
    auto t0 = std::chrono::steady_clock::now();
    /// 自瞄核心逻辑
    // 从检测器获取带有调试信息的结果（图像、装甲板列表、时间戳）
    auto [img, armors, t] = detector.debug_pop();
    // 获取大约在图像采集时间点的IMU数据（四元数）
    Eigen::Quaterniond q = cboard.imu_at(t - 1ms);
    // 获取当前机器人模式
    mode = cboard.mode;

    // 如果模式发生变化，记录日志
 
  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  while (!exiter.exit()) {
    auto t0 = std::chrono::steady_clock::now();
    /// 自瞄核心逻辑
    auto [img, armors, t] = detector.debug_pop();
    Eigen::Quaterniond q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;

 
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

 
    // 在解算器中设置云台到世界坐标系的旋转矩阵
    solver.set_R_gimbal2world(q);

    // 计算云台的欧拉角 (yaw, pitch, roll)
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // 使用跟踪器处理检测到的装甲板，更新目标状态
    auto targets = tracker.track(armors, t);

    // 将跟踪到的目标、时间戳、弹速和云台姿态推送到指令生成器
    commandgener.push(targets, t, cboard.bullet_speed, ypr);  // 发送给决策线程

    /// 调试信息显示
    // 在图像上绘制跟踪器的当前状态
    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    // 创建json对象用于绘图和调试
    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

    // 记录原始装甲板观测数据
 
    solver.set_R_gimbal2world(q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto targets = tracker.track(armors, t);

    commandgener.push(targets, t, cboard.bullet_speed, ypr);  // 发送给决策线程

    /// debug
    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

    // 装甲板原始观测数据
 
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
 
      // 找到最左边的装甲板作为参考
  
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }  //always left
 
      // 解算该装甲板在世界坐标系下的位置
  
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    }

 
    // 如果有跟踪到的目标
    if (!targets.empty()) {
      auto target = targets.front();

      // 将目标的装甲板列表重投影到图像上并绘制
 
    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
 
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
 
        tools::draw_points(img, image_points, {0, 255, 0}); // 绿色绘制
      }

      // 获取并绘制瞄准点
 
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
 
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
 
        tools::draw_points(img, image_points, {0, 0, 255}); // 红色绘制有效瞄准点
      else
        tools::draw_points(img, image_points, {255, 0, 0}); // 蓝色绘制无效瞄准点

      // 记录观测器（EKF）的内部状态数据
 
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      // 观测器内部数据
 
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
 
      data["a"] = x[6] * 57.3; // 角度转为度
 
      data["a"] = x[6] * 57.3;
 
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

 
      // 记录卡方检验相关数据
 
      // 卡方检验数据
 
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

 
    // 记录云台响应情况
 
    // 云台响应情况
 
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_pitch"] = ypr[1] * 57.3;
    data["bullet_speed"] = cboard.bullet_speed;

 
    // 将所有调试数据发送到绘图工具
    plotter.plot(data);

    // 调整图像大小并显示
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    // 等待按键，如果按下 'q' 则退出循环
 
    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
 
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

 
  // 等待检测线程结束
  
  detect_thread.join();

  return 0;
}