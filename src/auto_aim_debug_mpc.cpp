#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include "tasks/auto_aim/target.hpp"
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
  "{@config-path   | configs/standard3.yaml | 位置参数yaml配置文件路径 }";

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

int main(int argc, char * argv[]) {
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  // 添加原子变量来存储开火状态，确保线程安全
  std::atomic<bool> allow_fire = false;
  
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      // 更新开火状态
      allow_fire = plan.fire;
      
      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = rad2deg(plan.target_yaw);
      data["target_pitch"] = rad2deg(plan.target_pitch);

      data["plan_yaw"] = rad2deg(plan.yaw);
      data["plan_yaw_vel"] = rad2deg(plan.yaw_vel);
      data["plan_yaw_acc"] = rad2deg(plan.yaw_acc);

      data["plan_pitch"] = rad2deg(plan.pitch);
      data["plan_pitch_vel"] = rad2deg(plan.pitch_vel);
      data["plan_pitch_acc"] = rad2deg(plan.pitch_acc);

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

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

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      int armor_idx = 0;
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
        if (image_points.empty()) {
          ++armor_idx;
          continue;
        }

        // 在每个装甲板正下方绘制解算结果: yaw, x, y, z

        float min_x = image_points.front().x;
        float max_y = image_points.front().y;
        for (const auto & pt : image_points) {
          min_x = std::min(min_x, pt.x);
          max_y = std::max(max_y, pt.y);
        }

        int text_x = static_cast<int>(min_x);
        int text_y = static_cast<int>(max_y) + 22;
        text_x = std::max(0, std::min(text_x, img.cols - 220));
        text_y = std::max(30, std::min(text_y, img.rows - 130));

        // 竖排显示每个装甲板的解算值，使用高对比颜色（洋红）并加黑色描边
        std::vector<std::string> armor_lines = {
          fmt::format("armor:{}", armor_idx),
          fmt::format("yaw: {:.1f}", rad2deg(xyza[3])),
          fmt::format("x: {:.2f}", xyza[1]),
          fmt::format("y: {:.2f}", xyza[0]),
          fmt::format("z: {:.2f}", xyza[2]),
        };
        const double font_scale_local = 0.50;
        const int line_gap = 22;
        for (size_t line_i = 0; line_i < armor_lines.size(); ++line_i) {
          cv::Point org(text_x, text_y + static_cast<int>(line_i) * line_gap);
          cv::putText(
            img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX, font_scale_local,
            cv::Scalar(0, 0, 0), 3);
          cv::putText(
            img, armor_lines[line_i], org, cv::FONT_HERSHEY_SIMPLEX, font_scale_local,
            cv::Scalar(255, 0, 255), 2);
        }
        ++armor_idx;
      }

      // 预测装甲板转换轨迹：先细采样，再按像素位移稀疏化，确保平移时也能看见箭头
      auto target_future = target;
      constexpr int kRawTrajSteps = 18;
      constexpr double kRawTrajDt = 0.03;
      constexpr double kArrowMinPixelStep = 8.0;
      std::vector<cv::Point> raw_traj_centers;
      std::vector<int> raw_traj_ids;
      raw_traj_centers.reserve(kRawTrajSteps);
      raw_traj_ids.reserve(kRawTrajSteps);

      for (int step = 0; step < kRawTrajSteps; ++step) {
        const auto future_xyza_list = target_future.armor_xyza_list();
        if (future_xyza_list.empty()) break;

        int best_id = 0;
        double min_dist = 1e10;
        for (int i = 0; i < static_cast<int>(future_xyza_list.size()); ++i) {
          const double dist = future_xyza_list[i].head<2>().norm();
          if (dist < min_dist) {
            min_dist = dist;
            best_id = i;
          }
        }

        const auto & best_xyza = future_xyza_list[best_id];
        auto pred_points = solver.reproject_armor(
          best_xyza.head(3), best_xyza[3], target.armor_type, target.name);
        if (!pred_points.empty()) {
          cv::Point2f center(0.0f, 0.0f);
          for (const auto & pt : pred_points) {
            center.x += pt.x;
            center.y += pt.y;
          }
          center.x /= static_cast<float>(pred_points.size());
          center.y /= static_cast<float>(pred_points.size());

          raw_traj_centers.emplace_back(static_cast<int>(center.x), static_cast<int>(center.y));
          raw_traj_ids.push_back(best_id);
        }

        target_future.predict(kRawTrajDt);
      }

      std::vector<cv::Point> traj_centers;
      std::vector<int> traj_ids;
      if (!raw_traj_centers.empty()) {
        traj_centers.push_back(raw_traj_centers.front());
        traj_ids.push_back(raw_traj_ids.front());

        for (size_t i = 1; i < raw_traj_centers.size(); ++i) {
          const bool switched = raw_traj_ids[i] != traj_ids.back();
          const double pixel_step = cv::norm(raw_traj_centers[i] - traj_centers.back());
          const bool is_last = i + 1 == raw_traj_centers.size();
          if (switched || pixel_step >= kArrowMinPixelStep || is_last) {
            traj_centers.push_back(raw_traj_centers[i]);
            traj_ids.push_back(raw_traj_ids[i]);
          }
        }
      }

      for (size_t i = 0; i < traj_centers.size(); ++i) {
        cv::circle(img, traj_centers[i], 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        cv::circle(img, traj_centers[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      }

      for (size_t i = 1; i < traj_centers.size(); ++i) {
        const bool switched = traj_ids[i] != traj_ids[i - 1];
        const cv::Scalar traj_color = switched ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 0);

        cv::arrowedLine(
          img, traj_centers[i - 1], traj_centers[i], cv::Scalar(0, 0, 0), 5, cv::LINE_AA, 0, 0.32);
        cv::arrowedLine(
          img, traj_centers[i - 1], traj_centers[i], traj_color, 2, cv::LINE_AA, 0, 0.32);

        if (switched) {
          cv::putText(
            img, "switch", traj_centers[i] + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(0, 165, 255), 1);
        }
      }

      if (!traj_centers.empty()) {
        cv::putText(
          img, "pred traj", traj_centers.front() + cv::Point(8, -10), cv::FONT_HERSHEY_SIMPLEX,
          0.5, cv::Scalar(255, 255, 0), 2);
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }
    
    // 创建一个新的图像来显示信息
    cv::Mat display_img;
    cv::resize(img, display_img, {}, 0.7, 0.7);  // 显示时缩小图片尺寸
    
    // 获取当前规划器输出的控制指令
    auto gs = gimbal.state();
    // 由于我们无法直接访问plan变量，我们需要重新规划一次来获取当前的计划值
    std::optional<auto_aim::Target> current_target;
    if (!targets.empty()) {
        current_target = targets.front();
    }
    auto current_plan = planner.plan(current_target, gs.bullet_speed);
    
    // 在图像上绘制文本信息
    int baseline = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.4;
    cv::Scalar color = cv::Scalar(0, 255, 255);  // 黄色
    int thickness = 1;
    

    // 画面中心上方显示开火提示（仅可开火时显示）
    if (current_plan.fire) {
      std::string fire_text = "fire!";
      int fire_baseline = 0;
      cv::Size fire_size =
        cv::getTextSize(fire_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &fire_baseline);
      cv::Point fire_org((display_img.cols - fire_size.width) / 2, fire_size.height + 10);
      cv::putText(
        display_img, fire_text, fire_org, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    // 右上角显示当前端到端延迟（从图像时间戳到当前显示）
    double latency_ms = tools::delta_time(std::chrono::steady_clock::now(), t) * 1000.0;
    std::string latency_info = fmt::format("latency: {:.2f} ms", latency_ms);
    int latency_baseline = 0;
    cv::Size latency_size =
      cv::getTextSize(latency_info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &latency_baseline);
    cv::Point latency_org(display_img.cols - latency_size.width - 10, latency_size.height + 10);
    cv::putText(
      display_img, latency_info, latency_org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 255, 255), 1);

    cv::imshow("Auto Aim Debug", display_img);  // 单一窗口显示所有信息
    auto key = cv::waitKey(1); 
    if (key == 'q') break; 
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
