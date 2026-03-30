#include "commandgener.hpp"

#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"  // 添加yaml头文件

namespace auto_aim
{
namespace multithread
{

CommandGener::CommandGener(
  auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
  tools::Plotter & plotter, bool debug)
: shooter_(shooter), aimer_(aimer), cboard_(cboard), plotter_(plotter), 
  stop_(false), debug_(debug), last_sent_command_({false, false, 0, 0})
{
  // 从配置文件加载阈值参数（假设从aimer或shooter的配置中获取）
  // 这里简化处理，实际应该从配置文件读取
  yaw_threshold_ = 0.2;  // 单位：弧度
  pitch_threshold_ = 0.2; // 单位：弧度
  // 移除四舍五入初始化
  thread_ = std::thread(&CommandGener::generate_command, this);
}

CommandGener::~CommandGener()
{
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void CommandGener::push(
  const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
  double bullet_speed, const Eigen::Vector3d & gimbal_pos)
{
  std::lock_guard<std::mutex> lock(mtx_);
  latest_ = {targets, t, bullet_speed, gimbal_pos};
  cv_.notify_one();
}

void CommandGener::generate_command()
{
  auto t0 = std::chrono::steady_clock::now();
  while (!stop_) {
    std::optional<Input> input;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (latest_ && tools::delta_time(std::chrono::steady_clock::now(), latest_->t) < 0.2) {
        input = latest_;
      } else
        input = std::nullopt;
    }
    if (input) {
      auto command = aimer_.aim(input->targets_, input->t, input->bullet_speed);
      command.shoot = shooter_.shoot(command, aimer_, input->targets_, input->gimbal_pos);
      command.horizon_distance = input->targets_.empty()
                                   ? 0
                                   : std::sqrt(
                                       tools::square(input->targets_.front().ekf_x()[0]) +
                                       tools::square(input->targets_.front().ekf_x()[2]));
      
      // 实现范围阈值逻辑，移除四舍五入操作
      if (command.control) {
        // 计算yaw和pitch的变化量
        double yaw_diff = std::abs(command.yaw - last_sent_command_.yaw);
        double pitch_diff = std::abs(command.pitch - last_sent_command_.pitch);
        
        // 如果变化量小于阈值，不再进行四舍五入
        
        // 只有当命令与上一次发送的命令不同时，才发送新命令
        if (command.yaw != last_sent_command_.yaw || 
            command.pitch != last_sent_command_.pitch ||
            command.shoot != last_sent_command_.shoot) {
          cboard_.send(command);
          last_sent_command_ = command;
        }
      } else {
        // 当不控制时，直接发送并更新last_sent_command_
        cboard_.send(command);
        last_sent_command_ = command;
      }
      
      if (debug_) {
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["cmd_yaw"] = command.yaw * 57.3;
        data["cmd_pitch"] = command.pitch * 57.3;
        data["shoot"] = command.shoot;
        data["horizon_distance"] = command.horizon_distance;
        plotter_.plot(data);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));  //approximately 500Hz
  }
}

}  // namespace multithread

}  // namespace auto_aim