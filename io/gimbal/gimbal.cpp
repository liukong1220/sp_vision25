#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/runtime_params.hpp"
#include "tools/yaml.hpp"

namespace io
{
  // 帧率计算相关变量
  auto frame_start_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  double fps = 0.0;
  auto last_fps_update = std::chrono::steady_clock::now();

// 构造函数：从配置文件初始化云台串口连接并启动读取线程
Gimbal::Gimbal(const std::string & config_path)
: config_path_(tools::resolve_config_path_string(config_path))
{
  current_com_port_ = resolve_desired_port();

  open_serial(current_com_port_);
  thread_ = std::thread(&Gimbal::read_thread, this);
}

// 析构函数：安全关闭云台连接和线程
Gimbal::~Gimbal()
{
  quit_ = true;
  close_serial();
  if (thread_.joinable()) thread_.join();
}

// 获取当前云台工作模式（线程安全）
GimbalMode Gimbal::mode() const
{
  // 使用互斥锁保护模式变量的并发访问
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

// 获取当前云台状态信息（线程安全）
GimbalState Gimbal::state() const
{
  // 使用互斥锁保护状态变量的并发访问
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

// 将云台模式枚举转换为可读字符串
std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";        // 空闲模式
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";    // 自动瞄准模式
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";  // 小能量机关模式
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";    // 大能量机关模式
    default:
      return "INVALID";     // 无效模式
  }
}

// 根据时间点获取插值后的四元数姿态
Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  constexpr auto kWaitTimeout = std::chrono::milliseconds(100);

  while (!quit_) {
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      if (prev_sample_.has_value() && latest_sample_.has_value()) {
        const auto & [q_a, t_a] = *prev_sample_;
        const auto & [q_b, t_b] = *latest_sample_;
        if (t_a < t && t <= t_b) {
          const auto t_ab = tools::delta_time(t_a, t_b);
          if (std::abs(t_ab) > 1e-6) {
            const auto t_ac = tools::delta_time(t_a, t);
            const double k = std::clamp(t_ac / t_ab, 0.0, 1.0);
            last_returned_q_ = q_a.slerp(k, q_b).normalized();
            return last_returned_q_;
          }
          last_returned_q_ = q_b.normalized();
          return last_returned_q_;
        }

        if (t <= t_a) {
          last_returned_q_ = q_a.normalized();
          return last_returned_q_;
        }
      }

      if (latest_sample_.has_value()) {
        last_returned_q_ = std::get<0>(*latest_sample_).normalized();
      }
    }

    auto sample = queue_.pop_for(kWaitTimeout);
    if (!sample) {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      if (latest_sample_.has_value()) return last_returned_q_;
      continue;
    }

    std::lock_guard<std::mutex> lock(sample_mutex_);
    if (!latest_sample_.has_value()) {
      latest_sample_ = *sample;
      last_returned_q_ = std::get<0>(*sample).normalized();
      return last_returned_q_;
    }
    prev_sample_ = latest_sample_;
    latest_sample_ = *sample;
  }

  std::lock_guard<std::mutex> lock(sample_mutex_);
  return last_returned_q_;
}

// 向云台发送控制指令
void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  // 设置控制模式
  tx_data_.mode = VisionToGimbal.mode;
  // 设置偏航角目标值
  tx_data_.yaw = VisionToGimbal.yaw;
  // 设置偏航角速度目标值
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  // 设置偏航角加速度目标值
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  // 设置俯仰角目标值
  tx_data_.pitch = VisionToGimbal.pitch;
  // 设置俯仰角速度目标值
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  // 设置俯仰角加速度目标值
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  // 计算CRC16校验和（排除校验和字段本身）
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));
  try {
    // 通过串口发送控制数据
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    // 串口写入失败时记录警告信息（不退出程序）
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  // 设置控制模式：根据control和fire参数确定模式值
  // control为false -> 模式0（不控制）
  // control为true且fire为false -> 模式1（控制云台但不开火）
  // control为true且fire为true -> 模式2（控制云台且开火）
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  
  // 设置偏航角相关参数
  tx_data_.yaw = yaw;           // 目标偏航角（单位：弧度）
  tx_data_.yaw_vel = yaw_vel;   // 目标偏航角速度（单位：弧度/秒）
  tx_data_.yaw_acc = yaw_acc;   // 目标偏航角加速度（单位：弧度/秒²）
  
  // 设置俯仰角相关参数
  tx_data_.pitch = pitch;       // 目标俯仰角（单位：弧度）
  tx_data_.pitch_vel = pitch_vel; // 目标俯仰角速度（单位：弧度/秒）
  tx_data_.pitch_acc = pitch_acc; // 目标俯仰角加速度（单位：弧度/秒²）
  
  // 计算CRC16校验和，确保数据传输的完整性
  // 计算范围不包括crc16字段本身（sizeof(tx_data_) - sizeof(tx_data_.crc16)）
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  // uint8_t mode_value = tx_data_.mode;
  // tools::logger()->info("[Gimbal] 发送到下位机 :  yaw: {:.2f}°, pitch: {:.2f}°, mode: {}", 
  //   yaw, pitch, mode_value);

  // 尝试通过串口发送控制数据
  try {
    // 将tx_data_结构体转换为字节流并发送
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    // 捕获并记录串口写入失败的错误信息
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    if (!serial_.isOpen()) return false;
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

std::string Gimbal::resolve_desired_port() const
{
  if (tools::runtime_params::is_registered(config_path_)) {
    return tools::runtime_params::get_string(config_path_, "com_port");
  }

  const auto yaml = tools::load(config_path_);
  return tools::read<std::string>(yaml, "com_port");
}

bool Gimbal::open_serial(const std::string & port)
{
  try {
    serial::Timeout timeout = serial::Timeout::simpleTimeout(20);
    serial_.setPort(port);
    serial_.setBaudrate(921600);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial_.setTimeout(timeout);
    serial_.open();
    current_com_port_ = port;
    tools::logger()->info("[Gimbal] Serial opened on {}", current_com_port_);
    return true;
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to open serial {}: {}", port, e.what());
    return false;
  }
}

void Gimbal::close_serial()
{
  try {
    if (serial_.isOpen()) serial_.close();
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to close serial: {}", e.what());
  }
}

void Gimbal::reset_sample_cache()
{
  std::lock_guard<std::mutex> lock(sample_mutex_);
  prev_sample_.reset();
  latest_sample_.reset();
  last_returned_q_ = Eigen::Quaterniond::Identity();
  queue_.clear();
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  uint8_t sync_buffer[1];

  while (!quit_) {
    frame_start_time = std::chrono::steady_clock::now();

    if (!serial_.isOpen()) {
      reconnect();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    bool frame_synced = false;
    while (!frame_synced && !quit_) {
      if (!read(sync_buffer, 1)) {
        error_count++;
        if (!serial_.isOpen()) break;
        continue;
      }

      if (sync_buffer[0] == 0x5A) {
        frame_synced = true;
        rx_data_.head = 0x5A;
      } else {
        tools::logger()->debug("[Gimbal] Sync byte: 0x{:02X}", sync_buffer[0]);
      }
    }

    if (quit_) break;
    if (!serial_.isOpen()) continue;

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
      tools::logger()->warn("[Gimbal] CRC16 check failed. Frame discarded.");
      error_count++;
      continue;
    }

    error_count = 0;
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);
    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    frame_count++;
    auto current_time = std::chrono::steady_clock::now();
    auto time_diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_update).count();
    if (time_diff > 0) {
      fps = frame_count * 1000.0 / time_diff;
      std::fflush(stdout);
      frame_count = 0;
      last_fps_update = current_time;
    }

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;

  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    const auto desired_port = resolve_desired_port();
    if (desired_port != current_com_port_) {
      tools::logger()->warn(
        "[Gimbal] Detected com_port change: {} -> {}", current_com_port_, desired_port);
    }

    close_serial();
    reset_sample_cache();

    tools::logger()->warn(
      "[Gimbal] Reconnecting serial on {} (attempt {}/{})...",
      desired_port, i + 1, max_retry_count);

    if (open_serial(desired_port)) return;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

}  // namespace io
