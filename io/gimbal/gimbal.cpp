#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
// 构造函数：从配置文件初始化云台串口连接并启动读取线程
Gimbal::Gimbal(const std::string & config_path)
{
  // 加载YAML配置文件
  auto yaml = tools::load(config_path);
  // 从配置文件中读取串口端口号
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    // 设置串口端口并打开连接
    serial_.setPort(com_port);
    serial_.open();
  } catch (const std::exception & e) {
    // 串口打开失败时记录错误并退出程序
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  // 启动数据读取线程，执行read_thread函数
  thread_ = std::thread(&Gimbal::read_thread, this);

  // 等待并弹出第一个四元数数据，确保连接正常
  queue_.pop();
  // 记录成功接收第一个四元数的日志
  tools::logger()->info("[Gimbal] First q received.");
}

// 析构函数：安全关闭云台连接和线程
Gimbal::~Gimbal()
{
  // 设置退出标志，通知线程停止
  quit_ = true;
  // 等待读取线程结束（如果可连接）
  if (thread_.joinable()) thread_.join();
  // 关闭串口连接
  serial_.close();
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
  while (true) {
    // 从队列中获取最早的四元数数据和时间点
    auto [q_a, t_a] = queue_.pop();
    // 查看队列前端的最新四元数数据和时间点
    auto [q_b, t_b] = queue_.front();
    // 计算两个数据点之间的时间间隔
    auto t_ab = tools::delta_time(t_a, t_b);
    // 计算目标时间与最早数据点的时间间隔
    auto t_ac = tools::delta_time(t_a, t);
    // 计算插值比例系数
    auto k = t_ac / t_ab;
    // 使用球面线性插值计算目标时刻的四元数，并进行标准化
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    // 如果目标时间早于最早数据点，直接返回插值结果
    if (t < t_a) return q_c;
    // 如果目标时间不在当前两个数据点之间，继续循环获取新数据
    if (!(t_a < t && t <= t_b)) continue;

    // 返回插值后的四元数
    return q_c;
  }
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
  tx_data_.yaw = yaw;           // 目标偏航角（单位：度）
  tx_data_.yaw_vel = yaw_vel;   // 目标偏航角速度（单位：度/秒）
  tx_data_.yaw_acc = yaw_acc;   // 目标偏航角加速度（单位：度/秒²）
  
  // 设置俯仰角相关参数
  tx_data_.pitch = pitch;       // 目标俯仰角（单位：度）
  tx_data_.pitch_vel = pitch_vel; // 目标俯仰角速度（单位：度/秒）
  tx_data_.pitch_acc = pitch_acc; // 目标俯仰角加速度（单位：度/秒²）
  
  // 计算CRC16校验和，确保数据传输的完整性
  // 计算范围不包括crc16字段本身（sizeof(tx_data_) - sizeof(tx_data_.crc16)）
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

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
    // 尝试从串口读取指定大小的数据到缓冲区
    // 如果实际读取的字节数等于请求的字节数，则返回true，否则返回false
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // 捕获串口读取过程中可能出现的异常
    // 记录警告日志，显示具体的错误信息
    tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    // 发生异常时返回false表示读取失败
    return false;
  }
}

void Gimbal::read_thread()
{
  // 记录读取线程启动日志
  tools::logger()->info("[Gimbal] read_thread started.");
  // 错误计数器，用于跟踪连续读取失败的次数
  int error_count = 0;
  // 缓冲区用于帧同步
  uint8_t sync_buffer[1];
  // 数据包大小检测标志
  bool packet_size_detected = false;
  size_t expected_packet_size = sizeof(rx_data_); // 初始期望大小

  // 主循环：持续读取云台数据直到收到退出信号
  while (!quit_) {
    // // 如果错误计数超过阈值（5000次），执行重连操作
    // if (error_count > 5000) {
    //   error_count = 0;
    //   tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
    //   // 调用重连函数尝试重新建立串口连接
    //   reconnect();
    //   continue;
    // }

    // 帧同步：逐个字节读取直到找到正确的帧头
    bool frame_synced = false;
    while (!frame_synced && !quit_) {
      // 读取单个字节进行帧同步
      if (!read(sync_buffer, 1)) {
        error_count++;
        continue;
      }

      // 检查是否为正确的帧头
      if (sync_buffer[0] == 0x5A) {
        frame_synced = true;
        rx_data_.head = 0x5A;  // 设置正确的帧头
      } else {
        // 记录无效帧头用于调试
        tools::logger()->debug("[Gimbal] Sync byte: 0x{:02X}", sync_buffer[0]);
      }
    }
    
    if (quit_) break;

    // 记录当前时间戳，用于后续数据同步
    auto t = std::chrono::steady_clock::now();

    // 如果还没有检测到数据包大小，先尝试读取完整数据包进行检测
    if (!packet_size_detected) {
      // 尝试读取较大的缓冲区来检测实际数据包大小
      uint8_t large_buffer[64]; // 足够大的缓冲区
      large_buffer[0] = 0x5A; // 设置帧头
      
      // 尝试读取剩余数据
      if (read(large_buffer + 1, 63)) {
        // 分析数据包大小：寻找可能的CRC校验码位置
        // 从你的cutecom数据看，数据包大小是48字节
        // 让我们检查48字节位置是否有合理的CRC值
        size_t test_size = 48;
        if (test_size <= 64) {
          // 检查CRC校验
          if (tools::check_crc16(large_buffer, test_size)) {
            expected_packet_size = test_size;
            packet_size_detected = true;
            tools::logger()->info("[Gimbal] Detected packet size: {} bytes", expected_packet_size);
            
            // 将检测到的数据复制到rx_data_
            if (expected_packet_size <= sizeof(rx_data_)) {
              memcpy(&rx_data_, large_buffer, expected_packet_size);
            } else {
              // 如果检测到的包比预期大，只复制预期大小的部分
              memcpy(&rx_data_, large_buffer, sizeof(rx_data_));
            }
          } else {
            tools::logger()->warn("[Gimbal] CRC check failed for size {}", test_size);
          }
        }
      }
      
      // 如果检测失败，继续使用默认大小
      if (!packet_size_detected) {
        tools::logger()->warn("[Gimbal] Packet size detection failed, using default size: {} bytes", expected_packet_size);
        // 读取剩余数据
        if (!read(
              reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
              expected_packet_size - sizeof(rx_data_.head))) {
          tools::logger()->warn("[Gimbal] Failed to read remaining {} bytes", expected_packet_size - sizeof(rx_data_.head));
          error_count++;
          continue;
        }
      }
    } else {
      // 已经检测到数据包大小，使用检测到的大小读取
      if (!read(
            reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
            expected_packet_size - sizeof(rx_data_.head))) {
        tools::logger()->warn("[Gimbal] Failed to read remaining {} bytes", expected_packet_size - sizeof(rx_data_.head));
        error_count++;
        continue;
      }
    }

    // 验证数据帧的CRC校验和是否正确
    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), expected_packet_size)) {
      tools::logger()->warn("[Gimbal] CRC16 check failed. Frame discarded.");
      error_count++;
      continue;
    }

    // 数据验证通过，重置错误计数器
    error_count = 0;
    
    // 从接收数据中提取四元数姿态信息
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    // 将四元数和时间戳推入队列，供其他线程使用
    queue_.push({q, t});

    // 获取互斥锁以安全地更新云台状态
    std::lock_guard<std::mutex> lock(mutex_);

    // 更新云台状态信息
    state_.yaw = rx_data_.yaw;           // 偏航角
    state_.yaw_vel = rx_data_.yaw_vel;   // 偏航角速度
    state_.pitch = rx_data_.pitch;       // 俯仰角
    state_.pitch_vel = rx_data_.pitch_vel; // 俯仰角速度
    state_.bullet_speed = rx_data_.bullet_speed; // 子弹速度
    state_.bullet_count = rx_data_.bullet_count; // 子弹计数

    // 使用临时变量输出
    tools::logger()->info("yaw: {:.2f}°, pitch: {:.2f}°\n yaw_vel: {:.2f}°m/s pitch_vel: {:.2f}°m/s\n bullet_speed: {:.2f}m/s bullet_count: {}", 
      state_.yaw, state_.pitch, state_.yaw_vel, state_.pitch_vel, state_.bullet_speed, state_.bullet_count);
      
    // 根据接收到的模式值设置云台工作模式
    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;        // 空闲模式
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;    // 自动瞄准模式
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;  // 小能量机关模式
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;    // 大能量机关模式
        break;
      default:
        mode_ = GimbalMode::IDLE;        // 默认设为空闲模式
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  // 记录读取线程停止日志
  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  // 设置最大重连尝试次数
  int max_retry_count = 10;
  
  // 循环尝试重连，直到达到最大尝试次数或收到退出信号
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    // 记录当前重连尝试的日志信息
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    
    // 尝试关闭串口连接
    try {
      serial_.close();
      // 等待1秒，给硬件设备足够的时间重置
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
      // 忽略关闭串口时可能出现的任何异常
    }

    // 尝试重新打开串口连接
    try {
      serial_.open();  // 尝试重新打开串口
      // 清空数据队列，避免使用旧的缓存数据
      queue_.clear();
      // 记录成功重连的日志信息
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      // 重连成功，跳出循环
      break;
    } catch (const std::exception & e) {
      // 捕获并记录重连失败的具体错误信息
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      // 等待1秒后继续下一次重连尝试
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io