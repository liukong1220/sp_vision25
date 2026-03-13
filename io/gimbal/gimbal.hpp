#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
// 云台到视觉系统的通信数据结构
// 使用packed属性确保结构体在内存中紧密排列，避免字节对齐问题
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head = 0x5A;  // 帧头标识，固定为5A
  uint8_t mode;  // 云台工作模式：0-空闲, 1-自瞄, 2-小符, 3-大符
  float q[4];    // 四元数姿态数据，wxyz顺序
  float yaw;     // 偏航角（单位：弧度）
  float yaw_vel; // 偏航角速度（单位：弧度/秒）
  float pitch;   // 俯仰角（单位：弧度）
  float pitch_vel; // 俯仰角速度（单位：弧度/秒）
  float bullet_speed; // 子弹速度（单位：m/s）
  uint16_t bullet_count;  // 子弹累计发送次数
  uint16_t crc16; // CRC16校验和，用于数据完整性验证
};

// 编译时断言，确保数据结构大小不超过64字节
static_assert(sizeof(GimbalToVision) <= 64);

// 视觉系统到云台的通信数据结构
struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head = 0xA5; // 帧头标识，固定为A5
  uint8_t mode;  // 控制模式：0-不控制, 1-控制云台但不开火，2-控制云台且开火
  float yaw;     // 目标偏航角（单位：弧度）
  float yaw_vel; // 目标偏航角速度（单位：弧度/秒）
  float yaw_acc; // 目标偏航角加速度（单位：弧度/秒²）
  float pitch;   // 目标俯仰角（单位：弧度）
  float pitch_vel; // 目标俯仰角速度（单位：弧度/秒）
  float pitch_acc; // 目标俯仰角加速度（单位：弧度/秒²）
  uint16_t crc16; // CRC16校验和
};

// 编译时断言，确保数据结构大小不超过64字节
static_assert(sizeof(VisionToGimbal) <= 64);

// 云台工作模式枚举定义
enum class GimbalMode
{
  IDLE,        // 空闲模式：云台不进行任何控制
  AUTO_AIM,    // 自动瞄准模式：用于自动跟踪目标
  SMALL_BUFF,  // 小能量机关模式：用于打击小能量机关
  BIG_BUFF     // 大能量机关模式：用于打击大能量机关
};

// 云台状态数据结构
struct GimbalState
{
  float yaw;           // 当前偏航角
  float yaw_vel;       // 当前偏航角速度
  float pitch;         // 当前俯仰角
  float pitch_vel;     // 当前俯仰角速度
  float bullet_speed;  // 当前子弹速度
  uint16_t bullet_count; // 当前子弹计数
};

// 云台控制类，负责与云台硬件进行通信和控制
class Gimbal
{
public:
  // 构造函数：通过配置文件路径初始化云台
  Gimbal(const std::string & config_path);

  // 析构函数：清理资源，停止线程
  ~Gimbal();

  // 获取当前云台工作模式
  GimbalMode mode() const;
  
  // 获取当前云台状态信息
  GimbalState state() const;
  
  // 将云台模式枚举转换为字符串表示
  std::string str(GimbalMode mode) const;
  
  // 根据时间戳获取插值后的四元数姿态
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  // 发送控制指令到云台（参数化版本）
  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  // 发送控制指令到云台（结构体版本）
  void send(io::VisionToGimbal VisionToGimbal);

private:
  serial::Serial serial_;  // 串口通信对象

  std::thread thread_;     // 数据读取线程
  std::atomic<bool> quit_ = false; // 线程退出标志
  mutable std::mutex mutex_; // 线程安全互斥锁

  GimbalToVision rx_data_;  // 接收数据缓冲区
  VisionToGimbal tx_data_;  // 发送数据缓冲区

  GimbalMode mode_ = GimbalMode::IDLE; // 当前云台模式
  GimbalState state_;                  // 当前云台状态
  
  // 线程安全队列，存储四元数和时间戳对，容量为1000
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  // 从串口读取指定大小的数据到缓冲区
  bool read(uint8_t * buffer, size_t size);
  
  // 数据读取线程函数
  void read_thread();
  
  // 串口重连函数
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP