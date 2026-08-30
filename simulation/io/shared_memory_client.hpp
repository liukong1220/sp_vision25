#ifndef SIMULATION_IO__SHARED_MEMORY_CLIENT_HPP
#define SIMULATION_IO__SHARED_MEMORY_CLIENT_HPP

// bevy_robomaster_simulator 本地共享内存通道的 C++ 消费端。
//
// 两个 mmap 区域（memmap2 把它们实现为 /tmp 下的普通文件，不是 POSIX shm 对象）：
//   /tmp/talos_ipc_meta        ShmMetaRegion，3712 字节
//   /tmp/talos_ipc_image_pool  3 * 1440*1080*3 字节的裸像素池
//
// 必须整帧消费。simulator 的 try_publish_synchronized_image 在
// 图像 + poses[0..=Camera] 全部清掉 FLAG_NEW 之前不会发布下一帧，
// 所以少消费任何一个通道都会让仿真端静默停止出图，表现为“卡住”而不是报错。
// consume_frame() 因此总是把这 5 个通道一起排空，即使中途发现数据不合约。

#include <cstddef>
#include <cstdint>
#include <string>

#include "shared_memory_layout.hpp"

namespace sim_io
{
enum class ConsumeStatus
{
  NotConnected,     // 尚未 open() 或已断开
  NoFrame,          // 没有新帧，正常空转
  Ok,               // 完整且同帧一致
  PoseMissing,      // 图像到了但某个必需 pose 通道没有新数据
  PoseSeqMismatch,  // pose 的 frame_seq 与图像不一致，该帧必须丢弃
  SeqRegressed,     // 帧号未前进（重复或倒退），通常意味着仿真端重启
  ImageInvalid,     // 分辨率/格式/buffer_id 超出协议约定
  Corrupted,        // 三缓冲索引越界，共享内存已不可信
};

const char * to_string(ConsumeStatus status);

struct FrameBundle
{
  std::uint64_t frame_seq = 0;
  std::uint64_t timestamp_ns = 0;  // 源端 wall clock，原样保留
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t image_format = 0;
  std::uint8_t buffer_id = 0;

  // 指向图像池中的槽位，零拷贝。有效期仅到下一次 consume_frame()：
  // 我们持有的 meta 槽位保护着这一帧，而 producer 每帧把 buffer_id 前移一格，
  // 且必须等我们排空 meta 才能再发一帧，所以它最多领先一帧、不会回头写到
  // 我们正在读的 buffer。调用方必须在本次调用返回后立刻拷走。
  const std::uint8_t * pixels = nullptr;
  std::size_t pixel_bytes = 0;

  PoseMeta poses[POSE_CHANNEL_COUNT]{};
  bool pose_present[POSE_CHANNEL_COUNT]{};

  const PoseMeta & gimbal() const { return poses[static_cast<int>(PoseIndex::Gimbal)]; }
  const PoseMeta & odom() const { return poses[static_cast<int>(PoseIndex::Odom)]; }
  const PoseMeta & muzzle() const { return poses[static_cast<int>(PoseIndex::Muzzle)]; }
  const PoseMeta & camera() const { return poses[static_cast<int>(PoseIndex::Camera)]; }
};

class SharedMemoryClient
{
public:
  struct Options
  {
    std::string dir = SHM_DIR;
    std::string meta_name = SHM_NAME_META;
    std::string image_pool_name = SHM_NAME_IMAGE_POOL;
  };

  SharedMemoryClient() = default;
  explicit SharedMemoryClient(Options options) : options_(std::move(options)) {}
  ~SharedMemoryClient();

  SharedMemoryClient(const SharedMemoryClient &) = delete;
  SharedMemoryClient & operator=(const SharedMemoryClient &) = delete;

  // 打开并校验两个区域：文件长度、magic、version、图像分辨率。
  // 失败时把原因写入 error 并保持未连接状态。
  bool open(std::string * error);
  void close();
  bool connected() const { return meta_ != nullptr; }

  // 整帧消费。返回 NoFrame 时 bundle 不被修改。
  ConsumeStatus consume_frame(FrameBundle * bundle);

  // 发布云台命令。distance_m = -1 是仿真端识别的“无控制”编码。
  bool publish_gimbal_cmd(const GimbalCmd & cmd);

  std::uint64_t heartbeat_ns() const;
  std::uint64_t created_ns() const;
  std::uint32_t magic() const;
  std::uint32_t version() const;

  const CameraInfo * camera_info() const;
  const RuntimeState * runtime_state() const;
  bool read_chassis_observation(ChassisObservation * out) const;

  // 仅供评估器使用，禁止进入算法输入。
  bool read_ground_truth(GroundTruthBatch * out) const;

  // 统计量，供最终报告使用。
  std::uint64_t consumed_frames() const { return consumed_frames_; }
  std::uint64_t dropped_frames() const { return dropped_frames_; }
  std::uint64_t skipped_frames() const { return skipped_frames_; }
  std::uint64_t regressed_frames() const { return regressed_frames_; }
  std::uint64_t corrupted_events() const { return corrupted_events_; }
  bool has_last_seq() const { return has_last_seq_; }
  // 观测到的仿真端重启次数（header.created_ns 变化）。
  std::uint64_t publisher_restarts() const { return publisher_restarts_; }
  std::uint64_t last_seq() const { return last_seq_; }

private:
  ShmMetaRegion * meta_ = nullptr;
  std::uint8_t * image_pool_ = nullptr;
  std::size_t meta_bytes_ = 0;
  std::size_t image_pool_bytes_ = 0;
  Options options_;

  std::uint64_t consumed_frames_ = 0;
  std::uint64_t dropped_frames_ = 0;
  std::uint64_t skipped_frames_ = 0;
  std::uint64_t regressed_frames_ = 0;
  std::uint64_t corrupted_events_ = 0;
  std::uint64_t last_seq_ = 0;
  bool has_last_seq_ = false;
  std::uint64_t publisher_created_ns_ = 0;
  std::uint64_t publisher_restarts_ = 0;

  // 无论图像是否合法都要排空 pose 通道，否则仿真端背压会锁死。
  ConsumeStatus drain_poses(FrameBundle * bundle, std::uint64_t image_seq);
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SHARED_MEMORY_CLIENT_HPP
