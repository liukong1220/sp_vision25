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

#include <chrono>
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
  EpochChanged,     // header.created_ns 变化：发布端换代，本帧必须丢弃并复位上层状态
  Remapped,         // 共享内存文件 inode 变化并已重新 mmap，本帧必须丢弃
};

const char * to_string(ConsumeStatus status);

// 把能力位掩码写成人能读的名字，例如 "ground_truth|muzzle_world_pose"。
// 掩码为 0 时返回 "none"，出现未知位时附上 "0x.." 十六进制残余。
std::string describe_capabilities(std::uint32_t caps);

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

// 一帧的两个时间点。二者必须分开保存：把它们混成一个"时间戳"是
// FAULT_STATE_STALE 恒亮的根因（见 SimGimbal::update 的说明）。
struct FrameStamps
{
  // 源端采样时刻，映射到本地 steady clock（clock_.to_steady(bundle.timestamp_ns)）。
  // 语义是"这一帧描述的世界是多久以前的"。受源端时钟跳变影响。
  std::chrono::steady_clock::time_point source{};
  // 本帧在**本进程**被取到的 steady 时刻。语义是"我们手上这份状态放了多久"。
  // 不受源端时钟影响，看门狗只能用它。
  std::chrono::steady_clock::time_point arrival{};
};

class SharedMemoryClient
{
public:
  struct Options
  {
    std::string dir = SHM_DIR;
    std::string meta_name = SHM_NAME_META;
    std::string image_pool_name = SHM_NAME_IMAGE_POOL;

    // open() 时必须由发布端声明、缺一位就拒绝连接的能力位。
    //
    // 默认只含 CAP_MUZZLE_WORLD_POSE，因为它是唯一"缺位时零值会被当成合法 v3
    // 世界量继续用下去"的通道：PoseIndex::Muzzle 在 v2 里是相对云台的局部平移、
    // v3 起是枪口的**世界**位置，两者都是三个 float，没有任何形状差别可供识别。
    // 一个只填 v2 语义（或干脆不填）的发布端会让消费端把 (0,0,0) 当作"枪口就在
    // 世界原点"，muzzle_position() 一路进到弹道与误差分解里，算出来的每一个角度
    // 都是错的，而且没有任何一项统计会变成 0 去提示这件事。所以这一位只能在
    // 连接时拒绝，不能降级。
    //
    // 其余能力位是"可选区域"：缺位时对应的 read_* 返回 false 并累加各自的
    // unsupported 计数，绝不交出全零数据冒充真实观测。需要更严的调用方可以把它们
    // 也加进来（例如 closed_loop 要求 CAP_RUNTIME_STATE）。
    std::uint32_t required_capabilities = CAP_MUZZLE_WORLD_POSE;
  };

  SharedMemoryClient() = default;
  explicit SharedMemoryClient(Options options) : options_(std::move(options)) {}
  ~SharedMemoryClient();

  SharedMemoryClient(const SharedMemoryClient &) = delete;
  SharedMemoryClient & operator=(const SharedMemoryClient &) = delete;

  // 打开并校验两个区域：文件长度、magic、version、图像分辨率。
  //
  // **事务化**：先把两个区域映射到临时变量并跑完全部校验，只有全部通过才 close()
  // 旧映射并换上新映射。任何一步失败都只回收新映射，已有连接保持原样，error 写入
  // 原因。
  //
  // 这一点是必须的，而不是"更干净"：原来第一句是 close()，于是"meta 已建好、pool
  // 还没 ftruncate"这个必然存在的窗口里一次失败，就会把仍然可用的旧映射拆掉。之后
  // connected() 恒为 false，而所有重试入口（SimCamera::try_read 的第一行、
  // paths_changed()）都以 connected() 为前提，重试路径永久不可达——一次瞬时失败变成
  // 永久失联，只能重启进程。
  bool open(std::string * error);
  void close();
  bool connected() const { return meta_ != nullptr; }

  // 发布端**正常退出**会 unlink 掉 /tmp 下的两个文件（ShmRegion::Drop ->
  // remove_file），重新拉起时 create() 用 O_TRUNC 建的是**新 inode**。此时本进程
  // 手里的旧映射既看不到新帧，也看不到新的 created_ns——created_ns 换代检测只能
  // 覆盖 SIGKILL（同 inode 被复用）那一路。
  //
  // 所以这里记下 open() 时的 (st_dev, st_ino)，由 consume_frame() 在长时间没有
  // 新帧时比对；真变了就 remap()：重新 mmap 并清空帧号水位线与换代基准，让下一帧
  // 走全新的 Remapped/EpochChanged 路径。
  bool paths_changed() const;
  // 重新映射。成功时返回 true 并把 remaps() 加一；失败时**保持原有映射不变**并把
  // remap_failures() 加一，调用方可以直接重试。
  bool remap(std::string * error);
  std::uint64_t remaps() const { return remaps_; }
  std::uint64_t remap_failures() const { return remap_failures_; }

  // 整帧消费。返回 NoFrame 时 bundle 不被修改。
  ConsumeStatus consume_frame(FrameBundle * bundle);

  // 发布云台命令。distance_m = -1 是仿真端识别的“无控制”编码。
  bool publish_gimbal_cmd(const GimbalCmd & cmd);

  std::uint64_t heartbeat_ns() const;
  std::uint64_t created_ns() const;
  std::uint32_t magic() const;
  std::uint32_t version() const;

  // 发布端声明的能力位（CAP_*）。未连接时返回 0。
  std::uint32_t capabilities() const;
  bool has_capability(std::uint32_t cap) const { return (capabilities() & cap) == cap; }

  const CameraInfo * camera_info() const;

  // 发布端未声明 CAP_RUNTIME_STATE 时返回 **nullptr** 并累加
  // runtime_state_unsupported()，绝不返回一块恒零的 RuntimeState。
  //
  // 返回零值是有害的：following=0 会被读成"仿真端没订阅云台命令"，于是排查方向
  // 被引到"去按 F5 / 设 DAEDALUS_FORCE_AUTO_AIM"上，而真实原因是发布端根本不报
  // 这个字段。调用方必须把 nullptr 当作"不可知"，而不是"没订阅"。
  const RuntimeState * runtime_state() const;

  // 发布端未声明 CAP_CHASSIS_OBSERVATION 时返回 false 并累加
  // chassis_observation_unsupported()。全零的 ChassisObservation 是一份合法读数
  // （车停着），所以缺位必须靠能力位识别，不能靠"读出来是零"。
  bool read_chassis_observation(ChassisObservation * out) const;

  // 仅供评估器使用，禁止进入算法输入。
  //
  // 发布端没有声明 CAP_GROUND_TRUTH 时直接返回 false 并累加
  // ground_truth_unsupported()，而不是把一块合法的全零数据当真值交出去。
  bool read_ground_truth(GroundTruthBatch * out) const;
  std::uint64_t ground_truth_unsupported() const { return ground_truth_unsupported_; }

  // 与最近一次 consume_frame() 处理的那一帧**同一次发布事务**的真值批次。
  //
  // consume_frame() 会在"图像 FLAG_NEW 已清、pose 通道还没排空"这个窗口里把真值
  // 拷下来（见其实现里的说明）。这一段时间发布端一定还堵在同一帧上，所以拷到的
  // 就是该帧的批次。调用方**必须**用这个入口，不要在流水线后段再去
  // read_ground_truth()：检测一帧 ~250 ms，那时背压早已放开，槽位可能已经被后面
  // 若干帧覆盖。
  //
  // 帧号一致性仍然由上层（GroundTruthEvaluator::fetch）判定，好让 seq_mismatches
  // 与 seq_skew 只在一处计数；这里只负责"在正确的时刻取到一份完整的批次"。
  // 返回 false = 本帧没能拷到（发布端未声明 CAP_GROUND_TRUTH，或 seqlock 连续
  // 8 次都撞上写入）。
  bool frame_ground_truth(GroundTruthBatch * out) const;
  // 成功在事务窗口里拷到批次的帧数。与 consumed_frames() 之差就是"取不到真值"的帧数。
  std::uint64_t ground_truth_captures() const { return ground_truth_captures_; }
  std::uint64_t runtime_state_unsupported() const { return runtime_state_unsupported_; }
  std::uint64_t chassis_observation_unsupported() const
  {
    return chassis_observation_unsupported_;
  }

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
  bool stat_ids(const std::string & path, std::uint64_t * dev, std::uint64_t * ino) const;

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
  std::uint64_t remaps_ = 0;
  std::uint64_t remap_failures_ = 0;
  mutable std::uint64_t ground_truth_unsupported_ = 0;
  GroundTruthBatch frame_ground_truth_{};
  bool has_frame_ground_truth_ = false;
  std::uint64_t ground_truth_captures_ = 0;
  mutable std::uint64_t runtime_state_unsupported_ = 0;
  mutable std::uint64_t chassis_observation_unsupported_ = 0;
  // open()/remap() 时记录的文件身份，用于发现"文件被删掉又重建"。
  std::uint64_t meta_dev_ = 0;
  std::uint64_t meta_ino_ = 0;
  std::uint64_t pool_dev_ = 0;
  std::uint64_t pool_ino_ = 0;

  // 无论图像是否合法都要排空 pose 通道，否则仿真端背压会锁死。
  ConsumeStatus drain_poses(FrameBundle * bundle, std::uint64_t image_seq);
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SHARED_MEMORY_CLIENT_HPP
