#ifndef SIMULATION_IO__TESTING__FAKE_PUBLISHER_HPP
#define SIMULATION_IO__TESTING__FAKE_PUBLISHER_HPP

// 仅供测试使用：用 C++ 复刻 Rust ShmPublisher 的生产者行为，
// 使 loopback 与故障注入测试不必启动整个 Bevy 仿真器。
//
// 这里刻意逐条对齐 crates/talos-ipc/src/publisher.rs：
//   - create() 零填充后必须把三缓冲重置为 state=1 / write_idx=0 / read_idx=2
//   - poses 在图像 meta 提交之前发布（图像 meta 是提交标记）
//   - try_publish_synchronized_image 的两个前置条件：帧号前进 + 上一帧已被排空
// 任何一条走偏，测试就会验证一个并不存在的协议。

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "simulation/io/shared_memory_layout.hpp"

namespace sim_io::testing
{
class FakePublisher
{
public:
  struct Options
  {
    std::string dir = SHM_DIR;
    std::string meta_name = SHM_NAME_META;
    std::string image_pool_name = SHM_NAME_IMAGE_POOL;
    // 协议版本与能力位都可覆盖，用来伪造"版本对得上但不写真值区"的旧发布端。
    std::uint32_t version = SHM_VERSION;
    std::uint32_t capabilities = SIMULATOR_CAPABILITIES;
  };

  FakePublisher() = default;
  explicit FakePublisher(Options options) : options_(std::move(options)) {}
  ~FakePublisher() { destroy(); }

  FakePublisher(const FakePublisher &) = delete;
  FakePublisher & operator=(const FakePublisher &) = delete;

  static std::uint64_t now_ns()
  {
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count());
  }

  std::string meta_path() const { return options_.dir + "/" + options_.meta_name; }
  std::string image_pool_path() const { return options_.dir + "/" + options_.image_pool_name; }

  bool create(std::string * error = nullptr)
  {
    if (!create_meta_only(error)) return false;
    if (!create_image_pool(error)) {
      destroy();
      return false;
    }
    return true;
  }

  // 只创建 meta 区，**不**创建图像池。
  //
  // 真实发布端的两个文件不可能同时出现（ShmPublisher::create 是分步的 open+
  // ftruncate+mmap），消费端必然会撞上"meta 已在、pool 还没有"这个中间态。
  // 分步接口让这个中间态可以被测试直接构造出来。
  bool create_meta_only(std::string * error = nullptr)
  {
    destroy();
    if (!make_region(meta_path(), sizeof(ShmMetaRegion), &meta_addr_, error)) return false;
    meta_ = static_cast<ShmMetaRegion *>(meta_addr_);

    meta_->header.magic = SHM_MAGIC;
    meta_->header.version = options_.version;
    meta_->header.capabilities = options_.capabilities;
    meta_->header.created_ns = now_ns();
    meta_->header.heartbeat_ns = now_ns();
    meta_->header.image_width = IMAGE_WIDTH;
    meta_->header.image_height = IMAGE_HEIGHT;

    // 零填充是非法初始状态，必须显式重置。
    triple_buffer_init(meta_->image);
    for (std::size_t i = 0; i < POSE_CHANNEL_COUNT; ++i) triple_buffer_init(meta_->poses[i]);
    triple_buffer_init(meta_->gimbal_cmd);

    current_buffer_id_ = 0;
    has_last_sync_seq_ = false;
    last_sync_seq_ = 0;
    gt_seq_ = 0;
    return true;
  }

  // 补上图像池。已经建好时是幂等的。
  bool create_image_pool(std::string * error = nullptr)
  {
    if (pool_addr_ != nullptr) return true;
    if (!make_region(image_pool_path(), IMAGE_POOL_SIZE, &pool_addr_, error)) return false;
    pool_ = static_cast<std::uint8_t *>(pool_addr_);
    return true;
  }

  void set_capabilities(std::uint32_t caps)
  {
    options_.capabilities = caps;
    if (meta_ != nullptr) meta_->header.capabilities = caps;
  }

  void destroy()
  {
    if (meta_addr_ != nullptr) {
      ::munmap(meta_addr_, sizeof(ShmMetaRegion));
      meta_addr_ = nullptr;
      meta_ = nullptr;
    }
    if (pool_addr_ != nullptr) {
      ::munmap(pool_addr_, IMAGE_POOL_SIZE);
      pool_addr_ = nullptr;
      pool_ = nullptr;
    }
  }

  void unlink_files() const
  {
    ::unlink(meta_path().c_str());
    ::unlink(image_pool_path().c_str());
  }

  bool created() const { return meta_ != nullptr; }
  ShmMetaRegion * meta() { return meta_; }
  std::uint8_t * pool() { return pool_; }

  void update_heartbeat()
  {
    __atomic_store_n(&meta_->header.heartbeat_ns, now_ns(), __ATOMIC_RELEASE);
  }

  void set_heartbeat_ns(std::uint64_t value)
  {
    __atomic_store_n(&meta_->header.heartbeat_ns, value, __ATOMIC_RELEASE);
  }

  void set_camera_info(const CameraInfo & info) { meta_->camera_info = info; }
  void set_runtime_state(const RuntimeState & state) { meta_->runtime_state = state; }
  void set_chassis_observation(const ChassisObservation & obs)
  {
    meta_->chassis_observation = obs;
  }
  // 与 Rust 侧 TalosPublisher::publish_ground_truth 相同的 seqlock 提交序：
  // 写前置奇、写后置偶，两端各一道 release 栅栏。消费端读到奇数或前后不等就重试。
  // marker 只能被原子访问。载荷 memcpy 必须恰好停在 seqlock 之前
  // （GROUND_TRUTH_PAYLOAD_BYTES），不能整结构体拷过去：
  //   * sizeof(GroundTruthBatch) 覆盖 seqlock 本身，等于用一次非原子写覆盖 marker，
  //     把刚刚置好的奇数序号擦成 batch 里带的值。若 batch.seqlock 恰好是偶数，
  //     读端就会看到"偶数 marker + 正在被写的载荷 + 同一个偶数 marker"，
  //     前后相等的判据直接通过，撕裂数据被当成完好数据交出去。
  //   * 之前靠 `staged.seqlock = begin` 绕开，那是让 marker 的正确性依赖于每个调用
  //     点都记得改一份副本；写法本身仍是非原子的，且 TSan 会如实报 race。
  void set_ground_truth(const GroundTruthBatch & batch)
  {
    const std::uint32_t begin = (gt_seq_ + 1) | 1u;
    store_seq(begin);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    std::memcpy(&meta_->ground_truth, &batch, GROUND_TRUTH_PAYLOAD_BYTES);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    gt_seq_ = begin + 1;
    store_seq(gt_seq_);
  }

  // 故意把真值区停在"写一半"的状态：置奇序号并写入 batch，但**不**收尾置偶。
  // 用来验证消费端的 seqlock 真的会拒绝撕裂数据，而不是靠"memcpy 前后 frame_seq
  // 相等"这种近似判断——同一帧号内重发时 frame_seq 不变、body 却在改，那种判据
  // 会把撕裂当成完好。调用之后必须再调一次 set_ground_truth 才能恢复可读。
  void begin_torn_ground_truth(const GroundTruthBatch & batch)
  {
    const std::uint32_t begin = (gt_seq_ + 1) | 1u;
    store_seq(begin);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    std::memcpy(&meta_->ground_truth, &batch, GROUND_TRUTH_PAYLOAD_BYTES);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    gt_seq_ = begin;  // 停在奇数：区域标记为"正在写"
  }

  std::uint32_t ground_truth_seq() const
  {
    return __atomic_load_n(&meta_->ground_truth.seqlock, __ATOMIC_ACQUIRE);
  }

  void publish_pose(
    PoseIndex index, const float position[3], const float quaternion[4], std::uint64_t frame_seq,
    std::uint64_t timestamp_ns)
  {
    PoseMeta pose{};
    pose.frame_seq = frame_seq;
    pose.timestamp_ns = timestamp_ns;
    for (int i = 0; i < 3; ++i) pose.position[i] = position[i];
    for (int i = 0; i < 4; ++i) pose.quaternion[i] = quaternion[i];
    triple_buffer_publish(meta_->poses[static_cast<std::size_t>(index)], pose);
  }

  // 与 capture.rs 的 publish_pose_data 一致（协议 v3）：
  //   Odom   世界位置 + 单位四元数
  //   Gimbal 位置为零占位 + 枪口世界姿态 [w,x,y,z]
  //   Muzzle **世界**位置 + 与 Gimbal 相同的世界姿态
  //   Camera 相对云台的局部平移 + 单位四元数（与 t_camera2gimbal 外参对比用）
  // Muzzle 之所以发世界量，见 PoseIndex 的契约注释：局部平移会被调用方直接加到
  // 世界坐标上，那是错的，而错法在编译期完全看不出来。
  void publish_pose_bundle(
    std::uint64_t frame_seq, std::uint64_t timestamp_ns, const float gimbal_quat_wxyz[4],
    const float odom_position[3] = nullptr, const float muzzle_world[3] = nullptr,
    const float camera_rel[3] = nullptr)
  {
    static const float zero3[3] = {0.0f, 0.0f, 0.0f};
    static const float identity4[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    publish_pose(
      PoseIndex::Odom, odom_position != nullptr ? odom_position : zero3, identity4, frame_seq,
      timestamp_ns);
    publish_pose(PoseIndex::Gimbal, zero3, gimbal_quat_wxyz, frame_seq, timestamp_ns);
    publish_pose(
      PoseIndex::Muzzle, muzzle_world != nullptr ? muzzle_world : zero3, gimbal_quat_wxyz,
      frame_seq, timestamp_ns);
    publish_pose(
      PoseIndex::Camera, camera_rel != nullptr ? camera_rel : zero3, identity4, frame_seq,
      timestamp_ns);
    publish_pose(PoseIndex::ChassisObservation, zero3, identity4, frame_seq, timestamp_ns);
  }

  // 无条件发布图像（对应 publish_image_with），供“覆盖未消费帧”这类故障注入使用。
  void publish_image(
    const std::uint8_t * rgb, std::uint64_t seq, std::uint64_t timestamp_ns,
    std::uint32_t width = IMAGE_WIDTH, std::uint32_t height = IMAGE_HEIGHT,
    std::uint8_t format = IMAGE_FORMAT_RGB8)
  {
    const std::uint8_t buffer_id = current_buffer_id_;
    current_buffer_id_ = static_cast<std::uint8_t>((current_buffer_id_ + 1) % IMAGE_SLOT_COUNT);

    if (rgb != nullptr) {
      std::memcpy(pool_ + static_cast<std::size_t>(buffer_id) * IMAGE_SIZE, rgb, IMAGE_SIZE);
    }

    ImageMeta meta{};
    meta.seq = seq;
    meta.timestamp_ns = timestamp_ns;
    meta.width = width;
    meta.height = height;
    meta.buffer_id = buffer_id;
    meta.format = format;
    triple_buffer_publish(meta_->image, meta);
  }

  bool synchronized_frame_consumed() const
  {
    if ((triple_buffer_load_state(meta_->image) & FLAG_NEW) != 0) return false;
    // 只有 0..=Camera 参与握手；slot 4 是遗留通道，刻意不参与。
    for (std::size_t i = 0; i <= static_cast<std::size_t>(PoseIndex::Camera); ++i) {
      if ((triple_buffer_load_state(meta_->poses[i]) & FLAG_NEW) != 0) return false;
    }
    return true;
  }

  // 完整复刻 try_publish_synchronized_image：先判据，再发 pose 与真值，
  // **最后**提交图像 meta。
  //
  // 真值必须在图像 meta 之前写入：图像 meta 是消费端唯一的提交标记，回调里发布的
  // 一切都严格先于它可见。这是"图像 seq=k 可见 ⇒ 真值槽位里已经是 seq=k"这条协议
  // 保证的全部依据，仿真端 capture.rs 的 before_commit 回调做的就是这件事。
  // ground_truth == nullptr 时按"本帧没有真值"处理（发布端未订阅真值通道）。
  bool try_publish_synchronized_frame(
    const std::uint8_t * rgb, std::uint64_t seq, std::uint64_t timestamp_ns,
    const float gimbal_quat_wxyz[4], const float odom_position[3] = nullptr,
    const float muzzle_world[3] = nullptr, const float camera_rel[3] = nullptr,
    const GroundTruthBatch * ground_truth = nullptr)
  {
    if (has_last_sync_seq_ && seq <= last_sync_seq_) return false;
    if (!synchronized_frame_consumed()) return false;

    publish_pose_bundle(
      seq, timestamp_ns, gimbal_quat_wxyz, odom_position, muzzle_world, camera_rel);
    if (ground_truth != nullptr) set_ground_truth(*ground_truth);
    publish_image(rgb, seq, timestamp_ns);

    last_sync_seq_ = seq;
    has_last_sync_seq_ = true;
    return true;
  }

  // 模拟仿真端 process_subscription 的读取侧。
  bool recv_gimbal_cmd(GimbalCmd * out)
  {
    bool corrupted = false;
    const GimbalCmd * cmd = triple_buffer_consume(meta_->gimbal_cmd, &corrupted);
    if (corrupted || cmd == nullptr) return false;
    *out = *cmd;
    return true;
  }

  bool has_gimbal_cmd() const
  {
    return (triple_buffer_load_state(meta_->gimbal_cmd) & FLAG_NEW) != 0;
  }

private:
  Options options_;
  void * meta_addr_ = nullptr;
  void * pool_addr_ = nullptr;
  ShmMetaRegion * meta_ = nullptr;
  std::uint8_t * pool_ = nullptr;
  std::uint8_t current_buffer_id_ = 0;
  std::uint64_t last_sync_seq_ = 0;
  std::uint32_t gt_seq_ = 0;
  bool has_last_sync_seq_ = false;

  void store_seq(std::uint32_t value)
  {
    __atomic_store_n(&meta_->ground_truth.seqlock, value, __ATOMIC_RELEASE);
  }

  static bool make_region(
    const std::string & path, std::size_t bytes, void ** addr, std::string * error)
  {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
      if (error) *error = "无法创建 " + path + ": " + std::strerror(errno);
      return false;
    }
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      if (error) *error = "无法 ftruncate " + path + ": " + std::strerror(errno);
      ::close(fd);
      return false;
    }
    void * mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
      if (error) *error = "无法 mmap " + path + ": " + std::strerror(errno);
      return false;
    }
    std::memset(mapped, 0, bytes);  // 对齐 Rust 侧的零填充语义
    *addr = mapped;
    return true;
  }
};

}  // namespace sim_io::testing

#endif  // SIMULATION_IO__TESTING__FAKE_PUBLISHER_HPP
