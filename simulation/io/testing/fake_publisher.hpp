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
    destroy();
    if (!make_region(meta_path(), sizeof(ShmMetaRegion), &meta_addr_, error)) return false;
    if (!make_region(image_pool_path(), IMAGE_POOL_SIZE, &pool_addr_, error)) {
      ::munmap(meta_addr_, sizeof(ShmMetaRegion));
      meta_addr_ = nullptr;
      return false;
    }

    meta_ = static_cast<ShmMetaRegion *>(meta_addr_);
    pool_ = static_cast<std::uint8_t *>(pool_addr_);

    meta_->header.magic = SHM_MAGIC;
    meta_->header.version = SHM_VERSION;
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
    return true;
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
  void set_ground_truth(const GroundTruthBatch & batch) { meta_->ground_truth = batch; }

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

  // 与 capture.rs 的 publish_pose_data 一致：Odom 带位置+单位四元数，
  // Gimbal 位置为零、带真实四元数 [w,x,y,z]，Muzzle/Camera 为相对平移。
  void publish_pose_bundle(
    std::uint64_t frame_seq, std::uint64_t timestamp_ns, const float gimbal_quat_wxyz[4],
    const float odom_position[3] = nullptr, const float muzzle_rel[3] = nullptr,
    const float camera_rel[3] = nullptr)
  {
    static const float zero3[3] = {0.0f, 0.0f, 0.0f};
    static const float identity4[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    publish_pose(
      PoseIndex::Odom, odom_position != nullptr ? odom_position : zero3, identity4, frame_seq,
      timestamp_ns);
    publish_pose(PoseIndex::Gimbal, zero3, gimbal_quat_wxyz, frame_seq, timestamp_ns);
    publish_pose(
      PoseIndex::Muzzle, muzzle_rel != nullptr ? muzzle_rel : zero3, identity4, frame_seq,
      timestamp_ns);
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

  // 完整复刻 try_publish_synchronized_image：先判据，再发 pose，最后提交图像 meta。
  bool try_publish_synchronized_frame(
    const std::uint8_t * rgb, std::uint64_t seq, std::uint64_t timestamp_ns,
    const float gimbal_quat_wxyz[4], const float odom_position[3] = nullptr,
    const float muzzle_rel[3] = nullptr, const float camera_rel[3] = nullptr)
  {
    if (has_last_sync_seq_ && seq <= last_sync_seq_) return false;
    if (!synchronized_frame_consumed()) return false;

    publish_pose_bundle(seq, timestamp_ns, gimbal_quat_wxyz, odom_position, muzzle_rel, camera_rel);
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
  bool has_last_sync_seq_ = false;

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
