#include "shared_memory_client.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace sim_io
{
const char * to_string(ConsumeStatus status)
{
  switch (status) {
    case ConsumeStatus::NotConnected:
      return "NotConnected";
    case ConsumeStatus::NoFrame:
      return "NoFrame";
    case ConsumeStatus::Ok:
      return "Ok";
    case ConsumeStatus::PoseMissing:
      return "PoseMissing";
    case ConsumeStatus::PoseSeqMismatch:
      return "PoseSeqMismatch";
    case ConsumeStatus::SeqRegressed:
      return "SeqRegressed";
    case ConsumeStatus::ImageInvalid:
      return "ImageInvalid";
    case ConsumeStatus::Corrupted:
      return "Corrupted";
    case ConsumeStatus::EpochChanged:
      return "EpochChanged";
    case ConsumeStatus::Remapped:
      return "Remapped";
  }
  return "Unknown";
}

namespace
{
struct MapResult
{
  void * addr = nullptr;
  std::size_t bytes = 0;
  std::uint64_t dev = 0;
  std::uint64_t ino = 0;
};

bool map_region(
  const std::string & path, std::size_t required_bytes, MapResult * out, std::string * error)
{
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    if (error) *error = "无法打开 " + path + ": " + std::strerror(errno);
    return false;
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    if (error) *error = "无法 fstat " + path + ": " + std::strerror(errno);
    ::close(fd);
    return false;
  }

  const std::size_t actual = static_cast<std::size_t>(st.st_size);
  if (actual < required_bytes) {
    if (error) {
      *error = path + " 太小: 需要 " + std::to_string(required_bytes) + " 字节, 实际 " +
               std::to_string(actual) + " 字节 (仿真端版本不匹配?)";
    }
    ::close(fd);
    return false;
  }

  void * addr = ::mmap(nullptr, required_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  // mmap 建立映射后 fd 即可关闭，映射本身持有引用。
  ::close(fd);

  if (addr == MAP_FAILED) {
    if (error) *error = "无法 mmap " + path + ": " + std::strerror(errno);
    return false;
  }

  out->addr = addr;
  out->bytes = required_bytes;
  out->dev = static_cast<std::uint64_t>(st.st_dev);
  out->ino = static_cast<std::uint64_t>(st.st_ino);
  return true;
}
}  // namespace

SharedMemoryClient::~SharedMemoryClient() { close(); }

std::string describe_capabilities(std::uint32_t caps)
{
  if (caps == 0u) return "none";
  static const struct
  {
    std::uint32_t bit;
    const char * name;
  } NAMES[] = {
    {CAP_GROUND_TRUTH, "ground_truth"},
    {CAP_MUZZLE_WORLD_POSE, "muzzle_world_pose"},
    {CAP_CHASSIS_OBSERVATION, "chassis_observation"},
    {CAP_RUNTIME_STATE, "runtime_state"},
  };

  std::string out;
  std::uint32_t rest = caps;
  for (const auto & e : NAMES) {
    if ((rest & e.bit) == 0u) continue;
    if (!out.empty()) out += "|";
    out += e.name;
    rest &= ~e.bit;
  }
  if (rest != 0u) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08X", rest);
    if (!out.empty()) out += "|";
    out += buf;
  }
  return out;
}

bool SharedMemoryClient::open(std::string * error)
{
  // 注意：这里**不能**先 close()。见头文件里对事务化的说明——先拆旧映射会把
  // "meta 已建好、pool 还没定长" 这个瞬时窗口里的一次失败变成永久失联。
  const std::string meta_path = options_.dir + "/" + options_.meta_name;
  const std::string pool_path = options_.dir + "/" + options_.image_pool_name;

  MapResult meta_map;
  if (!map_region(meta_path, sizeof(ShmMetaRegion), &meta_map, error)) return false;

  MapResult pool_map;
  if (!map_region(pool_path, IMAGE_POOL_SIZE, &pool_map, error)) {
    ::munmap(meta_map.addr, meta_map.bytes);
    return false;
  }

  auto * meta = static_cast<ShmMetaRegion *>(meta_map.addr);

  // magic/version 先于任何字段解释：布局一旦改版，按旧布局读出的都是垃圾。
  std::string reason;
  if (meta->header.magic != SHM_MAGIC) {
    char buf[128];
    std::snprintf(
      buf, sizeof(buf), "magic 不匹配: 期望 0x%08X, 实际 0x%08X", SHM_MAGIC, meta->header.magic);
    reason = buf;
  } else if (meta->header.version != SHM_VERSION) {
    reason = "version 不匹配: 期望 " + std::to_string(SHM_VERSION) + ", 实际 " +
             std::to_string(meta->header.version);
  } else if (
    meta->header.image_width != IMAGE_WIDTH || meta->header.image_height != IMAGE_HEIGHT) {
    reason = "图像分辨率不匹配: 期望 " + std::to_string(IMAGE_WIDTH) + "x" +
             std::to_string(IMAGE_HEIGHT) + ", 实际 " + std::to_string(meta->header.image_width) +
             "x" + std::to_string(meta->header.image_height);
  } else if (
    (meta->header.capabilities & options_.required_capabilities) !=
    options_.required_capabilities) {
    // 必需能力位缺失只能在连接时拒绝。见 Options::required_capabilities：
    // 这些通道缺位时读出来的零值与合法读数在形状上完全一样，放过去就会被当成
    // 真实的 v3 世界量一路用下去。
    const std::uint32_t missing = options_.required_capabilities & ~meta->header.capabilities;
    reason = "缺少必需能力位 " + describe_capabilities(missing) + "（发布端声明 " +
             describe_capabilities(meta->header.capabilities) + "，本消费端要求 " +
             describe_capabilities(options_.required_capabilities) + "）";
  }

  if (!reason.empty()) {
    if (error) *error = reason;
    ::munmap(meta_map.addr, meta_map.bytes);
    ::munmap(pool_map.addr, pool_map.bytes);
    return false;
  }

  // 校验全部通过，到这一步才允许动现有映射。
  close();

  meta_ = meta;
  meta_bytes_ = meta_map.bytes;
  image_pool_ = static_cast<std::uint8_t *>(pool_map.addr);
  image_pool_bytes_ = pool_map.bytes;
  meta_dev_ = meta_map.dev;
  meta_ino_ = meta_map.ino;
  pool_dev_ = pool_map.dev;
  pool_ino_ = pool_map.ino;
  return true;
}

bool SharedMemoryClient::stat_ids(
  const std::string & path, std::uint64_t * dev, std::uint64_t * ino) const
{
  struct stat st{};
  if (::stat(path.c_str(), &st) != 0) return false;
  *dev = static_cast<std::uint64_t>(st.st_dev);
  *ino = static_cast<std::uint64_t>(st.st_ino);
  return true;
}

bool SharedMemoryClient::paths_changed() const
{
  if (!connected()) return false;

  std::uint64_t dev = 0;
  std::uint64_t ino = 0;
  // 文件当前不存在（发布端已退出、还没重建）不算变化：此时旧映射仍然是我们唯一
  // 的真相来源，且里面不会再有新帧，上层按心跳超时处理即可。只有"存在且身份不同"
  // 才说明发布端已经重建过文件。
  if (stat_ids(options_.dir + "/" + options_.meta_name, &dev, &ino)) {
    if (dev != meta_dev_ || ino != meta_ino_) return true;
  }
  if (stat_ids(options_.dir + "/" + options_.image_pool_name, &dev, &ino)) {
    if (dev != pool_dev_ || ino != pool_ino_) return true;
  }
  return false;
}

bool SharedMemoryClient::remap(std::string * error)
{
  // open() 是事务化的：失败时现有映射原样保留，connected() 不变，上层下个周期
  // 可以继续重试。
  if (!open(error)) {
    ++remap_failures_;
    return false;
  }
  // 换了文件就等于换了发布端：帧号水位线、换代基准全部作废。
  has_last_seq_ = false;
  last_seq_ = 0;
  publisher_created_ns_ = 0;
  ++remaps_;
  return true;
}

void SharedMemoryClient::close()
{
  if (meta_ != nullptr) {
    ::munmap(meta_, meta_bytes_);
    meta_ = nullptr;
    meta_bytes_ = 0;
  }
  if (image_pool_ != nullptr) {
    ::munmap(image_pool_, image_pool_bytes_);
    image_pool_ = nullptr;
    image_pool_bytes_ = 0;
  }
  // 断开后手里那份同帧真值不再属于任何一帧图像。
  has_frame_ground_truth_ = false;
}

ConsumeStatus SharedMemoryClient::drain_poses(FrameBundle * bundle, std::uint64_t image_seq)
{
  const int required = static_cast<int>(PoseIndex::Camera);  // 0..=3 参与背压握手
  bool missing = false;
  bool mismatch = false;

  for (int i = 0; i < static_cast<int>(POSE_CHANNEL_COUNT); ++i) {
    bool corrupted = false;
    const PoseMeta * pose = triple_buffer_consume(meta_->poses[i], &corrupted);
    if (corrupted) {
      ++corrupted_events_;
      return ConsumeStatus::Corrupted;
    }

    bundle->pose_present[i] = pose != nullptr;
    if (pose != nullptr) {
      bundle->poses[i] = *pose;
      if (i <= required && pose->frame_seq != image_seq) mismatch = true;
    } else {
      bundle->poses[i] = PoseMeta{};
      if (i <= required) missing = true;
    }
  }

  if (missing) return ConsumeStatus::PoseMissing;
  if (mismatch) return ConsumeStatus::PoseSeqMismatch;
  return ConsumeStatus::Ok;
}

ConsumeStatus SharedMemoryClient::consume_frame(FrameBundle * bundle)
{
  if (!connected() || bundle == nullptr) return ConsumeStatus::NotConnected;

  // 仿真端重启检测。ShmHeader::created_ns 在 ShmPublisher 构造时写入，重启后必变，
  // 而共享内存文件是按同一路径打开的（不是重建 inode），所以本进程原有的映射就能
  // 看到新值。
  //
  // 不做这件事的后果实测过：kill 掉仿真端再拉起来，新进程的 FRAME_SEQ 从 0 开始，
  // 下面的单调性门限会把每一帧都判成 regressed，直到新计数器爬过旧的最高水位才
  // 恢复——实测 rejected 426 帧、约 60s 才重新出图（然后 Tracker 报
  // "Large dt: 44.9s"）。期间行为是安全的（禁火 + 持续安全停止，faults 里带
  // target_lost），只是恢复得毫无必要地慢。
  //
  // 这里在换代时清掉水位线即可。注意**只清水位线**：不清 frame_age / 时钟映射
  // 之外的任何状态，也不把旧命令或旧帧当成有效数据。
  const std::uint64_t created_ns = __atomic_load_n(&meta_->header.created_ns, __ATOMIC_ACQUIRE);
  if (publisher_created_ns_ == 0) {
    publisher_created_ns_ = created_ns;
  } else if (created_ns != publisher_created_ns_) {
    publisher_created_ns_ = created_ns;
    ++publisher_restarts_;
    has_last_seq_ = false;
    last_seq_ = 0;
    // 换代必须显式上报，不能悄悄清掉水位线就继续。上层要据此把 Tracker/EKF/
    // Planner 的锁定、上一帧位姿、上一条命令全部作废，并在拿到新的完整同步帧
    // 且目标重新连续确认之前禁止开火。本帧丢弃：pose 通道大概率还是旧发布端
    // 留下的残留，同帧一致性无从谈起。
    return ConsumeStatus::EpochChanged;
  }

  // 上一帧的同帧真值到此作废。放在这里（而不是拷到新批次的地方）是为了让所有
  // 提前 return 的分支都不会把旧真值留给下一帧当"同帧"数据。
  has_frame_ground_truth_ = false;

  bool corrupted = false;
  const ImageMeta * image = triple_buffer_consume(meta_->image, &corrupted);
  if (corrupted) {
    ++corrupted_events_;
    return ConsumeStatus::Corrupted;
  }
  if (image == nullptr) return ConsumeStatus::NoFrame;

  const ImageMeta meta = *image;  // 先拷出来，后面所有判断都基于这份快照

  // 帧号必须前进。仿真端 try_publish_synchronized_image 保证了这一点，
  // 所以一旦看到不前进，说明仿真端重启（FRAME_SEQ 归零）或共享内存被重建。
  bool regressed = false;
  if (has_last_seq_) {
    if (meta.seq <= last_seq_) {
      ++regressed_frames_;
      regressed = true;
    } else {
      skipped_frames_ += meta.seq - last_seq_ - 1;
    }
  }

  // 同帧真值必须在这里拷走，窗口的两个边界都不是随手挑的：
  //   上界：triple_buffer_consume(meta_->image) 刚刚清掉图像的 FLAG_NEW；
  //   下界：drain_poses() 才会清掉 poses[0..=Camera] 的 FLAG_NEW。
  // 发布端的 synchronized_frame_consumed() 要求这两组标记**全部**清掉才允许发下
  // 一帧，所以在这中间它一定还堵在第 meta.seq 帧上，真值槽位里就是这一帧的批次。
  // 发布端那侧把真值放进图像提交前的 before_commit 回调里（事务化发布），保证
  // "图像 seq=k 可见"时真值已经就位。
  //
  // 以前是让评估器在流水线后段自己去 read_ground_truth()：检测一帧 ~250 ms，
  // 那时 pose 通道早已排空、背压放开，槽位已被后面若干帧覆盖，于是同帧校验恒不
  // 命中。这是 seq_mismatches 的第二个来源（第一个在发布端的时序）。
  {
    GroundTruthBatch captured{};
    if (read_ground_truth(&captured)) {
      frame_ground_truth_ = captured;
      has_frame_ground_truth_ = true;
      ++ground_truth_captures_;
    }
  }

  // 无论这一帧是否可用，pose 通道都必须排空，否则背压会让仿真端停止出图。
  const ConsumeStatus pose_status = drain_poses(bundle, meta.seq);

  bundle->frame_seq = meta.seq;
  bundle->timestamp_ns = meta.timestamp_ns;
  bundle->width = meta.width;
  bundle->height = meta.height;
  bundle->image_format = meta.format;
  bundle->buffer_id = meta.buffer_id;
  bundle->pixels = nullptr;
  bundle->pixel_bytes = 0;

  if (!regressed) {
    last_seq_ = meta.seq;
    has_last_seq_ = true;
  }

  if (pose_status == ConsumeStatus::Corrupted) return ConsumeStatus::Corrupted;

  if (
    meta.width != IMAGE_WIDTH || meta.height != IMAGE_HEIGHT ||
    meta.format != IMAGE_FORMAT_RGB8 || meta.buffer_id >= IMAGE_SLOT_COUNT) {
    ++dropped_frames_;
    return ConsumeStatus::ImageInvalid;
  }

  bundle->pixels = image_pool_ + static_cast<std::size_t>(meta.buffer_id) * IMAGE_SIZE;
  bundle->pixel_bytes = IMAGE_SIZE;

  if (regressed) {
    ++dropped_frames_;
    return ConsumeStatus::SeqRegressed;
  }
  if (pose_status != ConsumeStatus::Ok) {
    ++dropped_frames_;
    return pose_status;
  }

  ++consumed_frames_;
  return ConsumeStatus::Ok;
}

bool SharedMemoryClient::publish_gimbal_cmd(const GimbalCmd & cmd)
{
  if (!connected()) return false;
  triple_buffer_publish(meta_->gimbal_cmd, cmd);
  return true;
}

std::uint64_t SharedMemoryClient::heartbeat_ns() const
{
  if (!connected()) return 0;
  return __atomic_load_n(&meta_->header.heartbeat_ns, __ATOMIC_ACQUIRE);
}

std::uint64_t SharedMemoryClient::created_ns() const
{
  return connected() ? meta_->header.created_ns : 0;
}

std::uint32_t SharedMemoryClient::magic() const { return connected() ? meta_->header.magic : 0; }

std::uint32_t SharedMemoryClient::version() const
{
  return connected() ? meta_->header.version : 0;
}

std::uint32_t SharedMemoryClient::capabilities() const
{
  return connected() ? meta_->header.capabilities : 0u;
}

const CameraInfo * SharedMemoryClient::camera_info() const
{
  return connected() ? &meta_->camera_info : nullptr;
}

const RuntimeState * SharedMemoryClient::runtime_state() const
{
  if (!connected()) return nullptr;
  // 缺位时返回 nullptr，不返回恒零的 RuntimeState。理由见头文件：零值会把
  // "发布端不报这个字段"误诊成"仿真端没订阅云台命令"。
  if (!has_capability(CAP_RUNTIME_STATE)) {
    ++runtime_state_unsupported_;
    return nullptr;
  }
  return &meta_->runtime_state;
}

bool SharedMemoryClient::read_chassis_observation(ChassisObservation * out) const
{
  if (!connected() || out == nullptr) return false;
  // 全零的底盘观测是合法读数（车停着，v=0、w=0），所以"读出来是零"区分不了
  // "确实静止"和"发布端根本不填这个区"。只能看能力位。
  if (!has_capability(CAP_CHASSIS_OBSERVATION)) {
    ++chassis_observation_unsupported_;
    return false;
  }
  if (meta_->chassis_observation.timestamp_ns == 0) return false;
  *out = meta_->chassis_observation;
  return true;
}

bool SharedMemoryClient::frame_ground_truth(GroundTruthBatch * out) const
{
  if (out == nullptr || !has_frame_ground_truth_) return false;
  *out = frame_ground_truth_;
  return true;
}

bool SharedMemoryClient::read_ground_truth(GroundTruthBatch * out) const
{
  if (!connected() || out == nullptr) return false;

  // 发布端必须显式声明自己发真值。全零的 GroundTruthBatch 是一份合法数据
  // （target_count = 0，"这一帧没看见任何目标"），所以"读出来是空的"区分不了
  // "确实没目标"和"这个发布端根本不填真值"。不看能力位就会静默失去真值。
  if (!has_capability(CAP_GROUND_TRUTH)) {
    ++ground_truth_unsupported_;
    return false;
  }

  // seqlock 读端：奇数表示发布端正在写；前后两次读到同一个偶数才说明这份拷贝
  // 完整。发布端在 publish_ground_truth 里 odd -> body -> even，配对的 release/
  // acquire 保证 body 的写入不会被重排到序号之后。
  //
  // 不再用 frame_seq 是否为 0 判断"有没有数据"：frame_seq==0 是合法帧号（仿真端
  // FRAME_SEQ 从 0 开始），把它当哨兵会丢掉第一帧的真值。改用 seqlock != 0 —
  // 发布端至少提交过一次，seqlock 才会离开初始的 0。
  //
  // 只拷 GROUND_TRUTH_PAYLOAD_BYTES 字节的前缀，绝不整块 memcpy：标记正被发布端
  // 并发修改，用非原子读去读它本身就是数据竞争（UB），编译器也可以把这次读与下面
  // 那次 __atomic_load_n 合并或重排。标记之后只有 pad_，前缀已覆盖全部有效字段。
  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint32_t before = __atomic_load_n(&meta_->ground_truth.seqlock, __ATOMIC_ACQUIRE);
    if ((before & 1u) != 0u) continue;  // 写入进行中
    std::memcpy(out, &meta_->ground_truth, GROUND_TRUTH_PAYLOAD_BYTES);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const std::uint32_t after = __atomic_load_n(&meta_->ground_truth.seqlock, __ATOMIC_ACQUIRE);
    if (before != after) continue;
    // out 是调用方的私有内存，这里补齐标记与填充，让整块自洽。
    out->seqlock = before;
    std::memset(out->pad_, 0, sizeof(out->pad_));
    return before != 0u;
  }
  return false;
}

}  // namespace sim_io
