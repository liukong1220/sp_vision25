#include "shared_memory_client.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
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
  }
  return "Unknown";
}

namespace
{
struct MapResult
{
  void * addr = nullptr;
  std::size_t bytes = 0;
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
  return true;
}
}  // namespace

SharedMemoryClient::~SharedMemoryClient() { close(); }

bool SharedMemoryClient::open(std::string * error)
{
  close();

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
  }

  if (!reason.empty()) {
    if (error) *error = reason;
    ::munmap(meta_map.addr, meta_map.bytes);
    ::munmap(pool_map.addr, pool_map.bytes);
    return false;
  }

  meta_ = meta;
  meta_bytes_ = meta_map.bytes;
  image_pool_ = static_cast<std::uint8_t *>(pool_map.addr);
  image_pool_bytes_ = pool_map.bytes;
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
  const std::uint64_t created_ns = meta_->header.created_ns;
  if (publisher_created_ns_ == 0) {
    publisher_created_ns_ = created_ns;
  } else if (created_ns != publisher_created_ns_) {
    publisher_created_ns_ = created_ns;
    ++publisher_restarts_;
    has_last_seq_ = false;
    last_seq_ = 0;
  }

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

const CameraInfo * SharedMemoryClient::camera_info() const
{
  return connected() ? &meta_->camera_info : nullptr;
}

const RuntimeState * SharedMemoryClient::runtime_state() const
{
  return connected() ? &meta_->runtime_state : nullptr;
}

bool SharedMemoryClient::read_chassis_observation(ChassisObservation * out) const
{
  if (!connected() || out == nullptr) return false;
  if (meta_->chassis_observation.timestamp_ns == 0) return false;
  *out = meta_->chassis_observation;
  return true;
}

bool SharedMemoryClient::read_ground_truth(GroundTruthBatch * out) const
{
  if (!connected() || out == nullptr) return false;

  // ground truth 是整块覆盖写、没有三缓冲保护，读取可能撕裂。
  // 用 frame_seq 前后一致来近似判断整块稳定；仅评估器使用，代价可以接受。
  for (int attempt = 0; attempt < 4; ++attempt) {
    const std::uint64_t before =
      __atomic_load_n(&meta_->ground_truth.frame_seq, __ATOMIC_ACQUIRE);
    std::memcpy(out, &meta_->ground_truth, sizeof(GroundTruthBatch));
    const std::uint64_t after = __atomic_load_n(&meta_->ground_truth.frame_seq, __ATOMIC_ACQUIRE);
    if (before == after && out->frame_seq == before) {
      return before != 0 || out->timestamp_ns != 0;
    }
  }
  return false;
}

}  // namespace sim_io
