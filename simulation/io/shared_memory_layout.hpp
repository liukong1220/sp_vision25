#ifndef SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP
#define SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP

// 本文件是 bevy_robomaster_simulator `crates/talos-ipc/src/layout.rs` 里
// Rust `repr(C)` 结构体的 C++ 镜像。
//
// 字段顺序、类型、对齐和填充中的任何一项不一致都会造成静默的跨进程读错位，
// 而不是一个显式的错误。因此这里对每个结构体都做 sizeof / alignof / offsetof
// 编译期断言，不允许只靠“字段名看起来一样”来判定 ABI 兼容。
//
// 上游基线：SHM_MAGIC = 0x54414C05，SHM_VERSION = 4。
//
// 三缓冲的 `state` 在 Rust 侧是 `AtomicU8`。这里刻意用裸 `std::uint8_t` 保持
// 结构体是纯 POD（可 offsetof、可 static_assert 标准布局），原子语义由
// triple_buffer_* 函数用 `__atomic_*` builtin 显式提供，内存序与 Rust 侧一一对应。

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cstring>

#if !defined(__GNUC__) && !defined(__clang__)
#error "sim_io shared-memory layout relies on __atomic_* builtins (GCC/Clang)."
#endif

namespace sim_io
{
// ---------------------------------------------------------------- 协议常量 --

constexpr std::uint32_t SHM_MAGIC = 0x54414C05u;
// 协议版本。v2 -> v3 保留世界枪口语义，v4 增加原子单槽快照与命令消费回执。
// v3 的 poses[Muzzle] 由“云台局部平移 + 单位四元数”改为**完整世界
// 位姿**（见 PoseIndex 的注释），并启用 ShmHeader::capabilities 能力位。
//
// 字节布局没有变，所以这次升版号纯粹为了拦住语义不兼容：旧发布端 + 新消费端能
// mmap 成功、能读出数字，只是把局部偏移当世界坐标用，得到的是一个静默的错误答案。
constexpr std::uint32_t SHM_VERSION = 4u;

// ShmHeader::capabilities 的位定义。
//
// 版本号表达“协议第几代”，表达不了“这一代的发布端实际填了哪些可选区域”。真值区
// 就是典型：布局里一直有 ground_truth，但只有仿真器本体会填，测试用的精简发布端
// 不会。不看能力位就只能靠“读出来是否全零”去猜，而全零同样是合法数据，于是
// **新消费端对着不发真值的发布端会静默失去真值**。
constexpr std::uint32_t CAP_GROUND_TRUTH = 1u << 0;
// poses[Muzzle] 是世界位姿（v3/v4 语义）。未置位表示 v2 的局部平移语义。
constexpr std::uint32_t CAP_MUZZLE_WORLD_POSE = 1u << 1;
constexpr std::uint32_t CAP_CHASSIS_OBSERVATION = 1u << 2;
constexpr std::uint32_t CAP_RUNTIME_STATE = 1u << 3;

// 仿真器本体（TalosPlugin）声明的全集，镜像 talos-ipc 的 SIMULATOR_CAPABILITIES。
// 测试用的精简发布端可以只声明其中一部分。
constexpr std::uint32_t SIMULATOR_CAPABILITIES =
  CAP_GROUND_TRUTH | CAP_MUZZLE_WORLD_POSE | CAP_CHASSIS_OBSERVATION | CAP_RUNTIME_STATE;

constexpr std::uint32_t IMAGE_WIDTH = 1440u;
constexpr std::uint32_t IMAGE_HEIGHT = 1080u;
constexpr std::uint32_t IMAGE_CHANNELS = 3u;
constexpr std::size_t IMAGE_SIZE =
  static_cast<std::size_t>(IMAGE_WIDTH) * IMAGE_HEIGHT * IMAGE_CHANNELS;
constexpr std::size_t IMAGE_SLOT_COUNT = 3;
constexpr std::size_t IMAGE_POOL_SIZE = IMAGE_SIZE * IMAGE_SLOT_COUNT;

// Rust 侧 ShmRegion 把名字当作 /tmp 下的文件名，而不是 POSIX shm 对象。
constexpr const char * SHM_NAME_META = "talos_ipc_meta";
constexpr const char * SHM_NAME_IMAGE_POOL = "talos_ipc_image_pool";
constexpr const char * SHM_DIR = "/tmp";

constexpr std::uint8_t FLAG_NEW = 0x80u;
constexpr std::uint8_t INDEX_MASK = 0x03u;
constexpr std::size_t TRIPLE_SLOT_COUNT = 3;

// ImageMeta.format：0 = RGB8。上游目前只发布这一种。
constexpr std::uint8_t IMAGE_FORMAT_RGB8 = 0u;

constexpr std::size_t POSE_CHANNEL_COUNT = 5;
constexpr std::size_t GROUND_TRUTH_MAX_TARGETS = 16;
constexpr std::size_t GROUND_TRUTH_MAX_RUNES = 4;

// PoseIndex，对应 Rust `#[repr(u8)] enum PoseIndex`。
//
// 全部已转到 ROS 约定（x 前、y 左、z 上），四元数按 [w, x, y, z] 存放。
//
// **各通道参考系不同，混用会静默出错**，逐条写明：
//
//   Gimbal : position 恒为 [0,0,0]（占位，不是坐标）
//            quaternion = 云台/枪管的**世界**姿态 world <- gimbal
//   Odom   : position  = 云台回转中心的**世界**位置；quaternion = 单位四元数（占位）
//   Muzzle : position  = 枪口的**世界**位置；quaternion = 枪口的**世界**姿态（同 Gimbal）
//   Camera : position  = 相机相对云台的**局部**平移；quaternion = 单位四元数（占位）
//
// Camera 刻意保持局部：消费端拿它和自己配置里的 t_camera2gimbal 外参做自检，
// 那本来就是一个局部量（见 sim_auto_aim.cpp 的外参自检）。
//
// Muzzle 在 v2 里也是局部平移，v3 起改为世界位姿，能力位 CAP_MUZZLE_WORLD_POSE。
// 改的原因是局部量太容易被误用：消费端看到 Odom 是世界位置、Muzzle 是“偏移”，
// 最自然的写法就是 odom + muzzle，而那是把一个**未经云台旋转**的局部平移直接加到
// 世界坐标上。yaw=90° 时 0.11 m 的局部 +X 实际指向世界 +Y，误差等于偏移量全长且
// 随姿态变化，在报表上看起来就像闭环残差。直接发世界量让消费端无从误用。
enum class PoseIndex : std::uint8_t
{
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3,
  // 遗留通道：新接入应读 ShmMetaRegion::chassis_observation。
  ChassisObservation = 4,
};

// ---------------------------------------------------------------- 数据结构 --

struct alignas(32) ImageMeta
{
  std::uint64_t seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::uint8_t pad_[6];
};

struct alignas(64) PoseMeta
{
  std::uint64_t frame_seq;
  float position[3];
  // Rust 侧发布顺序是 [w, x, y, z]。
  float quaternion[4];
  std::uint8_t pad0_[4];
  std::uint64_t timestamp_ns;
  std::uint8_t pad_[16];
};

struct alignas(32) GimbalCmd
{
  std::uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  std::uint8_t fire_advice;
  std::uint8_t pad_[3];
  std::uint64_t command_seq;
};

struct alignas(64) CameraInfo
{
  std::uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t pad_[24];
};

struct alignas(64) ChassisObservation
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  float dt_s;
  float v_body[2];
  float wz_radps;
  float wheel_linear_mps[4];
  float wheel_angular_radps[4];
  float a_body[2];
  float alpha_z_radps2;
  float rpy_rad[3];
  float gyro_xyz_radps[3];
  float accel_xyz_mps2[3];
  std::uint32_t seqlock;
  std::uint8_t pad_[12];
};

template <typename Slot>
struct alignas(64) TripleBuffer
{
  std::uint8_t state;
  std::uint8_t write_idx;
  std::uint8_t read_idx;
  std::uint8_t pad1_[61];
  Slot slots[TRIPLE_SLOT_COUNT];
};

using ImageTripleBuffer = TripleBuffer<ImageMeta>;
using PoseTripleBuffer = TripleBuffer<PoseMeta>;
using GimbalTripleBuffer = TripleBuffer<GimbalCmd>;

struct alignas(64) ShmHeader
{
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t image_width;
  std::uint32_t image_height;
  // 可选区域能力位，见 CAP_* 常量。占用原 pad_[32] 的前 4 字节，
  // 结构体大小与其余偏移不变。
  std::uint32_t capabilities;
  std::uint8_t pad_[28];
};

struct alignas(32) GroundTruthTarget
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t is_outpost;
  std::uint8_t pad1_;
  float position[3];
  float vyaw;
  float yaw;
  // 被选中装甲板板心在 odom 系下的真值位置。整车中心（position）不是算法瞄的点：
  // 装甲板偏心半径 r 约 0.2m、板心比车心高约 0.06m，在 1.5m 距离上就是 2 度量级，
  // 拿整车中心算瞄准误差会把这段几何差当成闭环残差。发布端置 valid=0 时该字段
  // 无意义。占用原 pad_[24] 的前 16 字节，结构体大小和其余偏移不变。
  float armor_position[3];
  std::uint8_t armor_position_valid;
  // 前哨站少于三块同半径板、或没有相机参考时的退化标记；valid 保持独立。
  std::uint8_t armor_position_degraded;
  std::uint16_t identity;
  std::uint8_t pad_[8];
};

struct alignas(64) GroundTruthRune
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t rune_mode;
  std::uint8_t mechanism_state;
  std::uint8_t pad0_;
  float r_center_odom[3];
  float radius;
  float current_angle;
  float v_roll;
  std::int32_t direction;
  float sin_amplitude;
  float sin_omega;
  float sin_phase;
  float sin_offset;
  float relative_time;
  std::int32_t blade_id;
  std::uint8_t target_activations[5];
  std::uint8_t pad_act_[3];
  float target_point_odom[3];
  std::uint16_t identity;
  std::uint8_t pad_[34];
};

struct alignas(64) GroundTruthBatch
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t rune_count;
  std::uint8_t pad_before_targets[8];
  GroundTruthTarget targets[GROUND_TRUTH_MAX_TARGETS];
  std::uint8_t pad_before_runes[32];
  GroundTruthRune runes[GROUND_TRUTH_MAX_RUNES];
  // seqlock 序号。发布端写前置奇、写完置偶；读端读到奇数或前后不一致就重试。
  // 原来靠 "memcpy 前后读 frame_seq 相等" 近似判断整块稳定，这不是同步保证：
  // 覆盖写可能只改了 targets 的中段而 frame_seq 恰好没动（同一帧内重发），
  // 也可能 frame_seq 先写、body 后写，读端于是拿到半新半旧的一批目标。
  // 占用原 pad_[64] 的前 4 字节，结构体大小和其余偏移不变。
  std::uint32_t seqlock;
  std::uint8_t pad_[60];
};

// seqlock 标记之前的 payload 字节数，即整块里“真正的数据”。
//
// 两端拷贝 payload 时都只拷这段前缀，标记本身只用 __atomic_* 访问。整块 memcpy
// 会连带碰到那 4 字节标记：读端等于用非原子读去读一个正在被并发修改的原子变量，
// 写端等于用非原子写踩自己的同步变量。标记之后只有 pad_，所以这段前缀已覆盖
// 全部有效字段。
constexpr std::size_t GROUND_TRUTH_PAYLOAD_BYTES = 1600;
constexpr std::size_t CHASSIS_OBSERVATION_PAYLOAD_BYTES = 112;

struct alignas(64) RuntimeState
{
  std::uint64_t timestamp_ns;
  std::uint8_t following;
  std::uint8_t pad0_[3];
  std::uint32_t projectile_launch;
  std::uint32_t projectile_hit;
  std::uint32_t consumed_commands;
  std::uint32_t consumed_control_commands;
  std::uint32_t consumed_fire_commands;
  std::uint64_t frame_seq;
  std::uint64_t last_command_seq;
  std::uint64_t last_command_consume_timestamp_ns;
  std::uint32_t seqlock;
  std::uint8_t pad_[4];
};

constexpr std::size_t RUNTIME_STATE_PAYLOAD_BYTES = 56;

struct ShmMetaRegion
{
  ShmHeader header;
  ImageTripleBuffer image;
  PoseTripleBuffer poses[POSE_CHANNEL_COUNT];
  GimbalTripleBuffer gimbal_cmd;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  GroundTruthBatch ground_truth;
  RuntimeState runtime_state;
};

// ------------------------------------------------------------ ABI 编译期断言 --
//
// 下面的期望值全部来自 talos-ipc `layout.rs` 中的 `const _: () = assert!(...)`，
// 以及按 Rust repr(C) 布局规则推导出的字段偏移。断言失败即表示 C++ 镜像和
// Rust 发布端不再是同一份协议，必须先修 ABI，不允许改断言绕过。

#define SIM_IO_ASSERT_LAYOUT(type, expected_size, expected_align)             \
  static_assert(sizeof(type) == (expected_size), #type " size mismatch");     \
  static_assert(alignof(type) == (expected_align), #type " align mismatch");  \
  static_assert(std::is_standard_layout<type>::value, #type " not standard layout")

#define SIM_IO_ASSERT_OFFSET(type, field, expected)                           \
  static_assert(offsetof(type, field) == (expected), #type "::" #field " offset mismatch")

SIM_IO_ASSERT_LAYOUT(ImageMeta, 32, 32);
SIM_IO_ASSERT_OFFSET(ImageMeta, seq, 0);
SIM_IO_ASSERT_OFFSET(ImageMeta, timestamp_ns, 8);
SIM_IO_ASSERT_OFFSET(ImageMeta, width, 16);
SIM_IO_ASSERT_OFFSET(ImageMeta, height, 20);
SIM_IO_ASSERT_OFFSET(ImageMeta, buffer_id, 24);
SIM_IO_ASSERT_OFFSET(ImageMeta, format, 25);

SIM_IO_ASSERT_LAYOUT(PoseMeta, 64, 64);
SIM_IO_ASSERT_OFFSET(PoseMeta, frame_seq, 0);
SIM_IO_ASSERT_OFFSET(PoseMeta, position, 8);
SIM_IO_ASSERT_OFFSET(PoseMeta, quaternion, 20);
SIM_IO_ASSERT_OFFSET(PoseMeta, pad0_, 36);
SIM_IO_ASSERT_OFFSET(PoseMeta, timestamp_ns, 40);
SIM_IO_ASSERT_OFFSET(PoseMeta, pad_, 48);
static_assert(sizeof(PoseMeta{}.pad_) == 16, "PoseMeta pad occupies former implicit hole");

SIM_IO_ASSERT_LAYOUT(GimbalCmd, 32, 32);
SIM_IO_ASSERT_OFFSET(GimbalCmd, timestamp_ns, 0);
SIM_IO_ASSERT_OFFSET(GimbalCmd, yaw_deg, 8);
SIM_IO_ASSERT_OFFSET(GimbalCmd, pitch_deg, 12);
SIM_IO_ASSERT_OFFSET(GimbalCmd, distance_m, 16);
SIM_IO_ASSERT_OFFSET(GimbalCmd, fire_advice, 20);
SIM_IO_ASSERT_OFFSET(GimbalCmd, command_seq, 24);

SIM_IO_ASSERT_LAYOUT(CameraInfo, 128, 64);
SIM_IO_ASSERT_OFFSET(CameraInfo, timestamp_ns, 0);
SIM_IO_ASSERT_OFFSET(CameraInfo, fx, 8);
SIM_IO_ASSERT_OFFSET(CameraInfo, fy, 16);
SIM_IO_ASSERT_OFFSET(CameraInfo, cx, 24);
SIM_IO_ASSERT_OFFSET(CameraInfo, cy, 32);
SIM_IO_ASSERT_OFFSET(CameraInfo, distortion, 40);
SIM_IO_ASSERT_OFFSET(CameraInfo, width, 80);
SIM_IO_ASSERT_OFFSET(CameraInfo, height, 84);

SIM_IO_ASSERT_LAYOUT(ChassisObservation, 128, 64);
SIM_IO_ASSERT_OFFSET(ChassisObservation, frame_seq, 0);
SIM_IO_ASSERT_OFFSET(ChassisObservation, timestamp_ns, 8);
SIM_IO_ASSERT_OFFSET(ChassisObservation, dt_s, 16);
SIM_IO_ASSERT_OFFSET(ChassisObservation, v_body, 20);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wz_radps, 28);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wheel_linear_mps, 32);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wheel_angular_radps, 48);
SIM_IO_ASSERT_OFFSET(ChassisObservation, a_body, 64);
SIM_IO_ASSERT_OFFSET(ChassisObservation, alpha_z_radps2, 72);
SIM_IO_ASSERT_OFFSET(ChassisObservation, rpy_rad, 76);
SIM_IO_ASSERT_OFFSET(ChassisObservation, gyro_xyz_radps, 88);
SIM_IO_ASSERT_OFFSET(ChassisObservation, accel_xyz_mps2, 100);
SIM_IO_ASSERT_OFFSET(ChassisObservation, seqlock, 112);

SIM_IO_ASSERT_LAYOUT(ImageTripleBuffer, 192, 64);
SIM_IO_ASSERT_LAYOUT(PoseTripleBuffer, 256, 64);
SIM_IO_ASSERT_LAYOUT(GimbalTripleBuffer, 192, 64);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, state, 0);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, write_idx, 1);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, read_idx, 2);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, slots, 64);
SIM_IO_ASSERT_OFFSET(PoseTripleBuffer, slots, 64);
SIM_IO_ASSERT_OFFSET(GimbalTripleBuffer, slots, 64);

SIM_IO_ASSERT_LAYOUT(ShmHeader, 64, 64);
SIM_IO_ASSERT_OFFSET(ShmHeader, magic, 0);
SIM_IO_ASSERT_OFFSET(ShmHeader, version, 4);
SIM_IO_ASSERT_OFFSET(ShmHeader, created_ns, 8);
SIM_IO_ASSERT_OFFSET(ShmHeader, heartbeat_ns, 16);
SIM_IO_ASSERT_OFFSET(ShmHeader, image_width, 24);
SIM_IO_ASSERT_OFFSET(ShmHeader, image_height, 28);
SIM_IO_ASSERT_OFFSET(ShmHeader, capabilities, 32);

SIM_IO_ASSERT_LAYOUT(GroundTruthTarget, 64, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, team, 16);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_label, 17);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, is_outpost, 18);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, position, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, vyaw, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, yaw, 36);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_position, 40);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_position_valid, 52);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_position_degraded, 53);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, identity, 54);

SIM_IO_ASSERT_LAYOUT(GroundTruthRune, 128, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, r_center_odom, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, radius, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, current_angle, 36);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, v_roll, 40);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, direction, 44);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, sin_amplitude, 48);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, relative_time, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, blade_id, 68);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, target_activations, 72);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, target_point_odom, 80);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, pad0_, 19);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, pad_act_, 77);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, identity, 92);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, pad_, 94);

SIM_IO_ASSERT_LAYOUT(GroundTruthBatch, 1664, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, target_count, 16);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, rune_count, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, pad_before_targets, 24);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, targets, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, pad_before_runes, 1056);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, runes, 1088);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, seqlock, 1600);
static_assert(
  GROUND_TRUTH_PAYLOAD_BYTES == offsetof(GroundTruthBatch, seqlock),
  "GROUND_TRUTH_PAYLOAD_BYTES 必须正好等于 seqlock 的偏移");

SIM_IO_ASSERT_LAYOUT(RuntimeState, 64, 64);
SIM_IO_ASSERT_OFFSET(RuntimeState, following, 8);
SIM_IO_ASSERT_OFFSET(RuntimeState, pad0_, 9);
SIM_IO_ASSERT_OFFSET(RuntimeState, projectile_launch, 12);
SIM_IO_ASSERT_OFFSET(RuntimeState, projectile_hit, 16);
SIM_IO_ASSERT_OFFSET(RuntimeState, consumed_commands, 20);
SIM_IO_ASSERT_OFFSET(RuntimeState, consumed_control_commands, 24);
SIM_IO_ASSERT_OFFSET(RuntimeState, consumed_fire_commands, 28);
SIM_IO_ASSERT_OFFSET(RuntimeState, frame_seq, 32);
SIM_IO_ASSERT_OFFSET(RuntimeState, last_command_seq, 40);
SIM_IO_ASSERT_OFFSET(RuntimeState, last_command_consume_timestamp_ns, 48);
SIM_IO_ASSERT_OFFSET(RuntimeState, seqlock, 56);
static_assert(
  CHASSIS_OBSERVATION_PAYLOAD_BYTES == offsetof(ChassisObservation, seqlock),
  "ChassisObservation payload must stop before seqlock");
static_assert(
  RUNTIME_STATE_PAYLOAD_BYTES == offsetof(RuntimeState, seqlock),
  "RuntimeState payload must stop before seqlock");

SIM_IO_ASSERT_LAYOUT(ShmMetaRegion, 3712, 64);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, header, 0);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, image, 64);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, poses, 256);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, gimbal_cmd, 1536);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, camera_info, 1728);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, chassis_observation, 1856);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, ground_truth, 1984);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, runtime_state, 3648);

static_assert(IMAGE_SIZE == 4665600, "image size mismatch");
static_assert(IMAGE_POOL_SIZE == 13996800, "image pool size mismatch");

// ------------------------------------------------------- 共享内存原子操作原语 --

// 固定单槽快照的 payload 不能用普通 memcpy：seqlock 能检测一次拷贝跨越写入，
// 但不能把冲突的非原子访问变成语言层有定义的行为。协议 v4 规定这些共享字节
// 只能通过原子 byte 操作访问；本地临时结构仍是普通对象。
inline void atomic_copy_to_shared(void * destination, const void * source, std::size_t bytes)
{
  auto * dst = static_cast<std::uint8_t *>(destination);
  const auto * src = static_cast<const std::uint8_t *>(source);
  for (std::size_t i = 0; i < bytes; ++i)
    __atomic_store_n(dst + i, src[i], __ATOMIC_RELAXED);
}

inline void atomic_copy_from_shared(void * destination, const void * source, std::size_t bytes)
{
  auto * dst = static_cast<std::uint8_t *>(destination);
  const auto * src = static_cast<const std::uint8_t *>(source);
  for (std::size_t i = 0; i < bytes; ++i)
    dst[i] = __atomic_load_n(src + i, __ATOMIC_RELAXED);
}

inline void store_le_u16(std::uint8_t * p, std::uint16_t v)
{
  p[0] = static_cast<std::uint8_t>(v);
  p[1] = static_cast<std::uint8_t>(v >> 8);
}

inline void store_le_u32(std::uint8_t * p, std::uint32_t v)
{
  p[0] = static_cast<std::uint8_t>(v);
  p[1] = static_cast<std::uint8_t>(v >> 8);
  p[2] = static_cast<std::uint8_t>(v >> 16);
  p[3] = static_cast<std::uint8_t>(v >> 24);
}

inline void store_le_u64(std::uint8_t * p, std::uint64_t v)
{
  store_le_u32(p, static_cast<std::uint32_t>(v));
  store_le_u32(p + 4, static_cast<std::uint32_t>(v >> 32));
}

inline void store_le_i32(std::uint8_t * p, std::int32_t v)
{
  store_le_u32(p, static_cast<std::uint32_t>(v));
}

inline void store_le_f32(std::uint8_t * p, float v)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  store_le_u32(p, bits);
}

inline std::uint16_t load_le_u16(const std::uint8_t * p)
{
  return static_cast<std::uint16_t>(
    static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8));
}

inline std::uint32_t load_le_u32(const std::uint8_t * p)
{
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
    (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t load_le_u64(const std::uint8_t * p)
{
  return static_cast<std::uint64_t>(load_le_u32(p)) |
    (static_cast<std::uint64_t>(load_le_u32(p + 4)) << 32);
}

inline std::int32_t load_le_i32(const std::uint8_t * p)
{
  return static_cast<std::int32_t>(load_le_u32(p));
}

inline float load_le_f32(const std::uint8_t * p)
{
  const std::uint32_t bits = load_le_u32(p);
  float v = 0.0f;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

inline void encode_ground_truth_target(const GroundTruthTarget & src, std::uint8_t * dst)
{
  std::size_t off = 0;
  store_le_u64(dst + off, src.frame_seq); off += 8;
  store_le_u64(dst + off, src.timestamp_ns); off += 8;
  dst[off++] = src.team;
  dst[off++] = src.armor_label;
  dst[off++] = src.is_outpost;
  dst[off++] = src.pad1_;
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.position[i]); off += 4; }
  store_le_f32(dst + off, src.vyaw); off += 4;
  store_le_f32(dst + off, src.yaw); off += 4;
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.armor_position[i]); off += 4; }
  dst[off++] = src.armor_position_valid;
  dst[off++] = src.armor_position_degraded;
  store_le_u16(dst + off, src.identity); off += 2;
  std::memcpy(dst + off, src.pad_, 8); off += 8;
  (void)off;
}

inline GroundTruthTarget decode_ground_truth_target(const std::uint8_t * src)
{
  GroundTruthTarget out{};
  std::size_t off = 0;
  out.frame_seq = load_le_u64(src + off); off += 8;
  out.timestamp_ns = load_le_u64(src + off); off += 8;
  out.team = src[off++];
  out.armor_label = src[off++];
  out.is_outpost = src[off++];
  out.pad1_ = src[off++];
  for (int i = 0; i < 3; ++i) { out.position[i] = load_le_f32(src + off); off += 4; }
  out.vyaw = load_le_f32(src + off); off += 4;
  out.yaw = load_le_f32(src + off); off += 4;
  for (int i = 0; i < 3; ++i) { out.armor_position[i] = load_le_f32(src + off); off += 4; }
  out.armor_position_valid = src[off++];
  out.armor_position_degraded = src[off++];
  out.identity = load_le_u16(src + off); off += 2;
  std::memcpy(out.pad_, src + off, 8); off += 8;
  (void)off;
  return out;
}

inline void encode_ground_truth_rune(const GroundTruthRune & src, std::uint8_t * dst)
{
  std::size_t off = 0;
  store_le_u64(dst + off, src.frame_seq); off += 8;
  store_le_u64(dst + off, src.timestamp_ns); off += 8;
  dst[off++] = src.team;
  dst[off++] = src.rune_mode;
  dst[off++] = src.mechanism_state;
  dst[off++] = src.pad0_;
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.r_center_odom[i]); off += 4; }
  store_le_f32(dst + off, src.radius); off += 4;
  store_le_f32(dst + off, src.current_angle); off += 4;
  store_le_f32(dst + off, src.v_roll); off += 4;
  store_le_i32(dst + off, src.direction); off += 4;
  store_le_f32(dst + off, src.sin_amplitude); off += 4;
  store_le_f32(dst + off, src.sin_omega); off += 4;
  store_le_f32(dst + off, src.sin_phase); off += 4;
  store_le_f32(dst + off, src.sin_offset); off += 4;
  store_le_f32(dst + off, src.relative_time); off += 4;
  store_le_i32(dst + off, src.blade_id); off += 4;
  std::memcpy(dst + off, src.target_activations, 5); off += 5;
  std::memcpy(dst + off, src.pad_act_, 3); off += 3;
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.target_point_odom[i]); off += 4; }
  store_le_u16(dst + off, src.identity); off += 2;
  std::memcpy(dst + off, src.pad_, 34); off += 34;
  (void)off;
}

inline GroundTruthRune decode_ground_truth_rune(const std::uint8_t * src)
{
  GroundTruthRune out{};
  std::size_t off = 0;
  out.frame_seq = load_le_u64(src + off); off += 8;
  out.timestamp_ns = load_le_u64(src + off); off += 8;
  out.team = src[off++];
  out.rune_mode = src[off++];
  out.mechanism_state = src[off++];
  out.pad0_ = src[off++];
  for (int i = 0; i < 3; ++i) { out.r_center_odom[i] = load_le_f32(src + off); off += 4; }
  out.radius = load_le_f32(src + off); off += 4;
  out.current_angle = load_le_f32(src + off); off += 4;
  out.v_roll = load_le_f32(src + off); off += 4;
  out.direction = load_le_i32(src + off); off += 4;
  out.sin_amplitude = load_le_f32(src + off); off += 4;
  out.sin_omega = load_le_f32(src + off); off += 4;
  out.sin_phase = load_le_f32(src + off); off += 4;
  out.sin_offset = load_le_f32(src + off); off += 4;
  out.relative_time = load_le_f32(src + off); off += 4;
  out.blade_id = load_le_i32(src + off); off += 4;
  std::memcpy(out.target_activations, src + off, 5); off += 5;
  std::memcpy(out.pad_act_, src + off, 3); off += 3;
  for (int i = 0; i < 3; ++i) { out.target_point_odom[i] = load_le_f32(src + off); off += 4; }
  out.identity = load_le_u16(src + off); off += 2;
  std::memcpy(out.pad_, src + off, 34); off += 34;
  (void)off;
  return out;
}

inline void encode_ground_truth_batch(
  const GroundTruthBatch & src, std::uint8_t dst[GROUND_TRUTH_PAYLOAD_BYTES])
{
  std::size_t off = 0;
  store_le_u64(dst + off, src.frame_seq); off += 8;
  store_le_u64(dst + off, src.timestamp_ns); off += 8;
  store_le_u32(dst + off, src.target_count); off += 4;
  store_le_u32(dst + off, src.rune_count); off += 4;
  std::memcpy(dst + off, src.pad_before_targets, 8); off += 8;
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_TARGETS; ++i) {
    encode_ground_truth_target(src.targets[i], dst + off);
    off += 64;
  }
  std::memcpy(dst + off, src.pad_before_runes, 32); off += 32;
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_RUNES; ++i) {
    encode_ground_truth_rune(src.runes[i], dst + off);
    off += 128;
  }
  (void)off;
}

inline GroundTruthBatch decode_ground_truth_batch(
  const std::uint8_t src[GROUND_TRUTH_PAYLOAD_BYTES])
{
  GroundTruthBatch out{};
  std::size_t off = 0;
  out.frame_seq = load_le_u64(src + off); off += 8;
  out.timestamp_ns = load_le_u64(src + off); off += 8;
  out.target_count = load_le_u32(src + off); off += 4;
  out.rune_count = load_le_u32(src + off); off += 4;
  std::memcpy(out.pad_before_targets, src + off, 8); off += 8;
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_TARGETS; ++i) {
    out.targets[i] = decode_ground_truth_target(src + off);
    off += 64;
  }
  std::memcpy(out.pad_before_runes, src + off, 32); off += 32;
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_RUNES; ++i) {
    out.runes[i] = decode_ground_truth_rune(src + off);
    off += 128;
  }
  (void)off;
  return out;
}

inline void encode_chassis_observation(
  const ChassisObservation & src, std::uint8_t dst[CHASSIS_OBSERVATION_PAYLOAD_BYTES])
{
  std::size_t off = 0;
  store_le_u64(dst + off, src.frame_seq); off += 8;
  store_le_u64(dst + off, src.timestamp_ns); off += 8;
  store_le_f32(dst + off, src.dt_s); off += 4;
  for (int i = 0; i < 2; ++i) { store_le_f32(dst + off, src.v_body[i]); off += 4; }
  store_le_f32(dst + off, src.wz_radps); off += 4;
  for (int i = 0; i < 4; ++i) { store_le_f32(dst + off, src.wheel_linear_mps[i]); off += 4; }
  for (int i = 0; i < 4; ++i) { store_le_f32(dst + off, src.wheel_angular_radps[i]); off += 4; }
  for (int i = 0; i < 2; ++i) { store_le_f32(dst + off, src.a_body[i]); off += 4; }
  store_le_f32(dst + off, src.alpha_z_radps2); off += 4;
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.rpy_rad[i]); off += 4; }
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.gyro_xyz_radps[i]); off += 4; }
  for (int i = 0; i < 3; ++i) { store_le_f32(dst + off, src.accel_xyz_mps2[i]); off += 4; }
  (void)off;
}

inline ChassisObservation decode_chassis_observation(
  const std::uint8_t src[CHASSIS_OBSERVATION_PAYLOAD_BYTES])
{
  ChassisObservation out{};
  std::size_t off = 0;
  out.frame_seq = load_le_u64(src + off); off += 8;
  out.timestamp_ns = load_le_u64(src + off); off += 8;
  out.dt_s = load_le_f32(src + off); off += 4;
  for (int i = 0; i < 2; ++i) { out.v_body[i] = load_le_f32(src + off); off += 4; }
  out.wz_radps = load_le_f32(src + off); off += 4;
  for (int i = 0; i < 4; ++i) { out.wheel_linear_mps[i] = load_le_f32(src + off); off += 4; }
  for (int i = 0; i < 4; ++i) { out.wheel_angular_radps[i] = load_le_f32(src + off); off += 4; }
  for (int i = 0; i < 2; ++i) { out.a_body[i] = load_le_f32(src + off); off += 4; }
  out.alpha_z_radps2 = load_le_f32(src + off); off += 4;
  for (int i = 0; i < 3; ++i) { out.rpy_rad[i] = load_le_f32(src + off); off += 4; }
  for (int i = 0; i < 3; ++i) { out.gyro_xyz_radps[i] = load_le_f32(src + off); off += 4; }
  for (int i = 0; i < 3; ++i) { out.accel_xyz_mps2[i] = load_le_f32(src + off); off += 4; }
  (void)off;
  return out;
}

inline void encode_runtime_state(
  const RuntimeState & src, std::uint8_t dst[RUNTIME_STATE_PAYLOAD_BYTES])
{
  std::size_t off = 0;
  store_le_u64(dst + off, src.timestamp_ns); off += 8;
  dst[off++] = src.following;
  std::memcpy(dst + off, src.pad0_, 3); off += 3;
  store_le_u32(dst + off, src.projectile_launch); off += 4;
  store_le_u32(dst + off, src.projectile_hit); off += 4;
  store_le_u32(dst + off, src.consumed_commands); off += 4;
  store_le_u32(dst + off, src.consumed_control_commands); off += 4;
  store_le_u32(dst + off, src.consumed_fire_commands); off += 4;
  store_le_u64(dst + off, src.frame_seq); off += 8;
  store_le_u64(dst + off, src.last_command_seq); off += 8;
  store_le_u64(dst + off, src.last_command_consume_timestamp_ns); off += 8;
  (void)off;
}

inline RuntimeState decode_runtime_state(const std::uint8_t src[RUNTIME_STATE_PAYLOAD_BYTES])
{
  RuntimeState out{};
  std::size_t off = 0;
  out.timestamp_ns = load_le_u64(src + off); off += 8;
  out.following = src[off++];
  std::memcpy(out.pad0_, src + off, 3); off += 3;
  out.projectile_launch = load_le_u32(src + off); off += 4;
  out.projectile_hit = load_le_u32(src + off); off += 4;
  out.consumed_commands = load_le_u32(src + off); off += 4;
  out.consumed_control_commands = load_le_u32(src + off); off += 4;
  out.consumed_fire_commands = load_le_u32(src + off); off += 4;
  out.frame_seq = load_le_u64(src + off); off += 8;
  out.last_command_seq = load_le_u64(src + off); off += 8;
  out.last_command_consume_timestamp_ns = load_le_u64(src + off); off += 8;
  (void)off;
  return out;
}

// ------------------------------------------------------- 三缓冲原子操作原语 --
//
// 与 talos-ipc `triple_buffer.rs` 严格对应：
//   producer: old = state.swap(write_idx | FLAG_NEW, AcqRel); write_idx = old & INDEX_MASK
//   consumer: 最多两次 compare_exchange_weak(expected, read_idx, AcqRel, Acquire)
// 内存序映射：Acquire -> __ATOMIC_ACQUIRE，AcqRel -> __ATOMIC_ACQ_REL。
//
// state 的低两位可以表示 3，而槽位只有 3 个。Rust 侧靠数组越界 panic 兜底，
// 这里改成显式拒绝，避免共享内存被写坏时读到区域外的数据。

inline std::uint8_t triple_buffer_load_state(const std::uint8_t * state)
{
  return __atomic_load_n(state, __ATOMIC_ACQUIRE);
}

inline bool triple_buffer_has_new_data(const std::uint8_t * state)
{
  return (triple_buffer_load_state(state) & FLAG_NEW) != 0;
}

// 初始化到 Rust `init_triple_buffer` 的状态：state=1（就绪槽 1，无 FLAG_NEW）、
// write_idx=0、read_idx=2。零填充不是合法初始状态，仅测试用发布端需要它。
template <typename Slot>
inline void triple_buffer_init(TripleBuffer<Slot> & buf)
{
  buf.write_idx = 0;
  buf.read_idx = 2;
  __atomic_store_n(&buf.state, static_cast<std::uint8_t>(1), __ATOMIC_RELEASE);
}

// 消费一个槽位。返回 nullptr 表示没有新数据，或 state 已损坏。
// 返回的指针在下一次对同一通道调用本函数之前保持有效：CAS 成功后生产者的
// write_idx 会变成我们刚归还的槽位，不会是我们正在持有的 ready 槽位。
// 便捷重载：直接对整个三缓冲取状态字。
template <typename Slot>
inline std::uint8_t triple_buffer_load_state(const TripleBuffer<Slot> & buf)
{
  return triple_buffer_load_state(&buf.state);
}

template <typename Slot>
inline bool triple_buffer_has_new_data(const TripleBuffer<Slot> & buf)
{
  return triple_buffer_has_new_data(&buf.state);
}

template <typename Slot>
inline const Slot * triple_buffer_consume(TripleBuffer<Slot> & buf, bool * corrupted = nullptr)
{
  if (corrupted != nullptr) *corrupted = false;

  std::uint8_t expected = __atomic_load_n(&buf.state, __ATOMIC_ACQUIRE);

  for (int attempt = 0; attempt < 2; ++attempt) {
    if ((expected & FLAG_NEW) == 0) return nullptr;

    const std::uint8_t ready_idx = expected & INDEX_MASK;
    if (ready_idx >= TRIPLE_SLOT_COUNT || buf.read_idx >= TRIPLE_SLOT_COUNT) {
      if (corrupted != nullptr) *corrupted = true;
      return nullptr;
    }

    const std::uint8_t desired = buf.read_idx;
    if (__atomic_compare_exchange_n(
          &buf.state, &expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buf.read_idx = ready_idx;
      return &buf.slots[ready_idx];
    }
    // CAS 失败时 expected 已被更新为当前值，直接进入第二次尝试。
  }

  return nullptr;
}

// 发布一个槽位。先写槽位内容再翻转 state，state 是消费端观察到的提交点。
template <typename Slot>
inline bool triple_buffer_publish(TripleBuffer<Slot> & buf, const Slot & value)
{
  if (buf.write_idx >= TRIPLE_SLOT_COUNT) return false;

  const std::uint8_t idx = buf.write_idx;
  buf.slots[idx] = value;

  const std::uint8_t old = __atomic_exchange_n(
    &buf.state, static_cast<std::uint8_t>(idx | FLAG_NEW), __ATOMIC_ACQ_REL);
  buf.write_idx = old & INDEX_MASK;
  if (buf.write_idx >= TRIPLE_SLOT_COUNT) buf.write_idx = 0;
  return true;
}

#undef SIM_IO_ASSERT_LAYOUT
#undef SIM_IO_ASSERT_OFFSET

}  // namespace sim_io

#endif  // SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP
