#include "tools/runtime_params.hpp"

#include "tools/logger.hpp"
#include "tools/path.hpp"
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/core.h>

namespace fs = std::filesystem;

namespace tools::runtime_params
{
namespace
{
using json = nlohmann::json;

enum class ParamType
{
  kDouble,
  kInt,
  kBool,
  kDoubleArray,
  kString,
  kStringEnum,
};

struct ParamSpec
{
  std::string key;
  std::string group_id;
  std::string group_label;
  std::string label;
  std::string description;
  std::string unit;
  ParamType type = ParamType::kDouble;
  int array_size = 0;
  json default_value = nullptr;
  std::vector<std::string> enum_values;
};

struct SessionState
{
  std::string config_path;
  json base_values = json::object();
  json effective_values = json::object();
  json overrides = json::object();
  uint64_t version = 1;
  int64_t last_update_unix_ms = 0;
  std::string session_log_path;
  std::string snapshot_path;
};

std::mutex g_mutex;
std::unordered_map<std::string, SessionState> g_sessions;

int64_t unix_time_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

std::vector<ParamSpec> build_specs()
{
  return {
    {
      "min_confidence", "yolo", "YOLO筛选", "最小置信度",
      "识别结果进入装甲板链路前的最低置信度。", "", ParamType::kDouble,
    },
    {
      "use_traditional", "yolo", "YOLO筛选", "传统二次矫正",
      "YOLOv5 后处理时是否启用传统方法微调角点。", "", ParamType::kBool,
      0, false,
    },
    {
      "use_roi", "yolo", "YOLO筛选", "启用 ROI",
      "只在 ROI 内做识别，缩小搜索区域。", "", ParamType::kBool,
      0, false,
    },
    {
      "roi.x", "yolo", "YOLO筛选", "ROI X",
      "ROI 左上角横坐标。", "px", ParamType::kInt,
      0, 420,
    },
    {
      "roi.y", "yolo", "YOLO筛选", "ROI Y",
      "ROI 左上角纵坐标。", "px", ParamType::kInt,
      0, 50,
    },
    {
      "roi.width", "yolo", "YOLO筛选", "ROI 宽度",
      "ROI 宽度，-1 表示自动铺满。", "px", ParamType::kInt,
      0, 600,
    },
    {
      "roi.height", "yolo", "YOLO筛选", "ROI 高度",
      "ROI 高度，-1 表示自动铺满。", "px", ParamType::kInt,
      0, 600,
    },
    {
      "threshold", "detector", "传统检测", "二值化阈值",
      "传统检测使用的灰度阈值。", "", ParamType::kDouble,
    },
    {
      "max_angle_error", "detector", "传统检测", "最大角误差",
      "灯条允许的最大倾斜误差。", "deg", ParamType::kDouble,
    },
    {
      "min_lightbar_ratio", "detector", "传统检测", "最小灯条比",
      "灯条长宽比下限。", "", ParamType::kDouble,
    },
    {
      "max_lightbar_ratio", "detector", "传统检测", "最大灯条比",
      "灯条长宽比上限。", "", ParamType::kDouble,
    },
    {
      "min_lightbar_length", "detector", "传统检测", "最小灯条长度",
      "过滤过短灯条。", "px", ParamType::kDouble,
    },
    {
      "min_armor_ratio", "detector", "传统检测", "最小装甲板比",
      "装甲板宽高比下限。", "", ParamType::kDouble,
    },
    {
      "max_armor_ratio", "detector", "传统检测", "最大装甲板比",
      "装甲板宽高比上限。", "", ParamType::kDouble,
    },
    {
      "max_side_ratio", "detector", "传统检测", "最大边长差比",
      "左右灯条长度差异阈值。", "", ParamType::kDouble,
    },
    {
      "max_rectangular_error", "detector", "传统检测", "最大矩形误差",
      "装甲板矩形性误差阈值。", "deg", ParamType::kDouble,
    },
    {
      "enemy_color", "tracker", "跟踪", "敌方颜色",
      "用于过滤敌我颜色。", "", ParamType::kStringEnum,
      0, nullptr, {"red", "blue"},
    },
    {
      "com_port", "gimbal", "云台串口", "串口设备名",
      "云台串口设备路径，重连时会按当前值尝试重新打开。", "", ParamType::kStringEnum,
      0, nullptr,
      {
        "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3",
      },
    },
    {
      "min_detect_count", "tracker", "跟踪", "最小确认帧数",
      "进入 tracking 前需要连续观测的次数。", "frame", ParamType::kInt,
    },
    {
      "max_temp_lost_count", "tracker", "跟踪", "最大临时丢失帧数",
      "普通目标 temp_lost 状态允许持续的最大帧数。", "frame", ParamType::kInt,
    },
    {
      "outpost_max_temp_lost_count", "tracker", "跟踪", "前哨站丢失帧数",
      "前哨站 temp_lost 状态允许持续的最大帧数。", "frame", ParamType::kInt,
    },
    {
      "outpost_radius", "tracker", "跟踪", "前哨站半径",
      "前哨站模型半径。", "m", ParamType::kDouble,
      0, 0.2765,
    },
    {
      "outpost_spin_speed_lock", "tracker", "跟踪", "前哨站转速锁",
      "前哨站收敛后的角速度锁定值。", "rad/s", ParamType::kDouble,
      0, 2.51,
    },
    {
      "outpost_fixed_center_rotation_model", "tracker", "跟踪", "固定中心模型",
      "前哨站是否使用固定旋转中心模型。", "", ParamType::kBool,
      0, true,
    },
    {
      "outpost_armor_z_offsets", "tracker", "跟踪", "前哨站高度偏置",
      "三块前哨站装甲板相对中心的高度偏置。", "m", ParamType::kDoubleArray,
      3, json::array({0.0, -0.102, 0.102}),
    },
    {
      "tracker_acceleration_variance", "estimator", "状态估计", "平移加速度方差",
      "整车中心常速度模型的白噪声加速度方差。", "m^2/s^4", ParamType::kDouble,
      0, 100.0,
    },
    {
      "tracker_yaw_acceleration_variance", "estimator", "状态估计", "角加速度方差",
      "整车 yaw 常角速度模型的白噪声角加速度方差。", "rad^2/s^4", ParamType::kDouble,
      0, 400.0,
    },
    {
      "tracker_roll_pitch_random_walk", "estimator", "状态估计", "Roll/Pitch 随机游走",
      "SO(3) 误差状态中 roll/pitch 每秒累积的过程噪声。", "rad^2/s", ParamType::kDouble,
      0, 2e-3,
    },
    {
      "tracker_geometry_random_walk", "estimator", "状态估计", "几何随机游走",
      "车体对数半径和高度差的单位时间过程噪声。", "m^2/s", ParamType::kDouble,
      0, 1e-4,
    },
    {
      "tracker_uvl_angle_variance", "estimator", "状态估计", "灯条角度方差",
      "UVL 图像观测中单灯条角度的量测方差。", "rad^2", ParamType::kDouble,
      0, 2.5e-3,
    },
    {
      "tracker_uvl_center_variance", "estimator", "状态估计", "灯条中心方差",
      "UVL 图像观测中灯条中心 u/v 的像素方差。", "px^2", ParamType::kDouble,
      0, 9.0,
    },
    {
      "tracker_uvl_length_variance", "estimator", "状态估计", "灯条长度方差",
      "UVL 图像观测中灯条投影长度的像素方差。", "px^2", ParamType::kDouble,
      0, 9.0,
    },
    {
      "tracker_nis_gate", "estimator", "状态估计", "NIS 门限",
      "8 维双灯条 UVL 创新的卡方门限；20.090 对应 99% 置信度。", "score",
      ParamType::kDouble, 0, 20.090,
    },
    {
      "inference_max_inflight", "inference", "并发推理", "最大在途帧数",
      "异步推理允许同时在队列中的最大帧数，满载时在启动推理前丢弃新帧。", "frame",
      ParamType::kInt, 0, 3,
    },
    {
      "outpost_coming_angle", "planner", "规划/MPC", "前哨站进入窗口角",
      "前哨站选板进入击打窗口角度，0 表示沿用通用窗口。", "deg", ParamType::kDouble,
      0, 70.0,
    },
    {
      "outpost_leaving_angle", "planner", "规划/MPC", "前哨站离开窗口角",
      "前哨站选板离开击打窗口角度，0 表示沿用通用窗口。", "deg", ParamType::kDouble,
      0, 30.0,
    },
    {
      "outpost_delay_time", "planner", "规划/MPC", "前哨站固定延迟",
      "前哨站专用预测延迟，0 表示沿用通用高低速延迟。", "s", ParamType::kDouble,
      0, 0.0,
    },
    {
      "outpost_fire_z_compensation", "planner", "规划/MPC", "前哨站击打高度补偿",
      "前哨站三块装甲板的额外火控高度补偿。", "m", ParamType::kDoubleArray,
      3, json::array({0.0, 0.0, 0.0}),
    },
    {
      "bullet_speed_min", "planner", "规划/MPC", "子弹速度最小有效值",
      "低于该值时认为串口测速异常，Planner 将使用回退速度。", "m/s", ParamType::kDouble,
      0, 10.0,
    },
    {
      "bullet_speed_max", "planner", "规划/MPC", "子弹速度最大有效值",
      "高于该值时认为串口测速异常，Planner 将使用回退速度。", "m/s", ParamType::kDouble,
      0, 25.0,
    },
    {
      "bullet_speed_fallback", "planner", "规划/MPC", "子弹速度回退值",
      "测速异常时 Planner 使用的默认子弹速度。", "m/s", ParamType::kDouble,
      0, 22.0,
    },
    {
      "yaw_offset", "planner", "规划/MPC", "Yaw 零偏",
      "枪口/相机在 yaw 方向的补偿偏置。", "deg", ParamType::kDouble,
    },
    {
      "pitch_offset", "planner", "规划/MPC", "Pitch 零偏",
      "枪口/相机在 pitch 方向的补偿偏置。", "deg", ParamType::kDouble,
    },
    {
      "coming_angle", "planner", "规划/MPC", "进入窗口角",
      "小陀螺切板时允许进入击打窗口的角度。", "deg", ParamType::kDouble,
    },
    {
      "leaving_angle", "planner", "规划/MPC", "离开窗口角",
      "小陀螺切板时允许离开击打窗口的角度。", "deg", ParamType::kDouble,
    },
    {
      "decision_speed", "planner", "规划/MPC", "高速判定角速",
      "根据目标角速度切换高低速延迟的阈值。", "rad/s", ParamType::kDouble,
    },
    {
      "high_speed_delay_time", "planner", "规划/MPC", "高速延迟补偿",
      "高速转动目标时使用的预测延迟。", "s", ParamType::kDouble,
    },
    {
      "low_speed_delay_time", "planner", "规划/MPC", "低速延迟补偿",
      "低速转动目标时使用的预测延迟。", "s", ParamType::kDouble,
    },
    {
      "fire_thresh", "planner", "规划/MPC", "击发阈值",
      "预测轨迹与 MPC 轨迹的最大允许误差。", "rad", ParamType::kDouble,
    },
    {
      "max_yaw_acc", "planner", "规划/MPC", "Yaw 最大角加速度",
      "Yaw MPC 的输入约束。", "deg/s^2", ParamType::kDouble,
    },
    {
      "Q_yaw", "planner", "规划/MPC", "Yaw Q",
      "Yaw MPC 状态权重。", "", ParamType::kDoubleArray,
      2,
    },
    {
      "R_yaw", "planner", "规划/MPC", "Yaw R",
      "Yaw MPC 控制权重。", "", ParamType::kDoubleArray,
      1,
    },
    {
      "max_pitch_acc", "planner", "规划/MPC", "Pitch 最大角加速度",
      "Pitch MPC 的输入约束。", "deg/s^2", ParamType::kDouble,
    },
    {
      "Q_pitch", "planner", "规划/MPC", "Pitch Q",
      "Pitch MPC 状态权重。", "", ParamType::kDoubleArray,
      2,
    },
    {
      "R_pitch", "planner", "规划/MPC", "Pitch R",
      "Pitch MPC 控制权重。", "", ParamType::kDoubleArray,
      1,
    },
    {
      "first_tolerance", "shooter", "射击判定", "近距离容差",
      "近距离射击容差，单位度。", "deg", ParamType::kDouble,
    },
    {
      "second_tolerance", "shooter", "射击判定", "远距离容差",
      "远距离射击容差，单位度。", "deg", ParamType::kDouble,
    },
    {
      "judge_distance", "shooter", "射击判定", "距离分界",
      "切换近远距离容差的距离阈值。", "m", ParamType::kDouble,
    },
    {
      "auto_fire", "shooter", "射击判定", "自动开火",
      "是否允许链路自动触发射击。", "", ParamType::kBool,
      0, true,
    },
    {
      "fire_gap_time", "buff", "BUFF参数", "Buff 最小击发间隔",
      "两次 Buff 击发之间的最小时间间隔。", "s", ParamType::kDouble,
    },
    {
      "predict_time", "buff", "BUFF参数", "Buff 预测提前量",
      "Buff 预测时间补偿。", "s", ParamType::kDouble,
    },
    {
      "R_gimbal2imubody", "calibration", "标定/外参", "云台到IMU旋转",
      "云台坐标系到IMU机体系的旋转矩阵，按行展开。", "", ParamType::kDoubleArray,
      9,
    },
    {
      "camera_matrix", "calibration", "标定/外参", "相机内参矩阵",
      "相机内参矩阵，按行展开。", "", ParamType::kDoubleArray,
      9,
    },
    {
      "distort_coeffs", "calibration", "标定/外参", "畸变系数",
      "相机畸变系数。", "", ParamType::kDoubleArray,
      5,
    },
    {
      "R_camera2gimbal", "calibration", "标定/外参", "相机到云台旋转",
      "相机坐标系到云台坐标系的旋转矩阵，按行展开。", "", ParamType::kDoubleArray,
      9,
    },
    {
      "t_camera2gimbal", "calibration", "标定/外参", "相机到云台平移",
      "相机坐标系到云台坐标系的平移向量。", "m", ParamType::kDoubleArray,
      3,
    },
    {
      "gimbal_state_unit", "debug", "调试显示", "云台状态单位",
      "网页与本地可视化解释串口姿态字段时采用的单位。", "", ParamType::kStringEnum,
      0, nullptr, {"auto", "deg", "rad"},
    },
    {
      "show_local", "debug", "调试显示", "本地窗口",
      "是否启用本地 OpenCV 调试窗口。", "", ParamType::kBool,
      0, false,
    },
    {
      "web_fps", "debug", "调试显示", "网页刷新帧率",
      "网页主图和弹道图的推流帧率。", "fps", ParamType::kDouble,
      0, 30.0,
    },
    {
      "web_scale", "debug", "调试显示", "显示缩放",
      "调试输出图像的显示缩放系数。", "", ParamType::kDouble,
      0, 0.8,
    },
    {
      "web_jpeg_quality", "debug", "调试显示", "JPEG质量",
      "网页图像压缩质量。", "", ParamType::kInt,
      0, 55,
    },
    {
      "web_client_ttl_ms", "debug", "调试显示", "网页保活时间",
      "最近访问后继续保持渲染的时间窗口。", "ms", ParamType::kInt,
      0, 1000,
    },
    {
      "record_raw_video", "debug_record", "调试录制", "录制原始图像",
      "是否录制原始相机画面。", "", ParamType::kBool,
      0, false,
    },
    {
      "record_debug_video", "debug_record", "调试录制", "录制调试画面",
      "是否录制带调试叠加的输出画面。", "", ParamType::kBool,
      0, false,
    },
    {
      "record_debug_fps", "debug_record", "调试录制", "录制帧率",
      "调试录制输出帧率。", "fps", ParamType::kDouble,
      0, 30.0,
    },
    {
      "record_debug_dir", "debug_record", "调试录制", "录制目录",
      "调试录制输出目录。", "", ParamType::kString,
      0, "records",
    },
  };
}

const std::vector<ParamSpec> & specs()
{
  static const std::vector<ParamSpec> kSpecs = build_specs();
  return kSpecs;
}

const ParamSpec & spec_for(const std::string & key)
{
  const auto & all_specs = specs();
  const auto it = std::find_if(
    all_specs.begin(), all_specs.end(),
    [&](const ParamSpec & spec) {return spec.key == key;});
  if (it == all_specs.end()) {
    throw std::runtime_error("Unknown runtime parameter key: " + key);
  }
  return *it;
}

double ui_step_for(const ParamSpec & spec)
{
  if (spec.type == ParamType::kInt) return 1.0;

  if (spec.key == "min_confidence") return 0.01;
  if (spec.key == "threshold") return 1.0;
  if (spec.key == "fire_thresh") return 0.0001;
  if (
    spec.key == "fire_gap_time" || spec.key == "predict_time" ||
    spec.key == "record_debug_fps")
  {
    return 0.001;
  }
  if (
    spec.key == "high_speed_delay_time" || spec.key == "low_speed_delay_time" ||
    spec.key == "outpost_delay_time" || spec.key == "outpost_radius")
  {
    return 0.001;
  }
  if (spec.key == "web_fps") return 1.0;
  if (spec.key == "web_scale") return 0.05;
  if (spec.key == "web_jpeg_quality" || spec.key == "web_client_ttl_ms") return 1.0;
  if (
    spec.key == "R_gimbal2imubody" || spec.key == "camera_matrix" ||
    spec.key == "distort_coeffs" || spec.key == "R_camera2gimbal" ||
    spec.key == "t_camera2gimbal")
  {
    return 0.000001;
  }
  if (
    spec.key == "bullet_speed_min" || spec.key == "bullet_speed_max" ||
    spec.key == "bullet_speed_fallback")
  {
    return 0.1;
  }
  if (spec.key == "outpost_spin_speed_lock") return 0.01;
  if (spec.key == "decision_speed") return 0.05;
  if (spec.key == "yaw_offset" || spec.key == "pitch_offset") return 0.05;
  if (
    spec.key == "coming_angle" || spec.key == "leaving_angle" ||
    spec.key == "outpost_coming_angle" || spec.key == "outpost_leaving_angle")
  {
    return 0.1;
  }
  if (spec.key == "max_yaw_acc" || spec.key == "max_pitch_acc") return 1.0;
  if (spec.unit == "deg") return 0.1;
  if (spec.unit == "rad") return 0.0001;
  if (spec.unit == "rad/s") return 0.05;
  if (spec.unit == "m") return 0.001;
  if (spec.type == ParamType::kDouble) return 0.01;

  return 1.0;
}

std::optional<double> ui_min_for(const ParamSpec & spec)
{
  if (spec.key == "min_confidence") return 0.0;
  if (spec.key == "threshold") return 0.0;
  if (
    spec.key == "roi.width" || spec.key == "roi.height" || spec.key == "roi.x" ||
    spec.key == "roi.y")
  {
    return spec.key == "roi.width" || spec.key == "roi.height" ? -1.0 : 0.0;
  }
  if (spec.key == "web_client_ttl_ms" || spec.key == "web_jpeg_quality") return 0.0;
  if (
    spec.key == "min_detect_count" || spec.key == "max_temp_lost_count" ||
    spec.key == "outpost_max_temp_lost_count" || spec.key == "inference_max_inflight")
  {
    return spec.key == "inference_max_inflight" ? 1.0 : 0.0;
  }
  if (
    spec.key == "min_lightbar_ratio" || spec.key == "max_lightbar_ratio" ||
    spec.key == "min_lightbar_length" || spec.key == "min_armor_ratio" ||
    spec.key == "max_armor_ratio" || spec.key == "max_side_ratio" ||
    spec.key == "max_angle_error" || spec.key == "max_rectangular_error" ||
    spec.key == "outpost_radius" || spec.key == "outpost_spin_speed_lock" ||
    spec.key == "bullet_speed_min" || spec.key == "bullet_speed_max" ||
    spec.key == "bullet_speed_fallback" ||
    spec.key == "decision_speed" || spec.key == "high_speed_delay_time" ||
    spec.key == "outpost_delay_time" ||
    spec.key == "low_speed_delay_time" || spec.key == "fire_thresh" ||
    spec.key == "fire_gap_time" || spec.key == "predict_time" ||
    spec.key == "judge_distance" || spec.key == "record_debug_fps" ||
    spec.key == "web_fps" || spec.key == "web_scale" ||
    spec.key == "max_yaw_acc" || spec.key == "max_pitch_acc" ||
    spec.key == "coming_angle" || spec.key == "leaving_angle" ||
    spec.key == "first_tolerance" || spec.key == "second_tolerance" ||
    spec.key == "outpost_coming_angle" || spec.key == "outpost_leaving_angle" ||
    spec.group_id == "estimator")
  {
    return 0.0;
  }
  return std::nullopt;
}

std::optional<double> ui_max_for(const ParamSpec & spec)
{
  if (spec.key == "min_confidence") return 1.0;
  if (spec.key == "threshold") return 255.0;
  if (spec.key == "web_fps") return 60.0;
  if (spec.key == "web_scale") return 1.0;
  if (spec.key == "web_jpeg_quality") return 95.0;
  if (spec.key == "web_client_ttl_ms") return 10000.0;
  if (spec.key == "inference_max_inflight") return 16.0;
  if (spec.key == "tracker_nis_gate") return 100.0;
  if (spec.key == "max_angle_error" || spec.key == "max_rectangular_error") return 90.0;
  if (spec.key == "first_tolerance" || spec.key == "second_tolerance") return 45.0;
  if (
    spec.key == "coming_angle" || spec.key == "leaving_angle" ||
    spec.key == "outpost_coming_angle" || spec.key == "outpost_leaving_angle")
  {
    return 180.0;
  }
  return std::nullopt;
}

int ui_precision_for(const ParamSpec & spec)
{
  if (spec.type == ParamType::kInt) return 0;

  if (spec.key == "threshold") return 0;
  if (spec.key == "min_confidence") return 2;
  if (spec.key == "fire_thresh") return 4;
  if (spec.group_id == "estimator") return 6;
  if (spec.key == "web_scale") return 2;
  if (spec.key == "web_fps" || spec.key == "record_debug_fps") return 1;
  if (
    spec.key == "R_gimbal2imubody" || spec.key == "camera_matrix" ||
    spec.key == "distort_coeffs" || spec.key == "R_camera2gimbal" ||
    spec.key == "t_camera2gimbal")
  {
    return 6;
  }
  if (
    spec.key == "high_speed_delay_time" || spec.key == "low_speed_delay_time" ||
    spec.key == "outpost_delay_time" || spec.key == "outpost_radius" ||
    spec.key == "fire_gap_time" || spec.key == "predict_time")
  {
    return 3;
  }
  if (spec.unit == "deg" || spec.unit == "rad/s" || spec.unit == "deg/s^2") return 2;
  if (spec.unit == "m" || spec.unit == "rad") return 3;
  if (spec.type == ParamType::kDoubleArray) return 3;
  return 2;
}

std::string trim_copy(std::string value)
{
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string::npos) return "";
  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

std::string strip_inline_comment(std::string line)
{
  bool in_single_quote = false;
  bool in_double_quote = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"' && !in_single_quote) in_double_quote = !in_double_quote;
    if (ch == '\'' && !in_double_quote) in_single_quote = !in_single_quote;
    if (ch == '#' && !in_single_quote && !in_double_quote) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string strip_quotes(std::string value)
{
  value = trim_copy(std::move(value));
  if (value.size() >= 2) {
    const char first = value.front();
    const char last = value.back();
    if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
      return value.substr(1, value.size() - 2);
    }
  }
  return value;
}

std::unordered_map<std::string, std::string> parse_config_text(const std::string & config_path)
{
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open config text: " + config_path);
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  std::string active_section;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    line = strip_inline_comment(line);
    const auto first_content = line.find_first_not_of(" \t");
    if (first_content == std::string::npos) continue;

    const bool is_nested = first_content > 0;
    const auto trimmed = trim_copy(line.substr(first_content));
    const auto colon = trimmed.find(':');
    if (colon == std::string::npos) continue;

    const auto raw_key = trim_copy(trimmed.substr(0, colon));
    const auto raw_value = trim_copy(trimmed.substr(colon + 1));
    if (!is_nested) {
      active_section.clear();
      if (raw_value.empty()) {
        active_section = raw_key;
        continue;
      }
      values[raw_key] = raw_value;
      continue;
    }

    if (active_section.empty()) continue;
    values[active_section + "." + raw_key] = raw_value;
  }
  return values;
}

json read_value_from_text(
  const std::unordered_map<std::string, std::string> & values, const ParamSpec & spec)
{
  auto it = values.find(spec.key);
  if (it == values.end() || trim_copy(it->second).empty()) {
    if (!spec.default_value.is_null()) return spec.default_value;
    throw std::runtime_error("Runtime parameter missing in YAML: " + spec.key);
  }

  const std::string raw_value = trim_copy(it->second);

  switch (spec.type) {
    case ParamType::kDouble:
      return std::stod(raw_value);
    case ParamType::kInt:
      return std::stoi(raw_value);
    case ParamType::kBool: {
      std::string lowered = raw_value;
      std::transform(
        lowered.begin(), lowered.end(), lowered.begin(),
        [](unsigned char ch) {return static_cast<char>(std::tolower(ch));});
      if (lowered == "true") return true;
      if (lowered == "false") return false;
      throw std::runtime_error(spec.key + " expects true/false");
    }
    case ParamType::kString:
      return strip_quotes(raw_value);
    case ParamType::kStringEnum:
      return strip_quotes(raw_value);
    case ParamType::kDoubleArray: {
      const auto start = raw_value.find('[');
      const auto end = raw_value.rfind(']');
      if (start == std::string::npos || end == std::string::npos || end < start) {
        throw std::runtime_error(spec.key + " expects [a, b, c]");
      }
      const auto inner = raw_value.substr(start + 1, end - start - 1);
      std::stringstream stream(inner);
      std::string token;
      json result = json::array();
      while (std::getline(stream, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;
        result.push_back(std::stod(token));
      }
      if (spec.array_size > 0 && result.size() != static_cast<size_t>(spec.array_size)) {
        throw std::runtime_error(
          spec.key + " expects " + std::to_string(spec.array_size) + " values");
      }
      return result;
    }
  }
  throw std::runtime_error("Unhandled runtime parameter type");
}

json normalize_value(const json & incoming, const ParamSpec & spec)
{
  switch (spec.type) {
    case ParamType::kDouble:
      if (!incoming.is_number()) {
        throw std::runtime_error(spec.key + " expects a number");
      }
      return incoming.get<double>();
    case ParamType::kInt:
      if (!incoming.is_number_integer() && !incoming.is_number()) {
        throw std::runtime_error(spec.key + " expects an integer");
      }
      return static_cast<int>(std::lround(incoming.get<double>()));
    case ParamType::kBool:
      if (!incoming.is_boolean()) {
        throw std::runtime_error(spec.key + " expects a boolean");
      }
      return incoming.get<bool>();
    case ParamType::kString:
      if (!incoming.is_string()) {
        throw std::runtime_error(spec.key + " expects a string");
      }
      return incoming.get<std::string>();
    case ParamType::kStringEnum: {
      if (!incoming.is_string()) {
        throw std::runtime_error(spec.key + " expects a string");
      }
      const auto value = incoming.get<std::string>();
      if (
        !spec.enum_values.empty() &&
        std::find(spec.enum_values.begin(), spec.enum_values.end(), value) == spec.enum_values.end())
      {
        throw std::runtime_error(spec.key + " value out of range");
      }
      return value;
    }
    case ParamType::kDoubleArray: {
      if (!incoming.is_array()) {
        throw std::runtime_error(spec.key + " expects an array");
      }
      json normalized = json::array();
      for (const auto & item : incoming) {
        if (!item.is_number()) {
          throw std::runtime_error(spec.key + " expects a numeric array");
        }
        normalized.push_back(item.get<double>());
      }
      if (spec.array_size > 0 && normalized.size() != static_cast<size_t>(spec.array_size)) {
        throw std::runtime_error(
          spec.key + " expects " + std::to_string(spec.array_size) + " values");
      }
      return normalized;
    }
  }
  throw std::runtime_error("Unhandled runtime parameter type");
}

std::string yaml_scalar(const json & value)
{
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_string()) return fmt::format("\"{}\"", value.get<std::string>());
  return value.dump();
}

json nest_flat_key_values(const json & flat_values)
{
  json root = json::object();
  for (auto it = flat_values.begin(); it != flat_values.end(); ++it) {
    json * current = &root;
    const std::string key = it.key();
    size_t start = 0;
    while (start < key.size()) {
      const auto dot = key.find('.', start);
      const auto token = key.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
      if (dot == std::string::npos) {
        (*current)[token] = it.value();
        break;
      }
      if (!(*current).contains(token) || !(*current)[token].is_object()) {
        (*current)[token] = json::object();
      }
      current = &(*current)[token];
      start = dot + 1;
    }
  }
  return root;
}

void append_yaml_lines(
  const json & value, const std::string & key, int indent, std::vector<std::string> & lines)
{
  const std::string padding(static_cast<size_t>(indent), ' ');
  if (value.is_object()) {
    lines.push_back(fmt::format("{}{}:", padding, key));
    for (auto it = value.begin(); it != value.end(); ++it) {
      append_yaml_lines(it.value(), it.key(), indent + 2, lines);
    }
    return;
  }

  if (value.is_array()) {
    lines.push_back(fmt::format("{}{}: {}", padding, key, value.dump()));
    return;
  }

  lines.push_back(fmt::format("{}{}: {}", padding, key, yaml_scalar(value)));
}

std::string build_export_yaml(const json & overrides)
{
  if (!overrides.is_object() || overrides.empty()) return "";

  const auto nested = nest_flat_key_values(overrides);
  std::vector<std::string> lines;
  for (auto it = nested.begin(); it != nested.end(); ++it) {
    append_yaml_lines(it.value(), it.key(), 0, lines);
  }

  std::string output;
  for (size_t i = 0; i < lines.size(); ++i) {
    output += lines[i];
    if (i + 1 < lines.size()) output += "\n";
  }
  return output;
}

bool write_all(int fd, const std::string & text)
{
  const char * cursor = text.data();
  size_t remaining = text.size();
  while (remaining > 0) {
    const ssize_t written = ::write(fd, cursor, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
  return true;
}

void durable_write_append(const fs::path & path, const std::string & text)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
  if (fd < 0) {
    throw std::runtime_error("open append failed: " + path.string());
  }

  const bool ok = write_all(fd, text) && (::fsync(fd) == 0);
  ::close(fd);
  if (!ok) {
    throw std::runtime_error("append/fsync failed: " + path.string());
  }
}

void durable_write_overwrite(const fs::path & path, const std::string & text)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644);
  if (fd < 0) {
    throw std::runtime_error("open overwrite failed: " + path.string());
  }

  const bool ok = write_all(fd, text) && (::fsync(fd) == 0);
  ::close(fd);
  if (!ok) {
    throw std::runtime_error("overwrite/fsync failed: " + path.string());
  }
}

std::pair<std::string, std::string> build_log_paths(const std::string & config_path)
{
  const auto config = fs::path(config_path);
  std::error_code ec;
  fs::create_directories("logs/web_params", ec);

  const auto timestamp =
    fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  const auto stem = config.stem().string();

  const auto session_log =
    fs::absolute(fs::path("logs/web_params") / (timestamp + "_" + stem + ".jsonl")).string();
  const auto snapshot =
    fs::absolute(fs::path("logs/web_params") / ("latest_" + stem + ".runtime.json")).string();
  return {session_log, snapshot};
}

SessionState build_session(const std::string & config_path)
{
  SessionState session;
  session.config_path = tools::resolve_config_path_string(config_path);

  const auto config_values = parse_config_text(session.config_path);
  for (const auto & spec : specs()) {
    const auto value = normalize_value(read_value_from_text(config_values, spec), spec);
    session.base_values[spec.key] = value;
    session.effective_values[spec.key] = value;
  }

  session.last_update_unix_ms = unix_time_ms();
  const auto [session_log_path, snapshot_path] = build_log_paths(session.config_path);
  session.session_log_path = session_log_path;
  session.snapshot_path = snapshot_path;
  return session;
}

SessionState & require_session_locked(const std::string & config_path)
{
  const auto resolved = tools::resolve_config_path_string(config_path);
  const auto it = g_sessions.find(resolved);
  if (it == g_sessions.end()) {
    throw std::runtime_error("Runtime parameter session not registered: " + resolved);
  }
  return it->second;
}

json build_response_locked(const SessionState & session)
{
  json response;
  response["enabled"] = true;
  response["config_path"] = session.config_path;
  response["version"] = session.version;
  response["last_update_unix_ms"] = session.last_update_unix_ms;
  response["session_log_path"] = session.session_log_path;
  response["snapshot_path"] = session.snapshot_path;
  response["override_count"] = session.overrides.size();
  response["export_yaml"] = build_export_yaml(session.overrides);
  response["overrides"] = nest_flat_key_values(session.overrides);
  response["groups"] = json::array();

  std::unordered_map<std::string, size_t> group_index;
  for (const auto & spec : specs()) {
    size_t index = 0;
    const auto existing = group_index.find(spec.group_id);
    if (existing == group_index.end()) {
      index = response["groups"].size();
      group_index[spec.group_id] = index;
      response["groups"].push_back({
        {"id", spec.group_id},
        {"label", spec.group_label},
        {"items", json::array()},
      });
    } else {
      index = existing->second;
    }

    json item = {
      {"key", spec.key},
      {"label", spec.label},
      {"description", spec.description},
      {"type",
        spec.type == ParamType::kDouble ? "number" :
        spec.type == ParamType::kInt ? "integer" :
        spec.type == ParamType::kBool ? "boolean" :
        spec.type == ParamType::kDoubleArray ? "number_array" :
        spec.type == ParamType::kString ? "string" :
        "enum"},
      {"unit", spec.unit},
      {"value", session.effective_values.at(spec.key)},
      {"base_value", session.base_values.at(spec.key)},
      {"overridden", session.overrides.contains(spec.key)},
    };
    if (
      spec.type == ParamType::kDouble || spec.type == ParamType::kInt ||
      spec.type == ParamType::kDoubleArray)
    {
      item["step"] = ui_step_for(spec);
      item["display_precision"] = ui_precision_for(spec);
      if (const auto min_value = ui_min_for(spec)) item["min"] = *min_value;
      if (const auto max_value = ui_max_for(spec)) item["max"] = *max_value;
    }
    if (!spec.enum_values.empty()) item["choices"] = spec.enum_values;
    response["groups"][index]["items"].push_back(item);
  }

  return response;
}

void persist_snapshot_locked(const SessionState & session)
{
  json payload = {
    {"config_path", session.config_path},
    {"version", session.version},
    {"saved_unix_ms", session.last_update_unix_ms},
    {"overrides", nest_flat_key_values(session.overrides)},
    {"flat_overrides", session.overrides},
    {"effective", nest_flat_key_values(session.effective_values)},
    {"export_yaml", build_export_yaml(session.overrides)},
  };
  durable_write_overwrite(session.snapshot_path, payload.dump(2) + "\n");
}

void persist_change_event_locked(
  const SessionState & session, const std::string & source, const json & changes)
{
  if (changes.empty()) return;

  json event = {
    {"unix_ms", session.last_update_unix_ms},
    {"source", source},
    {"config_path", session.config_path},
    {"version", session.version},
    {"changes", changes},
    {"overrides", nest_flat_key_values(session.overrides)},
    {"flat_overrides", session.overrides},
  };
  durable_write_append(session.session_log_path, event.dump() + "\n");
}

json disabled_response(const std::string & config_path, const std::string & reason)
{
  return {
    {"enabled", false},
    {"config_path", tools::resolve_config_path_string(config_path)},
    {"error", reason},
  };
}
}  // namespace

void register_config(const std::string & config_path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  if (g_sessions.find(resolved) != g_sessions.end()) return;

  try {
    auto session = build_session(resolved);
    persist_snapshot_locked(session);
    tools::logger()->info(
      "[RuntimeParams] registered {} -> log: {} snapshot: {}",
      session.config_path, session.session_log_path, session.snapshot_path);
    g_sessions.emplace(resolved, std::move(session));
  } catch (const std::exception & e) {
    tools::logger()->warn("[RuntimeParams] failed to register {}: {}", resolved, e.what());
  }
}

bool is_registered(const std::string & config_path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  return g_sessions.find(resolved) != g_sessions.end();
}

uint64_t version(const std::string & config_path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  const auto it = g_sessions.find(resolved);
  return it == g_sessions.end() ? 0 : it->second.version;
}

double get_double(const std::string & config_path, const std::string & key)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto & session = require_session_locked(config_path);
  return session.effective_values.at(key).get<double>();
}

int get_int(const std::string & config_path, const std::string & key)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto & session = require_session_locked(config_path);
  return session.effective_values.at(key).get<int>();
}

bool get_bool(const std::string & config_path, const std::string & key)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto & session = require_session_locked(config_path);
  return session.effective_values.at(key).get<bool>();
}

std::string get_string(const std::string & config_path, const std::string & key)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto & session = require_session_locked(config_path);
  return session.effective_values.at(key).get<std::string>();
}

std::vector<double> get_number_array(const std::string & config_path, const std::string & key)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto & session = require_session_locked(config_path);
  return session.effective_values.at(key).get<std::vector<double>>();
}

nlohmann::json describe(const std::string & config_path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  const auto it = g_sessions.find(resolved);
  if (it == g_sessions.end()) {
    return disabled_response(config_path, "runtime parameter session is not registered");
  }
  return build_response_locked(it->second);
}

nlohmann::json apply(
  const std::string & config_path, const nlohmann::json & request, const std::string & source)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  auto it = g_sessions.find(resolved);
  if (it == g_sessions.end()) {
    return disabled_response(config_path, "runtime parameter session is not registered");
  }

  const auto & updates = request.contains("updates") ? request.at("updates") : request;
  if (!updates.is_object()) {
    throw std::runtime_error("runtime parameter updates must be an object");
  }

  auto & session = it->second;
  json changes = json::array();
  bool changed = false;
  for (auto update_it = updates.begin(); update_it != updates.end(); ++update_it) {
    const auto & spec = spec_for(update_it.key());
    const auto normalized = normalize_value(update_it.value(), spec);
    const auto & old_value = session.effective_values.at(spec.key);
    if (old_value == normalized) continue;

    changes.push_back({
      {"key", spec.key},
      {"old_value", old_value},
      {"new_value", normalized},
      {"base_value", session.base_values.at(spec.key)},
    });

    session.effective_values[spec.key] = normalized;
    if (normalized == session.base_values.at(spec.key)) {
      session.overrides.erase(spec.key);
    } else {
      session.overrides[spec.key] = normalized;
    }
    changed = true;
  }

  if (changed) {
    session.last_update_unix_ms = unix_time_ms();
    ++session.version;
    persist_change_event_locked(session, source, changes);
    persist_snapshot_locked(session);
  }

  return build_response_locked(session);
}

nlohmann::json reset(
  const std::string & config_path, const std::vector<std::string> & keys, const std::string & source)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto resolved = tools::resolve_config_path_string(config_path);
  auto it = g_sessions.find(resolved);
  if (it == g_sessions.end()) {
    return disabled_response(config_path, "runtime parameter session is not registered");
  }

  auto & session = it->second;
  json changes = json::array();
  bool changed = false;

  std::vector<std::string> targets = keys;
  if (targets.empty()) {
    for (auto override_it = session.overrides.begin(); override_it != session.overrides.end(); ++override_it) {
      targets.push_back(override_it.key());
    }
  }

  for (const auto & key : targets) {
    const auto & spec = spec_for(key);
    const auto current_it = session.effective_values.find(spec.key);
    if (current_it == session.effective_values.end()) continue;
    const auto & base_value = session.base_values.at(spec.key);
    if (current_it.value() == base_value) {
      session.overrides.erase(spec.key);
      continue;
    }

    changes.push_back({
      {"key", spec.key},
      {"old_value", current_it.value()},
      {"new_value", base_value},
      {"base_value", base_value},
    });
    session.effective_values[spec.key] = base_value;
    session.overrides.erase(spec.key);
    changed = true;
  }

  if (changed) {
    session.last_update_unix_ms = unix_time_ms();
    ++session.version;
    persist_change_event_locked(session, source, changes);
    persist_snapshot_locked(session);
  }

  return build_response_locked(session);
}

}  // namespace tools::runtime_params
