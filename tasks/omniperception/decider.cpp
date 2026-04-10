#include "decider.hpp"

#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace omniperception
{
Decider::Decider(const std::string & config_path) : detector_(config_path), count_(0)
{
  auto yaml = tools::load(config_path);
  img_width_ = yaml["image_width"].as<double>();
  img_height_ = yaml["image_height"].as<double>();
  fov_h_ = yaml["fov_h"].as<double>();
  fov_v_ = yaml["fov_v"].as<double>();
  new_fov_h_ = yaml["new_fov_h"].as<double>();
  new_fov_v_ = yaml["new_fov_v"].as<double>();
  enemy_color_ =
    (yaml["enemy_color"].as<std::string>() == "red") ? auto_aim::Color::red : auto_aim::Color::blue;
  mode_ = yaml["mode"].as<double>();

  const auto vision_fusion = yaml["vision_fusion"];
  if (vision_fusion) {
    enable_goal_suggestion_ = vision_fusion["enable_goal_suggestion"]
                                ? vision_fusion["enable_goal_suggestion"].as<bool>()
                                : enable_goal_suggestion_;
    left_goal_index_ =
      vision_fusion["left_goal_index"] ? vision_fusion["left_goal_index"].as<int>() : left_goal_index_;
    front_goal_index_ = vision_fusion["front_goal_index"]
                          ? vision_fusion["front_goal_index"].as<int>()
                          : front_goal_index_;
    right_goal_index_ = vision_fusion["right_goal_index"]
                          ? vision_fusion["right_goal_index"].as<int>()
                          : right_goal_index_;
    close_range_goal_index_ = vision_fusion["close_range_goal_index"]
                                ? vision_fusion["close_range_goal_index"].as<int>()
                                : close_range_goal_index_;
    front_yaw_abs_threshold_rad_ =
      vision_fusion["front_yaw_abs_threshold_deg"]
        ? vision_fusion["front_yaw_abs_threshold_deg"].as<double>() / 57.3
        : front_yaw_abs_threshold_rad_;
    close_range_threshold_m_ = vision_fusion["close_range_threshold_m"]
                                 ? vision_fusion["close_range_threshold_m"].as<double>()
                                 : close_range_threshold_m_;
  }
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
  io::USBCamera & usbcam2, io::Camera & back_camera)
{
  Eigen::Vector2d delta_angle;
  io::USBCamera * cams[] = {&usbcam1, &usbcam2};

  cv::Mat usb_img;
  std::chrono::steady_clock::time_point timestamp;
  if (count_ < 0 || count_ > 2) {
    throw std::runtime_error("count_ out of valid range [0,2]");
  }
  if (count_ == 2) {
    back_camera.read(usb_img, timestamp);
  } else {
    cams[count_]->read(usb_img, timestamp);
  }
  auto armors = yolo.detect(usb_img);
  auto empty = armor_filter(armors);

  if (!empty) {
    if (count_ == 2) {
      delta_angle = this->delta_angle(armors, "back");
    } else {
      delta_angle = this->delta_angle(armors, cams[count_]->device_name);
    }

    tools::logger()->debug(
      "[{} camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      (count_ == 2 ? "back" : cams[count_]->device_name), delta_angle[0], delta_angle[1],
      armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    count_ = (count_ + 1) % 3;

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  count_ = (count_ + 1) % 3;
  // 如果没有找到目标，返回默认命令
  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera)
{
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  back_cammera.read(img, timestamp);
  auto armors = yolo.detect(img);
  auto empty = armor_filter(armors);

  if (!empty) {
    auto delta_angle = this->delta_angle(armors, "back");
    tools::logger()->debug(
      "[back camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      delta_angle[0], delta_angle[1], armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) {
    return io::Command{false, false, 0, 0};
  }

  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  tools::logger()->info(
    "omniperceptron find {},delta yaw is {:.4f}", auto_aim::ARMOR_NAMES[dr.armors.front().name],
    dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
};

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  Eigen::Vector2d delta_angle;
  if (camera == "left") {
    delta_angle[0] = 62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
    return delta_angle;
  }

  else if (camera == "right") {
    delta_angle[0] = -62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
    return delta_angle;
  }

  else {
    delta_angle[0] = 170 + (54.2 / 2) - armors.front().center_norm.x * 54.2;
    delta_angle[1] = armors.front().center_norm.y * 44.5 - 44.5 / 2;
    return delta_angle;
  }
}

bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return true;
  // 过滤非敌方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 25赛季没有5号装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::five; });
  // 不打工程
  // armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::two; });
  // 不打前哨站
  armors.remove_if(
    [&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::outpost; });

  // 过滤掉刚复活无敌的装甲板
  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
           invincible_armor_.end();
  });

  return armors.empty();
}

void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return;

  const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;

  if (!armors.empty()) {
    for (auto & armor : armors) {
      armor.priority = priority_map.at(armor.name);
    }
  }
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return;

  // 对每个 DetectionResult 调用 armor_filter 和 set_priority
  for (auto & dr : detection_queue) {
    armor_filter(dr.armors);
    set_priority(dr.armors);

    // 对每个 DetectionResult 中的 armors 进行排序
    dr.armors.sort(
      [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
  }

  // 根据优先级对 DetectionResult 进行排序
  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      return a.armors.front().priority < b.armors.front().priority;
    });
}

VisionTargetInfo Decider::get_target_info(
  const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets)
{
  // 当前没有装甲板或目标轨迹时，直接返回空结果，
  // 让上层融合节点自己决定是否继续保持接管。
  if (armors.empty() || targets.empty()) return VisionTargetInfo{};

  auto target = targets.front();

  for (const auto & armor : armors) {
    if (armor.name == target.name) {
      VisionTargetInfo info;
      info.valid = true;
      // 这里延续原有协议的 +1 约定，
      // 这样不会和枚举下标产生歧义，也便于旧链路平滑迁移。
      info.target_id = static_cast<int>(armor.name) + 1;  // 避免歧义 +1（与原协议保持一致）
      info.confidence = 1.0;
      info.position_gimbal = armor.xyz_in_gimbal;
      return info;
    }
  }

  return VisionTargetInfo{};
}

int Decider::suggest_goal_index(const VisionTargetInfo & target_info) const
{
  // 第一代策略只在“目标位置可信”时给导航建议，
  // 如果视觉当前只有云台角指令，没有稳定空间位置，就让行为树回退到锚点。
  if (!enable_goal_suggestion_ || !target_info.valid) return -1;

  const double x = target_info.position_gimbal.x();
  const double y = target_info.position_gimbal.y();
  const double distance = target_info.position_gimbal.norm();

  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(distance) || distance <= 1e-6) {
    return -1;
  }

  // gimbal 坐标系下使用 x 前、y 左 的常见约定，
  // 因此 yaw > 0 表示目标落在左前方，yaw < 0 表示右前方。
  const double yaw = std::atan2(y, x);

  // 近距离目标优先推荐贴近压制位，避免已经贴脸的目标还继续走大范围巡逻。
  if (close_range_goal_index_ >= 0 && distance <= close_range_threshold_m_) {
    return close_range_goal_index_;
  }

  // 前向目标直接推荐正面压制位；左右侧目标则按符号分流到左右支援点。
  if (std::abs(yaw) <= front_yaw_abs_threshold_rad_) {
    return front_goal_index_;
  }
  return yaw > 0.0 ? left_goal_index_ : right_goal_index_;
}

io::VisionTargetState Decider::build_vision_target_state(
  const io::Command & command, const VisionTargetInfo & target_info) const
{
  io::VisionTargetState state;
  state.tracking = command.control;
  state.nav_hold = command.control;
  state.fire_permitted = command.shoot;
  state.target_id = target_info.target_id;
  state.confidence = target_info.confidence;
  state.target_yaw = command.yaw;
  state.target_pitch = command.pitch;

  if (!target_info.valid) {
    return state;
  }

  state.target_distance = target_info.position_gimbal.norm();
  state.target_position_gimbal = target_info.position_gimbal;
  state.suggested_goal_index = suggest_goal_index(target_info);
  return state;
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
  invincible_armor_.clear();

  if (invincible_enemy_ids.empty()) return;

  for (const auto & id : invincible_enemy_ids) {
    tools::logger()->info("invincible armor id: {}", id);
    invincible_armor_.push_back(auto_aim::ArmorName(id - 1));
  }
}

void Decider::get_auto_aim_target(
  std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
  if (auto_aim_target.empty()) return;

  std::vector<auto_aim::ArmorName> auto_aim_targets;

  for (const auto & target : auto_aim_target) {
    if (target <= 0 || static_cast<size_t>(target) > auto_aim::ARMOR_NAMES.size()) {
      tools::logger()->warn("Received invalid auto_aim target value: {}", int(target));
      continue;
    }
    auto_aim_targets.push_back(static_cast<auto_aim::ArmorName>(target - 1));
    tools::logger()->info("nav send auto_aim target is {}", auto_aim::ARMOR_NAMES[target - 1]);
  }

  if (auto_aim_targets.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
           auto_aim_targets.end();
  });
}

}  // namespace omniperception
