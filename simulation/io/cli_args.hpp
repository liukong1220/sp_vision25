#ifndef SIMULATION_IO__CLI_ARGS_HPP
#define SIMULATION_IO__CLI_ARGS_HPP

// sim_* 可执行文件共用的命令行取值解析。
//
// 单独成一个头文件的唯一原因是**可单测**：这些函数原来定义在 sim_auto_aim.cpp 的匿
// 名 namespace 里，只能靠端到端跑一遍闭环来验证，而它们要覆盖的恰恰是一堆边界取值
// （空串、"nan"、"90abc"、无穷、只给一半）。驻留角解析错的那一次代价很直接：云台被
// 摆到 yaw=0 而目标在 yaw≈+90，detected_frames 从 447 掉到 4，看起来像感知坏了。

#include <cmath>
#include <cstdint>
#include <exception>
#include <string>

#include "sim_ground_truth.hpp"

namespace sim_io
{
// 解析驻留角。空 = 未给定；解析失败或非有限值也算未给定（由调用方报错）。
//
// 必须自己解析字符串，不能用 OpenCV CommandLineParser 的 get<double>()：它表达不了
// "这个数值键没有被给出"。给定 "nan" 作默认值时 get<double>() 会静默返回 0.0，而
// has() 对任何带默认值的键恒为 true，两条路都拿不回"用户没给"。所以数值可选参数在
// 那个 parser 里只能声明成**字符串**键、以空串为默认值，再走这里。
//
// 拒绝尾部残留（used != size）：把 "90abc" 悄悄当成 90 会让一个明显的笔误变成一个
// 看起来合理的指向。"nan" / "inf" 能被 stod 接受，所以还要显式挡掉非有限值——驻留角
// 是要被送进三角函数和命令编码的。
inline bool parse_park_angle(const std::string & text, double * out)
{
  if (text.empty()) return false;
  try {
    std::size_t used = 0;
    const double v = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(v)) return false;
    *out = v;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

// 真值评估的敌方队伍必须显式确定：仿真场景里红蓝三号步兵共用 armor label=3，
// 只按 label 匹配会把自家车当评估对象（见 GroundTruthEvaluator 的说明）。
// "any" 只允许用于诊断，正式评估必须指定一方。
inline bool parse_enemy_team(const std::string & text, std::uint8_t * out)
{
  if (text == "red") {
    *out = GT_TEAM_RED;
    return true;
  }
  if (text == "blue") {
    *out = GT_TEAM_BLUE;
    return true;
  }
  if (text == "any") {
    *out = GT_TEAM_ANY;
    return true;
  }
  return false;
}

}  // namespace sim_io

#endif  // SIMULATION_IO__CLI_ARGS_HPP
