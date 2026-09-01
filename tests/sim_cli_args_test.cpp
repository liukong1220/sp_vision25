// sim_io::parse_park_angle / parse_enemy_team 的独立单元测试。
//
// 这两个函数原来藏在 sim_auto_aim.cpp 的匿名 namespace 里，唯一的"验证"方式是跑一遍
// 闭环看 detected_frames 像不像话——而它们真正要挡的是一串边界取值：空串、尾部残留、
// nan/inf、大小写。闭环跑绿只说明主路径对，说明不了 "90abc" 被静默截成 90。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

#include "simulation/io/cli_args.hpp"

namespace
{
int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string & name, const std::string & detail = "")
{
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("%-56s %s", name.c_str(), ok ? "ok" : "失败");
  if (!detail.empty()) std::printf("  %s", detail.c_str());
  std::printf("\n");
}

// 期望解析成功且取值相等（按位比较：这些都是十进制字面量，不涉及累积误差）。
void expect_angle(const std::string & text, double want)
{
  // 哨兵值取一个绝不会是合法解析结果的数，才能区分"函数返回 false 但仍写了 out"
  // 与"函数确实没碰 out"。
  double got = -12345.0;
  const bool ok = sim_io::parse_park_angle(text, &got);
  char detail[96];
  std::snprintf(detail, sizeof(detail), "want=%.10g got=%.10g", want, got);
  check(ok && got == want, "接受 \"" + text + "\"", detail);
}

void expect_angle_rejected(const std::string & text)
{
  double got = -12345.0;
  const bool ok = sim_io::parse_park_angle(text, &got);
  // 除了返回 false，还必须没有污染 out：调用方在失败分支里保留的是自己的默认值。
  check(!ok && got == -12345.0, "拒绝 \"" + text + "\" 且不写 out");
}

void expect_team(const std::string & text, std::uint8_t want)
{
  std::uint8_t got = 0xEE;
  const bool ok = sim_io::parse_enemy_team(text, &got);
  char detail[64];
  std::snprintf(detail, sizeof(detail), "want=%u got=%u", want, got);
  check(ok && got == want, "阵营接受 \"" + text + "\"", detail);
}

void expect_team_rejected(const std::string & text)
{
  std::uint8_t got = 0xEE;
  const bool ok = sim_io::parse_enemy_team(text, &got);
  check(!ok && got == 0xEE, "阵营拒绝 \"" + text + "\" 且不写 out");
}
}  // namespace

int main()
{
  std::printf("== parse_park_angle：合法取值 ==\n");
  expect_angle("90", 90.0);
  expect_angle("-2", -2.0);
  expect_angle("0", 0.0);
  expect_angle("90.0", 90.0);
  expect_angle("-2.5", -2.5);
  expect_angle("+90", 90.0);
  // 允许前导空白是 std::stod 的既有行为，这里固定住它，避免以后换实现时静默改变。
  expect_angle("  90", 90.0);
  expect_angle("1e2", 100.0);
  // 超出物理量程但仍是有限数：解析层放行，量程检查是调用方的事。
  expect_angle("720", 720.0);
  expect_angle("-720", -720.0);

  std::printf("\n== parse_park_angle：必须拒绝 ==\n");
  // 空串是"用户没给这个参数"的编码，绝不能被当成 0。
  expect_angle_rejected("");
  // 尾部残留：静默截断会把明显的笔误变成一个看起来合理的指向。
  expect_angle_rejected("90abc");
  expect_angle_rejected("90 deg");
  expect_angle_rejected("90,0");
  expect_angle_rejected("90.0.0");
  expect_angle_rejected("--90");
  // 非有限值：stod 接受它们，而驻留角要进三角函数和命令编码。
  expect_angle_rejected("nan");
  expect_angle_rejected("NaN");
  expect_angle_rejected("inf");
  expect_angle_rejected("-inf");
  expect_angle_rejected("infinity");
  // 纯空白、纯符号、纯单位：全都不是数值。
  expect_angle_rejected("   ");
  expect_angle_rejected("+");
  expect_angle_rejected("-");
  expect_angle_rejected("deg");
  expect_angle_rejected("abc");
  // 超出 double 量程：stod 抛 out_of_range，必须被 catch 成"未给定"而不是逃出去。
  expect_angle_rejected("1e400");
  expect_angle_rejected("-1e400");

  std::printf("\n== parse_park_angle：解析结果本身必须有限 ==\n");
  {
    double got = 0.0;
    const bool ok = sim_io::parse_park_angle("90", &got);
    check(ok && std::isfinite(got), "成功路径产出有限值");
  }

  std::printf("\n== parse_enemy_team ==\n");
  expect_team("red", sim_io::GT_TEAM_RED);
  expect_team("blue", sim_io::GT_TEAM_BLUE);
  expect_team("any", sim_io::GT_TEAM_ANY);
  // 三个取值必须互不相同，否则"指定一方"根本没起作用。
  check(
    sim_io::GT_TEAM_RED != sim_io::GT_TEAM_BLUE && sim_io::GT_TEAM_RED != sim_io::GT_TEAM_ANY &&
      sim_io::GT_TEAM_BLUE != sim_io::GT_TEAM_ANY,
    "GT_TEAM_RED / BLUE / ANY 三者互异");

  // 大小写与空白都不做归一化：红蓝三号步兵共用 armor label=3，把 "Red" 悄悄当成
  // 未给定（走 ANY）会让评估器把自家车当成评估对象。宁可报错退出。
  expect_team_rejected("");
  expect_team_rejected("Red");
  expect_team_rejected("RED");
  expect_team_rejected("Blue");
  expect_team_rejected("ANY");
  expect_team_rejected(" red");
  expect_team_rejected("red ");
  expect_team_rejected("r");
  expect_team_rejected("redblue");
  expect_team_rejected("0");
  expect_team_rejected("none");

  std::printf("\n共 %d 项检查，%d 项失败\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
