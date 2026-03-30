# auto_aim/planner/tinympc 分析

## 1. 目录职责
该目录是独立的 TinyMPC 求解器实现，给 `Planner` 提供在线优化能力（状态跟踪 + 输入约束）。

## 2. 文件级说明

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 构建静态库 `tinympcstatic`，并支持可选代码生成文件拷贝 |
| `types.hpp` | 定义核心数据结构：`TinySolver/TinyWorkspace/TinySettings/TinyCache/TinySolution` |
| `tiny_api_constants.hpp` | 默认求解参数宏（残差阈值、最大迭代、约束开关） |
| `tiny_api.hpp` | 对外 C 接口声明：`tiny_setup/tiny_solve/tiny_set_*` |
| `tiny_api.cpp` | API 实现，完成求解器初始化、维度校验、参数写入、参考轨迹写入 |
| `admm.hpp` | ADMM求解流程与投影函数声明 |
| `admm.cpp` | 核心求解循环：backward/forward/slack/dual/cost update + termination |
| `rho_benchmark.hpp` | 自适应 rho 相关结构与函数声明 |
| `rho_benchmark.cpp` | 计算原始/对偶残差并估计新 rho，更新缓存矩阵 |
| `codegen.hpp` | 代码生成接口声明 |
| `codegen.cpp` | 导出嵌入式使用的 solver 数据与示例代码 |
| `error.hpp` | 统一错误输出宏 |

## 3. 算法结构（控制重点）
`admm.cpp` 中主循环逻辑：

1. `backward_pass_grad`：Riccati 反向递推更新线性项
2. `forward_pass`：基于 LQR 反馈滚动得到 `x/u`
3. `update_slack`：对 box / SOC / linear 约束做投影
4. `update_dual`：更新增广拉格朗日对偶变量
5. `update_linear_cost`：刷新 ADMM 下一次线性代价项
6. `termination_condition`：检查原始/对偶残差是否收敛
7. （可选）`adaptive_rho`：周期性重估 rho 并更新缓存矩阵

这一套是 `Planner` 能做实时 MPC 的基础。

## 4. 与 `Planner` 的接口关系
- `Planner::setup_yaw_solver/setup_pitch_solver` 会调用：
  - `tiny_setup`
  - `tiny_set_bound_constraints`
  - `tiny_set_x0`
  - `tiny_solve`
- `Planner` 每一帧把参考轨迹写入 `work->Xref`，然后执行求解读取中点控制结果。

## 5. 备注
- 该子目录是通用 MPC 求解器实现，不含装甲板业务逻辑。
- 业务逻辑（目标预测、弹道、开火门限）都在上一层 `planner.cpp`。

