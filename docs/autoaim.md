# auto_aim 模块分析（task 第一阶段）

## 1. 模块目标
`tasks/auto_aim` 负责从检测结果出发，完成：
- 装甲板空间解算（PnP + 坐标变换）
- 目标状态估计（EKF）
- 预测瞄准点选择与弹道补偿
- 击发判定与指令输出

本分析按你的要求，弱化模型调用细节（OpenVINO/ONNX 推理），重点说明算法与控制结算链路。

## 2. 目录内主链路（控制视角）
典型运行链路（见 `src/standard.cpp` / `src/mt_standard.cpp` 调用）：

1. `YOLO` 或 `Detector` 输出 `std::list<Armor>`
2. `Solver::solve(armor)` 将 2D 角点解算到世界坐标
3. `Tracker` 进行目标初始化/关联/EKF 更新，输出 `std::list<Target>`
4. `Aimer` 选择装甲板 + 弹道迭代，输出 `io::Command(yaw, pitch)`
5. `Shooter` 二次击发判定（抖动/容差门控），决定 `shoot`
6. 由上层线程直接 `cboard.send(command)` 或经 `multithread/CommandGener` 发送

## 3. 关键数据对象

### `Armor`（`armor.hpp/.cpp`）
- 含检测角点、类别、颜色、置信度、3D解算结果（`xyz_in_world`, `ypr_in_world`）。
- 既支持传统灯条构造，也支持 YOLO keypoints 构造。
- 是检测层与跟踪层之间的统一中间表示。

### `Target`（`target.hpp/.cpp`）
- 跟踪对象，内部是 EKF 状态。
- 状态向量：`x=[cx,vx, cy,vy, cz,vz, a,w, r,l,h]`。
  - `a,w`：车体角与角速度
  - `r,l,h`：半径、长短轴差、高度差
- 提供 `armor_xyza_list()`，用于给 `Aimer/Planner` 选择可打击装甲板。

## 4. 顶层文件逐一说明

| 文件 | 核心内容 | 在链路中的作用 | 直接依赖 |
|---|---|---|---|
| `CMakeLists.txt` | 构建 `auto_aim` object library，并链接 `tinympcstatic` | 组织模块编译 | OpenVINO、tinympc |
| `armor.hpp` | 颜色/类型/编号枚举；`Lightbar`、`Armor` 数据结构 | 统一检测-解算-跟踪数据接口 | OpenCV/Eigen |
| `armor.cpp` | 传统/YOLO多种 `Armor` 构造逻辑与几何量计算 | 将检测输出规整为可跟踪目标 | `armor.hpp` |
| `detector.hpp` | 传统视觉检测类接口 | 传统方案入口 | `classifier.hpp` |
| `detector.cpp` | 灰度阈值、轮廓、灯条配对、几何筛选、图案分类、重叠消歧、可视化 | 传统检测主流程 | OpenCV, Classifier |
| `classifier.hpp` | 分类器接口（DNN/OpenVINO） | 图案数字识别接口层 | OpenVINO |
| `classifier.cpp` | 图案预处理 + softmax 分类 | 为 `Detector/YOLOV8` 提供编号 | 模型文件 |
| `yolo.hpp` | `YOLOBase` 抽象与 `YOLO` 分发器 | 统一神经网络检测接口 | yolos 子目录 |
| `yolo.cpp` | 按配置选择 `YOLOV5/YOLOV8/YOLO11` | 解耦模型版本 | YAML |
| `solver.hpp` | 位姿解算与重投影接口 | 从像素到世界坐标的关键桥梁 | `armor.hpp` |
| `solver.cpp` | PnP 解算、坐标系变换、yaw优化、重投影误差 | 为 tracker/aimer 提供稳定 3D 观测 | OpenCV/Eigen |
| `target.hpp` | EKF 目标类接口 | 跟踪状态容器 | `extended_kalman_filter` |
| `target.cpp` | 预测模型、观测模型、关联更新、发散检测 | 跟踪核心算法 | math_tools/EKF |
| `tracker.hpp` | 目标状态机与跟踪接口 | 将多装甲观测收敛成单目标 | `solver.hpp`, `target.hpp` |
| `tracker.cpp` | `lost/detecting/tracking/temp_lost/switching` 状态机、目标初始化和更新 | 自瞄控制入口的“目标决策层” | solver/target |
| `aimer.hpp` | 瞄准解算接口 | 由目标状态转云台角 | `target.hpp`, `io::Command` |
| `aimer.cpp` | 装甲板选择策略 + 弹道时间迭代 + yaw/pitch 补偿 | 最终瞄准角结算核心 | trajectory/math_tools |
| `shooter.hpp` | 击发判定接口 | 开火安全门控 | `aimer.hpp`, `io::Command` |
| `shooter.cpp` | 基于指令变化、云台误差、容差的开火判定 | 输出 `shoot` 布尔值 | logger/math_tools |
| `voter.hpp` | 投票计数接口 | 轻量统计工具 | `armor.hpp` |
| `voter.cpp` | `color+name+type` 组合计数 | 可用于稳定识别统计 | - |

## 5. 关键算法与结算逻辑（重点）

### 5.1 解算层：`Solver`
- `solvePnP(IPPE)` 得到 `rvec/tvec`。
- 通过外参把 `camera -> gimbal -> world`。
- `optimize_yaw()`：在给定搜索角域里，按重投影误差选更稳定 `yaw`，降低姿态抖动。

### 5.2 跟踪层：`Target + Tracker`
- `Tracker::set_target()` 按装甲类型设置不同初始化参数（如 outpost/base/balance）。
- `Target::predict()` 使用常速度模型 + 分段白噪声过程噪声。
- `Target::update()` 先在候选装甲板中做关联，再用 `z=[yaw,pitch,distance,angle]` EKF 更新。
- `Tracker` 状态机保证短时丢失不立即丢目标，兼顾稳定与响应。
- `Target::diverged()` 对 `r/l/h` 做物理边界约束，异常即丢目标。

### 5.3 瞄准层：`Aimer`
- `choose_aim_point()` 根据旋转中心角度关系选可击中装甲板。
- 小陀螺时，按“来向角/离向角”策略倾向选择更可击中的板。
- 先估一次弹道飞行时间，再做最多10次“目标预测-弹道重解”迭代，直到飞行时间收敛（阈值 `0.001s`）。
- 输出 `yaw/pitch`（含标定偏移与发射延时补偿）。

### 5.4 击发层：`Shooter`
- `control=true` 且目标有效时，按距离切换容差。
- 仅当：
  - 当前指令相对上一帧变化不突变
  - 云台当前角接近上次指令角
  - `aimer.debug_aim_point.valid=true`
  才允许 `shoot=true`。

## 6. 与子目录关系
- `yolos/`：神经网络检测实现，输出 `Armor`（见子文档）。
- `multithread/`：异步检测与异步发指令（见子文档）。
- `planner/`：MPC 版本控制链路（与 Aimer 并行方案，不是同一控制器）（见子文档）。

## 7. 当前代码中与控制相关的注意点
- `Tracker::track(..., bool use_enemy_color)` 参数目前未实际使用开关逻辑（内部仍固定按 `enemy_color_` 过滤）。
- `Detector::detect()` 中 `cv::imshow("binary_img", ...)` 是无条件显示，部署无图形环境时可能影响运行。
- `Tracker::update_target()` 中前哨站高度补偿写在双层 `for (armors)` 结构里，会重复对全部装甲板遍历补偿，建议后续单独复核。

## 8. 下一步建议（针对你当前拆分方式）
- 下一阶段可继续对 `tasks/auto_buff` 按同样格式拆分。
- 或先针对本目录做“调用时序图 + 配置参数清单（yaml键到逻辑）”补充，便于调参。

