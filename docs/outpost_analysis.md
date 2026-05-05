# Outpost 识别、跟踪与控制算法解析

以下分析完全基于当前 `/home/aw/sp_vision25` 仓库代码整理，目标是把项目里与 `outpost` 相关的识别、解算、跟踪、控制链路完整拆开，方便后续做参数整定、算法验收和实车定位。

## 1. 结论先行

当前项目里的 `outpost` 不是一套完全独立的新系统，而是在通用自动瞄准链路上做了三层特化：

1. 识别层：
   `outpost` 首先只是一个 `ArmorName::outpost` 类别，检测结果仍然是 4 个角点的装甲板观测，没有直接输出转轴中心或角速度。
2. 跟踪层：
   这里是前哨算法最核心的特化。项目把前哨建模为 `3` 块装甲板、同一半径、不同高度的旋转目标，并为它写了专用的匹配、物理板号重映射、固定中心预测和角速度锁定逻辑。
3. 控制层：
   当前项目同时保留 `Aimer` 和 `Planner(MPC)` 两套火控思路。
   其中前哨相关策略最完整的是 `Planner` 分支：支持前哨独立角窗口、独立延迟、板级高度补偿、命中时刻迭代和开火相位门。

一句话概括当前工程：

```text
前哨 = 通用检测/解算 + 前哨专用 3 板跟踪 + 火控框架中的前哨专用策略
普通装甲板 = 通用检测/解算 + 2/4 板目标跟踪 + 通用火控
```

## 2. 代码入口与主链路

当前工程自动瞄准主要有三条和前哨强相关的入口。

### 2.1 `src/standard.cpp`

这是普通自瞄入口，使用：

- `io::CBoard`
- `auto_aim::YOLO`
- `auto_aim::Solver`
- `auto_aim::Tracker`
- `auto_aim::Aimer`

当前这条链路的实际调用是：

```text
相机图像
  -> YOLO 检测
  -> Solver 解算
  -> Tracker 跟踪
  -> Aimer 计算 yaw/pitch
  -> CBoard.send(command)
```

要特别注意：

1. `standard.cpp` 虽然实例化了 `Shooter`，但当前并没有真正调用 `shooter.shoot()`；
2. 这意味着这条入口主要负责出瞄准角，不是当前项目里前哨验收最完整的分支。

### 2.2 `src/standard_mpc.cpp`

这是 MPC 控制入口，使用：

- `io::Gimbal`
- `auto_aim::YOLO`
- `auto_aim::Solver`
- `auto_aim::Tracker`
- `auto_aim::Planner`

链路形式是：

```text
感知线程：相机 -> YOLO -> Solver -> Tracker -> Target
规划线程：Target -> Planner(MPC) -> Gimbal.send(...)
```

### 2.3 `src/auto_aim_debug_mpc.cpp`

这是最适合做前哨调试和验收的入口。它在 `standard_mpc.cpp` 的基础上额外提供了：

- 网页调试器
- 弹道诊断面板
- `planner` 和 `tracker` 内部量可视化
- 前哨选板、物理板号、命中时刻、相位门、匹配得分等调试量输出

因此，当前工程里 `outpost` 的完整路径可以概括为：

```text
YOLO / Detector 标出 outpost
  -> Solver 把 2D 角点解算成 3D 装甲板观测
  -> Tracker 按前哨 3 板模型跟踪
  -> Planner 或 Aimer 选择应击打的板
  -> 输出 yaw / pitch
  -> 若走 Planner 分支，再额外给出 fire
```

## 3. 识别层解析

### 3.1 类别定义

当前项目的装甲板类别定义在 `tasks/auto_aim/armor.hpp`：

```cpp
enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};
```

同时，`armor_properties` 把前哨明确登记为：

```text
{blue/red/extinguish, outpost, small}
```

这意味着当前工程里：

1. 前哨在检测层就是一个 `ArmorName::outpost`
2. 它在几何模板上按小装甲板处理

### 3.2 检测器如何识别 outpost

项目通过 `tasks/auto_aim/yolo.cpp` 统一封装检测器，根据配置里的 `yolo_name` 选择：

- `YOLOV5`
- `YOLOV8`
- `YOLO11`

当前主配置 `configs/standard3.yaml` 使用的是：

```yaml
yolo_name: yolov5
device: GPU
use_traditional: true
```

也就是说，主链路实际是：

```text
YOLOV5 网络检测
  + 可选传统方法二次角点矫正
```

以 `tasks/auto_aim/yolos/yolov5.cpp` 为例，网络输出包含：

1. 颜色类别
2. 编号类别
3. 4 个角点
4. 置信度

对前哨来说，识别层输出的仍然只是：

```text
“这是一块 outpost 装甲板 + 这块板的四角点”
```

并不会直接给出：

- 前哨转轴中心
- 三块板的整体编号
- 角速度

### 3.3 识别后处理

`YOLOV5::parse()` 的主要流程是：

1. 从网络输出中读取颜色、类别、角点和置信度；
2. 通过 NMS 过滤重复检测；
3. 构造 `Armor`；
4. 用 `check_name()` 过滤 `not_armor` 与低置信度目标；
5. 用 `check_type()` 检查类别和大小装甲板的一致性；
6. 如果启用了 `use_traditional`，用 `detector_.detect(*it, bgr_img)` 做角点二次矫正；
7. 生成归一化中心点 `center_norm`。

与前哨直接相关的点有三个：

1. `outpost` 在类别系统里是合法目标；
2. `check_type()` 会禁止 `outpost` 走大装甲板类型；
3. 传统检测器在这里承担的是“角点优化”或独立传统检测，不是前哨专用识别器。

### 3.4 识别层对 outpost 的真实作用

从工程角度看，检测层对前哨的职责只有三件事：

1. 把目标标成 `ArmorName::outpost`
2. 给出尽量稳定的四角点
3. 给后续 `Solver` 和 `Tracker` 提供一块可解算的装甲板观测

也就是说，当前工程里前哨的主要难点不在 detector，而在：

1. 3 板几何恢复
2. 板号与物理高度重映射
3. 选板与开火时机控制

## 4. 跟踪层解析

### 4.1 跟踪器状态机

`tasks/auto_aim/tracker.cpp` 的外层状态机包括：

- `lost`
- `detecting`
- `tracking`
- `temp_lost`
- `switching`

其中前四个是自动瞄准主链路常用状态，`switching` 主要用于全向感知切目标流程。

基本逻辑是：

1. `lost -> detecting`
   首次找到可信目标
2. `detecting -> tracking`
   连续检测次数达到 `min_detect_count`
3. `tracking -> temp_lost`
   当前帧没找到匹配目标
4. `temp_lost -> lost`
   临时丢失超过阈值

对前哨还有一个专门差异：

1. `temp_lost` 阈值会切到 `outpost_max_temp_lost_count`
2. 也就是前哨允许比普通目标更长时间的短暂丢失

### 4.2 初始化

`Tracker::set_target()` 会根据装甲板类型构造不同 `Target`：

1. 平衡步兵：
   `2` 板模型
2. 前哨：
   `3` 板模型
3. 基地：
   `3` 板模型
4. 其他普通目标：
   `4` 板模型

对前哨的初始化参数是：

```text
P0 diag = [1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0]
radius = outpost_radius
armor_num = 3
armor_z_offsets = outpost_armor_z_offsets
fixed_center_rotation_model = outpost_fixed_center_rotation_model
spin_speed_lock = outpost_spin_speed_lock
```

默认主配置 `configs/standard3.yaml` 中对应的是：

```yaml
outpost_radius: 0.2765
outpost_armor_z_offsets: [0.0, -0.102, 0.102]
outpost_fixed_center_rotation_model: true
outpost_spin_speed_lock: 2.51
```

### 4.3 状态定义

`Target` 内部使用统一的 11 维 EKF 状态：

```text
x = [cx, vcx, cy, vcy, cz, vcz, yaw, vyaw, r, l, h]
```

含义如下：

- `cx, cy, cz`：旋转中心位置
- `vcx, vcy, vcz`：中心速度
- `yaw`：参考装甲板相位
- `vyaw`：角速度
- `r`：基础半径
- `l, h`：
  对普通 4 板目标表示长短半径差和高度差

对前哨要特别注意：

1. 当前前哨仍然复用这套 11 维状态接口；
2. 但三块板的真实高度关系，主要不是靠 `h` 建模，而是靠 `outpost_armor_z_offsets`；
3. 也就是说，前哨在状态向量上“共用外形”，在几何解释上“走专门分支”。

### 4.4 outpost 的装甲板几何建模

`Target::h_armor_xyz()` 里，普通 4 板目标采用：

```text
偶数板：半径 r，高度 cz
奇数板：半径 r + l，高度 cz + h
```

而前哨由于 `armor_num = 3`，不会进入 `use_l_h` 分支，所以它的三块板几何是：

```text
x_i = cx - r * cos(yaw_i)
y_i = cy - r * sin(yaw_i)
z_i = cz + armor_z_offset(i)
yaw_i = yaw + i * 2pi / 3
```

这里的关键点有两个：

1. 三块板共用同一个旋转半径 `r`
2. 三块板的高度差来自 `armor_z_offset(i)`，而不是状态里的 `h`

更进一步，前哨还引入了：

```text
local armor id
vs
physical armor id
```

对应接口是：

- `physical_armor_id()`
- `set_armor_id_offset()`
- `armor_z_offset()`

这样做的目的，是把：

1. “当前几何顺序中的第几块板”
2. “真实物理上高板/低板是哪一块”

这两件事分开处理。

### 4.5 outpost 的姿态建模

当前工程里，前哨与普通装甲板最重要的姿态差异发生在 `Solver::reproject_armor()`：

```text
outpost      -> pitch = -15°
普通装甲板   -> pitch = +15°
```

这不是 EKF 状态里单独估计的 pitch，而是重投影和 yaw 优化时的几何假设。

它会直接影响：

1. 重投影误差
2. tracker 匹配质量
3. 板位空间估计

因此，当前工程里前哨并不是“普通小装甲板换个标签”这么简单，至少从重投影几何开始就已经分叉。

### 4.6 预测模型

`Target::predict()` 中，前哨有两种工作模式。

#### 模式 A：固定中心旋转模型

当 `outpost_fixed_center_rotation_model=true` 时：

1. `vx, vy, vz` 会被压回 `0`
2. 中心默认不平移
3. 只保留 `yaw += vyaw * dt`
4. 过程噪声采用一组比普通目标更小的固定值

这等价于假设：

```text
前哨主要绕固定中心转，而不是像车体那样在平面上机动
```

#### 模式 B：普通常速度模型

如果关闭固定中心模型，就退回统一常速度 EKF 预测。

此外，前哨还有一个关键逻辑：

1. 当目标已经收敛；
2. 且目标是前哨；
3. 且 `|vyaw| > 2`

就会把角速度锁到：

```text
+outpost_spin_speed_lock 或 -outpost_spin_speed_lock
```

这可以抑制前哨角速度估计在稳定跟踪后继续漂动。

### 4.7 过程噪声

当前工程里，前哨过程噪声不是旧项目里的一组 `qxyz_outpost / qyaw_output / q_outpost_dz` 风格参数，而是写在 `Target::predict()` 里的两套实现。

对前哨：

1. 如果走固定中心模型，过程噪声直接硬编码为一组很小的对角项；
2. 如果走普通常速度模型，则使用：
   `v1 = 10`
   `v2 = 0.1`

对普通目标：

```text
v1 = 100
v2 = 400
```

这说明当前工程的设计倾向非常明确：

1. 相信前哨中心比普通车更稳定
2. 相信前哨角速度变化远小于普通小陀螺目标

### 4.8 观测模型

`Target::update_ypda()` 的观测量是：

```text
z = [yaw_los, pitch_los, distance, armor_yaw]
```

其中：

1. `yaw_los / pitch_los / distance` 来自 `armor.ypd_in_world`
2. `armor_yaw` 来自 `armor.ypr_in_world[0]`

当前工程里观测噪声不是固定常数，而是经验相关项：

```text
R_dig = [
  4e-3,
  4e-3,
  log(|delta_angle| + 1) + 1,
  log(|distance| + 1) / 200 + 9e-2
]
```

这意味着：

1. 前两项角度观测给固定噪声；
2. 距离噪声会随板相位偏角变化；
3. 装甲板 yaw 观测噪声会随距离变化。

从工程上看，这是一种“轻量自适应”的经验建模，不是旧文档里的单独前哨量测噪声参数化。

### 4.9 匹配与门控

这里是当前工程里前哨最专门化的部分。

普通目标的更新方式更简单：

1. 先预测 `target_`
2. 找到同名同类型候选
3. `solver.solve()`
4. 调 `target_.update(armor)`，由默认就近关联完成匹配

前哨则不会直接走这条路，而是先执行：

```text
select_best_outpost_match()
```

它会同时枚举：

1. 当前局部板号 `id`
2. 物理板号偏移 `offset`

对每种组合计算综合代价，代价项包括：

1. 重投影误差
2. 视线 yaw 误差
3. pitch 误差
4. 距离误差
5. 装甲板 yaw 误差
6. xy 位置误差
7. z 位置误差
8. 与上一帧 `id`、`offset` 的连续性惩罚

然后再用 `accept_outpost_match()` 做门限筛选：

```text
reprojection_error < 90 px
xy_error < 0.40 m
z_error < 0.20 m
score < 36
```

只有通过筛选后，才会：

1. `set_armor_id_offset(best_match.offset, target_.last_id)`
2. `target_.update(*best_match.armor_it, best_match.id)`

这一步把：

1. 哪块局部板被看到
2. 当前物理高低板映射关系

放到了同一个匹配框架里统一求解。

### 4.10 ROI 预测

当前工程没有旧项目里那种“根据上一帧目标预测自适应 ROI”的单独前哨策略。

目前实际生效的是 YOLO 分支里的静态 ROI：

- `use_roi`
- `roi.x`
- `roi.y`
- `roi.width`
- `roi.height`

它由配置和运行时参数共同控制，属于检测加速手段，不是 tracker 驱动的目标预测 ROI。

因此，这一版工程里前哨 ROI 的主要特点是：

1. 可以配置
2. 可以热更新
3. 但不是按目标状态动态滑窗

## 5. 控制层解析

### 5.1 控制入口

当前工程有两套自动瞄准控制入口。

#### 入口 A：`Aimer`

主要在：

- `src/standard.cpp`
- `src/uav.cpp`
- `src/sentry.cpp`

这条链路以“单次求 yaw/pitch”为主，更接近传统火控。

#### 入口 B：`Planner(MPC)`

主要在：

- `src/standard_mpc.cpp`
- `src/auto_aim_debug_mpc.cpp`
- `src/auto_debug.cpp`

这条链路会：

1. 构造未来参考轨迹
2. 分别对 yaw / pitch 解 MPC
3. 同时输出位置、速度、加速度和 `fire`

对前哨来说，这一分支明显更完整。

### 5.2 弹道求解

两套方案都基于 `tools::Trajectory` 做弹道解算，但用法不同。

#### `Aimer`

`Aimer::aim()` 的流程是：

1. 按 `decision_speed` 选择高低速延迟；
2. 预测目标到当前决策时刻；
3. 用 `choose_aim_point()` 选板；
4. 用 `tools::Trajectory` 计算飞行时间；
5. 最多做 10 次“预测到命中时刻 -> 重选板 -> 重算飞行时间”的迭代；
6. 输出最终 yaw / pitch。

#### `Planner`

`Planner::plan()` 的流程是：

1. 先按 `delay_time` 预测到当前决策时刻；
2. 用 `solve_hit_target()` 做“选板 -> 算飞行时间 -> 预测到命中时刻 -> 再选板”的固定点迭代；
3. 以命中时刻目标为基准生成控制轨迹；
4. 再解 MPC。

当前实现里：

1. `Aimer` 更像“单次求解 + 轻量迭代”
2. `Planner` 更像“命中时刻求解 + 连续控制”

### 5.3 装甲板选择策略

#### `Aimer::choose_aim_point()`

当前逻辑是：

1. 如果 `target.jumped == false`，直接打当前板；
2. 非小陀螺时，只在 `60°` 可见角内选板，并尽量锁住 `lock_id_`；
3. 小陀螺时，用 `comming_angle / leaving_angle` 选“正在进入窗口”的板；
4. 如果窗口内没有合适候选，退化为 `fallback_to_closest()`。

对前哨，`Aimer` 还有一个现状：

```text
前哨被强制视为 spin gate 目标
coming/leaving 角仍然写死为 70° / 30°
```

也就是说，`Aimer` 分支里的前哨窗口目前还是硬编码。

#### `Planner::choose_aim_selection()`

`Planner` 的选板逻辑和 `Aimer` 框架相似，但更完整：

1. 仍然区分“未 jump”和“已 jump”
2. 仍然支持正常可见角选板和 spin gate 选板
3. 前哨窗口来自：
   `outpost_comming_angle`
   `outpost_leaving_angle`
4. 选板不是只看当前时刻，而是会进入命中时刻固定点迭代

因此，当前工程里前哨“未来命中板选择”真正落地的是 `Planner` 分支，而不是 `Aimer` 分支。

### 5.4 控制点生成

#### `Aimer`

`Aimer` 最终只输出一个命中点对应的：

- `yaw`
- `pitch`

没有显式轨迹，也没有前哨板级高度补偿接口。

#### `Planner`

`Planner` 在求控制点时，会先通过：

```text
resolve_aim_xyz()
```

把选中的板中心位置取出来，再叠加：

```text
outpost_fire_z_compensation[physical_armor_id]
```

也就是说，当前前哨在 `Planner` 分支里可以做到：

1. 选中不同物理板
2. 对不同物理板加不同的额外击打高度补偿

这是当前工程相对完整的一项前哨专用火控能力。

### 5.5 轨迹生成与限加速度

这一节主要对应 `Planner`。

`Planner::get_trajectory()` 会：

1. 把目标从当前时刻回滚到轨迹起点；
2. 逐步向前预测；
3. 每一帧重新求 `yaw/pitch`；
4. 生成未来 `HORIZON` 长度的参考轨迹；
5. 如果切板造成的 `yaw_acc` 尖峰过大，就做一次轻量抑制。

随后：

1. yaw 和 pitch 分别进入 `TinyMPC`
2. 最大角加速度由：
   `max_yaw_acc`
   `max_pitch_acc`
   约束

因此，当前前哨控制不是简单的“算一个角度打过去”，而是已经包含：

1. 飞行时间补偿
2. 连续轨迹参考
3. 控制输入限幅

### 5.6 开火门控

这部分必须按入口区分。

#### `standard.cpp`

当前 `standard.cpp` 里虽然创建了 `Shooter`，但没有真正调用 `shooter.shoot()`。

因此这条链路的现状是：

1. `Aimer` 输出 `Command`
2. 直接 `cboard.send(command)`
3. 不存在当前入口内额外的前哨专用开火门

#### `uav/sentry` 等分支

这些入口会调用 `Shooter::shoot()`，其门控条件主要是：

1. 当前命令不能突变太大
2. 当前云台角要接近上一帧命令
3. `aimer.debug_aim_point.valid == true`

它是通用门控，不是前哨专用相位门。

#### `Planner`

当前 `Planner` 分支对前哨已经有更完整的开火门控：

1. 先计算轨迹跟踪误差 `tracking_error`
2. 普通目标只看 `tracking_error < fire_thresh`
3. 前哨目标还必须同时满足：
   - `hit_solution.converged == true`
   - 选中板命中时刻的相位角落入更紧的击打窗口
   - 如果目标已经 `jumped`，则必须是真正通过 `spin gate` 选中的板，而不是 fallback 板

当前前哨击打窗口的推导规则是：

```text
fire_phase_limit =
  clamp(outpost_leaving_angle * 0.5, 4°, 12°)
  再限制不超过 outpost_leaving_angle 本身
```

这意味着当前工程已经显式区分：

1. 控制参考连续性
2. 实际是否允许开火

## 6. outpost 专用逻辑总结

### 6.1 感知层

当前工程对前哨在感知层的特化主要是：

1. 网络类别里有 `outpost`
2. `outpost` 被定义为小装甲板
3. 可以使用 YOLO + 传统角点二次矫正

但没有：

1. 独立前哨 detector
2. 直接输出整体转轴或相位的识别器

### 6.2 几何层

当前工程对前哨的几何特化包括：

1. 3 板模型
2. 同一半径 `r`
3. 三板高度由 `outpost_armor_z_offsets` 给出
4. 重投影时使用 `-15°` pitch 假设
5. 通过 `physical armor id` 管理真实高低板

### 6.3 动力学层

当前工程对前哨的动力学特化包括：

1. 可选固定中心旋转模型
2. 收敛后的角速度锁
3. 单独更大的 `temp_lost` 容忍
4. 专用 `id + offset` 联合匹配

### 6.4 控制层

当前工程对前哨的控制特化主要集中在 `Planner`：

1. 前哨独立 `coming/leaving` 角窗口
2. 前哨独立 `delay_time`
3. 前哨板级 `z` 补偿
4. 命中时刻固定点迭代
5. 前哨专用开火相位门

相对而言，`Aimer` 分支对前哨的支持仍然偏简化。

## 7. 当前实现中值得优先关注的问题

下面这些点是当前工程里仍然值得重点关注的风险，它们不一定都是“错误”，但都可能直接影响前哨验收结果。

### 7.1 `Aimer` 与 `Planner` 的前哨策略并不一致

当前 `Aimer` 分支和 `Planner` 分支对前哨的支持程度不一样：

1. `Aimer` 仍然使用硬编码 `70° / 30°`
2. `Aimer` 没有板级 `z` 补偿
3. `Aimer` 没有当前 `Planner` 这套更严格的前哨相位开火门

因此，如果实车主要用的是 `Planner`，就不能再沿用旧的 `Aimer` 经验去理解问题。

### 7.2 `standard.cpp` 入口当前并不代表完整前哨火控能力

当前 `standard.cpp` 没有真正接入 `Shooter::shoot()`。

这意味着：

1. 这条入口更像“出瞄准角”
2. 不是当前前哨击发逻辑最完整的验收入口
3. 如果把这条入口和 `auto_aim_debug_mpc` 的表现混着看，很容易误判问题来源

### 7.3 `Planner` 的开火效果仍高度依赖 `outpost_delay_time`

当前前哨在 `Planner` 分支里虽然已经有：

1. 命中时刻迭代
2. 相位门
3. convergence 门

但如果 `outpost_delay_time` 偏差较大，仍然会出现：

1. 总体偏早
2. 总体偏晚
3. 相位门长期不过

所以这仍然是实车验收中最敏感的参数之一。

### 7.4 前哨匹配门限仍然是硬编码经验值

`select_best_outpost_match()` 和 `accept_outpost_match()` 的门限当前写在代码里，不是运行时参数。

这意味着：

1. 远距离
2. 画面边缘
3. 角点退化

这几种工况下，如果需要放宽或收紧门限，就必须改代码，而不能只靠网页调参。

### 7.5 `outpost_armor_z_offsets` 与 `outpost_fire_z_compensation` 必须配套看

这两个量解决的是不同问题：

1. `outpost_armor_z_offsets`
   描述几何模型里三块板本身的真实高度关系
2. `outpost_fire_z_compensation`
   描述火控层额外补偿给某块物理板的击打高度修正

如果它们混用，现场就容易出现：

1. tracker 看起来稳定
2. planner 选板也对
3. 但实际总打高或打低

### 7.6 初始阶段仍然更相信“当前看到的这块板”

当前无论是 `Aimer` 还是 `Planner`，在 `target.jumped == false` 时都更相信当前观测板。

这是合理的，但也意味着：

1. 前哨刚建链时，相位是“当前板优先”
2. 如果初始化板本身就带较大观测误差，第一段参考可能仍然偏

### 7.7 文档验收时要区分“算法问题”和“入口问题”

当前工程里前哨相关表现至少会同时受到三类因素影响：

1. 跟踪是否稳
2. 当前实际跑的是 `Aimer` 还是 `Planner`
3. 当前入口有没有把开火门真正接进去

因此，文档和实车验收时一定要先确认运行入口，再谈算法效果。

## 8. 面向 outpost 的优化建议

### 8.1 第一优先级：先用 `Planner` 分支做前哨验收

当前工程里最完整的前哨方案已经在：

- `planner.cpp`
- `auto_aim_debug_mpc.cpp`
- `auto_debug.cpp`

所以如果目标是“先确认当前工程前哨算法是否合理”，最优先应该验这条链路。

### 8.2 第二优先级：统一 `Aimer` 与 `Planner` 的前哨策略

如果后续仍需要保留 `Aimer` 入口，建议至少补齐：

1. 前哨可配置窗口，而不是硬编码 `70/30`
2. 前哨板级高度补偿
3. 前哨开火相位门

否则不同入口的表现会长期不一致。

### 8.3 第三优先级：把前哨匹配门限做成运行时参数

建议优先考虑把下面几类量参数化：

1. `accept_outpost_match()` 的总分门限
2. 重投影误差门限
3. `xy/z` 误差门限

这样前哨实车调试就不必每次改代码重编译。

### 8.4 第四优先级：把前哨验收重点集中到“相位、延迟、物理板号”

当前工程里，前哨最值得关注的不是 detector，而是：

1. `tracker` 是否把局部板号和物理板号映射对了
2. `planner` 的命中时刻是否收敛
3. `delay_time` 是否让板真正落入击打相位

### 8.5 第五优先级：如果还要继续提升，再考虑自适应 ROI 和距离相关门限

当前项目没有目标驱动的动态 ROI，也没有前哨匹配的距离自适应门限。

如果前哨问题主要出现在：

1. 远距离
2. 边缘角度
3. 大视角切板

这两块会是后续继续提升的方向。

## 9. 建议的调参顺序

如果目的是做当前工程前哨验收，建议按下面顺序推进。

1. 先确认入口
   当前到底跑的是 `standard.cpp`、`standard_mpc.cpp`、`auto_aim_debug_mpc.cpp` 还是 `auto_debug.cpp`。
2. 再看检测是否稳定
   `outpost` 是否能稳定识别为 `ArmorName::outpost`，角点是否平稳。
3. 再看 tracker
   重点关注 `tracker_match_valid / tracker_match_score / tracker_reprojection_px`，确认 3 板匹配和物理板号映射是否稳定。
4. 再看 planner 命中时刻求解
   重点关注 `planner_hit_converged / planner_hit_iters / planner_selected_physical_armor`。
5. 再调前哨延迟和相位
   重点关注 `outpost_delay_time / planner_selected_delta_deg / planner_fire_phase_ready / planner_fire_phase_limit_deg`。
6. 最后看板级高度补偿
   如果相位已经对了但仍总打高或打低，再调 `outpost_fire_z_compensation`。

从验收角度，当前建议的通过标准可以先定成：

1. 前哨识别稳定，不频繁掉成别的类别；
2. `tracker_match_valid` 大部分时间为真；
3. `planner_hit_converged` 大部分时间为真；
4. 空发不再呈现“固定每切一次板就空一枪”的规律性节奏；
5. 若仍空枪，其原因能从相位、延迟、弹道、机械零偏中明确归类。

## 10. 最终评价

对当前 `sp_vision25` 来说：

1. 前哨识别层仍然是通用检测方案，特化重点不在 detector；
2. 前哨跟踪层已经有比较明确的工程化建模：
   `3` 板模型、固定中心预测、角速度锁、专用匹配、物理板号重映射；
3. 前哨控制层里最完整的实现已经在 `Planner` 分支，不再是旧文档里那种“只会选当前最正脸板”的状态。

当前这套实现的整体评价是：

1. 跟踪层设计已经比较像真正面向前哨的方案；
2. `Planner` 分支的火控闭环已经具备前哨专用相位控制雏形；
3. 但不同入口之间前哨能力仍不完全一致；
4. 实车验收时必须优先确认入口，并用调试量去拆分问题来源。

## 11. 2026-04-22 前哨实车补充分析与代码修改记录

以下内容为基于近期排查追加的补充说明，不改动上文主结论，只记录本次前哨问题定位、代码修改和后续验收重点。

本节当前也作为哨兵分支 `/home/aw/ATS_2026_snetry_test/src/sp_vision25` 同步迁移前哨优化时的记录基线使用。

### 11.1 实车现象补充

当前反馈现象可以概括为：

1. 距离约 4 米、低弹频条件下，识别与跟踪本身相对稳定；
2. `tracker` 可视化观测稳定，问题不像是检测抖动或频繁丢板；
3. 实际更像是：
   先命中当前装甲板
   -> 紧接着空发一枪
   -> 再命中下一块装甲板

因此，本轮排查重点放在：

1. `planner` 前哨选板
2. 命中时刻固定点迭代
3. 最终 `fire` 判定

而不是先回头怀疑 detector。

### 11.2 本次新增判断：为什么会出现“中一发、空一发”

本次复查确认，造成这种节奏的核心风险是：

1. `planner` 负责选“哪块板作为控制参考”；
2. 但如果“控制参考连续性”和“真正允许开火”没有分开约束；
3. 那么切板空窗期也可能被误当成可开火区间。

对前哨来说，这种风险尤其明显，因为：

1. 板切换频率高；
2. `fallback_to_closest()` 对控制参考是有价值的；
3. 但 fallback 目标并不一定已经进入稳定击打相位。

### 11.3 本次确认并修复的算法问题

这次不是简单调参数，而是确认并补上了几个算法层面的关键收口。

#### 1. 前哨缺少“开火相位门”

原先前哨虽然已经有：

1. `outpost_comming_angle / outpost_leaving_angle`
2. 选板窗口

但最终开火若只看轨迹跟踪误差，就会把：

1. 切板空窗期的参考板
2. fallback 维持连续参考的板

也误当成可开火目标。

修复后：

1. 前哨在开火前除了要满足 `tracking_error < fire_thresh`
2. 还必须满足命中时刻的 `|delta_angle|` 落入更紧的击打窗口
3. 这个窗口由 `outpost_leaving_angle` 推导：
   `leaving_angle * 0.5`
   并限制在 `4° ~ 12°`
   同时不超过 `leaving_angle` 本身

补充收口：

1. 如果目标已经发生过 `jumped`
2. 但当前命中时刻求解并没有通过 `spin gate` 选到真正进入窗口的板
3. 而只是退化到 `fallback_to_closest`
4. 那么系统现在仍然允许维持连续控制参考，但不再允许开火

#### 2. 命中时刻迭代未收敛时，不能再允许前哨开火

当前 `Planner` 里前哨会做：

```text
选板 -> 算飞行时间 -> 预测到命中时刻 -> 再选板
```

如果这一步本身都还没收敛，就说明：

1. 命中时刻还在跳
2. 命中板号还在跳

这时允许开火风险很大。

因此现在的收口是：

1. 对前哨目标，只有 `hit_solution.converged == true` 时才允许开火；
2. 若未收敛，系统仍可继续控制瞄准，但不放枪。

#### 3. 轨迹规划起点优先继承命中时刻求解出的板号

当前实现里，轨迹规划会优先使用：

```text
hit_solution.selection.armor_id
```

作为后续 `solve_aim_command()` 和 `get_trajectory()` 的初始板号。

这样做的意义是：

1. 命中时刻求解结果
2. 后续 MPC 轨迹起点

两者更一致，不容易出现“算的是这块板，轨迹却又从另一块板起步”的轻微偏差。

### 11.4 本次实际修改的代码文件

本次与前哨问题直接相关的修改主要在：

1. `tasks/auto_aim/planner/planner.hpp`
2. `tasks/auto_aim/planner/planner.cpp`
3. `src/auto_aim_debug_mpc.cpp`
4. `src/auto_debug.cpp`

补充说明：

1. 当前哨兵分支里的 `planner.hpp` 已经预先具备本轮需要的调试字段；
2. 因此这次迁移时，算法差异主要实际落在 `planner.cpp` 与两个 debug 入口；
3. 文档里仍保留 `planner.hpp` 在清单中，是为了对应这套前哨优化逻辑的完整落点。

### 11.5 本次新增的调试量

为了便于后续实车验证，本次已补充并确认以下调试量：

1. `planner_selected_delta_deg`
   当前被选中装甲板在命中时刻的相位角
2. `planner_fire_tracking_error_deg`
   当前 `fire` 判定所看到的轨迹跟踪误差
3. `planner_fire_phase_limit_deg`
   当前前哨击打相位门限
4. `planner_fire_track_ready`
   轨迹误差是否满足开火要求
5. `planner_fire_phase_ready`
   选中板是否进入相位开火窗口
6. `planner_hit_converged`
   命中时刻固定点迭代是否收敛
7. `planner_selected_physical_armor`
   当前被击打判定引用的是哪块物理板

这些量已经接入：

1. `auto_aim_debug_mpc`
2. `auto_debug`

的 plot / web state。

### 11.6 预期效果

本轮修改后的预期是：

1. `planner` 仍然可以在切板间隙保持控制参考连续；
2. 但不再把切板空窗期参考板直接当成可开火板；
3. 低弹频条件下，原来那种“命中一发后紧跟一发空枪”的固定节奏应明显减少；
4. 开火会更偏向“少打一发，但发出去的枪更像有效枪”。

### 11.7 这次修改后仍需要继续观察的点

本次修复的是算法逻辑漏洞，但不代表前哨所有问题都已经结束，后续仍建议重点观察：

1. `outpost_delay_time`
   如果实车总是统一偏早或统一偏晚，仍然需要继续测和调；
2. `planner_selected_delta_deg`
   如果它已经稳定进入小角度窗口但依旧空枪，则问题更可能转向弹道、零偏或机械时延；
3. `planner_hit_converged`
   如果它经常不收敛，说明命中时刻固定点求解本身还需要继续优化；
4. `planner_spin_gate`
   如果目标已经 `jumped`，但这个量经常为 `false`，说明切板窗口判定或角速度符号稳定性还不够好；
5. `planner_fire_phase_ready`
   如果它长期为 `false`，要结合 `planner_spin_gate` 一起看：
   `spin_gate = 0` 更像 fallback 禁火在生效；
   `spin_gate = 1` 但仍长期为 `false`，则更像相位门偏严或 `delay_time` 有偏差。

### 11.8 当前状态说明

本次先不做进一步实车回归，这里记录当前状态：

1. 已完成代码修改；
2. 已完成文档记录同步；
3. 已完成代码复查；
4. 尚未完成当前哨兵分支的本地编译验证；
5. 尚未完成新的实车回归验证。

从当前工程验收角度，建议下一步按下面顺序进行：

1. 先在 `auto_aim_debug_mpc` 或 `auto_debug` 上车看：
   `planner_selected_delta_deg`
   `planner_fire_phase_ready`
   `planner_hit_converged`
   `planner_selected_physical_armor`
2. 再看空发是否已经从“稳定每切板一次出现一枪”下降到“偶发”；
3. 最后再决定是继续调 `outpost_delay_time`、`outpost_fire_z_compensation`，还是继续细化前哨专用选板逻辑。
