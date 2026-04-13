
# Outpost 识别、跟踪与控制算法解析

以下分析基于/home/ats/awakening仓库代码整理，目标是把项目里与 `outpost` 相关的识别、跟踪、控制链路完整拆开，方便后续做参数整定、算法替换和结构优化。

## 1. 结论先行

当前项目里的 `outpost` 并不是一套完全独立的模块，而是复用了整条自动瞄准链路，只在几个关键位置做了专门建模：

1. 识别层：
   `outpost` 主要体现在类别标签上，检测器负责把目标标成 `ArmorClass::OUTPOST` 并输出 4 个角点；检测阶段没有专门的 outpost 几何或旋转预测。
2. 跟踪层：
   真正的 outpost 特化在这里。项目把 outpost 视为 `3` 块装甲板组成的旋转目标，使用 11 维状态的 ESEKF，并对运动模型、过程噪声、装甲板高度差、装甲 pitch 角做了专门分支。
3. 控制层：
   `VeryAimer` 基本沿用通用的弹道解算和轨迹生成，但在装甲板选择逻辑上没有为 outpost 做充分专门化，很多模式会退化成“选当前最正对相机的那块板”。

## 2. 代码入口与主链路

主运行链路在 `src/runtime/auto_aim.cpp`：

```text
图像输入
  -> push_common_frame
      -> 根据上一帧目标预测 ROI
  -> ArmorDetector
      -> 网络检测
      -> 数字分类
      -> 灯条颜色分类
  -> tracker
      -> 颜色筛选
      -> ArmorTracker::track
      -> AutoAimFsmController::update
  -> solver
      -> VeryAimer::very_aim
      -> 下发云台控制/开火建议
```

关键调用关系：

- ROI 预测与裁剪：`src/runtime/auto_aim.cpp:300-338`
- 检测：`src/runtime/auto_aim.cpp:435-460`
- 跟踪：`src/runtime/auto_aim.cpp:463-507`
- 控制求解：`src/runtime/auto_aim.cpp:508-556`

因此，`outpost` 的完整路径是：

```text
ArmorInfer 标出 OUTPOST
  -> ArmorDetector 整理角点/颜色/数字
  -> ArmorTracker 按 outpost 的 3 板模型建 EKF
  -> VeryAimer 按预测状态做弹道和轨迹规划
  -> 输出 yaw/pitch/fire_advice
```

## 3. 识别层解析

### 3.1 类别定义

装甲板类别定义在 `src/tasks/auto_aim/type.hpp:15`：

```cpp
enum class ArmorClass : int { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
```

`armor_num_by_armor_class()` 在 `src/tasks/auto_aim/type.hpp:154-157` 中把 `OUTPOST` 映射为 `3`，这决定了后续跟踪器会把它当作 3 板旋转体处理。

### 3.2 检测器如何识别 outpost

当前配置 `config/omni.yaml` 与 `config/test.yaml` 都使用：

```yaml
armor_detector:
  armor_infer:
    model_type: tup
```

对应 `ArmorInfer` 中的 `TUP` 模式：

- 输入尺寸：`416 x 416`
- 输出内容：4 个角点 + 置信度 + 颜色 + 数字类别
- 类别映射里包含 `ArmorClass::OUTPOST`

代码位置：

- `src/tasks/auto_aim/armor_detect/armor_infer.cpp:27-43`
- `src/tasks/auto_aim/armor_detect/armor_infer.cpp:258-337`

对于当前项目，网络输出的是装甲板四角点，而不是 outpost 中心、转轴或角速度。

### 3.3 识别后处理

`ArmorDetector::detect()` 的流程在 `src/tasks/auto_aim/armor_detect/armor_detector.cpp:330-360`：

1. 对 ROI 做网络推理。
2. `ArmorInfer::process()` 解码得到 `Armor::net`。
3. 如果启用数字分类器：
   截取数字 ROI，做二值化，再用 `mlp.onnx` 分类。
4. 如果启用颜色分类器：
   取左右灯条区域，比较 R/B 均值判断颜色。
5. `armor.tidy()`：
   把 `net` 结果整合成最终 `Armor`。
6. 把角点从网络输入尺度映射回原图，再加上 ROI 偏移。

与 outpost 直接相关的地方有两处：

1. 数字分类标签中包含 `OUTPOST`
   代码在 `src/tasks/auto_aim/armor_detect/armor_detector.cpp:296-299`
2. `ArmorClass::OUTPOST` 被当作小装甲板
   `armor_type_by_armor_class()` 只把 `NO1` 当大装甲板，`OUTPOST` 会走 `SimpleSmall`

### 3.4 识别层对 outpost 的真实作用

从工程角度看，检测层对 outpost 的职责只有三件事：

1. 把目标标成 `OUTPOST`
2. 给出 4 个稳定角点
3. 给后续 PnP / EKF 一个可用的装甲板观测

也就是说，当前 outpost 的“识别难点”并没有在检测层被单独建模。真正的难点被推给了跟踪层和控制层。

## 4. 跟踪层解析

### 4.1 跟踪器状态机

`ArmorTracker` 的外层跟踪状态机在 `src/tasks/auto_aim/armor_tracker/armor_tracker.cpp:146-186`，状态包括：

- `LOST`
- `DETECTING`
- `TRACKING`
- `TEMP_LOST`

逻辑很典型：

- 初次看到目标后进入 `DETECTING`
- 连续命中超过阈值进入 `TRACKING`
- 短时丢失进入 `TEMP_LOST`
- 丢失过久回 `LOST`

### 4.2 初始化

`ArmorTarget` 构造函数在 `src/tasks/auto_aim/armor_tracker/armor_target.cpp:13-116`。

初始化做了几件关键事：

1. 使用角点 + 相机内参做 `solvePnP(IPPE)`，得到当前装甲板在相机坐标系下的 pose。
2. 变换到 `odom` 坐标系。
3. 从当前观测装甲板反推旋转中心 `(cx, cy, cz)`。
4. 设定不同目标类型的初始协方差 `P0` 和初始半径 `r_pre`。

对 outpost 的专门初始化：

- `P0`：
  `diag = [1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.1, 0.1]`
- `r_pre = 0.2765`

对应代码：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:30-33`

### 4.3 状态定义

当前实际使用的是 `motion_model_point.hpp` 里的模型，命名空间为 `awakening::armor_point_motion_model`。

状态维度：

```text
x = [cx, vcx, cy, vcy, cz, vcz, yaw, vyaw, r, p1, p2]^T
```

代码位置：

- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:17-28`

其中：

- `cx, cy, cz`：目标旋转中心位置
- `vcx, vcy, vcz`：中心速度
- `yaw`：参考装甲板相位
- `vyaw`：角速度
- `r`：基础半径
- 对普通四板目标：
  `p1 = l`，`p2 = h`
- 对 outpost：
  `p1 = outpost01DZ`，`p2 = outpost02DZ`

也就是说，这个模型是“通用底盘模型”和“outpost 三板模型”复用同一个 11 维状态，只是最后两个维度的物理意义不同。

### 4.4 outpost 的装甲板几何建模

装甲板 `id` 对应的 pose 由 `Measure::armor_pose()` 给出，代码在：

- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:228-247`

核心公式如下。

装甲板相位：

```text
yaw_i = yaw + id * 2pi / armor_num
```

其中 outpost 有：

```text
armor_num = 3
```

装甲板平面位置：

```text
ax = cx - r_i * cos(yaw_i)
ay = cy - r_i * sin(yaw_i)
```

对 outpost：

```text
r_i = r
z_0 = cz
z_1 = cz + dz1
z_2 = cz + dz2
```

对普通四板目标：

```text
偶数板: r_i = r,     z_i = cz
奇数板: r_i = r + l, z_i = cz + h
```

这说明项目对 outpost 的建模假设是：

1. 三块装甲板在水平面上等角间隔 `120°`
2. 三块板使用同一个旋转半径 `r`
3. 三块板允许不同高度，但只显式建两个高度偏移量 `dz1/dz2`

### 4.5 outpost 的姿态建模

在观测模型里，装甲板的 pitch 会按目标类型切换：

```text
outpost      -> pitch = -15°
普通装甲板   -> pitch = +15°
```

代码位置：

- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:174-181`
- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:276-299`
- `src/tasks/auto_aim/armor_tracker/armor_tracker.cpp:294-310`

这一步非常关键，因为投影模型和 PnP 初始化都默认装甲板不是纯竖直，而是带固定 pitch 倾角。对于 outpost，这个倾角方向和普通装甲板相反。

### 4.6 预测模型

预测模型定义在 `Predict` 中，代码在：

- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:47-91`

支持 3 种模型：

- `CONSTANT_VELOCITY`
- `CONSTANT_ROTATION`
- `CONSTANT_VEL_ROT`

真正给 outpost 使用的是：

```cpp
MotionModel::CONSTANT_ROTATION
```

对应代码：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:227-235`

其含义是：

1. 中心位置不按速度积分更新
2. 线速度项会被直接清零
3. `yaw` 按 `vyaw` 积分更新

也就是把 outpost 当作“中心近似固定，只绕中心旋转”的目标。

这个假设对前哨站非常合理，也是当前项目里最像“专门算法”的一部分。

### 4.7 过程噪声

`ArmorTarget::process_noise()` 在 `src/tasks/auto_aim/armor_tracker/armor_target.cpp:128-170`。

outpost 使用专门的过程噪声参数：

- `qxyz_output`
- `qyaw_output`
- `q_outpost_dz`

配置位置：

- `config/omni.yaml:72-94`
- `config/test.yaml:72-94`

含义如下：

1. `qxyz_output`
   outpost 旋转中心位置噪声
2. `qyaw_output`
   outpost 角加速度噪声
3. `q_outpost_dz`
   两个高度偏移量 `dz1/dz2` 的过程噪声

注意这里的命名里 `output` 实际上指的是 outpost，而不是输出量。

当前配置：

```yaml
qxyz_output: [10.0, 10.0, 0.5]
qyaw_output: 0.01
q_outpost_dz: 0.5
```

这组参数体现出设计者的意图：

- 相信 outpost 旋转中心比较稳定
- 强烈相信 outpost 角速度变化不大
- 允许高度偏移量有一定自由度

### 4.8 观测模型

当前项目并没有用“yaw / pitch / distance”的低维观测，而是直接用 4 个角点的像素坐标作为 8 维观测：

```text
z = [lt_x, lt_y, lb_x, lb_y, rb_x, rb_y, rt_x, rt_y]^T
```

代码位置：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:172-184`
- `src/tasks/auto_aim/armor_tracker/motion_model_point.hpp:167-215`

观测生成过程：

1. 根据状态 `x` 生成指定 `id` 装甲板在 `odom` 中的 pose
2. 变换到 `camera_cv`
3. 使用相机内参、畸变系数做投影
4. 得到 4 个角点像素位置

这种建模的优点是：

1. 比直接用中心点更充分利用了角点几何信息
2. 更适合 outpost 这种姿态变化明显的目标
3. 可以把固定 pitch 倾角和相机畸变一起纳入观测方程

### 4.9 匹配与门控

匹配逻辑在 `ArmorTarget::match()`：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:275-354`

做法是：

1. 对每个观测装甲板 `j`
2. 对每个假设装甲板 `id`
3. 用当前 EKF 状态预测对应的 8 维角点观测 `z_pred`
4. 计算残差 `nu = z_meas - z_pred`
5. 用 `R` 计算 Mahalanobis 距离 `d2`
6. 小于 `match_gate` 的保留
7. 再做一次贪心分配

当前观测噪声是固定对角阵：

```text
R = diag(r_uv, r_uv, ..., r_uv)
```

代码位置：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:117-127`

当前配置：

```yaml
r_uv: 50.0
match_gate: 1000.0
```

### 4.10 ROI 预测

`ArmorTarget::expanded_one_one()` 在：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:355-468`

它会把目标中心附近的一个 3D 正方体投影到图像上，得到下一帧检测 ROI。

这对 outpost 也生效，因为 outpost 的中心和半径来自同一个 EKF 状态。好处是：

1. 跟踪稳定后可以缩小检测范围
2. 降低误检
3. 提升推理速度

## 5. 控制层解析

### 5.1 控制入口

控制入口是：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:600-836`

执行顺序大致如下：

1. 选一块当前较合适的装甲板
2. 估计飞行时间
3. 预测目标到子弹到达时刻
4. 重新选板并计算目标控制点
5. 在采样窗口内生成一串离散控制点
6. 用五次多项式约束加速度，生成平滑轨迹
7. 计算当前应发给云台的 yaw/pitch、速度、加速度
8. 计算开火允许窗口 `fire_advice`

### 5.2 弹道求解

`BallisticTrajectory` 在 `src/tasks/base/ballistic_trajectory.hpp`。

这里使用的是带阻力的一维近似模型：

```text
t = (exp(r * distance) - 1) / (r * v0 * cos(theta))
y = v0 * sin(theta) * t - 0.5 * g * t^2
```

控制层分别会求：

- `solve_pitch()`
- `solve_flytime()`

这意味着 outpost 的控制不是纯视觉角度控制，而是已经包含了飞行时间补偿。

### 5.3 装甲板选择策略

`select_armor()` 在：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:448-548`

它首先计算每块板相对于目标中心朝向的相位偏差：

```text
delta_i = normalize(armor_yaw_i - center_yaw)
```

然后按不同 FSM 模式选板。

但要注意，代码里明确把 outpost 排除在部分逻辑之外：

```cpp
if (auto_aim_fsm == AIM_SINGLE_ARMOR
    && target.target_number != ArmorClass::OUTPOST)
```

以及：

```cpp
if (auto_aim_fsm == AIM_WHOLE_CAR_PAIR
    && target.target_number != ArmorClass::OUTPOST)
```

这意味着：

1. outpost 不走“单板锁定”专用逻辑
2. outpost 不走“双板 pair”逻辑
3. 大多数情况下会退化成“在所有板里选 `|delta_i|` 最小的那块”

从效果上说，就是：当前 outpost 更像“选当前最正脸的一块板”，而不是“按未来过中线时刻选择将被击中的那块板”。

### 5.4 控制点生成

`get_control_point()` 在：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:549-575`

对选中的装甲板：

1. 用 `(x, y, z)` 算水平偏航 `control_yaw`
2. 用弹道模型解俯仰 `pitch`
3. 记录 `aim_point`
4. 保存装甲板法向与视线之间的夹角 `d_angle`

其中：

```text
d_angle = shortest_angular_distance(control_yaw, armor_yaw)
```

这个量后面会用于缩放开火窗口，因为越是侧着的装甲板，等效可打宽度越小。

### 5.5 轨迹生成与限加速度

`VeryAimer` 并不是只给一个目标角，而是会在前后一个采样窗口内生成控制轨迹：

- `sample_total_time = 2.0`
- `sample_horizon = 500`

配置位置：

- `config/omni.yaml:106-136`
- `config/test.yaml:106-136`

生成方法：

1. 在 `[-1s, +1s]` 附近采样多个时刻
2. 每个时刻都预测目标状态
3. 每个时刻都重新选板并求控制点
4. 用五次多项式连接相邻点
5. 如果某段角加速度过大，就自动扩展拼接区间

核心代码：

- 轨迹段与限加速度：`src/tasks/auto_aim/armor_control/very_aimer.cpp:55-407`
- 构建采样轨迹：`src/tasks/auto_aim/armor_control/very_aimer.cpp:638-749`

这部分设计对于 outpost 的价值很大，因为 outpost 本质上是高速周期目标，单点命令容易抖动，轨迹式控制更稳。

### 5.6 开火门控

开火逻辑在：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:775-833`

思路是：

1. 根据目标距离和装甲板宽高，计算可接受的 yaw / pitch 误差窗口
2. 再乘以姿态因子
   - `yaw_factor = cos(d_angle)`
   - `pitch_factor = cos(15°)`
3. 比较“控制轨迹当前角度”和“目标轨迹当前角度”的偏差
4. 如果偏差在窗口内，则允许开火

这套逻辑对 outpost 同样适用，但因为 outpost 旋转快，最终效果高度依赖：

1. 飞行时间估计是否准确
2. 选板是否面向未来时刻而不是当前时刻
3. 控制延迟是否建模准确

## 6. outpost 专用逻辑总结

把所有 outpost 特化点汇总起来，当前项目对 outpost 的处理可以概括成：

### 6.1 感知层

1. 网络类别中有 `OUTPOST`
2. 数字分类器中有 `OUTPOST`
3. 装甲板按小装甲板尺寸建模

### 6.2 几何层

1. outpost 共有 3 块板
2. 三块板角间隔固定为 `120°`
3. 三块板共享基础半径 `r`
4. 三块板允许不同高度：`cz / cz + dz1 / cz + dz2`
5. 装甲板固定 pitch 为 `-15°`

### 6.3 动力学层

1. 使用 `CONSTANT_ROTATION`
2. 线速度直接衰减为 0
3. 只保留中心近似固定、相位持续变化
4. 使用更小的 `qyaw_output`

### 6.4 控制层

1. 复用通用弹道与轨迹控制
2. 没有独立的 outpost 相位控制器
3. 很多模式最终退化成最小 `|delta|` 选板

## 7. 当前实现中值得优先关注的问题

下面这些点我建议优先处理，因为它们会直接影响 outpost 命中率。

### 7.1 部分灯条观测逻辑实际上没有生效

在 `ArmorTarget::update()` 中先计算了 `MeasureType mt`：

- 只看到左灯条时想用 `L_LIGHT`
- 只看到右灯条时想用 `R_LIGHT`

代码位置：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:248-257`

但后面真正更新时调用的是：

```cpp
auto measurement = get_measurement(armor);
target_state.x = esekf.update(measurement);
```

而不是：

```cpp
get_measurement(armor, z_pred, mt)
```

对应代码：

- `src/tasks/auto_aim/armor_tracker/armor_target.cpp:265-270`

这意味着：

1. `L_LIGHT / R_LIGHT` 只是算了但没用
2. 遮挡半板时并没有真正退化成部分观测更新
3. outpost 在高速旋转、边缘出视野、灯条缺失时更容易丢失

### 7.2 飞行时间“迭代预测”没有真正迭代

`VeryAimer::very_aim()` 中看起来想做飞行时间迭代：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:612-625`

但循环里只计算了 `iter_fly_time`，没有回写给 `prev_fly_time`：

```cpp
double prev_fly_time = ...
for (...) {
    ...
    double iter_fly_time = ...
}
const double predict_time = prev_fly_time + ...
```

结果是：

1. 预测时刻只用了第一次粗略飞行时间
2. 没有形成“预测 -> 重选板 -> 重估飞行时间 -> 再预测”的闭环
3. 对 outpost 这种高速旋转目标会更伤

### 7.3 `fire_advice` 存在角度单位混用风险

代码中先把允许误差转成了角度制：

```cpp
cmd.enable_yaw_diff = angles::to_degrees(enable_diff.first);
cmd.enable_pitch_diff = angles::to_degrees(enable_diff.second);
```

但紧接着又拿弧度结果去和它比较：

```cpp
abs(shortest_angular_distance(rad, rad)) < cmd.enable_yaw_diff
```

位置：

- `src/tasks/auto_aim/armor_control/very_aimer.cpp:803-815`

这会导致：

1. 判断式左侧是弧度
2. 右侧是角度数值
3. 窗口被放大约 57.3 倍

虽然延迟校验分支里又用了弧度版本，但当前帧 `cmd.fire_advice` 本身还是有单位不一致问题。

### 7.4 outpost 没有专门的“未来命中板选择”策略

当前 outpost 选板大多基于“此刻最正对相机的板”，而不是“子弹到达时最接近准线的板”。

对于 outpost，更合理的策略通常是：

1. 用 `t_hit = fly_time + control_delay + prediction_delay`
2. 预测每块板在 `t_hit` 时刻的相位
3. 选未来最接近准线的板
4. 若旋转很快，则直接进入“过中心开火”或“固定相位窗口开火”

当前代码里虽然有时间预测和轨迹规划，但 `select_armor()` 本身还不是基于 `t_hit` 的相位最优选择器。

### 7.5 outpost 的高度偏移量是自由随机游走，缺少物理约束

当前模型里：

```text
z_0 = cz
z_1 = cz + dz1
z_2 = cz + dz2
```

并且 `dz1 / dz2` 只走随机游走噪声，没有额外约束。

这会带来两个风险：

1. 长时间跟踪后高度偏移可能漂移
2. 三块板之间的几何关系不够“刚”

如果你后续发现 outpost 俯仰抖动明显，这里很值得改。

### 7.6 outpost 初始目标选择过于粗糙

`init_target()` 只是取第一个“不是 NONE / PURPLE”的装甲板作为初始化目标：

- `src/tasks/auto_aim/armor_tracker/armor_tracker.cpp:71-90`

没有按以下因素做排序：

1. 置信度
2. 与上一 ROI 中心距离
3. PnP 重投影误差
4. 是否更像前哨站的角点结构

这会让 outpost 初始相位和中心解算更容易跳。

### 7.7 数字分类标签映射建议核对

代码里的 `label_map` 只有 8 类：

- `src/tasks/auto_aim/armor_detect/armor_detector.cpp:296-299`

但 `model/label.txt` 有 9 行，包含 `negative`：

- `model/label.txt:1-9`

这不一定一定有错，但建议确认：

1. `mlp.onnx` 的输出维度到底是 8 还是 9
2. 若有 `negative` 类，当前代码是否会把它吞成 `UNKNOWN`
3. outpost 和哨兵之间是否存在混淆

## 8. 面向 outpost 的优化建议

下面给一版我认为更适合你当前工程结构的优化路线。

### 8.1 第一优先级：先修实现问题

建议先修以下 4 个点：

1. 让 `L_LIGHT / R_LIGHT` 观测真正参与 EKF 更新
2. 修复飞行时间迭代不回写的问题
3. 统一 `fire_advice` 的弧度/角度单位
4. 核对数字分类标签映射

这四个点都属于“低成本高收益”，先修它们，outpost 效果通常会明显稳定。

### 8.2 第二优先级：为 outpost 单独做选板和开火策略

建议新增一个 outpost 专用策略，而不是完全复用普通车。

推荐思路：

1. 保留现有 EKF，不必大改状态估计框架。
2. 在 `VeryAimer` 里单独写 `select_outpost_armor(t_hit)`。
3. 以未来击中时刻 `t_hit` 为准，选择最接近准线的板。
4. 开火时不要只看当前角度误差，而要看“未来 `control_delay` 后”的误差是否仍然在窗内。
5. 如果 outpost 转速很高，可以直接在中心模式下工作，并增加一个固定相位窗口。

一个更适合 outpost 的简化思路是：

```text
phi_i(t_hit) = yaw(t_hit) + i * 2pi / 3
pick i = argmin |normalize(phi_i(t_hit) - yaw_los)|
if |...| < fire_window:
    shoot
```

这里的 `yaw_los` 是瞄准线对应的参考相位。

### 8.3 第三优先级：提高 outpost 状态模型的刚性

如果你希望进一步提稳定性，可以考虑：

1. 把 `dz1 / dz2` 改成固定标定值，或者强约束参数
2. 给 outpost 单独的状态向量
   比如只保留 `[cx, cy, cz, yaw, vyaw, r]`
3. 如果 outpost 在场地中几乎静止，甚至可以把 `(cx, cy)` 做成极小随机游走

这样做的好处是：

1. 参数更少
2. 可观测性更强
3. 俯仰抖动更小
4. 更接近 outpost 的真实运动学

### 8.4 第四优先级：把观测噪声做成距离/角度相关

当前 `R = diag(r_uv)` 太简单了。

更好的做法是把下列因素纳入 `R`：

1. 图像尺度
2. 装甲板长宽比
3. 出视野程度
4. `d_angle`
5. PnP 重投影误差

经验上，outpost 在边缘角度大时角点更容易飘，固定 `r_uv` 不够鲁棒。

### 8.5 第五优先级：改进 ROI

当前 ROI 是按整个目标外接正方形投影得到的。

可以进一步优化为：

1. 对 outpost 单独扩大切边余量
2. 在高角速度时沿旋转方向增加前向余量
3. 用 `vyaw * dt` 预测下一帧可见板区域

这样能减少高速旋转时“板刚好切出 ROI”的问题。

## 9. 建议的调参顺序

如果你准备在自己项目里复用这套思路，我建议按下面顺序调：

1. 先确认识别：
   outpost 是否能稳定分成 `OUTPOST`，角点是否平稳。
2. 再确认观测：
   `solvePnP` 和 8 维角点残差是否稳定，重投影误差是否可接受。
3. 再调跟踪：
   先调 `r_uv` 和 `match_gate`，再调 `qyaw_output`、`qxyz_output`、`q_outpost_dz`。
4. 再调时延：
   调 `prediction_delay`、`control_delay` 和弹速。
5. 最后调开火：
   调 `shooting_range_*`、`min_enable_*`，以及 outpost 专用相位窗口。

## 10. 最终评价

这套实现的优点是：

1. 对 outpost 已经有明确的三板旋转体建模
2. 观测模型不是低维角度，而是完整四角点投影，信息利用比较充分
3. 控制层不是简单 PID 指向，而是带弹道和轨迹平滑的预测控制风格

它当前的主要短板也很明确：

1. outpost 专用控制策略还不够完整
2. 有几处实现细节会直接影响命中率
3. 选板逻辑更偏“当前最正”而不是“未来命中最优”

如果你下一步要做自己的项目优化，我最推荐的切入点是：

1. 先修正观测更新和飞行时间迭代
2. 再单独写 outpost 的相位式选板/开火逻辑
3. 最后再收紧状态模型和噪声参数

这三步做完，通常就能把 outpost 的命中稳定性拉高一个层级。
