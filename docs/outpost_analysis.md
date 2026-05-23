# 前哨站跟踪匹配与 EKF 预测算法详解

本文档只包含前哨站（outpost）专用算法，详细说明跟踪匹配、EKF 预测、板号映射的完整代码实现。

## 1. 前哨站跟踪匹配流程

### 1.1 入口：update_target()

代码位置：[tracker.cpp:459-521](tasks/auto_aim/tracker.cpp#L459-L521)

```cpp
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);  // [1] EKF 预测

  int found_count = 0;
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
  }

  if (target_.name == ArmorName::outpost) {
    // [2] 前哨专用匹配
    const auto best_match = select_best_outpost_match(target_, solver_, armors);
    // [3] 门限筛选
    const bool accept = accept_outpost_match(best_match);
    if (!accept) {
      return false;
    }
    // [4] 设置物理板号映射
    target_.set_armor_id_offset(best_match.offset, target_.last_id);
    // [5] 用指定的局部板号做 EKF 更新
    target_.update(*best_match.armor_it, best_match.id);
    return true;
  }

  // 普通目标：直接就近关联
  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    solver_.solve(armor);
    target_.update(armor);
  }
  return true;
}
```

**关键区别**：普通目标直接调用 `target_.update(armor)` 走默认就近关联；前哨站必须经过 `select_best_outpost_match()` + `accept_outpost_match()` 两步专用匹配。

---

### 1.2 核心匹配：select_best_outpost_match()

代码位置：[tracker.cpp:50-129](tasks/auto_aim/tracker.cpp#L50-L129)

这是前哨匹配的核心函数。它对每一块检测到的装甲板，枚举所有可能的 `(id, offset)` 组合，计算综合代价，返回得分最低者。

```cpp
OutpostMatchResult select_best_outpost_match(
  const Target & target, Solver & solver, std::list<Armor> & armors)
{
  OutpostMatchResult best_match;
  const int armor_num = static_cast<int>(target.armor_xyza_list().size());
  if (armor_num <= 0) return best_match;

  // 各代价项的归一化门限
  constexpr double kYawGate = 6.0 / 57.3;
  constexpr double kPitchGate = 5.0 / 57.3;
  constexpr double kArmorYawGate = 14.0 / 57.3;
  constexpr double kDistanceGate = 0.16;
  constexpr double kXYGate = 0.12;
  constexpr double kZGate = 0.05;
  constexpr double kReprojectionGate = 18.0;

  for (auto it = armors.begin(); it != armors.end(); ++it) {
    auto & armor = *it;
    if (armor.name != target.name || armor.type != target.armor_type) continue;

    solver.solve(armor);
    const Eigen::Vector3d observed_xyz = armor.xyz_in_world;
    const Eigen::Vector3d observed_ypd = armor.ypd_in_world;

    // 枚举所有 (id, offset) 组合
    for (int id = 0; id < armor_num; ++id) {
      for (int offset = 0; offset < armor_num; ++offset) {
        // 创建副本，设置物理板号偏移
        Target mapped_target = target;
        mapped_target.set_armor_id_offset(offset, target.last_id);

        // 获取第 id 块板的预测位置
        const auto predicted_armors = mapped_target.armor_xyza_list();
        const auto & predicted_xyza = predicted_armors[id];
        const Eigen::Vector3d predicted_xyz = predicted_xyza.head<3>();
        const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(predicted_xyz);
        const auto predicted_points =
          solver.reproject_armor(predicted_xyz, predicted_xyza[3], armor.type, armor.name);

        // 计算各项误差
        const double reprojection_error = average_point_error(armor.points, predicted_points);
        const double los_yaw_error =
          std::abs(tools::limit_rad(observed_ypd[0] - predicted_ypd[0]));
        const double pitch_error = std::abs(observed_ypd[1] - predicted_ypd[1]);
        const double distance_error = std::abs(observed_ypd[2] - predicted_ypd[2]);
        const double armor_yaw_error =
          std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_xyza[3]));
        const double xy_error = (observed_xyz.head<2>() - predicted_xyz.head<2>()).norm();
        const double z_error = std::abs(observed_xyz.z() - predicted_xyz.z());

        // 连续性惩罚：惩罚 id 跳变和 offset 跳变
        double continuity_penalty = 0.0;
        if (id != target.last_id) {
          const int step = cyclic_id_distance(id, target.last_id, armor_num);
          continuity_penalty += target.jumped ? 0.8 * step : 0.35 * step;
        }
        if (offset != target.armor_id_offset()) {
          const int step = cyclic_id_distance(offset, target.armor_id_offset(), armor_num);
          continuity_penalty += target.jumped ? 0.22 * step : 0.08 * step;
        }

        // 加权求和
        const double score =
          std::pow(reprojection_error / kReprojectionGate, 2) +
          std::pow(los_yaw_error / kYawGate, 2) +
          std::pow(pitch_error / kPitchGate, 2) +
          std::pow(distance_error / kDistanceGate, 2) +
          std::pow(armor_yaw_error / kArmorYawGate, 2) +
          std::pow(xy_error / kXYGate, 2) +
          std::pow(z_error / kZGate, 2) + continuity_penalty;

        if (score >= best_match.score) continue;

        best_match.armor_it = it;
        best_match.id = id;
        best_match.offset = offset;
        best_match.physical_id = mapped_target.physical_armor_id(id);
        best_match.score = score;
        best_match.reprojection_error = reprojection_error;
        best_match.xy_error = xy_error;
        best_match.z_error = z_error;
        best_match.valid = true;
      }
    }
  }

  return best_match;
}
```

**算法要点**：

1. **三重循环**：对每个检测到的 armor，枚举 `id ∈ {0,1,2}`（局部板号）× `offset ∈ {0,1,2}`（物理板号偏移），共 9 种组合
2. **每种组合**：创建 `mapped_target` 副本，调用 `set_armor_id_offset(offset)` 设置物理板号映射，然后取第 `id` 块板的预测位置
3. **代价计算**：7 项几何误差 + 1 项连续性惩罚，各项除以门限后平方求和
4. **连续性惩罚**：`jumped` 后惩罚更重（0.8/0.22），未 jump 时惩罚较轻（0.35/0.08）
5. **返回**：得分最低的 `(id, offset, armor)` 组合

---

### 1.3 门限筛选：accept_outpost_match()

代码位置：[tracker.cpp:131-137](tasks/auto_aim/tracker.cpp#L131-L137)

```cpp
bool accept_outpost_match(const OutpostMatchResult & match)
{
  return
    match.valid && std::isfinite(match.score) && std::isfinite(match.reprojection_error) &&
    match.reprojection_error < 90.0 && match.xy_error < 0.40 && match.z_error < 0.20 &&
    match.score < 36.0;
}
```

四个硬门限：

| 条件 | 阈值 | 含义 |
|------|------|------|
| `reprojection_error` | < 90 px | 预测角点与观测角点的像素距离 |
| `xy_error` | < 0.40 m | 水平面位置误差 |
| `z_error` | < 0.20 m | 高度误差 |
| `score` | < 36.0 | 综合加权得分 |

---

## 2. 物理板号映射机制

### 2.1 两套板号

前哨站有 3 块装甲板，但存在两套编号：

- **局部板号 (local id)**：EKF 状态中几何顺序的第几块板（0, 1, 2）
- **物理板号 (physical id)**：真实物理上的高板/低板是哪一块（0, 1, 2）

转换关系：`physical_id = (local_id + offset) % 3`

### 2.2 set_armor_id_offset() — 设置物理板号映射

代码位置：[target.cpp:325-342](tasks/auto_aim/target.cpp#L325-L342)

```cpp
void Target::set_armor_id_offset(int offset, int reference_id)
{
  if (armor_num_ <= 0) return;

  const int normalized_offset = normalize_armor_id(offset);
  if (normalized_offset == armor_id_offset_) return;

  const int reference_local_id = normalize_armor_id(reference_id);
  const double old_reference_z_offset = armor_z_offset(reference_local_id);
  armor_id_offset_ = normalized_offset;
  const double new_reference_z_offset = armor_z_offset(reference_local_id);

  // 保持当前跟踪的参考板高度连续
  if (ekf_.x.size() > 4) {
    ekf_.x[4] += old_reference_z_offset - new_reference_z_offset;
  }
}
```

**关键设计**：切换物理板号映射时，调整 `ekf_.x[4]`（中心 z 位置），使得当前参考板的实际高度保持连续。这避免了切板时中心高度的跳变。

**示例**：假设当前 `offset=0`，参考板 id=0，其物理板号为 0，高度偏移为 0.0m。切换到 `offset=1` 后，参考板 id=0 的物理板号变为 1，高度偏移变为 -0.102m。此时 `ekf_.x[4]` 会增加 `0.0 - (-0.102) = 0.102m`，使得参考板的实际高度不变。

### 2.3 physical_armor_id() — 获取物理板号

代码位置：[target.cpp:318-321](tasks/auto_aim/target.cpp#L318-L321)

```cpp
int Target::physical_armor_id(int id) const
{
  return normalize_armor_id(normalize_armor_id(id) + armor_id_offset_);
}
```

### 2.4 armor_z_offset() — 获取板高度偏移

代码位置：[target.cpp:312-316](tasks/auto_aim/target.cpp#L312-L316)

```cpp
double Target::armor_z_offset(int id) const
{
  if (armor_z_offsets_.empty() || armor_num_ <= 0) return 0.0;
  return armor_z_offsets_[physical_armor_id(id)];
}
```

**注意**：`armor_z_offset(id)` 接受局部板号，内部通过 `physical_armor_id(id)` 转换为物理板号后查表。三块板的高度偏移来自配置：

| 物理板号 | 高度偏移 | 含义 |
|---------|---------|------|
| 0 | 0.0 m | 基准板 |
| 1 | -0.102 m | 低板 |
| 2 | +0.102 m | 高板 |

---

## 3. EKF 预测模型

### 3.1 11 维状态向量

代码位置：[target.cpp:49](tasks/auto_aim/target.cpp#L49)

```
x = [cx, vcx, cy, vcy, cz, vcz, yaw, vyaw, r, l, h]
      0    1    2    3    4    5    6     7    8   9  10
```

| 索引 | 符号 | 含义 |
|-----|------|------|
| x[0] | cx | 旋转中心 x 坐标 |
| x[1] | vcx | 中心 x 方向速度 |
| x[2] | cy | 旋转中心 y 坐标 |
| x[3] | vcy | 中心 y 方向速度 |
| x[4] | cz | 旋转中心 z 坐标 |
| x[5] | vcz | 中心 z 方向速度 |
| x[6] | yaw | 参考装甲板相位角 |
| x[7] | vyaw | 角速度 |
| x[8] | r | 旋转半径 |
| x[9] | l | 长短半径差（前哨不使用） |
| x[10] | h | 高度差（前哨不使用） |

### 3.2 固定中心旋转模型（前哨默认）

代码位置：[target.cpp:98-123](tasks/auto_aim/target.cpp#L98-L123)

```cpp
void Target::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f =
    [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (fixed_center_rotation_model_) {
    const double dt_abs = std::max(std::abs(dt), 1e-3);

    // 状态转移矩阵 F
    F(6, 7) = dt;    // yaw += vyaw * dt
    F(1, 1) = 0.0;   // 速度不传播到位置
    F(3, 3) = 0.0;
    F(5, 5) = 0.0;

    // 过程噪声 Q
    Q(0, 0) = 1e-5 * dt_abs;   // 位置噪声
    Q(1, 1) = 1e-4 * dt_abs;   // 速度噪声
    Q(2, 2) = 1e-5 * dt_abs;
    Q(3, 3) = 1e-4 * dt_abs;
    Q(4, 4) = 1e-5 * dt_abs;
    Q(5, 5) = 1e-4 * dt_abs;
    Q(6, 6) = 2e-3 * dt_abs;   // 角度噪声
    Q(7, 7) = 5e-3 * dt_abs;   // 角速度噪声

    // 状态转移函数 f
    f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::VectorXd x_prior = x;
      x_prior[1] = 0.0;   // 速度压零
      x_prior[3] = 0.0;
      x_prior[5] = 0.0;
      x_prior[6] = tools::limit_rad(x[6] + x[7] * dt);  // 角度积分
      return x_prior;
    };
  }

  // 角速度锁定
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2) {
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? spin_speed_lock_ : -spin_speed_lock_;
  }

  ekf_.predict(F, Q, f);
}
```

**算法要点**：

1. **位置不移动**：`F(1,1)=F(3,3)=F(5,5)=0`，速度不传播到位置，中心保持固定
2. **速度压零**：`f` 中将 `x[1], x[3], x[5]` 设为 0，等价于"前哨不平移"
3. **角度积分**：`yaw += vyaw × dt`，只旋转不平移
4. **小过程噪声**：位置 1e-5、速度 1e-4、角度 2e-3、角速度 5e-3，反映"前哨中心稳定"的假设
5. **角速度锁定**：收敛后若 `|vyaw| > 2`，锁定到 `±2.51 rad/s`

### 3.3 普通常速度模型（备选）

代码位置：[target.cpp:123-167](tasks/auto_aim/target.cpp#L123-L167)

```cpp
  } else {
    // 状态转移矩阵
    F <<
      1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1;

    // Piecewise White Noise Model
    double v1, v2;
    if (name == ArmorName::outpost) {
      v1 = 10;   // 前哨站加速度方差
      v2 = 0.1;  // 前哨站角加速度方差
    } else {
      v1 = 100;  // 加速度方差
      v2 = 400;  // 角加速度方差
    }
    // ... Q 矩阵构建
  }
```

前哨在此模式下的过程噪声仍比普通目标小一个数量级。

---

## 4. EKF 量测更新

### 4.1 观测模型

代码位置：[target.cpp:231-288](tasks/auto_aim/target.cpp#L231-L288)

```cpp
void Target::update_ypda(const Armor & armor, int id)
{
  // 参数保护：检查半径、l、h 是否越界
  if (this->convergened()) {
    if (ekf_.x[8] < 0.18 || ekf_.x[8] > 0.35) {
      ekf_.x[8] = 0.25;  // 半径越界重置
    }
    if (std::abs(ekf_.x[9]) > 0.20) {
      ekf_.x[9] = 0.1;   // l 越界重置
    }
    if (std::abs(ekf_.x[10]) > 0.20) {
      ekf_.x[10] = 0.0;  // h 越界重置
    }
  }

  // 观测雅可比矩阵
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  // 自适应观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 非线性观测函数 h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 角度差值函数（防止 2π 跳变）
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  // 观测量
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}
```

**观测量**：`z = [yaw_los, pitch_los, distance, armor_yaw]`

| 分量 | 来源 | 含义 |
|------|------|------|
| `ypd[0]` | `armor.ypd_in_world[0]` | 视线水平角 |
| `ypd[1]` | `armor.ypd_in_world[1]` | 视线垂直角 |
| `ypd[2]` | `armor.ypd_in_world[2]` | 目标距离 |
| `ypr[0]` | `armor.ypr_in_world[0]` | 装甲板朝向角 |

**自适应噪声**：

```
R_diag = [4e-3, 4e-3, log(|delta_angle|+1)+1, log(|distance|+1)/200 + 9e-2]
```

- 角度噪声固定 4e-3
- 距离噪声随板相位偏角增大（板面倾斜时测距更不准）
- 装甲板 yaw 噪声随距离增大（远距离角点退化）

### 4.2 装甲板几何模型：h_armor_xyz()

代码位置：[target.cpp:382-393](tasks/auto_aim/target.cpp#L382-L393)

```cpp
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4] + armor_z_offset(id);

  return {armor_x, armor_y, armor_z};
}
```

对前哨（`armor_num_ = 3`），`use_l_h` 恒为 `false`，因此：

```
第 i 块板的位置：
  angle_i = yaw + i × 2π/3
  x_i = cx - r × cos(angle_i)
  y_i = cy - r × sin(angle_i)
  z_i = cz + armor_z_offset(i)
```

三块板共用半径 `r`，高度差完全由 `armor_z_offsets` 决定。

### 4.3 观测雅可比矩阵：h_jacobian()

代码位置：[target.cpp:395-432](tasks/auto_aim/target.cpp#L395-L432)

```cpp
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // 装甲板位置对状态的雅可比
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };

  // xyz -> ypd 的雅可比
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };

  return H_armor_ypda * H_armor_xyza;
}
```

对前哨（`use_l_h = false`），`dx_dl = dy_dl = dz_dh = 0`，雅可比矩阵简化为：

```
H_armor_xyza = [
  [1, 0, 0, 0, 0, 0, r·sin(a), 0, -cos(a), 0, 0],
  [0, 0, 1, 0, 0, 0, -r·cos(a), 0, -sin(a), 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]
```

---

## 5. 前哨站初始化

### 5.1 Target 构造函数

代码位置：[target.cpp:12-60](tasks/auto_aim/target.cpp#L12-L60)

```cpp
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, const std::vector<double> & armor_z_offsets,
  bool fixed_center_rotation_model, double spin_speed_lock)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  fixed_center_rotation_model_(fixed_center_rotation_model),
  spin_speed_lock_(spin_speed_lock),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  armor_z_offsets_.assign(armor_num_, 0.0);
  for (int i = 0; i < std::min<int>(armor_num_, armor_z_offsets.size()); ++i) {
    armor_z_offsets_[i] = armor_z_offsets[i];
  }

  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 从观测装甲板位置沿 yaw 方向回推旋转中心
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // 初始化状态向量
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}
```

### 5.2 Tracker::set_target() 中的前哨初始化

代码位置：[tracker.cpp:439-443](tasks/auto_aim/tracker.cpp#L439-L443)

```cpp
else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(
      armor, t, outpost_radius_, 3, P0_dig, outpost_armor_z_offsets_,
      outpost_fixed_center_rotation_model_, outpost_spin_speed_lock_);
}
```

| 参数 | 配置项 | 默认值 |
|-----|-------|-------|
| 半径 | `outpost_radius` | 0.2765 m |
| 装甲板数 | 硬编码 | 3 |
| 高度偏移 | `outpost_armor_z_offsets` | [0.0, -0.102, 0.102] |
| 固定中心模型 | `outpost_fixed_center_rotation_model` | true |
| 角速度锁定 | `outpost_spin_speed_lock` | 2.51 rad/s |
| 初始协方差 | 硬编码 | [1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0] |

---

## 6. 前哨站 Planner 火控

### 6.1 命中时刻固定点迭代：solve_hit_target()

代码位置：[planner.cpp:179-230](tasks/auto_aim/planner/planner.cpp#L179-L230)

```cpp
HitTargetSolution solve_hit_target(
  const Target & base_target, double bullet_speed, double coming_angle, double leaving_angle,
  int initial_lock_id,
  const std::function<Eigen::Vector3d(const Target &, const AimSelection &)> & resolve_aim_xyz)
{
  HitTargetSolution solution;
  int working_lock_id = initial_lock_id;
  double fly_time = 0.0;
  int previous_armor_id = -1;

  for (int iter = 0; iter < kHitTimeIterMax; ++iter) {
    Target iter_target = base_target;
    if (fly_time > 0.0) {
      iter_target.predict(fly_time);  // 预测到命中时刻
    }

    // 在预测后的 target 上选板
    auto selection =
      choose_aim_selection(iter_target, coming_angle, leaving_angle, working_lock_id);
    if (!selection.valid) return solution;

    // 解弹道得到飞行时间
    const Eigen::Vector3d xyz = resolve_aim_xyz(iter_target, selection);
    const double dist_xy = xyz.head<2>().norm();
    auto bullet_traj = tools::Trajectory(bullet_speed, dist_xy, xyz.z());
    if (bullet_traj.unsolvable) return solution;

    // 再预测到命中时刻
    Target hit_target = base_target;
    hit_target.predict(bullet_traj.fly_time);

    // 收敛判定：飞行时间变化 < 1ms 且板号不变
    if (
      iter > 0 && std::abs(bullet_traj.fly_time - fly_time) < kHitTimeTol &&
      selection.armor_id == previous_armor_id)
    {
      solution.valid = true;
      solution.target_at_hit = hit_target;
      solution.selection = selection;
      solution.fly_time = bullet_traj.fly_time;
      solution.iter_count = iter + 1;
      solution.converged = true;
      return solution;
    }

    fly_time = bullet_traj.fly_time;
    previous_armor_id = selection.armor_id;
    solution.valid = true;
    solution.target_at_hit = hit_target;
    solution.selection = selection;
    solution.fly_time = bullet_traj.fly_time;
    solution.iter_count = iter + 1;
  }

  return solution;
}
```

**迭代过程**：
1. 将 target 预测 `fly_time` 秒
2. 在预测后的 target 上执行 `choose_aim_selection()` 选板
3. 解弹道得到新的 `fly_time`
4. 若 `|new_fly_time - old_fly_time| < 1ms` 且板号不变 → 收敛
5. 否则用新的 `fly_time` 继续迭代（最多 6 次）

### 6.2 选板策略：choose_aim_selection()

代码位置：[planner.cpp:48-167](tasks/auto_aim/planner/planner.cpp#L48-L167)

```cpp
AimSelection choose_aim_selection(
  const Target & target, double coming_angle, double leaving_angle, int & lock_id)
{
  AimSelection selection;
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return selection;

  const Eigen::VectorXd ekf_x = target.ekf_x();
  selection.center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  selection.delta_angle_list.reserve(armor_xyza_list.size());

  // 计算每块板相对于中心的相位角
  for (const auto & xyza : armor_xyza_list) {
    selection.delta_angle_list.push_back(
      tools::limit_rad(xyza[3] - selection.center_yaw));
  }

  auto fill_selection = [&](int armor_id, bool used_spin_gate) {
    selection.valid = true;
    selection.armor_id = armor_id;
    selection.used_spin_gate = used_spin_gate;
    selection.xyza = armor_xyza_list[armor_id];
    selection.selected_delta_angle = resolve_selected_delta_angle(selection);
  };

  auto fallback_to_closest = [&]() {
    int best_id = 0;
    double best_score = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(selection.delta_angle_list.size()); ++i) {
      const double score = std::abs(selection.delta_angle_list[i]);
      if (score < best_score) {
        best_score = score;
        best_id = i;
      }
    }
    lock_id = -1;
    fill_selection(best_id, false);
  };

  // 未 jump 时锁定当前板
  if (!target.jumped) {
    const int armor_count = static_cast<int>(armor_xyza_list.size());
    const int observed_id = (target.last_id % armor_count + armor_count) % armor_count;
    lock_id = observed_id;
    fill_selection(observed_id, false);
    return selection;
  }

  const double target_w = ekf_x[7];
  const bool use_spin_gate =
    std::abs(target_w) > kSpinSpeedThreshold || target.name == ArmorName::outpost;

  // 小陀螺或前哨：选正在进入击打窗口的板
  int best_id = -1;
  double best_score = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(selection.delta_angle_list.size()); ++i) {
    const double delta_angle = selection.delta_angle_list[i];
    if (std::abs(delta_angle) > coming_angle) continue;

    bool entering_window = false;
    if (target_w > 0) entering_window = delta_angle < leaving_angle;
    if (target_w < 0) entering_window = delta_angle > -leaving_angle;
    if (!entering_window) continue;

    const double score = std::abs(delta_angle);
    if (score < best_score) {
      best_score = score;
      best_id = i;
    }
  }

  if (best_id == -1) {
    fallback_to_closest();
    return selection;
  }

  lock_id = -1;
  fill_selection(best_id, true);
  return selection;
}
```

**前哨选板逻辑**：

1. **未 jump**：锁定 `last_id`，不切板
2. **Spin Gate**（前哨始终启用）：选择正在进入 `leaving_angle` 窗口的板
   - 正转：`delta_angle < leaving_angle`
   - 反转：`delta_angle > -leaving_angle`
3. **Fallback**：若无候选，选绝对 `delta_angle` 最小的板

### 6.3 开火相位门

代码位置：[planner.cpp:389-397](tasks/auto_aim/planner/planner.cpp#L389-L397)

```cpp
  bool fire_ready = debug_fire_track_ready;
  if (target.name == ArmorName::outpost) {
    const bool spin_gate_ready = !target.jumped || selection.used_spin_gate;
    debug_fire_phase_limit = resolve_outpost_fire_phase_angle(leaving_angle);
    debug_fire_phase_ready =
      spin_gate_ready && std::abs(selection.selected_delta_angle) <= debug_fire_phase_limit;
    fire_ready = fire_ready && hit_solution.converged && debug_fire_phase_ready;
  }
```

**前哨开火必须同时满足**：

1. 轨迹跟踪误差 < `fire_thresh`
2. 命中时刻迭代收敛：`hit_solution.converged == true`
3. 相位门通过：`|delta_angle|` ≤ `fire_phase_limit`
4. 若已 `jumped`，必须通过 spin gate 选板（非 fallback）

**相位门限计算**：

代码位置：[planner.cpp:28-35](tasks/auto_aim/planner/planner.cpp#L28-L35)

```cpp
double resolve_outpost_fire_phase_angle(double leaving_angle)
{
  if (leaving_angle <= 1e-6) return 8.0 / 57.3;
  const double fire_angle =
    std::clamp(leaving_angle * 0.5, kMinOutpostFireAngle, kMaxOutpostFireAngle);
  return std::min(fire_angle, leaving_angle);
}
```

其中 `kMinOutpostFireAngle = 4°`，`kMaxOutpostFireAngle = 12°`。

### 6.4 板级高度补偿

代码位置：[planner.cpp:611-621](tasks/auto_aim/planner/planner.cpp#L611-L621)

```cpp
double Planner::resolve_aim_z_compensation(const Target & target, int armor_id) const
{
  if (target.name != ArmorName::outpost || armor_id < 0) return 0.0;
  if (outpost_fire_z_compensation_.size() != 3) return 0.0;

  const int physical_id = target.physical_armor_id(armor_id);
  if (physical_id < 0 || physical_id >= static_cast<int>(outpost_fire_z_compensation_.size())) {
    return 0.0;
  }
  return outpost_fire_z_compensation_[physical_id];
}
```

不同物理板可以有不同的额外击打高度补偿，配置项 `outpost_fire_z_compensation` 默认为 `[0.0, 0.0, 0.0]`。

---

## 7. 前哨站专用逻辑总结

| 环节 | 代码位置 | 前哨特殊处理 |
|------|---------|-------------|
| 初始化 | `tracker.cpp:439` | `armor_num=3`，固定中心模型，专用半径和高度偏移 |
| 匹配 | `tracker.cpp:50-129` | 枚举 3×3 种 `(id, offset)` 组合，综合 7 项几何误差 + 连续性惩罚 |
| 门限 | `tracker.cpp:131` | `reproj<90px, xy<0.4m, z<0.2m, score<36` |
| 板号映射 | `target.cpp:325` | `set_armor_id_offset()` 调整 `ekf_.x[4]` 保持高度连续 |
| 几何模型 | `target.cpp:382` | 三板共用半径，高度差由 `armor_z_offsets` 决定 |
| 预测 | `target.cpp:98` | 固定中心旋转，速度压零，只积分角度 |
| 角速度锁 | `target.cpp:169` | 收敛后 `|vyaw|>2` 时锁到 `±2.51 rad/s` |
| 量测更新 | `target.cpp:231` | 4 维观测量 `[yaw, pitch, distance, armor_yaw]`，自适应噪声 |
| 收敛判定 | `target.cpp:374` | 前哨需要 10 次更新（普通目标 3 次） |
| 选板 | `planner.cpp:48` | 前哨始终走 spin gate 选板，未 jump 时锁定当前板 |
| 命中迭代 | `planner.cpp:179` | 固定点迭代：选板→弹道→预测→再选板 |
| 开火门 | `planner.cpp:389` | 相位门 `4°~12°` + 收敛门 + spin gate 门 |
| 高度补偿 | `planner.cpp:611` | 按物理板号查表补偿 |
