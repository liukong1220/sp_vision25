# 标定流程与参数回填清单

## 1. 结论先看（你最关心的）

1. `camera_matrix`、`distort_coeffs`：只填 `calibrate_camera` 的输出。
2. `R_camera2gimbal`、`t_camera2gimbal`：优先填 `calibrate_robotworld_handeye` 的输出。
3. `calibrate_handeye` 的结果建议作为对照，不建议单独作为最终上车值。
4. 运行时生效配置是具体车型配置（如 `configs/standard3.yaml`），不是 `configs/calibration.yaml`。

## 2. 代码链路与合理性评审

### 2.1 当前链路（合理）

1. `capture`：采集 `N.jpg + N.txt`（`txt` 为 `w x y z` 四元数）。
2. `calibrate_camera`：用圆点板求内参 + 畸变。
3. `calibrate_handeye`/`calibrate_robotworld_handeye`：用内参 + 四元数 + 标定板观测求外参。
4. `auto_aim::Solver`：运行时读取 YAML 的 `camera_matrix/distort_coeffs/R_camera2gimbal/t_camera2gimbal` 做 PnP 与坐标变换。

### 2.2 已做的必要改进（本次已改代码）

1. `capture.cpp`：按 `s` 时仅在圆点板识别成功才保存，避免无效样本污染数据集。
2. `calibrate_handeye.cpp`：`solvePnP` 失败样本直接跳过。
3. `calibrate_robotworld_handeye.cpp`：`solvePnP` 失败样本直接跳过。

### 2.3 仍需你们关注的风险点

1. `calibrate_handeye` 中 `t_gimbal2world` 固定为 0，平移解对数据质量更敏感，容易出现量级异常。
2. 标定程序按 `1.jpg,2.jpg,...` 连续编号读取，中间断号会提前停止读取。
3. 外参平移异常大（例如 `|t_camera2gimbal| > 0.3m`）通常说明数据或坐标定义有问题，不建议直接上车。

## 3. 详细操作步骤（推荐流程）

### 3.1 先准备配置

编辑 `configs/calibration.yaml`：

- `pattern_cols`、`pattern_rows`：圆点板列数/行数（必须和实物一致）。
- `center_distance_mm`：圆点中心距（mm）。
- `R_gimbal2imubody`：云台坐标系到 IMU 体坐标系的旋转（安装定义，非标定自动求解）。
- 相机与串口基础参数（`camera_name`、`exposure_ms`、`gain`、`com_port`）。

### 3.2 采集数据

```bash
./build/capture -c configs/calibration.yaml -o assets/calib_2026_03_12
```

采集要求：

1. 角度覆盖：至少覆盖明显的 yaw/pitch 变化，不要只在正前方。
2. 空间覆盖：标定板尽量覆盖图像中心和四角。
3. 距离覆盖：近中远都要有。
4. 每次按 `s` 保存前，确认窗口中圆点识别是成功状态。
5. 建议至少 20 组有效样本（手眼最少 6 组，内参最少 5 组）。

### 3.3 内参标定

```bash
./build/calibrate_camera assets/calib_2026_03_12 -c configs/calibration.yaml
```

记录输出：

- `camera_matrix`
- `distort_coeffs`
- 重投影误差（建议 `<= 0.35px`）

### 3.4 外参标定（优先 robotworld）

```bash
./build/calibrate_robotworld_handeye assets/calib_2026_03_12 -c configs/calibration.yaml
```

记录输出：

- `R_camera2gimbal`
- `t_camera2gimbal`（程序已转成 m）
- 注释中的相机偏角、标定板姿态与距离信息（用于人工验算）

### 3.5 对照跑一遍 handeye（用于交叉验证）

```bash
./build/calibrate_handeye assets/calib_2026_03_12 -c configs/calibration.yaml
```

若两者差异很大：

1. 先排查样本质量（覆盖/模糊/识别成功率）。
2. 再排查 `R_gimbal2imubody` 和四元数定义（是否 `wxyz`、轴向是否一致）。
3. 通过后再决定是否重采，不建议盲目选“看起来更顺眼”的结果。

## 4. 参数回填映射（填到哪里）

回填目标：具体车型配置文件（如 `configs/standard3.yaml`、`configs/standard4.yaml`）。

| 运行参数键 | 来自哪个程序 | 应填内容 |
|---|---|---|
| `camera_matrix` | `calibrate_camera` | 同名输出数组 |
| `distort_coeffs` | `calibrate_camera` | 同名输出数组 |
| `R_camera2gimbal` | `calibrate_robotworld_handeye`（优先） | 同名输出数组 |
| `t_camera2gimbal` | `calibrate_robotworld_handeye`（优先） | 同名输出数组（单位 m） |
| `R_gimbal2imubody` | 非本流程自动估计 | 按安装定义人工维护 |

注意：

1. 一套内参/外参必须来自同一批次数据，避免混用。
2. 不同机器人不可直接拷贝外参。

## 5. 结果验收清单（上车前）

- [ ] 有效样本数满足要求（建议 >=20）。
- [ ] 内参重投影误差在可接受范围（建议 <=0.35px）。
- [ ] 外参旋转接近安装直觉（相机偏角通常是几度以内到十来度）。
- [ ] `t_camera2gimbal` 量级合理（多数车是厘米级到十几厘米）。
- [ ] 上车后近距/中距静止目标，左右和高低偏差无明显系统性偏置。
- [ ] 切换配置文件时未串用他车参数。

## 6. 常见错误

1. 把 `calibration.yaml` 当成运行配置，忘记回填到车型配置。
2. 四元数顺序弄错（本项目采集与读取均按 `w x y z`）。
3. 图像编号断号导致只读取到前半段样本。
4. 只采中心区域、角度变化太小，导致手眼结果不稳定。
5. 外参异常仍直接上车，不做量级与实测检查。
