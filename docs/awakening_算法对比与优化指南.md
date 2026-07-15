# sp_vision25 与 awakening：算法对比、推导与优化迁移指南

> 分析日期：2026-07-15
>
> `awakening` 基线：`d7d5ed2ed868e802bc1e3394f4590f59d15cc00f`（2026-07-14）
>
> `sp_vision25` 基线：提交 `a08a9d92cc2a068fcf71bf4e970053ccec3d5680` 加当前工作树改动
>
> 13 维实现状态：2026-07-15 已迁移到当前工作树，并通过合成 SO(3)/UVL 测试和离线程序链路验证；尚未完成实车精度验收。

本文基于两个仓库的实际源码与 `awakening/docs/algorithm/`，目标不是把 `awakening` 整体移植进来，而是回答四个工程问题：

1. `sp_vision25` 与 `awakening` 在整车状态估计、轨迹优化、并发推理和 Web 可视化上到底有什么差异。
2. 两套算法的状态、预测、观测、EKF/NIS 和轨迹公式如何推导。
3. 当前工作树已经完成了哪些改造，哪些仍只是界面预留或迁移建议。
4. 如何通过离线回放和可量化实验学习这些算法，而不是凭现场感觉调参。

文中的状态标签含义如下：

| 标签 | 含义 |
| --- | --- |
| **已落地** | 当前 `sp_vision25` 工作树中已有对应源码；不等于已经通过实车验收或已经提交发布 |
| **参考实现** | `awakening` 当前源码中存在，但 `sp_vision25` 尚未实现 |
| **建议** | 本文给出的迁移或实验方案，不能当作当前能力 |
| **未知** | 仅凭仓库无法确认，需要实车、数据集、固件协议或作者说明 |

## 1. 先给结论

1. `sp_vision25` 当前已使用 **13 维 SO(3) 误差状态整车模型**：完整目标保留 roll/pitch/yaw，姿态以旋转向量存储并采用右乘注入；前哨和基地按参考实现退化为 yaw-only。普通目标的半径以 `log_r1/log_r2` 存储，观测已从 PnP 后的 YPD 改为单块装甲左右灯条组成的 8 维 UVL。
2. 与 `awakening` 相比，当前迁移的是状态、SO(3) 注入、完整刚体投影、UVL、NIS 和 Joseph 更新；尚未迁移 **同帧联合 `update_multi`、单灯条/深度差观测、预测 ROI 反哺检测、Ceres 自动微分**。因此不能将当前实现描述为与 awakening 功能完全等价。
3. `sp_vision25` 的 TinyMPC 与 `awakening::LimitTrajectory` 解决的是相近但不相同的问题。前者通过带加速度约束的二次优化生成可执行参考；后者用五次多项式和局部时间扩展解析地平滑切板。**不建议立刻替换 TinyMPC**，更合理的是先让 `LimitTrajectory` 以 shadow/A-B 模式运行，再决定作为默认、前置整形或求解失败回退。
4. `awakening` 也不是“只用五次多项式”。`very_aimer.type == mpc` 时，它会改走 `DualSmallMpcTrajectory`；非 MPC 分支才查询 `LimitTrajectory`。因此不能把 `LimitTrajectory` 描述为 `awakening` 唯一控制方案。
5. 并发优化的目标应是降低“图像采集到指令发送”的数据年龄，而不是只提高 FPS。`sp_vision25` 当前仅在 `mt_standard` 的 OpenVINO 异步链路上加入有界在途请求和过载丢帧；`standard_mpc` 的检测仍是同步调用。
6. `sp_vision25` 的 Web 已经具备比 `awakening` 更直接的运行时热调链路；`awakening` 值得借鉴的是共享内存解耦和独立 Web 进程。当前新增的估计/MPC 诊断已经有 C++ 数据源；`auto_aim_debug_mpc` 和 `auto_aim_test_web` 也已提供同步 YOLO/Tracker/Planner 分段耗时，但真正使用异步队列的 `mt_standard` 尚未把队列、丢帧和吞吐数据送入 Web。

## 2. 架构与能力差异

| 维度 | `sp_vision25` 当前源码 | `awakening` 当前源码 | 对 `sp_vision25` 的判断 |
| --- | --- | --- | --- |
| 语言与主要依赖 | C++17、OpenCV、Eigen、OpenVINO；ROS2 依赖按环境可选 | C++23、OpenCV、Eigen、Ceres、TBB；OpenVINO/TensorRT 可选，ROS2/Rerun/Daedalus 可选 | 不能直接复制依赖 C++23、Ceres Jet、TBB 或 CUDA 的实现 |
| 运行时组织 | 多个入口各自写循环；部分入口手工创建规划/命令线程 | `Scheduler` 注册 typed source/task，TBB task group 执行，控制/调试使用独立 rate source | 先借鉴背压与数据契约，不宜先整体替换调度器 |
| 标准 MPC 主链 | 相机、同步 YOLO、Tracker 在主线程；容量 1 的 latest-target 队列交给规划线程 | 相机 source、并发 Detector、OrderedQueue、串行 Tracker、SWMR 目标快照、1000 Hz solver source | `standard_mpc` 尚未获得并发检测收益 |
| 异步检测入口 | `mt_standard` 为单生产线程 + OpenVINO async request FIFO；按 FIFO 取最老请求并等待 | task 级并发 + 非阻塞信号量；结果按 frame id 重排 | SP 已有最小背压，但仍有队首阻塞和请求反复创建 |
| 推理后端 | 当前异步类只封装 OpenVINO | OpenVINO `InferRequest` 池；TensorRT 独立 execution context、buffer、stream 池 | 后端资源所有权是最值得迁移的部分 |
| 整车状态 | 13 维：中心位置/速度、SO(3) 旋转向量、yaw 角速度、对数半径/高度槽位 | 13 维：中心位置/速度、SO(3) 旋转向量、yaw 角速度、对数半径/高度槽位 | 已统一普通目标状态语义；前哨的 dz 槽位仍保留 SP 固定高度偏置 |
| 特殊目标 | 前哨可用固定中心旋转模型、固定转速、3 块板独立 z 偏置 | 前哨固定半径、方向投票、3 块板独立 dz；基地/前哨姿态退化为主要 yaw | 两边语义相近但状态槽位和编号映射不同 |
| 常规观测 | 一块完整板的左右灯条 `[angle,u,v,length]` 堆叠为 8 维 UVL | 实际更新主要用灯条 `[angle,u,v,length]`，单板时补左右灯条深度差 | 已避免把 PnP 深度抖动直接写入滤波；SP 尚未使用单灯条/深度差辅助观测 |
| 同帧更新 | 每块匹配装甲板做一次 8 维顺序更新；前哨取最佳匹配 | 完整板拆两条灯条，未占用单灯条也可加入，堆叠后 `update_multi` | SP 的多板顺序更新不等价于同帧联合更新 |
| 滤波形式 | 误差状态 EKF，SO(3) 右乘注入、数值雅可比、NIS 门控、Joseph 形式 | 误差状态 EKF，SO(3) 右乘注入，支持迭代和联合观测 | 当前主要差异是联合观测、自动微分和观测覆盖范围 |
| 匹配/门控 | 最近板候选 + 角度代价；前哨另有重投影/位置综合匹配；更新前做 NIS | 可见面预测 + 四边形中心/边角/周长代价 + 贪心匹配；灯条另设几何门限 | 两级门控更稳：先关联门控，再统计门控 |
| 目标反哺检测 | 常规入口仍以完整图像/既有 ROI 配置为主 | 整车状态预测所有灯条，生成随丢失时间扩张的网络/CV ROI | 属于高价值的第二阶段迁移项 |
| 轨迹优化 | yaw/pitch 分轴 TinyMPC，100 点、10 ms；输出中点状态 | 默认可用解析五次多项式 `LimitTrajectory`，也保留 TinyMPC/双轴 MPC 可选实现 | 应先 A-B 验证，不应按项目名称直接替换 |
| Web 数据路径 | C++ 内置 HTTP，内存 JSON/JPEG，MJPEG + polling，支持参数 POST/reset | C++ 写 `/dev/shm` JPEG/JSON，独立 Flask 进程读取并提供 MJPEG/JSON | SP 热调更完整；awakening 的进程解耦更强 |
| 参数在线生效 | `runtime_params` 维护 effective value、override、version，各模块主动刷新 | 当前 Web 主要读共享内存数据，未见等价的通用热调写回链 | 不应为了参考 awakening 放弃 SP 已有热调能力 |
| 可视化后端 | Web + Plotter/OpenCV 调试画面 | Web + 可选 ROS2/Rerun | 是否引入 Rerun 取决于记录/回放需求，不是自瞄算法前置条件 |

两个主链路可以简化为：

```text
sp_vision25 / standard_mpc
Camera -> synchronous YOLO -> Tracker -> latest Target queue(size=1)
                                      -> Planner thread -> Gimbal

sp_vision25 / mt_standard
Camera thread -> OpenVINO async request FIFO(cap=N) -> pop oldest + wait
              -> Tracker -> latest command input -> CommandGener thread

awakening / standard
Camera source -> Scheduler detector tasks -> semaphore -> backend resource pool
              -> OrderedQueue(frame id) -> serialized Tracker -> SWMR target
              -> 1000 Hz solver source -> serial
              -> 45 Hz debug source -> shared memory -> Flask Web
```

## 3. `sp_vision25` 当前的 13 维 SO(3) 整车滤波

### 3.1 当前实现、公式与约束

当前状态顺序与 awakening 对齐：

$$
\mathbf{x}=
[c_x,v_x,c_y,v_y,c_z,v_z,\phi_z,\omega_z,\log r_1,\log r_2,h,\phi_y,\phi_x]^T.
$$

其中旋转向量按 $[\phi_x,\phi_y,\phi_z]^T$ 重组：

$$
\mathbf{R}_{car}^{world}=\operatorname{Exp}_{SO(3)}([\phi_x,\phi_y,\phi_z]^T).
$$

普通四装甲目标使用 $r_i=\exp(\log r_1)$ 或 $\exp(\log r_2)$，奇数板有高度偏置 $h$。第 $i$ 块板的刚体位姿为：

$$
\theta_i=\frac{2\pi i}{N},\qquad
{}^{car}\mathbf{p}_i=[-r_i\cos\theta_i,-r_i\sin\theta_i,dz_i]^T,
$$

$$
{}^{world}\mathbf{p}_i=\mathbf{c}+\mathbf{R}_{car}^{world}{}^{car}\mathbf{p}_i,
\quad
{}^{world}\mathbf{R}_{armor,i}=\mathbf{R}_{car}^{world}
\mathbf{R}_z(\theta_i)\mathbf{R}_y(15^\circ).
$$

前哨装甲板倾角为 $-15^\circ$，并继续使用 `outpost_armor_z_offsets` 保留 SP 原有三板固定高度模型。基地和前哨只保留 yaw；普通目标才更新完整 roll/pitch。当前 13 维并没有 roll/pitch 的角速度，故它们是随机游走估计量，适合跟随和补偿缓慢姿态变化，**不能提前外推快速俯仰或侧倾**。

预测使用常速度和平面 yaw 角速度：

$$
\mathbf{c}^+=\mathbf{c}+\mathbf{v}\Delta t,
\qquad
\mathbf{R}^+=\mathbf{R}\operatorname{Exp}([0,0,\omega_z\Delta t]^T).
$$

误差注入与差分为右扰动：

$$
\mathbf{R}\boxplus\delta\boldsymbol\phi=
\mathbf{R}\operatorname{Exp}(\delta\boldsymbol\phi),\qquad
\delta\boldsymbol\phi=
\operatorname{Log}(\mathbf{R}_{nominal}^T\mathbf{R}_{value}).
$$

平移、yaw/角速度的过程噪声采用常加速度离散块；roll/pitch 加 `tracker_roll_pitch_random_walk\,|\Delta t|`；半径随机游走在对数空间传播：

$$
Q_{\log r\log r}=q_g|\Delta t|/r^2,
\qquad Q_{hh}=q_g|\Delta t|.
$$

观测不再使用 YPD。每一灯条观测为

$$
\mathbf{z}_{UVL}=[\beta,u_c,v_c,L]^T,
\quad
\beta=\operatorname{atan2}(u_t-u_b,v_t-v_b),
$$

$$
[u_c,v_c]^T=\tfrac12([u_t,v_t]^T+[u_b,v_b]^T),
\qquad L=\|[u_t-u_b,v_t-v_b]^T\|.
$$

完整装甲板的左右灯条组成 8 维量测。由完整三维刚体位姿经 `Solver::reproject_armor()` 投影得到预测值；观测雅可比以中心有限差分计算，并对姿态列使用上述 $\boxplus$。`tracker_uvl_angle_variance`、`tracker_uvl_center_variance`、`tracker_uvl_length_variance` 构成对角 $\mathbf R$。投影或雅可比不完整/非有限时直接拒绝本次更新。

创新、NIS、Joseph 协方差更新仍为：

$$
\boldsymbol\nu=\mathbf z-h(\hat{\mathbf x}^{-}),\quad
\epsilon=\boldsymbol\nu^T(\mathbf H\mathbf P^-\mathbf H^T+\mathbf R)^{-1}\boldsymbol\nu,
$$

$$
\mathbf P^+=(\mathbf I-\mathbf K\mathbf H)\mathbf P^-(\mathbf I-\mathbf K\mathbf H)^T+
\mathbf K\mathbf R\mathbf K^T.
$$

当前默认 `tracker_nis_gate=20.090`，是 $\chi^2_8$ 的 99% 门限（95% 为 15.507，99.9% 为 26.125）。通过门控后，状态以 $\boxplus$ 注入并对协方差对称化；拒绝时状态、协方差与跟踪元数据均不推进。

源码与验证证据：[`target.cpp`](../tasks/auto_aim/target.cpp)、[`solver.cpp`](../tasks/auto_aim/solver.cpp)、[`tracker.cpp`](../tasks/auto_aim/tracker.cpp)、[`target_13d_test.cpp`](../tests/target_13d_test.cpp)。

### 3.2 迁移前的 11 维模型（历史基线）

以下模型用于理解改造前差异，**不再是当前 `Target` 源码的状态或观测实现**。

源码状态顺序为：

$$
\mathbf{x}=
[c_x,v_x,c_y,v_y,c_z,v_z,\theta,\omega,r,l,h]^T.
$$

其中：

| 状态 | 含义 |
| --- | --- |
| $c_x,c_y,c_z$ | 整车旋转中心在 world 系的位置 |
| $v_x,v_y,v_z$ | 整车中心平移速度 |
| $\theta$ | 第 0 块装甲板对应的车体 yaw 相位 |
| $\omega$ | yaw 角速度 |
| $r$ | 第一组装甲板半径 |
| $l$ | 四装甲目标第二组半径与第一组半径之差，即 $r_2=r+l$ |
| $h$ | 四装甲目标第二组相对高度差 |

第 $i$ 块装甲板的相位为：

$$
\theta_i=\operatorname{wrap}\left(\theta+\frac{2\pi i}{N}\right).
$$

对普通四装甲目标，令 $b_i=1$ 表示 $i\in\{1,3\}$，否则 $b_i=0$，则：

$$
r_i=r+b_i l,
$$

$$
\mathbf{p}_i=
\begin{bmatrix}
c_x-r_i\cos\theta_i\\
c_y-r_i\sin\theta_i\\
c_z+b_i h
\end{bmatrix}.
$$

前哨站使用 3 块板时不使用上述奇偶 $l/h$ 结构，而是：

$$
p_{i,z}=c_z+\Delta z_i,
$$

其中 $\Delta z_i$ 来自 `outpost_armor_z_offsets`。`armor_id_offset_` 还负责把滤波器局部编号映射到物理板编号，并在重映射时保持当前参考板高度连续。

源码证据：[`target.cpp`](../tasks/auto_aim/target.cpp)、[`target.hpp`](../tasks/auto_aim/target.hpp)。

#### 3.2.1 预测模型

普通目标使用常速度/常角速度模型：

$$
\begin{aligned}
c_x^+ &= c_x+v_x\Delta t, & v_x^+&=v_x,\\
c_y^+ &= c_y+v_y\Delta t, & v_y^+&=v_y,\\
c_z^+ &= c_z+v_z\Delta t, & v_z^+&=v_z,\\
\theta^+&=\operatorname{wrap}(\theta+\omega\Delta t), & \omega^+&=\omega,\\
r^+&=r, & l^+&=l,\quad h^+=h.
\end{aligned}
$$

对任意一个位置/速度二元组 $[p,v]^T$，状态转移块为：

$$
\mathbf{F}_{pv}=
\begin{bmatrix}
1&\Delta t\\
0&1
\end{bmatrix}.
$$

若把一个采样周期内的未知加速度近似成白噪声 $a_k$，且离散增量为

$$
\Delta\mathbf{x}=
\begin{bmatrix}
\frac{1}{2}\Delta t^2\\
\Delta t
\end{bmatrix}a_k,
$$

当 $\operatorname{Var}(a_k)=q_a$ 时，过程噪声块为：

$$
\mathbf{Q}_{pv}=q_a
\begin{bmatrix}
\frac{\Delta t^4}{4}&\frac{\Delta t^3}{2}\\
\frac{\Delta t^3}{2}&\Delta t^2
\end{bmatrix}.
$$

平移三个轴共用 `tracker_acceleration_variance`，yaw/角速度块使用 `tracker_yaw_acceleration_variance`。几何状态当前加入随机游走：

$$
Q_{rr}=Q_{ll}=Q_{hh}=q_g|\Delta t|,
$$

其中 $q_g$ 对应 `tracker_geometry_random_walk`。

前哨站有两个必须注意的分支：

1. 非固定中心模型仍走通用 $F/Q$，但平移加速度方差上限被限制为 `10`，yaw 角加速度方差上限被限制为 `0.1`。
2. `outpost_fixed_center_rotation_model=true` 时，代码每次预测把 $v_x,v_y,v_z$ 置零，只推进 $\theta$；该分支使用内部固定的对角过程噪声，**不会使用新加入的平移/yaw/几何过程噪声参数**。量测噪声和 NIS 门限仍会生效。

因此，不能看到 Web 上出现过程噪声参数就推断它们对固定中心前哨分支全部有效。

#### 3.2.2 观测模型

`sp_vision25` 把单块装甲板 PnP/坐标转换后的量组织为：

$$
\mathbf{z}=[\psi,\varphi,d,\alpha]^T,
$$

分别表示视线 yaw、视线 pitch、距离和装甲板 yaw。

对预测装甲板位置 $\mathbf{p}_i=[X,Y,Z]^T$，令

$$
\rho=\sqrt{X^2+Y^2},\qquad d=\sqrt{X^2+Y^2+Z^2},
$$

则非线性观测函数为：

$$
\mathbf{h}_i(\mathbf{x})=
\begin{bmatrix}
\operatorname{atan2}(Y,X)\\
\operatorname{atan2}(Z,\rho)\\
d\\
\theta_i
\end{bmatrix}.
$$

从笛卡尔位置到 YPD 的雅可比为：

$$
\mathbf{J}_{ypd}=
\begin{bmatrix}
-Y/\rho^2 & X/\rho^2 & 0\\
-XZ/(\rho d^2) & -YZ/(\rho d^2) & \rho/d^2\\
X/d & Y/d & Z/d
\end{bmatrix}.
$$

装甲板笛卡尔位置对整车状态的关键偏导为：

$$
\frac{\partial X}{\partial c_x}=1,\quad
\frac{\partial Y}{\partial c_y}=1,\quad
\frac{\partial Z}{\partial c_z}=1,
$$

$$
\frac{\partial X}{\partial\theta}=r_i\sin\theta_i,\qquad
\frac{\partial Y}{\partial\theta}=-r_i\cos\theta_i,
$$

$$
\frac{\partial X}{\partial r}=-\cos\theta_i,\qquad
\frac{\partial Y}{\partial r}=-\sin\theta_i,
$$

奇数组装甲板还具有相同的 $\partial/\partial l$，以及 $\partial Z/\partial h=1$。最终：

$$
\mathbf{H}_i=
\frac{\partial[\psi,\varphi,d,\alpha]}{\partial[X,Y,Z,\theta_i]}
\frac{\partial[X,Y,Z,\theta_i]}{\partial\mathbf{x}}.
$$

代码中的 `h_jacobian()` 正是这两个雅可比的链式相乘。

当前量测协方差为对角阵：

$$
\mathbf{R}=\operatorname{diag}
(R_\psi,R_\varphi,R_d,R_\alpha),
$$

其中：

$$
R_\psi=q_{\psi},\qquad R_\varphi=q_{\varphi},
$$

$$
R_d=\log(1+|\alpha_{obs}-\psi_{center}|)+q_d,
$$

$$
R_\alpha=\frac{\log(1+|d_{obs}|)}{200}+q_\alpha.
$$

这是迁移前的量测定义；当前配置改为 `tracker_uvl_*_variance`，仅两条灯条 angle 残差使用 $[-\pi,\pi)$ 归一化。

#### 3.2.3 EKF、NIS 与 Joseph 更新

预测为：

$$
\hat{\mathbf{x}}_k^-=f(\hat{\mathbf{x}}_{k-1}^+),
$$

$$
\mathbf{P}_k^-=\mathbf{F}_k\mathbf{P}_{k-1}^+\mathbf{F}_k^T+\mathbf{Q}_k.
$$

更新前先计算先验创新：

$$
\boldsymbol{\nu}_k=\mathbf{z}_k-h(\hat{\mathbf{x}}_k^-),
$$

$$
\mathbf{S}_k=\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T+\mathbf{R}_k.
$$

归一化创新平方 NIS 为：

$$
\epsilon_k=\boldsymbol{\nu}_k^T\mathbf{S}_k^{-1}\boldsymbol{\nu}_k.
$$

当前实现用 Eigen LDLT 求解 $\mathbf{S}^{-1}\nu$ 和 $\mathbf{S}^{-1}(PH^T)^T$，没有显式求逆。若 LDLT 失败、NIS 非有限或

$$
\epsilon_k>\gamma,
$$

则拒绝整次量测，状态与协方差保持预测值。4 维创新在理想高斯且模型一致时近似服从 $\chi^2_4$：

| 置信水平 | $\chi^2_4$ 门限 |
| --- | ---: |
| 95% | 9.488 |
| 99% | 13.277 |
| 99.9% | 18.467 |

历史模型的默认值为 `13.277`；当前 8 维 UVL 使用 `tracker_nis_gate=20.090`。门限不是“越大越好”：门限增大会减少误拒绝，同时也会放入更多错配或图像离群值。

通过门控后：

$$
\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^T\mathbf{S}_k^{-1},
$$

$$
\delta\mathbf{x}_k=\mathbf{K}_k\boldsymbol{\nu}_k,
$$

$$
\hat{\mathbf{x}}_k^+=\hat{\mathbf{x}}_k^-\boxplus\delta\mathbf{x}_k.
$$

历史模型中 $\boxplus$ 是普通加法，只对 $\theta$ 做角度归一化；当前实现改为第 3.1 节的 SO(3) 右扰动。协方差使用 Joseph 形式：

$$
\mathbf{P}_k^+=(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}_k^-
\cdot(\mathbf{I}-\mathbf{K}\mathbf{H})^T+\mathbf{K}\mathbf{R}\mathbf{K}^T,
$$

随后强制对称化：

$$
\mathbf{P}\leftarrow\frac{1}{2}(\mathbf{P}+\mathbf{P}^T).
$$

源码证据：[`extended_kalman_filter.cpp`](../tools/extended_kalman_filter.cpp)、[`estimator_pipeline_test.cpp`](../tests/estimator_pipeline_test.cpp)。

#### 3.2.4 统计量的正确解释

**已落地：**

- `nis`、`nis_fail`、`update_accepted`。
- 累计 `accepted_updates/rejected_updates`。
- 最近最多 100 次的拒绝率 `recent_nis_failures`。
- 被拒绝量测不会推进 `last_id`、`jumped`、`switch_count_` 或 `update_count_`，也不会触发几何状态重置。
- Tracker 只把被滤波器接受的量测视为 `found`。

**必须纠正一个名称：**当前 `data["nees"]` 计算的是

$$
\delta\mathbf{x}^T(\mathbf{P}^+)^{-1}\delta\mathbf{x},
$$

即“本次修正量的归一化能量”，不是统计学定义的 NEES。真正的 NEES 需要真值：

$$
\operatorname{NEES}=(\hat{\mathbf{x}}-\mathbf{x}_{truth})^T
\mathbf{P}^{-1}(\hat{\mathbf{x}}-\mathbf{x}_{truth}).
$$

在没有 mocap、仿真真值或可靠离线标注时，Web 上的 `nees` 不能用于宣称滤波一致性。建议后续把现字段改名为 `correction_energy`，另建真正的离线 NEES 评估。

## 4. `awakening` 的 13 维 SO(3) 误差状态

### 4.1 精确状态顺序

`awakening` 源码枚举的存储顺序为：

$$
\mathbf{x}=
[c_x,v_x,c_y,v_y,c_z,v_z,\phi_z,\omega_z,\log r_1,p_1,p_2,\phi_y,\phi_x]^T.
$$

为了理解姿态，应把非连续存储的 $[\phi_x,\phi_y,\phi_z]^T$ 重新组合成 SO(3) 旋转向量：

$$
\mathbf{R}_{car}^{odom}=\operatorname{Exp}_{SO(3)}
([\phi_x,\phi_y,\phi_z]^T).
$$

状态槽位会按目标类型复用：

| 目标 | $p_1$ | $p_2$ | 半径处理 |
| --- | --- | --- | --- |
| 普通四装甲目标 | $\log r_2$ | 高度差 $h$ | $r_j=\exp(\log r_j)>0$ |
| 前哨站 | 第 1 块板 dz | 第 2 块板 dz | $\log r_1$ 被固定到前哨半径 |
| 基地 | 结构槽位仍存在 | 结构槽位仍存在 | `armor_radius()` 对基地返回 0 |

对普通目标，第 $i$ 块板在车体系的相对位姿为：

$$
\theta_i=\frac{2\pi i}{N},
$$

$$
{}^{car}\mathbf{p}_{armor,i}=
[-r_i\cos\theta_i,-r_i\sin\theta_i,dz_i]^T,
$$

$$
{}^{odom}\mathbf{T}_{armor,i}=
{}^{odom}\mathbf{T}_{car}(\mathbf{x})
{}^{car}\mathbf{T}_{armor,i}.
$$

源码证据：[`motion_model.hpp`](../../awakening/src/tasks/auto_aim/armor_track/motion_model.hpp)。

### 4.2 为什么是误差状态

旋转向量可以用 3 个数存储，但旋转本身不属于普通三维向量空间。`awakening` 对平移、速度、角速度和几何量做加法，对姿态做右乘误差注入：

$$
\mathbf{R}\boxplus\delta\boldsymbol\phi
=\mathbf{R}\operatorname{Exp}(\delta\boldsymbol\phi).
$$

两个名义姿态之间的误差定义为：

$$
\delta\boldsymbol\phi
=\operatorname{Log}(\mathbf{R}_{nominal}^T\mathbf{R}_{value}).
$$

于是：

$$
\mathbf{x}_{value}=\mathbf{x}_{nominal}\boxplus\delta\mathbf{x},
$$

$$
\delta\mathbf{x}=\mathbf{x}_{value}\boxminus\mathbf{x}_{nominal}.
$$

这样协方差描述的是当前姿态切空间内的小误差，而不是把三个旋转向量分量当作全球一致的欧氏坐标直接相加。

需要特别限定：普通目标会构造完整 SO(3) 姿态；`car_rotation()` 对前哨站和基地只使用 $\phi_z$，因此“13 维”不代表所有目标都有效估计 roll/pitch。

### 4.3 预测和过程噪声

平移仍是常速度：

$$
\mathbf{c}^+=\mathbf{c}+\mathbf{v}\Delta t.
$$

姿态预测为：

$$
\mathbf{R}^+=\mathbf{R}\operatorname{Exp}([0,0,\omega_z\Delta t]^T).
$$

前哨站方向投票明确后，$\omega_z$ 被设为 $\pm2.51\ \mathrm{rad/s}$；基地约束 $\omega_z=0$。

平移加速度方差先定义在车体系，再旋转到 odom：

$$
\mathbf{Q}_{a}^{odom}=\mathbf{R}\mathbf{Q}_{a}^{body}\mathbf{R}^T.
$$

位置/速度块仍由

$$
\begin{bmatrix}
\frac14\Delta t^4\mathbf{Q}_a & \frac12\Delta t^3\mathbf{Q}_a\\
\frac12\Delta t^3\mathbf{Q}_a & \Delta t^2\mathbf{Q}_a
\end{bmatrix}
$$

构造。yaw/角速度使用同型标量块，roll/pitch 误差另加 `q_wpr * dt` 随机游走。

半径使用 $s=\log r$。由一阶误差传播

$$
\delta s\approx\frac{\delta r}{r},
$$

若半径噪声方差为 $q_r$，则代码写入

$$
Q_{ss}=\frac{q_r}{r^2}.
$$

注意该项在当前实现中没有再乘 $\Delta t$，所以 `awakening.q_r` 与 `sp_vision25.tracker_geometry_random_walk` 的单位/采样周期语义不同，不能复制数值。

### 4.4 图像平面观测与多观测更新

`awakening` 当前自动瞄准更新实际使用的主要观测是 `UVLMeasure`：

$$
\mathbf{z}_{UVL}=[\beta,u_c,v_c,L]^T,
$$

其中：

$$
\beta=\operatorname{atan2}(u_t-u_b,v_t-v_b),
$$

$$
[u_c,v_c]^T=\frac12([u_t,v_t]^T+[u_b,v_b]^T),
$$

$$
L=\|[u_t-u_b,v_t-v_b]^T\|.
$$

预测值由整车状态生成装甲板/灯条三维端点，再经过相机外参、内参和畸变模型投影：

$$
\hat{\mathbf{z}}=
\Pi(\mathbf{K},\mathbf{D},
({}^{odom}\mathbf{T}_{camera})^{-1}
{}^{odom}\mathbf{T}_{armor,i}(\mathbf{x})\mathbf{P}_{light}).
$$

完整装甲板会拆成左右两条 UVL 观测；没有被完整装甲板占用的单灯条也可以加入同一帧更新。只有一块完整装甲板时，还可从 IPPE 结果取左右灯条中心深度差：

$$
z_{diff}=Z_{left}-Z_{right},
$$

用一维 `DiffMeasure` 补充斜视姿态约束。它不是把整套 PnP 位姿强行作为观测写入滤波器。

所有观测堆叠：

$$
\mathbf{r}=
[\mathbf{r}_1^T,\ldots,\mathbf{r}_M^T]^T,
\quad
\mathbf{H}=
[\mathbf{H}_1^T,\ldots,\mathbf{H}_M^T]^T,
$$

$$
\mathbf{R}=\operatorname{blkdiag}(\mathbf{R}_1,\ldots,\mathbf{R}_M).
$$

`ErrorStateEKF::update_multi()` 在误差状态上求雅可比、计算 $K$、迭代修正，再把误差注入名义状态，最后使用 Joseph 形式更新协方差。当前 manifold 分支的联合观测雅可比使用中心有限差分；预测雅可比使用 Ceres Jet 自动微分。

`YPDMeasure` 虽然定义在 `motion_model.hpp`，但当前自动瞄准 `ArmorTarget::update()` 的调用路径未使用它做常规更新。不能只凭类型定义声称该观测已经上线。

源码证据：[`armor_target.cpp`](../../awakening/src/tasks/auto_aim/armor_track/armor_target.cpp)、[`error_state_extended_kalman_filter.hpp`](../../awakening/3rdparty/KalmanHyLib/error_state_extended_kalman_filter.hpp)、[`estimation.md`](../../awakening/docs/algorithm/estimation.md)。

### 4.5 当前移植与 awakening 的剩余差异

| 问题 | 当前 `sp_vision25` | `awakening` 参考 |
| --- | --- | --- |
| 状态与姿态 | 13 维、SO(3) 右扰动、完整目标估计 RPY | 同构 13 维、SO(3) 右扰动 |
| 半径正值约束 | `log_r1/log_r2`，正值由指数映射保证 | 同样使用 `log_r` |
| 图像观测 | 单个完整板的 8 维左右 UVL，中心数值雅可比 | 可堆叠完整板、单灯条和深度差观测；联合更新 |
| 预测/雅可比 | 常速度 + yaw 角速度；数值预测和观测雅可比 | 车体系过程噪声旋转、Ceres Jet 预测雅可比，观测可迭代 |
| 前哨高度 | 保留 `outpost_armor_z_offsets` 固定三板偏置 | 利用状态槽位学习 dz |
| NIS | 8 维门控，默认 $\chi^2_8$ 99% | 自动瞄准常用几何关联门限，未见等价 NIS 暴露 |

### 4.6 推荐的估计迁移顺序

**阶段 E0，已落地：13 维 SO(3) 与单板 UVL。**

- 先验创新 NIS 门控，而不是更新后残差打分。
- LDLT 替代显式矩阵求逆。
- Joseph 协方差更新与对称化。
- `log_r1/log_r2`、完整刚体重投影、8 维 UVL、数值雅可比与 8 维 NIS。
- Web 显示 NIS、门限、门控结果、累计计数和拒绝率。

**阶段 E1，建议：扩展同帧观测而不是盲目加维。**

1. 对多个完整板、未被占用单灯条实现同帧堆叠和 block-diagonal $R$。
2. 实现 `DiffMeasure` 或等价的斜视辅助观测，并先以 shadow 指标验证。
3. 用离线数据标定 UVL 方差，验证远距/斜视残差分布和 NIS 覆盖率。
4. 实现同帧联合更新，保留全局或分组 NIS。
5. 用观测信息矩阵 $\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}$ 的特征值/条件数观察哪些状态可观。

**阶段 E2，建议：补齐前哨状态复用和可观性分析。**

当前前哨仍使用 SP 的固定 `armor_z_offsets`。若要向 awakening 靠齐，应将状态槽位 9/10 复用于物理板 dz，同时验证编号重映射、观测可观性和丢失重捕获稳定性。

**阶段 E3，若需要预测快速 RPY，再扩展角速度状态。**

只有出现以下证据才值得从 13 维扩展到至少 15 维（增加 pitch/roll 角速度）：

- 坡地或底盘姿态变化场景中，11 维模型存在稳定、可重复的重投影结构残差。
- 灯条/多板观测对 roll/pitch 的局部信息矩阵不退化。
- 手眼标定、IMU 时间对齐和坐标系约定已经通过独立检查。
- 离线 A-B 中当前 13 维模型只能跟随、无法及时预测快速 RPY，且扩维降低命中时刻位置误差。

不要直接复制 `ErrorStateEKF` 后改数组长度。SO(3) 迁移要求预测、`boxplus/boxminus`、过程噪声所在坐标系、观测残差和协方差全部使用同一个误差定义。

## 5. TinyMPC 与五次多项式 `LimitTrajectory`

### 5.1 `sp_vision25` 当前 TinyMPC 的数学形式

每个轴独立使用二阶状态：

$$
\mathbf{s}_k=[q_k,\dot q_k]^T,
$$

代码采用离散模型：

$$
\mathbf{s}_{k+1}=
\underbrace{\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}}_{A}
\mathbf{s}_k+
\underbrace{\begin{bmatrix}0\\\Delta t\end{bmatrix}}_{B}u_k,
$$

其中 $u_k$ 被解释为角加速度，并约束：

$$
-a_{max}\le u_k\le a_{max}.
$$

有限时域目标可写成：

$$
J=\sum_{k=0}^{N-1}
(\mathbf{s}_k-\mathbf{s}_{k}^{ref})^TQ
(\mathbf{s}_k-\mathbf{s}_{k}^{ref})
+\sum_{k=0}^{N-2}(u_k-u_k^{ref})^TR(u_k-u_k^{ref}).
$$

当前 `DT=0.01 s`、`HORIZON=100`，yaw/pitch 分别求解，最大迭代 10 次。规划器输出第 50 个点的位置、速度和加速度。

这里有两个建模事实：

1. $B=[0,\Delta t]^T$ 忽略同一采样内的 $\frac12a\Delta t^2$ 位置增量，是一阶离散近似；若以后改变 `DT`，要重新评估，而不能只改采样周期。
2. 求解器的初始状态取参考轨迹最前端状态，而不是实时测得的云台状态。因此这层 TinyMPC 更接近“受约束轨迹整形器”，真正的闭环跟踪仍依赖下位机。

参考轨迹在进入 TinyMPC 前已包含：控制延迟预测、最多 6 次命中飞行时间固定点迭代、选板/锁板、弹道 pitch 和未来 100 点采样。当前工作树还记录求解状态和迭代次数；任一轴 `tiny_solve()` 非 0 时禁止开火。

`max_yaw_acc/max_pitch_acc` 的 YAML/Web 单位是 `deg/s^2`，传入求解器前已除以 `57.3` 转成 `rad/s^2`。

源码证据：[`planner.cpp`](../tasks/auto_aim/planner/planner.cpp)、[`planner.hpp`](../tasks/auto_aim/planner/planner.hpp)、[`tiny_api.cpp`](../tasks/auto_aim/planner/tinympc/tiny_api.cpp)、[`admm.cpp`](../tasks/auto_aim/planner/tinympc/admm.cpp)。

### 5.2 `LimitTrajectory` 的五次多项式推导

一段持续时间 $T$ 的位置轨迹写成：

$$
p(t)=c_0+c_1t+c_2t^2+c_3t^3+c_4t^4+c_5t^5.
$$

给定两端的位置、速度、加速度

$$
[p_0,v_0,a_0],\qquad[p_1,v_1,a_1],
$$

由 $t=0$ 可直接得到：

$$
c_0=p_0,\qquad c_1=v_0,\qquad c_2=\frac12a_0.
$$

定义终点相对常加速度外推的误差：

$$
\Delta p=p_1-(p_0+v_0T+\tfrac12a_0T^2),
$$

$$
\Delta v=v_1-(v_0+a_0T),\qquad \Delta a=a_1-a_0.
$$

解三元线性方程可得：

$$
c_3=\frac{10\Delta p-4\Delta vT+\frac12\Delta aT^2}{T^3},
$$

$$
c_4=\frac{-15\Delta p+7\Delta vT-\Delta aT^2}{T^4},
$$

$$
c_5=\frac{6\Delta p-3\Delta vT+\frac12\Delta aT^2}{T^5}.
$$

因此每段精确满足六个边界约束。相邻段若共享同一个节点 $[p,v,a]$，则整体至少是 $C^2$ 连续。

对普通连续节点，`awakening` 先计算左右段平均速度：

$$
\bar v_L=\frac{p_i-p_{i-1}}{T_L},\qquad
\bar v_R=\frac{p_{i+1}-p_i}{T_R},
$$

再估计节点状态：

$$
v_i=\frac{T_R\bar v_L+T_L\bar v_R}{T_L+T_R},
$$

$$
a_i=\frac{2(\bar v_R-\bar v_L)}{T_L+T_R}.
$$

切板时 `aim_id` 改变。`LimitTrajectory` 找到离当前时间最近的切板区间，并向左右扩大过渡时间，直到解析最大加速度满足约束或到达可扩展边界。

五次位置多项式的加速度和 jerk 为：

$$
a(t)=2c_2+6c_3t+12c_4t^2+20c_5t^3,
$$

$$
j(t)=6c_3+24c_4t+60c_5t^2.
$$

$|a(t)|$ 的最大值只可能出现在 $t=0$、$t=T$ 或区间内满足

$$
60c_5t^2+24c_4t+6c_3=0
$$

的实根上，所以不需要用固定步长采样猜最大加速度。

源码证据：[`dta_utils.hpp`](../../awakening/src/tasks/base/dta_utils.hpp)、[`traj_opt.md`](../../awakening/docs/algorithm/traj_opt.md)、[`very_aimer.cpp`](../../awakening/src/tasks/auto_aim/armor_control/very_aimer.cpp)。

### 5.3 适用性对比

| 维度 | TinyMPC | `LimitTrajectory` |
| --- | --- | --- |
| 求解方式 | 迭代二次优化/ADMM | 边界条件闭式解 + 局部区间搜索 |
| 主要约束 | 输入加速度上下界；可继续扩展状态/线性约束 | 解析检查最大加速度，未直接优化状态误差代价 |
| 计算耗时 | 与 horizon、迭代次数和收敛有关 | 通常更稳定、更轻，但仍需扫描切换区间和构段 |
| 求解失败 | 可能不收敛，需要状态和回退策略 | 无通用优化器不收敛，但可能扩展到边界仍无法满足限制 |
| 切板处理 | 通过整条参考和约束平滑 | 明确识别 `aim_id` 跳变，只扩展切换附近时间 |
| 参考跟随 | $Q/R$ 可调，能表达“跟踪 vs 控制代价” | 控制点固定，过渡段可能偏离原始射击轨迹 |
| 参数可解释性 | `Q/R/a_max/horizon` 相互耦合 | `a_max`、采样和区间范围更直观 |
| 当前 SP 集成成本 | 已上线并接入开火判断/Web | 尚未实现 |

### 5.4 推荐的轨迹迁移方式

**不要先删除 TinyMPC。**建议按以下顺序实施：

1. 抽象统一的 `AxisTrajectory::state_at(t) -> {p,v,a,on_target}` 接口。
2. 在 `sp_vision25` 内独立实现最小 `QuinticSegment`，先写边界、连续性和解析最大加速度单测。
3. 用现有 `Planner::get_trajectory()` 的相同离散目标点同时喂给 TinyMPC 和五次多项式，shadow 记录但不下发。
4. 离线比较参考误差、峰值加速度、jerk、计算时间、切板过渡长度和开火可用率。
5. 若解析轨迹稳定，可先作为 TinyMPC 求解失败回退；再做可配置 A-B，而不是一次性切换默认。
6. 若两者各有优势，可采用“切板区间用五次多项式整形，正常区间继续 TinyMPC”的混合方案，但必须避免对同一加速度约束重复收紧。

## 6. 并发推理与数据新鲜度

### 6.1 `sp_vision25` 当前已落地范围

`standard_mpc`：

- 图像读取、YOLO 推理和 Tracker 仍在主线程顺序执行。
- 规划线程从容量 1、满时覆盖旧值的 `target_queue` 读取最新目标。
- 当前改为 `pop_for(20ms)` 原子取出，避免先 `empty()` 再 `front()` 的竞态式接口组合。

`mt_standard`：

- 相机线程创建 `ov::InferRequest`、`start_async()`，主线程按 FIFO `pop()` 后等待。
- `inference_max_inflight` 默认 3，队列满时在创建/启动新请求前拒绝新帧。
- 记录 `submitted/dropped/pending`，每隔约 100 次丢帧输出一次过载日志。
- 切换模式时清空未消费请求，idle/buff 分支加入短 sleep，避免空转占满 CPU。
- `ThreadSafeQueue::try_push/size/pop_for` 已有针对容量和覆盖语义的单测。

边界必须写清：

1. 这些异步推理改造只被 `mt_standard` 使用，`standard_mpc` 不会自动变成并发检测。
2. FIFO 保证提交顺序，但主线程会等待最老请求，存在 head-of-line blocking。
3. 每帧仍创建新的 `InferRequest`，没有类似 awakening 的请求资源池。
4. `pending()` 是队列深度，不是严格区分 waiting/running/completed 的执行状态。
5. `inference_max_inflight` 当前只在 `MultiThreadDetector` 构造时读取。虽然它出现在 Web 参数表中，在线修改不会重建队列，也不会立即生效。
6. 仓库中存在 `tools::OrderedQueue` 和通用 `ThreadPool`，但当前 `mt_standard` 这条路径没有使用它们；“类存在”不等于“生产链已接入”。

源码证据：[`mt_detector.cpp`](../tasks/auto_aim/multithread/mt_detector.cpp)、[`mt_standard.cpp`](../src/mt_standard.cpp)、[`standard_mpc.cpp`](../src/standard_mpc.cpp)、[`thread_safe_queue.hpp`](../tools/thread_safe_queue.hpp)、[`thread_pool.hpp`](../tools/thread_pool.hpp)。

### 6.2 `awakening` 的三层并发控制

`awakening` 将并发分为三层：

1. **任务调度层**：`Scheduler` 把 typed source/task 克隆后提交给 TBB `task_group`。
2. **运行时限流层**：检测任务使用 `std::counting_semaphore::try_acquire()`；满载时不阻塞，当前帧产生空检测结果。
3. **后端资源层**：`ResourcePool<T>` 通过原子 CAS 独占不可重入的推理上下文。

并发完成顺序可能与 frame id 不同，因此检测结果进入 `OrderedQueue`。被信号量跳过的帧仍带原 frame id 入队空结果，使后续已完成帧不会永远卡在编号缺口后面。Tracker 对有序批次逐帧更新，并用互斥锁保证单序列状态更新。

OpenVINO 后端预创建多个 `InferRequest`；池耗尽时当前源码会临时创建请求作为兜底，所以它不是严格的硬并发上限。TensorRT 后端则为每个 `Ctx` 分配独立的 execution context、GPU buffer、host output buffer 和 CUDA stream；拿不到 `Ctx` 时返回空输出。创建额外 TensorRT context 前还会检查剩余显存比例。

源码证据：[`scheduler.hpp`](../../awakening/src/utils/scheduler/scheduler.hpp)、[`buffer.hpp`](../../awakening/src/utils/buffer.hpp)、[`standard.cpp`](../../awakening/src/runtime/standard.cpp)、[`net_detector_openvino.cpp`](../../awakening/src/utils/net_detector/openvino/net_detector_openvino.cpp)、[`net_detector_tensorrt.cpp`](../../awakening/src/utils/net_detector/tensorrt/net_detector_tensorrt.cpp)、[`concurrent_inference.md`](../../awakening/docs/algorithm/concurrent_inference.md)。

### 6.3 推荐迁移路线

**阶段 C0，已落地：最小背压。**

- 有界队列、满载前拒绝、丢帧计数、模式切换清理。
- latest-only 规划输入，避免旧目标堆积。

**阶段 C1，建议：先补完整遥测。**每帧记录：

```text
frame_id
t_capture
t_infer_submit
t_infer_done
t_order_release
t_tracker_done
t_plan_done
t_send
drop_reason
```

核心指标不是单次 infer time，而是：

$$
Age_{send}=t_{send}-t_{capture}.
$$

至少统计 p50/p95/p99、吞吐、丢帧率、队列峰值和最长编号阻塞时间。

**阶段 C2，建议：请求池 + 完成回调。**

1. 启动时按配置创建固定数量 `InferRequest`。
2. 提交时非阻塞申请独占请求；失败则为该 frame id 生成 `drop/resource_busy` 占位结果。
3. 完成回调中做轻量封装并入有序队列；较重后处理交给受控线程池。
4. Tracker 只消费连续 frame id 批次。
5. 退出/切模式时先停止接收、等待回调或使用 generation id 丢弃旧代结果，再释放模型和队列。

**阶段 C3，建议：再决定是否迁移 Scheduler。**

只有当自瞄、打符、全向感知和记录任务的手工线程关系已经难以维护时，才值得引入 typed task graph。直接复制 awakening Scheduler 还会带来 C++23/TBB、对象 clone、队列生命周期和停止语义的额外验证成本。

## 7. Web 可视化链路与当前优化

### 7.1 `sp_vision25` 当前链路

调试入口在进程内创建 `WebDebugger`：

```text
algorithm/debug entry
  -> update_state(JSON)       -> GET /api/state
  -> update_plot_sample(JSON) -> GET /data
  -> update_log(JSON)         -> GET /log
  -> update_*_frame(cv::Mat)  -> MJPEG /stream/*.mjpg or JPEG /api/frames/*.jpg
  -> runtime_params           -> GET/POST /api/params, POST /api/params/reset
```

此外还提供 `/api/mode`、`/api/overlay` 和 `/healthz`。前端通过轮询 JSON、MJPEG 和 Canvas 曲线完成展示。参数写入后，`runtime_params` 增加版本号，各模块在下一次主流程中主动刷新；影响模型结构一致性的 Tracker 参数会触发 tracker reset。

源码证据：[`web_debugger.cpp`](../tools/web_debugger.cpp)、[`runtime_params.cpp`](../tools/runtime_params.cpp)、[`网页热调参数链路说明.md`](./网页热调参数链路说明.md)。

### 7.2 `awakening` 可借鉴的 Web 解耦

`awakening` 的 C++ 调试任务：

- 把 JPEG 写入 2 MiB 共享内存 `/awakening_frame`。
- 把时序数据和日志原子替换写到 `/dev/shm/awakening_data.json`、`awakening_log.json`。
- 独立 `web.py`/Flask 进程 mmap 图像并缓存 JSON，提供 `/video`、`/data`、`/log`。

优势是 Web 进程崩溃或重启不必结束核心视觉进程，也能分别限制 Web 资源。代价是多一个 Python/Flask 部署单元、固定共享内存协议和文件轮询一致性问题。它当前没有 `sp_vision25` 这套通用参数 POST/reset 能力。

源码证据：[`web.cpp`](../../awakening/src/tasks/base/web.cpp)、[`web.hpp`](../../awakening/src/tasks/base/web.hpp)、[`web.py`](../../awakening/web.py)。

### 7.3 当前工作树中已经落地的 Web 改造

**已落地：**

- 保留 `Artisans` 品牌文案，布局改为更紧凑的工具型界面，移除装饰性渐变和环境光元素，并增加桌面/窄屏响应式约束。
- 新增“状态估计 / MPC 求解 / 并发推理”诊断区。
- `estimator_to_json()` 已在 `auto_aim_debug_mpc`、`auto_debug`、`sentry_debug` 和 `auto_aim_test_web` 接入，状态面板可显示 NIS、门限、接受/拒绝和累计计数。
- `mpc_to_json()` 已接入上述入口，状态面板可显示 yaw/pitch 求解状态、迭代数和是否收敛。
- 曲线字典新增 NIS 门控、MPC 迭代和 pipeline 指标分组。
- `auto_aim_test_web` 已把 NIS、MPC 迭代和串行 pipeline 耗时写入 plot sample。
- `auto_aim_debug_mpc` 与 `auto_aim_test_web` 已生成 `web_state["pipeline"]`：当前同步链路按 `max_inflight=1` 展示 YOLO、Tracker 和 Planner 分段耗时。

**尚未完成：**

- `auto_debug`、`sentry_debug` 目前没有生成 pipeline 状态；这些入口的并发诊断字段仍可能显示 `--`。
- 已接入的两个入口是同步处理，`pending=0`、`inflight=0`、`max_inflight=1`、`dropped=0` 不能代表 `mt_standard` 的异步队列状态。
- 真正使用 `MultiThreadDetector` 的 `mt_standard` 没有创建 `WebDebugger`，其 `submitted/dropped/pending` 尚未进入 `web_state.pipeline`，吞吐字段也没有生产者。
- 各调试入口没有全部把 estimator/MPC/pipeline 字段送入 `/data` 时序曲线；曲线可用字段随入口不同。
- `inference_max_inflight` 不支持运行中重建，Web 修改只更新 runtime parameter session，不能改变已经构造的 `MultiThreadDetector`。
- 没有 `schema_version`，前端目前通过多个候选 path 兼容字段；长期会增加协议漂移风险。
- Web 服务绑定 `0.0.0.0` 且参数 POST 没有认证/TLS。只能在可信比赛局域网使用，不能直接暴露到公网。

源码证据：[`debug.cpp`](../tools/debug.cpp)、[`index.html`](../assets/web_debugger/index.html)、[`main.js`](../assets/web_debugger/static/js/main.js)、[`chart_logic.js`](../assets/web_debugger/static/js/chart_logic.js)。

### 7.4 Web 下一阶段建议

1. 定义版本化状态协议：`schema_version`、`source_entry`、`frame_id`、`sample_time`、`publish_time`。
2. 给 `MultiThreadDetector` 增加只读 telemetry snapshot，并由真正使用它的入口写入 `state.pipeline`；不要由前端猜测。
3. 把 plot sample 的公共字段集中到一个 C++ builder，避免四个调试入口各自漏字段。
4. 对构造期参数在 UI 标注 `restart_required=true`；或实现安全的 generation-based 重建后再允许热生效。
5. 若实测 Web 编码/客户端线程影响主链 p99 延迟，再把 JPEG/JSON 发布层迁到共享内存独立进程；不要在没有性能证据时先增加部署复杂度。
6. 给参数写接口增加至少一层只读/可写模式和局域网 token；比赛模式默认只读。

## 8. 当前工作树“已落地 / 未落地”清单

| 项目 | 状态 | 证据与限制 |
| --- | --- | --- |
| 13 维 SO(3) 状态与对数半径 | **已落地** | `target_state`、`so3_exp/so3_log`、完整刚体装甲板位姿；前哨仍使用固定高度偏置 |
| 8 维 UVL 先验 NIS 门控 | **已落地** | 默认 20.090；拒绝量测不更新状态/元数据 |
| LDLT + Joseph + 协方差对称化 | **已落地** | 单测覆盖正常量测、离群量测和协方差对称性 |
| 真正 NEES | **未落地** | 当前 `nees` 是 correction energy，没有 ground truth |
| 图像 UVL 完整板观测 | **已落地** | 左右灯条各 4 维，完整板更新为 8 维 |
| 同帧多板/单灯条联合观测 | **未落地** | 当前为逐板顺序更新，未实现 `update_multi` 和深度差辅助量测 |
| TinyMPC 求解状态/迭代遥测 | **已落地** | 求解失败禁止开火；状态面板已接入 |
| 最大角加速度单位修正 | **已落地** | 配置 deg/s² 转内部 rad/s² |
| 五次多项式 `LimitTrajectory` | **未落地** | 不存在 SP C++ 实现，不能写成可用 |
| `standard_mpc` latest-only 规划输入 | **已落地** | 容量 1 覆盖队列 + `pop_for` |
| `mt_standard` 有界异步请求 | **已落地** | 默认 3、过载丢新帧、计数和清理；仅该入口 |
| InferRequest 资源池/完成回调/按 id 重排 | **未落地** | 当前仍每帧创建 request、FIFO 等待 |
| Web 估计与 MPC 状态诊断 | **已落地** | 四个调试入口接状态 JSON |
| Web 同步 pipeline 分段耗时 | **部分落地** | `auto_aim_debug_mpc`、`auto_aim_test_web` 已接；是 `max_inflight=1` 的串行链路 |
| Web 异步 pipeline 队列遥测 | **未落地** | `mt_standard` 的 submitted/dropped/pending 尚未接 Web，吞吐也未计算 |
| 参数热调持久记录 | **已有并保留** | `runtime_params` 维护 override、version、JSONL/snapshot |
| 本轮实车性能/命中率验收 | **未知** | 仓库内没有本轮同条件 A-B 结果 |

## 9. 参数学习与实验方法

### 9.1 先固定实验基础

任何滤波或轨迹调参之前，先固定以下变量：

1. 相机内参、畸变、手眼外参和坐标轴方向。
2. 图像时间戳与 IMU/云台姿态对齐误差。
3. 弹速来源、异常回退值和弹道模型。
4. 同一段原始图像/姿态/串口记录，禁止比较时更换数据。
5. 目标类别、装甲板尺寸、编号映射和前哨物理板高度。

建议建立至少四类回放集：静止/慢速平移、普通旋转、前哨固定旋转、遮挡/误检/重新捕获。训练式调参与最终验收数据必须分开。

### 9.2 用创新统计调 $R$，用动态失配调 $Q$

先在静止或低动态、关联可靠的数据上记录创新 $\nu$。量测噪声初值可用稳健尺度估计，而不是直接被离群值污染的普通方差。例如单维残差：

$$
\hat\sigma\approx1.4826\operatorname{median}(|\nu-\operatorname{median}(\nu)|).
$$

然后令对应 $R_{ii}\approx\hat\sigma_i^2$，再用动态数据调 $Q$。评价 NIS 时：

$$
\overline\epsilon=\frac1N\sum_{k=1}^N\epsilon_k\approx m,
$$

其中当前完整板 UVL 观测 $m=8$；并检查经验覆盖率：

$$
\hat C_\alpha=\frac1N\sum_k
\mathbf{1}[\epsilon_k\le\chi^2_{m,\alpha}]\approx\alpha.
$$

解释：

- NIS 长期过高：模型/Q 太自信、R 太小、时间同步差、关联错误或观测有偏。
- NIS 长期过低：P/R 过大，滤波器过度保守；不代表一定“更稳”。
- NIS 只在切板尖峰：优先检查编号关联和角度 wrap，不要先整体放大 R。
- 拒绝率突然上升：同时看距离、斜视角、检测置信度和帧龄，不能只放宽 gate。

参数方向可作初步参考：

| 参数 | 增大后的典型效果 | 风险 |
| --- | --- | --- |
| `tracker_acceleration_variance` | 更快跟随平移机动 | 中心速度/位置更抖，远期预测方差增大 |
| `tracker_yaw_acceleration_variance` | 更快跟随角速度变化 | 角速度更敏感、切板相位更抖 |
| `tracker_geometry_random_walk` | 半径/高度更容易重新学习 | 几何状态被单帧误差拖动 |
| `tracker_uvl_*_variance` | 更少相信对应 UVL 量测、输出更平滑 | 响应变慢，也可能掩盖系统性标定误差 |
| `tracker_nis_gate` | 接受更多量测 | 错配/离群值更容易进入滤波器 |

### 9.3 轨迹 A-B 指标

对同一离散目标轨迹同时计算 TinyMPC 与 Quintic shadow，至少记录：

$$
e_q(t)=\operatorname{wrap}(q_{control}(t)-q_{target}(t)),
$$

$$
a_{peak}=\max_t|a(t)|,
$$

$$
J_{jerk}=\int |j(t)|^2dt.
$$

还应记录：规划耗时 p50/p95/p99、TinyMPC 迭代数/失败率、切板过渡时长、发射延迟区间内全部采样点是否保持在可击打窗口，以及实车云台实际角与命令角的误差。

只比较“曲线看起来更平滑”是不够的。轨迹必须同时满足：

- 控制可执行。
- 命中点误差可接受。
- 开火窗口没有因过度平滑而显著缩短。
- 计算时间对控制周期稳定。

### 9.4 并发负载扫描

固定模型、分辨率和相机输入，依次测试 `max_inflight=1,2,3,4,...`。每档至少运行 2 分钟，记录：

- 推理吞吐 FPS。
- `Age_send` p50/p95/p99。
- dropped/submitted。
- 队列峰值和 frame-id 等待时间。
- Tracker 更新间隔和临时丢失次数。
- CPU、GPU/NPU 占用和内存。

选择满足控制延迟约束的最小并发度，而不是吞吐最大的并发度。若并发增加后 FPS 上升但 `Age_send` p99 变差，应降低并发或改为更激进的 latest-only/drop 策略。

### 9.5 Web 作为实验仪器

Web 面板应服务于假设验证：

1. 先写下假设，例如“斜视时距离 R 太小导致 NIS 尖峰”。
2. 固定数据集，只改一个参数族。
3. 同时导出 override、原始配置 hash、程序 commit、数据集 id 和指标摘要。
4. 用 NIS 覆盖、预测误差、求解失败率、数据年龄等指标判定，不以单次命中作为结论。
5. 验收后把有效值回填 YAML，并清空网页 override，防止“现场有效但配置不可复现”。

## 10. 分阶段总迁移路线

| 阶段 | 目标 | 交付物 | 进入下一阶段的门槛 |
| --- | --- | --- | --- |
| P0：正确性补强 | 稳住现有 EKF/MPC/队列 | 当前 NIS、Joseph、参数化、求解状态、有界队列、Web 诊断 | 单测通过，离线回放无明显回归 |
| P1：可观测与可测量 | 补 pipeline 全链时间、统一 Web schema、收集创新 | 版本化 telemetry、数据年龄分位数、NIS 报告 | 能用数据定位主要瓶颈 |
| P2：观测升级 | 扩展当前 13 维 UVL 为同帧多观测，并做预测 ROI | 联合更新、单灯条/深度差、ROI A-B | 远距/斜视误差和召回有稳定收益 |
| P3：轨迹 A-B | 引入本地 Quintic shadow/fallback | 单测、回放对比、可配置切换 | 控制误差/开火窗口/耗时达到验收门槛 |
| P4：并发后端 | 请求池、完成回调、有序批次 | OpenVINO resource pool、drop reason、退出测试 | p99 数据年龄下降且 Tracker 不乱序 |
| P5：状态预测扩展 | 证据充分时加入 roll/pitch 角速度或自动微分 | 15 维以上模块、数值/自动微分检查、A-B 报告 | 快速 RPY 场景真实收益，普通场景不回归 |
| P6：运行时解耦 | 必要时拆 Web 或引入 task graph | 共享内存协议或自有 scheduler 设计 | 有性能/维护证据，而非为了架构一致 |

## 11. 为什么不能直接复制

即使两个顶层仓库都是 MIT License，工程上仍不能把 awakening 的模块直接粘贴后宣称完成：

1. **状态槽位仍有差异**：当前 SP 已采用 awakening 风格的 13 维索引，但前哨固定高度偏置、平衡步兵二板模型和基地语义没有直接复用 awakening 的全部槽位。
2. **观测空间不同**：SP 使用 world YPD/PnP，awakening 更新主要使用 camera image UVL；$R$ 的单位也不同。
3. **坐标系不同**：awakening 显式维护 odom、gimbal_odom、camera、camera_cv、shoot 的时变 TF；SP 当前接口不能直接满足相同前提。
4. **误差定义不同**：SO(3) 右扰动要求 $Q/P/H$ 都在同一切空间，普通加法 EKF 不能只替换 `x_add()`。
5. **依赖和标准不同**：C++17/OpenVINO 与 C++23/Ceres/TBB/TensorRT 的构建、对象生命周期和硬件路径不同。
6. **控制协议和单位不同**：SP 内部以 rad 为主、部分 YAML 为 deg；awakening 的 `GimbalCmd` 对外字段以 degree 转换，符号和坐标约定需逐项核对。
7. **目标枚举/编号不同**：前哨、基地、平衡步兵的半径、高度槽位和物理编号映射不能复用常量。
8. **配置快照可能不一致**：当前 awakening 的部分 YAML 仍使用 `match_gate_armor/r_uv_at_1m` 等旧名字，而 `ArmorTrackerCfg::load()` 读取 `armor_match_gate`、权重和 `r_sigma_*`；必须以实际运行配置和源码版本联合验证。
9. **可选路径不等于默认路径**：`LimitTrajectory`、TinyMPC、双轴 MPC 同时存在，具体走哪条由配置决定；部分 YAML 是否通过合并配置补全 `type`，仅凭当前文件无法确认。
10. **第三方来源要单独审计**：若 vendoring `3rdparty/KalmanHyLib` 或 TinyMPC 相关文件，应保留原作者和许可证声明，并核对具体文件的许可证，而不只看顶层仓库 LICENSE。

正确方式是迁移思想和接口契约，按 SP 自身坐标系、编译标准、硬件和测试重新实现；若复用原代码，则保留许可证/归属并做独立适配层与测试。

## 12. 局限与未知

- 本文完成的是源码静态分析，没有在当前机器上复现 awakening README 的 `250 Hz / 2-8 ms` 性能数据，因此不把它作为已验证对比结论。
- 当前没有本轮改造前后的同数据集回放、实车命中率、控制延迟分位数或长期稳定性结果。
- 未知下位机对 yaw/pitch 位置、速度、加速度前馈的精确闭环实现；这会影响 TinyMPC 与 Quintic 的最终优劣。
- 当前实现没有实车 A-B、命中率和控制延迟分位数，不能仅凭合成投影或离线程序不崩溃宣称已经更精准。
- awakening 的算法文档与源码整体一致，但配置键和可选路径存在版本演进痕迹；关键结论应以本文列出的实际实现文件为准。
- 当前 Web pipeline 面板在两个同步入口已有分段耗时，但不是异步推理链路已经完成的证据。

## 13. 主要源码索引

### `sp_vision25`

- 整车状态与观测：[`tasks/auto_aim/target.cpp`](../tasks/auto_aim/target.cpp)
- 跟踪状态机与参数刷新：[`tasks/auto_aim/tracker.cpp`](../tasks/auto_aim/tracker.cpp)
- EKF/NIS：[`tools/extended_kalman_filter.cpp`](../tools/extended_kalman_filter.cpp)
- TinyMPC Planner：[`tasks/auto_aim/planner/planner.cpp`](../tasks/auto_aim/planner/planner.cpp)
- MPC 求解器：[`tasks/auto_aim/planner/tinympc/`](../tasks/auto_aim/planner/tinympc)
- 异步 OpenVINO：[`tasks/auto_aim/multithread/mt_detector.cpp`](../tasks/auto_aim/multithread/mt_detector.cpp)
- 两个关键入口：[`src/standard_mpc.cpp`](../src/standard_mpc.cpp)、[`src/mt_standard.cpp`](../src/mt_standard.cpp)
- 队列基础设施：[`tools/thread_safe_queue.hpp`](../tools/thread_safe_queue.hpp)、[`tools/thread_pool.hpp`](../tools/thread_pool.hpp)
- Web 后端与参数：[`tools/web_debugger.cpp`](../tools/web_debugger.cpp)、[`tools/runtime_params.cpp`](../tools/runtime_params.cpp)
- Web 前端：[`assets/web_debugger/`](../assets/web_debugger)
- 本轮聚焦单测：[`tests/estimator_pipeline_test.cpp`](../tests/estimator_pipeline_test.cpp)、[`tests/target_13d_test.cpp`](../tests/target_13d_test.cpp)

### `awakening`

- 13 维运动/观测模型：[`src/tasks/auto_aim/armor_track/motion_model.hpp`](../../awakening/src/tasks/auto_aim/armor_track/motion_model.hpp)
- 整车目标与联合更新：[`src/tasks/auto_aim/armor_track/armor_target.cpp`](../../awakening/src/tasks/auto_aim/armor_track/armor_target.cpp)
- ESEKF：[`3rdparty/KalmanHyLib/error_state_extended_kalman_filter.hpp`](../../awakening/3rdparty/KalmanHyLib/error_state_extended_kalman_filter.hpp)
- 五次多项式与可选 MPC：[`src/tasks/base/dta_utils.hpp`](../../awakening/src/tasks/base/dta_utils.hpp)
- 自瞄控制集成：[`src/tasks/auto_aim/armor_control/very_aimer.cpp`](../../awakening/src/tasks/auto_aim/armor_control/very_aimer.cpp)
- Scheduler 与并发容器：[`src/utils/scheduler/scheduler.hpp`](../../awakening/src/utils/scheduler/scheduler.hpp)、[`src/utils/buffer.hpp`](../../awakening/src/utils/buffer.hpp)
- 标准运行时：[`src/runtime/standard.cpp`](../../awakening/src/runtime/standard.cpp)
- 推理后端：[`src/utils/net_detector/`](../../awakening/src/utils/net_detector)
- Web C++/Python：[`src/tasks/base/web.cpp`](../../awakening/src/tasks/base/web.cpp)、[`web.py`](../../awakening/web.py)
- awakening 原始算法文档：[`estimation.md`](../../awakening/docs/algorithm/estimation.md)、[`traj_opt.md`](../../awakening/docs/algorithm/traj_opt.md)、[`concurrent_inference.md`](../../awakening/docs/algorithm/concurrent_inference.md)

本文对核心状态、公式和当前调用路径的置信度为 **高**；对实车收益、实际部署配置和跨硬件性能的置信度为 **低至中**，必须通过第 9 节实验补齐。
