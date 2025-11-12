#include "target.hpp"

#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
  // 初始化成员变量
  : name(armor.name),                         // 装甲板名称
    armor_type(armor.type),                   // 装甲板类型
    jumped(false),                            // 是否发生跳跃（装甲板切换）
    last_id(0),                               // 上一次装甲板ID
    update_count_(0),                         // 更新计数器
    armor_num_(armor_num),                    // 装甲板数量
    t_(t),                                    // 当前时间戳
    is_switch_(false),                        // 是否发生装甲板切换
    is_converged_(false),                     // 是否收敛
    switch_count_(0)                          // 切换次数计数器
{
  auto r = radius;                            // 获取半径参数
  priority = armor.priority;                  // 设置优先级

  // 获取装甲板在世界坐标系中的位置和角度
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 根据装甲板位置和角度计算旋转中心坐标
  // 这里假设装甲板围绕一个中心点旋转，通过当前装甲板位置和角度反推中心点
  auto center_x = xyz[0] + r * std::cos(ypr[0]);  // X坐标
  auto center_y = xyz[1] + r * std::sin(ypr[0]);  // Y坐标
  auto center_z = xyz[2];                         // Z坐标

  // 状态向量定义：x vx y vy z vz a w r l h
  // x, y, z: 旋转中心坐标
  // vx, vy, vz: 旋转中心速度
  // a: 当前装甲板角度（偏航角）
  // w: 角速度
  // r: 旋转半径
  // l: 半径差（长装甲板与短装甲板的半径差）
  // h: 高度差（长装甲板与短装甲板的高度差）
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();   // 使用输入参数初始化协方差矩阵

  // 定义状态向量加法函数，确保角度值在[-π, π]范围内
  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);  // 限制第6个元素（角度）在合理范围内
    return c;
  };

  // 使用计算得到的初始状态和协方差初始化扩展卡尔曼滤波器
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

// 构造函数定义，接受目标初始x坐标、yaw角速度、半径和高度作为参数
Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  // 初始化状态向量x0，包含11个状态变量：
  // x(0): x坐标, 0(1): x方向速度
  // 0(2): y坐标, 0(3): y方向速度
  // 0(4): z坐标, 0(5): z方向速度
  // 0(6): yaw角度, vyaw(7): yaw角速度
  // radius(8): 半径, 0(9): 半径差, h(10): 高度差
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  
  // 初始化协方差矩阵对角线元素为0
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  
  // 将向量转换为对角矩阵形式
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 定义状态向量加法函数，确保角度值在[-π, π]范围内
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    // 先执行普通加法
    Eigen::VectorXd c = a + b;
    // 对第6个元素(yaw角度)进行限制，防止超出范围
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  // 使用初始化的状态向量、协方差矩阵和加法函数创建扩展卡尔曼滤波器
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

/**
 * @brief 根据给定的时间点预测目标状态
 * @param t 输入的时间点，用于计算时间差并更新目标状态
 */
void Target::predict(std::chrono::steady_clock::time_point t)
{
  // 计算当前时间点与上一次预测时间点的时间差
  auto dt = tools::delta_time(t, t_);
  // 使用计算得到的时间差进行预测
  predict(dt);
  // 更新上一次预测时间点为当前时间点
  t_ = t;
}

/**
 * @brief 预测目标状态
 * @param dt 时间间隔
 */
/**
 * @brief 根据时间间隔预测目标状态
 * @param dt 时间间隔
 */
void Target::predict(double dt)
{
  // 状态转移矩阵F，描述状态如何随时间变化
  // 状态向量为 [x, vx, y, vy, z, vz, a, w, r, l, h]
  // clang-format off
  Eigen::MatrixXd F{
    // x = x + vx*dt
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    // vx = vx (速度保持不变，实际会受噪声影响)
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    // y = y + vy*dt
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    // vy = vy
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    // z = z + vz*dt
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    // vz = vz
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    // a = a + w*dt (角度 = 角度 + 角速度*dt)
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    // w = w (角速度保持不变)
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    // r = r (半径保持不变)
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    // l = l (半径差保持不变)
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    // h = h (高度差保持不变)
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model (分段白噪声模型)
  // 用于建模系统过程噪声，参考链接:
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  // 根据目标类型设置不同的过程噪声参数
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差较小，运动相对稳定
    v2 = 0.1;  // 前哨站角加速度方差也较小
  } else {
    v1 = 100;  // 普通目标加速度方差较大，运动更剧烈
    v2 = 400;  // 普通目标角加速度方差较大
  }
  
  // 计算噪声协方差矩阵Q所需系数
  auto a = dt * dt * dt * dt / 4;  // dt^4/4
  auto b = dt * dt * dt / 2;       // dt^3/2
  auto c = dt * dt;                // dt^2
  
  // 预测过程噪声偏差的方差矩阵Q
  // clang-format off
  Eigen::MatrixXd Q{
    // x方向位置和速度噪声
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    // y方向位置和速度噪声
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    // z方向位置和速度噪声
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    // 角度和角速度噪声
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    // 半径噪声
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    // 半径差噪声
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    // 高度差噪声
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 定义状态转移函数f，用于非线性预测
  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    // 先应用线性状态转移矩阵
    Eigen::VectorXd x_prior = F * x;
    // 对角度值进行限制，确保在[-π, π]范围内
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判：限制前哨站最大角速度
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    // 将前哨站角速度限制在±2.51范围内
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  // 调用扩展卡尔曼滤波器的预测步骤
  ekf_.predict(F, Q, f);
}

/**
 * @brief 更新目标状态，根据装甲板信息进行匹配和追踪
 * @param armor 输入的装甲板信息，包含世界坐标系下的角度和位置数据
 */
/**
 * @brief 更新目标状态，通过匹配装甲板来确定目标
 * 
 * @param armor 输入的装甲板信息，包含世界坐标系下的角度和位置数据
 */
/**
 * @brief 更新目标状态，通过匹配装甲板来确定目标
 * 
 * @param armor 输入的装甲板信息，包含世界坐标系下的角度和位置数据
 */
void Target::update(const Armor & armor)
{
  // 装甲板匹配：通过角度误差找到最匹配的装甲板
  int id;  // 最匹配装甲板的ID
  auto min_angle_error = 1e10;  // 初始化最小角度误差为一个很大的值
  // 获取所有装甲板的xyz坐标和角度信息（基于当前估计状态）
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  // 创建装甲板列表，每个元素包含装甲板信息和其原始索引
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  // 遍历所有装甲板
  for (int i = 0; i < armor_num_; i++) {
    // 将装甲板信息和对应索引组成pair存入列表
    xyza_i_list.push_back({xyza_list[i], i});
  }

  // 按照深度（距离相机的距离）对装甲板进行排序，最近的排在前面
  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      // 将xyz坐标转换为ypd坐标（yaw, pitch, distance）
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      // 按距离（depth）排序
      return ypd1[2] < ypd2[2];
    });

  // 只取前3个距离最近的装甲板进行匹配，提高效率
  for (int i = 0; i < 3; i++) {
    // 获取当前装甲板的坐标和角度信息
    const auto & xyza = xyza_i_list[i].first;
    // 将xyz坐标转换为ypd坐标
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    
    // 计算角度误差，包括两个部分：
    // 1. 当前观测装甲板角度与预测装甲板角度的差异
    // 2. 当前观测装甲板yaw与预测装甲板yaw的差异
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    // 如果当前角度误差小于已记录的最小角度误差
    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      // 更新最匹配装甲板的ID
      id = xyza_i_list[i].second;
      // 更新最小角度误差
      min_angle_error = angle_error;
    }
  }

  // 如果ID不为0，表示目标发生了跳变（从一个装甲板跳到另一个）
  if (id != 0) jumped = true;

  // 检查是否切换了目标（当前匹配的装甲板与上次不同）
  if (id != last_id) {
    is_switch_ = true;  // 标记为切换目标
  } else {
    is_switch_ = false;  // 未切换目标
  }

  // 如果发生了切换，增加切换计数
  if (is_switch_) switch_count_++;

  // 更新最后匹配的ID和更新计数
  last_id = id;
  update_count_++;

  // 使用匹配到的装甲板更新EKF状态
  update_ypda(armor, id);
}

/**
 * @brief 使用观测到的装甲板信息更新目标状态（EKF更新步骤）
 * 
 * @param armor 观测到的装甲板信息
 * @param id 匹配到的装甲板ID
 */
void Target::update_ypda(const Armor & armor, int id)
{
  // 计算观测雅可比矩阵H，描述状态变量对观测值的偏导数
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  
  // 计算目标中心的偏航角（yaw）
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  // 计算装甲板角度与中心角度的差值，并限制在[-π, π]范围内
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  
  // 构造观测噪声协方差矩阵R的对角线元素
  // 噪声值根据角度差和距离动态调整
  Eigen::VectorXd R_dig{
    {
      4e-3,  // yaw观测噪声
      4e-3,  // pitch观测噪声
      log(std::abs(delta_angle) + 1) + 1,  // 距离观测噪声，与角度差相关
      log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2  // 角度观测噪声，与距离相关
    }
  };

  // 将观测噪声向量转换为对角矩阵
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性观测函数h: x -> z，将状态向量映射到观测空间
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    // 根据状态向量计算指定ID装甲板的3D坐标
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    // 将3D坐标转换为yaw-pitch-distance坐标
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    // 计算装甲板的角度（考虑装甲板ID）
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    // 返回观测向量[yaw, pitch, distance, angle]
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 定义观测向量的减法函数，确保角度差值在[-π, π]范围内
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    // 先执行普通减法
    Eigen::VectorXd c = a - b;
    // 对yaw、pitch和angle分量进行角度限制
    c[0] = tools::limit_rad(c[0]);  // yaw角度差
    c[1] = tools::limit_rad(c[1]);  // pitch角度差
    c[3] = tools::limit_rad(c[3]);  // 装甲板角度差
    return c;
  };

  // 获取观测到的装甲板的yaw-pitch-distance坐标
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  // 获取观测到的装甲板的yaw-pitch-roll坐标
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  // 构造观测向量[z_yaw, z_pitch, z_distance, z_angle]
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  // 调用扩展卡尔曼滤波器的更新步骤
  ekf_.update(z, H, R, h, z_subtract);
}

  // 返回扩展卡尔曼滤波器的当前状态向量
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

/**
 * @brief 获取扩展卡尔曼滤波器对象的常量引用
 * 
 * @return const tools::ExtendedKalmanFilter& 扩展卡尔曼滤波器对象的常量引用
 * 
 * 该函数提供对内部扩展卡尔曼滤波器对象的只读访问，
 * 允许外部代码获取滤波器的完整状态，包括：
 * - 状态向量 x
 * - 协方差矩阵 P
 * - 以及其他滤波器内部参数和方法
 */
  // 返回扩展卡尔曼滤波器对象的常量引用
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }


/**
 * @brief 根据当前估计状态计算所有装甲板的坐标和角度
 * 
 * @return std::vector<Eigen::Vector4d> 包含所有装甲板的[x, y, z, angle]信息的向量
 * 
 * 该函数基于当前EKF估计的状态，计算所有装甲板在世界坐标系中的位置和角度，
 * 用于装甲板匹配和可视化等目的。
 */
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  // 创建存储装甲板信息的向量
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  // 遍历所有装甲板
  for (int i = 0; i < armor_num_; i++) {
    // 计算第i个装甲板的角度，并限制在[-π, π]范围内
    // 装甲板角度 = 中心角度 + 装甲板ID * 2π / 装甲板总数
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    
    // 根据当前状态向量和装甲板ID计算装甲板的3D坐标
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    
    // 将装甲板的坐标和角度信息存入向量
    // 格式为[x, y, z, angle]
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  
  // 返回所有装甲板的坐标和角度信息
  return _armor_xyza_list;
}

/**
 * @brief 检查目标状态估计是否发散
 * 
 * 通过检查旋转半径和半径差的合理性来判断滤波器是否发散。
 * 如果估计值超出合理范围，则认为滤波器发散。
 * 
 * @return bool 如果发散返回true，否则返回false
 */
bool Target::diverged() const
{
  // 检查主旋转半径是否在合理范围内 (0.05m - 0.5m)
  // 这个范围基于实际机器人装甲板旋转半径的物理约束
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  
  // 检查考虑半径差后的总半径是否在合理范围内
  // 对于4装甲板目标，长装甲板和短装甲板的半径差也要考虑在内
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  // 如果半径和半径差都在合理范围内，则未发散
  if (r_ok && l_ok) return false;

  // 如果发散，记录调试日志，显示当前的半径和半径差值
  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  
  // 返回true表示滤波器已发散
  return true;
}

/**
 * @brief 检查目标状态估计是否已经收敛
 * 
 * 通过检查更新次数和是否发散来判断滤波器是否收敛。
 * 不同类型的目标（普通目标 vs 前哨站）有不同的收敛判断标准。
 * 
 * @return bool 如果已收敛返回true，否则返回false
 */
bool Target::convergened()
{
  // 对于非前哨站目标：更新次数超过3次且未发散时认为已收敛
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    // 设置收敛标志为true
    is_converged_ = true;
  }

  // 前哨站特殊判断：更新次数超过10次且未发散时认为已收敛
  // 前哨站需要更多更新次数才能认为收敛，因为其运动模式可能更复杂
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    // 设置收敛标志为true
    is_converged_ = true;
  }

  // 返回当前的收敛状态
  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
/**
 * @brief 根据状态向量和装甲板ID计算装甲板在世界坐标系中的3D坐标
 * 
 * 该函数实现了从状态向量到装甲板坐标的非线性观测模型的正向计算。
 * 考虑了4装甲板目标中长装甲板和短装甲板的几何差异。
 * 
 * @param x 状态向量 [x, vx, y, vy, z, vz, a, w, r, l, h]
 * @param id 装甲板ID (0-3)
 * @return Eigen::Vector3d 装甲板的3D坐标 [x, y, z]
 */
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  // 计算装甲板的角度 = 中心角度 + 相对角度偏移，并限制在[-π, π]范围内
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  
  // 判断是否需要考虑半径差和高度差
  // 对于4装甲板目标，ID为1和3的装甲板是长装甲板，需要考虑半径差和高度差
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  // 计算装甲板的旋转半径
  // 对于长装甲板，半径 = 基础半径 + 半径差
  // 对于短装甲板，半径 = 基础半径
  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  
  // 计算装甲板的X坐标：中心X坐标 - 半径 * cos(角度)
  // 注意这里使用减法是因为装甲板位于中心点的径向方向上
  auto armor_x = x[0] - r * std::cos(angle);
  
  // 计算装甲板的Y坐标：中心Y坐标 - 半径 * sin(角度)
  auto armor_y = x[2] - r * std::sin(angle);
  
  // 计算装甲板的Z坐标
  // 对于长装甲板，Z坐标 = 中心Z坐标 + 高度差
  // 对于短装甲板，Z坐标 = 中心Z坐标
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  // 返回装甲板的3D坐标
  return {armor_x, armor_y, armor_z};
}

/**
 * @brief 计算观测函数的雅可比矩阵（观测矩阵H）
 * 
 * 该函数计算观测函数h(x)在给定点x处的雅可比矩阵，
 * 描述状态变量对观测值的偏导数关系。
 * 
 * @param x 状态向量 [x, vx, y, vy, z, vz, a, w, r, l, h]
 * @param id 装甲板ID (0-3)
 * @return Eigen::MatrixXd 观测雅可比矩阵 H (4x11)
 */
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  // 计算装甲板的角度，并限制在[-π, π]范围内
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  
  // 判断是否为长装甲板（需要考虑半径差和高度差）
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  // 计算装甲板的旋转半径
  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  
  // 计算X坐标对角度a的偏导数: ∂x/∂a = r * sin(angle)
  auto dx_da = r * std::sin(angle);
  
  // 计算Y坐标对角度a的偏导数: ∂y/∂a = -r * cos(angle)
  auto dy_da = -r * std::cos(angle);

  // 计算X坐标对半径r的偏导数: ∂x/∂r = -cos(angle)
  auto dx_dr = -std::cos(angle);
  
  // 计算Y坐标对半径r的偏导数: ∂y/∂r = -sin(angle)
  auto dy_dr = -std::sin(angle);
  
  // 计算X坐标对半径差l的偏导数
  // 对于长装甲板: ∂x/∂l = -cos(angle)
  // 对于短装甲板: ∂x/∂l = 0
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  
  // 计算Y坐标对半径差l的偏导数
  // 对于长装甲板: ∂y/∂l = -sin(angle)
  // 对于短装甲板: ∂y/∂l = 0
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  // 计算Z坐标对高度差h的偏导数
  // 对于长装甲板: ∂z/∂h = 1
  // 对于短装甲板: ∂z/∂h = 0
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // 构造装甲板坐标对状态变量的雅可比矩阵 H_armor_xyza (4x11)
  // 行表示: [x, y, z, angle]观测值
  // 列表示: [x, vx, y, vy, z, vz, a, w, r, l, h]状态变量
  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    // x观测值对各状态变量的偏导数
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    // y观测值对各状态变量的偏导数
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    // z观测值对各状态变量的偏导数
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    // angle观测值对各状态变量的偏导数（只有对角度a的偏导数为1）
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  // 计算装甲板的3D坐标（用于后续计算）
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  
  // 计算从3D坐标到yaw-pitch-distance坐标的雅可比矩阵
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  
  // 构造从yaw-pitch-distance-angle到观测值的雅可比矩阵 H_armor_ypda (4x4)
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    // yaw对xyz的偏导数，angle对angle的偏导数为1
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    // pitch对xyz的偏导数，其余为0
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    // distance对xyz的偏导数，其余为0
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    // angle对angle的偏导数为1，其余为0
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  // 返回完整的观测雅可比矩阵 = H_armor_ypda * H_armor_xyza
  return H_armor_ypda * H_armor_xyza;
}

/**
 * @brief 检查目标是否已初始化
 * 
 * @return bool 如果已初始化返回true，否则返回false
 */
bool Target::checkinit() { 
  // 返回初始化状态标志
  return isinit; 
}

}  // namespace auto_aim
