# Driving Control vs Parking Control 算法深度对比分析

> 基于 `/home/sonzhen/12-mmt_whole_code_072_12f/pangu_ad_sw/maf-control/` 和 `maf-planning/` 代码的完整分析

---

## 目录

- [一、横向控制（Lateral Control）算法对比](#一横向控制lateral-control算法对比)
  - [1.1 Driving Control 横向算法](#11-driving-control-横向算法)
  - [1.2 Parking Control 横向算法](#12-parking-control-横向算法)
  - [1.3 横向控制对比总结](#13-横向控制对比总结)
- [二、纵向控制（Longitudinal Control）算法对比](#二纵向控制longitudinal-control算法对比)
  - [2.1 Driving Control 纵向算法](#21-driving-control-纵向算法)
  - [2.2 Parking Control 纵向算法](#22-parking-control-纵向算法)
  - [2.3 纵向控制对比总结](#23-纵向控制对比总结)
- [三、规划轨迹精细度对比](#三规划轨迹精细度对比)
  - [3.1 Planning 输入数据结构](#31-planning-输入数据结构)
  - [3.2 Driving 轨迹 vs Parking 轨迹](#32-driving-轨迹-vs-parking-轨迹)
- [四、Driving Planning 中的 MPC 与优化算法](#四driving-planning-中的-mpc-与优化算法)
  - [4.1 cp_planning：显式 MPC](#41-cp_planning显式-mpc)
  - [4.2 所有模块：iLQR 优化](#42-所有模块ilqr-优化)
  - [4.3 OSQP 二次规划](#43-osqp-二次规划)
  - [4.4 各模块优化技术汇总](#44-各模块优化技术汇总)
- [五、关键概念解释](#五关键概念解释)
  - [5.1 侧偏角 β（Sideslip Angle）](#51-侧偏角-βsideslip-angle)
  - [5.2 协方差（Covariance）](#52-协方差covariance)
  - [5.3 RLS（递推最小二乘法）](#53-rls递推最小二乘法)
  - [5.4 OCP（最优控制问题）](#54-ocp最优控制问题)
- [六、Planning MPC vs Control MPC：为什么需要两层 MPC？](#六planning-mpc-vs-control-mpc为什么需要两层-mpc)
- [七、Parking Control 是否需要 RLS 在线辨识？](#七parking-control-是否需要-rls-在线辨识)
- [八、Parking Planning 的轨迹优化深度分析](#八parking-planning-的轨迹优化深度分析)
- [九、全局总结](#九全局总结)

---

## 一、横向控制（Lateral Control）算法对比

### 1.1 Driving Control 横向算法

代码位置：`maf-control/driving_control/src/control_interface/`

#### 1.1.1 ACADO MPC（横向 MPC 1.0 和 2.0）

- **文件**：`include/acado_lat_mpc.h`，`src/acado_lat_mpc.cpp`
- **算法**：ACADO 工具包代码生成的 MPC 求解器，QP 后端为 qpOASES

**MPC 1.0 运动学自行车模型**：
```
状态: [dx, dy, dphi, delta]
控制: [omega]（转向角速率 rad/s）

dy/dt     = v * sin(theta)
dtheta/dt = curv_factor * v * tan(delta)
ddelta/dt = omega
```
采用 RK4 积分。

**MPC 2.0 动力学模型**：
```
状态: [dx, dy, dphi, yawrate, delta]（5维）
控制: [omega]

dy/dt     = v * sin(theta)
dtheta/dt = r（横摆角速度）
dr/dt     = low_speed_tao_factor * ((w1/v + w2*v)*r + w3*delta)
ddelta/dt = omega
```
其中 w1, w2, w3 为车辆动力学参数，w2 可通过 RLS 在线辨识。

**参考轨迹生成**（`control_preprocess_rads.cpp: CalcLatMpcReferenceRads()`）：
1. 路径点 (x, y) 对弧长 s 做样条拟合
2. 曲率对弧长 s 做样条拟合
3. 对 MPC 预测步长（horizon=25, dt=0.1s）：
   - dx_ref, dy_ref 来自样条在 s 处的求值
   - dphi_ref 来自 `atan2(dy_spline_deriv, dx_spline_deriv)`
   - curv_ref 来自曲率样条
   - vel_ref 来自速度样条

#### 1.1.2 iLQR MPC（横向）

- **文件**：`include/ilqr_lat_mpc.h`，`src/ilqr_lat_mpc.cpp`
- **算法**：自定义迭代线性二次调节器（iLQR），支持约束变体 CiLQR（barrier cost）

```
状态: [Y, THETA, R, DELTA]（横向偏差、航向、横摆角速度、前轮转角）
控制: [OMEGA]（转向速率）
配置: horizon=25, dt=0.1s, max_iter=10
```

**代价项**：
- `PathRefCostTerm`：0.5 * W_REF_Y * (REF_Y - Y)² * cos²(REF_THETA)
- `HeadRefCostTerm`：0.5 * W_REF_THETA * (REF_THETA - THETA)²
- `DYBoundCostTerm`：横向走廊约束（CiLQR barrier）
- `OmegaCostTerm`：控制量正则化
- 各种边界约束的 barrier cost

#### 1.1.3 曲率接口 iLQR MPC（Harz 项目）

- **文件**：`include/ilqr_lat_curv_mpc.h`
- **状态**：[Y, THETA, CURV]（横向偏差、航向、曲率）
- **控制**：[CURV_RATE]（曲率变化率）
- 输出曲率指令而非转向角指令（用于曲率接口车辆如 Harz/MMA）

#### 1.1.4 非线性积分滑模控制（横向力矩叠加）

- **文件**：`src/library/advanced_ctrl_lib/include/nonlinear_integral_smc.h`
- **算法**：滑模控制器用于横向力矩反馈

```
滑模面: s = k1*e + integral(k0*rho_dot) + k2*k2_gain*e_dot
控制律: ufb = -(k0*rho_dot + k1*e_dot)/(k2*b) - ufb_max * Sat(s/phi)
其中 b = gm/jeq（蜗轮蜗杆比/等效惯量）
```

#### 1.1.5 力矩前馈 + NESO

- **文件**：`include/lat_smc.h`，`src/lat_smc.cpp`
- **算法**：非线性扩展状态观测器（NESO）估计转向摩擦

```
tfc_comp_ = tc * tanh(kfc * target_steer_rate) + kv * c * tanh(kfc * target_steer_rate) / is_
参数: jeq（等效惯量）, beq, keq, gm（蜗轮蜗杆比）, is_（转向比）
```

#### 1.1.6 横向误差 PI 补偿

- **位置**：`ControlLoop::LatErrPICompensation()`
- 简单 PI 控制器，增益与速度成反比：`kp = Kp_param / v`，仅在弯道（曲率超阈值）时激活

#### 1.1.7 横向执行器后处理链

```
MPC输出 → 曲率接口滤波 → 转向角转换(×转向比) → 低通滤波
→ DOB扰动观测补偿 → Kalman转向偏移补偿 → 力矩前馈+NESO
→ 速率限幅 → 人机共驾（可选）→ 最终转向指令
```

### 1.2 Parking Control 横向算法

代码位置：`maf-control/parking_control/src/control_interface/`

#### 1.2.1 ACADO MPC

- 与 driving_control 共享同一 ACADO 框架
- **模型**：运动学自行车模型
- **状态**：[dx, dy, dphi, delta]（4维，无横摆角速度）
- 用于行车模式（HNP/CP/UNP）和 APA 泊车的 fallback

#### 1.2.2 iLQR APA MPC

- **文件**：`src/ilqr_lat_mpc_apa.cpp`，`src/library/ilqr_lat_apa/src/lat_ilqr_model.cpp`
- **用途**：APA 泊车专用
- **算法**：约束 iLQR（CiLQR），惩罚法

```
模型: 运动学自行车 + RK4 积分
状态: [dx, dy, dphi, delta]（4维）
控制: [omega]（1维）
配置: horizon=25, dt=0.1s, max_iter=10
      max_outer_iterations=10, penalty_factor=20.0
代价: q_y（横向误差）, q_phi（航向误差）, q_omega（转向速率）
      终端代价: q_y_terminal, q_phi_terminal
```

#### 1.2.3 Crab 模式 MPC（四轮转向）

- **文件**：`src/library/crab_lat_apa/src/lat_crab_model.cpp`
- **用途**：四轮转向车辆的蟹行泊车

```
状态: [dx, dy, dphi, delta_f, delta_r]（5维，前后轮转角）
控制: [omega_f, omega_r]（2维，前后轮转角速率）
动力学: 航向角速率用 tan(delta_f) - tan(delta_r) 计算
```

#### 1.2.4 力矩前馈 + ESO

- 与 driving_control 类似但简化版本的 ESO 转向摩擦补偿

### 1.3 横向控制对比总结

| 特性 | Driving Control | Parking Control |
|------|----------------|-----------------|
| **主力算法** | ACADO MPC / iLQR MPC | iLQR APA MPC / ACADO MPC |
| **车辆模型** | MPC 2.0: 动力学（含横摆角速度，5维） | 运动学自行车（4维） |
| **参数在线辨识** | RLS 在线辨识 w2 参数 | 无 |
| **力矩前馈** | NESO 非线性扩展状态观测器 | 简化版 ESO |
| **特殊模式** | 曲率接口 iLQR（Harz） | Crab 模式 MPC（4轮转向） |
| **后处理** | DOB + Kalman + 低通 + 人机共驾 | 相对简单 |

**关键差异**：Driving Control 采用动力学模型 + RLS 在线辨识 + 完善的观测器/滤波器链；Parking Control 采用运动学模型，因为低速下动力学效应不显著。

---

## 二、纵向控制（Longitudinal Control）算法对比

### 2.1 Driving Control 纵向算法

#### 2.1.1 CasADi 力矩 OCP MPC

- **文件**：`include/lon_mpc_cp.h`，`src/lon_mpc_cp.cpp`
- **算法**：CasADi 代码生成 + OSQP QP 求解器的力矩优化 MPC

```
状态: [s, v, M]（位置、速度、力矩，3维）
控制: [u = dM/dt]（力矩变化率）
预测: horizon=30, dt=0.1s = 3秒

车辆模型:
  dv/dt = p0*M + p1*v² + p2*θ + p3
  （区分驱动模型和制动模型，不同的 p0/p1/p2/p3 和 sigma/theta 参数）

特性:
  - 多次 OSQP 迭代 + 线性化点更新
  - 驱动/制动模型自动切换
  - 松弛变量处理力矩切换和速度边界
  - 模型扰动观测器补偿 (坡度、阻力等)
```

#### 2.1.2 OSQP 加速度 MPC

- **文件**：`include/osqp_long_mpc.h`，`src/osqp_long_mpc.cpp`
- **算法**：基于 Jerk 优化的加速度 MPC

```
状态: [s, v, a, a_r]（位置、速度、加速度、参考加速度，4维）
控制: [jerk]

连续动力学:
  Ac = [[0,1,0,0],
        [0,0,1,0],
        [0,0,-1/tau,1/tau],
        [0,0,0,0]]
  Bc = [0,0,0,1]'
  其中 tau 为执行器时间常数

离散化: 双线性变换 Ad = (I - 0.5*Ac*Ts)^-1 * (I + 0.5*Ac*Ts)

初始状态:
  s = 自车在路径上的投影位置
  v = 测量速度
  a = 测量加速度 - ESO 扰动估计
  a_r = 上一周期加速度指令

ESO 扰动观测器: 估计未建模力（坡度、阻力）
```

#### 2.1.3 PI/Lead 速度控制

- **位置**：`ControlLoop::LongitudeVelControl()`
- **算法**：经典 PI 补偿器（Lead/Lag 型），带大量增益调度

```
数据流:
  位置环输出 → 低通滤波 → 转向减速限制 → 坡度补偿(DOB)
  → 速度误差计算 → 误差滤波(低通+死区) → 动态增益调度
  → PI 补偿器 → 加速度前馈 → 最终加速度指令

增益调度维度:
  - 坡度（上坡/下坡）
  - 车速（高速增益）
  - 车型
```

#### 2.1.4 RX5 PID 控制器（策略层）

- **文件**：`src/strategy/src/msquare/actuators/rx5/pid.hpp`
- 车辆级 PID，将加速度指令转为油门/制动
- 速度相关 kp/ki 插值表 + 误差相关增益因子 + 积分抗饱和

### 2.2 Parking Control 纵向算法

#### 2.2.1 PI 级联控制（位置环 + 速度环）— APA 主力

- **文件**：`src/control_loop_apa.cpp`
- **架构**：

```
位置环: remain_s → 速度指令（基于距离的减速策略 v² / (2*remain_s)）
速度环: PI 控制 + 坡度补偿(DOB) + 滤波（Butterworth/Notch/Lead）

APA 控制状态机:
  PID     → LongitudePosControlApa() → LongitudeVelControlApa() → 横向MPC
  STARTING → VehicleStartApa() → 横向MPC
  STOPPING → VehicleStopApa() → 横向MPC
  BREAK    → 位置/速度控制 + 特殊制动策略
```

#### 2.2.2 OSQP 纵向 MPC（SOP 架构）

- **文件**：`src/osqp_long_mpc.cpp`
- 与 driving_control 的 OSQP MPC 结构相同
- **关键差异**：**horizon=10**（仅 1 秒预测，driving 为 30 步 3 秒）

#### 2.2.3 CasADi 纵向 OCP（行车模式）

- 用于 parking_control 中的行车模式（HNP/CP/UNP），非泊车模式
- 与 driving_control 的 OCP 结构相同

### 2.3 纵向控制对比总结

| 特性 | Driving Control | Parking Control (APA) |
|------|----------------|----------------------|
| **主力算法** | CasADi 力矩 OCP + OSQP 加速度 MPC | PI 级联控制（位置环+速度环） |
| **OCP 力矩优化** | 有，[s,v,M] 3维，horizon=30 | 无（APA 模式） |
| **OSQP MPC** | [s,v,a,a_r] 4维，**horizon=30** | [s,v,a,a_r] 4维，**horizon=10** |
| **在线模型辨识** | RLS 辨识 [p0,p1,p2,p3] | 无 |
| **核心策略** | 跟踪精细的 s-v-a 轨迹 | 基于 remain_s 距离的减速控制 |

---

## 三、规划轨迹精细度对比

### 3.1 Planning 输入数据结构

定义在 `maf_interface/maf_planning.h`：

```cpp
struct Trajectory {
  std::vector<PathPoint> path{};            // 最多 320 点
  Velocity velocity{};
  Acceleration acceleration{};
  std::vector<FrenetState> frenet_state{};  // 最多 320 点
};

struct PathPoint {
  Point2d position_enu{};   // ENU 坐标 (x, y)
  double heading_yaw{};     // 航向角
  double curvature{};       // 曲率
  double path_follow_strength{};
};

struct VelocityPoint {
  double target_velocity{}; // 目标速度 (m/s)
  double relative_time{};   // 相对 planning 起点的时间 (s)
  double distance{};        // 相对 planning 起点的距离 (m)
};

struct AccelerationPoint {
  double acc{};             // 目标加速度 (m/s²)
  double jerk{};            // 目标加加速度 (m/s³)
};
```

### 3.2 Driving 轨迹 vs Parking 轨迹

| 特性 | Driving Planning 轨迹 | Parking Planning 轨迹 |
|------|----------------------|----------------------|
| **采样方式** | **时域采样** 40 Hz（0.025s 间隔） | **空域采样**（0.05~0.2m 间隔） |
| **最大点数** | 320 点 = 8 秒前瞻 | 取决于路径长度（通常 50~200 点） |
| **速度剖面** | 密集 VelocityPoint 时序（320点），含 target_velocity、relative_time、distance | 单一 target_value + piecewise-jerk 速度剖面 |
| **加速度剖面** | 密集 AccelerationPoint（320点），含 acc 和 jerk | 有但点数更少 |
| **曲率信息** | 每个路径点含 curvature | 每个路径点含 curvature |
| **运输延迟补偿** | 有（transport_delay 时间偏移） | 无（低速不敏感） |
| **轨迹更新频率** | 高频重规划（~100-200ms） | 整段路径一次规划，仅换挡时更新 |

**Control 端预处理**（两者共享相同的样条拟合流程）：
1. 找到自车在规划路径上的匹配点（最近点搜索）
2. 投影计算横向误差和航向误差
3. 构建弧长参数化样条（路径点）
4. 构建时间参数化样条（速度/加速度点）
5. 生成 MPC 参考向量：横向 25 步 × 0.1s = 2.5s；纵向 30 步 × 0.1s = 3s

---

## 四、Driving Planning 中的 MPC 与优化算法

代码位置：`/home/sonzhen/12-mmt_whole_code_072_12f/pangu_ad_sw/maf-planning/`

### 4.1 cp_planning：显式 MPC

**文件**：`cp_planning/src/cp_planning/src/planner/motion_planner/optimizers/optimal_speed_planner.cpp`

`OptimalSpeedPlanner` 类通过 `dlopen` 加载 `libmotion_planner_for_dpmpc.so`，调用 `run_speed_planner_mpc`：

```
MPC 配置:
  预测步长: MPC_N = 25
  时间步长: delta_t = 0.2s
  预测窗口: 5 秒

状态向量: [s_frenet, r_frenet, theta_error, vel, acc, omega]
控制输出: [omega_rate（转向速率）, jerk]

在线数据:
  参考位置 sr, 参考速度 vr
  位置上下界, 曲率, 加速度上下界

代价权重:
  s_reference: 5.0, v_reference: 0.0
  accel: 1.0, jerk: 5.0, curvature: 0.0

输出: 优化后的 s, velocity, acceleration, jerk 剖面
```

### 4.2 所有模块：iLQR 优化

iLQR（迭代线性二次调节器）是一种与 MPC 密切相关的非线性轨迹优化方法，是所有 driving planning 模块的主力优化技术。

#### adas_planning — CiLQR 路径规划

- **文件**：`adas_planning/src/planning_stack/path_planner/cilqr_planner/`
- 增广拉格朗日 iLQR + 回溯线搜索
- 状态: (x, y, heading, velocity)，控制: (steering, acceleration)
- 代价组件：动力学模型、轨迹代价、轨迹约束、障碍物代价、参考路径代价

#### unp_planning — 最广泛的 iLQR 使用（134 个相关文件）

- **时空 iLQR**（横向路径优化）：`unp_planning/core/modules/path_planner/spatio_temporal_planner/`
- **纵向 iLQR v4**（速度优化）：`unp_planning/core/modules/speed_planner/`
- **SV iLQR**（速度-位置联合优化）：`unp_planning/core/modules/sv_planner_v4/`
- **JP iLQR**（博弈论多智能体交互）：`unp_planning/core/modules/jp_lon_planner/`

#### cp_planning — CiLQR 速度规划

- **文件**：`cp_planning/src/planning_module/planning_stack/speed_planner/cilqr_speed_planner/`
- 约束 iLQR 用于城市驾驶速度规划

### 4.3 OSQP 二次规划

#### 参考线平滑（cp_planning）

- **文件**：`cp_planning/src/cp_planning/src/common/refline/discretized_points_smoothing/fem_pos_deviation_osqp_interface.cpp`
- FEM 位置偏差平滑器，最小化位置偏差 + 路径长度 + 参考偏差 + 航向代价
- 权重：relaxation=5e5, fem_pos_deviation=1e2, path_length=1.0, ref_deviation=1.0, heading=1e5

#### Piecewise Jerk 速度优化（adas_planning & cp_planning）

- 优化变量：[s_0..s_n, v_0..v_n, a_0..a_n, jerk_0..jerk_n]
- OSQP 求解，带连续性约束和动力学约束

### 4.4 各模块优化技术汇总

| 模块 | 横向/路径优化 | 纵向/速度优化 | 参考线平滑 |
|------|-------------|-------------|-----------|
| **adas_planning** | CiLQR（增广拉格朗日 iLQR） | Piecewise Jerk QP（OSQP） | 无 |
| **cp_planning** | 数据驱动 + 规则 | MPC + Piecewise Jerk QP + CiLQR | FEM OSQP |
| **unp_planning** | 时空 iLQR | 纵向 iLQR v4、SV iLQR、博弈 iLQR | 无 |
| **pnp_planning** | Ceres clothoid + iLQR | 复用 unp 纵向优化器 | 无 |

### Driving Planning 完整轨迹处理流程

```
原始路径 → 参考线 OSQP 平滑 → iLQR/MPC 横向路径优化 → iLQR/MPC/QP 纵向速度优化
→ 输出精细的 (x, y, θ, κ, v, a, jerk) 轨迹
```

---

## 五、关键概念解释

### 5.1 侧偏角 β（Sideslip Angle）

**侧偏角 β** 是车辆速度方向与车身纵向轴之间的夹角。

更准确地说：

- 车身朝向由航向角 `yaw / theta / phi` 描述
- 车辆实际运动方向由速度矢量方向描述
- 两者不完全一致时，就存在侧偏角 `β`

平面示意：

```
车身纵向轴方向:      →
实际速度方向:         ↗
两者夹角:             β
```

它反映的是轮胎侧偏力已经建立、车辆不再“指哪打哪”的程度。

在车辆动力学里：

- `β = 0`：速度方向与车头方向一致，更接近纯运动学状态
- `β > 0` 或 `β < 0`：说明车辆在“滑着走”，即使车头指向某个方向，实际速度方向会有偏差

对控制的意义：

- 低速泊车时，`β` 通常很小，运动学模型往往足够
- 中高速转弯时，`β` 不可忽略，需要动力学模型描述轮胎侧偏力
- `β` 增大时，车辆更接近极限工况，横向稳定性控制和高精度 MPC 都会更关注它

经典线性自行车模型中，状态常取为 `β` 与 `r`：

$$
\dot{\beta} = f(\beta, r, \delta, v)
$$

$$
\dot{r} = g(\beta, r, \delta, v)
$$

文档前面写成单状态横摆模型：

$$
\dot r = \text{low\_speed\_tao\_factor} \cdot \left(\left(\frac{w_1}{v} + w_2 v\right) r + w_3 \delta\right)
$$

本质上就是把完整的 `β-r` 二自由度模型做了简化，把 `β` 的影响折算进等效参数里。

### 5.2 协方差（Covariance）

**协方差** 用来描述“估计值有多不确定”。

在 RLS、Kalman Filter、状态观测器里，协方差不是车辆的物理量，而是算法内部对“我现在有多确定”的量化。

如果只看一个待辨识参数，比如 `w2`，协方差可以理解为：

- 协方差大：当前对 `w2` 的估计不太有把握，新数据来了会改得更多
- 协方差小：当前对 `w2` 的估计比较有把握，新数据只能小幅修正

如果把多个参数一起估计，例如：

$$
  heta = [w_1, w_2, w_3]^T
$$

那么协方差矩阵 `P` 的含义是：

- 对角线元素：每个参数自己的不确定度
- 非对角线元素：不同参数估计误差之间的相关性

直观上，协方差矩阵回答的是两个问题：

1. 哪个参数目前最不确定
2. 某个参数变动时，会不会“连带影响”另一个参数的估计

在 RLS 中：

- 初始 `P` 往往设得较大，表示“刚开始我没什么把握”
- 数据持续进入后，`P` 一般会收缩，表示估计越来越稳定
- 如果引入遗忘因子，旧数据会逐渐淡化，`P` 不会无限收缩到完全不动

### 5.3 RLS（递推最小二乘法）

**RLS = Recursive Least Squares（递推最小二乘法）**，是一种在线参数辨识算法。

**核心思想**：车辆的真实物理参数（质量、轮胎刚度、路面摩擦等）会随工况变化，无法提前标定准确。RLS 通过实时采集传感器数据，递推更新模型参数的估计值。

在 driving_control 中的具体应用：

| 辨识对象 | 数学表达 | 用途 |
|---------|---------|------|
| 横向动力学参数 w2 | `dr/dt = (w1/v + w2*v)*r + w3*delta` | w2 与轮胎侧偏刚度相关，随路面/轮胎状态变化 |
| 纵向驱动模型 [p0,p1,p2,p3] | `dv/dt = p0*M + p1*v² + p2*θ + p3` | p0 与质量相关，p1 与风阻相关，p2 与坡度相关 |
| 整车负载率 | — | 估计车辆当前载重 |

递推公式本质：每个控制周期用新测量值修正参数估计，使模型始终贴近真实车辆状态。

#### 横向动力学里 w1 / w2 / w3 的含义

对文档前面的横摆角速度动力学：

$$
\dot r = \alpha \cdot \left(\left(\frac{w_1}{v} + w_2 v\right) r + w_3 \delta\right)
$$

其中：

- `r`：横摆角速度 yaw rate
- `v`：车速
- `delta`：前轮转角
- `alpha`：`low_speed_tao_factor`

三个参数的直观意义如下：

1. `w1`

表示与 `1/v` 成正比的横摆阻尼项，低速时影响更强。它更多反映车辆几何和轮胎恢复力带来的基础横摆阻尼。

2. `w2`

表示与 `v` 成正比的速度相关稳定性项。速度越高，这一项对横摆动态影响越明显。它对轮胎侧偏刚度、路面附着和载荷转移最敏感，因此最适合在线辨识。

3. `w3`

表示转向输入 `delta` 到横摆角加速度 `dot(r)` 的输入增益。直观上，它决定“同样的前轮打角，能建立多大的横摆响应”。

从完整的线性自行车模型角度看，`w1 / w2 / w3` 不是原始物理常数，而是把以下因素合并后的等效参数：

- 前后轮侧偏刚度 `Cf / Cr`
- 前后轴到质心距离 `lf / lr`
- 整车质量 `m`
- 绕 z 轴转动惯量 `Iz`

因此它们不是完全固定不变的“几何量”，而是会随轮胎、路面、载荷等工况发生偏移的等效动力学参数。

#### 为什么常用 RLS 在线辨识 w2

因为 `w2` 对高速横摆稳定性最敏感，也最容易随以下因素变化：

- 路面附着系数变化
- 轮胎温度、磨损、胎压变化
- 前后轴载荷变化
- 高速工况下的侧偏刚度变化

相比之下，`w1` 和 `w3` 往往可以通过离线标定给出较稳定的初值，而 `w2` 更适合在线自适应更新。

#### 如何把模型写成 RLS 的回归形式

如果只辨识 `w2`，将模型整理为：

$$
\frac{\dot r}{\alpha} - \frac{w_1}{v}r - w_3 \delta = w_2 (v r)
$$

定义：

$$
y_k = \frac{\dot r_k}{\alpha_k} - \frac{w_1}{v_k}r_k - w_3 \delta_k
$$

$$
\phi_k = v_k r_k
$$

则得到标准的一维线性回归：

$$
y_k = \phi_k w_2
$$

这时 RLS 每个周期用新的测量值 `(v_k, r_k, dot(r)_k, delta_k)` 更新参数估计 `hat(w2)`。

#### 标量 RLS 更新公式

设：

- `hat(w2)_(k-1)` 是上一个周期的参数估计
- `P_(k-1)` 是上一个周期的协方差
- `lambda` 是遗忘因子

则：

$$
K_k = \frac{P_{k-1}\phi_k}{\lambda + \phi_k^2 P_{k-1}}
$$

$$
\hat w_{2,k} = \hat w_{2,k-1} + K_k \left(y_k - \phi_k \hat w_{2,k-1}\right)
$$

$$
P_k = \frac{1}{\lambda}(1 - K_k \phi_k) P_{k-1}
$$

其中：

- `K_k` 是本周期更新增益
- 括号里的残差 `y_k - phi_k * hat(w2)` 表示“模型预测误差”
- `P_k` 越大，说明当前估计越不确定，后续更新会更激进

#### 实际工程里如何做

理论公式很短，真正难的是数据质量控制。一般会这样做：

1. 使用滤波后的 `r`

因为 `dot(r)` 常由差分得到，噪声会被放大，因此通常先对 `r` 做低通滤波。

2. 使用实际轮角而不是方向盘角

模型里的 `delta` 是前轮转角，不是方向盘手轮角；中间还需要考虑转向比和执行器特性。

3. 只在有效工况更新

例如：

- 车速高于阈值
- 转向输入不为零
- 传感器状态正常
- ABS/ESP 等强干预未激活

4. 给参数加边界

防止传感器异常或瞬时噪声把 `w2` 拉到不合理区间。

5. 要有足够激励

车辆长期直线行驶时，`r` 很小，`phi = v*r` 接近零，几乎无法从数据中辨识出 `w2`。

### 5.4 OCP（最优控制问题）

**OCP = Optimal Control Problem（最优控制问题）**，是一个数学优化框架。

在 driving_control 纵向控制中的具体形式：

```
最小化:  J = Σ [ w_s·(s-s_ref)² + w_v·(v-v_ref)² + w_m·(M-M_ref)² + w_u·u² ]
                                                       ↑ 力矩误差      ↑ 力矩变化率

约束:    ds/dt = v                          （位置动力学）
         dv/dt = p0·M + p1·v² + p2·θ + p3   （车辆纵向动力学）
         dM/dt = u                           （力矩变化率为控制量）
         M_min ≤ M ≤ M_max                   （力矩饱和）
         u_min ≤ u ≤ u_max                   （力矩变化率限制）
```

**OCP vs MPC 的关系**：

| | OCP | MPC |
|---|---|---|
| **定义** | 数学优化问题的**形式化描述** | 一种**滚动求解** OCP 的控制策略 |
| **关系** | 是"题目" | 是"解题方法" |

MPC 的本质：每个控制周期重新求解一个有限时域的 OCP，只执行第一步控制量，然后滚动前进。代码中的 `LonMpcCp` 本质是 MPC 控制器——把纵向控制建模为 OCP，用 CasADi 代码生成 + OSQP 求解器实时求解。

---

## 六、Planning MPC vs Control MPC：为什么需要两层 MPC？

### 两个 MPC 解决的问题完全不同

```
Planning MPC："应该走什么样的轨迹？" → 生成轨迹（决策层）
Control MPC："怎么操控方向盘和油门/刹车来跟上这条轨迹？" → 跟踪轨迹（执行层）
```

### 核心差异对比

| 维度 | Planning MPC | Control MPC |
|------|-------------|-------------|
| **目标** | 生成避障、合规、舒适的轨迹 | 精确跟踪给定轨迹 |
| **输出** | 轨迹点 (x, y, v, a, jerk) | 方向盘转角/力矩、油门/制动力矩 |
| **约束** | 道路边界、障碍物、交通规则 | 执行器饱和、转向速率限制、舒适性 |
| **车辆模型** | 简化（运动学/质点） | 详细（动力学+执行器延迟+摩擦） |
| **运行频率** | ~5-10 Hz, dt=0.2s, 预测 5 秒 | ~10-50 Hz, dt=0.1s, 预测 2.5-3 秒 |

### 五个必要性理由

**1. 模型精度不同**

Planning 使用简化模型（运动学/质点），因为需要快速搜索大量候选轨迹。Control 使用精细模型（动力学 + 执行器延迟 + 轮胎非线性）。Planning 认为"可行"的轨迹，实际车辆不一定能精确执行。

**2. 运行频率不同**

Planning 每 100-200ms 重规划一次。两次规划之间，车辆可能已偏离计划位置。Control MPC 在每个控制周期（10-100ms）实时修正偏差。

**3. 反馈 vs 前馈**

```
Planning：本质是"前馈"（feedforward）→ 基于当前感知预测未来，输出开环轨迹
Control：本质是"反馈"（feedback）→ 实时测量误差，闭环修正
```

**4. 外部扰动**

即使轨迹完美，仍需 Control MPC 处理：传感器延迟、侧风、路面不平、坡度突变、轮胎磨损、载重变化、转向间隙、制动响应延迟。

**5. 如果去掉 Control MPC**

直接把 Planning 输出当执行器指令：无闭环反馈导致误差累积发散；无执行器模型导致延迟偏差；无在线辨识导致系统性偏差。车辆会逐渐偏离规划轨迹，尤其在高速弯道。

### 直觉类比

> **Planning MPC** = 导航软件规划的路线（"沿这条路走，在这里转弯，保持这个速度"）
>
> **Control MPC** = 驾驶员的手和脚（实时调整方向盘应对侧风、修正车道偏移、控制油门应对坡度）
>
> **RLS** = 驾驶员的学习能力（换了更重的车或路面变滑，自适应调整操作力度）

---

## 七、Parking Control 是否需要 RLS 在线辨识？

### 现状：未使用 RLS

泊车场景特点使得 RLS 非必须：
- **低速**（0.3~1.0 m/s）→ 空气阻力 `p1·v²` 几乎为零
- **动力学效应弱** → 轮胎侧偏力小，运动学模型够用
- **控制器简单**（PI 级联）→ 积分项可缓慢消除稳态误差
- **精度容忍度相对高** → 厘米级够用（对比高速行车的毫米级）

### 引入 RLS 的潜在好处

| 问题场景 | RLS 能做什么 |
|---------|-------------|
| **坡道泊车** | 辨识坡度参数 `p2·θ`，停车场常有坡道，坡度变化导致纵向控制偏差 |
| **载重变化** | 辨识质量参数 `p0`，满载/空载差异大（SUV 后备箱满载可差几百公斤），影响制动距离 |
| **路面差异** | 辨识摩擦参数，地库光滑地面 vs 室外粗糙路面，转向响应差异明显 |
| **转向非线性** | 辨识横向模型参数，低速时转向摩擦占主导，不同车辆/温度差异大 |
| **精准停车** | 泊车最后阶段需精确到 ±5cm，模型失配直接导致停不准 |

### 最值得引入的两项

- **纵向**：坡度/质量辨识 → 解决坡道上"停不住"或"提前停"的问题，RLS 辨识坡度后可做前馈补偿
- **横向**：低速转向摩擦辨识 → 改善 iLQR MPC 预测精度，减少泊车轨迹跟踪的横向偏差

---

## 八、Parking Planning 的轨迹优化深度分析

代码位置：`maf-planning/apa_planning/`

### 8.1 路径规划算法

| 路径规划器 | 原理 | 输出特点 |
|-----------|------|---------|
| **Hybrid A\*** | 栅格搜索 + 运动学扩展 | 栅格分辨率 0.2m，`MultiCircleFootprintModel` 碰撞检测 |
| **Target Hybrid RS Star** | Reeds-Shepp 曲线搜索 + 双向搜索 | 圆弧+直线组合，曲率不连续（跳变） |
| **Clothoid / Euler Spiral** | Fresnel 积分计算 Euler 螺线 | 仅用于端点连接过渡，非全局优化 |
| **Tail Curve Join** | A* 搜索最优 Euler Spiral + 直线 | 端点特定平滑，非全局路径优化 |
| **DLP 深度学习** | `minfer_v5` 推理引擎 + 编码器-解码器 | 依赖训练数据质量 |

### 8.2 路径后处理优化：**没有**

关键发现：APA 泊车管线**不对路径做后处理优化**。

**不存在的优化**：
- 无 IPOPT / 非线性优化
- 无 FEM 位置平滑
- 无通用 QP 路径平滑
- 无 MPC 轨迹精修
- 无 iLQR 路径精修（iLQR 仅用于 AVP 巡航场景，非 APA 泊车）

路径直接从几何搜索规划器输出到控制模块。

### 8.3 速度优化：**有，且较精细**

速度规划并非简单梯形剖面，有两种优化方法：

**方法一：OSQP Piecewise-Jerk QP**
```
优化变量: 3N 个 (s, v=ds, a=dds) 在每个节点
目标函数: w_s·‖s-s_ref‖² + w_v·‖v-v_ref‖² + w_a·‖a‖² + w_j·‖jerk‖²
约束:     连续性约束, jerk 边界, 速度/加速度边界
求解器:   OSQP（max_iter=500, polish=true）
```

**方法二：iLQR SV 优化**
```
状态: [v², a]
控制: [da/ds]
预测步: 40
CiLQR: max_outer_iterations=5, penalty_factor=20, init_rho=1000
代价项: VSqr, A, DaDs, VSqrBound, ABound, DaDsBound, VSqrKappa, VSqrObs
速度限制来源: 曲率、障碍物距离、TTC、剩余距离
暖启动: Bang-Bang 正反向平滑
```

**速度管线**（`parking_speed_planner_v2.cpp`）：
- 多重速度限制器：前车、自由空间、行人避让、剩余距离、动态障碍物、预测、自行车、APA、余量
- 配置选择 `compute_speed_use_ilqr` 或 `compute_speed_use_osqp`

### 8.4 与 Driving Planning 的对比

```
Driving Planning 流程:
  原始路径 → OSQP 参考线平滑 → iLQR/MPC 横向路径优化 → iLQR/MPC/QP 纵向速度优化
  路径: ✅ 多层优化          速度: ✅ 多层优化

Parking Planning 流程:
  Hybrid A*/RS 曲线 → (无路径优化) → OSQP/iLQR 纵向速度优化
  路径: ❌ 无后处理优化       速度: ✅ 有优化
```

### 8.5 Parking Planning 路径不优化的原因

这是**刻意的架构选择**，而非疏忽：
- 低速场景动力学不关键，几何精度更重要
- 泊车路径含换挡和曲率不连续，难以用光滑轨迹优化处理
- Hybrid A* + RS 曲线已内置运动学约束（最小转弯半径、clothoid 过渡）
- QP/iLQR 速度优化确保了舒适的速度剖面

### 8.6 Parking 路径粗糙的具体表现

1. **曲率不连续**：RS 曲线在圆弧和直线交界处曲率跳变（从 0 突变到 1/R），导致方向盘突然打角
2. **路径锯齿**：Hybrid A* 的栅格分辨率限制（0.2m）导致路径不够平滑
3. **无动力学可行性保证**：路径只满足运动学约束（最小转弯半径），不考虑转向速率、加速度等

这正是为什么 parking control 的横向 iLQR MPC 承担了更多"平滑补偿"的角色——它不仅要跟踪轨迹，还要在跟踪过程中平滑掉路径本身的几何粗糙度。

---

## 九、全局总结

### 算法复杂度层级

```
              Driving                    Parking
Planning:  MPC + iLQR + QP（多层优化） → Hybrid A* + RS（几何搜索）+ QP/iLQR（速度）
              ↓                             ↓
           精细轨迹                       粗糙路径 + 优化速度
              ↓                             ↓
Control:   MPC(动力学) + RLS + ESO/DOB  → iLQR(运动学) + PI级联
              ↓                             ↓
           精确跟踪                       鲁棒跟踪 + 路径平滑
```

### 核心结论

1. **Driving Planning 的轨迹确实更精细**：40Hz 时域采样，完整的 s-v-a-jerk 耦合剖面，经过 MPC/iLQR/QP 多层优化
2. **Parking Planning 的路径确实更粗糙**：纯几何搜索输出，无后处理路径优化，曲率不连续
3. **Driving Planning 确实使用了 MPC**：cp_planning 的 `OptimalSpeedPlanner` 直接使用 25 步 MPC 优化速度；所有模块广泛使用 iLQR
4. **两层 MPC 各司其职**：Planning MPC 生成可行轨迹（前馈），Control MPC 精确跟踪（反馈闭环）
5. **Parking Control 的 MPC 额外承担了路径平滑责任**：弥补 Planning 路径的几何粗糙度
