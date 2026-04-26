# 📖 学习笔记：E2E Parking 项目知识点 Q&A 汇总

---

## 目录

1. [Python 基础](#1-python-基础)
2. [CARLA 仿真器基础](#2-carla-仿真器基础)
3. [数据采集流程](#3-数据采集流程)
4. [Token 化控制信号](#4-token-化控制信号)
5. [模型架构与张量形状](#5-模型架构与张量形状)
6. [Embedding 与位置编码](#6-embedding-与位置编码)
7. [相机与深度图](#7-相机与深度图)
8. [LSS 算法（Lift-Splat-Shoot）](#8-lss-算法lift-splat-shoot)
9. [BEV 分割](#9-bev-分割)
10. [训练机制](#10-训练机制)
11. [数据采集细节（DataGenerator）](#11-数据采集细节datagenerator)
12. [FeatureFusion 与 Motion 编码](#12-featurefusion-与-motion-编码)
13. [BevEncoder 与 ResNet18](#13-bevencoder-与-resnet18)

---

## 1. Python 基础

### Q: `str2bool` 函数有什么用？

用于 `argparse` 参数解析。`argparse` 默认的 `type=bool` 有 bug——`bool('False')` 返回 `True`（非空字符串都是 True）。
`str2bool` 手动把字符串映射到布尔值：

```python
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
```

### Q: `.lower()` 是什么？

Python 字符串方法，将所有字符转为小写：`'True'.lower()` → `'true'`。
用于忽略大小写比较，这样用户输入 `True`、`TRUE`、`true` 都能正确解析。

### Q: `args.host` 是什么？

`argparse` 解析后的命令行参数对象的属性。`--host` 参数 → `args.host`。
默认值 `'127.0.0.1'`（本机回环地址），即 CARLA 服务器在本机运行。

### Q: `gamma` 参数是什么？

Gamma 校正，用于调整图像亮度的非线性映射：$\text{output} = \text{input}^{\gamma}$

- $\gamma < 1$：图像变亮（暗部细节增强）
- $\gamma = 1$：无变化
- $\gamma > 1$：图像变暗

代码中默认 `gamma=0.0`，实际在 CameraManager 中会特殊处理（0 表示不做校正）。

---

## 2. CARLA 仿真器基础

### Q: CARLA 是什么架构？

**Server-Client 架构**：
- **Server**：UE4（虚幻引擎4）渲染的 3D 世界，启动命令 `./CarlaUE4.sh -opengl`
- **Client**：Python 脚本，通过 TCP 连接 Server，发送控制指令、接收传感器数据
- **通信**：默认端口 2000，`carla.Client(host, port)`

### Q: `-opengl` 是什么意思？

CARLA Server（UE4 引擎）的渲染后端选择：
- **OpenGL**：更通用的图形API，兼容性好，适合无显示器的服务器（配合 Xvfb 虚拟显示）
- **Vulkan**：默认后端，性能更好但兼容性要求更高
- 与 Python 客户端无关，客户端只通过 TCP 收发数据

### Q: `tick_busy_loop(60)` 是什么？

pygame 的帧率控制方法。参数 60 表示限制 pygame 渲染循环最多 60 FPS。
与 CARLA 仿真帧率（30 Hz，由 `fixed_delta_seconds = 1/30` 决定）是两个独立的时钟：
- **CARLA 仿真**：30 Hz，物理世界精确推进
- **pygame 显示**：≤60 Hz，用于人工操作员看到的界面刷新

`tick_busy_loop` vs `tick`：前者用 CPU 忙等待，计时更精确。

---

## 3. 数据采集流程

### Q: 整体数据采集流程是什么？

```
人工驾驶（键盘 WASD）
    ↓
game_loop 主循环（每帧）：
    ① world_tick()     → CARLA 仿真推进一步（1/30 秒）
    ② tick(clock)      → 检查碰撞 → 每3帧保存传感器数据 → 检查是否到达目标
    ③ render(display)  → 渲染 pygame 窗口
    ↓
停稳 2 秒后 → save_sensor_data() → 多线程写磁盘
    ↓
restart() → 换下一个车位 → 继续
```

### Q: 保存到磁盘的数据结构？

```
save_path/Town04_Opt/MM_DD_HH_MM_SS/
├── task0/
│   ├── rgb_front/        0000.png, 0001.png, ...
│   ├── rgb_left/         ...
│   ├── rgb_right/        ...
│   ├── rgb_rear/         ...
│   ├── depth_front/      ...
│   ├── depth_left/       ...
│   ├── depth_right/      ...
│   ├── depth_rear/       ...
│   ├── lidar/            0000.ply, ...
│   ├── measurements/     0000.json, 0001.json, ...  （位置/速度/控制量）
│   ├── parking_goal/     0001.json  （目标车位坐标）
│   └── topdown/          encoded_0000.png, ...  （BEV 俯视图）
├── task1/
│   └── ...
```

每个 measurements JSON 包含：
- 位置 (x, y, z)、姿态 (pitch, yaw, roll)
- 速度 (speed, km/h)
- 控制量 (Throttle, Steer, Brake, Reverse, Gear)
- IMU (加速度、角速度、指南针)
- GNSS (经纬度)

---

## 4. Token 化控制信号

### Q: 为什么要把控制信号 token 化？

将连续的控制量（throttle、steer、brake）离散化为 token，用 **Transformer Decoder 自回归预测**，就像 GPT 生成文字一样"生成"控制序列。

### Q: token_nums = 204 怎么来的？

| 值 | 含义 |
|----|------|
| 0 ~ 199 | 200 个有效控制值（half_token=100，范围 [-1,1] 映射到 [0,199]） |
| 200 | 保留（未使用） |
| 201 | **BOS**（Begin of Sequence，序列开始标记） |
| 202 | **EOS**（End of Sequence，序列结束标记） |
| 203 | **PAD**（填充标记） |

### Q: tokenize 过程？

以 steer = -0.35 为例：
1. `steer` 范围 [-1, 1]
2. 加 1 → 0.65，映射到 [0, 2]
3. 除以 2 → 0.325，归一化到 [0, 1]
4. 乘以 199 → 64.675
5. 四舍五入 → 65
6. 最终 token_id = 65

### Q: 为什么 steer 和 reverse 共享 token 空间？

因为 token 只是个"编号"，模型通过 **位置（序列中的第几个）** 来区分含义：
- 位置 0：BOS
- 位置 1：throttle/brake（第1帧）
- 位置 2：steer（第1帧）
- 位置 3：reverse（第1帧）
- 位置 4~6：第2帧的三个控制量
- ...
- 位置 13：EOS
- 位置 14：PAD

同一个 token_id=65 在位置 2 表示 steer=-0.35，在位置 3 表示 reverse 的某个值——**模型通过位置编码区分**。

### Q: 完整的 gt_control 序列？

`future_frame_nums = 4`，每帧 3 个控制量：

```
[BOS, t1,s1,r1, t2,s2,r2, t3,s3,r3, t4,s4,r4, EOS, PAD]
  0    1  2  3   4  5  6   7  8  9  10 11 12   13   14
```

总长度 = 1 + 4×3 + 1 + 1 = **15**（对应 `tf_de_tgt_dim = 15`）

---

## 5. 模型架构与张量形状

### Q: 整体模型架构？

```
输入：4 路 RGB 图像 + 相机内外参 + 目标车位坐标 + 自车运动
  ↓
CamEncoder（EfficientNet-B4）→ 图像特征 + 深度分布
  ↓
BevModel（LSS: Lift-Splat-Shoot）→ BEV 鸟瞰图特征
  ↓
拼接目标车位热力图 → [B, 65, 200, 200]
  ↓
BevEncoder（ResNet18）→ 压缩 BEV 特征 [B, 256, 256]
  ↓
FeatureFusion 拼接 ego motion → Transformer Encoder, 4层 → [B, 256, 258]
  ├→ SegmentationHead → BEV 分割预测 [B, 3, 200, 200]（辅助任务）
  ↓
ControlPredict（Transformer Decoder, 4层）→ [B, 14, 204]
  ↓
输出：14 个位置各 204 类的 logits → argmax 得到 token → detokenize 得到控制量
```

注意：4 路深度图是训练深度辅助任务的真值，不是 `ParkingModel` 的直接输入。

### Q: Transformer Decoder 的输入输出形状？

| 阶段 | 形状 | 说明 |
|------|------|------|
| gt_control（训练时的 target） | [B, 15] | token_id 序列 |
| Embedding(204, 258) | [B, 15, 258] → 取前14个 [B, 14, 258] | token 嵌入 |
| + pos_embed | [B, 14, 258] | 加上位置编码 |
| Transformer Decoder | [B, 14, 258] | 4层解码 |
| Linear(258, 204) | **[B, 14, 204]** | 每个位置预测 204 类 |

为什么输出是 14 而不是 15？因为 **Teacher Forcing**：用前 14 个 token 预测后 14 个 token（即位置 1~14），第 15 个（PAD）不需要预测。

### Q: FeatureFusion 的输入 `[B, 256, 258]` 是怎么来的？

BevEncoder 输出：

```python
bev_down_sample: [B, 256, 256]
```

这里第一个 256 是 ResNet18 layer3 的通道数，第二个 256 是 `16×16` 个 BEV 空间位置。

进入 `FeatureFusion` 后先转置：

```python
[B, C=256, S=256] -> [B, S=256, C=256]
```

`ego_motion = [speed, acc_x, acc_y]` 经过 MLP 后变成 2 个附加特征维度：

```python
ego_motion:     [B, 1, 3]
motion_feature: [B, 256, 2]
```

最后在特征维拼接：

```python
BEV token 特征: [B, 256, 256]
motion 特征:    [B, 256, 2]
拼接后:         [B, 256, 258]
```

所以这里没有 66 个 token，也没有把 `target_point` 当作 token 输入 Transformer。目标车位坐标在进入 BevEncoder 前已经被画成了 1 通道热力图，与 64 通道 BEV 特征拼成 `[B, 65, 200, 200]`。

---

## 6. Embedding 与位置编码

### Q: `Embedding(204, 258)` 是什么意思？

```
nn.Embedding(num_embeddings=204, embedding_dim=258)
```

- **204**：词汇表大小（token_id 范围 0~203）
- **258**：每个 token 映射为 258 维向量

内部是一个 **查找表**（lookup table），形状 `[204, 258]`：
- 输入 token_id=65 → 返回第 65 行，一个 258 维向量
- 这 258 个值是**可学习参数**，随训练更新

### Q: Embedding 的向量初始值是什么？

默认用 **标准正态分布** $\mathcal{N}(0, 1)$ 随机初始化。训练过程中通过反向传播更新。

### Q: 位置编码 pos_embed 和 Embedding 是什么关系？

它们是**两个独立的信息源，相加合并**：

```python
x = self.embed(token_ids)    # [B, 14, 258]  ← "我是什么 token"
x = x + self.pos_embed       # [B, 14, 258]  ← "我在序列的哪个位置"
```

- `embed(token_id)` 编码**语义**：token_id=65 代表 steer=-0.35 这个"含义"
- `pos_embed(position)` 编码**位置**：第 2 个位置 → 这里应该放 steer

**同一个 token_id 在不同位置，加的 pos_embed 不同，所以最终向量不同。**

### Q: BOS 的 token_id 和位置有区别吗？

| 概念 | 值 | 作用 |
|------|-----|------|
| BOS 的 token_id | 201 | 输入 Embedding 查表，得到"序列开始"的语义向量 |
| BOS 的位置 | 0 | 输入 pos_embed 查表，得到"第0个位置"的位置向量 |

两者相加后，Decoder 就知道"这是序列开头"。

### Q: 常见的 Embedding 方式有哪些？

| 方法 | 特点 |
|------|------|
| `nn.Embedding` | 查找表，全可学习，本项目使用 |
| One-Hot | 稀疏，维度 = 词汇量，不可学习 |
| Word2Vec / GloVe | 预训练词向量，NLP 常用 |
| 正弦位置编码 | 固定公式，不可学习，原版 Transformer 使用 |
| 可学习位置编码 | `nn.Parameter`，本项目的 pos_embed 使用 |

---

## 7. 相机与深度图

### Q: 项目用了几个相机？

4 路 RGB + 4 路 Depth（前/左/右/后），共 8 个相机传感器。
每个 RGB 相机有一个对应的 Depth 相机，安装位置和参数完全一致。

### Q: 深度图是怎么获取的？

在 CARLA 仿真器中，深度图是**直接由引擎提供的 Ground Truth**——UE4 渲染时本身就有 Z-Buffer（深度缓冲），CARLA 的 `sensor.camera.depth` 直接导出这个数据。

这是仿真的优势：真实世界需要 LiDAR、双目相机、或深度估计网络才能获取深度。

### Q: 相机内参矩阵为什么是 3×3？

内参矩阵（Intrinsic Matrix）描述**相机内部光学参数**：

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- $f_x, f_y$：焦距（像素单位）
- $c_x, c_y$：主点（图像中心）
- 3×3 是因为它映射 3D 点到 2D 像素：$\begin{bmatrix}u\\v\\1\end{bmatrix} = K \cdot \begin{bmatrix}X\\Y\\Z\end{bmatrix}$

### Q: 外参矩阵为什么是 4×4？

外参矩阵（Extrinsic Matrix）描述**相机在世界中的位姿**（旋转 + 平移）：

$$T = \begin{bmatrix} R_{3\times3} & t_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix}$$

4×4 是因为使用了**齐次坐标**，可以用一次矩阵乘法同时完成旋转和平移。

### Q: 齐次坐标为什么第 4 维是 1？

3D 点 $(X, Y, Z)$ → 齐次坐标 $(X, Y, Z, 1)$

第 4 维 = 1 是一个**数学约定**，使得平移可以用矩阵乘法表示：

$$\begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} RX + t \\ 1 \end{bmatrix}$$

如果第 4 维不是 1，平移项 $t$ 就无法正确作用。这是计算机图形学和机器人学的通用标准。

### Q: 这些矩阵维度是通用标准吗？

是的，**全行业通用**：
- 内参 3×3：OpenCV、MATLAB、所有视觉库都用
- 外参 4×4：机器人学（变换矩阵）、图形学（Model-View 矩阵）、SLAM 都用
- 不是 CARLA 独有的

---

## 8. LSS 算法（Lift-Splat-Shoot）

### Q: LSS 算法是什么？

LSS（Lift-Splat-Shoot）是将多相机 2D 图像特征**投影到 BEV（鸟瞰图）** 的算法，分三步：

### Step 1: Lift（提升）

为每个像素创建一个**深度假设分布**：
```
图像特征 [H, W, C] → [H, W, D, C]
```
- 在每个像素的射线方向上，假设 D 个可能的深度值
- 每个深度值对应 3D 空间中的一个点
- 用网络预测每个深度值的概率权重

### Step 2: Splat（铺展/溅射）

将 3D 点"投射"到 BEV 网格上：
```
所有相机的 3D 点云 → 落入 BEV 网格的各个格子中 → 同一格子内的特征求和
```
- 用相机外参矩阵将相机坐标系的点转换到世界坐标系
- 再投影到 BEV 平面（x-y 平面）
- `VoxelsSumming` 实现：同一个 BEV 格子内所有点的特征相加

### Step 3: Shoot（执行）

在 BEV 特征图上做下游任务：
- 本项目：BEV 语义分割 + 特征压缩 → Transformer → 控制预测

### Q: BEV 网格的分辨率？

```yaml
bev_x_bound: [-10, 10, 0.1]   # x 方向：-10m 到 +10m，步长 0.1m → 200 格
bev_y_bound: [-10, 10, 0.1]   # y 方向：同上 → 200 格
bev_z_bound: [-10, 10, 20.0]  # z 方向：整个高度范围合为 1 层
```

BEV 特征图大小：**200 × 200**，每个格子代表 0.1m × 0.1m 的真实世界区域。

---

## 9. BEV 分割

### Q: BEV 分割的 3 个类别是什么？

| 类别 ID | 含义 | 说明 |
|---------|------|------|
| 0 | 背景 | 道路、空地等 |
| 1 | 车辆 | 停车场中的其他车辆 |
| 2 | 目标车位 | 需要停入的空车位 |

这是一个**辅助任务**——训练时帮助 BEV 特征学习到场景的空间结构，推理时可以不用分割头的输出。

---

## 10. 训练机制

### Q: batch_size=12 意味着什么？

每次前向传播同时处理 12 个样本。每个样本是一个**时刻的快照**，包含：
- 4 路 RGB 图像（前/左/右/后）
- 4 路深度图
- 目标车位坐标（target_point）
- 未来 4 帧的控制量（gt_control，15 个 token）
- BEV 分割标签

### Q: epoch 是什么？

一个 epoch = 遍历整个训练集一次。假设有 1200 个样本，batch_size=12：
- 每个 epoch = 1200 / 12 = **100 次迭代**
- 总训练 155 个 epoch = 15500 次迭代

### Q: Teacher Forcing 是什么？

训练时 Decoder 的输入用 **Ground Truth**（正确答案），而不是模型自己上一步的预测：
- **训练**：输入 gt_control 的前 14 个 token，预测后 14 个 token（与 GT 比较计算 loss）
- **推理**：输入 BOS，预测第 1 个 token → 把预测结果输入，预测第 2 个 token → ...自回归生成

---

## 11. 数据采集细节（DataGenerator）

### Q: `_parking_goal_index = 17` 是什么？

停车场布局定义在 `parking_position.py` 中，共 4 排 × 16 个车位 = 64 个位置。
索引 17 = Row 2 起始（16）+ 1 = **2-2 号车位**。

采集时**奇数号**车位放 NPC 静态车辆，**偶数号**车位留空作为目标：
```
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ 2-1  │ │ 2-2  │ │ 2-3  │ │ 2-4  │ ...
│ NPC车│ │🅿目标│ │ NPC车│ │🅿目标│
└──────┘ └──────┘ └──────┘ └──────┘
```

每完成一个任务，`_parking_goal_index += 2`，切换到下一个空车位。

### Q: `_save_frequency = 3` 的含义？

仿真 30 Hz，每 3 帧保存一次 → **保存频率 10 Hz**（每 0.1 秒一帧）。
目的：降低数据冗余，相邻帧变化极小，10 Hz 足够。

### Q: `_num_frames_goal_needed = 2 * 30 = 60` 的作用？

车辆必须**连续 60 帧**（2 秒）都满足停车条件（距离 < 0.5m 且角度 < 0.5°），才判定任务完成。
防止车辆只是**路过**目标车位时被误判为停好。中间任何一帧不满足条件，计数器归零重新计时。

---

## 12. FeatureFusion 与 Motion 编码

### Q: ego_motion 是什么？有几个维度？

ego_motion 是自车当前的运动状态，包含 3 个物理量：

| 分量 | 含义 | 来源 |
|------|------|------|
| speed | 车速（km/h） | CARLA IMU |
| acc_x | 纵向加速度 | CARLA IMU |
| acc_y | 横向加速度 | CARLA IMU |

在 `carla_dataset.py` 中组装为 `[1, 3]` 张量输入模型。

### Q: 3 维 ego_motion 怎么升维到 256 维？

通过 `feature_fusion.py` 中的 MLP（多层感知机）：

```
输入: [B, 1, 3]        ← 3 维运动特征
  ↓ Linear(3 → 64)     ← 第1次升维
  ↓ ReLU
  ↓ Linear(64 → 128)   ← 第2次升维
  ↓ ReLU
  ↓ Linear(128 → 256)  ← 第3次升维
  ↓ ReLU
输出: [B, 1, 256]       ← 256 维特征
```

`motion_encoder` 这一小段的参数量：

```text
只算 weight:
3×64 + 64×128 + 128×256
= 192 + 8192 + 32768
= 41,152

加上 bias:
64 + 128 + 256 = 448
41,152 + 448 = 41,600
```

注意：这里的 **41,600** 只是 `FeatureFusion.motion_encoder` 这个小 MLP 的参数量，不是整个模型的总参数量。整个 `ParkingModel` 约 **28M parameters**。

### Q: 为什么需要 ReLU（非线性激活）？

如果没有 ReLU，多层 Linear 会退化为一个等效的 Linear：

$$W_3 \cdot (W_2 \cdot (W_1 \cdot x)) = (W_3 W_2 W_1) \cdot x = W_{equiv} \cdot x$$

三层线性变换等价于一层！加了 ReLU 后，每层之间引入非线性"折叠"，使得网络能学到**复杂的非线性映射**，而非简单的矩阵乘法。

### Q: 为什么 motion 特征要扩展成 2 个维度？

代码中 motion 经过 MLP 后被 `expand(-1, -1, 2)` 扩展为每个 BEV token 的 2 个附加特征维度：

```python
motion_feature = self.motion_encoder(ego_motion)                  # [B, 1, 256]
motion_feature = motion_feature.transpose(1, 2).expand(-1, -1, 2) # [B, 256, 2]
fuse_feature = torch.cat([bev_feature, motion_feature], dim=2)     # [B, 256, 258]
```

**根本原因：Transformer 多头注意力的整除约束**。

Transformer Encoder 的 `d_model` 必须能被 `nhead` 整除：

$$d_{model} \mod n_{head} = 0$$

本项目配置：
- BEV token 数量 = 256
- nhead = 6
- BEV 每个 token 原本是 256 维特征
- d_model = 256 + 额外 motion 特征维度数

需要让 $(256 + n) \mod 6 = 0$，可选值：

| motion 额外维度数 n | d_model = 256+n | 能否被 6 整除 |
|---|---|---|
| 0 | 256 | 256/6 = 42.67 ❌ |
| 1 | 257 | 257/6 = 42.83 ❌ |
| **2** | **258** | **258/6 = 43** ✅ |
| 3 | 259 | 259/6 = 43.17 ❌ |
| 4 | 260 | 260/6 = 43.33 ❌ |

**2 是满足整除条件的最小值**。所以最终每个 BEV token 从 256 维变成 258 维，其中最后 2 维携带自车运动信息。

### Q: 两个相同的 motion 特征维度不会冗余吗？

这里不是 2 个 motion token，而是每个空间 token 末尾的 2 个特征维度。虽然 `expand` 后这两个维度初始数值相同，但它们位于特征向量的不同维度上，后续的 Q/K/V 线性投影会给不同维度使用不同权重；再加上 `pos_embed` 也在每个特征维度上都有独立参数，因此 Transformer 可以学习到不同的使用方式。

可以把它理解为：作者用 2 个额外 feature slot 把运动状态塞进每个 BEV 空间 token 中，而不是在序列尾部追加新的 token。

---

## 13. BevEncoder 与 ResNet18

### Q: BEV 特征为什么是 256 维？

BevEncoder 使用 ResNet18 的前 3 个 stage，输出 `[B, 256, 256]`。这里有**两个 256**，来源不同：

**第一个 256（dim=1）= 特征维度**：

ResNet18 各 layer 的通道数是固定的：

| Layer | 输出通道 |
|-------|---------|
| layer1 | 64 |
| layer2 | 128 |
| layer3 | **256** ← 用到这里 |
| layer4 | 512（不使用） |

所以特征维度 256 是 **ResNet18 架构决定的**。

**第二个 256（dim=2）= 序列长度**：

BEV 特征图 200×200 先被 `interpolate` 到 256×256，然后经过 4 次 stride=2 下采样：

$$256 \div 2^4 = 16, \quad 16 \times 16 = 256$$

所以 256 个空间位置展平为 256 个 token。

### Q: 两个 256 恰好相等是巧合吗？

**不是巧合**，是刻意设计。作者选择 `interpolate(size=(256,256))` 正是为了让序列长度也等于 256，使输出成为 `[B, 256, 256]` 的对称方阵，简化后续处理。

如果不做 interpolate（直接 200×200 输入），下采样后得到 12×12=144，输出就是 `[B, 256, 144]`，需要额外处理维度不对齐的问题。

### Q: 为什么不用 layer4（512 通道）？

| 方案 | 输出形状 | flatten 后 | 问题 |
|------|----------|-----------|------|
| 用到 layer3 | [256, 16, 16] | [256, 256] | ✅ 256 个 token，空间信息充足 |
| 用到 layer4 | [512, 8, 8] | [512, 64] | ❌ 只有 64 个 token，空间信息损失严重 |

8×8 的分辨率对于停车场景来说太粗糙了（相当于每个 token 覆盖 ~2.5m×2.5m），无法区分相邻车位。layer3 的 16×16（每个 token 覆盖 ~1.25m×1.25m）是更好的平衡。

> 详细的 ResNet18 架构解析见 [RESNET18_GUIDE.md](RESNET18_GUIDE.md)。

---

## 附录：关键配置速查

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `token_nums` | 204 | 控制 token 词汇表大小 |
| `half_token` | 100 | 有效控制值的一半（[-1,1] 映射到 [0,199]） |
| `tf_de_dim` | 258 | Transformer 隐藏维度 |
| `tf_de_tgt_dim` | 15 | 控制序列长度 |
| `tf_de_heads` | 6 | 注意力头数 |
| `tf_de_layers` | 4 | Transformer 层数 |
| `batch_size` | 12 | 训练批大小 |
| `future_frame_nums` | 4 | 预测未来帧数 |
| `image_crop` | 256 | 图像裁剪尺寸 |
| `seg_classes` | 3 | BEV 分割类别数 |
| `epochs` | 155 | 训练总轮数 |
| `bev_x/y_bound` | [-10, 10, 0.1] | BEV 网格范围和分辨率 |
| `bev_encoder_in_channel` | 64 | BEV 编码器输入通道 |
| `bev_encoder_out_channel` | 258 | BEV 编码器输出通道（256+2） |
| `backbone` | efficientnet-b4 | 图像编码器骨干网络 |
| `tf_en_dim` | 258 | Transformer Encoder 的 d_model |
| `tf_en_heads` | 6 | Transformer Encoder 注意力头数 |
| `tf_en_layers` | 4 | Transformer Encoder 层数 |
| `tf_en_bev_length` | 256 | BEV token 序列长度（16×16） |
| `tf_en_motion_length` | 3 | 运动特征维度（speed, acc_x, acc_y） |
| `weight_decay` | 0.0001 | Adam 优化器 L2 正则化系数 |
| 仿真帧率 | 30 Hz | `fixed_delta_seconds = 1/30` |
| 数据保存频率 | 10 Hz | 每 3 帧保存一次 |
