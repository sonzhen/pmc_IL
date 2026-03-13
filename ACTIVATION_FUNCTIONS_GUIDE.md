# ⚡ 激活函数详解：ReLU 及其变种

---

## 目录

1. [为什么需要激活函数？](#1-为什么需要激活函数)
2. [Sigmoid 和 Tanh（早期激活函数）](#2-sigmoid-和-tanh早期激活函数)
3. [ReLU（修正线性单元）](#3-relu修正线性单元)
4. [Leaky ReLU](#4-leaky-relu)
5. [PReLU（参数化 ReLU）](#5-prelu参数化-relu)
6. [ELU（指数线性单元）](#6-elu指数线性单元)
7. [SELU（自归一化 ELU）](#7-selu自归一化-elu)
8. [GELU（高斯误差线性单元）](#8-gelu高斯误差线性单元)
9. [Swish / SiLU](#9-swish--silu)
10. [Mish](#10-mish)
11. [对比总结](#11-对比总结)
12. [本项目中的使用](#12-本项目中的使用)

---

## 1. 为什么需要激活函数？

### 没有激活函数的网络 = 线性变换

假设一个 3 层网络，每层都是线性变换（矩阵乘法）：

$$y = W_3 \cdot (W_2 \cdot (W_1 \cdot x)) = (W_3 W_2 W_1) \cdot x = W_{equiv} \cdot x$$

无论堆叠多少层，最终都等价于**一个矩阵乘法**——网络只能学到线性关系（直线/超平面）。

### 激活函数引入非线性

在每层后面加激活函数 $\sigma$：

$$y = W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x))$$

由于 $\sigma$ 是非线性的，多层组合后网络可以逼近**任意复杂函数**（万能近似定理）。

### 对激活函数的要求

| 要求 | 原因 |
|------|------|
| **非线性** | 否则多层网络退化为单层 |
| **可导** | 反向传播需要计算梯度 |
| **计算高效** | 每个神经元每次前向/反向都要调用 |
| **不会导致梯度消失/爆炸** | 否则深层网络无法训练 |

---

## 2. Sigmoid 和 Tanh（早期激活函数）

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
输出范围: (0, 1)

     1.0 ─────────────────────────╭──────
                                ╱
     0.5 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─
                            ╱
     0.0 ──────────────────╯─────────────
         -6   -4   -2    0    2    4    6
```

**导数**：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$，最大值 0.25（在 $x=0$ 时）

**问题**：
- ❌ **梯度消失**：$|x|$ 较大时，导数趋近于 0，梯度几乎无法传播
- ❌ **非零中心**：输出恒正 $(0,1)$，导致梯度更新方向受限（zigzag 问题）
- ❌ **指数运算**：$e^{-x}$ 计算较慢

**现在的用途**：几乎不用于隐藏层，仅用于：
- 二分类输出层（概率输出）
- 注意力机制中的门控（如 LSTM、GRU 的门）

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

```
输出范围: (-1, 1)

     1.0 ─────────────────────────╭──────
                                ╱
     0.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─
                            ╱
    -1.0 ──────────────────╯─────────────
         -6   -4   -2    0    2    4    6
```

**导数**：$\tanh'(x) = 1 - \tanh^2(x)$，最大值 1（在 $x=0$ 时）

**相比 Sigmoid 的改进**：
- ✅ **零中心**：输出范围 $(-1, 1)$，解决了 zigzag 问题
- ❌ 仍然有梯度消失问题（$|x|$ 大时导数趋近于 0）

---

## 3. ReLU（修正线性单元）

> Rectified Linear Unit，2010 年由 Nair & Hinton 提出，**深度学习最重要的激活函数之一**

### 公式

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

```
输出范围: [0, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────────╱────────
    -1 ─
         -4   -2    0    2    4    6
```

### 导数

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

（$x=0$ 处不可导，实践中定义为 0 或 1）

### PyTorch 使用

```python
import torch.nn as nn

# 方式 1：作为模块
relu = nn.ReLU(inplace=True)  # inplace=True 节省内存
y = relu(x)

# 方式 2：函数式
import torch.nn.functional as F
y = F.relu(x)
```

### 优点

| 优点 | 说明 |
|------|------|
| ✅ **计算极快** | 只需比较和赋值，无指数运算 |
| ✅ **正区间梯度恒为 1** | 不会梯度消失（正区间内） |
| ✅ **稀疏激活** | 约 50% 神经元输出为 0，增强特征稀疏性 |
| ✅ **收敛速度快** | 比 Sigmoid/Tanh 快 6 倍（Krizhevsky 2012） |

### 缺点：Dead ReLU（死亡 ReLU）

```
问题：x < 0 时，ReLU(x) = 0，梯度也为 0

假设某个神经元的输入总是负数：
  → 输出恒为 0
  → 梯度恒为 0
  → 权重永远无法更新
  → 这个神经元"死了"，永远不会被激活
```

**触发条件**：
- 学习率过大 → 权重更新过猛 → 偏置变成很大的负数 → 输入恒负
- 不当的权重初始化

**经验**：训练后期约 10%~40% 的 ReLU 神经元可能死亡。

---

## 4. Leaky ReLU

> 2013 年由 Maas 提出，解决 Dead ReLU 问题

### 公式

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

其中 $\alpha$ 是一个小常数，通常 $\alpha = 0.01$。

```
输出范围: (-∞, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────────╱────────
   -0.1 ─ ╲（斜率很小 = 0.01）
         -4   -2    0    2    4    6
```

### 导数

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

### PyTorch 使用

```python
nn.LeakyReLU(negative_slope=0.01, inplace=True)
```

### 与 ReLU 对比

- ✅ 负区间有小梯度（$\alpha = 0.01$），**不会"死亡"**
- ❌ $\alpha$ 是超参数，需要手动选择
- 实践中效果**通常与 ReLU 差不多**，但在某些任务（如 GAN）中更稳定

---

## 5. PReLU（参数化 ReLU）

> Parametric ReLU，2015 年由何恺明提出（就是 ResNet 的作者）

### 公式

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ a \cdot x & \text{if } x \leq 0 \end{cases}$$

与 Leaky ReLU **形状完全一样**，唯一区别：$a$ 不是固定值，而是**可学习参数**。

### 导数

$$\text{PReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ a & \text{if } x \leq 0 \end{cases}$$

$$\frac{\partial \text{PReLU}}{\partial a} = \begin{cases} 0 & \text{if } x > 0 \\ x & \text{if } x \leq 0 \end{cases}$$

$a$ 通过反向传播学习，每个通道可以有自己的 $a$。

### PyTorch 使用

```python
nn.PReLU(num_parameters=1)     # 所有通道共享一个 a
nn.PReLU(num_parameters=64)    # 每个通道有自己的 a（64 通道）
```

### 特点

- ✅ $a$ 自适应学习，无需手动调参
- ✅ 论文显示在 ImageNet 上比 ReLU 提升 1.05%
- ❌ 增加了少量参数（每通道 1 个标量）
- 当 $a=0$ 时退化为 ReLU，$a=0.01$ 时退化为 Leaky ReLU

---

## 6. ELU（指数线性单元）

> Exponential Linear Unit，2015 年由 Clevert 提出

### 公式

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

通常 $\alpha = 1.0$。

```
输出范围: (-α, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────────╱────────
    -α ─  ╲____ （平滑趋近 -α）
         -4   -2    0    2    4    6
```

### 导数

$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

### PyTorch 使用

```python
nn.ELU(alpha=1.0, inplace=True)
```

### 对比 ReLU / Leaky ReLU

| 特性 | ReLU | Leaky ReLU | ELU |
|------|------|-----------|-----|
| 负区间输出 | 0 | $\alpha x$（线性） | $\alpha(e^x-1)$（指数曲线） |
| 输出均值 | > 0 | ≈ 0 | **更接近 0** |
| 0 处连续 | ✅ | ✅ | ✅ |
| 0 处可导 | ❌（不光滑） | ❌（不光滑） | **✅（光滑）** |
| 计算速度 | 最快 | 快 | 较慢（有 $e^x$） |

**核心优势**：
- ✅ **零中心输出**：负区间有负值输出，使得激活值均值接近 0，加速收敛
- ✅ **光滑曲线**：在 $x=0$ 处连续可导（不像 ReLU 有"折角"），有助于优化
- ✅ **负区间饱和**：$x \to -\infty$ 时输出趋近 $-\alpha$，对噪声有鲁棒性
- ❌ **有指数运算**：比 ReLU 慢

---

## 7. SELU（自归一化 ELU）

> Scaled ELU，2017 年由 Klambauer 提出

### 公式

$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

其中 $\lambda \approx 1.0507$，$\alpha \approx 1.6733$（这两个值是**精确推导**出来的，不是经验选择）。

### PyTorch 使用

```python
nn.SELU(inplace=True)
```

### 特殊之处

SELU 有一个数学证明的性质：**自归一化（Self-Normalizing）**。

如果网络只用全连接层 + SELU + Lecun 初始化，则各层的激活值会自动收敛到均值 0、方差 1，**不需要 BatchNorm**。

**限制**：
- 仅对全连接网络有理论保证
- CNN 和 RNN 上没有理论保证
- 实际效果通常不如 BN + ReLU

---

## 8. GELU（高斯误差线性单元）

> Gaussian Error Linear Unit，2016 年由 Hendrycks & Gimpel 提出
> **GPT、BERT、ViT 等 Transformer 模型的默认激活函数**

### 公式

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准正态分布的**累积分布函数（CDF）**。

**近似公式**（计算更快）：

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]$$

```
输出范围: (≈ -0.17, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────╲_╱──────────  ← x≈-0.17 处有极小值
    -0.2 ─
         -4   -2    0    2    4    6
```

### PyTorch 使用

```python
nn.GELU()
# 或
F.gelu(x)
```

### 直觉理解

GELU 可以理解为一种**"软门控"**：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

- $\Phi(x)$ 是一个 0 到 1 之间的值，表示"保留 $x$ 的概率"
- $x$ 很大（正）→ $\Phi(x) \approx 1$ → 几乎全部保留
- $x$ 很小（负）→ $\Phi(x) \approx 0$ → 几乎全部丢弃
- $x$ 在 0 附近 → 部分保留，平滑过渡

相比 ReLU 的"硬门控"（$x>0$ 全留，$x\leq0$ 全丢），GELU 是**概率性的软判断**。

### 为什么 Transformer 偏好 GELU？

- ✅ **处处光滑可导**：没有 ReLU 在 0 处的不可导点
- ✅ **非单调**：$x \approx -0.17$ 处有极小值，可以编码更复杂的模式
- ✅ **与 Dropout 的思想一致**：都是"以某种概率保留/丢弃"
- 经验上在 NLP 和 Vision Transformer 中表现优于 ReLU

---

## 9. Swish / SiLU

> 2017 年由 Google Brain 通过**自动搜索**发现（Ramachandran 等人）
> PyTorch 中称为 **SiLU**（Sigmoid Linear Unit）

### 公式

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

当 $\beta = 1$（最常用）：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```
输出范围: (≈ -0.278, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────╲_╱──────────  ← x≈-0.278 处有极小值
   -0.3 ─
         -4   -2    0    2    4    6
```

### 导数

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$

### PyTorch 使用

```python
nn.SiLU(inplace=True)
# 等价于
F.silu(x)
```

### 与 GELU 的关系

Swish 和 GELU 形状**极其相似**，可以看作同一思路的不同实现：

| 特性 | GELU | Swish/SiLU |
|------|------|-----------|
| 门控函数 | 正态 CDF: $\Phi(x)$ | Sigmoid: $\sigma(x)$ |
| 公式 | $x \cdot \Phi(x)$ | $x \cdot \sigma(x)$ |
| 极小值位置 | $x \approx -0.17$ | $x \approx -0.278$ |
| 计算速度 | 需要 erf 或近似 | 只需 sigmoid（略快） |
| 主要用户 | BERT, GPT, ViT | **EfficientNet**, YOLOv5 |

### 与 ReLU 的极端情况

- 当 $\beta \to \infty$：$\sigma(\beta x)$ 变成阶跃函数 → Swish 退化为 ReLU
- 当 $\beta = 0$：$\sigma(0) = 0.5$ → Swish 退化为 $0.5x$（线性函数）
- 当 $\beta = 1$：标准 SiLU

### 本项目中的 Swish

本项目的 CamEncoder 使用 **EfficientNet-B4** 作为骨干网络，EfficientNet 内部大量使用 Swish/SiLU 激活函数（替代 ReLU），这是 EfficientNet 的标准配置。

---

## 10. Mish

> 2019 年由 Diganta Misra 提出

### 公式

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

```
输出范围: (≈ -0.31, +∞)

     4 ─                              ╱
     3 ─                            ╱
     2 ─                          ╱
     1 ─                        ╱
     0 ─────────────────────╲_╱──────────
   -0.3 ─
         -4   -2    0    2    4    6
```

### PyTorch 使用

```python
nn.Mish(inplace=True)
# 或
F.mish(x)
```

### 特点

- 形状类似 Swish/GELU，但用 $\tanh(\text{softplus}(x))$ 替代 $\sigma(x)$
- ✅ 处处光滑、非单调、自正则化
- 用于 YOLOv4、YOLOv5 等目标检测网络
- 计算比 Swish 略慢（多了 softplus + tanh）

---

## 11. 对比总结

### 公式一览

| 激活函数 | 公式 | 正区间 | 负区间 |
|---------|------|--------|--------|
| **ReLU** | $\max(0, x)$ | $x$ | $0$ |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $x$ | $\alpha x$ |
| **PReLU** | $\max(ax, x)$ | $x$ | $ax$（$a$ 可学习） |
| **ELU** | — | $x$ | $\alpha(e^x - 1)$ |
| **GELU** | $x \cdot \Phi(x)$ | ≈ $x$ | ≈ $0$（软过渡） |
| **Swish/SiLU** | $x \cdot \sigma(x)$ | ≈ $x$ | 小负值（软过渡） |
| **Mish** | $x \cdot \tanh(\text{sp}(x))$ | ≈ $x$ | 小负值（软过渡） |

### 性能对比

| 激活函数 | 计算速度 | 梯度消失 | Dead 神经元 | 零中心 | 光滑 |
|---------|---------|---------|-----------|--------|------|
| Sigmoid | 慢 | ❌ 严重 | — | ❌ | ✅ |
| Tanh | 慢 | ❌ 有 | — | ✅ | ✅ |
| **ReLU** | **最快** | ✅ 正区间无 | ❌ 有 | ❌ | ❌ |
| Leaky ReLU | 快 | ✅ | ✅ 无 | ≈✅ | ❌ |
| PReLU | 快 | ✅ | ✅ 无 | ≈✅ | ❌ |
| ELU | 较慢 | ✅ | ✅ 无 | ✅ | ✅ |
| **GELU** | 中等 | ✅ | ✅ 无 | ≈✅ | **✅** |
| **Swish/SiLU** | 中等 | ✅ | ✅ 无 | ≈✅ | **✅** |
| Mish | 较慢 | ✅ | ✅ 无 | ≈✅ | ✅ |

### 选择指南

```
你在做什么？
  │
  ├─ CNN（图像分类/检测/分割）
  │    ├─ 标准网络（ResNet 等）→ ReLU  ← 简单高效，默认选择
  │    ├─ EfficientNet → Swish/SiLU  ← 架构自带
  │    └─ YOLO → Leaky ReLU 或 Mish
  │
  ├─ Transformer（NLP/ViT）
  │    └─ GELU  ← 几乎是标准配置
  │
  ├─ GAN（生成对抗网络）
  │    └─ Leaky ReLU  ← 判别器中防止死亡神经元
  │
  └─ 全连接网络
       └─ ReLU 或 ELU
```

### 发展时间线

```
2010  ReLU ──────── 深度学习复兴的基石
2013  Leaky ReLU ── 修复 Dead ReLU
2015  PReLU ─────── 让负斜率可学习（何恺明）
2015  ELU ─────────  指数平滑 + 零中心
2016  GELU ─────── Transformer 时代的选择
2017  SELU ─────── 自归一化（理论优美但实用受限）
2017  Swish/SiLU ── Google 自动搜索发现
2019  Mish ─────── 社区贡献，用于 YOLO
```

---

## 12. 本项目中的使用

### 使用位置

| 模块 | 激活函数 | 原因 |
|------|---------|------|
| **BevEncoder** (ResNet18) | **ReLU** | ResNet 标准配置 |
| **FeatureFusion** (motion MLP) | **ReLU** | 简单 MLP，ReLU 足够 |
| **CamEncoder** (EfficientNet-B4) | **Swish/SiLU** | EfficientNet 架构自带 |
| **SegmentationHead** | **ReLU** | 上采样卷积中使用 |
| **ControlPredict** (Transformer Decoder) | Transformer 内部默认 | 前馈网络中使用 |

### BevEncoder 中的 ReLU

```python
class BevEncoder(nn.Module):
    def __init__(self, in_channel):
        trunk = resnet18(pretrained=False)
        self.relu = trunk.relu           # ← ResNet18 自带的 ReLU

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)                # ← 头部使用 ReLU
        x = self.max_pool(x)
        # layer1~3 内部的 BasicBlock 也都使用 ReLU
```

### FeatureFusion 中的 ReLU

```python
self.motion_encoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(inplace=True),      # ← 3→64 后激活
    nn.Linear(64, 128),
    nn.ReLU(inplace=True),      # ← 64→128 后激活
    nn.Linear(128, 256),
    nn.ReLU(inplace=True),      # ← 128→256 后激活
)
```

这里用 ReLU 而不是 Swish/GELU 的原因：
- MLP 只有 3 层，简单任务不需要复杂激活函数
- ReLU 计算最快，参数为零
- 效果差异在小型 MLP 上可忽略

### EfficientNet 中的 Swish

EfficientNet 在每个 MBConv Block 内部使用 Swish：

```python
# EfficientNet 内部（简化）
class MBConvBlock:
    def forward(self, x):
        # 1×1 升维
        x = self.expand_conv(x)
        x = self.bn0(x)
        x = swish(x)              # ← Swish 激活

        # Depthwise 卷积
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = swish(x)              # ← Swish 激活

        # SE 注意力
        x = self.se(x)

        # 1×1 降维（无激活函数）
        x = self.project_conv(x)
        x = self.bn2(x)
        return x
```

Google 在设计 EfficientNet 时通过 NAS（神经架构搜索）发现 Swish 比 ReLU 在该架构上效果更好，这与 Swish 论文中"自动搜索发现"的起源一脉相承。
