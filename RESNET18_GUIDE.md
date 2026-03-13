# 🏗️ ResNet18 架构详解

---

## 目录

1. [背景：为什么需要 ResNet？](#1-背景为什么需要-resnet)
2. [核心思想：残差连接](#2-核心思想残差连接)
3. [BasicBlock 详解](#3-basicblock-详解)
4. [ResNet18 完整架构](#4-resnet18-完整架构)
5. [逐层维度追踪](#5-逐层维度追踪)
6. [ResNet 家族对比](#6-resnet-家族对比)
7. [本项目中的 ResNet18 使用](#7-本项目中的-resnet18-使用)
8. [关键概念补充](#8-关键概念补充)

---

## 1. 背景：为什么需要 ResNet？

### 深度网络的退化问题

2015 年之前，人们发现一个反直觉的现象：**网络越深，效果越差**。

```
20 层 CNN → 训练误差 5.0%
56 层 CNN → 训练误差 5.5%  ← 更深但更差！
```

这不是过拟合（过拟合是测试集差、训练集好），而是**训练集上就更差**，叫做**退化问题（Degradation Problem）**。

### 直觉理解

理论上，56 层网络至少不该比 20 层差——因为可以让多余的 36 层学成"恒等映射"（什么也不做，原样输出）。但实际中，**让网络学习恒等映射比学习其他变换更困难**。

### ResNet 的解决方案

何恺明（Kaiming He）等人在 2015 年提出 ResNet：**不让网络直接学输出，而是学"残差"**。

> 论文：*Deep Residual Learning for Image Recognition*（CVPR 2016 Best Paper）
> 作者：何恺明、张翔宇、任少卿、孙剑（微软亚洲研究院）

---

## 2. 核心思想：残差连接

### 普通网络 vs 残差网络

```
普通网络：                    残差网络：
输入 x                        输入 x ─────────────────┐
  ↓                             ↓                     │（shortcut / skip connection）
Conv → BN → ReLU              Conv → BN → ReLU        │
  ↓                             ↓                     │
Conv → BN                     Conv → BN               │
  ↓                             ↓                     │
ReLU                           + ←────────────────────┘
  ↓                             ↓
输出 H(x)                     ReLU
                                ↓
                              输出 H(x) = F(x) + x
```

### 数学表达

- **普通网络**：学习目标映射 $H(x)$
- **残差网络**：学习残差 $F(x) = H(x) - x$，最终输出 $H(x) = F(x) + x$

### 为什么残差更容易学？

如果理想映射是恒等变换（$H(x) = x$）：
- 普通网络要学 $H(x) = x$：需要权重精确配合
- 残差网络只需学 $F(x) = 0$：权重全部趋近于 0 即可（更容易！）

**核心直觉**：将"学习完整变换"简化为"学习微小修正"。

### 跳跃连接（Skip Connection）

那条从输入直接连到加法的线叫 **skip connection**（也叫 shortcut connection）：

```python
# 伪代码
out = conv2(relu(conv1(x)))   # F(x): 经过两层卷积
out = out + x                  # F(x) + x: 加上原始输入（跳跃连接）
out = relu(out)                # 最终激活
```

这条线让梯度可以**直接回传到浅层**（不经过中间的卷积层），大大缓解了**梯度消失**问题。

---

## 3. BasicBlock 详解

ResNet18 和 ResNet34 使用 **BasicBlock**（基础残差块），每块包含 2 个 3×3 卷积：

```
输入 x [C_in, H, W]
  │
  ├──────────────────────────────────────────┐
  ↓                                          │ shortcut
  Conv2d(3×3, stride, padding=1)             │
  ↓                                          │
  BatchNorm2d                                │
  ↓                                          │
  ReLU                                       │
  ↓                                          │
  Conv2d(3×3, stride=1, padding=1)           │
  ↓                                          │
  BatchNorm2d                                │
  ↓                                          │
  +  ←───────────────────────────────────────┘
  ↓
  ReLU
  ↓
  输出 [C_out, H', W']
```

### shortcut 的两种情况

**情况 1：维度不变**（C_in = C_out，stride=1）
```python
shortcut = x  # 直接传过去
```

**情况 2：维度改变**（C_in ≠ C_out 或 stride=2，需要下采样）
```python
shortcut = nn.Sequential(
    nn.Conv2d(C_in, C_out, kernel_size=1, stride=2),  # 1×1 卷积调通道+下采样
    nn.BatchNorm2d(C_out)
)
```

### PyTorch 源码中的 BasicBlock

```python
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道 = 输入通道 × expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # shortcut 上的下采样层（如果需要）

    def forward(self, x):
        identity = x                      # 保存输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:    # 如果维度不匹配
            identity = self.downsample(x)  # 用 1×1 卷积调整 shortcut

        out += identity                    # 残差连接: F(x) + x
        out = self.relu(out)

        return out
```

---

## 4. ResNet18 完整架构

### 架构总览

ResNet18 = **1 个头部** + **4 个 layer**（每个 layer 含 2 个 BasicBlock）+ **1 个分类头**

```
输入 [3, 224, 224]
  ↓
┌─────────────────── 头部（Stem）───────────────────┐
│  conv1:  Conv2d(3→64, 7×7, stride=2, pad=3)      │
│  bn1:    BatchNorm2d(64)                           │
│  relu:   ReLU                                      │
│  maxpool: MaxPool2d(3×3, stride=2, pad=1)          │
└────────────────────────────────────────────────────┘
  ↓ [64, 56, 56]
┌─────────── layer1: 2 × BasicBlock(64→64) ──────────┐
│  Block1: Conv(64→64) → BN → ReLU → Conv(64→64)    │
│  Block2: Conv(64→64) → BN → ReLU → Conv(64→64)    │
└─────────────────────────────────────────────────────┘
  ↓ [64, 56, 56]   ← 尺寸不变
┌─────────── layer2: 2 × BasicBlock(64→128) ─────────┐
│  Block1: Conv(64→128, stride=2) → ... → Conv(128)  │  ← stride=2 下采样
│  Block2: Conv(128→128) → ... → Conv(128)            │
└─────────────────────────────────────────────────────┘
  ↓ [128, 28, 28]
┌─────────── layer3: 2 × BasicBlock(128→256) ────────┐
│  Block1: Conv(128→256, stride=2) → ... → Conv(256) │  ← stride=2 下采样
│  Block2: Conv(256→256) → ... → Conv(256)            │
└─────────────────────────────────────────────────────┘
  ↓ [256, 14, 14]
┌─────────── layer4: 2 × BasicBlock(256→512) ────────┐
│  Block1: Conv(256→512, stride=2) → ... → Conv(512) │  ← stride=2 下采样
│  Block2: Conv(512→512) → ... → Conv(512)            │
└─────────────────────────────────────────────────────┘
  ↓ [512, 7, 7]
┌─────────────── 分类头（Head）──────────────────────┐
│  avgpool: AdaptiveAvgPool2d(1×1) → [512, 1, 1]    │
│  flatten: → [512]                                   │
│  fc:      Linear(512, 1000) → [1000]               │  ← ImageNet 1000 类
└─────────────────────────────────────────────────────┘
```

### 关键数字

| 属性 | 值 |
|------|-----|
| 总层数 | 18（含卷积层，不含 BN/ReLU/池化） |
| 参数量 | **11.7M**（1170 万） |
| BasicBlock 数量 | 2+2+2+2 = **8 个** |
| 卷积层数量 | 1(头部) + 2×8(8个Block) + 1(未使用的fc前无卷积) = **17 个卷积 + 1 个全连接 = 18** |

### "18" 的来源

```
头部 conv1:                1 层
layer1: 2 blocks × 2 conv = 4 层
layer2: 2 blocks × 2 conv = 4 层
layer3: 2 blocks × 2 conv = 4 层
layer4: 2 blocks × 2 conv = 4 层
全连接 fc:                 1 层
                    合计 = 18 层
```

---

## 5. 逐层维度追踪

以标准 ImageNet 输入 224×224 为例：

| 层 | 操作 | 输出形状 | 空间下采样 | 通道变化 |
|----|------|----------|-----------|---------|
| 输入 | — | [3, 224, 224] | — | — |
| conv1 | 7×7, s=2, p=3 | [64, 112, 112] | ÷2 | 3→64 |
| bn1+relu | — | [64, 112, 112] | — | — |
| maxpool | 3×3, s=2, p=1 | [64, 56, 56] | ÷2 | — |
| layer1.block1 | 3×3×2 | [64, 56, 56] | ×1 | 64→64 |
| layer1.block2 | 3×3×2 | [64, 56, 56] | ×1 | 64→64 |
| layer2.block1 | 3×3×2, s=2 | [**128**, 28, 28] | ÷2 | 64→128 |
| layer2.block2 | 3×3×2 | [128, 28, 28] | ×1 | 128→128 |
| layer3.block1 | 3×3×2, s=2 | [**256**, 14, 14] | ÷2 | 128→256 |
| layer3.block2 | 3×3×2 | [256, 14, 14] | ×1 | 256→256 |
| layer4.block1 | 3×3×2, s=2 | [**512**, 7, 7] | ÷2 | 256→512 |
| layer4.block2 | 3×3×2 | [512, 7, 7] | ×1 | 512→512 |
| avgpool | adaptive 1×1 | [512, 1, 1] | →1×1 | — |
| fc | linear | [1000] | — | 512→1000 |

**总下采样倍数**：$2 \times 2 \times 2 \times 2 \times 2 = 32$ 倍（224 → 7）

**通道递增规律**：每经过一个 layer（除 layer1），通道数翻倍：64 → 128 → 256 → 512

---

## 6. ResNet 家族对比

### 不同深度的 ResNet

| 模型 | 层数配置 [l1, l2, l3, l4] | Block 类型 | 参数量 | Top-1 准确率 |
|------|--------------------------|-----------|--------|-------------|
| **ResNet18** | [2, 2, 2, 2] | BasicBlock | 11.7M | 69.8% |
| ResNet34 | [3, 4, 6, 3] | BasicBlock | 21.8M | 73.3% |
| ResNet50 | [3, 4, 6, 3] | **Bottleneck** | 25.6M | 76.1% |
| ResNet101 | [3, 4, 23, 3] | Bottleneck | 44.5M | 77.4% |
| ResNet152 | [3, 8, 36, 3] | Bottleneck | 60.2M | 78.3% |

### BasicBlock vs Bottleneck

```
BasicBlock（ResNet18/34 使用）:        Bottleneck（ResNet50+ 使用）:
  x                                      x
  ↓                                      ↓
  Conv 3×3 (C → C)                       Conv 1×1 (C → C/4)     ← 降维
  BN → ReLU                              BN → ReLU
  Conv 3×3 (C → C)                       Conv 3×3 (C/4 → C/4)   ← 处理
  BN                                     BN → ReLU
  ↓                                      Conv 1×1 (C/4 → C)     ← 升维
  + x → ReLU                             BN
                                          ↓
  2 层卷积                                + x → ReLU
                                          3 层卷积（"瓶颈"结构）
```

Bottleneck 通过 1×1 卷积先降维再升维，减少 3×3 卷积的计算量，适合更深的网络。

---

## 7. 本项目中的 ResNet18 使用

### 7.1 BevEncoder：BEV 特征压缩

BevEncoder **借用** ResNet18 的前 3 个 stage（不用 layer4 和分类头）：

```python
class BevEncoder(nn.Module):
    def __init__(self, in_channel):  # in_channel = 64
        trunk = resnet18(pretrained=False)

        # 替换 conv1：输入通道从 3 改为 65（64 + 1 通道目标车位热力图）
        self.conv1 = nn.Conv2d(in_channel + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 复用 ResNet18 的 BN、ReLU、MaxPool
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.max_pool = trunk.maxpool

        # 复用前 3 个 layer（不用 layer4）
        self.layer1 = trunk.layer1   # 64 → 64
        self.layer2 = trunk.layer2   # 64 → 128
        self.layer3 = trunk.layer3   # 128 → 256
        # self.layer4 = trunk.layer4   ← 不用！256 → 512 太压缩了
```

**为什么不用 layer4？**
- layer4 输出 [512, 8, 8]，flatten 后只有 64 个 token，空间信息损失太大
- layer3 输出 [256, 16, 16]，flatten 后 256 个 token，保留更多空间细节

**逐层维度变化**（以 BEV 输入为例）：

| 层 | 输出形状 | 说明 |
|----|----------|------|
| 输入 | [B, 65, 200, 200] | 64ch BEV + 1ch 目标车位热力图 |
| interpolate | [B, 65, 256, 256] | 双线性插值到 256×256 |
| conv1 (7×7, s=2) | [B, 64, 128, 128] | 65→64 通道 |
| bn1 + relu + maxpool (s=2) | [B, 64, 64, 64] | |
| layer1 (2×BasicBlock) | [B, 64, 64, 64] | 通道不变 |
| layer2 (2×BasicBlock, s=2) | [B, 128, 32, 32] | 64→128 |
| layer3 (2×BasicBlock, s=2) | [B, **256**, 16, 16] | 128→**256** |
| flatten(dim=2) | [B, **256**, **256**] | 16×16=256 展平 |

**最终输出 `[B, 256, 256]`**：
- 第一个 256 = **特征维度**（ResNet18 layer3 固定输出 256 通道）
- 第二个 256 = **序列长度**（16×16=256 个空间位置展平为 token 序列）

### 7.2 为什么两个 256 恰好相等？

这**不是巧合**，是刻意设计：
- ResNet18 layer3 输出 256 通道 → 由 ResNet 架构决定，不可改
- 空间尺寸 16×16=256 → 由 `interpolate(size=(256, 256))` 刻意控制

作者选择 interpolate 到 256×256，**正是为了让 flatten 后序列长度也等于 256**：

$$256 \div 2^4 = 16, \quad 16 \times 16 = 256$$

如果不做 interpolate（直接用 200×200 输入），经过 4 次 stride=2 得到 12×12=144，序列长度就是 144 ≠ 256，和特征维度不对齐。

### 7.3 为什么用 `pretrained=False`？

```python
trunk = resnet18(pretrained=False, zero_init_residual=True)
```

- **pretrained=False**：不加载 ImageNet 预训练权重，因为输入不是 3 通道 RGB 而是 65 通道 BEV 特征图，预训练权重不适用
- **zero_init_residual=True**：将每个 BasicBlock 的最后一个 BN 层的 γ 初始化为 0，使得初始时 $F(x) = 0$，残差块退化为恒等映射，有助于训练稳定性

### 7.4 CamEncoder 中的 ResNet18？

CamEncoder **不使用 ResNet18**，而是使用 **EfficientNet-B4** 作为骨干网络。只有 BevEncoder 使用 ResNet18。

---

## 8. 关键概念补充

### 8.1 BatchNorm（批归一化）

```python
nn.BatchNorm2d(num_features)
```

对每个通道，在 batch 维度上做归一化：

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

- $\mu_B, \sigma_B$：当前 batch 的均值和标准差
- $\gamma, \beta$：可学习的缩放和偏移参数
- 作用：**加速训练收敛、允许更大学习率、轻微正则化效果**

### 8.2 为什么 Conv 后面总跟 BN？

```
Conv → BN → ReLU   （标准顺序，称为 "CBR"）
```

- Conv 输出的数值范围不固定，可能很大或很小
- BN 将数值归一化到均值 0、方差 1 附近
- ReLU 在 0 附近有最大梯度，归一化后训练更高效
- 使用 BN 后，Conv 可以省略 bias（`bias=False`），因为 BN 的 β 参数等效于 bias

### 8.3 stride=2 的下采样机制

```python
nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
```

- **stride=2**：卷积核每次移动 2 步（而非默认的 1 步）
- 效果：输出的空间尺寸减半（H÷2, W÷2）
- 替代方案：池化层（MaxPool）也可以下采样，但 stride=2 的卷积可以**同时学习特征和下采样**

### 8.4 7×7 大卷积核的作用

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
```

- 头部使用 7×7 大卷积核（而非 3×3）
- **感受野更大**：一次卷积就能看到 7×7 的区域，快速捕获低层纹理
- **配合 stride=2**：空间尺寸减半，同时 3→64 大幅增加通道数
- 后续层都用 3×3 小卷积核（计算效率更高）

### 8.5 AdaptiveAvgPool2d(1) 的作用

```python
nn.AdaptiveAvgPool2d((1, 1))  # 输出固定为 1×1
```

- 无论输入空间尺寸是多少（7×7、8×8、16×16...），都输出 1×1
- 对每个通道的所有空间位置取**全局平均**
- 作用：将空间信息压缩为一个标量，只保留通道维度的语义信息
- 本项目 BevEncoder **不用此层**（因为需要保留空间信息给 Transformer）

### 8.6 ResNet 为什么这么流行？

| 优势 | 说明 |
|------|------|
| **解决退化问题** | 残差连接使得 100+ 层网络可以正常训练 |
| **梯度流通** | skip connection 让梯度直接传回浅层，不会消失 |
| **即插即用** | 可以轻松替换 conv1 和 fc 层，适配不同任务 |
| **预训练权重** | ImageNet 预训练广泛可用，迁移学习方便 |
| **结构简洁** | 堆叠相同的 Block，代码实现清晰 |

ResNet 至今仍是计算机视觉的**默认骨干网络**之一，大量项目（包括本项目）将其作为特征提取器使用。

---

## 附录：ResNet18 参数量计算

| 层 | 参数量 | 计算 |
|----|--------|------|
| conv1 | 9,408 | 3×64×7×7 |
| layer1 | 147,968 | 2×(64×64×3×3×2) |
| layer2 | 525,568 | (64×128×3×3 + 128×128×3×3) + (128×128×3×3×2) + downsample |
| layer3 | 2,099,712 | 类似 layer2，通道翻倍 |
| layer4 | 8,394,752 | 类似 layer3，通道翻倍 |
| fc | 513,000 | 512×1000 + 1000 |
| BN 参数 | ~9,600 | 各层 γ, β |
| **总计** | **~11.7M** | |

> 本项目 BevEncoder 不用 layer4 和 fc，实际参数约 **2.8M**（conv1 改为 65→64 后略有变化）。
