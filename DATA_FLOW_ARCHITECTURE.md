# 项目数据处理流与模型架构总览

本文按代码执行顺序梳理整个项目的数据流：从 CARLA 采集原始相机图像，到训练时读取样本、深度监督、LSS/BEV 投射、ResNet BEV Encoder、Transformer Encoder/Decoder、模仿学习真值处理，以及最终输出形状。

默认配置来自 `config/training.yaml`：

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `batch_size` | 12 | 训练 batch 大小 |
| `image_crop` | 256 | 相机图像中心裁剪尺寸 |
| `future_frame_nums` | 4 | 每个样本监督未来 4 帧控制 |
| `token_nums` | 204 | 控制 token 词表大小 |
| `bev_encoder_in_channel` | 64 | LSS 输出 BEV 特征通道数 |
| `bev_x_bound` | `[-10, 10, 0.1]` | BEV 前后范围和分辨率 |
| `bev_y_bound` | `[-10, 10, 0.1]` | BEV 左右范围和分辨率 |
| `bev_z_bound` | `[-10, 10, 20]` | BEV 高度范围，仅 1 个 z bin |
| `d_bound` | `[0.5, 12.5, 0.25]` | 深度范围，48 个 depth bin |
| `final_dim` | `[256, 256]` | 模型输入相机图像尺寸 |
| `bev_down_sample` | 8 | 相机特征图相对输入图的下采样倍数 |
| `tf_en_bev_length` | 256 | Transformer Encoder 的 BEV token 数 |
| `tf_en_dim` | 258 | Transformer Encoder token 维度 |
| `tf_de_tgt_dim` | 15 | 控制序列最大长度 |
| `tf_de_dim` | 258 | Transformer Decoder token 维度 |

下文用 `B` 表示 batch size，默认训练时 `B=12`；用 `N=4` 表示 4 路相机。

## 1. 数据采集与落盘

入口主要在 `carla_data_gen.py -> data_generation/data_generator.py -> data_generation/world.py`。

CARLA 中安装 4 路 RGB 相机和 4 路 Depth 相机：

| 相机 | 位置和朝向 | 原始尺寸 | FOV |
|---|---|---:|---:|
| `front` | `x=1.5, y=0, z=1.5, yaw=0` | 400x300 | 100 |
| `left` | `x=0, y=-0.8, z=1.5, yaw=-90, pitch=-40` | 400x300 | 100 |
| `right` | `x=0, y=0.8, z=1.5, yaw=90, pitch=-40` | 400x300 | 100 |
| `rear` | `x=-2.2, y=0, z=1.5, yaw=180, pitch=-30` | 400x300 | 100 |

采集阶段每 3 个仿真 step 保存一次数据。一个 task 目录大致包含：

```text
taskN/
├── rgb_front/0000.png
├── rgb_left/0000.png
├── rgb_right/0000.png
├── rgb_rear/0000.png
├── depth_front/0000.png
├── depth_left/0000.png
├── depth_right/0000.png
├── depth_rear/0000.png
├── measurements/0000.json
├── parking_goal/0001.json
└── topdown/encoded_0000.png
```

`measurements/*.json` 保存当前帧自车状态和专家控制：

```text
位置姿态: x, y, z, pitch, yaw, roll
运动量: speed, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
控制量: Throttle, Brake, Steer, Reverse
```

`parking_goal/0001.json` 保存目标车位世界坐标：

```text
{x, y, yaw}
```

`topdown/encoded_*.png` 是数据采集时渲染出的 BEV 语义图。采集阶段先渲染一个 15 通道的局部 BEV：

```text
[1, 15, 500, 500]
```

分辨率是 `5 px/m`，也就是 `1 px = 0.2 m`，局部范围约 `100m x 100m`。实际主要使用：

```text
ch0: road
ch1: lane
ch5: vehicle
```

保存时用 `data_generation/tools.py::encode_npy_to_pil()` 把最多 15 个二值通道压缩到 RGB 三个 uint8 通道里。

## 2. Dataset 读取一个训练样本

训练读取在 `dataset/carla_dataset.py::CarlaDataset`。

一个样本使用当前帧的传感器输入，同时使用未来 `future_frame_nums=4` 帧的专家控制作为模仿学习真值。有效帧范围是：

```text
frame in [hist_frame_nums, total_frames - future_frame_nums)
```

注意：`hist_frame_nums=10` 在这里主要用于跳过前 10 帧，当前模型并没有把历史图像序列作为输入。

`__getitem__()` 返回的主要字段如下：

| key | 单样本形状 | batch 后形状 | 含义 |
|---|---:|---:|---|
| `image` | `[4, 3, 256, 256]` | `[B, 4, 3, 256, 256]` | 4 路 RGB 图像 |
| `depth` | `[4, 256, 256]` | `[B, 4, 256, 256]` | 4 路米制深度图 |
| `intrinsics` | `[4, 3, 3]` | `[B, 4, 3, 3]` | 裁剪后的相机内参 |
| `extrinsics` | `[4, 4, 4]` | `[B, 4, 4, 4]` | 相机外参，veh->cam/pixel 坐标系 |
| `segmentation` | `[1, 200, 200]` | `[B, 1, 200, 200]` | BEV 分割 GT，3 类 |
| `target_point` | `[3]` | `[B, 3]` | 目标车位自车系坐标 `[x, y, yaw_diff]` |
| `ego_motion` | `[1, 3]` | `[B, 1, 3]` | `[speed, acc_x, acc_y]` |
| `gt_control` | `[15]` | `[B, 15]` | token 化控制序列 |
| `gt_acc` | `[4]` | `[B, 4]` | 未来 4 帧原始油门/刹车合并值 |
| `gt_steer` | `[4]` | `[B, 4]` | 未来 4 帧原始方向盘 |
| `gt_reverse` | `[4]` | `[B, 4]` | 未来 4 帧原始倒档标记 |

代码注释里有些地方把 depth 写成 `[4, 1, 256, 256]`，但按 `get_depth()` 和 `torch.cat(dim=0)` 的实际行为，单样本实际是 `[4, 256, 256]`。

## 3. RGB 图像处理

对应 `ProcessImage`：

1. 读取 CARLA 保存的 RGB PNG，原始尺寸 `400x300`。
2. 中心裁剪为 `256x256`。
3. 转成 tensor，通道顺序为 `[3, 256, 256]`。
4. 使用 ImageNet 均值方差归一化：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

裁剪偏移：

```text
原始宽 400，高 300
left_crop = (400 - 256) / 2 = 72
top_crop  = (300 - 256) / 2 = 22
```

4 路图像分别处理后拼接：

```text
front/left/right/rear 每路: [1, 3, 256, 256]
cat 后单样本 image:      [4, 3, 256, 256]
batch 后:               [B, 4, 3, 256, 256]
```

## 4. 相机内参与外参

原始相机内参按 `width=400, height=300, fov=100` 计算：

```text
fx = fy = width / (2 * tan(fov / 2))
cx = width / 2 = 200
cy = height / 2 = 150
```

裁剪后主点偏移：

```text
cx = 200 - 72 = 128
cy = 150 - 22 = 128
```

所以裁剪后图像中心正好对应主点 `(128, 128)`。

外参来自每个相机相对自车的安装位姿。代码先得到 `cam2veh`，再取逆得到 `veh2cam`，并左乘一个 CARLA 相机坐标到像素相机坐标的转换矩阵：

```text
cam2pixel =
[[0,  1,  0, 0],
 [0,  0, -1, 0],
 [1,  0,  0, 0],
 [0,  0,  0, 1]]
```

训练时传入：

```text
intrinsics: [B, 4, 3, 3]
extrinsics: [B, 4, 4, 4]
```

## 5. 深度图处理与深度监督

CARLA depth PNG 把深度编码在 RGB 三个通道里。`get_depth()` 中的转换公式是：

```text
normalized = (R + G * 256 + B * 65536) / (256^3 - 1)
depth_m    = 1000 * normalized
```

处理流程：

```text
depth PNG 400x300
-> 中心裁剪 256x256
-> 转成米制深度
-> 单路 [1, 256, 256]
-> 4 路 cat 后 [4, 256, 256]
-> batch 后 [B, 4, 256, 256]
```

深度监督在 `loss/depth_loss.py::DepthLoss`：

1. 将 GT depth 按 `bev_down_sample=8` 划成 `8x8` patch。
2. 每个 patch 取最小非零深度，得到低分辨率深度图：

```text
[B, 4, 256, 256] -> [B*4, 32, 32]
```

3. 按 `d_bound=[0.5, 12.5, 0.25]` 离散成 48 个深度 bin：

```text
(12.5 - 0.5) / 0.25 = 48
```

4. 转成 one-hot：

```text
depth label: [B*4*32*32, 48]
```

模型预测的深度分布来自 `CamEncoder`：

```text
pred_depth: [B*4, 48, 32, 32]
flatten 后: [B*4*32*32, 48]
```

loss 用 BCE 对齐预测深度分布和 one-hot 深度标签。

## 6. BEV 分割真值处理

对应 `ProcessSemantic`。

采集阶段保存的 topdown 是编码 RGB 图，训练时：

1. 读取 `topdown/encoded_*.png`。
2. 转灰度图。
3. 调用 `scale_and_crop_image(image, scale=0.5, crop=200)`。
4. 根据目标车位自车系坐标，在图上绘制目标车位区域，像素值设为 `255`。
5. 转成 3 类语义图：

```text
0: background
1: vehicle，来自 pixel == 75
2: target slot，来自 pixel == 255
```

按 `scale_and_crop_image()` 的实际代码，`scale=0.5` 会把 `500x500` 放大到 `1000x1000`，再中心裁剪 `200x200`。这会把采集阶段 `0.2m/px` 的 BEV 变成训练分割图中的 `0.1m/px`，刚好对应模型的 BEV 范围：

```text
x: -10m 到 10m，步长 0.1m -> 200
y: -10m 到 10m，步长 0.1m -> 200
```

最终：

```text
segmentation: [B, 1, 200, 200]
```

## 7. 控制真值与模仿学习

这个项目的控制学习是模仿学习/行为克隆：模型观察当前帧传感器输入，学习预测未来专家控制。

每个样本读取未来 4 帧控制量：

```text
frame+1, frame+2, frame+3, frame+4
```

每一帧控制量包括：

```text
Throttle: [0, 1]
Brake:    [0, 1]
Steer:    [-1, 1]
Reverse:  {0, 1}
```

`tokenize()` 把一帧控制压成 3 个 token：

```text
[throttle_brake_token, steer_token, reverse_token]
```

其中 `token_nums=204`：

```text
valid_token = token_nums - 4 = 200
half_token  = 100
BOS = 201
EOS = 202
PAD = 203
```

4 帧控制一共 `4 * 3 = 12` 个控制 token。再加特殊 token：

```text
gt_control = [BOS, 12 个控制 token, EOS, PAD]
```

所以：

```text
gt_control: [15]
batch 后:  [B, 15]
```

训练时 `ControlPredict.forward()` 使用 teacher forcing：

```text
decoder 输入: gt_control[:, :-1] -> [B, 14]
训练目标:     gt_control[:, 1:]  -> [B, 14]
```

也就是让 decoder 在每个位置预测下一个 token。

## 8. ParkingModel 总体前向

核心入口在 `model/parking_model.py`。

训练前向：

```text
data dict
-> ParkingModel.encoder(data)
-> ControlPredict(fuse_feature, gt_control)
-> pred_control, pred_segmentation, pred_depth
```

输入字段核心形状：

```text
image:        [B, 4, 3, 256, 256]
intrinsics:   [B, 4, 3, 3]
extrinsics:   [B, 4, 4, 4]
target_point: [B, 3]
ego_motion:   [B, 1, 3]
gt_control:   [B, 15]
```

训练输出：

```text
pred_control:      [B, 14, 204]
pred_segmentation: [B, 3, 200, 200]
pred_depth:        [B*4, 48, 32, 32]
```

## 9. CamEncoder：图像特征与深度分布

对应 `model/cam_encoder.py`。

输入 4 路图像先合并 batch 和 camera 维：

```text
images: [B, 4, 3, 256, 256]
view -> [B*4, 3, 256, 256]
```

`CamEncoder` 使用 `efficientnet-b4`，并截断到 `bev_down_sample=8` 对应的特征层。输出两个分支：

```text
feature: [B*4, 64, 32, 32]
depth:   [B*4, 48, 32, 32]
```

这里 `32 = 256 / 8`。`48` 是深度 bin 数。

深度和图像特征绑定的方式在 `BevModel.encoder_forward()`：

```text
depth_prob = softmax(depth, dim=1)

feature.unsqueeze(2):    [B*4, 64, 1,  32, 32]
depth_prob.unsqueeze(1): [B*4, 1,  48, 32, 32]
相乘后:                  [B*4, 64, 48, 32, 32]
```

含义是：每个像素位置的 64 维图像特征，按 48 个深度概率分配到不同深度层上。

随后 reshape 和 permute：

```text
[B*4, 64, 48, 32, 32]
-> [B, 4, 64, 48, 32, 32]
-> [B, 4, 48, 32, 32, 64]
```

最后这个张量可以理解为：

```text
B 个样本
4 路相机
48 个深度层
32x32 个图像特征位置
每个点 64 维特征
```

## 10. LSS：从透视图到 BEV

对应 `model/bev_model.py`。它实现的是 Lift-Splat-Shoot 思路。

### 10.1 构建 frustum

`create_frustum()` 根据相机特征图大小和深度范围构建视锥点：

```text
final_dim = [256, 256]
bev_down_sample = 8
feature map = 32x32
depth bins = 48

frustum: [48, 32, 32, 3]
```

最后一维 3 表示：

```text
[pixel_x, pixel_y, depth]
```

### 10.2 几何投影到自车坐标系

`get_geometry(intrinsics, extrinsics)` 使用相机内外参，把每个 frustum 点从图像/相机坐标转换到自车坐标：

```text
geom: [B, 4, 48, 32, 32, 3]
```

最后一维是自车坐标系下的：

```text
[x, y, z]
```

### 10.3 Splat 到 BEV 网格

BEV 网格参数：

```text
x: [-10, 10), step=0.1 -> 200 cells
y: [-10, 10), step=0.1 -> 200 cells
z: [-10, 10), step=20  -> 1 cell
```

所以 `bev_dim` 是：

```text
[200, 200, 1]
```

对每个 batch 样本，所有相机、深度、像素点展平：

```text
4 * 48 * 32 * 32 = 196608 个 3D 点
```

每个点携带 64 维特征。代码把落在 BEV 范围内的点筛出来，根据 voxel index 排序，然后用 `VoxelsSumming` 把同一个 BEV cell 里的特征累加。

输出：

```text
bev_feature: [B, 64, 200, 200]
pred_depth:  [B*4, 48, 32, 32]
```

## 11. 目标车位热力图拼接

对应 `ParkingModel.add_target_bev()`。

模型从 `parking_goal/0001.json` 读到世界系目标车位，然后在 Dataset 中通过 `convert_slot_coord()` 转到当前自车系：

```text
target_point: [x, y, yaw_diff]
```

进入模型后，`add_target_bev()` 将 `[x, y]` 转成 BEV 像素坐标：

```text
x_pixel = h / 2 + target_x / 0.1
y_pixel = w / 2 + target_y / 0.1
```

然后额外生成一个目标车位热力图：

```text
bev_target: [B, 1, 200, 200]
```

目标点附近 `8x8` 区域置为 1，其余为 0。训练时还会加一个大约 `[-5, 5]` 像素的随机噪声，起到目标点扰动增强的作用。

拼接后：

```text
LSS BEV feature: [B, 64, 200, 200]
target heatmap:  [B, 1,  200, 200]
cat 后:          [B, 65, 200, 200]
```

这 65 个通道可以理解为：

```text
64 个网络学习出来的环境 BEV 特征
+ 1 个明确告诉模型目标车位在哪里的热力图
```

## 12. BevEncoder：ResNet18 压缩 BEV

对应 `model/bev_encoder.py`。

输入：

```text
[B, 65, 200, 200]
```

先插值到 ResNet 更方便处理的尺寸：

```text
[B, 65, 200, 200] -> [B, 65, 256, 256]
```

然后走 ResNet18 的前几层：

| 阶段 | 输出形状 |
|---|---:|
| input | `[B, 65, 256, 256]` |
| `conv1`, stride=2, 65->64 | `[B, 64, 128, 128]` |
| `maxpool`, stride=2 | `[B, 64, 64, 64]` |
| `layer1` | `[B, 64, 64, 64]` |
| `layer2`, stride=2 | `[B, 128, 32, 32]` |
| `layer3`, stride=2 | `[B, 256, 16, 16]` |
| `flatten(x, 2)` | `[B, 256, 256]` |

最后一个 `[B, 256, 256]` 的含义是：

```text
通道维: 256
空间 token 数: 16 * 16 = 256
```

注意：代码里注册了 `layer4`，但 `forward()` 没有使用。停在 `layer3` 是为了保留 `16x16=256` 个 BEV token，对应配置里的 `tf_en_bev_length=256`。

## 13. FeatureFusion：进入 Transformer Encoder

对应 `model/feature_fusion.py`。

`BevEncoder` 输出：

```text
bev_down_sample: [B, 256, 256]
```

进入 `FeatureFusion.forward()` 后先转置：

```text
[B, C=256, S=256] -> [B, S=256, C=256]
```

自车运动输入：

```text
ego_motion: [B, 1, 3]
```

MLP 将 `[speed, acc_x, acc_y]` 编码成长度为 256 的运动序列：

```text
[B, 1, 3]
-> [B, 1, 64]
-> [B, 1, 128]
-> [B, 1, 256]
-> transpose -> [B, 256, 1]
-> expand -> [B, 256, 2]
```

这里 `expand(-1, -1, 2)` 把运动信息复制成 2 个附加特征维度。

然后把 BEV token 特征和 motion 特征在最后一维拼接：

```text
BEV:    [B, 256, 256]
motion: [B, 256, 2]
cat:    [B, 256, 258]
```

加位置编码：

```text
pos_embed: [1, 256, 258]
```

PyTorch Transformer Encoder 需要 `[S, B, E]`：

```text
[B, 256, 258] -> [256, B, 258]
TransformerEncoder
-> [256, B, 258]
-> [B, 256, 258]
```

最终融合特征：

```text
fuse_feature: [B, 256, 258]
```

含义：

```text
256 个 BEV 空间 token
每个 token 258 维
其中 256 维来自 BEV 视觉特征，2 维来自自车运动特征
```

## 14. SegmentationHead：辅助 BEV 分割输出

对应 `model/segmentation_head.py`。

输入：

```text
fuse_feature: [B, 256, 258]
```

先还原成 2D BEV 特征图：

```text
transpose: [B, 258, 256]
reshape:   [B, 258, 16, 16]
```

然后逐步上采样：

| 阶段 | 输出形状 |
|---|---:|
| `c5_conv`, 258->64 | `[B, 64, 16, 16]` |
| upsample x2 | `[B, 64, 32, 32]` |
| upsample x2 | `[B, 64, 64, 64]` |
| upsample x2 | `[B, 64, 128, 128]` |
| interpolate 到 200x200 | `[B, 64, 200, 200]` |
| segmentation head | `[B, 3, 200, 200]` |

最终：

```text
pred_segmentation: [B, 3, 200, 200]
```

3 个通道对应：

```text
0: background
1: vehicle
2: target slot
```

训练时 `SegmentationLoss` 会把它和 GT：

```text
target segmentation: [B, 1, 200, 200]
```

做交叉熵。

## 15. ControlPredict：Transformer Decoder

对应 `model/control_predict.py`。

### 15.1 训练时

输入：

```text
encoder_out / memory: [B, 256, 258]
gt_control:           [B, 15]
```

`forward()` 先丢掉最后一个 token：

```text
tgt = gt_control[:, :-1] -> [B, 14]
```

embedding：

```text
[B, 14] -> [B, 14, 258]
```

加位置编码：

```text
pos_embed: [1, 14, 258]
```

送入 PyTorch Transformer Decoder 前转成 `[T, B, E]`：

```text
tgt_embedding: [B, 14, 258] -> [14, B, 258]
encoder_out:   [B, 256, 258] -> [256, B, 258]
```

Decoder 输出：

```text
[14, B, 258] -> [B, 14, 258]
```

线性分类到 204 个 token：

```text
pred_control: [B, 14, 204]
```

训练目标是：

```text
gt_control[:, 1:] -> [B, 14]
```

所以训练目标序列是：

```text
12 个未来控制 token + EOS + PAD
```

`ControlLoss` 用交叉熵，忽略 `PAD=203`。

### 15.2 推理时

推理入口在 `ParkingModel.predict()`，初始 decoder 输入只有：

```text
gt_control = [BOS]
shape: [B, 1]
```

然后自回归循环 3 次，每次预测下一个 token：

```text
for i in range(3):
    pred_token = ControlPredict.predict(fuse_feature, pred_multi_controls)
    pred_multi_controls = cat([pred_multi_controls, pred_token], dim=1)
```

最终得到：

```text
pred_multi_controls: [B, 4]
```

内容是：

```text
[BOS, throttle_brake_token, steer_token, reverse_token]
```

再由 `detokenize()` 转回 CARLA 控制：

```text
[throttle, brake, steer, reverse]
```

注意：训练时监督未来 4 帧、共 12 个控制 token；当前推理代码每次只生成 1 帧控制所需的 3 个 token，然后在 CARLA 中每 3 个仿真 step 重新推理一次。

## 16. Loss 汇总

训练 step 在 `trainer/pl_trainer.py::ParkingTrainingModule.training_step()`。

模型输出：

```text
pred_control:      [B, 14, 204]
pred_segmentation: [B, 3, 200, 200]
pred_depth:        [B*4, 48, 32, 32]
```

对应三个 loss：

| loss | 预测 | 真值 | 说明 |
|---|---:|---:|---|
| `ControlLoss` | `[B, 14, 204]` | `[B, 14]` | token 分类交叉熵，忽略 PAD |
| `SegmentationLoss` | `[B, 1, 3, 200, 200]` | `[B, 1, 200, 200]` | 3 类 BEV 分割交叉熵 |
| `DepthLoss` | `[B*4*32*32, 48]` | `[B*4*32*32, 48]` | 48 个 depth bin 的 BCE |

总训练损失：

```text
train_loss = control_loss + segmentation_loss + depth_loss
```

验证时除了分割和深度，还会把控制 token 反离散化，额外计算：

```text
acc_steer_val_loss: throttle/brake 与 steer 的 SmoothL1
reverse_val_loss:   reverse 二分类交叉熵
```

## 17. 在线推理数据流

入口是 `carla_parking_eva.py -> agent/parking_agent.py`。

`ParkingAgent.get_model_data()` 从实时 CARLA 传感器组装模型输入：

```text
image:        [1, 4, 3, 256, 256]
intrinsics:   [1, 4, 3, 3]
extrinsics:   [1, 4, 4, 4]
ego_motion:   [1, 1, 3]
target_point: [1, 3]
gt_control:   [1, 1]，只含 BOS
```

`ParkingModel.predict()` 输出：

```text
pred_multi_controls: [1, 4]
pred_segmentation:   [1, 3, 200, 200]
pred_depth:          [4, 48, 32, 32]
target_bev:          [1, 1, 200, 200]
```

然后：

```text
pred_multi_controls[0][1:]
-> detokenize()
-> throttle, brake, steer, reverse
-> carla.VehicleControl
-> player.apply_control()
```

在线评估还有一个小机制：`ParkingAgent.save_prev_target()` 会从上一帧预测的 `pred_segmentation` 中找出目标车位区域中心，并在下一帧覆盖 `target_point` 的 `[x, y]`，起到目标点平滑/自反馈作用。

## 18. 一条完整形状链路

下面是一条最核心的训练前向形状链：

```text
RGB images
[B, 4, 3, 256, 256]
    |
    | merge camera into batch
    v
[B*4, 3, 256, 256]
    |
    | CamEncoder EfficientNet-B4 + heads
    v
image feature: [B*4, 64, 32, 32]
depth logits:  [B*4, 48, 32, 32]
    |
    | depth softmax binds feature to depth bins
    v
[B, 4, 48, 32, 32, 64]
    |
    | geometry projection + voxel summing
    v
LSS BEV feature
[B, 64, 200, 200]
    |
    | add target slot heatmap
    v
[B, 65, 200, 200]
    |
    | BevEncoder / ResNet18 until layer3
    v
[B, 256, 256]
    |
    | transpose + concat ego motion
    v
[B, 256, 258]
    |
    | Transformer Encoder
    v
fuse_feature
[B, 256, 258]
    |                         |
    |                         | SegmentationHead
    |                         v
    |                  pred_segmentation
    |                  [B, 3, 200, 200]
    |
    | Transformer Decoder with teacher forcing
    v
pred_control
[B, 14, 204]
```

同时深度辅助输出：

```text
pred_depth: [B*4, 48, 32, 32]
```

## 19. 模型参数量与权重大小

按当前源码和默认配置估算，完整 `ParkingModel` 约：

```text
27,987,243 parameters
≈ 28.0M parameters
```

如果只保存 FP32 权重，每个参数占 4 bytes：

```text
27,987,243 * 4 = 111,948,972 bytes
≈ 112 MB
≈ 106.8 MiB
```

主要模块拆分如下：

| 模块 | 参数量约值 | 说明 |
|---|---:|---|
| `bev_model` | 4.64M | EfficientNet-B4 截断骨干 + 图像特征头 + 深度头 |
| `bev_encoder` | 11.37M | ResNet18 BEV 编码器；当前代码注册了 `layer4` |
| `feature_fusion` | 5.42M | motion MLP + Transformer Encoder |
| `control_predict` | 6.49M | token embedding + Transformer Decoder + 输出分类头 |
| `segmentation_head` | 0.07M | BEV 3 类分割头 |

这里容易混淆的是 `BevEncoder.layer4`：代码中它被注册进模块，但 `forward()` 实际没有调用。因此：

```text
checkpoint/state_dict 会包含 layer4 参数
实际前向计算图不会经过 layer4
```

如果只数真正参与前向的部分，模型参数量会更小，约 **19.6M**；但按“最终训练保存的模型参数”理解，应按包含已注册参数的 `state_dict` 计算，也就是约 **28.0M**。

另外，Lightning 的 `.ckpt` 通常不只保存模型权重，还会保存 optimizer、scheduler、epoch/global_step 等训练状态。尤其 Adam 会额外保存一阶/二阶动量，所以实际 `.ckpt` 文件可能明显大于单纯 FP32 权重的 112 MB。

## 20. 对应代码索引

| 内容 | 文件 |
|---|---|
| 数据采集调度、保存 task 数据 | `data_generation/data_generator.py` |
| CARLA 传感器配置、相机尺寸、外参 | `data_generation/world.py` |
| BEV 语义图渲染 | `data_generation/bev_render.py` |
| BEV 多通道压缩到 RGB | `data_generation/tools.py` |
| Dataset 读取、图像/深度/分割/控制真值处理 | `dataset/carla_dataset.py` |
| DataLoader | `dataset/dataloader.py` |
| LSS BEV 投射 | `model/bev_model.py` |
| 图像编码与深度预测 | `model/cam_encoder.py` |
| BEV ResNet18 编码 | `model/bev_encoder.py` |
| Transformer Encoder 融合 | `model/feature_fusion.py` |
| Transformer Decoder 控制预测 | `model/control_predict.py` |
| BEV 分割头 | `model/segmentation_head.py` |
| 训练 step 和 loss 组合 | `trainer/pl_trainer.py` |
| 控制 loss | `loss/control_loss.py` |
| 深度 loss | `loss/depth_loss.py` |
| 分割 loss | `loss/seg_loss.py` |
| 在线推理 agent | `agent/parking_agent.py` |

## 21. 几个容易混淆的点

1. `hist_frame_nums=10` 并不代表模型输入 10 帧历史图像；当前代码只是从第 10 帧后开始取样。
2. 训练时监督未来 4 帧控制，共 12 个控制 token；推理时每次只自回归生成 3 个 token，即当前要执行的一帧控制。
3. `BevEncoder` 注册了 ResNet18 的 `layer4`，但 `forward()` 实际只跑到 `layer3`，因此输出保留 `16x16=256` 个 BEV token。
4. `FeatureFusion` 中的 `258` 维来自 `256` 维 BEV 特征加 `2` 维运动特征。
5. `DepthLoss` 的 GT depth 实际 batch 形状是 `[B, 4, 256, 256]`，不是 `[B, 4, 1, 256, 256]`。
6. `ProcessSemantic(scale=0.5)` 按实际函数行为是放大 2 倍后裁剪，使分割 GT 与模型 BEV 的 `0.1m/px` 分辨率对齐。
