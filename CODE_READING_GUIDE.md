# 📚 代码阅读路线图

已完成 `carla_data_gen.py`，按**调用链由浅入深**的原则，建议按以下顺序逐步推进：

---

## 🔵 第一阶段：数据采集流水线（跟着 `carla_data_gen.py` 的调用链走）

| 顺序 | 文件 | 原因 |
|------|------|------|
| ✅ 1 | `carla_data_gen.py` | **已完成** — 入口，`game_loop()` |
| **👉 2** | `data_generation/data_generator.py` | `game_loop()` 直接创建的 `DataGenerator`，是数据采集的核心调度器 |
| 3 | `data_generation/world.py` | `DataGenerator` 内部创建的 `World`，封装 CARLA 世界（车辆、传感器、天气） |
| 4 | `data_generation/sensors.py` | `World` 内部创建的 `CameraManager` + `CollisionSensor` |
| 5 | `data_generation/keyboard_control.py` | 键盘控制逻辑，相对独立，较简单 |
| 6 | `data_generation/hud.py` | HUD 显示，相对独立 |
| 7 | `data_generation/bev_render.py` | 俯视图渲染，相对独立 |
| 8 | `data_generation/parking_position.py` | 停车位坐标定义，纯数据文件 |
| 9 | `data_generation/tools.py` | 工具函数 |

---

## 🟢 第二阶段：数据集与训练入口（从磁盘数据 → 模型训练）

| 顺序 | 文件 | 原因 |
|------|------|------|
| 10 | `tool/config.py` | 配置类，后续所有模块都依赖它 |
| 11 | `dataset/carla_dataset.py` | 数据集类，理解训练数据如何从磁盘加载 & tokenize |
| 12 | `dataset/dataloader.py` | DataLoader 封装 |
| 13 | `pl_train.py` | 训练入口 |
| 14 | `trainer/pl_trainer.py` | PyTorch Lightning Module，`training_step()` 在这里 |

---

## 🔴 第三阶段：模型架构（自顶向下）

| 顺序 | 文件 | 原因 |
|------|------|------|
| 15 | `model/parking_model.py` | 顶层模型，`forward()` 串联所有子模块 |
| 16 | `model/cam_encoder.py` | EfficientNet-B4 图像编码器 |
| 17 | `model/bev_model.py` | LSS 算法核心（Lift-Splat-Shoot） |
| 18 | `model/bev_encoder.py` | BEV 特征压缩（ResNet18） |
| 19 | `model/feature_fusion.py` | Transformer Encoder 特征融合 |
| 20 | `model/control_predict.py` | Transformer Decoder 控制预测 |
| 21 | `model/segmentation_head.py` | 分割头 |
| 22 | `model/convolutions.py` | 卷积基础模块 |

---

## 🟡 第四阶段：损失函数与评估

| 顺序 | 文件 | 原因 |
|------|------|------|
| 23 | `loss/control_loss.py` | 控制 loss（CrossEntropy） |
| 24 | `loss/depth_loss.py` | 深度 loss |
| 25 | `loss/seg_loss.py` | 分割 loss |
| 26 | `carla_parking_eva.py` | 评估入口 |
| 27 | `data_generation/network_evaluator.py` | 在线评估逻辑 |
| 28 | `agent/parking_agent.py` | 推理时的 Agent |
| 29 | `tool/geometry.py` & `tool/metric.py` | 几何计算 & 评估指标 |

---

## 🎯 核心思路

```
采集数据（第一阶段）→ 构建数据集（第二阶段）→ 训练模型（第三阶段）→ 评估模型（第四阶段）
```

## 📝 关键配置速查

| 配置项 | 值 | 含义 |
|--------|-----|------|
| token_nums | 204 | 控制 token 词汇表大小（200有效 + BOS/EOS/PAD/保留） |
| tf_de_dim | 258 | Transformer 隐藏维度（256 ResNet + 2 motion） |
| tf_de_tgt_dim | 15 | 控制序列长度（BOS + 4帧×3token + EOS + PAD） |
| batch_size | 12 | 训练批大小 |
| future_frame_nums | 4 | 预测未来帧数 |
| image_crop | 256 | 图像裁剪尺寸 |
| seg_classes | 3 | BEV分割类别（背景/车辆/目标车位） |
| epochs | 155 | 训练总轮数 |
| 仿真帧率 | 30 Hz | CARLA fixed_delta_seconds = 1/30 |
| 数据保存频率 | 10 Hz | 每3帧保存一次 |
