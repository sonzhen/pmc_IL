# =============================================================================
# 核心神经网络：ParkingModel
#
# 整体数据流（encoder → decoder）：
#
#  [4路环视图像]  [相机内/外参]                          [自车运动(速度/加速度)]
#       │               │                                        │
#       ▼               ▼                                        │
#   CamEncoder ─── BevModel ────► [BEV特征图 200x200]            │
#   (EfficientNet      │           + 目标车位热力图叠加            │
#    + 深度预测)        │ add_target_bev()                        │
#                      │                                        │
#                      ▼                                        │
#                  BevEncoder ──────────────────────────────────┤
#                  (ResNet18压缩)                                │
#                      │                                        │
#                      └──────────────► FeatureFusion ◄─────────┘
#                                       (Transformer Encoder)
#                                              │
#                    ┌─────────────────────────┤
#                    ▼                         ▼
#           SegmentationHead           ControlPredict
#           (预测BEV停车位分割图)       (Transformer Decoder)
#                    │                         │
#                    ▼                         ▼
#             [3类分割图输出]     [tokenized控制序列: throttle/brake/steer/reverse]
#
# forward()  用于训练：teacher-forcing，输入gt_control作为decoder的ground truth序列
# predict()  用于推理：自回归解码，逐步生成4个控制token
# =============================================================================
import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        self.segmentation_head = SegmentationHead(self.cfg)

    def add_target_bev(self, bev_feature, target_point):
        b, c, h, w = bev_feature.shape
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)

        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise

        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
                             target_point_batch[1] - 4:target_point_batch[1] + 4] = 1.0

        bev_feature = torch.cat([bev_feature, bev_target], dim=1)
        return bev_feature, bev_target

    def encoder(self, data):
        """编码阶段（训练和推理共用）：图像 → BEV特征 → Transformer融合特征"""
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)

        # Step1: 4路环视图像 + 相机参数 → 鸟瞰图(BEV)特征 + 深度图（LSS: Lift-Splat-Shoot）
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # Step2: 将目标停车位坐标渲染为热力图，叠加到BEV特征上（带随机噪声增广）
        bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)

        # Step3: ResNet18对BEV特征下采样压缩（200x200 → 序列）
        bev_down_sample = self.bev_encoder(bev_feature)

        # Step4: Transformer Encoder 将BEV序列与自车运动(速度/加速度)融合
        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion)

        # Step5（辅助任务）: 从融合特征解码BEV分割图（背景/车辆/目标车位，3类）
        pred_segmentation = self.segmentation_head(fuse_feature)

        return fuse_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        """训练时调用：teacher-forcing模式，gt_control作为Decoder输入"""
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        # ControlPredict: Transformer Decoder，输入gt序列预测下一个control token
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        return pred_control, pred_segmentation, pred_depth

    def predict(self, data):
        """推理时调用：自回归解码，以BOS token起始，逐步预测4个控制token
        输出 pred_multi_controls 形如 [BOS, throttle_tok, brake_tok, steer_tok, reverse_tok]
        需用 detokenize() 将 token 还原为实际控制值
        """
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()  # 初始：[BOS_token]
        for i in range(3):  # 自回归生成3个控制token（throttle/brake/steer; reverse由steer决定）
            pred_control = self.control_predict.predict(fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        return pred_multi_controls, pred_segmentation, pred_depth, bev_target
