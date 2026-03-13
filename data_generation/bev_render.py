# =============================================================================
# BEV（鸟瞰图）渲染模块
#
# 职责：将 CARLA 世界的道路、车道线、车辆等信息渲染为多通道 BEV 语义图
#
# 核心类：
#   BevRender  — 对外接口，管理全局地图和车辆渲染
#   Renderer   — 底层渲染引擎，处理坐标变换和仿射采样
#   MapImage   — 从 CARLA HD Map 生成道路/车道线的像素图
#
# 通道分配（15 通道，实际只用了 3 个）：
#   ch0: 道路（road）     ← 从 HD Map 渲染
#   ch1: 车道线（lane）   ← 从 HD Map 渲染
#   ch2~4: 保留（红/黄/绿灯，本项目未使用）
#   ch5: 车辆（vehicle）  ← 动态渲染其他车辆
#   ch6~14: 保留（未使用）
#
# 调用关系：
#   DataGenerator → World → BevRender.render_BEV() / get_bev_states()
#                                ↓
#                          Renderer.get_local_birdview()  ← 裁剪以自车为中心的局部视图
#                          Renderer.render_agent_bv()     ← 在 BEV 上绘制其他车辆
# =============================================================================
import torch
import pygame
import numpy as np
import torch.nn.functional as F
import carla
from PIL import Image
import os

# Global Flags
PIXELS_PER_METER = 5  # 每米对应 5 个像素，即 1 像素 = 0.2m

# ---- pygame 调色板常量（用于 MapImage 绘制道路/车道线/HUD 可视化）----
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_TRAFFIC_RED = pygame.Color(255, 0, 0)
COLOR_TRAFFIC_YELLOW = pygame.Color(0, 255, 0)
COLOR_TRAFFIC_GREEN = pygame.Color(0, 0, 255)


class BevRender:
    """
    BEV 渲染器对外接口
    负责：1) 初始化时从 HD Map 生成全局道路/车道线地图
          2) 每帧裁剪以自车为中心的局部 BEV，并在上面绘制其他车辆
    """
    def __init__(self, world, device):
        self._device = device         # 'cpu' 或 'cuda'
        self._world = world.world     # carla.World 对象
        self._vehicle = world.player  # 自车 Actor
        self._actors = None

        # 从 CARLA 获取 OpenDRIVE 格式的高精地图
        hd_map = self._map = self._world.get_map().to_opendrive()
        self.world_map = carla.Map("RouteMap", hd_map)

        # 默认车辆模板：22×9 像素的矩形（约 4.4m × 1.8m）
        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self._device)

        # ---- 从 HD Map 生成全局静态地图 ----
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        # pygame Surface → numpy 灰度图
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)  # 道路区域（白色道路，黑色背景）
        lane = make_image(map_image.lane_surface)  # 车道线

        # 构建 15 通道全局语义地图 [1, 15, H_map, W_map]
        # 实际只用了 ch0（道路）和 ch1（车道线），其余通道为 0
        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.   # 归一化到 [0, 1]
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device=self._device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self._device, dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]  # 全局地图的像素尺寸 (H, W)

        # 底层渲染器：处理世界坐标→像素坐标变换、仿射采样
        self.renderer = Renderer(world_offset, self.map_dims, data_generation=True, device=self._device)

        self.detection_radius = 50.0  # 只渲染自车 50m 半径内的其他车辆

    def set_player(self, player):
        self._vehicle = player

    def render_BEV(self):
        """
        实时渲染当前帧的 BEV 语义图（直接从 CARLA 世界读取最新状态）
        
        流程：
            1. 从全局地图中裁剪以自车为中心的局部区域（道路+车道线）
            2. 遍历 50m 内的其他车辆，按其 bounding box 大小绘制到 ch5
        
        Returns:
            birdview: tensor [1, 15, crop_H, crop_W]，以自车为中心的局部 BEV
        """
        ego_t = self._vehicle.get_transform()
        semantic_grid = self.global_map

        # Step 1: 从全局地图裁剪局部视图（旋转对齐自车朝向）
        ego_pos = torch.tensor([ego_t.location.x, ego_t.location.y],
                               device=self._device, dtype=torch.float32)
        ego_yaw = torch.tensor([ego_t.rotation.yaw / 180 * np.pi], device=self._device,
                               dtype=torch.float32)  # 度 → 弧度
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        # Step 2: 在 BEV 上绘制 50m 内的其他车辆（写入 channel=5）
        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if vehicle.get_location().distance(ego_t.location) < self.detection_radius:
                if vehicle.id != self._vehicle.id:  # 排除自车
                    pos = torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y],
                                       device=self._device, dtype=torch.float32)
                    yaw = torch.tensor([vehicle.get_transform().rotation.yaw / 180 * np.pi], device=self._device,
                                       dtype=torch.float32)
                    # 根据车辆实际尺寸创建矩形模板（extent 是半长/半宽，×2 得全尺寸）
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device=self._device)
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5  # 车辆渲染到通道 5
                    )

        return birdview

    def get_bev_states(self):
        """
        快照当前帧所有车辆的位姿（Transform），用于后续离线渲染 BEV
        这样在 save_sensor_data 时可以用 render_BEV_from_state() 重新渲染，
        而不需要 CARLA 世界仍处于那个时刻
        
        Returns:
            dict: {'ego_t': 自车Transform, 'vehicle_ts': [所有车辆的Transform列表]}
        """
        def get_element_ts(keyword):
            elements = self._world.get_actors().filter(keyword)
            ts = [carla.Transform(element.get_transform().location, element.get_transform().rotation)
                  for element in elements]
            return ts

        return {
            "ego_t": carla.Transform(self._vehicle.get_transform().location,
                                     self._vehicle.get_transform().rotation),
            "vehicle_ts": get_element_ts("*vehicle*"),
        }

    def render_BEV_from_state(self, state):
        """
        从保存的状态快照（get_bev_states 的返回值）离线渲染 BEV
        逻辑与 render_BEV() 完全相同，只是数据来源从实时查询变为历史快照
        在 save_sensor_data() 的多线程保存中被调用
        """
        ego_t = state["ego_t"]

        semantic_grid = self.global_map

        # fetch local birdview per agent
        ego_pos = torch.tensor([ego_t.location.x, ego_t.location.y],
                               device=self._device, dtype=torch.float32)
        ego_yaw = torch.tensor([ego_t.rotation.yaw / 180 * np.pi], device=self._device,
                               dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        # vehicle is only used for id and bbox, which is never changed during the play
        # WARNING: the order of filter('*vehicle*') can't be changed or bug occurs here
        for vehicle_t, vehicle in zip(state["vehicle_ts"], self._world.get_actors().filter('*vehicle*')):
            if vehicle.id != self._vehicle.id:
                if vehicle_t.location.distance(ego_t.location) < self.detection_radius:
                    pos = torch.tensor([vehicle_t.location.x, vehicle_t.location.y],
                                       device=self._device, dtype=torch.float32)
                    yaw = torch.tensor([vehicle_t.rotation.yaw / 180 * np.pi], device=self._device,
                                       dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device=self._device)
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        return birdview


class Renderer():
    """
    底层 BEV 渲染引擎
    核心功能：
        1. get_local_birdview(): 从全局地图裁剪以自车为中心的局部视图（旋转对齐自车朝向）
        2. render_agent_bv(): 将其他车辆绘制到局部 BEV 上
    
    坐标变换链：
        世界坐标 (m) → world_to_pix() → 全局像素坐标 → world_to_rel() → [-1,1] 归一化坐标
        → F.affine_grid() + F.grid_sample() 实现旋转裁剪
    """
    def __init__(self, map_offset, map_dims, data_generation=True, device='cpu'):
        self.args = {'device': device}
        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0    # 数据采集模式：自车在 BEV 中心
            self.local_view_dims = (500, 500)  # 局部视图尺寸 500×500 像素
            self.crop_dims = (500, 500)         # = 500/5 = 100m × 100m 范围
        else:
            self.PIXELS_AHEAD_VEHICLE = 100 + 10  # 评估模式：自车偏下方（前方可视更远）
            self.local_view_dims = (320, 320)
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.local_view_scale = (
            self.local_view_dims[1] / self.map_dims[1],
            self.local_view_dims[0] / self.map_dims[0]
        )
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

    def world_to_pix(self, pos):
        """世界坐标 (m) → 全局地图像素坐标，减去偏移后乘以 5 px/m"""
        pos_px = (pos - self.map_offset) * PIXELS_PER_METER
        return pos_px

    def world_to_pix_crop_batched(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        # TODO: should be able to handle batches

        # # FIXME: why do we need to do this everywhere?
        crop_yaw = crop_yaw + np.pi / 2
        batch_size = crop_pos.shape[0]

        # transform to crop pose
        rotation = torch.stack(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
             torch.sin(crop_yaw), torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1).unsqueeze(1) @ \
                       (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2],
                                                  device=self.args['device'])

        return pos_px_crop

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        """
        将世界坐标转换为局部裁剪图的像素坐标
        
        变换步骤：
            1. 世界坐标 → 全局像素坐标（world_to_pix）
            2. 减去裁剪中心（自车）的像素坐标，得到相对偏移
            3. 用自车朝向的逆旋转矩阵旋转，消除自车朝向
            4. 加上裁剪图半宽/半高的偏移，将原点从中心移到左上角
        
        Returns:
            pos_px_crop: [x, y] 在局部裁剪图中的像素坐标
        """
        crop_yaw = crop_yaw + np.pi / 2  # CARLA yaw 坐标系修正

        # 自车朝向的旋转矩阵
        rotation = torch.tensor(
            [[torch.cos(crop_yaw), -torch.sin(crop_yaw)],
             [torch.sin(crop_yaw), torch.cos(crop_yaw)]],
            device=self.args['device'],
        )

        crop_pos_px = self.world_to_pix(crop_pos)  # 自车的全局像素坐标

        # 数据采集模式下 PIXELS_AHEAD_VEHICLE=0，此项为零向量
        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)  # 查询点的全局像素坐标

        # rotation.T = 逆旋转，将 (查询点 - 自车) 的偏移从世界坐标系旋转到自车坐标系
        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift

        # 将原点从裁剪图中心移到左上角
        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2],
                                                  device=self.args['device'])

        return pos_px_crop

    def world_to_rel(self, pos):
        """
        世界坐标 → [-1, 1] 归一化坐标（用于 F.affine_grid/F.grid_sample）
        步骤：world → pixel → [0,1] → [-1,1]
        """
        pos_px = self.world_to_pix(pos)
        # 除以地图像素尺寸，归一化到 [0, 1]
        pos_rel = pos_px / torch.tensor([self.map_dims[1], self.map_dims[0]], device=self.args['device'])
        # 映射到 [-1, 1]，这是 PyTorch grid_sample 的标准坐标范围
        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def render_agent(self, grid, vehicle, position, orientation):
        """
        """
        orientation = orientation - np.pi / 2  # TODO
        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # convert position from world to relative image coordinates
        position = self.world_to_rel(position) * -1

        # TODO: build composite transform directly
        # build individual transforms
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        rotation_transform = torch.tensor(
            [[torch.cos(orientation), torch.sin(orientation), 0],
             [-torch.sin(orientation), torch.cos(orientation), 0],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, position[0]],
             [0, 1, position[1]],
             [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        # chain transforms
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # expects Nx2x3
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        grid[:, 5, ...] += vehicle_rendering.squeeze()

        return grid

    def render_agent_bv(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
            state=None,  # traffic light_state
    ):
        """
        将单个车辆/物体绘制到局部 BEV 上的指定通道
        
        使用 STN（Spatial Transformer Network）的仿射变换：
            1. 计算车辆在局部 BEV 中的相对坐标
            2. 构建 缩放×旋转×平移 的仿射矩阵
            3. F.grid_sample() 将矩形模板变换到正确位置和朝向
            4. 累加到 grid[:, channel, ...] 上
        
        Args:
            grid: [1, 15, H, W] 局部 BEV 张量（会被原地修改）
            grid_pos: 自车世界坐标
            grid_orientation: 自车朝向（弧度）
            vehicle: [1, 1, veh_H, veh_W] 车辆矩形模板
            position: 目标车辆世界坐标
            orientation: 目标车辆朝向（弧度）
            channel: 写入哪个通道（默认 5=车辆）
        """
        orientation = orientation + np.pi / 2  # CARLA yaw 坐标系修正

        # 将目标车辆世界坐标转为局部裁剪图的像素坐标
        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)

        # 像素坐标 → [-1, 1] STN 归一化坐标
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])  # [0, 1]
        pos_rel_bv = pos_rel_bv * 2 - 1   # [-1, 1]
        pos_rel_bv = pos_rel_bv * -1       # STN 坐标轴方向翻转

        # 缩放比：车辆模板 → BEV 网格的比例
        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # ---- 构建 缩放 × 旋转 × 平移 的仿射矩阵 ----
        # 缩放变换：将小的车辆模板缩放到 BEV 网格尺寸
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # 旋转变换：目标车辆朝向 - 自车朝向 = 相对朝向
        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [[torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
             [- torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # 平移变换：将车辆放到 BEV 中的正确位置
        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]],
             [0, 1, pos_rel_bv[1]],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # 链式复合：scale @ rotation @ translation
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        # 生成采样网格并执行 grid_sample
        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # 取 2×3 子矩阵
            (1, 1, grid.shape[2], grid.shape[3]),  # 输出尺寸与 BEV 相同
            align_corners=True,
        )

        # 将车辆模板（全1矩形）变换到目标位置和朝向
        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        # 交通灯状态决定写入哪个通道（本项目停车场景未使用）
        if state == 'Green':
            channel = 4
        elif state == 'Yellow':
            channel = 3
        elif state == 'Red':
            channel = 2

        # 累加到 BEV 的对应通道上（可能多辆车重叠）
        grid[:, channel, ...] += vehicle_rendering.squeeze()

    def render_agent_bv_batched(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
    ):
        """
        """
        # FIXME: why do we need to do this everywhere?
        orientation = orientation + np.pi / 2
        batch_size = position.shape[0]

        pos_pix_bv = self.world_to_pix_crop_batched(position, grid_pos, grid_orientation)

        # to centered relative coordinates for STN
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])  # normalize over h and w
        pos_rel_bv = pos_rel_bv * 2 - 1  # change domain from [0, 1] to [-1, 1]
        pos_rel_bv = pos_rel_bv * -1  # Because the STN coordinates are weird

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])

        # TODO: build composite transform directly
        # build individual transforms
        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
             [0, scale_h, 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3).expand(batch_size, -1, -1)

        # this is the inverse of the rotation matrix for the visibility check
        # because now we want crop coordinates instead of world coordinates
        grid_orientation = grid_orientation + np.pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)
        rotation_transform = torch.stack(
            [torch.cos(angle_delta), torch.sin(angle_delta), zeros,
             -torch.sin(angle_delta), torch.cos(angle_delta), zeros,
             zeros, zeros, ones],
            dim=-1
        ).view(batch_size, 3, 3)

        translation_transform = torch.stack(
            [ones, zeros, pos_rel_bv[..., 0:1],
             zeros, ones, pos_rel_bv[..., 1:2],
             zeros, zeros, ones],
            dim=-1,
        ).view(batch_size, 3, 3)

        # chain transforms
        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],  # expects Nx2x3
            (batch_size, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        for i in range(batch_size):
            grid[:, int(channel[i].item()), ...] += vehicle_rendering[i].squeeze()

    def get_local_birdview(self, grid, position, orientation):
        """
        从全局地图中裁剪以自车为中心的局部 BEV 视图
        
        使用仿射变换实现旋转裁剪：全局地图按自车朝向旋转，
        然后以自车位置为中心裁剪 crop_dims 大小的区域
        
        Args:
            grid: [1, 15, H_map, W_map] 全局语义地图
            position: 自车世界坐标 [x, y]
            orientation: 自车朝向（弧度）
        Returns:
            local_view: [1, 15, crop_H, crop_W] 局部 BEV
        """

        # ---- Step 1: 世界坐标 → [-1,1] 归一化坐标 ----
        position = self.world_to_rel(position)
        orientation = orientation + np.pi / 2  # CARLA yaw 坐标系修正

        # ---- Step 2: 构建仿射变换 = 缩放 × 平移 × 旋转 ----
        # 缩放：将全局地图缩放到 crop_dims 大小的窗口
        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
             [0, self.crop_scale[0], 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # 旋转：按自车朝向旋转，使 BEV 始终与自车对齐
        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0],
             [torch.sin(orientation), torch.cos(orientation), 0],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # 数据采集模式下 PIXELS_AHEAD_VEHICLE=0，shift 为零
        shift = torch.tensor([0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]], device=self.args['device'])
        position = position + rotation_transform[0, 0:2, 0:2] @ shift

        # 平移：将裁剪窗口中心移到自车位置
        translation_transform = torch.tensor(
            [[1, 0, position[0] / self.crop_scale[0]],
             [0, 1, position[1] / self.crop_scale[1]],
             [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        # ---- Step 3: 链式复合仿射变换，然后用 grid_sample 执行裁剪 ----
        # 变换顺序：先旋转 → 再平移 → 再缩放
        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        # 生成采样网格：每个输出像素对应输入图中的坐标
        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],  # 取 2×3 子矩阵
            (1, 1, self.crop_dims[0], self.crop_dims[0]),  # 输出尺寸
            align_corners=True,
        )

        # 双线性插值采样：从全局地图中提取局部视图
        # 输入 grid [1,15,H_map,W_map] → 输出 [1,15,crop_H,crop_W]
        local_view = F.grid_sample(
            grid,
            affine_grid,
            align_corners=True,
        )

        return local_view

    def step(self, actions):
        """
        """
        # 1. update ego agent
        print(self.ego.state, actions)
        # actions['steer'] = torch.Tensor([0.])
        self.ego.set_state(self.ego.motion_model(self.ego.state, actions=actions))
        # self.ego.state['yaw'] *= 0
        # self.ego.state['yaw'] += np.pi * self.timestep / 100
        # self.ego.set_state(self.ego.state)
        self.adv.set_state(self.adv.motion_model(self.adv.state))

        # 2. update adversarial agents
        # ...
        self.timestep += 1

    def visualize_grid(self, grid, type='LTS_Reduced'):
        """
        将多通道 BEV tensor 可视化为彩色 PIL 图像（用于 pygame 显示和调试）
        根据 type 参数选择不同的语义类别着色方案
        """
        if type == 'LTS_Reduced':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
            ]

        elif type == 'Trajectory_planner':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                # (0, 0, 142), # vehicle
                # (220, 20, 60), # pedestrian
            ]

        elif type == 'LTS_Full':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                (204, 6, 5),  # red light
                (250, 210, 1),  # yellow light
                (39, 232, 51),  # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
            ]
        elif type == 'LTS_FullFuture':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                (204, 6, 5),  # red light
                (250, 210, 1),  # yellow light
                (39, 232, 51),  # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],  # vehicle future
            ]
        elif type == 'LTS_ReducedFuture':
            colors = [
                (102, 102, 102),  # road
                (253, 253, 17),  # lane
                # (204, 6, 5), # red light
                # (250, 210, 1), # yellow light
                # (39, 232, 51), # green light
                (0, 0, 142),  # vehicle
                (220, 20, 60),  # pedestrian
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],  # vehicle future
            ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [0, 47, 0]

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def bev_to_gray_img(self, grid):
        """
        将多通道 BEV tensor 转为灰度图（每个语义类别用不同灰度值 1~7 表示）
        """
        colors = [
            1,  # road
            2,  # lane
            3,  # red light
            4,  # yellow light
            5,  # green light
            6,  # vehicle
            7,  # pedestrian
        ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4]), dtype=np.uint8)

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img


class ModuleManager(object):
    """简易模块管理器，用于管理 pygame 渲染模块的生命周期（本项目中仅用于 MapImage 初始化时的临时管理）"""
    def __init__(self):
        self.modules = []

    def register_module(self, module):
        self.modules.append(module)

    def clear_modules(self):
        del self.modules[:]

    def tick(self, clock):
        # Update all the modules
        for module in self.modules:
            module.tick(clock)

    def render(self, display, snapshot=None):
        display.fill(COLOR_ALUMINIUM_4)
        for module in self.modules:
            module.render(display, snapshot=snapshot)

    def get_module(self, name):
        for module in self.modules:
            if module.name == name:
                return module

    def start_modules(self):
        for module in self.modules:
            module.start()


module_manager = ModuleManager()


class MapImage(object):
    """
    从 CARLA HD Map（OpenDRIVE 格式）生成全局道路/车道线的像素图
    
    输出两个 pygame.Surface：
        map_surface  — 道路区域（白色多边形填充）
        lane_surface — 车道线（实线/虚线）
    
    这些 Surface 被 BevRender 转为 numpy 数组，存入 global_map 的 ch0/ch1
    """
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 不需要真实显示器，用虚拟 SDL 驱动

        module_manager.clear_modules()

        pygame.init()
        display = pygame.display.set_mode((320, 320), 0, 32)
        # pygame.display.flip()

        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        # ---- 计算地图边界 ----
        # 每隔 2m 在所有道路上生成 waypoint，找出 x/y 的极值
        waypoints = carla_map.generate_waypoints(2)
        margin = 50  # 四周留 50m 边距
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        # 取 x/y 跨度的较大值作为正方形地图的边长
        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)  # 左上角世界坐标，后续 world_to_pix 要减去

        width_in_pixels = int(self._pixels_per_meter * self.width)  # 地图像素边长

        # 创建两个等大的 pygame Surface
        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()   # 道路
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()  # 车道线
        self.draw_road_map(
            self.big_map_surface, self.big_lane_surface,
            carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        """
        从 CARLA 地图拓扑中绘制道路多边形和车道线
        
        核心算法：
            1. 获取地图拓扑（topology）：所有道路段的起始 waypoint
            2. 沿每条道路向前采样，收集左/右边界点
            3. 将边界点围成多边形，填充到 map_surface（道路区域）
            4. 根据车道线类型（实线/虚线）绘制到 lane_surface
        """
        map_surface.fill(COLOR_BLACK)  # 黑色背景
        precision = 0.05  # 沿道路采样间隔 0.05m（精度很高）

        def draw_lane_marking(surface, points, solid=True):
            """绘制车道线：实线用连续线段，虚线用间断线段"""
            if solid and len(points) > 1:
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
            else:
                # 每 20 个点一组，每 3 组取 1 组绘制 → 虚线效果
                broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            start = transform.location
            end = start + 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def draw_stop(surface, font_surface, transform, color=COLOR_ALUMINIUM_2):
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        # ---- 获取地图拓扑：每条道路段的起始 waypoint ----
        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)  # 按高度排序

        for waypoint in topology:
            # 从起始 waypoint 向前采样，直到道路段结束
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            # 计算每个采样点的左边界和右边界位置（横向偏移 ±半车道宽）
            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            # 左边界 + 右边界反转 = 闭合多边形
            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]  # 世界坐标 → 像素坐标

            if len(polygon) > 2:
                # 先画轮廓（线宽10），再填充（实心白色）→ 道路区域
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            # 在非交叉口区域绘制车道线（交叉口内不画）
            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]  # 取中间采样点判断线型
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1))  # True=实线
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1))

                # Dian: Do not draw them arrows
                # for n, wp in enumerate(waypoints):
                #     if (n % 400) == 0:
                #         draw_arrow(map_surface, wp.transform)

        actors = carla_world.get_actors()
        stops_transform = [actor.get_transform() for actor in actors if 'stop' in actor.type_id]
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)
        font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        font_surface = pygame.transform.scale(font_surface, (font_surface.get_width(), font_surface.get_height() * 2))

        # Dian: do not draw stop sign

        # for stop in stops_transform:
        #     draw_stop(map_surface,font_surface, stop)

    def world_to_pixel(self, location, offset=(0, 0)):
        """CARLA 世界坐标 → pygame Surface 像素坐标"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
