# =============================================================================
# 训练数据集模块
#
# 职责：从磁盘加载数据采集阶段保存的原始数据，并预处理为训练所需的格式
#
# 核心类：
#   CarlaDataset     — PyTorch Dataset，加载全部训练/验证数据
#   ProcessSemantic  — BEV 语义分割 GT 处理器（解码→裁剪→画目标车位→分3类）
#   ProcessImage     — RGB 图像处理器（裁剪→归一化）
#
# 关键工具函数：
#   convert_slot_coord() — 目标车位坐标：世界系→自车系
#   tokenize()           — 控制信号离散化：连续值→token
#   detokenize()         — 控制信号反离散化：token→连续值
#   get_depth()          — CARLA RGB 深度图→米制单通道深度
#
# 数据流：
#   磁盘文件（data_generation 阶段产出）
#     ├─ rgb_{front,left,right,rear}/  → 4路 RGB 图像
#     ├─ depth_{front,left,right,rear}/ → 4路深度图（RGB编码）
#     ├─ topdown/encoded_*.png          → BEV 语义图（15通道压缩为RGB）
#     ├─ measurements/*.json           → 自车位姿+速度+加速度+控制量
#     └─ parking_goal/0001.json        → 目标车位世界坐标 {x,y,yaw}
#                    ↓
#     CarlaDataset.__getitem__() 返回 dict：
#       image:        [4, 3, crop, crop]   4路裁剪归一化RGB
#       depth:        [4, 1, crop, crop]   4路米制深度
#       extrinsics:   [4, 4, 4]            相机外参（veh→cam）
#       intrinsics:   [4, 3, 3]            相机内参（裁剪后）
#       segmentation: [1, 200, 200]        BEV语义GT（0=背景,1=车辆,2=目标车位）
#       target_point: [3]                  目标车位自车系坐标 [x,y,yaw]
#       ego_motion:   [1, 3]               [速度, acc_x, acc_y]
#       gt_control:   [15]                 token化的控制序列（含BOS/EOS/PAD）
#       gt_acc/gt_steer/gt_reverse:        原始控制值（用于辅助loss）
# =============================================================================
import json
import os
import carla
import torch.utils.data
import numpy as np
import torchvision.transforms

from PIL import Image
from loguru import logger


def convert_slot_coord(ego_trans, target_point):
    """
    将目标车位坐标从世界坐标系转换到自车坐标系
    
    Args:
        ego_trans: carla.Transform，自车的 veh→world 变换
        target_point: [x, y, yaw] 目标车位在世界坐标系下的位置和朝向（度）
    Returns:
        [x, y, yaw_diff] 目标车位在自车坐标系下的位置和相对朝向差（度）
    """
    # 位置转换：世界系 → 自车系
    target_point_self_veh = convert_veh_coord(target_point[0], target_point[1], 1.0, ego_trans)

    # 朝向转换：计算目标车位与自车的 yaw 差值，并规范化到 [-180, 180]
    yaw_diff = target_point[2] - ego_trans.rotation.yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    target_point = [target_point_self_veh[0], target_point_self_veh[1], yaw_diff]

    return target_point


def convert_veh_coord(x, y, z, ego_trans):
    """
    将世界坐标系中的点 (x,y,z) 转换到自车坐标系
    
    原理：ego_trans 是 veh→world 的变换矩阵，取其逆矩阵得到 world→veh
    用齐次坐标 [x,y,z,1] 乘以 world→veh 矩阵完成转换
    """
    world2veh = np.array(ego_trans.get_inverse_matrix())  # 4×4 逆变换矩阵
    target_array = np.array([x, y, z, 1.0], dtype=float)  # 齐次坐标
    target_point_self_veh = world2veh @ target_array
    return target_point_self_veh


def scale_and_crop_image(image, scale=1.0, crop=256):
    """
    缩放并中心裁剪 PIL 图像
    
    Args:
        image: PIL Image 对象
        scale: 缩小倍数（>1 表示缩小）
        crop:  裁剪后的正方形边长（像素）
    Returns:
        cropped_image: numpy 数组 [crop, crop] 或 [crop, crop, C]
    """
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), resample=Image.NEAREST)
    image = np.asarray(im_resized)
    # 从中心裁剪 crop×crop 区域
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop].copy()
    return cropped_image


def tokenize(throttle, brake, steer, reverse, token_nums=200):
    """
    将连续控制信号离散化为 token（类似 NLP 词表）
    token 编号规则（以 token_nums=204 为例，valid_token=200, half_token=100）：
      throttle_brake（油门/刹车共用一个 token，互斥）：
        [0,  100) : brake 区间   (brake=1→0, brake=0→100)
        [100, 200] : throttle 区间 (throttle=0→1, throttle=100→200)
      [0, 200] : steer     ([-1,1] 映射到 [0, valid_token])
      token_nums-3 : BOS (Begin Of Sequence，解码起始符)
      token_nums-2 : EOS (End Of Sequence)
      token_nums-1 : PAD (填充符)
    reverse 由 steer 的高低区间隐式编码：
      steer = 0  → reverse=False
      steer = 200  → reverse=True
    :param throttle: [0,1]
    :param brake: [0,1]
    :param steer: [-1,1]
    :param reverse: {0,1}
    :param token_nums: size of token
    :return: tokenized control range [0, token_nums-4]
    """

    valid_token = token_nums - 4
    half_token = valid_token / 2

    if brake != 0.0:
        throttle_brake_token = int(half_token * (-brake + 1))
    else:
        throttle_brake_token = int(half_token * (throttle + 1))
    steer_token = int((steer + 1) * half_token) #100回正
    reverse_token = int(reverse * valid_token)
    return [throttle_brake_token, steer_token, reverse_token]


def detokenize(token_list, token_nums=200):
    """
    Detokenize control signals
    :param token_list: [throttle_brake, steer, reverse]
    :param token_nums: size of token number
    :return: control signal values
    """

    valid_token = token_nums - 4
    half_token = float(valid_token / 2)

    if token_list[0] > half_token:
        throttle = token_list[0] / half_token - 1
        brake = 0.0
    else:
        throttle = 0.0
        brake = -(token_list[0] / half_token - 1)

    steer = (token_list[1] / half_token) - 1
    reverse = (True if token_list[2] > half_token else False)

    return [throttle, brake, steer, reverse]


def get_depth(depth_image_path, crop):
    """
    将 CARLA 的 RGB 编码深度图转换为米制单通道深度图
    
    CARLA 深度编码规则：
        深度值被编码到 RGB 三个通道中：
        depth = (R + G*256 + B*65536) / (256^3 - 1) * 1000m
        其中 R 是低位，B 是高位，最大表示 1000m
    
    Args:
        depth_image_path: CARLA 深度图文件路径（RGB 编码格式）
        crop: 裁剪尺寸
    Returns:
        tensor [1, crop, crop]，单通道深度值（单位：米）
    """
    depth_image = Image.open(depth_image_path).convert('RGB')

    data = np.array(scale_and_crop_image(depth_image, scale=1.0, crop=crop))

    data = data.astype(np.float32)

    # RGB → 归一化深度值：R*1 + G*256 + B*65536，然后除以 (256^3-1)
    normalized = np.dot(data, [1.0, 256.0, 65536.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized  # 归一化值 × 1000 = 米

    return torch.from_numpy(in_meters).unsqueeze(0)


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    """
    根据图像裁剪和缩放更新相机内参矩阵
    
    裁剪会改变主点（cx, cy）位置，缩放会改变焦距和主点
    
    内参矩阵格式：
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    update_intrinsic = intrinsics.clone()

    # 缩放：焦距和主点同步缩放
    update_intrinsic[0, 0] *= scale_width   # fx
    update_intrinsic[0, 2] *= scale_width   # cx
    update_intrinsic[1, 1] *= scale_height  # fy
    update_intrinsic[1, 2] *= scale_height  # cy

    # 裁剪：主点位置偏移
    update_intrinsic[0, 2] -= left_crop     # cx 减去左裁剪量
    update_intrinsic[1, 2] -= top_crop      # cy 减去上裁剪量

    return update_intrinsic


def add_raw_control(data, throttle_brake, steer, reverse):
    """
    收集原始控制信号（未 token 化），用于训练时的辅助 loss
    throttle_brake 合并为单值：刹车为负，油门为正，范围 [-1, 1]
    """
    if data['Brake'] != 0.0:
        throttle_brake.append(-data['Brake'])  # 刹车取负
    else:
        throttle_brake.append(data['Throttle'])  # 油门为正
    steer.append(data['Steer'])
    reverse.append(int(data['Reverse']))


class CarlaDataset(torch.utils.data.Dataset):
    """
    停车场景训练数据集
    
    加载流程：
        __init__  → init_camera_config()  → get_data()
        init_camera_config(): 构建 4 个相机的内参和外参矩阵
        get_data():          遍历所有 task 目录，预加载文件路径和数值数据到内存
        __getitem__():       按索引读取图像并处理，返回训练所需的 dict
    
    关键设计：
        - 文件路径和数值数据在 __init__ 时全部预加载到内存（np.array）
        - 图像的实际读取和处理延迟到 __getitem__ 时执行（懒加载）
        - 控制序列组装为 [BOS, tok1, tok2, ..., tok12, EOS, PAD]，长度固定 15
    """
    def __init__(self, root_dir, is_train, config):
        super(CarlaDataset, self).__init__()
        self.cfg = config

        # 特殊 token：BOS=201, EOS=202, PAD=203（当 token_nums=204 时）
        self.BOS_token = self.cfg.token_nums - 3
        self.EOS_token = self.BOS_token + 1
        self.PAD_token = self.EOS_token + 1

        self.root_dir = root_dir
        self.is_train = is_train

        # ---- 相机配置 ----
        self.image_crop = self.cfg.image_crop  # 裁剪后的图像尺寸（默认 256）
        self.intrinsic = None                  # [4, 3, 3] 裁剪后内参
        self.veh2cam_dict = {}                 # 各相机的 veh→cam 外参
        self.extrinsic = None                  # [4, 4, 4] 外参矩阵
        self.image_process = ProcessImage(self.image_crop)    # RGB 图像处理器
        self.semantic_process = ProcessSemantic(self.cfg)      # BEV 语义图处理器

        self.init_camera_config()  # 构建内参和外参

        # ---- 数据存储容器（预加载到内存）----
        # 4路 RGB 图像路径
        self.front = []
        self.left = []
        self.right = []
        self.rear = []

        # 4路深度图路径
        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.rear_depth = []

        # 控制信号（token 化 + 原始值）
        self.control = []

        # 运动特征
        self.velocity = []
        self.acc_x = []
        self.acc_y = []

        # 原始控制值（未 token 化，用于辅助 loss）
        self.throttle_brake = []
        self.steer = []
        self.reverse = []

        # 目标车位坐标（自车系）
        self.target_point = []

        # BEV 语义图路径
        self.topdown = []

        self.get_data()  # 预加载所有数据

    def init_camera_config(self):
        """
        构建 4 个相机的内参和外参矩阵
        
        相机配置：400×300, FOV=100°
        布局：
            front: 车头正前 (1.5, 0, 1.5), yaw=0
            left:  左侧   (0, -0.8, 1.5), yaw=-90, pitch=-40
            right: 右侧   (0, 0.8, 1.5),  yaw=90,  pitch=-40
            rear:  车尾   (-2.2, 0, 1.5), yaw=180, pitch=-30
        
        内参构建：
            fx = fy = w / (2 * tan(fov/2))，主点 (cx,cy) = (w/2, h/2)
            然后根据裁剪量更新 cx, cy
        
        外参构建：
            cam2veh → 取逆 → veh2cam，再左乘 cam2pixel 坐标系转换矩阵
        """
        cam_config = {'width': 400, 'height': 300, 'fov': 100}

        # 四个相机的安装位置和朝向（相对于自车坐标系）
        cam_specs = {
            'rgb_front': {  # 前视相机：车头正前方
                'x': 1.5, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_left': {  # 左视相机：左侧向下俑视
                'x': 0.0, 'y': -0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': -90.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_right': {  # 右视相机：右侧向下俑视
                'x': 0.0, 'y': 0.8, 'z': 1.5,
                'roll': 0.0, 'pitch': -40.0, 'yaw': 90.0,
                'type': 'sensor.camera.rgb',
            },
            'rgb_rear': {  # 后视相机：车尾向下俑视
                'x': -2.2, 'y': 0.0, 'z': 1.5,
                'roll': 0.0, 'pitch': -30.0, 'yaw': 180.0,
                'type': 'sensor.camera.rgb',
            },
        }

        # ---- 构建内参矩阵 ----
        w = cam_config['width']    # 400
        h = cam_config['height']   # 300
        fov = cam_config['fov']    # 100°
        # 焦距计算：f = w / (2 * tan(fov/2))
        f = w / (2 * np.tan(fov * np.pi / 360))
        Cu = w / 2  # 主点 x
        Cv = h / 2  # 主点 y
        intrinsic_original = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        # 根据图像裁剪更新内参（主点偏移）
        self.intrinsic = update_intrinsics(
            torch.from_numpy(intrinsic_original).float(),
            (h - self.image_crop) / 2,   # 上下裁剪量 = (300-256)/2 = 22
            (w - self.image_crop) / 2,   # 左右裁剪量 = (400-256)/2 = 72
            scale_width=1,
            scale_height=1
        )
        # 扩展为 [4, 3, 3]，4 个相机共用同一内参
        self.intrinsic = self.intrinsic.unsqueeze(0).expand(4, 3, 3)

        # ---- 构建外参矩阵 ----
        # CARLA 相机坐标系 → 图像像素坐标系的转换矩阵
        # CARLA: x=前, y=右, z=上  →  图像: x=右, y=下, z=前
        cam2pixel = np.array([
            [0, 1, 0, 0],   # 图像 x = CARLA y
            [0, 0, -1, 0],  # 图像 y = -CARLA z
            [1, 0, 0, 0],   # 图像 z = CARLA x
            [0, 0, 0, 1],
        ], dtype=float)
        # 对每个相机：cam2veh 取逆 → veh2cam，再左乘 cam2pixel
        for cam_id, cam_spec in cam_specs.items():
            cam2veh = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                      carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                     roll=cam_spec['roll']))
            veh2cam = cam2pixel @ np.array(cam2veh.get_inverse_matrix())  # [4, 4]
            self.veh2cam_dict[cam_id] = veh2cam
        # 拼接为 [4, 4, 4] 张量
        front_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_front']).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_left']).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_right']).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(self.veh2cam_dict['rgb_rear']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

    def get_data(self):
        """
        遍历磁盘上所有 task 目录，预加载文件路径和数值数据
        
        目录结构：
            root_dir / {train|val}_town / seed_xxx / task_yyy /
                ├─ rgb_front/      0001.png ~ NNNN.png
                ├─ rgb_left/       ...
                ├─ rgb_right/      ...
                ├─ rgb_rear/       ...
                ├─ depth_front/    ...
                ├─ depth_left/     ...
                ├─ depth_right/    ...
                ├─ depth_rear/     ...
                ├─ topdown/        encoded_0001.png ~ ...
                ├─ measurements/   0001.json ~ ...
                └─ parking_goal/   0001.json
        
        帧索引范围：[hist_frame_nums, total_frames - future_frame_nums)
            因为每个样本需要读取未来 future_frame_nums 帧的控制量作为 GT
        """
        val_towns = self.cfg.validation_map
        train_towns = self.cfg.training_map
        train_data = os.path.join(self.root_dir, train_towns)
        val_data = os.path.join(self.root_dir, val_towns)

        town_dir = train_data if self.is_train == 1 else val_data

        # 收集所有停车任务的路径
        root_dirs = os.listdir(town_dir)
        all_tasks = []
        for root_dir in root_dirs:
            root_path = os.path.join(town_dir, root_dir)
            for task_dir in os.listdir(root_path):
                task_path = os.path.join(root_path, task_dir)
                all_tasks.append(task_path)

        for task_path in all_tasks:
            total_frames = len(os.listdir(task_path + "/measurements/"))
            # 遍历每个有效帧（需要预留未来帧给 GT 控制序列）
            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                # ---- 当前帧数据 ----
                filename = f"{str(frame).zfill(4)}.png"  # 例如 "0005.png"
                # 4路 RGB 图像路径
                self.front.append(task_path + "/rgb_front/" + filename)
                self.left.append(task_path + "/rgb_left/" + filename)
                self.right.append(task_path + "/rgb_right/" + filename)
                self.rear.append(task_path + "/rgb_rear/" + filename)

                # 4路深度图路径
                self.front_depth.append(task_path + "/depth_front/" + filename)
                self.left_depth.append(task_path + "/depth_left/" + filename)
                self.right_depth.append(task_path + "/depth_right/" + filename)
                self.rear_depth.append(task_path + "/depth_rear/" + filename)

                # BEV 语义图路径（encode_npy_to_pil 的产出）
                self.topdown.append(task_path + "/topdown/encoded_" + filename)

                # 读取当前帧的测量数据（位姿 + 运动 + 控制）
                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # 自车位姿（用于将目标车位从世界系转为自车系）
                ego_trans = carla.Transform(carla.Location(x=data['x'], y=data['y'], z=data['z']),
                                            carla.Rotation(yaw=data['yaw'], pitch=data['pitch'], roll=data['roll']))

                # 运动特征：速度 + 加速度
                self.velocity.append(data['speed'])
                self.acc_x.append(data['acc_x'])
                self.acc_y.append(data['acc_y'])

                # ---- 未来 future_frame_nums 帧的控制量作为 GT 标签 ----
                # 每帧 3 个 token: [throttle_brake, steer, reverse]
                # future_frame_nums=4 → 4×3=12 个 token
                controls = []
                throttle_brakes = []
                steers = []
                reverse = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + 1 + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                    controls.append(
                        tokenize(data['Throttle'], data["Brake"], data["Steer"], data["Reverse"], self.cfg.token_nums))
                    add_raw_control(data, throttle_brakes, steers, reverse)

                # 拼接并添加特殊 token: [BOS, tok1...tok12, EOS, PAD] → 长度 15
                controls = [item for sublist in controls for item in sublist]  # 展平 4×3=12 个 token
                controls.insert(0, self.BOS_token)   # 头部插入 BOS
                controls.append(self.EOS_token)       # 末尾添加 EOS
                controls.append(self.PAD_token)       # 再添加 PAD
                self.control.append(controls)          # 总长度 = 1+12+1+1 = 15

                self.throttle_brake.append(throttle_brakes)  # 原始 throttle/brake
                self.steer.append(steers)                     # 原始 steer
                self.reverse.append(reverse)                  # 原始 reverse

                # ---- 目标车位（从 parking_goal.json 读取，转换到自车系）----
                # 注意：全部帧共用同一个目标车位（世界坐标固定），但自车坐标系下不同
                with open(task_path + f"/parking_goal/0001.json", "r") as read_file:
                    data = json.load(read_file)
                parking_goal = [data['x'], data['y'], data['yaw']]  # 世界坐标
                parking_goal = convert_slot_coord(ego_trans, parking_goal)  # → 自车坐标系
                self.target_point.append(parking_goal)

        # ---- 转为 numpy 数组，加速后续访问 ----
        # 图像路径存为 bytes 类型（np.string_）
        self.front = np.array(self.front).astype(np.string_)
        self.left = np.array(self.left).astype(np.string_)
        self.right = np.array(self.right).astype(np.string_)
        self.rear = np.array(self.rear).astype(np.string_)

        self.front_depth = np.array(self.front_depth).astype(np.string_)
        self.left_depth = np.array(self.left_depth).astype(np.string_)
        self.right_depth = np.array(self.right_depth).astype(np.string_)
        self.rear_depth = np.array(self.rear_depth).astype(np.string_)

        self.topdown = np.array(self.topdown).astype(np.string_)

        self.velocity = np.array(self.velocity).astype(np.float32)
        self.acc_x = np.array(self.acc_x).astype(np.float32)
        self.acc_y = np.array(self.acc_y).astype(np.float32)

        self.control = np.array(self.control).astype(np.int64)

        self.throttle_brake = np.array(self.throttle_brake).astype(np.float32)
        self.steer = np.array(self.steer).astype(np.float32)
        self.reverse = np.array(self.reverse).astype(np.int64)

        self.target_point = np.array(self.target_point).astype(np.float32)

        logger.info('Preloaded {} sequences', str(len(self.front)))

    def __len__(self):
        return len(self.front)

    def __getitem__(self, index):
        """
        按索引加载一个训练样本
        
        返回 dict 包含：
            image:        [4, 3, crop, crop]   4路裁剪归一化 RGB
            depth:        [4, 1, crop, crop]   4路米制深度
            extrinsics:   [4, 4, 4]            veh→cam 外参
            intrinsics:   [4, 3, 3]            裁剪后内参
            segmentation: [1, 200, 200]        BEV 语义 GT (0/1/2)
            target_point: [3]                  目标车位自车系 [x,y,yaw]
            ego_motion:   [1, 3]               [速度, acc_x, acc_y]
            gt_control:   [15]                 token序列 [BOS,tok*12,EOS,PAD]
            gt_acc:       [4]                  原始 throttle/brake
            gt_steer:     [4]                  原始 steer
            gt_reverse:   [4]                  原始 reverse
        """
        data = {}
        keys = ['image', 'depth', 'extrinsics', 'intrinsics', 'target_point', 'ego_motion', 'segmentation',
                'gt_control', 'gt_acc', 'gt_steer', 'gt_reverse']
        for key in keys:
            data[key] = []

        # ---- 4路 RGB 图像：读取→裁剪→归一化→拼接 ----
        images = [self.image_process(self.front[index])[0], self.image_process(self.left[index])[0],
                  self.image_process(self.right[index])[0], self.image_process(self.rear[index])[0]]
        images = torch.cat(images, dim=0)  # [4, 3, crop, crop]
        data['image'] = images

        data['extrinsics'] = self.extrinsic  # [4, 4, 4] 共用
        data['intrinsics'] = self.intrinsic  # [4, 3, 3] 共用

        # ---- 4路深度图：RGB编码→米制单通道 ----
        depths = [get_depth(self.front_depth[index], self.image_crop),
                  get_depth(self.left_depth[index], self.image_crop),
                  get_depth(self.right_depth[index], self.image_crop),
                  get_depth(self.rear_depth[index], self.image_crop)]
        depths = torch.cat(depths, dim=0)  # [4, 1, crop, crop]
        data['depth'] = depths

        # ---- BEV 语义分割 GT ----
        # 读取编码BEV→解码→裁剪→画目标车位→分3类 (0=背景, 1=车辆, 2=目标车位)
        # 0.2m一个像素点，scale=0.5后0.4m一个像素点，crop=200后覆盖80m×80m范围
        segmentation = self.semantic_process(self.topdown[index], scale=0.5, crop=200,
                                             target_slot=self.target_point[index])
        data['segmentation'] = torch.from_numpy(segmentation).long().unsqueeze(0)  # [1, 200, 200]

        # ---- 目标车位坐标（自车系）----
        data['target_point'] = torch.from_numpy(self.target_point[index])  # [3]

        # ---- 运动特征 ----
        ego_motion = np.column_stack((self.velocity[index], self.acc_x[index], self.acc_y[index]))
        data['ego_motion'] = torch.from_numpy(ego_motion)  # [1, 3]

        # ---- GT 控制信号 ----
        data['gt_control'] = torch.from_numpy(self.control[index])    # [15] token序列

        # 原始控制值（用于辅助 loss）
        data['gt_acc'] = torch.from_numpy(self.throttle_brake[index])  # [4]
        data['gt_steer'] = torch.from_numpy(self.steer[index])        # [4]
        data['gt_reverse'] = torch.from_numpy(self.reverse[index])    # [4]

        return data


class ProcessSemantic:    """
    BEV 语义分割 GT 处理器
    
    输入：编码后的 BEV PNG 图像（encode_npy_to_pil 产出，15通道压缩为 RGB）
    输出：[200, 200] 的整数语义图，值为 0/1/2
    
    处理流程：
        1. 读取编码 PNG → 转灰度 → 缩放裁剪 (scale=0.5, crop=200)
        2. draw_target_slot(): 在灰度图上绘制目标车位矩形 (pixel=255)
        3. 分 3 类：pixel==75 → class 1(车辆), pixel==255 → class 2(目标车位), 其余 → class 0(背景)
    
    关于 pixel==75 的来源：
        encode_npy_to_pil 将 ch5(车辆) 编码到 G 通道的低位，灰度图中值为约 75
        （具体：ch5 存在 G 通道的 bit2，灰度值 = (R+G+B)/3 ≈ (0+4×..+0)/3 → 某个特定值）
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image, scale, crop, target_slot):
        """
        将编码 BEV 图像处理为 3 类语义分割 GT
        
        Args:
            image: PIL Image 或图像路径（编码后的 BEV PNG）
            scale: 缩小因子（0.5 → 500×500 变为 250×250）
            crop:  裁剪尺寸（200 → 中心裁剪 200×200）
            target_slot: 目标车位自车系坐标 [x, y, yaw_diff]
        Returns:
            semantics: [200, 200] numpy 数组，值为 0(背景)/1(车辆)/2(目标车位)
        """
        # 读取编码 BEV 图像
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert('L')  # 转灰度图（RGB 平均）

        # 缩放并中心裁剪
        cropped_image = scale_and_crop_image(image, scale, crop)

        # 在 BEV 图上绘制目标车位（像素值设为 255）
        cropped_image = self.draw_target_slot(cropped_image, target_slot)

        # 语义分类：根据像素值分 3 类
        h, w = cropped_image.shape
        vehicle_index = cropped_image == 75    # 车辆像素值≈ 75
        target_index = cropped_image == 255    # 目标车位像素值 = 255
        semantics = np.zeros((h, w))           # 默认背景 = 0
        semantics[vehicle_index] = 1           # 车辆 = 1
        semantics[target_index] = 2            # 目标车位 = 2
        # BEV 图上下翻转，使车头朝向图像 x 轴正方向（LSS 约定）
        semantics = semantics[::-1]

        return semantics.copy()

    def draw_target_slot(self, image, target_slot):
        """
        在 BEV 灰度图上绘制目标车位矩形（像素值设为 255）
        
        车位大小：55×31 像素 = range(-27,28) × range(-15,16)
        在 scale=0.5 下：每像素 = 0.2m / 0.5 = 0.4m
        实际尺寸 = 55×0.4 × 31×0.4 ≈ 22m × 12.4m （较大，因为包含整个车位区域）
        
        流程：
            1. 将目标车位的米制坐标转为像素坐标
            2. 生成 55×31 个点的矩形模板
            3. 按目标车位的 yaw 角旋转这些点
            4. 平移到车位中心位置，在图像上标记为 255
        """
        size = image.shape[0]  # 200

        # 米制坐标 → 像素坐标（除以 BEV 分辨率，默认 0.1m/px）
        x_pixel = target_slot[0] / self.cfg.bev_x_bound[2]
        y_pixel = target_slot[1] / self.cfg.bev_y_bound[2]
        # 自车在图像中心，x向前 = 像素向上（减），y向右 = 像素向右（加）
        target_point = np.array([size / 2 - x_pixel, size / 2 + y_pixel], dtype=int)

        # 生成车位矩形的所有点（55×31 = 1705 个点）
        slot_points = []
        for x in range(-27, 28):
            for y in range(-15, 16):
                slot_points.append(np.array([x, y, 1, 1], dtype=int))  # 齐次坐标

        # 按目标车位的 yaw 旋转这些点（用 carla.Transform 矩阵）
        slot_trans = np.array(
            carla.Transform(carla.Location(), carla.Rotation(yaw=float(-target_slot[2]))).get_matrix())
        slot_points = np.vstack(slot_points).T  # [4, 1705]
        slot_points_ego = (slot_trans @ slot_points)[0:2].astype(int)  # 取前 2 行 (x, y)

        # 平移到车位中心坐标
        slot_points_ego[0] += target_point[0]
        slot_points_ego[1] += target_point[1]

        # 在图像上标记为 255（后续会被识别为 class 2 = 目标车位）
        image[tuple(slot_points_ego)] = 255

        return image


class ProcessImage:
    """
    RGB 图像处理器
    流程：读取图像 → 中心裁剪 crop×crop → ImageNet 归一化
    支持输入：文件路径（训练时）或 carla.Image 对象（评估时）
    """
    def __init__(self, crop):
        self.crop = crop

        # ImageNet 标准归一化（因为使用预训练 ResNet18）
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),  # [0,255] uint8 → [0,1] float
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

    def __call__(self, image):
        """
        Args:
            image: 文件路径（str/bytes）或 carla.Image 对象
        Returns:
            (normalized_tensor, crop_array):
                normalized_tensor: [1, 3, crop, crop] 归一化后的图像
                crop_array:        [crop, crop, 3] 原始裁剪 numpy 数组
        """
        if isinstance(image, carla.Image):
            # CARLA 实时数据：BGRA → BGR → RGB → PIL
            image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            image = image[:, :, :3]     # 去掉 Alpha 通道
            image = image[:, :, ::-1]   # BGR → RGB
            image = Image.fromarray(image)
        else:
            image = Image.open(image).convert('RGB')

        crop_image = scale_and_crop_image(image, scale=1.0, crop=self.crop)

        return self.normalise_image(np.array(crop_image)).unsqueeze(0), crop_image
