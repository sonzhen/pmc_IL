import sys
import json
import math
import random
import logging
import pathlib

import numpy as np
import cv2

from datetime import datetime
from threading import Thread

from data_generation import parking_position
from data_generation.tools import encode_npy_to_pil
from data_generation.world import World


class DataGenerator:
    """
    数据采集核心调度器
    职责：管理停车任务的生命周期（初始化→采集→判停→保存→切换下一任务）
    调用关系：carla_data_gen.py → DataGenerator → World → Sensors/BEV
    """
    def __init__(self, carla_world, args):
        self._seed = args.random_seed
        random.seed(args.random_seed)

        self._world = World(carla_world, args)  # 封装 CARLA 世界（车辆、传感器、天气）

        #只泊入中间一排的车位，总共4排
        self._parking_goal_index = 17  # 2-2; index 15+2=17
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]  # 目标车位的世界坐标
        self._ego_transform_generator = parking_position.EgoPosTown04()  # 自车初始位姿生成器（随机起点）

        # 用时间戳生成唯一目录名，如 "03_12_14_30_05"
        now = datetime.now()
        result_dir = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        # 最终路径: save_path/Town04_Opt/03_12_14_30_05/
        self._save_path = pathlib.Path(args.save_path) / args.map / result_dir
        self._save_path.mkdir(parents=True, exist_ok=False)

        #降低存储频率，保证每个task的数据量在合理范围内（每个task约1.5GB）
        self._save_frequency = 3  # save sensor data for every 3 steps 0.1s

        self._num_tasks = args.task_num
        self._task_index = 0

        # 当前帧与最近目标车位的距离/角度差（每帧更新）
        self._distance_diff_to_goal = 10000
        self._rotation_diff_to_goal = 10000
        # 判定"停好"的阈值：距离 < 0.5m 且 角度 < 0.5°
        self._goal_reach_distance_diff = 0.5  # meter
        self._goal_reach_rotation_diff = 0.5  # degree

        # number of frames needs to get into the parking goal in order to consider task completed
        #停稳2s
        self._num_frames_goal_needed = 2 * 30  # 2s * 30Hz
        self._num_frames_in_goal = 0

        # collected sensor data to output in disk at last
        self._batch_data_frames = []

        self.init()

    @property
    def world(self):
        return self._world

    def world_tick(self):
        self._world.world_tick()

    def render(self, display):
        self._world.render(display)

    def init(self):
        """初始化一个停车任务的完整环境：清理旧状态 → 放NPC车 → 放自车 → 装传感器 → 设天气"""
        logging.info('*****Init environment for task %d*****', self._task_index)

        # clear all previous setting
        self.destroy()

        # 在停车场奇数号车位放置静态NPC车辆（目标车位 parking_goal_index 保持空闲）
        self._world.init_static_npc(self._seed, self._parking_goal_index)

        # 在目标车位附近随机生成自车起始位置
        self._ego_transform_generator.update_data_gen_goal_y(self._parking_goal.y)
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()
        self._world.init_ego_vehicle(ego_transform)

        # 安装传感器（4路RGB + 4路Depth + LiDAR + IMU + GNSS + 碰撞检测）
        self._world.init_sensors()

        self._world.next_weather()  # 切换天气（如果启用了 shuffle_weather）

        logging.info('*****Init environment for task %d done*****', self._task_index)

    def destroy(self):
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0

        self._world.destroy()

    def soft_destroy(self):
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0

        self._world.soft_destroy()

    def tick(self, clock):
        """
        每帧调用一次，执行：碰撞检测 → 数据采集（每3帧）→ 到达目标检测
        由 carla_data_gen.py 的 game_loop 主循环调用
        """
        # 传递距离/角度差给 World，用于 HUD 显示
        self._world.distance_diff_to_goal = self._distance_diff_to_goal
        self._world.rotation_diff_to_goal = self._rotation_diff_to_goal

        # 推进 World 状态并检测碰撞，碰撞则 soft_restart（重置自车位置，不重建NPC）
        is_collision = self._world.tick(clock, self._parking_goal_index)
        if is_collision:
            self.soft_restart()
            return

        # 每 3 帧保存一次传感器数据（30Hz 仿真 → 10Hz 保存）
        step = self._world.step
        if step % self._save_frequency == 0:
            sensor_data_frame = self._world.sensor_data_frame  # dict: rgb/depth/lidar/imu/gnss/控制量等
            sensor_data_frame['bev_state'] = self._world.bev_state  # 追加 BEV 语义状态
            self._batch_data_frames.append(sensor_data_frame.copy())  # 暂存内存，任务完成后统一写盘

        # 检查是否到达目标车位
        self.check_goal()

    def check_goal(self):
        """
        检查自车是否已停入目标车位
        判定条件：连续 60 帧（2秒）距离 < 0.5m 且 角度 < 0.5°
        满足后触发 save_sensor_data() 保存整个任务的数据
        """
        t = self._world.ego_transform
        p = t.location   # 自车当前位置 (x, y, z)
        r = t.rotation   # 自车当前姿态 (pitch, yaw, roll)

        all_parking_goals = self._world.all_parking_goals  # 所有空车位的坐标

        # 遍历所有空车位，找到距离自车最近的那个
        self._distance_diff_to_goal = sys.float_info.max
        closest_goal = [0.0, 0.0, 0.0]  # (x, y, yaw)
        for parking_goal in all_parking_goals:
            if p.distance(parking_goal) < self._distance_diff_to_goal:
                self._distance_diff_to_goal = p.distance(parking_goal)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y
                closest_goal[2] = r.yaw

        # 计算角度差（处理 yaw 在 ±180° 边界的跳变，并考虑 roll/pitch）
        self._rotation_diff_to_goal = math.sqrt(min(abs(r.yaw), 180 - abs(r.yaw)) ** 2 + r.roll ** 2 + r.pitch ** 2)

        # check if goal is reached
        if self._distance_diff_to_goal < self._goal_reach_distance_diff and \
                self._rotation_diff_to_goal < self._goal_reach_rotation_diff:
            self._num_frames_in_goal += 1
        else:
            self._num_frames_in_goal = 0

        if self._num_frames_in_goal > self._num_frames_goal_needed:
            logging.info('task %d goal reached; ready to save sensor data', self._task_index)
            self.save_sensor_data(closest_goal)
            logging.info('*****task %d done*****', self._task_index)
            self._task_index += 1
            if self._task_index >= self._num_tasks:
                logging.info('completed all tasks; Thank you!')
                exit(0)
            self.restart()

    def restart(self):
        """
        切换到下一个停车任务：更换目标车位 + 重新随机自车起点 + 重排NPC车辆
        目标车位按偶数索引递增：17→19→21→...→47，超过16个任务后回到17
        """
        logging.info('*****Config environment for task %d*****', self._task_index)

        # clear all previous setting
        self.soft_destroy()

        # 切换目标车位：每次跳 2 个索引（跳过NPC占据的奇数号车位）
        if self._task_index >= 16:
            self._parking_goal_index = 17    # 超过16个任务后回到 2-2 号车位
        else:
            self._parking_goal_index += 2    # 17→19→21→...（2-2→2-4→2-6→...）

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.update_data_gen_goal_y(self._parking_goal.y)
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()

        self._seed += 1
        self._world.restart(self._seed, self._parking_goal_index, ego_transform)

        logging.info('*****Config environment for task %d done*****', self._task_index)

    def soft_restart(self):
        """碰撞后的轻量重启：只重置自车位置和已采集数据，不重建NPC车辆"""
        logging.info('*****Restart task %d*****', self._task_index)
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()  # 重新随机起点
        self._world.soft_restart(ego_transform)

        # clear cache
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0

        logging.info('*****Restart task %d done*****', self._task_index)

    def save_sensor_data(self, parking_goal):
        """
        将整个任务的采集数据批量写入磁盘，使用10线程并行加速IO
        
        磁盘目录结构:
            taskN/
            ├── rgb_front/     0000.png ~ XXXX.png  (CARLA Image → PNG)
            ├── rgb_left/      ...
            ├── rgb_right/     ...
            ├── rgb_rear/      ...
            ├── depth_front/   0000.png ~ XXXX.png  (CARLA Depth → PNG)
            ├── depth_left/    ...
            ├── depth_right/   ...
            ├── depth_rear/    ...
            ├── lidar/         0000.ply ~ XXXX.ply  (点云 PLY 格式)
            ├── measurements/  0000.json ~ XXXX.json (位置/速度/控制量/IMU/GNSS)
            ├── parking_goal/  0001.json  (目标车位坐标 x,y,yaw)
            └── topdown/       encoded_0000.png ~ (BEV语义图，位运算压缩为RGB)
        """
        # 创建目录
        cur_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        cur_save_path.mkdir(parents=True, exist_ok=False)
        (cur_save_path / 'measurements').mkdir()
        (cur_save_path / 'lidar').mkdir()
        (cur_save_path / 'parking_goal').mkdir()
        (cur_save_path / 'topdown').mkdir()
        # 动态创建 rgb_front/ rgb_left/ depth_front/ 等目录
        for sensor in self._batch_data_frames[0].keys():
            if sensor.startswith('rgb') or sensor.startswith('depth'):
                (cur_save_path / sensor).mkdir()

        # 多线程并行保存（IO 密集型，多线程有效）
        total_frames = len(self._batch_data_frames)
        thread_num = 10
        frames_for_thread = total_frames // thread_num
        thread_list = []
        for t_idx in range(thread_num):
            start = t_idx * frames_for_thread
            if t_idx == thread_num - 1:
                end = total_frames      # 最后一个线程处理余数部分
            else:
                end = (t_idx + 1) * frames_for_thread
            t = Thread(target=self.save_unit_data, args=(start, end, cur_save_path))
            t.start()
            thread_list.append(t)

        for thread in thread_list:
            thread.join()  # 等待所有线程完成

        # save Parking Goal
        measurements_file = cur_save_path / 'parking_goal' / '0001.json'
        with open(measurements_file, 'w') as f:
            data = {'x': parking_goal[0],
                    'y': parking_goal[1],
                    'yaw': parking_goal[2]}
            json.dump(data, f, indent=4)

        # save vehicle video
        self._world.save_video(cur_save_path)

        logging.info('saved sensor data for task %d in %s', self._task_index, str(cur_save_path))

    def save_unit_data(self, start, end, cur_save_path):
        """单个线程的保存任务：处理 [start, end) 范围内的帧"""
        for index in range(start, end):
            data_frame = self._batch_data_frames[index]

            # ---- 1. 保存传感器原始数据（carla.Image/LidarMeasurement 自带 save_to_disk 方法）----
            for sensor in data_frame.keys():
                if sensor.startswith('rgb'):
                    # data_frame[sensor] 是 carla.Image 对象 → 保存为 PNG
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                elif sensor.startswith('depth'):
                    # 深度图也是 carla.Image 对象 → 保存为 PNG（16bit 灰度编码）
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                elif sensor.startswith('lidar'):
                    # LiDAR 点云 → 保存为 PLY 格式
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / 'lidar' / ('%04d.ply' % index)))

            # ---- 2. 保存测量数据为 JSON ----
            imu_data = data_frame['imu']              # 惯性测量单元（加速度 + 角速度 + 指南针）
            gnss_data = data_frame['gnss']             # 全球定位（经纬度）
            vehicle_transform = data_frame['veh_transfrom']  # 自车位姿 (x,y,z,pitch,yaw,roll)
            vehicle_velocity = data_frame['veh_velocity']    # 自车速度向量 (vx,vy,vz) m/s
            vehicle_control = data_frame['veh_control']      # 自车控制量 (throttle,steer,brake,...)

            data = {
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z,
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw,
                'roll': vehicle_transform.rotation.roll,
                # 速度: m/s → km/h（乘3.6），三维向量求模
                'speed': (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)),
                'Throttle': vehicle_control.throttle,
                'Steer': vehicle_control.steer,
                'Brake': vehicle_control.brake,
                'Reverse': vehicle_control.reverse,
                'Hand brake': vehicle_control.hand_brake,
                'Manual': vehicle_control.manual_gear_shift,
                'Gear': {-1: 'R', 0: 'N'}.get(vehicle_control.gear, vehicle_control.gear),  # -1→'R'倒挡, 0→'N'空挡, 其他保留数字
                'acc_x': imu_data.accelerometer.x,
                'acc_y': imu_data.accelerometer.y,
                'acc_z': imu_data.accelerometer.z,
                'gyr_x': imu_data.gyroscope.x,
                'gyr_y': imu_data.gyroscope.y,
                'gyr_z': imu_data.gyroscope.z,
                'compass': imu_data.compass,
                'lat': gnss_data.latitude,
                'lon': gnss_data.longitude
            }

            measurements_file = cur_save_path / 'measurements' / ('%04d.json' % index)
            with open(measurements_file, 'w') as f:
                json.dump(data, f, indent=4)

            # ---- 3. 保存 BEV 俯视图（多通道语义图 → 位运算压缩为 RGB PNG）----
            def save_img(image, keyword=""):
                img_save = np.moveaxis(image, 0, 2)  # [C,W,H] → [W,H,C]，转为 OpenCV 格式
                save_path = str(cur_save_path / 'topdown' / ('encoded_%04d' % index + keyword + '.png'))
                cv2.imwrite(save_path, img_save)

            keyword = ""
            # 从 bev_state 渲染 BEV 语义图，形状 [1, C, W, H]（C 为语义类别数，每个通道是二值图）
            bev_view1 = self._world.render_BEV_from_state(data_frame['bev_state'])
            # squeeze(): [1,C,W,H]→[C,W,H]  .cpu(): GPU→CPU  np.asarray: tensor→numpy
            # encode_npy_to_pil: 用位运算将最多15个二值通道压缩到3个uint8通道(RGB)
            #   通道0~4 → R的bit7~bit3, 通道5~9 → G的bit7~bit3, 通道10~14 → B的bit7~bit3
            img1 = encode_npy_to_pil(np.asarray(bev_view1.squeeze().cpu()))
            save_img(img1, keyword)
