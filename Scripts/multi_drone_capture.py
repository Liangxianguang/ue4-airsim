import os
import time
import json
import csv
import cv2
import numpy as np
import airsim
import argparse
import datetime
import threading

# --- 配置 ---
PREFERRED_VEHICLES = ["Drone1", "Drone2", "Drone3", "Drone4"]
CAM_IDX = 0
OUT_DIR = "output"
# 目标采集帧率（视频编码器也会按该帧率写入）；降低帧率以提高稳定性
CAPTURE_FPS = 20
FLIGHT_SPEED = 3.0  # 降低速度以减少计算负荷
FLIGHT_ALT = -5.0  # 飞得更高以获得更广的视野
# 可选：启用一个外部观察者摄像头（需在 AirSim settings.json 中配置 ExternalCameras -> "Observer"）
ENABLE_OBSERVER_CAMERA = True
OBSERVER_CAMERA_NAME = "0"          # 外部摄像机的相机索引名，通常为字符串"0"
OBSERVER_VEHICLE_NAME = "Observer"  # 外部摄像机在 settings.json 中的名称
OBSERVER_COMPRESS = False            # 改为False，提高图像质量和处理速度
OBSERVER_DEDICATED_THREAD = True     # 为Observer启用专用采集+编码线程，获得更稳定的“相机式”录制
OBSERVER_FPS = 30  # 降低Observer帧率以减少系统压力                   
ENABLE_DEPTH_SEGMENTATION = False    # 启用深度图和分割掩码采集（仅对Observer有效）
ENABLE_DRONE_DEPTH_SEGMENTATION = False  # 启用无人机深度图和分割掩码采集（会显著增加数据量）

# Observer 缩放/距离控制（解决“越拍越远”）：
# lock: 固定距离与高度
# bounded: 自适应但限制在最小/最大范围并做EMA平滑
# auto: 原始逻辑，完全自适应
OBSERVER_ZOOM_MODE = "lock"          # 改为lock模式，固定距离
OBSERVER_FIXED_DIST = 35.0           # 固定水平距离
OBSERVER_FIXED_HEIGHT = 20.0         # 固定高度
OBSERVER_MIN_DIST = 22.0             # bounded 模式距离/高度范围
OBSERVER_MAX_DIST = 45.0
OBSERVER_MIN_HEIGHT = 18.0
OBSERVER_MAX_HEIGHT = 35.0
OBSERVER_ZOOM_SMOOTH = 0.2          # 从0.4改为0.2，减少缩放变化
OBSERVER_PHASE_SWITCH_SEC = 10.0     # 延长近景锁定时间

# Observer 地面观察阶段参数（先水平/近景观察起飞，后平滑升空到空中视角）
OBSERVER_GROUND_HEIGHT = 2.0         # 地面观察高度（米）
OBSERVER_GROUND_DIST = 10.0          # 地面观察时与质心的水平距离（米）

# Observer 摄像机平滑与限制（cinematic）
# 每 N 帧更新一次观察者摄像机（降低开销），值越小更新越频繁
OBSERVER_UPDATE_EVERY_N = 5           # 从2改为5，减少更新频率
# 平滑时间常数（秒）：小值更紧贴目标，大值更平滑
CAM_SMOOTH_POS_TAU = 1.2              # 从0.4改为1.2，增加位置平滑
CAM_SMOOTH_ROT_TAU = 0.8              # 从0.18改为0.8，增加旋转平滑
# 最大位置速度（单位：世界坐标/秒）与最大角速度（度/秒）
CAM_MAX_POS_SPEED = 10.0              # 从30改为10，降低移动速度
CAM_MAX_ANG_SPEED_DEG = 30.0          # 从120改为30，降低旋转速度
# 可选的FOV平滑（启用可随缩放改变视场）
OBSERVER_FOV_ENABLE = False           # 禁用FOV变化，减少视觉跳动
OBSERVER_FOV_MIN = 30.0
OBSERVER_FOV_MAX = 75.0
OBSERVER_FOV_SMOOTH = 0.12

# 当循环性能不足导致实际帧率低于CAPTURE_FPS时，是否通过重复写入上一帧来拉齐视频播放时长
MATCH_REALTIME_OUTPUT = True  # 启用补帧，保持视频时间与真实时间一致
# 为防止极端卡顿导致一次性重复过多帧，设置每tick最大补帧数上限
MAX_DUP_FRAMES_PER_TICK = 10  # 降低到10，避免大量补帧

# 性能优化选项
# 0 表示不单独落盘 PNG（仅写视频）；>0 表示每 N 帧落一张 PNG
SAVE_PNG_EVERY_N = 0
# 传感器（IMU/GPS/磁力计/气压计）记录的降采样因子（每 N 帧记录一次）
SENSOR_LOG_EVERY_N = 1
# 是否保存 Observer 的视频
SAVE_OBSERVER_VIDEO = True

# 拦截跟踪相关配置
ENABLE_BBOX_ANNOTATION = True       # 启用边界框标注功能
ENABLE_DEPTH_SEGMENTATION = True   # 启用深度图和分割掩码采集
BBOX_LOG_EVERY_N = 1               # 边界框标注记录频率（每N帧记录一次）
DRONE_BBOX_SIZE = (2.0, 2.0, 1.0)  # 无人机边界框尺寸 (长x宽x高)，单位米

import math
# 自动生成复杂轨迹（圆形、螺旋、S型、交错等）
def generate_path(pattern, center, radius, alt, num_points=40):
    path = []
    if pattern == "circle":
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = float(center[0] + radius * math.cos(theta))
            y = float(center[1] + radius * math.sin(theta))
            path.append((x, y, float(alt)))
    elif pattern == "spiral":
        for i in range(num_points):
            r = radius * (i / num_points)
            theta = 4 * math.pi * i / num_points
            x = float(center[0] + r * math.cos(theta))
            y = float(center[1] + r * math.sin(theta))
            z = float(alt + i * 0.2)  # 螺旋上升
            path.append((x, y, z))
    elif pattern == "sine":
        for i in range(num_points):
            x = float(center[0] + i * radius / num_points)
            y = float(center[1] + math.sin(i * 2 * math.pi / num_points) * radius / 2)
            path.append((x, y, float(alt)))
    elif pattern == "zigzag":
        for i in range(num_points):
            x = float(center[0] + i * radius / num_points)
            y = float(center[1] + (radius if i % 2 == 0 else -radius))
            path.append((x, y, float(alt)))
    elif pattern == "evasive":  # 规避机动 - 急转弯 + 随机扰动
        for i in range(num_points):
            base_angle = 6 * math.pi * i / num_points  # 快速转向
            noise_x = float(np.random.uniform(-radius*0.3, radius*0.3))  # 随机扰动
            noise_y = float(np.random.uniform(-radius*0.3, radius*0.3))
            x = float(center[0] + radius * math.cos(base_angle) + noise_x)
            y = float(center[1] + radius * math.sin(base_angle) + noise_y)
            z = float(alt + np.random.uniform(-1, 1))  # 高度随机变化
            path.append((x, y, z))
    elif pattern == "intercept":  # 拦截轨迹 - 高速直线冲刺
        for i in range(num_points):
            progress = i / (num_points - 1)
            # 加速段 + 匀速段 + 减速段
            if progress < 0.3:
                speed_factor = progress / 0.3  # 加速
            elif progress > 0.7:
                speed_factor = (1 - progress) / 0.3  # 减速
            else:
                speed_factor = 1.0  # 匀速
            
            x = float(center[0] + radius * 2 * progress)  # 直线运动
            y = float(center[1] + math.sin(progress * math.pi) * radius * 0.2)  # 轻微摆动
            z = float(alt - progress * radius * 0.1)  # 轻微俯冲
            path.append((x, y, z))
    elif pattern == "swarm_break":  # 编队解散 - 从集中到分散
        for i in range(num_points):
            progress = i / (num_points - 1)
            spread_factor = progress * 2.0  # 逐渐分散
            angle = 2 * math.pi * i / num_points
            x = float(center[0] + radius * spread_factor * math.cos(angle))
            y = float(center[1] + radius * spread_factor * math.sin(angle))
            z = float(alt + spread_factor * radius * 0.1)
            path.append((x, y, z))
    elif pattern == "combat_turn":  # 战斗转弯 - 高G机动
        for i in range(num_points):
            # 紧急转弯，高角速度
            theta = 8 * math.pi * i / num_points  
            # 模拟高G转弯的轨迹特征
            turn_radius = radius * (0.5 + 0.5 * math.sin(4 * math.pi * i / num_points))
            x = float(center[0] + turn_radius * math.cos(theta))
            y = float(center[1] + turn_radius * math.sin(theta))
            z = float(alt + math.sin(theta) * radius * 0.15)  # 垂直机动
            path.append((x, y, z))
    else:
        for i in range(num_points):
            path.append((float(center[0] + i), float(center[1]), float(alt)))
    return path

def quaternion_to_rotation_matrix(q):
    """将四元数(airsim.Quaternionr)转换为旋转矩阵 R (3x3)."""
    w = q.w_val; x = q.x_val; y = q.y_val; z = q.z_val
    # 归一化（防止累计误差）
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w /= norm; x /= norm; y /= norm; z /= norm
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

# 3D坐标转换为Observer相机的2D边界框（改进版：使用真实相机位姿与姿态旋转到相机坐标系）
def project_3d_to_2d_bbox(world_pos, world_size, camera_pose, camera_fov_deg, img_width, img_height):
    """
    改进投影：使用 camera_pose (airsim.Pose) 中的 position 与 orientation。
    在相机坐标系下假定前向为 +X，右为 +Y，向下为 +Z（AirSim / Unreal NED）。
    步骤：
      1. 计算相对向量 rel = P_world - P_cam
      2. 通过 R_cam^T * rel 得到相机坐标 p_cam
      3. 若 p_cam.x <= 0 视为在相机后方
      4. 使用水平 FOV 计算焦距 fx = W / (2*tan(FOV/2))，假定 fy = fx
      5. 像素投影：u = fx * (p_cam.y / p_cam.x) + cx, v = fx * (-p_cam.z / p_cam.x) + cy
      6. 利用物理尺寸近似像素尺寸：size_x_px = fx * world_size[0] / p_cam.x (同理 size_y_px)
    
    改进：
      - 降低最小尺寸阈值（0.5px 而非 1px），捕获远距离目标
      - 改进可见性计算：基于中心点投影有效性 + 部分可见性
      - 支持部分裁剪的 bbox（只要有部分在视野内）
      - 返回可见性值用于后续过滤
    
    返回：bbox 或 None, visibility(0-1), distance
    """
    try:
        cam_pos = camera_pose.position
        cam_quat = camera_pose.orientation
        rel = np.array([
            world_pos[0] - cam_pos.x_val,
            world_pos[1] - cam_pos.y_val,
            world_pos[2] - cam_pos.z_val
        ], dtype=np.float32)

        distance = float(np.linalg.norm(rel))
        if distance < 0.5:  # 太近不稳定
            return None, 0.0, distance

        R_cam = quaternion_to_rotation_matrix(cam_quat)
        # 世界 -> 相机坐标（相机朝向 +X）
        p_cam = R_cam.T.dot(rel)

        forward_x = p_cam[0]
        if forward_x <= 0.01:  # 在后方或几乎共面
            return None, 0.0, distance

        fov_rad = math.radians(camera_fov_deg)
        fx = img_width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # 近似方形像素
        cx = img_width / 2.0
        cy = img_height / 2.0

        u = fx * (p_cam[1] / forward_x) + cx
        v = fy * (-p_cam[2] / forward_x) + cy  # NED z向下：像素 y 方向取负

        # 近似尺寸（使用长宽，忽略高度）
        size_x_px = fx * world_size[0] / forward_x
        size_y_px = fy * world_size[1] / forward_x
        
        # ===== 改进：降低最小尺寸阈值，捕获远距离小目标 =====
        # 0.5 像素的目标仍然有投影意义，可以用于跟踪
        if size_x_px < 0.5 or size_y_px < 0.5:
            return None, 0.0, distance  # 太小，无法投影

        # 计算原始 bbox（未裁剪）
        x_min_raw = u - size_x_px/2.0
        y_min_raw = v - size_y_px/2.0
        x_max_raw = u + size_x_px/2.0
        y_max_raw = v + size_y_px/2.0

        # ===== 改进：允许部分裁剪，只要有部分在图像内 =====
        # 检查是否与图像区域有交集
        x_min = max(0.0, x_min_raw)
        y_min = max(0.0, y_min_raw)
        x_max = min(img_width - 1.0, x_max_raw)
        y_max = min(img_height - 1.0, y_max_raw)

        if x_max < 0 or y_max < 0 or x_min >= img_width or y_min >= img_height:
            # 完全在图像外
            return None, 0.0, distance
        
        if x_max <= x_min or y_max <= y_min:
            # 无有效交集
            return None, 0.0, distance

        # ===== 改进：可见性计算更合理 =====
        # 1. 中心点投影有效性（是否指向相机前方）
        center_valid = 1.0 if (0 <= u < img_width and 0 <= v < img_height) else 0.7
        
        # 2. 可见面积比（bbox 中有多少在图像内）
        clipped_area = (x_max - x_min) * (y_max - y_min)
        original_area = size_x_px * size_y_px
        clipped_ratio = clipped_area / original_area if original_area > 0 else 0
        
        # 3. 距离因子（更宽松：200m 视为满分）
        distance_factor = min(1.0, 200.0 / max(distance, 1.0))
        
        # 综合可见性（中心投影 * 裁剪比例 * 距离因子）
        visibility = float(center_valid * clipped_ratio * distance_factor)
        
        # ===== 改进：即使低可见性也返回 bbox（让导出工具决定是否过滤） =====
        # 这样后续可以在导出时根据阈值灵活过滤
        return (int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))), visibility, distance
    except Exception as e:
        print(f"Projection error: {e}")
        return None, 0.0, 0.0

# 计算集群相对运动特征
def compute_swarm_dynamics(poses, velocities=None):
    """
    计算无人机集群的动力学特征
    
    Returns:
        dict: 包含质心、分散度、密集度、相对速度等信息
    """
    if not poses or len(poses) < 2:
        return {}
        
    # 质心和基本几何特征
    center, radius = compute_swarm_center_and_radius(poses)
    
    # 计算分散度（标准差）
    positions = np.array([[p.position.x_val, p.position.y_val, p.position.z_val] for p in poses])
    dispersion = np.std(positions, axis=0)  # 每个轴的分散度
    avg_dispersion = float(np.mean(dispersion))
    
    # 密集度（相对于边界框的填充率）
    if len(poses) > 1:
        bbox_volume = (np.max(positions, axis=0) - np.min(positions, axis=0))
        bbox_vol = float(np.prod(bbox_volume + 1e-6))  # 避免除零
        density = float(len(poses) / bbox_vol) if bbox_vol > 0 else 0.0
    else:
        density = 0.0
    
    # 相对距离矩阵
    relative_distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            relative_distances.append(dist)
    
    dynamics = {
        'center': center,
        'radius': radius,
        'dispersion': avg_dispersion,
        'density': density,
        'num_drones': len(poses),
        'min_distance': float(min(relative_distances)) if relative_distances else 0.0,
        'max_distance': float(max(relative_distances)) if relative_distances else 0.0,
        'avg_distance': float(np.mean(relative_distances)) if relative_distances else 0.0
    }
    
    return dynamics

# 计算无人机集群的质心与范围（最大半径）
def compute_swarm_center_and_radius(poses):
    if not poses:
        return (0.0, 0.0, FLIGHT_ALT), 0.0
    xs, ys, zs = [], [], []
    for p in poses:
        xs.append(p.position.x_val)
        ys.append(p.position.y_val)
        zs.append(p.position.z_val)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    cz = float(np.mean(zs))
    # 半径取最大到质心的水平距离
    r = 0.0
    for x, y in zip(xs, ys):
        r = max(r, math.hypot(x - cx, y - cy))
    return (cx, cy, cz), r


def _wrap_pi(a):
    """Wrap angle to [-pi, pi)"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _exp_lerp(prev, target, dt, tau):
    """Exponential low-pass: move prev toward target using time-constant tau.
    If tau <= 0, jump to target. dt is elapsed time in seconds.
    """
    if prev is None:
        return target
    if tau is None or tau <= 0.0 or dt <= 0.0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return prev + alpha * (target - prev)

# 将观察者摄像机移动到能覆盖集群的位置，并指向质心 
def update_observer_camera_stable(client, vehicles, strength=2.0):
    """Observer相机控制，减少晃动和距离变化"""
    try:
        if not ENABLE_OBSERVER_CAMERA:
            return
        
        # 获取所有无人机的位姿
        poses = []
        for v in vehicles:
            try:
                poses.append(client.simGetVehiclePose(vehicle_name=v))
            except Exception:
                pass
        
        if not poses:
            return
            
        center, radius = compute_swarm_center_and_radius(poses)
        cx, cy, cz = center

        # 全局状态
        global obs_cam_pos, obs_cam_yaw, obs_cam_pitch, obs_last_t
        global swarm_center_ema

        now = time.perf_counter()
        dt = (now - obs_last_t) if obs_last_t else (1.0 / max(OBSERVER_FPS, 30))
        
        # 平滑质心（减少抖动）
        if swarm_center_ema is None:
            swarm_center_ema = (cx, cy, cz)
        else:
            # 更强的平滑
            tau = CAM_SMOOTH_POS_TAU * 3  # 三倍平滑
            swarm_center_ema = (
                _exp_lerp(swarm_center_ema[0], cx, dt, tau),
                _exp_lerp(swarm_center_ema[1], cy, dt, tau),
                _exp_lerp(swarm_center_ema[2], cz, dt, tau)
            )

        # 固定距离和高度 - 分阶段控制
        global observer_phase_start
        
        # 确定当前观察阶段
        elapsed = time.time() - observer_phase_start if observer_phase_start else 0
        in_ground_phase = elapsed < OBSERVER_PHASE_SWITCH_SEC
        
        if in_ground_phase:
            # 地面/水平观察阶段：相机在地面附近，观察起飞
            # 关键：在 NED 坐标系中，Z=0 是地面，Z<0 是空中
            # 我们希望相机接近地面（Z 接近 0），而集群在空中（Z < 0）
            horiz_dist = float(OBSERVER_GROUND_DIST)  # ~10m 水平距离
            # 相机高度：从地面算起的高度（绝对高度）
            camera_absolute_height = OBSERVER_GROUND_HEIGHT  # 2m 高度
            # 在 NED 中，地面上 2m 的 Z 坐标是 -2
            camera_z_absolute = -camera_absolute_height
        else:
            # 空中观察阶段：相机在集群上方
            horiz_dist = float(OBSERVER_FIXED_DIST)
            # 相机在集群上方 OBSERVER_FIXED_HEIGHT 米
            camera_z_absolute = swarm_center_ema[2] - OBSERVER_FIXED_HEIGHT
        
        # 目标位置
        target_pos = (
            swarm_center_ema[0] - horiz_dist, 
            swarm_center_ema[1], 
            camera_z_absolute  # 使用绝对 Z 坐标
        )

        # 目标朝向（指向集群中心）- 对于地面和空中阶段都动态计算
        dir_x = swarm_center_ema[0] - target_pos[0]
        dir_y = swarm_center_ema[1] - target_pos[1]
        dir_z = swarm_center_ema[2] - target_pos[2]
        target_yaw = math.atan2(dir_y, dir_x)
        
        # 目标俯仰角：两个阶段都动态计算，确保始终指向集群中心
        # 在 NED 坐标系中：
        #   - 相机在 (x, y, z_cam)，集群在 (x', y', z_swarm)
        #   - dir_z = z_swarm - z_cam
        #   - 如果 z_cam < z_swarm (相机在更高处/Z更小)，dir_z > 0，需要向下看
        #   - 如果 z_cam > z_swarm (相机在更低处/Z更大)，dir_z < 0，需要向上看
        # 在 AirSim pitch 约定中：pitch < 0 向下，pitch > 0 向上
        # 因此：target_pitch = atan2(-dir_z, horizontal_dist)
        #   当 dir_z > 0 时，pitch < 0（向下），正确
        #   当 dir_z < 0 时，pitch > 0（向上），正确
        target_pitch = math.atan2(-dir_z, math.hypot(dir_x, dir_y))

        # 初始化位置（首次设置）
        if obs_cam_pos is None:
            obs_cam_pos = target_pos
            obs_cam_yaw = target_yaw
            obs_cam_pitch = target_pitch
        else:
            # 平滑移动到目标位置（更强的平滑）
            pos_tau = CAM_SMOOTH_POS_TAU * 4  # 四倍平滑
            rot_tau = CAM_SMOOTH_ROT_TAU * 4  # 四倍平滑
            
            obs_cam_pos = (
                _exp_lerp(obs_cam_pos[0], target_pos[0], dt, pos_tau),
                _exp_lerp(obs_cam_pos[1], target_pos[1], dt, pos_tau),
                _exp_lerp(obs_cam_pos[2], target_pos[2], dt, pos_tau)
            )
            
            # 平滑旋转
            obs_cam_yaw = _exp_lerp(obs_cam_yaw, target_yaw, dt, rot_tau)
            obs_cam_pitch = _exp_lerp(obs_cam_pitch, target_pitch, dt, rot_tau)

        # 应用相机姿态
        pose = airsim.Pose(
            airsim.Vector3r(obs_cam_pos[0], obs_cam_pos[1], obs_cam_pos[2]),
            airsim.to_quaternion(obs_cam_pitch, 0.0, obs_cam_yaw)
        )
        
        try:
            client.simSetCameraPose(OBSERVER_CAMERA_NAME, pose, vehicle_name=OBSERVER_VEHICLE_NAME)
        except Exception:
            pass

        obs_last_t = now
        
    except Exception:
        # 忽略任何单次更新异常，保持主流程稳定
        pass

# 为每架机分配不同轨迹 - 针对拦截场景优化
PATHS = {
    "Drone1": generate_path("evasive", center=(8, 0), radius=10, alt=FLIGHT_ALT, num_points=80),      # 规避机动
    "Drone2": generate_path("intercept", center=(0, 8), radius=15, alt=FLIGHT_ALT, num_points=60),    # 拦截轨迹  
    "Drone3": generate_path("combat_turn", center=(-5, 8), radius=15, alt=FLIGHT_ALT, num_points=70), # 战斗转弯
    "Drone4": generate_path("swarm_break", center=(8, -5), radius=10, alt=FLIGHT_ALT, num_points=60)  # 编队解散
}

# --- 全局变量 ---
client = None
VEHICLES = []
video_writers = {}
csv_files = {}
csv_writers = {}
CSV_HEADERS = [
    "timestamp_utc", "timestamp_ms", "timestamp_us", "frame_idx",
    "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z",
    "gps_lat", "gps_lon", "gps_alt",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "baro_altitude", "baro_pressure",
    "mag_x", "mag_y", "mag_z"
]

# Observer视角的边界框标注CSV头部
OBSERVER_BBOX_HEADERS = [
    "timestamp_utc", "timestamp_ms", "timestamp_us", "frame_idx",
    "swarm_center_x", "swarm_center_y", "swarm_center_z",
    "swarm_radius", "swarm_dispersion", "swarm_density", "swarm_num_drones",
    "swarm_min_distance", "swarm_max_distance", "swarm_avg_distance"
]

# 为每架无人机动态添加边界框字段
def get_observer_bbox_headers_with_drones(vehicle_names):
    headers = OBSERVER_BBOX_HEADERS.copy()
    for v in vehicle_names:
        headers.extend([
            f"{v}_bbox_x_min", f"{v}_bbox_y_min", f"{v}_bbox_x_max", f"{v}_bbox_y_max",
            f"{v}_visibility", f"{v}_distance", f"{v}_world_x", f"{v}_world_y", f"{v}_world_z"
        ])
    return headers

# --- Observer 专用线程的共享状态 ---
observer_stop_event = None
observer_lock = threading.Lock()
observer_latest_frame = None  # 最新采集到但尚未被编码线程消耗的帧
observer_latest_ts = None     # 该帧的UTC采集时间
observer_capture_thread = None
observer_encode_thread = None
observer_bbox_thread = None   # 新增：后台 bbox 计算线程
observer_frame_buffer = {}    # 新增：缓存帧数据 {frame_idx: {'frame': ..., 'ts': ..., 'done': False}}
observer_buffer_lock = threading.Lock()  # 新增：保护 frame_buffer 的锁
observer_threads_started = False

# Observer 缩放EMA状态
observer_dist_ema = None
observer_hgt_ema = None
observer_phase_start = None  # 分段起始时间

# Cinematic 状态（位置/角度/FOV 平滑）
obs_cam_pos = None    # tuple (x,y,z)
obs_cam_yaw = None
obs_cam_pitch = None
obs_last_t = None
swarm_center_ema = None
swarm_radius_ema = None
obs_fov_deg = None

# 视频文件路径存储（用于多模态编码器）
video_paths = {}

# --- 函数 ---

def initialize_client():
    global client, VEHICLES
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
    
    available = client.listVehicles()
    print("Available vehicles:", available)

    VEHICLES = [v for v in PREFERRED_VEHICLES if v in available]
    if not VEHICLES:
        if available:
            print("Warning: Preferred vehicles not found. Using all available vehicles.")
            VEHICLES = available
        else:
            raise RuntimeError("No vehicles available. Is AirSim running?")
    print("Using vehicles:", VEHICLES)

    # 为路径不存在的车辆创建备用路径
    for v in VEHICLES:
        if v not in PATHS:
            template = list(PATHS.values())[0] if PATHS else [(10, 0, FLIGHT_ALT)]
            # 修复：确保备用路径是元组列表，而不是Vector3r对象列表
            PATHS[v] = [(float(p[0] + np.random.uniform(-5, 5)), float(p[1] + np.random.uniform(-5, 5)), float(p[2])) for p in template]
            print(f"Created fallback path for {v}")

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    for v in VEHICLES:
        for subdir in ["images", "videos", "logs"]:
            os.makedirs(os.path.join(OUT_DIR, v, subdir), exist_ok=True)
    if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO:
        for subdir in ["images", "videos", "logs"]:
            os.makedirs(os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, subdir), exist_ok=True)

def initialize_recorders(img_size=(640, 480), observer_img_size=None):
    global video_paths
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for v in VEHICLES:
        # RGB Video Writer
        video_path = os.path.join(OUT_DIR, v, "videos", f"{v}_{ts}.mp4")
        video_paths[v] = video_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writers[v] = cv2.VideoWriter(video_path, fourcc, CAPTURE_FPS, img_size)
        print(f"[{v}] RGB video writer initialized at {video_path}")
        
        # 多模态视频编码器（如果启用）
        if ENABLE_DRONE_DEPTH_SEGMENTATION:
            # 深度视频编码器 - 使用更兼容的编码器
            depth_path = os.path.join(OUT_DIR, v, "videos", f"{v}_{ts}_depth.avi")
            depth_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writers[f"{v}_depth"] = cv2.VideoWriter(depth_path, depth_fourcc, CAPTURE_FPS, img_size, True)  # 3通道彩色（深度已转为BGR）
            print(f"[{v}] Depth video writer initialized at {depth_path}")
            
            # 分割视频编码器 - 使用更兼容的编码器
            seg_path = os.path.join(OUT_DIR, v, "videos", f"{v}_{ts}_segmentation.avi")
            seg_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writers[f"{v}_segmentation"] = cv2.VideoWriter(seg_path, seg_fourcc, CAPTURE_FPS, img_size)  # 彩色
            print(f"[{v}] Segmentation video writer initialized at {seg_path}")

        # CSV Logger
        csv_path = os.path.join(OUT_DIR, v, "logs", f"{v}_{ts}.csv")
        csv_files[v] = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writers[v] = csv.writer(csv_files[v])
        csv_writers[v].writerow(CSV_HEADERS)
        print(f"[{v}] CSV logger initialized at {csv_path}")

    # 为 Observer 初始化视频与日志（仅当启用）
    if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO:
        # 使用 MJPG + AVI 提升实时性，并采用独立的 OBSERVER_FPS
        obs_size = observer_img_size if observer_img_size else img_size
        video_ext = ".avi"
        video_path = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "videos", f"{OBSERVER_VEHICLE_NAME}_{ts}{video_ext}")
        video_paths[OBSERVER_VEHICLE_NAME] = video_path
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writers[OBSERVER_VEHICLE_NAME] = cv2.VideoWriter(video_path, fourcc, OBSERVER_FPS, obs_size)
        print(f"[{OBSERVER_VEHICLE_NAME}] Video writer initialized at {video_path} (fps={OBSERVER_FPS})")

        # Observer基础日志
        csv_path = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "logs", f"{OBSERVER_VEHICLE_NAME}_{ts}.csv")
        csv_files[OBSERVER_VEHICLE_NAME] = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writers[OBSERVER_VEHICLE_NAME] = csv.writer(csv_files[OBSERVER_VEHICLE_NAME])
        csv_writers[OBSERVER_VEHICLE_NAME].writerow(["timestamp_utc", "timestamp_ms", "timestamp_us", "frame_idx"]) 
        print(f"[{OBSERVER_VEHICLE_NAME}] CSV logger initialized at {csv_path}")
        
        # 边界框标注日志（用于目标检测训练）
        if ENABLE_BBOX_ANNOTATION:
            bbox_csv_path = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "logs", f"{OBSERVER_VEHICLE_NAME}_bbox_{ts}.csv")
            csv_files[f"{OBSERVER_VEHICLE_NAME}_bbox"] = open(bbox_csv_path, 'w', newline='', encoding='utf-8')
            csv_writers[f"{OBSERVER_VEHICLE_NAME}_bbox"] = csv.writer(csv_files[f"{OBSERVER_VEHICLE_NAME}_bbox"])
            bbox_headers = get_observer_bbox_headers_with_drones(VEHICLES)
            csv_writers[f"{OBSERVER_VEHICLE_NAME}_bbox"].writerow(bbox_headers)
            print(f"[{OBSERVER_VEHICLE_NAME}] Bbox annotation logger initialized at {bbox_csv_path}")

def _observer_capture_loop(dt):
    global observer_latest_frame, observer_latest_ts
    # 在专用线程内使用独立的 AirSim 客户端，避免与主线程RPC冲突
    try:
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
    except Exception:
        local_client = None
    # 采用固定节拍进行采集；若 simGetImages 本身较慢，会自然降频
    next_t = time.perf_counter()
    while not observer_stop_event.is_set():
        try:
            # 若本地客户端不可用则跳过本轮
            if local_client is None:
                raise RuntimeError("Observer local AirSim client not available")
            
            # 多模态图像采集：RGB + 深度 + 分割
            image_requests = [
                airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.Scene, pixels_as_float=False, compress=OBSERVER_COMPRESS),
            ]
            
            if ENABLE_DEPTH_SEGMENTATION:
                image_requests.extend([
                    airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
                    airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)
                ])
            
            resp = local_client.simGetImages(image_requests, vehicle_name=OBSERVER_VEHICLE_NAME)
            
            if resp and len(resp) > 0:
                # RGB帧
                rgb_frame = frame_from_response(resp[0]) if resp[0] else None
                depth_frame = None
                seg_frame = None
                
                if ENABLE_DEPTH_SEGMENTATION and len(resp) >= 3:
                    # 深度图处理
                    if resp[1] and resp[1].image_data_float:
                        depth_data = np.array(resp[1].image_data_float, dtype=np.float32)
                        depth_frame = depth_data.reshape(resp[1].height, resp[1].width)
                        # 将深度转换为可视化图像（伪彩色）
                        depth_normalized = np.clip(depth_frame / 100.0, 0, 1)  # 假设100米为最大深度
                        depth_frame = (depth_normalized * 255).astype(np.uint8)
                        # 确保深度图像是3通道的（复制灰度通道为RGB）
                        if len(depth_frame.shape) == 2:
                            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                    
                    # 分割掩码处理
                    if resp[2]:
                        seg_frame = frame_from_response(resp[2])
                
                if rgb_frame is not None:
                    with observer_lock:
                        # 存储多模态数据
                        observer_latest_frame = {
                            'rgb': rgb_frame,
                            'depth': depth_frame,
                            'segmentation': seg_frame,
                            'timestamp': datetime.datetime.now(datetime.timezone.utc)
                        }
                        observer_latest_ts = observer_latest_frame['timestamp']
        except Exception:
            # 采集错误不致命，继续尝试
            pass
        # 基于目标帧率的简单节拍
        next_t += dt
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # 跑不满就重置节拍，避免持续漂移
            next_t = time.perf_counter()

def _observer_bbox_compute_loop(dt):
    """后台线程：异步计算 bbox，不阻塞视频编码。确保视频始终按恒定帧率输出。"""
    try:
        bbox_client = airsim.MultirotorClient()
        bbox_client.confirmConnection()
    except Exception:
        bbox_client = None
    
    next_t = time.perf_counter()
    while not observer_stop_event.is_set():
        try:
            # 从缓存中取出待处理的帧
            with observer_buffer_lock:
                # 找到第一个 'done'=False 的帧
                pending_frame_idx = None
                for fidx in sorted(observer_frame_buffer.keys()):
                    if not observer_frame_buffer[fidx].get('done', False):
                        pending_frame_idx = fidx
                        break
                
                if pending_frame_idx is None:
                    # 暂无待处理帧，略过此轮
                    pass
                else:
                    frame_entry = observer_frame_buffer[pending_frame_idx]
                    frame_idx = pending_frame_idx
                    ts_utc = frame_entry['ts_utc']
                    ts_ms = frame_entry['ts_ms']
                    ts_us = frame_entry['ts_us']
                    frame_data = frame_entry['frame']
            
            if pending_frame_idx is not None and bbox_client is not None and ENABLE_BBOX_ANNOTATION:
                try:
                    # 获取相机信息与FOV
                    cam_info = bbox_client.simGetCameraInfo(OBSERVER_CAMERA_NAME, vehicle_name=OBSERVER_VEHICLE_NAME)
                    camera_pose = cam_info.pose
                    current_fov = None
                    try:
                        if hasattr(cam_info, 'fov') and cam_info.fov:
                            current_fov = float(cam_info.fov)
                    except Exception:
                        current_fov = None
                    if not current_fov:
                        current_fov = OBSERVER_FOV_MAX

                    # 获取图像尺寸
                    if frame_data is not None and isinstance(frame_data, dict) and 'rgb' in frame_data and frame_data['rgb'] is not None:
                        img_h, img_w = frame_data['rgb'].shape[:2]
                    else:
                        img_w, img_h = 1280, 720

                    # 收集无人机位姿
                    drone_poses = []
                    for v in VEHICLES:
                        try:
                            pose = bbox_client.simGetVehiclePose(vehicle_name=v)
                            drone_poses.append((v, pose))
                        except Exception:
                            pass

                    # 群体动力学
                    poses_only = [p for (_, p) in drone_poses]
                    swarm_dynamics = compute_swarm_dynamics(poses_only)

                    # 组装bbox行
                    bbox_row = [
                        ts_utc, ts_ms, ts_us, frame_idx,
                        swarm_dynamics.get('center', (0,0,0))[0],
                        swarm_dynamics.get('center', (0,0,0))[1],
                        swarm_dynamics.get('center', (0,0,0))[2],
                        swarm_dynamics.get('radius', 0),
                        swarm_dynamics.get('dispersion', 0),
                        swarm_dynamics.get('density', 0),
                        swarm_dynamics.get('num_drones', 0),
                        swarm_dynamics.get('min_distance', 0),
                        swarm_dynamics.get('max_distance', 0),
                        swarm_dynamics.get('avg_distance', 0)
                    ]

                    for v in VEHICLES:
                        # 查找位姿
                        drone_pose = None
                        for name, pose in drone_poses:
                            if name == v:
                                drone_pose = pose
                                break
                        if drone_pose is None:
                            bbox_row.extend([-1, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0])
                            continue

                        world_pos = (drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val)
                        bbox_2d, visibility, distance = project_3d_to_2d_bbox(world_pos, DRONE_BBOX_SIZE, camera_pose, current_fov, img_w, img_h)
                        if bbox_2d:
                            bbox_row.extend([bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3], visibility, distance, world_pos[0], world_pos[1], world_pos[2]])
                        else:
                            bbox_row.extend([-1, -1, -1, -1, 0.0, distance, world_pos[0], world_pos[1], world_pos[2]])

                    # 写入 CSV
                    if f"{OBSERVER_VEHICLE_NAME}_bbox" in csv_writers:
                        csv_writers[f"{OBSERVER_VEHICLE_NAME}_bbox"].writerow(bbox_row)
                        if frame_idx % 60 == 0:
                            try:
                                csv_files[f"{OBSERVER_VEHICLE_NAME}_bbox"].flush()
                            except Exception:
                                pass
                    
                    # 标记为已处理
                    with observer_buffer_lock:
                        if pending_frame_idx in observer_frame_buffer:
                            observer_frame_buffer[pending_frame_idx]['done'] = True
                            
                except Exception as e:
                    # bbox 计算错误不应阻塞，标记为已处理以继续
                    with observer_buffer_lock:
                        if pending_frame_idx in observer_frame_buffer:
                            observer_frame_buffer[pending_frame_idx]['done'] = True
                    
        except Exception:
            pass
        
        # 基于目标帧率的节拍（但 bbox 计算通常较慢，可能跟不上）
        next_t += dt * 2  # 降低优先级，2倍时间间隔
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.perf_counter()

def _observer_encode_loop(dt):
    frame_idx = 0
    prev_frame = None
    next_t = time.perf_counter()
    # 在编码线程内也创建一个本地客户端，用于逐帧计算 bbox（确保每一帧都有详细标注）
    try:
        enc_client = airsim.MultirotorClient()
        enc_client.confirmConnection()
    except Exception:
        enc_client = None
    
    # 为多模态数据创建额外的视频编码器
    depth_writer = None
    seg_writer = None
    if ENABLE_DEPTH_SEGMENTATION and OBSERVER_VEHICLE_NAME in video_writers:
        if OBSERVER_VEHICLE_NAME in video_paths:
            base_path = os.path.splitext(video_paths[OBSERVER_VEHICLE_NAME])[0]
            depth_path = f"{base_path}_depth.avi"
            seg_path = f"{base_path}_segmentation.avi"
            
            # 获取Observer视频的尺寸信息
            obs_size = (1280, 720)  # 默认Observer尺寸
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            depth_writer = cv2.VideoWriter(depth_path, fourcc, OBSERVER_FPS, obs_size, True)  # 3通道彩色（深度已转为BGR）
            seg_writer = cv2.VideoWriter(seg_path, fourcc, OBSERVER_FPS, obs_size)  # 彩色
    
    while not observer_stop_event.is_set():
        # 使用最新帧；若暂无新帧则复用上一帧，保证CFR输出
        with observer_lock:
            frame_data = observer_latest_frame if observer_latest_frame is not None else prev_frame
            
        if frame_data is not None:
            # 处理RGB帧
            if isinstance(frame_data, dict) and 'rgb' in frame_data:
                rgb_frame = frame_data['rgb']
                depth_frame = frame_data.get('depth')
                seg_frame = frame_data.get('segmentation')
            else:
                # 兼容旧格式（直接是numpy数组）
                rgb_frame = frame_data
                depth_frame = None
                seg_frame = None
            
            # 写入RGB视频（更健壮的写入：检查writer、格式并捕获异常）
            if rgb_frame is not None and OBSERVER_VEHICLE_NAME in video_writers:
                try:
                    writer = video_writers[OBSERVER_VEHICLE_NAME]
                    # 检查 writer 是否打开
                    if hasattr(writer, 'isOpened') and not writer.isOpened():
                        # 尝试重建writer（保守方式）
                        if OBSERVER_VEHICLE_NAME in video_paths:
                            path = video_paths[OBSERVER_VEHICLE_NAME]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            obs_h, obs_w = rgb_frame.shape[0], rgb_frame.shape[1]
                            try:
                                new_writer = cv2.VideoWriter(path, fourcc, OBSERVER_FPS, (obs_w, obs_h))
                                video_writers[OBSERVER_VEHICLE_NAME] = new_writer
                                writer = new_writer
                                print(f"Reinitialized Observer VideoWriter at {path}")
                            except Exception as e:
                                print(f"Failed to reinitialize Observer writer: {e}")

                    # 确保图像类型为uint8且为3通道BGR
                    if not isinstance(rgb_frame, np.ndarray):
                        raise TypeError("rgb_frame is not a numpy array")
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
                    if rgb_frame.ndim == 2:
                        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
                    elif rgb_frame.shape[2] == 4:
                        # 若意外是4通道，转换为BGR
                        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)

                    # 尺寸调整：若与常用Observer分辨率不同，调整到写入器期望的分辨率（默认1280x720）
                    try:
                        expected_size = (1280, 720)
                        if (rgb_frame.shape[1], rgb_frame.shape[0]) != expected_size:
                            rgb_frame = cv2.resize(rgb_frame, expected_size)
                    except Exception:
                        pass

                    writer.write(rgb_frame)
                except Exception as e:
                    print(f"Observer video write error at frame {frame_idx}: {e}")
                
            # 写入深度视频
            if depth_frame is not None and depth_writer is not None:
                # 确保尺寸匹配
                if depth_frame.shape[:2] == (720, 1280):  # (height, width)
                    depth_writer.write(depth_frame)
                else:
                    # 调整尺寸
                    depth_resized = cv2.resize(depth_frame, (1280, 720))
                    depth_writer.write(depth_resized)
                
            # 写入分割视频
            if seg_frame is not None and seg_writer is not None:
                # 确保尺寸匹配
                if seg_frame.shape[:2] == (720, 1280):  # (height, width)
                    seg_writer.write(seg_frame)
                else:
                    # 调整尺寸
                    seg_resized = cv2.resize(seg_frame, (1280, 720))
                    seg_writer.write(seg_resized)
                
            prev_frame = frame_data
            
        # 写Observer的时间戳日志（与视频帧一一对应），并在同一时间戳下写入bbox
        if OBSERVER_VEHICLE_NAME in csv_writers:
            utc_now = datetime.datetime.now(datetime.timezone.utc)
            ts_utc = utc_now.isoformat()
            epoch_ns = time.time_ns()
            ts_ms = int(epoch_ns // 1_000_000)
            ts_us = int(epoch_ns // 1_000)
            csv_writers[OBSERVER_VEHICLE_NAME].writerow([ts_utc, ts_ms, ts_us, frame_idx])
            if OBSERVER_VEHICLE_NAME in csv_files and frame_idx % 60 == 0:
                try:
                    csv_files[OBSERVER_VEHICLE_NAME].flush()
                except Exception:
                    pass

            # 每帧写入 bbox（若启用），保证“真实每帧详细信息”
            if ENABLE_BBOX_ANNOTATION and f"{OBSERVER_VEHICLE_NAME}_bbox" in csv_writers and enc_client is not None:
                try:
                    # 获取相机信息与FOV
                    cam_info = enc_client.simGetCameraInfo(OBSERVER_CAMERA_NAME, vehicle_name=OBSERVER_VEHICLE_NAME)
                    camera_pose = cam_info.pose
                    current_fov = None
                    try:
                        if hasattr(cam_info, 'fov') and cam_info.fov:
                            current_fov = float(cam_info.fov)
                    except Exception:
                        current_fov = None
                    if not current_fov:
                        current_fov = OBSERVER_FOV_MAX

                    # 获取图像尺寸（优先从当前帧）
                    if frame_data is not None and isinstance(frame_data, dict) and 'rgb' in frame_data and frame_data['rgb'] is not None:
                        img_h, img_w = frame_data['rgb'].shape[:2]
                    else:
                        img_w, img_h = 1280, 720

                    # 收集无人机位姿
                    drone_poses = []
                    for v in VEHICLES:
                        try:
                            pose = enc_client.simGetVehiclePose(vehicle_name=v)
                            drone_poses.append((v, pose))
                        except Exception:
                            pass

                    # 群体动力学
                    poses_only = [p for (_, p) in drone_poses]
                    swarm_dynamics = compute_swarm_dynamics(poses_only)

                    # 组装bbox行
                    bbox_row = [
                        ts_utc, ts_ms, ts_us, frame_idx,
                        swarm_dynamics.get('center', (0,0,0))[0],
                        swarm_dynamics.get('center', (0,0,0))[1],
                        swarm_dynamics.get('center', (0,0,0))[2],
                        swarm_dynamics.get('radius', 0),
                        swarm_dynamics.get('dispersion', 0),
                        swarm_dynamics.get('density', 0),
                        swarm_dynamics.get('num_drones', 0),
                        swarm_dynamics.get('min_distance', 0),
                        swarm_dynamics.get('max_distance', 0),
                        swarm_dynamics.get('avg_distance', 0)
                    ]

                    for v in VEHICLES:
                        # 查找位姿
                        drone_pose = None
                        for name, pose in drone_poses:
                            if name == v:
                                drone_pose = pose
                                break
                        if drone_pose is None:
                            bbox_row.extend([-1, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0])
                            continue

                        world_pos = (drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val)
                        bbox_2d, visibility, distance = project_3d_to_2d_bbox(world_pos, DRONE_BBOX_SIZE, camera_pose, current_fov, img_w, img_h)
                        if bbox_2d:
                            bbox_row.extend([bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3], visibility, distance, world_pos[0], world_pos[1], world_pos[2]])
                        else:
                            bbox_row.extend([-1, -1, -1, -1, 0.0, distance, world_pos[0], world_pos[1], world_pos[2]])

                    csv_writers[f"{OBSERVER_VEHICLE_NAME}_bbox"].writerow(bbox_row)
                    # 定期flush
                    key = f"{OBSERVER_VEHICLE_NAME}_bbox"
                    if key in csv_files and frame_idx % 60 == 0:
                        try:
                            csv_files[key].flush()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Observer per-frame bbox write error at frame {frame_idx}: {e}")

        frame_idx += 1
        # 固定节拍输出
        next_t += dt
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # 写入线程落后时重置节拍，避免连续负漂移
            next_t = time.perf_counter()
    
    # 清理多模态视频编码器
    if depth_writer is not None:
        depth_writer.release()
    if seg_writer is not None:
        seg_writer.release()

def start_observer_pipeline():
    """启动Observer的专用采集与编码线程，确保CFR并尽可能贴近"真实相机"体验。"""
    global observer_stop_event, observer_capture_thread, observer_encode_thread, observer_threads_started
    if not (ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO and OBSERVER_DEDICATED_THREAD):
        return
    if observer_threads_started:
        return
    observer_stop_event = threading.Event()
    dt = 1.0 / OBSERVER_FPS if OBSERVER_FPS > 0 else 1.0 / 20.0
    observer_capture_thread = threading.Thread(target=_observer_capture_loop, args=(dt,), name="ObserverCapture", daemon=True)
    observer_encode_thread = threading.Thread(target=_observer_encode_loop, args=(dt,), name="ObserverEncode", daemon=True)
    observer_capture_thread.start()
    observer_encode_thread.start()
    observer_threads_started = True
    
    # 初始化分阶段观察起始时间
    global observer_phase_start
    observer_phase_start = time.time()

def stop_observer_pipeline():
    global observer_threads_started
    if not observer_threads_started:
        return
    if observer_stop_event:
        observer_stop_event.set()
    if observer_capture_thread:
        observer_capture_thread.join(timeout=2.0)
    if observer_encode_thread:
        observer_encode_thread.join(timeout=2.0)
    observer_threads_started = False

def cleanup_recorders():
    for writer in video_writers.values():
        writer.release()
    for f in csv_files.values():
        f.close()
    print("All recorders cleaned up.")


def export_dataset_summary(base_output_dir=OUT_DIR):
    """生成数据集摘要信息"""
    summary = {
        "export_time": datetime.datetime.now().isoformat(),
        "vehicles": VEHICLES,
        "observer_enabled": ENABLE_OBSERVER_CAMERA,
        "capture_fps": CAPTURE_FPS,
        "observer_fps": OBSERVER_FPS if ENABLE_OBSERVER_CAMERA else None,
        "depth_segmentation_enabled": ENABLE_DEPTH_SEGMENTATION,
        "drone_depth_segmentation_enabled": ENABLE_DRONE_DEPTH_SEGMENTATION,
        "flight_altitude": FLIGHT_ALT,
        "flight_speed": FLIGHT_SPEED,
        "bbox_size": DRONE_BBOX_SIZE,
        "sequences": []
    }
    
    # 扫描输出目录收集序列信息
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path) and item in VEHICLES + [OBSERVER_VEHICLE_NAME]:
            video_dir = os.path.join(item_path, "videos")
            log_dir = os.path.join(item_path, "logs")
            
            sequence_info = {
                "name": item,
                "type": "observer" if item == OBSERVER_VEHICLE_NAME else "drone",
                "video_files": [],
                "log_files": []
            }
            
            if os.path.exists(video_dir):
                sequence_info["video_files"] = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            
            if os.path.exists(log_dir):
                sequence_info["log_files"] = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
            
            summary["sequences"].append(sequence_info)
    
    # 保存摘要文件
    summary_file = os.path.join(base_output_dir, "dataset_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"数据集摘要已保存至: {summary_file}")
    return summary

def frame_from_response(resp):
    """从 AirSim ImageResponse 中解析为 BGR 图像（numpy 数组）。
    优先处理未压缩三通道/四通道的情况，回退到解码。
    """
    try:
        if not (hasattr(resp, 'image_data_uint8') and resp.image_data_uint8):
            return None
        buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        # 优先未压缩路径
        if hasattr(resp, 'width') and hasattr(resp, 'height') and resp.width and resp.height:
            if buf.size == resp.width * resp.height * 3:
                return buf.reshape((resp.height, resp.width, 3))
            if buf.size == resp.width * resp.height * 4:
                img4 = buf.reshape((resp.height, resp.width, 4))
                return cv2.cvtColor(img4, cv2.COLOR_BGRA2BGR)
        # 回退：按压缩图像解码
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def arm_and_takeoff():
    print("Arming and taking off...")
    futures = []
    for v in VEHICLES:
        client.enableApiControl(True, vehicle_name=v)
        client.armDisarm(True, vehicle_name=v)
        futures.append(client.takeoffAsync(vehicle_name=v))
    
    for i, f in enumerate(futures):
        f.join()
        print(f"[{VEHICLES[i]}] has taken off.")
    
    print("All drones are airborne. Moving to initial positions...")
    move_futures = [client.moveToPositionAsync(PATHS[v][0][0], PATHS[v][0][1], PATHS[v][0][2], FLIGHT_SPEED, vehicle_name=v) for v in VEHICLES]
    for i, f in enumerate(move_futures):
        f.join()
        print(f"[{VEHICLES[i]}] reached initial waypoint.")

def follow_paths_and_capture(duration_seconds):
    start_time = time.time()
    dt = 1.0 / CAPTURE_FPS
    frame_idx = 0
    path_indices = {v: 0 for v in VEHICLES}
    # 记录上一帧写入时间与最后一帧（用于必要时补帧）
    last_tick_time = time.time()
    last_frames = {v: None for v in VEHICLES}
    last_frames[OBSERVER_VEHICLE_NAME] = None
    # 缓存每个无人机的上一行传感器数据（用于CFR补帧时复制）
    last_sensor_rows = {v: None for v in VEHICLES}

    while time.time() - start_time < duration_seconds:
        tick_start = time.time()
        # 统一本帧的时间戳（毫秒/微秒）
        epoch_ns = time.time_ns()
        ts_ms_frame = int(epoch_ns // 1_000_000)
        ts_us_frame = int(epoch_ns // 1_000)
        # 构建多模态图像采集请求
        requests = [airsim.ImageRequest(CAM_IDX, airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
        if ENABLE_DRONE_DEPTH_SEGMENTATION:
            requests.extend([
                airsim.ImageRequest(CAM_IDX, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
                airsim.ImageRequest(CAM_IDX, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)
            ])
        future_responses = {}
        for v in VEHICLES:
            sensor_this_frame = (frame_idx % SENSOR_LOG_EVERY_N == 0)
            future_responses[v] = {
                "image": client.simGetImages(requests, vehicle_name=v),
                "state": client.getMultirotorState(vehicle_name=v)
            }
            if sensor_this_frame:
                future_responses[v]["imu"] = client.getImuData(vehicle_name=v)
                future_responses[v]["baro"] = client.getBarometerData(vehicle_name=v)
                future_responses[v]["mag"] = client.getMagnetometerData(vehicle_name=v)
                future_responses[v]["gps"] = client.getGpsData(vehicle_name=v)

            # 采集 Observer 图像（如启用）
            observer_resp = None
            if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO and not OBSERVER_DEDICATED_THREAD:
                observer_resp = client.simGetImages(
                    [airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.Scene, pixels_as_float=False, compress=OBSERVER_COMPRESS)],
                    vehicle_name=OBSERVER_VEHICLE_NAME
                )

        for v in VEHICLES:
            img_resp = future_responses[v]["image"]
            state = future_responses[v]["state"]
            imu = future_responses[v].get("imu")
            baro = future_responses[v].get("baro")
            mag = future_responses[v].get("mag")
            gps = future_responses[v].get("gps")

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            ts_utc = utc_now.isoformat()
            ts_ms = ts_ms_frame
            ts_us = ts_us_frame

            # 保存多模态图像数据并写入视频
            if img_resp and img_resp[0]:
                # RGB图像处理
                frame = frame_from_response(img_resp[0])
                depth_frame = None
                seg_frame = None
                
                # 处理深度和分割数据
                if ENABLE_DRONE_DEPTH_SEGMENTATION and len(img_resp) >= 3:
                    # 深度图处理
                    if img_resp[1] and img_resp[1].image_data_float:
                        depth_data = np.array(img_resp[1].image_data_float, dtype=np.float32)
                        depth_frame = depth_data.reshape(img_resp[1].height, img_resp[1].width)
                        # 将深度转换为可视化图像
                        depth_normalized = np.clip(depth_frame / 100.0, 0, 1)
                        depth_frame = (depth_normalized * 255).astype(np.uint8)
                        # 确保深度图像是3通道的（复制灰度通道为RGB）
                        if len(depth_frame.shape) == 2:
                            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                    
                    # 分割掩码处理
                    if img_resp[2]:
                        seg_frame = frame_from_response(img_resp[2])
                
                # 写入RGB视频
                if frame is not None and v in video_writers:
                    video_writers[v].write(frame)
                    last_frames[v] = frame
                    
                # 写入深度和分割视频（如果启用）
                if ENABLE_DRONE_DEPTH_SEGMENTATION:
                    if depth_frame is not None and f"{v}_depth" in video_writers:
                        # 确保尺寸匹配（无人机默认640x480）
                        if depth_frame.shape[:2] == (480, 640):  # (height, width)
                            video_writers[f"{v}_depth"].write(depth_frame)
                        else:
                            depth_resized = cv2.resize(depth_frame, (640, 480))
                            video_writers[f"{v}_depth"].write(depth_resized)
                    if seg_frame is not None and f"{v}_segmentation" in video_writers:
                        # 确保尺寸匹配
                        if seg_frame.shape[:2] == (480, 640):  # (height, width)
                            video_writers[f"{v}_segmentation"].write(seg_frame)
                        else:
                            seg_resized = cv2.resize(seg_frame, (640, 480))
                            video_writers[f"{v}_segmentation"].write(seg_resized)
                    if SAVE_PNG_EVERY_N and (frame_idx % SAVE_PNG_EVERY_N == 0):
                        img_fname = os.path.join(OUT_DIR, v, "images", f"{frame_idx:06d}.png")
                        cv2.imwrite(img_fname, frame)

            # 记录CSV（对未取样传感器使用 NaN 占位，避免 NoneType 访问）
            row = [
                ts_utc, ts_ms, ts_us, frame_idx,
                state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val,
                state.kinematics_estimated.orientation.w_val, state.kinematics_estimated.orientation.x_val, state.kinematics_estimated.orientation.y_val, state.kinematics_estimated.orientation.z_val,
                (gps.gnss.geo_point.latitude if gps else float('nan')),
                (gps.gnss.geo_point.longitude if gps else float('nan')),
                (gps.gnss.geo_point.altitude if gps else float('nan')),
                (imu.linear_acceleration.x_val if imu else float('nan')),
                (imu.linear_acceleration.y_val if imu else float('nan')),
                (imu.linear_acceleration.z_val if imu else float('nan')),
                (imu.angular_velocity.x_val if imu else float('nan')),
                (imu.angular_velocity.y_val if imu else float('nan')),
                (imu.angular_velocity.z_val if imu else float('nan')),
                (baro.altitude if baro else float('nan')),
                (baro.pressure if baro else float('nan')),
                (mag.magnetic_field_body.x_val if mag else float('nan')),
                (mag.magnetic_field_body.y_val if mag else float('nan')),
                (mag.magnetic_field_body.z_val if mag else float('nan'))
            ]
            if v in csv_writers:
                csv_writers[v].writerow(row)
                # 缓存本行用于CFR补帧（除时间戳和frame_idx外的传感器数据）
                if v not in last_sensor_rows:
                    last_sensor_rows[v] = {}
                last_sensor_rows[v] = row[4:]  # 保存位置、姿态、GPS、IMU等所有传感器数据
                # 降低缓存风险：每隔30帧刷新一次
                if frame_idx % 30 == 0 and v in csv_files:
                    try:
                        csv_files[v].flush()
                    except Exception:
                        pass

        # 写入 Observer 视频与时间戳（如启用）
        if (observer_resp and observer_resp[0]) and (not OBSERVER_DEDICATED_THREAD):
            frame = frame_from_response(observer_resp[0])
            if frame is not None and OBSERVER_VEHICLE_NAME in video_writers:
                try:
                    writer = video_writers[OBSERVER_VEHICLE_NAME]
                    if hasattr(writer, 'isOpened') and not writer.isOpened():
                        if OBSERVER_VEHICLE_NAME in video_paths:
                            path = video_paths[OBSERVER_VEHICLE_NAME]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            h, w = frame.shape[0], frame.shape[1]
                            try:
                                new_writer = cv2.VideoWriter(path, fourcc, OBSERVER_FPS, (w, h))
                                video_writers[OBSERVER_VEHICLE_NAME] = new_writer
                                writer = new_writer
                                print(f"Reinitialized Observer VideoWriter at {path}")
                            except Exception as e:
                                print(f"Failed to reinitialize Observer writer: {e}")

                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    try:
                        expected_size = (1280, 720)
                        if (frame.shape[1], frame.shape[0]) != expected_size:
                            frame = cv2.resize(frame, expected_size)
                    except Exception:
                        pass
                    writer.write(frame)
                except Exception as e:
                    print(f"Observer video write error (main loop) at frame {frame_idx}: {e}")
                last_frames[OBSERVER_VEHICLE_NAME] = frame
                if SAVE_PNG_EVERY_N and (frame_idx % SAVE_PNG_EVERY_N == 0):
                    img_fname = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "images", f"{frame_idx:06d}.png")
                    cv2.imwrite(img_fname, frame)
            # 日志仅写时间戳/帧号
            if (OBSERVER_VEHICLE_NAME in csv_writers):
                utc_now = datetime.datetime.now(datetime.timezone.utc)
                ts_utc = utc_now.isoformat()
                epoch_ns2 = time.time_ns()
                ts_ms2 = int(epoch_ns2 // 1_000_000)
                ts_us2 = int(epoch_ns2 // 1_000)
                csv_writers[OBSERVER_VEHICLE_NAME].writerow([ts_utc, ts_ms2, ts_us2, frame_idx])
                if OBSERVER_VEHICLE_NAME in csv_files and frame_idx % 60 == 0:
                    try:
                        csv_files[OBSERVER_VEHICLE_NAME].flush()
                    except Exception:
                        pass
        # 动态计算目标点，实现复杂运动
        for v in VEHICLES:
            # 每帧都更新目标点，实现平滑运动
            idx = (frame_idx + abs(hash(v))) % len(PATHS[v])
            next_pos = PATHS[v][idx]
            
            # === 拦截场景特化的机动动作(并行发送命令) ===
            elapsed_time = time.time() - start_time
            
            # Drone1: 规避机动 - 随机急转弯和突然加减速
            if v == "Drone1":
                if frame_idx % 45 == 0:  # 每2.25秒一次急转弯
                    # 随机急转弯
                    turn_angle = float(np.random.uniform(-90, 90))  # 大角度转弯
                    client.rotateByYawRateAsync(turn_angle * 2, 1.0, vehicle_name=v)
                elif frame_idx % 60 == 30:  # 突然加速
                    accel_vel = float(np.random.uniform(8, 15))  # 高速机动
                    rand_direction = float(np.random.uniform(0, 2*math.pi))
                    vx = accel_vel * math.cos(rand_direction)
                    vy = accel_vel * math.sin(rand_direction)
                    client.moveByVelocityAsync(vx, vy, 0, 1.5, vehicle_name=v)
                else:
                    # 正常轨迹跟随但速度更高
                    client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED * 1.8, vehicle_name=v)
            
            # Drone2: 拦截者 - 高速直线冲刺和悬停
            elif v == "Drone2":
                if frame_idx % 100 == 0:  # 每5秒一次拦截冲刺
                    # 计算目标方向(朝向集群中心) - 使用已获取的state数据而非额外RPC
                    try:
                        positions = []
                        for vv in VEHICLES:
                            if vv != v and vv in future_responses:
                                resp_state = future_responses[vv]["state"]
                                positions.append((
                                    resp_state.kinematics_estimated.position.x_val,
                                    resp_state.kinematics_estimated.position.y_val,
                                    resp_state.kinematics_estimated.position.z_val
                                ))
                        if positions:
                            center = tuple(sum(x)/len(positions) for x in zip(*positions))
                            current_pos = future_responses[v]["state"].kinematics_estimated.position
                            dx = center[0] - current_pos.x_val
                            dy = center[1] - current_pos.y_val
                            norm = math.sqrt(dx**2 + dy**2)
                            if norm > 0:
                                intercept_speed = 20.0  # 拦截速度
                                vx = intercept_speed * dx / norm
                                vy = intercept_speed * dy / norm
                                client.moveByVelocityAsync(vx, vy, -2, 3.0, vehicle_name=v)
                        else:
                            client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED * 1.5, vehicle_name=v)
                    except Exception:
                        client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED * 1.5, vehicle_name=v)
                elif frame_idx % 100 == 80:  # 冲刺后短暂悬停
                    client.hoverAsync(vehicle_name=v)
                else:
                    client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED * 1.5, vehicle_name=v)
            
            # Drone3: 战斗机动 - 垂直机动和滚转
            elif v == "Drone3":
                if frame_idx % 80 == 0:  # 垂直机动
                    climb_speed = float(np.random.choice([-8, 8]))  # 快速爬升或俯冲
                    client.moveByVelocityAsync(0, 0, climb_speed, 1.0, vehicle_name=v)
                elif frame_idx % 40 == 20:  # 高速转弯
                    yaw_rate = float(np.random.uniform(-180, 180))  # 高角速度转弯
                    client.rotateByYawRateAsync(yaw_rate, 0.8, vehicle_name=v)
                else:
                    client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED * 1.3, vehicle_name=v)
            
            # Drone4: 编队机动 - 协同和分离
            elif v == "Drone4":
                if frame_idx % 120 == 0:  # 编队分离
                    # 远离其他无人机
                    separation_force = (0, 0, 0)
                    try:
                        current_pos = client.simGetVehiclePose(vehicle_name=v).position
                        for other_v in VEHICLES:
                            if other_v != v:
                                other_pos = client.simGetVehiclePose(vehicle_name=other_v).position
                                dx = current_pos.x_val - other_pos.x_val
                                dy = current_pos.y_val - other_pos.y_val
                                dist = math.sqrt(dx**2 + dy**2)
                                if dist > 0:
                                    # 排斥力
                                    separation_force = (
                                        separation_force[0] + dx/dist * 5,
                                        separation_force[1] + dy/dist * 5,
                                        separation_force[2]
                                    )
                        if abs(separation_force[0]) > 0 or abs(separation_force[1]) > 0:
                            client.moveByVelocityAsync(separation_force[0], separation_force[1], 0, 2.0, vehicle_name=v)
                    except Exception:
                        pass
                else:
                    client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED, vehicle_name=v)
            
            else:
                # 默认行为
                client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED, vehicle_name=v)

        # 每隔若干帧更新一次观察者摄像机位置，以降低开销
        if ENABLE_OBSERVER_CAMERA and (frame_idx % OBSERVER_UPDATE_EVERY_N == 0):
            update_observer_camera_stable(client, VEHICLES, strength=2.2)

        # 如实际循环间隔大于目标dt，则按比例补写上一帧，尽量与真实时间一致
        if MATCH_REALTIME_OUTPUT:
            now = time.time()
            loop_dt = now - last_tick_time
            dup = int(round(loop_dt / dt)) - 1
            
            if dup > 0:
                dup = min(dup, MAX_DUP_FRAMES_PER_TICK)
                for _ in range(dup):
                    # 为补帧生成新的时间戳，复制上一帧图像和传感器数据
                    epoch_ns_dup = time.time_ns()
                    ts_ms_dup = int(epoch_ns_dup // 1_000_000)
                    ts_us_dup = int(epoch_ns_dup // 1_000)
                    ts_utc_dup = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    
                    for v in VEHICLES:
                        # 复制视频帧
                        lf = last_frames.get(v)
                        if lf is not None and v in video_writers:
                            video_writers[v].write(lf)
                        
                        # 复制传感器CSV行（使用缓存的传感器数据 + 新时间戳）
                        if v in csv_writers and last_sensor_rows.get(v) is not None:
                            dup_row = [ts_utc_dup, ts_ms_dup, ts_us_dup, frame_idx] + list(last_sensor_rows[v])
                            csv_writers[v].writerow(dup_row)
                    
                    # Observer补帧
                    if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO and (not OBSERVER_DEDICATED_THREAD):
                        lf = last_frames.get(OBSERVER_VEHICLE_NAME)
                        if lf is not None and OBSERVER_VEHICLE_NAME in video_writers:
                            video_writers[OBSERVER_VEHICLE_NAME].write(lf)
                        if OBSERVER_VEHICLE_NAME in csv_writers:
                            csv_writers[OBSERVER_VEHICLE_NAME].writerow([ts_utc_dup, ts_ms_dup, ts_us_dup, frame_idx])
                    
                    frame_idx += 1
            # 无论是否补帧，都更新last_tick_time为当前时间
            last_tick_time = now

        frame_idx += 1
        elapsed = time.time() - tick_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("Warning: Loop is running slower than target FPS.")
    
    # 循环结束：不要在这里关闭文件/编码器（Observer 编码线程可能仍在写入）
    # 将关闭责任交给外层的 cleanup_recorders()，以避免竞争性关闭导致的 I/O 错误。
    print(f"Recording completed. Total frames captured: {frame_idx}")
    # 尝试刷新所有打开的文件/编码器以确保缓冲区写出，但不要关闭它们
    for name, writer in list(video_writers.items()):
        try:
            # VideoWriter 没有通用 flush 接口；尝试调用属性或忽略错误
            if hasattr(writer, 'release'):
                # 不释放，这里仅打印状态
                pass
        except Exception:
            pass
    for name, f in list(csv_files.items()):
        try:
            f.flush()
        except Exception:
            pass

    print("Recording finished; files flushed. Final cleanup will be done after observer threads stop.")

def landing_and_cleanup():
    print("Landing all drones...")
    # 使用简单的time.sleep等待,避免IOLoop冲突
    for v in VEHICLES:
        try:
            client.landAsync(vehicle_name=v)
            print(f"[{v}] landing command sent.")
        except Exception as e:
            print(f"[{v}] landing failed: {e}")
    
    # 等待所有无人机降落(简单延迟)
    print("Waiting for drones to land...")
    time.sleep(5)
    
    # 解除武装
    for v in VEHICLES:
        try:
            client.armDisarm(False, vehicle_name=v)
            client.enableApiControl(False, vehicle_name=v)
            print(f"[{v}] disarmed and API control disabled.")
        except Exception as e:
            print(f"[{v}] cleanup failed: {e}")
    
    print("All drones landed and disarmed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-drone data capture script for AirSim.")
    parser.add_argument('--duration', type=int, default=60, help='Duration of the capture in seconds.')
    parser.add_argument('--export-mot', action='store_true', help='Export captured data to MOT format after recording.')
    parser.add_argument('--export-coco', action='store_true', help='Export captured data to COCO format after recording.')
    parser.add_argument('--export-summary', action='store_true', help='Generate dataset summary after recording.')
    parser.add_argument('--export-only', action='store_true', help='Skip recording, only export existing data.')
    args = parser.parse_args()

    try:
        initialize_client()
        ensure_dirs()
        
        # 获取图像尺寸用于视频编码器（分别获取无人机与 Observer 的分辨率）
        resp = client.simGetImages([airsim.ImageRequest(CAM_IDX, airsim.ImageType.Scene, pixels_as_float=False, compress=False)], vehicle_name=VEHICLES[0])
        img_height = resp[0].height
        img_width = resp[0].width

        observer_size = None
        if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO:
            try:
                oresp = client.simGetImages([airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.Scene, pixels_as_float=False, compress=False)], vehicle_name=OBSERVER_VEHICLE_NAME)
                if oresp and oresp[0]:
                    observer_size = (oresp[0].width, oresp[0].height)
            except Exception as e:
                print(f"Observer size probe failed: {e}")
        
        initialize_recorders(img_size=(img_width, img_height), observer_img_size=observer_size)

        arm_and_takeoff()
        # 起飞后再启动Observer专用录制线程，减小起始时间差
        start_observer_pipeline()
        follow_paths_and_capture(duration_seconds=args.duration)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print("完整错误堆栈:")
        traceback.print_exc()
    finally:
        # 先停止Observer线程，避免与降落并发RPC导致 IOLoop 冲突
        stop_observer_pipeline()
        if client:
            landing_and_cleanup()
        cleanup_recorders()
        print("录制完成。")
        print("所有任务完成。")