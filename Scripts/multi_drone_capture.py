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
# 目标采集帧率（视频编码器也会按该帧率写入）；提升帧率的同时请确保机器性能足够
CAPTURE_FPS = 20
FLIGHT_SPEED = 5.0
FLIGHT_ALT = -10.0  # 飞得更高以获得更广的视野
# 可选：启用一个外部观察者摄像头（需在 AirSim settings.json 中配置 ExternalCameras -> "Observer"）
ENABLE_OBSERVER_CAMERA = True
OBSERVER_CAMERA_NAME = "0"          # 外部摄像机的相机索引名，通常为字符串"0"
OBSERVER_VEHICLE_NAME = "Observer"  # 外部摄像机在 settings.json 中的名称
OBSERVER_COMPRESS = True             # 为Observer使用压缩图，降低高分辨率下的带宽与拷贝开销
OBSERVER_DEDICATED_THREAD = True     # 为Observer启用专用采集+编码线程，获得更稳定的“相机式”录制
OBSERVER_FPS = 60                    # Observer 专用恒定输出帧率（建议高于/等于 CAPTURE_FPS）

# Observer 缩放/距离控制（解决“越拍越远”）：
# lock: 固定距离与高度
# bounded: 自适应但限制在最小/最大范围并做EMA平滑
# auto: 原始逻辑，完全自适应
OBSERVER_ZOOM_MODE = "bounded"      # 可选 "lock" | "bounded" | "auto"
OBSERVER_FIXED_DIST = 30.0           # ZOOM_MODE=lock 时水平距离
OBSERVER_FIXED_HEIGHT = 25.0         # ZOOM_MODE=lock 时高度（正值，内部转换为AirSim坐标系）
OBSERVER_MIN_DIST = 22.0             # bounded 模式距离/高度范围
OBSERVER_MAX_DIST = 45.0
OBSERVER_MIN_HEIGHT = 18.0
OBSERVER_MAX_HEIGHT = 35.0
OBSERVER_ZOOM_SMOOTH = 0.4          # EMA 平滑系数 0-1
OBSERVER_PHASE_SWITCH_SEC = 5.0     # 前N秒近景lock，之后切换到bounded全景

# Observer 摄像机平滑与限制（cinematic）
# 每 N 帧更新一次观察者摄像机（降低开销），值越小更新越频繁
OBSERVER_UPDATE_EVERY_N = 2
# 平滑时间常数（秒）：小值更紧贴目标，大值更平滑
CAM_SMOOTH_POS_TAU = 0.4
CAM_SMOOTH_ROT_TAU = 0.18
# 最大位置速度（单位：世界坐标/秒）与最大角速度（度/秒）
CAM_MAX_POS_SPEED = 30.0
CAM_MAX_ANG_SPEED_DEG = 120.0
# 可选的FOV平滑（启用可随缩放改变视场）
OBSERVER_FOV_ENABLE = True
OBSERVER_FOV_MIN = 30.0
OBSERVER_FOV_MAX = 75.0
OBSERVER_FOV_SMOOTH = 0.12

# 当循环性能不足导致实际帧率低于CAPTURE_FPS时，是否通过重复写入上一帧来拉齐视频播放时长
MATCH_REALTIME_OUTPUT = True
# 为防止极端卡顿导致一次性重复过多帧，设置每tick最大补帧数上限
MAX_DUP_FRAMES_PER_TICK = 5

# 性能优化选项
# 0 表示不单独落盘 PNG（仅写视频）；>0 表示每 N 帧落一张 PNG
SAVE_PNG_EVERY_N = 0
# 传感器（IMU/GPS/磁力计/气压计）记录的降采样因子（每 N 帧记录一次）
SENSOR_LOG_EVERY_N = 5
# 是否保存 Observer 的视频
SAVE_OBSERVER_VIDEO = True

import math
# 自动生成复杂轨迹（圆形、螺旋、S型、交错等）
def generate_path(pattern, center, radius, alt, num_points=40):
    path = []
    if pattern == "circle":
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            path.append((x, y, alt))
    elif pattern == "spiral":
        for i in range(num_points):
            r = radius * (i / num_points)
            theta = 4 * math.pi * i / num_points
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            z = alt + i * 0.2  # 螺旋上升
            path.append((x, y, z))
    elif pattern == "sine":
        for i in range(num_points):
            x = center[0] + i * radius / num_points
            y = center[1] + math.sin(i * 2 * math.pi / num_points) * radius / 2
            path.append((x, y, alt))
    elif pattern == "zigzag":
        for i in range(num_points):
            x = center[0] + i * radius / num_points
            y = center[1] + (radius if i % 2 == 0 else -radius)
            path.append((x, y, alt))
    else:
        for i in range(num_points):
            path.append((center[0] + i, center[1], alt))
    return path

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
def update_observer_camera(client, vehicles, strength=2.0):
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
        center, radius = compute_swarm_center_and_radius(poses)
        cx, cy, cz = center

        # 全局状态（zoom EMA 与 cinematic）
        global observer_dist_ema, observer_hgt_ema, observer_phase_start
        global swarm_center_ema, swarm_radius_ema
        global obs_cam_pos, obs_cam_yaw, obs_cam_pitch, obs_last_t, obs_fov_deg

        now = time.perf_counter()
        # 初始时间
        if observer_phase_start is None:
            observer_phase_start = now

        # 平滑质心与半径（避免快速跳动导致相机抖动）
        if swarm_center_ema is None:
            swarm_center_ema = (cx, cy, cz)
        else:
            dt_tmp = max(1e-6, (now - obs_last_t) if obs_last_t else (1.0 / max(OBSERVER_FPS, 30)))
            swarm_center_ema = (
                _exp_lerp(swarm_center_ema[0], cx, dt_tmp, CAM_SMOOTH_POS_TAU),
                _exp_lerp(swarm_center_ema[1], cy, dt_tmp, CAM_SMOOTH_POS_TAU),
                _exp_lerp(swarm_center_ema[2], cz, dt_tmp, CAM_SMOOTH_POS_TAU)
            )
        swarm_radius_ema = _exp_lerp(swarm_radius_ema, radius, (now - obs_last_t) if obs_last_t else (1.0 / max(OBSERVER_FPS, 30)), CAM_SMOOTH_POS_TAU)

        # 根据缩放模式确定距离/高度（使用原有EMA以控制zoom响应）
        effective_mode = OBSERVER_ZOOM_MODE
        if OBSERVER_ZOOM_MODE == "bounded" and (now - observer_phase_start) < OBSERVER_PHASE_SWITCH_SEC:
            effective_mode = "lock"

        if effective_mode == "lock":
            horiz_dist = float(OBSERVER_FIXED_DIST)
            height = float(OBSERVER_FIXED_HEIGHT)
        else:
            raw_dist = max(25.0, swarm_radius_ema * strength)
            raw_hgt = max(20.0, swarm_radius_ema * 1.2)
            if effective_mode == "bounded":
                raw_dist = float(np.clip(raw_dist, OBSERVER_MIN_DIST, OBSERVER_MAX_DIST))
                raw_hgt = float(np.clip(raw_hgt, OBSERVER_MIN_HEIGHT, OBSERVER_MAX_HEIGHT))
            if observer_dist_ema is None:
                observer_dist_ema, observer_hgt_ema = raw_dist, raw_hgt
            else:
                a = float(np.clip(OBSERVER_ZOOM_SMOOTH, 0.0, 1.0))
                observer_dist_ema = (1 - a) * observer_dist_ema + a * raw_dist
                observer_hgt_ema = (1 - a) * observer_hgt_ema + a * raw_hgt
            horiz_dist = observer_dist_ema
            height = observer_hgt_ema

        # 目标位置（以质心为基准向 -X 偏移）
        target_pos = (swarm_center_ema[0] - horiz_dist, swarm_center_ema[1], swarm_center_ema[2] - height)

        # 目标朝向（指向质心）
        dir_x = swarm_center_ema[0] - target_pos[0]
        dir_y = swarm_center_ema[1] - target_pos[1]
        dir_z = swarm_center_ema[2] - target_pos[2]
        target_yaw = math.atan2(dir_y, dir_x)
        target_pitch = -math.atan2(dir_z, math.hypot(dir_x, dir_y))
        roll = 0.0

        # 估计 dt
        dt = (now - obs_last_t) if obs_last_t else (1.0 / max(OBSERVER_FPS, 30))

        # 初始化观察者状态（首次直接跳到目标）
        if obs_cam_pos is None:
            obs_cam_pos = target_pos
        else:
            # 先做指数低通，再限制最大速度
            new_x = _exp_lerp(obs_cam_pos[0], target_pos[0], dt, CAM_SMOOTH_POS_TAU)
            new_y = _exp_lerp(obs_cam_pos[1], target_pos[1], dt, CAM_SMOOTH_POS_TAU)
            new_z = _exp_lerp(obs_cam_pos[2], target_pos[2], dt, CAM_SMOOTH_POS_TAU)
            # 限速
            dx = new_x - obs_cam_pos[0]
            dy = new_y - obs_cam_pos[1]
            dz = new_z - obs_cam_pos[2]
            move_len = math.hypot(math.hypot(dx, dy), dz)
            max_move = CAM_MAX_POS_SPEED * dt
            if move_len > max_move and move_len > 1e-6:
                scale = max_move / move_len
                new_x = obs_cam_pos[0] + dx * scale
                new_y = obs_cam_pos[1] + dy * scale
                new_z = obs_cam_pos[2] + dz * scale
            obs_cam_pos = (new_x, new_y, new_z)

        # 角度限速（按最短方向）
        if obs_cam_yaw is None:
            obs_cam_yaw = target_yaw
        else:
            diff = _wrap_pi(target_yaw - obs_cam_yaw)
            max_ang = math.radians(CAM_MAX_ANG_SPEED_DEG) * dt
            step = math.copysign(min(abs(diff), max_ang), diff)
            obs_cam_yaw = _wrap_pi(obs_cam_yaw + step)

        if obs_cam_pitch is None:
            obs_cam_pitch = target_pitch
        else:
            pdiff = _wrap_pi(target_pitch - obs_cam_pitch)
            pmax = math.radians(CAM_MAX_ANG_SPEED_DEG) * dt
            pstep = math.copysign(min(abs(pdiff), pmax), pdiff)
            obs_cam_pitch = _wrap_pi(obs_cam_pitch + pstep)

        # FOV 平滑（可选）
        if OBSERVER_FOV_ENABLE:
            # 根据 horiz_dist 映射到 FOV 范围
            t = float(np.clip((horiz_dist - OBSERVER_MIN_DIST) / max(1e-6, (OBSERVER_MAX_DIST - OBSERVER_MIN_DIST)), 0.0, 1.0))
            target_fov = OBSERVER_FOV_MIN + t * (OBSERVER_FOV_MAX - OBSERVER_FOV_MIN)
            if obs_fov_deg is None:
                obs_fov_deg = target_fov
            else:
                obs_fov_deg = _exp_lerp(obs_fov_deg, target_fov, dt, OBSERVER_FOV_SMOOTH)
            try:
                client.simSetCameraFov(OBSERVER_CAMERA_NAME, float(obs_fov_deg), vehicle_name=OBSERVER_VEHICLE_NAME)
            except Exception:
                pass

        # 应用相机姿态
        pose = airsim.Pose(
            airsim.Vector3r(obs_cam_pos[0], obs_cam_pos[1], obs_cam_pos[2]),
            airsim.to_quaternion(obs_cam_pitch, roll, obs_cam_yaw)
        )
        try:
            client.simSetCameraPose(OBSERVER_CAMERA_NAME, pose, vehicle_name=OBSERVER_VEHICLE_NAME)
        except Exception:
            pass

        obs_last_t = now
    except Exception:
        # 忽略任何单次更新异常，保持主流程稳定
        pass

# 为每架机分配不同轨迹
PATHS = {
    "Drone1": generate_path("circle", center=(5, 0), radius=10, alt=FLIGHT_ALT, num_points=60),
    "Drone2": generate_path("spiral", center=(5, 0), radius=15, alt=FLIGHT_ALT, num_points=60),
    "Drone3": generate_path("sine", center=(0, 8), radius=15, alt=FLIGHT_ALT, num_points=60),
    "Drone4": generate_path("zigzag", center=(8, 0), radius=10, alt=FLIGHT_ALT, num_points=60)
}

# --- 全局变量 ---
client = None
VEHICLES = []
video_writers = {}
csv_files = {}
csv_writers = {}
CSV_HEADERS = [
    "timestamp_utc", "timestamp_ms", "frame_idx",
    "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z",
    "gps_lat", "gps_lon", "gps_alt",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "baro_altitude", "baro_pressure",
    "mag_x", "mag_y", "mag_z"
]

# --- Observer 专用线程的共享状态 ---
observer_stop_event = None
observer_lock = threading.Lock()
observer_latest_frame = None  # 最新采集到但尚未被编码线程消耗的帧
observer_latest_ts = None     # 该帧的UTC采集时间
observer_capture_thread = None
observer_encode_thread = None
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
            PATHS[v] = [(p[0] + np.random.uniform(-5, 5), p[1] + np.random.uniform(-5, 5), p[2]) for p in template]
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for v in VEHICLES:
        # Video Writer
        video_path = os.path.join(OUT_DIR, v, "videos", f"{v}_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writers[v] = cv2.VideoWriter(video_path, fourcc, CAPTURE_FPS, img_size)
        print(f"[{v}] Video writer initialized at {video_path}")

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
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writers[OBSERVER_VEHICLE_NAME] = cv2.VideoWriter(video_path, fourcc, OBSERVER_FPS, obs_size)
        print(f"[{OBSERVER_VEHICLE_NAME}] Video writer initialized at {video_path} (fps={OBSERVER_FPS})")

        csv_path = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "logs", f"{OBSERVER_VEHICLE_NAME}_{ts}.csv")
        csv_files[OBSERVER_VEHICLE_NAME] = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writers[OBSERVER_VEHICLE_NAME] = csv.writer(csv_files[OBSERVER_VEHICLE_NAME])
        # 仅写通用头（没有机体状态可写时也能对齐时间戳）
        csv_writers[OBSERVER_VEHICLE_NAME].writerow(["timestamp_utc", "timestamp_ms", "frame_idx"]) 
        print(f"[{OBSERVER_VEHICLE_NAME}] CSV logger initialized at {csv_path}")

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
            resp = local_client.simGetImages(
                [airsim.ImageRequest(OBSERVER_CAMERA_NAME, airsim.ImageType.Scene, pixels_as_float=False, compress=OBSERVER_COMPRESS)],
                vehicle_name=OBSERVER_VEHICLE_NAME
            )
            frame = frame_from_response(resp[0]) if (resp and resp[0]) else None
            if frame is not None:
                with observer_lock:
                    observer_latest_frame = frame
                    observer_latest_ts = datetime.datetime.now(datetime.timezone.utc)
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

def _observer_encode_loop(dt):
    frame_idx = 0
    prev_frame = None
    next_t = time.perf_counter()
    while not observer_stop_event.is_set():
        # 使用最新帧；若暂无新帧则复用上一帧，保证CFR输出
        with observer_lock:
            frame = observer_latest_frame if observer_latest_frame is not None else prev_frame
        if frame is not None and OBSERVER_VEHICLE_NAME in video_writers:
            video_writers[OBSERVER_VEHICLE_NAME].write(frame)
            prev_frame = frame
        # 写Observer的时间戳日志（与视频帧一一对应）
        if OBSERVER_VEHICLE_NAME in csv_writers:
            utc_now = datetime.datetime.now(datetime.timezone.utc)
            ts_utc = utc_now.isoformat()
            ts_ms = int(utc_now.timestamp() * 1000)
            csv_writers[OBSERVER_VEHICLE_NAME].writerow([ts_utc, ts_ms, frame_idx])

        frame_idx += 1
        # 固定节拍输出
        next_t += dt
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # 写入线程落后时重置节拍，避免连续负漂移
            next_t = time.perf_counter()

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

    while time.time() - start_time < duration_seconds:
        tick_start = time.time()
        # 使用未压缩图像以避免每帧 PNG 解码开销
        requests = [airsim.ImageRequest(CAM_IDX, airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
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
            ts_ms = int(utc_now.timestamp() * 1000)

            # 保存图像并写入视频
            if img_resp and img_resp[0]:
                frame = frame_from_response(img_resp[0])
                if frame is not None and v in video_writers:
                    video_writers[v].write(frame)
                    last_frames[v] = frame
                    if SAVE_PNG_EVERY_N and (frame_idx % SAVE_PNG_EVERY_N == 0):
                        img_fname = os.path.join(OUT_DIR, v, "images", f"{frame_idx:06d}.png")
                        cv2.imwrite(img_fname, frame)

            # 记录CSV（对未取样传感器使用 NaN 占位，避免 NoneType 访问）
            row = [
                ts_utc, ts_ms, frame_idx,
                state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val,
                state.kinematics_estimated.orientation.w_val, state.kinematics_estimated.orientation.x_val, state.kinematics_estimated.orientation.y_val, state.kinematics_estimated.orientation.z_val,
                (gps.gnss.geo_point.latitude if gps else np.nan),
                (gps.gnss.geo_point.longitude if gps else np.nan),
                (gps.gnss.geo_point.altitude if gps else np.nan),
                (imu.linear_acceleration.x_val if imu else np.nan),
                (imu.linear_acceleration.y_val if imu else np.nan),
                (imu.linear_acceleration.z_val if imu else np.nan),
                (imu.angular_velocity.x_val if imu else np.nan),
                (imu.angular_velocity.y_val if imu else np.nan),
                (imu.angular_velocity.z_val if imu else np.nan),
                (baro.altitude if baro else np.nan),
                (baro.pressure if baro else np.nan),
                (mag.magnetic_field_body.x_val if mag else np.nan),
                (mag.magnetic_field_body.y_val if mag else np.nan),
                (mag.magnetic_field_body.z_val if mag else np.nan)
            ]
            if v in csv_writers:
                csv_writers[v].writerow(row)

        # 写入 Observer 视频与时间戳（如启用）
        if (observer_resp and observer_resp[0]) and (not OBSERVER_DEDICATED_THREAD):
            frame = frame_from_response(observer_resp[0])
            if frame is not None and OBSERVER_VEHICLE_NAME in video_writers:
                video_writers[OBSERVER_VEHICLE_NAME].write(frame)
                last_frames[OBSERVER_VEHICLE_NAME] = frame
                if SAVE_PNG_EVERY_N and (frame_idx % SAVE_PNG_EVERY_N == 0):
                    img_fname = os.path.join(OUT_DIR, OBSERVER_VEHICLE_NAME, "images", f"{frame_idx:06d}.png")
                    cv2.imwrite(img_fname, frame)
            # 日志仅写时间戳/帧号
            if (OBSERVER_VEHICLE_NAME in csv_writers):
                utc_now = datetime.datetime.now(datetime.timezone.utc)
                ts_utc = utc_now.isoformat()
                ts_ms = int(utc_now.timestamp() * 1000)
                csv_writers[OBSERVER_VEHICLE_NAME].writerow([ts_utc, ts_ms, frame_idx])

        # 动态计算目标点，实现复杂运动
        for v in VEHICLES:
            # 每帧都更新目标点，实现平滑运动
            idx = (frame_idx + hash(v)) % len(PATHS[v])
            next_pos = PATHS[v][idx]
            # --- 插入简单动作 ---
            # 例如：每隔30帧悬停1秒、每隔50帧旋转、每隔40帧抖动
            if frame_idx % 30 == 0 and v == "Drone1":
                # 非阻塞悬停：不再 sleep，避免心跳中断
                client.hoverAsync(vehicle_name=v)
            elif frame_idx % 50 == 0 and v == "Drone2":
                client.rotateByYawRateAsync(60, 1, vehicle_name=v)  # 非阻塞
            elif frame_idx % 40 == 0 and v == "Drone1":
                client.moveByVelocityAsync(1, 0, 0, 0.3, vehicle_name=v)  # 非阻塞
            else:
                client.moveToPositionAsync(next_pos[0], next_pos[1], next_pos[2], FLIGHT_SPEED, vehicle_name=v)

        # 每隔若干帧更新一次观察者摄像机位置，以降低开销
        if ENABLE_OBSERVER_CAMERA and (frame_idx % OBSERVER_UPDATE_EVERY_N == 0):
            update_observer_camera(client, VEHICLES, strength=2.2)

        # 如实际循环间隔大于目标dt，则按比例补写上一帧，尽量与真实时间一致
        if MATCH_REALTIME_OUTPUT:
            now = time.time()
            loop_dt = now - last_tick_time
            dup = int(round(loop_dt / dt)) - 1
            if dup > 0:
                dup = min(dup, MAX_DUP_FRAMES_PER_TICK)
                for _ in range(dup):
                    for v in VEHICLES:
                        lf = last_frames.get(v)
                        if lf is not None and v in video_writers:
                            video_writers[v].write(lf)
                    if ENABLE_OBSERVER_CAMERA and SAVE_OBSERVER_VIDEO and (not OBSERVER_DEDICATED_THREAD):
                        lf = last_frames.get(OBSERVER_VEHICLE_NAME)
                        if lf is not None and OBSERVER_VEHICLE_NAME in video_writers:
                            video_writers[OBSERVER_VEHICLE_NAME].write(lf)
            last_tick_time = now

        frame_idx += 1
        elapsed = time.time() - tick_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("Warning: Loop is running slower than target FPS.")

def landing_and_cleanup():
    print("Landing all drones...")
    land_futures = [client.landAsync(vehicle_name=v) for v in VEHICLES]
    for i, f in enumerate(land_futures):
        f.join()
        print(f"[{VEHICLES[i]}] has landed.")
        client.armDisarm(False, vehicle_name=VEHICLES[i])
        client.enableApiControl(False, vehicle_name=VEHICLES[i])
    print("All drones landed and disarmed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-drone data capture script for AirSim.")
    parser.add_argument('--duration', type=int, default=60, help='Duration of the capture in seconds.')
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
        # 启动Observer专用录制线程（如启用）
        start_observer_pipeline()

        arm_and_takeoff()
        follow_paths_and_capture(duration_seconds=args.duration)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 先停止Observer线程，避免与降落并发RPC导致 IOLoop 冲突
        stop_observer_pipeline()
        if client:
            landing_and_cleanup()
        cleanup_recorders()
        print("Done.")