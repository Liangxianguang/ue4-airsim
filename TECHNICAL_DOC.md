# 系统技术文档

## 🏗️ 系统架构

### 整体架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UE4 Engine    │    │   AirSim API    │    │  Python Client  │
│                 │◄──►│                 │◄──►│                 │
│  - 物理仿真      │    │  - 多机管理      │    │  - 数据采集      │
│  - 渲染引擎      │    │  - 传感器接口    │    │  - 视频编码      │
│  - 环境建模      │    │  - 摄像头控制    │    │  - 标注生成      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                        ┌───────┴───────┐
                        │               │
                 ┌──────▼──────┐ ┌──────▼──────┐
                 │ Multi-Drone │ │  Observer   │
                 │   System    │ │   Camera    │
                 │             │ │             │
                 └─────────────┘ └─────────────┘
```

## 🔧 核心模块设计

### 1. 多无人机管理模块

#### 轨迹生成器
```python
def generate_path(pattern, center, radius, alt, num_points=40):
    """动态轨迹生成
    
    支持的模式:
    - circle: 圆形轨迹
    - spiral: 螺旋轨迹  
    - sine: 正弦波轨迹
    - figure8: 8字形轨迹
    - zigzag: 之字形轨迹
    - evasive: 规避机动
    - intercept: 拦截轨迹
    - combat_turn: 战斗转弯
    - swarm_break: 编队解散
    """
```

#### 编队控制算法
```python
# 分离力计算 - 避免碰撞
def compute_separation_force(own_pos, other_positions, min_distance=8.0):
    separation_force = np.array([0.0, 0.0])
    for other_pos in other_positions:
        diff = np.array(own_pos[:2]) - np.array(other_pos[:2])
        distance = np.linalg.norm(diff)
        if distance < min_distance and distance > 0:
            force = diff / distance * (min_distance - distance)
            separation_force += force * 0.5  # 分离强度
    return separation_force
```

### 2. Observer摄像头控制系统

#### 智能跟踪算法
```python
class ObserverCameraController:
    def __init__(self):
        self.smooth_pos_tau = 1.2      # 位置平滑时间常数
        self.smooth_rot_tau = 0.8      # 旋转平滑时间常数
        self.max_pos_speed = 10.0      # 最大位置速度
        self.max_ang_speed = 30.0      # 最大角速度
        
    def compute_target_pose(self, swarm_center, swarm_radius):
        """计算目标摄像头位置和朝向"""
        # 固定距离模式
        horiz_dist = OBSERVER_FIXED_DIST
        height = OBSERVER_FIXED_HEIGHT
        
        # 目标位置计算
        target_pos = (
            swarm_center[0] - horiz_dist,
            swarm_center[1],
            swarm_center[2] - height
        )
        
        # 目标朝向计算
        target_yaw = math.atan2(
            swarm_center[1] - target_pos[1],
            swarm_center[0] - target_pos[0]
        )
        target_pitch = -math.atan2(
            swarm_center[2] - target_pos[2],
            math.hypot(swarm_center[0] - target_pos[0],
                      swarm_center[1] - target_pos[1])
        )
        
        return target_pos, target_yaw, target_pitch
```

#### 平滑滤波器
```python
def _exp_lerp(current, target, dt, tau):
    """指数移动平均滤波器
    
    Args:
        current: 当前值
        target: 目标值
        dt: 时间间隔
        tau: 时间常数 (越大越平滑)
        
    Returns:
        平滑后的值
    """
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    return current + alpha * (target - current)
```

### 3. 数据采集管道

#### 多线程架构
```python
class DataCaptureSystem:
    def __init__(self):
        self.drone_threads = {}      # 无人机采集线程
        self.observer_thread = None  # Observer专用线程
        self.encoding_queue = None   # 编码队列
        
    def start_capture(self):
        """启动多线程数据采集"""
        # 为每架无人机创建独立线程
        for vehicle in VEHICLES:
            thread = threading.Thread(
                target=self.drone_capture_loop,
                args=(vehicle,)
            )
            self.drone_threads[vehicle] = thread
            thread.start()
            
        # Observer专用线程
        if ENABLE_OBSERVER_CAMERA and OBSERVER_DEDICATED_THREAD:
            self.observer_thread = threading.Thread(
                target=self.observer_capture_loop
            )
            self.observer_thread.start()
```

#### 图像处理管道
```python
def process_frame_with_annotations(frame, vehicles_poses, camera_pose):
    """图像帧处理与标注生成
    
    处理流程:
    1. 世界坐标到相机坐标转换
    2. 3D到2D投影
    3. 边界框生成
    4. 可见性检测
    5. 遮挡状态判断
    """
    annotations = []
    
    for vehicle_name, pose in vehicles_poses.items():
        # 坐标转换
        world_pos = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        bbox_2d, visibility = project_3d_to_2d(world_pos, camera_pose, frame.shape)
        
        if visibility > 0:
            annotations.append({
                'vehicle': vehicle_name,
                'bbox': bbox_2d,
                'visibility': visibility,
                'world_pos': world_pos,
                'distance': calculate_distance(world_pos, camera_pose.position)
            })
            
    return frame, annotations
```

### 4. 传感器数据系统

#### 多模态传感器接口
```python
def collect_comprehensive_sensor_data(client, vehicle_name):
    """全方位传感器数据收集"""
    try:
        # IMU数据
        imu = client.getImuData(vehicle_name=vehicle_name)
        
        # GPS数据  
        gps = client.getGpsData(vehicle_name=vehicle_name)
        
        # 气压计数据
        baro = client.getBarometerData(vehicle_name=vehicle_name)
        
        # 磁力计数据
        mag = client.getMagnetometerData(vehicle_name=vehicle_name)
        
        # 位姿数据
        pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
        
        return {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'position': [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            'orientation': [pose.orientation.w_val, pose.orientation.x_val, 
                          pose.orientation.y_val, pose.orientation.z_val],
            'gps': [gps.gnss.geo_point.latitude, gps.gnss.geo_point.longitude, gps.gnss.geo_point.altitude],
            'imu_linear_acc': [imu.linear_acceleration.x_val, imu.linear_acceleration.y_val, imu.linear_acceleration.z_val],
            'imu_angular_vel': [imu.angular_velocity.x_val, imu.angular_velocity.y_val, imu.angular_velocity.z_val],
            'barometer': [baro.altitude, baro.pressure],
            'magnetometer': [mag.magnetic_field_body.x_val, mag.magnetic_field_body.y_val, mag.magnetic_field_body.z_val]
        }
    except Exception as e:
        return None
```

## 🎯 拦截场景模拟算法

### 1. 规避机动算法
```python
def generate_evasive_maneuver(center, radius, alt, phase=0):
    """生成规避机动轨迹
    
    特点:
    - 随机性强
    - 速度变化大
    - 方向突变
    - 垂直机动
    """
    path = []
    for i in range(num_points):
        # 基础圆形轨迹
        theta = 2 * math.pi * i / num_points + phase
        
        # 添加随机扰动
        noise_x = np.random.normal(0, radius * 0.1)
        noise_y = np.random.normal(0, radius * 0.1)
        
        # 速度变化
        speed_factor = 1.0 + 0.3 * math.sin(theta * 3)
        
        x = center[0] + radius * math.cos(theta) * speed_factor + noise_x
        y = center[1] + radius * math.sin(theta) * speed_factor + noise_y
        z = alt + 2 * math.sin(theta * 2)  # 垂直机动
        
        path.append((float(x), float(y), float(z)))
    
    return path
```

### 2. 拦截轨迹算法
```python
def generate_intercept_trajectory(target_path, intercept_speed=1.5):
    """生成拦截轨迹
    
    算法:
    1. 预测目标未来位置
    2. 计算最佳拦截点
    3. 规划最短拦截路径
    4. 考虑速度限制
    """
    intercept_path = []
    
    for i, target_point in enumerate(target_path):
        # 前导预测
        if i < len(target_path) - 5:
            predicted_pos = target_path[i + 5]  # 预测5步后位置
        else:
            predicted_pos = target_point
            
        # 拦截向量计算
        intercept_vector = calculate_intercept_vector(
            current_pos, predicted_pos, intercept_speed
        )
        
        # 路径平滑
        if i > 0:
            smooth_pos = smooth_trajectory_point(
                intercept_path[-1], intercept_vector, 0.7
            )
        else:
            smooth_pos = intercept_vector
            
        intercept_path.append(smooth_pos)
    
    return intercept_path
```

### 3. 集群行为分析
```python
def analyze_swarm_behavior(vehicles_poses):
    """集群行为分析
    
    计算指标:
    - 集群中心
    - 集群半径
    - 分散度
    - 密度
    - 相对距离
    """
    positions = [pose.position for pose in vehicles_poses.values()]
    
    # 集群中心
    center = np.mean(positions, axis=0)
    
    # 集群半径
    distances = [np.linalg.norm(pos - center) for pos in positions]
    radius = np.max(distances)
    
    # 分散度（标准差）
    dispersion = np.std(distances)
    
    # 密度（相对于理想编队）
    ideal_radius = len(positions) * 2  # 假设理想间距2米
    density = ideal_radius / (radius + 1e-6)
    
    # 最小/最大/平均相对距离
    pairwise_distances = []
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i+1:], i+1):
            dist = np.linalg.norm(pos1 - pos2)
            pairwise_distances.append(dist)
    
    return {
        'center': center.tolist(),
        'radius': float(radius),
        'dispersion': float(dispersion),
        'density': float(density),
        'min_distance': float(np.min(pairwise_distances)) if pairwise_distances else 0,
        'max_distance': float(np.max(pairwise_distances)) if pairwise_distances else 0,
        'avg_distance': float(np.mean(pairwise_distances)) if pairwise_distances else 0,
        'num_drones': len(positions)
    }
```

## 📊 数据格式标准

### MOT Challenge格式
```python
# MOT标准格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
def export_mot_format(annotations, output_path):
    """导出MOT Challenge标准格式"""
    with open(output_path, 'w') as f:
        for frame_idx, frame_annotations in annotations.items():
            for ann in frame_annotations:
                # MOT格式行
                mot_line = f"{frame_idx},{ann['id']},{ann['bbox'][0]},{ann['bbox'][1]}," \
                          f"{ann['bbox'][2]},{ann['bbox'][3]},{ann['confidence']}," \
                          f"{ann['world_pos'][0]},{ann['world_pos'][1]},{ann['world_pos'][2]}\n"
                f.write(mot_line)
```

### COCO检测格式
```python
def export_coco_format(annotations, output_path):
    """导出COCO检测格式"""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "drone", "supercategory": "vehicle"}]
    }
    
    annotation_id = 1
    for frame_idx, frame_annotations in annotations.items():
        # 图像信息
        image_info = {
            "id": frame_idx,
            "width": 1280,
            "height": 720,
            "file_name": f"frame_{frame_idx:06d}.jpg"
        }
        coco_data["images"].append(image_info)
        
        # 标注信息
        for ann in frame_annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            annotation = {
                "id": annotation_id,
                "image_id": frame_idx,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "visibility": ann['visibility']
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
```

## ⚡ 性能优化策略

### 1. 多线程优化
```python
class ThreadSafeVideoWriter:
    """线程安全的视频写入器"""
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.lock = threading.Lock()
        
    def write(self, frame):
        with self.lock:
            self.writer.write(frame)
```

### 2. 内存池管理
```python
class FrameBufferPool:
    """帧缓冲池，减少内存分配"""
    def __init__(self, pool_size=10, frame_shape=(720, 1280, 3)):
        self.pool = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(pool_size)]
        self.available = list(range(pool_size))
        self.lock = threading.Lock()
        
    def get_buffer(self):
        with self.lock:
            if self.available:
                return self.pool[self.available.pop()]
            else:
                # 动态扩展
                return np.zeros(self.frame_shape, dtype=np.uint8)
                
    def return_buffer(self, buffer_idx):
        with self.lock:
            if buffer_idx < len(self.pool):
                self.available.append(buffer_idx)
```

### 3. 异步I/O
```python
import asyncio
import aiofiles

async def async_save_frame(frame, filepath):
    """异步保存图像帧"""
    loop = asyncio.get_event_loop()
    
    # 在线程池中执行CPU密集型操作
    encoded_frame = await loop.run_in_executor(
        None, cv2.imencode, '.jpg', frame
    )
    
    # 异步写入文件
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(encoded_frame[1].tobytes())
```

## 🔍 调试和监控

### 1. 性能监控
```python
class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.frame_times = []
        self.processing_times = []
        self.memory_usage = []
        
    def log_frame_time(self, frame_time):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
            
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
    def get_stats(self):
        return {
            'avg_fps': self.get_fps(),
            'avg_processing_time': np.mean(self.processing_times),
            'memory_usage_mb': self.get_memory_usage()
        }
```

### 2. 日志系统
```python
import logging

def setup_logging():
    """配置详细的日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('capture_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # 为不同模块创建专用logger
    loggers = {
        'camera': logging.getLogger('camera_controller'),
        'capture': logging.getLogger('data_capture'),
        'processing': logging.getLogger('frame_processing')
    }
    
    return loggers
```

这个技术文档详细说明了系统的核心架构、算法实现和优化策略，为深入理解和扩展系统提供了完整的技术参考。