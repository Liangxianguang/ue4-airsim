# UAV Swarm Interception Tracking System

## 🎯 项目概述

本项目是一个基于AirSim的**无人机集群拦截目标跟踪研究平台**，专门设计用于开发和测试面向无人机集群拦截的目标跟踪方法。系统提供了完整的多无人机仿真环境、数据采集管道和分析工具。

### 🌟 核心特性

- **多无人机编队仿真** - 支持4架无人机同时飞行，每架具有独特的飞行轨迹
- **智能Observer摄像头** - 专业的第三人称观察视角，具有平滑跟踪和稳定控制
- **全方位数据采集** - RGB图像、深度图、分割掩码、传感器数据、边界框标注
- **实时边界框标注** - 自动生成目标检测数据，支持多目标跟踪(MOT)格式
- **高质量视频录制** - 支持多格式编码，可配置帧率和分辨率
- **拦截场景模拟** - 内置多种飞行模式：规避、拦截、战斗转弯、编队解散

## 📁 项目结构

```
uav/
├── Scripts/                    # 主要脚本文件
│   ├── multi_drone_capture.py # 🔥 主要捕获系统
│   ├── check_and_capture.py   # 基础功能检查
│   └── validate_settings.py   # 配置验证工具
├── output/                     # 数据输出目录
│   ├── Drone1/                # 各无人机数据
│   ├── Drone2/
│   ├── Drone3/
│   ├── Drone4/
│   └── Observer/              # Observer摄像头数据
├── Content/                   # UE4内容资源
├── Config/                    # 引擎配置文件
├── Plugins/AirSim/           # AirSim插件
├── settings.json             # AirSim主配置文件
└── uav.uproject             # UE4项目文件
```

## 🚀 快速开始

### 环境要求

- **操作系统**: Windows 10/11
- **Unreal Engine**: 4.27+
- **Python**: 3.8+ 
- **AirSim**: 最新版本
- **依赖包**: `airsim`, `opencv-python`, `numpy`

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd uav
   ```

2. **配置Python环境**
   ```bash
   conda create -n airsim python=3.8
   conda activate airsim
   pip install airsim opencv-python numpy
   ```

3. **启动AirSim仿真**
   - 在UE4中打开 `uav.uproject`
   - 点击Play启动仿真环境

4. **运行数据采集**
   ```bash
   cd Scripts
   python multi_drone_capture.py
   ```

## 🛩️ 飞行模式详解

系统内置了4种针对拦截场景优化的飞行模式：

| 无人机 | 模式 | 描述 | 轨迹特点 |
|--------|------|------|----------|
| Drone1 | **规避机动** (Evasive) | 模拟目标无人机执行规避动作 | 紧密圆形机动，半径12m |
| Drone2 | **拦截轨迹** (Intercept) | 模拟拦截无人机追击目标 | 大幅度椭圆轨迹，半径20m |
| Drone3 | **战斗转弯** (Combat Turn) | 模拟空战中的战术机动 | 复杂转弯轨迹，半径15m |
| Drone4 | **编队解散** (Swarm Break) | 模拟编队突然解散 | 螺旋扩散轨迹，半径10m |

### 轨迹类型

- **Circle**: 标准圆形轨迹
- **Spiral**: 螺旋上升/下降轨迹
- **Sine**: S型波浪轨迹
- **Figure8**: 8字型轨迹
- **Zigzag**: 之字型机动轨迹

## 📸 Observer摄像头系统

### 核心特性

Observer摄像头是本系统的核心组件，专为无人机集群跟踪研究设计：

#### 🎥 智能跟踪
- **集群中心计算** - 自动计算多机编队的几何中心
- **平滑跟踪** - 使用指数移动平均(EMA)实现平滑跟踪
- **稳定控制** - 多层次的平滑算法消除摄像头抖动

#### 🔧 视角控制
```python
# 摄像头控制参数
OBSERVER_ZOOM_MODE = "lock"          # 固定距离模式
OBSERVER_FIXED_DIST = 35.0           # 水平距离35米
OBSERVER_FIXED_HEIGHT = 20.0         # 垂直高度20米
CAM_SMOOTH_POS_TAU = 1.2            # 位置平滑系数
CAM_SMOOTH_ROT_TAU = 0.8            # 旋转平滑系数
```

#### 🎬 录制参数
- **帧率**: 30-60 FPS可配置
- **分辨率**: 1280x720 (可调整)
- **编码**: H.264/MJPEG多格式支持
- **质量**: 无损/高质量模式

## 📊 数据采集系统

### 多模态数据

系统同时采集多种类型的数据，为深度学习研究提供丰富的训练素材：

#### 🖼️ 图像数据
- **RGB图像** - 高清彩色图像
- **深度图** - 距离信息，用于3D重建
- **分割掩码** - 语义分割信息

#### 📡 传感器数据
- **IMU数据** - 线性加速度、角速度
- **GPS数据** - 经纬度、海拔高度
- **气压计** - 高度和气压信息
- **磁力计** - 三轴磁场强度

#### 🎯 目标检测数据
- **边界框标注** - 自动生成的bounding box
- **目标可见性** - 遮挡状态判断
- **相对位置** - 目标间的空间关系
- **MOT格式** - 多目标跟踪标准格式

### 数据格式

#### CSV传感器日志
```csv
timestamp_utc,timestamp_ms,frame_idx,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,
gps_lat,gps_lon,gps_alt,imu_lin_acc_x,imu_lin_acc_y,imu_lin_acc_z,
imu_ang_vel_x,imu_ang_vel_y,imu_ang_vel_z,baro_altitude,baro_pressure,
mag_x,mag_y,mag_z
```

#### 边界框标注格式
```csv
timestamp_utc,frame_idx,swarm_center_x,swarm_center_y,swarm_center_z,
swarm_radius,Drone1_bbox_x_min,Drone1_bbox_y_min,Drone1_bbox_x_max,Drone1_bbox_y_max,
Drone1_visibility,Drone1_distance,Drone1_world_x,Drone1_world_y,Drone1_world_z
```

## ⚙️ 配置选项

### 基础配置

```python
# 飞行参数
FLIGHT_SPEED = 5.0              # 飞行速度 (m/s)
FLIGHT_ALT = -10.0              # 飞行高度 (负值表示高度)
CAPTURE_FPS = 20                # 视频帧率

# Observer配置
ENABLE_OBSERVER_CAMERA = True   # 启用Observer摄像头
OBSERVER_FPS = 30               # Observer帧率
OBSERVER_COMPRESS = False       # 禁用压缩，提高质量
```

### 高级配置

```python
# 数据采集频率
SENSOR_LOG_EVERY_N = 5         # 传感器数据降采样
BBOX_LOG_EVERY_N = 1           # 边界框标注频率
SAVE_PNG_EVERY_N = 0           # PNG保存频率（0=仅视频）

# 性能优化
MATCH_REALTIME_OUTPUT = True   # 实时输出匹配
MAX_DUP_FRAMES_PER_TICK = 5    # 最大补帧数
```

### Observer摄像头优化

```python
# 抖动控制
OBSERVER_UPDATE_EVERY_N = 5     # 更新频率
CAM_MAX_POS_SPEED = 10.0       # 最大移动速度
CAM_MAX_ANG_SPEED_DEG = 30.0   # 最大角速度

# 平滑参数
CAM_SMOOTH_POS_TAU = 1.2       # 位置平滑时间常数
CAM_SMOOTH_ROT_TAU = 0.8       # 旋转平滑时间常数
```

## 🔬 研究应用

### 目标跟踪算法测试

本系统特别适合以下研究方向：

#### 1. **多目标跟踪(MOT)**
- 自动生成的MOT格式标注数据
- 复杂遮挡场景模拟
- 多种运动模式测试

#### 2. **目标检测**
- 丰富的边界框标注
- 多角度、多距离的目标视图
- 动态背景下的检测鲁棒性

#### 3. **深度估计**
- 深度图与RGB图像对应
- 已知的真实深度信息
- 动态场景的深度变化

#### 4. **语义分割**
- 精确的分割掩码
- 多类别目标分割
- 时序一致性验证

### 拦截场景研究

#### 🎯 拦截策略优化
```python
# 拦截轨迹参数
"intercept": {
    "approach_angle": 45,      # 接近角度
    "lead_prediction": True,   # 前导预测
    "collision_avoidance": True # 碰撞避免
}
```

#### 🛡️ 规避机动分析
```python
# 规避策略参数  
"evasive": {
    "evasion_radius": 12,      # 规避半径
    "frequency": 2.5,          # 机动频率
    "unpredictability": 0.8    # 不可预测性
}
```

## 📈 性能优化

### 系统性能调优

#### 🖥️ 硬件建议
- **CPU**: Intel i7/AMD Ryzen 7 或更高
- **GPU**: RTX 3060/4060 或更高 (用于实时渲染)
- **内存**: 16GB+ RAM
- **存储**: SSD推荐 (大量数据I/O)

#### ⚡ 软件优化
```python
# 关键优化参数
OBSERVER_DEDICATED_THREAD = True    # 专用线程录制
SAVE_PNG_EVERY_N = 0               # 禁用PNG保存
SENSOR_LOG_EVERY_N = 5             # 传感器数据降采样
OBSERVER_COMPRESS = False          # 关闭压缩提升速度
```

### 内存管理

系统采用多种策略优化内存使用：

- **帧缓冲管理** - 循环使用图像缓冲区
- **数据流水线** - 异步数据处理和I/O
- **智能采样** - 按需降采样减少数据量

## 🐛 故障排除

### 常见问题

#### 1. **Observer摄像头抖动**
```python
# 增加平滑参数
CAM_SMOOTH_POS_TAU = 1.2  # 更大的值 = 更平滑
CAM_SMOOTH_ROT_TAU = 0.8
```

#### 2. **帧率不稳定**
```python
# 降低计算复杂度
OBSERVER_UPDATE_EVERY_N = 5    # 减少更新频率
OBSERVER_FPS = 30              # 降低目标帧率
ENABLE_DEPTH_SEGMENTATION = False  # 禁用重型计算
```

#### 3. **视频编码错误**
```python
# 使用更兼容的编码器
OBSERVER_COMPRESS = False      # 禁用压缩
# 或在代码中修改编码器选择
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPEG
```

#### 4. **连接失败**
```bash
# 检查AirSim连接
python check_and_capture.py
# 验证settings.json配置
python validate_settings.py
```

## 📚 扩展开发

### 添加新的飞行模式

```python
def generate_custom_path(center, radius, alt, num_points):
    """自定义轨迹生成器"""
    path = []
    for i in range(num_points):
        # 实现您的轨迹逻辑
        x, y, z = custom_trajectory_function(i, center, radius, alt)
        path.append((x, y, z))
    return path

# 在PATHS字典中添加新模式
PATHS["Drone5"] = generate_custom_path(center=(10, 10), radius=8, alt=FLIGHT_ALT)
```

### 自定义数据处理

```python
def custom_frame_processor(frame, metadata):
    """自定义帧处理函数"""
    # 添加您的图像处理逻辑
    processed_frame = your_processing_function(frame)
    
    # 添加自定义元数据
    metadata['custom_metric'] = calculate_metric(frame)
    
    return processed_frame, metadata
```

### 添加新的传感器

```python
# 在传感器数据收集函数中添加
def collect_sensor_data(client, vehicle_name):
    # 现有传感器...
    
    # 添加自定义传感器
    custom_sensor_data = client.getCustomSensorData(vehicle_name)
    return {**existing_data, 'custom_sensor': custom_sensor_data}
```

## 🤝 贡献指南

欢迎为项目做出贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📜 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 📞 联系方式

- **项目维护者**: [Liangxianguang]
- **邮箱**: [2811306715@qq.com]
- **研究机构**: [nwpu]

## 🙏 致谢

- [Microsoft AirSim](https://github.com/microsoft/AirSim) - 强大的无人机仿真平台
- [Unreal Engine](https://www.unrealengine.com/) - 高质量的渲染引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

**⚡ 开始您的无人机集群拦截研究之旅！**

如果您在使用过程中遇到任何问题或有改进建议，请随时创建Issue或联系项目维护者。