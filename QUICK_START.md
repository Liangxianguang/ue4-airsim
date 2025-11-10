# 快速入门指南

## ⚡ 5分钟快速上手

### 第一步: 检查环境
```bash
# 检查Python环境
python --version  # 需要Python 3.8+

# 检查依赖包
pip list | grep -E "(airsim|opencv|numpy)"
```

### 第二步: 启动AirSim
1. 双击 `uav.uproject` 打开UE4项目
2. 点击 **Play** 按钮启动仿真
3. 等待加载完成（看到4架无人机和环境）

### 第三步: 运行数据采集
```bash
cd Scripts
python multi_drone_capture.py
```

### 第四步: 查看结果
```bash
# 检查输出文件
ls output/Observer/
# 应该看到: video.mp4, sensors.csv, bbox.csv等文件
```

---

## 🎮 基础操作

### 启动系统
```bash
# 基础启动
python multi_drone_capture.py

# 指定运行时间（秒）
python multi_drone_capture.py --duration 60

# 修改输出目录
python multi_drone_capture.py --output_dir custom_output
```

### 实时监控
```bash
# 查看实时日志
tail -f capture_system.log

# 监控输出文件大小
watch -n 1 'du -sh output/*'
```

---

## 📋 常用配置

### 快速配置模板

#### 🎯 高质量录制模式
```python
# 在multi_drone_capture.py中修改
CAPTURE_FPS = 30                    # 稳定帧率
OBSERVER_COMPRESS = False           # 无压缩
SAVE_PNG_EVERY_N = 1               # 保存PNG帧
ENABLE_DEPTH_SEGMENTATION = True   # 启用深度/分割
```

#### ⚡ 性能优化模式
```python
CAPTURE_FPS = 20                    # 降低帧率
OBSERVER_COMPRESS = True            # 启用压缩
SAVE_PNG_EVERY_N = 0               # 禁用PNG
SENSOR_LOG_EVERY_N = 10            # 降低传感器采样率
```

#### 🔬 研究模式
```python
ENABLE_BBOX_ANNOTATION = True      # 启用边界框
BBOX_LOG_EVERY_N = 1               # 每帧标注
MATCH_REALTIME_OUTPUT = True       # 实时输出
OBSERVER_DEDICATED_THREAD = True   # 专用线程
```

---

## 🛠️ 常见任务

### 1. 修改飞行轨迹
```python
# 在PATHS字典中修改
PATHS = {
    "Drone1": generate_path("circle", center=(10, 0), radius=15, alt=FLIGHT_ALT),
    "Drone2": generate_path("spiral", center=(0, 10), radius=20, alt=FLIGHT_ALT),
    # 添加更多无人机...
}
```

### 2. 调整Observer视角
```python
# 观察距离和高度
OBSERVER_FIXED_DIST = 40.0      # 距离40米
OBSERVER_FIXED_HEIGHT = 25.0    # 高度25米

# 平滑参数
CAM_SMOOTH_POS_TAU = 1.0       # 位置平滑
CAM_SMOOTH_ROT_TAU = 0.6       # 旋转平滑
```

### 3. 修改数据采集频率
```python
CAPTURE_FPS = 25               # 视频帧率
SENSOR_LOG_EVERY_N = 5         # 传感器每5帧记录一次
BBOX_LOG_EVERY_N = 2           # 边界框每2帧记录一次
```

---

## 📊 数据检查

### 验证数据完整性
```bash
# 检查视频文件
ls -la output/Observer/video.mp4

# 检查CSV文件行数
wc -l output/Observer/*.csv

# 检查图像帧数量
ls output/Observer/frames/ | wc -l
```

### 查看传感器数据
```python
import pandas as pd

# 读取传感器数据
df = pd.read_csv('output/Observer/sensors.csv')
print(f"记录数量: {len(df)}")
print(f"时间跨度: {df['timestamp_ms'].max() - df['timestamp_ms'].min()}ms")
print(df.head())
```

### 分析边界框数据
```python
# 读取边界框标注
bbox_df = pd.read_csv('output/Observer/bbox.csv')
print(f"标注帧数: {len(bbox_df)}")
print(f"无人机可见性统计:")
for drone in ['Drone1', 'Drone2', 'Drone3', 'Drone4']:
    visible_frames = (bbox_df[f'{drone}_visibility'] > 0).sum()
    print(f"  {drone}: {visible_frames}/{len(bbox_df)} frames")
```

---

## 🔧 故障诊断

### 连接问题
```bash
# 测试AirSim连接
python -c "import airsim; client = airsim.MultirotorClient(); print('连接成功' if client.ping() else '连接失败')"

# 检查无人机状态
python check_and_capture.py
```

### 性能问题
```python
# 降低计算负载
OBSERVER_UPDATE_EVERY_N = 10        # 减少更新频率
ENABLE_DRONE_DEPTH_SEGMENTATION = False  # 禁用重型计算
MAX_DUP_FRAMES_PER_TICK = 2         # 减少补帧
```

### 视频问题
```python
# 使用更兼容的编码器
# 在代码中找到VideoWriter部分，修改为:
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPEG
```

---

## 📈 优化建议

### 硬件优化
- **CPU**: 使用多核处理器，至少8核
- **GPU**: RTX 3060或更高，用于UE4渲染
- **内存**: 16GB+，大内存缓解I/O压力
- **存储**: SSD，提升大文件写入速度

### 软件优化
```python
# 启用所有优化选项
OBSERVER_DEDICATED_THREAD = True   # 专用线程
MATCH_REALTIME_OUTPUT = True       # 实时匹配
```

### 数据管理
```bash
# 定期清理旧数据
rm -rf output_backup/
mv output/ output_backup/
mkdir output/

# 压缩存储数据
tar -czf experiment_$(date +%Y%m%d).tar.gz output/
```

---

## 🎯 实验模板

### 基础跟踪实验
```python
# 30秒短时实验
python multi_drone_capture.py --duration 30

# 检查结果
ls output/Observer/
ffmpeg -i output/Observer/video.mp4  # 查看视频信息
```

### 长时间数据采集
```python
# 5分钟长实验
python multi_drone_capture.py --duration 300

# 后台运行
nohup python multi_drone_capture.py --duration 600 > capture.log 2>&1 &
```

### 多场景测试
```bash
# 场景1: 密集编队
# 修改PATHS为较小半径
python multi_drone_capture.py --output_dir output_dense

# 场景2: 分散编队  
# 修改PATHS为较大半径
python multi_drone_capture.py --output_dir output_sparse
```

---

## 📝 记录实验

### 创建实验日志
```bash
# 创建实验记录
echo "实验时间: $(date)" > experiment_log.txt
echo "配置: 4机编队, Observer固定距离35m" >> experiment_log.txt
echo "目标: 测试拦截场景下的目标跟踪" >> experiment_log.txt

# 运行实验
python multi_drone_capture.py >> experiment_log.txt 2>&1
```

### 数据备份
```bash
# 自动备份脚本
mkdir -p experiments/$(date +%Y%m%d_%H%M%S)
cp -r output/ experiments/$(date +%Y%m%d_%H%M%S)/
cp experiment_log.txt experiments/$(date +%Y%m%d_%H%M%S)/
```

---

**🚁 现在您已经准备好开始无人机集群拦截研究了！**

如有问题，请查阅完整的README.md和技术文档，或检查系统日志获取详细错误信息。