# 数据质量检验工具使用指南

## 📋 概述

`validate_data_quality.py` 是一个完整的数据质量检验工具，用于验证多无人机数据采集系统的数据完整性、一致性和准确性。

## 🛠️ 功能特性

### 1. **数据完整性检查**
- ✅ 验证视频文件存在性和可读性
- ✅ 验证CSV日志文件
- ✅ 检查数据缺失情况
- ✅ 检测异常值和离群点

### 2. **时间同步验证**
- ✅ 视频帧数与CSV帧数对齐
- ✅ 时间戳连续性检查
- ✅ 视频时长与CSV时间范围匹配
- ✅ 跨vehicle时间同步

### 3. **数据质量评估**
- ✅ 传感器覆盖率统计
- ✅ 边界框标注准确性验证
- ✅ 无人机可见性分析
- ✅ 坐标有效性检查

### 4. **可视化验证**
- ✅ 将边界框叠加到视频上
- ✅ 生成可视化验证视频
- ✅ 保存采样帧进行人工检查
- ✅ 颜色编码的无人机识别

### 5. **报告生成**
- ✅ JSON格式的详细验证报告
- ✅ 纯文本统计报告
- ✅ 问题汇总和建议
- ✅ 性能指标统计

## 📦 依赖安装

```bash
# 安装必要的包
pip install pandas opencv-python numpy
```

## 🚀 快速使用

### 基础使用 - 数据验证

```bash
# 进入Scripts目录
cd Scripts

# 运行基础验证
python validate_data_quality.py --output_dir output
```

**输出结果：**
- `validation_report.json` - 详细的JSON验证报告
- `statistics_report.txt` - 纯文本统计报告
- `data_validation.log` - 详细的验证日志

### 高级使用 - 生成可视化

```bash
# 生成Observer的边界框可视化视频
python validate_data_quality.py --output_dir output --visualize --vehicle Observer

# 处理前100帧
python validate_data_quality.py --output_dir output --visualize --num-frames 100

# 从第50帧开始，处理300帧
python validate_data_quality.py --output_dir output --visualize --start-frame 50 --num-frames 300

# 生成采样帧（保存关键帧的边界框图）
python validate_data_quality.py --output_dir output --visualize --sample-frames
```

## 📊 输出文件说明

### 1. validation_report.json

完整的JSON格式验证报告，包含：

```json
{
  "validation_time": "2025-11-10T10:30:45.123456",
  "total_vehicles": 5,
  "total_issues": 3,
  "details": {
    "Drone1": {
      "video": {
        "exists": true,
        "frames": 600,
        "fps": 20.0,
        "duration": 30.0,
        "resolution": [1280, 720],
        "issues": []
      },
      "csv": {
        "sensor_exists": true,
        "sensor_rows": 600,
        "sensor_coverage": 100,
        "sensor_issues": [],
        ...
      },
      "time_sync": {...}
    },
    ...
  }
}
```

### 2. statistics_report.txt

易读的统计报告示例：

```
================================================================================
无人机数据采集统计报告
生成时间: 2025-11-10 10:30:45
================================================================================

────────────────────────────────────────────────────────────────────────────────
无人机: Drone1
────────────────────────────────────────────────────────────────────────────────

【视频信息】
  总帧数: 600 帧
  帧率: 20.00 FPS
  总时长: 30.00 秒
  分辨率: 1280x720

【传感器数据】
  数据行数: 600 行
  覆盖率: 100%

【时间同步】
  帧数对齐: ✓
  视频帧数: 600
  CSV帧数: 600
  帧数差异: 0
  时间范围匹配: ✓

...

【总体统计】
════════════════════════════════════════════════════════════════════════════════

检验的vehicle数: 5
发现的问题数: 3

✓ 所有数据质量检查通过！
```

### 3. bbox_visualization.mp4

边界框可视化视频，特点：
- 红色框：Drone1
- 绿色框：Drone2
- 蓝色框：Drone3
- 青色框：Drone4
- 灰色圆：集群中心和范围
- 显示可见性百分比

### 4. bbox_frames/ 目录

包含采样帧的PNG图像：
- `frame_000000_bbox.png` - 第0帧
- `frame_000050_bbox.png` - 第50帧
- `frame_000100_bbox.png` - 第100帧
- 等等...

## 🔍 检查项详解

### 视频验证

```
✓ 视频帧数: 600
✓ 帧率: 20.00 FPS
✓ 分辨率: 1280x720
✓ 时长: 30.00 秒
```

**可能的问题：**
- `视频帧数过少` - 采集时间太短
- `无法打开视频文件` - 文件格式问题或损坏

### 传感器数据验证

```
✓ 传感器数据行数: 600
✓ 覆盖率: 100%
⚠️ 传感器列缺失率: 5.00%
⚠️ 包含离群点: 10 个
```

**常见问题：**
- `缺失率过高` - 数据采集不稳定
- `时间戳间隙过大` - 存在数据跳跃
- `离群点过多` - 传感器异常

### 时间同步验证

```
✓ 视频和CSV帧数对齐 (视频: 600, CSV: 600)
✓ 视频时长和CSV时间范围匹配
```

**同步问题排查：**
- 帧数差异 > 5 帧：可能存在丢帧
- 时间范围不匹配 > 1s：采集不同步

### 边界框标注验证

```
✓ Drone1 可见性: 580/600 (96.67%)
✓ Drone2 可见性: 575/600 (95.83%)
⚠️ 边界框列包含无效值: 5 个 (max < min)
```

**标注质量指标：**
- **可见性**: 高于90%为正常
- **无效坐标**: 应该为0
- **覆盖率**: 应该是100%

## 📈 数据质量等级

根据检查结果分类：

| 等级 | 标准 | 说明 |
|------|------|------|
| ✅ 优秀 | 0 个问题 | 数据完全符合要求 |
| ✓ 良好 | 1-3 个轻微问题 | 可用于训练，建议改进 |
| ⚠️ 一般 | 4-10 个问题 | 需要修复后再用 |
| ❌ 差 | 10+ 个问题 | 不建议使用 |

## 🛠️ 常见问题解决

### 问题1: 视频和CSV帧数不对齐

**症状：**
```
⚠️ 视频和CSV帧数不对齐 (差异: 5 帧)
```

**原因：**
- 视频编码丢帧
- CSV记录不完整

**解决方案：**
```bash
# 检查视频文件
ffmpeg -i output/Observer/video.mp4

# 检查CSV内容
wc -l output/Observer/sensors.csv
```

### 问题2: 时间戳间隙过大

**症状：**
```
⚠️ 时间戳最大间隙: 2500ms (存在数据跳跃)
```

**原因：**
- 系统卡顿
- 网络延迟

**解决方案：**
```python
# 查看间隙位置
import pandas as pd
df = pd.read_csv('output/Observer/sensors.csv')
gaps = df['timestamp_ms'].diff()
print(gaps[gaps > 1000])
```

### 问题3: 边界框坐标无效

**症状：**
```
⚠️ 边界框列包含无效值: 10 个 (max < min)
```

**原因：**
- 标注算法错误
- 投影计算失败

**解决方案：**
- 检查投影坐标计算代码
- 验证相机参数设置

## 📝 脚本扩展

### 添加自定义检查

```python
class CustomValidator(DataValidator):
    def _validate_custom_metric(self, vehicle_dir):
        """添加自定义检查"""
        # 您的检查逻辑
        pass
    
    def validate_all(self):
        super().validate_all()
        for vehicle in self.vehicles:
            vehicle_dir = self.output_dir / vehicle
            self._validate_custom_metric(vehicle_dir)
```

### 修改可视化样式

```python
# 在BboxVisualizer中修改
drone_colors = {
    "Drone1": (0, 0, 255),      # 修改颜色
    "Drone2": (0, 255, 0),
    "Drone3": (255, 0, 0),
    "Drone4": (255, 255, 0)
}
```

## 🎯 最佳实践

1. **定期检验** - 每次采集后都运行一次
2. **保存报告** - 建立检验历史记录
3. **可视化验证** - 人工检查几个关键帧
4. **问题追踪** - 记录发现的问题和解决方案

## 💡 性能优化

对于大型数据集：

```bash
# 只验证前100帧
python validate_data_quality.py --output_dir output --num-frames 100

# 跳过可视化生成
python validate_data_quality.py --output_dir output  # 不加--visualize

# 只检查特定vehicle
python validate_data_quality.py --output_dir output/Observer
```

## 📞 获取帮助

查看详细日志：
```bash
tail -f data_validation.log
```

生成的报告包含了所有问题的详细描述，可据此进行问题诊断和改进。

---

**🎉 数据质量检验完成后，您就可以放心地用于下一步的目标检测和跟踪研究了！**