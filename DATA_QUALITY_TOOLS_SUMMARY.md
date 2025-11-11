# ✅ 数据质量检验工具 - 完整方案

## 📦 已创建的工具清单

### 1. **validate_data_quality.py** (主验证脚本)
**功能模块：**
- `DataValidator` 类 - 核心验证器
  - `_validate_video()` - 视频完整性检验
  - `_validate_csv()` - CSV数据验证
  - `_validate_time_sync()` - 时间同步检验
  - `_validate_observer_special()` - Observer特殊数据验证
  - `_validate_cross_vehicle_sync()` - 跨vehicle同步检验
  - `generate_report()` - 生成JSON报告
  - `generate_statistics_report()` - 生成统计报告

- `BboxVisualizer` 类 - 可视化工具
  - `visualize_bboxes()` - 生成边界框视频
  - `visualize_frame_range()` - 保存采样帧

**输出：**
- `validation_report.json` - 详细验证报告
- `statistics_report.txt` - 统计报告
- `bbox_visualization.mp4` - 可视化视频
- `bbox_frames/` - 采样帧图像
- `data_validation.log` - 详细日志

---

### 2. **run_validation_demo.py** (交互式演示脚本)
**功能：**
- 友好的命令行界面
- 6种预定义任务：
  1. 基础验证
  2. 生成报告
  3. 可视化验证
  4. 采样帧验证
  5. 完整检验
  6. 自定义检验
- 实时进度显示
- 汇总报告

**使用方法：**
```bash
python run_validation_demo.py
# 然后选择任务 (1-6)
```

---

### 3. **validation_config.py** (配置管理)
**功能：**
- 集中的配置管理
- 4种验证预设：
  - `quick` - 快速检验
  - `standard` - 标准检验
  - `strict` - 严格检验
  - `research` - 研究用检验
- 改进建议数据库
- 配置保存/加载

**使用方法：**
```python
from validation_config import load_config, get_suggestion

# 加载预设配置
config = load_config("strict")

# 获取改进建议
suggestion = get_suggestion("time_sync_mismatch")
```

---

### 4. **DATA_VALIDATION_GUIDE.md** (完整使用指南)
**内容：**
- 功能特性详解
- 快速使用示例
- 输出文件说明
- 检查项详细解释
- 数据质量等级标准
- 常见问题解决方案
- 脚本扩展方法
- 最佳实践

---

## 🚀 快速开始

### 方式1：交互式演示（推荐新手）
```bash
cd Scripts
python run_validation_demo.py
# 按提示选择任务
```

### 方式2：直接命令行
```bash
cd Scripts

# 基础验证
python validate_data_quality.py --output_dir output

# 生成可视化
python validate_data_quality.py --output_dir output --visualize

# 生成采样帧
python validate_data_quality.py --output_dir output --visualize --sample-frames
```

### 方式3：Python代码调用
```python
from validate_data_quality import DataValidator, BboxVisualizer

# 创建验证器
validator = DataValidator("output")
report = validator.validate_all()

# 生成报告
validator.generate_report()
validator.generate_statistics_report()

# 可视化
visualizer = BboxVisualizer("output", "Observer")
visualizer.visualize_bboxes(start_frame=0, num_frames=300)
```

---

## 📊 验证流程图

```
输入数据
  ↓
validate_data_quality.py
  ├─ 视频验证
  │  ├─ 文件存在性
  │  ├─ 帧数、fps
  │  └─ 分辨率
  │
  ├─ CSV验证
  │  ├─ 传感器数据完整性
  │  ├─ 异常值检测
  │  └─ 边界框数据质量
  │
  ├─ 时间同步验证
  │  ├─ 帧数对齐
  │  ├─ 时间戳连续性
  │  └─ 跨vehicle同步
  │
  └─ 生成报告
     ├─ validation_report.json
     ├─ statistics_report.txt
     └─ data_validation.log
         ↓
     ✓ 数据质量检验完成
         ↓
     [BboxVisualizer]
         ├─ 生成可视化视频
         ├─ 保存采样帧
         └─ ✓ 可视化验证完成
```

---

## 🔍 检验覆盖范围

### ✅ 验证的数据类型
- [x] RGB视频
- [x] 深度图
- [x] 分割掩码
- [x] 传感器CSV (IMU, GPS, 气压计, 磁力计)
- [x] 边界框标注
- [x] 时间戳

### ✅ 检查的问题
- [x] 文件完整性
- [x] 数据缺失
- [x] 异常值和离群点
- [x] 时间同步不一致
- [x] 帧数不对齐
- [x] 坐标有效性
- [x] 可见性不足
- [x] 编码错误

### ✅ 生成的输出
- [x] JSON详细报告
- [x] 文本统计报告
- [x] 日志文件
- [x] 可视化视频
- [x] 采样帧图像

---

## 📈 数据质量评分标准

| 问题数 | 等级 | 说明 | 建议 |
|--------|------|------|------|
| 0 | ✅ 优秀 | 完全符合要求 | 可直接用于训练 |
| 1-3 | ✓ 良好 | 轻微问题 | 可用，建议改进 |
| 4-10 | ⚠️ 一般 | 明显问题 | 修复后再用 |
| 10+ | ❌ 差 | 严重问题 | 不建议使用 |

---

## 🎯 第一次使用步骤

### Step 1: 准备环境
```bash
# 安装依赖
pip install pandas opencv-python numpy

# 确认输出目录存在
ls output/
```

### Step 2: 运行验证
```bash
cd Scripts
python validate_data_quality.py --output_dir output
```

### Step 3: 查看报告
```bash
# 查看统计报告
cat ../output/statistics_report.txt

# 查看JSON报告（可用文本编辑器打开）
cat ../output/validation_report.json
```

### Step 4: 可视化验证
```bash
# 生成可视化视频（处理前300帧）
python validate_data_quality.py --output_dir output --visualize --num-frames 300

# 播放可视化视频
# 使用任何视频播放器打开 output/Observer/bbox_visualization.mp4
```

### Step 5: 人工检查
- 打开生成的可视化视频
- 检查边界框是否准确
- 检查无人机是否被正确识别
- 查看是否有标注错误

---

## 💡 常见任务命令

### 快速检验（仅检查，不可视化）
```bash
python validate_data_quality.py --output_dir output
```

### 检验特定vehicle
```bash
python validate_data_quality.py --output_dir output --visualize --vehicle Drone1
```

### 只处理某个帧范围
```bash
python validate_data_quality.py --output_dir output --visualize --start-frame 100 --num-frames 200
```

### 保存采样帧
```bash
python validate_data_quality.py --output_dir output --visualize --sample-frames
```

### 使用交互界面
```bash
python run_validation_demo.py
```

---

## 📋 文件清单

### 新增脚本
- `Scripts/validate_data_quality.py` (750+ 行)
- `Scripts/run_validation_demo.py` (300+ 行)
- `Scripts/validation_config.py` (400+ 行)

### 新增文档
- `Scripts/DATA_VALIDATION_GUIDE.md` (本文档的完整版)

### 生成的输出
- `output/validation_report.json` - 验证报告
- `output/statistics_report.txt` - 统计报告
- `output/data_validation.log` - 日志文件
- `output/Observer/bbox_visualization.mp4` - 可视化视频
- `output/Observer/bbox_frames/` - 采样帧目录

---

## 🔧 高级用法

### 自定义配置
```python
from validation_config import load_config, PRESETS

# 使用严格模式
strict_config = load_config("strict")

# 修改阈值
strict_config["validation_thresholds"]["csv"]["max_missing_ratio"] = 0.02

# 创建新预设
custom_config = {
    **PRESETS["standard"],
    "output_dir": "output_custom",
    "vehicles": {k: v for k, v in PRESETS["standard"]["vehicles"].items() if k != "Drone4"}
}
```

### 编写自定义验证函数
```python
from validate_data_quality import DataValidator

class CustomValidator(DataValidator):
    def validate_custom_metric(self):
        """添加自定义检查"""
        logger.info("执行自定义检查...")
        # 您的检查逻辑
```

### 批量验证
```bash
# 为多个output目录验证
for dir in output_*; do
    python validate_data_quality.py --output_dir $dir
done
```

---

## 📞 故障排除

### 问题1: "找不到模块 pandas"
**解决方案：**
```bash
pip install pandas
```

### 问题2: 可视化视频编码失败
**解决方案：**
修改 `validate_data_quality.py` 中的编码器：
```python
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 改用MJPEG
```

### 问题3: 内存不足
**解决方案：**
- 减少处理帧数：`--num-frames 100`
- 只验证不可视化
- 分批处理不同vehicle

---

## 🎉 下一步行动

验证完成后：

1. ✅ **检查报告** - 查看 statistics_report.txt
2. ✅ **观看视频** - 播放 bbox_visualization.mp4
3. ✅ **评估质量** - 根据标准等级判断
4. ✅ **解决问题** - 根据建议改进
5. ✅ **重新验证** - 改进后再次验证

---

## 📚 相关文档

- `README.md` - 项目总体介绍
- `README_CN.md` - 中文简洁版
- `TECHNICAL_DOC.md` - 技术细节
- `QUICK_START.md` - 快速入门
- `DATA_VALIDATION_GUIDE.md` - 完整使用指南

---

**🚀 现在您拥有了完整的数据质量检验工具链！**

可以放心进行下一阶段的工作：
- 目标检测模型训练
- 目标跟踪算法开发
- 多目标跟踪研究
- 拦截策略优化