"""
数据质量检验配置文件示例

将此文件复制并修改参数以自定义验证流程
"""

import json
from pathlib import Path


# ============================================================================
# 1. 基础配置
# ============================================================================

CONFIG = {
    # 输出目录
    "output_dir": "output",
    
    # 验证报告输出文件名
    "report_files": {
        "validation": "validation_report.json",
        "statistics": "statistics_report.txt",
        "log": "data_validation.log"
    },
    
    # ========================================================================
    # 2. 数据验证阈值
    # ========================================================================
    
    "validation_thresholds": {
        # 视频验证
        "video": {
            "min_frames": 10,                    # 最小帧数
            "max_time_gap_ms": 1000,           # 最大时间间隙（毫秒）
        },
        
        # CSV数据验证
        "csv": {
            "max_missing_ratio": 0.05,         # 最大缺失率 (5%)
            "max_outlier_ratio": 0.01,         # 最大离群点比率 (1%)
            "outlier_threshold_sigma": 3,      # 离群点阈值（标准差倍数）
        },
        
        # 时间同步验证
        "time_sync": {
            "max_frame_diff": 5,               # 最大帧差
            "max_duration_diff_sec": 1.0,      # 最大时长差（秒）
            "max_cross_vehicle_diff_ms": 100,  # 跨vehicle最大时间差（毫秒）
        },
        
        # 边界框验证
        "bbox": {
            "min_visibility": 0.80,            # 最小可见性 (80%)
            "coord_tolerance": 1,              # 坐标容差（像素）
        }
    },
    
    # ========================================================================
    # 3. 可视化配置
    # ========================================================================
    
    "visualization": {
        # 无人机颜色设置 (BGR格式)
        "drone_colors": {
            "Drone1": [0, 0, 255],      # 红色
            "Drone2": [0, 255, 0],      # 绿色
            "Drone3": [255, 0, 0],      # 蓝色
            "Drone4": [255, 255, 0]     # 青色
        },
        
        # 集群中心颜色
        "swarm_color": [200, 200, 200],  # 灰色
        "swarm_thickness": 2,
        
        # 字体设置
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.5,
        "font_thickness": 2,
        "font_color": [255, 255, 255],   # 白色
        
        # 可视化视频输出
        "output_video": "bbox_visualization.mp4",
        "output_codec": "mp4v",          # 编码器: mp4v, MJPG, XVID等
        
        # 采样帧输出
        "sample_frames_dir": "bbox_frames",
        "sample_frame_format": "png"
    },
    
    # ========================================================================
    # 4. 检验对象
    # ========================================================================
    
    "vehicles": {
        "Drone1": {
            "enabled": True,
            "check_video": True,
            "check_sensors": True,
            "check_time_sync": True
        },
        "Drone2": {
            "enabled": True,
            "check_video": True,
            "check_sensors": True,
            "check_time_sync": True
        },
        "Drone3": {
            "enabled": True,
            "check_video": True,
            "check_sensors": True,
            "check_time_sync": True
        },
        "Drone4": {
            "enabled": True,
            "check_video": True,
            "check_sensors": True,
            "check_time_sync": True
        },
        "Observer": {
            "enabled": True,
            "check_video": True,
            "check_sensors": True,
            "check_time_sync": True,
            "check_bbox": True,              # Observer特有
            "check_depth_segmentation": True # Observer特有
        }
    },
    
    # ========================================================================
    # 5. 报告生成设置
    # ========================================================================
    
    "report_settings": {
        "include_summary": True,           # 生成汇总
        "include_detailed_stats": True,    # 生成详细统计
        "include_issue_list": True,        # 包含问题列表
        "include_recommendations": True,   # 包含改进建议
        "max_issues_to_report": 100        # 最多报告的问题数
    },
    
    # ========================================================================
    # 6. 日志设置
    # ========================================================================
    
    "logging": {
        "level": "INFO",                   # DEBUG, INFO, WARNING, ERROR
        "console_output": True,            # 输出到控制台
        "file_output": True,               # 输出到文件
        "verbose": True                    # 详细输出
    }
}


# ============================================================================
# 7. 预定义的检验配置文件
# ============================================================================

PRESETS = {
    # 快速检验 - 只验证关键指标
    "quick": {
        "output_dir": "output",
        "skip_visualization": True,
        "validation_thresholds": {
            "video": {"min_frames": 10},
            "csv": {"max_missing_ratio": 0.10},  # 更宽松
            "time_sync": {"max_frame_diff": 10}
        }
    },
    
    # 标准检验 - 平衡性能和准确性
    "standard": CONFIG,
    
    # 严格检验 - 最高质量要求
    "strict": {
        "output_dir": "output",
        "validation_thresholds": {
            "video": {"min_frames": 100},
            "csv": {"max_missing_ratio": 0.01},  # 更严格
            "time_sync": {"max_frame_diff": 1}
        },
        "visualization": {
            "output_video": "bbox_visualization_strict.mp4"
        }
    },
    
    # 研究用检验 - 全面检验+完整可视化
    "research": {
        "output_dir": "output",
        "validation_thresholds": CONFIG["validation_thresholds"],
        "visualization": {
            **CONFIG["visualization"],
            "output_video": "bbox_visualization_research.mp4"
        },
        "report_settings": {
            "include_summary": True,
            "include_detailed_stats": True,
            "include_issue_list": True,
            "include_recommendations": True,
            "max_issues_to_report": 1000
        }
    }
}


# ============================================================================
# 8. 改进建议数据库
# ============================================================================

IMPROVEMENT_SUGGESTIONS = {
    "video_frame_count_low": {
        "issue": "视频帧数过少",
        "suggestion": "延长采集时间或降低帧率以增加总帧数",
        "action": "修改 CAPTURE_FPS 或延长采集时间"
    },
    "time_sync_mismatch": {
        "issue": "视频和CSV帧数不对齐",
        "suggestion": "检查视频编码器设置或CSV记录逻辑",
        "action": "检查 multi_drone_capture.py 中的帧同步逻辑"
    },
    "sensor_missing_ratio_high": {
        "issue": "传感器数据缺失率过高",
        "suggestion": "增加传感器采样率或检查AirSim连接",
        "action": "降低 SENSOR_LOG_EVERY_N 参数"
    },
    "timestamp_gap_large": {
        "issue": "时间戳间隙过大",
        "suggestion": "系统卡顿或存在数据跳跃，检查系统性能",
        "action": "降低 CAPTURE_FPS 或关闭其他耗时操作"
    },
    "bbox_invalid_coords": {
        "issue": "边界框坐标无效",
        "suggestion": "检查坐标投影算法",
        "action": "验证相机内参和投影矩阵"
    },
    "drone_visibility_low": {
        "issue": "无人机可见性低于预期",
        "suggestion": "调整观察视角或飞行轨迹",
        "action": "修改 OBSERVER_FIXED_DIST 或 OBSERVER_FIXED_HEIGHT"
    }
}


# ============================================================================
# 实用函数
# ============================================================================

def load_config(preset_name: str = "standard") -> dict:
    """加载预定义的配置"""
    return PRESETS.get(preset_name, CONFIG)


def save_config(config: dict, filename: str = "validation_config.json"):
    """保存配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置已保存: {filename}")


def get_suggestion(issue_key: str) -> dict:
    """获取特定问题的改进建议"""
    return IMPROVEMENT_SUGGESTIONS.get(issue_key, {})


def print_config_summary():
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("📋 数据质量检验配置摘要")
    print("=" * 70)
    
    print("\n【验证阈值】")
    for category, thresholds in CONFIG["validation_thresholds"].items():
        print(f"\n  {category}:")
        for key, value in thresholds.items():
            print(f"    - {key}: {value}")
    
    print("\n【检验对象】")
    enabled = [v for v, settings in CONFIG["vehicles"].items() if settings["enabled"]]
    print(f"  启用: {', '.join(enabled)}")
    
    print("\n【可视化设置】")
    print(f"  输出视频: {CONFIG['visualization']['output_video']}")
    print(f"  编码器: {CONFIG['visualization']['output_codec']}")
    print(f"  采样帧目录: {CONFIG['visualization']['sample_frames_dir']}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # 示例：打印配置摘要
    print_config_summary()
    
    # 示例：保存不同预设
    for preset_name in PRESETS.keys():
        config = load_config(preset_name)
        print(f"预设 '{preset_name}' 已加载")
    
    # 示例：获取改进建议
    suggestion = get_suggestion("time_sync_mismatch")
    print(f"\n改进建议示例:")
    print(f"  问题: {suggestion.get('issue')}")
    print(f"  建议: {suggestion.get('suggestion')}")
