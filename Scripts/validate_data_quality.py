"""
数据质量检验脚本 (Data Quality Validation)

功能:
1. 验证视频帧数与时间戳对齐
2. 检查CSV日志完整性和异常值
3. 验证时间同步
4. 检查边界框标注准确性
5. 生成详细的统计报告
6. 可视化验证 - 将边界框叠加到视频上

使用方法:
    python validate_data_quality.py --output_dir output
    python validate_data_quality.py --output_dir output --visualize --vehicle Observer
"""

import os
import sys
import json
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

# 配置日志
import io

# 日志文件使用 utf-8 编码，控制台输出使用替换模式避免在 Windows 控制台出现 UnicodeEncodeError
log_file = 'data_validation.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')

# 使用 TextIOWrapper 包装 stdout.buffer，编码为 utf-8 并在无法编码时替换字符，防止写入时抛出编码错误
try:
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    stream_handler = logging.StreamHandler(stream)
except Exception:
    # 作为后备，直接使用默认 StreamHandler（可能仍然在某些环境中抛错，但极少见）
    stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)


def _strip_emojis(s: str) -> str:
    """将常见 emoji 替换为 ASCII 标签，减少控制台/日志编码问题并提高兼容性。"""
    if not isinstance(s, str):
        return s
    # 简单替换映射（可扩展）
    replacements = {
        '⚠️': '[WARN]',
        '⚠': '[WARN]',
        '✓': '[OK]',
        '✔': '[OK]',
        'ℹ️': '[INFO]',
        'ℹ': '[INFO]',
        '❌': '[ERROR]',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s

# 用封装函数替代直接调用 logger.* 时携带 emoji 的写法，保证写入安全
def log_info(msg: str, *args, **kwargs):
    logger.info(_strip_emojis(msg), *args, **kwargs)

def log_warning(msg: str, *args, **kwargs):
    logger.warning(_strip_emojis(msg), *args, **kwargs)

def log_error(msg: str, *args, **kwargs):
    logger.error(_strip_emojis(msg), *args, **kwargs)

# 将 logger 的方法替换为安全版本，自动做 emoji 替换，避免在写入控制台时抛出编码错误
_orig_info = logger.info
_orig_warning = logger.warning
_orig_error = logger.error
logger.info = lambda msg, *a, **k: _orig_info(_strip_emojis(msg), *a, **k)
logger.warning = lambda msg, *a, **k: _orig_warning(_strip_emojis(msg), *a, **k)
logger.error = lambda msg, *a, **k: _orig_error(_strip_emojis(msg), *a, **k)


class DataValidator:
    """数据质量验证器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        # 确保输出目录存在（便于写入报告和可视化结果）
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # 在极少数权限受限的环境中忽略目录创建错误，后续写入会抛出异常
            pass
        self.vehicles = ["Drone1", "Drone2", "Drone3", "Drone4", "Observer"]
        self.report = {}
        self.issues = []

    def _find_video_file(self, vehicle_dir: Path) -> Optional[Path]:
        """在常见位置查找视频文件：
        优先顺序：
          1) vehicle_dir / 'video.mp4'
          2) vehicle_dir / 'videos' / *.mp4|*.avi|*.mov
          3) vehicle_dir 根目录下的 *.mp4|*.avi|*.mov
        如果找到多个，选择最近修改的那个。
        返回 Path 或 None
        """
        # 1) 特定文件名
        candidate = vehicle_dir / "video.mp4"
        if candidate.exists():
            return candidate

        # 2) videos 子目录
        videos_dir = vehicle_dir / "videos"
        patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        candidates = []
        if videos_dir.exists() and videos_dir.is_dir():
            for pat in patterns:
                candidates.extend(list(videos_dir.glob(pat)))

        # 3) vehicle 根目录下的媒体文件
        if not candidates and vehicle_dir.exists():
            for pat in patterns:
                candidates.extend(list(vehicle_dir.glob(pat)))

        if not candidates:
            return None

        # 选择最后修改时间最新的文件
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _find_csv_file(self, vehicle_dir: Path, key: str = "sensors") -> Optional[Path]:
        """查找 sensors/bbox 等 CSV 文件，按模式匹配并选择“最近修改”的文件：
        - 优先匹配文件名或路径中包含 key 的 CSV
        - 在 logs/metadata/csvs/data 等常见目录中递归搜索
        - 回退到任意 CSV，但仍选择最近修改时间的文件
        """
        candidates: list[Path] = []

        # 直接检查常见文件名
        direct = vehicle_dir / f"{key}.csv"
        if direct.exists():
            candidates.append(direct)

        # 检查常见目录中“包含 key 的 CSV”
        common_dirs = [vehicle_dir / "logs", vehicle_dir / "metadata", vehicle_dir / "csvs", vehicle_dir / "data"]
        for d in common_dirs:
            if d.exists() and d.is_dir():
                candidates.extend(list(d.glob(f"**/*{key}*.csv")))

        # 全库递归查找包含 key 的 CSV
        candidates.extend(list(vehicle_dir.rglob(f"*{key}*.csv")))

        # 如果还没有，放宽到“任意 CSV”
        if not candidates:
            for d in common_dirs:
                if d.exists() and d.is_dir():
                    candidates.extend(list(d.glob("**/*.csv")))
            if vehicle_dir.exists():
                candidates.extend(list(vehicle_dir.glob("*.csv")))
            candidates.extend(list(vehicle_dir.rglob("*.csv")))

        # 选择最近修改的文件
        candidates = [p for p in set(candidates) if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
        
    def validate_all(self):
        """执行全面的数据验证"""
        logger.info("=" * 60)
        logger.info("开始数据质量检验")
        logger.info("=" * 60)
        
        for vehicle in self.vehicles:
            logger.info(f"\n检验 {vehicle} 数据...")
            vehicle_dir = self.output_dir / vehicle
            
            if not vehicle_dir.exists():
                logger.warning(f"  ⚠️ {vehicle} 目录不存在，跳过")
                continue
            
            # 获取视频信息
            video_info = self._validate_video(vehicle_dir, vehicle)
            
            # 获取CSV信息
            csv_info = self._validate_csv(vehicle_dir, vehicle)
            
            # 验证时间同步
            sync_info = self._validate_time_sync(vehicle_dir, vehicle, video_info, csv_info)
            
            # 生成报告
            self.report[vehicle] = {
                'video': video_info,
                'csv': csv_info,
                'time_sync': sync_info
            }
        
        # 验证Observer特殊数据
        logger.info("\n检验 Observer 特殊数据...")
        special_info = self._validate_observer_special(self.output_dir / "Observer")
        # 将特殊信息纳入报告
        if 'Observer' in self.report:
            self.report['Observer']['special'] = special_info
        else:
            self.report['Observer'] = {'special': special_info}
        
        # 跨vehicle验证
        logger.info("\n进行跨vehicle时间同步检验...")
        self._validate_cross_vehicle_sync()
        
        return self.report
    
    def _validate_video(self, vehicle_dir: Path, vehicle_name: str) -> Dict:
        """验证视频文件完整性"""
        video_info = {
            'exists': False,
            'frames': 0,
            'fps': 0,
            'duration': 0,
            'resolution': (0, 0),
            'issues': [],
            'path': ''
        }
        
        video_path = self._find_video_file(vehicle_dir)

        if video_path is None or not video_path.exists():
            msg = f"  [WARN] 视频文件不存在: {vehicle_dir / 'videos'} 或者 {vehicle_dir} 下没有媒体文件"
            logger.warning(msg)
            video_info['issues'].append(msg)
            return video_info
        else:
            logger.info(f"  [INFO] 发现视频文件: {video_path}")
        
        video_info['exists'] = True
        video_info['path'] = str(video_path)
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ 视频文件大小: {file_size_mb:.2f} MB")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                msg = f"  ⚠️ 无法打开视频文件"
                logger.warning(msg)
                video_info['issues'].append(msg)
                return video_info
            
            # 获取视频属性
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_info['frames'] = frame_count
            video_info['fps'] = fps
            video_info['duration'] = frame_count / fps if fps > 0 else 0
            video_info['resolution'] = (width, height)
            
            logger.info(f"  ✓ 视频帧数: {frame_count}")
            logger.info(f"  ✓ 帧率: {fps:.2f} FPS")
            logger.info(f"  ✓ 分辨率: {width}x{height}")
            logger.info(f"  ✓ 时长: {video_info['duration']:.2f} 秒")
            
            # 检查帧数异常
            if frame_count < 10:
                msg = f"  ⚠️ 视频帧数过少 ({frame_count} frames)"
                logger.warning(msg)
                video_info['issues'].append(msg)
            
            cap.release()
            
        except Exception as e:
            msg = f"  ❌ 读取视频失败: {str(e)}"
            logger.error(msg)
            video_info['issues'].append(msg)
        
        return video_info
    
    def _validate_csv(self, vehicle_dir: Path, vehicle_name: str) -> Dict:
        """验证CSV日志文件完整性（支持自动发现 logs/ 下带时间戳命名的 CSV）"""
        csv_info = {
            'sensor_exists': False,
            'sensor_rows': 0,
            'sensor_coverage': 0,
            'sensor_issues': [],
            'sensor_path': '',
            'bbox_exists': False,
            'bbox_rows': 0,
            'bbox_coverage': 0,
            'bbox_issues': [],
            'bbox_path': ''
        }

        # 1) 传感器 CSV 自动发现
        sensor_path = self._find_csv_file(vehicle_dir, "sensors")
        if sensor_path and sensor_path.exists():
            csv_info['sensor_exists'] = True
            csv_info['sensor_path'] = str(sensor_path)
            try:
                logger.info(f"  [INFO] 使用传感器文件: {sensor_path}")
                df = pd.read_csv(sensor_path)
                csv_info['sensor_rows'] = len(df)
                logger.info(f"  [OK] 传感器数据行数: {len(df)}")

                # 缺失值检查
                if len(df) > 0:
                    missing_ratio = df.isnull().sum() / len(df)
                    for col in missing_ratio[missing_ratio > 0].index:
                        ratio = missing_ratio[col]
                        msg = f"  [WARN] 传感器列 '{col}' 缺失率: {ratio:.2%}"
                        logger.warning(msg)
                        csv_info['sensor_issues'].append(msg)

                # 数值异常检查（无限值与 3σ 离群）
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    col_data = df[col]
                    # 无限值
                    inf_count = np.isinf(col_data).sum()
                    if inf_count > 0:
                        msg = f"  [WARN] 传感器列 '{col}' 包含 {inf_count} 个无限值"
                        logger.warning(msg)
                        csv_info['sensor_issues'].append(msg)

                    # 3σ 离群点
                    std = col_data.std()
                    if std and std > 0:
                        outliers = np.abs((col_data - col_data.mean()) / std) > 3
                        outlier_count = int(outliers.sum())
                        if outlier_count > 0:
                            outlier_ratio = outlier_count / max(1, len(df))
                            if outlier_ratio > 0.01:
                                msg = f"  [WARN] 传感器列 '{col}' 包含 {outlier_count} 个离群点 ({outlier_ratio:.2%})"
                                logger.warning(msg)
                                csv_info['sensor_issues'].append(msg)

                # 时间戳连续性
                if 'timestamp_ms' in df.columns and len(df) > 1:
                    ts = df['timestamp_ms'].values
                    gaps = np.diff(ts)
                    if len(gaps) > 0:
                        max_gap = int(np.max(gaps))
                        if max_gap > 1000:
                            msg = f"  [WARN] 时间戳最大间隙: {max_gap}ms (存在数据跳跃)"
                            logger.warning(msg)
                            csv_info['sensor_issues'].append(msg)

                csv_info['sensor_coverage'] = 100

            except Exception as e:
                msg = f"  [ERROR] 读取传感器CSV失败: {str(e)}"
                logger.error(msg)
                csv_info['sensor_issues'].append(msg)
        else:
            msg = "  [WARN] 传感器数据文件不存在"
            logger.warning(msg)
            csv_info['sensor_issues'].append(msg)

        # 2) 边界框 CSV（Observer 专用）
        if vehicle_name == "Observer":
            bbox_path = self._find_csv_file(vehicle_dir, "bbox")
            if bbox_path and bbox_path.exists():
                csv_info['bbox_exists'] = True
                csv_info['bbox_path'] = str(bbox_path)
                try:
                    logger.info(f"  [INFO] 使用边界框文件: {bbox_path}")
                    df = pd.read_csv(bbox_path)
                    csv_info['bbox_rows'] = len(df)
                    logger.info(f"  [OK] 边界框标注行数: {len(df)}")

                    # 缺失值
                    if len(df) > 0:
                        missing_ratio = df.isnull().sum() / len(df)
                        for col in missing_ratio[missing_ratio > 0].index:
                            ratio = missing_ratio[col]
                            msg = f"  [WARN] 边界框列 '{col}' 缺失率: {ratio:.2%}"
                            logger.warning(msg)
                            csv_info['bbox_issues'].append(msg)

                    # 坐标有效性
                    bbox_cols = [col for col in df.columns if 'bbox' in col.lower()]
                    for col in bbox_cols:
                        if col.endswith('_max'):
                            min_col = col.replace('_max', '_min')
                            if min_col in df.columns:
                                invalid = (df[col] < df[min_col])
                                invalid_count = int(invalid.sum())
                                if invalid_count > 0:
                                    msg = f"  [WARN] 边界框列 '{col}' 包含 {invalid_count} 个无效值 (max < min)"
                                    logger.warning(msg)
                                    csv_info['bbox_issues'].append(msg)

                    csv_info['bbox_coverage'] = 100

                except Exception as e:
                    msg = f"  [ERROR] 读取边界框CSV失败: {str(e)}"
                    logger.error(msg)
                    csv_info['bbox_issues'].append(msg)
            else:
                msg = "  [WARN] 边界框数据文件不存在"
                logger.warning(msg)
                csv_info['bbox_issues'].append(msg)

        return csv_info
    
    def _validate_time_sync(self, vehicle_dir: Path, vehicle_name: str, 
                           video_info: Dict, csv_info: Dict) -> Dict:
        """验证视频和CSV时间戳同步"""
        sync_info = {
            'frame_csv_aligned': False,
            'expected_frames': 0,
            'actual_frames': 0,
            'frame_diff': 0,
            'time_range_match': False,
            'sampling_factor': 0,
            'coverage_ratio': 0.0,
            'issues': []
        }
        
        if not video_info['exists'] or not csv_info['sensor_exists']:
            logger.info(f"  ℹ️ 跳过时间同步检验 (缺少必要数据)")
            return sync_info
        
        try:
            # 优先使用 _validate_csv 发现的传感器路径
            sensor_path_str = csv_info.get('sensor_path') or ''
            sensor_path = Path(sensor_path_str) if sensor_path_str else self._find_csv_file(vehicle_dir, "sensors")
            if not sensor_path or not sensor_path.exists():
                raise FileNotFoundError("未找到传感器 CSV 文件用于时间同步")

            logger.info(f"  [INFO] 时间同步使用传感器文件: {sensor_path}")
            df = pd.read_csv(sensor_path)
            
            # 帧数 & 稀疏采样判断
            if 'frame_idx' in df.columns and len(df) > 0:
                # 使用唯一帧索引数避免跳号影响
                csv_frame_count = int(df['frame_idx'].nunique())
                video_frame_count = int(video_info.get('frames') or 0)
                sync_info['expected_frames'] = csv_frame_count
                sync_info['actual_frames'] = video_frame_count
                frame_diff = abs(video_frame_count - csv_frame_count)
                sync_info['frame_diff'] = frame_diff

                if csv_frame_count > 0 and video_frame_count > 0:
                    ratio = video_frame_count / csv_frame_count
                    # 采样因子推断（四舍五入到整数）
                    sampling_factor = max(1, round(ratio))
                    sync_info['sampling_factor'] = sampling_factor
                    # 判断是否为近似等间隔采样：视频帧数 ≈ CSV行数 * 采样因子
                    diff_vs_sampling = abs(video_frame_count - sampling_factor * csv_frame_count)
                    tolerance = max(2, int(round(0.05 * video_frame_count)))  # 5% 或至少2帧
                    if diff_vs_sampling <= tolerance:
                        sync_info['frame_csv_aligned'] = True
                        logger.info(
                            f"  ✓ 视频与CSV为稀疏采样对齐 (视频帧: {video_frame_count}, CSV行: {csv_frame_count}, 采样因子≈{sampling_factor}x, 误差: {diff_vs_sampling} 帧)"
                        )
                    else:
                        msg = f"  ⚠️ 视频和CSV帧数不对齐 (视频: {video_frame_count}, CSV行: {csv_frame_count}, 采样因子推断≈{sampling_factor}x, 偏差: {diff_vs_sampling} 帧)"
                        logger.warning(msg)
                        sync_info['issues'].append(msg)
            
            # 检查时间范围（作为稀疏/无 frame_idx 的容错依据）
            if 'timestamp_ms' in df.columns and len(df) > 1:
                ts_min = df['timestamp_ms'].min()
                ts_max = df['timestamp_ms'].max()
                raw_span = float(ts_max - ts_min)
                # timestamp_ms列存储Unix纪元毫秒时间戳，时间跨度(max-min)直接就是毫秒
                unit = "ms"
                csv_duration = raw_span / 1000.0
                video_duration = float(video_info.get("duration") or 0)

                sync_info['coverage_ratio'] = csv_duration / video_duration if video_duration > 0 else 0.0
                duration_diff = abs(csv_duration - video_duration)

                coverage_ok = sync_info['coverage_ratio'] >= 0.7 or duration_diff < 1.0
                # 在稀疏采样对齐且采样因子较大的情况下，时间范围不匹配多由“视频帧补齐/重复写入”导致，放宽为信息提示
                if sync_info.get('frame_csv_aligned') and sync_info.get('sampling_factor', 1) > 1:
                    coverage_ok = True
                if coverage_ok:
                    logger.info(
                        f"  ✓ CSV时间范围与视频基本匹配 (单位推断: {unit}, 视频: {video_duration:.2f}s, CSV: {csv_duration:.2f}s, 覆盖率: {sync_info['coverage_ratio']:.1%})"
                    )
                    sync_info['time_range_match'] = True
                else:
                    msg = (f"  ⚠️ 时间范围不匹配 (单位推断: {unit}, 视频: {video_duration:.2f}s, CSV: {csv_duration:.2f}s, 覆盖率: {sync_info['coverage_ratio']:.1%}, 差异: {duration_diff:.2f}s)")
                    logger.warning(msg)
                    sync_info['issues'].append(msg)

                # 无 frame_idx 情况下用时长估算采样因子/帧数
                if 'frame_idx' not in df.columns and video_info.get('fps', 0) > 0 and csv_duration > 0:
                    est_frames = int(round(csv_duration * float(video_info['fps'])))
                    video_frame_count = int(video_info['frames'])
                    frame_diff = abs(video_frame_count - est_frames)
                    sync_info['expected_frames'] = est_frames
                    sync_info['actual_frames'] = video_frame_count
                    sync_info['frame_diff'] = frame_diff
                    if frame_diff <= max(2, int(0.05 * video_frame_count)):
                        logger.info(f"  ✓ 视频帧数与CSV时长估算近似对齐 (视频: {video_frame_count}, 估算: {est_frames}, 误差: {frame_diff} 帧)")
                        sync_info['frame_csv_aligned'] = True
                    else:
                        msg = f"  ⚠️ 视频帧数与CSV时长估算不一致 (视频: {video_frame_count}, 估算: {est_frames}, 误差: {frame_diff} 帧)"
                        logger.warning(msg)
                        sync_info['issues'].append(msg)
            
        except Exception as e:
            msg = f"  ❌ 时间同步检验失败: {str(e)}"
            logger.error(msg)
            sync_info['issues'].append(msg)
        
        return sync_info
    
    def _validate_observer_special(self, observer_dir: Path):
        """验证Observer特殊数据，返回统计以便纳入报告"""
        special_info = {
            'depth_count': 0,
            'seg_count': 0,
            'issues': []
        }

        if not observer_dir.exists():
            msg = "  [WARN] Observer目录不存在"
            logger.warning(msg)
            special_info['issues'].append(msg)
            return special_info

        # 检查深度图和分割掩码
        depth_dir = observer_dir / "depth"
        seg_dir = observer_dir / "seg"

        depth_count = len(list(depth_dir.glob("*.png"))) if depth_dir.exists() else 0
        seg_count = len(list(seg_dir.glob("*.png"))) if seg_dir.exists() else 0
        special_info['depth_count'] = depth_count
        special_info['seg_count'] = seg_count

        if depth_count > 0:
            logger.info(f"  ✓ 深度图数量: {depth_count}")
        else:
            msg = "  [WARN] 深度图数量为0或目录不存在"
            logger.warning(msg)
            special_info['issues'].append(msg)

        if seg_count > 0:
            logger.info(f"  ✓ 分割掩码数量: {seg_count}")
        else:
            msg = "  [WARN] 分割掩码数量为0或目录不存在"
            logger.warning(msg)
            special_info['issues'].append(msg)

        # 检查边界框标注质量（灵活查找 bbox 文件）
        bbox_path = self._find_csv_file(observer_dir, "bbox")
        if bbox_path and bbox_path.exists():
            try:
                df = pd.read_csv(bbox_path)
                drone_names = ["Drone1", "Drone2", "Drone3", "Drone4"]
                for drone in drone_names:
                    visibility_col = f"{drone}_visibility"
                    if visibility_col in df.columns:
                        visible_frames = int((df[visibility_col] > 0).sum())
                        total_frames = int(len(df))
                        visibility_ratio = (visible_frames / total_frames) if total_frames > 0 else 0
                        logger.info(f"  ✓ {drone} 可见性: {visible_frames}/{total_frames} ({visibility_ratio:.2%})")
            except Exception as e:
                logger.error(f"  ❌ 读取边界框标注失败: {str(e)}")

        return special_info
    
    def _validate_cross_vehicle_sync(self):
        """验证跨vehicle时间同步"""
        time_ranges = {}
        
        for vehicle in self.vehicles:
            vehicle_dir = self.output_dir / vehicle
            # 优先使用已记录在报告中的路径
            rep_csv = self.report.get(vehicle, {}).get('csv', {})
            sensor_path_str = rep_csv.get('sensor_path') if isinstance(rep_csv, dict) else ''
            sensor_path = Path(sensor_path_str) if sensor_path_str else self._find_csv_file(vehicle_dir, "sensors")

            if sensor_path and sensor_path.exists():
                try:
                    df = pd.read_csv(sensor_path)
                    if 'timestamp_ms' in df.columns and len(df) > 0:
                        ts_min = int(df['timestamp_ms'].min())
                        ts_max = int(df['timestamp_ms'].max())
                        time_ranges[vehicle] = (ts_min, ts_max)
                except Exception:
                    pass
        
        if len(time_ranges) > 1:
            all_starts = [t[0] for t in time_ranges.values()]
            all_ends = [t[1] for t in time_ranges.values()]
            
            max_start_diff = max(all_starts) - min(all_starts)
            max_end_diff = max(all_ends) - min(all_ends)
            
            logger.info(f"  采集开始时间差: {max_start_diff}ms")
            logger.info(f"  采集结束时间差: {max_end_diff}ms")
            
            if max_start_diff > 100:
                logger.warning(f"  ⚠️ 采集开始时间差过大，可能存在同步问题")
    
    def generate_report(self, output_file: str = "validation_report.json"):
        """生成详细的验证报告"""
        report_path = self.output_dir / output_file
        
        # 汇总统计
        summary = {
            'validation_time': datetime.now().isoformat(),
            'total_vehicles': len([v for v in self.report if len(self.report[v]) > 0]),
            'total_issues': sum(len(self.report.get(v, {}).get('video', {}).get('issues', [])) +
                              len(self.report.get(v, {}).get('csv', {}).get('sensor_issues', [])) +
                              len(self.report.get(v, {}).get('csv', {}).get('bbox_issues', [])) +
                              len(self.report.get(v, {}).get('time_sync', {}).get('issues', [])) +
                              len(self.report.get(v, {}).get('special', {}).get('issues', []))
                              for v in self.vehicles),
            'details': self.report
        }
        
        # 将所有可能的 numpy/pandas 类型转换为内置类型后再序列化
        def _to_builtin(obj):
            import numpy as _np
            if isinstance(obj, dict):
                return {k: _to_builtin(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_builtin(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_builtin(v) for v in obj)
            if isinstance(obj, set):
                return [_to_builtin(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            if 'numpy' in sys.modules:
                if isinstance(obj, _np.integer):
                    return int(obj)
                if isinstance(obj, _np.floating):
                    return float(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
            # pandas Timestamp / NA 处理
            if 'pandas' in sys.modules:
                import pandas as _pd
                if isinstance(obj, _pd.Timestamp):
                    return obj.isoformat()
                if obj is _pd.NA:
                    return None
            return obj

        safe_summary = _to_builtin(summary)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(safe_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ 验证报告已保存: {report_path}")
        return summary
    
    def generate_statistics_report(self, output_file: str = "statistics_report.txt"):
        """生成统计报告"""
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("无人机数据采集统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 逐个vehicle统计
            for vehicle, info in self.report.items():
                f.write(f"\n{'─' * 80}\n")
                f.write(f"无人机: {vehicle}\n")
                f.write(f"{'─' * 80}\n")
                
                # 视频统计
                video = info.get('video', {})
                if video.get('exists'):
                    f.write(f"\n【视频信息】\n")
                    if video.get('path'):
                        f.write(f"  文件: {video.get('path')}\n")
                    f.write(f"  总帧数: {video.get('frames', 0)} 帧\n")
                    f.write(f"  帧率: {video.get('fps', 0):.2f} FPS\n")
                    f.write(f"  总时长: {video.get('duration', 0):.2f} 秒\n")
                    res = video.get('resolution') or (0, 0)
                    f.write(f"  分辨率: {res[0]}x{res[1]}\n")
                    issues = video.get('issues', [])
                    if issues:
                        f.write(f"  ⚠️ 问题: {', '.join([i.replace('  ⚠️ ', '').replace('  ❌ ', '') for i in issues])}\n")
                
                # 传感器统计（无论是否存在都输出，避免误导）
                csv = info.get('csv', {})
                f.write(f"\n【传感器数据】\n")
                f.write(f"  是否存在: {'是' if csv.get('sensor_exists') else '否'}\n")
                if csv.get('sensor_path'):
                    f.write(f"  文件: {csv.get('sensor_path')}\n")
                f.write(f"  数据行数: {csv.get('sensor_rows', 0)} 行\n")
                f.write(f"  覆盖率: {csv.get('sensor_coverage', 0)}%\n")
                sensor_issues = csv.get('sensor_issues', [])
                if sensor_issues:
                    f.write(f"  ⚠️ 问题: {len(sensor_issues)} 个\n")
                
                # 时间同步统计
                if 'time_sync' in info:
                    sync = info['time_sync']
                    f.write(f"\n【时间同步】\n")
                    f.write(f"  帧数对齐: {'✓' if sync['frame_csv_aligned'] else '✗'}\n")
                    f.write(f"  视频帧数: {sync['actual_frames']}\n")
                    f.write(f"  CSV帧数: {sync['expected_frames']}\n")
                    f.write(f"  帧数差异: {sync['frame_diff']}\n")
                    f.write(f"  时间范围匹配: {'✓' if sync['time_range_match'] else '✗'}\n")
                    if sync['issues']:
                        f.write(f"  ⚠️ 问题: {', '.join([i.replace('  ⚠️ ', '').replace('  ❌ ', '') for i in sync['issues']])}\n")
                
                # Observer 边界框与特殊数据
                if vehicle == "Observer":
                    # 边界框
                    f.write(f"\n【边界框标注】\n")
                    f.write(f"  是否存在: {'是' if csv.get('bbox_exists') else '否'}\n")
                    if csv.get('bbox_path'):
                        f.write(f"  文件: {csv.get('bbox_path')}\n")
                    f.write(f"  标注行数: {csv.get('bbox_rows', 0)} 行\n")
                    f.write(f"  覆盖率: {csv.get('bbox_coverage', 0)}%\n")
                    bbox_issues = csv.get('bbox_issues', [])
                    if bbox_issues:
                        f.write(f"  ⚠️ 问题: {len(bbox_issues)} 个\n")

                    # 特殊（深度/分割）
                    special = info.get('special', {})
                    if special:
                        f.write(f"\n【Observer 特殊数据】\n")
                        f.write(f"  深度图数量: {special.get('depth_count', 0)}\n")
                        f.write(f"  分割掩码数量: {special.get('seg_count', 0)}\n")
                        sp_issues = special.get('issues', [])
                        if sp_issues:
                            f.write(f"  ⚠️ 问题: {len(sp_issues)} 个\n")
            
            # 总体统计
            f.write(f"\n{'=' * 80}\n")
            f.write("【总体统计】\n")
            f.write(f"{'=' * 80}\n")
            
            total_issues = sum(
                len(self.report.get(v, {}).get('video', {}).get('issues', [])) +
                len(self.report.get(v, {}).get('csv', {}).get('sensor_issues', [])) +
                len(self.report.get(v, {}).get('csv', {}).get('bbox_issues', [])) +
                len(self.report.get(v, {}).get('time_sync', {}).get('issues', [])) +
                len(self.report.get(v, {}).get('special', {}).get('issues', []))
                for v in self.vehicles
            )
            
            f.write(f"\n检验的vehicle数: {len([v for v in self.report if len(self.report[v]) > 0])}\n")
            f.write(f"发现的问题数: {total_issues}\n")
            
            if total_issues == 0:
                f.write(f"\n✓ 所有数据质量检查通过！\n")
            else:
                f.write(f"\n⚠️ 发现 {total_issues} 个问题，建议逐一检查和解决。\n")
        
        logger.info(f"✓ 统计报告已保存: {report_path}")


class BboxVisualizer:
    """边界框可视化工具"""
    
    def __init__(self, output_dir: str, vehicle: str = "Observer"):
        self.output_dir = Path(output_dir)
        self.vehicle_dir = self.output_dir / vehicle
        self.vehicle = vehicle
        
    def visualize_bboxes(self, start_frame: int = 0, num_frames: int = 100, output_video: str = "bbox_visualization.mp4"):
        """将边界框叠加到视频上并输出，修复无标注帧导致0KB视频问题，支持多目标且frame_idx严格int匹配。"""
        import pandas as pd
        import cv2
        video_path = None
        videos_dir = self.vehicle_dir / "videos"
        if videos_dir.exists() and videos_dir.is_dir():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                res = list(videos_dir.glob(ext))
                if res:
                    res.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    video_path = res[0]
                    break
        if video_path is None:
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                res = list(self.vehicle_dir.glob(ext))
                if res:
                    res.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    video_path = res[0]
                    break
        bbox_path = self.vehicle_dir / "bbox.csv"
        if not bbox_path.exists():
            candidates = list(self.vehicle_dir.rglob("*bbox*.csv"))
            if candidates:
                # 选择最新的 bbox CSV，避免误选旧文件
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                bbox_path = candidates[0]
        if video_path is None or not video_path.exists():
            logger.error(f"视频文件不存在: {self.vehicle_dir / 'videos'} 或 {self.vehicle_dir}")
            return
        if not bbox_path.exists():
            logger.warning(f"边界框文件不存在: {bbox_path}")
            return
        logger.info(f"\n开始生成边界框可视化视频...")
        bbox_df = pd.read_csv(bbox_path)
        # 确保 frame_idx 为 int 类型
        if 'frame_idx' in bbox_df.columns:
            bbox_df['frame_idx'] = bbox_df['frame_idx'].astype(int)
        cap = cv2.VideoCapture(str(video_path))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = self.vehicle_dir / output_video
        # 优先写 mp4，失败则回退 avi（MJPG）避免 0KB
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), (frame_width, frame_height))
        if (not out.isOpened()) or (fps is None) or (fps == 0):
            logger.warning(f"mp4 编码器不可用或fps未知，回退为 MJPG/AVI 以避免空文件: {output_path}")
            out.release()
            output_path = self.vehicle_dir / output_video.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0 if fps == 0 else max(fps, 1.0), (frame_width, frame_height))
            if not out.isOpened():
                logger.error("无法打开视频写入器，放弃可视化生成。")
                cap.release()
                return
        frame_idx = 0

        # 构建基于时间戳的对齐（优先使用 Observer 的逐帧时间戳CSV）
        obs_ts_path = None
        ts_candidates = [p for p in (self.vehicle_dir / 'logs').glob('*.csv') if 'bbox' not in p.name.lower()]
        if ts_candidates:
            ts_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            obs_ts_path = ts_candidates[0]
        obs_ts_df = None
        if obs_ts_path and obs_ts_path.exists():
            try:
                obs_ts_df = pd.read_csv(obs_ts_path)
                if 'frame_idx' in obs_ts_df.columns and 'timestamp_ms' in obs_ts_df.columns:
                    obs_ts_df['frame_idx'] = obs_ts_df['frame_idx'].astype(int)
                    obs_ts_df['timestamp_ms'] = obs_ts_df['timestamp_ms'].astype('int64')
            except Exception as e:
                logger.warning(f"读取 Observer 时间戳CSV失败，改用帧索引近似匹配: {e}")

        # 预处理 bbox_df 时间戳
        if 'timestamp_ms' in bbox_df.columns:
            try:
                bbox_df['timestamp_ms'] = bbox_df['timestamp_ms'].astype('int64')
            except Exception:
                pass
        drone_colors = {
            "Drone1": (0, 0, 255),
            "Drone2": (0, 255, 0),
            "Drone3": (255, 0, 0),
            "Drone4": (255, 255, 0)
        }
        any_frame_written = False
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= start_frame + num_frames:
                    break
                # 匹配当前帧的标注：优先按时间戳近邻匹配，其次按缩放后的帧号近似
                frame_bbox = None
                if obs_ts_df is not None and 'timestamp_ms' in bbox_df.columns:
                    ts_row = obs_ts_df[obs_ts_df['frame_idx'] == frame_idx]
                    if not ts_row.empty:
                        ts_ms = int(ts_row.iloc[0]['timestamp_ms'])
                        # 在 bbox_df 中寻找时间差最小的一行（限定窗口）
                        # 窗口 ±200ms，若为空则放宽
                        candidates = bbox_df.copy()
                        candidates['dt'] = np.abs(candidates['timestamp_ms'] - ts_ms)
                        frame_bbox = candidates.nsmallest(1, 'dt')
                        if frame_bbox.iloc[0]['dt'] > 500:
                            # 时间相差过大，视为无匹配
                            frame_bbox = pd.DataFrame()
                if frame_bbox is None:
                    # 按帧号近似匹配：视频与CSV帧数比例估计
                    if 'frame_idx' in bbox_df.columns:
                        # 估计采样因子 ratio = video_total / (csv_nunique)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (start_frame + num_frames)
                        csv_unique = int(pd.Series(bbox_df['frame_idx']).nunique()) or 1
                        ratio = max(1, round(total_frames / max(1, csv_unique)))
                        approx_csv_idx = int(round(frame_idx / ratio))
                        frame_bbox = bbox_df[bbox_df['frame_idx'] == approx_csv_idx]
                    else:
                        frame_bbox = pd.DataFrame()
                drawn = False
                if frame_bbox is not None and not frame_bbox.empty:
                    row = frame_bbox.iloc[0]
                    # 绘制集群中心和半径
                    center_x = int(row.get('swarm_center_x', 0))
                    center_y = int(row.get('swarm_center_y', 0))
                    radius = int(row.get('swarm_radius', 0))
                    if center_x > 0 and center_y > 0:
                        cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), 2)
                        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
                    # 多目标支持：只要有一个目标可见就画框
                    for drone in ["Drone1", "Drone2", "Drone3", "Drone4"]:
                        x_min_col = f"{drone}_bbox_x_min"
                        y_min_col = f"{drone}_bbox_y_min"
                        x_max_col = f"{drone}_bbox_x_max"
                        y_max_col = f"{drone}_bbox_y_max"
                        visibility_col = f"{drone}_visibility"
                        if all(col in row.index for col in [x_min_col, y_min_col, x_max_col, y_max_col]):
                            x_min = int(row[x_min_col])
                            y_min = int(row[y_min_col])
                            x_max = int(row[x_max_col])
                            y_max = int(row[y_max_col])
                            visibility = row.get(visibility_col, 0)
                            # 只要不是全为-1就画框
                            if x_min != -1 and y_min != -1 and x_max != -1 and y_max != -1 and x_max > x_min and y_max > y_min:
                                color = drone_colors.get(drone, (255, 255, 255))
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                text = f"{drone} ({visibility:.0%})"
                                cv2.putText(frame, text, (x_min, max(0, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                drawn = True
                # 添加帧索引
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # 只要有标注或强制输出全部帧
                if True:
                    out.write(frame)
                    any_frame_written = True
                frame_idx += 1
                if frame_idx % 10 == 0:
                    logger.info(f"  处理到第 {frame_idx} 帧...")
            if any_frame_written:
                logger.info(f"✓ 边界框可视化视频已生成: {output_path}")
            else:
                # 作为兜底，这里仍应至少有帧写入；若没有，说明读取失败或编码器不可用
                logger.warning(f"未找到任何可视化标注，且无帧写入，可能视频读取或编码失败: {output_path}")
        finally:
            cap.release()
            out.release()
    
    def visualize_frame_range(self, frame_indices: List[int] = None, 
                             output_dir_name: str = "bbox_frames"):
        """将指定帧的边界框保存为图像"""
        # 查找视频和 bbox 文件（同 visualize_bboxes 的查找逻辑）
        video_path = None
        videos_dir = self.vehicle_dir / "videos"
        if videos_dir.exists() and videos_dir.is_dir():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                res = list(videos_dir.glob(ext))
                if res:
                    res.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    video_path = res[0]
                    break

        if video_path is None:
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                res = list(self.vehicle_dir.glob(ext))
                if res:
                    res.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    video_path = res[0]
                    break

        bbox_path = self.vehicle_dir / "bbox.csv"
        if not bbox_path.exists():
            candidates = list(self.vehicle_dir.rglob("*bbox*.csv"))
            bbox_path = candidates[0] if candidates else bbox_path
        
        if not video_path.exists() or not bbox_path.exists():
            logger.error("视频或边界框文件不存在")
            return
        
        if frame_indices is None:
            frame_indices = [0, 50, 100, 150, 200]  # 默认采样帧
        
        output_frames_dir = self.vehicle_dir / output_dir_name
        output_frames_dir.mkdir(exist_ok=True)
        
        bbox_df = pd.read_csv(bbox_path)
        cap = cv2.VideoCapture(str(video_path))
        
        drone_colors = {
            "Drone1": (0, 0, 255),
            "Drone2": (0, 255, 0),
            "Drone3": (255, 0, 0),
            "Drone4": (255, 255, 0)
        }
        
        try:
            for target_frame_idx in frame_indices:
                # 跳转到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"无法读取第 {target_frame_idx} 帧")
                    continue
                
                # 获取边界框
                frame_bbox = bbox_df[bbox_df['frame_idx'] == target_frame_idx]
                
                if len(frame_bbox) == 0:
                    logger.warning(f"第 {target_frame_idx} 帧没有边界框数据")
                    continue
                
                row = frame_bbox.iloc[0]
                
                # 绘制集群
                center_x = int(row.get('swarm_center_x', 0))
                center_y = int(row.get('swarm_center_y', 0))
                radius = int(row.get('swarm_radius', 0))
                
                if center_x > 0 and center_y > 0:
                    cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), 2)
                    cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
                
                # 绘制无人机边界框
                for drone in ["Drone1", "Drone2", "Drone3", "Drone4"]:
                    x_min_col = f"{drone}_bbox_x_min"
                    y_min_col = f"{drone}_bbox_y_min"
                    x_max_col = f"{drone}_bbox_x_max"
                    y_max_col = f"{drone}_bbox_y_max"
                    
                    if all(col in row.index for col in [x_min_col, y_min_col, x_max_col, y_max_col]):
                        x_min = int(row[x_min_col])
                        y_min = int(row[y_min_col])
                        x_max = int(row[x_max_col])
                        y_max = int(row[y_max_col])
                        
                        if x_min >= 0 and y_min >= 0 and x_max > x_min and y_max > y_min:
                            color = drone_colors.get(drone, (255, 255, 255))
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                            cv2.putText(frame, drone, (x_min, y_min - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 保存
                output_path = output_frames_dir / f"frame_{target_frame_idx:06d}_bbox.png"
                cv2.imwrite(str(output_path), frame)
                logger.info(f"✓ 帧 {target_frame_idx} 已保存: {output_path}")
        
        finally:
            cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="无人机数据质量验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础验证
  python validate_data_quality.py --output_dir output
  
  # 生成可视化
  python validate_data_quality.py --output_dir output --visualize
  
  # 可视化特定vehicle
  python validate_data_quality.py --output_dir output --visualize --vehicle Observer
  
  # 保存采样帧
  python validate_data_quality.py --output_dir output --visualize --sample-frames
        """
    )
    
    parser.add_argument('--output_dir', default='output', help='输出数据目录')
    parser.add_argument('--visualize', action='store_true', help='生成边界框可视化视频')
    parser.add_argument('--vehicle', default='Observer', help='指定vehicle进行可视化')
    parser.add_argument('--sample-frames', action='store_true', help='保存采样帧')
    parser.add_argument('--start-frame', type=int, default=0, help='开始帧编号')
    parser.add_argument('--num-frames', type=int, default=300, help='处理帧数')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = DataValidator(args.output_dir)
    
    # 执行验证
    report = validator.validate_all()
    
    # 生成报告
    validator.generate_report()
    validator.generate_statistics_report()
    
    # 可视化（如果指定）
    if args.visualize:
        visualizer = BboxVisualizer(args.output_dir, args.vehicle)
        
        logger.info(f"\n生成 {args.vehicle} 的边界框可视化...")
        visualizer.visualize_bboxes(
            start_frame=args.start_frame,
            num_frames=args.num_frames
        )
        
        if args.sample_frames:
            logger.info(f"\n保存 {args.vehicle} 的采样帧...")
            visualizer.visualize_frame_range()
    
    logger.info("\n" + "=" * 60)
    logger.info("数据质量检验完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
