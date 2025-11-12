"""
独立的数据导出工具（MOT / COCO）
从原始的 `multi_drone_capture.py` 中提取出的导出功能，作为独立脚本使用。
用法示例：
    python export_tools.py --base-output ../output --export-mot

该脚本尽量兼容原始 CSV 列名（例如 Drone1_bbox_x_min/_y_min/_x_max/_y_max 或 Drone1_bbox_2d_x1/_y1/_x2/_y2），
并在没有运行时初始化 VEHICLES 的情况下从 CSV header 推断车辆名。
"""

import os
import csv
import json
import datetime
import argparse
import re

PREFERRED_VEHICLES = ["Drone1", "Drone2", "Drone3", "Drone4"]
DEFAULT_OBSERVER_NAME = "Observer"


def export_to_mot_format(base_output_dir=r"D:\Documents\Unreal Projects\uav\output", export_dir="mot_export",
                         vehicles=None, enable_observer_camera=True,
                         observer_vehicle_name=DEFAULT_OBSERVER_NAME,
                         preferred_vehicles=PREFERRED_VEHICLES,
                         observer_frame_rate=30,
                         capture_fps=20,
                         csv_file=None,
                         preview=0,
                         verbose=False,
                         visibility_threshold=0.3):
    """导出 MOT 格式（gt.txt + seqinfo.ini）。
    参数:
      base_output_dir: 原始数据根目录（包含每个 vehicle 的 logs/ 子目录）
      export_dir: 导出目录（相对于 base_output_dir）
      vehicles: 列表或 None；若 None，从 CSV header 回退推断
      enable_observer_camera: 是否处理 Observer 的数据
      observer_vehicle_name: Observer 在 output 下的文件夹名
      visibility_threshold: 可见性阈值（0.0-1.0），低于此值的 bbox 将被过滤（default=0.3）
    """
    """
    执行导出并返回导出目录与统计信息。
    返回 (export_path, summary_dict)
    summary_dict 示例: {"observer": {"gt_count": 120, "seq_length": 120}, "vehicles": {"Drone1_ego": True}}
    """
    print(f"开始导出MOT格式数据... (visibility_threshold={visibility_threshold})")
    export_path = os.path.join(base_output_dir, export_dir)
    os.makedirs(export_path, exist_ok=True)

    summary = {"observer": None, "vehicles": {}}

    # Observer
    if enable_observer_camera:
        observer_dir = os.path.join(base_output_dir, observer_vehicle_name)
        if os.path.exists(observer_dir):
            try:
                gt_count, seq_len = _export_observer_to_mot(observer_dir, export_path, vehicles=vehicles,
                                                           preferred_vehicles=preferred_vehicles,
                                                           sequence_name="observer_sequence",
                                                           observer_frame_rate=observer_frame_rate,
                                                           csv_file=csv_file,
                                                           preview=preview,
                                                           verbose=verbose,
                                                           visibility_threshold=visibility_threshold)
                summary["observer"] = {"gt_count": gt_count, "seq_length": seq_len}
            except Exception as e:
                print(f"Error exporting observer MOT: {e}")

    # Vehicles (ego view) - 仅生成 seqinfo
    if vehicles is None:
        vehicles = preferred_vehicles
    for v in vehicles:
        vehicle_dir = os.path.join(base_output_dir, v)
        if os.path.exists(vehicle_dir):
            try:
                _export_vehicle_to_mot(vehicle_dir, export_path, v, cap_fps=capture_fps)
                summary["vehicles"][f"{v}_ego"] = True
            except Exception as e:
                summary["vehicles"][f"{v}_ego"] = False
                print(f"Error exporting {v} ego MOT: {e}")

    print(f"MOT格式数据导出完成，保存至: {export_path}")
    return export_path, summary


def _export_observer_to_mot(observer_dir, export_path, vehicles=None, preferred_vehicles=None,
                            sequence_name="observer_sequence", observer_frame_rate=30,
                            csv_file=None, preview=0, verbose=False, visibility_threshold=0.3):
    """从 Observer 的 bbox CSV 生成 MOT 的 gt.txt 与 seqinfo.ini。"""
    mot_seq_dir = os.path.join(export_path, sequence_name)
    os.makedirs(mot_seq_dir, exist_ok=True)

    log_dir = os.path.join(observer_dir, "logs")
    if not os.path.exists(log_dir):
        print(f"Observer log dir not found: {log_dir}")
        return

    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Observer logs empty: {log_dir}")
        return 0, 0

    # 如果用户指定了 csv_file，优先使用（可指定绝对路径或相对于 log_dir 的路径）
    if csv_file:
        if os.path.isabs(csv_file) and os.path.exists(csv_file):
            csv_path = csv_file
        else:
            candidate = os.path.join(log_dir, csv_file)
            if os.path.exists(candidate):
                csv_path = candidate
            else:
                # 回退到 latest 并警告
                latest_csv = sorted(csv_files)[-1]
                csv_path = os.path.join(log_dir, latest_csv)
                print(f"指定的 CSV 文件不存在，回退到最新: {csv_path}")
    else:
        latest_csv = sorted(csv_files)[-1]
        csv_path = os.path.join(log_dir, latest_csv)

    if verbose:
        print(f"Using Observer CSV: {csv_path}")

    gt_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # 推断 vehicles
        if vehicles:
            vehicles_list = list(vehicles)
        else:
            vehicles_found = set()
            for fn in fieldnames:
                m = re.match(r'^(Drone\d+)_', fn)
                if m:
                    vehicles_found.add(m.group(1))
            if vehicles_found:
                vehicles_list = sorted(vehicles_found)
            else:
                vehicles_list = list(preferred_vehicles) if preferred_vehicles else list(PREFERRED_VEHICLES)

        frame_id = 1
        for row in reader:
            for i, vehicle in enumerate(vehicles_list, 1):
                x1_keys = [f'{vehicle}_bbox_2d_x1', f'{vehicle}_bbox_x_min', f'{vehicle}_bbox_x1', f'{vehicle}_bbox_left']
                y1_keys = [f'{vehicle}_bbox_2d_y1', f'{vehicle}_bbox_y_min', f'{vehicle}_bbox_y1', f'{vehicle}_bbox_top']
                x2_keys = [f'{vehicle}_bbox_2d_x2', f'{vehicle}_bbox_x_max', f'{vehicle}_bbox_x2', f'{vehicle}_bbox_right']
                y2_keys = [f'{vehicle}_bbox_2d_y2', f'{vehicle}_bbox_y_max', f'{vehicle}_bbox_y2', f'{vehicle}_bbox_bottom']
                vis_keys = [f'{vehicle}_visibility', f'{vehicle}_bbox_visibility', f'{vehicle}_bbox_vis']

                def find_first(keys):
                    for k in keys:
                        if k in row:
                            return row.get(k)
                    return None

                try:
                    x1v = find_first(x1_keys)
                    y1v = find_first(y1_keys)
                    x2v = find_first(x2_keys)
                    y2v = find_first(y2_keys)

                    if x1v is None or y1v is None or x2v is None or y2v is None:
                        continue

                    bb_left = float(x1v)
                    bb_top = float(y1v)
                    bb_right = float(x2v)
                    bb_bottom = float(y2v)

                    # 对于调试目的，如果所有bbox都是-1，我们生成一些基础统计信息
                    if bb_left < 0 or bb_top < 0 or bb_right < 0 or bb_bottom < 0:
                        # 记录不可见帧数但跳过这些bbox
                        if verbose:
                            print(f"Frame {frame_id}: {vehicle} not visible (bbox=-1)")
                        continue

                    bb_width = bb_right - bb_left
                    bb_height = bb_bottom - bb_top

                    if bb_width > 0 and bb_height > 0:
                        visv = find_first(vis_keys)
                        try:
                            conf = float(visv) if visv is not None else 1.0
                        except (ValueError, TypeError):
                            conf = 1.0

                        # ===== 改进：根据可见性阈值过滤 bbox =====
                        if conf < visibility_threshold:
                            if verbose:
                                print(f"Frame {frame_id}: {vehicle} visibility={conf:.3f} < {visibility_threshold}, filtered out")
                            continue

                        object_class = 1
                        visibility = conf if conf <= 1.0 else 1.0

                        gt_data.append(f"{frame_id},{i},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf},{object_class},{visibility}")
                except (ValueError, TypeError):
                    continue
            frame_id += 1

    gt_file = os.path.join(mot_seq_dir, "gt.txt")
    with open(gt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gt_data))

    if preview and gt_data:
        print(f"--- preview first {preview} gt entries ---")
        for line in gt_data[:preview]:
            print(line)
        print("--- end preview ---")

    seqinfo = f"""[Sequence]
name={sequence_name}
imDir=img1
frameRate={observer_frame_rate}
seqLength={frame_id-1}
imWidth=1280
imHeight=720
imExt=.jpg
"""
    seqinfo_file = os.path.join(mot_seq_dir, 'seqinfo.ini')
    with open(seqinfo_file, 'w', encoding='utf-8') as f:
        f.write(seqinfo)

    print(f"MOT序列导出: {sequence_name} ({frame_id-1} 帧)")
    # 返回已写入的GT条目数与序列长度
    return len(gt_data), (frame_id - 1)


def _export_vehicle_to_mot(vehicle_dir, export_path, vehicle_name, cap_fps=20):
    mot_seq_dir = os.path.join(export_path, f"{vehicle_name}_ego")
    os.makedirs(mot_seq_dir, exist_ok=True)

    seqinfo = f"""[Sequence]
name={vehicle_name}_ego
imDir=img1
frameRate={cap_fps}
seqLength=1000
imWidth=640
imHeight=480
imExt=.jpg
"""
    seqinfo_file = os.path.join(mot_seq_dir, 'seqinfo.ini')
    with open(seqinfo_file, 'w', encoding='utf-8') as f:
        f.write(seqinfo)
    print(f"MOT序列导出: {vehicle_name}_ego (seqinfo written)")


def export_to_coco_format(base_output_dir=r"D:\Documents\Unreal Projects\uav\output", export_dir="coco_export",
                          enable_observer_camera=True, observer_vehicle_name=DEFAULT_OBSERVER_NAME):
    """简化的 COCO 导出：仅导出 Observer 的 bbox 到 annotations.json（不打包 images）。"""
    print("开始导出COCO格式数据...")
    export_path = os.path.join(base_output_dir, export_dir)
    os.makedirs(export_path, exist_ok=True)

    if not enable_observer_camera:
        print("Observer camera disabled; skipping COCO export.")
        return

    observer_dir = os.path.join(base_output_dir, observer_vehicle_name)
    if not os.path.exists(observer_dir):
        print(f"Observer dir not found: {observer_dir}")
        return

    log_dir = os.path.join(observer_dir, 'logs')
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')] if os.path.exists(log_dir) else []
    if not csv_files:
        print('No observer csv files for COCO export')
        return

    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(log_dir, latest_csv)
    print(f"Using Observer CSV for COCO: {csv_path}")

    coco = {
        'info': {'description': 'UAV Swarm Tracking Dataset', 'year': datetime.datetime.now().year},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'drone'}]
    }

    img_id = 1
    ann_id = 1
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 生成基于 csv 的 annotation（如果存在 bbox 列）
            for v in PREFERRED_VEHICLES:
                x1 = row.get(f'{v}_bbox_x_min') or row.get(f'{v}_bbox_2d_x1')
                y1 = row.get(f'{v}_bbox_y_min') or row.get(f'{v}_bbox_2d_y1')
                x2 = row.get(f'{v}_bbox_x_max') or row.get(f'{v}_bbox_2d_x2')
                y2 = row.get(f'{v}_bbox_y_max') or row.get(f'{v}_bbox_2d_y2')
                if x1 and y1 and x2 and y2:
                    try:
                        x1f = float(x1); y1f = float(y1); x2f = float(x2); y2f = float(y2)
                        if x1f < 0 or y1f < 0 or x2f < 0 or y2f < 0:
                            continue
                        w = x2f - x1f; h = y2f - y1f
                        if w <= 0 or h <= 0:
                            continue
                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': 1,
                            'bbox': [x1f, y1f, w, h],
                            'area': w * h,
                            'iscrowd': 0
                        })
                        ann_id += 1
                    except ValueError:
                        continue
            img_id += 1

    coco_file = os.path.join(export_path, 'annotations.json')
    with open(coco_file, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"COCO导出完成: {coco_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export tools for AirSim capture data')
    parser.add_argument('--base-output', default=r'D:\Documents\Unreal Projects\uav\output', help='Base output directory produced by capture scripts')
    parser.add_argument('--export-mot', action='store_true', help='Export MOT format')
    parser.add_argument('--export-coco', action='store_true', help='Export COCO format')
    parser.add_argument('--observer-name', default=DEFAULT_OBSERVER_NAME, help='Observer folder name')
    parser.add_argument('--vehicles', nargs='*', help='List of vehicle names to export (overrides header inference)')
    parser.add_argument('--csv-file', default=None, help='Observer bbox CSV file to use (filename or absolute path). If omitted, the latest CSV is used.')
    parser.add_argument('--preview', type=int, default=0, help='Print first N GT entries as preview')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--visibility-threshold', type=float, default=0.3, help='Visibility threshold for filtering bbox (0.0-1.0, default=0.3). Lower values = include more distant/partial targets.')
    args = parser.parse_args()

    if args.export_mot:
        export_to_mot_format(base_output_dir=args.base_output, vehicles=args.vehicles,
                             observer_vehicle_name=args.observer_name, csv_file=args.csv_file, preview=args.preview, verbose=args.verbose,
                             visibility_threshold=args.visibility_threshold)
    if args.export_coco:
        export_to_coco_format(base_output_dir=args.base_output, observer_vehicle_name=args.observer_name)
