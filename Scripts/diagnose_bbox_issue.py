"""
诊断脚本：分析为什么前 10 秒 bbox 为 -1
"""
import csv
import json

def analyze_observer_bbox_csv(csv_path):
    """分析 Observer bbox CSV，找出 -1 的原因"""
    print(f"\n=== 分析文件: {csv_path} ===\n")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        total_frames = 0
        invalid_bbox_frames = 0
        phase_transition_frame = None
        
        # 按阶段统计
        ground_phase_invalid = 0
        ground_phase_valid = 0
        air_phase_invalid = 0
        air_phase_valid = 0
        
        for row in reader:
            total_frames += 1
            frame_idx = int(row['frame_idx'])
            
            # 假设 30 fps，10 秒 = 300 帧
            is_ground_phase = frame_idx < 300
            
            # 检查所有无人机的 bbox
            has_invalid = False
            drone_status = {}
            
            for drone_name in ['Drone1', 'Drone2', 'Drone3', 'Drone4']:
                x_min_key = f"{drone_name}_bbox_x_min"
                if x_min_key in row:
                    x_min = int(row[x_min_key])
                    visibility = float(row[f"{drone_name}_visibility"])
                    distance = float(row[f"{drone_name}_distance"])
                    
                    is_invalid = (x_min == -1)
                    drone_status[drone_name] = {
                        'invalid': is_invalid,
                        'visibility': visibility,
                        'distance': distance
                    }
                    
                    if is_invalid:
                        has_invalid = True
            
            if has_invalid:
                invalid_bbox_frames += 1
                if is_ground_phase:
                    ground_phase_invalid += 1
                else:
                    air_phase_invalid += 1
                
                # 打印前几帧的详细信息
                if frame_idx < 20 or (295 < frame_idx < 305):
                    print(f"帧 {frame_idx} ({'地面阶段' if is_ground_phase else '空中阶段'}):")
                    for drone, status in drone_status.items():
                        if status['invalid']:
                            print(f"  {drone}: bbox=-1, vis={status['visibility']:.3f}, dist={status['distance']:.2f}m")
            else:
                if is_ground_phase:
                    ground_phase_valid += 1
                    if phase_transition_frame is None:
                        print(f"\n✓ 首次有效 bbox 出现在帧 {frame_idx} (地面阶段)")
                else:
                    air_phase_valid += 1
                    if phase_transition_frame is None and frame_idx >= 300:
                        phase_transition_frame = frame_idx
                        print(f"\n✓ 空中阶段开始，帧 {frame_idx}")
    
    print(f"\n=== 统计结果 ===")
    print(f"总帧数: {total_frames}")
    print(f"包含无效 bbox 的帧数: {invalid_bbox_frames} ({invalid_bbox_frames/total_frames*100:.1f}%)")
    print(f"\n地面阶段 (0-300 帧, 0-10秒):")
    print(f"  有效帧: {ground_phase_valid}")
    print(f"  无效帧: {ground_phase_invalid} ({ground_phase_invalid/(ground_phase_valid+ground_phase_invalid)*100:.1f}%)")
    print(f"\n空中阶段 (300+ 帧, 10秒+):")
    print(f"  有效帧: {air_phase_valid}")
    print(f"  无效帧: {air_phase_invalid} ({air_phase_invalid/(air_phase_valid+air_phase_invalid)*100:.1f}% if air_phase_valid+air_phase_invalid > 0 else 0)")

if __name__ == "__main__":
    import os
    import glob
    
    # 查找最新的 Observer bbox CSV
    bbox_csv_pattern = "../output/Observer/logs/Observer_bbox_*.csv"
    csv_files = glob.glob(bbox_csv_pattern)
    
    if not csv_files:
        print(f"错误: 未找到文件 {bbox_csv_pattern}")
        print("请先运行 multi_drone_capture.py 生成数据")
    else:
        latest_csv = max(csv_files, key=os.path.getctime)
        analyze_observer_bbox_csv(latest_csv)
        
        print("\n=== 可能的原因分析 ===")
        print("1. 相机姿态问题：地面阶段相机可能朝向错误，无人机在相机后方")
        print("2. 投影尺寸问题：近距离但角度不佳导致投影尺寸 < 0.5px")
        print("3. FOV 问题：相机视场角可能不正确")
        print("4. 相机初始化延迟：前几帧相机姿态还未稳定")
        print("\n建议：")
        print("- 检查 update_observer_camera_stable() 中的相机定位逻辑")
        print("- 验证地面阶段的 pitch 角度是否正确（应该能看到起飞点）")
        print("- 增加调试日志输出相机位置和无人机位置")
