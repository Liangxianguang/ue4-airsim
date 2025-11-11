"""
数据验证工具 - 快速示例脚本

演示如何使用validate_data_quality.py进行数据检验和可视化
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "=" * 70)
    print(f"📌 {description}")
    print("=" * 70)
    print(f"执行: {cmd}")
    print("-" * 70)
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} 完成")
        else:
            print(f"❌ {description} 失败 (返回码: {result.returncode})")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行错误: {str(e)}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🚁 无人机数据质量检验 - 快速示例  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 检查dependencies
    print("\n📦 检查依赖...")
    required_packages = ['pandas', 'cv2', 'numpy']
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return
    
    # 示例任务
    print("\n\n📋 可用的验证任务:")
    print("""
    1️⃣  基础验证 - 检查所有数据的完整性和一致性
    2️⃣  生成报告 - 创建详细的统计报告
    3️⃣  可视化验证 - 生成边界框可视化视频
    4️⃣  采样帧验证 - 保存关键帧进行人工检查
    5️⃣  完整检验 - 执行所有验证任务
    6️⃣  自定义检验 - 针对特定vehicle的检验
    """)
    
    choice = input("\n请选择任务 (1-6) [默认: 1]: ").strip() or "1"
    
    output_dir = "output"
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 检查输出目录
    if not Path(output_dir).exists():
        print(f"\n❌ 输出目录不存在: {output_dir}")
        return
    
    quoted_output_dir = f'"{output_dir}"'

    # 统一构建 Python 与脚本的绝对路径，避免因工作目录不同导致找不到脚本
    python_exe = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    validator_py = scripts_dir / "validate_data_quality.py"
    quoted_python = f'"{python_exe}"'
    quoted_validator = f'"{str(validator_py)}"'

    # 结果跟踪，便于末尾只汇报成功产物
    success_report = False
    success_visual = False
    success_frames = False
    if choice == "1":
        # 基础验证
        success_report = run_command(
            f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir}",
            "基础数据验证"
        )
    elif choice == "2":
        # 生成报告
        success_report = run_command(
            f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir}",
            "生成验证报告"
        )
        print("\n📄 已生成以下报告:")
        print(f"  - {output_dir}/validation_report.json")
        print(f"  - {output_dir}/statistics_report.txt")
    elif choice == "3":
        # 可视化验证
        print("\n请选择处理帧数范围:")
        start = input("  开始帧数 [默认: 0]: ").strip() or "0"
        num = input("  处理帧数 [默认: 300]: ").strip() or "300"
        success_visual = run_command(
            f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir} "
            f"--visualize --start-frame {start} --num-frames {num}",
            f"生成边界框可视化视频 (帧 {start}-{int(start)+int(num)})"
        )
        print("\n📹 已生成可视化视频:")
        print(f"  - {output_dir}/Observer/bbox_visualization.mp4")
    elif choice == "4":
        # 采样帧验证
        success_frames = run_command(
            f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir} "
            f"--visualize --sample-frames --num-frames 0",
            "生成采样帧"
        )
        print("\n🖼️  已保存采样帧:")
        print(f"  - {output_dir}/Observer/bbox_frames/")
    elif choice == "5":
        # 完整检验
        print("\n⏳ 执行完整检验，这可能需要几分钟...")
        tasks = [
            ("基础验证", f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir}"),
            ("生成可视化", f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir} --visualize --num-frames 300"),
            ("保存采样帧", f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir} --visualize --sample-frames --num-frames 0"),
        ]
        results = []
        for desc, cmd in tasks:
            success = run_command(cmd, desc)
            results.append((desc, success))
            if desc == "基础验证":
                success_report = success
            elif desc == "生成可视化":
                success_visual = success
            elif desc == "保存采样帧":
                success_frames = success
        print("\n" + "=" * 70)
        print("📊 完整检验结果汇总:")
        print("=" * 70)
        for desc, success in results:
            status = "✅" if success else "❌"
            print(f"  {status} {desc}")
    elif choice == "6":
        # 自定义检验
        print("\n选择要检验的vehicle:")
        vehicles = ["Drone1", "Drone2", "Drone3", "Drone4", "Observer"]
        for i, v in enumerate(vehicles, 1):
            print(f"  {i}. {v}")
        vehicle_choice = input("\n请选择 (1-5): ").strip()
        try:
            vehicle = vehicles[int(vehicle_choice) - 1]
        except (ValueError, IndexError):
            vehicle = "Observer"
        success_visual = run_command(
            f"{quoted_python} {quoted_validator} --output_dir {quoted_output_dir} "
            f"--visualize --vehicle {vehicle} --num-frames 200",
            f"检验 {vehicle} 的数据"
        )
    else:
        print("❌ 无效选择")
        return
    
    # 生成汇总
    print("\n\n" + "=" * 70)
    print("📈 验证完成！")
    print("=" * 70)
    print("\n生成的文件:")
    if success_report:
        print(f"  📄 {output_dir}/validation_report.json - 详细的验证报告")
        print(f"  📄 {output_dir}/statistics_report.txt - 统计报告")
    else:
        print("  （报告未生成或本次未执行）")
    if success_visual:
        print(f"  📹 {output_dir}/Observer/bbox_visualization.mp4 - 可视化视频")
    if success_frames:
        print(f"  📋 {output_dir}/Observer/bbox_frames/ - 采样帧")
    
    print("\n\n💡 后续步骤:")
    print("""
    1. 查看 statistics_report.txt 了解数据质量
    2. 检查 validation_report.json 中的问题列表
    3. 观看 bbox_visualization.mp4 验证边界框准确性
    4. 根据发现的问题进行改进
    5. 如果数据质量满足要求，可进行目标检测训练
    """)
    
    print("=" * 70)
    print("✨ 数据验证工具使用完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
