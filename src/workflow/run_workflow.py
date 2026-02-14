"""
三步骤工作流执行脚本
Step1: DataReadOp - 数据读取
Step2: IdentifyOp - 参数辨识
Step3: UncertaintyOp - 不确定性分析
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 兼容两种运行方式
try:
    from src.workflow.ops import DataReadOp, IdentifyOp, UncertaintyOp
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.workflow.ops import DataReadOp, IdentifyOp, UncertaintyOp


def run_workflow(
    mat_path: str = "/data/B0005.mat",
    cycle_n: int = 1,
    output_dir: str = "outputs/workflow",
    current_threshold: float = 0.05,
    min_duration: float = 60.0,
    n_bootstrap: int = 50
):
    """
    运行完整的三步骤工作流
    
    参数:
        mat_path: B0005.mat 文件路径
        cycle_n: 循环编号
        output_dir: 输出目录
        current_threshold: 恒流段电流阈值
        min_duration: 最小持续时间
        n_bootstrap: Bootstrap 重采样次数
    """
    
    print("\n" + "="*70)
    print("ECM Parameter Identification Workflow")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Data file: {mat_path}")
    print(f"  Cycle: {cycle_n}")
    print(f"  Output directory: {output_dir}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print("="*70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Step 1: 数据读取
    # ================================================================
    segment_csv = DataReadOp.execute(
        mat_path=mat_path,
        cycle_n=cycle_n,
        current_threshold=current_threshold,
        min_duration=min_duration,
        output_path=str(output_path / "segment.csv")
    )
    
    # ================================================================
    # Step 2: 参数辨识
    # ================================================================
    identify_outputs = IdentifyOp.execute(
        segment_csv=segment_csv,
        output_dir=str(output_path)
    )
    
    # ================================================================
    # Step 3: 不确定性分析
    # ================================================================
    uncertainty_outputs = UncertaintyOp.execute(
        segment_csv=segment_csv,
        params_json=identify_outputs['params_json'],
        output_dir=str(output_path),
        n_bootstrap=n_bootstrap
    )
    
    # ================================================================
    # 工作流完成
    # ================================================================
    print("\n" + "="*70)
    print("Workflow Completed Successfully!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll output files saved to: {output_path}")
    print("\nGenerated files:")
    print(f"  Step 1 - Data:")
    print(f"    - segment.csv")
    print(f"  Step 2 - Identification:")
    print(f"    - params.json")
    print(f"    - fit_metrics.json")
    print(f"    - fit_curve.png")
    print(f"  Step 3 - Uncertainty:")
    print(f"    - ci_table.csv")
    print(f"    - sensitivity.png")
    print(f"    - bootstrap_params.csv")
    print(f"    - bootstrap_analysis.png")
    print("="*70 + "\n")
    
    return {
        'segment_csv': segment_csv,
        **identify_outputs,
        **uncertainty_outputs
    }


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description='Run ECM parameter identification workflow (3 steps)'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='/data/B0005.mat',
        help='Path to B0005.mat file (default: /data/B0005.mat)'
    )
    
    parser.add_argument(
        '--cycle', '-c',
        type=int,
        default=1,
        help='Discharge cycle number (default: 1)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/workflow',
        help='Output directory (default: outputs/workflow)'
    )
    
    parser.add_argument(
        '--bootstrap', '-b',
        type=int,
        default=50,
        help='Number of bootstrap iterations (default: 50)'
    )
    
    parser.add_argument(
        '--current-threshold',
        type=float,
        default=0.05,
        help='Current threshold for CC segment (default: 0.05)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=float,
        default=60.0,
        help='Minimum duration for CC segment in seconds (default: 60.0)'
    )
    
    args = parser.parse_args()
    
    # 运行工作流
    results = run_workflow(
        mat_path=args.data,
        cycle_n=args.cycle,
        output_dir=args.output,
        current_threshold=args.current_threshold,
        min_duration=args.min_duration,
        n_bootstrap=args.bootstrap
    )
    
    return results


if __name__ == "__main__":
    main()
