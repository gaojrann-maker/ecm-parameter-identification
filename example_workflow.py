"""
示例：如何使用三步骤工作流
"""

from pathlib import Path
import sys

from src.workflow.ops import DataReadOp, IdentifyOp, UncertaintyOp


def example_full_workflow():
    """
    示例1：运行完整的三步骤工作流
    """
    print("\n" + "="*70)
    print("Example 1: Full Workflow")
    print("="*70)
    
    output_dir = "outputs/example_full"
    
    # Step 1: 数据读取
    segment_csv = DataReadOp.execute(
        mat_path="/data/B0005.mat",
        cycle_n=1,
        output_path=f"{output_dir}/segment.csv"
    )
    
    # Step 2: 参数辨识
    identify_outputs = IdentifyOp.execute(
        segment_csv=segment_csv,
        output_dir=output_dir
    )
    
    # Step 3: 不确定性分析
    uncertainty_outputs = UncertaintyOp.execute(
        segment_csv=segment_csv,
        params_json=identify_outputs['params_json'],
        output_dir=output_dir,
        n_bootstrap=30
    )
    
    print(f"\n✓ All outputs saved to: {output_dir}")


def example_step_by_step():
    """
    示例2：逐步运行工作流（可以在步骤之间检查结果）
    """
    print("\n" + "="*70)
    print("Example 2: Step-by-Step Workflow")
    print("="*70)
    
    output_dir = "outputs/example_steps"
    
    # Step 1: 只运行数据读取
    print("\n>>> Running Step 1 only...")
    segment_csv = DataReadOp.execute(
        mat_path="/data/B0005.mat",
        cycle_n=1,
        output_path=f"{output_dir}/segment.csv"
    )
    
    print(f"\n✓ Step 1 complete. Data saved to: {segment_csv}")
    print("  You can now inspect segment.csv before proceeding.")
    
    # 用户可以在这里检查 segment.csv
    # input("Press Enter to continue to Step 2...")
    
    # Step 2: 只运行参数辨识
    print("\n>>> Running Step 2 only...")
    identify_outputs = IdentifyOp.execute(
        segment_csv=segment_csv,
        output_dir=output_dir
    )
    
    print(f"\n✓ Step 2 complete.")
    print(f"  Parameters: {identify_outputs['params_json']}")
    print(f"  Metrics: {identify_outputs['fit_metrics_json']}")
    
    # 用户可以在这里检查参数和拟合结果
    # input("Press Enter to continue to Step 3...")
    
    # Step 3: 只运行不确定性分析
    print("\n>>> Running Step 3 only...")
    uncertainty_outputs = UncertaintyOp.execute(
        segment_csv=segment_csv,
        params_json=identify_outputs['params_json'],
        output_dir=output_dir,
        n_bootstrap=30
    )
    
    print(f"\n✓ Step 3 complete.")
    print(f"  CI table: {uncertainty_outputs['ci_table_csv']}")
    print(f"  Sensitivity: {uncertainty_outputs['sensitivity_png']}")
    print(f"  Bootstrap: {uncertainty_outputs['bootstrap_params_csv']}")


def example_using_run_workflow():
    """
    示例3：使用便捷函数运行完整工作流
    """
    print("\n" + "="*70)
    print("Example 3: Using run_workflow() Function")
    print("="*70)
    
    from src.workflow import run_workflow
    
    results = run_workflow(
        mat_path="/data/B0005.mat",
        cycle_n=1,
        output_dir="outputs/example_convenient",
        n_bootstrap=30
    )
    
    print("\n✓ Workflow complete!")
    print(f"  Generated {len(results)} output files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Workflow Examples')
    parser.add_argument(
        '--example', '-e',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Which example to run (1: full, 2: step-by-step, 3: convenient)'
    )
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_full_workflow()
    elif args.example == 2:
        example_step_by_step()
    elif args.example == 3:
        example_using_run_workflow()
