"""
完整流程脚本：ECM 参数辨识与不确定性分析
功能：从数据加载到输出结果的完整自动化流程
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# 兼容两种运行方式
try:
    from src.ecm.loader import load_discharge_cc_segment
    from src.ecm.ocv import fit_ocv_curve, calculate_soc
    from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage
    from src.identification.fit import fit_ecm_params, plot_fit_results
    from src.analysis.ci import analyze_parameter_uncertainty
    from src.analysis.bootstrap import residual_bootstrap, plot_bootstrap_results
    from src.analysis.sensitivity import local_sensitivity_analysis, plot_sensitivity_results
except ModuleNotFoundError:
    # 如果作为脚本直接运行，添加项目根目录到路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ecm.loader import load_discharge_cc_segment
    from src.ecm.ocv import fit_ocv_curve, calculate_soc
    from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage
    from src.identification.fit import fit_ecm_params, plot_fit_results
    from src.analysis.ci import analyze_parameter_uncertainty
    from src.analysis.bootstrap import residual_bootstrap, plot_bootstrap_results
    from src.analysis.sensitivity import local_sensitivity_analysis, plot_sensitivity_results


def create_output_directory(cycle_number: int, base_dir: str = "outputs") -> Path:
    """
    创建输出目录
    
    参数:
        cycle_number: 循环编号
        base_dir: 基础输出目录
    
    返回:
        output_dir: 输出目录路径
    """
    output_dir = Path(base_dir) / f"cycle_{cycle_number:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_parameters(params: ECM2RCParams, output_dir: Path):
    """
    保存参数到 JSON 文件
    
    参数:
        params: ECM 参数
        output_dir: 输出目录
    """
    params_dict = {
        'R0': float(params.R0),
        'R1': float(params.R1),
        'C1': float(params.C1),
        'R2': float(params.R2),
        'C2': float(params.C2),
        'tau1': float(params.R1 * params.C1),
        'tau2': float(params.R2 * params.C2)
    }
    
    output_file = output_dir / 'params.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=4, ensure_ascii=False)
    
    print(f"  参数已保存: {output_file}")


def save_metrics(metrics: dict, output_dir: Path):
    """
    保存拟合指标到 JSON 文件
    
    参数:
        metrics: 拟合指标字典
        output_dir: 输出目录
    """
    # 转换为可序列化的格式
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    
    output_file = output_dir / 'fit_metrics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=4, ensure_ascii=False)
    
    print(f"  拟合指标已保存: {output_file}")


def save_ci_table(ci_results: dict, output_dir: Path):
    """
    保存置信区间表格到 CSV 文件
    
    参数:
        ci_results: 置信区间分析结果
        output_dir: 输出目录
    """
    import csv
    
    ci_dict = ci_results['confidence_intervals']
    
    output_file = output_dir / 'ci_table.csv'
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['参数', '估计值', '标准差', '相对标准差(%)', 'CI下界', 'CI上界'])
        
        for i, name in enumerate(ci_dict['param_names']):
            writer.writerow([
                name,
                f"{ci_dict['estimates'][i]:.6e}",
                f"{ci_dict['std_errors'][i]:.6e}",
                f"{ci_dict['relative_std'][i]:.2f}",
                f"{ci_dict['ci_lower'][i]:.6e}",
                f"{ci_dict['ci_upper'][i]:.6e}"
            ])
    
    print(f"  置信区间表已保存: {output_file}")


def run_pipeline(
    mat_path: str,
    cycle_number: int = 1,
    output_base_dir: str = "outputs",
    current_threshold: float = 0.05,
    min_duration: float = 60.0,
    n_bootstrap: int = 50,
    verbose: bool = True
):
    """
    运行完整的 ECM 参数辨识与不确定性分析流程
    
    参数:
        mat_path: B0005.mat 文件路径
        cycle_number: 放电循环编号
        output_base_dir: 输出基础目录
        current_threshold: 恒流段电流标准差阈值
        min_duration: 恒流段最小持续时间
        n_bootstrap: Bootstrap 重采样次数
        verbose: 是否显示详细输出
    """
    
    if verbose:
        print("\n" + "="*70)
        print("ECM 参数辨识与不确定性分析流程")
        print("="*70)
        print(f"\n配置:")
        print(f"  数据文件: {mat_path}")
        print(f"  放电循环: 第 {cycle_number} 次")
        print(f"  输出目录: {output_base_dir}/cycle_{cycle_number:03d}")
        print(f"  Bootstrap 次数: {n_bootstrap}")
    
    # 创建输出目录
    output_dir = create_output_directory(cycle_number, output_base_dir)
    
    # ================================================================
    # 步骤 1: 加载数据
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print("[步骤 1/6] 加载数据...")
        print("-"*70)
    
    t, i, v_measured, info = load_discharge_cc_segment(
        mat_path, n=cycle_number,
        current_threshold=current_threshold,
        min_duration=min_duration
    )
    
    if verbose:
        print(f"  恒流段持续时间: {info['duration']:.2f} s")
        print(f"  平均电流: {info['mean_current']:.3f} A")
        print(f"  数据点数: {len(t)}")
    
    # ================================================================
    # 步骤 2: 计算 SOC 和拟合 OCV
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print("[步骤 2/6] 计算 SOC 和拟合 OCV...")
        print("-"*70)
    
    # 计算 SOC
    capacity_ah = info['total_capacity_ah']
    dt = np.gradient(t)
    charge_ah = np.cumsum(-i * dt) / 3600
    soc = 1.0 - charge_ah / capacity_ah
    soc = np.clip(soc, 0.0, 1.0)
    
    # 使用采样方法创建 OCV 函数
    soc_samples = np.linspace(soc.min(), soc.max(), 15)
    v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
    ocv_func = fit_ocv_curve(soc_samples, v_samples, method='linear')
    
    if verbose:
        print(f"  SOC 范围: [{soc.min():.4f}, {soc.max():.4f}]")
        print(f"  OCV 函数已创建（使用 {len(soc_samples)} 个采样点）")
    
    # ================================================================
    # 步骤 3: 参数辨识
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print("[步骤 3/6] 参数辨识...")
        print("-"*70)
    
    # 设置初始猜测和边界
    x0 = np.array([1e-4, 1e-4, 1e6, 1e-4, 1e6])
    bounds = (
        [1e-4, 1e-4, 1e1, 1e-4, 1e1],
        [1e0, 1e0, 1e6, 1e0, 1e6]
    )
    
    # 使用最小二乘法辨识
    params_fitted, fit_result = fit_ecm_params(
        t, i, v_measured, soc, ocv_func,
        method='least_squares',
        x0=x0,
        bounds=bounds,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"\n  辨识参数:")
        print(f"    {params_fitted}")
        print(f"\n  拟合指标:")
        print(f"    RMSE = {fit_result['metrics']['RMSE']:.6f} V")
        print(f"    MAE  = {fit_result['metrics']['MAE']:.6f} V")
        print(f"    R2   = {fit_result['metrics']['R2']:.6f}")
    
    # 保存参数和指标
    save_parameters(params_fitted, output_dir)
    save_metrics(fit_result['metrics'], output_dir)
    
    # 绘制拟合结果
    fig = plot_fit_results(
        t, v_measured, fit_result['v_pred'], fit_result['residuals'],
        params_fitted, fit_result['metrics'], i
    )
    fig.savefig(output_dir / 'fit_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  拟合曲线已保存: {output_dir / 'fit_curve.png'}")
    
    # 单独保存残差图
    fig_residual, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, fit_result['residuals'], 'b-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual (V)')
    ax.set_title(f'Residual Analysis (RMSE={fit_result["metrics"]["RMSE"]:.6f} V)')
    ax.grid(True, alpha=0.3)
    fig_residual.savefig(output_dir / 'residual.png', dpi=150, bbox_inches='tight')
    plt.close(fig_residual)
    
    if verbose:
        print(f"  残差图已保存: {output_dir / 'residual.png'}")
    
    # ================================================================
    # 步骤 4: 置信区间分析
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print("[步骤 4/6] 置信区间分析...")
        print("-"*70)
    
    # 创建残差函数（simulate_voltage 已经在文件顶部导入）
    def residual_func(theta):
        params_test = ECM2RCParams.from_array(theta)
        v_pred_test = simulate_voltage(t, i, soc, params_test, ocv_func)
        return v_pred_test - v_measured
    
    ci_results = analyze_parameter_uncertainty(
        residual_func=residual_func,
        params=params_fitted,
        residuals=fit_result['residuals'],
        confidence_level=0.95,
        use_stored_jacobian=None
    )
    
    # 保存置信区间表
    save_ci_table(ci_results, output_dir)
    
    # ================================================================
    # 步骤 5: Bootstrap 分析
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print(f"[步骤 5/6] Bootstrap 分析（{n_bootstrap} 次重采样）...")
        print("-"*70)
    
    bootstrap_results = residual_bootstrap(
        t, i, v_measured, soc, ocv_func,
        params_fitted=params_fitted,
        v_pred=fit_result['v_pred'],
        residuals=fit_result['residuals'],
        fit_function=fit_ecm_params,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        seed=42,
        verbose=verbose
    )
    
    # 绘制 Bootstrap 结果
    fig_bootstrap = plot_bootstrap_results(bootstrap_results, ci_results)
    fig_bootstrap.savefig(output_dir / 'bootstrap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bootstrap)
    
    if verbose:
        print(f"  Bootstrap 分析图已保存: {output_dir / 'bootstrap_analysis.png'}")
    
    # ================================================================
    # 步骤 6: 敏感性分析
    # ================================================================
    if verbose:
        print("\n" + "-"*70)
        print("[步骤 6/6] 敏感性分析...")
        print("-"*70)
    
    sensitivity_results = local_sensitivity_analysis(
        t, i, soc, ocv_func,
        params=params_fitted,
        perturbation=0.01,
        v_baseline=fit_result['v_pred']
    )
    
    # 绘制敏感性分析结果
    fig_sensitivity = plot_sensitivity_results(sensitivity_results)
    fig_sensitivity.savefig(output_dir / 'sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close(fig_sensitivity)
    
    if verbose:
        print(f"  敏感性分析图已保存: {output_dir / 'sensitivity.png'}")
    
    # ================================================================
    # 生成总结报告
    # ================================================================
    if verbose:
        print("\n" + "="*70)
        print("流程完成！")
        print("="*70)
        print(f"\n输出文件:")
        print(f"  - {output_dir / 'params.json'}")
        print(f"  - {output_dir / 'fit_metrics.json'}")
        print(f"  - {output_dir / 'fit_curve.png'}")
        print(f"  - {output_dir / 'residual.png'}")
        print(f"  - {output_dir / 'ci_table.csv'}")
        print(f"  - {output_dir / 'bootstrap_analysis.png'}")
        print(f"  - {output_dir / 'sensitivity.png'}")
        print("\n" + "="*70 + "\n")
    
    # 返回所有结果
    return {
        'params': params_fitted,
        'metrics': fit_result['metrics'],
        'ci_results': ci_results,
        'bootstrap_results': bootstrap_results,
        'sensitivity_results': sensitivity_results,
        'output_dir': output_dir
    }


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description='ECM 参数辨识与不确定性分析流程'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='/data/B0005.mat',
        help='B0005.mat 文件路径'
    )
    
    parser.add_argument(
        '--cycle', '-c',
        type=int,
        default=1,
        help='放电循环编号（默认: 1）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='输出基础目录（默认: outputs）'
    )
    
    parser.add_argument(
        '--bootstrap', '-b',
        type=int,
        default=50,
        help='Bootstrap 重采样次数（默认: 50）'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式（不显示详细输出）'
    )
    
    args = parser.parse_args()
    
    # 运行流程
    results = run_pipeline(
        mat_path=args.data,
        cycle_number=args.cycle,
        output_base_dir=args.output,
        n_bootstrap=args.bootstrap,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()
