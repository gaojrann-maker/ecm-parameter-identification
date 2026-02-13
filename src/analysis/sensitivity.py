"""
敏感性分析模块
功能：
1. 局部敏感性分析（参数扰动）
2. 敏感性指标计算
3. 敏感性可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecm.ecm2rc import ECM2RCParams, simulate_voltage


def local_sensitivity_analysis(
    t: np.ndarray,
    i: np.ndarray,
    soc: np.ndarray,
    ocv_func: Callable,
    params: ECM2RCParams,
    perturbation: float = 0.01,
    v_baseline: np.ndarray = None
) -> Dict:
    """
    局部敏感性分析
    
    方法：
    对每个参数进行微小扰动（如±1%），观察电压输出的变化
    
    敏感性定义：
        S_j(t) ≈ [V(θ_j') - V(θ)] / (θ_j' - θ_j)
    
    其中 θ_j' = θ_j * (1 + ε)
    
    参数:
        t: 时间序列
        i: 电流序列
        soc: SOC序列
        ocv_func: OCV函数
        params: ECM参数
        perturbation: 扰动比例（默认1%）
        v_baseline: 基准电压（如果None则计算）
    
    返回:
        results: 敏感性分析结果
    """
    print("\n" + "="*60)
    print(f"局部敏感性分析（扰动 ±{perturbation*100:.1f}%）")
    print("="*60)
    
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']
    theta = params.to_array()
    n_params = len(theta)
    n_samples = len(t)
    
    # 计算基准电压
    if v_baseline is None:
        v_baseline = simulate_voltage(t, i, soc, params, ocv_func)
    
    # 存储敏感性曲线
    sensitivity_curves = np.zeros((n_samples, n_params))
    
    # 存储扰动后的电压
    v_perturbed_positive = np.zeros((n_samples, n_params))
    v_perturbed_negative = np.zeros((n_samples, n_params))
    
    print("\n计算各参数敏感性...")
    
    for j in range(n_params):
        # 正向扰动
        theta_plus = theta.copy()
        theta_plus[j] *= (1 + perturbation)
        params_plus = ECM2RCParams.from_array(theta_plus)
        v_plus = simulate_voltage(t, i, soc, params_plus, ocv_func)
        v_perturbed_positive[:, j] = v_plus
        
        # 负向扰动
        theta_minus = theta.copy()
        theta_minus[j] *= (1 - perturbation)
        params_minus = ECM2RCParams.from_array(theta_minus)
        v_minus = simulate_voltage(t, i, soc, params_minus, ocv_func)
        v_perturbed_negative[:, j] = v_minus
        
        # 计算敏感性（使用中心差分）
        delta_theta = theta_plus[j] - theta_minus[j]
        delta_v = v_plus - v_minus
        sensitivity_curves[:, j] = delta_v / delta_theta
        
        print(f"  {param_names[j]}: 完成")
    
    # 计算敏感性指标
    # 1. RMS敏感性
    sensitivity_rms = np.sqrt(np.mean(sensitivity_curves**2, axis=0))
    
    # 2. 最大绝对敏感性
    sensitivity_max = np.max(np.abs(sensitivity_curves), axis=0)
    
    # 3. 平均绝对敏感性
    sensitivity_mean = np.mean(np.abs(sensitivity_curves), axis=0)
    
    # 4. 归一化敏感性（相对于参数值）
    sensitivity_normalized = sensitivity_rms * np.abs(theta)
    
    # 打印结果
    print("\n敏感性指标汇总:")
    print("-" * 90)
    print(f"{'参数':<6} {'参数值':>12} {'RMS敏感性':>15} {'最大敏感性':>15} "
          f"{'平均敏感性':>15} {'归一化敏感性':>15}")
    print("-" * 90)
    
    for j, name in enumerate(param_names):
        print(f"{name:<6} {theta[j]:>12.6e} {sensitivity_rms[j]:>15.6e} "
              f"{sensitivity_max[j]:>15.6e} {sensitivity_mean[j]:>15.6e} "
              f"{sensitivity_normalized[j]:>15.6e}")
    
    # 排序找出最敏感的参数
    sorted_indices = np.argsort(sensitivity_rms)[::-1]
    print("\n参数敏感性排序（从高到低）:")
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {param_names[idx]}: RMS敏感性 = {sensitivity_rms[idx]:.6e}")
    
    # 物理解释
    print("\n物理意义解释:")
    print("  R0: 欧姆内阻，主要影响瞬时压降（I*R0项）")
    print("  R1, C1: 快速极化过程，影响短时间动态响应（时间常数τ1 = R1*C1）")
    print("  R2, C2: 慢速极化过程，影响长时间动态响应（时间常数τ2 = R2*C2）")
    
    # 组织返回结果
    results = {
        'param_names': param_names,
        'param_values': theta,
        'sensitivity_curves': sensitivity_curves,
        'v_baseline': v_baseline,
        'v_perturbed_positive': v_perturbed_positive,
        'v_perturbed_negative': v_perturbed_negative,
        'sensitivity_rms': sensitivity_rms,
        'sensitivity_max': sensitivity_max,
        'sensitivity_mean': sensitivity_mean,
        'sensitivity_normalized': sensitivity_normalized,
        'perturbation': perturbation,
        'sensitivity_ranking': [param_names[i] for i in sorted_indices]
    }
    
    return results


def compute_voltage_impact(
    sensitivity_curves: np.ndarray,
    param_values: np.ndarray,
    param_changes: np.ndarray
) -> np.ndarray:
    """
    计算参数变化对电压的影响
    
    参数:
        sensitivity_curves: 敏感性曲线，形状 (n_samples, n_params)
        param_values: 参数值
        param_changes: 参数变化量（绝对值或百分比）
    
    返回:
        voltage_impact: 电压变化
    """
    # 线性近似：ΔV ≈ Σ (∂V/∂θ_j) * Δθ_j
    voltage_impact = sensitivity_curves @ param_changes
    return voltage_impact


def relative_sensitivity_analysis(
    sensitivity_curves: np.ndarray,
    param_values: np.ndarray,
    v_baseline: np.ndarray
) -> Dict:
    """
    相对敏感性分析
    
    相对敏感性 = (∂V/∂θ) * (θ/V)
    表示参数相对变化1%导致电压相对变化多少%
    
    参数:
        sensitivity_curves: 敏感性曲线
        param_values: 参数值
        v_baseline: 基准电压
    
    返回:
        results: 相对敏感性结果
    """
    n_samples, n_params = sensitivity_curves.shape
    
    # 计算相对敏感性
    relative_sensitivity = np.zeros_like(sensitivity_curves)
    
    for j in range(n_params):
        # 避免除零
        v_safe = np.where(np.abs(v_baseline) < 1e-10, 1e-10, v_baseline)
        relative_sensitivity[:, j] = (sensitivity_curves[:, j] * param_values[j]) / v_safe
    
    # 计算相对敏感性指标
    relative_rms = np.sqrt(np.mean(relative_sensitivity**2, axis=0))
    relative_max = np.max(np.abs(relative_sensitivity), axis=0)
    
    results = {
        'relative_sensitivity_curves': relative_sensitivity,
        'relative_rms': relative_rms,
        'relative_max': relative_max
    }
    
    return results


def sensitivity_summary(sensitivity_results: Dict):
    """
    打印敏感性分析摘要
    
    参数:
        sensitivity_results: 敏感性分析结果
    """
    print("\n" + "="*60)
    print("敏感性分析摘要")
    print("="*60)
    
    param_names = sensitivity_results['param_names']
    sensitivity_rms = sensitivity_results['sensitivity_rms']
    
    print(f"\n参数敏感性排序（RMS敏感性）:")
    ranking = sensitivity_results['sensitivity_ranking']
    for i, name in enumerate(ranking):
        idx = param_names.index(name)
        print(f"  {i+1}. {name}: {sensitivity_rms[idx]:.6e}")
    
    print(f"\n最敏感参数: {ranking[0]}")
    print(f"最不敏感参数: {ranking[-1]}")
    
    # 敏感性比值
    max_sens = sensitivity_rms[param_names.index(ranking[0])]
    min_sens = sensitivity_rms[param_names.index(ranking[-1])]
    ratio = max_sens / min_sens if min_sens > 0 else np.inf
    print(f"敏感性比值（最大/最小）: {ratio:.2f}")


def plot_sensitivity_results(sensitivity_results: Dict) -> plt.Figure:
    """
    Plot sensitivity analysis results
    
    Parameters:
        sensitivity_results: Sensitivity analysis results
    
    Returns:
        fig: matplotlib figure object
    """
    param_names = sensitivity_results['param_names']
    sensitivity_curves = sensitivity_results['sensitivity_curves']
    sensitivity_rms = sensitivity_results['sensitivity_rms']
    t = np.arange(len(sensitivity_curves))
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Subplot 1: Sensitivity curves
    ax1 = plt.subplot(2, 2, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, name in enumerate(param_names):
        ax1.plot(t, sensitivity_curves[:, i], label=name, 
                linewidth=1.5, color=colors[i])
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Sensitivity (V/param)')
    ax1.set_title('Parameter Sensitivity Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: RMS sensitivity bar chart
    ax2 = plt.subplot(2, 2, 2)
    bars = ax2.bar(param_names, sensitivity_rms, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMS Sensitivity')
    ax2.set_title('Parameter Sensitivity Ranking (RMS)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, sensitivity_rms):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Maximum sensitivity bar chart
    ax3 = plt.subplot(2, 2, 3)
    sensitivity_max = sensitivity_results['sensitivity_max']
    bars = ax3.bar(param_names, sensitivity_max, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Max Sensitivity')
    ax3.set_title('Parameter Maximum Sensitivity')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Normalized sensitivity
    ax4 = plt.subplot(2, 2, 4)
    sensitivity_normalized = sensitivity_results['sensitivity_normalized']
    bars = ax4.bar(param_names, sensitivity_normalized, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    ax4.set_ylabel('Normalized Sensitivity')
    ax4.set_title('Normalized Parameter Sensitivity')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
