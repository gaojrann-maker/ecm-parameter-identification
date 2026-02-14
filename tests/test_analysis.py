"""
测试不确定性分析模块
验证：
1. 置信区间分析（基于雅可比矩阵）
2. Bootstrap分析
3. 敏感性分析
4. 可视化所有结果
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 添加 src 到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ecm.loader import load_discharge_cc_segment
from ecm.ocv import calculate_soc, fit_ocv_curve
from identification.fit import fit_ecm_params
from analysis.ci import analyze_parameter_uncertainty
from analysis.bootstrap import residual_bootstrap, compare_ci_methods
from analysis.sensitivity import local_sensitivity_analysis, sensitivity_summary

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_fit_data():
    """加载数据并进行参数辨识"""
    print("="*60)
    print("加载数据并进行参数辨识")
    print("="*60)
    
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    
    # 加载数据
    t, i, v_measured, info = load_discharge_cc_segment(str(mat_path), n=1)
    capacity = info['total_capacity_ah']
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    
    # 拟合OCV
    sample_indices = np.linspace(0, len(t)-1, 15, dtype=int)
    ocv_func = fit_ocv_curve(soc[sample_indices], v_measured[sample_indices])
    
    # 参数辨识
    params, results = fit_ecm_params(
        t, i, v_measured, soc, ocv_func,
        method='least_squares',
        verbose=0
    )
    
    print(f"\n数据点数: {len(t)}")
    print(f"拟合 RMSE: {results['metrics']['RMSE']:.6f} V")
    
    return t, i, v_measured, soc, ocv_func, params, results


def test_confidence_interval_analysis(t, i, v_measured, soc, ocv_func, params, results):
    """测试置信区间分析"""
    print("\n" + "="*60)
    print("测试 1: 置信区间分析（基于雅可比矩阵）")
    print("="*60)
    
    # 创建残差函数
    def residual_func(theta):
        from ecm.ecm2rc import ECM2RCParams, simulate_voltage
        params_temp = ECM2RCParams.from_array(theta)
        v_pred = simulate_voltage(t, i, soc, params_temp, ocv_func)
        return v_pred - v_measured
    
    # 进行不确定性分析
    start_time = time.time()
    ci_results = analyze_parameter_uncertainty(
        residual_func,
        params,
        results['residuals'],
        confidence_level=0.95
    )
    elapsed = time.time() - start_time
    print(f"\n分析耗时: {elapsed:.2f} s")
    
    return ci_results


def test_bootstrap_analysis(t, i, v_measured, soc, ocv_func, params, results):
    """测试Bootstrap分析"""
    print("\n" + "="*60)
    print("测试 2: Bootstrap 分析")
    print("="*60)
    
    start_time = time.time()
    
    # Bootstrap分析（使用较少的迭代次数以加快速度）
    bootstrap_results = residual_bootstrap(
        t, i, v_measured, soc, ocv_func,
        params,
        results['v_pred'],
        results['residuals'],
        fit_ecm_params,
        n_bootstrap=50,  # 实际应用中可以用更多次，如100-1000
        seed=42,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n分析耗时: {elapsed:.2f} s")
    
    return bootstrap_results


def test_sensitivity_analysis(t, i, soc, ocv_func, params, results):
    """测试敏感性分析"""
    print("\n" + "="*60)
    print("测试 3: 敏感性分析")
    print("="*60)
    
    start_time = time.time()
    
    # 敏感性分析
    sensitivity_results = local_sensitivity_analysis(
        t, i, soc, ocv_func, params,
        perturbation=0.01,
        v_baseline=results['v_pred']
    )
    
    elapsed = time.time() - start_time
    print(f"\n分析耗时: {elapsed:.2f} s")
    
    # 打印摘要
    sensitivity_summary(sensitivity_results)
    
    return sensitivity_results


def visualize_confidence_intervals(ci_results):
    """可视化置信区间"""
    print("\n" + "="*60)
    print("生成置信区间可视化")
    print("="*60)
    
    ci_dict = ci_results['confidence_intervals']
    corr_matrix = ci_results['correlation_matrix']
    
    fig = plt.figure(figsize=(14, 5))
    
    # 子图1: 参数估计与置信区间
    ax1 = plt.subplot(1, 2, 1)
    param_names = ci_dict['param_names']
    x = np.arange(len(param_names))
    estimates = ci_dict['estimates']
    ci_lower = ci_dict['ci_lower']
    ci_upper = ci_dict['ci_upper']
    errors = np.array([estimates - ci_lower, ci_upper - estimates])
    
    ax1.errorbar(x, estimates, yerr=errors, fmt='o', markersize=8, 
                 capsize=5, capthick=2, linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names)
    ax1.set_ylabel('参数值', fontsize=12)
    ax1.set_title('参数估计与95%置信区间', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # 子图2: 相关系数矩阵热力图
    ax2 = plt.subplot(1, 2, 2)
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加文本标注
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax2.set_xticks(np.arange(len(param_names)))
    ax2.set_yticks(np.arange(len(param_names)))
    ax2.set_xticklabels(param_names)
    ax2.set_yticklabels(param_names)
    ax2.set_title('参数相关系数矩阵', fontsize=13)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('相关系数', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 保存
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "analysis_ci.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_bootstrap_results(bootstrap_results, ci_results):
    """可视化Bootstrap结果"""
    print("\n" + "="*60)
    print("生成Bootstrap结果可视化")
    print("="*60)
    
    param_names = bootstrap_results['param_names']
    bootstrap_samples = bootstrap_results['bootstrap_samples']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Bootstrap 参数分布', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        # 直方图
        ax.hist(bootstrap_samples[:, i], bins=30, color='skyblue', 
               edgecolor='black', alpha=0.7, density=True)
        
        # 原始估计
        original = bootstrap_results['original_estimates'][i]
        ax.axvline(original, color='red', linestyle='--', linewidth=2, 
                  label=f'原估计: {original:.2e}')
        
        # Bootstrap均值
        boot_mean = bootstrap_results['bootstrap_mean'][i]
        ax.axvline(boot_mean, color='green', linestyle='--', linewidth=2,
                  label=f'Boot均值: {boot_mean:.2e}')
        
        # 置信区间
        ci_lower = bootstrap_results['ci_lower'][i]
        ci_upper = bootstrap_results['ci_upper'][i]
        ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=1.5,
                  label=f'95% CI')
        ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel(f'{name}', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(f'{name} 分布', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏第6个子图（只有5个参数）
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "analysis_bootstrap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_sensitivity_results(t, sensitivity_results):
    """可视化敏感性分析结果"""
    print("\n" + "="*60)
    print("生成敏感性分析可视化")
    print("="*60)
    
    param_names = sensitivity_results['param_names']
    sensitivity_curves = sensitivity_results['sensitivity_curves']
    sensitivity_rms = sensitivity_results['sensitivity_rms']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('敏感性分析结果', fontsize=16, fontweight='bold')
    
    # 子图1: 敏感性曲线
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, name in enumerate(param_names):
        axes[0].plot(t, sensitivity_curves[:, i], color=colors[i], 
                    linewidth=2, label=name, alpha=0.7)
    
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('敏感性 dV/dθ (V/单位)', fontsize=12)
    axes[0].set_title('敏感性曲线（参数扰动对电压的影响）', fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 子图2: RMS敏感性柱状图
    x = np.arange(len(param_names))
    bars = axes[1].bar(x, sensitivity_rms, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, sensitivity_rms)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=10)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names)
    axes[1].set_ylabel('RMS 敏感性', fontsize=12)
    axes[1].set_title('参数敏感性汇总（RMS值）', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "analysis_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("开始测试不确定性分析模块")
    print("="*60)
    
    try:
        # 加载数据并拟合
        t, i, v_measured, soc, ocv_func, params, results = load_and_fit_data()
        
        # 测试1: 置信区间分析
        ci_results = test_confidence_interval_analysis(
            t, i, v_measured, soc, ocv_func, params, results
        )
        
        # 测试2: Bootstrap分析
        bootstrap_results = test_bootstrap_analysis(
            t, i, v_measured, soc, ocv_func, params, results
        )
        
        # 比较两种置信区间方法
        compare_ci_methods(ci_results['confidence_intervals'], bootstrap_results)
        
        # 测试3: 敏感性分析
        sensitivity_results = test_sensitivity_analysis(
            t, i, soc, ocv_func, params, results
        )
        
        # 可视化
        visualize_confidence_intervals(ci_results)
        visualize_bootstrap_results(bootstrap_results, ci_results)
        visualize_sensitivity_results(t, sensitivity_results)
        
        print("\n" + "="*60)
        print("[OK] 所有测试完成！")
        print("="*60)
        print("\n不确定性分析总结:")
        print(f"1. 置信区间: 基于雅可比矩阵，快速准确")
        print(f"2. Bootstrap: {bootstrap_results['n_bootstrap']}次重采样，更稳健")
        print(f"3. 敏感性: {sensitivity_results['sensitivity_ranking'][0]} 最敏感")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
