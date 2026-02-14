"""
完整流程测试：验证从数据加载到不确定性分析的全流程
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ecm.loader import load_discharge_cc_segment
from ecm.ocv import fit_ocv_from_full_cycle
from ecm.ecm2rc import ECM2RCParams
from identification.fit import fit_ecm_params
from analysis.ci import analyze_parameter_uncertainty
from analysis.bootstrap import residual_bootstrap
from analysis.sensitivity import local_sensitivity_analysis


def main():
    print("\n" + "="*70)
    print("完整流程测试：ECM 参数辨识 + 不确定性分析")
    print("="*70)
    
    # ================================================================
    # 步骤 1: 加载数据
    # ================================================================
    print("\n[步骤 1/6] 加载数据...")
    
    mat_path = "data/B0005.mat"
    
    # 加载恒流段
    t, i, v_measured, info = load_discharge_cc_segment(
        mat_path, n=1,
        current_threshold=0.05,
        min_duration=60.0
    )
    
    print(f"  恒流段持续时间: {info['duration']:.2f} s")
    print(f"  平均电流: {info['mean_current']:.3f} A")
    print(f"  数据点数: {len(t)}")
    
    # ================================================================
    # 步骤 2: 拟合 OCV-SOC 曲线
    # ================================================================
    print("\n[步骤 2/6] 拟合 OCV-SOC 曲线...")
    
    # 从恒流段计算 SOC
    capacity_ah = info['total_capacity_ah']
    # 修正 SOC 计算：从高到低，使用负电流
    dt = np.gradient(t)
    charge_ah = np.cumsum(-i * dt) / 3600  # 放电时i为正，所以取负
    soc = 1.0 - charge_ah / capacity_ah
    # 确保 SOC 在 [0, 1] 范围内
    soc = np.clip(soc, 0.0, 1.0)
    
    # 使用充电循环拟合 OCV（因为放电循环静置段较少）
    # 注意：由于没有充电循环函数，这里使用采样方法
    
    # 创建采样 OCV 点（简化方法）
    soc_samples = np.linspace(soc.min(), soc.max(), 15)
    v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
    
    from ecm.ocv import fit_ocv_curve
    
    ocv_func = fit_ocv_curve(
        soc_samples, v_samples,
        method='linear'
    )
    
    print(f"  使用采样方法创建 OCV 函数")
    print(f"  SOC 范围: [{soc.min():.4f}, {soc.max():.4f}]")
    
    # ================================================================
    # 步骤 3: 参数辨识
    # ================================================================
    print("\n[步骤 3/6] 参数辨识...")
    
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
        verbose=0
    )
    
    print(f"\n  辨识参数:")
    print(f"    {params_fitted}")
    print(f"\n  拟合指标:")
    print(f"    RMSE = {fit_result['metrics']['RMSE']:.6f} V")
    print(f"    MAE  = {fit_result['metrics']['MAE']:.6f} V")
    print(f"    R2   = {fit_result['metrics']['R2']:.6f}")
    
    # ================================================================
    # 步骤 4: 置信区间分析
    # ================================================================
    print("\n[步骤 4/6] 置信区间分析...")
    
    # 创建残差函数
    from ecm.ecm2rc import simulate_voltage
    
    def residual_func(theta):
        params_test = ECM2RCParams.from_array(theta)
        v_pred_test = simulate_voltage(t, i, soc, params_test, ocv_func)
        return v_pred_test - v_measured
    
    ci_results = analyze_parameter_uncertainty(
        residual_func=residual_func,
        params=params_fitted,
        residuals=fit_result['residuals'],
        confidence_level=0.95,
        use_stored_jacobian=None  # 重新计算
    )
    
    # ================================================================
    # 步骤 5: Bootstrap 分析
    # ================================================================
    print("\n[步骤 5/6] Bootstrap 分析...")
    
    bootstrap_results = residual_bootstrap(
        t, i, v_measured, soc, ocv_func,
        params_fitted=params_fitted,
        v_pred=fit_result['v_pred'],
        residuals=fit_result['residuals'],
        fit_function=fit_ecm_params,
        n_bootstrap=30,  # 减少次数以加快测试
        confidence_level=0.95,
        seed=42,
        verbose=False
    )
    
    print(f"\n  Bootstrap 完成: {bootstrap_results['n_bootstrap']} 次")
    
    # ================================================================
    # 步骤 6: 敏感性分析
    # ================================================================
    print("\n[步骤 6/6] 敏感性分析...")
    
    sensitivity_results = local_sensitivity_analysis(
        t, i, soc, ocv_func,
        params=params_fitted,
        perturbation=0.01,
        v_baseline=fit_result['v_pred']
    )
    
    print(f"\n  敏感性排序:")
    for i, param_name in enumerate(sensitivity_results['sensitivity_ranking'], 1):
        idx = sensitivity_results['param_names'].index(param_name)
        sens_val = sensitivity_results['sensitivity_rms'][idx]
        print(f"    {i}. {param_name}: {sens_val:.6e}")
    
    # ================================================================
    # 生成综合报告图
    # ================================================================
    print("\n[生成综合报告图]...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 拟合结果
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, v_measured, 'k-', label='实测', linewidth=1.5)
    ax1.plot(t, fit_result['v_pred'], 'r--', label='预测', linewidth=1.5)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('电压 (V)')
    ax1.set_title('拟合结果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, fit_result['residuals'], 'b-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('残差 (V)')
    ax2.set_title(f'残差 (RMSE={fit_result["metrics"]["RMSE"]:.6f} V)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 参数置信区间
    ax3 = plt.subplot(3, 3, 3)
    ci_dict = ci_results['confidence_intervals']
    param_names = ci_dict['param_names']
    estimates = ci_dict['estimates']
    ci_lower = ci_dict['ci_lower']
    ci_upper = ci_dict['ci_upper']
    
    y_pos = np.arange(len(param_names))
    errors = np.array([estimates - ci_lower, ci_upper - estimates])
    
    ax3.errorbar(estimates, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(param_names)
    ax3.set_xlabel('参数值')
    ax3.set_title('95% 置信区间')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 相关性矩阵
    ax4 = plt.subplot(3, 3, 4)
    corr_matrix = ci_results['correlation_matrix']
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(param_names)))
    ax4.set_yticks(range(len(param_names)))
    ax4.set_xticklabels(param_names)
    ax4.set_yticklabels(param_names)
    ax4.set_title('参数相关系数矩阵')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. Bootstrap 参数分布 (R0)
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(bootstrap_results['bootstrap_samples'][:, 0], bins=20, alpha=0.7, edgecolor='black')
    ax5.axvline(bootstrap_results['bootstrap_mean'][0], color='r', linestyle='--', linewidth=2, label='均值')
    ax5.axvline(bootstrap_results['ci_lower'][0], color='g', linestyle='--', linewidth=1.5, label='95% CI')
    ax5.axvline(bootstrap_results['ci_upper'][0], color='g', linestyle='--', linewidth=1.5)
    ax5.set_xlabel('R0 (Ω)')
    ax5.set_ylabel('频数')
    ax5.set_title('Bootstrap: R0 分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Bootstrap 参数分布 (R1)
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(bootstrap_results['bootstrap_samples'][:, 1], bins=20, alpha=0.7, edgecolor='black')
    ax6.axvline(bootstrap_results['bootstrap_mean'][1], color='r', linestyle='--', linewidth=2, label='均值')
    ax6.axvline(bootstrap_results['ci_lower'][1], color='g', linestyle='--', linewidth=1.5, label='95% CI')
    ax6.axvline(bootstrap_results['ci_upper'][1], color='g', linestyle='--', linewidth=1.5)
    ax6.set_xlabel('R1 (Ω)')
    ax6.set_ylabel('频数')
    ax6.set_title('Bootstrap: R1 分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. 敏感性曲线 (R0, R1, R2)
    ax7 = plt.subplot(3, 3, 7)
    sens_curves = sensitivity_results['sensitivity_curves']
    ax7.plot(t, sens_curves[:, 0], label='R0', linewidth=1.5)
    ax7.plot(t, sens_curves[:, 1], label='R1', linewidth=1.5)
    ax7.plot(t, sens_curves[:, 3], label='R2', linewidth=1.5)
    ax7.set_xlabel('时间 (s)')
    ax7.set_ylabel('敏感性 (V/参数)')
    ax7.set_title('参数敏感性曲线')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 敏感性指标对比
    ax8 = plt.subplot(3, 3, 8)
    sens_rms = sensitivity_results['sensitivity_rms']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax8.bar(param_names, sens_rms, color=colors, edgecolor='black', linewidth=1.5)
    ax8.set_ylabel('RMS 敏感性')
    ax8.set_title('参数敏感性排序')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, sens_rms):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 9. 分析结论文本框
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 构建结论文本
    conclusion_text = "不确定性分析结论:\n\n"
    conclusion_text += f"1. 拟合质量:\n"
    conclusion_text += f"   RMSE = {fit_result['metrics']['RMSE']:.6f} V\n"
    conclusion_text += f"   R2 = {fit_result['metrics']['R2']:.6f}\n\n"
    
    conclusion_text += f"2. 参数可辨识性:\n"
    if ci_results['high_corr_pairs']:
        conclusion_text += f"   发现高相关参数对:\n"
        for p1, p2, corr in ci_results['high_corr_pairs'][:2]:
            conclusion_text += f"   {p1}-{p2}: {corr:.3f}\n"
    else:
        conclusion_text += f"   参数独立性良好\n"
    
    conclusion_text += f"\n3. 参数敏感性:\n"
    for i, param_name in enumerate(sensitivity_results['sensitivity_ranking'][:3], 1):
        conclusion_text += f"   {i}. {param_name}\n"
    
    conclusion_text += f"\n4. 建议:\n"
    conclusion_text += f"   - 使用更丰富的激励信号\n"
    conclusion_text += f"   - 考虑正则化约束\n"
    conclusion_text += f"   - 验证物理合理性"
    
    ax9.text(0.05, 0.95, conclusion_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'outputs' / 'full_pipeline_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  综合报告已保存: {output_path}")
    
    # ================================================================
    # 测试总结
    # ================================================================
    print("\n" + "="*70)
    print("完整流程测试完成!")
    print("="*70)
    print("\n主要结果:")
    print(f"  1. 参数辨识成功: RMSE = {fit_result['metrics']['RMSE']:.6f} V")
    print(f"  2. 置信区间分析完成: {len(ci_results['high_corr_pairs'])} 个高相关参数对")
    print(f"  3. Bootstrap 分析完成: {bootstrap_results['n_bootstrap']} 次重采样")
    print(f"  4. 敏感性分析完成: 最敏感参数为 {sensitivity_results['sensitivity_ranking'][0]}")
    print(f"\n报告图已生成: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
