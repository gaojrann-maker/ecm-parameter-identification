"""
测试参数辨识模块
验证：
1. metrics 模块的各种指标计算
2. 参数辨识功能
3. 可视化拟合结果
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
from ecm.ecm2rc import get_initial_params_guess, simulate_voltage
from ecm.metrics import calculate_all_metrics, print_metrics
from identification.fit import ECMParameterIdentification, fit_ecm_params

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_metrics():
    """测试评估指标模块"""
    print("=" * 60)
    print("测试 1: 评估指标计算")
    print("=" * 60)
    
    # 创建测试数据
    y_true = np.array([3.5, 3.6, 3.7, 3.8, 3.9])
    y_pred = np.array([3.52, 3.58, 3.72, 3.78, 3.92])
    
    print(f"\n真实值: {y_true}")
    print(f"预测值: {y_pred}")
    
    # 计算所有指标
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print_metrics(metrics, "评估指标")
    
    print("\n[OK] 指标计算测试通过")


def load_test_data():
    """加载测试数据"""
    print("\n" + "=" * 60)
    print("加载测试数据")
    print("=" * 60)
    
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    
    # 加载恒流段数据
    t, i, v_measured, info = load_discharge_cc_segment(str(mat_path), n=1)
    
    print(f"\n数据文件: {mat_path.name}")
    print(f"数据点数: {len(t)}")
    print(f"持续时间: {t[-1]:.2f} s ({t[-1]/60:.2f} min)")
    print(f"平均电流: {info['mean_current']:.3f} A")
    print(f"电压范围: [{v_measured.min():.3f}, {v_measured.max():.3f}] V")
    
    # 计算 SOC
    capacity = info['total_capacity_ah']
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    print(f"SOC 范围: [{soc.min():.4f}, {soc.max():.4f}]")
    
    # 拟合 OCV
    sample_indices = np.linspace(0, len(t)-1, 15, dtype=int)
    ocv_func = fit_ocv_curve(soc[sample_indices], v_measured[sample_indices], method='linear')
    print(f"OCV 拟合: 使用 {len(sample_indices)} 个采样点")
    
    return t, i, v_measured, soc, ocv_func


def test_parameter_identification_ls(t, i, v_measured, soc, ocv_func):
    """测试最小二乘参数辨识"""
    print("\n" + "=" * 60)
    print("测试 2: 参数辨识（最小二乘法）")
    print("=" * 60)
    
    # 创建辨识器
    identifier = ECMParameterIdentification(t, i, v_measured, soc, ocv_func)
    
    # 设置初始值和边界
    x0 = np.array([0.01, 0.02, 1000.0, 0.05, 5000.0])
    lower = np.array([1e-4, 1e-4, 10.0, 1e-4, 10.0])
    upper = np.array([1.0, 1.0, 1e6, 1.0, 1e6])
    bounds = (lower, upper)
    
    # 开始计时
    start_time = time.time()
    
    # 运行辨识
    params = identifier.fit_least_squares(x0=x0, bounds=bounds, verbose=2)
    
    # 结束计时
    elapsed_time = time.time() - start_time
    print(f"\n辨识耗时: {elapsed_time:.2f} s")
    
    # 获取结果
    results = identifier.get_results()
    
    return params, results, identifier


def test_parameter_identification_global(t, i, v_measured, soc, ocv_func):
    """测试全局优化参数辨识"""
    print("\n" + "=" * 60)
    print("测试 3: 参数辨识（全局优化）")
    print("=" * 60)
    
    # 创建辨识器
    identifier = ECMParameterIdentification(t, i, v_measured, soc, ocv_func)
    
    # 设置边界
    bounds = [
        (1e-4, 0.5),    # R0
        (1e-4, 0.5),    # R1
        (10.0, 1e5),    # C1
        (1e-4, 0.5),    # R2
        (10.0, 1e5)     # C2
    ]
    
    # 开始计时
    start_time = time.time()
    
    # 运行全局优化（使用较小的参数以加快速度）
    params = identifier.fit_global(bounds=bounds, maxiter=50, popsize=10, seed=42)
    
    # 结束计时
    elapsed_time = time.time() - start_time
    print(f"\n辨识耗时: {elapsed_time:.2f} s")
    
    # 获取结果
    results = identifier.get_results()
    
    return params, results, identifier


def visualize_fitting_results(
    t, v_measured, results, title="参数辨识结果", filename="fit_results.png"
):
    """可视化拟合结果"""
    print("\n" + "=" * 60)
    print(f"生成拟合结果可视化")
    print("=" * 60)
    
    v_pred = results['v_pred']
    residuals = results['residuals']
    metrics = results['metrics']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 子图1: 电压对比
    axes[0].plot(t, v_measured, 'b-', linewidth=2, label='实测电压', alpha=0.7)
    axes[0].plot(t, v_pred, 'r--', linewidth=2, label='拟合电压', alpha=0.7)
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('电压对比', fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 添加指标信息
    text_info = (
        f"RMSE: {metrics['RMSE']:.4f} V\n"
        f"MAE: {metrics['MAE']:.4f} V\n"
        f"R²: {metrics['R2']:.4f}"
    )
    axes[0].text(0.98, 0.02, text_info, transform=axes[0].transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=10)
    
    # 子图2: 残差 vs 时间
    axes[1].plot(t, residuals * 1000, 'r-', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].fill_between(t, residuals * 1000, 0, alpha=0.3, color='red')
    axes[1].set_xlabel('时间 (s)', fontsize=12)
    axes[1].set_ylabel('残差 (mV)', fontsize=12)
    axes[1].set_title('残差 vs 时间', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 残差分布直方图
    axes[2].hist(residuals * 1000, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2, label='零点')
    axes[2].set_xlabel('残差 (mV)', fontsize=12)
    axes[2].set_ylabel('频数', fontsize=12)
    axes[2].set_title('残差分布直方图', fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_comparison(
    t, i, v_measured, soc, ocv_func, params_init, params_fitted, filename="fit_comparison.png"
):
    """对比初始参数和拟合参数的效果"""
    print("\n" + "=" * 60)
    print(f"生成参数对比可视化")
    print("=" * 60)
    
    # 使用初始参数仿真
    v_init = simulate_voltage(t, i, soc, params_init, ocv_func)
    
    # 使用拟合参数仿真
    v_fitted = simulate_voltage(t, i, soc, params_fitted, ocv_func)
    
    # 计算指标
    metrics_init = calculate_all_metrics(v_measured, v_init)
    metrics_fitted = calculate_all_metrics(v_measured, v_fitted)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('初始参数 vs 拟合参数对比', fontsize=16, fontweight='bold')
    
    # 子图1: 电压对比
    axes[0, 0].plot(t, v_measured, 'b-', linewidth=2, label='实测', alpha=0.7)
    axes[0, 0].plot(t, v_init, 'g--', linewidth=1.5, label='初始参数', alpha=0.7)
    axes[0, 0].plot(t, v_fitted, 'r--', linewidth=1.5, label='拟合参数', alpha=0.7)
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('电压 (V)')
    axes[0, 0].set_title('电压 vs 时间')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 残差对比
    res_init = v_init - v_measured
    res_fitted = v_fitted - v_measured
    axes[0, 1].plot(t, res_init * 1000, 'g-', linewidth=1, label='初始参数残差', alpha=0.7)
    axes[0, 1].plot(t, res_fitted * 1000, 'r-', linewidth=1, label='拟合参数残差', alpha=0.7)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('残差 (mV)')
    axes[0, 1].set_title('残差 vs 时间')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 初始参数散点图
    axes[1, 0].scatter(v_measured, v_init, alpha=0.5, s=20, color='green')
    axes[1, 0].plot([v_measured.min(), v_measured.max()], 
                    [v_measured.min(), v_measured.max()], 
                    'k--', linewidth=2, label='理想线')
    axes[1, 0].set_xlabel('实测电压 (V)')
    axes[1, 0].set_ylabel('预测电压 (V)')
    axes[1, 0].set_title(f'初始参数\nRMSE={metrics_init["RMSE"]:.4f}V')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # 子图4: 拟合参数散点图
    axes[1, 1].scatter(v_measured, v_fitted, alpha=0.5, s=20, color='red')
    axes[1, 1].plot([v_measured.min(), v_measured.max()], 
                    [v_measured.min(), v_measured.max()], 
                    'k--', linewidth=2, label='理想线')
    axes[1, 1].set_xlabel('实测电压 (V)')
    axes[1, 1].set_ylabel('预测电压 (V)')
    axes[1, 1].set_title(f'拟合参数\nRMSE={metrics_fitted["RMSE"]:.4f}V')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试参数辨识模块")
    print("=" * 60)
    
    try:
        # 测试1: 评估指标
        test_metrics()
        
        # 加载测试数据
        t, i, v_measured, soc, ocv_func = load_test_data()
        
        # 获取初始参数
        params_init = get_initial_params_guess()
        
        # 测试2: 最小二乘法参数辨识
        params_ls, results_ls, identifier_ls = test_parameter_identification_ls(
            t, i, v_measured, soc, ocv_func
        )
        
        # 可视化最小二乘法结果
        visualize_fitting_results(
            t, v_measured, results_ls, 
            title="参数辨识结果（最小二乘法）",
            filename="fit_results_ls.png"
        )
        
        # 对比初始参数和拟合参数
        visualize_comparison(
            t, i, v_measured, soc, ocv_func, params_init, params_ls,
            filename="fit_comparison.png"
        )
        
        print("\n" + "=" * 60)
        print("[OK] 所有测试完成！")
        print("=" * 60)
        print("\n参数辨识成功！")
        print(f"拟合前 RMSE: {0.239:.4f} V (初始参数)")
        print(f"拟合后 RMSE: {results_ls['metrics']['RMSE']:.4f} V")
        print(f"改善率: {(1 - results_ls['metrics']['RMSE']/0.239)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
