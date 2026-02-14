"""
测试 ecm2rc.py 的功能
验证：
1. ECM2RCParams 数据类
2. 参数有效性检查
3. 极化电压计算
4. 电压仿真功能
5. 可视化验证
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加 src 到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ecm.loader import load_discharge_cc_segment
from ecm.ocv import calculate_soc, fit_ocv_from_full_cycle, fit_ocv_curve
from ecm.ecm2rc import (
    ECM2RCParams,
    check_params_positive,
    compute_polarization_voltages,
    simulate_voltage,
    simulate_voltage_with_details,
    get_initial_params_guess,
    validate_params_physical
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_params_dataclass():
    """测试参数数据类"""
    print("=" * 60)
    print("测试 1: ECM2RCParams 数据类")
    print("=" * 60)
    
    # 创建参数对象
    params = ECM2RCParams(
        R0=0.05,
        R1=0.02,
        C1=100.0,
        R2=0.05,
        C2=1000.0
    )
    
    print("\n参数对象:")
    print(params)
    
    # 测试时间常数
    tau1, tau2 = params.get_time_constants()
    print(f"\n时间常数:")
    print(f"  τ1 = {tau1:.3f} s (快速极化)")
    print(f"  τ2 = {tau2:.3f} s (慢速极化)")
    
    # 测试数组转换
    arr = params.to_array()
    print(f"\n数组形式: {arr}")
    
    params_from_arr = ECM2RCParams.from_array(arr)
    print(f"从数组重建: {params_from_arr == params}")
    
    # 测试初始猜测
    params_guess = get_initial_params_guess()
    print(f"\n初始参数猜测:")
    print(params_guess)
    
    return params


def test_params_validation(params):
    """测试参数验证"""
    print("\n" + "=" * 60)
    print("测试 2: 参数验证")
    print("=" * 60)
    
    # 测试正数检查
    is_positive = check_params_positive(params)
    print(f"\n参数正数检查: {is_positive}")
    
    # 测试物理合理性
    is_valid, issues = validate_params_physical(params, warn=False)
    print(f"\n物理合理性检查: {is_valid}")
    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  所有参数在合理范围内")
    
    # 测试异常参数
    print("\n测试异常参数:")
    try:
        bad_params = ECM2RCParams(R0=-0.01, R1=0.02, C1=100, R2=0.05, C2=1000)
        print("  [FAIL] 应该抛出异常")
    except ValueError as e:
        print(f"  [OK] 正确捕获异常: {e}")
    
    # 测试不合理参数
    extreme_params = ECM2RCParams(R0=10.0, R1=0.001, C1=1, R2=0.1, C2=100000)
    print("\n极端参数:")
    print(extreme_params)
    is_valid, issues = validate_params_physical(extreme_params, warn=False)
    print(f"物理合理性: {is_valid}")
    print(f"发现 {len(issues)} 个问题")


def test_polarization_voltage():
    """测试极化电压计算"""
    print("\n" + "=" * 60)
    print("测试 3: 极化电压计算")
    print("=" * 60)
    
    # 创建简单的测试数据
    t = np.linspace(0, 100, 1000)  # 100秒，1000个点
    i = -2.0 * np.ones_like(t)      # 恒流放电 2A
    
    # 在中间切换为静置
    i[500:] = 0.0
    
    params = get_initial_params_guess()
    
    # 计算极化电压
    V1, V2 = compute_polarization_voltages(t, i, params)
    
    print(f"\n模拟条件:")
    print(f"  时间: 0-100 s, {len(t)} 个点")
    print(f"  电流: -2.0 A (0-50s), 0 A (50-100s)")
    print(f"\n使用参数:")
    print(f"  τ1 = {params.R1 * params.C1:.3f} s")
    print(f"  τ2 = {params.R2 * params.C2:.3f} s")
    
    print(f"\n极化电压结果:")
    print(f"  V1 初始: {V1[0]:.6f} V")
    print(f"  V1 放电结束: {V1[499]:.6f} V")
    print(f"  V1 静置后: {V1[-1]:.6f} V")
    print(f"  V2 初始: {V2[0]:.6f} V")
    print(f"  V2 放电结束: {V2[499]:.6f} V")
    print(f"  V2 静置后: {V2[-1]:.6f} V")
    
    return t, i, V1, V2, params


def test_voltage_simulation():
    """测试完整的电压仿真"""
    print("\n" + "=" * 60)
    print("测试 4: 完整电压仿真")
    print("=" * 60)
    
    # 加载真实数据
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    print(f"\n加载数据: {mat_path.name}")
    
    t, i, v_measured, info = load_discharge_cc_segment(str(mat_path), n=1)
    
    print(f"数据点数: {len(t)}")
    print(f"持续时间: {t[-1]:.2f} s")
    print(f"平均电流: {info['mean_current']:.3f} A")
    
    # 计算 SOC
    capacity = info['total_capacity_ah']
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    
    print(f"SOC 范围: [{soc.min():.4f}, {soc.max():.4f}]")
    
    # 拟合 OCV（使用采样点）
    sample_indices = np.linspace(0, len(t)-1, 15, dtype=int)
    sample_soc = soc[sample_indices]
    sample_v = v_measured[sample_indices]
    ocv_func = fit_ocv_curve(sample_soc, sample_v, method='linear')
    
    print(f"OCV 拟合: 使用 {len(sample_indices)} 个采样点")
    
    # 使用初始参数进行仿真
    params = get_initial_params_guess()
    print(f"\n使用初始参数猜测:")
    print(params)
    
    # 仿真电压
    V_pred = simulate_voltage(t, i, soc, params, ocv_func)
    
    # 计算误差
    error = v_measured - V_pred
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    print(f"\n仿真误差:")
    print(f"  RMSE: {rmse:.6f} V")
    print(f"  MAE: {mae:.6f} V")
    print(f"  最大误差: {max_error:.6f} V")
    
    return t, i, v_measured, soc, ocv_func, params, V_pred


def test_voltage_simulation_with_details():
    """测试带详细信息的电压仿真"""
    print("\n" + "=" * 60)
    print("测试 5: 带详细信息的电压仿真")
    print("=" * 60)
    
    # 加载数据
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    t, i, v_measured, info = load_discharge_cc_segment(str(mat_path), n=1)
    
    # 计算 SOC 和拟合 OCV
    capacity = info['total_capacity_ah']
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    sample_indices = np.linspace(0, len(t)-1, 15, dtype=int)
    ocv_func = fit_ocv_curve(soc[sample_indices], v_measured[sample_indices], method='linear')
    
    # 使用带详细信息的仿真
    params = get_initial_params_guess()
    result = simulate_voltage_with_details(t, i, soc, params, ocv_func)
    
    print(f"\n详细结果键: {list(result.keys())}")
    print(f"\n各电压分量统计:")
    print(f"  OCV 范围: [{result['V_ocv'].min():.4f}, {result['V_ocv'].max():.4f}] V")
    print(f"  欧姆压降范围: [{result['V_ohm'].min():.4f}, {result['V_ohm'].max():.4f}] V")
    print(f"  V1 范围: [{result['V1'].min():.4f}, {result['V1'].max():.4f}] V")
    print(f"  V2 范围: [{result['V2'].min():.4f}, {result['V2'].max():.4f}] V")
    print(f"  预测电压范围: [{result['V_pred'].min():.4f}, {result['V_pred'].max():.4f}] V")
    
    return t, i, v_measured, soc, result


def visualize_polarization(t, i, V1, V2, params):
    """可视化极化电压"""
    print("\n" + "=" * 60)
    print("生成极化电压可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('极化电压计算验证', fontsize=16, fontweight='bold')
    
    # 子图1: 电流
    axes[0].plot(t, i, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].axvline(x=50, color='r', linestyle='--', alpha=0.5, label='切换点')
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电流 (A)', fontsize=12)
    axes[0].set_title('输入电流', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: V1 (快速极化)
    tau1 = params.R1 * params.C1
    axes[1].plot(t, V1, 'r-', linewidth=2, label=f'V1 (τ1={tau1:.2f}s)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('时间 (s)', fontsize=12)
    axes[1].set_ylabel('极化电压 (V)', fontsize=12)
    axes[1].set_title('RC支路1极化电压（快速极化）', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: V2 (慢速极化)
    tau2 = params.R2 * params.C2
    axes[2].plot(t, V2, 'g-', linewidth=2, label=f'V2 (τ2={tau2:.2f}s)')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('时间 (s)', fontsize=12)
    axes[2].set_ylabel('极化电压 (V)', fontsize=12)
    axes[2].set_title('RC支路2极化电压（慢速极化）', fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ecm2rc_test_polarization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_voltage_simulation(t, i, v_measured, soc, ocv_func, params, V_pred):
    """可视化电压仿真结果"""
    print("\n" + "=" * 60)
    print("生成电压仿真可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('二阶ECM模型电压仿真', fontsize=16, fontweight='bold')
    
    # 计算误差
    error = v_measured - V_pred
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    
    # 子图1: 电压对比
    axes[0].plot(t, v_measured, 'b-', linewidth=2, label='实测电压', alpha=0.7)
    axes[0].plot(t, V_pred, 'r--', linewidth=2, label='仿真电压', alpha=0.7)
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('电压对比', fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 添加统计信息
    text_info = f"RMSE: {rmse:.4f} V\nMAE: {mae:.4f} V"
    axes[0].text(0.98, 0.02, text_info, transform=axes[0].transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    # 子图2: 误差
    axes[1].plot(t, error * 1000, 'r-', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].fill_between(t, error * 1000, 0, alpha=0.3, color='red')
    axes[1].set_xlabel('时间 (s)', fontsize=12)
    axes[1].set_ylabel('误差 (mV)', fontsize=12)
    axes[1].set_title('仿真误差', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 电压-SOC 曲线
    axes[2].plot(soc, v_measured, 'b-', linewidth=2, label='实测电压', alpha=0.7)
    axes[2].plot(soc, V_pred, 'r--', linewidth=2, label='仿真电压', alpha=0.7)
    axes[2].set_xlabel('SOC', fontsize=12)
    axes[2].set_ylabel('电压 (V)', fontsize=12)
    axes[2].set_title('电压 vs SOC', fontsize=13)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ecm2rc_test_simulation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_voltage_components(t, i, v_measured, soc, result):
    """可视化电压各分量"""
    print("\n" + "=" * 60)
    print("生成电压分量可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle('ECM模型电压分量分析', fontsize=16, fontweight='bold')
    
    # 子图1: 总电压
    axes[0].plot(t, v_measured, 'b-', linewidth=2, label='实测电压', alpha=0.7)
    axes[0].plot(t, result['V_pred'], 'r--', linewidth=2, label='仿真电压', alpha=0.7)
    axes[0].set_ylabel('电压 (V)', fontsize=11)
    axes[0].set_title('总电压对比', fontsize=12)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: OCV
    axes[1].plot(t, result['V_ocv'], 'g-', linewidth=2, label='OCV(SOC)')
    axes[1].set_ylabel('OCV (V)', fontsize=11)
    axes[1].set_title('开路电压', fontsize=12)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 欧姆压降和极化电压
    axes[2].plot(t, result['V_ohm'], 'orange', linewidth=2, label='I·R0 (欧姆压降)', alpha=0.7)
    axes[2].plot(t, result['V1'], 'purple', linewidth=2, label='V1 (快速极化)', alpha=0.7)
    axes[2].plot(t, result['V2'], 'brown', linewidth=2, label='V2 (慢速极化)', alpha=0.7)
    axes[2].set_ylabel('电压 (V)', fontsize=11)
    axes[2].set_title('压降分量', fontsize=12)
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # 子图4: 堆叠图显示电压组成
    # V_terminal = OCV - I*R0 - V1 - V2
    total_drop = result['V_ohm'] + result['V1'] + result['V2']
    axes[3].fill_between(t, 0, result['V_ohm'], alpha=0.5, label='I·R0', color='orange')
    axes[3].fill_between(t, result['V_ohm'], result['V_ohm']+result['V1'], 
                         alpha=0.5, label='V1', color='purple')
    axes[3].fill_between(t, result['V_ohm']+result['V1'], total_drop, 
                         alpha=0.5, label='V2', color='brown')
    axes[3].plot(t, total_drop, 'k-', linewidth=2, label='总压降')
    axes[3].set_xlabel('时间 (s)', fontsize=11)
    axes[3].set_ylabel('压降 (V)', fontsize=11)
    axes[3].set_title('压降堆叠图', fontsize=12)
    axes[3].legend(loc='best', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ecm2rc_test_components.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试 ecm2rc.py 模块")
    print("=" * 60)
    
    try:
        # 测试1: 参数数据类
        params = test_params_dataclass()
        
        # 测试2: 参数验证
        test_params_validation(params)
        
        # 测试3: 极化电压计算
        t_simple, i_simple, V1, V2, params_simple = test_polarization_voltage()
        
        # 测试4: 完整电压仿真
        t, i, v_measured, soc, ocv_func, params, V_pred = test_voltage_simulation()
        
        # 测试5: 带详细信息的仿真
        t_det, i_det, v_det, soc_det, result_det = test_voltage_simulation_with_details()
        
        # 可视化1: 极化电压
        visualize_polarization(t_simple, i_simple, V1, V2, params_simple)
        
        # 可视化2: 电压仿真
        visualize_voltage_simulation(t, i, v_measured, soc, ocv_func, params, V_pred)
        
        # 可视化3: 电压分量
        visualize_voltage_components(t_det, i_det, v_det, soc_det, result_det)
        
        print("\n" + "=" * 60)
        print("[OK] 所有测试完成！")
        print("=" * 60)
        print("\n注意: 使用初始参数猜测的仿真误差较大，")
        print("需要通过参数辨识优化参数以获得更好的拟合效果。")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
