"""
测试 ocv.py 的功能
验证：
1. SOC 计算的正确性
2. 静置段提取功能
3. OCV-SOC 曲线拟合
4. 一站式函数
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

from ecm.loader import load_b0005_cycles, get_nth_discharge
from ecm.ocv import (
    calculate_soc,
    extract_rest_segments,
    fit_ocv_curve,
    fit_ocv_from_full_cycle
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_soc_calculation():
    """测试 SOC 计算"""
    print("=" * 60)
    print("测试 1: SOC 计算")
    print("=" * 60)
    
    # 加载第1次放电数据
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    cycles = load_b0005_cycles(str(mat_path))
    t, i, v, capacity = get_nth_discharge(cycles, n=1)
    
    # 计算 SOC（从满电开始放电）
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    
    print(f"电池容量: {capacity:.4f} Ah")
    print(f"初始 SOC: {soc[0]:.4f}")
    print(f"结束 SOC: {soc[-1]:.4f}")
    print(f"SOC 范围: [{soc.min():.4f}, {soc.max():.4f}]")
    print(f"放电深度: {(soc[0] - soc[-1]) * 100:.2f}%")
    
    # 计算理论放电容量
    theoretical_capacity = (soc[0] - soc[-1]) * capacity
    print(f"理论放电容量: {theoretical_capacity:.4f} Ah")
    print(f"额定容量: {capacity:.4f} Ah")
    
    return t, i, v, soc, capacity


def test_rest_extraction(t, i, v, soc):
    """测试静置段提取"""
    print("\n" + "=" * 60)
    print("测试 2: 静置段提取")
    print("=" * 60)
    
    # 提取静置段（使用较宽松的阈值，因为放电过程中静置较少）
    rest_soc, rest_v, segments_info = extract_rest_segments(
        t, i, v, soc,
        current_threshold=0.1,  # 0.1 A
        min_duration=60.0,      # 60 秒
        method='end'
    )
    
    print(f"找到 {len(rest_soc)} 个静置段")
    
    if len(rest_soc) > 0:
        print(f"\n静置段详细信息:")
        for i, info in enumerate(segments_info):
            print(f"\n  静置段 {i+1}:")
            print(f"    持续时间: {info['duration']:.2f} s")
            print(f"    数据点数: {info['num_points']}")
            print(f"    SOC: {info['soc']:.4f}")
            print(f"    平均电压: {info['v_mean']:.4f} V")
            print(f"    电压标准差: {info['v_std']:.6f} V")
            print(f"    平均电流: {info['i_mean']:.6f} A")
            print(f"    最大电流: {info['i_max']:.6f} A")
    else:
        print("警告: 未找到静置段，将尝试从整个数据集中采样")
    
    return rest_soc, rest_v, segments_info


def test_ocv_fitting_from_samples():
    """测试 OCV 拟合（使用采样点模拟静置段）"""
    print("\n" + "=" * 60)
    print("测试 3: OCV 拟合（使用采样点）")
    print("=" * 60)
    
    # 加载数据
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    cycles = load_b0005_cycles(str(mat_path))
    t, i, v, capacity = get_nth_discharge(cycles, n=1)
    soc = calculate_soc(t, i, capacity, initial_soc=1.0)
    
    # 由于放电过程中可能没有真正的静置段，我们在电流较小的点采样
    # 找到电流相对较小的点
    current_threshold = 0.5  # 更宽松的阈值
    low_current_mask = np.abs(i) > current_threshold
    
    if np.sum(low_current_mask) > 10:
        # 均匀采样
        sample_indices = np.linspace(0, len(t)-1, 15, dtype=int)
        sample_soc = soc[sample_indices]
        sample_v = v[sample_indices]
        
        print(f"从数据中采样 {len(sample_soc)} 个点用于 OCV 拟合")
        print(f"SOC 范围: [{sample_soc.min():.4f}, {sample_soc.max():.4f}]")
        print(f"电压范围: [{sample_v.min():.4f}, {sample_v.max():.4f}] V")
        
        # 测试不同的拟合方法
        methods = ['linear', 'cubic']
        ocv_funcs = {}
        
        for method in methods:
            try:
                ocv_func = fit_ocv_curve(sample_soc, sample_v, method=method)
                ocv_funcs[method] = ocv_func
                print(f"\n{method} 插值拟合成功")
                
                # 测试插值
                test_soc = np.array([0.2, 0.5, 0.8])
                test_v = ocv_func(test_soc)
                print(f"  测试点: SOC={test_soc}, V={test_v}")
            except Exception as e:
                print(f"\n{method} 插值失败: {e}")
        
        return sample_soc, sample_v, ocv_funcs, soc, v
    else:
        print("数据中电流过大，无法进行有效采样")
        return None, None, None, soc, v


def test_full_cycle_with_charge():
    """测试完整充电循环的 OCV 拟合（充电过程有更多静置段）"""
    print("\n" + "=" * 60)
    print("测试 4: 使用充电循环拟合 OCV（包含静置段）")
    print("=" * 60)
    
    # 加载数据
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    cycles = load_b0005_cycles(str(mat_path))
    
    # 找第一个充电循环
    charge_idx = None
    for i, c in enumerate(cycles):
        if c["type"][0] == "charge":
            charge_idx = i
            break
    
    if charge_idx is None:
        print("未找到充电循环")
        return None
    
    # 获取充电数据
    c = cycles[charge_idx]
    d = c["data"][0, 0]
    
    t = d["Time"].flatten().astype(float)
    v = d["Voltage_measured"].flatten().astype(float)
    i_meas = d["Current_measured"].flatten().astype(float)
    
    # 时间归零
    t = t - t[0]
    
    print(f"充电数据点数: {len(t)}")
    print(f"持续时间: {t[-1]:.2f} s ({t[-1]/60:.2f} min)")
    print(f"电流范围: [{i_meas.min():.4f}, {i_meas.max():.4f}] A")
    print(f"电压范围: [{v.min():.4f}, {v.max():.4f}] V")
    
    # 计算 SOC（充电从低 SOC 开始）
    capacity = 2.0  # 额定容量
    soc = calculate_soc(t, i_meas, capacity, initial_soc=0.2)
    
    # 提取静置段
    rest_soc, rest_v, segments_info = extract_rest_segments(
        t, i_meas, v, soc,
        current_threshold=0.05,
        min_duration=30.0,
        method='end'
    )
    
    print(f"\n找到 {len(rest_soc)} 个静置段")
    
    if len(rest_soc) >= 2:
        # 拟合 OCV 曲线
        ocv_func = fit_ocv_curve(rest_soc, rest_v, method='linear')
        print("OCV 曲线拟合成功")
        
        return t, i_meas, v, soc, rest_soc, rest_v, ocv_func, segments_info
    else:
        print("静置段不足，无法拟合 OCV 曲线")
        return t, i_meas, v, soc, rest_soc, rest_v, None, segments_info


def visualize_soc_calculation(t, i, v, soc, capacity):
    """可视化 SOC 计算结果"""
    print("\n" + "=" * 60)
    print("生成 SOC 计算可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('SOC 计算验证', fontsize=16, fontweight='bold')
    
    # 子图1: 电压 vs 时间
    axes[0].plot(t, v, 'b-', linewidth=2)
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('电压 vs 时间', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 电流 vs 时间
    axes[1].plot(t, i, 'r-', linewidth=2)
    axes[1].set_xlabel('时间 (s)', fontsize=12)
    axes[1].set_ylabel('电流 (A)', fontsize=12)
    axes[1].set_title('电流 vs 时间', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: SOC vs 时间
    axes[2].plot(t, soc * 100, 'g-', linewidth=2)
    axes[2].set_xlabel('时间 (s)', fontsize=12)
    axes[2].set_ylabel('SOC (%)', fontsize=12)
    axes[2].set_title('SOC vs 时间', fontsize=13)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ocv_test_soc_calculation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_ocv_fitting(sample_soc, sample_v, ocv_funcs, full_soc, full_v):
    """可视化 OCV 拟合结果"""
    print("\n" + "=" * 60)
    print("生成 OCV 拟合可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('OCV-SOC 曲线拟合', fontsize=16, fontweight='bold')
    
    # 生成密集的 SOC 点用于绘制拟合曲线
    soc_dense = np.linspace(sample_soc.min(), sample_soc.max(), 200)
    
    # 子图1: OCV 拟合曲线
    axes[0].scatter(sample_soc, sample_v, color='red', s=100, 
                    label='采样点（用于拟合）', zorder=5, marker='o')
    
    colors = {'linear': 'blue', 'cubic': 'green'}
    for method, ocv_func in ocv_funcs.items():
        v_fit = ocv_func(soc_dense)
        axes[0].plot(soc_dense, v_fit, color=colors.get(method, 'black'), 
                    linewidth=2, label=f'{method} 插值', alpha=0.7)
    
    axes[0].set_xlabel('SOC', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('OCV-SOC 拟合曲线', fontsize=13)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 拟合误差
    if 'linear' in ocv_funcs:
        ocv_func = ocv_funcs['linear']
        v_pred = ocv_func(sample_soc)
        residual = sample_v - v_pred
        
        axes[1].scatter(sample_soc, residual, color='red', s=80, marker='o')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('SOC', fontsize=12)
        axes[1].set_ylabel('残差 (V)', fontsize=12)
        axes[1].set_title('拟合残差（线性插值）', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        rmse = np.sqrt(np.mean(residual**2))
        mae = np.mean(np.abs(residual))
        text_info = f"RMSE: {rmse:.6f} V\nMAE: {mae:.6f} V"
        axes[1].text(0.98, 0.97, text_info, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ocv_test_fitting.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def visualize_charge_cycle_ocv(t, i, v, soc, rest_soc, rest_v, ocv_func, segments_info):
    """可视化充电循环的 OCV 提取"""
    print("\n" + "=" * 60)
    print("生成充电循环 OCV 提取可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('充电循环静置段提取与 OCV 拟合', fontsize=16, fontweight='bold')
    
    # 子图1: 电流 vs 时间，标记静置段
    axes[0].plot(t, i, 'b-', linewidth=1, alpha=0.5, label='电流')
    for info in segments_info:
        start_t = t[info['start_idx']]
        end_t = t[info['end_idx']]
        axes[0].axvspan(start_t, end_t, alpha=0.3, color='yellow', label='静置段' if info == segments_info[0] else '')
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电流 (A)', fontsize=12)
    axes[0].set_title('电流曲线与静置段', fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 电压 vs SOC，显示原始数据和静置点
    axes[1].plot(soc, v, 'b-', linewidth=1, alpha=0.3, label='原始数据')
    axes[1].scatter(rest_soc, rest_v, color='red', s=100, 
                   label='静置段（OCV 点）', zorder=5, marker='o')
    axes[1].set_xlabel('SOC', fontsize=12)
    axes[1].set_ylabel('电压 (V)', fontsize=12)
    axes[1].set_title('电压 vs SOC', fontsize=13)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: OCV 拟合曲线
    if ocv_func is not None and len(rest_soc) >= 2:
        soc_dense = np.linspace(rest_soc.min(), rest_soc.max(), 200)
        v_fit = ocv_func(soc_dense)
        
        axes[2].scatter(rest_soc, rest_v, color='red', s=100, 
                       label='静置段点', zorder=5, marker='o')
        axes[2].plot(soc_dense, v_fit, 'g-', linewidth=2, label='OCV 拟合曲线')
        axes[2].set_xlabel('SOC', fontsize=12)
        axes[2].set_ylabel('电压 (V)', fontsize=12)
        axes[2].set_title('OCV-SOC 拟合曲线', fontsize=13)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, '静置段不足，无法拟合', 
                    transform=axes[2].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "ocv_test_charge_cycle.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试 ocv.py 模块")
    print("=" * 60)
    
    try:
        # 测试1: SOC 计算
        t, i, v, soc, capacity = test_soc_calculation()
        
        # 测试2: 静置段提取（放电循环）
        rest_soc, rest_v, segments_info = test_rest_extraction(t, i, v, soc)
        
        # 测试3: OCV 拟合（使用采样点）
        sample_soc, sample_v, ocv_funcs, full_soc, full_v = test_ocv_fitting_from_samples()
        
        # 可视化1: SOC 计算
        visualize_soc_calculation(t, i, v, soc, capacity)
        
        # 可视化2: OCV 拟合
        if sample_soc is not None and ocv_funcs:
            visualize_ocv_fitting(sample_soc, sample_v, ocv_funcs, full_soc, full_v)
        
        # 测试4: 充电循环（包含更多静置段）
        result = test_full_cycle_with_charge()
        if result is not None and len(result) == 8:
            t_charge, i_charge, v_charge, soc_charge, rest_soc_charge, rest_v_charge, ocv_func_charge, seg_info_charge = result
            
            # 可视化3: 充电循环 OCV 提取
            if len(rest_soc_charge) > 0:
                visualize_charge_cycle_ocv(t_charge, i_charge, v_charge, soc_charge, 
                                          rest_soc_charge, rest_v_charge, ocv_func_charge, seg_info_charge)
        
        print("\n" + "=" * 60)
        print("[OK] 所有测试完成！")
        print("=" * 60)
        print("\n注意: NASA B0005 数据中放电循环的静置段较少，")
        print("实际应用中建议使用包含充放电循环的完整数据。")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
