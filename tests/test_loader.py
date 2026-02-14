"""
测试 loader.py 的功能
验证：
1. 能否正确加载 B0005.mat 数据
2. 能否提取指定次数的放电循环
3. 能否正确截取恒流段
4. 可视化验证结果
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加 src 到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ecm.loader import (
    load_b0005_cycles,
    list_discharge_indices,
    get_nth_discharge,
    extract_constant_current_segment,
    load_discharge_cc_segment
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_basic_loading():
    """测试基本数据加载"""
    print("=" * 60)
    print("测试 1: 基本数据加载")
    print("=" * 60)
    
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    print(f"数据路径: {mat_path}")
    
    # 加载所有 cycles
    cycles = load_b0005_cycles(str(mat_path))
    print(f"总共 {len(cycles)} 个循环")
    
    # 统计循环类型
    types = {}
    for c in cycles:
        cycle_type = c["type"][0]
        types[cycle_type] = types.get(cycle_type, 0) + 1
    
    print(f"循环类型统计: {types}")
    
    # 找出所有放电循环
    discharge_idx = list_discharge_indices(cycles)
    print(f"放电循环数量: {len(discharge_idx)}")
    print(f"前10个放电循环索引: {discharge_idx[:10]}")
    
    return cycles, discharge_idx


def test_discharge_extraction(cycles, n=1):
    """测试提取指定放电循环"""
    print("\n" + "=" * 60)
    print(f"测试 2: 提取第 {n} 次放电循环")
    print("=" * 60)
    
    t, i_meas, v, cap_ah = get_nth_discharge(cycles, n)
    
    print(f"放电容量: {cap_ah:.4f} Ahr")
    print(f"数据点数: {len(t)}")
    print(f"持续时间: {t[-1] - t[0]:.2f} s ({(t[-1] - t[0])/60:.2f} min)")
    print(f"电流范围: [{i_meas.min():.4f}, {i_meas.max():.4f}] A")
    print(f"电压范围: [{v.min():.4f}, {v.max():.4f}] V")
    
    return t, i_meas, v, cap_ah


def test_cc_extraction(t, i_meas, v):
    """测试恒流段提取"""
    print("\n" + "=" * 60)
    print("测试 3: 提取恒流段")
    print("=" * 60)
    
    t_cc, i_cc, v_cc, info = extract_constant_current_segment(
        t, i_meas, v,
        current_threshold=0.05,
        min_duration=60.0
    )
    
    print(f"恒流段统计信息:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return t_cc, i_cc, v_cc, info


def test_integrated_function(n=1):
    """测试一站式函数"""
    print("\n" + "=" * 60)
    print(f"测试 4: 一站式函数加载第 {n} 次放电恒流段")
    print("=" * 60)
    
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    t_cc, i_cc, v_cc, info = load_discharge_cc_segment(
        str(mat_path), n,
        current_threshold=0.05,
        min_duration=60.0
    )
    
    print(f"完整信息:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return t_cc, i_cc, v_cc, info


def visualize_results(t, i_meas, v, t_cc, i_cc, v_cc, info, n=1):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'B0005 第 {n} 次放电循环数据分析', fontsize=16, fontweight='bold')
    
    # 子图1: 电压曲线
    ax1 = axes[0]
    ax1.plot(t, v, 'b-', alpha=0.5, label='原始数据', linewidth=1)
    ax1.plot(t_cc, v_cc, 'r-', label='恒流段', linewidth=2)
    ax1.axvline(t[info['start_idx']], color='g', linestyle='--', alpha=0.7, label='恒流段起点')
    ax1.axvline(t[info['end_idx']], color='orange', linestyle='--', alpha=0.7, label='恒流段终点')
    ax1.set_xlabel('时间 (s)', fontsize=12)
    ax1.set_ylabel('电压 (V)', fontsize=12)
    ax1.set_title('电压 vs 时间', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 电流曲线
    ax2 = axes[1]
    ax2.plot(t, i_meas, 'b-', alpha=0.5, label='原始数据', linewidth=1)
    ax2.plot(t_cc + t[info['start_idx']], i_cc, 'r-', label='恒流段', linewidth=2)
    ax2.axvline(t[info['start_idx']], color='g', linestyle='--', alpha=0.7)
    ax2.axvline(t[info['end_idx']], color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(info['mean_current'], color='purple', linestyle=':', alpha=0.7, 
                label=f"平均电流: {info['mean_current']:.3f} A")
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('电流 (A)', fontsize=12)
    ax2.set_title('电流 vs 时间', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 电压-容量曲线（恒流段）
    ax3 = axes[2]
    # 计算累积容量 (Ah)
    capacity_cc = np.zeros(len(t_cc))
    for i in range(1, len(t_cc)):
        dt = t_cc[i] - t_cc[i-1]  # 时间间隔 (s)
        capacity_cc[i] = capacity_cc[i-1] + abs(i_cc[i]) * dt / 3600  # 累积容量 (Ah)
    
    ax3.plot(capacity_cc, v_cc, 'r-', linewidth=2, marker='o', markersize=2)
    ax3.set_xlabel('放电容量 (Ah)', fontsize=12)
    ax3.set_ylabel('电压 (V)', fontsize=12)
    ax3.set_title('恒流段放电曲线', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    # 添加文本信息
    text_info = (
        f"恒流段统计:\n"
        f"持续时间: {info['duration']:.2f} s\n"
        f"数据点数: {info['num_points']}\n"
        f"平均电流: {info['mean_current']:.3f} A\n"
        f"电流标准差: {info['std_current']:.4f} A\n"
        f"电压范围: [{info['voltage_range'][0]:.3f}, {info['voltage_range'][1]:.3f}] V"
    )
    ax3.text(0.98, 0.97, text_info, transform=ax3.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"loader_test_cycle_{n}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    plt.show()


def test_multiple_cycles():
    """测试多个不同循环"""
    print("\n" + "=" * 60)
    print("测试 5: 比较不同放电循环")
    print("=" * 60)
    
    mat_path = project_root / "data" / "raw" / "B0005.mat"
    test_cycles = [1, 10, 50]  # 测试第1、10、50次放电
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('不同循环次数的恒流段对比', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red']
    
    for idx, n in enumerate(test_cycles):
        try:
            t_cc, i_cc, v_cc, info = load_discharge_cc_segment(str(mat_path), n)
            
            # 计算累积容量
            capacity_cc = np.zeros(len(t_cc))
            for i in range(1, len(t_cc)):
                dt = t_cc[i] - t_cc[i-1]
                capacity_cc[i] = capacity_cc[i-1] + abs(i_cc[i]) * dt / 3600
            
            # 绘制电压-时间
            axes[0].plot(t_cc, v_cc, color=colors[idx], label=f'第{n}次放电', linewidth=2)
            
            # 绘制电压-容量
            axes[1].plot(capacity_cc, v_cc, color=colors[idx], label=f'第{n}次放电', linewidth=2)
            
            print(f"第{n}次放电: 容量={info['total_capacity_ah']:.4f} Ah, "
                  f"平均电流={info['mean_current']:.3f} A")
            
        except Exception as e:
            print(f"第{n}次放电提取失败: {e}")
    
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('电压 vs 时间', fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('放电容量 (Ah)', fontsize=12)
    axes[1].set_ylabel('电压 (V)', fontsize=12)
    axes[1].set_title('放电曲线', fontsize=13)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = project_root / "tests" / "outputs"
    output_path = output_dir / "loader_test_multiple_cycles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")
    
    plt.show()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试 loader.py 模块")
    print("=" * 60)
    
    try:
        # 测试1: 基本加载
        cycles, discharge_idx = test_basic_loading()
        
        # 测试2-3: 提取第1次放电并截取恒流段
        n = 1
        t, i_meas, v, cap_ah = test_discharge_extraction(cycles, n)
        t_cc, i_cc, v_cc, info = test_cc_extraction(t, i_meas, v)
        
        # 测试4: 一站式函数
        t_cc2, i_cc2, v_cc2, info2 = test_integrated_function(n)
        
        # 验证两种方法结果一致
        assert np.allclose(t_cc, t_cc2), "两种方法得到的时间序列不一致"
        assert np.allclose(i_cc, i_cc2), "两种方法得到的电流序列不一致"
        assert np.allclose(v_cc, v_cc2), "两种方法得到的电压序列不一致"
        print(f"\n[OK] 两种提取方法结果一致性验证通过")
        
        # 可视化
        visualize_results(t, i_meas, v, t_cc, i_cc, v_cc, info2, n)
        
        # 测试5: 多循环对比
        test_multiple_cycles()
        
        print("\n" + "=" * 60)
        print("[OK] 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
