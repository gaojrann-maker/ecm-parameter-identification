"""
数据加载模块
功能：
1. 读取 NASA B0005 电池数据
2. 提取指定次数的放电循环
3. 截取恒流段数据
"""

import numpy as np
from scipy.io import loadmat
from typing import Tuple, List, Optional
import warnings


def load_b0005_cycles(
    mat_path: str = r"C:\Users\GaoJi\Desktop\ecm-identification\data\raw\B0005.mat",
    key: str = "B0005",
):
    """
    加载 B0005.mat 文件，返回所有 cycle 结构体数组
    
    参数:
        mat_path: .mat 文件路径
        key: MATLAB 文件中的顶层变量名
    
    返回:
        cycles: numpy 数组，每个元素是一个 cycle 结构体
    """
    mat = loadmat(mat_path)
    battery = mat[key][0, 0]          # MATLAB struct
    cycles = battery["cycle"][0]      # 1D array, 每个元素是一个 cycle struct
    return cycles


def list_discharge_indices(cycles) -> List[int]:
    """
    列出所有放电循环的索引
    
    参数:
        cycles: load_b0005_cycles 返回的 cycle 数组
    
    返回:
        idx: 放电循环的索引列表
    """
    idx = []
    for i, c in enumerate(cycles):
        # c["type"] 是个数组，比如 array(['discharge'], dtype='<U9')
        if c["type"][0] == "discharge":
            idx.append(i)
    return idx


def get_nth_discharge(cycles, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    获取第 n 次放电循环的原始数据
    
    参数:
        cycles: load_b0005_cycles 返回的 cycle 数组
        n: 放电循环序号（从 1 开始，符合人类习惯）
    
    返回:
        t: 时间序列 (s)
        i_meas: 电流序列 (A)
        v: 电压序列 (V)
        cap_ah: 该次放电容量 (Ahr)
    """
    discharge_idx = list_discharge_indices(cycles)
    if n < 1 or n > len(discharge_idx):
        raise ValueError(f"n={n} 超出范围，有效范围为 1 到 {len(discharge_idx)}")
    
    i = discharge_idx[n - 1]          # 第 n 次放电对应 cycles 的真实下标
    c = cycles[i]
    d = c["data"][0, 0]               # 真正的数据 struct

    t = d["Time"].flatten().astype(float)
    v = d["Voltage_measured"].flatten().astype(float)
    i_meas = d["Current_measured"].flatten().astype(float)

    cap_ah = float(np.array(d["Capacity"]).squeeze())  # 标量 Ahr
    return t, i_meas, v, cap_ah


def extract_constant_current_segment(
    t: np.ndarray, 
    i_meas: np.ndarray, 
    v: np.ndarray,
    current_threshold: float = 0.05,
    min_duration: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    从放电数据中截取恒流段
    
    原理：
    - 恒流段定义为电流变化小于阈值的连续区间
    - 需要排除开始和结束时的非稳态区间
    
    参数:
        t: 时间序列 (s)
        i_meas: 电流序列 (A)
        v: 电压序列 (V)
        current_threshold: 电流标准差阈值 (A)，判断是否为恒流
        min_duration: 最小恒流段持续时间 (s)
    
    返回:
        t_cc: 恒流段时间序列 (s)，重新从0开始
        i_cc: 恒流段电流序列 (A)
        v_cc: 恒流段电压序列 (V)
        info: 包含恒流段统计信息的字典
    """
    # 计算电流绝对值（放电电流可能为负）
    i_abs = np.abs(i_meas)
    
    # 使用滑动窗口检测恒流段
    window_size = 50  # 滑动窗口大小
    if len(i_abs) < window_size:
        warnings.warn(f"数据点太少（{len(i_abs)}），无法可靠检测恒流段")
        window_size = max(10, len(i_abs) // 10)
    
    # 计算滑动标准差
    i_std = np.array([
        np.std(i_abs[max(0, i-window_size//2):min(len(i_abs), i+window_size//2)])
        for i in range(len(i_abs))
    ])
    
    # 找到恒流段：标准差小于阈值
    is_constant = i_std < current_threshold
    
    # 找到最长的连续恒流段
    segments = []
    start = None
    for i, is_cc in enumerate(is_constant):
        if is_cc and start is None:
            start = i
        elif not is_cc and start is not None:
            if t[i-1] - t[start] >= min_duration:
                segments.append((start, i-1))
            start = None
    # 处理最后一个段
    if start is not None and t[-1] - t[start] >= min_duration:
        segments.append((start, len(t)-1))
    
    if not segments:
        # 如果没有找到恒流段，尝试更宽松的标准
        warnings.warn(f"未找到满足条件的恒流段，尝试使用更宽松的阈值")
        current_threshold *= 2
        is_constant = i_std < current_threshold
        segments = []
        start = None
        for i, is_cc in enumerate(is_constant):
            if is_cc and start is None:
                start = i
            elif not is_cc and start is not None:
                if t[i-1] - t[start] >= min_duration:
                    segments.append((start, i-1))
                start = None
        if start is not None and t[-1] - t[start] >= min_duration:
            segments.append((start, len(t)-1))
    
    if not segments:
        raise ValueError(
            f"无法找到恒流段。电流标准差范围：{i_std.min():.4f} - {i_std.max():.4f} A，"
            f"阈值：{current_threshold} A"
        )
    
    # 选择最长的段
    longest_segment = max(segments, key=lambda seg: seg[1] - seg[0])
    start_idx, end_idx = longest_segment
    
    # 提取恒流段数据
    t_cc = t[start_idx:end_idx+1]
    i_cc = i_meas[start_idx:end_idx+1]
    v_cc = v[start_idx:end_idx+1]
    
    # 时间序列重新从0开始
    t_cc = t_cc - t_cc[0]
    
    # 统计信息
    info = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "duration": t_cc[-1],
        "num_points": len(t_cc),
        "mean_current": np.mean(np.abs(i_cc)),
        "std_current": np.std(np.abs(i_cc)),
        "voltage_range": (v_cc.min(), v_cc.max()),
        "num_segments_found": len(segments)
    }
    
    return t_cc, i_cc, v_cc, info


def load_discharge_cc_segment(
    mat_path: str,
    n: int,
    current_threshold: float = 0.05,
    min_duration: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    一站式函数：加载第 n 次放电并提取恒流段
    
    参数:
        mat_path: .mat 文件路径
        n: 放电循环序号（从 1 开始）
        current_threshold: 电流标准差阈值 (A)
        min_duration: 最小恒流段持续时间 (s)
    
    返回:
        t_cc: 恒流段时间序列 (s)
        i_cc: 恒流段电流序列 (A)
        v_cc: 恒流段电压序列 (V)
        info: 包含原始容量和恒流段统计信息的字典
    """
    # 加载数据
    cycles = load_b0005_cycles(mat_path)
    t, i_meas, v, cap_ah = get_nth_discharge(cycles, n)
    
    # 提取恒流段
    t_cc, i_cc, v_cc, cc_info = extract_constant_current_segment(
        t, i_meas, v, current_threshold, min_duration
    )
    
    # 合并信息
    info = {
        "cycle_number": n,
        "total_capacity_ah": cap_ah,
        **cc_info
    }
    
    return t_cc, i_cc, v_cc, info
