"""
OCV-SOC 特性拟合模块
功能：
1. 从数据中提取静置段（电流接近0的片段）
2. 计算静置段的平均电压和对应的 SOC
3. 拟合 OCV-SOC 曲线
4. 提供 OCV 插值函数
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from typing import Tuple, List, Optional, Callable, Dict
import warnings


def calculate_soc(
    t: np.ndarray,
    i: np.ndarray,
    capacity_ah: float,
    initial_soc: float = 1.0
) -> np.ndarray:
    """
    根据电流积分计算 SOC（State of Charge）
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)，放电为负，充电为正
        capacity_ah: 电池额定容量 (Ah)
        initial_soc: 初始 SOC（0-1之间），默认为 1.0（满电）
    
    返回:
        soc: SOC 序列（0-1之间）
    """
    # 使用梯形法则计算累积容量 (Ah)
    capacity_consumed = np.zeros(len(t))
    for i_idx in range(1, len(t)):
        dt = t[i_idx] - t[i_idx-1]  # 时间间隔 (s)
        # 注意：放电电流为负，所以 -i 表示消耗的容量
        capacity_consumed[i_idx] = capacity_consumed[i_idx-1] - (i[i_idx] + i[i_idx-1]) / 2 * dt / 3600
    
    # 计算 SOC
    soc = initial_soc - capacity_consumed / capacity_ah
    
    # 限制在 [0, 1] 范围内
    soc = np.clip(soc, 0, 1)
    
    return soc


def extract_rest_segments(
    t: np.ndarray,
    i: np.ndarray,
    v: np.ndarray,
    soc: np.ndarray,
    current_threshold: float = 0.01,
    min_duration: float = 300.0,
    method: str = 'end'
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    从数据中提取静置段（电流接近0的片段）
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)
        v: 电压序列 (V)
        soc: SOC 序列（0-1之间）
        current_threshold: 电流阈值 (A)，小于此值认为是静置
        min_duration: 最小静置时间 (s)
        method: SOC 取值方法
            - 'end': 取静置段结束时的 SOC（默认）
            - 'mean': 取静置段平均 SOC
            - 'middle': 取静置段中间时刻的 SOC
    
    返回:
        rest_soc: 静置段对应的 SOC 数组
        rest_v: 静置段对应的平均电压数组
        segments_info: 每个静置段的详细信息列表
    """
    # 找到电流小于阈值的点
    is_rest = np.abs(i) < current_threshold
    
    # 找到所有连续的静置段
    segments = []
    start = None
    
    for idx in range(len(is_rest)):
        if is_rest[idx] and start is None:
            start = idx
        elif not is_rest[idx] and start is not None:
            # 静置段结束
            if t[idx-1] - t[start] >= min_duration:
                segments.append((start, idx-1))
            start = None
    
    # 处理最后一个段
    if start is not None and t[-1] - t[start] >= min_duration:
        segments.append((start, len(t)-1))
    
    if not segments:
        warnings.warn(
            f"未找到满足条件的静置段（电流阈值: {current_threshold} A，"
            f"最小持续时间: {min_duration} s）"
        )
        return np.array([]), np.array([]), []
    
    # 提取每个静置段的特征
    rest_soc_list = []
    rest_v_list = []
    segments_info = []
    
    for start_idx, end_idx in segments:
        # 提取该段数据
        t_seg = t[start_idx:end_idx+1]
        i_seg = i[start_idx:end_idx+1]
        v_seg = v[start_idx:end_idx+1]
        soc_seg = soc[start_idx:end_idx+1]
        
        # 计算平均电压
        v_mean = np.mean(v_seg)
        
        # 根据方法选择 SOC
        if method == 'end':
            soc_value = soc_seg[-1]
        elif method == 'mean':
            soc_value = np.mean(soc_seg)
        elif method == 'middle':
            soc_value = soc_seg[len(soc_seg)//2]
        else:
            raise ValueError(f"未知的 SOC 取值方法: {method}")
        
        rest_soc_list.append(soc_value)
        rest_v_list.append(v_mean)
        
        # 保存段信息
        segments_info.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration': t_seg[-1] - t_seg[0],
            'num_points': len(t_seg),
            'soc': soc_value,
            'v_mean': v_mean,
            'v_std': np.std(v_seg),
            'i_mean': np.mean(np.abs(i_seg)),
            'i_max': np.max(np.abs(i_seg))
        })
    
    return np.array(rest_soc_list), np.array(rest_v_list), segments_info


def fit_ocv_curve(
    rest_soc: np.ndarray,
    rest_v: np.ndarray,
    method: str = 'linear',
    smoothing: Optional[float] = None,
    extrapolate: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    """
    拟合 OCV-SOC 曲线
    
    参数:
        rest_soc: 静置段 SOC 数组（0-1之间）
        rest_v: 静置段电压数组 (V)
        method: 插值方法
            - 'linear': 线性插值（默认，稳定可靠）
            - 'cubic': 三次样条插值（平滑但可能振荡）
            - 'spline': 可调平滑参数的样条拟合
        smoothing: 样条拟合的平滑参数（仅当 method='spline' 时有效）
            - None: 自动选择
            - 0: 精确通过所有点
            - >0: 更大的值产生更平滑的曲线
        extrapolate: 是否允许外推
            - True: 超出数据范围使用外推（可能不准确）
            - False: 超出范围返回边界值
    
    返回:
        ocv_func: OCV 插值函数，输入 SOC（标量或数组），输出电压 (V)
    """
    if len(rest_soc) == 0:
        raise ValueError("没有静置段数据，无法拟合 OCV 曲线")
    
    if len(rest_soc) != len(rest_v):
        raise ValueError(f"SOC 和电压数据长度不匹配: {len(rest_soc)} vs {len(rest_v)}")
    
    # 按 SOC 排序
    sorted_idx = np.argsort(rest_soc)
    soc_sorted = rest_soc[sorted_idx]
    v_sorted = rest_v[sorted_idx]
    
    # 去除重复的 SOC 点（取平均电压）
    unique_soc = []
    unique_v = []
    i = 0
    while i < len(soc_sorted):
        current_soc = soc_sorted[i]
        # 找到所有相同 SOC 的点
        same_soc_mask = np.abs(soc_sorted - current_soc) < 1e-6
        # 取平均电压
        mean_v = np.mean(v_sorted[same_soc_mask])
        unique_soc.append(current_soc)
        unique_v.append(mean_v)
        # 跳过相同的点
        i += np.sum(same_soc_mask)
    
    soc_sorted = np.array(unique_soc)
    v_sorted = np.array(unique_v)
    
    if len(soc_sorted) < 2:
        raise ValueError(f"拟合需要至少2个不同的 SOC 点，当前只有 {len(soc_sorted)} 个")
    
    # 根据方法选择插值方式
    if method == 'linear':
        fill_value = 'extrapolate' if extrapolate else (v_sorted[0], v_sorted[-1])
        ocv_func = interp1d(
            soc_sorted, v_sorted,
            kind='linear',
            bounds_error=False,
            fill_value=fill_value
        )
    
    elif method == 'cubic':
        if len(soc_sorted) < 4:
            warnings.warn(
                f"三次样条插值需要至少4个点，当前只有 {len(soc_sorted)} 个，"
                f"将使用线性插值"
            )
            fill_value = 'extrapolate' if extrapolate else (v_sorted[0], v_sorted[-1])
            ocv_func = interp1d(
                soc_sorted, v_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=fill_value
            )
        else:
            fill_value = 'extrapolate' if extrapolate else (v_sorted[0], v_sorted[-1])
            ocv_func = interp1d(
                soc_sorted, v_sorted,
                kind='cubic',
                bounds_error=False,
                fill_value=fill_value
            )
    
    elif method == 'spline':
        if len(soc_sorted) < 4:
            warnings.warn(
                f"样条拟合需要至少4个点，当前只有 {len(soc_sorted)} 个，"
                f"将使用线性插值"
            )
            fill_value = 'extrapolate' if extrapolate else (v_sorted[0], v_sorted[-1])
            ocv_func = interp1d(
                soc_sorted, v_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=fill_value
            )
        else:
            # 使用 UnivariateSpline，可以控制平滑度
            s = smoothing if smoothing is not None else len(soc_sorted)
            spline = UnivariateSpline(soc_sorted, v_sorted, s=s, ext='extrapolate' if extrapolate else 'const')
            ocv_func = lambda soc: spline(soc)
    
    else:
        raise ValueError(f"未知的插值方法: {method}，支持的方法: 'linear', 'cubic', 'spline'")
    
    return ocv_func


def fit_ocv_from_full_cycle(
    t: np.ndarray,
    i: np.ndarray,
    v: np.ndarray,
    capacity_ah: float,
    initial_soc: float = 1.0,
    current_threshold: float = 0.01,
    min_duration: float = 300.0,
    soc_method: str = 'end',
    fit_method: str = 'linear',
    smoothing: Optional[float] = None,
    extrapolate: bool = True
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict]:
    """
    从完整循环数据中拟合 OCV-SOC 曲线（一站式函数）
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)
        v: 电压序列 (V)
        capacity_ah: 电池额定容量 (Ah)
        initial_soc: 初始 SOC（0-1之间）
        current_threshold: 静置判断的电流阈值 (A)
        min_duration: 静置段最小持续时间 (s)
        soc_method: SOC 取值方法 ('end', 'mean', 'middle')
        fit_method: 拟合方法 ('linear', 'cubic', 'spline')
        smoothing: 样条拟合的平滑参数
        extrapolate: 是否允许外推
    
    返回:
        ocv_func: OCV 插值函数
        info: 包含拟合信息的字典
            - rest_soc: 静置段 SOC 数组
            - rest_v: 静置段电压数组
            - num_segments: 静置段数量
            - segments_info: 每个静置段的详细信息
            - soc_range: SOC 范围
            - voltage_range: 电压范围
    """
    # 计算 SOC
    soc = calculate_soc(t, i, capacity_ah, initial_soc)
    
    # 提取静置段
    rest_soc, rest_v, segments_info = extract_rest_segments(
        t, i, v, soc,
        current_threshold=current_threshold,
        min_duration=min_duration,
        method=soc_method
    )
    
    if len(rest_soc) == 0:
        raise ValueError("未找到静置段，无法拟合 OCV 曲线")
    
    # 拟合 OCV 曲线
    ocv_func = fit_ocv_curve(
        rest_soc, rest_v,
        method=fit_method,
        smoothing=smoothing,
        extrapolate=extrapolate
    )
    
    # 构建返回信息
    info = {
        'rest_soc': rest_soc,
        'rest_v': rest_v,
        'num_segments': len(rest_soc),
        'segments_info': segments_info,
        'soc_range': (rest_soc.min(), rest_soc.max()),
        'voltage_range': (rest_v.min(), rest_v.max()),
        'fit_method': fit_method,
        'soc_method': soc_method,
        'current_threshold': current_threshold,
        'min_duration': min_duration
    }
    
    return ocv_func, info
