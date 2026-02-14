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
    auto_download: bool = False,
):
    """
    加载 B0005.mat 文件，返回所有 cycle 结构体数组
    
    支持两种格式：
    - MATLAB v7.2 及更早版本（使用 scipy.io.loadmat）
    - MATLAB v7.3 (HDF5格式，使用 h5py)
    
    参数:
        mat_path: .mat 文件路径
        key: MATLAB 文件中的顶层变量名
        auto_download: 如果文件不存在或损坏，是否尝试自动下载
    
    返回:
        cycles: numpy 数组，每个元素是一个 cycle 结构体
    """
    import os
    
    # 如果文件不存在且允许自动下载
    if not os.path.exists(mat_path) and auto_download:
        print(f"[INFO] Data file not found, attempting to download...", flush=True)
        _download_b0005_data(mat_path)
    
    # 打印文件信息用于调试
    if os.path.exists(mat_path):
        file_size = os.path.getsize(mat_path)
        print(f"[DEBUG] File exists: {mat_path}", flush=True)
        print(f"[DEBUG] File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)", flush=True)
        
        # 检查文件大小是否合理（B0005.mat 应该在 15MB 左右）
        expected_min_size = 14 * 1024 * 1024  # 14 MB
        expected_max_size = 20 * 1024 * 1024  # 20 MB
        if file_size < expected_min_size:
            print(f"[WARNING] File size ({file_size/1024/1024:.2f} MB) is smaller than expected (>14 MB)", flush=True)
            print(f"[WARNING] The file might be incomplete or corrupted!", flush=True)
        
        # 读取文件头部信息
        try:
            with open(mat_path, 'rb') as f:
                header = f.read(200)
                # 打印前32字节的十六进制
                hex_header = ' '.join(f'{b:02x}' for b in header[:32])
                print(f"[DEBUG] File header (hex): {hex_header}", flush=True)
                # 尝试解析为文本
                try:
                    text_header = header[:116].decode('latin-1', errors='ignore')
                    print(f"[DEBUG] File header (text): {repr(text_header[:80])}", flush=True)
                    
                    # 检测 Git LFS 指针文件
                    if text_header.startswith('version https://git-lfs.github.com'):
                        raise ValueError(
                            f"ERROR: {mat_path} is a Git LFS pointer file, not the actual data!\n"
                            f"Solutions:\n"
                            f"  1. Run 'git lfs pull' to download the actual file\n"
                            f"  2. Or manually upload B0005.mat to the platform and mount it to /data/\n"
                            f"  3. Or set ECM_DATA_PATH to point to the correct data file location"
                        )
                except:
                    pass
        except ValueError:
            raise  # 重新抛出 Git LFS 错误
        except Exception as e:
            print(f"[DEBUG] Cannot read file header: {e}", flush=True)
    else:
        raise FileNotFoundError(f"Data file not found: {mat_path}")
    
    try:
        # 首先尝试使用 scipy.io.loadmat（适用于 v7.2 及更早版本）
        mat = loadmat(mat_path)
        battery = mat[key][0, 0]          # MATLAB struct
        cycles = battery["cycle"][0]      # 1D array, 每个元素是一个 cycle struct
        print(f"[INFO] Successfully loaded using scipy.io.loadmat", flush=True)
        return cycles
    except ValueError as e:
        if "Unknown mat file type" in str(e) or "version" in str(e):
            # 如果是版本问题，尝试使用 h5py 读取 MATLAB v7.3 格式
            print(f"[INFO] Detected MATLAB v7.3 format, using h5py to load...", flush=True)
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "MATLAB v7.3 format detected but h5py is not installed. "
                    "Please install it: pip install h5py"
                ) from e
            
            return _load_b0005_cycles_hdf5(mat_path, key)
        else:
            # 其他错误直接抛出
            raise
    except Exception as e:
        # 捕获 zlib 解压错误（文件损坏）
        import zlib
        if isinstance(e, zlib.error) or "decompressing data" in str(e):
            raise ValueError(
                f"ERROR: Failed to decompress {mat_path} - the file appears to be corrupted or incomplete!\n"
                f"File size: {os.path.getsize(mat_path)/1024/1024:.2f} MB (expected: ~15 MB)\n"
                f"Error: {e}\n\n"
                f"=== SOLUTION FOR BOHRIUM PLATFORM ===\n"
                f"The file in /appcode/ was corrupted during folder upload.\n"
                f"Please use external data mounting instead:\n\n"
                f"1. Go to Bohrium 'Data Management' or 'Storage' section\n"
                f"2. Upload B0005.mat (15.22 MB) separately as a dataset\n"
                f"3. In App settings, mount the dataset to /data/ or /share/\n"
                f"4. The code will auto-detect the file in mounted directories\n\n"
                f"Alternative: Set environment variable ECM_DATA_PATH=/data/B0005.mat\n"
                f"Download original file: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
            ) from e
        else:
            # 其他未知错误
            raise


def _load_b0005_cycles_hdf5(mat_path: str, key: str = "B0005"):
    """
    使用 h5py 加载 MATLAB v7.3 (HDF5) 格式的 B0005.mat 文件
    
    参数:
        mat_path: .mat 文件路径
        key: MATLAB 文件中的顶层变量名
    
    返回:
        cycles: list，每个元素是一个包含 cycle 数据的字典
    """
    import h5py
    
    with h5py.File(mat_path, 'r') as f:
        battery = f[key]
        cycle_refs = battery['cycle'][0]  # 获取 cycle 引用数组
        
        cycles = []
        for ref in cycle_refs:
            cycle_obj = f[ref]
            
            # 提取 cycle 数据
            cycle_data = {}
            
            # type: 'discharge' 或 'charge'
            if 'type' in cycle_obj:
                type_data = cycle_obj['type'][:]
                cycle_data['type'] = ''.join(chr(c) for c in type_data.flatten() if c != 0)
            
            # data: 包含时间、电压、电流等
            if 'data' in cycle_obj:
                data_ref = cycle_obj['data'][0, 0]
                data_obj = f[data_ref]
                
                cycle_data['data'] = {}
                # 读取常见字段
                for field in ['Time', 'Voltage_measured', 'Current_measured', 
                             'Temperature_measured', 'Voltage_load', 'Current_load',
                             'Voltage_battery', 'Current_battery', 'Capacity']:
                    if field in data_obj:
                        # HDF5 存储是转置的，需要转回来
                        value = data_obj[field][:]
                        # Capacity 可能是标量，其他是向量
                        if value.size == 1:
                            cycle_data['data'][field] = value.flatten()[0]
                        else:
                            cycle_data['data'][field] = value.T.flatten()
            
            # 转换为类似 scipy.loadmat 的结构
            class CycleStruct:
                pass
            
            cycle_struct = CycleStruct()
            cycle_struct.type = np.array([[cycle_data.get('type', '')]], dtype=object)
            
            if 'data' in cycle_data:
                class DataStruct:
                    pass
                data_struct = DataStruct()
                for field, value in cycle_data['data'].items():
                    setattr(data_struct, field, value)
                cycle_struct.data = np.array([[data_struct]], dtype=object)
            
            cycles.append(cycle_struct)
        
        return np.array(cycles, dtype=object)


def _download_b0005_data(save_path: str):
    """
    从备用源下载 B0005.mat 数据文件
    
    参数:
        save_path: 保存路径
    """
    import urllib.request
    import os
    
    # NASA 官方数据集通常需要手动下载，这里提供一个备用方案
    print("[INFO] Downloading B0005.mat from backup source...", flush=True)
    print("[WARNING] Auto-download is not fully implemented yet.", flush=True)
    print("[INFO] Please manually download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository", flush=True)
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 这里可以添加实际的下载逻辑，例如从 S3、OSS 或其他可访问的备用源
    # url = "https://your-backup-source/B0005.mat"
    # urllib.request.urlretrieve(url, save_path)
    
    raise FileNotFoundError(
        f"Auto-download is not available. Please manually download B0005.mat and place it at: {save_path}"
    )


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
        # 兼容两种数据结构：
        # 1. scipy.loadmat: c["type"] 是个数组，比如 array(['discharge'], dtype='<U9')
        # 2. h5py: c.type 是个数组
        try:
            cycle_type = c["type"][0]
        except (TypeError, KeyError):
            # 如果是 h5py 读取的自定义对象
            cycle_type = c.type[0, 0] if hasattr(c, 'type') else None
        
        if cycle_type == "discharge" or (isinstance(cycle_type, str) and "discharge" in cycle_type.lower()):
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
    
    # 兼容两种数据结构
    try:
        # scipy.loadmat 格式
        d = c["data"][0, 0]               # 真正的数据 struct
    except (TypeError, KeyError):
        # h5py 格式
        d = c.data[0, 0]
    
    # 提取数据（兼容两种访问方式）
    def get_field(obj, field_name):
        """兼容字典访问和属性访问"""
        try:
            return obj[field_name]
        except (TypeError, KeyError):
            return getattr(obj, field_name)
    
    t = np.array(get_field(d, "Time")).flatten().astype(float)
    v = np.array(get_field(d, "Voltage_measured")).flatten().astype(float)
    i_meas = np.array(get_field(d, "Current_measured")).flatten().astype(float)
    
    # Capacity 可能是标量或数组
    cap = get_field(d, "Capacity")
    cap_ah = float(np.array(cap).squeeze())  # 标量 Ahr
    
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
