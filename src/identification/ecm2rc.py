"""
二阶 RC 等效电路模型（ECM）模块
功能：
1. 定义二阶 ECM 参数结构
2. 参数有效性检查
3. 电压仿真函数
4. 极化电压计算
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import warnings


@dataclass
class ECM2RCParams:
    """
    二阶 RC 等效电路模型参数
    
    模型结构：
        OCV(SOC) - R0 - [R1||C1] - [R2||C2]
    
    参数说明：
        R0: 欧姆内阻 (Ω)
        R1: 第一个 RC 支路的电阻 (Ω)
        C1: 第一个 RC 支路的电容 (F)
        R2: 第二个 RC 支路的电阻 (Ω)
        C2: 第二个 RC 支路的电容 (F)
    
    物理意义：
        - R0: 瞬时响应，欧姆压降
        - R1, C1: 快速极化过程（电化学极化）
        - R2, C2: 慢速极化过程（浓差极化）
    """
    R0: float  # 欧姆内阻 (Ω)
    R1: float  # RC支路1电阻 (Ω)
    C1: float  # RC支路1电容 (F)
    R2: float  # RC支路2电阻 (Ω)
    C2: float  # RC支路2电容 (F)
    
    def __post_init__(self):
        """参数验证"""
        if not check_params_positive(self):
            raise ValueError(
                f"所有参数必须为正数: R0={self.R0}, R1={self.R1}, "
                f"C1={self.C1}, R2={self.R2}, C2={self.C2}"
            )
    
    def to_array(self) -> np.ndarray:
        """转换为数组形式，便于优化算法使用"""
        return np.array([self.R0, self.R1, self.C1, self.R2, self.C2])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ECM2RCParams':
        """从数组创建参数对象"""
        if len(arr) != 5:
            raise ValueError(f"参数数组长度必须为5，当前为{len(arr)}")
        return cls(R0=arr[0], R1=arr[1], C1=arr[2], R2=arr[3], C2=arr[4])
    
    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"ECM2RCParams(\n"
            f"  R0 = {self.R0:.6f} Ω\n"
            f"  R1 = {self.R1:.6f} Ω\n"
            f"  C1 = {self.C1:.6f} F  (τ1 = {self.R1*self.C1:.3f} s)\n"
            f"  R2 = {self.R2:.6f} Ω\n"
            f"  C2 = {self.C2:.6f} F  (τ2 = {self.R2*self.C2:.3f} s)\n"
            f")"
        )
    
    def get_time_constants(self) -> Tuple[float, float]:
        """获取两个时间常数"""
        tau1 = self.R1 * self.C1
        tau2 = self.R2 * self.C2
        return tau1, tau2


def check_params_positive(
    params: ECM2RCParams,
    min_value: float = 1e-10
) -> bool:
    """
    检查参数是否为正数
    
    参数:
        params: ECM2RC 参数对象
        min_value: 最小允许值，避免数值问题
    
    返回:
        bool: 所有参数都为正数返回 True，否则返回 False
    """
    return all([
        params.R0 > min_value,
        params.R1 > min_value,
        params.C1 > min_value,
        params.R2 > min_value,
        params.C2 > min_value
    ])


def compute_polarization_voltages(
    t: np.ndarray,
    i: np.ndarray,
    params: ECM2RCParams,
    V1_init: float = 0.0,
    V2_init: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个 RC 支路的极化电压
    
    使用离散化迭代公式：
        a_k = exp(-dt / (R_k * C_k))
        b_k = R_k * (1 - a_k)
        V_k[n+1] = a_k * V_k[n] + b_k * I[n]
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)，放电为负，充电为正
        params: ECM2RC 参数
        V1_init: RC支路1的初始极化电压 (V)
        V2_init: RC支路2的初始极化电压 (V)
    
    返回:
        V1: RC支路1的极化电压序列 (V)
        V2: RC支路2的极化电压序列 (V)
    """
    n = len(t)
    V1 = np.zeros(n)
    V2 = np.zeros(n)
    
    # 初始值
    V1[0] = V1_init
    V2[0] = V2_init
    
    # 计算时间常数
    tau1 = params.R1 * params.C1
    tau2 = params.R2 * params.C2
    
    # 迭代计算极化电压
    for k in range(n - 1):
        # 时间间隔
        dt = t[k + 1] - t[k]
        
        # 防止时间间隔为负或为零
        if dt <= 0:
            warnings.warn(f"时间间隔异常: dt={dt} at k={k}")
            dt = 1e-6  # 使用一个很小的正值
        
        # RC支路1
        a1 = np.exp(-dt / tau1)
        b1 = params.R1 * (1 - a1)
        V1[k + 1] = a1 * V1[k] + b1 * i[k]
        
        # RC支路2
        a2 = np.exp(-dt / tau2)
        b2 = params.R2 * (1 - a2)
        V2[k + 1] = a2 * V2[k] + b2 * i[k]
    
    return V1, V2


def simulate_voltage(
    t: np.ndarray,
    i: np.ndarray,
    soc: np.ndarray,
    params: ECM2RCParams,
    ocv_func: Callable[[np.ndarray], np.ndarray],
    V1_init: float = 0.0,
    V2_init: float = 0.0
) -> np.ndarray:
    """
    使用二阶 ECM 模型仿真端电压
    
    模型方程：
        V_terminal = OCV(SOC) - I*R0 - V1 - V2
    
    其中：
        - OCV(SOC): 开路电压，由 SOC 决定
        - I*R0: 欧姆压降
        - V1, V2: 两个 RC 支路的极化电压
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)，放电为负，充电为正
        soc: SOC 序列（0-1之间）
        params: ECM2RC 参数
        ocv_func: OCV 函数，输入 SOC，输出电压 (V)
        V1_init: RC支路1的初始极化电压 (V)
        V2_init: RC支路2的初始极化电压 (V)
    
    返回:
        V_pred: 预测的端电压序列 (V)
    """
    # 验证输入
    if len(t) != len(i) or len(t) != len(soc):
        raise ValueError(
            f"输入长度不匹配: t={len(t)}, i={len(i)}, soc={len(soc)}"
        )
    
    # 计算 OCV
    V_ocv = ocv_func(soc)
    
    # 计算极化电压
    V1, V2 = compute_polarization_voltages(t, i, params, V1_init, V2_init)
    
    # 计算端电压
    # 注意：放电时电流为负，所以 -I*R0 实际上减小了电压
    V_pred = V_ocv - i * params.R0 - V1 - V2
    
    return V_pred


def simulate_voltage_with_details(
    t: np.ndarray,
    i: np.ndarray,
    soc: np.ndarray,
    params: ECM2RCParams,
    ocv_func: Callable[[np.ndarray], np.ndarray],
    V1_init: float = 0.0,
    V2_init: float = 0.0
) -> dict:
    """
    使用二阶 ECM 模型仿真端电压，并返回详细信息
    
    参数:
        同 simulate_voltage()
    
    返回:
        result: 包含详细仿真结果的字典
            - V_pred: 预测的端电压 (V)
            - V_ocv: 开路电压 (V)
            - V_ohm: 欧姆压降 (V)
            - V1: RC支路1的极化电压 (V)
            - V2: RC支路2的极化电压 (V)
            - params: 使用的参数
    """
    # 验证输入
    if len(t) != len(i) or len(t) != len(soc):
        raise ValueError(
            f"输入长度不匹配: t={len(t)}, i={len(i)}, soc={len(soc)}"
        )
    
    # 计算 OCV
    V_ocv = ocv_func(soc)
    
    # 计算欧姆压降
    V_ohm = i * params.R0
    
    # 计算极化电压
    V1, V2 = compute_polarization_voltages(t, i, params, V1_init, V2_init)
    
    # 计算端电压
    V_pred = V_ocv - V_ohm - V1 - V2
    
    # 返回详细结果
    result = {
        'V_pred': V_pred,       # 预测的端电压
        'V_ocv': V_ocv,         # 开路电压
        'V_ohm': V_ohm,         # 欧姆压降
        'V1': V1,               # RC支路1极化电压
        'V2': V2,               # RC支路2极化电压
        'params': params        # 使用的参数
    }
    
    return result


def get_initial_params_guess() -> ECM2RCParams:
    """
    获取参数的初始猜测值
    
    基于典型锂离子电池的经验值：
        - R0: 0.01-0.1 Ω
        - R1: 0.01-0.05 Ω, τ1: 1-10 s (快速极化)
        - R2: 0.02-0.1 Ω, τ2: 10-100 s (慢速极化)
    
    返回:
        初始参数猜测
    """
    return ECM2RCParams(
        R0=0.05,      # 50 mΩ
        R1=0.02,      # 20 mΩ
        C1=100.0,     # 100 F, τ1 = 2 s
        R2=0.05,      # 50 mΩ
        C2=1000.0     # 1000 F, τ2 = 50 s
    )


def validate_params_physical(
    params: ECM2RCParams,
    warn: bool = True
) -> Tuple[bool, list]:
    """
    验证参数的物理合理性
    
    参数:
        params: ECM2RC 参数
        warn: 是否输出警告信息
    
    返回:
        is_valid: 参数是否物理合理
        issues: 不合理之处的列表
    """
    issues = []
    
    # 检查电阻范围（典型值：0.001-1 Ω）
    if params.R0 < 0.001 or params.R0 > 1.0:
        issues.append(f"R0={params.R0:.6f} Ω 超出典型范围 [0.001, 1.0]")
    
    if params.R1 < 0.001 or params.R1 > 1.0:
        issues.append(f"R1={params.R1:.6f} Ω 超出典型范围 [0.001, 1.0]")
    
    if params.R2 < 0.001 or params.R2 > 1.0:
        issues.append(f"R2={params.R2:.6f} Ω 超出典型范围 [0.001, 1.0]")
    
    # 检查电容范围（典型值：10-10000 F）
    if params.C1 < 10 or params.C1 > 10000:
        issues.append(f"C1={params.C1:.2f} F 超出典型范围 [10, 10000]")
    
    if params.C2 < 10 or params.C2 > 10000:
        issues.append(f"C2={params.C2:.2f} F 超出典型范围 [10, 10000]")
    
    # 检查时间常数（τ1 应该小于 τ2）
    tau1, tau2 = params.get_time_constants()
    if tau1 >= tau2:
        issues.append(
            f"时间常数顺序异常: τ1={tau1:.3f} s 应小于 τ2={tau2:.3f} s"
        )
    
    # 检查时间常数范围（典型值：τ1: 0.1-10 s, τ2: 10-1000 s）
    if tau1 < 0.1 or tau1 > 100:
        issues.append(f"τ1={tau1:.3f} s 超出典型范围 [0.1, 100]")
    
    if tau2 < 1 or tau2 > 10000:
        issues.append(f"τ2={tau2:.3f} s 超出典型范围 [1, 10000]")
    
    # 输出警告
    if warn and issues:
        warnings.warn(
            f"参数物理合理性检查发现 {len(issues)} 个问题:\n" + 
            "\n".join(f"  - {issue}" for issue in issues)
        )
    
    return len(issues) == 0, issues
