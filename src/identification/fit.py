"""
参数辨识模块
功能：
1. 定义残差函数
2. 使用最小二乘法优化ECM参数
3. 计算拟合指标
4. 可视化拟合结果
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution
from typing import Tuple, Callable, Dict, Optional
import warnings

import sys
from pathlib import Path

from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage
from src.ecm.metrics import calculate_all_metrics, print_metrics


class ECMParameterIdentification:
    """
    ECM 参数辨识类
    """
    
    def __init__(
        self,
        t: np.ndarray,
        i: np.ndarray,
        v_measured: np.ndarray,
        soc: np.ndarray,
        ocv_func: Callable[[np.ndarray], np.ndarray]
    ):
        """
        初始化参数辨识器
        
        参数:
            t: 时间序列 (s)
            i: 电流序列 (A)
            v_measured: 实测电压序列 (V)
            soc: SOC 序列
            ocv_func: OCV 函数
        """
        self.t = t
        self.i = i
        self.v_measured = v_measured
        self.soc = soc
        self.ocv_func = ocv_func
        
        # 辨识结果
        self.params_identified = None
        self.v_pred = None
        self.residuals = None
        self.metrics = None
        self.optimization_result = None
    
    def residual_function(self, theta: np.ndarray) -> np.ndarray:
        """
        残差函数（用于最小二乘优化）
        
        参数:
            theta: 参数数组 [R0, R1, C1, R2, C2]
        
        返回:
            residuals: 残差向量 (V_pred - V_measured)
        """
        try:
            # 从数组创建参数对象
            params = ECM2RCParams.from_array(theta)
            
            # 仿真电压
            v_pred = simulate_voltage(
                self.t, self.i, self.soc, params, self.ocv_func
            )
            
            # 返回残差
            return v_pred - self.v_measured
            
        except Exception as e:
            # 如果参数不合理，返回很大的残差
            warnings.warn(f"参数不合理: {e}")
            return np.ones_like(self.v_measured) * 1e6
    
    def fit_least_squares(
        self,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        method: str = 'trf',
        verbose: int = 1
    ) -> ECM2RCParams:
        """
        使用最小二乘法辨识参数
        
        参数:
            x0: 初始参数猜测 [R0, R1, C1, R2, C2]
            bounds: 参数边界 (lower, upper)
            method: 优化方法 ('trf', 'dogbox', 'lm')
            verbose: 详细程度 (0=silent, 1=minimal, 2=detailed)
        
        返回:
            params: 辨识得到的参数
        """
        # 默认初始值
        if x0 is None:
            x0 = np.array([0.01, 0.02, 1000.0, 0.05, 5000.0])
        
        # 默认边界
        if bounds is None:
            lower = np.array([1e-4, 1e-4, 10.0, 1e-4, 10.0])
            upper = np.array([1.0, 1.0, 1e6, 1.0, 1e6])
            bounds = (lower, upper)
        
        print("\n" + "="*60)
        print("开始参数辨识（最小二乘法）")
        print("="*60)
        print(f"初始参数: {x0}")
        print(f"参数下界: {bounds[0]}")
        print(f"参数上界: {bounds[1]}")
        
        # 运行优化
        result = least_squares(
            self.residual_function,
            x0,
            bounds=bounds,
            method=method,
            verbose=verbose,
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8
        )
        
        self.optimization_result = result
        
        # 提取最优参数
        theta_opt = result.x
        self.params_identified = ECM2RCParams.from_array(theta_opt)
        
        # 计算最优电压
        self.v_pred = simulate_voltage(
            self.t, self.i, self.soc, self.params_identified, self.ocv_func
        )
        
        # 计算残差
        self.residuals = self.v_pred - self.v_measured
        
        # 计算评估指标
        self.metrics = calculate_all_metrics(self.v_measured, self.v_pred)
        
        # 打印结果
        print("\n" + "="*60)
        print("优化结果")
        print("="*60)
        print(f"优化状态: {result.message}")
        print(f"函数评估次数: {result.nfev}")
        print(f"成本函数值: {result.cost:.6e}")
        print(f"优化是否成功: {result.success}")
        
        print(f"\n辨识得到的参数:")
        print(self.params_identified)
        
        print_metrics(self.metrics, "拟合指标")
        
        return self.params_identified
    
    def fit_global(
        self,
        bounds: Optional[list] = None,
        maxiter: int = 100,
        popsize: int = 15,
        seed: Optional[int] = None
    ) -> ECM2RCParams:
        """
        使用全局优化（差分进化）辨识参数
        
        参数:
            bounds: 参数边界列表 [(min, max), ...]
            maxiter: 最大迭代次数
            popsize: 种群大小
            seed: 随机种子
        
        返回:
            params: 辨识得到的参数
        """
        # 默认边界
        if bounds is None:
            bounds = [
                (1e-4, 1.0),    # R0
                (1e-4, 1.0),    # R1
                (10.0, 1e6),    # C1
                (1e-4, 1.0),    # R2
                (10.0, 1e6)     # C2
            ]
        
        print("\n" + "="*60)
        print("开始参数辨识（全局优化）")
        print("="*60)
        print(f"参数边界: {bounds}")
        print(f"最大迭代次数: {maxiter}")
        print(f"种群大小: {popsize}")
        
        # 定义目标函数（最小化RMSE）
        def objective(theta):
            residuals = self.residual_function(theta)
            return np.sum(residuals**2)
        
        # 运行全局优化
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            disp=True,
            polish=True,  # 最后用局部优化抛光
            atol=1e-8,
            tol=1e-8
        )
        
        self.optimization_result = result
        
        # 提取最优参数
        theta_opt = result.x
        self.params_identified = ECM2RCParams.from_array(theta_opt)
        
        # 计算最优电压
        self.v_pred = simulate_voltage(
            self.t, self.i, self.soc, self.params_identified, self.ocv_func
        )
        
        # 计算残差
        self.residuals = self.v_pred - self.v_measured
        
        # 计算评估指标
        self.metrics = calculate_all_metrics(self.v_measured, self.v_pred)
        
        # 打印结果
        print("\n" + "="*60)
        print("优化结果")
        print("="*60)
        print(f"优化状态: {result.message}")
        print(f"函数评估次数: {result.nfev}")
        print(f"最优目标函数值: {result.fun:.6e}")
        print(f"优化是否成功: {result.success}")
        
        print(f"\n辨识得到的参数:")
        print(self.params_identified)
        
        print_metrics(self.metrics, "拟合指标")
        
        return self.params_identified
    
    def get_results(self) -> Dict:
        """
        获取辨识结果
        
        返回:
            results: 包含所有结果的字典
        """
        if self.params_identified is None:
            raise ValueError("尚未进行参数辨识，请先调用 fit_least_squares 或 fit_global")
        
        return {
            'params': self.params_identified,
            'v_pred': self.v_pred,
            'residuals': self.residuals,
            'metrics': self.metrics,
            'optimization_result': self.optimization_result
        }


def fit_ecm_params(
    t: np.ndarray,
    i: np.ndarray,
    v_measured: np.ndarray,
    soc: np.ndarray,
    ocv_func: Callable[[np.ndarray], np.ndarray],
    method: str = 'least_squares',
    x0: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    verbose: int = 1
) -> Tuple[ECM2RCParams, Dict]:
    """
    ECM 参数辨识的便捷函数
    
    参数:
        t: 时间序列 (s)
        i: 电流序列 (A)
        v_measured: 实测电压序列 (V)
        soc: SOC 序列
        ocv_func: OCV 函数
        method: 优化方法 ('least_squares' 或 'global')
        x0: 初始参数（仅用于least_squares）
        bounds: 参数边界
        verbose: 详细程度
    
    返回:
        params: 辨识得到的参数
        results: 辨识结果字典
    """
    identifier = ECMParameterIdentification(t, i, v_measured, soc, ocv_func)
    
    if method == 'least_squares':
        params = identifier.fit_least_squares(x0=x0, bounds=bounds, verbose=verbose)
    elif method == 'global':
        # 转换边界格式
        if bounds is not None:
            bounds_list = [(bounds[0][i], bounds[1][i]) for i in range(5)]
        else:
            bounds_list = None
        params = identifier.fit_global(bounds=bounds_list)
    else:
        raise ValueError(f"未知的方法: {method}，支持 'least_squares' 或 'global'")
    
    results = identifier.get_results()
    
    return params, results


def plot_fit_results(
    t: np.ndarray,
    v_measured: np.ndarray,
    v_pred: np.ndarray,
    residuals: np.ndarray,
    params: ECM2RCParams,
    metrics: Dict,
    i: np.ndarray = None
) -> plt.Figure:
    """
    Plot parameter identification results
    
    Parameters:
        t: Time series
        v_measured: Measured voltage
        v_pred: Predicted voltage
        residuals: Residuals
        params: Identified parameters
        metrics: Fitting metrics
        i: Current series (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    if i is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Voltage comparison
    ax1 = axes[0]
    ax1.plot(t, v_measured, 'k-', label='Measured', linewidth=1.5, alpha=0.8)
    ax1.plot(t, v_pred, 'r--', label='Predicted', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'Voltage Fitting (RMSE={metrics["RMSE"]:.6f} V, R2={metrics["R2"]:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Residuals
    ax2 = axes[1]
    ax2.plot(t, residuals, 'b-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax2.fill_between(t, residuals, 0, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual (V)')
    ax2.set_title(f'Residual Analysis (MAE={metrics["MAE"]:.6f} V, Max={metrics["MaxAbsError"]:.6f} V)')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Current (if provided)
    if i is not None:
        ax3 = axes[2]
        ax3.plot(t, i, 'g-', linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Current (A)')
        ax3.set_title('Current Profile')
        ax3.grid(True, alpha=0.3)
    
    # Add parameter info text box
    param_text = f"Identified Parameters:\n"
    param_text += f"R0 = {params.R0:.6f} Ohm\n"
    param_text += f"R1 = {params.R1:.6f} Ohm\n"
    param_text += f"C1 = {params.C1:.2f} F\n"
    param_text += f"R2 = {params.R2:.6f} Ohm\n"
    param_text += f"C2 = {params.C2:.2f} F\n"
    param_text += f"tau1 = {params.R1*params.C1:.2f} s\n"
    param_text += f"tau2 = {params.R2*params.C2:.2f} s"
    
    axes[0].text(0.02, 0.98, param_text, transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    
    plt.tight_layout()
    return fig
