"""
模型评估指标模块
功能：
1. 计算回归模型的各种评估指标
2. 提供统一的评估接口
"""

import numpy as np
from typing import Dict, Optional
import warnings


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Square Error)
    
    公式: RMSE = sqrt(mean((y_true - y_pred)^2))
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        rmse: 均方根误差
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    residuals = y_true - y_pred
    return np.sqrt(np.mean(residuals ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    公式: MAE = mean(|y_true - y_pred|)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        mae: 平均绝对误差
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    residuals = y_true - y_pred
    return np.mean(np.abs(residuals))


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算最大绝对误差 (Maximum Absolute Error)
    
    公式: MaxAE = max(|y_true - y_pred|)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        max_ae: 最大绝对误差
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    residuals = y_true - y_pred
    return np.max(np.abs(residuals))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 (Mean Square Error)
    
    公式: MSE = mean((y_true - y_pred)^2)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        mse: 均方误差
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    residuals = y_true - y_pred
    return np.mean(residuals ** 2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R² Score / Coefficient of Determination)
    
    公式: R² = 1 - SS_res / SS_tot
    其中: SS_res = sum((y_true - y_pred)^2)
         SS_tot = sum((y_true - mean(y_true))^2)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        r2: 决定系数（越接近1越好）
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        warnings.warn("SS_tot为0，无法计算R²")
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    公式: MAPE = mean(|y_true - y_pred| / |y_true|) * 100%
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小常数
    
    返回:
        mape: 平均绝对百分比误差（百分比）
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # 避免除零
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    include_mape: bool = False
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        include_mape: 是否包含MAPE（对于电压值MAPE意义不大）
    
    返回:
        metrics: 包含所有指标的字典
    """
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MaxAbsError': max_abs_error(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    if include_mape:
        metrics['MAPE'] = mape(y_true, y_pred)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: Optional[str] = None):
    """
    格式化打印评估指标
    
    参数:
        metrics: 指标字典
        title: 标题（可选）
    """
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)
    
    for key, value in metrics.items():
        if key == 'R2':
            print(f"  {key:15s}: {value:.6f}")
        elif key == 'MAPE':
            print(f"  {key:15s}: {value:.4f}%")
        else:
            print(f"  {key:15s}: {value:.6f} V")


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    计算相对误差序列
    
    公式: relative_error = (y_pred - y_true) / y_true * 100%
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        rel_error: 相对误差序列（百分比）
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # 避免除零
    epsilon = 1e-10
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    return (y_pred - y_true) / y_true_safe * 100


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    计算残差序列
    
    公式: residual = y_pred - y_true
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        residuals: 残差序列
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    return y_pred - y_true
