"""
置信区间分析模块
功能：
1. 基于雅可比矩阵的参数协方差估计
2. 计算参数置信区间
3. 参数相关性分析
"""

import numpy as np
from scipy.optimize import approx_fprime
from typing import Callable, Dict, Tuple
import warnings

import sys
from pathlib import Path

from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage


def compute_jacobian_numerical(
    residual_func: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    epsilon: float = None
) -> np.ndarray:
    """
    数值计算雅可比矩阵（残差对参数的导数）
    
    参数:
        residual_func: 残差函数，输入参数theta，输出残差向量
        theta: 参数向量
        epsilon: 有限差分步长（如果为None，自动选择相对步长）
    
    返回:
        J: 雅可比矩阵，形状 (n_samples, n_params)
    """
    n_params = len(theta)
    r0 = residual_func(theta)
    n_samples = len(r0)
    
    J = np.zeros((n_samples, n_params))
    
    for i in range(n_params):
        # 自动选择步长（相对步长）
        if epsilon is None:
            # 使用相对步长：max(sqrt(eps) * |theta_i|, sqrt(eps))
            h = max(np.sqrt(np.finfo(float).eps) * abs(theta[i]), np.sqrt(np.finfo(float).eps))
        else:
            h = epsilon
        
        theta_perturbed = theta.copy()
        theta_perturbed[i] += h
        r_perturbed = residual_func(theta_perturbed)
        J[:, i] = (r_perturbed - r0) / h
    
    return J


def compute_parameter_covariance(
    J: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    """
    计算参数协方差矩阵
    
    基于线性近似：
        Cov(θ) ≈ σ² (J^T J)^(-1)
    其中：
        σ² = SSE / (n - p)  # 残差方差
        SSE = sum(residuals²)
        n = 样本数
        p = 参数数
    
    参数:
        J: 雅可比矩阵，形状 (n_samples, n_params)
        residuals: 残差向量
    
    返回:
        cov_matrix: 参数协方差矩阵
    """
    n_samples, n_params = J.shape
    
    # 计算残差方差
    SSE = np.sum(residuals ** 2)
    sigma2 = SSE / (n_samples - n_params)
    
    # 计算 J^T J
    JTJ = J.T @ J
    
    # 检查条件数，避免数值问题
    cond_number = np.linalg.cond(JTJ)
    if cond_number > 1e10:
        warnings.warn(
            f"J^T J 的条件数很大 ({cond_number:.2e})，"
            f"协方差估计可能不准确。可能存在参数不可辨识问题。"
        )
    
    # 计算协方差矩阵
    try:
        cov_matrix = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        warnings.warn("J^T J 奇异，使用伪逆")
        cov_matrix = sigma2 * np.linalg.pinv(JTJ)
    
    return cov_matrix


def compute_confidence_intervals(
    params: ECM2RCParams,
    cov_matrix: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    计算参数置信区间
    
    参数:
        params: ECM参数
        cov_matrix: 参数协方差矩阵
        confidence_level: 置信水平（默认95%）
    
    返回:
        ci_dict: 包含置信区间信息的字典
    """
    from scipy import stats
    
    # 转换为数组
    theta = params.to_array()
    n_params = len(theta)
    
    # 计算参数标准差
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # 计算t分布的临界值（样本量大时接近正态分布的1.96）
    # 这里使用正态分布的分位数（大样本近似）
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # 计算置信区间
    ci_lower = theta - z_score * std_errors
    ci_upper = theta + z_score * std_errors
    
    # 组织结果
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']
    
    ci_dict = {
        'param_names': param_names,
        'estimates': theta,
        'std_errors': std_errors,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'relative_std': std_errors / np.abs(theta) * 100  # 相对标准差（%）
    }
    
    return ci_dict


def compute_correlation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """
    从协方差矩阵计算相关系数矩阵
    
    参数:
        cov_matrix: 协方差矩阵
    
    返回:
        corr_matrix: 相关系数矩阵
    """
    std = np.sqrt(np.diag(cov_matrix))
    
    # 处理标准差为0或非常小的情况
    std_safe = np.where(std < 1e-20, 1e-20, std)
    
    # 计算相关系数矩阵
    corr_matrix = cov_matrix / np.outer(std_safe, std_safe)
    
    # 确保对角线为1（数值误差）
    np.fill_diagonal(corr_matrix, 1.0)
    
    # 将非常大或非常小的值替换为NaN或0
    corr_matrix = np.where(np.abs(corr_matrix) > 1e10, np.nan, corr_matrix)
    corr_matrix = np.where(np.isnan(corr_matrix), 0.0, corr_matrix)
    
    # 限制在[-1, 1]范围内
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    return corr_matrix


def analyze_parameter_uncertainty(
    residual_func: Callable[[np.ndarray], np.ndarray],
    params: ECM2RCParams,
    residuals: np.ndarray,
    confidence_level: float = 0.95,
    use_stored_jacobian: np.ndarray = None
) -> Dict:
    """
    完整的参数不确定性分析
    
    参数:
        residual_func: 残差函数
        params: ECM参数
        residuals: 残差向量
        confidence_level: 置信水平
        use_stored_jacobian: 如果提供，使用存储的雅可比矩阵（避免重复计算）
    
    返回:
        results: 包含完整分析结果的字典
    """
    print("\n" + "="*60)
    print("参数不确定性分析（基于雅可比矩阵）")
    print("="*60)
    
    theta = params.to_array()
    
    # 计算雅可比矩阵
    if use_stored_jacobian is not None:
        print("使用提供的雅可比矩阵")
        J = use_stored_jacobian
    else:
        print("数值计算雅可比矩阵...")
        J = compute_jacobian_numerical(residual_func, theta)
        print(f"雅可比矩阵形状: {J.shape}")
    
    # 计算协方差矩阵
    print("计算参数协方差矩阵...")
    cov_matrix = compute_parameter_covariance(J, residuals)
    
    # 计算置信区间
    print(f"计算 {confidence_level*100:.0f}% 置信区间...")
    ci_dict = compute_confidence_intervals(params, cov_matrix, confidence_level)
    
    # 计算相关系数矩阵
    print("计算参数相关系数矩阵...")
    corr_matrix = compute_correlation_matrix(cov_matrix)
    
    # 打印结果
    print("\n参数估计与置信区间:")
    print("-" * 80)
    print(f"{'参数':<6} {'估计值':>12} {'标准差':>12} {'相对标准差':>12} "
          f"{'CI下界':>12} {'CI上界':>12}")
    print("-" * 80)
    
    for i, name in enumerate(ci_dict['param_names']):
        print(f"{name:<6} {ci_dict['estimates'][i]:>12.6e} "
              f"{ci_dict['std_errors'][i]:>12.6e} "
              f"{ci_dict['relative_std'][i]:>11.2f}% "
              f"{ci_dict['ci_lower'][i]:>12.6e} "
              f"{ci_dict['ci_upper'][i]:>12.6e}")
    
    print("\n参数相关系数矩阵:")
    print("-" * 60)
    param_names = ci_dict['param_names']
    print(f"{'':>6}", end='')
    for name in param_names:
        print(f"{name:>10}", end='')
    print()
    print("-" * 60)
    for i, name in enumerate(param_names):
        print(f"{name:>6}", end='')
        for j in range(len(param_names)):
            val = corr_matrix[i, j]
            if np.isnan(val) or np.isinf(val):
                print(f"{'N/A':>10}", end='')
            else:
                print(f"{val:>10.3f}", end='')
        print()
    
    # 检查高相关性参数
    print("\n参数可辨识性检查:")
    high_corr_threshold = 0.95
    high_corr_pairs = []
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            val = corr_matrix[i, j]
            if not np.isnan(val) and not np.isinf(val) and abs(val) > high_corr_threshold:
                high_corr_pairs.append((param_names[i], param_names[j], val))
    
    if high_corr_pairs:
        print(f"  发现高度相关的参数对（|相关系数| > {high_corr_threshold}）:")
        for p1, p2, corr in high_corr_pairs:
            print(f"    {p1} - {p2}: {corr:.4f}")
        print("  这些参数可能存在可辨识性问题。")
    else:
        print(f"  未发现高度相关的参数对（|相关系数| < {high_corr_threshold}）")
        print("  参数可辨识性良好。")
    
    # 组织返回结果
    results = {
        'jacobian': J,
        'covariance_matrix': cov_matrix,
        'correlation_matrix': corr_matrix,
        'confidence_intervals': ci_dict,
        'high_corr_pairs': high_corr_pairs
    }
    
    return results


def print_ci_table(ci_dict: Dict):
    """
    打印置信区间表格
    
    参数:
        ci_dict: 置信区间字典
    """
    print("\n" + "="*60)
    print(f"参数置信区间表 ({ci_dict['confidence_level']*100:.0f}% 置信水平)")
    print("="*60)
    print(f"{'参数':<6} {'估计值':>12} {'标准差':>12} {'CI下界':>12} {'CI上界':>12}")
    print("-" * 60)
    
    for i, name in enumerate(ci_dict['param_names']):
        print(f"{name:<6} {ci_dict['estimates'][i]:>12.6e} "
              f"{ci_dict['std_errors'][i]:>12.6e} "
              f"{ci_dict['ci_lower'][i]:>12.6e} "
              f"{ci_dict['ci_upper'][i]:>12.6e}")
    
    print("="*60)
