"""
Bootstrap 不确定性分析模块
功能：
1. 残差Bootstrap重采样
2. 参数分布估计
3. Bootstrap置信区间
"""

import numpy as np
from typing import Callable, Dict, Tuple
import warnings
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecm.ecm2rc import ECM2RCParams


def residual_bootstrap(
    t: np.ndarray,
    i: np.ndarray,
    v_measured: np.ndarray,
    soc: np.ndarray,
    ocv_func: Callable,
    params_fitted: ECM2RCParams,
    v_pred: np.ndarray,
    residuals: np.ndarray,
    fit_function: Callable,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: int = None,
    verbose: bool = True
) -> Dict:
    """
    残差Bootstrap分析
    
    方法：
    1. 使用拟合得到的残差
    2. 重采样残差（有放回）
    3. 构造新的观测值 V_new = V_pred + r*
    4. 重新拟合参数
    5. 重复B次，得到参数分布
    
    参数:
        t: 时间序列
        i: 电流序列
        v_measured: 实测电压
        soc: SOC序列
        ocv_func: OCV函数
        params_fitted: 拟合得到的参数
        v_pred: 拟合的预测电压
        residuals: 残差向量
        fit_function: 参数拟合函数
        n_bootstrap: Bootstrap迭代次数
        confidence_level: 置信水平
        seed: 随机种子
        verbose: 是否显示进度
    
    返回:
        results: Bootstrap结果字典
    """
    print("\n" + "="*60)
    print(f"Bootstrap 不确定性分析（{n_bootstrap} 次重采样）")
    print("="*60)
    
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(residuals)
    n_params = 5
    
    # 存储Bootstrap参数样本
    bootstrap_params = np.zeros((n_bootstrap, n_params))
    success_count = 0
    
    # Bootstrap循环
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap进度") if verbose else range(n_bootstrap)
    
    for b in iterator:
        try:
            # 1. 重采样残差（有放回）
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            resampled_residuals = residuals[indices]
            
            # 2. 构造新的观测值
            v_bootstrap = v_pred + resampled_residuals
            
            # 3. 重新拟合参数
            # 使用拟合得到的参数作为初始值
            x0 = params_fitted.to_array()
            
            # 调用拟合函数
            params_b, _ = fit_function(
                t, i, v_bootstrap, soc, ocv_func,
                method='least_squares',
                x0=x0,
                verbose=0
            )
            
            # 存储参数
            bootstrap_params[b, :] = params_b.to_array()
            success_count += 1
            
        except Exception as e:
            if verbose:
                warnings.warn(f"Bootstrap 迭代 {b} 失败: {e}")
            # 失败的迭代用原参数填充
            bootstrap_params[b, :] = params_fitted.to_array()
    
    print(f"\n成功的Bootstrap迭代: {success_count}/{n_bootstrap} ({success_count/n_bootstrap*100:.1f}%)")
    
    # 计算统计量
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']
    
    # 均值和标准差
    bootstrap_mean = np.mean(bootstrap_params, axis=0)
    bootstrap_std = np.std(bootstrap_params, axis=0, ddof=1)
    
    # 百分位数置信区间
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_params, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(bootstrap_params, (1 - alpha/2) * 100, axis=0)
    
    # 打印结果
    print("\nBootstrap 参数估计:")
    print("-" * 80)
    print(f"{'参数':<6} {'原估计':>12} {'Bootstrap均值':>15} {'Bootstrap标准差':>15} "
          f"{'CI下界':>12} {'CI上界':>12}")
    print("-" * 80)
    
    original = params_fitted.to_array()
    for i, name in enumerate(param_names):
        print(f"{name:<6} {original[i]:>12.6e} {bootstrap_mean[i]:>15.6e} "
              f"{bootstrap_std[i]:>15.6e} {ci_lower[i]:>12.6e} {ci_upper[i]:>12.6e}")
    
    # 计算偏差
    bias = bootstrap_mean - original
    print("\nBootstrap 偏差分析:")
    print("-" * 60)
    print(f"{'参数':<6} {'偏差':>12} {'相对偏差(%)':>15}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        rel_bias = bias[i] / original[i] * 100 if original[i] != 0 else 0
        print(f"{name:<6} {bias[i]:>12.6e} {rel_bias:>14.2f}%")
    
    # 组织结果
    results = {
        'bootstrap_samples': bootstrap_params,
        'param_names': param_names,
        'original_estimates': original,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'bias': bias,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'success_count': success_count
    }
    
    return results


def compare_ci_methods(ci_jacobian: Dict, ci_bootstrap: Dict):
    """
    比较雅可比方法和Bootstrap方法的置信区间
    
    参数:
        ci_jacobian: 雅可比方法的置信区间
        ci_bootstrap: Bootstrap方法的置信区间
    """
    print("\n" + "="*60)
    print("置信区间方法比较")
    print("="*60)
    
    param_names = ci_jacobian['param_names']
    
    print(f"\n{'参数':<6} {'估计值':>12} {'方法':>10} {'CI下界':>12} {'CI上界':>12} {'CI宽度':>12}")
    print("-" * 80)
    
    for i, name in enumerate(param_names):
        estimate = ci_jacobian['estimates'][i]
        
        # 雅可比方法
        jac_lower = ci_jacobian['ci_lower'][i]
        jac_upper = ci_jacobian['ci_upper'][i]
        jac_width = jac_upper - jac_lower
        
        print(f"{name:<6} {estimate:>12.6e} {'Jacobian':>10} "
              f"{jac_lower:>12.6e} {jac_upper:>12.6e} {jac_width:>12.6e}")
        
        # Bootstrap方法
        boot_lower = ci_bootstrap['ci_lower'][i]
        boot_upper = ci_bootstrap['ci_upper'][i]
        boot_width = boot_upper - boot_lower
        
        print(f"{'':>6} {estimate:>12} {'Bootstrap':>10} "
              f"{boot_lower:>12.6e} {boot_upper:>12.6e} {boot_width:>12.6e}")
        
        # 宽度比较
        width_ratio = boot_width / jac_width if jac_width > 0 else np.nan
        print(f"{'':>6} {'':>12} {'Width Ratio':>10} {width_ratio:>12.2f}")
        print("-" * 80)
