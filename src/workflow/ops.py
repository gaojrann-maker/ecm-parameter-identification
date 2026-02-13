"""
Workflow Operations: 三步骤工作流定义
Step1: DataReadOp - 数据读取和预处理
Step2: IdentifyOp - 参数辨识
Step3: UncertaintyOp - 不确定性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecm.loader import load_discharge_cc_segment
from ecm.ocv import fit_ocv_curve
from ecm.ecm2rc import ECM2RCParams, simulate_voltage
from identification.fit import fit_ecm_params, plot_fit_results
from analysis.ci import analyze_parameter_uncertainty, compute_jacobian_numerical
from analysis.bootstrap import residual_bootstrap, plot_bootstrap_results
from analysis.sensitivity import local_sensitivity_analysis, plot_sensitivity_results


class DataReadOp:
    """
    Step 1: 数据读取操作
    
    输入:
        - mat_path: B0005.mat 文件路径
        - cycle_n: 循环编号
        - current_threshold: 恒流段电流阈值
        - min_duration: 最小持续时间
    
    输出:
        - segment.csv: 包含 t, I, SOC, V 的数据文件
    """
    
    @staticmethod
    def execute(
        mat_path: str,
        cycle_n: int,
        current_threshold: float = 0.05,
        min_duration: float = 60.0,
        output_path: str = "segment.csv"
    ) -> str:
        """
        执行数据读取和预处理
        
        参数:
            mat_path: B0005.mat 文件路径
            cycle_n: 循环编号
            current_threshold: 恒流段电流阈值
            min_duration: 最小持续时间
            output_path: 输出文件路径
        
        返回:
            output_path: 输出文件路径
        """
        print("\n" + "="*70)
        print(f"Step 1: Data Read Operation")
        print("="*70)
        print(f"  Input: {mat_path}, Cycle {cycle_n}")
        print(f"  Output: {output_path}")
        
        # 加载恒流段
        t, i, v_measured, info = load_discharge_cc_segment(
            mat_path, n=cycle_n,
            current_threshold=current_threshold,
            min_duration=min_duration
        )
        
        # 计算 SOC
        capacity_ah = info['total_capacity_ah']
        dt = np.gradient(t)
        charge_ah = np.cumsum(-i * dt) / 3600
        soc = 1.0 - charge_ah / capacity_ah
        soc = np.clip(soc, 0.0, 1.0)
        
        # 保存为 CSV
        df = pd.DataFrame({
            't': t,
            'I': i,
            'SOC': soc,
            'V': v_measured
        })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"  Data points: {len(t)}")
        print(f"  Duration: {info['duration']:.2f} s")
        print(f"  SOC range: [{soc.min():.4f}, {soc.max():.4f}]")
        print(f"  Saved to: {output_file}")
        print("="*70)
        
        return str(output_file)


class IdentifyOp:
    """
    Step 2: 参数辨识操作
    
    输入:
        - segment.csv: 数据文件（t, I, SOC, V）
    
    输出:
        - params.json: 辨识的参数
        - fit_metrics.json: 拟合指标
        - fit_curve.png: 拟合曲线图
    """
    
    @staticmethod
    def execute(
        segment_csv: str,
        output_dir: str = ".",
        x0: list = None,
        bounds: tuple = None
    ) -> Dict[str, str]:
        """
        执行参数辨识
        
        参数:
            segment_csv: 输入数据文件
            output_dir: 输出目录
            x0: 初始参数猜测
            bounds: 参数边界
        
        返回:
            output_files: 输出文件路径字典
        """
        print("\n" + "="*70)
        print(f"Step 2: Parameter Identification Operation")
        print("="*70)
        print(f"  Input: {segment_csv}")
        
        # 读取数据
        df = pd.read_csv(segment_csv)
        t = df['t'].values
        i = df['I'].values
        soc = df['SOC'].values
        v_measured = df['V'].values
        
        print(f"  Loaded {len(t)} data points")
        
        # 创建 OCV 函数（使用采样方法）
        soc_samples = np.linspace(soc.min(), soc.max(), 15)
        v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
        ocv_func = fit_ocv_curve(soc_samples, v_samples, method='linear')
        
        # 设置默认参数
        if x0 is None:
            x0 = [1e-4, 1e-4, 1e6, 1e-4, 1e6]
        if bounds is None:
            bounds = (
                [1e-4, 1e-4, 1e1, 1e-4, 1e1],
                [1e0, 1e0, 1e6, 1e0, 1e6]
            )
        
        # 参数辨识
        print("  Running least squares optimization...")
        params_fitted, fit_result = fit_ecm_params(
            t, i, v_measured, soc, ocv_func,
            method='least_squares',
            x0=x0,
            bounds=bounds,
            verbose=1
        )
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存参数
        params_dict = {
            'R0': float(params_fitted.R0),
            'R1': float(params_fitted.R1),
            'C1': float(params_fitted.C1),
            'R2': float(params_fitted.R2),
            'C2': float(params_fitted.C2),
            'tau1': float(params_fitted.R1 * params_fitted.C1),
            'tau2': float(params_fitted.R2 * params_fitted.C2)
        }
        
        params_file = output_path / 'params.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=4)
        
        # 保存拟合指标
        metrics_dict = {k: float(v) for k, v in fit_result['metrics'].items()}
        
        metrics_file = output_path / 'fit_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # 保存拟合曲线图
        fig = plot_fit_results(
            t, v_measured, fit_result['v_pred'], fit_result['residuals'],
            params_fitted, fit_result['metrics'], i
        )
        
        curve_file = output_path / 'fit_curve.png'
        fig.savefig(curve_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  RMSE: {fit_result['metrics']['RMSE']:.6f} V")
        print(f"  R2: {fit_result['metrics']['R2']:.6f}")
        print(f"  Output files:")
        print(f"    - {params_file}")
        print(f"    - {metrics_file}")
        print(f"    - {curve_file}")
        print("="*70)
        
        return {
            'params_json': str(params_file),
            'fit_metrics_json': str(metrics_file),
            'fit_curve_png': str(curve_file)
        }


class UncertaintyOp:
    """
    Step 3: 不确定性分析操作
    
    输入:
        - segment.csv: 数据文件
        - params.json: 辨识的参数
    
    输出:
        - ci_table.csv: 置信区间表
        - sensitivity.png: 敏感性分析图
        - bootstrap_params.csv: Bootstrap 参数样本
    """
    
    @staticmethod
    def execute(
        segment_csv: str,
        params_json: str,
        output_dir: str = ".",
        n_bootstrap: int = 50
    ) -> Dict[str, str]:
        """
        执行不确定性分析
        
        参数:
            segment_csv: 输入数据文件
            params_json: 参数文件
            output_dir: 输出目录
            n_bootstrap: Bootstrap 重采样次数
        
        返回:
            output_files: 输出文件路径字典
        """
        print("\n" + "="*70)
        print(f"Step 3: Uncertainty Analysis Operation")
        print("="*70)
        print(f"  Input: {segment_csv}, {params_json}")
        
        # 读取数据
        df = pd.read_csv(segment_csv)
        t = df['t'].values
        i = df['I'].values
        soc = df['SOC'].values
        v_measured = df['V'].values
        
        # 读取参数
        with open(params_json, 'r') as f:
            params_dict = json.load(f)
        
        params_fitted = ECM2RCParams(
            R0=params_dict['R0'],
            R1=params_dict['R1'],
            C1=params_dict['C1'],
            R2=params_dict['R2'],
            C2=params_dict['C2']
        )
        
        # 创建 OCV 函数
        soc_samples = np.linspace(soc.min(), soc.max(), 15)
        v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
        ocv_func = fit_ocv_curve(soc_samples, v_samples, method='linear')
        
        # 计算预测值和残差
        v_pred = simulate_voltage(t, i, soc, params_fitted, ocv_func)
        residuals = v_pred - v_measured
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ================================================================
        # 3.1 置信区间分析
        # ================================================================
        print("  Running confidence interval analysis...")
        
        def residual_func(theta):
            params_test = ECM2RCParams.from_array(theta)
            v_pred_test = simulate_voltage(t, i, soc, params_test, ocv_func)
            return v_pred_test - v_measured
        
        ci_results = analyze_parameter_uncertainty(
            residual_func=residual_func,
            params=params_fitted,
            residuals=residuals,
            confidence_level=0.95,
            use_stored_jacobian=None
        )
        
        # 保存置信区间表
        ci_dict = ci_results['confidence_intervals']
        ci_df = pd.DataFrame({
            'Parameter': ci_dict['param_names'],
            'Estimate': ci_dict['estimates'],
            'Std_Error': ci_dict['std_errors'],
            'Relative_Std_%': ci_dict['relative_std'],
            'CI_Lower': ci_dict['ci_lower'],
            'CI_Upper': ci_dict['ci_upper']
        })
        
        ci_file = output_path / 'ci_table.csv'
        ci_df.to_csv(ci_file, index=False)
        
        # ================================================================
        # 3.2 敏感性分析
        # ================================================================
        print("  Running sensitivity analysis...")
        
        sensitivity_results = local_sensitivity_analysis(
            t, i, soc, ocv_func,
            params=params_fitted,
            perturbation=0.01,
            v_baseline=v_pred
        )
        
        # 保存敏感性分析图
        fig_sens = plot_sensitivity_results(sensitivity_results)
        sens_file = output_path / 'sensitivity.png'
        fig_sens.savefig(sens_file, dpi=150, bbox_inches='tight')
        plt.close(fig_sens)
        
        # ================================================================
        # 3.3 Bootstrap 分析
        # ================================================================
        print(f"  Running bootstrap analysis ({n_bootstrap} iterations)...")
        
        bootstrap_results = residual_bootstrap(
            t, i, v_measured, soc, ocv_func,
            params_fitted=params_fitted,
            v_pred=v_pred,
            residuals=residuals,
            fit_function=fit_ecm_params,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            seed=42,
            verbose=False
        )
        
        # 保存 Bootstrap 参数样本
        bootstrap_df = pd.DataFrame(
            bootstrap_results['bootstrap_samples'],
            columns=bootstrap_results['param_names']
        )
        
        bootstrap_file = output_path / 'bootstrap_params.csv'
        bootstrap_df.to_csv(bootstrap_file, index=False)
        
        # 保存 Bootstrap 分析图
        fig_boot = plot_bootstrap_results(bootstrap_results, ci_results)
        boot_fig_file = output_path / 'bootstrap_analysis.png'
        fig_boot.savefig(boot_fig_file, dpi=150, bbox_inches='tight')
        plt.close(fig_boot)
        
        print(f"  Output files:")
        print(f"    - {ci_file}")
        print(f"    - {sens_file}")
        print(f"    - {bootstrap_file}")
        print(f"    - {boot_fig_file}")
        print("="*70)
        
        return {
            'ci_table_csv': str(ci_file),
            'sensitivity_png': str(sens_file),
            'bootstrap_params_csv': str(bootstrap_file),
            'bootstrap_analysis_png': str(boot_fig_file)
        }


# 导出所有 OP
__all__ = ['DataReadOp', 'IdentifyOp', 'UncertaintyOp']
