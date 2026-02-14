"""
dflow 工作流定义
将数据读取、参数辨识、不确定性分析封装成 3 个 dflow Step

============================================================================
运行环境说明
============================================================================
- 提交端（你运行 python dflow_example.py submit 的地方）：玻尔平台节点
- 执行端（OP 实际运行的地方）：Argo/K8s Pod，使用你构建的镜像

关键前提：镜像中必须已通过 .pth 文件将 /opt/ECM-dflow 加入 sys.path
（运行 setup_image.sh 会自动完成）
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, List

from dflow import (
    Workflow,
    Step,
    upload_artifact,
    download_artifact,
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    Artifact,
    Parameter,
)


# =============================================================================
# 配置区域
# =============================================================================

# 默认镜像 - 必须改成你在玻尔平台构建的镜像地址
DEFAULT_IMAGE = os.environ.get(
    "ECM_DOCKER_IMAGE",
    "registry.dp.tech/dptech/dp/native/prod-2679131/ecm-dflow:v1"
)


# ========================================================================
# Step 1: 数据读取 OP
# ========================================================================
class DataReadOP(OP):
    """
    数据读取操作：加载 B0005.mat，提取恒流段，计算 SOC，拟合 OCV
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return {
            "data_path": Parameter(str),
            "cycle_n": Parameter(int, default=1),
            "current_threshold": Parameter(float, default=0.05),
            "min_duration": Parameter(float, default=60.0),
        }

    @classmethod
    def get_output_sign(cls):
        return {
            "segment_csv": Artifact(Path),
            "soc_range": Parameter(List[float]),
            "data_points": Parameter(int),
        }

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        import numpy as np
        import pandas as pd
        from src.ecm.loader import load_discharge_cc_segment
        from src.ecm.ocv import fit_ocv_curve

        print(f"[Step 1] 数据读取开始...", flush=True)
        print(f"  数据路径: {op_in['data_path']}", flush=True)
        print(f"  循环编号: {op_in['cycle_n']}", flush=True)

        # 1. 加载恒流段数据
        t, i, v_measured, info = load_discharge_cc_segment(
            op_in['data_path'],
            n=op_in['cycle_n'],
            current_threshold=op_in['current_threshold'],
            min_duration=op_in['min_duration']
        )
        print(f"  加载成功: {len(t)} 个数据点", flush=True)

        # 2. 计算 SOC
        capacity_ah = info['total_capacity_ah']
        dt = np.gradient(t)
        charge_ah = np.cumsum(-i * dt) / 3600
        soc = 1.0 - charge_ah / capacity_ah
        soc = np.clip(soc, 0.0, 1.0)
        soc_range = [float(soc.min()), float(soc.max())]
        print(f"  SOC 范围: [{soc_range[0]:.4f}, {soc_range[1]:.4f}]", flush=True)

        # 3. 拟合 OCV-SOC 曲线
        ocv_func = fit_ocv_curve(soc, v_measured, method='cubic')
        print(f"  OCV 曲线拟合完成", flush=True)

        # 4. 保存到 CSV
        output_csv = Path("segment.csv")
        df = pd.DataFrame({
            't': t, 'i': i, 'soc': soc,
            'v_measured': v_measured, 'ocv': ocv_func(soc)
        })
        df.to_csv(output_csv, index=False)
        print(f"  数据已保存: {output_csv}", flush=True)

        return OPIO({
            "segment_csv": output_csv,
            "soc_range": soc_range,
            "data_points": len(t),
        })


# ========================================================================
# Step 2: 参数辨识 OP
# ========================================================================
class IdentifyOP(OP):
    """
    参数辨识操作：使用最小二乘法拟合 ECM 参数
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return {
            "segment_csv": Artifact(Path),
            "x0": Parameter(List[float], default=None),
            "bounds": Parameter(List[List[float]], default=None),
        }

    @classmethod
    def get_output_sign(cls):
        return {
            "params_json": Artifact(Path),
            "fit_metrics_json": Artifact(Path),
            "rmse": Parameter(float),
            "r2": Parameter(float),
        }

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        import numpy as np
        import pandas as pd
        from scipy.interpolate import interp1d
        from src.identification.fit import fit_ecm_params

        print(f"[Step 2] 参数辨识开始...", flush=True)

        # 1. 读取数据
        df = pd.read_csv(op_in['segment_csv'])
        t = df['t'].values
        i = df['i'].values
        soc = df['soc'].values
        v_measured = df['v_measured'].values
        ocv = df['ocv'].values

        # 重建 OCV 插值函数
        ocv_func = interp1d(soc, ocv, kind='cubic', fill_value='extrapolate')
        print(f"  数据加载完成: {len(t)} 个数据点", flush=True)

        # 2. 设置初始值和边界
        x0 = op_in['x0'] if op_in['x0'] is not None else [1e-4, 1e-4, 1e6, 1e-4, 1e6]
        if op_in['bounds'] is not None:
            bounds = tuple(op_in['bounds'])
        else:
            bounds = ([1e-4, 1e-4, 10, 1e-4, 10], [1.0, 1.0, 1e6, 1.0, 1e6])

        # 3. 参数辨识
        #    fit_ecm_params 返回 (params: ECM2RCParams, results: dict)
        print(f"  执行参数辨识...", flush=True)
        params, fit_result = fit_ecm_params(
            t, i, v_measured, soc, ocv_func,
            method='least_squares',
            x0=x0,
            bounds=bounds,
            verbose=1
        )

        metrics = fit_result['metrics']
        print(f"  辨识完成: RMSE={metrics['RMSE']:.6f} V, R2={metrics['R2']:.6f}", flush=True)

        # 4. 保存结果
        params_file = Path("params.json")
        with open(params_file, 'w') as f:
            json.dump({
                'R0': params.R0, 'R1': params.R1, 'C1': params.C1,
                'R2': params.R2, 'C2': params.C2,
            }, f, indent=2)

        metrics_file = Path("fit_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        return OPIO({
            "params_json": params_file,
            "fit_metrics_json": metrics_file,
            "rmse": float(metrics['RMSE']),
            "r2": float(metrics['R2']),
        })


# ========================================================================
# Step 3: 不确定性分析 OP
# ========================================================================
class UncertaintyOP(OP):
    """
    不确定性分析操作：置信区间 + Bootstrap + 敏感性分析
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return {
            "segment_csv": Artifact(Path),
            "params_json": Artifact(Path),
            "n_bootstrap": Parameter(int, default=50),
        }

    @classmethod
    def get_output_sign(cls):
        return {
            "ci_table_csv": Artifact(Path),
            "bootstrap_params_csv": Artifact(Path),
            "sensitivity_data_json": Artifact(Path),
            "ci_summary": Parameter(Dict),
        }

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        import numpy as np
        import pandas as pd
        from scipy.interpolate import interp1d
        from src.analysis.ci import analyze_parameter_uncertainty
        from src.analysis.bootstrap import residual_bootstrap
        from src.analysis.sensitivity import local_sensitivity_analysis
        from src.identification.fit import fit_ecm_params
        from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage

        print(f"[Step 3] 不确定性分析开始...", flush=True)

        # 1. 读取数据
        df = pd.read_csv(op_in['segment_csv'])
        t = df['t'].values
        i = df['i'].values
        soc = df['soc'].values
        v_measured = df['v_measured'].values
        ocv = df['ocv'].values
        ocv_func = interp1d(soc, ocv, kind='cubic', fill_value='extrapolate')

        # 2. 读取参数
        with open(op_in['params_json'], 'r') as f:
            params_dict = json.load(f)
        params = ECM2RCParams(
            R0=params_dict['R0'], R1=params_dict['R1'], C1=params_dict['C1'],
            R2=params_dict['R2'], C2=params_dict['C2'],
        )

        # 3. 计算预测值和残差
        v_pred = simulate_voltage(t, i, soc, params, ocv_func)
        residuals = v_pred - v_measured
        print(f"  参数加载完成", flush=True)

        # 4. 置信区间分析（基于雅可比矩阵）
        print(f"  执行置信区间分析...", flush=True)

        def residual_func(theta):
            p = ECM2RCParams.from_array(theta)
            v = simulate_voltage(t, i, soc, p, ocv_func)
            return v - v_measured

        ci_results = analyze_parameter_uncertainty(
            residual_func=residual_func,
            params=params,
            residuals=residuals,           # 修复：之前缺少此参数
            confidence_level=0.95,
            use_stored_jacobian=None
        )

        # 保存置信区间表格
        #   ci_results['confidence_intervals'] 包含正确的键
        ci_dict = ci_results['confidence_intervals']
        ci_file = Path("ci_table.csv")
        param_names = ci_dict['param_names']
        ci_data = []
        for idx, name in enumerate(param_names):
            ci_data.append({
                'Parameter': name,
                'Value': ci_dict['estimates'][idx],
                'StdError': ci_dict['std_errors'][idx],
                'CI_Lower': ci_dict['ci_lower'][idx],
                'CI_Upper': ci_dict['ci_upper'][idx],
            })
        pd.DataFrame(ci_data).to_csv(ci_file, index=False)
        print(f"  置信区间已保存: {ci_file}", flush=True)

        # 5. Bootstrap 分析
        print(f"  执行 Bootstrap 分析 ({op_in['n_bootstrap']} 次)...", flush=True)
        bootstrap_results = residual_bootstrap(
            t, i, v_measured, soc, ocv_func,
            params_fitted=params,
            v_pred=v_pred,
            residuals=residuals,
            fit_function=fit_ecm_params,   # 修复：之前缺少此参数
            n_bootstrap=op_in['n_bootstrap'],
            confidence_level=0.95,
            seed=42,
            verbose=False
        )

        bootstrap_file = Path("bootstrap_params.csv")
        bootstrap_df = pd.DataFrame(
            bootstrap_results['bootstrap_samples'],
            columns=bootstrap_results['param_names']
        )
        bootstrap_df.to_csv(bootstrap_file, index=False)
        print(f"  Bootstrap 结果已保存: {bootstrap_file}", flush=True)

        # 6. 敏感性分析
        print(f"  执行敏感性分析...", flush=True)
        sensitivity_results = local_sensitivity_analysis(
            t, i, soc, ocv_func,
            params=params,
            perturbation=0.01,
            v_baseline=v_pred
        )

        sensitivity_file = Path("sensitivity_data.json")
        sensitivity_data = {
            'rms_sensitivity': sensitivity_results['sensitivity_rms'].tolist(),      # 修复键名
            'max_sensitivity': sensitivity_results['sensitivity_max'].tolist(),      # 修复键名
            'sensitivity_ranking': sensitivity_results['sensitivity_ranking'],
            'param_names': param_names,
        }
        with open(sensitivity_file, 'w') as f:
            json.dump(sensitivity_data, f, indent=2)
        print(f"  敏感性分析已保存: {sensitivity_file}", flush=True)

        # 7. 生成摘要
        ci_summary = {
            'mean_std_error': float(np.mean(ci_dict['std_errors'])),
            'max_std_error': float(np.max(ci_dict['std_errors'])),
            'bootstrap_success_rate': float(
                bootstrap_results['success_count'] / op_in['n_bootstrap']
            ),
            'most_sensitive_param': sensitivity_results['sensitivity_ranking'][0],
        }

        print(f"  不确定性分析完成", flush=True)
        print(f"    Bootstrap 成功率: {ci_summary['bootstrap_success_rate']*100:.1f}%", flush=True)
        print(f"    最敏感参数: {ci_summary['most_sensitive_param']}", flush=True)

        return OPIO({
            "ci_table_csv": ci_file,
            "bootstrap_params_csv": bootstrap_file,
            "sensitivity_data_json": sensitivity_file,
            "ci_summary": ci_summary,
        })


# ========================================================================
# 构建完整的 dflow 工作流
# ========================================================================
def build_ecm_workflow(
    data_path: str = "/data/B0005.mat",
    cycle_n: int = 1,
    n_bootstrap: int = 50,
    workflow_name: str = "ecm-identification"
) -> Workflow:
    """
    构建 ECM 参数辨识工作流（3 个 Step 串行）
    """
    wf = Workflow(name=workflow_name)

    # Step 1: 数据读取
    step1 = Step(
        name="data-read",
        template=PythonOPTemplate(
            DataReadOP,
            image=DEFAULT_IMAGE,
            command=["python3"],
            python_packages=[],
        ),
        parameters={
            "data_path": data_path,
            "cycle_n": cycle_n,
            "current_threshold": 0.05,
            "min_duration": 60.0,
        },
    )
    wf.add(step1)

    # Step 2: 参数辨识（依赖 Step 1 的 segment_csv）
    step2 = Step(
        name="identify",
        template=PythonOPTemplate(
            IdentifyOP,
            image=DEFAULT_IMAGE,
            command=["python3"],
            python_packages=[],
        ),
        artifacts={
            "segment_csv": step1.outputs.artifacts["segment_csv"]
        },
    )
    wf.add(step2)

    # Step 3: 不确定性分析（依赖 Step 1 和 Step 2）
    step3 = Step(
        name="uncertainty",
        template=PythonOPTemplate(
            UncertaintyOP,
            image=DEFAULT_IMAGE,
            command=["python3"],
            python_packages=[],
        ),
        parameters={
            "n_bootstrap": n_bootstrap,
        },
        artifacts={
            "segment_csv": step1.outputs.artifacts["segment_csv"],
            "params_json": step2.outputs.artifacts["params_json"],
        },
    )
    wf.add(step3)

    return wf


# ========================================================================
# 提交工作流
# ========================================================================
def submit_workflow(
    data_path: str = "/data/B0005.mat",
    cycle_n: int = 1,
    n_bootstrap: int = 50
):
    """提交工作流到 Argo"""
    print("=" * 70, flush=True)
    print("构建 ECM 参数辨识工作流", flush=True)
    print("=" * 70, flush=True)
    print(f"  数据路径: {data_path}", flush=True)
    print(f"  循环编号: {cycle_n}", flush=True)
    print(f"  Bootstrap 次数: {n_bootstrap}", flush=True)
    print(f"  镜像: {DEFAULT_IMAGE}", flush=True)
    print("=" * 70, flush=True)

    wf = build_ecm_workflow(
        data_path=data_path,
        cycle_n=cycle_n,
        n_bootstrap=n_bootstrap
    )

    wf.submit()

    print(f"\n工作流已提交!", flush=True)
    print(f"  ID: {wf.id}", flush=True)
    print(f"  步骤: data-read -> identify -> uncertainty", flush=True)

    return wf
