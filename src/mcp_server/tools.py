"""
MCP Tools 实现
将 ECM 参数辨识能力封装为 MCP 标准工具

设计原则：
1. 所有文件操作基于当前工作目录（用户会话空间）
2. 不依赖任何本地项目路径
3. 内置资源通过 _get_bundled_resource_path() 获取并复制到用户空间
"""

import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# 模块路径（用于定位打包的内置资源，不用于数据文件查找）
_MODULE_DIR = Path(__file__).parent
_PACKAGE_ROOT = _MODULE_DIR.parent.parent  # src/mcp_server -> src -> project_root

# 添加到 sys.path 仅用于模块导入
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _get_bundled_resource_path(resource_name: str) -> Optional[Path]:
    """
    获取内置资源文件的路径（打包在 Agent 中的资源）
    
    这些资源随 Agent 一起部署，用于初始化用户空间
    """
    # 内置资源存放在 package 的 resources 目录
    bundled_path = _PACKAGE_ROOT / "resources" / resource_name
    if bundled_path.exists():
        return bundled_path
    
    # 兼容：也检查 data 目录（开发时）
    dev_path = _PACKAGE_ROOT / "data" / resource_name
    if dev_path.exists():
        return dev_path
    
    return None


def _ensure_default_data_file():
    """
    确保用户工作空间中存在默认数据文件
    
    如果用户空间的 data/B0005.mat 不存在，从内置资源复制一份
    """
    cwd = Path.cwd()
    user_data_dir = cwd / "data"
    user_data_file = user_data_dir / "B0005.mat"
    
    print(f"[_ensure_default_data_file] cwd={cwd}")
    print(f"[_ensure_default_data_file] user_data_file={user_data_file}, exists={user_data_file.exists()}")
    print(f"[_ensure_default_data_file] _PACKAGE_ROOT={_PACKAGE_ROOT}")
    
    # 如果已存在，不覆盖
    if user_data_file.exists():
        print(f"[_ensure_default_data_file] 文件已存在，跳过")
        return
    
    # 从内置资源复制
    bundled = _get_bundled_resource_path("B0005.mat")
    print(f"[_ensure_default_data_file] bundled={bundled}, exists={bundled.exists() if bundled else 'N/A'}")
    
    if bundled and bundled.exists():
        user_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, user_data_file)
        print(f"[_ensure_default_data_file] 已复制: {bundled} -> {user_data_file}")
    else:
        print(f"[_ensure_default_data_file] 内置资源不存在，无法复制")


def _resolve_data_path(data_path: str) -> Path:
    """
    解析数据文件路径 - 仅在用户工作空间（当前工作目录）内查找
    
    注意：此函数不会访问任何本地项目路径，只使用 Path.cwd()
    """
    path = Path(data_path)
    cwd = Path.cwd()
    
    # 提取文件名
    filename = path.name if ("/" in data_path or "\\" in data_path) else data_path
    
    # 1. 如果是绝对路径且在工作空间内，使用它
    if path.is_absolute():
        try:
            path.relative_to(cwd)
            if path.exists() and path.is_file():
                return path
        except ValueError:
            pass
        # 绝对路径不在工作空间内，忽略，用文件名继续查找
    
    # 2. 相对于当前工作目录的完整路径
    cwd_path = cwd / path
    if cwd_path.exists() and cwd_path.is_file():
        return cwd_path
    
    # 3. data 子目录
    data_path_candidate = cwd / "data" / filename
    if data_path_candidate.exists() and data_path_candidate.is_file():
        return data_path_candidate
    
    # 4. 直接在当前目录
    direct_path = cwd / filename
    if direct_path.exists() and direct_path.is_file():
        return direct_path
    
    # 返回预期路径，让调用方处理不存在的情况
    return cwd_path

from src.mcp_server.schemas import (
    IdentifyInput, UncertaintyInput, PipelineInput,
    IdentifyResult, PipelineResult, UncertaintyResult,
    ECMParams, FitMetrics, ConfidenceInterval, SensitivityInfo, BootstrapSummary
)


def _params_to_schema(params) -> ECMParams:
    """将 ECM2RCParams 转换为 Pydantic 模型"""
    return ECMParams(
        R0=float(params.R0),
        R1=float(params.R1),
        C1=float(params.C1),
        R2=float(params.R2),
        C2=float(params.C2),
        tau1=float(params.R1 * params.C1),
        tau2=float(params.R2 * params.C2)
    )


def _metrics_to_schema(metrics: Dict[str, Any]) -> FitMetrics:
    """将指标字典转换为 Pydantic 模型"""
    return FitMetrics(
        RMSE=float(metrics.get('RMSE', 0)),
        MAE=float(metrics.get('MAE', 0)),
        R2=float(metrics.get('R2', 0)),
        MSE=float(metrics.get('MSE', 0)),
        MAPE=float(metrics.get('MAPE', 0))
    )


def _ci_to_schema(ci_results: Dict) -> list:
    """将置信区间结果转换为 Pydantic 模型列表"""
    ci_dict = ci_results.get('confidence_intervals', {})
    if not ci_dict:
        return []
    
    param_names = ci_dict.get('param_names', [])
    estimates = ci_dict.get('estimates', [])
    std_errors = ci_dict.get('std_errors', [])
    relative_stds = ci_dict.get('relative_std', [])
    ci_lowers = ci_dict.get('ci_lower', [])
    ci_uppers = ci_dict.get('ci_upper', [])
    
    result = []
    for i, name in enumerate(param_names):
        result.append(ConfidenceInterval(
            param_name=name,
            estimate=float(estimates[i]) if i < len(estimates) else 0.0,
            std_error=float(std_errors[i]) if i < len(std_errors) else 0.0,
            relative_std=float(relative_stds[i]) if i < len(relative_stds) else 0.0,
            ci_lower=float(ci_lowers[i]) if i < len(ci_lowers) else 0.0,
            ci_upper=float(ci_uppers[i]) if i < len(ci_uppers) else 0.0
        ))
    return result


def _sensitivity_to_schema(sensitivity_results: Dict) -> list:
    """将敏感性分析结果转换为 Pydantic 模型列表"""
    ranking = sensitivity_results.get('ranking', [])
    rms_values = sensitivity_results.get('rms_sensitivity', {})
    max_values = sensitivity_results.get('max_sensitivity', {})
    
    result = []
    for rank, name in enumerate(ranking, 1):
        result.append(SensitivityInfo(
            param_name=name,
            rms_sensitivity=float(rms_values.get(name, 0)),
            max_sensitivity=float(max_values.get(name, 0)),
            rank=rank
        ))
    return result


def _bootstrap_to_schema(bootstrap_results: Dict) -> BootstrapSummary:
    """将 Bootstrap 结果转换为 Pydantic 模型"""
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']
    
    means = bootstrap_results.get('param_means', [])
    stds = bootstrap_results.get('param_stds', [])
    ci_lower = bootstrap_results.get('ci_lower', [])
    ci_upper = bootstrap_results.get('ci_upper', [])
    
    return BootstrapSummary(
        n_bootstrap=int(bootstrap_results.get('n_bootstrap', 0)),
        success_rate=float(bootstrap_results.get('success_rate', 0)),
        param_means={name: float(means[i]) if i < len(means) else 0.0 
                     for i, name in enumerate(param_names)},
        param_stds={name: float(stds[i]) if i < len(stds) else 0.0 
                    for i, name in enumerate(param_names)},
        ci_lower={name: float(ci_lower[i]) if i < len(ci_lower) else 0.0 
                  for i, name in enumerate(param_names)},
        ci_upper={name: float(ci_upper[i]) if i < len(ci_upper) else 0.0 
                  for i, name in enumerate(param_names)}
    )


def _collect_artifacts(output_dir: Path) -> Dict[str, str]:
    """
    收集输出目录中的所有文件
    
    返回相对路径，避免暴露本地绝对路径给用户
    路径相对于当前工作目录（用户会话 files 目录）
    """
    artifacts = {}
    if output_dir.exists():
        cwd = Path.cwd()
        for f in output_dir.iterdir():
            if f.is_file():
                try:
                    # 尝试返回相对于工作目录的路径
                    rel_path = f.relative_to(cwd)
                    artifacts[f.name] = str(rel_path)
                except ValueError:
                    # 如果无法计算相对路径，返回相对于输出目录的路径
                    artifacts[f.name] = f"{output_dir.name}/{f.name}"
    return artifacts


def identify(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs",
    current_threshold: float = 0.05,
    min_duration: float = 60.0
) -> dict:
    """
    执行 ECM 参数辨识
    
    Args:
        data_path: B0005.mat 数据文件路径
        cycle_number: 放电循环编号（从1开始）
        output_dir: 输出目录路径
        current_threshold: 恒流段电流标准差阈值
        min_duration: 恒流段最小持续时间（秒）
    
    Returns:
        参数辨识结果，包含 ECM 参数和拟合指标
    """
    try:
        import numpy as np
        from src.ecm.loader import load_discharge_cc_segment
        from src.ecm.ocv import fit_ocv_curve
        from src.ecm.ecm2rc import ECM2RCParams
        from src.identification.fit import fit_ecm_params, plot_fit_results
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print(f"\n{'='*60}")
        print(f"ECM 参数辨识 - MCP Tool")
        print(f"{'='*60}")
        
        # 解析数据文件路径（确保默认数据文件存在）
        _ensure_default_data_file()
        resolved_path = _resolve_data_path(data_path)
        
        # 只在日志中打印，不返回给用户
        print(f"[内部日志] 数据文件: {resolved_path}")
        print(f"[内部日志] 循环编号: {cycle_number}")
        
        # 检查数据文件 - 不暴露本地路径
        if not resolved_path.exists():
            display_name = Path(data_path).name
            return IdentifyResult(
                status="failed",
                message=f"数据文件 '{display_name}' 不存在。请在文件浏览器中上传数据文件到 data 文件夹。",
                error="FileNotFoundError"
            ).model_dump()
        
        # 使用解析后的路径
        data_path = str(resolved_path)
        
        # 创建输出目录
        out_path = Path(output_dir) / f"cycle_{cycle_number:03d}"
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        t, i, v_measured, info = load_discharge_cc_segment(
            data_path, n=cycle_number,
            current_threshold=current_threshold,
            min_duration=min_duration
        )
        
        # 计算 SOC 和 OCV
        capacity_ah = info['total_capacity_ah']
        dt = np.gradient(t)
        charge_ah = np.cumsum(-i * dt) / 3600
        soc = 1.0 - charge_ah / capacity_ah
        soc = np.clip(soc, 0.0, 1.0)
        
        soc_samples = np.linspace(soc.min(), soc.max(), 15)
        v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
        ocv_func = fit_ocv_curve(soc_samples, v_samples, method='linear')
        
        # 参数辨识
        x0 = np.array([1e-4, 1e-4, 1e6, 1e-4, 1e6])
        bounds = ([1e-4, 1e-4, 1e1, 1e-4, 1e1], [1e0, 1e0, 1e6, 1e0, 1e6])
        
        params_fitted, fit_result = fit_ecm_params(
            t, i, v_measured, soc, ocv_func,
            method='least_squares', x0=x0, bounds=bounds, verbose=0
        )
        
        # 保存结果
        import json
        params_dict = {
            'R0': float(params_fitted.R0),
            'R1': float(params_fitted.R1),
            'C1': float(params_fitted.C1),
            'R2': float(params_fitted.R2),
            'C2': float(params_fitted.C2),
            'tau1': float(params_fitted.R1 * params_fitted.C1),
            'tau2': float(params_fitted.R2 * params_fitted.C2)
        }
        with open(out_path / 'params.json', 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=4, ensure_ascii=False)
        
        metrics_dict = {k: float(v) for k, v in fit_result['metrics'].items()}
        with open(out_path / 'fit_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        
        # 绘图
        fig = plot_fit_results(
            t, v_measured, fit_result['v_pred'], fit_result['residuals'],
            params_fitted, fit_result['metrics'], i
        )
        fig.savefig(out_path / 'fit_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n辨识完成！参数: {params_fitted}")
        print(f"RMSE = {fit_result['metrics']['RMSE']:.6f} V")
        
        return IdentifyResult(
            status="success",
            message="参数辨识完成",
            params=_params_to_schema(params_fitted),
            metrics=_metrics_to_schema(fit_result['metrics']),
            artifacts=_collect_artifacts(out_path)
        ).model_dump()
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"错误: {e}")
        return IdentifyResult(
            status="failed",
            message=f"参数辨识失败: {str(e)}",
            error=error_msg
        ).model_dump()


def uncertainty(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs",
    n_bootstrap: int = 50,
    confidence_level: float = 0.95,
    current_threshold: float = 0.05,
    min_duration: float = 60.0
) -> dict:
    """
    执行不确定性分析
    
    Args:
        data_path: B0005.mat 数据文件路径
        cycle_number: 放电循环编号
        output_dir: 输出目录路径
        n_bootstrap: Bootstrap 重采样次数
        confidence_level: 置信水平
        current_threshold: 恒流段电流标准差阈值
        min_duration: 恒流段最小持续时间
    
    Returns:
        不确定性分析结果，包含置信区间、Bootstrap统计和敏感性排名
    """
    try:
        import numpy as np
        from src.ecm.loader import load_discharge_cc_segment
        from src.ecm.ocv import fit_ocv_curve
        from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage
        from src.identification.fit import fit_ecm_params
        from src.analysis.ci import analyze_parameter_uncertainty
        from src.analysis.bootstrap import residual_bootstrap, plot_bootstrap_results
        from src.analysis.sensitivity import local_sensitivity_analysis, plot_sensitivity_results
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print(f"\n{'='*60}")
        print(f"不确定性分析 - MCP Tool")
        print(f"{'='*60}")
        
        # 确保默认数据文件存在并解析路径
        _ensure_default_data_file()
        resolved_path = _resolve_data_path(data_path)
        
        if not resolved_path.exists():
            display_name = Path(data_path).name
            return UncertaintyResult(
                status="failed",
                message=f"数据文件 '{display_name}' 不存在。请在文件浏览器中上传数据文件到 data 文件夹。",
                error="FileNotFoundError"
            ).model_dump()
        
        # 使用解析后的路径
        data_path = str(resolved_path)
        
        out_path = Path(output_dir) / f"cycle_{cycle_number:03d}"
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据和辨识参数（与 identify 相同）
        t, i, v_measured, info = load_discharge_cc_segment(
            data_path, n=cycle_number,
            current_threshold=current_threshold,
            min_duration=min_duration
        )
        
        capacity_ah = info['total_capacity_ah']
        dt = np.gradient(t)
        charge_ah = np.cumsum(-i * dt) / 3600
        soc = 1.0 - charge_ah / capacity_ah
        soc = np.clip(soc, 0.0, 1.0)
        
        soc_samples = np.linspace(soc.min(), soc.max(), 15)
        v_samples = np.interp(soc_samples, soc[::-1], v_measured[::-1])
        ocv_func = fit_ocv_curve(soc_samples, v_samples, method='linear')
        
        x0 = np.array([1e-4, 1e-4, 1e6, 1e-4, 1e6])
        bounds = ([1e-4, 1e-4, 1e1, 1e-4, 1e1], [1e0, 1e0, 1e6, 1e0, 1e6])
        
        params_fitted, fit_result = fit_ecm_params(
            t, i, v_measured, soc, ocv_func,
            method='least_squares', x0=x0, bounds=bounds, verbose=0
        )
        
        # 置信区间分析
        def residual_func(theta):
            params_test = ECM2RCParams.from_array(theta)
            v_pred_test = simulate_voltage(t, i, soc, params_test, ocv_func)
            return v_pred_test - v_measured
        
        ci_results = analyze_parameter_uncertainty(
            residual_func=residual_func,
            params=params_fitted,
            residuals=fit_result['residuals'],
            confidence_level=confidence_level,
            use_stored_jacobian=None
        )
        
        # Bootstrap 分析
        bootstrap_results = residual_bootstrap(
            t, i, v_measured, soc, ocv_func,
            params_fitted=params_fitted,
            v_pred=fit_result['v_pred'],
            residuals=fit_result['residuals'],
            fit_function=fit_ecm_params,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=42,
            verbose=True
        )
        
        fig_bootstrap = plot_bootstrap_results(bootstrap_results, ci_results)
        fig_bootstrap.savefig(out_path / 'bootstrap_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig_bootstrap)
        
        # 敏感性分析
        sensitivity_results = local_sensitivity_analysis(
            t, i, soc, ocv_func,
            params=params_fitted,
            perturbation=0.01,
            v_baseline=fit_result['v_pred']
        )
        
        fig_sensitivity = plot_sensitivity_results(sensitivity_results)
        fig_sensitivity.savefig(out_path / 'sensitivity.png', dpi=150, bbox_inches='tight')
        plt.close(fig_sensitivity)
        
        # 保存 CI 表
        import csv
        ci_dict = ci_results['confidence_intervals']
        with open(out_path / 'ci_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['参数', '估计值', '标准差', '相对标准差(%)', 'CI下界', 'CI上界'])
            for idx, name in enumerate(ci_dict['param_names']):
                writer.writerow([
                    name,
                    f"{ci_dict['estimates'][idx]:.6e}",
                    f"{ci_dict['std_errors'][idx]:.6e}",
                    f"{ci_dict['relative_std'][idx]:.2f}",
                    f"{ci_dict['ci_lower'][idx]:.6e}",
                    f"{ci_dict['ci_upper'][idx]:.6e}"
                ])
        
        uncertainty_result = UncertaintyResult(
            confidence_intervals=_ci_to_schema(ci_results),
            sensitivity_ranking=_sensitivity_to_schema(sensitivity_results),
            bootstrap_summary=_bootstrap_to_schema(bootstrap_results)
        )
        
        return PipelineResult(
            status="success",
            message="不确定性分析完成",
            params=_params_to_schema(params_fitted),
            metrics=_metrics_to_schema(fit_result['metrics']),
            uncertainty=uncertainty_result,
            artifacts=_collect_artifacts(out_path)
        ).model_dump()
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"错误: {e}")
        return PipelineResult(
            status="failed",
            message=f"不确定性分析失败: {str(e)}",
            error=error_msg
        ).model_dump()


def pipeline(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs",
    n_bootstrap: int = 50,
    current_threshold: float = 0.05,
    min_duration: float = 60.0
) -> dict:
    """
    执行完整的 ECM 参数辨识与不确定性分析流程
    
    Args:
        data_path: B0005.mat 数据文件路径
        cycle_number: 放电循环编号
        output_dir: 输出目录路径
        n_bootstrap: Bootstrap 重采样次数
        current_threshold: 恒流段电流标准差阈值
        min_duration: 恒流段最小持续时间
    
    Returns:
        完整流程结果，包含参数、指标、不确定性分析和所有输出文件
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        from src.pipeline.run_pipeline import run_pipeline as _run_pipeline
        
        print(f"\n{'='*60}")
        print(f"ECM 完整流程 - MCP Tool")
        print(f"{'='*60}")
        
        # 确保默认数据文件存在并解析路径
        _ensure_default_data_file()
        resolved_path = _resolve_data_path(data_path)
        
        if not resolved_path.exists():
            display_name = Path(data_path).name
            return PipelineResult(
                status="failed",
                message=f"数据文件 '{display_name}' 不存在。请在文件浏览器中上传数据文件到 data 文件夹。",
                error="FileNotFoundError"
            ).model_dump()
        
        # 使用解析后的路径
        data_path = str(resolved_path)
        
        # 调用现有的 run_pipeline
        results = _run_pipeline(
            mat_path=data_path,
            cycle_number=cycle_number,
            output_base_dir=output_dir,
            current_threshold=current_threshold,
            min_duration=min_duration,
            n_bootstrap=n_bootstrap,
            verbose=True
        )
        
        # 转换结果
        params = results['params']
        metrics = results['metrics']
        ci_results = results['ci_results']
        bootstrap_results = results['bootstrap_results']
        sensitivity_results = results['sensitivity_results']
        out_path = results['output_dir']
        
        uncertainty_result = UncertaintyResult(
            confidence_intervals=_ci_to_schema(ci_results),
            sensitivity_ranking=_sensitivity_to_schema(sensitivity_results),
            bootstrap_summary=_bootstrap_to_schema(bootstrap_results)
        )
        
        return PipelineResult(
            status="success",
            message="完整流程执行成功",
            params=_params_to_schema(params),
            metrics=_metrics_to_schema(metrics),
            uncertainty=uncertainty_result,
            artifacts=_collect_artifacts(out_path)
        ).model_dump()
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"错误: {e}")
        return PipelineResult(
            status="failed",
            message=f"流程执行失败: {str(e)}",
            error=error_msg
        ).model_dump()


def register_tools(mcp):
    """
    注册所有 MCP 工具到指定的 MCP 服务实例
    
    Args:
        mcp: FastMCP 实例
    """
    # 注册 identify 工具
    mcp.tool(
        name="identify",
        description="ECM 参数辨识：对电池放电数据进行二阶RC等效电路模型参数辨识，返回 R0/R1/C1/R2/C2 五个参数及拟合指标"
    )(identify)
    
    # 注册 uncertainty 工具
    mcp.tool(
        name="uncertainty",
        description="不确定性分析：对 ECM 参数进行置信区间、Bootstrap 重采样和敏感性分析"
    )(uncertainty)
    
    # 注册 pipeline 工具
    mcp.tool(
        name="pipeline",
        description="完整流程：一键执行数据加载、参数辨识、置信区间、Bootstrap和敏感性分析的完整流程"
    )(pipeline)
    
    print("[INFO] Registered MCP tools: identify, uncertainty, pipeline")
