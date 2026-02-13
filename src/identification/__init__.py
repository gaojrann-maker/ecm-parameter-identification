"""
ECM (Equivalent Circuit Model) 模块

包含电池等效电路模型相关的功能：
- loader: 数据加载和预处理
- ocv: OCV-SOC 特性拟合
- ecm2rc: 二阶RC等效电路模型
- metrics: 模型评估指标
"""

from .loader import (
    load_b0005_cycles,
    list_discharge_indices,
    get_nth_discharge,
    extract_constant_current_segment,
    load_discharge_cc_segment
)

from .ocv import (
    calculate_soc,
    extract_rest_segments,
    fit_ocv_curve,
    fit_ocv_from_full_cycle
)

from .ecm2rc import (
    ECM2RCParams,
    check_params_positive,
    compute_polarization_voltages,
    simulate_voltage,
    simulate_voltage_with_details,
    get_initial_params_guess,
    validate_params_physical
)

from .metrics import (
    rmse,
    mae,
    max_abs_error,
    mse,
    r2_score,
    mape,
    calculate_all_metrics,
    print_metrics,
    relative_error,
    residuals
)

__all__ = [
    # loader 模块
    'load_b0005_cycles',
    'list_discharge_indices',
    'get_nth_discharge',
    'extract_constant_current_segment',
    'load_discharge_cc_segment',
    # ocv 模块
    'calculate_soc',
    'extract_rest_segments',
    'fit_ocv_curve',
    'fit_ocv_from_full_cycle',
    # ecm2rc 模块
    'ECM2RCParams',
    'check_params_positive',
    'compute_polarization_voltages',
    'simulate_voltage',
    'simulate_voltage_with_details',
    'get_initial_params_guess',
    'validate_params_physical',
    # metrics 模块
    'rmse',
    'mae',
    'max_abs_error',
    'mse',
    'r2_score',
    'mape',
    'calculate_all_metrics',
    'print_metrics',
    'relative_error',
    'residuals'
]

__version__ = '0.4.0'
