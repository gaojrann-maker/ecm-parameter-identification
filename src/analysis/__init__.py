"""
不确定性分析模块

包含参数不确定性和敏感性分析的功能：
- ci: 基于雅可比矩阵的置信区间
- bootstrap: Bootstrap重采样分析
- sensitivity: 敏感性分析
"""

from .ci import (
    compute_jacobian_numerical,
    compute_parameter_covariance,
    compute_confidence_intervals,
    compute_correlation_matrix,
    analyze_parameter_uncertainty,
    print_ci_table
)

from .bootstrap import (
    residual_bootstrap,
    compare_ci_methods
)

from .sensitivity import (
    local_sensitivity_analysis,
    compute_voltage_impact,
    relative_sensitivity_analysis,
    sensitivity_summary
)

__all__ = [
    # ci 模块
    'compute_jacobian_numerical',
    'compute_parameter_covariance',
    'compute_confidence_intervals',
    'compute_correlation_matrix',
    'analyze_parameter_uncertainty',
    'print_ci_table',
    # bootstrap 模块
    'residual_bootstrap',
    'compare_ci_methods',
    # sensitivity 模块
    'local_sensitivity_analysis',
    'compute_voltage_impact',
    'relative_sensitivity_analysis',
    'sensitivity_summary'
]

__version__ = '0.1.0'
