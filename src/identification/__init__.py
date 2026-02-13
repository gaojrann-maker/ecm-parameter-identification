"""
参数辨识模块

包含ECM模型参数辨识的功能：
- fit: 参数辨识算法（最小二乘、全局优化）
"""

from .fit import (
    ECMParameterIdentification,
    fit_ecm_params
)

__all__ = [
    'ECMParameterIdentification',
    'fit_ecm_params'
]

__version__ = '0.1.0'
