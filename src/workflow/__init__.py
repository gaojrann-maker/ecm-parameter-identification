"""
Workflow 模块：三步骤工作流
Step1: DataReadOp - 数据读取和预处理
Step2: IdentifyOp - 参数辨识
Step3: UncertaintyOp - 不确定性分析

支持两种执行方式：
1. 直接调用 (ops.py + run_workflow.py)
2. dflow 工作流 (dflow_workflow.py)
"""

from .ops import DataReadOp, IdentifyOp, UncertaintyOp
from .run_workflow import run_workflow

# dflow 工作流（可选，需要安装 dflow）
try:
    from .dflow_workflow import (
        DataReadOP,
        IdentifyOP,
        UncertaintyOP,
        build_ecm_workflow,
        submit_workflow
    )
    __all__ = [
        'DataReadOp',
        'IdentifyOp', 
        'UncertaintyOp',
        'run_workflow',
        'DataReadOP',
        'IdentifyOP',
        'UncertaintyOP',
        'build_ecm_workflow',
        'submit_workflow',
    ]
except ImportError:
    # 如果没有安装 dflow，只导出基本功能
    __all__ = [
        'DataReadOp',
        'IdentifyOp', 
        'UncertaintyOp',
        'run_workflow'
    ]

__version__ = '1.0.0'

