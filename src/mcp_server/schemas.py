"""
MCP 服务的 Pydantic 数据模型定义
用于结构化输入参数和输出结果
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class IdentifyInput(BaseModel):
    """参数辨识输入参数"""
    data_path: str = Field(
        default="/data/B0005.mat",
        description="B0005.mat 数据文件路径"
    )
    cycle_number: int = Field(
        default=1,
        description="放电循环编号（从1开始）",
        ge=1
    )
    output_dir: str = Field(
        default="outputs",
        description="输出目录路径"
    )
    current_threshold: float = Field(
        default=0.05,
        description="恒流段电流标准差阈值"
    )
    min_duration: float = Field(
        default=60.0,
        description="恒流段最小持续时间（秒）"
    )


class UncertaintyInput(BaseModel):
    """不确定性分析输入参数"""
    data_path: str = Field(
        default="/data/B0005.mat",
        description="B0005.mat 数据文件路径"
    )
    cycle_number: int = Field(
        default=1,
        description="放电循环编号（从1开始）",
        ge=1
    )
    output_dir: str = Field(
        default="outputs",
        description="输出目录路径"
    )
    n_bootstrap: int = Field(
        default=50,
        description="Bootstrap 重采样次数",
        ge=10,
        le=1000
    )
    confidence_level: float = Field(
        default=0.95,
        description="置信水平（如0.95表示95%置信区间）",
        gt=0,
        lt=1
    )
    current_threshold: float = Field(
        default=0.05,
        description="恒流段电流标准差阈值"
    )
    min_duration: float = Field(
        default=60.0,
        description="恒流段最小持续时间（秒）"
    )


class PipelineInput(BaseModel):
    """完整流程输入参数"""
    data_path: str = Field(
        default="/data/B0005.mat",
        description="B0005.mat 数据文件路径"
    )
    cycle_number: int = Field(
        default=1,
        description="放电循环编号（从1开始）",
        ge=1
    )
    output_dir: str = Field(
        default="outputs",
        description="输出目录路径"
    )
    n_bootstrap: int = Field(
        default=50,
        description="Bootstrap 重采样次数",
        ge=10,
        le=1000
    )
    current_threshold: float = Field(
        default=0.05,
        description="恒流段电流标准差阈值"
    )
    min_duration: float = Field(
        default=60.0,
        description="恒流段最小持续时间（秒）"
    )


class ECMParams(BaseModel):
    """ECM 二阶RC模型参数"""
    R0: float = Field(description="欧姆内阻 (Ω)")
    R1: float = Field(description="第一RC环节电阻 (Ω)")
    C1: float = Field(description="第一RC环节电容 (F)")
    R2: float = Field(description="第二RC环节电阻 (Ω)")
    C2: float = Field(description="第二RC环节电容 (F)")
    tau1: float = Field(description="第一RC时间常数 τ1 = R1×C1 (s)")
    tau2: float = Field(description="第二RC时间常数 τ2 = R2×C2 (s)")


class FitMetrics(BaseModel):
    """拟合质量指标"""
    RMSE: float = Field(description="均方根误差 (V)")
    MAE: float = Field(description="平均绝对误差 (V)")
    R2: float = Field(description="决定系数 R²")
    MSE: float = Field(description="均方误差 (V²)")
    MAPE: float = Field(description="平均绝对百分比误差 (%)")


class ConfidenceInterval(BaseModel):
    """单个参数的置信区间"""
    param_name: str = Field(description="参数名称")
    estimate: float = Field(description="参数估计值")
    std_error: float = Field(description="标准误差")
    relative_std: float = Field(description="相对标准差 (%)")
    ci_lower: float = Field(description="置信区间下界")
    ci_upper: float = Field(description="置信区间上界")


class SensitivityInfo(BaseModel):
    """敏感性分析结果"""
    param_name: str = Field(description="参数名称")
    rms_sensitivity: float = Field(description="RMS敏感性指标")
    max_sensitivity: float = Field(description="最大敏感性")
    rank: int = Field(description="敏感性排名（1为最敏感）")


class BootstrapSummary(BaseModel):
    """Bootstrap分析摘要"""
    n_bootstrap: int = Field(description="重采样次数")
    success_rate: float = Field(description="成功率 (%)")
    param_means: Dict[str, float] = Field(description="参数均值")
    param_stds: Dict[str, float] = Field(description="参数标准差")
    ci_lower: Dict[str, float] = Field(description="Bootstrap置信区间下界")
    ci_upper: Dict[str, float] = Field(description="Bootstrap置信区间上界")


class UncertaintyResult(BaseModel):
    """不确定性分析完整结果"""
    confidence_intervals: List[ConfidenceInterval] = Field(
        description="各参数的置信区间"
    )
    sensitivity_ranking: List[SensitivityInfo] = Field(
        description="敏感性分析结果（按敏感性排序）"
    )
    bootstrap_summary: Optional[BootstrapSummary] = Field(
        default=None,
        description="Bootstrap分析摘要"
    )


class IdentifyResult(BaseModel):
    """参数辨识结果"""
    status: str = Field(description="执行状态: success/failed")
    message: str = Field(description="状态消息")
    params: Optional[ECMParams] = Field(default=None, description="辨识的ECM参数")
    metrics: Optional[FitMetrics] = Field(default=None, description="拟合指标")
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="输出文件路径（绝对路径）"
    )
    error: Optional[str] = Field(default=None, description="错误信息（如果失败）")


class PipelineResult(BaseModel):
    """完整流程结果"""
    status: str = Field(description="执行状态: success/failed")
    message: str = Field(description="状态消息")
    params: Optional[ECMParams] = Field(default=None, description="辨识的ECM参数")
    metrics: Optional[FitMetrics] = Field(default=None, description="拟合指标")
    uncertainty: Optional[UncertaintyResult] = Field(
        default=None,
        description="不确定性分析结果"
    )
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="所有输出文件路径（绝对路径）"
    )
    error: Optional[str] = Field(default=None, description="错误信息（如果失败）")
