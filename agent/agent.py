"""
ECM 参数辨识 Agent
使用 Google ADK + LiteLLM 创建可对话的智能 Agent
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# 配置日志 - 只在需要时
logger = logging.getLogger(__name__)

# 模块路径
_MODULE_DIR = Path(__file__).parent
_PACKAGE_ROOT = _MODULE_DIR.parent

# 延迟导入的全局变量
_Agent = None
_LiteLlm = None


def _lazy_import():
    """延迟导入 Google ADK，避免模块加载时的网络请求"""
    global _Agent, _LiteLlm
    if _Agent is None:
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        _Agent = Agent
        _LiteLlm = LiteLlm
    return _Agent, _LiteLlm


def _load_env():
    """加载环境变量"""
    from dotenv import load_dotenv
    env_file = _PACKAGE_ROOT / '.env'
    if env_file.exists():
        load_dotenv(env_file)


# Agent 指令（系统提示词）
AGENT_INSTRUCTION = """你是一个专业的电池等效电路模型（ECM）参数辨识助手。

你的能力：
1. **参数辨识 (identify)**：对电池放电数据进行二阶RC等效电路模型参数辨识
   - 输入：数据文件路径、放电循环编号
   - 输出：五个ECM参数 (R0, R1, C1, R2, C2) 和拟合质量指标 (RMSE, R², MAE)

2. **不确定性分析 (uncertainty)**：评估辨识参数的可信度
   - 置信区间分析：基于雅可比矩阵计算95%置信区间
   - Bootstrap重采样：通过残差重采样估计参数分布
   - 敏感性分析：评估各参数对模型输出的影响

3. **完整流程 (pipeline)**：一键执行全部分析
   - 包括数据加载、参数辨识、所有不确定性分析
   - 生成完整的结果文件和可视化图表

使用说明：
- 默认数据路径：data/B0005.mat（NASA B0005电池数据）
- 默认输出目录：outputs/
- 可以指定不同的放电循环编号进行分析
- 用户可以上传自己的 .mat 数据文件

当用户请求进行ECM分析时，请主动使用相应的工具，并清晰地解释结果。
如果用户上传了数据文件，请使用上传文件的路径进行分析。
"""


def identify_ecm_params(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs"
) -> str:
    """
    执行 ECM 参数辨识
    
    Args:
        data_path: B0005.mat 数据文件路径
        cycle_number: 放电循环编号（从1开始）
        output_dir: 输出目录路径
    
    Returns:
        辨识结果描述
    """
    import json
    from src.mcp_server.tools import identify, _ensure_default_data_file
    
    _ensure_default_data_file()
    
    result = identify(
        data_path=data_path,
        cycle_number=cycle_number,
        output_dir=output_dir
    )
    
    if result['status'] == 'success':
        params = result['params']
        metrics = result['metrics']
        return f"""ECM 参数辨识完成！

**辨识参数：**
- R0 (欧姆内阻): {params['R0']:.6e} Ω
- R1 (第一RC电阻): {params['R1']:.6e} Ω  
- C1 (第一RC电容): {params['C1']:.2f} F
- R2 (第二RC电阻): {params['R2']:.6e} Ω
- C2 (第二RC电容): {params['C2']:.2f} F

**拟合质量：**
- RMSE: {metrics['RMSE']:.6f} V
- R²: {metrics['R2']:.6f}
- MAE: {metrics['MAE']:.6f} V

**输出文件：**
{json.dumps(result.get('artifacts', {}), indent=2, ensure_ascii=False)}
"""
    else:
        return f"辨识失败: {result.get('message', '未知错误')}"


def analyze_uncertainty(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs",
    n_bootstrap: int = 50
) -> str:
    """
    执行不确定性分析
    
    Args:
        data_path: 数据文件路径
        cycle_number: 放电循环编号
        output_dir: 输出目录路径
        n_bootstrap: Bootstrap 重采样次数
    
    Returns:
        不确定性分析结果描述
    """
    import json
    from src.mcp_server.tools import uncertainty, _ensure_default_data_file
    
    _ensure_default_data_file()
    
    result = uncertainty(
        data_path=data_path,
        cycle_number=cycle_number,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap
    )
    
    if result['status'] == 'success':
        unc = result.get('uncertainty', {})
        ci_list = unc.get('confidence_intervals', [])
        sens_list = unc.get('sensitivity_ranking', [])
        
        ci_text = "\n".join([
            f"  - {ci['param_name']}: {ci['estimate']:.6e} ± {ci['std_error']:.6e}"
            for ci in ci_list
        ])
        
        sens_text = "\n".join([
            f"  {s['rank']}. {s['param_name']}"
            for s in sens_list
        ])
        
        return f"""不确定性分析完成！

**95% 置信区间：**
{ci_text}

**敏感性排名：**
{sens_text}
"""
    else:
        return f"分析失败: {result.get('message', '未知错误')}"


def run_full_pipeline(
    data_path: str = "data/B0005.mat",
    cycle_number: int = 1,
    output_dir: str = "outputs",
    n_bootstrap: int = 50
) -> str:
    """
    执行完整的 ECM 参数辨识与不确定性分析流程
    
    Args:
        data_path: 数据文件路径
        cycle_number: 放电循环编号
        output_dir: 输出目录路径
        n_bootstrap: Bootstrap 重采样次数
    
    Returns:
        完整流程结果描述
    """
    import json
    from src.mcp_server.tools import pipeline, _ensure_default_data_file
    
    _ensure_default_data_file()
    
    result = pipeline(
        data_path=data_path,
        cycle_number=cycle_number,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap
    )
    
    if result['status'] == 'success':
        params = result['params']
        metrics = result['metrics']
        
        return f"""完整流程执行成功！

**ECM 参数：**
- R0: {params['R0']:.6e} Ω
- R1: {params['R1']:.6e} Ω, C1: {params['C1']:.2f} F
- R2: {params['R2']:.6e} Ω, C2: {params['C2']:.2f} F

**拟合质量：**
- RMSE: {metrics['RMSE']:.6f} V, R²: {metrics['R2']:.6f}
"""
    else:
        return f"流程执行失败: {result.get('message', '未知错误')}"


def list_available_cycles(data_path: str = "data/B0005.mat") -> str:
    """列出数据文件中可用的放电循环"""
    try:
        from src.mcp_server.tools import _resolve_data_path, _ensure_default_data_file
        import h5py
        
        _ensure_default_data_file()
        resolved_path = _resolve_data_path(data_path)
        
        if not resolved_path.exists():
            return f"数据文件不存在。请先上传数据文件。"
        
        try:
            with h5py.File(str(resolved_path), 'r') as f:
                if 'cycle' in f:
                    n_cycles = len(f['cycle']['type'][()])
                    return f"数据文件包含 {n_cycles} 个循环。"
        except:
            pass
        
        return f"数据文件已就绪。"
    
    except Exception as e:
        return f"读取数据文件时出错: {str(e)}"


def create_agent(ak: str = None, app_key: str = None, project_id: int = None):
    """
    创建 ECM 参数辨识 Agent - SDK 标准接口
    
    Args:
        ak: Bohrium Access Key（可选）
        app_key: Bohrium App Key（可选）
        project_id: Bohrium Project ID（可选）
    
    Returns:
        配置好的 Agent 实例
    """
    # 加载环境变量
    _load_env()
    
    # 延迟导入 ADK
    Agent, LiteLlm = _lazy_import()
    
    # 从环境变量获取 project_id（如果未传入）
    if project_id is None:
        env_project_id = os.getenv('BOHR_PROJECT_ID') or os.getenv('PROJECT_ID')
        if env_project_id:
            try:
                project_id = int(env_project_id)
            except ValueError:
                project_id = 1
        else:
            project_id = 1
    
    # 获取模型配置
    model_type = os.getenv('MODEL', 'deepseek/deepseek-chat')
    
    logger.info(f"创建 Agent: model={model_type}, project_id={project_id}")
    
    # 保存认证信息
    agent_project_id = project_id
    
    def show_agent_info() -> str:
        """显示 Agent 信息"""
        return f"""ECM 参数辨识 Agent v1.0

Project ID: {agent_project_id}

可用功能:
- identify_ecm_params: ECM 参数辨识
- analyze_uncertainty: 不确定性分析
- run_full_pipeline: 完整分析流程
- list_available_cycles: 查看可用循环
"""
    
    # 创建 Agent
    agent = Agent(
        name="ecm_identification_agent",
        model=LiteLlm(model=model_type),
        instruction=AGENT_INSTRUCTION,
        tools=[
            show_agent_info,
            identify_ecm_params,
            analyze_uncertainty,
            run_full_pipeline,
            list_available_cycles
        ]
    )
    
    logger.info("Agent 创建成功")
    return agent


# 导出
__all__ = ['create_agent']
