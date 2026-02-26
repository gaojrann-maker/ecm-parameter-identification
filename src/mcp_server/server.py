"""
ECM Parameter Identification MCP Server
使用 bohr-agent-sdk / mcp 提供 MCP 标准服务

启动方式:
    python -m src.mcp_server.server
    或
    python main.py
"""

import os
import sys
import signal
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务实例
mcp = FastMCP(
    name="ecm-identification-agent",
    host="0.0.0.0",
    port=50001,
    log_level="INFO"
)

# 导入并注册工具
from src.mcp_server.tools import register_tools
register_tools(mcp)


def signal_handler(sig, frame):
    """优雅关闭处理"""
    print("\n[INFO] Shutting down MCP server...")
    sys.exit(0)


def main():
    """启动 MCP Server"""
    print("="*60)
    print("ECM Parameter Identification MCP Server")
    print("="*60)
    print()
    print("可用工具 (MCP Tools):")
    print("  - identify:    ECM 参数辨识")
    print("  - uncertainty: 不确定性分析（CI/Bootstrap/Sensitivity）")
    print("  - pipeline:    完整流程（一键全分析）")
    print()
    print(f"传输模式: SSE")
    print(f"监听地址: 0.0.0.0:50001")
    print("="*60)
    print()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动 MCP Server
    print("[INFO] Starting MCP server with SSE transport on port 50001...")
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
