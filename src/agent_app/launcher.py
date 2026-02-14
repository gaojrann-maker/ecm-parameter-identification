"""
统一启动脚本
支持两种模式：
1. Agent 模式：批处理任务 + 健康检查（原 app.py）
2. Web 模式：Gradio Web 界面（新 web_app.py）
"""

import os
import sys
from pathlib import Path

# 兼容两种运行方式
try:
    from src.agent_app import app, web_app
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.agent_app import app, web_app


def main():
    """
    根据环境变量选择启动模式
    """
    mode = os.environ.get('ECM_MODE', 'web').lower()
    
    print("="*70, flush=True)
    print(f"ECM IDENTIFICATION AGENT - {mode.upper()} MODE", flush=True)
    print("="*70, flush=True)
    
    if mode == 'web':
        print("[INFO] Starting Web interface mode (Gradio)", flush=True)
        web_app.main()
    elif mode == 'agent':
        print("[INFO] Starting Agent mode (batch processing)", flush=True)
        # 使用原来的 app.py 逻辑
        # 但这里需要直接运行 app.py 的 __main__ 部分
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app_main",
            Path(__file__).parent / "app.py"
        )
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
    else:
        print(f"[ERROR] Unknown mode: {mode}", flush=True)
        print("[INFO] Valid modes: 'web' or 'agent'", flush=True)
        print("[INFO] Set environment variable: ECM_MODE=web or ECM_MODE=agent", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
