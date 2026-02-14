"""
dflow 工作流提交脚本
在玻尔平台节点里运行，向 Argo 提交 ECM 参数辨识工作流

使用方法：
  1. 先运行 setup_argo.py 配置 Argo 连接
  2. python dflow_example.py submit         提交工作流
  3. python dflow_example.py monitor        提交并监控
  4. python dflow_example.py query <id>     查询状态
"""

# 必须先初始化 dflow 配置
from src.workflow.dflow_config import init_dflow_config

if not init_dflow_config():
    print("dflow 配置失败，请先运行 setup_argo.py 或设置环境变量", flush=True)
    exit(1)

from src.workflow.dflow_workflow import submit_workflow


def cmd_submit():
    """提交工作流"""
    wf = submit_workflow(
        data_path="/opt/ECM-dflow/data/B0005.mat",
        cycle_n=1,
        n_bootstrap=50
    )
    print(f"\n工作流 ID: {wf.id}", flush=True)
    return wf


def cmd_monitor():
    """提交并监控工作流"""
    import time

    wf = cmd_submit()
    print("\n监控工作流执行...", flush=True)

    while True:
        status = wf.query_status()
        print(f"  状态: {status}", flush=True)
        if status in ['Succeeded', 'Failed', 'Error']:
            break
        time.sleep(10)

    if status == 'Succeeded':
        print("\n工作流执行成功!", flush=True)
    else:
        print(f"\n工作流执行失败: {status}", flush=True)


def cmd_query(workflow_id: str):
    """查询工作流状态"""
    from dflow import Workflow

    wf = Workflow(id=workflow_id)
    status = wf.query_status()
    print(f"工作流 {workflow_id}: {status}", flush=True)

    if status == 'Succeeded':
        steps = wf.query_step()
        for step in steps:
            print(f"  {step.name}: {step.phase}", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:", flush=True)
        print("  python dflow_example.py submit       提交工作流", flush=True)
        print("  python dflow_example.py monitor      提交并监控", flush=True)
        print("  python dflow_example.py query <id>   查询状态", flush=True)
        exit(0)

    command = sys.argv[1]

    if command == "submit":
        cmd_submit()
    elif command == "monitor":
        cmd_monitor()
    elif command == "query" and len(sys.argv) > 2:
        cmd_query(sys.argv[2])
    else:
        print(f"未知命令: {command}", flush=True)
