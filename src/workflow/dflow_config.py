"""
dflow 配置模块
在提交工作流前自动加载 Bohrium Argo 配置
"""

import os


def init_dflow_config():
    """
    初始化 dflow 配置
    从环境变量读取配置信息
    """
    # 读取环境变量
    username = os.environ.get('BOHRIUM_USERNAME')
    password = os.environ.get('BOHRIUM_PASSWORD')
    project_id = os.environ.get('BOHRIUM_PROJECT_ID')
    
    if not all([username, password, project_id]):
        print("⚠️  警告：未检测到 Bohrium 配置环境变量", flush=True)
        print("请先设置环境变量或运行 setup_argo.py", flush=True)
        print("", flush=True)
        print("快速设置：", flush=True)
        print('  export BOHRIUM_USERNAME="your@email.com"', flush=True)
        print('  export BOHRIUM_PASSWORD="your-password"', flush=True)
        print('  export BOHRIUM_PROJECT_ID="12345"', flush=True)
        print("", flush=True)
        return False
    
    try:
        from dflow import config, s3_config
        from dflow.plugins import bohrium
        from dflow.plugins.bohrium import TiefblueClient
    except ImportError as e:
        print(f"❌ 导入 dflow 失败: {e}", flush=True)
        return False
    
    # 配置 Argo 服务器
    config["host"] = "http://workflows.deepmodeling.com"
    config["token"] = os.environ.get('ARGO_TOKEN', "")
    config["k8s_api_server"] = "http://workflows.deepmodeling.com"
    config["namespace"] = "argo"
    
    # 配置 Bohrium 认证
    bohrium.config["username"] = username
    bohrium.config["password"] = password
    bohrium.config["project_id"] = int(project_id)
    
    # 配置 S3 存储（关键！）
    s3_config["repo_key"] = "oss-bohrium"
    s3_config["storage_client"] = TiefblueClient()
    
    print("✓ dflow 配置已加载", flush=True)
    print(f"  Bohrium 用户: {username}", flush=True)
    print(f"  项目 ID: {project_id}", flush=True)
    
    return True


# 自动初始化（导入时执行）
if __name__ != "__main__":
    # 只在被导入时自动初始化
    # 如果直接运行此文件，不自动初始化
    init_dflow_config()
