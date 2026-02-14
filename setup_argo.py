"""
Bohrium Argo 服务配置脚本
在节点里运行此脚本以配置 dflow 连接到 Argo

使用方法：
=========
1. 修改下面的配置信息（填入你的 Bohrium 账号信息）
2. 在节点里运行：python setup_argo.py
3. 验证配置：python setup_argo.py --test
4. 如果成功，可以提交工作流：python dflow_example.py
"""

import os
import sys
from pathlib import Path


def setup_dflow_config(
    bohrium_username: str,
    bohrium_password: str,
    bohrium_project_id: int,
    argo_token: str = ""
):
    """
    配置 dflow 连接到 Bohrium Argo 服务
    
    参数:
        bohrium_username: Bohrium 账号邮箱
        bohrium_password: Bohrium 密码
        bohrium_project_id: Bohrium 项目 ID（整数）
        argo_token: Argo token（通常为空字符串）
    """
    try:
        from dflow import config, s3_config
        from dflow.plugins import bohrium
        from dflow.plugins.bohrium import TiefblueClient
    except ImportError:
        print("❌ 错误：dflow 未安装", flush=True)
        print("请运行：pip install pydflow", flush=True)
        return False
    
    print("="*70, flush=True)
    print("配置 Bohrium Argo 服务", flush=True)
    print("="*70, flush=True)
    
    # 配置 Argo 服务器
    config["host"] = "http://workflows.deepmodeling.com"
    config["token"] = argo_token
    config["k8s_api_server"] = "http://workflows.deepmodeling.com"
    config["namespace"] = "argo"
    print(f"✓ Argo 服务器: {config['host']}", flush=True)
    print(f"✓ 命名空间: {config['namespace']}", flush=True)
    
    # 配置 Bohrium 认证
    bohrium.config["username"] = bohrium_username
    bohrium.config["password"] = bohrium_password
    bohrium.config["project_id"] = bohrium_project_id
    print(f"✓ Bohrium 用户: {bohrium_username}", flush=True)
    print(f"✓ 项目 ID: {bohrium_project_id}", flush=True)
    
    # 配置 S3 存储（用于 artifact）
    s3_config["repo_key"] = "oss-bohrium"
    s3_config["storage_client"] = TiefblueClient()
    print(f"✓ 存储后端: oss-bohrium (Tiefblue)", flush=True)
    
    print("="*70, flush=True)
    print("✅ 配置完成！", flush=True)
    print("="*70, flush=True)
    
    return True


def test_connection():
    """
    测试 Argo 连接是否正常
    """
    try:
        from dflow import config
        import requests
    except ImportError as e:
        print(f"❌ 导入失败: {e}", flush=True)
        return False
    
    print("\n" + "="*70, flush=True)
    print("测试 Argo 连接", flush=True)
    print("="*70, flush=True)
    
    # 检查配置
    if not config.get("host"):
        print("❌ 配置未初始化，请先运行 setup_dflow_config()", flush=True)
        return False
    
    print(f"正在连接: {config['host']}", flush=True)
    
    # 尝试访问 Argo API
    try:
        # 简单的健康检查
        response = requests.get(
            f"{config['host']}/",
            timeout=10
        )
        if response.status_code < 500:
            print("✅ Argo 服务器可访问", flush=True)
            print(f"   状态码: {response.status_code}", flush=True)
            return True
        else:
            print(f"⚠️  服务器返回错误: {response.status_code}", flush=True)
            return False
    except Exception as e:
        print(f"❌ 连接失败: {e}", flush=True)
        print("   可能原因：", flush=True)
        print("   1. 网络不通", flush=True)
        print("   2. 服务器地址错误", flush=True)
        print("   3. 防火墙阻止", flush=True)
        return False


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="配置 Bohrium Argo 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：

1. 配置 Argo（交互式）：
   python setup_argo.py

2. 配置 Argo（命令行参数）：
   python setup_argo.py --username your@email.com --password yourpass --project-id 12345

3. 测试连接：
   python setup_argo.py --test

4. 使用环境变量配置：
   export BOHRIUM_USERNAME="your@email.com"
   export BOHRIUM_PASSWORD="your-password"
   export BOHRIUM_PROJECT_ID="12345"
   python setup_argo.py --env
        """
    )
    
    parser.add_argument(
        '--username', '-u',
        type=str,
        help='Bohrium 账号邮箱'
    )
    parser.add_argument(
        '--password', '-p',
        type=str,
        help='Bohrium 密码'
    )
    parser.add_argument(
        '--project-id', '-i',
        type=int,
        help='Bohrium 项目 ID'
    )
    parser.add_argument(
        '--token', '-t',
        type=str,
        default="",
        help='Argo token（可选，通常为空）'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试 Argo 连接'
    )
    parser.add_argument(
        '--env',
        action='store_true',
        help='从环境变量读取配置'
    )
    
    args = parser.parse_args()
    
    # 只测试连接
    if args.test:
        success = test_connection()
        sys.exit(0 if success else 1)
    
    # 从环境变量读取
    if args.env:
        username = os.environ.get('BOHRIUM_USERNAME')
        password = os.environ.get('BOHRIUM_PASSWORD')
        project_id = os.environ.get('BOHRIUM_PROJECT_ID')
        
        if not all([username, password, project_id]):
            print("❌ 环境变量不完整", flush=True)
            print("需要设置：BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID", flush=True)
            sys.exit(1)
        
        project_id = int(project_id)
    
    # 从命令行参数读取
    elif all([args.username, args.password, args.project_id]):
        username = args.username
        password = args.password
        project_id = args.project_id
    
    # 交互式输入
    else:
        print("\n" + "="*70, flush=True)
        print("Bohrium Argo 配置向导", flush=True)
        print("="*70, flush=True)
        print("\n请输入您的 Bohrium 账号信息：\n", flush=True)
        
        username = input("Bohrium 邮箱: ").strip()
        password = input("Bohrium 密码: ").strip()
        project_id_str = input("项目 ID: ").strip()
        
        try:
            project_id = int(project_id_str)
        except ValueError:
            print("❌ 项目 ID 必须是整数", flush=True)
            sys.exit(1)
    
    # 执行配置
    success = setup_dflow_config(
        bohrium_username=username,
        bohrium_password=password,
        bohrium_project_id=project_id,
        argo_token=args.token
    )
    
    if success:
        # 测试连接
        print("\n正在测试连接...", flush=True)
        test_connection()
        
        print("\n" + "="*70, flush=True)
        print("✅ 配置成功！", flush=True)
        print("="*70, flush=True)
        print("", flush=True)
        print("重要：请设置环境变量（用于后续脚本）", flush=True)
        print("----------------------------------------------------------------------", flush=True)
        print(f'export BOHRIUM_USERNAME="{username}"', flush=True)
        print(f'export BOHRIUM_PASSWORD="{password}"', flush=True)
        print(f'export BOHRIUM_PROJECT_ID="{project_id}"', flush=True)
        print("----------------------------------------------------------------------", flush=True)
        print("", flush=True)
        print("或者将上面的命令添加到 ~/.bashrc 以永久保存：", flush=True)
        print("----------------------------------------------------------------------", flush=True)
        print(f'echo \'export BOHRIUM_USERNAME="{username}"\' >> ~/.bashrc', flush=True)
        print(f'echo \'export BOHRIUM_PASSWORD="{password}"\' >> ~/.bashrc', flush=True)
        print(f'echo \'export BOHRIUM_PROJECT_ID="{project_id}"\' >> ~/.bashrc', flush=True)
        print('source ~/.bashrc', flush=True)
        print("----------------------------------------------------------------------", flush=True)
        print("", flush=True)
        print("下一步：", flush=True)
        print("="*70, flush=True)
        print("1. 复制上面的 export 命令并执行", flush=True)
        print("2. 修改 dflow_workflow.py 中的 DEFAULT_IMAGE 变量（如果需要）", flush=True)
        print("3. 运行工作流：python dflow_example.py", flush=True)
        print("4. 查看 Argo UI：http://workflows.deepmodeling.com", flush=True)
        print("="*70, flush=True)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
