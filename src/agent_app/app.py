"""
Bohrium Agent 应用入口
功能：封装 ECM 参数辨识流程为 Bohrium Agent 应用
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer

# 兼容两种运行方式：
# 1. python -m src.agent_app.app (Bohrium平台)
# 2. python src/agent_app/app.py (本地直接运行)
try:
    from src.pipeline.run_pipeline import run_pipeline
except ModuleNotFoundError:
    # 如果作为脚本直接运行，添加项目根目录到路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.pipeline.run_pipeline import run_pipeline


class HealthHandler(BaseHTTPRequestHandler):
    """
    健康检查 HTTP 处理器
    用于 Bohrium 平台在 50001 端口做健康检查
    """
    def do_GET(self):
        """处理 GET 请求"""
        if self.path in ["/", "/health", "/healthz"]:
            # 返回健康状态
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "service": "ECM Parameter Identification Agent",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """禁用默认的访问日志（避免刷屏）"""
        pass


def start_http_server(port: int = 50001):
    """
    启动 HTTP 健康检查服务器
    
    参数:
        port: 监听端口（默认 50001）
    """
    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        print(f"[INFO] Health check server bound to port {port}", flush=True)
        server.serve_forever()
    except Exception as e:
        print(f"[ERROR] Failed to start health check server: {e}", flush=True)
        import traceback
        traceback.print_exc()


class ECMIdentificationAgent:
    """
    ECM 参数辨识 Agent
    """
    
    def __init__(self):
        """
        初始化 Agent
        """
        self.name = "ECM Parameter Identification Agent"
        self.version = "0.1.0"
        self.description = "二阶等效电路模型参数辨识与不确定性分析"
    
    def get_info(self) -> Dict[str, str]:
        """
        获取 Agent 信息
        
        返回:
            info: Agent 信息字典
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description
        }
    
    def run(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行 Agent 任务
        
        参数:
            input_params: 输入参数字典，包含:
                - data_path: B0005.mat 文件路径
                - cycle_number: 放电循环编号
                - output_dir: 输出目录
                - n_bootstrap: Bootstrap 重采样次数（可选）
                - current_threshold: 恒流段电流阈值（可选）
                - min_duration: 恒流段最小持续时间（可选）
        
        返回:
            results: 结果字典，包含:
                - status: 运行状态 ('success' 或 'failed')
                - message: 状态消息
                - output_dir: 输出目录路径
                - params: 辨识的参数（如果成功）
                - metrics: 拟合指标（如果成功）
                - error: 错误信息（如果失败）
        """
        try:
            # 提取输入参数
            data_path = input_params.get('data_path', '/data/B0005.mat')
            cycle_number = input_params.get('cycle_number', 1)
            output_dir = input_params.get('output_dir', 'outputs')
            n_bootstrap = input_params.get('n_bootstrap', 50)
            current_threshold = input_params.get('current_threshold', 0.05)
            min_duration = input_params.get('min_duration', 60.0)
            
            print(f"\n{'='*70}")
            print(f"{self.name} v{self.version}")
            print(f"{'='*70}")
            print(f"\n输入参数:")
            print(f"  数据文件: {data_path}")
            print(f"  循环编号: {cycle_number}")
            print(f"  输出目录: {output_dir}")
            print(f"  Bootstrap 次数: {n_bootstrap}")
            
            # 运行流程
            results = run_pipeline(
                mat_path=data_path,
                cycle_number=cycle_number,
                output_base_dir=output_dir,
                current_threshold=current_threshold,
                min_duration=min_duration,
                n_bootstrap=n_bootstrap,
                verbose=True
            )
            
            # 构造返回结果
            return {
                'status': 'success',
                'message': '参数辨识完成',
                'output_dir': str(results['output_dir']),
                'params': {
                    'R0': float(results['params'].R0),
                    'R1': float(results['params'].R1),
                    'C1': float(results['params'].C1),
                    'R2': float(results['params'].R2),
                    'C2': float(results['params'].C2)
                },
                'metrics': {k: float(v) for k, v in results['metrics'].items()}
            }
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            
            print(f"\n错误: {str(e)}")
            print(error_msg)
            
            return {
                'status': 'failed',
                'message': f'参数辨识失败: {str(e)}',
                'error': error_msg
            }


def main():
    """
    主函数：从环境变量或命令行参数读取输入
    """
    # 创建 Agent 实例
    agent = ECMIdentificationAgent()
    
    # 从环境变量读取输入参数（Bohrium 平台会设置这些环境变量）
    # 默认数据路径：智能检测
    # 1. 优先使用环境变量 ECM_DATA_PATH
    # 2. 自动检测：在常见位置查找 B0005.mat
    default_data = None
    if 'ECM_DATA_PATH' not in os.environ:
        # 尝试在多个位置查找数据文件
        possible_paths = [
            '/data/B0005.mat',
            '/share/B0005.mat',
            '/appcode/ECM-APPagent/data/B0005.mat',
            'data/B0005.mat',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                default_data = path
                print(f"[INFO] Auto-detected data file: {path}", flush=True)
                break
        
        if default_data is None:
            # 如果都找不到，使用默认值（会触发跳过逻辑）
            default_data = '/data/B0005.mat'
    
    input_params = {
        'data_path': os.environ.get('ECM_DATA_PATH', default_data),
        'cycle_number': int(os.environ.get('ECM_CYCLE_NUMBER', '1')),
        'output_dir': os.environ.get('ECM_OUTPUT_DIR', 'outputs'),
        'n_bootstrap': int(os.environ.get('ECM_N_BOOTSTRAP', '50')),
        'current_threshold': float(os.environ.get('ECM_CURRENT_THRESHOLD', '0.05')),
        'min_duration': float(os.environ.get('ECM_MIN_DURATION', '60.0'))
    }
    
    # 如果提供了 JSON 配置文件
    config_file = os.environ.get('ECM_CONFIG_FILE')
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            input_params.update(config)
    
    # 关键：schema 生成阶段通常没有数据文件，默认路径不存在就不要直接跑
    data_path_obj = Path(input_params['data_path'])
    
    # ========================================================================
    # 调试信息：打印路径和环境信息
    # ========================================================================
    print("\n" + "="*70, flush=True)
    print("DEBUG INFO", flush=True)
    print("="*70, flush=True)
    print("[DEBUG] cwd =", os.getcwd(), flush=True)
    print("[DEBUG] input data_path =", input_params['data_path'], flush=True)
    print("[DEBUG] abs path =", str(Path(input_params['data_path']).resolve()), flush=True)
    print("[DEBUG] exists? =", Path(input_params['data_path']).exists(), flush=True)
    
    # 检查常见的数据挂载目录
    for check_dir in ["/data", "/share", "/mnt", "/workspace", "/appcode"]:
        if os.path.exists(check_dir):
            try:
                contents = os.listdir(check_dir)
                print(f"[DEBUG] {check_dir} listing = {contents}", flush=True)
                # 如果目录不为空，递归列出所有 .mat 文件
                if contents:
                    import glob
                    mat_files = glob.glob(f"{check_dir}/**/*.mat", recursive=True)
                    if mat_files:
                        print(f"[DEBUG] Found .mat files in {check_dir}: {mat_files}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Cannot list {check_dir}: {e}", flush=True)
        else:
            print(f"[DEBUG] {check_dir} does not exist", flush=True)
    
    print("="*70, flush=True)
    
    # 只有当使用默认数据路径且文件不存在时，才跳过运行
    if input_params['data_path'] == default_data and (not data_path_obj.exists()):
        print("[INFO] Default data file not found:", input_params['data_path'], flush=True)
        print("       Skip running pipeline for schema generation stage.", flush=True)
        print("       Please provide ECM_DATA_PATH (e.g. /share/B0005.mat) when running the App.", flush=True)
        return 0
    
    # 运行 Agent
    results = agent.run(input_params)
    
    # 保存结果到 JSON 文件
    output_file = Path(input_params['output_dir']) / 'agent_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nAgent 结果已保存: {output_file}")
    
    # 返回状态码
    return 0 if results['status'] == 'success' else 1


if __name__ == "__main__":
    # ========================================================================
    # 第一步：启动健康检查 HTTP 服务（必须先启动，否则平台健康检查会失败）
    # ========================================================================
    print("="*70, flush=True)
    print("STARTING HEALTH CHECK SERVER", flush=True)
    print("="*70, flush=True)
    
    # 启动健康检查服务器（后台线程）
    health_thread = threading.Thread(target=start_http_server, args=(49999,), daemon=True)
    health_thread.start()
    print("[INFO] Health check thread started", flush=True)
    
    # 等待服务器启动
    print("[INFO] Waiting for server to bind to port 49999...", flush=True)
    time.sleep(3)
    
    # 验证健康检查服务是否启动成功
    server_ready = False
    for attempt in range(5):
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 49999))
            sock.close()
            if result == 0:
                print(f"[INFO] Health check server is READY on port 49999 (attempt {attempt+1})", flush=True)
                server_ready = True
                break
            else:
                print(f"[WARNING] Port 49999 not ready yet (attempt {attempt+1})", flush=True)
                time.sleep(1)
        except Exception as e:
            print(f"[WARNING] Connection test failed (attempt {attempt+1}): {e}", flush=True)
            time.sleep(1)
    
    if not server_ready:
        print("[ERROR] Health check server failed to start!", flush=True)
        print("[ERROR] Platform health check will fail!", flush=True)
    
    print("="*70, flush=True)
    
    # ========================================================================
    # 第二步：执行主任务
    # ========================================================================
    print("[INFO] Starting main task...", flush=True)
    exit_code = main()
    print(f"[INFO] Main task completed with exit code: {exit_code}", flush=True)
    
    # ========================================================================
    # 第三步：任务完成后，保持进程存活（模拟常驻服务）
    # ========================================================================
    if exit_code == 0:
        print("\n" + "="*70, flush=True)
        print("Agent task completed successfully!", flush=True)
        print("Entering service mode to keep the process alive...", flush=True)
        print("Health check server is still running on port 49999", flush=True)
        print("="*70, flush=True)
        
        try:
            heartbeat_count = 0
            while True:
                time.sleep(60)  # 每60秒心跳一次
                heartbeat_count += 1
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Service heartbeat #{heartbeat_count}", flush=True)
        except KeyboardInterrupt:
            print("\nAgent service stopped by user.", flush=True)
            sys.exit(0)
    else:
        # 如果任务失败，仍然保持服务运行（因为健康检查服务还在）
        print(f"\nAgent task failed with exit code: {exit_code}", flush=True)
        print("But health check server is still running on port 49999", flush=True)
        
        try:
            heartbeat_count = 0
            while True:
                time.sleep(60)
                heartbeat_count += 1
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Service heartbeat #{heartbeat_count} (task failed)", flush=True)
        except KeyboardInterrupt:
            print("\nAgent service stopped by user.", flush=True)
            sys.exit(1)
