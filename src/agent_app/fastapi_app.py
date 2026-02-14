"""
FastAPI Web 界面（简化版）
功能：提供 REST API 和简单的 HTML 界面进行 ECM 参数辨识
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 兼容两种运行方式
try:
    from src.pipeline.run_pipeline import run_pipeline
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.pipeline.run_pipeline import run_pipeline


# 创建 FastAPI 应用
app = FastAPI(
    title="ECM Parameter Identification",
    description="NASA B0005 电池数据的二阶等效电路模型参数辨识系统",
    version="1.0.0"
)


class AnalysisRequest(BaseModel):
    """分析请求参数"""
    data_path: str = "/data/B0005.mat"
    cycle_number: int = 1
    current_threshold: float = 0.05
    min_duration: float = 60.0
    optimization_method: str = "least-squares"
    n_bootstrap: int = 50


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页 - HTML 表单界面"""
    
    # 自动检测数据文件路径
    possible_paths = [
        '/data/B0005.mat',
        '/share/B0005.mat',
        '/appcode/ECM-APPagent/data/B0005.mat',
        'data/B0005.mat',
    ]
    default_data_path = '/data/B0005.mat'
    for path in possible_paths:
        if os.path.exists(path):
            default_data_path = path
            break
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECM Parameter Identification</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .form-group {{
                margin-bottom: 20px;
            }}
            label {{
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #333;
            }}
            input, select {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }}
            button {{
                background: #667eea;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
            }}
            button:hover {{
                background: #5568d3;
            }}
            .results {{
                display: none;
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            .loading {{
                display: none;
                text-align: center;
                padding: 20px;
            }}
            .spinner {{
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .download-link {{
                display: inline-block;
                margin: 5px;
                padding: 8px 15px;
                background: #28a745;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .download-link:hover {{
                background: #218838;
            }}
            .image-gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-gallery img {{
                width: 100%;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔋 ECM Parameter Identification</h1>
            <p>NASA B0005 电池数据的二阶等效电路模型参数辨识系统</p>
        </div>
        
        <div class="container">
            <h2>📋 输入参数</h2>
            <form id="analysisForm">
                <div class="form-group">
                    <label for="data_path">数据文件路径:</label>
                    <input type="text" id="data_path" name="data_path" value="{default_data_path}">
                </div>
                
                <div class="form-group">
                    <label for="cycle_number">放电循环编号 (1-168):</label>
                    <input type="number" id="cycle_number" name="cycle_number" value="1" min="1" max="168">
                </div>
                
                <div class="form-group">
                    <label for="current_threshold">电流阈值 (A):</label>
                    <input type="number" id="current_threshold" name="current_threshold" value="0.05" step="0.01" min="0.01">
                </div>
                
                <div class="form-group">
                    <label for="min_duration">最小持续时间 (s):</label>
                    <input type="number" id="min_duration" name="min_duration" value="60" step="10" min="10">
                </div>
                
                <div class="form-group">
                    <label for="optimization_method">优化方法:</label>
                    <select id="optimization_method" name="optimization_method">
                        <option value="least-squares">Least-Squares (快速)</option>
                        <option value="differential-evolution">Differential-Evolution (准确)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="n_bootstrap">Bootstrap 次数:</label>
                    <input type="number" id="n_bootstrap" name="n_bootstrap" value="50" step="10" min="10" max="200">
                </div>
                
                <button type="submit">🚀 开始分析</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>分析进行中，请稍候...</p>
            </div>
            
            <div class="results" id="results"></div>
        </div>
        
        <script>
            document.getElementById('analysisForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                // Convert numbers
                data.cycle_number = parseInt(data.cycle_number);
                data.current_threshold = parseFloat(data.current_threshold);
                data.min_duration = parseFloat(data.min_duration);
                data.n_bootstrap = parseInt(data.n_bootstrap);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                try {{
                    const response = await fetch('/api/analyze', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(data)
                    }});
                    
                    const result = await response.json();
                    
                    if (!response.ok) {{
                        throw new Error(result.detail || 'Analysis failed');
                    }}
                    
                    // Display results
                    displayResults(result);
                }} catch (error) {{
                    document.getElementById('results').innerHTML = `
                        <h3 style="color: red;">❌ 错误</h3>
                        <p>${{error.message}}</p>
                    `;
                    document.getElementById('results').style.display = 'block';
                }} finally {{
                    document.getElementById('loading').style.display = 'none';
                }}
            }});
            
            function displayResults(result) {{
                const resultsDiv = document.getElementById('results');
                let html = '<h3>✅ 分析完成</h3>';
                
                // Parameters
                html += '<h4>辨识参数</h4>';
                html += '<ul>';
                for (const [key, value] of Object.entries(result.params)) {{
                    html += `<li><strong>${{key}}</strong>: ${{value.toExponential(6)}}</li>`;
                }}
                html += '</ul>';
                
                // Metrics
                html += '<h4>拟合指标</h4>';
                html += '<ul>';
                for (const [key, value] of Object.entries(result.fit_metrics)) {{
                    html += `<li><strong>${{key}}</strong>: ${{value.toFixed(6)}}</li>`;
                }}
                html += '</ul>';
                
                // Downloads
                html += '<h4>📥 下载文件</h4>';
                html += `<a class="download-link" href="/output/${{result.output_dir}}/params.json" download>参数 JSON</a>`;
                html += `<a class="download-link" href="/output/${{result.output_dir}}/fit_metrics.json" download>拟合指标 JSON</a>`;
                html += `<a class="download-link" href="/output/${{result.output_dir}}/ci_table.csv" download>置信区间 CSV</a>`;
                html += `<a class="download-link" href="/output/${{result.output_dir}}/bootstrap_params.csv" download>Bootstrap 参数 CSV</a>`;
                
                // Images
                html += '<h4>📊 可视化结果</h4>';
                html += '<div class="image-gallery">';
                const images = [
                    'fit_curve.png',
                    'residual.png',
                    'bootstrap_analysis.png',
                    'sensitivity.png',
                    'correlation_matrix.png'
                ];
                for (const img of images) {{
                    html += `<div><img src="/output/${{result.output_dir}}/${{img}}" alt="${{img}}" onerror="this.parentElement.style.display='none'"></div>`;
                }}
                html += '</div>';
                
                resultsDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            }}
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """运行ECM参数辨识分析"""
    try:
        # 检查数据文件
        if not os.path.exists(request.data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {request.data_path}")
        
        # 设置输出目录
        output_base_dir = "outputs"
        output_dir = f"{output_base_dir}/web_cycle_{request.cycle_number:03d}"
        
        # 运行分析
        results = run_pipeline(
            mat_path=request.data_path,
            cycle_number=request.cycle_number,
            output_base_dir=output_base_dir,
            n_bootstrap=request.n_bootstrap,
            current_threshold=request.current_threshold,
            min_duration=request.min_duration,
            verbose=True
        )
        
        return {
            "status": "success",
            "message": "分析完成",
            "params": results['params'],
            "fit_metrics": results['metrics'],
            "output_dir": str(results['output_dir'])
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/output/{filepath:path}")
async def get_output_file(filepath: str):
    """获取输出文件"""
    full_path = Path(filepath)
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(full_path)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "ECM Parameter Identification", "version": "1.0.0"}


@app.get("/test")
async def test_page():
    """测试页面"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>✅ FastAPI 服务运行正常</h1>
        <p>服务端口: 50001</p>
        <p>访问主页: <a href="/">点击这里</a></p>
        <p>API 文档: <a href="/docs">点击这里</a></p>
    </body>
    </html>
    """)


def main():
    """启动服务"""
    print("="*70, flush=True)
    print("STARTING FASTAPI WEB SERVICE", flush=True)
    print("="*70, flush=True)
    print("[INFO] Server will be available at: http://0.0.0.0:50001", flush=True)
    print("[INFO] Health check endpoint: http://localhost:50001/health", flush=True)
    print("[INFO] Test endpoint: http://localhost:50001/test", flush=True)
    print("[INFO] API docs: http://localhost:50001/docs", flush=True)
    print("="*70, flush=True)
    print("", flush=True)
    print("访问方式:", flush=True)
    print("  - 在节点内: curl http://localhost:50001/test", flush=True)
    print("  - 浏览器: 使用 Bohrium 平台提供的公网访问地址", flush=True)
    print("="*70, flush=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
