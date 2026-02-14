"""
Gradio Web 界面
功能：提供友好的 Web 界面进行 ECM 参数辨识和不确定性分析
"""

import os
import sys
import json
import gradio as gr
from pathlib import Path
from typing import Tuple, List, Optional
import time

# 兼容两种运行方式
try:
    from src.pipeline.run_pipeline import run_pipeline
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.pipeline.run_pipeline import run_pipeline


def run_ecm_analysis(
    data_path: str,
    cycle_number: int,
    current_threshold: float,
    min_duration: float,
    optimization_method: str,
    n_bootstrap: int,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    """
    运行 ECM 参数辨识和不确定性分析
    
    返回:
        (状态信息, 参数JSON, 拟合指标JSON, 拟合曲线图, 残差图, 置信区间CSV, 
         Bootstrap分析图, 敏感性分析图, Bootstrap参数CSV, 相关性矩阵图)
    """
    try:
        progress(0, desc="初始化...")
        
        # 检查数据文件
        if not os.path.exists(data_path):
            return (
                f"❌ 错误：数据文件不存在: {data_path}",
                None, None, None, None, None, None, None, None, None
            )
        
        # 设置输出目录
        output_base_dir = "outputs"
        output_dir = f"{output_base_dir}/web_cycle_{cycle_number:03d}"
        
        progress(0.1, desc="加载数据...")
        
        # 运行分析
        results = run_pipeline(
            mat_path=data_path,
            cycle_number=cycle_number,
            output_base_dir=output_base_dir,
            n_bootstrap=n_bootstrap,
            current_threshold=current_threshold,
            min_duration=min_duration,
            verbose=True
        )
        
        progress(1.0, desc="完成！")
        
        # 构建状态信息
        status = f"""
## ✅ 分析完成

### 输入参数
- 数据文件: `{data_path}`
- 循环编号: {cycle_number}
- 电流阈值: {current_threshold} A
- 最小持续时间: {min_duration} s
- 优化方法: {optimization_method}
- Bootstrap 次数: {n_bootstrap}

### 输出目录
`{output_dir}`

### 辨识参数
- R0 = {results['params']['R0']:.6f} Ω
- R1 = {results['params']['R1']:.6f} Ω
- C1 = {results['params']['C1']:.2f} F
- R2 = {results['params']['R2']:.6f} Ω
- C2 = {results['params']['C2']:.2f} F

### 拟合指标
- RMSE = {results['metrics']['rmse']:.6f} V
- R² = {results['metrics']['r2']:.6f}
- MAE = {results['metrics']['mae']:.6f} V
"""
        
        # 准备输出文件路径
        params_json = os.path.join(output_dir, "params.json")
        metrics_json = os.path.join(output_dir, "fit_metrics.json")
        fit_curve = os.path.join(output_dir, "fit_curve.png")
        residual = os.path.join(output_dir, "residual.png")
        ci_table = os.path.join(output_dir, "ci_table.csv")
        bootstrap_plot = os.path.join(output_dir, "bootstrap_analysis.png")
        sensitivity_plot = os.path.join(output_dir, "sensitivity.png")
        bootstrap_params = os.path.join(output_dir, "bootstrap_params.csv")
        correlation_plot = os.path.join(output_dir, "correlation_matrix.png")
        
        return (
            status,
            params_json if os.path.exists(params_json) else None,
            metrics_json if os.path.exists(metrics_json) else None,
            fit_curve if os.path.exists(fit_curve) else None,
            residual if os.path.exists(residual) else None,
            ci_table if os.path.exists(ci_table) else None,
            bootstrap_plot if os.path.exists(bootstrap_plot) else None,
            sensitivity_plot if os.path.exists(sensitivity_plot) else None,
            bootstrap_params if os.path.exists(bootstrap_params) else None,
            correlation_plot if os.path.exists(correlation_plot) else None
        )
        
    except Exception as e:
        import traceback
        error_msg = f"""
## ❌ 分析失败

### 错误信息
```
{str(e)}
```

### 详细堆栈
```
{traceback.format_exc()}
```
"""
        return (error_msg, None, None, None, None, None, None, None, None, None)


def create_web_interface():
    """
    创建 Gradio Web 界面
    """
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
    
    with gr.Blocks(title="ECM Parameter Identification") as app:
        gr.Markdown("""
# 🔋 ECM 参数辨识与不确定性分析

NASA B0005 电池数据的二阶等效电路模型（ECM2RC）参数辨识系统

---
""")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 输入参数")
                
                # 数据配置
                with gr.Group():
                    gr.Markdown("#### 数据源")
                    data_path_input = gr.Textbox(
                        label="数据文件路径",
                        value=default_data_path,
                        placeholder="/data/B0005.mat",
                        info="NASA B0005.mat 文件路径"
                    )
                    cycle_number_input = gr.Slider(
                        label="放电循环编号",
                        minimum=1,
                        maximum=168,
                        step=1,
                        value=1,
                        info="选择第几次放电循环（1-168）"
                    )
                
                # 数据处理配置
                with gr.Group():
                    gr.Markdown("#### 数据处理")
                    current_threshold_input = gr.Slider(
                        label="电流阈值 (A)",
                        minimum=0.01,
                        maximum=0.5,
                        step=0.01,
                        value=0.05,
                        info="恒流段判定的电流变化阈值"
                    )
                    min_duration_input = gr.Slider(
                        label="最小持续时间 (s)",
                        minimum=10.0,
                        maximum=300.0,
                        step=10.0,
                        value=60.0,
                        info="恒流段最小持续时间"
                    )
                
                # 优化配置
                with gr.Group():
                    gr.Markdown("#### 优化设置")
                    optimization_method_input = gr.Radio(
                        label="优化方法",
                        choices=["Least-Squares", "Differential-Evolution"],
                        value="Least-Squares",
                        info="参数辨识算法"
                    )
                    n_bootstrap_input = gr.Slider(
                        label="Bootstrap 次数",
                        minimum=10,
                        maximum=200,
                        step=10,
                        value=50,
                        info="Bootstrap 重采样次数（越多越准确但越慢）"
                    )
                
                # 运行按钮
                run_button = gr.Button(
                    "🚀 开始分析",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 分析结果")
                
                # 状态信息
                status_output = gr.Markdown(
                    value="*等待输入参数并点击「开始分析」按钮...*"
                )
                
                # 下载区域
                with gr.Row():
                    params_json_output = gr.File(
                        label="📄 参数 JSON",
                        interactive=False
                    )
                    metrics_json_output = gr.File(
                        label="📄 拟合指标 JSON",
                        interactive=False
                    )
                    ci_csv_output = gr.File(
                        label="📄 置信区间 CSV",
                        interactive=False
                    )
                    bootstrap_csv_output = gr.File(
                        label="📄 Bootstrap 参数 CSV",
                        interactive=False
                    )
                
                # 图表展示
                with gr.Tabs():
                    with gr.Tab("📈 拟合曲线"):
                        fit_curve_output = gr.Image(
                            label="拟合曲线对比",
                            type="filepath"
                        )
                    
                    with gr.Tab("📉 残差分析"):
                        residual_output = gr.Image(
                            label="残差分布",
                            type="filepath"
                        )
                    
                    with gr.Tab("🎲 Bootstrap 分析"):
                        bootstrap_plot_output = gr.Image(
                            label="Bootstrap 参数分布",
                            type="filepath"
                        )
                    
                    with gr.Tab("🔍 敏感性分析"):
                        sensitivity_plot_output = gr.Image(
                            label="参数敏感性",
                            type="filepath"
                        )
                    
                    with gr.Tab("🔗 相关性矩阵"):
                        correlation_plot_output = gr.Image(
                            label="参数相关性",
                            type="filepath"
                        )
        
        # 绑定事件
        run_button.click(
            fn=run_ecm_analysis,
            inputs=[
                data_path_input,
                cycle_number_input,
                current_threshold_input,
                min_duration_input,
                optimization_method_input,
                n_bootstrap_input,
            ],
            outputs=[
                status_output,
                params_json_output,
                metrics_json_output,
                fit_curve_output,
                residual_output,
                ci_csv_output,
                bootstrap_plot_output,
                sensitivity_plot_output,
                bootstrap_csv_output,
                correlation_plot_output,
            ]
        )
        
        # 页脚
        gr.Markdown("""
---
### 📚 使用说明

1. **数据文件路径**：系统会自动检测挂载的数据文件，也可手动指定路径
2. **循环编号**：NASA B0005 数据集共有 168 次放电循环
3. **电流阈值**：用于判定恒流段，默认 0.05A 适用于大部分情况
4. **优化方法**：
   - **Least-Squares**：快速，适合初步分析
   - **Differential-Evolution**：全局优化，更准确但较慢
5. **Bootstrap 次数**：50-100 次通常足够，更多次数可提高置信区间准确性

### 📖 输出说明

- **参数 JSON**：辨识得到的 ECM 模型参数（R0, R1, C1, R2, C2）
- **拟合指标 JSON**：模型性能指标（RMSE, R², MAE 等）
- **置信区间 CSV**：参数的 95% 置信区间
- **Bootstrap 参数 CSV**：所有 Bootstrap 重采样得到的参数
- **拟合曲线**：实测电压 vs 模型电压对比
- **残差分析**：拟合误差分布
- **Bootstrap 分析**：参数的不确定性分布
- **敏感性分析**：各参数对模型输出的影响
- **相关性矩阵**：参数间的相关性

### ℹ️ 版本信息

ECM Parameter Identification Agent v1.0.0  
Powered by NASA B0005 Battery Dataset
""")
    
    return app


def main():
    """
    启动 Web 服务
    """
    print("="*70, flush=True)
    print("STARTING GRADIO WEB SERVICE", flush=True)
    print("="*70, flush=True)
    
    # 创建界面
    app = create_web_interface()
    
    # 启动服务
    print(f"[INFO] Starting Gradio on 0.0.0.0:50001", flush=True)
    
    app.queue()  # 启用队列以支持进度条
    app.launch(
        server_name="0.0.0.0",
        server_port=50001,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
        prevent_thread_lock=False,
        max_threads=40,
    )


if __name__ == "__main__":
    main()
