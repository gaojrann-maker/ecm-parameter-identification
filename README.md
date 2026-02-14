# ECM 参数辨识与不确定性分析

基于 NASA B0005 电池数据的二阶等效电路模型（ECM）参数辨识与不确定性分析项目。支持本地运行、玻尔平台 Agent 部署和 dflow 自动化工作流。

---

## 快速使用

### 方式一：本地一键运行（直接出结果）

```bash
pip install -r requirements.txt

# 运行完整流程（数据读取 → 参数辨识 → 不确定性分析）
python src/pipeline/run_pipeline.py --data data/raw/B0005.mat --cycle 1 --bootstrap 50

# 结果输出到 outputs/cycle_001/
```

输出文件：
- `params.json` — 辨识得到的五个参数 (R0, R1, C1, R2, C2)
- `fit_metrics.json` — 拟合指标 (RMSE, R², MAE 等)
- `fit_curve.png` — 拟合曲线对比图
- `ci_table.csv` — 参数 95% 置信区间表
- `bootstrap_analysis.png` — Bootstrap 参数分布图
- `sensitivity.png` — 敏感性分析图

### 方式二：三步骤工作流（本地逐步执行）

```bash
# 一次性执行三步
python src/workflow/run_workflow.py --data data/raw/B0005.mat --cycle 1 --bootstrap 50

# 结果输出到 outputs/workflow/
```

也可以在 Python 中逐步调用：

```python
from src.workflow.ops import DataReadOp, IdentifyOp, UncertaintyOp

# Step 1: 数据读取
segment_csv = DataReadOp.execute(mat_path="data/raw/B0005.mat", cycle_n=1,
                                  output_path="outputs/segment.csv")

# Step 2: 参数辨识
result = IdentifyOp.execute(segment_csv=segment_csv, output_dir="outputs")

# Step 3: 不确定性分析
UncertaintyOp.execute(segment_csv=segment_csv,
                      params_json=result['params_json'],
                      output_dir="outputs", n_bootstrap=50)
```

### 方式三：dflow 自动化工作流（玻尔平台 + Argo）

在玻尔平台节点上操作：

**第一步：构建镜像（只需一次）**

```bash
# 1. 上传代码和数据到节点的 /personal/ECM-dflow/
# 2. 运行镜像构建脚本
bash /personal/ECM-dflow/setup_image.sh
# 3. 在平台上将此节点保存为镜像，记下镜像地址
```

**第二步：提交工作流**

```bash
cd /personal/ECM-dflow

# 设置环境变量（或通过 setup_argo.py 交互配置）
export BOHRIUM_USERNAME="your@email.com"
export BOHRIUM_PASSWORD="your-password"
export BOHRIUM_PROJECT_ID="12345"

# 配置 Argo 连接
python setup_argo.py --env

# 提交工作流
python dflow_example.py submit

# 提交并实时监控
python dflow_example.py monitor

# 查询工作流状态
python dflow_example.py query <workflow_id>
```

工作流会在 Argo 上依次执行三个 Step：
1. **data-read** — 读取 B0005.mat，提取恒流段，计算 SOC
2. **identify** — 最小二乘参数辨识
3. **uncertainty** — 置信区间 + Bootstrap + 敏感性分析

---

## 项目结构

```
ecm-identification/
├── README.md                    # 项目说明
├── CHANGELOG.md                 # 变更日志
├── requirements.txt             # Python 依赖
├── setup_argo.py                # Argo 连接配置脚本
├── setup_image.sh               # 玻尔平台镜像构建脚本
├── dflow_example.py             # dflow 工作流提交脚本
├── example_workflow.py          # 本地工作流使用示例
├── data/                        # 数据目录
│   ├── raw/                     #   原始数据 (B0005.mat)
│   └── processed/               #   处理后的数据
├── src/                         # 源代码
│   ├── ecm/                     #   ECM 核心模块
│   │   ├── loader.py            #     数据加载与恒流段提取
│   │   ├── ocv.py               #     OCV-SOC 特性拟合
│   │   ├── ecm2rc.py            #     二阶 RC 等效电路模型
│   │   └── metrics.py           #     评估指标 (RMSE, R², MAE)
│   ├── identification/          #   参数辨识模块
│   │   └── fit.py               #     最小二乘法 / 全局优化
│   ├── analysis/                #   不确定性分析模块
│   │   ├── ci.py                #     置信区间（雅可比矩阵法）
│   │   ├── bootstrap.py         #     Bootstrap 重采样
│   │   └── sensitivity.py       #     局部敏感性分析
│   ├── pipeline/                #   完整流程
│   │   └── run_pipeline.py      #     端到端一键运行
│   ├── workflow/                #   工作流模块
│   │   ├── ops.py               #     三步骤操作定义
│   │   ├── run_workflow.py      #     本地执行工作流
│   │   ├── dflow_workflow.py    #     dflow OP 定义与工作流构建
│   │   └── dflow_config.py      #     dflow 配置加载
│   └── agent_app/               #   Bohrium Agent 应用
│       ├── app.py               #     Agent 入口
│       ├── fastapi_app.py       #     Web 界面 (FastAPI)
│       ├── web_app.py           #     Web 界面 (Gradio)
│       └── launcher.py          #     启动器
├── tests/                       # 测试代码
│   └── test_full_pipeline.py    #   完整流程测试
└── outputs/                     # 输出结果
```

---

## 技术方案

### 二阶 RC 等效电路模型

```
OCV(SOC) ── R0 ── [R1 ‖ C1] ── [R2 ‖ C2] ── V_terminal
```

端电压方程：`V = OCV(SOC) - I·R0 - V1 - V2`

其中 V1、V2 为 RC 支路极化电压，采用离散化递推：

```
a_k = exp(-dt / τ_k),   b_k = R_k · (1 - a_k)
V_k[n+1] = a_k · V_k[n] + b_k · I[n]
```

### 参数辨识

- **最小二乘法**（Trust Region Reflective）：快速局部优化
- **微分进化算法**：全局搜索，不依赖初值

### 不确定性分析

| 方法 | 原理 | 输出 |
|------|------|------|
| 置信区间 | 雅可比矩阵 → 协方差 → CI | 参数标准差、95%CI、相关矩阵 |
| Bootstrap | 残差重采样 → 反复拟合 → 分布 | 参数经验分布、百分位CI |
| 敏感性 | 参数扰动 ±1% → 电压变化 | RMS敏感性排序 |

### 主要发现

- R0（欧姆内阻）敏感性最高，辨识最可靠
- R1/R2 高度负相关（-1.0），C1/C2 高度负相关（-1.0），存在可辨识性问题
- 原因：恒流放电数据激励不够丰富，建议使用 HPPC 脉冲测试数据

---

## 数据说明

**数据来源**：NASA PCoE Battery Data Set — B0005

| 项目 | 值 |
|------|-----|
| 电池类型 | 锂离子电池 |
| 额定容量 | 2 Ahr |
| 充电方式 | 1.5A CC 至 4.2V，CV 至 20mA |
| 放电方式 | 2A CC 至 2.7V |
| 环境温度 | 室温 |

下载地址：[NASA Battery Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

将 `B0005.mat`（约 15 MB）放入 `data/raw/` 目录即可使用。

---

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, h5py, tqdm, pydflow

---

## 测试

```bash
# 完整流程测试
python tests/test_full_pipeline.py
```

---

## 许可证

MIT License
