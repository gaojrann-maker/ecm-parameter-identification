# ECM Parameter Identification Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bohrium Apps](https://img.shields.io/badge/Bohrium-Apps-green.svg)](https://bohrium.dp.tech)

基于 AI 对话的电池等效电路模型（ECM）参数辨识工具，部署于 Bohrium Apps 平台。

![ECM Model](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Equivalent_circuit_of_a_battery.svg/400px-Equivalent_circuit_of_a_battery.svg.png)

## 📋 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [本地开发](#本地开发)
- [Bohrium 部署](#bohrium-部署)
- [部署问题与解决方案](#部署问题与解决方案)
- [API 参考](#api-参考)
- [技术架构](#技术架构)
- [参考文献](#参考文献)
- [License](#license)

## ✨ 功能特性

### 核心功能

1. **ECM 参数辨识**
   - 二阶 RC 等效电路模型
   - 最小二乘法优化（Trust Region Reflective）
   - 辨识五个核心参数：R0, R1, C1, R2, C2

2. **不确定性分析**
   - 置信区间分析（基于雅可比矩阵）
   - Bootstrap 重采样
   - 敏感性分析

3. **智能对话交互**
   - 自然语言指令
   - 实时分析进度显示
   - 专业问题解答

4. **文件管理**
   - 支持上传自定义 `.mat` 数据
   - 结果文件下载
   - 可视化图表生成

### 支持的模型

二阶 RC 等效电路模型：

```
    R0      R1       R2
   ─┴─    ─┴─     ─┴─
    │      │       │
   ─┬─   ─┬─C1   ─┬─C2
    │      │       │
   ─┴──────┴───────┴─
```

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| R0 | 欧姆内阻 | 0.01 ~ 0.5 Ω |
| R1 | 快极化电阻 | 0.001 ~ 0.1 Ω |
| C1 | 快极化电容 | 100 ~ 100000 F |
| R2 | 慢极化电阻 | 0.001 ~ 0.1 Ω |
| C2 | 慢极化电容 | 1000 ~ 1000000 F |

## 📁 项目结构

```
ecm-identification-agent/
├── agent/                  # Agent 模块
│   └── agent.py           # Google ADK Agent 定义
├── src/                   # 核心源代码
│   ├── data/              # 数据加载与预处理
│   │   ├── loader.py      # MATLAB 数据加载器
│   │   └── preprocessor.py # 数据预处理
│   ├── models/            # ECM 模型
│   │   └── ecm.py         # 二阶 RC 模型实现
│   ├── identification/    # 参数辨识
│   │   └── identifier.py  # 最小二乘辨识器
│   ├── analysis/          # 不确定性分析
│   │   ├── ci.py          # 置信区间分析
│   │   ├── bootstrap.py   # Bootstrap 分析
│   │   └── sensitivity.py # 敏感性分析
│   ├── pipeline/          # 完整流程
│   │   └── run_pipeline.py
│   └── mcp_server/        # MCP 工具接口
│       └── tools.py
├── resources/             # 内置资源
│   └── B0005.mat          # NASA 电池数据集
├── docs/                  # 文档
│   ├── APP_INTRO.md       # App 介绍
│   └── HELP.md            # 帮助文档
├── config.json            # Agent 配置
├── requirements.txt       # Python 依赖
├── start.sh              # Bohrium 启动脚本
├── .env.example          # 环境变量示例
└── README.md
```

## 🚀 快速开始

### 在 Bohrium Apps 上使用

1. 访问 [Bohrium Apps](https://bohrium.dp.tech/apps)
2. 搜索「ECM 参数辨识助手」
3. 点击进入应用
4. 在对话框输入 `分析` 即可开始

### 使用示例

```
# 使用默认数据分析
分析

# 指定循环编号
对第五个循环进行分析

# 使用上传的数据
分析 mydata.mat 的第3个循环

# 咨询问题
R0 参数的物理意义是什么？
```

## 💻 本地开发

### 环境要求

- Python 3.10+
- pip

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-username/ecm-identification-agent.git
cd ecm-identification-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 DEEPSEEK_API_KEY
```

### 本地运行

```bash
# 使用 dp-agent 启动（推荐）
dp-agent run agent --config config.json --port 50001

# 或直接运行 start.sh
bash start.sh
```

访问 http://localhost:50001 即可使用。

### 单独测试核心功能

```python
from src.mcp_server.tools import pipeline

# 运行完整分析流程
result = pipeline(
    data_path="resources/B0005.mat",
    cycle_number=1,
    output_dir="outputs",
    n_bootstrap=50
)

print(result)
```

## ☁️ Bohrium 部署

### 部署步骤

1. **准备上传文件夹**

```bash
# 创建上传文件夹
mkdir ecm-agent-upload

# 复制必要文件
cp -r agent src resources ecm-agent-upload/
cp config.json requirements.txt start.sh .env ecm-agent-upload/
```

2. **在 Bohrium 开发者中心创建 App**
   - 应用类型：Agent App
   - 镜像：选择包含 `bohr-agent-sdk` 的镜像

3. **上传代码**
   - 上传 `ecm-agent-upload` 文件夹

4. **配置启动命令**

```bash
cd /appcode/ecm-agent-upload && bash start.sh
```

5. **上传默认数据文件**
   - 在「文件管理」中上传 `B0005.mat` 到 `/data` 目录

### 环境变量配置

在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
MODEL=deepseek/deepseek-chat
BOHR_PROJECT_ID=1
```

## 🔧 部署问题与解决方案

在部署过程中，我们遇到并解决了以下关键问题：

### 问题 1：服务启动后立即退出

**现象**：`Function instance exited unexpectedly(code 0)`

**原因**：`dp-agent run agent` 命令启动后，主进程退出导致容器终止。

**解决方案**：改用自定义 FastAPI 服务器，使用 `exec python3 /tmp/server.py` 保持主进程运行。

### 问题 2：Project ID 获取失败

**现象**：UI 显示 `请先设置项目 ID`

**原因**：Bohrium SDK 的认证机制在 Apps 环境中无法正常工作。

**解决方案**：实现自定义 API 接口，直接返回固定的 `project_id=1`：

```python
@app.get("/api/projects")
async def get_projects():
    return {"success": True, "projects": [{"id": 1, "name": "Default"}]}
```

### 问题 3：LiteLLM 启动超时

**现象**：服务启动时卡住，端口不监听

**原因**：LiteLLM 首次加载时尝试从 GitHub 获取模型价格配置，网络超时导致阻塞。

**解决方案**：
1. 设置环境变量 `LITELLM_LOCAL_MODEL_COST_MAP=true`
2. 在服务启动时预加载 LiteLLM

### 问题 4：数据文件损坏

**现象**：`Unable to synchronously open file (file signature not found)`

**原因**：大文件（>10MB）在 Bohrium 上传过程中可能被截断。

**解决方案**：
1. 通过 Bohrium「文件管理」功能上传数据文件到 `/data` 目录
2. `start.sh` 中添加多路径搜索逻辑：

```bash
DATA_PATHS=("/data/B0005.mat" "$APP_DIR/resources/B0005.mat")
for p in "${DATA_PATHS[@]}"; do
    if [ -f "$p" ]; then
        cp -f "$p" data/B0005.mat
        break
    fi
done
```

### 问题 5：循环编号未正确传递

**现象**：分析第1个和第5个循环得到相同结果

**原因**：`do_ecm` 函数未从用户消息中提取循环编号。

**解决方案**：实现 `extract_cycle_number()` 函数解析中文和数字：

```python
def extract_cycle_number(q: str) -> int:
    cn_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, ...}
    # 匹配 "第N个循环", "循环N", "cycle N" 等模式
    ...
```

### 问题 6：WebSocket 连接断开后消息丢失

**现象**：页面刷新后对话记录消失

**原因**：消息只保存在内存中。

**解决方案**：实现消息持久化，保存到 `.chat_messages.json` 文件：

```python
def save_messages():
    with open(MESSAGES_FILE, 'w') as f:
        json.dump(chat_messages[-50:], f)

def load_messages():
    if MESSAGES_FILE.exists():
        chat_messages = json.load(open(MESSAGES_FILE))
```

### 问题 7：Bohrium FC 60秒超时

**现象**：长时间分析后页面白屏，回复丢失

**原因**：Bohrium Function Compute 有 60 秒请求超时限制。

**解决方案**：
1. 减少 Bootstrap 次数：50 → 20
2. 减少 LLM max_tokens：1500 → 800
3. 分析前先发送"正在处理"消息

## 📚 API 参考

### MCP Tools

#### `pipeline()`

执行完整的 ECM 参数辨识与不确定性分析流程。

```python
result = pipeline(
    data_path="data/B0005.mat",  # 数据文件路径
    cycle_number=1,              # 放电循环编号
    output_dir="outputs",        # 输出目录
    n_bootstrap=50               # Bootstrap 次数
)
```

返回值：
```python
{
    "status": "success",
    "params": {"R0": 0.0001, "R1": 0.0001, ...},
    "metrics": {"RMSE": 0.0007, "R2": 0.9999, ...},
    "output_files": [...]
}
```

#### `identify()`

仅执行参数辨识。

#### `uncertainty()`

仅执行不确定性分析。

### WebSocket 消息格式

```javascript
// 发送消息
{type: "message", content: "分析"}

// 接收响应
{type: "assistant", content: "...", session_id: "main"}
{type: "complete", content: ""}
```

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   Bohrium Apps                       │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   UI 前端   │◄─┤  WebSocket  ├─►│  FastAPI    │ │
│  │  (SDK 提供) │  │   Server    │  │  Backend    │ │
│  └─────────────┘  └─────────────┘  └──────┬──────┘ │
│                                           │        │
│  ┌─────────────────────────────────────────┴──────┐ │
│  │                   Core Engine                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐ │ │
│  │  │  Data    │ │   ECM    │ │  Uncertainty   │ │ │
│  │  │  Loader  │ │ Identify │ │   Analysis     │ │ │
│  │  └──────────┘ └──────────┘ └────────────────┘ │ │
│  └─────────────────────────────────────────────────┘ │
│                         │                            │
│  ┌─────────────────────┴───────────────────────────┐ │
│  │                  LiteLLM                         │ │
│  │            (DeepSeek API)                        │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 技术栈

- **后端**：FastAPI + Uvicorn
- **AI**：Google ADK + LiteLLM + DeepSeek
- **数据处理**：NumPy, SciPy, h5py
- **可视化**：Matplotlib
- **部署**：Bohrium Apps (bohr-agent-sdk)

## 📖 参考文献

1. Hu, X., Li, S., & Peng, H. (2012). A comparative study of equivalent circuit models for Li-ion batteries. *Journal of Power Sources*, 198, 359-367.

2. He, H., Xiong, R., & Fan, J. (2011). Evaluation of lithium-ion battery equivalent circuit models for state of charge estimation. *Energies*, 4(4), 582-598.

3. Plett, G. L. (2004). Extended Kalman filtering for battery management systems. *Journal of Power Sources*, 134(2), 262-276.

4. NASA Prognostics Center of Excellence - Battery Data Set.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

欢迎贡献代码！请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/ecm-identification-agent/issues)
- **Email**: your-email@example.com
