#!/usr/bin/env bash
set -e

echo "=========================================="
echo "ECM Agent 启动 - $(date -u)"
echo "=========================================="

APP_DIR="/appcode/ecm-agent-upload"
cd "$APP_DIR"

[ -f ".env" ] && { set -a; . .env; set +a; echo "✓ .env"; }

export USER_WORKING_DIR="$APP_DIR"
export PYTHONPATH="$APP_DIR:${PYTHONPATH:-}"
export HOME="${HOME:-/tmp}"
export LITELLM_LOCAL_MODEL_COST_MAP=true

mkdir -p data outputs

# 查找默认数据文件
# 优先级: /data/B0005.mat (Bohrium文件管理) > resources/B0005.mat (ZIP包)
echo "查找默认数据文件..."

if [ -f "/data/B0005.mat" ]; then
    size=$(stat -c%s "/data/B0005.mat" 2>/dev/null || echo 0)
    echo "Bohrium /data/B0005.mat: $size bytes"
    if [ "$size" -gt 1000000 ]; then
        cp -f "/data/B0005.mat" data/B0005.mat
        echo "✓ 使用 Bohrium 文件管理中的数据"
    fi
fi

if [ ! -f "data/B0005.mat" ] && [ -f "resources/B0005.mat" ]; then
    cp -f resources/B0005.mat data/B0005.mat
    echo "✓ 使用 ZIP 包中的数据"
fi

if [ -f "data/B0005.mat" ]; then
    size=$(stat -c%s "data/B0005.mat" 2>/dev/null || echo 0)
    echo "data/B0005.mat: $size bytes ($(echo "scale=2; $size/1024/1024" | bc) MB)"
else
    echo "⚠ 未找到默认数据文件"
fi

echo ""
echo "data 目录:"
ls -la data/
echo ""

# 服务器
cat > /tmp/server.py << 'PYEOF'
#!/usr/bin/env python3
"""ECM Agent v12"""
import os, sys, json, uuid, logging, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

work_dir = Path(os.environ.get('USER_WORKING_DIR', '.')).resolve()
os.chdir(work_dir)
sys.path.insert(0, str(work_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

log.info("预加载 LiteLLM...")
from litellm import completion as llm_completion
log.info("✓ LiteLLM")

config = json.load(open(work_dir/"config.json")) if (work_dir/"config.json").exists() else {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SID = "main"
sessions = {SID: {"id": SID, "title": "ECM", "created_at": datetime.now().isoformat()}}
history: List[dict] = []  # LLM 对话历史
chat_messages: List[dict] = []  # UI 显示的消息历史

# 消息持久化文件
MESSAGES_FILE = work_dir / ".chat_messages.json"

def save_messages():
    """保存消息到文件"""
    try:
        with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_messages[-50:], f, ensure_ascii=False)  # 只保留最近50条
    except Exception as e:
        log.error(f"保存消息失败: {e}")

def load_messages():
    """从文件加载消息"""
    global chat_messages
    try:
        if MESSAGES_FILE.exists():
            with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
                chat_messages = json.load(f)
                log.info(f"加载了 {len(chat_messages)} 条历史消息")
    except Exception as e:
        log.error(f"加载消息失败: {e}")
        chat_messages = []

# 启动时加载消息
load_messages()

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'csv', 'json', 'xml', 'png', 'jpg', 'jpeg', 'gif', 'svg', 
    'py', 'js', 'md', 'yaml', 'yml', 'log', 'html', 'css',
    'mat', 'npy', 'npz', 'hdf5', 'h5', 'nc', 'dat', 'xls', 'xlsx'
}
MAX_FILE_SIZE = 100 * 1024 * 1024

WELCOME_MESSAGE = """你好！我是 **ECM 参数辨识助手**，专注于电池等效电路模型的参数辨识与不确定性分析。

## 🔋 我可以帮你做什么？

### 1. 参数辨识
对电池放电数据进行 **二阶 RC 等效电路模型** 参数辨识：
- 辨识五个核心参数：R0（欧姆内阻）、R1/C1（快极化）、R2/C2（慢极化）
- 输出拟合质量指标：RMSE、R²、MAE

### 2. 不确定性分析
- **置信区间分析**：基于雅可比矩阵计算 95% 置信区间
- **Bootstrap 重采样**：通过残差重采样估计参数分布
- **敏感性分析**：评估各参数对模型输出的影响程度

### 3. 完整流程
一键执行全部分析，生成完整的结果文件和可视化图表。

---

## 🚀 快速开始

**使用默认数据演示：**
- 输入 `分析` 或 `对第一个循环进行分析`

**使用自己的数据：**
1. 点击右侧文件浏览器上传你的 `.mat` 数据文件
2. 告诉我：`分析 mydata.mat` 或 `对第5个循环进行分析`

---

## 📊 默认数据说明
系统内置 NASA B0005 电池数据集，包含多个充放电循环的电压、电流、时间数据。

有什么我可以帮你的吗？"""

SYSTEM_PROMPT = """你是ECM参数辨识助手，专注于电池等效电路模型的参数辨识与分析。

## 专业知识
- **R0**: 欧姆内阻，即时电压响应
- **R1/C1**: 快极化，时间常数τ1=R1×C1（几十秒）
- **R2/C2**: 慢极化，时间常数τ2=R2×C2（几百秒）
- 辨识方法：最小二乘法 + Trust Region Reflective优化
- 不确定性：置信区间、Bootstrap重采样、敏感性分析

## 回答要求
- 专业准确，使用Markdown格式
- 解释参数的物理意义
- 分析请求由系统自动处理，你回答一般性问题"""

# ========== 文件上传 API ==========

@app.post("/api/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    """上传文件，保持原始文件名"""
    try:
        output_dir = work_dir / "data"
        output_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            if file_ext and file_ext not in ALLOWED_EXTENSIONS:
                return JSONResponse(content={"error": f"不支持的文件类型: {file_ext}"}, status_code=400)
            
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                return JSONResponse(content={"error": f"文件太大: {file.filename}"}, status_code=400)
            
            # 保持原始文件名
            safe_filename = file.filename.replace('/', '_').replace('\\', '_')
            file_path = output_dir / safe_filename
            
            # 如果同名文件存在，添加序号
            if file_path.exists():
                name_parts = safe_filename.rsplit('.', 1)
                counter = 1
                while file_path.exists():
                    if len(name_parts) == 2:
                        safe_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        safe_filename = f"{safe_filename}_{counter}"
                    file_path = output_dir / safe_filename
                    counter += 1
            
            file_path.write_bytes(content)
            log.info(f"上传: {safe_filename} ({len(content)/1024/1024:.2f} MB)")
            
            uploaded_files.append({
                "name": file.filename,
                "saved_name": safe_filename,
                "path": str(file_path),
                "relative_path": f"data/{safe_filename}",
                "url": f"/api/files/data/{safe_filename}",
                "size": len(content)
            })
        
        return JSONResponse({"success": True, "files": uploaded_files})
        
    except Exception as e:
        log.error(f"上传失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/files/upload")
async def upload_files_alt(request: Request, files: List[UploadFile] = File(...)):
    return await upload_files(request, files)

# ========== 其他 API ==========

@app.get("/api/projects")
async def get_projects(): 
    return {"success": True, "projects": [{"id": 1, "name": "Default"}]}

@app.get("/api/config")
async def get_config(): 
    return {**config, "projectId": 1}

@app.get("/api/status")
async def get_status(): 
    return {"status": "ok"}

@app.get("/api/files/tree")
async def files_tree():
    def scan(p, base):
        r = []
        if not p.exists(): return r
        for f in sorted(p.iterdir()):
            if f.name.startswith('.'): continue
            rel = str(f.relative_to(base))
            if f.is_dir():
                r.append({"name": f.name, "path": rel, "type": "directory", "isDirectory": True, "children": scan(f, base)})
            else:
                r.append({"name": f.name, "path": rel, "type": "file", "isDirectory": False, "size": f.stat().st_size})
        return r
    return [{"name": d, "path": d, "type": "directory", "isDirectory": True, "children": scan(work_dir/d, work_dir)} for d in ["data", "outputs"] if (work_dir/d).exists()]

@app.get("/api/download/file/{p:path}")
async def download_file(p: str):
    if ".." in p: raise HTTPException(403)
    fp = (work_dir/p).resolve()
    if not fp.exists(): raise HTTPException(404)
    return FileResponse(str(fp), filename=fp.name)

@app.get("/api/files/{p:path}")
async def get_file(p: str): 
    return await download_file(p)

@app.delete("/api/sessions/clear")
async def clear_sessions(): 
    history.clear()
    return {"success": True}

# ========== ECM 分析 ==========

def extract_cycle_number(q: str) -> int:
    """从用户消息中提取循环号"""
    import re
    # 中文数字映射
    cn_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
               '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5, '第六': 6, '第七': 7, '第八': 8, '第九': 9, '第十': 10}
    
    # 匹配 "第N个循环", "循环N", "cycle N" 等
    patterns = [
        (r'第([一二三四五六七八九十]+)个?(?:循环|周期)', 'cn'),
        (r'第\s*(\d+)\s*个?(?:循环|周期)', 'num'),
        (r'循环\s*(\d+)', 'num'),
        (r'cycle\s*(\d+)', 'num'),
        (r'(\d+)\s*(?:th|st|nd|rd)?\s*cycle', 'num'),
    ]
    
    for pattern, ptype in patterns:
        match = re.search(pattern, q.lower())
        if match:
            val = match.group(1)
            if ptype == 'cn':
                return cn_nums.get(val, 1)
            else:
                return int(val)
    return 1  # 默认第1个循环

def extract_filename_from_query(q: str) -> Optional[str]:
    """从用户消息中提取文件名"""
    import re
    patterns = [
        r'分析\s*["\']?([^\s"\']+\.mat)["\']?',
        r'["\']?([^\s"\']+\.mat)["\']?\s*进行',
        r'文件\s*["\']?([^\s"\']+\.mat)["\']?',
        r'使用\s*["\']?([^\s"\']+\.mat)["\']?',
        r'([^\s"\']+\.mat)',
    ]
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

async def do_ecm_streaming(q: str, websocket: WebSocket, session_id: str) -> Optional[str]:
    """处理 ECM 分析请求，流式输出过程"""
    keywords = ['分析', '辨识', 'analyze', 'ecm', 'pipeline', '运行', '识别']
    if not any(k in q.lower() for k in keywords): 
        return None
    
    async def send_step(msg: str):
        """发送步骤消息"""
        await websocket.send_json({"type": "assistant", "content": msg, "session_id": session_id})
    
    try:
        import asyncio
        data_dir = work_dir / "data"
        
        # 提取循环号
        cycle_number = extract_cycle_number(q)
        log.info(f"提取的循环号: {cycle_number}")
        
        # 尝试从消息中提取文件名
        specified_file = extract_filename_from_query(q)
        
        if specified_file:
            data_file = data_dir / specified_file
            if not data_file.exists():
                mat_files = [f.name for f in data_dir.glob("*.mat")]
                if mat_files:
                    return f"错误: 文件 '{specified_file}' 不存在。\n\n可用的数据文件:\n" + "\n".join(f"- {f}" for f in mat_files)
                return f"错误: 文件 '{specified_file}' 不存在。"
        else:
            data_file = data_dir / "B0005.mat"
            if not data_file.exists():
                mat_files = list(data_dir.glob("*.mat"))
                if mat_files:
                    data_file = mat_files[0]
                else:
                    return "错误: 没有找到数据文件。请先上传 .mat 数据文件。"
        
        fsize = data_file.stat().st_size
        output_dir = f"outputs/cycle_{cycle_number:03d}"
        rel_path = str(data_file.relative_to(work_dir))
        
        # 发送开始消息
        await send_step(f"""🚀 **开始 ECM 参数辨识流程**

📊 **分析配置**
- 数据文件: `{data_file.name}` ({fsize/1024/1024:.2f} MB)
- 放电循环: 第 {cycle_number} 次
- Bootstrap 次数: 20
- 输出目录: `{output_dir}/`""")
        
        await asyncio.sleep(0.1)
        
        # 调用 pipeline 执行完整分析（减少 bootstrap 次数以加快速度）
        from src.mcp_server.tools import pipeline
        
        log.info(f"调用 pipeline: data_path={rel_path}, cycle_number={cycle_number}")
        r = pipeline(data_path=rel_path, cycle_number=cycle_number, output_dir=output_dir, n_bootstrap=20)
        
        if r.get('status') == 'success':
            p = r.get('params', {})
            m = r.get('metrics', {})
            
            # 发送完成消息
            final_msg = f"""✅ **ECM 参数辨识流程完成！**

---

🔧 **辨识参数**

| 参数 | 含义 | 值 | 单位 |
|------|------|-----|------|
| R0 | 欧姆内阻 | {p.get('R0',0):.4e} | Ω |
| R1 | 快极化电阻 | {p.get('R1',0):.4e} | Ω |
| C1 | 快极化电容 | {p.get('C1',0):.2f} | F |
| R2 | 慢极化电阻 | {p.get('R2',0):.4e} | Ω |
| C2 | 慢极化电容 | {p.get('C2',0):.2f} | F |

---

📈 **拟合质量**
- **RMSE**: {m.get('RMSE',0):.6f} V（越小越好）
- **MAE**: {m.get('MAE',0):.6f} V
- **R²**: {m.get('R2',0):.6f}（越接近1越好）

---

📁 **输出文件** (位于 `{output_dir}/`)
| 文件名 | 说明 |
|--------|------|
| params.json | 辨识参数结果 |
| fit_metrics.json | 拟合质量指标 |
| fit_curve.png | 拟合曲线对比图 |
| residual.png | 残差分析图 |
| ci_table.csv | 置信区间表 |
| bootstrap_analysis.png | Bootstrap分析图 |
| sensitivity.png | 敏感性分析图 |

---

💡 **提示**: 刷新右侧文件浏览器可查看和下载结果文件。"""
            
            return final_msg
        else:
            return f"❌ 分析失败: {r.get('message', '未知错误')}"
        
    except Exception as e:
        log.error(traceback.format_exc())
        return f"❌ 分析出错: {e}"

# ========== WebSocket ==========

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    cid = uuid.uuid4().hex[:6]
    log.info(f"[{cid}] 连接")
    
    # 发送欢迎消息
    await websocket.send_json({"type": "welcome", "content": WELCOME_MESSAGE})
    await websocket.send_json({"type": "project_id_set", "project_id": 1})
    await websocket.send_json({"type": "sessions_list", "sessions": list(sessions.values()), "current_session_id": SID})
    
    # 恢复历史消息
    if chat_messages:
        log.info(f"[{cid}] 恢复 {len(chat_messages)} 条历史消息")
        for msg in chat_messages:
            await websocket.send_json(msg)
    
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            mt = data.get("type")
            
            if mt == "message":
                q = data.get("content", "").strip()
                if not q: continue
                
                log.info(f"[{cid}] Q: {q[:50]}")
                t0 = datetime.now()
                
                # 发送并立即保存用户消息
                user_msg = {"type": "user", "content": q, "session_id": SID}
                await websocket.send_json(user_msg)
                chat_messages.append(user_msg)
                save_messages()  # 立即保存用户消息
                
                # 检查是否是 ECM 分析请求
                ecm_keywords = ['分析', '辨识', 'analyze', 'ecm', 'pipeline', '运行', '识别']
                is_ecm_request = any(k in q.lower() for k in ecm_keywords)
                
                ans = None
                if is_ecm_request:
                    # 先发送一个"正在分析"的消息
                    processing_msg = {"type": "assistant", "content": "⏳ 正在进行 ECM 分析，请稍候...\n\n（分析过程可能需要 10-30 秒）", "session_id": SID}
                    await websocket.send_json(processing_msg)
                    
                    # 执行分析
                    ans = await do_ecm_streaming(q, websocket, SID)
                    if ans:
                        assistant_msg = {"type": "assistant", "content": ans, "session_id": SID}
                        await websocket.send_json(assistant_msg)
                        chat_messages.append(assistant_msg)
                        save_messages()
                else:
                    # 普通问题，使用 LLM
                    try:
                        model = os.environ.get('MODEL', 'deepseek/deepseek-chat')
                        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
                        for h in history[-4:]:
                            msgs.append(h)
                        msgs.append({"role": "user", "content": q})
                        
                        resp = llm_completion(model=model, messages=msgs, max_tokens=800, timeout=25)
                        ans = resp.choices[0].message.content
                        
                        history.append({"role": "user", "content": q})
                        history.append({"role": "assistant", "content": ans})
                        
                        assistant_msg = {"type": "assistant", "content": ans, "session_id": SID}
                        await websocket.send_json(assistant_msg)
                        chat_messages.append(assistant_msg)
                        save_messages()
                    except Exception as e:
                        log.error(f"[{cid}] LLM error: {e}")
                        err_msg = {"type": "assistant", "content": f"调用模型出错: {e}", "session_id": SID}
                        await websocket.send_json(err_msg)
                
                dt = (datetime.now() - t0).total_seconds()
                log.info(f"[{cid}] 完成 {dt:.1f}s")
                
                await websocket.send_json({"type": "complete", "content": ""})
                
            elif mt == "get_sessions":
                await websocket.send_json({"type": "sessions_list", "sessions": list(sessions.values()), "current_session_id": SID})
            elif mt in ["create_session", "switch_session", "delete_session"]:
                if mt == "delete_session":
                    history.clear()
                    chat_messages.clear()
                    save_messages()
                await websocket.send_json({"type": "sessions_list", "sessions": list(sessions.values()), "current_session_id": SID})
            elif mt == "set_project_id":
                pid = data.get("project_id", 1)
                await websocket.send_json({"type": "project_id_set", "project_id": pid})
                
    except WebSocketDisconnect:
        log.info(f"[{cid}] 断开")
    except Exception as e:
        log.error(f"[{cid}] 错误: {e}")

# ========== UI ==========

UI = "/opt/mamba/lib/python3.12/site-packages/dp/agent/cli/templates/ui/frontend/ui-static"
if os.path.exists(UI):
    app.mount("/", StaticFiles(directory=UI, html=True), name="ui")

if __name__ == "__main__":
    log.info("启动 0.0.0.0:50001")
    uvicorn.run(app, host="0.0.0.0", port=50001, log_level="warning")
PYEOF

echo "✓ Server v12"
exec python3 /tmp/server.py
