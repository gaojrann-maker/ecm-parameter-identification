# 贡献指南 (Contributing)

感谢你对 ECM 参数辨识助手项目的关注！欢迎通过 Issue、Pull Request 或文档改进等方式参与贡献。

## 如何贡献

### 报告问题 (Bug / 功能建议)

1. 在 [GitHub Issues] 中搜索是否已有类似问题。
2. 若无，新建 Issue，选择类型（Bug / Feature / Documentation）。
3. 请尽量提供：
   - 环境信息（Python 版本、操作系统）
   - 复现步骤或期望行为
   - 相关日志或截图（如有）

### 提交代码 (Pull Request)

1. **Fork 本仓库**，并在你的 Fork 中创建分支：
   ```bash
   git checkout -b feature/your-feature   # 或 fix/your-fix
   ```

2. **遵循项目结构**：
   - 核心逻辑放在 `src/` 对应子目录（如 `src/ecm/`、`src/identification/`）。
   - Agent 相关修改在 `agent/`。
   - 文档放在 `docs/`。

3. **代码风格**：
   - 使用 Python 3.10+ 语法。
   - 保持与现有代码风格一致（命名、缩进、注释）。
   - 新功能请补充 docstring 或必要注释。

4. **提交信息**：
   - 使用清晰的提交说明，例如：`feat: 添加 XXX 分析`、`fix: 修复循环编号解析`。

5. **发起 Pull Request**：
   - 目标分支为 `main`（或仓库默认分支）。
   - 在 PR 描述中说明改动动机、测试情况。
   - 如涉及文档，请同步更新 `README.md` 或 `docs/` 中的相关内容。

### 文档改进

- 错别字、表述不清、过时说明：可直接提 PR 修改 `README.md`、`docs/APP_INTRO.md`、`docs/HELP.md` 等。
- 新增教程或示例：可在 `docs/` 下新增文档，并在 README 中加上链接。

## 开发与测试

- 本地运行方式见 [README - 本地开发](README.md#-本地开发)。
- 修改后建议在本地跑通 `start.sh` 或 `dp-agent run agent`，确认 WebSocket 与 MCP 工具正常。

## 行为准则

- 尊重其他贡献者，保持友好、专业的讨论氛围。
- 贡献即表示你同意其内容以本项目所采用的 [MIT License](LICENSE) 发布。

再次感谢你的参与！
