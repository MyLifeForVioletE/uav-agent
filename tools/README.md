# tools

工具执行层：算法工具注册、参数校验与统一任务脚本写入。

## 文件

| 文件 | 作用 |
|------|------|
| `mcp_server.py` | MCP Server：读 `algorithms.json` → 注册算法为 MCP 工具 → JSON-RPC over stdio，由 `main.py` 以子进程启动。仅 `path_planning` 真实调用 exe |
| `stub_tools.py` | 为不真实执行 exe 的算法生成本地 mock 工具，与 MCP 真实工具合并后注入 `deps.tools` |
| `executor.py` | `execute_tool_calls()`：参数校验（占位文本检测 + 用户提及检测）→ 去重执行 → 结果输出 → LLM 总结 |
| `script_writer.py` | 统一任务脚本写入：每个已执行动作落盘为 JSON step，计算步骤间血缘依赖 |

## 算法工具

`algorithms.json` 共注册 11 个算法，其中：

- `path_planning`：通过 MCP 注册，真实调用 exe 执行路径规划
- 其余 10 个（communication、signalAnalysis、directionFinding 等）：本地 stub 工具，返回占位输出（"输出;字段名列表"），供上层建立 Redis 输出占位与脚本记录

## 脚本写入

`script_writer.py` 将执行过的原子动作统一写入 `output/actions_script_{session_id}.json`，包含 `steps` 和 `dependencies`（顺序边 + 数据血缘边，确定性匹配 + LLM 语义兜底）。
