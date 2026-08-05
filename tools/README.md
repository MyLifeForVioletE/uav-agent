# tools

工具执行层，包含算法工具执行器、场景构建器和 MCP Server。

## 文件

| 文件 | 作用 |
|------|------|
| `executor.py` | `execute_tool_calls()`：参数校验（占位文本检测 + 用户提及检测）→ 去重执行 → 结果输出 → LLM 总结 |
| `scene_builder.py` | `build_scene()`：LLM 提取实体 → 按 `schema/types/` 定义填 Excel 表，生成电磁场景配置文件 |
| `mcp_server.py` | MCP Server：读 `algorithms.json` → 注册算法为 MCP 工具 → JSON-RPC over stdio，由 `main.py` 以子进程启动 |

## 算法工具

`mcp_server.py` 注册的1个算法（定义在 `algorithms.json`）：

- `path_planning`：路径规划

