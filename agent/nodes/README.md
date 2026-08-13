# nodes

子 agent 执行图节点函数，每个文件对应图中的一个节点。

## 文件

| 文件 | 节点 | 作用 |
|------|------|------|
| `idle.py` | idle | 输入清洗、状态重置、记录 original_scenario |
| `router.py` | router | 规划确认/修改意图检测 + skill 注入 |
| `planning.py` | planning_prep | RAG 检索注入上下文（宏观场景+约束 / 详细阶段拆解） |
| `llm.py` | call_llm | 调 Ollama API + 响应解析 + 规划标记检测 |
| `tools.py` | execute_tools | 原子动作执行 / LLM 工具调用执行 |

## 依赖

所有节点依赖 `core.state`（AgentState, Deps），部分依赖 `core.config`、`core.prompts`、`core.ollama_utils`。
