# nodes

LangGraph 图节点函数，每个文件对应图中的一个节点。

## 文件

| 文件 | 节点 | 作用 |
|------|------|------|
| `idle.py` | idle | 输入清洗、状态重置 |
| `router.py` | router | LLM 意图分类 + 确认检测 + skill 注入 |
| `planning.py` | planning_prep | RAG 检索注入上下文 |
| `scene.py` | scene_node | 场景构建 |
| `llm.py` | call_llm | 调 Ollama API + 响应解析 |
| `tools.py` | execute_tools | 工具调用执行 |

## 依赖

所有节点依赖 `core.state`（AgentState, Deps），部分依赖 `core.config`、`core.prompts`、`core.ollama_utils`。
