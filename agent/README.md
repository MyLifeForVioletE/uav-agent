# agent

LangGraph 图结构层，包含图构建、节点路由和所有业务节点。

## 文件

| 文件 | 作用 |
|------|------|
| `graph.py` | `build_graph()`：构建 6 节点 StateGraph 并编译 |
| `routes.py` | 6 个路由函数：按 `_intent` 状态分派到下游节点 |
| `nodes/` | 图节点目录，每个节点一个文件 |

## 节点

| 节点 | 文件 | 作用 |
|------|------|------|
| `idle` | `nodes/idle.py` | 输入清洗、状态重置、记录 original_scenario |
| `router` | `nodes/router.py` | LLM 意图分类（scene/planning/calc/other）+ 宏观/详细规划确认检测 + skill 注入 |
| `planning_prep` | `nodes/planning.py` | 宏观阶段：RAG 双通道检索（场景+约束）；详细阶段：检索阶段分解+执行器规则+工具列表+schema |
| `scene_node` | `nodes/scene.py` | 调用 `build_scene()` 构建电磁场景 Excel |
| `call_llm` | `nodes/llm.py` | 调 Ollama API、tool_calls 格式转换、plan_generated/detail_plan_done 标记检测 |
| `execute_tools` | `nodes/tools.py` | 委托 executor 执行工具调用、scene_building 后暂停等待用户 |

## 图结构

```
idle → router → [scene_node / planning_prep / call_llm]
scene_node → idle
planning_prep → call_llm
call_llm → execute_tools / idle
execute_tools → idle / call_llm
```
