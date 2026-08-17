# agent

LangGraph 图结构层：指挥 agent 图、子 agent 执行图与子 agent 实例。

## 文件

| 文件 | 作用 |
|------|------|
| `coordinator/` | 指挥 agent 图（Coordinator）：8 节点，参数收集 → 任务拆解 → 分配 → 调度 → 聚合 |
| `graph.py` | `build_graph()`：UAV 子 agent 执行图，6 节点（宏观/详细规划 → 确认 → 执行） |
| `analyst_graph.py` | `build_analyst_graph()`：信息处理子 agent 图，3 节点（直接调算法，无规划/确认） |
| `sub_agent.py` | `SubAgent`：子 agent 实例，拥有独立状态与 Kafka 通信工具（send_message） |
| `routes.py` | 执行图路由函数：按 `_intent` 状态分派到下游节点 |
| `nodes/` | 执行图节点目录，每个节点一个文件 |
| `executors/` | 原子动作执行器分发：tool |

## Agent 角色

| Agent | 图 | 用途 |
|-------|----|------|
| 指挥 agent | `coordinator/` | 参数收集、拆解分配、缺参推导、规划确认转达、结果聚合 |
| UAV 子 agent | `graph.py` | 侦察/干扰子任务的规划与执行 |
| 信息处理子 agent | `analyst_graph.py` | 直接调用算法分析数据，产出最终结论 |

## 子 agent 执行图结构（`graph.py`）

```
idle → router → [planning_prep / call_llm]
planning_prep → call_llm
call_llm → execute_tools / planning_prep / idle
execute_tools → call_llm / idle
```

## 信息处理子 agent 图结构（`analyst_graph.py`）

```
idle → call_llm → execute_tools → call_llm / END
```
