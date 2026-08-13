# schema

JSON Schema 定义文件，约束 LLM 输出格式。

## 目录

| 目录 | 内容 |
|------|------|
| `plan/` | 规划相关 schema |
| `scene/` | 场景相关 schema（excel/ 表结构、types/ 实体定义） |
| `mission/` | 任务需求 schema |

## 关键 Schema

| 文件 | 约束 |
|------|------|
| `plan/macro_plan.json` | 宏观规划输出：`scenario` + `macro_phases[]`（phase_id, phase_name, goal） |
| `plan/atomic_action.json` | 详细规划输出：`actions[]`（action_id, phase_id, action_name, executor, required_inputs, done_when） |
| `plan/fleet_plan.json` | 多机子任务拆解输出：`scenario` + `uav_count` + `sub_tasks[]` |
| `mission/Reconnaissance.json` | 侦察任务需求字段定义 |
| `schema_registry.json` | schema 注册表 |

## 使用方式

- `macro_plan.json` / `atomic_action.json` 由 `agent/nodes/planning.py` 注入规划 prompt
- `fleet_plan.json` 由 `coordinator_nodes.task_decomposer_node` 注入子任务拆解 prompt
