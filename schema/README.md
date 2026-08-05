# schema

JSON Schema 定义文件，约束 LLM 输出格式。

## 目录

| 目录 | 内容 |
|------|------|
| `plan/` | 规划相关 schema |
| `scene/` | 场景相关 schema |
| `mission/` | 任务相关 schema |

## 关键 Schema

| 文件 | 约束 |
|------|------|
| `plan/macro_plan.json` | 宏观规划输出：`scenario` + `macro_phases[]`（phase_id, phase_name, goal） |
| `plan/atomic_action.json` | 详细规划输出：`actions[]`（action_id, phase_id, action_name, executor, required_inputs, done_when） |
| `schema_registry.json` | schema 注册表 |
