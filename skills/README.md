# skills

LLM 技能指令文件，作为 SystemMessage 注入对话，指导 LLM 按特定流程输出。

## 文件

| 文件 | 作用 |
|------|------|
| `task_planning.md` | 任务规划技能：定义 6 个参数维度、宏观/详细两层规划的 IF/ELSE 逻辑、JSON 输出要求 |

## 加载方式

`core.prompts.load_skill("task_planning.md")` 读取文件内容，由 `agent/nodes/router.py` 在检测到 planning 意图时注入为 SystemMessage。
