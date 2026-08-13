# core

核心框架层，提供全局配置、状态定义、提示词、Redis 黑板、Kafka 总线与 Ollama 工具函数。零业务逻辑，被所有其他模块依赖。

## 文件

| 文件 | 作用 |
|------|------|
| `config.py` | 全局常量：`BASE_DIR`、`OLLAMA_BASE`、`MODEL`、Redis/Kafka 连接、`COORDINATOR_ID` |
| `state.py` | `AgentState`（TypedDict）图状态定义 + `Deps` 依赖注入类 |
| `prompts.py` | `SYSTEM_PROMPT` / `ANALYST_SYSTEM_PROMPT` 系统提示词 + `load_skill()` 从 `skills/` 加载技能文件 |
| `ollama_utils.py` | `messages_to_ollama()` LangChain 消息→Ollama 格式转换；`tools_to_ollama()` 工具表→Ollama tools 格式；`filter_tools_for_role()` 按 agent 角色过滤工具表 |
| `redis_manager.py` | `RedisManager` 单例：会话/消息/子 agent 状态存储，结构化上下文黑板（uavs/targets/base）的深合并读写 |
| `kafka_bus.py` | `KafkaBus` 单例：指挥/子 agent 间消息总线，topic=`agent.inbox.<id>`，request/reply 用 correlation_id 配对 |
| `fleet_config.py` | 多机协同数据结构：`UAVConfig`、`SubTask`、`UAVRole`、`generate_fleet_id()` |
| `context_template.json` | 结构化上下文黑板模板（uavs/targets/base），`init_context()` 据此初始化 |

## 依赖关系

- `config.py`：无内部依赖，被所有模块引用
- `state.py`：无内部依赖
- `prompts.py`：依赖 `config.BASE_DIR`
- `ollama_utils.py`：依赖 `config.BASE_DIR`
- `redis_manager.py`：依赖 `config`，LangChain messages
- `kafka_bus.py`：依赖 `config`，aiokafka
