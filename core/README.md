# core

核心框架层，提供全局配置、状态定义、提示词和工具函数。零业务逻辑，被所有其他模块依赖。

## 文件

| 文件 | 作用 |
|------|------|
| `config.py` | 全局常量：`BASE_DIR`（项目根目录）、`OLLAMA_BASE`、`MODEL` |
| `state.py` | `AgentState`（TypedDict）图状态定义 + `Deps` 依赖注入类 |
| `prompts.py` | `SYSTEM_PROMPT` 系统提示词 + `load_skill()` 从 `skills/` 加载技能文件 |
| `ollama_utils.py` | `messages_to_ollama()` LangChain 消息→Ollama 格式转换；`tools_to_ollama()` 工具表→Ollama tools 格式 |

## 依赖关系

- `config.py`：无内部依赖，被所有模块引用
- `state.py`：无内部依赖
- `prompts.py`：依赖 `config.BASE_DIR`
- `ollama_utils.py`：无内部依赖
