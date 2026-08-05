# UAV Agent Planning

无人机电磁侦察任务规划与执行系统，基于 LangGraph + Ollama 本地大模型 + RAG 知识检索。

## 功能

- **任务规划**：用户描述侦察/巡检任务，系统自动提取参数、检索参考场景、生成宏观规划和详细规划（原子动作）
- **场景构建**：自然语言描述电磁场景，LLM 提取实体并生成标准 Excel 配置表
- **算法计算**：点场强、面场强、遮蔽角、路径规划等 9 种算法工具，通过 MCP 协议注册
- **RAG 知识检索**：宏观场景示例、约束条件、阶段拆解三层检索，辅助 LLM 生成高质量规划

## 架构

```
用户输入 → router(意图分类) → planning(RAG检索+上下文注入) → llm(调Ollama生成) → tools(执行工具) → 循环
```

LangGraph 6 节点图：`idle → router → [scene_node / planning_prep / call_llm] → execute_tools → idle`

## 项目结构

```
├── main.py                  # 程序入口
├── core/                    # 核心框架：配置、状态、提示词、工具函数
├── agent/                   # LangGraph 图：graph.py + routes.py + nodes/
├── tools/                   # 工具层：executor.py + scene_builder.py + mcp_server.py
├── rag/                     # RAG 规划器：检索 + 重排
├── algorithms.json          # 算法能力注册表（9个工具）
├── skills/                  # LLM 技能指令文件
├── rag_docs/                # RAG 知识库文档
├── schema/                  # JSON Schema 定义
├── chroma_db/               # 向量数据库（自动生成）
├── scene/                   # 场景输出目录
└── output/                  # 输出文件目录
```

## 环境要求

- Python 3.14+
- Ollama 本地服务（`http://127.0.0.1:11434`）
- 已拉取模型：`ExpedientFalcon/qwen3-4b-agent:latest`

## 安装依赖

```bash
pip install langgraph langchain-ollama langchain-core langchain-mcp-adapters chromadb openpyxl pydantic requests
```

## 启动

```bash
python main.py
```

启动后自动连接 MCP Server、初始化 RAG 索引、注册 9 个算法工具 + 1 个场景构建工具。

## 使用示例

```
>>> 定点侦察一个固定已知目标，需要实时回传，续航充裕
[Router] intent=planning
[RAG] 场景检索: 723字, phases=['任务输入和任务要素提取', ...]
# 系统生成宏观规划 JSON，等待确认

>>> 确认
# 系统生成详细规划 JSON（原子动作列表），等待确认

>>> 确认
# 规划完成，进入执行阶段

>>> 生成场景：一架无人机搭载雷达载荷，在城市峡谷区域执行巡检任务
# 调用 scene_building 工具，生成场景 Excel

>>> 点场强计算 -30dBm
# 直接调用算法工具计算
```

## 核心流程

1. **意图分类**（router）：判断用户要规划/建场景/计算/其他
2. **参数提取**：检查 6 个关键参数是否齐全（侦察类型、目标确定性、通信、续航、突发、协同）
3. **RAG 检索**：按参数检索 macro_examples（场景示例）+ constraint_sop（约束条件）
4. **宏观规划**：LLM 生成阶段列表（JSON），用户确认
5. **详细规划**：LLM 对每个阶段拆分原子动作（JSON），指定 executor 类型，用户确认
6. **执行**：按动作列表依次调用工具/系统记录

## 执行器类型

| executor | 用途 |
|----------|------|
| `tool` | 调用算法工具（path_planning、point_field_strength 等） |
| `external` | 无人机系统执行（起飞、降落、飞行、载荷操作） |
| `system` | 系统内部记录（日志、状态保存、确认核验） |
