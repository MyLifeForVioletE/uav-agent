# UAV Agent Planning

无人机电磁侦察任务规划与执行系统。基于 LangGraph + Ollama 本地大模型 + RAG 知识检索的多 agent 协作系统。

## 核心设计：每个任务都是多 agent 协作

所有任务统一走 **指挥 agent + 子 agent** 的协同流程。即使是单机任务，也会经过任务拆解，由指挥 agent 分配后交给 UAV 子 agent 独立规划与执行。

```
用户输入 → 指挥agent(参数收集 → 任务拆解 → 任务分配) → 子agent(规划→确认→执行) → 结果聚合
```

### Agent 角色

| Agent | 实现 | 职责 |
|-------|------|------|
| 指挥 agent | `agent/coordinator/` | 参数收集、子任务拆解与分配、缺参推导、规划确认转达、结果聚合 |
| UAV 子 agent | `agent/sub_agent.py` (mode=`uav`) | 对分配的侦察/干扰子任务做宏观规划 → 详细规划 → 执行原子动作 |
| 信息处理子 agent | `agent/sub_agent.py` (mode=`processor`) | 直接匹配算法工具分析无人机采集数据，无规划/确认环节 |

## 功能

- **任务规划**：用户描述侦察/巡检任务，系统自动提取参数、检索参考场景、生成宏观规划和详细规划（原子动作）
- **多 agent 协作**：指挥 agent 将任务拆解分配，UAV/信息处理子 agent 通过 Kafka 消息总线通信
- **算法计算**：11 种算法工具，通过 MCP 协议注册，`path_planning` 真实调用 exe，其余以本地 stub 暴露能力
- **RAG 知识检索**：宏观场景示例、约束条件、阶段拆解、任务分解四层检索，辅助 LLM 生成高质量规划

## 架构

### 入口图：Coordinator 图（8 节点）

```
idle → router → parameter_collector → task_decomposer → task_allocator
  → fleet_dispatcher → [conflict_resolver] → result_agg → idle
```

- `parameter_collector`：询问并收集任务参数，LLM 结构化提取存入 Redis 上下文（黑板）
- `task_decomposer`：LLM 将任务拆解为子任务（uav 采集类 / processor 分析类）
- `task_allocator`：拓扑排序 + 角色能力匹配，分配到具体 UAV
- `fleet_dispatcher`：创建子 agent 实例，Kafka 派发任务，tick 轮询推进
- `conflict_resolver` / `result_agg`：跨机冲突处理与结果聚合

### 子 agent 执行图：6 节点（`agent/graph.py`）

```
idle → router → [planning_prep / call_llm] → execute_tools → idle
```

每个 UAV 子 agent 复用此图完成 宏观规划 → 确认 → 详细规划 → 确认 → 执行。信息处理子 agent 使用 3 节点极简图（`agent/analyst_graph.py`）。

## 项目结构

```
├── main.py                  # 程序入口
├── core/                    # 核心框架：配置、状态、提示词、Redis、Kafka、工具函数
├── agent/
│   ├── coordinator/         # 指挥 agent 图（Coordinator）
│   ├── sub_agent.py         # 子 agent 实例（UAV / 信息处理）
│   ├── graph.py             # 子 agent 执行图（6 节点）
│   ├── analyst_graph.py     # 信息处理子 agent 图（3 节点）
│   ├── nodes/               # 执行图节点
│   └── executors/           # 原子动作执行器（tool / external / system）
├── fleet/                   # 多机协同：FleetManager + TaskAllocator
├── tools/                   # 工具层：mcp_server.py + stub_tools.py + executor.py + script_writer.py
├── rag/                     # RAG 规划器：检索 + 重排
├── algorithms.json          # 算法能力注册表（11 个工具）
├── skills/                  # LLM 技能指令文件
├── rag_docs/                # RAG 知识库文档
├── schema/                  # JSON Schema 定义
├── chroma_db/               # 向量数据库（自动生成）
└── output/                  # 输出文件目录
```

## 通信与状态

| 组件 | 用途 |
|------|------|
| Kafka | 指挥与子 agent 间消息总线（report / request / reply / result / task） |
| Redis | 会话状态、结构化上下文黑板（uavs / targets / base）、子 agent 状态 |
| MCP stdio | main.py 以子进程启动 `tools/mcp_server.py`，注册算法工具 |

## 环境要求

- Python 3.14+
- Ollama 本地服务（`http://127.0.0.1:11434`）
- 已拉取模型：`ExpedientFalcon/qwen3-4b-agent:latest`（对话）、`bge-m3:latest`（embedding）、`qllama/bge-reranker-v2-m3:latest`（重排）
- Redis（`127.0.0.1:6379`）、Kafka（`127.0.0.1:9092`）

## 安装依赖

```bash
pip install langgraph langchain-ollama langchain-core langchain-mcp-adapters chromadb openpyxl pydantic requests redis aiokafka
```

## 部署

按以下顺序启动依赖（先就绪再跑应用）：

### 1. Ollama（`http://127.0.0.1:11434`）

```bash
ollama serve
# 拉取项目依赖的 3 个模型
ollama pull ExpedientFalcon/qwen3-4b-agent:latest
ollama pull bge-m3:latest
ollama pull qllama/bge-reranker-v2-m3:latest
```

### 2. Redis（`127.0.0.1:6379`，Docker）

```powershell
docker run -d --name uav-redis -p 6379:6379 -v uav-redis-data:/data --restart unless-stopped redis:7-alpine
```

### 3. Kafka（`127.0.0.1:9092`，Docker）

```powershell
docker run -d --name uav-kafka -p 9092:9092 -v uav-kafka-data:/var/lib/kafka/data --restart unless-stopped -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:9092 apache/kafka:3.7.0
```

验证两个容器都已启动：

```powershell
docker ps --filter "name=uav-redis" --filter "name=uav-kafka"
```

> **注意**：`KAFKA_ADVERTISED_LISTENERS` 必须写 `127.0.0.1` 而不是 `localhost`。Windows 下 `localhost` 解析为 IPv6 `::1` 优先，而 Docker Desktop 只转发 IPv4，会导致 aiokafka 重连失败、消息收发全部挂起。broker 默认关闭 auto.create.topics，项目内 `_ensure_topic` 会自动创建 topic，无需额外配置。

### 4. 项目代码与数据

拷贝整个项目目录（含 `rag_docs/`、`skills/`、`algorithms.json`，`chroma_db/` 可不拷，启动时自动重建索引），然后：

```bash
pip install -r requirements.txt
python main.py
```

## 启动

```bash
python main.py
```

启动后自动连接 MCP Server、初始化 RAG 索引、注册 11 个算法工具。

## 使用示例

```
>>> 定点侦察一个固定已知目标，需要实时回传，续航充裕
[ParameterCollector] 请确认是否已提供所有必要信息？
>>> 确认
# 指挥 agent 拆解任务 → 分配到 UAV 子 agent
>>> 确认
# 子 agent 生成宏观规划 JSON，经指挥转达等待用户确认
>>> 确认
# 生成详细规划（原子动作列表），等待确认
>>> 确认
# 子 agent 依次执行原子动作（算法工具 / 外部系统 / 系统记录）
```

## 执行器类型

| executor | 用途 |
|----------|------|
| `tool` | 调用算法工具（path_planning、signalAnalysis 等，`agent/executors/tool_executor.py`） |
| `external` | 无人机系统执行（起飞、降落、飞行、载荷操作，`agent/executors/external_executor.py`） |
| `system` | 系统内部记录（日志、状态保存、确认核验，`agent/executors/system_executor.py`） |
