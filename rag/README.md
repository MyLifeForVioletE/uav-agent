# rag

RAG（Retrieval-Augmented Generation）规划器，负责从知识库中检索相关文档辅助 LLM 生成规划。

## 文件

| 文件 | 作用 |
|------|------|
| `planner.py` | `TaskPlanner`：ChromaDB 向量检索 → 余弦相似度重排 → 返回上下文文本 + 阶段名列表 |
| `retriever.py` | ChromaDB 向量检索器：文档分块 → Ollama embedding → 检索，manifest 增量索引 |
| `reranker.py` | `Reranker`：对检索结果按查询相似度重排序，返回 top-N |

## 四个独立索引

| 索引名 | 文档目录 | 用途 |
|--------|---------|------|
| `macro_rag` | `rag_docs/macro_examples/` | 宏观规划场景示例（定点侦察等） |
| `constraint_rag` | `rag_docs/constraint_sop/` | 约束条件文档（实时通信/续航/夜间/山区，不分块整篇检索） |
| `detail_rag` | `rag_docs/phase_decmposition/` | 详细阶段拆解基准 |
| `decomposer_rag` | `rag_docs/task_decomposition/` | 多机子任务拆解参考 |

## 调用方式

```python
planner = TaskPlanner(docs_dir=..., collection_name=..., glob_include=...)
planner.index_docs()  # 首次建索引，后续自动跳过（manifest 增量）

context, phases = planner.plan(query, retrieve_k=3, rerank_n=2)
# 约束类文档可用 score_threshold 过滤（不做重排）
context, _ = planner.plan(query, score_threshold=0.4)
```
