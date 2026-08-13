# rag_docs

RAG 知识库文档，供 `rag/` 模块检索使用。

## 目录

| 目录 | 内容 | 文件数 |
|------|------|--------|
| `macro_examples/` | 宏观规划场景示例，按侦察类型分类 | 1 个 .md |
| `constraint_sop/` | 约束条件与标准操作流程 | 5 个 .md |
| `phase_decmposition/` | 详细阶段拆解基准示例 | 2 个 .md |
| `task_decomposition/` | 多机子任务拆解参考 | 2 个 .md |
| `atomic_action/` | 原子动作执行器类型定义 | 1 个 .md |

## 使用注意

- 文档变更后需删除 `chroma_db/` 中对应索引目录，重启程序重建（或依赖 manifest 增量索引）
- `constraint_sop/` 使用 `no_chunk=True` 模式，整篇作为单条检索，配合 `score_threshold` 过滤
