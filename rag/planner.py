"""
任务规划模块：Retrieve → Rerank → 格式化上下文
"""
import re
from .retriever import Retriever
from .reranker import Reranker


class TaskPlanner:
    """任务规划器
    流程：向量检索 top-k → 向量余弦相似度重排序 top-n → 格式化上下文
    """

    def __init__(self, docs_dir: str, persist_dir: str = None, collection_name: str = "rag_docs", glob_include: list[str] = None, no_chunk: bool = False):
        self.retriever = Retriever(docs_dir, persist_dir, collection_name, no_chunk)
        self.reranker = Reranker()
        self.glob_include = glob_include

    def index_docs(self, force_reindex: bool = False):
        self.retriever.index_docs(force_reindex, self.glob_include)

    def plan(self, user_input: str, retrieve_k: int = 10, rerank_n: int = 6, score_threshold: float = None) -> tuple[str, list[str]]:
        """
        执行检索 + 重排序，返回 (格式化上下文文本, 阶段名称列表)
        上下文文本为空字符串表示无相关内容
        score_threshold: 按相似度阈值过滤（用于约束等独立类型文档），设置后不重排序
        """
        raw = self.retriever.retrieve(user_input, top_k=retrieve_k, score_threshold=score_threshold)
        if not raw:
            return ("", [])

        if score_threshold is not None:
            reranked = raw
        else:
            reranked = self.reranker.rerank(user_input, raw, top_n=rerank_n)

        contexts = []
        for i, item in enumerate(reranked):
            src = (item.get("metadata") or {}).get("source", "unknown")
            contexts.append(f"[参考 {i + 1}] 来源: {src}\n{item['text']}")

        phases = self._extract_phases(chunks=[r["text"] for r in reranked])

        return ("\n\n---\n\n".join(contexts), phases)

    @staticmethod
    def _extract_phases(chunks: list[str]) -> list[str]:
        """从 chunk 文本中提取阶段名称列表（累加所有 chunk）"""
        all_phases = []
        seen = set()
        for chunk in chunks:
            if "完整任务阶段" not in chunk:
                continue
            lines = chunk.split("\n")
            for line in lines:
                line = line.strip()
                m = re.match(r'[-－]\s*阶段\d+[：:]\s*(.+)', line)
                if m:
                    name = m.group(1).strip()
                    if name not in seen:
                        seen.add(name)
                        all_phases.append(name)
        return all_phases

    @property
    def doc_count(self) -> int:
        return self.retriever.doc_count
