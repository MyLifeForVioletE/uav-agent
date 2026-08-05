import requests


class Reranker:
    def __init__(self, ollama_base: str = "http://127.0.0.1:11434"):
        self.ollama_base = ollama_base.rstrip("/")
        self.model = "qllama/bge-reranker-v2-m3:latest"

    def _get_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        formatted = [f"query: {q} passage: {p}" for q, p in pairs]
        resp = requests.post(
            f"{self.ollama_base}/api/embed",
            json={"model": self.model, "input": formatted},
        )
        resp.raise_for_status()
        return [emb[0] for emb in resp.json()["embeddings"]]

    def rerank(self, query: str, results: list[dict], top_n: int = 3) -> list[dict]:
        if not results:
            return []

        pairs = [(query, item["text"]) for item in results]
        try:
            scores = self._get_scores(pairs)
        except Exception:
            return results[:top_n]

        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_n]]
