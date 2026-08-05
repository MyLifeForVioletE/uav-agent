"""
检索模块：从 ChromaDB 向量检索相关文档块，支持增量更新
"""
import hashlib
import json
import re
from pathlib import Path
import requests
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class OllamaEmbedding(EmbeddingFunction):
    def __init__(self, model: str = "bge-m3:latest", base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, input: Documents) -> Embeddings:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": input},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


class Retriever:
    """从 ChromaDB 检索文档块，支持分 collection 存储和增量更新"""

    def __init__(self, docs_dir: str, persist_dir: str = None, collection_name: str = "rag_docs", no_chunk: bool = False):
        self.docs_dir = Path(docs_dir)
        if persist_dir is None:
            persist_dir = str(self.docs_dir.parent / "chroma_db")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.no_chunk = no_chunk

        self.ollama_base = "http://127.0.0.1:11434"
        self.embed_fn = OllamaEmbedding(base_url=self.ollama_base)
        # 每个 collection 独立子目录，避免多 collection HNSW 冲突
        col_persist_dir = str(Path(persist_dir) / collection_name)
        self.client = chromadb.PersistentClient(path=col_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── manifest 管理 ──────────────────────────────────────

    def _manifest_path(self) -> Path:
        """manifest 文件路径：chroma_db/{collection_name}/manifest.json"""
        return Path(self.persist_dir) / self.collection_name / "manifest.json"

    def _load_manifest(self) -> dict[str, str]:
        """加载 manifest，返回 {文件路径: sha256}"""
        path = self._manifest_path()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, manifest: dict[str, str]):
        """保存 manifest"""
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _file_hash(fpath: Path) -> str:
        """计算文件 SHA256"""
        h = hashlib.sha256()
        h.update(fpath.read_bytes())
        return h.hexdigest()

    def _scan_files(self, glob_include: list[str] = None) -> dict[str, str]:
        """扫描 glob 匹配的文件，返回 {文件绝对路径(str): sha256}"""
        patterns = glob_include if glob_include is not None else ["*.md", "*.txt"]
        files = {}
        for pattern in patterns:
            for fpath in sorted(self.docs_dir.glob(pattern)):
                if fpath.is_file():
                    files[str(fpath)] = self._file_hash(fpath)
        return files

    # ── 索引 ──────────────────────────────────────────────

    def index_docs(self, force_reindex: bool = False, glob_include: list[str] = None):
        """索引文档，支持增量更新
        - force_reindex=True: 删全部重建
        - force_reindex=False: 对比 manifest，只处理新增/修改/删除的文件
        """
        if force_reindex:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._save_manifest({})
            self._do_index(glob_include, "全量重建")
            return

        manifest = self._load_manifest()
        current_files = self._scan_files(glob_include)

        # 分类：新增 / 修改 / 删除 / 未变
        new_files = [f for f in current_files if f not in manifest]
        modified_files = [f for f in current_files if f in manifest and current_files[f] != manifest[f]]
        deleted_files = [f for f in manifest if f not in current_files]
        unchanged = [f for f in current_files if f in manifest and current_files[f] == manifest[f]]

        # 无变化则跳过
        if not new_files and not modified_files and not deleted_files:
            total = sum(1 for _ in self.docs_dir.glob("*") if _.is_file())
            print(f"  [RAG] {self.collection_name}: {self.collection.count()}条，无变化，跳过索引")
            return

        # 删除已变和已删文件的旧 chunks
        for fpath in new_files + modified_files + deleted_files:
            self.collection.delete(where={"source": fpath})

        # 索引新增和修改的文件
        docs = []
        for fpath in new_files + modified_files:
            content = Path(fpath).read_text(encoding="utf-8")
            if self.no_chunk:
                docs.append({"text": content, "metadata": {"source": fpath, "section": 0, "chunk": 0}})
            else:
                chunks = self._chunk_text(content, source=fpath)
                docs.extend(chunks)

        if docs:
            self._add_docs(docs)

        # 更新 manifest
        for fpath in deleted_files:
            manifest.pop(fpath, None)
        for fpath in new_files + modified_files:
            manifest[fpath] = current_files[fpath]
        self._save_manifest(manifest)

        # 汇总日志
        n_new = len(new_files)
        n_mod = len(modified_files)
        n_del = len(deleted_files)
        n_skip = len(unchanged)
        print(f"  [RAG] {self.collection_name}: {self.collection.count()}条，{n_skip}未变，{n_mod}已更新，{n_new}新增，{n_del}删除")

    def _do_index(self, glob_include: list[str] = None, label: str = "已索引"):
        """全量索引并保存 manifest"""
        patterns = glob_include if glob_include is not None else ["*.md", "*.txt"]
        docs = []
        for pattern in patterns:
            for fpath in sorted(self.docs_dir.glob(pattern)):
                if not fpath.is_file():
                    continue
                content = fpath.read_text(encoding="utf-8")
                if self.no_chunk:
                    docs.append({"text": content, "metadata": {"source": str(fpath), "section": 0, "chunk": 0}})
                else:
                    chunks = self._chunk_text(content, source=str(fpath))
                    docs.extend(chunks)

        if not docs:
            print(f"  [RAG] 警告：rag_docs 目录为空，未索引任何文档")
            return

        self._add_docs(docs)

        # 保存 manifest
        manifest = {}
        for fpath in sorted(self.docs_dir.glob(patterns[0])):
            if fpath.is_file():
                manifest[str(fpath)] = self._file_hash(fpath)
        for pattern in patterns[1:]:
            for fpath in sorted(self.docs_dir.glob(pattern)):
                if fpath.is_file():
                    manifest[str(fpath)] = self._file_hash(fpath)
        self._save_manifest(manifest)

        print(f"  [RAG] {self.collection_name}: {label}，{len(docs)} 个文档块")

    def _add_docs(self, docs: list[dict]):
        """批量写入 ChromaDB，使用确定性 ID"""
        if not docs:
            return

        # 生成确定性 ID：doc_{source哈希前8位}_{section}_{chunk}
        ids = []
        for d in docs:
            src = d["metadata"]["source"]
            sec = d["metadata"]["section"]
            chk = d["metadata"]["chunk"]
            src_hash = hashlib.sha256(src.encode()).hexdigest()[:8]
            ids.append(f"doc_{src_hash}_{sec}_{chk}")

        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]

        batch_size = 10
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.add(
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

        self.collection.peek(1)  # 触发 HNSW 索引持久化

    # ── 分块 ──────────────────────────────────────────────

    def _chunk_text(self, text: str, source: str, chunk_size: int = 2000):
        sections = re.split(r'\r?\n(?=#{1,3}\s+)', text)
        chunks = []
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) > chunk_size * 1.5:
                paragraphs = [p.strip() for p in sec.split('\n\n') if p.strip()]
                for j, para in enumerate(paragraphs):
                    if para:
                        chunks.append({"text": para, "metadata": {"source": source, "section": i, "chunk": j}})
            else:
                chunks.append({"text": sec, "metadata": {"source": source, "section": i, "chunk": 0}})
        return chunks

    # ── 检索 ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = None) -> list[dict]:
        """向量检索，返回原始结果列表
        score_threshold: cosine distance 阈值（小于此值的保留），设置后检索全量并按阈值过滤
        """
        count = self.collection.count()
        if count == 0:
            return []

        if score_threshold is not None:
            n_results = count  # 检索全部，在结果中按阈值过滤
        else:
            n_results = min(top_k, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        out = []
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i] if results.get("distances") else None
            if score_threshold is not None and dist is not None and dist >= score_threshold:
                continue
            out.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": dist,
            })
        return out

    def get_embedding(self, text: str) -> list[float]:
        """获取单条文本的嵌入向量"""
        resp = requests.post(
            f"{self.ollama_base}/api/embed",
            json={"model": "bge-m3:latest", "input": [text]},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    @property
    def doc_count(self) -> int:
        return self.collection.count()
