"""
ChromaDB 向量存储封装。

用于语义检索：人物、世界观设定、情节、章节内容的向量嵌入和搜索。

RAG 实现说明：
    为避免 ChromaDB 无法持久化 embedding function 的固有限制，
    采用「代码显式生成向量」方案：写入和查询时都由本模块调用 embedding API
    生成向量，然后显式传给 ChromaDB。这样不依赖 ChromaDB 自动嵌入，重启后仍生效。
"""

import uuid
import threading
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.core.config import get_chroma_path, get_settings


class _EmbeddingClient:
    """封装 embedding API 调用（OpenAI 兼容），负责生成向量"""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.embedding_api_key
        self.base_url = settings.embedding_base_url
        self.model = settings.embedding_model
        self._client = None

        if not self.api_key:
            logger.warning("未配置 EMBEDDING_API_KEY，向量检索将不可用")
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"embedding 客户端已初始化: {self.model}")
            except Exception as e:
                logger.warning(f"embedding 客户端初始化失败: {e}")

    def embed(self, text: str) -> Optional[list[float]]:
        """对单条文本生成向量"""
        if self._client is None:
            return None
        try:
            r = self._client.embeddings.create(model=self.model, input=text)
            return r.data[0].embedding
        except Exception as e:
            logger.warning(f"生成向量失败: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        """对批量文本生成向量"""
        if self._client is None:
            return None
        try:
            r = self._client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in r.data]
        except Exception as e:
            logger.warning(f"批量生成向量失败: {e}")
            return None


class VectorStore:
    """
    ChromaDB 向量存储封装。

    Collections:
    - novel_characters: 人物描述嵌入
    - novel_settings: 世界观设定嵌入
    - novel_plots: 情节片段嵌入
    - chapter_content: 章节内容嵌入
    """

    COLLECTIONS = ["novel_characters", "novel_settings", "novel_plots", "chapter_content"]

    def __init__(self, persist_dir: Optional[str] = None):
        if persist_dir is None:
            persist_dir = str(get_chroma_path())
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # embedding 客户端（代码显式生成向量，不依赖 ChromaDB 自动嵌入）
        self.embedding = _EmbeddingClient()

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._ensure_collections()
        # 后台写入队列：向量写入（含 embedding API 调用）不阻塞主流程
        self._write_lock = threading.Lock()
        self._enabled = self.embedding._client is not None
        logger.info(
            f"向量存储已初始化: {persist_dir} "
            f"(embedding={'启用' if self._enabled else '未配置'})"
        )

    def _ensure_collections(self):
        """确保所有 collection 存在（不绑定 embedding function，向量由代码显式生成）"""
        for name in self.COLLECTIONS:
            try:
                self.client.get_collection(name)
            except Exception:
                # 新建 collection，不带 embedding function（向量由代码显式传入）
                self.client.create_collection(name=name)

    def get_collection(self, collection_name: str):
        return self.client.get_collection(collection_name)

    # ═══════════════════════════════════════════════════════
    # 写入
    # ═══════════════════════════════════════════════════════

    def add(self, collection_name: str, content: str, metadata: Optional[dict] = None,
            doc_id: Optional[str] = None, embedding: Optional[list[float]] = None) -> str:
        """
        向向量库添加数据（代码显式生成向量，保证 RAG 生效）。

        参数:
            collection_name: collection 名
            content: 文本内容
            metadata: 元数据（应包含 novel_id，用于按小说过滤检索）
            doc_id: 文档ID（默认自动生成）
            embedding: 若已提供向量则直接用，否则调用 embedding API 生成
        """
        if doc_id is None:
            doc_id = f"vec_{uuid.uuid4().hex[:16]}"
        collection = self.get_collection(collection_name)

        # 若未显式提供向量，调用 embedding API 生成
        if embedding is None:
            embedding = self.embedding.embed(content)

        kwargs = {
            "documents": [content],
            "ids": [doc_id],
        }
        if metadata:
            kwargs["metadatas"] = [metadata]
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        # upsert（而非 add）：同名/同 id 重复保存（如角色档案更新）时覆盖旧向量，
        # 否则 ChromaDB 对已存在的 id 会抛「ID already exists」，导致向量库与唯一权威库脱节。
        collection.upsert(**kwargs)
        return doc_id

    def add_background(self, collection_name: str, content: str,
                       metadata: Optional[dict] = None, doc_id: Optional[str] = None):
        """
        后台写入：不阻塞主流程，embedding 失败静默。

        保存侧（save_to_long_term/save_chapter）调用此方法同步维护向量索引，
        避免 embedding API 调用拖慢主流程；失败不影响 SQLite 已持久化的数据。
        """
        if not self._enabled:
            return

        def _worker():
            try:
                with self._write_lock:
                    self.add(collection_name, content, metadata=metadata, doc_id=doc_id)
            except Exception as e:
                logger.debug(f"向量后台写入失败（不影响主流程）: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def add_batch(self, collection_name: str, contents: list[str],
                  metadatas: Optional[list[dict]] = None,
                  ids: Optional[list[str]] = None,
                  embeddings: Optional[list[list[float]]] = None) -> list[str]:
        """批量添加（代码显式生成向量）"""
        if ids is None:
            ids = [f"vec_{uuid.uuid4().hex[:16]}" for _ in contents]
        collection = self.get_collection(collection_name)

        if embeddings is None:
            embeddings = self.embedding.embed_batch(contents)

        kwargs = {"documents": contents, "ids": ids}
        if metadatas and any(metadatas):
            kwargs["metadatas"] = metadatas
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        collection.upsert(**kwargs)
        logger.debug(f"批量写入 {len(contents)} 条到 {collection_name}")
        return ids

    # ═══════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════

    def search(self, collection_name: str, query: str, k: int = 5,
               embedding: Optional[list[float]] = None,
               where: Optional[dict] = None) -> list[dict]:
        """
        语义检索（代码显式生成 query 向量）。

        参数:
            collection_name: collection 名
            query: 查询文本
            k: 返回数量
            embedding: 若已提供 query 向量则直接用
            where: 过滤条件
        """
        collection = self.get_collection(collection_name)

        if embedding is None:
            embedding = self.embedding.embed(query)

        kwargs = {"n_results": k}
        if embedding is not None:
            kwargs["query_embeddings"] = [embedding]
        else:
            # 退化：无 embedding 时用文本查询（依赖 ChromaDB 默认，可能不可用）
            kwargs["query_texts"] = [query]
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        return self._format_results(results)

    def search_by_embedding(self, collection_name: str, query_embedding: list[float],
                            k: int = 5, where: Optional[dict] = None) -> list[dict]:
        """通过嵌入向量检索"""
        collection = self.get_collection(collection_name)
        kwargs = {"query_embeddings": [query_embedding], "n_results": k}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        return self._format_results(results)

    def _format_results(self, results: dict) -> list[dict]:
        """统一格式化 ChromaDB 查询结果"""
        output = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else None
                output.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "score": 1.0 - distance if distance else None,
                })
        return output

    # ═══════════════════════════════════════════════════════
    # 管理
    # ═══════════════════════════════════════════════════════

    def delete(self, collection_name: str, doc_id: str):
        """删除单条"""
        self.get_collection(collection_name).delete(ids=[doc_id])

    def delete_by_filter(self, collection_name: str, where: dict):
        """按条件删除（多字段自动包装成 $and，因 ChromaDB delete 只接受单个 operator）"""
        if len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}
        self.get_collection(collection_name).delete(where=where)

    def count(self, collection_name: str) -> int:
        """获取 collection 中的条目数"""
        return self.get_collection(collection_name).count()

    def list_collections(self) -> list[str]:
        """列出所有 collection"""
        return [c.name for c in self.client.list_collections()]
