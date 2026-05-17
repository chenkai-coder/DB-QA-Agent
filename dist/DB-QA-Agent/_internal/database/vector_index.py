import os
import json
from typing import List, Dict, Any


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VectorIndexService:
    """
    向量索引层：
    - SQLite 仍然保存权威数据
    - ChromaDB 只保存可语义检索的文本片段
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or os.path.join(
            _get_project_root(),
            "data",
            "chroma_index"
        )
        self.collection_name = "knowledge_chunks"
        self._client = None
        self._collection = None

    def _lazy_init(self):
        if self._collection is not None:
            return

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "未安装 chromadb，请先执行：pip install chromadb"
            )

        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    def _safe_json_loads(self, value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {}
        return {}

    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(0, end - overlap)

        return chunks

    def build_chunks(self, record_id: int, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将一条 knowledge_record 拆成多个可检索 chunk。
        不改变原数据库结构。
        """
        chunks = []

        title = record.get("title", "") or ""
        summary = record.get("summary", "") or ""
        raw_text = record.get("raw_text", "") or ""
        data_type = record.get("data_type", "") or ""
        category = record.get("category", "") or ""
        source = record.get("source", "") or ""
        author = record.get("author", "") or ""

        base_meta = {
            "record_id": str(record_id),
            "title": title,
            "data_type": data_type,
            "category": category,
            "source": source,
            "author": author,
        }

        # 1. 文档级摘要 chunk
        doc_text = "\n".join([
            f"标题：{title}",
            f"类型：{data_type}",
            f"分类：{category}",
            f"作者：{author}",
            f"来源：{source}",
            f"摘要：{summary}",
        ]).strip()

        if doc_text:
            chunks.append({
                "id": f"record_{record_id}_summary",
                "text": doc_text,
                "metadata": {
                    **base_meta,
                    "chunk_type": "summary"
                }
            })

        # 2. structured_json 中的 knowledge_units
        structured_json = self._safe_json_loads(record.get("structured_json", {}))
        knowledge_units = structured_json.get("knowledge_units", [])

        if isinstance(knowledge_units, list):
            for idx, unit in enumerate(knowledge_units):
                if not isinstance(unit, dict):
                    continue

                unit_type = unit.get("unit_type", "")
                name = unit.get("name", "")
                attributes = unit.get("attributes", {})
                relations = unit.get("relations", [])
                evidence = unit.get("evidence", "")

                unit_text = "\n".join([
                    f"知识单元类型：{unit_type}",
                    f"知识单元名称：{name}",
                    f"属性：{json.dumps(attributes, ensure_ascii=False)}",
                    f"关系：{json.dumps(relations, ensure_ascii=False)}",
                    f"证据：{evidence}",
                ]).strip()

                if unit_text:
                    chunks.append({
                        "id": f"record_{record_id}_unit_{idx}",
                        "text": unit_text,
                        "metadata": {
                            **base_meta,
                            "chunk_type": "knowledge_unit",
                            "unit_type": str(unit_type),
                            "unit_name": str(name),
                        }
                    })

        # 3. 原文 chunk
        raw_chunks = self._split_text(raw_text)
        for idx, chunk in enumerate(raw_chunks):
            chunks.append({
                "id": f"record_{record_id}_raw_{idx}",
                "text": chunk,
                "metadata": {
                    **base_meta,
                    "chunk_type": "raw_text",
                    "chunk_index": str(idx),
                }
            })

        return chunks

    def add_record(self, record_id: int, record: Dict[str, Any]) -> int:
        """
        将一条 SQLite 记录同步到向量索引。
        """
        self._lazy_init()

        chunks = self.build_chunks(record_id, record)
        if not chunks:
            return 0

        ids = [chunk["id"] for chunk in chunks]
        docs = [chunk["text"] for chunk in chunks]
        metas = [chunk["metadata"] for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas
        )

        return len(chunks)

    def delete_record(self, record_id: int) -> None:
        """
        删除某条记录对应的全部向量 chunk。
        """
        self._lazy_init()
        self._collection.delete(where={"record_id": str(record_id)})

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索。
        """
        self._lazy_init()

        query = (query or "").strip()
        if not query:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits = []
        for doc, meta, distance in zip(docs, metas, distances):
            hits.append({
                "score": float(distance),
                "record_id": meta.get("record_id"),
                "title": meta.get("title"),
                "chunk_type": meta.get("chunk_type"),
                "unit_type": meta.get("unit_type", ""),
                "unit_name": meta.get("unit_name", ""),
                "source": meta.get("source", ""),
                "author": meta.get("author", ""),
                "content": doc,
            })

        return hits


vector_index = VectorIndexService()