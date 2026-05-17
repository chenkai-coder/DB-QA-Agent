import os
import re
import chromadb
from chromadb.utils import embedding_functions


def _sanitize_text(text: str) -> str:
    cleaned = text or ''
    cleaned = re.sub(r'(?i)\b[a-z]:\\[^\s\'"<>|]+', '[已省略路径]', cleaned)
    cleaned = re.sub(r'(?i)\b[a-z]:/[^\s\'"<>|]+', '[已省略路径]', cleaned)
    cleaned = re.sub(r'(?i)\\\\[^\s\'"<>|]+', '[已省略路径]', cleaned)
    cleaned = re.sub(r'(?i)\B/[^\s\'"<>|]+', '[已省略路径]', cleaned)
    cleaned = re.sub(r'(source_path\s*[:=]\s*)([^\n,}\]]+)', r'\1[已省略路径]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class VectorMemory:
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "chat_memory"):
        """
        初始化 ChromaDB 本地向量库，用于存储历史问答。
        """
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_memory(self, session_id: str, query: str, answer: str):
        """保存 QA 到长期库"""
        memory_text = f"User Query: {_sanitize_text(query)}\nAgent Answer: {_sanitize_text(answer)}"
        doc_id = f"mem_{session_id}_{self.collection.count() + 1}"
        self.collection.add(
            documents=[memory_text],
            ids=[doc_id],
            metadatas=[{"type": "qa_pair", "session": session_id}]
        )

    def retrieve_memory(self, user_query: str, top_k: int = 2) -> str:
        """检索最相似的历史记忆"""
        if self.collection.count() == 0:
            return ""
        results = self.collection.query(
            query_texts=[user_query],
            n_results=min(top_k, self.collection.count())
        )
        recalled_docs = results.get('documents', [[]])[0]
        if not recalled_docs:
            return ""
        memory_context = "【召回的相关历史记忆】\n"
        for idx, doc in enumerate(recalled_docs):
            memory_context += f"记忆 {idx+1}:\n{_sanitize_text(doc)}\n\n"
        return memory_context
