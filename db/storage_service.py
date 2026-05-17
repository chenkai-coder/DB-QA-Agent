# 存储业务层：对上层提供统一的业务接口，组合知识记录、会话、日志三个仓库的操作
from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection
from db.log_repository import LogRepository
from db.knowledge_records_repository import KnowledgeRecordsRepository
from db.schema import init_db
from db.session_repository import SessionRepository


class StorageService:
    """
    对上层提供统一的业务接口。
    Agent、GUI、后端逻辑都可以只调用这一层，无需关心底层表结构。
    内部组合了 knowledge_records、sessions、ingest_logs 三个表的操作。
    """

    def __init__(self, db_name: str = "app.db"):
        self.db = DatabaseConnection(db_name=db_name)
        init_db(self.db)

        self.paper_repo = KnowledgeRecordsRepository(self.db)
        self.session_repo = SessionRepository(self.db)
        self.log_repo = LogRepository(self.db)

    # --------------------------
    # knowledge_records 相关
    # --------------------------
    def insert_record(self, data: Dict[str, Any]) -> int:
        """插入知识记录：先查重，若重复则抛异常并记录日志，否则写入并返回新记录 ID。"""
        title = str(data.get("title", "")).strip()
        data_type = data.get("data_type")
        author = data.get("author")
        source = data.get("source")
        source_path = data.get("source_path")

        if self.paper_repo.exists_record(
            title=title,
            data_type=data_type,
            author=author,
            source=source,
            source_path=source_path,
        ):
            self.log_repo.save_log(
                step="insert_record",
                status="duplicate",
                message=f"检测到重复记录: {title}",
                source_path=data.get("source_path"),
            )
            raise ValueError(f"记录已存在，拒绝重复插入 ({title})")

        record_id = self.paper_repo.insert_record(data)

        self.log_repo.save_log(
            step="insert_record",
            status="success",
            message=f"成功插入记录 id={record_id}",
            record_id=record_id,
            source_path=data.get("source_path"),
        )
        return record_id

    def query_record(self, condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按条件查询知识记录，支持 title/author/keyword 等多字段匹配。"""
        return self.paper_repo.query_record(condition)

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """按 ID 查询单条知识记录。"""
        return self.paper_repo.get_record_by_id(record_id)

    def list_records(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """分页查询知识记录，默认最近 20 条。"""
        return self.paper_repo.list_records(limit=limit, offset=offset)

    def update_record(self, record_id: int, data: Dict[str, Any]) -> bool:
        """更新指定 ID 的记录，只更新传入的字段，并记录操作日志。"""
        ok = self.paper_repo.update_record(record_id, data)
        self.log_repo.save_log(
            step="update_record",
            status="success" if ok else "failed",
            message=f"更新记录 id={record_id}",
            record_id=record_id,
        )
        return ok

    def delete_record(self, record_id: int) -> bool:
        """删除指定 ID 的记录，并记录操作日志。"""
        ok = self.paper_repo.delete_record(record_id)
        self.log_repo.save_log(
            step="delete_record",
            status="success" if ok else "failed",
            message=f"删除记录 id={record_id}",
            record_id=record_id,
        )
        return ok

    # --------------------------
    # sessions 相关
    # --------------------------
    def save_session(
        self,
        session_id: str,
        user_query: str,
        agent_intent: Optional[str] = None,
        response_text: Optional[str] = None,
        related_record_ids: Optional[str] = None,
    ) -> int:
        row_id = self.session_repo.save_session(
            session_id=session_id,
            user_query=user_query,
            agent_intent=agent_intent,
            response_text=response_text,
            related_record_ids=related_record_ids,
        )
        return row_id

    def get_sessions_by_session_id(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_repo.get_sessions_by_session_id(session_id)

    # --------------------------
    # logs 相关
    # --------------------------
    def save_log(
        self,
        step: str,
        status: str,
        message: Optional[str] = None,
        record_id: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> int:
        return self.log_repo.save_log(
            step=step,
            status=status,
            message=message,
            record_id=record_id,
            source_path=source_path,
        )

    def list_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.log_repo.list_logs(limit=limit)