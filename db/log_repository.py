# 日志仓库：入库处理日志的保存与查询
from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection


class LogRepository:
    """负责 ingest_logs 表的写入与读取，记录每次入库操作的状态。"""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def save_log(
        self,
        step: str,
        status: str,
        message: Optional[str] = None,
        record_id: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> int:
        """插入一条处理日志，返回新记录 ID。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ingest_logs (record_id, source_path, step, status, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record_id, source_path, step, status, message),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return int(row_id)

    def list_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """按时间倒序查询最近的入库日志，默认最近 50 条。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM ingest_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]