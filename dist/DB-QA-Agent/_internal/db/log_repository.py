from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection


class LogRepository:
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