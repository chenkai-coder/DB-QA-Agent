# 会话仓库：用户问答对话的保存与按会话 ID 查询
from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection


class SessionRepository:
    """负责 sessions 表的写入与查询，记录每轮问答的完整上下文。"""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def save_session(
        self,
        session_id: str,
        user_query: str,
        agent_intent: Optional[str] = None,
        response_text: Optional[str] = None,
        related_record_ids: Optional[str] = None,
    ) -> int:
        """插入一条问答会话记录，返回新记录 ID。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (session_id, user_query, agent_intent, response_text, related_record_ids)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_query, agent_intent, response_text, related_record_ids),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return int(row_id)

    def get_sessions_by_session_id(self, session_id: str) -> List[Dict[str, Any]]:
        """按会话 ID 查询历史对话轮次，按时间升序返回。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sessions
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]