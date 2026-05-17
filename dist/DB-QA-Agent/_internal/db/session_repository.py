from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection


class SessionRepository:
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