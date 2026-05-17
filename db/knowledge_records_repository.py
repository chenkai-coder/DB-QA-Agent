# 知识记录仓库：通用知识表的 CRUD 操作，支持多字段查询、更新、删除与关键词搜索
import json
from typing import Any, Dict, List, Optional

from db.connection import DatabaseConnection


class KnowledgeRecordsRepository:
    """
    为了兼容现有项目结构，保留旧文件名和类名。
    实际上本类已经升级为对 knowledge_records 通用知识表的访问封装。
    """

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def validate_record(self, data: Dict[str, Any]) -> None:
        """
        校验通用知识记录。
        最低要求：
        - data_type 非空
        - title 非空
        """
        data_type = str(data.get("data_type", "")).strip()
        title = str(data.get("title", "")).strip()

        if not data_type:
            raise ValueError("data_type 不能为空")

        if not title:
            raise ValueError("title 不能为空")

    def exists_record(
        self,
        title: str,
        data_type: Optional[str] = None,
        author: Optional[str] = None,
        source: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> bool:
        """
        判断是否存在重复记录。
        策略：只要来源文件路径存在且相同，即判定重复；
        否则按 title + data_type + author + source 联合匹配。
        注意：只有 title 非空时才回退到标题匹配，避免空 title 误匹配。
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        if source_path:
            cursor.execute(
                """
                SELECT 1
                FROM knowledge_records
                WHERE source_path = ?
                LIMIT 1
                """,
                (source_path,),
            )
            row = cursor.fetchone()
            # 如果明确提供了 source_path，则只以路径精确匹配为准，
            # 若路径不一致则不视为重复（避免不同文件但同标题被误判）。
            conn.close()
            return row is not None

        # 只有 title 有值时才做标题联合匹配，避免空字符串误判
        if title:
            cursor.execute(
                """
                SELECT 1
                FROM knowledge_records
                WHERE title = ?
                  AND IFNULL(data_type, '') = IFNULL(?, '')
                  AND IFNULL(author, '') = IFNULL(?, '')
                  AND IFNULL(source, '') = IFNULL(?, '')
                LIMIT 1
                """,
                (title, data_type, author, source),
            )
            row = cursor.fetchone()
            conn.close()
            return row is not None

        conn.close()
        return False

    def insert_record(self, data: Dict[str, Any]) -> int:
        """
        插入一条通用知识记录，返回新记录 id。
        """
        self.validate_record(data)

        data_type = str(data.get("data_type", "")).strip()
        title = str(data.get("title", "")).strip()
        category = data.get("category")
        tags = data.get("tags")
        summary = data.get("summary")
        raw_text = data.get("raw_text")
        structured_json = data.get("structured_json")

        author = data.get("author")
        source = data.get("source")
        created_date = data.get("created_date")
        event_date = data.get("event_date")

        keyword_text = data.get("keyword_text")
        entity_text = data.get("entity_text")

        source_type = data.get("source_type", "image")
        source_path = data.get("source_path")
        status = data.get("status", "normal")

        if isinstance(structured_json, (dict, list)):
            structured_json = json.dumps(structured_json, ensure_ascii=False)

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge_records
            (
                data_type, title, category, tags, summary, raw_text, structured_json,
                author, source, created_date, event_date,
                keyword_text, entity_text,
                source_type, source_path, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data_type, title, category, tags, summary, raw_text, structured_json,
                author, source, created_date, event_date,
                keyword_text, entity_text,
                source_type, source_path, status
            ),
        )
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        return int(record_id)

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """按 ID 查询单条知识记录，不存在返回 None。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM knowledge_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def list_records(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """分页查询知识记录，默认按 ID 倒序返回最近 20 条。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM knowledge_records
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def query_record(self, condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        通用条件查询。
        支持字段：
        - data_type
        - title
        - author
        - source
        - created_date
        - category
        - status
        - keyword（模糊匹配 title/summary/raw_text/tags/keyword_text/entity_text）
        """
        sql = "SELECT * FROM knowledge_records WHERE 1=1"
        params: List[Any] = []

        data_type = condition.get("data_type")
        if data_type:
            sql += " AND data_type = ?"
            params.append(data_type)

        title = condition.get("title")
        if title:
            sql += " AND title LIKE ?"
            params.append(f"%{title}%")

        author = condition.get("author")
        if author:
            sql += " AND author LIKE ?"
            params.append(f"%{author}%")

        source = condition.get("source")
        if source:
            sql += " AND source LIKE ?"
            params.append(f"%{source}%")

        created_date = condition.get("created_date")
        if created_date:
            sql += " AND created_date LIKE ?"
            params.append(f"%{created_date}%")

        category = condition.get("category")
        if category:
            sql += " AND category LIKE ?"
            params.append(f"%{category}%")

        status = condition.get("status")
        if status:
            sql += " AND status = ?"
            params.append(status)

        source_path = condition.get("source_path")
        if source_path:
            sql += " AND source_path = ?"
            params.append(source_path)

        keyword = condition.get("keyword")
        if keyword:
            sql += """
                AND (
                    title LIKE ?
                    OR summary LIKE ?
                    OR raw_text LIKE ?
                    OR tags LIKE ?
                    OR keyword_text LIKE ?
                    OR entity_text LIKE ?
                    OR author LIKE ?
                    OR source LIKE ?
                    OR structured_json LIKE ?
                )
            """
            like_keyword = f"%{keyword}%"
            params.extend([
                like_keyword, like_keyword, like_keyword, like_keyword,
                like_keyword, like_keyword, like_keyword, like_keyword,
                like_keyword
            ])

        sql += " ORDER BY id DESC"

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_record(self, record_id: int, data: Dict[str, Any]) -> bool:
        """
        更新记录。只更新传入的字段。
        """
        allowed_fields = {
            "data_type",
            "title",
            "category",
            "tags",
            "summary",
            "raw_text",
            "structured_json",
            "author",
            "source",
            "created_date",
            "event_date",
            "keyword_text",
            "entity_text",
            "source_type",
            "source_path",
            "status",
        }

        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        if not update_data:
            return False

        if "data_type" in update_data:
            data_type = str(update_data["data_type"]).strip()
            if not data_type:
                raise ValueError("data_type 不能为空")

        if "title" in update_data:
            title = str(update_data["title"]).strip()
            if not title:
                raise ValueError("title 不能为空")

        if "structured_json" in update_data and isinstance(update_data["structured_json"], (dict, list)):
            update_data["structured_json"] = json.dumps(update_data["structured_json"], ensure_ascii=False)

        update_data["updated_at"] = "CURRENT_TIMESTAMP"

        set_clauses = []
        params = []

        for field, value in update_data.items():
            if field == "updated_at" and value == "CURRENT_TIMESTAMP":
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            else:
                set_clauses.append(f"{field} = ?")
                params.append(value)

        params.append(record_id)

        sql = f"UPDATE knowledge_records SET {', '.join(set_clauses)} WHERE id = ?"

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def delete_record(self, record_id: int) -> bool:
        """删除指定 ID 的知识记录，返回是否成功。"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_records WHERE id = ?", (record_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0