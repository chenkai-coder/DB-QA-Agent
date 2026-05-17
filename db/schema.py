# 数据库表结构定义：知识记录表、会话表、日志表的创建与初始化
from db.connection import DatabaseConnection


def create_tables(db: DatabaseConnection) -> None:
    """
    创建系统所需的数据表。
    当前采用更通用的 knowledge_records 表，覆盖多类型私有数据。
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # 通用知识记录表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            data_type TEXT NOT NULL,
            title TEXT,
            category TEXT,
            tags TEXT,
            summary TEXT,
            raw_text TEXT,
            structured_json TEXT,

            author TEXT,
            source TEXT,
            created_date TEXT,
            event_date TEXT,

            keyword_text TEXT,
            entity_text TEXT,

            source_type TEXT DEFAULT 'image',
            source_path TEXT,
            status TEXT DEFAULT 'normal',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 高频查询索引，兼顾 Agent 查询性能
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_records(data_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge_records(title)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_author ON knowledge_records(author)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_records(source)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_created_date ON knowledge_records(created_date)"
    )

    # 会话记录表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            agent_intent TEXT,
            response_text TEXT,
            related_record_ids TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 入库/处理日志表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            source_path TEXT,
            step TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(record_id) REFERENCES knowledge_records(id)
        )
        """
    )

    conn.commit()
    conn.close()


def init_db(db: DatabaseConnection) -> None:
    """初始化数据库：创建全部表和索引。"""
    create_tables(db)