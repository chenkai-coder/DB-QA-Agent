"""测试数据库基本 CRUD 操作"""
import pytest
from db.storage_service import StorageService


@pytest.fixture
def storage():
    """创建 StorageService 实例"""
    return StorageService()


def test_insert_and_query(storage):
    """测试插入和查询记录"""
    record_data = {
        "data_type": "paper",
        "title": "TaaSDB: A Transaction Management Service Platform Supporting Cross-Model Transactions",
        "category": "academic",
        "tags": "数据库,事务,多模型,Agent",
        "summary": "一篇关于跨模型事务管理平台的论文。",
        "raw_text": "This paper presents TaaSDB...",
        "structured_json": {
            "authors": ["Yuan Teng"],
            "conference": "WISA 2025",
            "keywords": ["transaction", "multi-model", "database"]
        },
        "author": "Yuan Teng",
        "source": "WISA 2025",
        "created_date": "2025",
        "event_date": None,
        "keyword_text": "transaction multi-model database agent",
        "entity_text": "Yuan Teng TaaSDB WISA",
        "source_type": "image",
        "source_path": "demo_paper.png",
        "status": "normal"
    }

    try:
        record_id = storage.insert_record(record_data)
        assert record_id > 0
        print(f"插入成功，record_id = {record_id}")
    except ValueError as e:
        print(f"插入失败（可能已存在）：{e}")
        # 如果已存在，获取现有记录
        records = storage.list_records()
        assert len(records) > 0

    # 查询全部记录
    records = storage.list_records()
    assert len(records) > 0
    for record in records:
        assert "id" in record

    # 按作者查询
    result = storage.query_record({"author": "Yuan"})
    assert len(result) > 0

    # 按数据类型查询
    result = storage.query_record({"data_type": "paper"})
    assert len(result) > 0

    # 按关键词查询
    result = storage.query_record({"keyword": "database"})
    assert len(result) > 0

    # 更新记录
    if records:
        first_id = records[0]["id"]
        ok = storage.update_record(first_id, {"summary": "已更新的通用知识摘要内容"})
        assert ok
        updated = storage.get_record_by_id(first_id)
        assert updated is not None
        assert updated["summary"] == "已更新的通用知识摘要内容"

    # 保存会话
    session_row_id = storage.save_session(
        session_id="session_001",
        user_query="帮我查询 Yuan Teng 的论文",
        agent_intent="query_record_by_author",
        response_text="已查询到 1 条相关知识记录。",
        related_record_ids="1"
    )
    assert session_row_id > 0
    print(f"会话保存成功，id = {session_row_id}")

    # 查询会话
    sessions = storage.get_sessions_by_session_id("session_001")
    assert len(sessions) > 0

    # 查询日志
    logs = storage.list_logs()
    assert len(logs) > 0
