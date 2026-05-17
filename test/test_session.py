"""测试会话管理功能"""
from db.storage_service import StorageService


def test_session():
    """测试会话保存和查询"""
    storage = StorageService()

    session_row_id = storage.save_session(
        session_id="session_001",
        user_query="帮我查询 Yuan Teng 的论文",
        agent_intent="query_record_by_author",
        response_text="已查询到 1 条相关知识记录。",
        related_record_ids="1"
    )
    assert session_row_id > 0
    print(f"会话保存成功，id = {session_row_id}")

    print("\n=== 查询会话 ===")
    sessions = storage.get_sessions_by_session_id("session_001")
    assert len(sessions) > 0
    for session in sessions:
        print(session)
