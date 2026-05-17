"""测试 Agent 插入记录功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_insert(agent):
    """测试通过 Agent 插入记录"""
    intent_json = {
        "intent": "insert_record",
        "target": "knowledge_records",
        "params": {
            "data_type": "note",
            "title": "数据库系统复习笔记",
            "category": "course",
            "tags": "数据库,事务,索引",
            "summary": "包含事务、恢复与索引的整理笔记。",
            "raw_text": "事务、恢复、索引是数据库系统的重要部分。",
            "structured_json": {
                "chapters": ["事务", "恢复", "索引"]
            },
            "author": "用户本人",
            "source": "数据库系统原理",
            "created_date": "2026-04-15",
            "event_date": None,
            "keyword_text": "事务 恢复 索引 数据库",
            "entity_text": "数据库系统原理",
            "source_type": "manual",
            "source_path": "manual_input",
            "status": "normal"
        },
        "options": {},
        "response_mode": "raw"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="插入一条数据库系统复习笔记",
        session_id="agent_insert_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_insert ===")
    print(result)
