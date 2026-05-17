"""测试 Agent 控制器基本功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    """创建 AgentController 实例"""
    return AgentController()


def test_handle_user_input_query_by_author(agent):
    """测试1：按作者查询"""
    result = agent.handle_user_input(
        user_input="帮我查一下 Yanfeng Zhang 的论文",
        session_id="agent_session_001"
    )
    assert result is not None
    assert isinstance(result, str)
    print(f"\n{'=' * 20} 测试1：按作者查询 {'=' * 20}")
    print(result)


def test_handle_user_input_query_by_keyword(agent):
    """测试2：按关键词查询"""
    result = agent.handle_user_input(
        user_input="帮我找 GNN 论文",
        session_id="agent_session_001"
    )
    assert result is not None
    assert isinstance(result, str)
    print(f"\n{'=' * 20} 测试2：按关键词查询 {'=' * 20}")
    print(result)


def test_handle_user_input_summarize(agent):
    """测试3：总结类意图"""
    result = agent.handle_user_input(
        user_input="总结一下 graph 相关研究",
        session_id="agent_session_001"
    )
    assert result is not None
    assert isinstance(result, str)
    print(f"\n{'=' * 20} 测试3：总结类意图 {'=' * 20}")
    print(result)


def test_handle_intent_json_insert(agent):
    """测试4：插入记录"""
    insert_intent = {
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
        intent_json=insert_intent,
        user_input="插入一条数据库系统复习笔记",
        session_id="agent_session_002"
    )
    assert result is not None
    assert isinstance(result, str)
    print(f"\n{'=' * 20} 测试4：插入记录 {'=' * 20}")
    print(result)


def test_handle_intent_json_query(agent):
    """测试5：直接传 intent 查询"""
    query_intent = {
        "intent": "query_record",
        "target": "knowledge_records",
        "params": {
            "keyword": "数据库"
        },
        "options": {
            "limit": 5
        },
        "response_mode": "list"
    }

    result = agent.handle_intent_json(
        intent_json=query_intent,
        user_input="查询数据库相关记录",
        session_id="agent_session_002"
    )
    assert result is not None
    assert isinstance(result, str)
    print(f"\n{'=' * 20} 测试5：直接传 intent 查询 {'=' * 20}")
    print(result)
