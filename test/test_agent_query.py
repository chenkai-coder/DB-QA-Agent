"""测试 Agent 查询记录功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_query(agent):
    """测试通过 Agent 查询记录"""
    intent_json = {
        "intent": "query_record",
        "target": "knowledge_records",
        "params": {
            "author": "Yanfeng Zhang"
        },
        "options": {
            "limit": 10
        },
        "response_mode": "list"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="查询 Yanfeng Zhang 的记录",
        session_id="agent_query_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_query ===")
    print(result)
