"""测试 Agent 获取记录详情功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_detail(agent):
    """测试通过 Agent 获取记录详情"""
    intent_json = {
        "intent": "get_record_detail",
        "target": "knowledge_records",
        "params": {
            "record_id": 14
        },
        "options": {},
        "response_mode": "detail"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="查看 id=14 的详情",
        session_id="agent_detail_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_detail ===")
    print(result)
