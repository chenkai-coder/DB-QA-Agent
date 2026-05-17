"""测试 Agent 删除记录功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_delete(agent):
    """测试通过 Agent 删除记录"""
    intent_json = {
        "intent": "delete_record",
        "target": "knowledge_records",
        "params": {
            "record_id": 15
        },
        "options": {},
        "response_mode": "raw"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="删除 id=1 的记录",
        session_id="agent_delete_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_delete ===")
    print(result)
