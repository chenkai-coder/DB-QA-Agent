"""测试 Agent 更新记录功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_update(agent):
    """测试通过 Agent 更新记录"""
    intent_json = {
        "intent": "update_record",
        "target": "knowledge_records",
        "params": {
            "record_id": 15,
            "data": {
                "summary": "这是通过 Agent 更新后的摘要内容。",
                "tags": "数据库,事务,索引,已更新"
            }
        },
        "options": {},
        "response_mode": "raw"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="更新 id=1 的记录",
        session_id="agent_update_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_update ===")
    print(result)
