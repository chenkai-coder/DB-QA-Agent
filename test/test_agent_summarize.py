"""测试 Agent 总结记录功能"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def test_agent_summarize(agent):
    """测试通过 Agent 总结记录"""
    intent_json = {
        "intent": "summarize_records",
        "target": "knowledge_records",
        "params": {
            "data_type": "paper",
            "keyword": "GNN"
        },
        "options": {
            "limit": 5
        },
        "response_mode": "summary"
    }

    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input="总结一下 GNN 相关论文",
        session_id="agent_summarize_test"
    )

    assert result is not None
    assert isinstance(result, str)
    print("=== test_agent_summarize ===")
    print(result)
