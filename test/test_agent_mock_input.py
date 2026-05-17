"""测试 Agent 处理自然语言输入（Mock 意图识别）"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


TEST_CASES = [
    "帮我查一下 Yanfeng Zhang 的论文",
    "帮我找 GNN 论文",
    "查一下 2025 年的研究",
    "帮我找 ICDE 相关论文",
    "总结一下 graph 相关研究"
]


@pytest.mark.parametrize("user_input", TEST_CASES)
def test_agent_mock_input(agent, user_input):
    """测试各种自然语言输入"""
    print(f"\n--- 用户输入: {user_input} ---")
    result = agent.handle_user_input(
        user_input=user_input,
        session_id="agent_mock_input_test"
    )
    assert result is not None
    assert isinstance(result, str)
    print(result)
