"""测试 Agent 处理无效意图"""
import pytest
from agent.agent_controller import AgentController


@pytest.fixture
def agent():
    return AgentController()


def run_case(agent, title: str, intent_json: dict):
    print(f"\n=== {title} ===")
    result = agent.handle_intent_json(
        intent_json=intent_json,
        user_input=title,
        session_id="agent_invalid_test"
    )
    print(result)
    return result


def test_missing_intent(agent):
    """测试1：缺少 intent"""
    result = run_case(agent, "缺少 intent", {
        "target": "knowledge_records",
        "params": {}
    })
    assert "错误" in result or "缺少" in result


def test_unsupported_intent(agent):
    """测试2：不支持的 intent"""
    result = run_case(agent, "不支持的 intent", {
        "intent": "unknown_intent",
        "target": "knowledge_records",
        "params": {}
    })
    assert "错误" in result or "不支持" in result


def test_wrong_target(agent):
    """测试3：target 错误"""
    result = run_case(agent, "target 错误", {
        "intent": "query_record",
        "target": "papers",
        "params": {}
    })
    assert "错误" in result or "不支持" in result


def test_invalid_params_type(agent):
    """测试4：params 不是 dict"""
    result = run_case(agent, "params 类型错误", {
        "intent": "query_record",
        "target": "knowledge_records",
        "params": "not_a_dict"
    })
    assert "错误" in result or "必须" in result
