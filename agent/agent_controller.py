"""
AgentController - 统一的 Agent 控制器

提供 handle_intent_json 和 handle_user_input 两个核心接口，
供测试文件和外部调用方使用。
"""

import json
import queue
from typing import Any, Dict, Optional

from agent_core.smart_agent import QA_Agent_System
from db.storage_service import StorageService


class AgentController:
    """
    Agent 控制器，封装 QA_Agent_System 并提供结构化意图分发能力。
    """

    def __init__(self):
        self.agent = QA_Agent_System()
        self.storage = StorageService(db_name="app.db")

    # ==================== 公开接口 ====================

    def handle_user_input(self, user_input: str, session_id: str) -> str:
        """
        处理自然语言用户输入，通过 QA_Agent_System 进行意图识别与执行。

        Args:
            user_input: 用户输入的自然语言文本
            session_id: 会话 ID

        Returns:
            处理结果文本
        """
        ui_queue = queue.Queue()
        self.agent.stream_chat_query(user_query=user_input, ui_queue=ui_queue)

        # 收集最终结果
        final_answer = ""
        while True:
            try:
                msg = ui_queue.get(timeout=30)
                if msg["type"] == "final_answer":
                    final_answer = msg["data"]
                elif msg["type"] == "error":
                    final_answer = f"处理出错: {msg['data']}"
                    break
                elif msg["type"] == "end_stream":
                    break
            except queue.Empty:
                final_answer = "处理超时，请重试。"
                break

        return final_answer or "处理完成，但未生成有效回答。"

    def handle_intent_json(
        self,
        intent_json: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """
        处理结构化意图 JSON，直接分发到对应的数据库操作。

        Args:
            intent_json: 结构化意图字典，包含 intent、target、params 等字段
            user_input: 原始用户输入（用于日志/上下文）
            session_id: 会话 ID

        Returns:
            操作结果文本
        """
        # 参数校验
        intent = intent_json.get("intent")
        if not intent:
            return "错误：缺少 intent 字段"

        target = intent_json.get("target", "knowledge_records")
        params = intent_json.get("params", {})
        if not isinstance(params, dict):
            return "错误：params 必须是字典类型"

        if target != "knowledge_records":
            return f"错误：不支持的 target '{target}'，当前仅支持 'knowledge_records'"

        # 根据 intent 分发
        intent_map = {
            "insert_record": self._handle_insert,
            "query_record": self._handle_query,
            "get_record_detail": self._handle_detail,
            "update_record": self._handle_update,
            "delete_record": self._handle_delete,
            "summarize_records": self._handle_summarize,
        }

        handler = intent_map.get(intent)
        if handler is None:
            return f"错误：不支持的 intent '{intent}'"

        try:
            return handler(params, user_input, session_id)
        except Exception as e:
            return f"执行出错: {str(e)}"

    # ==================== 内部处理方法 ====================

    def _handle_insert(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理插入记录"""
        record_id = self.storage.insert_record(params)
        return json.dumps(
            {"status": "success", "message": "记录插入成功", "record_id": record_id},
            ensure_ascii=False,
        )

    def _handle_query(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理查询记录"""
        condition = {}
        if params.get("keyword"):
            condition["keyword"] = params["keyword"]
        if params.get("author"):
            condition["author"] = params["author"]
        if params.get("data_type"):
            condition["data_type"] = params["data_type"]

        records = self.storage.query_record(condition)
        return json.dumps(records, ensure_ascii=False, default=str)

    def _handle_detail(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理获取记录详情"""
        record_id = params.get("record_id")
        if record_id is None:
            return "错误：缺少 record_id 参数"

        record = self.storage.get_record_by_id(record_id)
        if record is None:
            return json.dumps(
                {"status": "error", "message": f"未找到 ID 为 {record_id} 的记录"},
                ensure_ascii=False,
            )
        return json.dumps(record, ensure_ascii=False, default=str)

    def _handle_update(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理更新记录"""
        record_id = params.get("record_id")
        if record_id is None:
            return "错误：缺少 record_id 参数"

        data = params.get("data", {})
        if not isinstance(data, dict) or not data:
            return "错误：缺少有效的 data 字段"

        self.storage.update_record(record_id, data)
        return json.dumps(
            {"status": "success", "message": f"记录 {record_id} 更新成功"},
            ensure_ascii=False,
        )

    def _handle_delete(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理删除记录"""
        record_id = params.get("record_id")
        if record_id is None:
            return "错误：缺少 record_id 参数"

        self.storage.delete_record(record_id)
        return json.dumps(
            {"status": "success", "message": f"记录 {record_id} 删除成功"},
            ensure_ascii=False,
        )

    def _handle_summarize(
        self,
        params: Dict[str, Any],
        user_input: str,
        session_id: str,
    ) -> str:
        """处理总结记录"""
        condition = {}
        if params.get("keyword"):
            condition["keyword"] = params["keyword"]
        if params.get("data_type"):
            condition["data_type"] = params["data_type"]

        records = self.storage.query_record(condition)

        if not records:
            return "未找到符合条件的记录，无法生成总结。"

        # 使用 agent 的 summarizer 生成总结
        summary_text = json.dumps(records, ensure_ascii=False, default=str)
        try:
            summarized = self.agent._normalize_final_answer(summary_text)
            return summarized
        except Exception:
            return f"找到 {len(records)} 条相关记录，但总结生成失败。"
