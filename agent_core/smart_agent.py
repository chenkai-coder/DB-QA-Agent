# 智能问答 Agent 核心：基于 ReAct 模式的多轮推理、工具调度与对话管理
import os
import json
import queue
import re
from typing import List, Dict
from qwen_agent.agents import Assistant

from tools.agent_tools import *
from agent_core.agent_helpers import (
    _PATH_PATTERNS,
    _sanitize_prompt_text,
    _sanitize_history,
    _trim_long_fields,
    _format_thinking_display,
)


class QA_Agent_System:
    """
    执行流程：
    1. 系统接收到用户的需求 (user_query)。
    2. 发起第一轮提问给大模型。
    3. 模型开始 "Thought(思考)" -> 发现缺数据 -> 产生 "Action(指定某个工具)" 和 "Action Input(输入参数)"。
    4. 核心系统拦截到上述标记，暂停模型，去执行本地 Python 函数 (比如查询 DB)。
    5. Python 返回执行结果，打上 "Observation:" 前缀，塞入对话历史，把结果 "喂" 回给大模型。
    6. 模型根据 Observation，继续 "Thought(思考)" -> 判断条件满足？ -> 输出 "Action: Finish" 完结。
    """
    def __init__(self):
        """初始化 Agent：配置 LLM、系统提示词、上传/问答/总结三个子 Agent。"""
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-5498ce6562884bcf81f3073e857c8a1b")
        self.text_llm_cfg = {
            'model': 'qwen-plus',
            'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': self.api_key,
        }

        sys_msg = """你是一位“高级科研数据管家”与“专业学术知识顾问”。你的核心职责是只依据检索到的数据内容、结构化记录和历史对话回答问题，不要把文件名、绝对路径、目录名、预览提示或其他无关噪声当作推理依据。

【核心职能】：
    1. 数据管理：当用户明确要求扫描入库、解析比对或修改记录时，调用对应工具完成任务，但只围绕数据内容本身工作，不要利用路径字符串做语义判断。
    2. 智能问答（极其重要）：当用户提出关于某篇文献、某张图表、某个概念的具体问题（例如“这个图用到卷积了吗？”或“对比一下这两者”）时，优先根据历史对话、最近轮次上下文、已检索到的结构化内容、摘要和原文片段回答，不要把路径、文件名或目录名当作答案。

【特别要求】：
    1. 遇到直接的专业提问，不要再用那些“系统拦截了重复插入”或“记录已存入数据库”等套话。要围绕数据内容回答技术点。
    2. 遇到组合问题，务必一步步去 Action。查不到结果不要瞎编，诚实告诉用户。
    3. 所有输入里的路径、预览提示、文件系统噪声都要忽略，只保留有效内容与历史记录。
    4. 如果任务来自当前选中的图片、文本文件或文件夹，而没有显式给出路径，请在 Action Input 中使用占位符“当前选中文件”或“当前选中文件夹”，由系统在后台补全真实路径，不要把真实绝对路径写进思考内容。
    5. 在最后一轮完结时（输出 Action: Finish 后），务必在 Action Input 中给出一段自然、专业的【人类语言】答复，严禁罗列生硬的机器日志。
    
【知识检索规则】：
当用户询问某个人、作者、论文、技术、课程、项目、关键词、概念时，必须优先调用知识库工具。
如果问题包含明确字段，例如标题、作者、会议、来源、年份、ID、统计、删除、更新，优先调用 SearchKnowledgeRecords 或其他精确工具。
如果问题是模糊语义查询、近义表达、概念相似、主题相关、技术含义相近，例如“有没有动态图实时处理相关内容”“和缓存管理有关的论文”“类似事务路由的研究”，优先调用 SemanticSearchKnowledge。
SearchKnowledgeRecords 负责关键词/字段检索。
SemanticSearchKnowledge 负责语义相似度检索。
如果一个工具结果不足，可以继续调用另一个工具补充。

【画图规则】：
当用户要求“统计、分析、画图、生成图表”时，必须调用 AnalyzeKnowledgeChart，禁止调用ListAllRecords。
Action Input 必须包含 query、dimension、chart_type。
如果用户没有明确主题，query 使用 "paper" 或 "知识库记录"。
dimension 只能从以下值中选择一个：
conference、source、author、year、keywords、data_type、category、tags。
注意：
1. 用户说“会议来源”“会议/期刊来源”“不同会议来源”时，dimension 必须传 "source" 或 "conference"，推荐传 "source"。
2. 严禁传 "conference/source"。
3. 严禁在 AnalyzeKnowledgeChart 失败后改用 ListAllRecords 判断是否有来源字段。

【专有名词保留规则】：
论文标题、作者姓名、会议/期刊名称、机构名称、系统名称、英文缩写必须严格保持数据库原文形式。
如果数据库返回的是英文作者名，最终回答也必须使用英文作者名，不得翻译成中文。
例如 Feng Yao 不能改写成 冯瑶。

【ReAct 输出格式强制要求】：
除非用户只是普通闲聊，否则你必须严格按照以下格式工作：
Thought: 简要说明你要做什么。
Action: 工具名称
Action Input: JSON格式参数
当工具返回 Observation 后，再根据 Observation 继续判断。
如果已经可以回答，必须输出：
Action: Finish
Action Input: 最终给用户的自然语言回答
禁止在没有 Action: Finish 的情况下直接输出最终答案。
"""
        
        self.agent = Assistant(
            llm=self.text_llm_cfg,
            system_message=sys_msg,
            function_list=[
                'ParseAndInsertImage', 'BatchImportFiles', 'ParseAndInsertTextFile',
                'CompareImageWithRecords', 'SearchKnowledgeRecords', 'SemanticSearchKnowledge',
                'AnalyzeKnowledgeChart',
                'QueryByAuthorOrTitle', 'ListAllRecords',
                'UpdatePaperRecord', 'DeletePaperRecord',
                'DeleteRecordsByKeyword', 'CountAndGroupStatistics'
            ]
        )
        self.upload_agent = Assistant(
            llm=self.text_llm_cfg,
            system_message=(
                "你正在处理上传、解析或入库任务。你必须完全忽略任何历史对话、向量记忆、数据库记录、路径噪声和旧答案。"
                "只能根据当前轮次提供的任务描述和工具返回结果工作。对于上传任务，优先调用对应工具；若返回解析结果，则只基于该结果进行自然语言转述，不能引用任何旧内容。"
                "如果用户提到‘对照’、‘对比’、‘相近记录’、‘数据库里有没有类似内容’，必须优先调用 CompareImageWithRecords，"
                "不要用 ParseAndInsertImage 代替对照分析。"
                "如果任务来自当前选中的图片、文本文件或文件夹，但没有显式给出绝对路径，必须在 Action Input 中使用占位符“当前选中文件”或“当前选中文件夹”，绝对不要输出空字符串、猜测路径或编造文件名。"
            ),
            function_list=[
                'ParseAndInsertImage', 'BatchImportFiles', 'ParseAndInsertTextFile',
                'CompareImageWithRecords'
            ]
        )
      
        self.response_summarizer = Assistant(
            llm=self.text_llm_cfg,
            system_message=(
                "你是一个最终答复整理器。输入内容可能包含工具原文、JSON、列表、日志或推理碎片。"
                "你的任务只有一个：把输入改写成给用户看的自然语言中文答复。"
                "要求：只输出最终答复正文，不要输出思考过程，不要输出JSON，不要输出代码块，不要复述工具调用细节。"
                "若输入中出现路径、文件名、目录名或其他噪声，只保留真正有助于回答的问题内容。"
                "若输入是记录列表、删除结果或批量导入结果，要把它们概括成一段通顺的话，而不是逐行复述。"
                "特别注意：论文标题、作者姓名、会议/期刊名称、机构名称、系统名称、英文缩写必须保持输入原文，不得翻译、音译或改写。"
                "例如 Feng Yao 不能改成 冯瑶。"
            )
            # system_message="你是一个最终答复整理器。输入内容可能包含工具原文、JSON、列表、日志或推理碎片。你的任务只有一个：把输入改写成给用户看的自然语言中文答复。要求：只输出最终答复正文，不要输出思考过程，不要输出JSON，不要输出代码块，不要复述工具调用细节。若输入中出现路径、文件名、目录名或其他噪声，只保留真正有助于回答的问题内容。若输入是记录列表、删除结果或批量导入结果，要把它们概括成一段通顺的话，而不是逐行复述。"
        )
        self.chat_history: List[Dict[str, str]] = []  # 这用于装载临时的 ReAct 内生推理链
        self.global_history: List[Dict[str, str]] = []
        self.max_history_turns = 5

    def _recent_global_history(self) -> List[Dict[str, str]]:
        """获取最近 N 轮全局对话历史，清洗路径噪声后返回。"""
        max_messages = self.max_history_turns * 2
        if len(self.global_history) <= max_messages:
            return _sanitize_history(self.global_history)
        return _sanitize_history(self.global_history[-max_messages:])

    def _normalize_final_answer(self, text: str) -> str:
        """清理模型输出，去除 ReAct 格式标记和代码块，必要时调用 summarizer 整理为自然语言。"""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return "我已经完成处理，但没有拿到可展示的结果。"

        cleaned_text = re.sub(r'(?is)^.*?Action Input:\s*', '', cleaned_text)
        cleaned_text = re.sub(r'(?is)^.*?Final Answer:\s*', '', cleaned_text)
        cleaned_text = re.sub(r'(?is)^.*?Thought:\s*', '', cleaned_text)
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()

        # 有时候模型还是会返回包含 "该图片已在数据库中存在" 的日志回复，这说明拦截了
        if "该图片已在数据库中存在" in cleaned_text:
            return cleaned_text

        if cleaned_text.startswith('[') or cleaned_text.startswith('{'):
            cleaned_text = f'请把下面这段结构化结果整理成自然语言汇报，不要逐条罗列：\n{cleaned_text}'

        needs_summarize = any(token in cleaned_text for token in ['{', '}', '[', ']', 'Observation:', 'Action:', 'title:', 'author:', 'summary:'])
        # 强制拦截被判定为重复图片的硬编码回复，不要让 summarizer 去“改写”（否则它会产生编造行为）
        if "该图片已在数据库中存在" in cleaned_text:
             return cleaned_text
             
        if len(cleaned_text) > 220 or needs_summarize:
            try:
                summary_messages = [
                    {'role': 'user', 'content': f'请把下面内容整理成简洁自然的中文答复，只输出答复正文，不要列项目符号，不要输出JSON：\n{cleaned_text}'}
                ]
                summary_chunks = list(self.response_summarizer.run(messages=summary_messages))
                if summary_chunks:
                    summarized_text = summary_chunks[-1][-1].get('content', '').strip()
                    if summarized_text:
                        return summarized_text
            except Exception:
                pass

        return cleaned_text

    def _build_working_history(self, user_query: str, ui_queue: queue.Queue):
        """构建工作历史：根据问题类型（上传/问答）决定是否带历史上下文，并返回工作 prompt。"""
        # Step 1: 先判断是否为上传/解析/入库类任务（基于文本关键词或路径）
        ui_queue.put({"type": "status", "data": "1. 判断是否为上传/解析任务..."})
        contains_path = any(re.search(p, user_query) for p in _PATH_PATTERNS)

        # 先清洗文本用于关键词匹配
        clean_user_query = _sanitize_prompt_text(user_query)

        # 判断是否为上传/解析/入库任务（包含路径或关键句）
        is_upload_task = contains_path or (
                "解析当前选中" in clean_user_query
                or "入库" in clean_user_query
                or "批量扫描" in clean_user_query
                or "对比当前选中" in clean_user_query
                or "对照当前选中" in clean_user_query
        )

        if is_upload_task:
            is_compare_task = (
                "对照" in clean_user_query
                or "对比" in clean_user_query
                or "相近记录" in clean_user_query
                or "类似记录" in clean_user_query
            )
            working_history = []  # 上传/解析图片时，不要带长期的 global_history
            prompt = (
                f"当前用户的查询任务是: {clean_user_query}\n\n"
                "【重要指令】：这是一项全新的数据录入、解析或比对任务。请务必完全忘掉之前的所有记忆、文献特征和检索到的内容数据。"
                "只允许严格调用对应的解析工具（例如 ParseAndInsertImage 或 CompareImageWithRecords），并按照系统返回的最新 Observation 原客观复述，绝不要生搬硬造或合并历史内容！\n"
                "【路径规则】：如果任务来自当前选中的图片、文本文件或文件夹，而没有显式给出绝对路径，Action Input 必须写“当前选中文件”或“当前选中文件夹”，不得留空。"
            )
            if is_compare_task:
                prompt += "\n【本轮强制要求】：这是图片对照任务，必须调用 CompareImageWithRecords，并把工具返回的图片解析结果和数据库相关内容直接汇报给用户。"

        else:
            if self._should_use_history(clean_user_query):
                working_history = self._recent_global_history()

                ui_queue.put({
                    "type": "think_chunk",
                    "data": "检测到当前问题可能依赖上下文，正在加载最近历史记忆...\n"
                })

                prompt = (
                    f"当前用户的查询任务是: {clean_user_query}\n\n"
                    "【重要提示】：本轮允许参考最近历史上下文，但历史只能用于理解代词、承接关系和上下文含义。"
                    "如果用户询问数据库、知识库、记录、论文、作者、标题、来源、会议、期刊、统计、筛选或列表结果，"
                    "仍然必须优先调用 SearchKnowledgeRecords 或其他合适工具获取当前数据库结果，不能仅凭历史回答。"
                    "历史对话不能作为数据库当前内容的事实依据。"
                )
            else:
                working_history = []

                ui_queue.put({
                    "type": "think_chunk",
                    "data": "当前问题将作为新的独立任务处理，不加载历史记忆...\n"
                })

                prompt = (
                    f"当前用户的查询任务是: {clean_user_query}\n\n"
                    "【重要提示】：本轮不使用历史记忆，请把当前问题作为独立问题处理。"
                    "如果用户询问数据库、知识库、记录、论文、作者、标题、来源、会议、期刊、统计、筛选或列表结果，"
                    "必须优先调用 SearchKnowledgeRecords 或其他合适工具获取当前数据库结果后再回答。"
                    "不得凭历史印象、上一轮回答或模型常识直接声称数据库中有什么或没有什么。"
                )

        working_history.append({'role': 'user', 'content': prompt})
        return is_upload_task, working_history

    def _should_use_history(self, user_query: str) -> bool:
        """判断当前问题是否需要历史上下文，默认只有连续追问时才使用。"""
        """
        判断当前问题是否需要使用最近历史上下文。
        默认不使用历史，只有明显属于连续追问时才使用历史，
        避免旧答案污染当前数据库检索结果。
        """
        query = _sanitize_prompt_text(user_query)

        follow_up_keywords = [
            "这个", "那个", "它", "他们", "这些", "上述", "上面",
            "刚才", "上一条", "前面", "继续", "进一步",
            "详细说", "展开说", "再说说", "对比一下",
            "为什么", "那它", "那这个", "第一个", "第二个",
            "第一篇", "第二篇", "这篇", "那篇"
        ]

        return any(keyword in query for keyword in follow_up_keywords)

    def _run_agent_once(self, loop_i: int, is_upload_task: bool, working_history: list, ui_queue: queue.Queue) -> str:
        """执行一轮 Agent 推理，流式接收模型输出并实时推送到 UI。"""
        ui_queue.put({"type": "stream_chunk", "data": f"\n\n======================\n▶ 脑回路 {loop_i} 判断中...\n"})
        ui_queue.put({"type": "think_chunk", "data": "正在理解问题，并判断本轮是否需要重新检索数据库或调用工具...\n"})

        active_agent = self.upload_agent if is_upload_task else self.agent
        generator = active_agent.run(messages=working_history)
        full_resp = ""
        last_len = 0

        for chunks in generator:
            current_str = chunks[-1]['content']
            diff_str = current_str[last_len:]

            if diff_str:
                # 保留阶段提示的同时，也恢复模型实时输出
                display_str = _format_thinking_display(diff_str)

                ui_queue.put({
                    "type": "think_chunk",
                    "data": display_str
                })

                full_resp += diff_str
                last_len = len(current_str)

        working_history.append({'role': 'assistant', 'content': full_resp})
        return full_resp

    def _handle_max_loop(self, loop_i: int, max_loops: int, full_resp: str, ui_queue: queue.Queue):
        """判断是否达到最大循环次数，达到则强制终止并返回当前结果。"""
        # 如果已经是最后一轮依然没有 Finish，强制总结并跳出
        if loop_i == max_loops and "Action: Finish" not in full_resp:
            final_answer = self._normalize_final_answer(full_resp)
            ui_queue.put({"type": "think_chunk", "data": f"\n\n⚠️ 运算超过最高设定的 {max_loops} 次循环阀值，安全阀切断运行。\n"})
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer
        return False, ""

    def _handle_finish_response(self, full_resp: str, ui_queue: queue.Queue):
        """检测模型是否输出 Finish 标记，若是则提取最终答案并终止循环。"""
        # 退出判断
        if "Action: Finish" in full_resp:
            ui_queue.put({
                "type": "think_chunk",
                "data": "模型已判断当前问题不需要继续调用工具，正在整理直接回答...\n"
            })

            finish_idx = full_resp.find("Action: Finish")
            following_text = full_resp[finish_idx + len("Action: Finish"):].strip()

            if following_text.startswith("Action Input:"):
                final_answer = following_text.replace("Action Input:", "", 1).strip()
            else:
                final_answer = following_text if following_text else "任务已结束。"

            if not final_answer:
                final_answer = "好的，我已经完成了您的指令任务。"

            final_answer = self._normalize_final_answer(final_answer)

            ui_queue.put({"type": "think_chunk", "data": "\n✅ 最终意图达成，终止判断引擎。\n"})
            ui_queue.put({"type": "think_chunk", "data": "🏁 当前问题已分析完成，正在生成最终回答...\n"})
            ui_queue.put({"type": "status", "data": "正在整理最终答案..."})
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer

        return False, ""

    def _parse_action(self, full_resp: str):
        """从模型输出中解析工具名称和参数。"""
        tool_match = re.search(r'Action:\s*(.*?)(?=\nAction Input:|$)', full_resp, re.IGNORECASE)
        param_match = re.search(r'Action Input:\s*(.*?)(?=\nObservation:|$)', full_resp, re.IGNORECASE)
        return tool_match, param_match

    def _execute_tool(self, tool_name: str, param_str: str, ui_queue: queue.Queue) -> str:
        """从 TOOL_REGISTRY 中获取工具实例并执行，返回观察结果。"""
        ui_queue.put({"type": "status", "data": f"系统调度底座: {tool_name}"})
        ui_queue.put({
            "type": "think_chunk",
            "data": f"\n\n[🔧 执行动作层] -> 工具: {tool_name}, 传入: {param_str}\n"
        })

        try:
            from qwen_agent.tools.base import TOOL_REGISTRY
            if tool_name in TOOL_REGISTRY:
                tool_instance = TOOL_REGISTRY[tool_name]()
                obs = tool_instance.call(param_str, ui_queue=ui_queue)
            else:
                obs = "Agent 未知异常：企图呼叫一个不存在的接口。"
        except Exception as e:
            obs = str(e)

        return obs

    def _append_observation(self, obs: str, working_history: list) -> str:
        """将工具执行结果包装为 Observation 格式并追加到工作历史。"""
        obs_feedback = _sanitize_prompt_text(
            f"Observation: {obs}\n请依据此处执行报告继续判断下一步策略。"
        )
        working_history.append({'role': 'user', 'content': obs_feedback})
        return obs_feedback

    def _handle_non_action_response(self, full_resp: str, working_history: list, ui_queue: queue.Queue):
        """处理没有触发工具调用的情况，强制模型整理为最终答复。"""
        # 如果没有任何 Action 这个动作，只是闲聊或失控，强制让他回复用户
        if "Action:" not in full_resp:
            ui_queue.put({
                "type": "think_chunk",
                "data": "模型未请求工具调用，正在将当前回答整理为最终结果...\n"
            })

            final_answer = self._normalize_final_answer(full_resp)

            ui_queue.put({"type": "think_chunk", "data": "✅ 当前问题已处理完成。\n"})
            ui_queue.put({"type": "status", "data": "正在整理最终答案..."})
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer

        working_history.append({
            'role': 'user',
            'content': "Observation: 没有识别到标准 ReAct 工具执行格式。若想回答用户请使用 Action: Finish 开头并附上 Action Input"
        })
        return False, ""

    def _save_global_history(self, user_query: str, final_answer: str, is_upload_task: bool):
        """将非上传任务的问答对保存到全局历史，用于后续上下文追溯。"""
        if not is_upload_task:
            self.global_history.append({'role': 'user', 'content': _sanitize_prompt_text(user_query)})
            self.global_history.append({'role': 'assistant', 'content': _sanitize_prompt_text(final_answer)})
            if len(self.global_history) > 10:
                self.global_history = self.global_history[-10:]

    def _try_direct_upload_dispatch(self, user_query: str, ui_queue: queue.Queue):
        """尝试直接执行上传类任务（对照/批量导入），跳过完整 ReAct 循环。"""
        clean_query = _sanitize_prompt_text(user_query)

        is_compare_task = (
            "对照" in clean_query
            or "对比" in clean_query
            or "相近记录" in clean_query
            or "类似记录" in clean_query
        )
        if is_compare_task:
            obs = self._execute_tool(
                "CompareImageWithRecords",
                json.dumps({"image_path": CURRENT_IMAGE_PLACEHOLDER}, ensure_ascii=False),
                ui_queue
            )
            final_answer = obs.strip() or "已完成图片对照，但没有拿到可展示的结果。"
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer

        is_batch_import_task = (
            ("批量" in clean_query or "扫描" in clean_query)
            and ("文件夹" in clean_query or "目录" in clean_query)
            and ("导入" in clean_query or "入库" in clean_query)
        )
        if is_batch_import_task:
            obs = self._execute_tool(
                "BatchImportFiles",
                json.dumps({"folder_path": CURRENT_FOLDER_PLACEHOLDER}, ensure_ascii=False),
                ui_queue
            )
            final_answer = self._normalize_final_answer(obs)
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer

        return False, ""

    def stream_chat_query(self, user_query: str, ui_queue: queue.Queue):
        """主入口：启动 ReAct 推理循环，通过 ui_queue 流式推送结果到 GUI。"""
        try:
            is_upload_task, working_history = self._build_working_history(user_query, ui_queue)

            ui_queue.put({"type": "start_stream"})
            ui_queue.put({"type": "status", "data": "🚀 进入独立思考循环..."})

            if is_upload_task:
                handled, final_answer = self._try_direct_upload_dispatch(user_query, ui_queue)
                if handled:
                    self._save_global_history(user_query, final_answer, is_upload_task)
                    ui_queue.put({"type": "end_stream"})
                    ui_queue.put({"type": "status", "data": "就绪"})
                    return

            max_loops = 5
            final_answer = ""

            for loop_i in range(1, max_loops + 1):
                full_resp = self._run_agent_once(loop_i, is_upload_task, working_history, ui_queue)

                is_finished, final_answer = self._handle_max_loop(loop_i, max_loops, full_resp, ui_queue)
                if is_finished:
                    break

                tool_match, param_match = self._parse_action(full_resp)

                ui_queue.put({
                    "type": "think_chunk",
                    "data": "已完成本轮判断，正在检查是否需要执行工具...\n"
                })

                if tool_match and param_match:
                    tool_name = tool_match.group(1).strip()
                    param_str = param_match.group(1).strip()

                    if tool_name == "Finish":
                        is_finished, final_answer = self._handle_finish_response(full_resp, ui_queue)
                        if is_finished:
                            break

                    obs = self._execute_tool(tool_name, param_str, ui_queue)
                    obs_feedback = self._append_observation(obs, working_history)

                    ui_queue.put({
                        "type": "think_chunk",
                        "data": f"\n[📢 报备反馈] -> {obs_feedback}\n"
                    })

                    ui_queue.put({
                        "type": "think_chunk",
                        "data": "\n[📢 工具执行完成，已收到返回结果。]\n"
                    })

                    if tool_name in (
                            # 'ListAllRecords',
                            'DeleteRecordsByKeyword',
                            'BatchImportFiles',
                            'ParseAndInsertImage',
                            'ParseAndInsertTextFile',
                            'CompareImageWithRecords',
                    ):
                        if tool_name == 'CompareImageWithRecords':
                            final_answer = obs.strip() or "已完成图片对照，但没有拿到可展示的结果。"
                        else:
                            final_answer = self._normalize_final_answer(obs)
                        ui_queue.put({"type": "final_answer", "data": final_answer})
                        break

                else:
                    is_finished, final_answer = self._handle_finish_response(full_resp, ui_queue)
                    if is_finished:
                        break

                    is_finished, final_answer = self._handle_non_action_response(
                        full_resp,
                        working_history,
                        ui_queue
                    )
                    if is_finished:
                        break

            self._save_global_history(user_query, final_answer, is_upload_task)

            ui_queue.put({"type": "end_stream"})
            ui_queue.put({"type": "status", "data": "就绪"})

        except Exception as e:
            ui_queue.put({"type": "error", "data": str(e)})
