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
        # self.answer_extractor = Assistant(
        #     llm=self.text_llm_cfg,
        #     system_message=(
        #         "你是一个通用知识抽取器。"
        #         "如果用户问题包含限定条件，例如标题、会议、期刊、作者、年份、地点、类别等，必须先按限定条件过滤记录，只输出满足条件的条目。"
        #         "你的任务不是总结整条记录，而是根据用户原始问题，从给定知识记录中抽取能直接回答问题的信息。"
        #         "所有知识都按“实体、属性、关系、证据”来理解。"
        #         "如果用户问某对象是什么，抽取定义、属性和证据；"
        #         "如果用户问某对象有哪些内容，抽取相关条目；"
        #         "如果用户问某人与某物的关系，抽取 subject-relation-object 关系；"
        #         "如果用户问原因、区别、流程、用途，抽取对应原因链、差异点、步骤或用途。"
        #         "必须基于给定记录，不得凭空补充。"
        #         "不要泛泛总结整条记录。"
        #         "专有名词必须保持原文，不得翻译、音译或改写。"
        #         "如果记录中没有依据，回答：根据当前检索记录无法确定。"
        #         "只输出抽取结果，不要输出推理过程，不要输出JSON。"
        #     )
        # )
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
        max_messages = self.max_history_turns * 2
        if len(self.global_history) <= max_messages:
            return _sanitize_history(self.global_history)
        return _sanitize_history(self.global_history[-max_messages:])

    def _normalize_final_answer(self, text: str) -> str:
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

    # def _collect_all_records_summary(self, query: str) -> str:
    #     try:
    #         from qwen_agent.tools.base import TOOL_REGISTRY
    #         tool_instance = TOOL_REGISTRY['ListAllRecords']()
    #         raw = tool_instance.call('{}')
    #         return self._normalize_final_answer(raw)
    #     except Exception:
    #         return f'我暂时没法完整列出数据库内容，但我可以继续按关键词 {query} 帮你查。'

    # def _build_generic_extraction_context(self, records: list) -> str:
    #     """
    #     将数据库返回的完整记录整理成更适合抽取层阅读的通用上下文。
    #     不限定论文结构，兼容笔记、报告、课程、项目、代码说明、会议纪要等内容。
    #     """
    #     context_blocks = []
    #
    #     for r in records:
    #         record_id = r.get("id", "")
    #         title = r.get("title", "")
    #         data_type = r.get("data_type", "")
    #         category = r.get("category", "")
    #         tags = r.get("tags", "")
    #         summary = r.get("summary", "")
    #         raw_text = r.get("raw_text", "")
    #         keyword_text = r.get("keyword_text", "")
    #         entity_text = r.get("entity_text", "")
    #         structured_json = r.get("structured_json", "")
    #
    #         if isinstance(structured_json, str):
    #             try:
    #                 structured_json = json.loads(structured_json)
    #             except Exception:
    #                 structured_json = {}
    #
    #         block = []
    #         block.append(f"记录ID: {record_id}")
    #         block.append(f"标题: {title}")
    #         block.append(f"类型: {data_type}")
    #         block.append(f"分类: {category}")
    #         block.append(f"标签: {tags}")
    #         block.append(f"关键词: {keyword_text}")
    #         block.append(f"实体: {entity_text}")
    #         block.append(f"摘要: {summary}")
    #
    #         # 通用结构化内容展开
    #         if isinstance(structured_json, dict):
    #             sections = structured_json.get("sections", [])
    #             items = structured_json.get("items", [])
    #             tables = structured_json.get("tables", [])
    #             important_facts = structured_json.get("important_facts", [])
    #
    #             if sections:
    #                 block.append("结构化章节:")
    #                 for sec in sections:
    #                     sec_title = sec.get("section_title", "")
    #                     block.append(f"- 章节: {sec_title}")
    #
    #                     content = sec.get("content", "")
    #                     if isinstance(content, list):
    #                         for item in content:
    #                             block.append(f"  - 条目: {json.dumps(item, ensure_ascii=False)}")
    #                     else:
    #                         block.append(f"  - 内容: {content}")
    #
    #             if items:
    #                 block.append("结构化条目:")
    #                 for item in items:
    #                     block.append(f"- {json.dumps(item, ensure_ascii=False)}")
    #
    #             if tables:
    #                 block.append("表格内容:")
    #                 for table in tables:
    #                     block.append(f"- {json.dumps(table, ensure_ascii=False)}")
    #
    #             if important_facts:
    #                 block.append("重要事实:")
    #                 for fact in important_facts:
    #                     block.append(f"- {fact}")
    #
    #         # 原文只截取，避免上下文爆炸
    #         if raw_text:
    #             raw_preview = raw_text[:5000]
    #             block.append(f"原文片段: {raw_preview}")
    #
    #         context_blocks.append("\n".join(block))
    #
    #     return "\n\n====================\n\n".join(context_blocks)
    #
    # def _extract_relevant_info(self, user_query: str, obs: str) -> str:
    #     """
    #     通用抽取层入口。
    #     数据库负责粗检索，本函数负责把检索到的记录转化为“可回答问题的证据”。
    #     不限定论文，适用于论文、笔记、报告、课程、项目、代码说明、会议纪要等。
    #     """
    #     try:
    #         records = json.loads(obs)
    #
    #         if not isinstance(records, list) or not records:
    #             return "根据当前检索记录无法确定。"
    #
    #         clean_context = self._build_generic_extraction_context(records)
    #
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     "用户原始问题：\n"
    #                     f"{user_query}\n\n"
    #                     "下面是从知识库检索到的相关记录，已经整理为通用上下文：\n"
    #                     f"{clean_context}\n\n"
    #                     "请根据用户问题，从上述记录中抽取直接相关的信息。\n"
    #                     "要求：\n"
    #                     "1. 不要总结整条记录。\n"
    #                     "2. 只抽取能回答用户问题的实体、属性、关系、步骤、原因、区别、用途或条目。\n"
    #                     "3. 如果用户问“谁/哪些/标题/作者/时间/地点/来源”，只返回对应字段或条目。\n"
    #                     "4. 如果用户问“为什么/怎么做/流程/区别/作用”，抽取对应原因、步骤、差异或用途。\n"
    #                     "5. 每条结论尽量带上记录ID或标题作为依据。\n"
    #                     "6. 专有名词保持原文，不得翻译。\n"
    #                     "7. 如果记录中没有依据，回答：根据当前检索记录无法确定。"
    #                 )
    #             }
    #         ]
    #
    #         chunks = list(self.answer_extractor.run(messages=messages))
    #
    #         if not chunks:
    #             return "未能从检索记录中抽取出相关信息。"
    #
    #         extracted = chunks[-1][-1].get("content", "").strip()
    #         return extracted or "未能从检索记录中抽取出相关信息。"
    #
    #     except Exception as e:
    #         return f"信息抽取异常: {str(e)}"
    #
    # def _summarize_extracted_answer(self, user_query: str, extracted_info: str) -> str:
    #     """
    #     将抽取层结果整理为最终用户回答。
    #     注意：这里只能基于抽取结果表达，不能再扩展、联想或总结原始记录。
    #     """
    #     try:
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     "用户原始问题：\n"
    #                     f"{user_query}\n\n"
    #                     "下面是抽取层已经提取出的相关信息：\n"
    #                     f"{extracted_info}\n\n"
    #                     "请基于抽取结果回答用户问题。"
    #                     "要求答案自然、准确、简洁。"
    #                     "如果抽取结果已经是列表，必须保持列表结构，不要改写成综述段落。"
    #                     "不要加入抽取结果之外的信息。"
    #                     "不要重新总结整条记录。"
    #                     "专有名词、论文标题、作者姓名、会议/期刊名称、系统名称、英文缩写必须保持原文。"
    #                 )
    #             }
    #         ]
    #
    #         chunks = list(self.response_summarizer.run(messages=messages))
    #         if not chunks:
    #             return extracted_info
    #
    #         final_text = chunks[-1][-1].get("content", "").strip()
    #         return final_text or extracted_info
    #
    #     except Exception:
    #         return extracted_info

    def _build_working_history(self, user_query: str, ui_queue: queue.Queue):
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
        # 如果已经是最后一轮依然没有 Finish，强制总结并跳出
        if loop_i == max_loops and "Action: Finish" not in full_resp:
            final_answer = self._normalize_final_answer(full_resp)
            ui_queue.put({"type": "think_chunk", "data": f"\n\n⚠️ 运算超过最高设定的 {max_loops} 次循环阀值，安全阀切断运行。\n"})
            ui_queue.put({"type": "final_answer", "data": final_answer})
            return True, final_answer
        return False, ""

    def _handle_finish_response(self, full_resp: str, ui_queue: queue.Queue):
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
        tool_match = re.search(r'Action:\s*(.*?)(?=\nAction Input:|$)', full_resp, re.IGNORECASE)
        param_match = re.search(r'Action Input:\s*(.*?)(?=\nObservation:|$)', full_resp, re.IGNORECASE)
        return tool_match, param_match

    def _execute_tool(self, tool_name: str, param_str: str, ui_queue: queue.Queue) -> str:
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
        obs_feedback = _sanitize_prompt_text(
            f"Observation: {obs}\n请依据此处执行报告继续判断下一步策略。"
        )
        working_history.append({'role': 'user', 'content': obs_feedback})
        return obs_feedback

    # def _handle_search_records_result(self, user_query: str, obs: str, ui_queue: queue.Queue) -> str:
    #     feedback_text = "\n[📢 检索结果]\n"
    #
    #     try:
    #         records = json.loads(obs)
    #         feedback_text += f"命中记录数量：{len(records)}\n"
    #
    #         for r in records[:3]:
    #             title = r.get("title", "未知标题")
    #             feedback_text += f"- 命中记录：{title}\n"
    #
    #         feedback_text += "正在从命中记录中抽取与问题直接相关的信息...\n"
    #
    #     except Exception:
    #         feedback_text += "检索完成，但返回内容不是标准 JSON。\n"
    #
    #     ui_queue.put({
    #         "type": "think_chunk",
    #         "data": feedback_text
    #     })
    #
    #     extracted_info = self._extract_relevant_info(user_query, obs)
    #
    #     ui_queue.put({
    #         "type": "think_chunk",
    #         "data": f"\n[🧩 抽取结果]\n{_format_thinking_display(extracted_info, max_field_len=200)}\n"
    #     })
    #
    #     final_answer = self._summarize_extracted_answer(user_query, extracted_info)
    #
    #     ui_queue.put({
    #         "type": "think_chunk",
    #         "data": "\n✅ 已完成抽取与总结。\n"
    #     })
    #
    #     ui_queue.put({
    #         "type": "think_chunk",
    #         "data": "🏁 当前问题已分析完成，正在生成最终回答...\n"
    #     })
    #
    #     ui_queue.put({
    #         "type": "status",
    #         "data": "正在整理最终答案..."
    #     })
    #
    #     ui_queue.put({
    #         "type": "final_answer",
    #         "data": final_answer
    #     })
    #     return final_answer

    def _handle_non_action_response(self, full_resp: str, working_history: list, ui_queue: queue.Queue):
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
        if not is_upload_task:
            self.global_history.append({'role': 'user', 'content': _sanitize_prompt_text(user_query)})
            self.global_history.append({'role': 'assistant', 'content': _sanitize_prompt_text(final_answer)})
            if len(self.global_history) > 10:
                self.global_history = self.global_history[-10:]

    def _try_direct_upload_dispatch(self, user_query: str, ui_queue: queue.Queue):
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
