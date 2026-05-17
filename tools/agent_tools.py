# Agent 工具集：图片解析入库、批量导入、知识检索、语义搜索、统计分析等注册工具
import json
import os
import re
import threading
import queue
import time
from typing import List
from PIL import Image, ImageStat
from qwen_agent.agents import Assistant

from database.vector_index import vector_index
from db.storage_service import StorageService
from qwen_agent.tools.base import BaseTool, register_tool

# 获取数据库统一访问对象
storage = StorageService(db_name="app.db")

# 配置视觉大模型与文本模型 
_DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-5498ce6562884bcf81f3073e857c8a1b")

# 必须显式指定为 `qwenvl_oai`
vl_llm_cfg = {
    'model': 'qwen-vl-plus',
    'model_type': 'qwenvl_oai',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': _DASHSCOPE_API_KEY,
    'generate_cfg': {
        'temperature': 0.1,
        'max_retries': 2,
    },
}

text_llm_cfg = {
    'model': 'qwen-plus',
    'model_type': 'oai',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': _DASHSCOPE_API_KEY,
    'generate_cfg': {
        'temperature': 0.2,
        'max_retries': 2,
    },
}
# 动态工厂函数，避免由于模型自身带有长期会话记忆导致解析新内容时混入旧内容
def get_vl_parser():
    """创建视觉解析器实例，用于图片内容结构化提取（JSON 格式）。"""
    return Assistant(
        llm=vl_llm_cfg,
        system_message="""你是一个专业的图像内容分析专家。请从图片中进行深度分析，提取所有关键信息并输出严格的JSON字符串。
【任务要求】：
1. 仔细分析图片中的所有内容、结构、文字、图表、公式等
1.1 严格只依据图片中真实可见的内容作答，绝对不要根据文件名、路径、目录名、历史对话或数据库记录推测内容
2. title: 提取或概括图片的主标题
3. author: 提取作者名称，若图中没有明确作者则为 Unknown
4. raw_text: 识别出的所有文本内容（完整保留，不截断）
5. summary: 这是最重要的字段，请生成【详细的内容总结】，包括：
   - 图片的主要内容和核心信息
   - 所有显著的技术细节、数据、参数
   - 图表、表格中的关键数据
   - 图中出现的所有专业术语和概念
   - 内容的实际应用场景或意义
   请确保 summary 足够详细（至少 200 字），让 AI 能充分理解图片内容
6. 如果图片主要是文档页、论文页、书页、截图、扫描件、课件或表格，请优先执行 OCR 和版面理解
7. 如果图片是空白图、纯色图、严重模糊图、严重乱码图、噪声图、损坏截图，或看不出稳定有效内容，必须明确拒绝入库：
   - is_rejected: true
   - reject_reason: 简明说明拒绝原因
   - raw_text 和 summary 不要编造
8. 如果某个字段无法从图片中直接确认，请返回 Unknown 或空字符串，不要编造
只输出 JSON，不要返回任何其他内容。"""
    )


def get_vl_ocr_parser():
    """创建严格 OCR 模式的解析器，只提取图片中真实可见的文字。"""
    return Assistant(
        llm=vl_llm_cfg,
        system_message="""你是一个严格的OCR与版面识别引擎。
【硬性要求】：
1. 只能输出图片中真实可见的文字与版面信息，禁止脑补不存在的图表、曲线、公式、流程、作者或结论。
2. 如果图片是文档页、论文页、书页、课件、截图、扫描件或表格，必须优先尽可能逐字提取正文、标题、小标题、表头、编号、批注等文字。
3. 看不清的内容可以省略或留空，但绝对不能编造。
4. 如果图片接近空白、纯色、严重噪声、严重模糊、文字大面积乱码，或无法稳定辨认有效内容，请返回 is_rejected=true，并给出 reject_reason。
5. 只输出严格 JSON，不要输出解释，不要输出 Markdown 代码块。"""
    )


def get_text_parser():
    """创建文本解析器实例，用于长文本内容的结构化提取。"""
    return Assistant(
        llm=text_llm_cfg,
        system_message="""你是一个专业的深度内容分析师。请对输入的文本进行全面深入的分析，返回严格的JSON字符串。
【分析要求】：
1. title: 提取或概括文本的标题/主题
2. author: 提取作者名称，若无则为 Unknown
3. summary: 这是最重要的字段，请生成【详尽的内容总结】，包括：
   - 文本的核心观点和主要内容
   - 所有关键数据、数字、统计信息
   - 重要的技术细节和专业概念
   - 作者的主要论证点和结论
   - 内容的实际应用价值和意义
   - 其他重要的细节信息
   请确保 summary 非常详细（至少 300 字），完全覆盖文本的所有重要信息，让 AI 能充分利用这些内容回答相关问题
只输出 JSON，不要返回其他内容。"""
    )

CURRENT_FILE_PLACEHOLDER = '当前选中文件'
CURRENT_FOLDER_PLACEHOLDER = '当前选中文件夹'
CURRENT_IMAGE_PLACEHOLDER = '当前选中的图片'
CURRENT_TEXT_PLACEHOLDER = '当前选中的文本'
CURRENT_SELECTED_FILE_PLACEHOLDER = '当前选中的文件'
CURRENT_SELECTED_FOLDER_PLACEHOLDER = '当前选中的文件夹'

_PENDING_PATHS = {
    'image_path': None,
    'file_path': None,
    'folder_path': None,
}


def set_pending_path(kind: str, path: str):
    """记录用户从文件选择器选中的路径，供 Agent 使用占位符时替换。"""
    if kind in _PENDING_PATHS:
        _PENDING_PATHS[kind] = path


def _resolve_pending_path(kind: str, value: str):
    """将 Agent 输出的占位符（如“当前选中文件”）替换为真实路径，替换后清空缓存。"""
    value = (value or '').strip()
    placeholder_values = {
        CURRENT_FILE_PLACEHOLDER,
        CURRENT_FOLDER_PLACEHOLDER,
        CURRENT_IMAGE_PLACEHOLDER,
        CURRENT_TEXT_PLACEHOLDER,
        CURRENT_SELECTED_FILE_PLACEHOLDER,
        CURRENT_SELECTED_FOLDER_PLACEHOLDER,
        '',
    }
    if value in placeholder_values or ('当前选中' in value and not os.path.isabs(value)):
        pending = _PENDING_PATHS.get(kind)
        if pending:
            _PENDING_PATHS[kind] = None  # 阅后即焚：一旦提取给 Agent，立刻清空该占位符缓存，防止后续错乱记忆重复调用
            return pending
    return value


def clear_pending_path(kind: str):
    """清空指定类型的待记录路径。"""
    if kind in _PENDING_PATHS:
        _PENDING_PATHS[kind] = None


def clear_all_pending_paths():
    """清空所有类型的待记录路径。"""
    for key in _PENDING_PATHS:
        _PENDING_PATHS[key] = None


def _parse_model_json(response_content: str) -> dict:
    """从模型输出中提取 JSON，清理 markdown 代码块和 think 标签。"""
    clean_str = _clean_model_output(response_content)
    if not clean_str:
        return {}

    json_candidate = _extract_first_json_object(clean_str)
    for candidate in [clean_str, json_candidate]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def _clean_model_output(response_content: str) -> str:
    """移除模型输出中的 think 标签和代码块包裹。"""
    text = str(response_content or "")
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """从文本中定位第一个完整 JSON 对象，处理嵌套和转义引号。"""
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def _is_unknown_text(value) -> bool:
    """判断文本是否为占位值（unknown/无法识别/空等）。"""
    text = str(value or "").strip()
    if not text:
        return True
    return text.lower() in {
        "unknown", "n/a", "na", "none", "null",
        "未识别", "无法识别", "未知", "无"
    }


def _unique_texts(items, max_items: int = 12) -> List[str]:
    """去重文本列表，过滤未知值，返回前 N 个唯一条目。"""
    results = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or _is_unknown_text(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
        if len(results) >= max_items:
            break
    return results


def _first_meaningful_line(text: str) -> str:
    """从多行文本中提取第一条有意义的行，用作标题候选。"""
    for line in re.split(r"[\r\n]+", str(text or "")):
        clean_line = re.sub(r"\s+", " ", line).strip(" -:;,.，。；：")
        if len(clean_line) >= 4 and not _is_unknown_text(clean_line):
            return clean_line[:80]
    return ""


def _derive_title_from_image_content(raw_text: str, summary: str) -> str:
    """从图片解析结果中智能提取标题，优先使用首行有意义文字。"""
    title = _first_meaningful_line(raw_text)
    if title:
        return title

    summary = re.sub(r"\s+", " ", str(summary or "")).strip()
    if summary and not _is_unknown_text(summary):
        for piece in re.split(r"[。！？.!?\n]", summary):
            clean_piece = piece.strip(" -:;,.，。；：")
            if len(clean_piece) >= 4:
                return clean_piece[:50]
    return ""


def _extract_image_terms(*texts, max_items: int = 12) -> List[str]:
    """从多段文本中提取中英文术语，过滤停用词。"""
    stopwords = {
        "unknown", "image", "summary", "title", "author", "content", "figure",
        "图片", "图像", "内容", "标题", "作者", "摘要", "总结", "分析", "结构",
        "显示", "可见", "信息", "技术", "方法", "结果", "研究", "模型", "数据",
    }
    candidates = []
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-+/]{2,}|[\u4e00-\u9fff]{2,12}", str(text or "")):
            cleaned = token.strip(" ,.;:()[]{}<>，。；：")
            if not cleaned or cleaned.isdigit() or cleaned.lower() in stopwords:
                continue
            candidates.append(cleaned)
    return _unique_texts(candidates, max_items=max_items)


def _normalize_image_parse_result(parsed_data: dict) -> dict:
    """标准化图片解析结果：补全缺失字段、生成关键词和实体列表、构建详细摘要。"""
    if not isinstance(parsed_data, dict):
        parsed_data = {}

    title = str(parsed_data.get("title", "") or "").strip()
    author = str(parsed_data.get("author", "") or "").strip()
    raw_text = str(parsed_data.get("raw_text", "") or "").strip()
    summary = str(parsed_data.get("summary", "") or "").strip()
    is_rejected = _boolish(parsed_data.get("is_rejected"))
    reject_reason = str(parsed_data.get("reject_reason", "") or "").strip()

    keywords = parsed_data.get("keywords", [])
    entities = parsed_data.get("entities", [])
    if not isinstance(keywords, list):
        keywords = _extract_image_terms(keywords, max_items=8)
    if not isinstance(entities, list):
        entities = _extract_image_terms(entities, max_items=8)

    if _is_unknown_text(title):
        title = ""
    if _is_unknown_text(author):
        author = "Unknown"
    if _is_unknown_text(raw_text):
        raw_text = ""
    if _is_unknown_text(summary):
        summary = ""

    if not title:
        title = _derive_title_from_image_content(raw_text, summary) or "未命名图片内容记录"

    if raw_text and summary and raw_text not in summary:
        detailed_summary = f"{summary}\n\n【识别的原始文本】:\n{raw_text}"
    elif summary:
        detailed_summary = summary
    elif raw_text:
        detailed_summary = f"图片中可识别的主要文字内容如下：\n{raw_text}"
    else:
        detailed_summary = "当前图片已完成解析，但未稳定识别出足够清晰的文字内容；建议更换更清晰的图片后再次解析。"

    keyword_terms = _unique_texts(
        list(keywords) + _extract_image_terms(title, author, raw_text, summary, max_items=12),
        max_items=12
    )
    entity_terms = _unique_texts(
        list(entities) + [title, author] + _extract_image_terms(title, raw_text, max_items=12),
        max_items=12
    )

    return {
        "title": title,
        "author": author or "Unknown",
        "raw_text": raw_text,
        "summary": summary,
        "is_rejected": is_rejected,
        "reject_reason": reject_reason,
        "detailed_summary": detailed_summary,
        "keywords": keyword_terms,
        "entities": entity_terms,
        "keyword_text": ",".join(keyword_terms),
        "entity_text": ",".join(entity_terms),
    }


def _build_image_messages(image_path: str, ocr_retry: bool = False):
    """构建图片解析消息，包含隔离指令和提取要求。"""
    isolation_note = (
        "请仅基于这张图片本身进行分析，不要参考任何数据库、以前的对话、系统中的其他记录、"
        "文件名、路径、目录名或外部常识。"
    )
    if ocr_retry:
        instruction = (
            "请执行严格的图片 OCR 与版面理解。"
            "如果这是一页文档、论文、书页、课件、截图、表格或扫描件，不要误判为空白图。"
            "务必优先识别正文、标题、小标题、表头、注释、编号、页内可见文字，并尽量逐字提取到 raw_text；"
            "title 只能来自图片中明确标题或基于图片可见内容做最小概括；"
            "summary 必须忠实概括图片里的实际文字内容、版面结构和主题，不能联想扩写。"
            "如果图片为空白图、纯色图、严重乱码图、严重模糊图、噪声图、损坏截图或没有稳定有效内容，"
            "必须返回 is_rejected=true 和 reject_reason，禁止为了入库而编造内容。"
            "请返回严格 JSON，字段必须包含 title、author、raw_text、summary、keywords、entities、is_rejected、reject_reason。"
        )
    else:
        instruction = (
            "请解析这张图片的真实内容，只输出严格 JSON。"
            "字段包含 title、author、raw_text、summary、keywords、entities、is_rejected、reject_reason。"
            "如果图片以文字内容为主，请优先做 OCR，raw_text 尽量完整保留主要正文；"
            "summary 必须详细描述图片中的可见结构、文字、关系、流程、图表、模块或公式；"
            "如果图片为空白图、纯色图、严重乱码图、严重模糊图、噪声图、损坏截图或没有稳定有效内容，"
            "必须返回 is_rejected=true 并说明 reject_reason；"
            "看不清或不存在的字段请写 Unknown 或空字符串，禁止编造。"
        )

    return [
        {
            "role": "system",
            "content": isolation_note + " 严格只基于图片中可见内容提取字段，禁止任何外部知识补充。"
        },
        {
            "role": "user",
            "content": [
                # 这里故意传本地路径给 qwen-agent，由它在底层自动转成 base64 后再发给模型。
                # 实测当前环境下手工 data:image;base64 方式会导致纯文字页 OCR 结果为空。
                {"image": image_path},
                {"text": instruction}
            ]
        }
    ]


def _build_image_ocr_messages(image_path: str):
    """构建严格 OCR 模式的消息，只允许提取图片中真实可见的文字。"""
    return [
        {
            "role": "system",
            "content": (
                "你现在处于严格 OCR 模式。"
                "只允许提取图片中真实可见的文字、标题、表头、编号、页眉页脚和简要版面属性。"
                "不要描述不存在的插图、流程、图表、作者、结论或背景知识。"
            )
        },
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {
                    "text": (
                        "请先判断这张图片是否为以文字为主的页面、截图、扫描件、书页、论文页、表格或混合内容页面。"
                        "如果存在文字，不论多少，都要优先做 OCR，不要因为排版简单、纯白背景或纯文字内容就返回空白。"
                        "如果图片接近空白、纯色、严重模糊、严重噪声、文字严重乱码，或根本没有稳定有效内容，"
                        "请明确拒绝入库，不要编造文字。"
                        "请尽可能逐字提取图片中真实可见的文字到 raw_text，保留换行、编号、标题和主要段落顺序。"
                        "并识别标题或页内主题到 title。"
                        "若图中没有明确作者，author 写 Unknown。"
                        "summary 只能基于 raw_text 做简短概括，不要补充任何图片外信息。"
                        "输出严格 JSON，格式为："
                        "{\"title\":\"\",\"author\":\"Unknown\",\"raw_text\":\"\",\"summary\":\"\",\"page_type\":\"\",\"is_text_dense\":true,\"is_rejected\":false,\"reject_reason\":\"\"}"
                    )
                }
            ]
        }
    ]


def _has_meaningful_raw_text(raw_text: str) -> bool:
    """判断 OCR 结果是否包含足够的有意义文字。"""
    text = re.sub(r"\s+", "", str(raw_text or ""))
    if len(text) < 12:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]{12,}", text))


def _parse_image_ocr(image_path: str) -> dict:
    """调用 OCR 模型提取图片文字，返回解析后的 JSON。"""
    chunks = list(get_vl_ocr_parser().run(messages=_build_image_ocr_messages(image_path)))
    parsed = _parse_model_json(chunks[-1][-1].get("content", "")) if chunks else {}
    if not isinstance(parsed, dict):
        return {}
    raw_text = str(parsed.get("raw_text", "") or "").strip()
    if not raw_text:
        # 少数情况下模型会把识别文字塞进 summary，这里兜底回填，避免后续误判为空白页。
        summary = str(parsed.get("summary", "") or "").strip()
        if len(re.sub(r"\s+", "", summary)) >= 12:
            parsed["raw_text"] = summary
    return parsed


def _summarize_ocr_text(raw_text: str) -> dict:
    """将 OCR 提取的原始文本交由文本模型整理为结构化 JSON。"""
    messages = [{
        "role": "user",
        "content": (
            "下面内容是从图片中OCR提取出的原始文本。"
            "请严格只基于这些文本整理 JSON，包含 title、author、summary、keywords、entities。"
            "不要补充图片外信息，不要猜作者，不要编造图表或结构。"
            "如果没有作者就返回 Unknown。只输出严格 JSON。\n\n"
            f"OCR原文：\n{raw_text[:8000]}"
        )
    }]
    chunks = list(get_text_parser().run(messages=messages))
    parsed = _parse_model_json(chunks[-1][-1].get("content", "")) if chunks else {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _image_parse_has_meaningful_content(parsed_data: dict) -> bool:
    """判断图片解析结果是否包含有效内容（title/raw_text/summary 至少一个）。"""
    if not isinstance(parsed_data, dict):
        return False
    for field in ("title", "raw_text", "summary"):
        value = str(parsed_data.get(field, "") or "").strip()
        if value and not _is_unknown_text(value) and len(value) >= 4:
            return True
    return False


def _parse_image_content(image_path: str) -> dict:
    """主图片解析流程：OCR → 文本整理 → 结构化生成，包含兜底重试逻辑。"""
    ocr_data = _parse_image_ocr(image_path)
    ocr_raw_text = str(ocr_data.get("raw_text", "") or "").strip()

    if _has_meaningful_raw_text(ocr_raw_text):
        text_summary_data = _summarize_ocr_text(ocr_raw_text)
        merged_data = dict(text_summary_data)
        merged_data["raw_text"] = ocr_raw_text

        if not str(merged_data.get("title", "") or "").strip():
            merged_data["title"] = ocr_data.get("title", "")
        if not str(merged_data.get("author", "") or "").strip():
            merged_data["author"] = ocr_data.get("author", "Unknown")
        if not str(merged_data.get("summary", "") or "").strip():
            merged_data["summary"] = ocr_data.get("summary", "")

        return _normalize_image_parse_result(merged_data)

    last_parsed = {}
    for ocr_retry in (False, True):
        messages = _build_image_messages(image_path, ocr_retry=ocr_retry)
        chunks = list(get_vl_parser().run(messages=messages))
        parsed = _parse_model_json(chunks[-1][-1].get("content", "")) if chunks else {}
        if parsed:
            last_parsed = parsed
        if _image_parse_has_meaningful_content(parsed):
            if ocr_raw_text and not parsed.get("raw_text"):
                parsed["raw_text"] = ocr_raw_text
            return _normalize_image_parse_result(parsed)

    if ocr_raw_text and not last_parsed.get("raw_text"):
        last_parsed["raw_text"] = ocr_raw_text
    if ocr_data.get("title") and not last_parsed.get("title"):
        last_parsed["title"] = ocr_data.get("title")
    if ocr_data.get("summary") and not last_parsed.get("summary"):
        last_parsed["summary"] = ocr_data.get("summary")
    return _normalize_image_parse_result(last_parsed)


def _build_image_record(parsed_data: dict, image_path: str) -> dict:
    """将图片解析结果组装为 knowledge_records 表的入库格式。"""
    normalized = _normalize_image_parse_result(parsed_data)
    return {
        "data_type": "image_doc",
        "title": normalized["title"],
        "category": "图片解析",
        "tags": "image,vision,ocr",
        "summary": normalized["detailed_summary"],
        "raw_text": normalized["raw_text"],
        "structured_json": normalized,
        "author": normalized["author"],
        "source": "",
        "created_date": "",
        "event_date": "",
        "keyword_text": normalized["keyword_text"],
        "entity_text": normalized["entity_text"],
        "source_type": "image",
        "source_path": image_path,
        "status": "normal",
    }


def _find_related_records_for_image(parsed_data: dict, keyword: str = "", limit: int = 5):
    """根据图片解析结果生成关键词，检索数据库中相关记录。"""
    normalized = _normalize_image_parse_result(parsed_data)
    compare_keywords = _unique_texts(
        [keyword, normalized["title"], normalized["author"]]
        + normalized.get("keywords", [])
        + _extract_image_terms(normalized["raw_text"], normalized["summary"], max_items=10),
        max_items=10
    )

    related_records = []
    existing_ids = set()
    for compare_keyword in compare_keywords:
        for record in storage.query_record({"keyword": compare_keyword})[:limit]:
            record_id = record.get("id")
            if record_id in existing_ids:
                continue
            existing_ids.add(record_id)
            related_records.append(record)
            if len(related_records) >= limit:
                return related_records, compare_keywords
    return related_records, compare_keywords


def _walk_supported_files(folder: str) -> List[str]:
    """递归扫描文件夹，收集所有支持的图片和文本文件路径。"""
    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    txt_exts = ('.txt', '.md')
    valid_exts = img_exts + txt_exts
    collected = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            if path.lower().endswith(valid_exts):
                collected.append(path)
    return collected


def _short_preview(text: str, limit: int = 180) -> str:
    """生成文本的简短预览，用于界面展示。"""
    preview = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def _build_batch_text_record(file_path: str) -> dict:
    """读取文本文件内容，调用 AI 提取结构化知识，组装为入库记录。"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    messages = [{
        'role': 'user',
        'content': (
            "请提取该文本的核心内容并返回严格 JSON，至少包含 title、author、summary、keywords、entities。\n"
            f"原文：\n{content[:4000]}"
        )
    }]
    chunks = list(get_text_parser().run(messages=messages))
    parsed_data = _parse_model_json(chunks[-1][-1]['content']) if chunks else {}
    if not isinstance(parsed_data, dict):
        parsed_data = {}

    title = str(parsed_data.get('title', '') or '').strip() or f"批处理文_{os.path.basename(file_path)}"
    author = str(parsed_data.get('author', '') or '').strip() or 'Unknown'
    summary = str(parsed_data.get('summary', '') or '').strip() or _short_preview(content, limit=500)

    keywords = parsed_data.get('keywords', [])
    entities = parsed_data.get('entities', [])
    if not isinstance(keywords, list):
        keywords = _extract_image_terms(keywords, summary, title, max_items=12)
    if not isinstance(entities, list):
        entities = _extract_image_terms(entities, title, content[:2000], max_items=12)

    keyword_terms = _unique_texts(list(keywords) + _extract_image_terms(title, summary, max_items=12), max_items=12)
    entity_terms = _unique_texts(list(entities) + _extract_image_terms(title, content[:2000], max_items=12), max_items=12)

    return {
        'data_type': 'text_doc',
        'title': title,
        'category': '批量文本导入',
        'tags': 'text,batch',
        'summary': summary,
        'raw_text': content,
        'structured_json': parsed_data,
        'author': author,
        'source': '',
        'created_date': '',
        'event_date': '',
        'keyword_text': ",".join(keyword_terms),
        'entity_text': ",".join(entity_terms),
        'source_type': 'text',
        'source_path': file_path,
        'status': 'normal',
    }


def _same_source_path(path1: str, path2: str) -> bool:
    """判断两个路径是否为同一个文件（规范化后比对）。"""
    left = os.path.abspath(str(path1 or "")).replace("\\", "/").lower()
    right = os.path.abspath(str(path2 or "")).replace("\\", "/").lower()
    return bool(left and right and left == right)


class ImageRejectError(ValueError):
    pass


def _boolish(value) -> bool:
    """将多种常见真值/假值字符串统一转换为 bool。"""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "是"}


def _image_basic_quality_check(image_path: str):
    """快速检测图片质量：判断是否为空白图、纯色图。"""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0]) if stat.mean else 0.0
            stddev = float(stat.stddev[0]) if stat.stddev else 0.0
            low, high = gray.getextrema()

        # 纯白、纯黑、近似纯色图直接拒绝。
        if high - low <= 2 and stddev <= 1.2:
            return False, "图片几乎为空白或纯色，没有可入库的有效信息。"
        if stddev <= 1.5 and (mean >= 252 or mean <= 3):
            return False, "图片几乎为空白或纯色，没有可入库的有效信息。"
    except Exception:
        pass
    return True, ""


def _looks_like_garbled_text(text: str) -> bool:
    """判断 OCR 识别的文字是否为严重乱码。"""
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) < 12:
        return False

    valid_chars = len(re.findall(r"[一-鿿A-Za-z0-9]", compact))
    weird_chars = len(re.findall(r"[^一-鿿A-Za-z0-9，。！？；：""''、,.!?:;()（）【】\[\]\-_/ ]", str(text or "")))
    valid_ratio = valid_chars / max(len(compact), 1)
    weird_ratio = weird_chars / max(len(str(text or "")), 1)

    repeated_noise = re.search(r"(.)\1{7,}", compact)
    return bool((valid_ratio < 0.45 and weird_ratio > 0.2) or repeated_noise)


def _image_reject_reason_from_model(parsed_data: dict) -> str:
    """从模型返回的摘要中提取拒绝入库的原因。"""
    reject_reason = str(parsed_data.get("reject_reason", "") or "").strip()
    if reject_reason:
        return reject_reason

    summary = str(parsed_data.get("summary", "") or "").strip()
    lower_summary = summary.lower()
    rejection_tokens = [
        "空白", "纯色", "乱码", "噪声", "模糊", "损坏", "无有效内容",
        "blank", "garbled", "noisy", "blur", "blurry", "corrupt", "invalid"
    ]
    if any(token in summary or token in lower_summary for token in rejection_tokens):
        return summary
    return ""


def _validate_image_for_ingest(image_path: str, parsed_data: dict):
    """综合图片质量和解析结果，决定是否允许入库。"""
    quality_ok, quality_reason = _image_basic_quality_check(image_path)
    if not quality_ok:
        raise ImageRejectError(f"拒绝入库：{quality_reason}")

    if _boolish(parsed_data.get("is_rejected")):
        reason = _image_reject_reason_from_model(parsed_data) or "图片为空白、乱码或缺少稳定有效内容。"
        raise ImageRejectError(f"拒绝入库：{reason}")

    normalized = _normalize_image_parse_result(parsed_data)
    raw_text = normalized.get("raw_text", "")
    summary = normalized.get("summary", "")

    if not _has_meaningful_raw_text(raw_text):
        if not summary or "未稳定识别" in normalized.get("detailed_summary", ""):
            raise ImageRejectError("拒绝入库：图片没有识别出足够稳定的有效内容，可能是空白图、严重模糊图或乱码图。")

    if raw_text and _looks_like_garbled_text(raw_text):
        raise ImageRejectError("拒绝入库：图片识别结果疑似严重乱码，当前内容不适合入库。")


def _precheck_image_before_model(image_path: str):
    """模型解析前的图片质量预检，不合格直接拦截。"""
    quality_ok, quality_reason = _image_basic_quality_check(image_path)
    if not quality_ok:
        raise ImageRejectError(f"拒绝入库：{quality_reason}")

# ===== 1. 单图提取入库：预检 → OCR/视觉解析 → 质量校验 → 入库 → 向量索引 =====
@register_tool('ParseAndInsertImage')
class ParseAndInsertImage(BaseTool):
    description = '接收单张图片路径，调用视觉大模型提取文献结构化信息并入库。'
    parameters = [{'name': 'image_path', 'type': 'string', 'description': '单张图片的绝对路径', 'required': True}]
    def call(self, params: str, **kwargs) -> str:
        try:
            path = _resolve_pending_path('image_path', json.loads(params).get('image_path'))
            path = os.path.abspath(path).replace("\\", "/") if path else "" # 规范化保证跨平台及查重有效
            if not os.path.exists(path): return "失败：找不到图片"
            
            ui_queue = kwargs.get('ui_queue', None)
            if ui_queue:
                ui_queue.put({'type': 'preview_file', 'data': path})

                
            # 提前查重，避免浪费大模型算力（一定要凑齐 title 和 source_path 避免抛错）
            if storage.paper_repo.exists_record(title="", source_path=path):
                return "该图片已在数据库中存在，已自动拦截重复插入。"

            _precheck_image_before_model(path)
            parsed_data = _parse_image_content(path)
            _validate_image_for_ingest(path, parsed_data)
            record = _build_image_record(parsed_data, path)
            try:
                new_id = storage.insert_record(record)
                try:
                    vector_chunk_count = vector_index.add_record(new_id, record)
                except Exception:
                    vector_chunk_count = 0
                # 将解析得到的核心信息返回给大脑以供查阅汇报
                return (
                    f"✅ 单图解析且入库成功！\n"
                    f"入库ID: {new_id}\n"
                    f"标题: {record['title']}\n"
                    f"作者: {record['author']}\n"
                    f"关键词: {record['keyword_text'] or '未稳定提取'}\n"
                    f"识别原文: {record['raw_text'] or '未稳定提取出清晰文字'}\n"
                    f"摘要总结: {record['summary']}\n"
                    f"向量索引片段数: {vector_chunk_count}"
                )
            except ValueError as ve: return str(ve)
        except Exception as e: return f"异常: {str(e)}"

# ===== 1.5 辅助工具: 文件名清理 =====
import traceback

# ===== 2. 文件夹批量混合入库：逐文件处理，带进度报告与超时保护 =====
@register_tool('BatchImportFiles')
class BatchImportFiles(BaseTool):
    description = '批量扫描指定文件夹下的所有图片及文本文件，分别由大模型解析提取特征后入库。'
    parameters = [{'name': 'folder_path', 'type': 'string', 'description': '文件夹的绝对路径', 'required': True}]
    
    def _process_single_file(self, fp: str) -> dict:
        """处理单个文件：查重 → 图片/文本分支解析 → 入库 → 同步向量索引。"""
        fp = os.path.abspath(fp).replace("\\", "/")
        result = {'status': 'failed', 'record': None, 'file_path': fp, 'error': ''}

        try:
            if storage.paper_repo.exists_record(title="", source_path=fp):
                result['status'] = 'skipped'
                result['error'] = '文件已在数据库中存在，已跳过重复入库'
                return result

            if fp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                _precheck_image_before_model(fp)
                parsed_data = _parse_image_content(fp)
                _validate_image_for_ingest(fp, parsed_data)
                record = _build_image_record(parsed_data, fp)
            else:
                record = _build_batch_text_record(fp)

            new_id = storage.insert_record(record)
            record['id'] = new_id

            try:
                vector_index.add_record(new_id, record)
            except Exception:
                pass

            result['status'] = 'success'
            result['record'] = record
            return result
        except ValueError as e:
            message = str(e)
            if '已存在' in message or '重复' in message:
                result['status'] = 'skipped'
            elif '拒绝入库' in message:
                result['status'] = 'rejected'
            result['error'] = message
            return result
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def call(self, params: str, **kwargs) -> str:
        """批量导入入口：扫描文件夹 → 逐文件处理 → 汇总成功/失败统计。"""
        try:
            folder = _resolve_pending_path('folder_path', json.loads(params).get('folder_path'))
            if not os.path.isdir(folder): return "失败：找不到目录"
            
            # 获取 UI 队列以发送进度（如果存在）
            ui_queue = kwargs.get('ui_queue', None)
            
            valid_files = _walk_supported_files(folder)
            if not valid_files: return "文件夹中未发现图文文件。"
            
            # 限制处理数量（防止太大的文件夹）
            MAX_FILES = 50
            if len(valid_files) > MAX_FILES:
                valid_files = valid_files[:MAX_FILES]
                msg = f"⚠️ 检测到超过 {MAX_FILES} 个文件，将仅处理前 {MAX_FILES} 个以防卡顿。"
                if ui_queue:
                    ui_queue.put({'type': 'status', 'data': msg})
            
            success_count = 0
            skipped_count = 0
            rejected_count = 0
            error_count = 0
            success_examples = []
            skipped_examples = []
            rejected_examples = []
            error_examples = []
            
            for idx, fp in enumerate(valid_files):
                # 发送进度更新
                progress = f"正在处理: ({idx+1}/{len(valid_files)}) {os.path.basename(fp)}"
                if ui_queue:
                    ui_queue.put({'type': 'status', 'data': progress})
                    ui_queue.put({'type': 'think_chunk', 'data': f"[{idx+1}/{len(valid_files)}] 正在解析入库: {os.path.basename(fp)}...\n"})
                    ui_queue.put({'type': 'preview_file', 'data': fp})
                
                result = self._process_single_file(fp)

                if result['status'] == 'success':
                    success_count += 1
                    record = result.get('record') or {}
                    success_examples.append(
                        f"{os.path.basename(fp)} ->《{record.get('title', '未命名')}》"
                    )
                elif result['status'] == 'skipped':
                    skipped_count += 1
                    skipped_examples.append(f"{os.path.basename(fp)}: {result.get('error', '已跳过')}")
                elif result['status'] == 'rejected':
                    rejected_count += 1
                    rejected_examples.append(f"{os.path.basename(fp)}: {result.get('error', '已拒绝入库')}")
                    if ui_queue and result.get('error'):
                        ui_queue.put({'type': 'stream_chunk', 'data': f"\n🚫 {os.path.basename(fp)}: {result.get('error', '已拒绝入库')}"})
                else:
                    error_count += 1
                    error_examples.append(f"{os.path.basename(fp)}: {result.get('error', '未知错误')}")
                    if ui_queue and result.get('error'):
                        ui_queue.put({'type': 'stream_chunk', 'data': f"\n⚠️ {os.path.basename(fp)}: {result.get('error', '未知错误')}"})

            summary_lines = [
                f"✅ 批量导入完成！共扫描 {len(valid_files)} 份数据。",
                f"成功入库 {success_count} 份，重复跳过 {skipped_count} 份，拒绝入库 {rejected_count} 份，失败 {error_count} 份。"
            ]
            if success_examples:
                summary_lines.append("成功样例：" + "；".join(success_examples[:5]))
            if skipped_examples:
                summary_lines.append("跳过样例：" + "；".join(skipped_examples[:5]))
            if rejected_examples:
                summary_lines.append("拒绝入库样例：" + "；".join(rejected_examples[:5]))
            if error_examples:
                summary_lines.append("失败原因样例：" + "；".join(error_examples[:5]))

            if ui_queue:
                ui_queue.put({'type': 'status', 'data': "当前 AI 状态: 批量导入完成"})
            
            return "\n".join(summary_lines)
        except Exception as e: 
            return f"异常: {str(e)}"


# ===== 2.1 删除匹配关键词的多条记录 =====
@register_tool('DeleteRecordsByKeyword')
class DeleteRecordsByKeyword(BaseTool):
    description = '按关键词删除所有匹配的记录，关键词会匹配标题和作者。适合批量清理重复主题记录。'
    parameters = [{'name': 'keyword', 'type': 'string', 'description': '要删除的关键词，例如 GTN', 'required': True}]
    def call(self, params: str, **kwargs) -> str:
        """按关键词检索并删除所有匹配记录。"""
        try:
            keyword = json.loads(params).get('keyword', '').strip()
            if not keyword:
                return '失败：关键词不能为空。'

            matched = storage.query_record({'keyword': keyword})
            if not matched:
                return f'未找到包含关键词 {keyword} 的记录。'

            deleted_ids = []
            for record in matched:
                if storage.delete_record(record['id']):
                    deleted_ids.append(record['id'])

            return f"已删除 {len(deleted_ids)} 条匹配 {keyword} 的记录，删除的记录 ID 为: {', '.join(map(str, deleted_ids))}。"
        except Exception as e:
            return f"删除异常: {str(e)}"

# ===== 3. 长文本解析入库：AI 提取结构化知识单元 → 组装 → 入库 → 向量索引 =====
@register_tool('ParseAndInsertTextFile')
class ParseAndInsertTextFile(BaseTool):
    description = '读取本地的 .txt 或 .md 文本文件，整理为通用知识记录后完整入库。'
    parameters = [{'name': 'file_path', 'type': 'string', 'description': '文本文件的绝对路径', 'required': True}]

    def call(self, params: str, **kwargs) -> str:
        """文本文件解析入库：读取内容 → AI 提取知识单元 → 组装记录 → 入库 → 向量索引。"""
        try:
            payload = json.loads(params)
            path = _resolve_pending_path('file_path', payload.get('file_path'))
            path = os.path.abspath(path).replace("\\", "/") if path else ""

            if not os.path.exists(path):
                return "失败：找不到文件"

            ui_queue = kwargs.get('ui_queue', None)
            if ui_queue:
                ui_queue.put({'type': 'preview_file', 'data': path})
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': '\n[📄 文本解析] 正在读取文本文件并构建通用知识结构...\n'
                })

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            messages = [{
                'role': 'user',
                'content': f"""
    请把下面内容整理成“通用知识记录 JSON”。

    重要说明：
    1. 不要只提取一个主标题或一个主对象。
    2. 文档中可能同时包含多种知识，例如论文、实验、代码模块、课程内容、需求、流程、表格、会议纪要、结论、问题与解决方案等。
    3. raw_text 不需要返回，原文系统会单独完整保存。
    4. summary 是文档级总体摘要。
    5. knowledge_units 是最重要的字段，用于保存文档内部的多个细粒度知识单元。
    6. knowledge_units 不局限于论文，必须根据原文内容自动识别知识单元。
    7. 每个 knowledge_unit 都要尽量包含 attributes、relations 和 evidence。
    8. 如果文档中有多个论文、多个实验、多个模块、多个流程或多个结论，必须分别放入 knowledge_units。
    9. evidence 必须来自原文片段，不能凭空编造。
    10. 只输出严格 JSON，不要解释，不要输出 Markdown 代码块。

    knowledge_units 的通用格式：
    {{
      "unit_type": "paper / concept / experiment / code_module / process / requirement / meeting_note / table / conclusion / problem_solution / person / organization / other",
      "name": "知识单元名称",
      "attributes": {{
        "任意属性名": "任意属性值"
      }},
      "relations": [
        {{
          "type": "关系类型",
          "target": "关联对象"
        }}
      ],
      "evidence": "原文证据片段"
    }}

    输出格式：
    {{
      "data_type": "text",
      "document_type": "mixed_knowledge",
      "title": "文档总体标题",
      "category": "分类",
      "tags": ["标签1", "标签2"],
      "author": "作者或 Unknown",
      "source": "来源或 Unknown",
      "created_date": "",
      "event_date": "",
      "keywords": ["关键词1", "关键词2"],
      "entities": ["实体1", "实体2"],
      "summary": "文档总体详细摘要",
      "sections": [],
      "knowledge_units": [],
      "items": [],
      "tables": [],
      "important_facts": []
    }}

    原文：
    {content[:8000]}
    """
            }]

            chunks = list(get_text_parser().run(messages=messages))
            if not chunks:
                parsed_data = {}
            else:
                response_content = chunks[-1][-1].get('content', '')
                parsed_data = _parse_model_json(response_content)

            if not isinstance(parsed_data, dict):
                parsed_data = {}

            summary = parsed_data.get("summary", "")
            if not summary:
                summary = content[:500]

            keywords = parsed_data.get("keywords", [])
            entities = parsed_data.get("entities", [])
            tags = parsed_data.get("tags", [])

            if isinstance(keywords, list):
                keyword_text = ",".join(map(str, keywords))
            else:
                keyword_text = str(keywords)

            if isinstance(entities, list):
                entity_text = ",".join(map(str, entities))
            else:
                entity_text = str(entities)

            if isinstance(tags, list):
                tags_text = ",".join(map(str, tags))
            else:
                tags_text = str(tags)

            knowledge_units = parsed_data.get("knowledge_units", [])
            if isinstance(knowledge_units, list):
                for unit in knowledge_units:
                    if not isinstance(unit, dict):
                        continue

                    unit_name = str(unit.get("name", ""))
                    unit_type = str(unit.get("unit_type", ""))
                    evidence = str(unit.get("evidence", ""))

                    if unit_name and unit_name not in entity_text:
                        entity_text = f"{entity_text},{unit_name}" if entity_text else unit_name

                    if unit_type and unit_type not in keyword_text:
                        keyword_text = f"{keyword_text},{unit_type}" if keyword_text else unit_type

                    attributes = unit.get("attributes", {})
                    if isinstance(attributes, dict):
                        for value in attributes.values():
                            if isinstance(value, list):
                                for item in value:
                                    item = str(item)
                                    if item and item not in entity_text:
                                        entity_text = f"{entity_text},{item}" if entity_text else item
                            else:
                                value = str(value)
                                if value and len(value) <= 80 and value not in entity_text:
                                    entity_text = f"{entity_text},{value}" if entity_text else value

                    if evidence and len(evidence) <= 120 and evidence not in keyword_text:
                        keyword_text = f"{keyword_text},{evidence}" if keyword_text else evidence

            record = {
                "data_type": parsed_data.get("data_type", "text"),
                "title": parsed_data.get("title", f"文本记录_{os.path.basename(path)}"),
                "category": parsed_data.get("category", "通用知识"),
                "tags": tags_text,
                "summary": summary,
                "raw_text": content,
                "structured_json": parsed_data,

                "author": parsed_data.get("author", "Unknown"),
                "source": parsed_data.get("source", ""),
                "created_date": parsed_data.get("created_date", ""),
                "event_date": parsed_data.get("event_date", ""),

                "keyword_text": keyword_text,
                "entity_text": entity_text,

                "source_type": "text",
                "source_path": path,
                "status": "normal"
            }

            try:
                new_id = storage.insert_record(record)

                try:
                    chunk_count = vector_index.add_record(new_id, record)
                except Exception as e:
                    chunk_count = 0
                    if ui_queue:
                        ui_queue.put({
                            'type': 'think_chunk',
                            'data': f'[⚠️ 向量索引] 同步失败：{str(e)}\n'
                        })

                unit_count = 0
                if isinstance(parsed_data.get("knowledge_units", []), list):
                    unit_count = len(parsed_data.get("knowledge_units", []))

                if ui_queue:
                    ui_queue.put({
                        'type': 'think_chunk',
                        'data': f'[📄 文本解析] 已生成通用知识单元 {unit_count} 个，"向量索引片段 {chunk_count}个\n"，并完成入库。\n'
                    })

                return (
                    f"✅ 文本知识入库成功！\n"
                    f"入库ID: {new_id}\n"
                    f"标题: {record['title']}\n"
                    f"分类: {record['category']}\n"
                    f"知识单元数量: {unit_count}\n"
                    f"关键词: {record['keyword_text']}\n"
                    f"实体: {record['entity_text']}\n"
                    f"摘要: {record['summary'][:200]}..."
                )
            except ValueError as ve:
                return str(ve)

        except Exception as e:
            return f"异常: {str(e)}"


# ===== 3.1 图片对照已有记录 =====
@register_tool('CompareImageWithRecords')
class CompareImageWithRecords(BaseTool):
    description = '解析图片并与数据库中相关记录进行自然语言对照汇报。'
    parameters = [
        {'name': 'image_path', 'type': 'string', 'description': '图片绝对路径', 'required': True},
        {'name': 'keyword', 'type': 'string', 'description': '可选的对照关键词，比如 GTN、GCN、Graph Transformer', 'required': False}
    ]
    def call(self, params: str, **kwargs) -> str:
        """图片对照：解析图片 → 关键词检索 → 匹配同源记录 → 返回自然语言对比结果。"""
        try:
            payload = json.loads(params)
            image_path = _resolve_pending_path('image_path', payload.get('image_path'))
            image_path = os.path.abspath(image_path).replace("\\", "/") if image_path else "" # 规范化保证跨平台及查重有效
            keyword = (payload.get('keyword') or '').strip()
            if not image_path or not os.path.exists(image_path):
                return '失败：找不到图片'
                
            ui_queue = kwargs.get('ui_queue', None)
            if ui_queue:
                ui_queue.put({'type': 'preview_file', 'data': image_path})

            _precheck_image_before_model(image_path)
            parsed_data = _parse_image_content(image_path)
            try:
                _validate_image_for_ingest(image_path, parsed_data)
            except ImageRejectError as e:
                return str(e)
            related_records, compare_keywords = _find_related_records_for_image(parsed_data, keyword=keyword, limit=5)

            # 强行查找是否本机已有同路径缓存（精确匹配双向兼容反斜杠），确保百分百能关联到已入库记录
            path_unix = image_path.replace("\\", "/")
            path_win = image_path.replace("/", "\\")
            exact_records = storage.query_record({'source_path': path_unix})
            if not exact_records:
                exact_records = storage.query_record({'source_path': path_win})
                
            if exact_records:
                existing_ids = {r['id'] for r in related_records}
                for er in exact_records:
                    if er['id'] not in existing_ids:
                        related_records.insert(0, er)

            related_lines = []
            for record in related_records:
                exact_tag = "同源精确匹配" if _same_source_path(record.get('source_path', ''), image_path) else "相关记录"
                summary_preview = _short_preview(record.get('summary', ''), limit=260)
                raw_preview = _short_preview(record.get('raw_text', ''), limit=180)
                if raw_preview and raw_preview not in summary_preview:
                    evidence_text = f"摘要：{summary_preview}\n原文片段：{raw_preview}"
                else:
                    evidence_text = f"摘要：{summary_preview}"
                related_lines.append(
                    f"【{exact_tag}】记录ID {record.get('id')}《{record.get('title', '未命名')}》"
                    f"，作者 {record.get('author', 'Unknown')}，类型 {record.get('data_type', 'Unknown')}\n"
                    f"{evidence_text}"
                )

            base_summary = parsed_data.get('detailed_summary') or parsed_data.get('summary') or '模型未给出完整摘要，但已完成图像解析。'
            title = parsed_data.get('title') or '未命名图片内容记录'
            author = parsed_data.get('author') or 'Unknown'
            raw_text = parsed_data.get('raw_text') or ''

            result = [
                "【图片解析结果】",
                f"这张图的核心内容可以概括为：{base_summary}",
                f"从结构化信息看，它的标题更接近“{title}”，作者信息为“{author}”。"
            ]
            if raw_text:
                result.append(f"图片中识别到的关键文字内容为：{_short_preview(raw_text, limit=600)}")
            else:
                result.append('当前图片未稳定提取出清晰文字，已尽量依据可见结构完成内容解析。')
            if compare_keywords:
                result.append("\n【数据库检索依据】")
                result.append('本次对照优先使用以下内容关键词检索数据库：' + '、'.join(compare_keywords[:6]))
            if related_lines:
                result.append("\n【数据库相关内容】")
                result.append(f"共找到 {len(related_records)} 条相关记录。")
                result.append('\n\n'.join(related_lines))
            else:
                result.append("\n【数据库相关内容】")
                result.append('数据库中暂未检索到足够明确的对照记录。')
            return '\n'.join(result)
        except Exception as e:
            return f'对照异常: {str(e)}'


# ===== 4. 记录管理：更新、查询、删除 =====
@register_tool('UpdatePaperRecord')
class UpdatePaperRecord(BaseTool):
    description = '修改/更新数据库中指定文献的各种字段数据（标题、作者、年份、摘要等）。允许用户对旧库修改。'
    parameters = [
        {'name': 'id', 'type': 'integer', 'description': '文献记录的数据库ID', 'required': True},
        {'name': 'update_fields', 'type': 'object', 'description': '这是一个JSON对象，里面是你希望修改的键对（例如 {"author": "李四", "summary": "新摘要"}）', 'required': True}
    ]
    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params)
            ok = storage.update_record(p['id'], p['update_fields'])
            return "✅ 数据更新成功。" if ok else "⚠️ 未找到对应ID的记录或更新失败。"
        except Exception as e: return f"更新异常: {str(e)}"

# ===== 5. 知识检索工具 =====
@register_tool('QueryByAuthorOrTitle')
class QueryByAuthorOrTitle(BaseTool):
    description = '旧版精确查询工具，仅用于按 title 或 author 字段查询；通用知识检索请优先使用 SearchKnowledgeRecords。'
    parameters = [{'name': 'keyword', 'type': 'string', 'description': '关键词', 'required': True}]
    def call(self, params: str, **kwargs) -> str:
        try:
            kw = json.loads(params).get('keyword', '')
            merged = {r['id']: r for r in storage.query_record({"title": kw}) + storage.query_record({"author": kw})}.values()
            if not merged: return "未找到匹配记录。"
            return json.dumps([{"id": r["id"], "title": r["title"], "author": r["author"]} for r in merged], ensure_ascii=False)
        except Exception as e: return str(e)

@register_tool('CountAndGroupStatistics')
class CountAndGroupStatistics(BaseTool):
    description = '统计按照年份或作者分组的数据。'
    parameters = [{'name': 'dimension', 'type': 'string', 'description': '维度(year或author)', 'required': True}]
    def call(self, params: str, **kwargs) -> str:
        try:
            dim = json.loads(params).get('dimension', 'year')
            key_dim = "created_date" if dim == "year" else "author"
            records = storage.list_records(limit=2000)
            stats = {}
            for r in records:
                val = r.get(key_dim, "Unknown")
                stats[val] = stats.get(val, 0) + 1
            return json.dumps([{dim: k, "count": v} for k, v in stats.items()], ensure_ascii=False)
        except Exception as e: return str(e)

@register_tool('ListAllRecords')
class ListAllRecords(BaseTool):
    description = '获取当前数据库中存在的所有记录概要（包含id、标题、作者）。用于让AI知道系统里有哪些数据。'
    parameters = []
    def call(self, params: str, **kwargs) -> str:
        try:
            records = storage.list_records(limit=100)
            if not records: return "数据库为空。"
            return json.dumps([{"id": r["id"], "title": r["title"], "author": r["author"]} for r in records], ensure_ascii=False)
        except Exception as e: return str(e)

@register_tool('DeletePaperRecord')
class DeletePaperRecord(BaseTool):
    description = '彻底删除一条数据。'
    parameters = [{'name': 'id', 'type': 'integer', 'description': '记录ID', 'required': True}]
    def call(self, p: str, **k) -> str:
        return "✅ 删除成功。" if storage.delete_record(json.loads(p)['id']) else "⚠️ 无效ID。"

@register_tool('SearchKnowledgeRecords')
class SearchKnowledgeRecords(BaseTool):
    description = '根据关键词搜索知识库记录'

    parameters = [
        {
            'name': 'keyword',
            'type': 'string',
            'description': '搜索关键词',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        """关键词检索：在 knowledge_records 表中进行多字段模糊匹配，返回最多 5 条记录。"""
        try:
            payload = json.loads(params)
            keyword = payload.get('keyword', '').strip()

            ui_queue = kwargs.get('ui_queue', None)

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'\n[🔍 知识库检索] 正在检索关键词：{keyword}\n'
                })

            records = storage.query_record({"keyword": keyword})

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'[🔍 知识库检索] 检索完成，命中 {len(records)} 条记录。\n'
                })

            return json.dumps(records[:5], ensure_ascii=False)

        except Exception as e:
            return f'知识检索异常: {str(e)}'


@register_tool('SemanticSearchKnowledge')
class SemanticSearchKnowledge(BaseTool):
    description = '使用向量相似度进行语义检索，适合查询同义表达、相近概念、模糊主题、技术含义相似的问题。'

    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '语义检索查询内容',
            'required': True
        },
        {
            'name': 'top_k',
            'type': 'integer',
            'description': '返回结果数量，默认5',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        """语义检索：在 ChromaDB 向量库中进行相似度搜索，返回最匹配的文本片段。"""
        try:
            payload = json.loads(params)
            query = payload.get('query', '').strip()
            top_k = int(payload.get('top_k', 5))

            ui_queue = kwargs.get('ui_queue', None)
            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'\n[🧠 向量检索] 正在进行语义检索：{query}\n'
                })

            hits = vector_index.search(query=query, top_k=top_k)

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'[🧠 向量检索] 检索完成，命中 {len(hits)} 个相关片段。\n'
                })

            if not hits:
                return "未检索到相关语义片段。"

            return json.dumps(hits, ensure_ascii=False)

        except Exception as e:
            return f"向量检索异常: {str(e)}"

# ===== 6. 知识库统计分析：语义检索 → 按维度统计 → 生成图表 =====
@register_tool('AnalyzeKnowledgeChart')
class AnalyzeKnowledgeChart(BaseTool):
    description = (
        '必须先按 query 对知识库做语义检索，再基于检索到的论文知识单元进行统计分析并生成图表。'
        '禁止直接全库统计。conference/source 维度按“每篇论文一个来源”统计。'
    )

    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '必须提供检索主题，例如 paper、ICDE论文、动态图计算、GraphRAG缓存管理。',
            'required': True
        },
        {
            'name': 'dimension',
            'type': 'string',
            'description': '统计维度，可选 conference/source/author/year/keywords/data_type/category/tags',
            'required': True
        },
        {
            'name': 'chart_type',
            'type': 'string',
            'description': '图表类型，可选 bar 或 pie，默认 bar',
            'required': False
        },
        {
            'name': 'top_k',
            'type': 'integer',
            'description': '语义检索返回片段数量，默认50',
            'required': False
        },
        {
            'name': 'top_n',
            'type': 'integer',
            'description': '图表展示前多少项，默认10',
            'required': False
        }
    ]

    def _extract_conferences(self, text: str):
        """从文本中提取学术会议/期刊名称，匹配预设的知名 venue 列表。"""
        text = text or ""
        known_venues = [
            "ICDE", "SIGMOD", "VLDB", "KDD", "WWW", "TKDE",
            "NeurIPS", "DASFAA", "FCS", "TACO", "ICML", "AAAI",
            "IJCAI", "CIKM", "SIGIR", "ACL", "EMNLP"
        ]

        results = []
        for venue in known_venues:
            pattern = rf'\b{venue}\s*(20\d{{2}})?\b'
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                year = match.group(1)
                results.append(f"{venue.upper()} {year}" if year else venue.upper())

        return results

    def _extract_years(self, text: str):
        """从文本中提取年份（20XX 格式）。"""
        return re.findall(r'\b20\d{2}\b', text or "")

    def _extract_authors_from_text(self, text: str):
        """从文本中提取作者名称，支持多种分隔符和格式。"""
        text = text or ""

        patterns = [
            r'"authors"\s*:\s*"([^"]+)"',
            r'"authors"\s*:\s*\[([^\]]+)\]',
            r'Authors:\s*([^\n]+)',
            r'作者[:：]\s*([^\n]+)'
        ]

        authors = []

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1)
                value = value.replace('"', '').replace("'", "")
                parts = re.split(r',|，|、| and ', value)
                for p in parts:
                    p = p.strip()
                    if p:
                        authors.append(p)

        return authors

    def _extract_keywords_from_text(self, text: str):
        """从文本中提取关键词，支持 JSON 格式和自然语言格式。"""
        text = text or ""

        patterns = [
            r'"keywords"\s*:\s*"([^"]+)"',
            r'"keywords"\s*:\s*\[([^\]]+)\]',
            r'Keywords:\s*([^\n]+)',
            r'关键词[:：]\s*([^\n]+)'
        ]

        keywords = []

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1)
                value = value.replace('"', '').replace("'", "")
                parts = re.split(r',|，|、|;', value)
                for p in parts:
                    p = p.strip()
                    if p:
                        keywords.append(p)

        return keywords

    def _make_chart(self, counter, dimension_name, chart_type, top_n, ui_queue):
        """根据统计数据生成柱状图或饼图，保存到 data/analysis_charts 目录。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from datetime import datetime

        if not counter:
            return None, []

        top_items = counter.most_common(top_n)
        labels = [str(item[0]) for item in top_items]
        values = [item[1] for item in top_items]

        chart_dir = os.path.join('data', 'analysis_charts')
        os.makedirs(chart_dir, exist_ok=True)

        filename = f"knowledge_{dimension_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = os.path.abspath(os.path.join(chart_dir, filename)).replace("\\", "/")

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(10, 6))
        title = f"知识库{dimension_name}分布统计"

        if chart_type == 'pie':
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title)
        else:
            plt.bar(labels, values)
            plt.title(title)
            plt.xlabel(dimension_name)
            plt.ylabel('论文数量')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        if ui_queue:
            ui_queue.put({
                'type': 'think_chunk',
                'data': f'\n[📊 知识分析] 已生成统计图：{chart_path}\n'
            })
            ui_queue.put({
                'type': 'preview_file',
                'data': chart_path
            })

        return chart_path, top_items

    def call(self, params: str, **kwargs) -> str:
        """入口：语义检索知识库 → 按维度统计 → 生成图表图片 → 返回统计结果。"""
        try:
            from collections import Counter

            payload = json.loads(params)

            query = (payload.get('query') or '').strip()
            dimension = (payload.get('dimension') or 'conference').strip()
            chart_type = (payload.get('chart_type') or 'bar').strip()
            top_k = int(payload.get('top_k', 50))
            top_n = int(payload.get('top_n', 10))

            if not query:
                return (
                    '统计失败：必须先提供 query 进行知识检索，不能直接全库统计。'
                    '例如：{"query":"paper", "dimension":"source", "chart_type":"bar"}'
                )

            dimension_names = {
                'conference': '会议/期刊',
                'source': '来源',
                'author': '作者',
                'year': '年份',
                'keywords': '关键词',
                'data_type': '数据类型',
                'category': '分类',
                'tags': '标签'
            }

            if dimension not in dimension_names:
                return "统计失败：dimension 只能是 conference/source/author/year/keywords/data_type/category/tags"

            ui_queue = kwargs.get('ui_queue', None)

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'\n[🔍 第一步] 正在按 query 检索知识库：{query}\n'
                })

            hits = vector_index.search(query=query, top_k=top_k)

            if not hits:
                return f"未检索到与“{query}”相关的知识片段，无法生成统计图。"

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': f'[🔍 第一步] 已召回 {len(hits)} 个相关片段，正在按论文去重并统计。\n'
                })

            knowledge_unit_hits = [
                h for h in hits
                if h.get('chunk_type') == 'knowledge_unit'
            ]

            analysis_hits = knowledge_unit_hits if knowledge_unit_hits else hits

            counter = Counter()
            seen_papers = set()
            evidence_count = 0

            for hit in analysis_hits:
                paper_key = (
                    hit.get('unit_name')
                    or hit.get('title')
                    or f"{hit.get('record_id')}_{hit.get('chunk_id', '')}"
                )

                if paper_key in seen_papers:
                    continue

                seen_papers.add(paper_key)

                content = hit.get('content', '') or ''
                source = hit.get('source', '') or ''
                author = hit.get('author', '') or ''
                unit_name = hit.get('unit_name', '') or ''

                combined_text = "\n".join([
                    str(unit_name),
                    str(source),
                    str(author),
                    str(content)
                ])

                evidence_count += 1

                if dimension in ('conference', 'source'):
                    values = self._extract_conferences(combined_text)

                    if not values and source:
                        values = self._extract_conferences(source)
                        if not values:
                            values = [source]

                    one_source = values[0] if values else 'Unknown'
                    counter.update([one_source])

                elif dimension == 'author':
                    values = self._extract_authors_from_text(combined_text)
                    if not values and author:
                        values = [author]
                    counter.update(values or ['Unknown'])

                elif dimension == 'year':
                    values = self._extract_years(combined_text)
                    one_year = values[0] if values else 'Unknown'
                    counter.update([one_year])

                elif dimension == 'keywords':
                    values = self._extract_keywords_from_text(combined_text)
                    counter.update(values or ['Unknown'])

                else:
                    value = hit.get(dimension, '') or 'Unknown'
                    counter[str(value).strip() or 'Unknown'] += 1

            if not counter:
                return f"没有可用于统计的 {dimension_names[dimension]} 数据。"

            if ui_queue:
                ui_queue.put({
                    'type': 'think_chunk',
                    'data': '[📊 第二步] 正在根据检索结果生成统计图...\n'
                })

            chart_path, top_items = self._make_chart(
                counter=counter,
                dimension_name=dimension_names[dimension],
                chart_type=chart_type,
                top_n=top_n,
                ui_queue=ui_queue
            )

            summary_lines = [
                "✅ 知识库统计图已生成。",
                "分析流程: 先按 query 检索知识库，再按论文去重统计画图",
                f"检索主题: {query}",
                f"统计维度: {dimension_names[dimension]}",
                f"图表类型: {chart_type}",
                f"参与统计的论文数: {evidence_count}",
                f"图表路径: {chart_path}",
                "",
                "Top 统计结果:"
            ]

            for label, count in top_items:
                summary_lines.append(f"- {label}: {count}")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"知识分析图表生成异常: {str(e)}"
