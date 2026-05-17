import re
from typing import List, Dict


_PATH_PATTERNS = [
    r'(?i)\b[a-z]:\\[^\s\'"<>|]+',
    r'(?i)\b[a-z]:/[^\s\'"<>|]+',
    r'(?i)\\\\[^\s\'"<>|]+',
    r'(?i)\B/[^\s\'"<>|]+',
]


def _sanitize_prompt_text(text: str) -> str:
    cleaned = text or ''
    for pattern in _PATH_PATTERNS:
        cleaned = re.sub(pattern, '[已省略路径]', cleaned)
    cleaned = re.sub(r'(source_path\s*[:=]\s*)([^\n,}\]]+)', r'\1[已省略路径]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _sanitize_history(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sanitized = []
    for record in records:
        sanitized.append({
            'role': record.get('role', 'user'),
            'content': _sanitize_prompt_text(record.get('content', '')),
        })
    return sanitized


def _trim_long_fields(text: str, max_field_len: int = 100) -> str:
    """
    只裁剪过长字段，不裁剪整个思考过程。
    优先保留关键词命中附近的片段。
    """
    long_fields = [
        "raw_text",
        "structured_json",
        "summary",
        "entity_text",
        "keyword_text",
        "tags",
    ]

    # 尽量从当前文本中提取检索关键词
    keyword = ""
    keyword_match = re.search(r'"keyword"\s*:\s*"([^"]+)"', text)
    if keyword_match:
        keyword = keyword_match.group(1).strip()

    def smart_cut(value: str, field: str) -> str:
        if len(value) <= max_field_len:
            return value

        # 如果能找到关键词，保留关键词前后片段
        if keyword and keyword in value:
            pos = value.find(keyword)
            half = max_field_len // 2
            start = max(0, pos - half)
            end = min(len(value), pos + half)

            prefix = "...【前文已裁剪】" if start > 0 else ""
            suffix = f"...【{field}字段过长，后文已裁剪】" if end < len(value) else ""

            return prefix + value[start:end] + suffix

        # 没有关键词时，保留开头和结尾
        head_len = max_field_len // 2
        tail_len = max_field_len // 2

        return (
                value[:head_len]
                + f"...【{field}字段过长，中间已裁剪】..."
                + value[-tail_len:]
        )

    for field in long_fields:
        pattern = rf'("{field}"\s*:\s*")(.+?)(")(\s*,|\s*}})'

        def repl(match):
            prefix = match.group(1)
            value = match.group(2)
            suffix = match.group(3)
            tail = match.group(4)

            value = smart_cut(value, field)

            return prefix + value + suffix + tail

        text = re.sub(pattern, repl, text, flags=re.DOTALL)

    return text

def _format_thinking_display(text: str, max_field_len: int = 300) -> str:
    """
    仅用于界面展示，不影响模型真实推理内容。
    目的：隐藏数据库原始 JSON、长字段、路径、工具返回细节，让思考过程更像人类可读日志。
    """
    if not text:
        return ""

    display_text = _sanitize_prompt_text(text)

    # 如果是大段 JSON 或数据库原始记录，不直接展示
    if (
        '"raw_text"' in display_text
        or '"structured_json"' in display_text
        or '"source_path"' in display_text
        or '"created_at"' in display_text
        or '"updated_at"' in display_text
    ):
        return "正在读取知识库记录，并筛选与问题相关的内容...\n"

    # 如果模型正在输出工具调用，只保留工具动作，不展示参数详情
    action_match = re.search(r'Action:\s*(.+)', display_text)
    if action_match:
        tool_name = action_match.group(1).strip()
        if tool_name and tool_name != "Finish":
            return f"准备调用工具：{tool_name}\n"

    # 如果是 Observation，不展示原始返回
    if "Observation:" in display_text:
        return "工具已返回结果，正在继续分析...\n"

    # 原有长字段裁剪逻辑继续复用
    display_text = _trim_long_fields(display_text, max_field_len=max_field_len)

    if len(display_text) > max_field_len:
        display_text = display_text[:max_field_len] + "...【内容较长，已折叠】"

    return display_text

