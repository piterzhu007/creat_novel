"""文本处理工具"""

import re


def clean_text(text: str) -> str:
    """清理文本中的非法 Unicode 字符"""
    # 移除私有区字符和代理对
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # 跳过代理对 (0xD800-0xDFFF) 和私有区
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if 0xE000 <= cp <= 0xF8FF:
            continue
        if 0xF0000 <= cp <= 0xFFFFD:
            continue
        if 0x100000 <= cp <= 0x10FFFD:
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def count_chinese_chars(text: str) -> int:
    """统计中文字符数"""
    return len(re.findall(r'[一-鿿㐀-䶿]', text))


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix
