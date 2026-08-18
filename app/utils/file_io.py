"""文件读写工具"""

import json
from pathlib import Path
from typing import Any, Optional

# 支持的文档输入格式
SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".markdown", ".yaml", ".yml", ".json", ".py", ".text"}


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(filepath: str) -> dict[str, Any]:
    """读取 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: str, data: dict[str, Any], indent: int = 2):
    """写入 JSON 文件"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_text(filepath: str) -> str:
    """读取文本文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_text(filepath: str, content: str):
    """写入文本文件"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def read_document(filepath: str | Path) -> str:
    """
    读取文档内容（支持 txt/md/yaml/json 等文本格式）。

    参数:
        filepath: 文档路径

    返回:
        文档文本内容

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的格式
    """
    p = Path(filepath).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if not p.is_file():
        raise ValueError(f"不是文件: {p}")
    if p.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
        raise ValueError(f"不支持的文档格式: {p.suffix}（支持: {sorted(SUPPORTED_DOC_EXTENSIONS)}）")
    return read_text(str(p))


def is_document_path(text: str) -> bool:
    """判断文本是否是一个可读的文档路径"""
    p = Path(text.strip().strip('"').strip("'")).expanduser()
    return p.exists() and p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS


def resolve_document_path(text: str) -> Optional[Path]:
    """将文本解析为文档路径（若合法），否则返回 None"""
    cleaned = text.strip().strip('"').strip("'")
    p = Path(cleaned).expanduser()
    if p.exists() and p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
        return p
    return None
