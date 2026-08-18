"""
提示词模块：统一管理所有智能体的系统提示词。

使用 ChatPromptTemplate 提供 langchain 统一的提示词接口。
每个智能体的系统提示词从 prompts.yaml 加载，并提供 ChatPromptTemplate 包装。
"""

from pathlib import Path
from typing import Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate

# ─── 提示词加载 ───────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "conf"
_PROMPTS_CACHE: dict[str, str] = {}


def _load_prompts_yaml() -> dict:
    """加载 prompts.yaml 文件"""
    yaml_path = _PROMPTS_DIR / "prompts.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_system_prompt(agent_name: str) -> str:
    """加载指定智能体的系统提示词文本（供 deepagents 使用）"""
    if agent_name not in _PROMPTS_CACHE:
        prompts_data = _load_prompts_yaml()
        agent_config = prompts_data.get(agent_name, {})
        _PROMPTS_CACHE[agent_name] = agent_config.get("system_prompt", "").strip()
    return _PROMPTS_CACHE[agent_name]


def create_chat_template(
    agent_name: str,
    user_template: str = "{input}",
) -> ChatPromptTemplate:
    """
    为指定智能体创建 ChatPromptTemplate。

    参数:
        agent_name: 智能体名称 (supervisor/architect/writer/editor/reader)
        user_template: 用户消息模板，默认 "{input}"

    返回:
        ChatPromptTemplate 实例，格式: [("system", system_prompt), ("user", user_template)]
    """
    system_prompt = load_system_prompt(agent_name)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_template),
    ])


def get_all_prompts() -> dict[str, str]:
    """获取所有智能体的系统提示词"""
    prompts_data = _load_prompts_yaml()
    return {
        agent: cfg.get("system_prompt", "").strip()
        for agent, cfg in prompts_data.items()
        if "system_prompt" in cfg
    }


# ─── 预加载提示词 ────────────────────────────────────────

SUPERVISOR_PROMPT: str = ""  # 延迟加载
ARCHITECT_PROMPT: str = ""
WRITER_PROMPT: str = ""
EDITOR_PROMPT: str = ""
READER_PROMPT: str = ""


def _init_prompts():
    """初始化所有提示词"""
    global SUPERVISOR_PROMPT, ARCHITECT_PROMPT, WRITER_PROMPT, EDITOR_PROMPT, READER_PROMPT
    prompts = get_all_prompts()
    SUPERVISOR_PROMPT = prompts.get("supervisor", "")
    ARCHITECT_PROMPT = prompts.get("architect", "")
    WRITER_PROMPT = prompts.get("writer", "")
    EDITOR_PROMPT = prompts.get("editor", "")
    READER_PROMPT = prompts.get("reader", "")


_init_prompts()
