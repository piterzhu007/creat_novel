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

# writing-style skill 的 SKILL.md 路径（内联进 writer 提示词，确保真正生效）
_WRITING_STYLE_SKILL = (
    Path(__file__).resolve().parent.parent.parent / "skills" / "writing-style" / "SKILL.md"
)
# novel-anti-ai-style skill 路径（内联进 writer + reader 提示词，确保真正生效）
# 注意目录名里的连字符是非断行连字符 U+2011（‑），非普通连字符 -
_ANTI_AI_STYLE_SKILL = (
    Path(__file__).resolve().parent.parent.parent / "skills" / "novel‑anti‑ai‑style" / "SKILL.md"
)


def _load_prompts_yaml() -> dict:
    """加载 prompts.yaml 文件"""
    yaml_path = _PROMPTS_DIR / "prompts.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_writing_style_skill() -> str:
    """读取 writing-style skill 的完整内容（内联进 writer 提示词）。

    说明：deepagents 的 skill 是「渐进式披露」机制，需要 agent 主动 read_file
    才读取 SKILL.md 全文。但 writer 在实际运行中从未主动读取，导致 skill 形同虚设。
    因此这里直接把 SKILL.md 全文内联进 writer 的 system prompt，保证 100% 生效。
    """
    if not _WRITING_STYLE_SKILL.exists():
        return ""
    try:
        return _WRITING_STYLE_SKILL.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_anti_ai_style_skill() -> str:
    """读取 novel-anti-ai-style skill 的完整内容（内联进 writer + reader 提示词）。

    writer 把它当作「必须规避的 AI 写作痕迹」硬规则；reader 把它当作
    「审核时逐项核对」的检查清单。同样因为 deepagents 的渐进式披露机制
    会导致 skill 不被主动读取，这里直接内联保证 100% 生效。
    """
    if not _ANTI_AI_STYLE_SKILL.exists():
        return ""
    try:
        return _ANTI_AI_STYLE_SKILL.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def load_system_prompt(agent_name: str) -> str:
    """加载指定智能体的系统提示词文本（供 deepagents 使用）"""
    if agent_name not in _PROMPTS_CACHE:
        prompts_data = _load_prompts_yaml()
        agent_config = prompts_data.get(agent_name, {})
        prompt = agent_config.get("system_prompt", "").strip()

        # writer 额外内联 writing-style + novel-anti-ai-style skill 全文，确保真正生效
        if agent_name == "writer":
            skill_text = _load_writing_style_skill()
            if skill_text:
                prompt += "\n\n## 写作风格方法论（writing-style skill，必须严格遵循）\n\n" + skill_text
            anti_ai_text = _load_anti_ai_style_skill()
            if anti_ai_text:
                prompt += "\n\n## 反AI写作痕迹规则（novel-anti-ai-style skill，必须严格遵循）\n\n" + anti_ai_text

        # reader 额外内联 novel-anti-ai-style skill（审核时按此识别 AI 写作痕迹并记问题）
        if agent_name == "reader":
            anti_ai_text = _load_anti_ai_style_skill()
            if anti_ai_text:
                prompt += "\n\n## 反AI写作痕迹检查清单（novel-anti-ai-style skill，审核时逐项核对并记问题）\n\n" + anti_ai_text

        _PROMPTS_CACHE[agent_name] = prompt
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
    """初始化所有提示词（走 load_system_prompt，writer 自动内联 writing-style skill）"""
    global SUPERVISOR_PROMPT, ARCHITECT_PROMPT, WRITER_PROMPT, EDITOR_PROMPT, READER_PROMPT
    SUPERVISOR_PROMPT = load_system_prompt("supervisor")
    ARCHITECT_PROMPT = load_system_prompt("architect")
    WRITER_PROMPT = load_system_prompt("writer")
    EDITOR_PROMPT = load_system_prompt("editor")
    READER_PROMPT = load_system_prompt("reader")


_init_prompts()
