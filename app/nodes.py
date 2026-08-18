"""
LangGraph 节点：基于 create_agent 的 subagent 化智能体。

设计：
- 每个智能体（supervisor/architect/writer/editor/reader）用 langchain.agents.create_agent
  创建为真正的 ReAct agent 子图，拥有自己的 system_prompt（prompts.yaml）、模型、工具子集，
  能自主循环调用工具（检索记忆、保存数据）。
- wrapper 节点负责「父 NovelState ↔ 子 agent messages」的双向桥接：
  1. 从父状态提取必要指针（novel_id、章节号等），组装成任务指令
  2. 调用子 agent 执行，agent 自主通过工具读写记忆库
  3. 从子 agent 的最终回复提取结果，写回父 NovelState
- 数据检索权交给 agent（按需检索，节约 token），而非代码硬编码注入全量数据。
"""

import json
from datetime import datetime
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.state import NovelState


class NodeContext:
    """节点共享的依赖上下文（模型注册表 + 记忆 + 提示词）"""

    def __init__(self, model_registry, ltm, stm, vs, prompts):
        self.model_registry = model_registry
        self.ltm = ltm
        self.stm = stm
        self.vs = vs
        self.prompts = prompts  # dict: agent_name -> system_prompt str

    def get_model(self, agent_name: str):
        return self.model_registry.get_model(agent_name)

    def system_prompt(self, agent_name: str) -> str:
        return self.prompts.get(agent_name, "")


# ─── 工具函数 ────────────────────────────────────────────

def _get_last_user_message(state: NovelState) -> str:
    """从消息列表提取最后一条用户消息"""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and getattr(msg, "type", "") == "human":
            return content
    return ""


def _last_ai_content(messages: list) -> str:
    """提取最后一条 AI 消息的文本内容"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "content", ""):
            return msg.content
    return ""


def _parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON（容错）"""
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def _profiles_to_dicts(profiles: list) -> list[dict]:
    """Pydantic 对象列表 → dict 列表"""
    result = []
    for p in profiles:
        if hasattr(p, "model_dump"):
            result.append(p.model_dump())
        elif isinstance(p, dict):
            result.append(p)
    return result


# ─── agent 工厂 ──────────────────────────────────────────

def _build_agent(ctx: NodeContext, agent_name: str, tools: list):
    """用 create_agent 创建 ReAct agent 子图"""
    model = ctx.get_model(agent_name)
    system_prompt = ctx.system_prompt(agent_name)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        name=agent_name,
    )


# ─── wrapper 节点 ────────────────────────────────────────

def create_novel_node(ctx: NodeContext):
    """创建小说项目节点（流程节点，非智能体）"""
    def node(state: NovelState) -> dict:
        user_msg = _get_last_user_message(state)
        title = state.get("novel_title", "")
        if not title:
            title = user_msg[:30] or f"未命名小说_{datetime.utcnow().strftime('%Y%m%d')}"

        genre = state.get("novel_genre", "")
        synopsis = state.get("novel_synopsis", user_msg[:200])
        target = state.get("target_chapters", 0)

        novel_id = ctx.ltm.create_novel(title, genre, synopsis, target)
        logger.info(f"[create_novel] 已创建小说: {title} (id={novel_id})")

        return {
            "novel_id": novel_id,
            "novel_title": title,
            "novel_genre": genre,
            "novel_synopsis": synopsis,
            "target_chapters": target,
            "phase": "design",
        }
    return node


def supervisor_node(ctx: NodeContext, agent):
    """supervisor 决策节点：agent 读状态摘要，动态决策下一步"""
    def node(state: NovelState) -> dict:
        summary = {
            "phase": state.get("phase", "design"),
            "current_chapter": state.get("current_chapter", 1),
            "novel_id": state.get("novel_id", ""),
            "novel_title": state.get("novel_title", ""),
            "target_chapters": state.get("target_chapters", 0),
            "has_characters": len(state.get("characters", [])) > 0,
            "has_outlines": len(state.get("outlines", [])) > 0,
            "has_writer_output": bool(state.get("writer_output", "")),
            "has_editor_report": bool(state.get("editor_report")),
            "has_reader_report": bool(state.get("reader_report")),
            "editor_score": state.get("editor_report", {}).get("overall_score"),
            "reader_consistent": state.get("reader_report", {}).get("is_consistent"),
            "has_editor_feedback": bool(state.get("editor_feedback", "")),
            "has_reader_feedback": bool(state.get("reader_feedback", "")),
        }

        task = (
            f"当前全局创作状态：\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
            f"请据此决策下一步该调用哪个智能体，输出 JSON：{{\"next\": \"...\", \"reason\": \"...\"}}"
        )

        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = _last_ai_content(result.get("messages", []))
        decision = _parse_json(content)

        next_node = decision.get("next", "finish")
        reason = decision.get("reason", "")
        logger.info(f"[supervisor] 决策 → {next_node}（{reason}）")

        return {
            "supervisor_decision": next_node,
            "supervisor_reason": reason,
        }
    return node


def architect_node(ctx: NodeContext, agent):
    """architect 设计节点：agent 自主设计人物/世界观/大纲，通过工具保存"""
    def node(state: NovelState) -> dict:
        novel_id = state.get("novel_id", "")
        title = state.get("novel_title", "")
        user_msg = _get_last_user_message(state)

        task = (
            f"请为小说《{title}》设计完整的人物体系、世界观设定和分章大纲。\n"
            f"小说ID：{novel_id}\n"
            f"用户需求：{user_msg}\n\n"
            f"要求：\n"
            f"1. 设计完成后，调用 save_to_long_term 把每个人物（category=character）、"
            f"每条世界观（category=setting）、每章大纲（用 get_novel_outline 前先 save_to_long_term category=plot 保存）保存到长期记忆\n"
            f"2. 角色名一旦确定就是最终名，全篇统一\n"
            f"3. 完成后用一句话总结你的设计"
        )

        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = _last_ai_content(result.get("messages", []))

        # 从记忆库读回设计结果（agent 通过工具保存的），写回共享状态
        characters = _profiles_to_dicts(ctx.ltm.get_characters(novel_id))
        world_settings = _profiles_to_dicts(ctx.ltm.get_world_settings(novel_id))
        outlines = _profiles_to_dicts(ctx.ltm.get_outlines(novel_id))

        logger.info(
            f"[architect] 完成设计：{len(characters)} 人物，{len(world_settings)} 世界观，{len(outlines)} 章大纲"
        )

        return {
            "characters": characters,
            "world_settings": world_settings,
            "outlines": outlines,
            "architect_output": content,
            "phase": "writing",
            "current_chapter": state.get("current_chapter", 1),
        }
    return node


def writer_node(ctx: NodeContext, agent):
    """writer 撰写节点：agent 自主检索上下文并撰写，通过工具保存"""
    def node(state: NovelState) -> dict:
        novel_id = state.get("novel_id", "")
        chapter = state.get("current_chapter", 1)
        editor_feedback = state.get("editor_feedback", "")
        reader_feedback = state.get("reader_feedback", "")

        task_parts = [
            f"请撰写第 {chapter} 章正文。\n",
            f"小说ID：{novel_id}\n",
            f"章节序号：{chapter}\n",
        ]
        if editor_feedback:
            task_parts.append(f"\n【编辑反馈】（必须针对性修改）\n{editor_feedback}\n")
        if reader_feedback:
            task_parts.append(f"\n【读者反馈】（必须修正矛盾点）\n{reader_feedback}\n")

        task = "".join(task_parts)

        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = _last_ai_content(result.get("messages", []))

        # 从记忆库读回最新草稿（agent 通过 save_chapter 保存的）
        draft = ctx.stm.get_latest_draft(novel_id, chapter)
        chapter_content = draft.content if draft else content
        title = draft.title if draft and draft.title else f"第{chapter}章"

        logger.info(f"[writer] 完成第{chapter}章《{title}》({len(chapter_content)}字)")

        return {
            "writer_output": chapter_content,
            "latest_chapter_content": chapter_content,
            "phase": "editing",
            "editor_feedback": "",
            "reader_feedback": "",
        }
    return node


def editor_node(ctx: NodeContext, agent):
    """editor 审核节点：agent 自主检索上下文审核，输出评分+反馈"""
    def node(state: NovelState) -> dict:
        novel_id = state.get("novel_id", "")
        chapter = state.get("current_chapter", 1)

        task = (
            f"请审核第 {chapter} 章的正文质量，从合理性、吸引力、价值观三个维度评估。\n"
            f"小说ID：{novel_id}\n"
            f"章节序号：{chapter}\n\n"
            f"要求：\n"
            f"1. 先调用 get_short_term_context 获取该章正文\n"
            f"2. 从合理性、吸引力、价值观三个维度评估\n"
            f"3. 输出 JSON：{{\"overall_score\": 7, \"issues\": [\"问题\"], \"suggestions\": [\"建议\"]}}\n"
            f"4. 若发现问题，调用 save_writing_issue 记录"
        )

        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = _last_ai_content(result.get("messages", []))
        report = _parse_json(content)

        score = report.get("overall_score", 7)
        logger.info(f"[editor] 第{chapter}章评分: {score}")

        # 构造反馈文本（供 writer 重写参考）
        feedback_parts = []
        issues = report.get("issues", [])
        suggestions = report.get("suggestions", [])
        if issues:
            feedback_parts.append("问题：" + "；".join(issues))
        if suggestions:
            feedback_parts.append("建议：" + "；".join(suggestions))
        feedback = "\n".join(feedback_parts) if feedback_parts else ""

        return {
            "editor_report": report,
            "editor_feedback": feedback,
            "phase": "checking",
        }
    return node


def reader_node(ctx: NodeContext, agent):
    """reader 一致性检查节点：agent 自主检索设定检查，输出矛盾点"""
    def node(state: NovelState) -> dict:
        novel_id = state.get("novel_id", "")
        chapter = state.get("current_chapter", 1)

        task = (
            f"请检查第 {chapter} 章正文与人物设定、世界观的一致性。\n"
            f"小说ID：{novel_id}\n"
            f"章节序号：{chapter}\n\n"
            f"要求：\n"
            f"1. 先调用 get_story_bible 获取权威设定卡（角色名表）\n"
            f"2. 调用 get_short_term_context 获取该章正文\n"
            f"3. 检查：角色名是否一致、人物行为是否符合性格、世界观规则是否被违背\n"
            f"4. 输出 JSON：{{\"is_consistent\": true, \"consistency_issues\": [\"矛盾点\"]}}\n"
            f"5. 若发现矛盾，调用 save_writing_issue 记录"
        )

        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = _last_ai_content(result.get("messages", []))
        report = _parse_json(content)

        issues = report.get("consistency_issues", [])
        is_consistent = report.get("is_consistent", True)
        logger.info(f"[reader] 第{chapter}章一致性: {'通过' if is_consistent else f'{len(issues)}个问题'}")

        feedback = "；".join(issues) if issues else ""

        return {
            "reader_report": report,
            "reader_feedback": feedback,
            "phase": "done",
            "needs_user_input": not is_consistent,
        }
    return node


def advance_node(ctx: NodeContext):
    """推进节点：导出本章终稿、更新进度（流程节点，非智能体）"""
    def node(state: NovelState) -> dict:
        chapter = state.get("current_chapter", 1)
        content = state.get("latest_chapter_content", "")
        outlines = state.get("outlines", [])
        novel_id = state.get("novel_id", "")

        # 本章通过审核，导出终稿
        editor_score = state.get("editor_report", {}).get("overall_score", 0)
        reader_ok = state.get("reader_report", {}).get("is_consistent", True)
        if content and (editor_score >= 6) and reader_ok:
            try:
                exported = ctx.stm.export_single_chapter(novel_id, chapter)
                if exported:
                    logger.info(f"[advance] 第{chapter}章终稿已导出: {exported['filepath']}")
            except Exception as e:
                logger.warning(f"[advance] 导出第{chapter}章失败: {e}")

        ending = content[-500:] if len(content) > 500 else content
        has_next = any(o.get("chapter_seq", 0) > chapter for o in outlines)

        if has_next:
            return {
                "previous_chapter_ending": ending,
                "current_chapter": chapter + 1,
                "phase": "writing",
                "writer_output": "",
                "latest_chapter_content": "",
                "editor_report": None,
                "reader_report": None,
            }
        else:
            return {
                "previous_chapter_ending": ending,
                "phase": "completed",
            }
    return node


def handle_error_node(ctx: NodeContext):
    """错误处理节点"""
    def node(state: NovelState) -> dict:
        error = state.get("error", "未知错误")
        logger.error(f"[handle_error] {error}")
        return {
            "needs_user_input": True,
            "user_message": f"处理出错：{error}",
            "error": None,
        }
    return node


# ─── 导出 ────────────────────────────────────────────────

__all__ = [
    "NodeContext",
    "create_novel_node",
    "supervisor_node",
    "architect_node",
    "writer_node",
    "editor_node",
    "reader_node",
    "advance_node",
    "handle_error_node",
]
