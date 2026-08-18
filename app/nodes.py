"""
LangGraph 节点：流程节点（非智能体）。

智能体（supervisor/architect/writer/editor/reader）已改用 create_agent 子图，
由 app/workflow.py 用 create_supervisor 统筹。本文件只保留纯流程节点：
- create_novel：创建小说项目（写数据库）
- advance：推进章节、导出终稿（流程控制）
- handle_error：错误处理
"""

from datetime import datetime

from loguru import logger

from app.state import NovelState


class NodeContext:
    """节点共享的依赖上下文（模型注册表 + 记忆 + 提示词）"""

    def __init__(self, model_registry, ltm, stm, vs, prompts):
        self.model_registry = model_registry
        self.ltm = ltm
        self.stm = stm
        self.vs = vs
        self.prompts = prompts

    def get_model(self, agent_name: str):
        return self.model_registry.get_model(agent_name)

    def system_prompt(self, agent_name: str) -> str:
        return self.prompts.get(agent_name, "")


def _get_last_user_message(state: NovelState) -> str:
    """从消息列表提取最后一条用户消息"""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and getattr(msg, "type", "") == "human":
            return content
    return ""


def create_novel_node(ctx: NodeContext):
    """创建小说项目节点（流程节点）"""
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
        }
    return node


def advance_node(ctx: NodeContext):
    """推进节点：导出本章终稿、更新进度（流程节点）"""
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
                "writer_output": "",
                "latest_chapter_content": "",
                "editor_report": None,
                "reader_report": None,
            }
        else:
            return {
                "previous_chapter_ending": ending,
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


__all__ = [
    "NodeContext",
    "create_novel_node",
    "advance_node",
    "handle_error_node",
]
