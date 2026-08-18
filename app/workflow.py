"""
LangGraph 工作流构建（Supervisor 模式，A2A + create_agent 子图）。

采用 LangGraph 标准的 supervisor pattern：
- supervisor 是主智能体（create_agent 子图），统筹全局、动态决策路由
- architect/writer/editor/reader 是子智能体（create_agent 子图），专注各自任务
- 所有智能体通过工具读写记忆库，共享全局状态中枢（NovelState）
- 子智能体间通过共享状态 + 记忆库实现 A2A 协作
"""

import sqlite3
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from app.state import NovelState
from app.nodes import (
    NodeContext,
    create_novel_node,
    supervisor_node,
    architect_node,
    writer_node,
    editor_node,
    reader_node,
    advance_node,
    handle_error_node,
    _build_agent,
)
from app.core.model_client import get_model_registry
from app.memory import LongTermMemory, ShortTermMemory, VectorStore
from app.prompts import (
    SUPERVISOR_PROMPT,
    ARCHITECT_PROMPT,
    WRITER_PROMPT,
    EDITOR_PROMPT,
    READER_PROMPT,
)
from app.tools import NovelMemoryTools


def _route_by_supervisor(state: NovelState) -> str:
    """根据 supervisor 的决策，路由到对应的 worker 节点"""
    decision = state.get("supervisor_decision", "finish")

    valid = {"architect", "writer", "editor", "reader", "advance", "finish"}
    if decision not in valid:
        logger.warning(f"supervisor 输出未知决策: {decision}，回退到 finish")
        return END

    if decision == "finish":
        return END

    return decision


def build_workflow(
    ltm: Optional[LongTermMemory] = None,
    stm: Optional[ShortTermMemory] = None,
    vs: Optional[VectorStore] = None,
    checkpoint_db_path: Optional[str] = None,
):
    """
    构建 supervisor 编排的小说创作 StateGraph。

    每个智能体节点都是 create_agent 生成的 agent 子图，拥有
    自己的 system_prompt、模型、工具子集，能自主循环调用工具。
    """
    if ltm is None:
        ltm = LongTermMemory()
    if stm is None:
        stm = ShortTermMemory()
    if vs is None:
        vs = VectorStore()

    model_registry = get_model_registry()
    model_registry.warmup_models()

    prompts = {
        "supervisor": SUPERVISOR_PROMPT,
        "architect": ARCHITECT_PROMPT,
        "writer": WRITER_PROMPT,
        "editor": EDITOR_PROMPT,
        "reader": READER_PROMPT,
    }
    ctx = NodeContext(model_registry, ltm, stm, vs, prompts)

    # ── 工具分配（最小权限原则） ──
    memory_tools = NovelMemoryTools(ltm, stm, vs)
    all_tools = memory_tools.get_tools()

    supervisor_tools = all_tools  # 主智能体，统筹全局，全部工具

    architect_tools = [t for t in all_tools if t.name in {
        "get_story_bible", "save_to_long_term",
        "get_novel_outline", "get_character_profile",
        "get_world_building", "list_novels",
    }]
    writer_tools = [t for t in all_tools if t.name in {
        "get_story_bible", "get_novel_outline", "get_character_profile",
        "get_world_building", "get_short_term_context",
        "search_similar_content", "save_chapter", "get_writing_issues",
    }]
    editor_tools = [t for t in all_tools if t.name in {
        "get_story_bible", "get_novel_outline", "get_character_profile",
        "get_short_term_context", "save_chapter",
        "save_writing_issue", "get_writing_issues",
    }]
    reader_tools = [t for t in all_tools if t.name in {
        "get_story_bible", "get_character_profile", "get_world_building",
        "search_similar_content", "save_writing_issue",
    }]

    # ── 创建 agent 子图 ──
    supervisor_agent = _build_agent(ctx, "supervisor", supervisor_tools)
    architect_agent = _build_agent(ctx, "architect", architect_tools)
    writer_agent = _build_agent(ctx, "writer", writer_tools)
    editor_agent = _build_agent(ctx, "editor", editor_tools)
    reader_agent = _build_agent(ctx, "reader", reader_tools)

    logger.info("5 个智能体子图已创建（supervisor/architect/writer/editor/reader）")

    # ── 构建父 StateGraph ──
    builder = StateGraph(NovelState)

    builder.add_node("create_novel", create_novel_node(ctx))
    builder.add_node("supervisor", supervisor_node(ctx, supervisor_agent))
    builder.add_node("architect", architect_node(ctx, architect_agent))
    builder.add_node("writer", writer_node(ctx, writer_agent))
    builder.add_node("editor", editor_node(ctx, editor_agent))
    builder.add_node("reader", reader_node(ctx, reader_agent))
    builder.add_node("advance", advance_node(ctx))
    builder.add_node("handle_error", handle_error_node(ctx))

    # 入口：先创建小说，再进入 supervisor 循环
    builder.add_edge(START, "create_novel")
    builder.add_edge("create_novel", "supervisor")

    # supervisor 动态路由到各 worker
    builder.add_conditional_edges("supervisor", _route_by_supervisor, {
        "architect": "architect",
        "writer": "writer",
        "editor": "editor",
        "reader": "reader",
        "advance": "advance",
        "handle_error": "handle_error",
    })

    # worker 完成后回到 supervisor（A2A 循环）
    builder.add_edge("architect", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("editor", "supervisor")
    builder.add_edge("reader", "supervisor")
    builder.add_edge("advance", "supervisor")
    builder.add_edge("handle_error", "supervisor")

    # 编译
    if checkpoint_db_path:
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = builder.compile(checkpointer=checkpointer)
        logger.info(f"工作流已编译（checkpoint: {checkpoint_db_path}）")
    else:
        graph = builder.compile()
        logger.info("工作流已编译（无 checkpoint）")

    return graph


def create_novel_workflow(checkpoint_db_path: Optional[str] = None):
    """创建默认的完整工作流"""
    return build_workflow(checkpoint_db_path=checkpoint_db_path)
