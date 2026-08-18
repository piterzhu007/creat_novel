"""
LangGraph 工作流构建（官方 supervisor 模式，节点=agent本身）。

架构：
- architect/writer/editor/reader 是 create_agent 生成的独立 agent 子图
- supervisor 用 create_supervisor 统筹，通过 handoff 工具直接调用子 agent（A2A）
- 全局共享状态中枢（NovelState）贯穿所有 agent
- 子 agent 完全通过工具读写记忆库，supervisor 通过工具了解全局状态

节点 = agent 本身，无 wrapper 手工解析层。
"""

import sqlite3
from typing import Optional

from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from loguru import logger

from app.state import NovelState
from app.nodes import (
    NodeContext,
    create_novel_node,
    advance_node,
    handle_error_node,
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


def build_workflow(
    ltm: Optional[LongTermMemory] = None,
    stm: Optional[ShortTermMemory] = None,
    vs: Optional[VectorStore] = None,
    checkpoint_db_path: Optional[str] = None,
):
    """
    构建 supervisor 编排的小说创作 StateGraph。

    每个智能体都是 create_agent 生成的 agent 子图，supervisor 通过
    handoff 工具直接调用子 agent，实现真正的 A2A 协作。
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

    # ── 工具分配（最小权限原则） ──
    memory_tools = NovelMemoryTools(ltm, stm, vs)
    all_tools = memory_tools.get_tools()

    supervisor_tools = all_tools  # 主智能体，统筹全局

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

    # ── 创建 4 个子 agent（节点 = agent 本身） ──
    architect_agent = create_agent(
        model=model_registry.get_model("architect"),
        tools=architect_tools,
        system_prompt=ARCHITECT_PROMPT,
        name="architect",
    )
    writer_agent = create_agent(
        model=model_registry.get_model("writer"),
        tools=writer_tools,
        system_prompt=WRITER_PROMPT,
        name="writer",
    )
    editor_agent = create_agent(
        model=model_registry.get_model("editor"),
        tools=editor_tools,
        system_prompt=EDITOR_PROMPT,
        name="editor",
    )
    reader_agent = create_agent(
        model=model_registry.get_model("reader"),
        tools=reader_tools,
        system_prompt=READER_PROMPT,
        name="reader",
    )

    logger.info("4 个子 agent 已创建（architect/writer/editor/reader）")

    # ── 创建 supervisor（统筹全局，handoff 工具调用子 agent） ──
    supervisor_graph = create_supervisor(
        agents=[architect_agent, writer_agent, editor_agent, reader_agent],
        model=model_registry.get_model("supervisor"),
        tools=supervisor_tools,
        prompt=SUPERVISOR_PROMPT,
        supervisor_name="supervisor",
    ).compile()  # create_supervisor 返回 StateGraph，需 compile 后才能挂载

    logger.info("supervisor 已创建（handoff 工具 + 全局工具）")

    # ── 构建父 StateGraph ──
    ctx = NodeContext(model_registry, ltm, stm, vs, prompts)

    builder = StateGraph(NovelState)
    builder.add_node("create_novel", create_novel_node(ctx))
    builder.add_node("supervisor", supervisor_graph)  # supervisor 子图
    builder.add_node("advance", advance_node(ctx))
    builder.add_node("handle_error", handle_error_node(ctx))

    builder.add_edge(START, "create_novel")
    builder.add_edge("create_novel", "supervisor")

    # supervisor 完成后结束（其内部的 handoff 已处理子 agent 调用）
    builder.add_conditional_edges("supervisor", _after_supervisor, {
        "advance": "advance",
        "handle_error": "handle_error",
    })
    builder.add_edge("advance", "supervisor")  # 推进后回到 supervisor 继续下一章
    builder.add_edge("handle_error", END)

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


def _after_supervisor(state: NovelState) -> str:
    """supervisor 子图执行完后，根据状态决定是否推进章节"""
    # 若一章已完成审核（有终稿产出），进入 advance 推进；否则结束
    if state.get("latest_chapter_content") and state.get("reader_report"):
        return "advance"
    if state.get("error"):
        return "handle_error"
    # 默认结束（supervisor 内部已自行循环调度）
    return "handle_error"


def create_novel_workflow(checkpoint_db_path: Optional[str] = None):
    """创建默认的完整工作流"""
    return build_workflow(checkpoint_db_path=checkpoint_db_path)
