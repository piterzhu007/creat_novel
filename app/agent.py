"""
Deep Agent 架构：create_deep_agent + SubAgent 子智能体。

架构：
- supervisor 是主 agent（create_deep_agent），统筹全局，通过 task 工具委派任务
- architect/writer/editor/reader 是 SubAgent，专注各自任务
- 子智能体通过工具读写记忆库，supervisor 通过工具了解全局状态
- 所有智能体共享记忆系统（SQLite + ChromaDB），实现 A2A 协作

token 优化：
- 子智能体按需通过 get_story_bible 精简卡检索，而非全量拉取
- supervisor 提示词强调「委派指令 + 指针」，不粘贴大段上下文
"""

import sqlite3
from typing import Optional

from deepagents import create_deep_agent, SubAgent
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from app.core.model_client import get_model_registry
from app.core.config import PROJECT_ROOT
from app.memory import LongTermMemory, ShortTermMemory, VectorStore
from app.prompts import (
    SUPERVISOR_PROMPT,
    ARCHITECT_PROMPT,
    WRITER_PROMPT,
    EDITOR_PROMPT,
    READER_PROMPT,
)
from app.tools import NovelMemoryTools


def _build_sub_agents(memory_tools: NovelMemoryTools, registry) -> list[SubAgent]:
    """构建 4 个专业子智能体（每个有独立模型 + 工具子集）"""
    all_tools = memory_tools.get_tools()

    # 工具分配（最小权限原则，减少 token 浪费）
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

    return [
        {
            "name": "architect",
            "description": (
                "资深小说大纲设计师和世界观构建师。负责设计人物体系、构建世界观、"
                "制定分章大纲。在需要创意设计和大纲规划时调用。"
            ),
            "system_prompt": ARCHITECT_PROMPT,
            "model": registry.get_model("architect"),
            "tools": architect_tools,
        },
        {
            "name": "writer",
            "description": (
                "专业网络小说写手，根据大纲和设定撰写具体章节正文。"
                "在需要撰写或修改章节内容时调用。"
            ),
            "system_prompt": WRITER_PROMPT,
            "model": registry.get_model("writer"),
            "tools": writer_tools,
            # 为撰写者配备写作风格 skill（人物/环境/语言/动作描写方法论）
            "skills": [str(PROJECT_ROOT / "skills" / "writing-style")],
        },
        {
            "name": "editor",
            "description": (
                "资深出版社编辑，审核章节质量和大纲，从合理性、吸引力、价值观维度评分。"
                "在制定大纲和章节完成后需要质量评估时调用。"
            ),
            "system_prompt": EDITOR_PROMPT,
            "model": registry.get_model("editor"),
            "tools": editor_tools,
        },
        {
            "name": "reader",
            "description": (
                "资深读者和评论家，检查设定一致性、逻辑严谨性、文笔。"
                "在需要一致性检查和读者视角反馈时调用。"
            ),
            "system_prompt": READER_PROMPT,
            "model": registry.get_model("reader"),
            "tools": reader_tools,
        },
    ]


def create_novel_agent(
    ltm: Optional[LongTermMemory] = None,
    stm: Optional[ShortTermMemory] = None,
    vs: Optional[VectorStore] = None,
    checkpoint_db_path: Optional[str] = None,
):
    """
    创建小说创作 Deep Agent（supervisor + 4 个子智能体）。

    返回编译后的 deep agent 图。
    """
    if ltm is None:
        ltm = LongTermMemory()
    if stm is None:
        stm = ShortTermMemory()
    if vs is None:
        vs = VectorStore()

    registry = get_model_registry()
    registry.warmup_models()

    memory_tools = NovelMemoryTools(ltm, stm, vs)
    all_tools = memory_tools.get_tools()

    # 构建子智能体
    sub_agents = _build_sub_agents(memory_tools, registry)
    logger.info(f"已创建 {len(sub_agents)} 个子智能体: {[sa['name'] for sa in sub_agents]}")

    # checkpoint
    checkpointer = None
    if checkpoint_db_path:
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    # supervisor 是主 agent，拥有全部工具（统筹全局）
    agent = create_deep_agent(
        model=registry.get_model("supervisor"),
        tools=all_tools,
        system_prompt=SUPERVISOR_PROMPT,
        subagents=sub_agents,
        checkpointer=checkpointer,
        name="novel_supervisor",
    )

    logger.info(
        f"小说创作 Deep Agent 已创建（工具 {len(all_tools)} 个，子智能体 {len(sub_agents)} 个）"
    )
    return agent
