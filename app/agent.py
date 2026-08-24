"""
Deep Agent 架构：create_deep_agent + SubAgent 子智能体。

架构：
- supervisor 是主 agent（create_deep_agent），统筹全局，通过 task 工具委派任务
- architect/writer/editor/reader 是 SubAgent，专注各自任务
- 所有工具统一通过 MCP 接口暴露（app/mcp/server.py 的 FastMCP stdio 服务器），
  任何智能体都可以自由使用全部工具——不再做 per-agent 最小权限分配
- 所有智能体共享记忆系统（SQLite + ChromaDB + MCP），实现 A2A 协作
"""

import sys
from typing import Optional

from deepagents import create_deep_agent, SubAgent
from loguru import logger

from app.core.model_client import get_model_registry
from app.core.config import PROJECT_ROOT
from app.prompts import (
    SUPERVISOR_PROMPT,
    ARCHITECT_PROMPT,
    WRITER_PROMPT,
    EDITOR_PROMPT,
    READER_PROMPT,
)


# 文件系统工具集合：这些工具在 Windows 上存在硬编码缺陷（deepagents 的
# validate_path 拒绝 D:\ 绝对路径），且源文档已由 read_source_docs 工具读取，
# 模型不需要 deepagents 内置的文件探索。从代码层面彻底移除，避免 read_file 死循环。
_FS_TOOLS = frozenset({
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
})

# 写工具集合：修改存储记忆的能力（硬约束）。
# 全部写工具只给 supervisor；子智能体按「写库职责边界」各自拿到少量写工具，
# 只写自己的产出（草稿），定稿/删除/覆盖权由 supervisor 独占。
_WRITE_TOOLS = frozenset({
    "create_novel", "update_novel_progress", "save_to_long_term",
    "save_chapter", "patch_chapter", "update_short_term",
    "save_writing_issue", "export_chapters", "delete_long_term_entry",
    "save_run_log", "lock_entry",
})

# 子智能体的「写库职责边界」——各自只能写自己的产出：
# architect 首建设计（人物/世界观/大纲），writer 写正文/子情节，editor/reader 记问题。
# 其余写工具（定稿锁、删除、patch、进度、导出、日志）由 supervisor 独占。
_ARCHITECT_WRITE_TOOLS = frozenset({"save_to_long_term"})
_WRITER_WRITE_TOOLS = frozenset({"save_chapter", "update_short_term"})
_EDITOR_WRITE_TOOLS = frozenset({"save_writing_issue"})


def _register_no_filesystem_profile():
    r"""
    从代码层面禁用所有文件系统工具 + 禁用 general-purpose 子智能体。

    根因：deepagents 自动添加的 general-purpose 子智能体拥有完整文件工具，
    在 Windows 上 read_file 会硬编码报错（validate_path 拒绝 D:\ 绝对路径），
    导致模型陷入「读→失败→换路径→再读」的死循环。此前用 middleware 限制
    只能影响 supervisor，管不住自动生成的 general-purpose 子智能体。

    这里用 HarnessProfile 在 provider 级别（deepseek）做全局排除，
    从物理上切断所有智能体（含 general-purpose）获取文件工具的能力。
    """
    from deepagents import HarnessProfile, GeneralPurposeSubagentProfile, register_harness_profile
    # 先触发内置 profile 懒加载 bootstrap，确保注册时序正确（不会被后续 bootstrap 覆盖）
    # 注意：_ensure_harness_profiles_loaded 是 deepagents 0.7.x 的私有符号，
    # 依赖内部实现（requirements 已 pin deepagents<0.8），升级前需验证是否仍存在。
    from deepagents.profiles.harness.harness_profiles import _ensure_harness_profiles_loaded
    _ensure_harness_profiles_loaded()

    register_harness_profile(
        "deepseek",
        HarnessProfile(
            excluded_tools=_FS_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    logger.info(f"已注册 HarnessProfile：禁用文件系统工具 {sorted(_FS_TOOLS)} + 禁用 general-purpose")


def _load_mcp_tools():
    """
    连接 MCP stdio 服务器，加载全部记忆工具为 langchain 工具。

    关键：必须用「持久 session」——若用 MultiServerMCPClient.get_tools()（内部传
    session=None），会导致每次工具调用都重新 spawn 子进程 + 重新初始化记忆后端
    （SQLite/ChromaDB/embedding 各初始化一次，数秒延迟）。这里手动建立一个持久
    session 并保持不关闭，所有工具共享它。

    所有工具统一通过 MCP 接口暴露，任何智能体自由使用全部工具。
    """
    from langchain_mcp_adapters.sessions import create_session
    from langchain_mcp_adapters.tools import load_mcp_tools
    from app.core.async_runtime import run

    connection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "app.mcp.server"],
        "cwd": str(PROJECT_ROOT),
    }

    async def _setup():
        # 手动进入 session context 并保持不退出（应用生命周期内复用）
        session_cm = create_session(connection)
        session = await session_cm.__aenter__()
        await session.initialize()
        # 把 context manager 挂到全局，防止被 GC 回收提前关闭
        _session_holder.append(session_cm)
        return await load_mcp_tools(session)

    tools = run(_setup())
    logger.info(f"已通过 MCP 接口加载 {len(tools)} 个工具（持久 session）: {[t.name for t in tools]}")
    return tools


# 持久 session 的 context manager 引用，保持 MCP 连接不关闭
_session_holder: list = []


def _build_sub_agents(all_tools: list, registry) -> list[SubAgent]:
    """构建 4 个专业子智能体（细粒度写库边界）。

    权限硬约束（方案 A）：
    - 读工具（get_*/search_*/list_novels/read_source_docs）所有子智能体共享。
    - 写工具按职责边界细分：architect 写 save_to_long_term（首建设计），
      writer 写 save_chapter/update_short_term（正文/子情节），
      editor/reader 写 save_writing_issue（问题）。
    - 定稿锁、删除、patch、进度、导出、日志等写工具由 supervisor 独占（子智能体拿不到）。

    不截断硬约束：子智能体不配任何 summarization 中间件，不中途压缩上下文。
    token 节约靠「产出下沉（子智能体写库，不进 supervisor 上下文）+ 源文档只读一次
    + 权威名表落库前校验（不删除重建）」实现。
    """
    read_tools = [t for t in all_tools if t.name not in _WRITE_TOOLS]

    def _tools_for(write_names: frozenset) -> list:
        """读工具 + 该子智能体的专属写工具"""
        return read_tools + [t for t in all_tools if t.name in write_names]

    return [
        {
            "name": "architect",
            "description": (
                "资深小说大纲设计师和世界观构建师。负责设计人物体系、构建世界观、"
                "制定分章大纲。在需要创意设计和大纲规划时调用。"
            ),
            "system_prompt": ARCHITECT_PROMPT,
            "model": registry.get_model("architect"),
            "tools": _tools_for(_ARCHITECT_WRITE_TOOLS),
        },
        {
            "name": "writer",
            "description": (
                "专业网络小说写手，根据大纲和设定撰写具体章节正文。"
                "在需要撰写或修改章节内容时调用。"
            ),
            "system_prompt": WRITER_PROMPT,
            "model": registry.get_model("writer"),
            "tools": _tools_for(_WRITER_WRITE_TOOLS),
        },
        {
            "name": "editor",
            "description": (
                "资深出版社编辑，审核章节质量和大纲，从合理性、吸引力、价值观维度评分。"
                "在制定大纲和章节完成后需要质量评估时调用。"
            ),
            "system_prompt": EDITOR_PROMPT,
            "model": registry.get_model("editor"),
            "tools": _tools_for(_EDITOR_WRITE_TOOLS),
        },
        {
            "name": "reader",
            "description": (
                "资深读者和评论家，检查设定一致性、逻辑严谨性、文笔。"
                "在需要一致性检查和读者视角反馈时调用。"
            ),
            "system_prompt": READER_PROMPT,
            "model": registry.get_model("reader"),
            "tools": _tools_for(_EDITOR_WRITE_TOOLS),
        },
    ]


def create_novel_agent(checkpoint_db_path: Optional[str] = None):
    """
    创建小说创作 Deep Agent（supervisor + 4 个子智能体）。

    所有工具统一通过 MCP 接口加载，任何智能体自由使用全部工具。
    返回编译后的 deep agent 图。
    """
    registry = get_model_registry()
    registry.warmup_models()

    # 通过 MCP 接口加载全部工具（单一大脑中枢，所有智能体共享）
    all_tools = _load_mcp_tools()

    # 配置本地文件系统 backend（deepagent 需要的文件系统 backend，文件工具已禁用）
    from deepagents.backends.filesystem import FilesystemBackend
    fs_backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=False)

    # 构建子智能体（子智能体只读工具，写工具归 supervisor）
    sub_agents = _build_sub_agents(all_tools, registry)
    logger.info(f"已创建 {len(sub_agents)} 个子智能体: {[sa['name'] for sa in sub_agents]}")

    # checkpoint
    # checkpoint：MCP 工具是 async-only，agent 用 astream 运行，
    # 因此必须用 AsyncSqliteSaver（同步 SqliteSaver 不支持 async 方法）。
    # 用持久 loop 建立 aiosqlite 连接，避免连接跨 loop 失效。
    checkpointer = None
    if checkpoint_db_path:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from app.core.async_runtime import run

        async def _make_saver():
            conn = await aiosqlite.connect(checkpoint_db_path)
            return AsyncSqliteSaver(conn)

        checkpointer = run(_make_saver())

    # 从代码层面禁用文件系统工具 + general-purpose 子智能体（根治 read_file 死循环）
    _register_no_filesystem_profile()

    # supervisor 是主 agent，拥有全部工具（含写工具，独占落库权限），不配上下文压缩
    supervisor_model = registry.get_model("supervisor")

    agent = create_deep_agent(
        model=supervisor_model,
        tools=all_tools,
        system_prompt=SUPERVISOR_PROMPT,
        subagents=sub_agents,
        checkpointer=checkpointer,
        backend=fs_backend,
        name="novel_supervisor",
    )

    logger.info(
        f"小说创作 Deep Agent 已创建（工具 {len(all_tools)} 个，子智能体 {len(sub_agents)} 个）"
    )
    return agent

