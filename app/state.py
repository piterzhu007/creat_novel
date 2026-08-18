"""
LangGraph 全局共享状态中枢（NovelState）。

设计原则（单一大脑中枢）：
- 所有与创作任务相关的数据都以标准化格式存储在这里
- 世界观设定、人物档案、大纲、章节正文、进度、校验记录等
- 工作流中的所有节点（architect/writer/editor/reader 等）共享同一份状态
- 每个节点可直接读取、写入、更新状态，实现无缝数据交接

状态持久化：
- 内存态（本文件）用于节点间流转
- 磁盘态（LongTermMemory/ShortTermMemory）用于跨会话持久化
- 节点函数负责「内存态 ↔ 磁盘态」的双向同步
"""

from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class NovelState(TypedDict, total=False):
    """小说创作全局共享状态"""

    # ─── 对话历史 ───────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ─── 小说标识 ───────────────────────────────────────
    novel_id: str
    novel_title: str
    novel_genre: str
    novel_synopsis: str
    target_chapters: int

    # ─── 共享数据中枢（权威数据源，节点间自动流转） ──────
    # 人物档案：由 architect 生成，writer/editor/reader 直接读取
    characters: list[dict]
    # 世界观设定：由 architect 生成，所有节点共享
    world_settings: list[dict]
    # 大纲：由 architect 生成，writer 据此创作
    outlines: list[dict]
    # 历史写作问题：editor/reader 记录，writer 规避
    writing_issues: list[dict]

    # ─── 当前进度 ───────────────────────────────────────
    current_chapter: int
    latest_chapter_content: str          # 当前章节最新正文
    previous_chapter_ending: str         # 上一章结尾（用于衔接）

    # ─── 各节点产出 ─────────────────────────────────────
    architect_output: str                # architect 的设计产出
    writer_output: str                   # writer 的正文产出
    editor_report: dict                  # editor 的审核报告
    reader_report: dict                  # reader 的一致性报告
    supervisor_decision: str             # supervisor 的决策（next 节点名）
    supervisor_reason: str               # supervisor 的决策理由
    # 反馈闭环：editor/reader 的具体意见，传回给 writer 用于重写
    editor_feedback: str                 # editor 的修改意见（供 writer 重写参考）
    reader_feedback: str                 # reader 的矛盾点（供 writer 修正参考）

    # ─── 流程控制 ───────────────────────────────────────
    phase: str                           # design / writing / done
    needs_user_input: bool
    user_message: str
    error: Optional[str]
    iteration_count: int


class NovelCreationInput(TypedDict, total=False):
    """用户初始输入"""
    title: str
    genre: str
    synopsis: str
    requirements: str
    target_chapters: int
