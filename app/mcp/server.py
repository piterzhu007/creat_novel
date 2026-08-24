"""
真正的 MCP 服务器（FastMCP + stdio 传输）。

将所有记忆工具通过标准 MCP 协议暴露，任何智能体（及外部 MCP 客户端）
都可以通过 stdio 连接并使用全部工具，实现「单一大脑中枢 + 工具全连通」。

用法（作为独立 stdio 子进程被 agent 启动）：
    python -m app.mcp.server

agent 侧通过 langchain_mcp_adapters 的 MultiServerMCPClient 连接本服务器，
加载全部工具为 langchain 工具后分发给所有智能体。
"""

from mcp.server.fastmcp import FastMCP
import sys
from loguru import logger

# 关键：MCP stdio 协议要求 stdout 只能有 JSON-RPC 消息。
# loguru 默认输出到 stderr，但为确保万无一失，显式移除默认 handler 并绑定 stderr，
# 避免任何日志污染 stdout 导致 MCP 通信中断。
logger.remove()
logger.add(sys.stderr, level="INFO")

from app.memory import LongTermMemory, ShortTermMemory, VectorStore
from app.tools.factory import NovelMemoryTools

# ─── 惰性单例记忆后端（首次工具调用时才初始化，import 不建 DB） ───
# 目的：`import app.mcp.server` 不再触发真实 DB 初始化，避免测试/导入时污染 data/ 工作区。

_factory = None


def _get_factory():
    global _factory
    if _factory is None:
        _ltm = LongTermMemory()
        _stm = ShortTermMemory()
        _vs = VectorStore()
        _factory = NovelMemoryTools(_ltm, _stm, _vs)
    return _factory

# ─── FastMCP 实例 ────────────────────────────────────────

mcp = FastMCP(
    "wangwen-creat-memory",
    instructions="小说创作多智能体系统的全局共享记忆工具集（长期/短期/向量记忆 + 创作调度）",
)


# ═══════════════════════════════════════════════════════════
# 以下逐个注册工具。每个工具直接委托给 NovelMemoryTools 工厂方法，
# 保留原有的闭包逻辑，只是改用 MCP 协议暴露。
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def create_novel(title: str, genre: str = "", synopsis: str = "", target_chapters: int = 0,
                 source_dir: str = "") -> str:
    """创建一个新的小说项目，返回 novel_id。创作第一步必做。
    source_dir：该小说自己的源文档目录（开新项目传新目录实现多项目隔离；留空用项目根目录下 4 个源文档）。"""
    return _get_factory()._create_novel(title, genre, synopsis, target_chapters, source_dir)


@mcp.tool()
def list_novels() -> str:
    """列出所有已创建的小说项目。"""
    return _get_factory()._list_novels()


@mcp.tool()
def get_novel_state(novel_id: str) -> str:
    """获取小说的「全局状态中枢」完整快照（世界观/人物/大纲/进度/校验历史一次拿全）。"""
    return _get_factory()._get_novel_state(novel_id)


@mcp.tool()
def update_novel_progress(novel_id: str, current_chapter: int) -> str:
    """更新小说的生成进度（当前章节序号），持久化到记忆库。"""
    return _get_factory()._update_novel_progress(novel_id, current_chapter)


@mcp.tool()
def get_story_bible(novel_id: str) -> str:
    """获取小说的「故事圣经」——精简权威设定卡（角色名表/世界观核心/主线进度）。"""
    return _get_factory()._get_story_bible(novel_id)


@mcp.tool()
def get_novel_outline(novel_id: str, chapter_seq: int = 0) -> str:
    """获取小说大纲。chapter_seq>0 时只返回该章所在卷的大纲（省 token），0 返回全部。"""
    return _get_factory()._get_novel_outline(novel_id, chapter_seq)


@mcp.tool()
def get_character_profile(novel_id: str, char_name: str = "") -> str:
    """获取人物档案（精简版）。char_name 留空返回所有人物，指定则返回该人物。"""
    return _get_factory()._get_character_profile(novel_id, char_name)


@mcp.tool()
def get_world_building(novel_id: str) -> str:
    """获取世界观设定（精简版）。"""
    return _get_factory()._get_world_building(novel_id)


@mcp.tool()
def save_to_long_term(novel_id: str, category: str, name: str, content: str,
                      metadata: str = "{}", locked: bool = False, allow_new: bool = False) -> str:
    """保存内容到长期记忆（category: character/setting/plot/outline，自动同步向量库）。
    locked=True 表示 supervisor 定稿（可覆盖已锁定条目）；allow_new=True 表示 supervisor 授权新增名表外角色。"""
    return _get_factory()._save_to_long_term(novel_id, category, name, content, metadata, locked, allow_new)


@mcp.tool()
def get_writing_context(novel_id: str, chapter_seq: int = 1) -> str:
    """获取写作上下文快照：一次返回写某章所需的全部状态（故事圣经+本章大纲+上一章结尾+历史问题），减少多次零散查询。"""
    return _get_factory()._get_writing_context(novel_id, chapter_seq)


@mcp.tool()
def get_novel_progress(novel_id: str) -> str:
    """获取小说的极简进度卡（进度+角色名表+大纲目录+未解决问题数，几 KB）。supervisor 日常推进优先用它，避免 get_novel_state 全量快照撑爆上下文。"""
    return _get_factory()._get_novel_progress(novel_id)


@mcp.tool()
def lock_entry(novel_id: str, category: str, name: str) -> str:
    """定稿加锁：把草稿条目 locked=True，不改内容、不重发全文。supervisor 审核通过后用它定稿（category: character/setting/plot/outline）。"""
    return _get_factory()._lock_entry(novel_id, category, name)


@mcp.tool()
def search_long_term_memory(query: str, novel_id: str, category: str = "all", k: int = 5) -> str:
    """语义检索长期记忆（人物/设定/情节）。"""
    return _get_factory()._search_long_term_memory(query, novel_id, category, k)


@mcp.tool()
def get_short_term_context(novel_id: str, chapter_seq: int) -> str:
    """获取当前章节的短期上下文（子情节 + 上一章结尾，用于章节衔接）。"""
    return _get_factory()._get_short_term_context(novel_id, chapter_seq)


@mcp.tool()
def get_chapter(novel_id: str, chapter_seq: int) -> str:
    """读取指定章节的完整正文。editor/reader 审核章节时用它读全文，避免误判正文缺失。"""
    return _get_factory()._get_chapter(novel_id, chapter_seq)


@mcp.tool()
def save_chapter(novel_id: str, chapter_seq: int, content: str, title: str = "",
                 feedback: str = "", quality_score: float = 0.0) -> str:
    """保存完成的章节正文（必须传完整 content，供后续章节衔接和审核）。"""
    return _get_factory()._save_chapter(novel_id, chapter_seq, content, title, feedback, quality_score)


@mcp.tool()
def patch_chapter(novel_id: str, chapter_seq: int, old_text: str, new_text: str = "",
                  replace_all: bool = True) -> str:
    """精准替换章节正文中的文本（不返回全文）。修正错字/角色名/单句时用它，省 token。"""
    return _get_factory()._patch_chapter(novel_id, chapter_seq, old_text, new_text, replace_all)


@mcp.tool()
def update_short_term(novel_id: str, category: str, content: str, chapter_seq: int = 0) -> str:
    """更新短期记忆（category: sub_plot/draft/log）。"""
    return _get_factory()._update_short_term(novel_id, category, content, chapter_seq)


@mcp.tool()
def search_similar_content(query: str, novel_id: str = "", collection: str = "chapter_content", k: int = 5) -> str:
    """向量语义检索相似内容（跨章节按语义回忆细节）。collection 可选 chapter_content(章节正文)/novel_characters(人物)/novel_settings(设定)/novel_plots(情节)。"""
    return _get_factory()._search_similar_content(query, novel_id, collection, k)


@mcp.tool()
def save_writing_issue(novel_id: str, issue_type: str, description: str = "",
                       suggestion: str = "", chapter_seq: int = 0,
                       found_by: str = "", severity: str = "medium") -> str:
    """记录一条写作历史问题（issue_type: 连贯性/人物一致性/逻辑/文笔/世界观/节奏）。"""
    return _get_factory()._save_writing_issue(novel_id, issue_type, description, suggestion, chapter_seq, found_by, severity)


@mcp.tool()
def get_writing_issues(novel_id: str, status: str = "open", limit: int = 30) -> str:
    """获取小说的历史写作问题列表（writer 写作前应阅读以规避）。"""
    return _get_factory()._get_writing_issues(novel_id, status, limit)


@mcp.tool()
def export_chapters(novel_id: str, output_dir: str = "") -> str:
    """导出小说所有章节正文为 .txt 文件（每章一个文件）。"""
    return _get_factory()._export_chapters(novel_id, output_dir)


@mcp.tool()
def read_source_docs(novel_id: str) -> str:
    """读取某小说自己的源文档（世界观设定/小说提纲/问题与建议，多项目隔离）。architect 首次设计时的唯一依据。"""
    return _get_factory()._read_source_docs(novel_id)


@mcp.tool()
def delete_long_term_entry(novel_id: str, category: str, name: str) -> str:
    """删除长期记忆条目（category: character/setting/plot/outline），消除新旧并存冲突。"""
    return _get_factory()._delete_long_term_entry(novel_id, category, name)


@mcp.tool()
def save_run_log(novel_id: str, stage: str, summary: str, detail: str = "") -> str:
    """把本次阶段性任务的运行日志以 txt 保存到专门文件夹（output/run_logs/<novel_id>/）。"""
    return _get_factory()._save_run_log(novel_id, stage, summary, detail)


# ─── stdio 入口 ──────────────────────────────────────────

def main():
    """启动 MCP stdio 服务器。"""
    logger.info("MCP 记忆中枢启动（stdio），暴露全部记忆工具")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
