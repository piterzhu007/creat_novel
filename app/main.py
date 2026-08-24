"""
小说创作智能体系统 — 主入口

使用方法:
    python -m app.main           # 交互模式
    python -m app.main --help    # 查看帮助
"""

from app.core.async_runtime import run as _async_run
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from loguru import logger

# ─── 环境加载（必须在其他导入之前） ───────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, encoding="utf-8")
else:
    load_dotenv(encoding="utf-8")

# ─── 修复 Windows 控制台编码 ─────────────────────────
# 避免 GBK 控制台无法编码 emoji/中文导致的 UnicodeEncodeError
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 流被重定向或非 TextIOWrapper 时忽略

# ─── 应用导入 ─────────────────────────────────────────

from app.workflow import create_novel_workflow
from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import setup_logging
from app.utils.file_io import (
    read_document,
    resolve_document_path,
    write_text,
)
from app.utils.text_processing import clean_text


def print_banner():
    """打印启动横幅"""
    banner = r"""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║      小说创作智能体系统 (Novel Creation System)       ║
║                                                      ║
║  架构: LangGraph StateGraph                           ║
║  模型: deepseek-v4-pro (max_tokens=200000)            ║
║  记忆: SQLite + ChromaDB (本地持久化)                 ║
║                                                      ║
║  智能体节点（共享状态中枢）:                           ║
║     Architect  — 大纲设计师 (t=1.5)                   ║
║     Writer     — 章节撰写者 (t=0.8)                   ║
║     Editor     — 出版社编辑 (t=0.2)                   ║
║     Reader     — 资深读者 (t=0.5)                     ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""
    print(banner)


def get_help_text() -> str:
    """帮助信息"""
    return """
📖 使用指南:

【创建阶段】
  创建小说    — 开始一个新的小说项目
  新项目      — 同上

【设计阶段】
  设计大纲    — 让大纲设计师制定小说大纲
  设计人物    — 让大纲设计师设计人物设定
  设定世界观  — 让大纲设计师构建世界观
  确认设定    — 保存当前设定到长期记忆

【写作阶段】
  开始写作    — 从第一章开始创作
  写第N章     — 撰写指定章节
  继续写      — 继续写下一章
  重写第N章   — 根据反馈重写指定章节

【审核阶段】
  审核        — 让出版社编辑审核当前章节
  检查        — 让资深读者检查设定一致性

【管理】
  查看大纲    — 显示当前小说的大纲
  查看人物    — 显示人物设定
  查看设定    — 显示世界观设定
  项目列表    — 列出所有小说项目
  切换项目    — 切换到另一个小说
  状态        — 查看当前工作状态

【系统】
  帮助        — 显示此帮助信息
  退出        — 退出系统
  新会话      — 开始一个全新的对话（清空当前上下文）
  历史会话    — 列出历史会话，可切换回之前的对话
  自动创作    — 输入创作要求后自动持续推进，直到任务完成（无需手动逐句续跑）
  续写        — 开干净新会话 + 从记忆库恢复进度继续写（省 token 的断点续传）

📎 文档输入:
  直接输入文档路径（如 output/大纲.txt、./设定.md）即可读取其内容作为输入。
  也可用 @路径 语法显式指定，例如: @./需求.txt 帮我创作这部小说

💡 自动创作示例:
  自动创作 @D:\Study Test\wangwen_creat\世界观与五行战力系统设定.txt ... 帮我生成600万字小说

💡 省 token 建议:
  每写一批章节后，用「续写」命令开新会话继续，而不是一直续接旧会话——
  新会话上下文干净、不重放历史正文，token 从零开始算，进度靠记忆库恢复。
"""


class NovelCreationCLI:
    """
    交互式 CLI 主循环。

    使用 DeepAgents 框架管理小说创作全流程。
    Supervisor 主智能体通过 task() 工具委派任务给子智能体。

    会话持久化：
    - session_id 保存在 data/ 目录下的 session_id.txt 中
    - 重启后自动续接上次的会话（通过 checkpointer 恢复对话历史）
    - 可用「新会话」命令开始全新对话，「历史会话」命令切换回旧对话
    """

    SESSION_FILE = PROJECT_ROOT / "data" / "session_id.txt"

    def __init__(self):
        self.settings = get_settings()
        self.session_id = self._load_or_create_session()
        self.graph = None
        # 硬约束：大纲定稿后是否已交用户审核（审核通过前不进入写正文）
        self._outline_reviewed = False
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 会话管理 ───────────────────────────────────────

    def _load_or_create_session(self) -> str:
        """加载上次会话 ID，若不存在或已失效则创建新会话"""
        if self.SESSION_FILE.exists():
            try:
                saved = self.SESSION_FILE.read_text(encoding="utf-8").strip()
                if saved and self._session_exists(saved):
                    logger.info(f"已续接上次会话: {saved}")
                    return saved
            except Exception:
                pass
        return self._new_session()

    def _new_session(self) -> str:
        """创建新会话并持久化"""
        new_id = f"session_{uuid.uuid4().hex[:12]}"
        try:
            self.SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.SESSION_FILE.write_text(new_id, encoding="utf-8")
        except Exception:
            pass
        logger.info(f"已创建新会话: {new_id}")
        return new_id

    def _session_exists(self, session_id: str) -> bool:
        """检查指定会话在 checkpoints 库中是否有历史记录"""
        checkpoint_path = PROJECT_ROOT / "data" / "checkpoints.db"
        if not checkpoint_path.exists():
            return False
        try:
            import sqlite3
            conn = sqlite3.connect(str(checkpoint_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (session_id,),
            )
            count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False

    def _list_sessions(self) -> list[tuple[str, int]]:
        """列出所有历史会话及各自的 checkpoint 数"""
        checkpoint_path = PROJECT_ROOT / "data" / "checkpoints.db"
        if not checkpoint_path.exists():
            return []
        try:
            import sqlite3
            conn = sqlite3.connect(str(checkpoint_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
            )
            rows = [(r[0], r[1]) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.warning(f"读取会话列表失败: {e}")
            return []

    def init_workflow(self):
        """初始化 LangGraph 工作流"""
        print("正在初始化系统...")
        checkpoint_path = str(PROJECT_ROOT / "data" / "checkpoints.db")
        self.graph = create_novel_workflow(checkpoint_db_path=checkpoint_path)
        logger.info("系统初始化完成")

    def _resolve_input(self, raw_input: str) -> str:
        """
        解析用户输入：提取其中的文档路径，读取文件内容后注入。

        支持：
        1. 纯路径输入：整条输入就是一个文档路径
        2. @路径语法：@./需求.txt 帮我创作
        3. 多路径混合：输入中含多个 .txt/.md 路径（含 Windows 绝对路径），全部读取后注入
        """
        import re

        stripped = raw_input.strip()

        # 形式 1：@路径 语法（可选后跟附加说明）
        if stripped.startswith("@"):
            parts = stripped[1:].split(None, 1)
            path_str = parts[0].strip().strip('"').strip("'")
            extra = parts[1].strip() if len(parts) > 1 else ""
            try:
                doc_content = read_document(path_str)
            except (FileNotFoundError, ValueError) as e:
                print(f"\n⚠️ {e}")
                return raw_input
            if extra:
                return f"以下是我提供的文档内容（来自 {path_str}）:\n\n{doc_content}\n\n我的要求：{extra}"
            return f"以下是我提供的文档内容（来自 {path_str}）:\n\n{doc_content}"

        # 形式 2：从输入中提取所有文档路径（支持 Windows 绝对路径 + 相对路径）
        # 关键：用户粘贴的路径常粘连（如 "...设定.txtD:\...问题.txt"），
        # 用「盘符开头 + 文件后缀结尾」精确提取每个完整路径。
        found_files = []
        seen = set()

        # 2a. 先按盘符切分 Windows 绝对路径（处理粘连场景）
        if re.search(r'[A-Za-z]:[\\/]', stripped):
            # 用盘符作为分隔符切分，每个片段是一个完整路径的候选
            segments = re.split(r'(?=[A-Za-z]:[\\/])', stripped)
            for seg in segments:
                seg = seg.strip()
                m = re.match(r'([A-Za-z]:[\\/].*?\.(?:txt|md|markdown))', seg)
                if m:
                    p = resolve_document_path(m.group(1))
                    if p is not None and str(p) not in seen:
                        seen.add(str(p))
                        try:
                            found_files.append((str(p), read_document(str(p))))
                        except (FileNotFoundError, ValueError):
                            pass

        # 2b. 再匹配相对路径 / 残留的 .txt 路径
        for m in re.findall(r'([^\s"\'，。；：]+\.(?:txt|md|markdown))', stripped):
            p = resolve_document_path(m)
            if p is not None and str(p) not in seen:
                seen.add(str(p))
                try:
                    found_files.append((str(p), read_document(str(p))))
                except (FileNotFoundError, ValueError):
                    pass

        if found_files:
            # 去掉输入中已识别的路径部分，剩下的作为指令
            remaining = stripped
            for path, _ in found_files:
                remaining = remaining.replace(path, " ")
            remaining = remaining.strip()

            parts = []
            for path, content in found_files:
                parts.append(f"【文档】{path}\n{content}")
            doc_block = "\n\n".join(parts)

            if remaining:
                return f"以下是用户提供的文档内容：\n\n{doc_block}\n\n【用户要求】{remaining}"
            return f"以下是用户提供的文档内容：\n\n{doc_block}"

        # 形式 3：纯路径输入（整条就是一个文档路径）
        doc_path = resolve_document_path(stripped)
        if doc_path is not None:
            doc_content = read_document(str(doc_path))
            return f"以下是我提供的文档内容（来自 {doc_path}）:\n\n{doc_content}"

        return raw_input

    def _save_output(self, content: str) -> Path:
        """将智能体产出保存到 output/ 目录（.txt 格式，带时间戳）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}_{uuid.uuid4().hex[:6]}.txt"
        output_path = OUTPUT_DIR / filename
        write_text(str(output_path), content)
        logger.info(f"产出已保存: {output_path}")
        return output_path

    def _show_sessions(self, config: dict):
        """展示历史会话列表，允许用户切换"""
        sessions = self._list_sessions()
        if not sessions:
            print("\n暂无历史会话。\n")
            return

        print("\n📚 历史会话列表:")
        for i, (sid, cnt) in enumerate(sessions, 1):
            marker = " ← 当前" if sid == self.session_id else ""
            print(f"  [{i}] {sid}（{cnt} 条记录）{marker}")

        print("\n输入序号切换会话，或输入其他内容取消：")
        choice = clean_text(input("切换到 > ")).strip()
        if not choice.isdigit():
            print("已取消。\n")
            return

        idx = int(choice)
        if 1 <= idx <= len(sessions):
            self.session_id = sessions[idx - 1][0]
            config["configurable"]["thread_id"] = self.session_id
            # 持久化当前会话
            try:
                self.SESSION_FILE.write_text(self.session_id, encoding="utf-8")
            except Exception:
                pass
            print(f"\n✅ 已切换到会话: {self.session_id}\n")
        else:
            print("无效的序号。\n")

    def run(self):
        """运行主循环"""
        print_banner()
        self.init_workflow()

        print("\n输入 '帮助' 查看命令列表，输入 '退出' 结束程序。")
        print("支持文档输入：直接输入 .txt/.md 等文档路径即可读取内容。\n")

        # 工作流配置 (thread_id 用于 checkpoint 会话隔离)
        config = {"configurable": {"thread_id": self.session_id}}

        while True:
            try:
                user_input = clean_text(input("📝 > ")).strip()
                if not user_input:
                    continue

                # 系统命令
                if user_input in ("退出", "exit", "quit"):
                    print("感谢使用，再见！")
                    break
                if user_input in ("帮助", "help"):
                    print(get_help_text())
                    continue
                if user_input in ("新会话", "new", "new session"):
                    self.session_id = self._new_session()
                    config = {"configurable": {"thread_id": self.session_id}}
                    print(f"\n✅ 已开启新会话: {self.session_id}")
                    print("现在开始全新的对话。\n")
                    continue
                if user_input in ("历史会话", "history", "sessions"):
                    self._show_sessions(config)
                    continue
                # 「续写」：开一个干净的新会话，让 supervisor 从记忆库恢复进度继续写，
                # 避免在旧会话上无限续接导致 token 越滚越大（断点续传的正确省钱姿势）。
                if user_input in ("续写", "继续写", "resume"):
                    self.session_id = self._new_session()
                    config = {"configurable": {"thread_id": self.session_id}}
                    print("\n📖 已开启续写会话（上下文干净，从记忆库恢复进度）。")
                    print("   supervisor 会先 list_novels + get_novel_state 恢复进度，再接着写。\n")
                    try:
                        _async_run(self._auto_advance(
                            "继续写之前的小说。请先调用 list_novels 找到已有小说，再用 "
                            "get_novel_state 恢复进度（当前写到第几章）、大纲和人物名表，"
                            "然后从下一章接着写。不要重新设计，不要重读源文档。",
                            config,
                        ))
                    except KeyboardInterrupt:
                        print("\n\n⏸️ 续写已暂停。输入「续写」继续。")
                    continue

                # 解析输入（可能包含文档路径）
                resolved_input = self._resolve_input(user_input)

                # 「自动创作」：持续自动推进直到完成（不靠手动逐句续跑）
                if user_input in ("自动创作", "auto", "自动生成"):
                    print("\n🚀 已开启自动创作模式，将持续推进直到任务完成。")
                    print("（可按 Ctrl+C 中断）\n")
                    try:
                        _async_run(self._auto_advance(resolved_input, config))
                    except KeyboardInterrupt:
                        print("\n\n⏸️ 自动创作已暂停。输入「自动创作」继续，或输入其他指令。")
                    continue

                # 包装用户输入并流式送入 Deep Agent
                try:
                    _async_run(self._handle_stream(resolved_input, config))
                except Exception as e:
                    logger.error(f"工作流执行出错: {e}")
                    print(f"\n⚠️ 处理出错: {e}")
                    print("请重试或输入其他指令。")

            except KeyboardInterrupt:
                print("\n\n已中断。输入 '退出' 结束程序。")
            except EOFError:
                print("\n感谢使用，再见！")
                break

    async def _handle_stream(self, user_content: str, config: dict, save_output: bool = True):
        """
        流式处理 deep agent 的执行结果。

        使用 stream_mode="messages" 逐 token 打印主智能体的回复，
        让用户实时看到进度。同时累积完整内容用于保存产出。

        注意：MCP 工具是 async-only 的（langchain_mcp_adapters 转换的工具只实现
        _arun），因此必须用 astream 而非同步 stream，否则报
        「StructuredTool does not support sync invocation」。

        参数:
            user_content: 用户输入内容
            config: 工作流配置（含 thread_id）
            save_output: 是否把最终文本保存为 output 文件（自动推进的中间步骤传 False，
                         避免堆积大量无意义的 output_*.txt）
        """
        from langchain_core.messages import HumanMessage, AIMessageChunk
        from app.core.tracing import get_tracer

        print()  # 空行分隔

        accumulated = []  # 累积 AI 文本

        # 注入运行追踪器回调，捕获 token 消耗 / 工具调用等技术细节
        tracer = get_tracer()
        tracer.clear()  # 每次任务重新统计，避免累积上一轮的数据
        _config = dict(config)
        _config["callbacks"] = [tracer]

        try:
            async for chunk, _metadata in self.graph.astream(
                {"messages": [HumanMessage(content=user_content)]},
                _config,
                stream_mode="messages",
            ):
                # 只打印 AI 文本 token（跳过工具调用等内部消息）
                if isinstance(chunk, AIMessageChunk):
                    content = chunk.content
                    if content:
                        print(content, end="", flush=True)
                        accumulated.append(content)
        except Exception as e:
            logger.error(f"工作流执行出错: {e}")
            print(f"\n⚠️ 处理出错: {e}")
            return ""

        # 流式结束后换行
        print()

        # 保存完整产出
        full_content = "".join(accumulated).strip()
        if save_output and full_content:
            output_path = self._save_output(full_content)
            print(f"\n💾 产出已保存至: {output_path}")

        print()

        # 返回累积的完整文本，供调用方（如自动推进）检查完成标记
        return full_content

    def _read_current_chapter(self) -> int:
        """读取当前活跃小说的进度（current_chapter），用于判断是否跨过情节单元边界。

        直接读 data/novels.db（主进程与 MCP 子进程共享同一 SQLite 文件），
        取所有小说里最大的 current_chapter 作为「活跃小说」的进度。
        """
        try:
            import sqlite3
            conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "novels.db"))
            cur = conn.cursor()
            cur.execute("SELECT MAX(current_chapter) FROM novels")
            row = cur.fetchone()
            conn.close()
            return int(row[0]) if row and row[0] else 0
        except Exception:
            return 0

    def _outline_ready_for_review(self) -> bool:
        """判断「大纲已定稿、但正文尚未开始」，此时必须交用户审核（硬约束）。"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "novels.db"))
            cur = conn.cursor()
            cur.execute("SELECT MAX(current_chapter) FROM novels")
            chapter = cur.fetchone()[0] or 0
            if chapter > 0:
                conn.close()
                return False  # 正文已开始，大纲阶段已过
            cur.execute("SELECT COUNT(*) FROM outlines WHERE locked = 1")
            locked = cur.fetchone()[0] or 0
            conn.close()
            return locked > 0  # 有 locked 的大纲 = 已定稿、待审核
        except Exception:
            return False

    def _read_outline_for_review(self) -> str:
        """读取落库的全书大纲（locked 条目）全文，供用户审核。"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "novels.db"))
            cur = conn.cursor()
            cur.execute(
                "SELECT title, summary, key_events FROM outlines "
                "WHERE locked = 1 ORDER BY chapter_seq"
            )
            rows = cur.fetchall()
            conn.close()
            parts = []
            for title, summary, key_events in rows:
                parts.append(f"## {title}\n{summary or ''}")
                if key_events:
                    parts.append(f"关键事件：{key_events}")
            return "\n\n".join(parts) if parts else "（未找到已定稿的大纲）"
        except Exception as e:
            return f"（读取大纲失败：{e}）"

    # 自动推进的终止信号（supervisor 输出中出现这些词时判定任务已全部完成）
    _DONE_MARKERS = ("创作总结", "全部完成", "已全部", "全部导出", "全部章节", "全部写完")

    # 情节单元大小（章）：supervisor 每推进约一个单元（3~5 章）就重置一次上下文，
    # 让消息历史保持有界、避免 O(N²) token 爆炸。进度已落库，重置是安全的（可从 DB 恢复）。
    UNIT_CHAPTERS = 5

    async def _auto_advance(self, initial_input: str, config: dict, max_steps: int = 50):
        """
        自动推进循环：持续喂给 supervisor「继续」指令，直到任务全部完成。

        背景：supervisor 是 ReAct 智能体，单次 stream 只推进「一步」（例如创建项目、
        或设计大纲、或写一章）。若不自动续跑，600万字/2000章的任务每步都要手动输入，
        不可能完成。本方法在每轮结束后自动续一句「继续推进」，直到 supervisor 输出
        明确的完成总结，或达到 max_steps 上限。

        参数:
            initial_input: 用户最初的创作指令（仅第一步喂入）
            config: 工作流配置
            max_steps: 最大自动推进轮数（防止真死循环时无限烧 token）
        """
        # 第一步：喂入用户原始指令（仅首步保存 output，后续避免堆积）
        last_text = await self._handle_stream(initial_input, config, save_output=True)
        # 记录本次上下文的「锚点」进度：之后每推进约一个情节单元就重置一次 supervisor 上下文
        last_reset_chapter = self._read_current_chapter()

        # 后续步骤：自动续跑
        for step in range(1, max_steps + 1):
            if any(marker in (last_text or "") for marker in self._DONE_MARKERS):
                print("\n✅ 检测到创作完成标记，自动推进结束。")
                break

            # ── 硬约束：大纲定稿后、正文开始前，把落库的全书大纲交用户审核，通过前不进入写作 ──
            if not self._outline_reviewed and self._outline_ready_for_review():
                outline = self._read_outline_for_review()
                print("\n" + "=" * 60)
                print("  📋 【硬约束】全书大纲已定稿，请审核以下落库大纲")
                print("=" * 60)
                print(outline)
                print("=" * 60)
                approval = clean_text(
                    input("大纲审核 > 输入「通过」开始写正文，或输入修改意见：")
                ).strip()
                if approval == "通过":
                    self._outline_reviewed = True
                    print("\n✅ 大纲审核通过，开始写正文。\n")
                    try:
                        last_text = await self._handle_stream(
                            "大纲审核已通过，开始写正文，按大纲里的情节单元批量推进。",
                            config, save_output=False,
                        )
                    except Exception as e:
                        logger.error(f"开始写正文出错: {e}")
                        print(f"\n⚠️ 出错: {e}")
                        break
                else:
                    print(f"\n📝 已记录修改意见，交 supervisor 修订大纲并重新定稿。\n")
                    try:
                        last_text = await self._handle_stream(
                            f"用户对大纲的审核意见（硬约束，必须据此修订）：{approval}。"
                            "请修改大纲并重新定稿，再交回审核。",
                            config, save_output=False,
                        )
                    except Exception as e:
                        logger.error(f"修订大纲出错: {e}")
                        print(f"\n⚠️ 出错: {e}")
                        break
                continue

            # 按单元重置上下文：进度跨过一个情节单元（UNIT_CHAPTERS 章）后，
            # 重启 supervisor 消息历史（换新 thread_id），用 get_novel_progress 从记忆库恢复，
            # 让上下文保持有界、避免 O(N²) token 爆炸。进度已落库，重置安全。
            current_chapter = self._read_current_chapter()
            if current_chapter > 0 and current_chapter - last_reset_chapter >= self.UNIT_CHAPTERS:
                self.session_id = self._new_session()
                config["configurable"]["thread_id"] = self.session_id
                last_reset_chapter = current_chapter
                print(f"\n🔁 已完成约一个情节单元（第 {current_chapter} 章），重置 supervisor 上下文（进度已落库，从记忆库恢复）。\n")
                try:
                    last_text = await self._handle_stream(
                        "继续写之前的小说。请先调用 get_novel_progress 恢复进度"
                        "（当前写到第几章）、大纲和角色名表，再从下一单元接着写。"
                        "不要重新设计，不要重读源文档。",
                        config,
                        save_output=False,
                    )
                except Exception as e:
                    logger.error(f"重置后恢复出错: {e}")
                    print(f"\n⚠️ 恢复中断: {e}")
                    break
                continue

            print(f"\n{'=' * 60}")
            print(f"  🔄 自动推进第 {step} 步（最多 {max_steps} 步）")
            print(f"{'=' * 60}\n")
            try:
                last_text = await self._handle_stream(
                    "继续推进创作任务。若上一阶段已完成，就进入下一阶段；"
                    "若整部小说已全部完成并导出，请输出「创作总结」。",
                    config,
                    save_output=False,
                )
            except Exception as e:
                logger.error(f"自动推进出错: {e}")
                print(f"\n⚠️ 自动推进中断: {e}")
                break
        else:
            print(f"\n⚠️ 已达最大自动推进步数（{max_steps}），暂停。"
                  f"可再次输入「自动创作」继续，或手动输入「继续」推进。")


def main():
    """主函数"""
    setup_logging()

    try:
        cli = NovelCreationCLI()
        cli.run()
    except ConfigurationError as e:
        logger.error(f"配置错误: {e}")
        print(f"\n❌ 配置错误: {e}")
        print("请确保 .env 文件存在且包含有效的 DEEPSEEK_API_KEY。")
        print("参考 .env.example 文件进行配置。")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未预期的错误: {e}")
        sys.exit(1)
    finally:
        # 优雅关闭持久 event loop（MCP session / aiosqlite 连接绑定其上）
        from app.core.async_runtime import close as _close_loop
        _close_loop()


if __name__ == "__main__":
    main()
