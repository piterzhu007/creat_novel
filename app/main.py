"""
小说创作智能体系统 — 主入口

使用方法:
    python -m app.main           # 交互模式
    python -m app.main --help    # 查看帮助
"""

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

📎 文档输入:
  直接输入文档路径（如 output/大纲.txt、./设定.md）即可读取其内容作为输入。
  也可用 @路径 语法显式指定，例如: @./需求.txt 帮我创作这部小说
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

                # 解析输入（可能包含文档路径）
                resolved_input = self._resolve_input(user_input)

                # 包装用户输入并流式送入 Deep Agent
                try:
                    self._handle_stream(resolved_input, config)
                except Exception as e:
                    logger.error(f"工作流执行出错: {e}")
                    print(f"\n⚠️ 处理出错: {e}")
                    print("请重试或输入其他指令。")

            except KeyboardInterrupt:
                print("\n\n已中断。输入 '退出' 结束程序。")
            except EOFError:
                print("\n感谢使用，再见！")
                break

    def _handle_stream(self, user_content: str, config: dict):
        """
        处理 deep agent 的执行结果。

        deepagents 架构：supervisor 是主 agent，通过 task 委派子智能体，
        结果都在 messages 里。这里提取最后一条 AI 消息展示给用户。
        """
        from langchain_core.messages import HumanMessage, AIMessage

        print()  # 空行分隔

        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_content)]},
                config,
            )
        except Exception as e:
            logger.error(f"工作流执行出错: {e}")
            print(f"\n⚠️ 处理出错: {e}")
            return

        # 提取最后一条 AI 消息（supervisor 的最终回复）
        messages = result.get("messages", [])
        last_ai = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_ai = m
                break

        if last_ai and last_ai.content:
            content = last_ai.content
            print("=" * 60)
            print("📖 回复：")
            print("=" * 60)
            if len(content) > 3000:
                print(content[:3000])
                print(f"\n... (共 {len(content)} 字符，已截断)")
            else:
                print(content)

            # 保存产出
            output_path = self._save_output(content)
            print(f"\n💾 产出已保存至: {output_path}")

        print()


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


if __name__ == "__main__":
    main()
