"""
短期记忆模块：管理当前创作会话中的子情节、章节草稿、智能体操作日志。

- SQLite 持久化，进程退出后数据保留
- 上下文窗口管理，避免超长历史
"""

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import PROJECT_ROOT as _PROJECT_ROOT
from app.models.novel import AgentLog, Base, ChapterDraft, SubPlot


class ShortTermMemory:
    """
    短期记忆管理器。

    存储内容：
    - 子情节/支线 (sub_plots)
    - 章节草稿 (chapter_drafts)
    - 智能体操作日志 (agent_logs)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(_PROJECT_ROOT / "data" / "short_term.db")
        self.db_path = db_path

        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"短期记忆已初始化: {db_path}")

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ═══════════════════════════════════════════════════════
    # 子情节 CRUD
    # ═══════════════════════════════════════════════════════

    def add_sub_plot(self, novel_id: str, chapter_id: str, content: str) -> str:
        """添加子情节"""
        sp_id = f"sp_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            sp = SubPlot(sub_plot_id=sp_id, novel_id=novel_id,
                         chapter_id=chapter_id, content=content)
            session.add(sp)
            session.commit()
        return sp_id

    def get_active_sub_plots(self, novel_id: str) -> list[dict]:
        """获取活跃的子情节"""
        with self.get_session() as session:
            plots = session.query(SubPlot).filter(
                SubPlot.novel_id == novel_id,
                SubPlot.status == "active",
            ).all()
        return [
            {"sub_plot_id": p.sub_plot_id, "chapter_id": p.chapter_id,
             "content": p.content, "created_at": str(p.created_at)}
            for p in plots
        ]

    def resolve_sub_plot(self, sub_plot_id: str):
        """标记子情节为已解决"""
        with self.get_session() as session:
            sp = session.query(SubPlot).filter(SubPlot.sub_plot_id == sub_plot_id).first()
            if sp:
                sp.status = "resolved"
                sp.resolved_at = str(datetime.utcnow())
                session.commit()

    # ═══════════════════════════════════════════════════════
    # 章节草稿 CRUD
    # ═══════════════════════════════════════════════════════

    def save_draft(self, novel_id: str, chapter_seq: int, content: str,
                   title: str = "", feedback: str = "",
                   quality_score: Optional[float] = None) -> str:
        """保存/更新章节草稿"""
        with self.get_session() as session:
            # 查找该章节的最新版本
            latest = (
                session.query(ChapterDraft)
                .filter(
                    ChapterDraft.novel_id == novel_id,
                    ChapterDraft.chapter_seq == chapter_seq,
                )
                .order_by(desc(ChapterDraft.version))
                .first()
            )
            version = (latest.version + 1) if latest else 1
            draft_id = f"draft_{uuid.uuid4().hex[:12]}"
            draft = ChapterDraft(
                draft_id=draft_id, novel_id=novel_id, chapter_seq=chapter_seq,
                version=version, title=title, content=content,
                word_count=len(content), feedback=feedback,
                quality_score=quality_score, status="drafting",
            )
            session.add(draft)
            session.commit()
        logger.info(f"草稿已保存: 第{chapter_seq}章 v{version} ({len(content)}字)")
        return draft_id

    def get_latest_draft(self, novel_id: str, chapter_seq: int) -> Optional[ChapterDraft]:
        """获取章节最新草稿"""
        with self.get_session() as session:
            return (
                session.query(ChapterDraft)
                .filter(
                    ChapterDraft.novel_id == novel_id,
                    ChapterDraft.chapter_seq == chapter_seq,
                )
                .order_by(desc(ChapterDraft.version))
                .first()
            )

    def get_draft_history(self, novel_id: str, chapter_seq: int) -> list[ChapterDraft]:
        """获取章节课稿版本历史"""
        with self.get_session() as session:
            return (
                session.query(ChapterDraft)
                .filter(
                    ChapterDraft.novel_id == novel_id,
                    ChapterDraft.chapter_seq == chapter_seq,
                )
                .order_by(ChapterDraft.version)
                .all()
            )

    # ═══════════════════════════════════════════════════════
    # 智能体日志 CRUD
    # ═══════════════════════════════════════════════════════

    def log_agent_action(self, session_id: str, agent_name: str, action: str,
                         input_summary: str = "", output_summary: str = "",
                         tokens_used: int = 0, success: bool = True,
                         error_msg: str = "", novel_id: str = "") -> str:
        """记录智能体操作"""
        log_id = f"log_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            log = AgentLog(
                log_id=log_id, session_id=session_id, novel_id=novel_id,
                agent_name=agent_name, action=action,
                input_summary=input_summary, output_summary=output_summary,
                tokens_used=tokens_used, success=success, error_msg=error_msg,
            )
            session.add(log)
            session.commit()
        return log_id

    def get_recent_logs(self, session_id: str, limit: int = 20) -> list[dict]:
        """获取最近的操作日志"""
        with self.get_session() as session:
            logs = (
                session.query(AgentLog)
                .filter(AgentLog.session_id == session_id)
                .order_by(desc(AgentLog.timestamp))
                .limit(limit)
                .all()
            )
        return [
            {"agent": l.agent_name, "action": l.action,
             "success": l.success, "timestamp": str(l.timestamp),
             "output_summary": l.output_summary[:200]}
            for l in reversed(logs)
        ]

    # ═══════════════════════════════════════════════════════
    # 章节导出
    # ═══════════════════════════════════════════════════════

    def export_chapters(self, novel_id: str, output_dir: str = "") -> list[dict]:
        """
        导出指定小说的所有章节（每章取最新版本）到 .txt 文件。

        参数:
            novel_id: 小说ID
            output_dir: 输出目录（默认 data/../output，即项目根目录下的 output）

        返回:
            导出的文件信息列表 [{chapter_seq, title, filepath, word_count}]
        """
        from pathlib import Path
        from app.core.config import PROJECT_ROOT

        if not output_dir:
            output_dir = str(PROJECT_ROOT / "output")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 获取该小说所有章节，按章节序号分组，取每章最新版本
        with self.get_session() as session:
            drafts = (
                session.query(ChapterDraft)
                .filter(ChapterDraft.novel_id == novel_id)
                .order_by(ChapterDraft.chapter_seq, ChapterDraft.version)
                .all()
            )

        # 每章取最新版本
        latest = {}
        for d in drafts:
            latest[d.chapter_seq] = d

        exported = []
        for seq in sorted(latest.keys()):
            d = latest[seq]
            # 文件名：章节序号 + 章节名
            safe_title = (d.title or f"第{seq}章").strip()
            # 清理文件名中的非法字符
            for ch in r'\/:*?"<>|':
                safe_title = safe_title.replace(ch, "_")
            filename = f"{seq:03d}_{safe_title}.txt"
            filepath = out_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"第{seq}章 {d.title}\n\n")
                f.write(d.content or "")

            exported.append({
                "chapter_seq": seq,
                "title": d.title or "",
                "filepath": str(filepath),
                "word_count": len(d.content or ""),
            })
            logger.info(f"章节已导出: {filename} ({len(d.content or '')}字)")

        return exported

    def export_single_chapter(self, novel_id: str, chapter_seq: int,
                              output_dir: str = "") -> Optional[dict]:
        """
        导出单章终稿到 .txt 文件（命名格式「章节数+章名」）。

        参数:
            novel_id: 小说ID
            chapter_seq: 章节序号
            output_dir: 输出目录（默认项目根目录下的 output）

        返回:
            {chapter_seq, title, filepath, word_count} 或 None（该章无草稿）
        """
        from pathlib import Path
        from app.core.config import PROJECT_ROOT

        if not output_dir:
            output_dir = str(PROJECT_ROOT / "output")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        draft = self.get_latest_draft(novel_id, chapter_seq)
        if draft is None:
            return None

        safe_title = (draft.title or f"第{chapter_seq}章").strip()
        for ch in r'\/:*?"<>|':
            safe_title = safe_title.replace(ch, "_")
        # 命名格式：章节数 + 章名
        filename = f"{chapter_seq:03d}_{safe_title}.txt"
        filepath = out_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"第{chapter_seq}章 {draft.title}\n\n")
            f.write(draft.content or "")

        logger.info(f"章节终稿已导出: {filename} ({len(draft.content or '')}字)")
        return {
            "chapter_seq": chapter_seq,
            "title": draft.title or "",
            "filepath": str(filepath),
            "word_count": len(draft.content or ""),
        }
