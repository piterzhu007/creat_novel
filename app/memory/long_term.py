"""
长期记忆模块：SQLite 持久化 + ChromaDB 向量检索。

管理小说主线内容：人物、世界观设定、大纲、情节弧等。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_db_path
from app.models.novel import (
    Base,
    Character,
    ChapterDraft,
    MainPlot,
    Novel,
    Outline,
    WorldSetting,
    WritingIssue,
)
from app.models.memory import CharacterProfile, MemoryEntry, MemoryType, NovelContext, OutlineEntry, WorldSettingProfile


class LongTermMemory:
    """
    长期记忆管理器。

    - SQLite 存储结构化数据（人物、设定、大纲、情节）
    - ChromaDB 提供向量语义检索（由 VectorStore 封装）
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(get_db_path())
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._ensure_tables()
        logger.info(f"长期记忆已初始化: {db_path}")

    def _ensure_tables(self):
        """确保所有表已创建"""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ═══════════════════════════════════════════════════════
    # 小说 CRUD
    # ═══════════════════════════════════════════════════════

    def create_novel(self, title: str, genre: str = "", synopsis: str = "",
                     target_chapters: int = 0) -> str:
        """创建新小说，返回 novel_id"""
        novel_id = f"novel_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            novel = Novel(
                novel_id=novel_id,
                title=title,
                genre=genre,
                synopsis=synopsis,
                target_chapters=target_chapters,
            )
            session.add(novel)
            session.commit()
        logger.info(f"小说已创建: {title} (id={novel_id})")
        return novel_id

    def get_novel(self, novel_id: str) -> Optional[Novel]:
        """获取小说信息"""
        with self.get_session() as session:
            return session.query(Novel).filter(Novel.novel_id == novel_id).first()

    def list_novels(self) -> list[Novel]:
        """列出所有小说"""
        with self.get_session() as session:
            return session.query(Novel).order_by(Novel.updated_at.desc()).all()

    def update_novel_status(self, novel_id: str, status: str):
        """更新小说状态"""
        with self.get_session() as session:
            novel = session.query(Novel).filter(Novel.novel_id == novel_id).first()
            if novel:
                novel.status = status
                novel.updated_at = datetime.utcnow()
                session.commit()

    # ═══════════════════════════════════════════════════════
    # 人物 CRUD
    # ═══════════════════════════════════════════════════════

    def save_character(self, novel_id: str, profile: CharacterProfile) -> str:
        """保存人物档案"""
        char_id = profile.char_id or f"char_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            existing = session.query(Character).filter(Character.char_id == char_id).first()
            if existing:
                # 更新
                for key, value in profile.model_dump(exclude={"char_id"}).items():
                    setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                char = Character(char_id=char_id, novel_id=novel_id, **profile.model_dump(exclude={"char_id"}))
                session.add(char)
            session.commit()
        return char_id

    def get_characters(self, novel_id: str) -> list[CharacterProfile]:
        """获取小说的所有人物"""
        with self.get_session() as session:
            chars = session.query(Character).filter(Character.novel_id == novel_id).all()
        return [
            CharacterProfile(
                char_id=c.char_id, name=c.name, role_type=c.role_type,
                gender=c.gender, age=c.age, appearance=c.appearance,
                personality=c.personality, background=c.background,
                motivation=c.motivation, abilities=c.abilities,
                relationships=c.relationships, arc_summary=c.arc_summary,
            )
            for c in chars
        ]

    def get_character(self, char_id: str) -> Optional[CharacterProfile]:
        """获取单个人物"""
        with self.get_session() as session:
            c = session.query(Character).filter(Character.char_id == char_id).first()
        if not c:
            return None
        return CharacterProfile(
            char_id=c.char_id, name=c.name, role_type=c.role_type,
            gender=c.gender, age=c.age, appearance=c.appearance,
            personality=c.personality, background=c.background,
            motivation=c.motivation, abilities=c.abilities,
            relationships=c.relationships, arc_summary=c.arc_summary,
        )

    # ═══════════════════════════════════════════════════════
    # 世界观设定 CRUD
    # ═══════════════════════════════════════════════════════

    def save_world_setting(self, novel_id: str, profile: WorldSettingProfile) -> str:
        """保存世界观设定"""
        sid = profile.setting_id or f"setting_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            existing = session.query(WorldSetting).filter(WorldSetting.setting_id == sid).first()
            if existing:
                existing.category = profile.category
                existing.name = profile.name
                existing.description = profile.description
                existing.details = profile.details
            else:
                setting = WorldSetting(
                    setting_id=sid, novel_id=novel_id, category=profile.category,
                    name=profile.name, description=profile.description,
                    details=profile.details,
                )
                session.add(setting)
            session.commit()
        return sid

    def get_world_settings(self, novel_id: str) -> list[WorldSettingProfile]:
        """获取小说的世界观设定"""
        with self.get_session() as session:
            settings = session.query(WorldSetting).filter(WorldSetting.novel_id == novel_id).all()
        return [
            WorldSettingProfile(
                setting_id=s.setting_id, category=s.category or "",
                name=s.name, description=s.description or "", details=s.details or "",
            )
            for s in settings
        ]

    # ═══════════════════════════════════════════════════════
    # 大纲 CRUD
    # ═══════════════════════════════════════════════════════

    def save_outline(self, novel_id: str, entry: OutlineEntry) -> str:
        """保存大纲条目"""
        oid = entry.outline_id or f"outline_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            existing = session.query(Outline).filter(Outline.outline_id == oid).first()
            if existing:
                existing.title = entry.title
                existing.summary = entry.summary
                existing.key_events = entry.key_events
                existing.foreshadowing = entry.foreshadowing
                existing.status = entry.status
            else:
                outline = Outline(
                    outline_id=oid, novel_id=novel_id, chapter_seq=entry.chapter_seq,
                    volume=entry.volume, title=entry.title, summary=entry.summary,
                    key_events=entry.key_events, foreshadowing=entry.foreshadowing,
                    status=entry.status,
                )
                session.add(outline)
            session.commit()
        return oid

    def save_outlines_batch(self, novel_id: str, entries: list[OutlineEntry]) -> list[str]:
        """批量保存大纲"""
        return [self.save_outline(novel_id, entry) for entry in entries]

    def get_outlines(self, novel_id: str) -> list[OutlineEntry]:
        """获取小说大纲（按章节序排序）"""
        with self.get_session() as session:
            outlines = (
                session.query(Outline)
                .filter(Outline.novel_id == novel_id)
                .order_by(Outline.chapter_seq)
                .all()
            )
        return [
            OutlineEntry(
                outline_id=o.outline_id, chapter_seq=o.chapter_seq,
                volume=o.volume, title=o.title, summary=o.summary or "",
                key_events=o.key_events or "", foreshadowing=o.foreshadowing or "",
                status=o.status,
            )
            for o in outlines
        ]

    # ═══════════════════════════════════════════════════════
    # 主线情节 CRUD
    # ═══════════════════════════════════════════════════════

    def save_main_plot(self, novel_id: str, arc_name: str, description: str,
                       start_chapter: int = 0, end_chapter: int = 0) -> str:
        """保存主线情节"""
        plot_id = f"plot_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            plot = MainPlot(
                plot_id=plot_id, novel_id=novel_id, arc_name=arc_name,
                description=description, start_chapter=start_chapter,
                end_chapter=end_chapter,
            )
            session.add(plot)
            session.commit()
        return plot_id

    def get_main_plots(self, novel_id: str) -> list[dict]:
        """获取主线情节"""
        with self.get_session() as session:
            plots = session.query(MainPlot).filter(MainPlot.novel_id == novel_id).all()
        return [
            {"plot_id": p.plot_id, "arc_name": p.arc_name, "description": p.description,
             "start_chapter": p.start_chapter, "end_chapter": p.end_chapter, "status": p.status}
            for p in plots
        ]

    # ═══════════════════════════════════════════════════════
    # 小说上下文组装
    # ═══════════════════════════════════════════════════════

    def get_novel_context(self, novel_id: str, current_chapter: int = 1) -> Optional[NovelContext]:
        """获取小说完整上下文（传给智能体）"""
        novel = self.get_novel(novel_id)
        if not novel:
            return None

        # 获取前一章摘要
        prev_summary = ""
        if current_chapter > 1:
            with self.get_session() as session:
                prev = (
                    session.query(Outline)
                    .filter(Outline.novel_id == novel_id, Outline.chapter_seq == current_chapter - 1)
                    .first()
                )
                if prev:
                    prev_summary = prev.summary or ""

        return NovelContext(
            novel_id=novel.novel_id,
            title=novel.title,
            genre=novel.genre or "",
            synopsis=novel.synopsis or "",
            characters=self.get_characters(novel_id),
            world_settings=self.get_world_settings(novel_id),
            outlines=self.get_outlines(novel_id),
            current_chapter=current_chapter,
            previous_chapter_summary=prev_summary,
            active_sub_plots=[],  # 短期记忆模块负责
        )

    def search_semantic(self, novel_id: str, query: str, category: str = "all",
                         k: int = 5) -> list[MemoryEntry]:
        """
        语义检索接口 —— 由 VectorStore 实现。
        此处作为 fallback 提供简单的关键词匹配。
        """
        # 降级方案: 简单的文本包含搜索
        results = []
        if category in ("all", "character"):
            chars = self.get_characters(novel_id)
            for c in chars:
                if query.lower() in f"{c.name}{c.personality}{c.background}{c.appearance}".lower():
                    results.append(MemoryEntry(
                        entry_id=c.char_id, memory_type=MemoryType.LONG_TERM,
                        novel_id=novel_id, category="character",
                        content=f"{c.name}: {c.personality}\n{c.background}",
                        metadata={"char_id": c.char_id},
                    ))
        if category in ("all", "setting"):
            settings = self.get_world_settings(novel_id)
            for s in settings:
                if query.lower() in f"{s.name}{s.description}".lower():
                    results.append(MemoryEntry(
                        entry_id=s.setting_id, memory_type=MemoryType.LONG_TERM,
                        novel_id=novel_id, category="setting",
                        content=f"{s.name}: {s.description}",
                        metadata={"setting_id": s.setting_id},
                    ))
        return results[:k]

    # ═══════════════════════════════════════════════════════
    # 写作问题库 CRUD（editor/reader 记录，writer 规避）
    # ═══════════════════════════════════════════════════════

    def save_writing_issue(self, novel_id: str, issue_type: str,
                           description: str, suggestion: str = "",
                           chapter_seq: int = 0, found_by: str = "",
                           severity: str = "medium") -> str:
        """保存一条写作问题到长期记忆"""
        issue_id = f"issue_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            issue = WritingIssue(
                issue_id=issue_id, novel_id=novel_id, issue_type=issue_type,
                chapter_seq=chapter_seq, description=description,
                suggestion=suggestion, found_by=found_by, severity=severity,
            )
            session.add(issue)
            session.commit()
        logger.info(f"写作问题已记录: [{issue_type}] {description[:40]}...")
        return issue_id

    def get_writing_issues(self, novel_id: str, status: str = "open",
                           limit: int = 50) -> list[dict]:
        """获取小说的写作问题列表（默认未解决的）"""
        with self.get_session() as session:
            query = session.query(WritingIssue).filter(
                WritingIssue.novel_id == novel_id
            )
            if status:
                query = query.filter(WritingIssue.status == status)
            issues = query.order_by(WritingIssue.created_at.desc()).limit(limit).all()
        return [
            {
                "issue_id": i.issue_id,
                "issue_type": i.issue_type,
                "chapter_seq": i.chapter_seq,
                "description": i.description,
                "suggestion": i.suggestion,
                "found_by": i.found_by,
                "severity": i.severity,
                "status": i.status,
            }
            for i in issues
        ]

    def resolve_writing_issue(self, issue_id: str) -> bool:
        """标记写作问题为已解决"""
        with self.get_session() as session:
            issue = session.query(WritingIssue).filter(
                WritingIssue.issue_id == issue_id
            ).first()
            if issue:
                issue.status = "resolved"
                session.commit()
                return True
        return False

    # ═══════════════════════════════════════════════════════
    # 故事圣经（Story Bible）—— 精简权威设定卡
    # ═══════════════════════════════════════════════════════

    def get_story_bible(self, novel_id: str) -> dict:
        """
        获取小说的「故事圣经」——一份精简的权威设定卡。

        这是所有智能体创作前必读的锚点，用于保证：
        1. 角色名、身份、关系的绝对统一（避免记忆混乱）
        2. 世界观核心规则的权威来源
        3. 主线与当前进度的快速对齐

        返回 dict，由工具层格式化为文本。
        """
        novel = self.get_novel(novel_id)
        if not novel:
            return {}

        characters = self.get_characters(novel_id)
        settings = self.get_world_settings(novel_id)
        outlines = self.get_outlines(novel_id)

        # 角色卡：只保留 姓名 / 角色类型 / 一句话核心身份，不塞长文
        char_cards = []
        for c in characters:
            # 角色类型映射为中文
            role_map = {
                "protagonist": "主角",
                "antagonist": "反派",
                "supporting": "配角",
                "love_interest": "恋人",
            }
            role = role_map.get(c.role_type, c.role_type)
            # 一句话身份：优先用 motivation（动机），其次用背景前 30 字
            identity = (c.motivation or "").strip()
            if not identity:
                identity = (c.background or "").strip()[:30]
            if not identity:
                identity = (c.personality or "").strip()[:30]
            char_cards.append(f"{c.name}（{role}）：{identity}")

        # 世界观卡：只保留 名称 + 一句话
        setting_cards = [
            f"{s.name}：{(s.description or '').strip()[:60]}"
            for s in settings
        ]

        # 主线卡：用小说简介 + 大纲首尾
        outline_summary = ""
        if outlines:
            first = outlines[0]
            last = outlines[-1]
            outline_summary = (
                f"共{len(outlines)}章，从「{first.title}」到「{last.title}」"
            )

        return {
            "novel_id": novel_id,
            "title": novel.title,
            "genre": novel.genre or "",
            "synopsis": novel.synopsis or "",
            "characters": char_cards,
            "world_settings": setting_cards,
            "outline_summary": outline_summary,
        }
