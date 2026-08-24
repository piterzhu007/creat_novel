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
    SourceDoc,
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
        """创建新小说，返回 novel_id（同名小说已存在则复用，避免重复从零设计）"""
        # 复用：同名小说已存在则直接返回已有 novel_id，
        # 让后续 architect 用 get_novel_state 接续上一轮已提炼的人物/世界观/大纲。
        with self.get_session() as session:
            existing = session.query(Novel).filter(Novel.title == title).first()
            if existing:
                logger.info(f"复用已有小说: {title} (id={existing.novel_id})")
                return existing.novel_id

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

    def update_novel_progress(self, novel_id: str, current_chapter: int):
        """更新小说生成进度（当前章节序号）"""
        with self.get_session() as session:
            novel = session.query(Novel).filter(Novel.novel_id == novel_id).first()
            if novel:
                novel.current_chapter = current_chapter
                novel.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"小说进度已更新: {novel.title} 第{current_chapter}章")

    def get_full_state(self, novel_id: str) -> dict:
        """
        获取小说的「全局状态中枢」完整快照。

        一次性返回世界观、人物、大纲、章节、进度、校验记录等所有
        与生成任务相关的重要数据，供任何智能体读取全局状态。

        返回 dict（标准化格式），各字段对工作流中所有智能体共享。
        """
        novel = self.get_novel(novel_id)
        if not novel:
            return {}

        state = {
            "novel_id": novel.novel_id,
            "title": novel.title,
            "genre": novel.genre or "",
            "synopsis": novel.synopsis or "",
            "status": novel.status,
            "target_chapters": novel.target_chapters,
            "current_chapter": novel.current_chapter,
            # 世界观设定
            "world_settings": [
                {
                    "setting_id": s.setting_id,
                    "category": s.category or "",
                    "name": s.name,
                    "description": s.description or "",
                }
                for s in self.get_world_settings(novel_id)
            ],
            # 人物档案
            "characters": [
                {
                    "char_id": c.char_id,
                    "name": c.name,
                    "role_type": c.role_type,
                    "personality": c.personality,
                    "motivation": c.motivation,
                }
                for c in self.get_characters(novel_id)
            ],
            # 大纲目录
            "outlines": [
                {
                    "outline_id": o.outline_id,
                    "chapter_seq": o.chapter_seq,
                    "title": o.title,
                    "summary": o.summary or "",
                    "status": o.status,
                }
                for o in self.get_outlines(novel_id)
            ],
            # 校验环节历史记录（写作问题库）
            "writing_issues": self.get_writing_issues(novel_id, status="open"),
        }
        return state

    # ═══════════════════════════════════════════════════════
    # 人物 CRUD
    # ═══════════════════════════════════════════════════════

    def save_character(self, novel_id: str, profile: CharacterProfile, locked: bool = False) -> str:
        """保存人物档案（按 novel_id + name 去重，同名则更新而非新建，保证角色档案唯一）。

        角色档案是全文唯一标准：同一小说内角色名唯一，重复保存同名角色会更新原条目，
        保留原 char_id，避免「删了重存导致 char_id 变化、向量/引用失效」的混乱。

        定稿锁：已 locked 的条目，只能被 locked=True（supervisor 定稿更新）覆盖；
        locked=False（子智能体草稿）的写入会抛 ValueError，防止子智能体覆盖定稿内容。
        """
        with self.get_session() as session:
            existing = (
                session.query(Character)
                .filter(Character.novel_id == novel_id, Character.name == profile.name)
                .first()
            )
            if existing and existing.locked and not locked:
                raise ValueError(f"角色「{profile.name}」已由 supervisor 定稿锁定，无权覆盖")
            if existing:
                # 同名角色已存在：更新（保留原 char_id）
                for key, value in profile.model_dump(exclude={"char_id"}).items():
                    setattr(existing, key, value)
                existing.locked = locked
                existing.updated_at = datetime.utcnow()
                char_id = existing.char_id
            else:
                char_id = profile.char_id or f"char_{uuid.uuid4().hex[:12]}"
                char = Character(char_id=char_id, novel_id=novel_id, locked=locked, **profile.model_dump(exclude={"char_id"}))
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

    def save_world_setting(self, novel_id: str, profile: WorldSettingProfile, locked: bool = False) -> str:
        """保存世界观设定（按 novel_id + name 去重，避免同名设定重复条目）。

        定稿锁：已 locked 的条目，只能被 locked=True（supervisor 定稿更新）覆盖。
        """
        with self.get_session() as session:
            existing = (
                session.query(WorldSetting)
                .filter(WorldSetting.novel_id == novel_id, WorldSetting.name == profile.name)
                .first()
            )
            if existing and existing.locked and not locked:
                raise ValueError(f"设定「{profile.name}」已由 supervisor 定稿锁定，无权覆盖")
            if existing:
                existing.category = profile.category
                existing.description = profile.description
                existing.details = profile.details
                existing.locked = locked
                sid = existing.setting_id
            else:
                sid = profile.setting_id or f"setting_{uuid.uuid4().hex[:12]}"
                setting = WorldSetting(
                    setting_id=sid, novel_id=novel_id, category=profile.category,
                    name=profile.name, description=profile.description,
                    details=profile.details, locked=locked,
                )
                session.add(setting)
            session.commit()
        return sid

    def get_world_settings(self, novel_id: str) -> list[WorldSettingProfile]:
        """获取小说的世界观设定（排除 character_registry 等内部记录，不对外展示）"""
        with self.get_session() as session:
            settings = (
                session.query(WorldSetting)
                .filter(WorldSetting.novel_id == novel_id, WorldSetting.category != "character_registry")
                .all()
            )
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

    def save_outline(self, novel_id: str, entry: OutlineEntry, locked: bool = False) -> str:
        """保存大纲条目。

        定稿锁：已 locked 的条目，只能被 locked=True（supervisor 定稿更新）覆盖。
        """
        oid = entry.outline_id or f"outline_{uuid.uuid4().hex[:12]}"
        with self.get_session() as session:
            existing = session.query(Outline).filter(Outline.outline_id == oid).first()
            if existing and existing.locked and not locked:
                raise ValueError(f"大纲「{entry.title}」已由 supervisor 定稿锁定，无权覆盖")
            if existing:
                existing.title = entry.title
                existing.summary = entry.summary
                existing.key_events = entry.key_events
                existing.foreshadowing = entry.foreshadowing
                existing.status = entry.status
                existing.locked = locked
            else:
                outline = Outline(
                    outline_id=oid, novel_id=novel_id, chapter_seq=entry.chapter_seq,
                    volume=entry.volume, title=entry.title, summary=entry.summary,
                    key_events=entry.key_events, foreshadowing=entry.foreshadowing,
                    status=entry.status, locked=locked,
                )
                session.add(outline)
            session.commit()
        return oid

    def save_outlines_batch(self, novel_id: str, entries: list[OutlineEntry]) -> list[str]:
        """批量保存大纲"""
        return [self.save_outline(novel_id, entry) for entry in entries]

    def get_outlines(self, novel_id: str) -> list[OutlineEntry]:
        """
        获取小说大纲（按章节序排序）。

        兼容两种情况：
        1. outlines 表（分章大纲，OutlineEntry 结构，chapter_seq/volume/title/summary）
        2. main_plots 表（分卷大纲，architect 用 save_to_long_term(category="plot") 保存）

        architect 实际是通过 save_to_long_term(category="plot") 保存大纲到 main_plots 表，
        而本方法原本只读 outlines 表，导致大纲「存了却读不到」（get_novel_outline 永远返回暂无大纲数据）。
        此处做 fallback：outlines 表为空时，从 main_plots 表构造等价条目返回。
        """
        with self.get_session() as session:
            outlines = (
                session.query(Outline)
                .filter(Outline.novel_id == novel_id)
                .order_by(Outline.chapter_seq)
                .all()
            )

        if outlines:
            return [
                OutlineEntry(
                    outline_id=o.outline_id, chapter_seq=o.chapter_seq,
                    volume=o.volume, title=o.title, summary=o.summary or "",
                    key_events=o.key_events or "", foreshadowing=o.foreshadowing or "",
                    status=o.status,
                )
                for o in outlines
            ]

        # fallback：从 main_plots 表读取（architect 存的分卷大纲）
        plots = self.get_main_plots(novel_id)
        if not plots:
            return []
        return [
            OutlineEntry(
                outline_id=p.get("plot_id", ""),
                chapter_seq=i + 1,
                volume=1,
                title=p.get("arc_name", ""),
                summary=p.get("description", ""),
                key_events="",
                foreshadowing="",
                status=p.get("status", "outlined"),
            )
            for i, p in enumerate(plots)
        ]

    # ═══════════════════════════════════════════════════════
    # 主线情节 CRUD
    # ═══════════════════════════════════════════════════════

    def save_main_plot(self, novel_id: str, arc_name: str, description: str,
                       start_chapter: int = 0, end_chapter: int = 0, locked: bool = False) -> str:
        """保存主线情节（按 novel_id + arc_name 去重，存在则更新，避免重复条目）。

        定稿锁：已 locked 的条目，只能被 locked=True（supervisor 定稿更新）覆盖。
        """
        with self.get_session() as session:
            existing = (
                session.query(MainPlot)
                .filter(MainPlot.novel_id == novel_id, MainPlot.arc_name == arc_name)
                .first()
            )
            if existing and existing.locked and not locked:
                raise ValueError(f"情节「{arc_name}」已由 supervisor 定稿锁定，无权覆盖")
            if existing:
                existing.description = description
                existing.start_chapter = start_chapter
                existing.end_chapter = end_chapter
                existing.locked = locked
                plot_id = existing.plot_id
            else:
                plot_id = f"plot_{uuid.uuid4().hex[:12]}"
                plot = MainPlot(
                    plot_id=plot_id, novel_id=novel_id, arc_name=arc_name,
                    description=description, start_chapter=start_chapter,
                    end_chapter=end_chapter, locked=locked,
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
    # 删除 CRUD（消除新旧条目并存的冲突）
    # ═══════════════════════════════════════════════════════

    def delete_character(self, novel_id: str, name: str) -> int:
        """按名称删除人物（返回删除数量）"""
        with self.get_session() as session:
            n = session.query(Character).filter(
                Character.novel_id == novel_id, Character.name == name
            ).delete()
            session.commit()
        if n:
            logger.info(f"已删除人物: {name}（{n} 条）")
        return n

    def delete_world_setting(self, novel_id: str, name: str) -> int:
        """按名称删除世界观设定"""
        with self.get_session() as session:
            n = session.query(WorldSetting).filter(
                WorldSetting.novel_id == novel_id, WorldSetting.name == name
            ).delete()
            session.commit()
        if n:
            logger.info(f"已删除设定: {name}（{n} 条）")
        return n

    def delete_main_plot(self, novel_id: str, arc_name: str) -> int:
        """按名称删除主线情节"""
        with self.get_session() as session:
            n = session.query(MainPlot).filter(
                MainPlot.novel_id == novel_id, MainPlot.arc_name == arc_name
            ).delete()
            session.commit()
        if n:
            logger.info(f"已删除情节: {arc_name}（{n} 条）")
        return n

    def delete_outline(self, novel_id: str, title: str) -> int:
        """按标题删除大纲条目"""
        with self.get_session() as session:
            n = session.query(Outline).filter(
                Outline.novel_id == novel_id, Outline.title == title
            ).delete()
            session.commit()
        if n:
            logger.info(f"已删除大纲: {title}（{n} 条）")
        return n

    def lock_entry(self, novel_id: str, category: str, name: str) -> int:
        """定稿加锁：把已有条目 locked=True（不改内容、不重发全文）。

        supervisor 审核通过草稿后调用，只需传条目名即可定稿，避免把草稿全文
        再次塞进 supervisor 上下文（这是 token 爆炸的主要来源之一）。

        参数:
            novel_id: 小说ID
            category: 分类 (character/setting/plot/outline)
            name: 条目名称（人物名/设定名/情节名/大纲标题）

        返回: 加锁的条目数量（0 表示未找到）
        """
        mapping = {
            "character": (Character, Character.name),
            "setting": (WorldSetting, WorldSetting.name),
            "plot": (MainPlot, MainPlot.arc_name),
            "outline": (Outline, Outline.title),
        }
        if category not in mapping:
            return 0
        model, col = mapping[category]
        with self.get_session() as session:
            rows = session.query(model).filter(
                model.novel_id == novel_id, col == name
            ).all()
            n = 0
            for r in rows:
                r.locked = True
                n += 1
            session.commit()
        if n:
            logger.info(f"已定稿加锁: {category}「{name}」（{n} 条）")
        return n

    def save_character_registry(self, novel_id: str, names: list[str]) -> None:
        """把权威角色名表持久化为一条世界设定记录（单一事实源）。

        存为 category=character_registry, name=authoritative_names，
        供 get_story_bible 读取权威名表——避免依赖可能被污染/遗漏的角色档案表
        （档案表只含「已落库」的角色，权威名表来自源文档，二者可能不一致）。
        """
        with self.get_session() as session:
            existing = (
                session.query(WorldSetting)
                .filter(WorldSetting.novel_id == novel_id, WorldSetting.name == "authoritative_names")
                .first()
            )
            desc = json.dumps(sorted(names), ensure_ascii=False)
            if existing:
                existing.category = "character_registry"
                existing.description = desc
            else:
                session.add(WorldSetting(
                    setting_id=f"setting_{uuid.uuid4().hex[:12]}",
                    novel_id=novel_id, category="character_registry",
                    name="authoritative_names", description=desc,
                ))
            session.commit()

    def get_character_registry(self, novel_id: str) -> list[str]:
        """读取权威角色名表（未持久化则返回空列表）。"""
        with self.get_session() as session:
            row = (
                session.query(WorldSetting)
                .filter(WorldSetting.novel_id == novel_id, WorldSetting.name == "authoritative_names")
                .first()
            )
        if not row or not row.description:
            return []
        try:
            names = json.loads(row.description)
            return [n for n in names if n]
        except Exception:
            return []

    def save_source_doc(self, novel_id: str, doc_name: str, content: str) -> None:
        """保存/更新某小说的一条源文档（多项目隔离，按 novel_id 存）。"""
        with self.get_session() as session:
            existing = (
                session.query(SourceDoc)
                .filter(SourceDoc.novel_id == novel_id, SourceDoc.doc_name == doc_name)
                .first()
            )
            if existing:
                existing.content = content
            else:
                session.add(SourceDoc(novel_id=novel_id, doc_name=doc_name, content=content))
            session.commit()

    def get_source_docs(self, novel_id: str) -> list[tuple[str, str]]:
        """读取某小说的所有源文档，返回 [(doc_name, content)]（按 doc_name 排序）。"""
        with self.get_session() as session:
            docs = (
                session.query(SourceDoc)
                .filter(SourceDoc.novel_id == novel_id)
                .order_by(SourceDoc.doc_name)
                .all()
            )
        return [(d.doc_name, d.content) for d in docs]

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
