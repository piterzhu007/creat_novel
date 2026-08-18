"""
数据模型：小说创作相关的 Pydantic 模型和 SQLAlchemy ORM 模型。

Pydantic 模型用于智能体间数据传递和序列化。
SQLAlchemy 模型用于持久化存储。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ═══════════════════════════════════════════════════════════
# 枚举类
# ═══════════════════════════════════════════════════════════

class NovelStatus(str, Enum):
    """小说状态"""
    PLANNING = "planning"
    WRITING = "writing"
    EDITING = "editing"
    COMPLETED = "completed"
    PAUSED = "paused"


class ChapterStatus(str, Enum):
    """章节状态"""
    OUTLINED = "outlined"
    DRAFTING = "drafting"
    WRITTEN = "written"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REVISION = "revision"


class AgentAction(str, Enum):
    """智能体动作类型"""
    ANALYSIS = "analysis"
    CREATION = "creation"
    REVIEW = "review"
    CHECK = "check"
    DECISION = "decision"


# ═══════════════════════════════════════════════════════════
# SQLAlchemy ORM 基类
# ═══════════════════════════════════════════════════════════

class Base(DeclarativeBase):
    pass


# ─── 长期记忆表 ─────────────────────────────────────────

class Novel(Base):
    """小说主表"""
    __tablename__ = "novels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    novel_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="小说唯一标识")
    title: Mapped[str] = mapped_column(String(256), comment="小说标题")
    genre: Mapped[str] = mapped_column(String(128), default="", comment="小说类型/流派")
    synopsis: Mapped[str] = mapped_column(Text, default="", comment="小说简介")
    status: Mapped[str] = mapped_column(String(32), default=NovelStatus.PLANNING.value, comment="小说状态")
    target_chapters: Mapped[int] = mapped_column(Integer, default=0, comment="目标章节数")
    current_chapter: Mapped[int] = mapped_column(Integer, default=0, comment="当前生成进度（章节序号，0=未开始）")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Character(Base):
    """人物表"""
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    char_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="人物唯一标识")
    novel_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属小说ID")
    name: Mapped[str] = mapped_column(String(128), comment="人物名称")
    role_type: Mapped[str] = mapped_column(String(64), default="supporting", comment="主角/配角/龙套")
    gender: Mapped[str] = mapped_column(String(16), default="", comment="性别")
    age: Mapped[str] = mapped_column(String(32), default="", comment="年龄")
    appearance: Mapped[str] = mapped_column(Text, default="", comment="外貌描述")
    personality: Mapped[str] = mapped_column(Text, default="", comment="性格特征")
    background: Mapped[str] = mapped_column(Text, default="", comment="背景故事")
    motivation: Mapped[str] = mapped_column(Text, default="", comment="核心动机")
    abilities: Mapped[str] = mapped_column(Text, default="", comment="能力/特长")
    relationships: Mapped[str] = mapped_column(Text, default="", comment="人物关系 (JSON)")
    arc_summary: Mapped[str] = mapped_column(Text, default="", comment="人物弧光概述")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorldSetting(Base):
    """世界观设定表"""
    __tablename__ = "world_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    setting_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="设定唯一标识")
    novel_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属小说ID")
    category: Mapped[str] = mapped_column(String(64), comment="分类: 时代/地理/社会/魔法/科技/种族")
    name: Mapped[str] = mapped_column(String(256), comment="设定名称")
    description: Mapped[str] = mapped_column(Text, default="", comment="设定详细描述")
    details: Mapped[str] = mapped_column(Text, default="", comment="补充详情 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Outline(Base):
    """大纲表"""
    __tablename__ = "outlines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    outline_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="大纲唯一标识")
    novel_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属小说ID")
    chapter_seq: Mapped[int] = mapped_column(Integer, comment="章节序号")
    volume: Mapped[int] = mapped_column(Integer, default=1, comment="卷号")
    title: Mapped[str] = mapped_column(String(256), default="", comment="章节标题")
    summary: Mapped[str] = mapped_column(Text, default="", comment="章节概要")
    key_events: Mapped[str] = mapped_column(Text, default="", comment="关键事件列表 (JSON)")
    foreshadowing: Mapped[str] = mapped_column(Text, default="", comment="伏笔设计")
    status: Mapped[str] = mapped_column(String(32), default=ChapterStatus.OUTLINED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MainPlot(Base):
    """主线情节表"""
    __tablename__ = "main_plots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plot_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="情节唯一标识")
    novel_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属小说ID")
    arc_name: Mapped[str] = mapped_column(String(256), comment="情节弧名称")
    description: Mapped[str] = mapped_column(Text, default="", comment="情节描述")
    start_chapter: Mapped[int] = mapped_column(Integer, default=0, comment="起始章节")
    end_chapter: Mapped[int] = mapped_column(Integer, default=0, comment="结束章节")
    status: Mapped[str] = mapped_column(String(32), default="active", comment="状态")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─── 短期记忆表 ─────────────────────────────────────────

class SubPlot(Base):
    """子情节/支线表"""
    __tablename__ = "sub_plots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sub_plot_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    novel_id: Mapped[str] = mapped_column(String(64), index=True)
    chapter_id: Mapped[str] = mapped_column(String(64), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active")
    resolved_at: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChapterDraft(Base):
    """章节草稿表"""
    __tablename__ = "chapter_drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draft_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    novel_id: Mapped[str] = mapped_column(String(64), index=True)
    chapter_seq: Mapped[int] = mapped_column(Integer, comment="章节序号")
    version: Mapped[int] = mapped_column(Integer, default=1, comment="版本号")
    title: Mapped[str] = mapped_column(String(256), default="")
    content: Mapped[str] = mapped_column(Text, default="", comment="章节正文")
    word_count: Mapped[int] = mapped_column(Integer, default=0, comment="字数")
    feedback: Mapped[str] = mapped_column(Text, default="", comment="审核反馈")
    quality_score: Mapped[Optional[float]] = mapped_column(Float, default=None, comment="质量评分")
    status: Mapped[str] = mapped_column(String(32), default=ChapterStatus.DRAFTING.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    """智能体操作日志"""
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    novel_id: Mapped[str] = mapped_column(String(64), default="")
    agent_name: Mapped[str] = mapped_column(String(64), comment="智能体名称")
    action: Mapped[str] = mapped_column(String(64), comment="操作类型")
    input_summary: Mapped[str] = mapped_column(Text, default="", comment="输入摘要")
    output_summary: Mapped[str] = mapped_column(Text, default="", comment="输出摘要")
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, comment="消耗的 token 数")
    success: Mapped[bool] = mapped_column(default=True, comment="是否成功")
    error_msg: Mapped[str] = mapped_column(Text, default="", comment="错误信息")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WritingIssue(Base):
    """写作问题库：editor/reader 发现的历史问题，供 writer 主动规避"""
    __tablename__ = "writing_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    novel_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属小说ID")
    issue_type: Mapped[str] = mapped_column(String(32), comment="问题类型: 连贯性/人物一致性/逻辑/文笔/世界观/节奏")
    chapter_seq: Mapped[int] = mapped_column(Integer, default=0, comment="发现问题的章节序号")
    description: Mapped[str] = mapped_column(Text, comment="问题描述")
    suggestion: Mapped[str] = mapped_column(Text, default="", comment="规避建议")
    found_by: Mapped[str] = mapped_column(String(32), default="", comment="发现者: editor/reader")
    severity: Mapped[str] = mapped_column(String(16), default="medium", comment="严重程度: low/medium/high")
    status: Mapped[str] = mapped_column(String(16), default="open", comment="状态: open/resolved")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
