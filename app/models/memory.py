"""
记忆相关的 Pydantic 数据模型。

用于智能体间的数据传输和序列化，与 ORM 模型解耦。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型"""
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    VECTOR = "vector"


class MemoryEntry(BaseModel):
    """通用记忆条目"""
    entry_id: str = Field(description="条目唯一标识")
    memory_type: MemoryType = Field(description="记忆类型")
    novel_id: str = Field(description="所属小说ID")
    category: str = Field(default="general", description="分类: character/setting/plot/draft")
    content: str = Field(description="记忆内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    embedding: Optional[list[float]] = Field(default=None, description="向量嵌入")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CharacterProfile(BaseModel):
    """人物档案（智能体间传输用）"""
    char_id: str = ""  # 空值表示新建，由 save_character 自动生成
    name: str
    role_type: str = "supporting"
    gender: str = ""
    age: str = ""
    appearance: str = ""
    personality: str = ""
    background: str = ""
    motivation: str = ""
    abilities: str = ""
    relationships: str = ""
    arc_summary: str = ""


class WorldSettingProfile(BaseModel):
    """世界观设定档案"""
    setting_id: str = ""  # 空值表示新建，由 save_world_setting 自动生成
    category: str
    name: str
    description: str
    details: str = ""


class OutlineEntry(BaseModel):
    """大纲条目"""
    outline_id: str = ""  # 空值表示新建，由 save_outline 自动生成
    chapter_seq: int
    volume: int = 1
    title: str = ""
    summary: str = ""
    key_events: str = ""
    foreshadowing: str = ""
    status: str = "outlined"


class NovelContext(BaseModel):
    """小说完整上下文（传给智能体）"""
    novel_id: str
    title: str
    genre: str = ""
    synopsis: str = ""
    characters: list[CharacterProfile] = Field(default_factory=list)
    world_settings: list[WorldSettingProfile] = Field(default_factory=list)
    outlines: list[OutlineEntry] = Field(default_factory=list)
    current_chapter: int = 1
    previous_chapter_summary: str = ""
    active_sub_plots: list[str] = Field(default_factory=list)
