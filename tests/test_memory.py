"""
测试：记忆模块 CRUD 操作
"""

import pytest
from pathlib import Path

from app.memory import LongTermMemory, ShortTermMemory, VectorStore
from app.models.memory import CharacterProfile, WorldSettingProfile, OutlineEntry


class TestLongTermMemory:
    """长期记忆测试"""

    @pytest.fixture
    def memory(self, tmp_path: Path):
        db_path = tmp_path / "test_novels.db"
        return LongTermMemory(str(db_path))

    def test_create_and_get_novel(self, memory: LongTermMemory):
        """测试创建和获取小说"""
        novel_id = memory.create_novel("测试小说", "玄幻", "一部测试小说", 10)
        novel = memory.get_novel(novel_id)
        assert novel is not None
        assert novel.title == "测试小说"
        assert novel.genre == "玄幻"

    def test_list_novels(self, memory: LongTermMemory):
        """测试列出所有小说"""
        memory.create_novel("小说A")
        memory.create_novel("小说B")
        novels = memory.list_novels()
        assert len(novels) >= 2

    def test_save_and_get_characters(self, memory: LongTermMemory):
        """测试保存和获取人物"""
        novel_id = memory.create_novel("人物测试小说")
        profile = CharacterProfile(
            name="张三", role_type="protagonist", gender="男",
            personality="勇敢正直", background="出身贫寒",
        )
        memory.save_character(novel_id, profile)
        chars = memory.get_characters(novel_id)
        assert len(chars) >= 1
        assert any(c.name == "张三" for c in chars)

    def test_save_and_get_world_settings(self, memory: LongTermMemory):
        """测试世界观设定"""
        novel_id = memory.create_novel("设定测试")
        profile = WorldSettingProfile(
            name="天元大陆", description="一个充满灵气的修仙世界",
            category="geography",
        )
        memory.save_world_setting(novel_id, profile)
        settings = memory.get_world_settings(novel_id)
        assert len(settings) >= 1

    def test_save_and_get_outlines(self, memory: LongTermMemory):
        """测试大纲 CRUD"""
        novel_id = memory.create_novel("大纲测试")
        entry = OutlineEntry(
            chapter_seq=1, title="序章", summary="故事的开端",
            key_events="主角登场", foreshadowing="隐藏身份",
        )
        memory.save_outline(novel_id, entry)
        outlines = memory.get_outlines(novel_id)
        assert len(outlines) >= 1
        assert outlines[0].title == "序章"

    def test_get_novel_context(self, memory: LongTermMemory):
        """测试获取小说完整上下文"""
        novel_id = memory.create_novel("上下文测试", "都市", "一个都市故事")
        ctx = memory.get_novel_context(novel_id)
        assert ctx is not None
        assert ctx.title == "上下文测试"
        assert ctx.genre == "都市"


class TestShortTermMemory:
    """短期记忆测试"""

    @pytest.fixture
    def memory(self, tmp_path: Path):
        db_path = tmp_path / "test_short_term.db"
        return ShortTermMemory(str(db_path))

    def test_add_and_get_sub_plots(self, memory: ShortTermMemory):
        """测试子情节"""
        sp_id = memory.add_sub_plot("novel_1", "ch_1", "主角在旅途中遇到神秘老人")
        active = memory.get_active_sub_plots("novel_1")
        assert len(active) >= 1

    def test_save_and_get_draft(self, memory: ShortTermMemory):
        """测试草稿保存"""
        d_id = memory.save_draft("novel_1", 1, "第一章正文内容...", "第一章 开端")
        draft = memory.get_latest_draft("novel_1", 1)
        assert draft is not None
        assert draft.content == "第一章正文内容..."

    def test_draft_versioning(self, memory: ShortTermMemory):
        """测试草稿版本管理"""
        memory.save_draft("novel_1", 1, "版本1")
        memory.save_draft("novel_1", 1, "版本2")
        draft = memory.get_latest_draft("novel_1", 1)
        assert draft.version == 2

    def test_agent_log(self, memory: ShortTermMemory):
        """测试智能体日志"""
        memory.log_agent_action("session_1", "architect", "create_characters",
                                output_summary="创建了3个人物")
        logs = memory.get_recent_logs("session_1")
        assert len(logs) >= 1


class TestVectorStore:
    """向量存储测试"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        persist_dir = tmp_path / "test_chroma"
        return VectorStore(str(persist_dir))

    def test_collections_exist(self, store: VectorStore):
        """测试所有 collection 已创建"""
        for name in VectorStore.COLLECTIONS:
            count = store.count(name)
            assert count >= 0  # collection exists

    def test_add_and_search(self, store: VectorStore):
        """测试添加和搜索"""
        store.add("novel_characters", "张三是一个勇敢的年轻人",
                  {"name": "张三", "role": "protagonist"})
        results = store.search("novel_characters", "勇敢的战士", k=3)
        assert isinstance(results, list)

    def test_delete(self, store: VectorStore):
        """测试删除"""
        doc_id = store.add("novel_settings", "测试设定内容")
        store.delete("novel_settings", doc_id)
        # 删除后应不报错
