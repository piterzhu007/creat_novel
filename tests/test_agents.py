"""
测试：智能体定义和工具配置
"""

import pytest
from unittest.mock import MagicMock, patch


class TestNovelMemoryTools:
    """记忆工具测试"""

    def test_tool_count(self, tmp_path):
        """测试工具数量正确"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        factory = NovelMemoryTools(ltm, stm, vs)
        tools = factory.get_tools()
        assert len(tools) == 25

    def test_tool_names(self, tmp_path):
        """测试所有工具名称正确"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        factory = NovelMemoryTools(ltm, stm, vs)
        tool_names = [t.name for t in factory.get_tools()]
        expected = [
            "search_long_term_memory",
            "save_to_long_term",
            "get_novel_outline",
            "get_character_profile",
            "get_world_building",
            "get_short_term_context",
            "update_short_term",
            "save_chapter",
            "search_similar_content",
            "list_novels",
            "save_writing_issue",
            "get_writing_issues",
            "export_chapters",
            "get_story_bible",
            "get_novel_state",
            "update_novel_progress",
            "create_novel",
            "save_run_log",
            "get_chapter",
            "read_source_docs",
            "delete_long_term_entry",
            "patch_chapter",
            "get_writing_context",
            "lock_entry",
            "get_novel_progress",
        ]
        for name in expected:
            assert name in tool_names, f"工具 {name} 未在列表中找到"


class TestWorkflowIntegration:
    """工作流集成测试（deepagents 架构 + MCP 工具）"""

    @patch("app.agent._load_mcp_tools")
    def test_build_workflow(self, mock_load_tools):
        """测试构建 deep agent 工作流（工具通过 mock 的 MCP 加载）"""
        from app.workflow import create_novel_workflow
        from app.core.model_client import reset_model_registry

        # mock MCP 加载返回一个哑工具，避免测试时启动真实子进程
        from langchain_core.tools import tool as langchain_tool
        @langchain_tool
        def dummy_tool(x: str) -> str:
            """哑工具"""
            return x
        dummy_tool.name = "dummy_tool"
        mock_load_tools.return_value = [dummy_tool]
        reset_model_registry()

        graph = create_novel_workflow(checkpoint_db_path=None)
        nodes = list(graph.get_graph().nodes.keys())
        # deepagents 图含 model/tools 节点
        assert "model" in nodes
        assert "tools" in nodes


class TestDeepAgent:
    """工作流创建测试"""

    @patch("app.agent._load_mcp_tools")
    def test_create_workflow_compiles(self, mock_load_tools):
        """测试创建 deep agent 工作流并编译成功"""
        from app.workflow import create_novel_workflow
        from app.core.model_client import reset_model_registry
        from langgraph.graph.state import CompiledStateGraph

        from langchain_core.tools import tool as langchain_tool
        @langchain_tool
        def dummy_tool(x: str) -> str:
            """哑工具"""
            return x
        dummy_tool.name = "dummy_tool"
        mock_load_tools.return_value = [dummy_tool]
        reset_model_registry()

        graph = create_novel_workflow(checkpoint_db_path=None)
        assert isinstance(graph, CompiledStateGraph)

    @patch("app.agent._load_mcp_tools")
    def test_graph_has_nodes(self, mock_load_tools):
        """测试 Graph 包含 ReAct 循环节点"""
        from app.workflow import create_novel_workflow
        from app.core.model_client import reset_model_registry

        from langchain_core.tools import tool as langchain_tool
        @langchain_tool
        def dummy_tool(x: str) -> str:
            """哑工具"""
            return x
        dummy_tool.name = "dummy_tool"
        mock_load_tools.return_value = [dummy_tool]
        reset_model_registry()

        graph = create_novel_workflow(checkpoint_db_path=None)
        nodes = list(graph.get_graph().nodes.keys())
        assert "model" in nodes
        assert "tools" in nodes


class TestConfig:
    """配置模块测试"""

    def test_model_settings(self):
        """测试模型基础配置"""
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.llm_provider == "deepseek"
        assert "deepseek" in settings.llm_model
        assert settings.max_tokens == 200000

    def test_model_registry_multi_slot(self):
        """测试模型注册表支持多供应商、多型号"""
        from app.core.model_client import get_model_registry, reset_model_registry
        reset_model_registry()
        registry = get_model_registry()

        # 模型池应包含多个槽位
        assert len(registry.slots) >= 2
        assert "deepseek_pro" in registry.slots
        assert "deepseek_flash" in registry.slots

        # 不同智能体绑定到不同槽位
        assert registry.bindings["supervisor"] == "deepseek_pro"
        assert registry.bindings["editor"] == "deepseek_flash"

        # pro 槽位被 supervisor/architect 共用，flash 槽位被 writer/editor/reader 共用
        assert registry.bindings["architect"] == "deepseek_pro"
        assert registry.bindings["writer"] == "deepseek_flash"

    def test_model_registry_per_agent_model(self):
        """测试每个智能体获取到独立绑定的模型"""
        from app.core.model_client import get_model_registry, reset_model_registry
        reset_model_registry()
        registry = get_model_registry()

        supervisor = registry.get_supervisor_model()
        editor = registry.get_editor_model()

        # 不同智能体使用不同型号
        assert supervisor.model_name == "deepseek-v4-pro"
        assert editor.model_name == "deepseek-v4-flash"

    def test_prompts_loaded(self):
        """测试提示词加载"""
        from app.prompts import (
            SUPERVISOR_PROMPT, ARCHITECT_PROMPT,
            WRITER_PROMPT, EDITOR_PROMPT, READER_PROMPT,
        )
        assert len(SUPERVISOR_PROMPT) > 0
        assert len(ARCHITECT_PROMPT) > 0
        assert len(WRITER_PROMPT) > 0
        assert len(EDITOR_PROMPT) > 0
        assert len(READER_PROMPT) > 0

    def test_chat_template_creation(self):
        """测试 ChatPromptTemplate 创建"""
        from app.prompts import create_chat_template
        tmpl = create_chat_template("architect", "{input}")
        assert "{input}" in str(tmpl.messages[-1])


class TestMCPIntegration:
    """MCP 服务器测试"""

    def test_mcp_server_has_all_tools(self):
        """测试 MCP server 暴露全部工具"""
        import asyncio
        from app.mcp.server import mcp
        tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        required = ["create_novel", "get_chapter", "read_source_docs", "delete_long_term_entry"]
        for name in required:
            assert name in tool_names, f"工具 {name} 未在 MCP server 中注册"


class TestToolPermission:
    """工具权限测试：细粒度写库边界 + supervisor 定稿独占（硬约束）"""

    def test_write_tools_set(self):
        """测试写工具集合正确（10 个写工具）"""
        from app.agent import _WRITE_TOOLS
        assert _WRITE_TOOLS == frozenset({
            "create_novel", "update_novel_progress", "save_to_long_term",
            "save_chapter", "patch_chapter", "update_short_term",
            "save_writing_issue", "export_chapters", "delete_long_term_entry",
            "save_run_log", "lock_entry",
        })

    def test_sub_agents_fine_grained_write(self, tmp_path):
        """测试子智能体只含各自的专属写工具，不含 supervisor 独占写工具"""
        from app.agent import _build_sub_agents
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        from app.tools import NovelMemoryTools

        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        factory = NovelMemoryTools(ltm, stm, vs)
        all_tools = factory.get_tools()

        registry = MagicMock()
        registry.get_model = MagicMock(return_value=MagicMock())

        sub_agents = _build_sub_agents(all_tools, registry)
        assert len(sub_agents) == 4

        # supervisor 独占的写工具（定稿锁、删除、patch、进度、导出、建项目、日志）
        supervisor_only = {
            "create_novel", "update_novel_progress", "patch_chapter",
            "export_chapters", "delete_long_term_entry", "save_run_log",
            "lock_entry",
        }
        # 每个子智能体的专属写工具（方案 A 细粒度边界）
        expected_write = {
            "architect": {"save_to_long_term"},
            "writer": {"save_chapter", "update_short_term"},
            "editor": {"save_writing_issue"},
            "reader": {"save_writing_issue"},
        }

        for sa in sub_agents:
            tool_names = set(t.name for t in sa["tools"])
            assert not (supervisor_only & tool_names), f"{sa['name']} 不应含 supervisor 独占写工具"
            own = expected_write[sa["name"]]
            assert own <= tool_names, f"{sa['name']} 应含专属写工具 {own}"
            for other_name, other_write in expected_write.items():
                if other_name != sa["name"]:
                    for w in (other_write - own):
                        assert w not in tool_names, f"{sa['name']} 不应含 {other_name} 的写工具 {w}"


class TestNameConfusionGuard:
    """角色名混乱防护：名表补全 + 写时硬校验 + 名表入 DB"""

    @staticmethod
    def _make_factory(tmp_path):
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        return NovelMemoryTools(ltm, stm, vs), ltm, stm

    def test_extract_includes_group_members(self):
        """名表应补全 F4/315 群体段落里的成员名（此前漏掉，被当名表外）"""
        from app.tools.factory import NovelMemoryTools
        text = open("人物角色与背景.txt", encoding="utf-8", errors="replace").read()
        names = NovelMemoryTools._extract_character_names(text)
        for n in ["赵志龙", "钟奕", "小雅", "魏大鹏", "宋伟", "刘松", "曾国欢", "曾国庆"]:
            assert n in names, f"名表应含群体成员 {n}"

    def test_save_chapter_corrects_confusable(self, tmp_path):
        """写时硬校验：正文里的同音/简繁错字自动回退为正确写法（字符级易混淆）"""
        factory, _, stm = self._make_factory(tmp_path)
        nid = factory._create_novel("测试", "都市", "", 100).split("novel_id=")[1].rstrip(")。")
        factory._save_chapter(nid, 1, "林峰看到林風和林楓。", "t")
        draft = stm.get_latest_draft(nid, 1)
        for bad in ("林峰", "林風", "林楓"):
            assert bad not in draft.content, f"正文不应残留 {bad}"
        assert "林枫" in draft.content  # 枫

    def test_registry_persisted_and_used_by_story_bible(self, tmp_path):
        """名表入 DB：create_novel 后 registry 持久化，get_story_bible 读它且不泄漏为世界观"""
        factory, ltm, _ = self._make_factory(tmp_path)
        nid = factory._create_novel("测试", "都市", "", 100).split("novel_id=")[1].rstrip(")。")
        reg = ltm.get_character_registry(nid)
        assert "赵志龙" in reg and "林枫" in reg
        bible = factory._get_story_bible(nid)
        assert "赵志龙" in bible  # 权威名表里应含补全的配角
        assert "authoritative_names" not in bible  # registry 不应泄漏进世界观核心

    def test_multi_project_isolation(self, tmp_path):
        """多项目隔离：不同小说各自用各自的源文档/名表，落库校验互不污染"""
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        from app.tools import NovelMemoryTools

        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "人物角色与背景.txt").write_text(
            "## 二、主要人物画像\n### 张三\n### 李四\n", encoding="utf-8")
        (dir_b / "人物角色与背景.txt").write_text(
            "## 二、主要人物画像\n### 王五\n### 赵六\n", encoding="utf-8")

        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        factory = NovelMemoryTools(ltm, stm, vs)

        nid_a = factory._create_novel("项目A", "都市", "", 100, source_dir=str(dir_a)).split("novel_id=")[1].rstrip(")。")
        nid_b = factory._create_novel("项目B", "都市", "", 100, source_dir=str(dir_b)).split("novel_id=")[1].rstrip(")。")

        names_a = factory._get_authoritative_names(nid_a)
        names_b = factory._get_authoritative_names(nid_b)
        assert "张三" in names_a and "王五" not in names_a
        assert "王五" in names_b and "张三" not in names_b

        # A 项目拒 B 项目的角色名，反之亦然；各自接受自己的角色名
        assert "已拒绝" in factory._save_to_long_term(nid_a, "character", "王五", "内容", "{}")
        assert "已拒绝" in factory._save_to_long_term(nid_b, "character", "张三", "内容", "{}")
        assert "人物已保存" in factory._save_to_long_term(nid_a, "character", "张三", "内容", "{}")
