"""
测试：智能体定义和工具配置
"""

import pytest
from unittest.mock import MagicMock, patch


class TestNovelMemoryTools:
    """记忆工具测试"""

    def test_tool_count(self):
        """测试工具数量正确"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory()
        stm = ShortTermMemory()
        vs = VectorStore()
        factory = NovelMemoryTools(ltm, stm, vs)
        tools = factory.get_tools()
        assert len(tools) == 14

    def test_tool_names(self):
        """测试所有工具名称正确"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory()
        stm = ShortTermMemory()
        vs = VectorStore()
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
        ]
        for name in expected:
            assert name in tool_names, f"工具 {name} 未在列表中找到"


class TestWorkflowIntegration:
    """工作流集成测试（LangGraph 显式架构）"""

    @patch("app.workflow.LongTermMemory")
    @patch("app.workflow.ShortTermMemory")
    @patch("app.workflow.VectorStore")
    def test_build_workflow(self, mock_vs, mock_stm, mock_ltm):
        """测试构建 LangGraph 工作流"""
        from app.workflow import build_workflow
        from app.core.model_client import reset_model_registry

        mock_ltm.return_value = MagicMock()
        mock_stm.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        reset_model_registry()

        graph = build_workflow(checkpoint_db_path=None)
        nodes = list(graph.get_graph().nodes.keys())
        assert "supervisor" in nodes
        assert "architect" in nodes
        assert "writer" in nodes
        assert "editor" in nodes
        assert "reader" in nodes


class TestDeepAgent:
    """工作流创建测试"""

    @patch("app.workflow.LongTermMemory")
    @patch("app.workflow.ShortTermMemory")
    @patch("app.workflow.VectorStore")
    def test_create_workflow_compiles(self, mock_vs, mock_stm, mock_ltm):
        """测试创建 LangGraph 工作流并编译成功"""
        from app.workflow import build_workflow
        from app.core.model_client import reset_model_registry
        from langgraph.graph.state import CompiledStateGraph

        mock_ltm.return_value = MagicMock()
        mock_stm.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        reset_model_registry()

        graph = build_workflow(checkpoint_db_path=None)
        assert isinstance(graph, CompiledStateGraph)

    @patch("app.workflow.LongTermMemory")
    @patch("app.workflow.ShortTermMemory")
    @patch("app.workflow.VectorStore")
    def test_graph_has_nodes(self, mock_vs, mock_stm, mock_ltm):
        """测试 Graph 包含所有节点"""
        from app.workflow import build_workflow
        from app.core.model_client import reset_model_registry

        mock_ltm.return_value = MagicMock()
        mock_stm.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        reset_model_registry()

        graph = build_workflow(checkpoint_db_path=None)
        nodes = list(graph.get_graph().nodes.keys())
        assert "create_novel" in nodes
        assert "writer" in nodes


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
    """MCP 集成测试"""

    def test_tool_registry(self):
        """测试工具注册表"""
        from app.mcp.tools import get_tool_registry
        registry = get_tool_registry()
        tools = registry.all_tools
        assert len(tools) == 14

    def test_tool_groups(self):
        """测试工具分组"""
        from app.mcp.tools import get_tool_registry
        registry = get_tool_registry()
        groups = registry.get_tools_by_group()
        assert "long_term_memory" in groups
        assert "short_term_memory" in groups
        assert "vector_search" in groups
