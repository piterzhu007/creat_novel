"""
测试：deepagents 架构工作流
"""

import pytest
from unittest.mock import MagicMock, patch


class TestWorkflow:
    """工作流测试"""

    @patch("app.agent._load_mcp_tools")
    def test_build_workflow_structure(self, mock_load_tools):
        """测试构建 deep agent 工作流图结构"""
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
        assert graph is not None

        nodes = list(graph.get_graph().nodes.keys())
        # deepagents 图含 model/tools 节点（ReAct 循环）
        for node in ["model", "tools"]:
            assert node in nodes, f"节点 {node} 未注册"

    @patch("app.agent._load_mcp_tools")
    def test_workflow_has_input_schema(self, mock_load_tools):
        """测试工作流有输入 schema"""
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
        assert hasattr(graph, "input_schema")

    def test_novel_memory_tools_has_all_tools(self, tmp_path):
        """测试工具工厂创建所有工具"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore

        ltm = LongTermMemory(str(tmp_path / "lt.db"))
        stm = ShortTermMemory(str(tmp_path / "st.db"))
        vs = VectorStore(str(tmp_path / "vec"))
        factory = NovelMemoryTools(ltm, stm, vs)
        tools = factory.get_tools()

        tool_names = [t.name for t in tools]
        required_tools = [
            "search_long_term_memory", "save_to_long_term",
            "get_novel_outline", "get_character_profile",
            "get_world_building", "get_short_term_context",
            "update_short_term", "save_chapter",
            "search_similar_content", "list_novels",
            "save_writing_issue", "get_writing_issues",
            "export_chapters", "get_story_bible",
            "get_chapter", "read_source_docs", "delete_long_term_entry",
        ]
        for name in required_tools:
            assert name in tool_names, f"工具 {name} 未找到"
