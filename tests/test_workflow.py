"""
测试：LangGraph 显式工作流
"""

import pytest
from unittest.mock import MagicMock, patch


class TestWorkflow:
    """工作流测试"""

    @patch("app.workflow.LongTermMemory")
    @patch("app.workflow.ShortTermMemory")
    @patch("app.workflow.VectorStore")
    def test_build_workflow_structure(self, mock_vs, mock_stm, mock_ltm):
        """测试构建工作流图结构"""
        from app.workflow import build_workflow
        from app.core.model_client import reset_model_registry

        mock_ltm.return_value = MagicMock()
        mock_stm.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        reset_model_registry()

        graph = build_workflow(checkpoint_db_path=None)
        assert graph is not None

        nodes = list(graph.get_graph().nodes.keys())
        # 父图只含 create_novel + supervisor 子图；子 agent 在 supervisor 内部
        for node in ["create_novel", "supervisor"]:
            assert node in nodes, f"节点 {node} 未注册"

    @patch("app.workflow.LongTermMemory")
    @patch("app.workflow.ShortTermMemory")
    @patch("app.workflow.VectorStore")
    def test_workflow_has_input_schema(self, mock_vs, mock_stm, mock_ltm):
        """测试工作流有输入 schema"""
        from app.workflow import build_workflow
        from app.core.model_client import reset_model_registry

        mock_ltm.return_value = MagicMock()
        mock_stm.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        reset_model_registry()

        graph = build_workflow(checkpoint_db_path=None)
        assert hasattr(graph, "input_schema")

    def test_state_schema(self):
        """测试共享状态 schema 包含核心字段"""
        from app.state import NovelState
        # 验证关键字段存在于 TypedDict 注解中
        annotations = NovelState.__annotations__
        for field in ["messages", "novel_id", "characters", "world_settings",
                      "outlines", "current_chapter", "writer_output",
                      "editor_report", "reader_report", "phase"]:
            assert field in annotations, f"状态字段 {field} 缺失"

    def test_novel_memory_tools_has_all_tools(self):
        """测试工具工厂创建所有工具"""
        from app.tools import NovelMemoryTools
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore

        ltm = LongTermMemory()
        stm = ShortTermMemory()
        vs = VectorStore()
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
        ]
        for name in required_tools:
            assert name in tool_names, f"工具 {name} 未找到"
