"""
MCP 工具注册表。

统一管理所有工具的定义、schema 和元数据。
已迁移到使用 NovelMemoryTools 工厂模式。
"""

from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from app.tools.factory import NovelMemoryTools


class ToolRegistry:
    """
    工具注册表 —— 提供工具发现和描述。

    用于：
    1. 向智能体注册可用工具
    2. 向 MCP Server 暴露工具列表
    3. 工具分组和访问控制
    """

    def __init__(self):
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._load_tools()

    def _load_tools(self):
        """加载所有 MCP 工具（通过 NovelMemoryTools 工厂）"""
        from app.memory import LongTermMemory, ShortTermMemory, VectorStore
        ltm = LongTermMemory()
        stm = ShortTermMemory()
        vs = VectorStore()
        factory = NovelMemoryTools(ltm, stm, vs)
        self._tools = factory.get_tools()
        self._tool_map = {t.name: t for t in self._tools}
        logger.info(f"工具注册表已加载 {len(self._tools)} 个工具")

    @property
    def all_tools(self) -> list[BaseTool]:
        return self._tools

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tool_map.get(name)

    def get_tools_by_group(self) -> dict[str, list[str]]:
        """按功能分组（供 UI 使用）"""
        return {
            "long_term_memory": [
                "search_long_term_memory", "save_to_long_term",
                "get_novel_outline", "get_character_profile", "get_world_building",
                "list_novels",
            ],
            "short_term_memory": [
                "get_short_term_context", "update_short_term", "save_chapter",
            ],
            "vector_search": [
                "search_similar_content",
            ],
        }

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """获取所有工具的描述信息"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": str(t.args_schema.schema()) if t.args_schema else None,
            }
            for t in self._tools
        ]


# ─── 全局单例 ─────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
