"""
MCP 模块初始化。

整合 MCP Server、Tool Registry 和工具工厂。
"""

from .server import get_mcp_server, MCPServer
from .tools import get_tool_registry, ToolRegistry

__all__ = [
    "MCPServer", "get_mcp_server",
    "ToolRegistry", "get_tool_registry",
]
