"""
MCP HTTP/stdio 服务器。

提供标准 MCP 协议端点，供外部客户端连接。
"""

import json
from typing import Any

from loguru import logger

from app.mcp.tools import get_tool_registry
from app.prompts import get_all_prompts


class MCPServer:
    """
    轻量级 MCP 服务器。

    当前实现为进程内服务器（智能体通过直接导入调用）。
    未来可扩展为完整的 stdio/HTTP MCP 服务器。
    """

    def __init__(self):
        self.tool_registry = get_tool_registry()
        self._started = False

    async def start(self):
        """启动 MCP 服务"""
        if self._started:
            return
        self._started = True
        logger.info(f"MCP 服务已启动 ({len(self.tool_registry.all_tools)} 个工具)")

    async def stop(self):
        """停止 MCP 服务"""
        self._started = False
        logger.info("MCP 服务已停止")

    def list_tools(self) -> list[dict[str, Any]]:
        """列出所有工具"""
        return self.tool_registry.get_tool_descriptions()

    def list_resources(self) -> list[dict[str, Any]]:
        """列出所有提示词资源"""
        prompts = get_all_prompts()
        return [
            {"name": f"prompt:{agent}", "type": "prompt", "description": f"{agent} 系统提示词"}
            for agent in prompts
        ]

    def call_tool(self, tool_name: str, **kwargs) -> str:
        """调用工具"""
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return json.dumps({"error": f"未知工具: {tool_name}"})
        try:
            result = tool.invoke(kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"工具调用失败: {tool_name} -> {e}")
            return json.dumps({"error": str(e)})


# ─── 全局单例 ─────────────────────────────────────────

_server: MCPServer | None = None


def get_mcp_server() -> MCPServer:
    global _server
    if _server is None:
        _server = MCPServer()
    return _server
