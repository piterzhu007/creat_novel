"""
MCP 适配器：将记忆模块功能封装为 MCP 工具。

已迁移到 app/tools/factory.py。
此文件保留作为兼容层，委托给新的工具工厂。
"""

from app.tools.factory import NovelMemoryTools

__all__ = ["NovelMemoryTools"]
