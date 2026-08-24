"""
全局持久异步运行时。

MCP 工具是 async-only 的（langchain_mcp_adapters 转换的工具只实现 _arun），
且 MCP session 和 AsyncSqliteSaver 都绑定创建它们的 event loop。

若每次用 asyncio.run() 新建 loop，会导致：
1. session 失效 → 每次工具调用重新 spawn 子进程 + 重新初始化记忆后端（数秒）
2. AsyncSqliteSaver 的 aiosqlite 连接跨 loop 报错

因此这里提供一个贯穿整个应用生命周期的持久 event loop，
所有 async 操作（加载 MCP session、建立 checkpointer、astream 流式推理）
都在这个 loop 上执行，保证 session/连接复用、不重复初始化。
"""

import asyncio

_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def get_loop() -> asyncio.AbstractEventLoop:
    """获取全局持久 event loop"""
    return _loop


def run(coro):
    """在全局持久 loop 上同步执行一个 coroutine，返回结果"""
    if _loop.is_running():
        raise RuntimeError(
            "持久事件循环正在运行中，禁止在 loop 内嵌套调用 run()——"
            "请检查是否在 async 回调里误用了同步 run()"
        )
    return _loop.run_until_complete(coro)


def close():
    """关闭全局持久 event loop（应用退出时调用一次）"""
    if _loop is not None and not _loop.is_closed():
        _loop.close()
