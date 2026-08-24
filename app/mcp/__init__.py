"""
MCP 模块。

工具通过 app/mcp/server.py 的 FastMCP stdio 服务器以标准 MCP 协议暴露。

注意：这里**不** eager import server 模块。原因——agent 通过
`python -m app.mcp.server` 启动子进程时，runpy 会先 import 父包 `app.mcp`
（若这里 import server 会触发一次模块顶层初始化），再把 server 作为 __main__
执行（又初始化一次），导致 SQLite/ChromaDB/embedding 双重初始化，白费数秒
且 ChromaDB 双实例有锁风险。故本包保持为空，server 由调用方显式 import。
"""
