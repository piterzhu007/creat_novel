"""
工作流入口（代理层）：委托给 deepagents 架构。

保留 create_novel_workflow 接口名，供 main.py 调用。
实际实现见 app/agent.py 的 create_novel_agent（create_deep_agent + SubAgent）。
"""

from typing import Optional

from app.agent import create_novel_agent


def create_novel_workflow(checkpoint_db_path: Optional[str] = None):
    """创建小说创作 Deep Agent 工作流"""
    return create_novel_agent(checkpoint_db_path=checkpoint_db_path)
