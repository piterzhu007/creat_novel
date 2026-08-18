"""
自定义异常类
"""


class AppException(Exception):
    """应用基础异常"""
    pass


class ConfigurationError(AppException):
    """配置错误"""
    pass


class MemoryError(AppException):
    """记忆模块错误"""
    pass


class AgentError(AppException):
    """智能体执行错误"""
    pass


class WorkflowError(AppException):
    """工作流错误"""
    pass


class MCPServerError(AppException):
    """MCP 服务错误"""
    pass
