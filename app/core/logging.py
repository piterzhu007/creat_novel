"""
日志模块：基于 loguru 的日志配置。

在应用入口处调用 setup_logging() 完成配置。
"""

import sys

from loguru import logger

from app.core.config import get_settings, PROJECT_ROOT


def setup_logging() -> None:
    """初始化日志配置"""
    settings = get_settings()

    # 移除默认的 handler
    logger.remove()

    # 添加控制台 handler（彩色输出）
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 添加文件 handler（按大小滚动）
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )

    logger.info(f"日志系统已初始化 (level={settings.log_level})")


def get_logger(name: str):
    """获取带有模块名的 logger"""
    return logger.bind(name=name)
