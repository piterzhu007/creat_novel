"""
配置模块：加载和管理应用配置。

使用 Pydantic Settings 从环境变量加载配置。
优先级：环境变量 > 默认值
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── 项目路径 ───────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# ─── 加载 .env 到 os.environ ──────────────────────────
# 保证无论从哪个入口导入（main.py 或直接 import app.core.config），
# .env 中的密钥都能被 os.getenv 正确读取。
try:
    from dotenv import load_dotenv

    if DEFAULT_ENV_PATH.exists():
        load_dotenv(DEFAULT_ENV_PATH, encoding="utf-8")
    else:
        load_dotenv(encoding="utf-8")
except ImportError:  # python-dotenv 未安装时静默跳过
    pass


# ─── 配置模型 ───────────────────────────────────────────


class Settings(BaseSettings):
    """应用总配置 —— 从环境变量加载"""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════
    # 模型供应商配置（兜底回退：模型池优先从 conf/app_config.yaml 加载，
    # 若无 models 段则使用这里的默认值）
    # ═══════════════════════════════════════════════════════

    llm_provider: str = Field(
        default="deepseek",
        alias="LLM_PROVIDER",
        description="模型供应商: deepseek / openai / anthropic 等",
    )
    llm_model: str = Field(
        default="deepseek-v4-pro",
        alias="LLM_MODEL",
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="LLM_BASE_URL",
    )
    max_tokens: int = Field(
        default=200000,
        alias="LLM_MAX_TOKENS",
        description="模型最大上下文窗口",
    )

    # 嵌入模型（ChromaDB 使用）
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="https://api.deepseek.com", alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL")

    # ── 智能体温度 ──
    architect_temperature: float = Field(default=1.5, alias="ARCHITECT_TEMPERATURE")
    writer_temperature: float = Field(default=0.8, alias="WRITER_TEMPERATURE")
    editor_temperature: float = Field(default=0.2, alias="EDITOR_TEMPERATURE")
    reader_temperature: float = Field(default=0.5, alias="READER_TEMPERATURE")
    supervisor_temperature: float = Field(default=0.3, alias="SUPERVISOR_TEMPERATURE")

    # ── 智能体 max_tokens（单次输出上限，按角色差异化）──
    # writer 写正文需要最大输出；editor/reader 只出报告，可设小值省成本
    architect_max_tokens: int = Field(default=200000, alias="ARCHITECT_MAX_TOKENS")
    writer_max_tokens: int = Field(default=200000, alias="WRITER_MAX_TOKENS")
    editor_max_tokens: int = Field(default=16384, alias="EDITOR_MAX_TOKENS")
    reader_max_tokens: int = Field(default=16384, alias="READER_MAX_TOKENS")
    supervisor_max_tokens: int = Field(default=16384, alias="SUPERVISOR_MAX_TOKENS")

    # ── 存储路径 ──
    sqlite_db_path: str = Field(default="data/novels.db", alias="SQLITE_DB_PATH")
    short_term_db_path: str = Field(default="data/short_term.db", alias="SHORT_TERM_DB_PATH")
    chroma_db_path: str = Field(default="data/vector_db/", alias="CHROMA_DB_PATH")

    # ── 日志 ──
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


# ─── 单例实例 ───────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取配置单例（延迟初始化，允许测试覆盖）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(env_file: Optional[str] = None) -> Settings:
    """重新加载配置（用于测试或运行时切换环境）"""
    global _settings
    _settings = Settings(_env_file=env_file) if env_file else Settings()
    return _settings


# ─── 路径工具 ────────────────────────────────────────


def _resolve_path(relative_path: str) -> Path:
    """将相对路径解析为绝对路径（相对于项目根目录）"""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_db_path() -> Path:
    """获取 SQLite 数据库的绝对路径"""
    settings = get_settings()
    return _resolve_path(settings.sqlite_db_path)


def get_chroma_path() -> Path:
    """获取 ChromaDB 持久化目录的绝对路径"""
    settings = get_settings()
    return _resolve_path(settings.chroma_db_path)


def get_short_term_db_path() -> Path:
    """获取短期记忆 SQLite 数据库的绝对路径"""
    settings = get_settings()
    return _resolve_path(settings.short_term_db_path)
