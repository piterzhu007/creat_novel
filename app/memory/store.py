"""
ORM Session 管理器。

提供统一的数据库会话获取和事务管理。
"""

from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_db_path


class StoreManager:
    """SQLAlchemy 存储管理器"""

    _instance: Optional["StoreManager"] = None

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(get_db_path())
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "StoreManager":
        """单例模式"""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试）"""
        cls._instance = None

    @contextmanager
    def session_scope(self) -> Session:
        """提供事务作用域 session"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
