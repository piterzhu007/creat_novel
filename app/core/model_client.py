"""
模型注册表：支持多供应商、多型号的模型池管理。

设计理念：
    参考 model_client/model.py 的「预定义多个模型实例」模式，
    但进一步做到「配置驱动」——每个智能体可独立指定供应商和型号，
    切换模型只需修改 YAML 配置，无需改动代码。

配置结构（conf/app_config.yaml）：
    models:
      deepseek_pro:                          # 模型槽位名（自定义）
        provider: deepseek                   # 供应商
        model: deepseek-v4-pro               # 型号
        base_url: https://api.deepseek.com   # API 端点
        api_key_env: DEEPSEEK_API_KEY        # 密钥对应的环境变量名
      deepseek_flash:
        provider: deepseek
        model: deepseek-v4-flash
        base_url: https://api.deepseek.com
        api_key_env: DEEPSEEK_API_KEY

    agents:
      supervisor:
        model: deepseek_pro                  # 引用上面的模型槽位
      writer:
        model: deepseek_flash                # 不同智能体可用不同型号
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from loguru import logger

from app.core.config import get_settings


@dataclass
class ModelSlot:
    """模型槽位：定义单个可用的模型（供应商 + 型号 + 端点 + 密钥来源）"""
    name: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""

    @property
    def api_key(self) -> str:
        """从环境变量读取密钥"""
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "")

    @property
    def identifier(self) -> str:
        """init_chat_model 使用的模型标识符（分离式参数不用这个）"""
        return f"{self.provider}:{self.model}"


class ModelRegistry:
    """
    模型注册表 —— 管理模型池和智能体模型绑定。

    核心能力：
    1. 从 YAML 加载多个模型槽位（不同供应商、同供应商不同型号均可）
    2. 每个智能体绑定到某个模型槽位
    3. 按需创建并缓存模型实例（带温度）
    """

    def __init__(self, settings=None, yaml_path: Optional[str] = None):
        self._settings = settings or get_settings()
        self._yaml_path = yaml_path
        self._slots: dict[str, ModelSlot] = {}
        self._bindings: dict[str, str] = {}  # agent_name -> slot_name
        self._models: dict[str, BaseChatModel] = {}  # 缓存键: slot_name@temperature

        self._load_model_pool()
        self._load_agent_bindings()

    # ── 配置加载 ────────────────────────────────────────

    def _resolve_yaml_path(self) -> Path:
        if self._yaml_path:
            return Path(self._yaml_path)
        from app.core.config import CONF_DIR
        return CONF_DIR / "app_config.yaml"

    def _load_yaml(self) -> dict:
        yaml_path = self._resolve_yaml_path()
        if not yaml_path.exists():
            logger.warning(f"配置文件不存在: {yaml_path}，使用空模型池")
            return {}
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_model_pool(self):
        """加载 models 段，构建模型槽位池"""
        data = self._load_yaml()
        models_cfg = data.get("models", {})

        if not models_cfg:
            # 回退：从 Settings 构建单个默认槽位，保持向后兼容
            default_slot = ModelSlot(
                name="default",
                provider=self._settings.llm_provider,
                model=self._settings.llm_model,
                base_url=self._settings.llm_base_url,
                api_key_env="DEEPSEEK_API_KEY",
            )
            self._slots["default"] = default_slot
            logger.info("未配置模型池，使用默认槽位 (Settings 来源)")
            return

        for slot_name, cfg in models_cfg.items():
            self._slots[slot_name] = ModelSlot(
                name=slot_name,
                provider=cfg.get("provider", "deepseek"),
                model=cfg.get("model", ""),
                base_url=cfg.get("base_url", ""),
                api_key_env=cfg.get("api_key_env", ""),
            )

        logger.info(f"模型池已加载 {len(self._slots)} 个槽位: {list(self._slots.keys())}")

    def _load_agent_bindings(self):
        """加载 agents 段，构建智能体 → 模型槽位映射"""
        data = self._load_yaml()
        agents_cfg = data.get("agents", {})

        if not agents_cfg:
            # 回退：所有智能体绑定到 default 槽位
            slot_name = next(iter(self._slots), "default")
            for agent in ["supervisor", "architect", "writer", "editor", "reader"]:
                self._bindings[agent] = slot_name
            logger.info("未配置智能体模型映射，全部使用默认槽位")
            return

        for agent_name, cfg in agents_cfg.items():
            slot_name = cfg.get("model", "default")
            if slot_name not in self._slots:
                logger.warning(
                    f"智能体 [{agent_name}] 绑定的槽位 [{slot_name}] 不存在，"
                    f"回退到第一个可用槽位"
                )
                slot_name = next(iter(self._slots), "default")
            self._bindings[agent_name] = slot_name

        logger.info(
            f"智能体模型映射已加载: "
            f"{ {a: s for a, s in self._bindings.items()} }"
        )

    # ── 温度映射 ────────────────────────────────────────

    def _temperature_for(self, agent_name: str) -> float:
        """从 Settings 获取智能体温度（.env 的 *_TEMPERATURE）"""
        mapping = {
            "supervisor": self._settings.supervisor_temperature,
            "architect": self._settings.architect_temperature,
            "writer": self._settings.writer_temperature,
            "editor": self._settings.editor_temperature,
            "reader": self._settings.reader_temperature,
        }
        return mapping.get(agent_name, 0.7)

    def _max_tokens_for(self, agent_name: str) -> int:
        """从 Settings 获取智能体的单次输出上限（.env 的 *_MAX_TOKENS）"""
        mapping = {
            "supervisor": self._settings.supervisor_max_tokens,
            "architect": self._settings.architect_max_tokens,
            "writer": self._settings.writer_max_tokens,
            "editor": self._settings.editor_max_tokens,
            "reader": self._settings.reader_max_tokens,
        }
        return mapping.get(agent_name, self._settings.max_tokens)

    # ── 公开 API ─────────────────────────────────────────

    def get_model(self, agent_name: str, force_reload: bool = False) -> BaseChatModel:
        """
        获取指定智能体的模型实例（带缓存）。

        参数:
            agent_name: 智能体名称 (supervisor/architect/writer/editor/reader)
            force_reload: 是否强制重建模型实例

        返回:
            BaseChatModel 实例
        """
        slot_name = self._bindings.get(agent_name)
        if slot_name is None:
            raise ValueError(f"未知智能体: {agent_name}（未配置模型绑定）")

        slot = self._slots[slot_name]
        temperature = self._temperature_for(agent_name)
        max_tokens = self._max_tokens_for(agent_name)
        cache_key = f"{slot_name}@{temperature}@{max_tokens}"

        if force_reload or cache_key not in self._models:
            self._models[cache_key] = self._build_model(slot, temperature, max_tokens)

        return self._models[cache_key]

    def warmup_models(self, agent_names: Optional[list[str]] = None):
        """
        并行预初始化所有智能体的模型实例，加速启动。

        模型创建之间无依赖，用线程池并行执行，避免串行逐个初始化的耗时。

        参数:
            agent_names: 要预初始化的智能体名列表（默认全部）
        """
        from concurrent.futures import ThreadPoolExecutor

        names = agent_names or list(self._bindings.keys())
        # 过滤掉已缓存的
        to_build = [n for n in names if self._build_cache_key(n) not in self._models]

        if not to_build:
            return

        def _build(name):
            slot_name = self._bindings[name]
            slot = self._slots[slot_name]
            return name, self._build_model(
                slot, self._temperature_for(name), self._max_tokens_for(name)
            )

        with ThreadPoolExecutor(max_workers=len(to_build)) as executor:
            results = executor.map(_build, to_build)

        for name, model in results:
            self._models[self._build_cache_key(name)] = model
            logger.info(f"模型已预热: {name}")

    def _build_cache_key(self, agent_name: str) -> str:
        """构建缓存键（与 get_model 保持一致）"""
        slot_name = self._bindings.get(agent_name)
        if slot_name is None:
            return ""
        return (
            f"{slot_name}@{self._temperature_for(agent_name)}"
            f"@{self._max_tokens_for(agent_name)}"
        )

    def get_model_by_slot(
        self,
        slot_name: str,
        temperature: float = 0.7,
    ) -> BaseChatModel:
        """按槽位名直接获取模型实例"""
        if slot_name not in self._slots:
            raise ValueError(f"未知模型槽位: {slot_name}")
        return self._build_model(self._slots[slot_name], temperature)

    def _build_model(self, slot: ModelSlot, temperature: float,
                     max_tokens: int = 200000) -> BaseChatModel:
        """通过 init_chat_model 构建模型实例（分离式参数，同参考文件）"""
        model = init_chat_model(
            model=slot.model,
            model_provider=slot.provider,
            api_key=slot.api_key,
            base_url=slot.base_url or None,
            temperature=temperature,
            max_tokens=max_tokens,
            # 禁用流式 chunk 超时：architect 读 20 万 token 源文档后单次生成超长
            # 结构化设定时，服务端可能在中途停顿 >120s（TCP 仍活着但暂不吐 chunk），
            # 默认的 120s 超时会误报「No streaming chunk received」中断任务。
            stream_chunk_timeout=None,
        )
        logger.info(
            f"模型实例已创建: [{slot.name}] {slot.identifier} "
            f"(t={temperature}, max_tokens={max_tokens})"
        )
        return model

    # ── 便捷方法（保持旧 API 兼容） ───────────────────────

    def get_supervisor_model(self) -> BaseChatModel:
        return self.get_model("supervisor")

    def get_architect_model(self) -> BaseChatModel:
        return self.get_model("architect")

    def get_writer_model(self) -> BaseChatModel:
        return self.get_model("writer")

    def get_editor_model(self) -> BaseChatModel:
        return self.get_model("editor")

    def get_reader_model(self) -> BaseChatModel:
        return self.get_model("reader")

    # ── 状态查询 ─────────────────────────────────────────

    @property
    def slots(self) -> dict[str, ModelSlot]:
        return self._slots

    @property
    def bindings(self) -> dict[str, str]:
        return self._bindings

    def summary(self) -> str:
        """返回模型配置摘要（展示每个智能体用的什么模型）"""
        parts = []
        for agent, slot_name in self._bindings.items():
            slot = self._slots.get(slot_name)
            if slot:
                parts.append(f"{agent}→{slot.identifier}")
        return " | ".join(parts)


# ─── 全局单例 ────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """获取模型注册表单例"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        logger.info(f"模型注册表已初始化: {_registry.summary()}")
    return _registry


def reset_model_registry():
    """重置模型注册表（测试用）"""
    global _registry
    _registry = None
