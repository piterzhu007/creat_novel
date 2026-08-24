"""
运行追踪器（RunTracer）：捕获项目运行的技术细节。

基于 LangChain 回调机制，实时记录两类事件：
1. LLM 调用事件 —— 智能体名、模型名、token 消耗（输入/输出/总计）、耗时
2. 工具调用事件 —— 工具名、输出摘要、耗时

这些事件与 loguru 运行日志一起，由 save_run_log 工具导出为 txt，
作为「整个项目运行状态」的完整技术记录（进度追踪 + 审计）。
"""

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import PROJECT_ROOT

# 事件缓冲上限（避免长期运行内存无限增长）
_MAX_EVENTS = 2000

# 跨进程共享的 tracer 快照文件：主进程每次事件后覆盖写，MCP 子进程的 save_run_log 读取
TRACING_FILE = PROJECT_ROOT / "data" / "tracing" / "latest.txt"


class RunTracer(BaseCallbackHandler):
    """LangChain 回调追踪器（进程内单例）"""

    def __init__(self, max_events: int = _MAX_EVENTS, log_file: Optional[Path] = None):
        self._lock = threading.Lock()
        self._events: deque[dict] = deque(maxlen=max_events)
        # run_id -> 起始信息（用于配对 start/end 计算耗时）
        self._llm_start: dict[str, dict] = {}
        self._tool_start: dict[str, dict] = {}
        self._log_file = log_file
        self._persist()

    def _persist(self):
        """把当前快照覆盖写到跨进程共享文件（供 MCP 子进程 save_run_log 读取）"""
        if not self._log_file:
            return
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.write_text(self.render(), encoding="utf-8")
        except Exception:
            pass

    # ── LangChain 回调 ────────────────────────────────

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = kwargs.get("run_id")
        if not run_id:
            return
        with self._lock:
            self._llm_start[run_id] = {
                "ts": time.time(),
                "agent": self._infer_agent(kwargs),
                "model": self._infer_model(serialized, kwargs),
            }

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        start = self._llm_start.pop(run_id, None) if run_id else None
        usage = self._extract_usage(response)
        with self._lock:
            self._events.append({
                "ts": time.time(),
                "kind": "llm",
                "agent": (start or {}).get("agent", "unknown"),
                "model": (start or {}).get("model", ""),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "duration_ms": self._duration(start),
            })
        self._persist()

    def on_llm_error(self, error, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id:
            self._llm_start.pop(run_id, None)
        with self._lock:
            self._events.append({
                "ts": time.time(),
                "kind": "error",
                "error": str(error)[:300],
            })
        self._persist()

    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = kwargs.get("run_id")
        if not run_id:
            return
        with self._lock:
            self._tool_start[run_id] = {
                "ts": time.time(),
                "name": serialized.get("name", "unknown"),
            }

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        start = self._tool_start.pop(run_id, None) if run_id else None
        with self._lock:
            self._events.append({
                "ts": time.time(),
                "kind": "tool",
                "name": (start or {}).get("name", "unknown"),
                "output": self._truncate(str(output), 300),
                "duration_ms": self._duration(start),
            })
        self._persist()

    def on_tool_error(self, error, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id:
            self._tool_start.pop(run_id, None)
        with self._lock:
            self._events.append({
                "ts": time.time(),
                "kind": "tool_error",
                "error": str(error)[:300],
            })
        self._persist()

    # ── 辅助方法 ──────────────────────────────────────

    @staticmethod
    def _infer_agent(kwargs) -> str:
        """尽量推断发起这次 LLM 调用的智能体名"""
        metadata = kwargs.get("metadata") or {}
        # deepagents 会在 metadata 里标记 lc_agent_name（主 agent / 子 agent 名）
        agent_name = metadata.get("lc_agent_name")
        if agent_name:
            return agent_name
        node = metadata.get("langgraph_node")
        if node and node not in ("model", "tools"):
            return node
        tags = kwargs.get("tags") or []
        for t in tags:
            if t in ("supervisor", "architect", "writer", "editor", "reader"):
                return t
        name = kwargs.get("name")
        if name:
            return name
        return "unknown"

    @staticmethod
    def _infer_model(serialized, kwargs) -> str:
        """从 invocation_params / serialized 推断模型名"""
        params = kwargs.get("invocation_params") or {}
        for k in ("model_name", "model", "model_id"):
            if params.get(k):
                return str(params[k])
        return serialized.get("name", "")

    @staticmethod
    def _extract_usage(response) -> dict:
        """从 LLMResult 提取 token 用量（兼容多种 provider）"""
        usage: dict[str, int] = {}
        try:
            token_usage = (response.llm_output or {}).get("token_usage")
            if token_usage:
                usage["input_tokens"] = token_usage.get("prompt_tokens", 0) or 0
                usage["output_tokens"] = token_usage.get("completion_tokens", 0) or 0
                usage["total_tokens"] = token_usage.get("total_tokens", 0) or 0
            if response.generations:
                msg = response.generations[0][0].message
                um = getattr(msg, "usage_metadata", None) or {}
                if um:
                    usage["input_tokens"] = usage.get("input_tokens") or um.get("input_tokens", 0)
                    usage["output_tokens"] = usage.get("output_tokens") or um.get("output_tokens", 0)
                    usage["total_tokens"] = usage.get("total_tokens") or um.get("total_tokens", 0)
        except Exception:
            pass
        return usage

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "…"

    @staticmethod
    def _duration(start: Optional[dict]) -> float:
        if not start or "ts" not in start:
            return 0.0
        return round((time.time() - start["ts"]) * 1000, 1)

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    # ── 导出 ──────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        """返回当前事件快照（按时间顺序）"""
        with self._lock:
            return list(self._events)

    def clear(self):
        """清空事件缓冲（切换新会话时调用）"""
        with self._lock:
            self._events.clear()
            self._llm_start.clear()
            self._tool_start.clear()
        self._persist()

    def render(self, title: str = "运行追踪") -> str:
        """将事件渲染为可读文本（供 save_run_log 使用）"""
        with self._lock:
            events = list(self._events)

        if not events:
            return ""

        lines = [f"## {title}"]

        # ── 统计 ──
        llm_events = [e for e in events if e["kind"] == "llm"]
        tool_events = [e for e in events if e["kind"] == "tool"]
        total_in = sum(e.get("input_tokens", 0) for e in llm_events)
        total_out = sum(e.get("output_tokens", 0) for e in llm_events)
        lines.append("### Token 消耗统计")
        lines.append(f"- LLM 调用次数: {len(llm_events)}")
        lines.append(f"- 输入 token 合计: {total_in}")
        lines.append(f"- 输出 token 合计: {total_out}")
        lines.append(f"- 总 token 合计: {total_in + total_out}")
        lines.append(f"- 工具调用次数: {len(tool_events)}")
        lines.append("")

        # ── LLM 明细 ──
        if llm_events:
            lines.append("### LLM 调用明细")
            for e in llm_events:
                lines.append(
                    f"- [{self._fmt_ts(e['ts'])}] {e['agent']} | {e['model']} | "
                    f"in={e['input_tokens']} out={e['output_tokens']} "
                    f"total={e['total_tokens']} ({e['duration_ms']}ms)"
                )
            lines.append("")

        # ── 工具明细 ──
        if tool_events:
            lines.append("### 工具调用明细")
            for e in tool_events:
                lines.append(
                    f"- [{self._fmt_ts(e['ts'])}] {e['name']} ({e['duration_ms']}ms)\n"
                    f"    输出: {e['output']}"
                )
            lines.append("")

        # ── 错误 ──
        errs = [e for e in events if e["kind"] in ("error", "tool_error")]
        if errs:
            lines.append("### 错误")
            for e in errs:
                lines.append(f"- {e.get('error', '')}")
            lines.append("")

        return "\n".join(lines)


# ─── 单例 ────────────────────────────────────────────

_tracer: Optional[RunTracer] = None


def get_tracer() -> RunTracer:
    """获取运行追踪器单例"""
    global _tracer
    if _tracer is None:
        _tracer = RunTracer(log_file=TRACING_FILE)
    return _tracer


def read_tracing_snapshot() -> str:
    """读取主进程 tracer 落盘的快照文本（供 MCP 子进程 save_run_log 读取）"""
    try:
        if TRACING_FILE.exists():
            return TRACING_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""
