"""logging.py —— 结构化日志（Section 5 核心，stdlib logging + JSON formatter）。

提供：
    * :class:`JsonFormatter` —— stdlib logging Formatter，输出 JSON 格式
      （每行 1 个 JSON object）；
    * :func:`setup_logging` —— 配置 root logger（级别 + JSON formatter）；
    * :func:`get_logger` —— 获取带 name 字段的 logger。

设计要点（与 spec.md "Section 5: 结构化日志" Requirement 严格对齐）：
    * **不引入新 pip**：纯 stdlib（``logging`` + ``json`` + ``datetime``）；
    * 关键路径（fold 评估、stacker 训完、artifact 落盘）改用 logger；
    * CLI ``--log-level {DEBUG, INFO, WARNING, ERROR}`` 默认 INFO。

公开 API：
    * :func:`setup_logging`
    * :func:`get_logger`
    * :class:`JsonFormatter`
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from typing import Any, Dict, Optional


# ============================================================
# JSON Formatter
# ============================================================


class JsonFormatter(logging.Formatter):
    """stdlib logging Formatter → JSON 行。

    输出格式（每行 1 个 JSON object）：
        {"timestamp": "...", "level": "info", "logger": "...",
         "message": "...", ...extra_fields}
    """

    DEFAULT_FIELDS = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """把 LogRecord 格式化成 JSON 行。"""
        # 基础字段
        out: Dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc,
            ).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 附加字段（用户传 extra=...）
        for k, v in record.__dict__.items():
            if k not in self.DEFAULT_FIELDS and not k.startswith("_"):
                try:
                    json.dumps(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = str(v)
        # 异常信息
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


# ============================================================
# setup_logging / get_logger
# ============================================================


def setup_logging(
    level: str = "INFO",
    stream: Optional[Any] = None,
) -> None:
    """配置 root logger（级别 + JSON formatter）。

    Args:
        level: 日志级别（DEBUG / INFO / WARNING / ERROR），默认 INFO。
        stream: 输出流（默认 ``sys.stdout``）。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    # 清掉旧 handler（避免重复配置）
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """获取 logger（用法与 stdlib 一致）。"""
    return logging.getLogger(name)
