"""models —— data_labeling 包的 dataclass 定义。

所有领域对象都使用 ``@dataclass(frozen=True)`` 修饰：
    * 不可变，避免下游误改内存对象后与数据库状态脱节；
    * 字段顺序与 spec.md "ADDED Requirements" 严格对齐；
    * 提供 ``from_row`` 类方法从 sqlite3.Row 还原实例，便于 DAO 透传。

``Label.value`` 字段约定（见 spec.md Requirement "标签持久化"）：
    * ``+1`` —— 买（看多）；
    * ``-1`` —— 卖（看空）；
    * ``0``  —— 清除 / 显式标记"不操作"。

注意：未打标签的 K 线**不会**出现在 ``labels`` 表中（见 spec.md），所以本类
仅描述"被打过的"标签；"未标注" 是数据库中**不存在**该行，**不是** ``value=0``。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

# ---- 标签字面量（不在 Python 层强约束类型，DB 层是 INTEGER，方便后续扩展） ----
LABEL_BUY: int = 1
LABEL_SELL: int = -1
LABEL_CLEAR: int = 0


@dataclass(frozen=True)
class Dataset:
    """数据集元数据，对应 ``datasets`` 表的一行。

    字段含义：
        id:           数据库自增主键。新建未入库时为 None。
        name:         唯一命名，格式 ``{SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{YYYYMMDD}``。
        symbol:       交易品种代码（"XAUUSDm" 等）。
        timeframe:    周期字符串（"M1" / "H1" / "D1" 等，与 mt5_data.parse_timeframe 输入一致）。
        date_from:    区间起始时间（UTC，naive datetime，与 fetch_rates 返回保持一致）。
        date_to:      区间结束时间（UTC，naive datetime）。当命名用 "latest" 时为 None。
        row_count:    蜡烛条数（与 len(candles) 一致）。
        created_at:   入库时间戳（UTC，naive datetime）。
        notes:        可选备注。
    """

    id: Optional[int]
    name: str
    symbol: str
    timeframe: str
    date_from: datetime
    date_to: Optional[datetime]
    row_count: int
    created_at: datetime
    notes: Optional[str] = None

    @classmethod
    def from_row(cls, row: Any) -> "Dataset":
        """从 sqlite3.Row 还原 Dataset 实例。"""
        # 兼容 sqlite3.Row / dict / tuple
        get = row["date_from"] if hasattr(row, "__getitem__") and "date_from" in row.keys() else None
        # sqlite3 没有原生 datetime，DAO 层负责把 TEXT 字段反序列化为 datetime。
        return cls(
            id=row["id"],
            name=row["name"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            date_from=_parse_iso(row["date_from"]),
            date_to=_parse_iso(row["date_to"]) if row["date_to"] else None,
            row_count=row["row_count"],
            created_at=_parse_iso(row["created_at"]),
            notes=row["notes"],
        )


@dataclass(frozen=True)
class Candle:
    """单根 K 线，对应 ``candles`` 表的一行。

    复合主键：``(dataset_id, time)``。蜡烛字段为必填数值（DB 层 NOT NULL）。
    """

    dataset_id: int
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int

    @classmethod
    def from_row(cls, row: Any) -> "Candle":
        return cls(
            dataset_id=row["dataset_id"],
            time=_parse_iso(row["time"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            tick_volume=row["tick_volume"],
        )


@dataclass(frozen=True)
class Label:
    """单条标签，对应 ``labels`` 表的一行。

    ``value`` 取 ``+1 / -1 / 0`` 三者之一。
    """

    dataset_id: int
    time: datetime
    value: int
    labeled_at: datetime
    note: Optional[str] = None

    @classmethod
    def from_row(cls, row: Any) -> "Label":
        return cls(
            dataset_id=row["dataset_id"],
            time=_parse_iso(row["time"]),
            value=row["label"],
            labeled_at=_parse_iso(row["labeled_at"]),
            note=row["note"],
        )


def _parse_iso(value: Any) -> datetime:
    """把 ISO 8601 字符串或 datetime 统一为 naive datetime（UTC）。"""
    if isinstance(value, datetime):
        # 去掉 tzinfo：DB 存的是 naive，与 fetch_rates 返回索引保持一致
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        # pandas / sqlite 都可能返回带 "T" 或带 "Z" 的字符串
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    raise TypeError(f"无法解析时间字段: {value!r}")
