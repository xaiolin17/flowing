"""persistence —— 高层落库入口。

两个公开函数：
    * :func:`make_dataset_name` —— 拼接 ``{SYMBOL}_{TF}_{YYYYMMDD}_{YYYYMMDD}`` 名字；
    * :func:`import_from_mt5` —— 调 fetch_rates 拉数据 + 创建 dataset + 批量写 candles。

设计要点：
    * persistence 依赖 :mod:`mt5_data.downloader`（单向依赖），不引入反向引用，
      避免 ``mt5_data`` 与 ``data_labeling`` 之间形成循环；
    * 名字生成与 DB 写入原子化：先尝试创建 dataset（依赖 UNIQUE 约束抢锁），
      重复名直接抛 ``ValueError`` 给上层处理；
    * DB 默认路径用 :data:`data_labeling.db.DEFAULT_DB_PATH`。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .db import DEFAULT_DB_PATH, CandleDAO, DatasetDAO, init_db
from .models import Candle, Dataset


def make_dataset_name(
    symbol: str,
    timeframe: str,
    date_from: datetime,
    date_to: Optional[datetime] = None,
) -> str:
    """生成 dataset 命名。

    格式：``{SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{YYYYMMDD|literal:'latest'}``。

    Args:
        symbol: 交易品种代码（"XAUUSDm"）。
        timeframe: 周期字符串（"M1" / "H1" / "D1" 等）。
        date_from: 区间起始。
        date_to: 区间结束；为 None 时用字面量 ``"latest"``。

    Returns:
        拼接好的名字。
    """
    if not symbol or not timeframe:
        raise ValueError("symbol 和 timeframe 不能为空")
    if not isinstance(date_from, datetime):
        raise TypeError("date_from 必须是 datetime 实例")
    if date_to is not None and not isinstance(date_to, datetime):
        raise TypeError("date_to 必须是 datetime 或 None")
    if date_to is not None and date_to < date_from:
        raise ValueError(f"date_to({date_to}) 早于 date_from({date_from})")

    # 规整化：去掉空白 + 统一大小写风格（SYMBOL 大写、TF 保持原样）
    sym = str(symbol).strip()
    tf = str(timeframe).strip()
    from_str = date_from.strftime("%Y%m%d")
    to_str = date_to.strftime("%Y%m%d") if date_to is not None else "latest"
    return f"{sym}_{tf}_{from_str}_{to_str}"


def _df_to_candles(dataset_id: int, df: pd.DataFrame) -> list[Candle]:
    """把 ``fetch_rates`` 返回的 DataFrame 转成 Candle 列表。

    期望的 DataFrame 索引是 ``time``（datetime），列固定为
    ``[open, high, low, close, tick_volume]``。
    """
    required = {"open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"fetch_rates 返回的 DataFrame 缺少必要列: {sorted(missing)}"
        )

    candles: list[Candle] = []
    # 索引是 time，转成 list 逐行打包
    for ts, row in df.iterrows():
        candles.append(
            Candle(
                dataset_id=dataset_id,
                time=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=int(row["tick_volume"]),
            )
        )
    return candles


def import_from_mt5(
    symbol: str,
    timeframe: str,
    date_from: datetime,
    date_to: Optional[datetime] = None,
    db_path: Optional[Path] = None,
    notes: Optional[str] = None,
) -> Dataset:
    """下载并落库。

    流程：
        1. 调 :func:`mt5_data.fetch_rates` 拉数据；
        2. 生成 dataset name，调 :class:`DatasetDAO.create` 插入元数据；
        3. 调 :class:`CandleDAO.bulk_insert` 写入 candles；
        4. 返回带 id 的 ``Dataset``。

    Args:
        symbol: 品种代码。
        timeframe: 周期。
        date_from: 区间起始。
        date_to: 区间结束（None = 拉最新）。
        db_path: SQLite 文件路径；None 用默认。
        notes: dataset 备注。

    Returns:
        新创建的 ``Dataset`` 实例（含数据库自增 id）。

    Raises:
        ValueError: name 重复 / fetch_rates 失败 / DataFrame 缺列。
    """
    # 延迟 import：避免在只跑 DAO 单测时强制拉入 MetaTrader5 SDK
    from mt5_data import fetch_rates  # type: ignore

    # 1) 拉数据（任何异常自然向上抛）
    df = fetch_rates(symbol, timeframe, date_from, date_to)

    # 2) 确保 DB 已初始化
    init_db(db_path)

    # 3) 命名 + 建 dataset 行（UNIQUE 冲突由 DAO 转 ValueError）
    name = make_dataset_name(symbol, timeframe, date_from, date_to)
    ds_dao = DatasetDAO(db_path)
    dataset = ds_dao.create(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        date_from=date_from,
        date_to=date_to,
        row_count=len(df),
        notes=notes,
    )

    # 4) 批量写 candles
    candles = _df_to_candles(dataset.id, df)
    CandleDAO(db_path).bulk_insert(candles)
    return dataset
