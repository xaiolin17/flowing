"""downloader —— 数据下载主入口。

本模块负责把 mt5 终端返回的原始 K 线数据转换为 ``pandas.DataFrame`` 供业务层使用。
约定：
    * 返回的 DataFrame 以 ``time``（UTC ``datetime``）为索引，按时间升序；
    * 任何下载失败均抛 ``RuntimeError`` 或 ``ValueError``，**不**静默返回空 DataFrame。
"""

from datetime import datetime
from typing import Optional, Union

import pandas as pd

import MetaTrader5 as mt5

from .connection import initialize
from .timeframes import parse_timeframe


def _ensure_initialized() -> None:
    """确保 mt5 终端已连接；未连接时尝试初始化一次。

    供 ``fetch_rates`` 内部调用，使调用方不必显式 ``initialize`` 也能使用。
    """
    if not initialize():
        # initialize() 内部已打印 last_error()，这里再补一句便于排障。
        raise RuntimeError("mt5 终端未连接，且 initialize() 失败；请检查终端是否启动。")


def fetch_rates(
    symbol: str,
    timeframe: Union[str, int],
    date_from: datetime,
    date_to: Optional[datetime] = None,
) -> pd.DataFrame:
    """下载指定品种在 [date_from, date_to] 区间内的 K 线数据。

    Args:
        symbol: 交易品种代码（如 ``"EURUSD"``、``"XAUUSD"``）。
        timeframe: 周期，支持字符串别名（``"M1"`` / ``"H1"`` / ``"D1"`` 等），
            也可直接传 ``mt5.TIMEFRAME_*`` 整型。
        date_from: 区间起始时间（含）。
        date_to: 区间结束时间（含）。为 ``None`` 时取到最新可用 K 线。

    Returns:
        以 ``time`` 为索引的 ``pandas.DataFrame``，列固定为
        ``[open, high, low, close, tick_volume, spread, real_volume]``。
        ``time`` 列为 ``datetime64[ns]``（UTC，无时区）。

    Raises:
        RuntimeError: 终端未连接 / 数据区间无效 / mt5 内部错误。
        ValueError: 品种无法加入 MarketWatch（broker 未提供 / 名称拼写错误）。
    """
    # Step 1: 确保终端连接可用；连接失败时主动抛错。
    _ensure_initialized()

    # Step 2: 解析周期字符串为 mt5 原生常量。
    # 解析失败会在内部抛 ValueError，向上自然透传。
    tf = parse_timeframe(timeframe)

    # Step 3: 把品种加入 MarketWatch。
    # mt5 的 MarketWatch 是惰性的：未启用的品种直接 copy_rates_range 会失败。
    # symbol_select 第二个参数 True 表示"加入 MarketWatch"。
    if not mt5.symbol_select(symbol, True):
        raise ValueError(
            f"无法在 MarketWatch 中启用品种 '{symbol}'：{mt5.last_error()}"
        )

    # Step 4: 拉取区间 K 线。
    # 官方文档：copy_rates_range 在区间为空、未来时间或终端无数据时返回 None。
    raw = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    if raw is None or len(raw) == 0:
        # 主动抛错而非返回空 DF：上层做回测时，空 DF 容易掩盖"拉错了品种/区间"等问题。
        raise RuntimeError(
            f"copy_rates_range('{symbol}', {tf}, {date_from}, {date_to}) 返回空："
            f"{mt5.last_error()}"
        )

    # Step 5: numpy 结构化数组 -> DataFrame。
    # mt5 返回的 time 字段是秒级 epoch（UTC），需显式转换。
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Step 6: 设 time 为索引并按时间升序。
    # 排序是防御性的：个别 broker 历史数据可能存在轻微乱序。
    df = df.set_index("time").sort_index()

    return df
