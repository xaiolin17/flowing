"""candle_features —— 虚拟 K 线窗口特征（Section 1 新增）。

提供：
    * :func:`compute_candle_window_features` —— 主入口：raw OHLC →
      2 × len(candle_windows) 列的 DataFrame。

设计要点（与 spec.md "Section 1: 虚拟 K 线特征" Requirement 严格对齐）：
    * 合并方式 = 虚拟 K 线（high=max, low=min, open=first, close=last）；
    * 窗口 N 取 fib 数列（默认 (1,2,3,5,8,13,21,34,55)）；
    * 指标 = 2 个：X1 = (high_v - low_v) / prev_close_v，
      X2 = (open_v - close_v) / prev_close_v；
    * prev_close_v = close[i-N]（窗口前一根 K 线的收盘价）；
    * **N=1 一致性**：与单根 K 线算法完全一致（关键正确性边界）；
    * rolling(N, min_periods=N) 前 N-1 行 NaN；shift(N) 前 N 行 NaN；
    * prev_close_v 接近 0 → 用 ``abs(close).clip(lower=eps)`` 保护；
    * 不引入新 pip：纯 pandas + numpy。

公开 API：
    * :func:`compute_candle_window_features`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 默认 candle_windows：fib 数列 9 元组（与 fib_lags 风格统一）
DEFAULT_CANDLE_WINDOWS: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55)


def compute_candle_window_features(
    ohlc_df: pd.DataFrame,
    candle_windows: tuple[int, ...] = DEFAULT_CANDLE_WINDOWS,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """计算虚拟 K 线窗口特征。

    对每个窗口 N ∈ candle_windows，输出两列：
        * ``cw_{N}_x1 = (max(high[i-N+1:i+1]) - min(low[i-N+1:i+1])) / prev_close_v``
        * ``cw_{N}_x2 = (open[i-N+1] - close[i]) / prev_close_v``

    其中 ``prev_close_v = close[i-N]``（窗口前一根 K 线的收盘价）。

    Args:
        ohlc_df: raw OHLC DataFrame，必含 ``open / high / low / close`` 4 列。
        candle_windows: 窗口 N 列表（默认 fib 9 元组）。
        eps: prev_close 接近 0 时的除零保护（默认 1e-12）。

    Returns:
        新 DataFrame，2 × len(candle_windows) 列，列名
        ``cw_{N}_x1`` / ``cw_{N}_x2``，长度与输入一致。
        前 N-1 行为 NaN（rolling + shift 双重 NaN 来源）；
        prev_close_v 接近 0 时用 ``abs(close).clip(lower=eps)`` 保护，
        不产生 inf / NaN。

    Raises:
        ValueError: 输入缺 ``open / high / low / close`` 4 列。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlc_df.columns)
    if missing:
        raise ValueError(
            f"compute_candle_window_features 缺必要列: {sorted(missing)}"
        )

    if not candle_windows:
        return pd.DataFrame(index=ohlc_df.index)

    open_s = ohlc_df["open"]
    high = ohlc_df["high"]
    low = ohlc_df["low"]
    close = ohlc_df["close"]
    # prev_close 接近 0 的除零保护：abs(close) 然后 clip 到 >= eps
    # 用 shift 后的 abs，因为 prev_close_v 在每个 i 是 close[i-N]
    safe_prev_close = close.abs().clip(lower=eps)

    pieces: dict[str, pd.Series] = {}
    for n in candle_windows:
        # 虚拟 K 线聚合
        high_v = high.rolling(window=n, min_periods=n).max()
        low_v = low.rolling(window=n, min_periods=n).min()
        # open_v = open[i-N+1]（窗口最左边的开盘价）
        open_v = open_s.shift(n - 1)
        # close_v = close[i]（当前 K 线的收盘价 = 窗口最右边的收盘价）
        close_v = close
        # prev_close_v = close[i-N]（窗口前一根 K 线的收盘价）
        prev_close_v = close.shift(n).abs().clip(lower=eps)

        x1 = (high_v - low_v) / prev_close_v
        x2 = (open_v - close_v) / prev_close_v

        pieces[f"cw_{n}_x1"] = x1
        pieces[f"cw_{n}_x2"] = x2

    return pd.DataFrame(pieces, index=ohlc_df.index)
