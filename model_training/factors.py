"""factors —— 自实现技术指标（FeatureSpec 驱动）。

提供：
    * :func:`compute_factors` —— 主入口：raw OHLCV → 含技术指标的 DataFrame；
    * 9 个内部辅助函数：``_sma / _ema / _macd / _boll / _rsi / _kdj / _cci / _wr / _atr / _obv``。

设计要点（与 spec.md "因子生成" Requirement 严格对齐）：
    * **FeatureSpec 真做配置**（v2 关键）：**所有**窗口从 ``factor_spec`` 字段读，
      **不**写死；
    * 保留原始 OHLCV 5 列 + 添加 29 个指标列（默认 FeatureSpec 下）；
    * 早期 NaN 来自预热期（前 60 行），下游 ``dropna()`` 兜底；
    * 全部自实现，**不**引入 ta-lib / pandas-ta。

严格公式（spec.md §9）：
    * MA / EMA / MACD / BOLL / RSI / CCI / WR / ATR —— 业内标准；
    * KDJ 严格：``RSV = (close - low_n) / (high_n - low_n) * 100``,
      ``K = SMA(RSV, m1)``，``D = SMA(K, m2)``，``J = 3K - 2D``，
      K/D 前一日默认 50；
    * OBV 严格：``obv[0] = 0``，
      ``obv[i] = obv[i-1] + sign(close[i] - close[i-1]) * tick_volume[i]``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import FeatureSpec

# ============================================================
# 内部辅助函数（自实现）
# ============================================================


def _sma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均：``series.rolling(n).mean()``。

    前 n-1 行 NaN。
    """
    return series.rolling(window=n, min_periods=n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均：``series.ewm(span=n, adjust=False).mean()``。

    ``adjust=False`` 保证：
        * 第 0 行 == series[0]；
        * 复现 EWM 标准递推；
    """
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    """MACD：返回 ``[dif, dea, hist]``。

    ``dif = EMA(close, fast) - EMA(close, slow)``
    ``dea = EMA(dif, signal)``
    ``hist = (dif - dea) * 2``（业内常见 ×2 因子，保持量级与柱状图一致）。
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def _boll(
    close: pd.Series, n: int, k: float
) -> pd.DataFrame:
    """BOLL：返回 ``[mid, up, low]``。

    ``mid = SMA(close, n)``
    ``up = mid + k * std``
    ``low = mid - k * std``
    """
    mid = _sma(close, n)
    std = close.rolling(window=n, min_periods=n).std(ddof=0)
    up = mid + k * std
    low = mid - k * std
    return pd.DataFrame({"mid": mid, "up": up, "low": low})


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑 RSI。

    ``delta = close.diff()``
    ``gain = max(delta, 0).ewm(alpha=1/n, adjust=False).mean()``
    ``loss = max(-delta, 0).ewm(alpha=1/n, adjust=False).mean()``
    ``RS = gain / loss``（loss=0 时 RS = inf，RSI = 100）
    ``RSI = 100 - 100 / (1 + RS)``（RS=0 时 RSI=0）
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # loss=0 时 rs=NaN → rsi=NaN，填 100（与业内 RSI 边界一致）
    rsi = rsi.fillna(100.0)
    return rsi


def _kdj(
    high: pd.Series, low: pd.Series, close: pd.Series,
    n: int, m1: int, m2: int
) -> pd.DataFrame:
    """KDJ：返回 ``[k, d, j]``。

    ``RSV = (close - low_n) / (high_n - low_n) * 100``
    ``K = SMA(RSV, m1)``，初始 K[0] = 50（业内标准）
    ``D = SMA(K, m2)``，初始 D[0] = 50
    ``J = 3K - 2D``
    """
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0.0, np.nan) * 100.0
    rsv = rsv.fillna(50.0)  # high==low 时 RSV 默认 50

    # 递推 SMA：K[i] = (RSV[i] + (m1-1) * K[i-1]) / m1
    # 等价于 K = rsv.ewm(alpha=1/m1, adjust=False).mean() 但需要特殊处理初始值
    k_arr = np.full(len(close), 50.0)
    rsv_arr = rsv.values
    for i in range(1, len(close)):
        if np.isnan(rsv_arr[i]):
            k_arr[i] = k_arr[i - 1]
        else:
            k_arr[i] = (rsv_arr[i] + (m1 - 1) * k_arr[i - 1]) / m1
    k = pd.Series(k_arr, index=close.index)

    d_arr = np.full(len(close), 50.0)
    k_vals = k.values
    for i in range(1, len(close)):
        if np.isnan(k_vals[i]):
            d_arr[i] = d_arr[i - 1]
        else:
            d_arr[i] = (k_vals[i] + (m2 - 1) * d_arr[i - 1]) / m2
    d = pd.Series(d_arr, index=close.index)

    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """CCI（Commodity Channel Index）。

    ``TP = (high + low + close) / 3``
    ``MA = SMA(TP, n)``
    ``MD = SMA(|TP - MA|, n)``
    ``CCI = (TP - MA) / (0.015 * MD)``
    """
    tp = (high + low + close) / 3.0
    ma = _sma(tp, n)
    md = tp.rolling(window=n, min_periods=n).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (tp - ma) / (0.015 * md.replace(0.0, np.nan))
    return cci


def _wr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Williams %R。

    ``WR = (high_n - close) / (high_n - low_n) * -100``
    """
    high_n = high.rolling(window=n, min_periods=n).max()
    low_n = low.rolling(window=n, min_periods=n).min()
    wr = (high_n - close) / (high_n - low_n).replace(0.0, np.nan) * -100.0
    return wr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑 ATR。

    ``TR = max(high - low, |high - close_prev|, |low - close_prev|)``
    ``ATR = TR.ewm(alpha=1/n, adjust=False).mean()``
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr


def _obv(close: pd.Series, tick_volume: pd.Series) -> pd.Series:
    """OBV（严格公式）。

    ``obv[0] = 0``
    ``obv[i] = obv[i-1] + sign(close[i] - close[i-1]) * tick_volume[i]``
    """
    delta = close.diff()
    sign = np.sign(delta.values)
    sign[0] = 0  # 第一个无前值，强制 0
    # NaN 处理：delta[0] = NaN → sign[0] = 0（已设）
    signed_vol = sign * tick_volume.values
    obv = pd.Series(signed_vol, index=close.index).cumsum()
    return obv


# ============================================================
# 主入口
# ============================================================


def compute_factors(factor_spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标，保留原始 OHLCV 列 + 添加指标列。

    输入 ``df`` 必须含 ``[open, high, low, close, tick_volume]`` 5 列（顺序无关）。
    所有指标窗口来自 ``factor_spec`` 字段。

    Args:
        factor_spec: 因子规格（含 12 字段，详见 :class:`FeatureSpec`）。
        df: raw OHLCV DataFrame，索引 time 或整数。

    Returns:
        新 DataFrame：原始 5 列 + 指标列。
        默认 FeatureSpec 下，shape = ``(n, 5 + 29) = (n, 34)``。
        前 60 行部分指标列 NaN（预热期），下游 ``dropna()`` 兜底。
    """
    required = {"open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_factors 缺必要列: {sorted(missing)}"
        )

    open_s = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df["tick_volume"]

    out = pd.DataFrame(index=df.index)
    # 保留原始列
    out["open"] = open_s
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["tick_volume"] = vol

    # ---- 趋势：MA / EMA / MACD / BOLL ----
    for n in factor_spec.ma_windows:
        out[f"ma{n}"] = _sma(close, n)
    for n in factor_spec.ema_windows:
        out[f"ema{n}"] = _ema(close, n)
    fast, slow, sig = factor_spec.macd_params
    macd_df = _macd(close, fast, slow, sig)
    out["macd_dif"] = macd_df["dif"]
    out["macd_dea"] = macd_df["dea"]
    out["macd_hist"] = macd_df["hist"]
    n_boll, k_boll = factor_spec.boll_params
    boll_df = _boll(close, n_boll, k_boll)
    out["boll_mid"] = boll_df["mid"]
    out["boll_up"] = boll_df["up"]
    out["boll_low"] = boll_df["low"]
    # BOLL 带宽（v2 RANGE 类，delta）
    # 钳位到 0：浮点误差下 std=0 时 up=mid, low=mid，差可能为 -0.0
    out["boll_bw_up"] = (out["boll_up"] - out["boll_mid"]).clip(lower=0.0)
    out["boll_bw_dn"] = (out["boll_mid"] - out["boll_low"]).clip(lower=0.0)

    # ---- 动量：RSI / KDJ / CCI / WR ----
    for n in factor_spec.rsi_windows:
        out[f"rsi{n}"] = _rsi(close, n)
    n_kdj, m1, m2 = factor_spec.kdj_params
    kdj_df = _kdj(high, low, close, n_kdj, m1, m2)
    out["kdj_k"] = kdj_df["k"]
    out["kdj_d"] = kdj_df["d"]
    out["kdj_j"] = kdj_df["j"]
    out[f"cci{factor_spec.cci_window}"] = _cci(high, low, close, factor_spec.cci_window)
    out[f"wr{factor_spec.wr_window}"] = _wr(high, low, close, factor_spec.wr_window)

    # ---- 波动：ATR ----
    out[f"atr{factor_spec.atr_window}"] = _atr(high, low, close, factor_spec.atr_window)

    # ---- 量：OBV / VMA ----
    out["obv"] = _obv(close, vol)
    for n in factor_spec.vma_windows:
        out[f"vma{n}"] = _sma(vol, n)

    return out
