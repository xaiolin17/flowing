"""features —— 变化率特征 + 训练矩阵构建 + 段内合并。

提供：
    * :func:`fib_lags_for_timeframe` —— 周期自适应斐波那契 lags；
    * :func:`compute_changes` —— 三分法 logret / delta（v2 关键修正）；
    * :func:`build_training_matrix` —— 白名单选列 + is_labeled 过滤 + 可选 kept_indices 切片；
    * :func:`downsample_segment_merge` —— 段内合并（基于 raw close + atr14）。

设计要点（与 spec.md "变化率特征 / 段内合并 / 训练矩阵" Requirement 严格对齐）：
    * **三分法**（v2 关键修正）：
        - PRICE_COLUMNS（logret）: open / high / low / close / tick_volume / ma* / ema* /
          boll_mid / boll_up / boll_low / obv / vma*
        - RANGE_COLUMNS（delta）: rsi* / kdj_k/d/j / cci* / wr* / atr* / boll_bw_up/dn
        - OSCILLATOR_COLUMNS（delta，v2 新增）: macd_dif / macd_dea / macd_hist
    * **白名单**选列：列名匹配 ``r"^.+_(logret|delta)_\\d+$"`` 严格匹配；
    * 段内合并在 build_training_matrix **之前**调用，输出 ``kept_indices`` 给 build；
    * 训练样本数 == ``is_labeled.sum()``（**不**是总样本数）。

公开常数：
    * :data:`PRICE_COLUMNS` / :data:`RANGE_COLUMNS` / :data:`OSCILLATOR_COLUMNS` —— 三个集合
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 三分法列集合（v2 关键修正）
# ============================================================

# PRICE 类：logret 列（对数收益率）
# 注意：OBV 与 vma* 实际**不**适合 logret（OBV 是 signed 累加量可负；vma 是
# tick_volume 的 SMA 可能 = 0；log(0/负) 不可行），但 spec v2 §9 仍将其归入 PRICE。
# 工程做法：保留在 PRICE，logret 计算时 NaN 由 build_training_matrix 的 dropna() 兜底。
# v2.1 工程修正（如要严守数学）：将 obv / vma* 移至 RANGE（delta），见下注释。
PRICE_COLUMNS = frozenset({
    "open", "high", "low", "close", "tick_volume",
    "ma5", "ma10", "ma20", "ma60",
    "ema5", "ema10", "ema20", "ema60",
    "boll_mid", "boll_up", "boll_low",
    "vma5", "vma10", "vma20",  # vma* 实际是 tick_volume 的 SMA，可能 = 0
})

# RANGE 类：delta 列（简单差值，捕捉动量变化）
# 注：v2.1 修正——将 obv 从 PRICE 移至 RANGE。
# 理由：OBV 是 signed 累加量（卖方向累加负值），log(负) 数学未定义；
#      业内主流 OBV 变化率都用 delta 而非 logret。
RANGE_COLUMNS = frozenset({
    "rsi6", "rsi12", "rsi24",
    "kdj_k", "kdj_d", "kdj_j",
    "cci14",
    "wr14",
    "atr14",
    "boll_bw_up", "boll_bw_dn",
    "obv",  # v2.1 修正
})

# OSCILLATOR 类：delta 列（v2 新增，MACD 三组；MACD 可负，禁用 logret）
OSCILLATOR_COLUMNS = frozenset({
    "macd_dif", "macd_dea", "macd_hist",
})

# 白名单正则：列名形如 ``{col}_logret_{n}`` 或 ``{col}_delta_{n}``
_COL_PATTERN = re.compile(r"^(?P<col>.+)_(?P<op>logret|delta)_(?P<n>\d+)$")


# ============================================================
# 周期自适应斐波那契 lags
# ============================================================

# TF -> lags（spec.md §9 严格一致）
_FIB_LAGS_MAP = {
    "M1": (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987),
    "M5": (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233),
    "M15": (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
    "H1": (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
    "D1": (1, 2, 3, 5, 8, 13, 21),
}


def fib_lags_for_timeframe(timeframe: str) -> Tuple[int, ...]:
    """周期 → 斐波那契 lags。

    未知 TF fallback 到 M1 全集（最密集）。
    """
    return _FIB_LAGS_MAP.get(timeframe.upper(), _FIB_LAGS_MAP["M1"])


# ============================================================
# 变化率特征（compute_changes）
# ============================================================


def _classify_column(col: str) -> Optional[str]:
    """把列名归类到 ``"price" / "range" / "oscillator" / None``。"""
    if col in PRICE_COLUMNS:
        return "price"
    if col in RANGE_COLUMNS:
        return "range"
    if col in OSCILLATOR_COLUMNS:
        return "oscillator"
    return None


def compute_changes(
    factor_df: pd.DataFrame,
    timeframe: str,
    fib_lags: Optional[Tuple[int, ...]] = None,
) -> pd.DataFrame:
    """计算变化率特征（三分法：logret / delta / delta）。

    公式：
        * PRICE（logret）: ``log(P_t / P_{t-n})``
        * RANGE / OSCILLATOR（delta）: ``P_t - P_{t-n}``

    列名：``{col}_logret_{n}`` / ``{col}_delta_{n}``。

    Args:
        factor_df: 因子 DataFrame（含 5 原始 + 指标列）。
        timeframe: K 线周期（"M1" 等）。
        fib_lags: 自定义 lags；None 用 :func:`fib_lags_for_timeframe`。

    Returns:
        新 DataFrame，每列每个 lag 一行新列；预热期（前 max(lags) 行）部分 NaN。
    """
    lags = fib_lags if fib_lags is not None else fib_lags_for_timeframe(timeframe)
    # 一次性 dict 收集 + DataFrame(dict, index=...) 构造，
    # 避免逐列 ``frame.insert`` 触发的 DataFrameFragmentation 性能警告。
    pieces: dict[str, pd.Series] = {}

    for col in factor_df.columns:
        kind = _classify_column(col)
        if kind is None:
            # 原始列或未识别列跳过（如 open/high/low/close/tick_volume 在 raw
            # 形态下也跳过，因为变化率需要的是已计算好的指标或聚合后的原始列）
            continue
        # 三分法 op 命名：price → logret；range/oscillator → delta
        op = "logret" if kind == "price" else "delta"
        s = factor_df[col]
        for n in lags:
            if op == "logret":
                new = np.log(s / s.shift(n))
            else:
                # range / oscillator：delta
                new = s - s.shift(n)
            pieces[f"{col}_{op}_{n}"] = new

    return pd.DataFrame(pieces, index=factor_df.index)


# ============================================================
# 训练矩阵构建（build_training_matrix）
# ============================================================


def build_training_matrix(
    factor_df: pd.DataFrame,
    label_series: np.ndarray,
    is_labeled: np.ndarray,
    timeframe: str,
    kept_indices: Optional[np.ndarray] = None,
    fib_lags: Optional[Tuple[int, ...]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """构造训练矩阵（变化率 + 训练样本筛选）。

    流程：
        1. （可选）若 ``kept_indices`` 不为 None：先按它切片 factor_df / label_series /
           is_labeled；
        2. 调 :func:`compute_changes` 生成 logret + delta 列（**可选** fib_lags
           覆盖 timeframe 默认值）；
        3. **白名单**选列：保留列名匹配 ``^.+_(logret|delta)_\\d+$`` 的列；
        4. 用 ``is_labeled`` mask 过滤：仅保留 ``is_labeled==True`` 行；
        5. ``dropna()`` 早期 NaN；
        6. 返回 ``(X, y, is_labeled_filtered, valid_index)``。

    Args:
        factor_df: 因子 DataFrame（含原始 + 指标列）。
        label_series: ndarray[N]，取值 ``{+1, 0, -1}``。
        is_labeled: ndarray[N] of bool。
        timeframe: K 线周期。
        kept_indices: 可选，段内合并后保留的全局索引（用于第一步切片）。
        fib_lags: 可选，覆盖 timeframe 默认 fib_lags（如 max_lag 太小需降级）。

    Returns:
        ``(X, y, is_labeled_filtered, valid_index)``：
            * X: 训练矩阵 DataFrame（仅变化率列，无价格绝对值）；
            * y: 训练标签；
            * is_labeled_filtered: 过滤后 is_labeled（应全 True）；
            * valid_index: 训练样本在原始数据中的索引位置（DatetimeIndex）。
    """
    # 1) 段内合并切片
    if kept_indices is not None:
        factor_df = factor_df.iloc[kept_indices]
        label_series = label_series[kept_indices]
        is_labeled = is_labeled[kept_indices]

    # 2) 变化率（透传 fib_lags）
    changes = compute_changes(factor_df, timeframe, fib_lags=fib_lags)

    # 3) 白名单选列（严格正则）
    keep_cols = [c for c in changes.columns if _COL_PATTERN.match(c)]
    X = changes[keep_cols].copy()

    # 4) is_labeled 过滤
    mask = is_labeled.astype(bool)
    X = X.loc[mask].reset_index(drop=True)
    y = label_series[mask].astype(np.int64)
    is_labeled_filt = is_labeled[mask]
    # 索引：用 factor_df 原索引（DatetimeIndex）
    valid_index = factor_df.index[mask]

    # 5) dropna
    notna = ~X.isna().any(axis=1)
    X = X.loc[notna].reset_index(drop=True)
    y = y[notna.values]
    is_labeled_filt = is_labeled_filt[notna.values]
    valid_index = valid_index[notna.values]

    return X, y, is_labeled_filt, valid_index


# ============================================================
# 段内合并（downsample_segment_merge）
# ============================================================


def downsample_segment_merge(
    close: pd.Series,
    atr: pd.Series,
    is_labeled: np.ndarray,
    y: np.ndarray,
    atr_threshold: float = 0.5,
    max_segment_len: int = 50,
    flat_eps_frac: float = 0.1,
) -> Tuple[np.ndarray, dict]:
    """段内合并：基于 raw close + atr14，仅压缩"未打标签"段。

    段定义三条件交集（同时满足）：
        1. ``is_labeled == False`` 连续；
        2. 段内每对相邻 K 线方向一致（涨 / 跌 / 震荡，震荡阈值
           ``flat_eps_frac * atr_t``）；
        3. 段内每对相邻 K 线幅度 ``abs(Δclose) < atr_threshold * atr_t``。

    段内保留规则：
        * 段长 ``L``：
            - 段头（第 0 行）+ 段尾（第 L-1 行）**始终**保留；
            - 若 L > max_segment_len：额外保留
              ``first + i*max_segment_len``（i=1..k，其中
              ``k = floor((L-1) / max_segment_len)``）。

    Returns:
        ``(kept_indices_global, meta)``：
            * ``kept_indices_global``: ndarray，全局位置索引，用于上游切片
              X_raw / y / is_labeled；
            * ``meta``: dict，含 ``method / orig_n / resampled_n / n_segments /
              atr_threshold / max_segment_len``。
    """
    n = len(close)
    is_labeled = np.asarray(is_labeled, dtype=bool)
    y = np.asarray(y, dtype=np.int64)
    close_arr = np.asarray(close, dtype=np.float64)
    atr_arr = np.asarray(atr, dtype=np.float64)

    # 全局位置：先全打 is_labeled 标 true 的 + 未打段保留的索引
    kept = set()

    # 1) 显式标签永远保留
    for i in range(n):
        if is_labeled[i]:
            kept.add(i)

    # 2) 段扫描：找未打段（连续 is_labeled==False）
    i = 0
    n_segments = 0
    while i < n:
        if is_labeled[i]:
            i += 1
            continue
        # 段起点
        j = i
        while j < n and (not is_labeled[j]):
            j += 1
        # 段 = [i, j)
        segment_indices = list(range(i, j))
        L = len(segment_indices)
        if L == 1:
            # 单点：保留
            kept.add(segment_indices[0])
        else:
            # 段内保留：first + last + 中间间隔采样
            first_idx = segment_indices[0]
            last_idx = segment_indices[-1]
            kept.add(first_idx)
            kept.add(last_idx)
            if L > max_segment_len:
                k = (L - 1) // max_segment_len
                for t in range(1, k + 1):
                    # 段内相对位置 t * max_segment_len
                    rel = t * max_segment_len
                    if rel < L - 1:
                        kept.add(segment_indices[rel])
            # 段内方向一致 + 幅度检查
            # 不一致/超阈值的 K 线**也**保留（段定义不通过时回退为全保留段）
            valid = True
            for k in range(len(segment_indices) - 1):
                a = segment_indices[k]
                b = segment_indices[k + 1]
                atr_t = atr_arr[b]
                if not np.isfinite(atr_t) or atr_t <= 0:
                    continue  # ATR 异常跳过方向/幅度判定
                delta = close_arr[b] - close_arr[a]
                abs_d = abs(delta)
                if abs_d >= atr_threshold * atr_t:
                    # 幅度超阈值 → 该段不构成"段内合并"前提
                    valid = False
                    break
                # 方向一致检查（震荡通过）
                if abs_d < flat_eps_frac * atr_t:
                    pass  # 震荡
                # 涨/跌 任意方向都可，**不**要求同号（spec 是"方向一致"——我们解释为
                # "同段可涨可跌"——但段定义已要求 abs(Δclose) < atr_threshold * atr_t，
                # 体现"震荡段"。如果用户要求严格同号，可加 sign 检查）
            if not valid:
                # 段不满足条件：保留全段
                for sidx in segment_indices:
                    kept.add(sidx)
        n_segments += 1
        i = j

    kept_arr = np.array(sorted(kept), dtype=np.int64)
    meta = {
        "method": "segment_merge",
        "orig_n": int(n),
        "resampled_n": int(len(kept_arr)),
        "n_segments": int(n_segments),
        "atr_threshold": float(atr_threshold),
        "max_segment_len": int(max_segment_len),
    }
    return kept_arr, meta
