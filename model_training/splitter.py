"""splitter —— 时间序列切分（外层 + 内层 + time_aware + TimeSeriesSplit 保留）。

提供：
    * :func:`walk_forward_splits` —— 外层（评估用）
    * :func:`inner_walk_forward_splits` —— 内层（OOF 用）
    * :func:`time_aware_splits` —— 处理非等间隔数据
    * :func:`list_splits` —— 导出 SplitSpec 列表（用于报告）

设计要点（与 spec.md "Walk-forward" Requirement 严格对齐）：
    * **禁 KFold / StratifiedKFold**（标准 K-fold 泄露未来信息）；
    * **保留 TimeSeriesSplit**（v2 适度放宽）—— 虽非推荐但作为 fallback；
    * 严格按时间顺序，``test_idx.min() > train_idx.max()``（无重叠）；
    * 末段尾部样本因 floor 丢失（**v2 文档化**）；
    * 内层 ``train_size_frac``（v2 重命名，与外层命名统一）；
    * 公共 API grep 验证：splitter.py **不**导出 ``KFold`` / ``StratifiedKFold``。

依赖（不强制导入）：
    * :class:`sklearn.model_selection.TimeSeriesSplit` —— 保留作为可选 fallback。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import SplitSpec


# ============================================================
# 外层 walk-forward（评估用）
# ============================================================


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    train_size_frac: float = 0.6,
    expanding: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """外层 walk-forward 时间序列切分（评估用）。

    算法：
        * ``initial_train = floor(n_samples * train_size_frac)``；
        * ``test_size = floor((n_samples - initial_train) / n_splits)``；
        * 末段尾部样本因 floor 丢失（**v2 文档化**）；
        * ``expanding=True``：train 窗口递增；
        * ``expanding=False``（sliding）：train 窗口固定，平移。

    Args:
        n_samples: 样本总数。
        n_splits: 折数。
        train_size_frac: 初始 train 大小（占总样本比例）。
        expanding: True = 扩展窗口；False = 滑动窗口。

    Returns:
        切分列表 ``[(train_idx, test_idx), ...]``，长度 == n_splits。

    Raises:
        ValueError: 参数不合法。
    """
    if n_samples < 2:
        raise ValueError(f"n_samples={n_samples} 必须 >= 2")
    if n_splits < 1:
        raise ValueError(f"n_splits={n_splits} 必须 >= 1")
    if not (0 < train_size_frac < 1):
        raise ValueError(f"train_size_frac={train_size_frac} 必须在 (0, 1) 区间")

    initial_train = int(math.floor(n_samples * train_size_frac))
    if initial_train < 1:
        raise ValueError(
            f"initial_train={initial_train} 必须 >= 1（n_samples={n_samples}, "
            f"train_size_frac={train_size_frac}）"
        )
    n_test_total = n_samples - initial_train
    if n_test_total < n_splits:
        raise ValueError(
            f"n_test_total={n_test_total} 必须 >= n_splits={n_splits}"
        )
    test_size = int(math.floor(n_test_total / n_splits))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        if expanding:
            train_start = 0
            train_end = initial_train + i * test_size
        else:
            train_start = i * test_size
            train_end = train_start + initial_train
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples:
            break
        train_idx = np.arange(train_start, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        # 严格无重叠
        if train_idx.size and test_idx.size:
            assert train_idx.max() < test_idx.min(), "train 与 test 重叠"
        splits.append((train_idx, test_idx))
    return splits


# ============================================================
# 内层 walk-forward（OOF 用）
# ============================================================


def inner_walk_forward_splits(
    n_train: int,
    n_inner_splits: int = 3,
    train_size_frac: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """内层 walk-forward 时间序列切分（OOF 生成用）。

    与外层同结构，更小窗口。``train_size_frac`` 与外层命名统一（v2 修正，
    旧名 ``min_train_frac`` 已弃）。

    Args:
        n_train: 外层某折 train 集的样本数。
        n_inner_splits: 内层折数。
        train_size_frac: 初始 train 大小比例（默认 0.5）。

    Returns:
        内层切分列表，长度 == n_inner_splits。

    Note:
        * 头部 ``initial_train_inner`` 行无 OOF（永远在 train 中，从不出现在 test）；
        * 上游 ``out_of_fold_predict`` 据此生成 ``valid_oof_mask`` 标记头部。
    """
    return walk_forward_splits(
        n_train, n_splits=n_inner_splits,
        train_size_frac=train_size_frac,
        expanding=True,
    )


# ============================================================
# 时间感知切分（处理非等间隔数据）
# ============================================================


def time_aware_splits(
    times: pd.DatetimeIndex,
    n_splits: int = 5,
    train_size_frac: float = 0.6,
    expanding: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """时间感知切分（处理节假日/停盘等非等间隔数据）。

    算法：
        1. 按 ``times`` 排序（按时间戳）；
        2. 走 :func:`walk_forward_splits`（按排序后的位置索引）。

    Args:
        times: DatetimeIndex（任意顺序均可，会内部排序）。
        n_splits: 折数。
        train_size_frac: 初始 train 大小比例。
        expanding: True = 扩展窗口；False = 滑动窗口。

    Returns:
        切分列表，索引是 ``times`` **排序后**的位置。
    """
    n = len(times)
    if n == 0:
        return []
    # 排序索引
    order = np.argsort(times.values)
    splits_positions = walk_forward_splits(
        n, n_splits=n_splits,
        train_size_frac=train_size_frac,
        expanding=expanding,
    )
    # 把位置索引映射回原始顺序
    out = []
    for train_pos, test_pos in splits_positions:
        train_idx = order[train_pos]
        test_idx = order[test_pos]
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        out.append((train_idx, test_idx))
    return out


# ============================================================
# SplitSpec 导出（用于报告）
# ============================================================


def list_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    label_prefix: str = "fold",
    times: Optional[pd.DatetimeIndex] = None,
) -> List[SplitSpec]:
    """把 ``(train_idx, test_idx)`` 列表转成 :class:`SplitSpec` 列表。

    Args:
        splits: walk_forward_splits 输出。
        label_prefix: SplitSpec.label 前缀（默认 "fold"）。
        times: 可选 DatetimeIndex，用于填 time_range_train / time_range_test。

    Returns:
        SplitSpec 列表。
    """
    out: List[SplitSpec] = []
    for i, (tr, te) in enumerate(splits):
        tr_range = (int(tr.min()), int(tr.max() + 1)) if tr.size else (0, 0)
        te_range = (int(te.min()), int(te.max() + 1)) if te.size else (0, 0)
        tr_time = ("", "")
        te_time = ("", "")
        if times is not None:
            if tr.size:
                tr_time = (str(times[tr[0]]), str(times[tr[-1]]))
            if te.size:
                te_time = (str(times[te[0]]), str(times[te[-1]]))
        out.append(
            SplitSpec(
                label=f"{label_prefix}_{i}",
                train_idx_range=tr_range,
                test_idx_range=te_range,
                time_range_train=tr_time,
                time_range_test=te_time,
                n_train=int(tr.size),
                n_test=int(te.size),
                meta={"fold_idx": i},
            )
        )
    return out


# ============================================================
# API 限制自检（v2 适度放宽）
# ============================================================

# 本模块**不**从 sklearn 导入 KFold / StratifiedKFold；
# 公共 API grep 验证：splitter.py 顶层无 `from sklearn.model_selection import KFold`。
# TimeSeriesSplit 保留（v2 适度放宽，作为可选 fallback）。
# 用户可手动：
#   from sklearn.model_selection import TimeSeriesSplit
# 但 walk_forward_splits 是推荐 API。
