"""imbalance —— 数据倾斜处理（3 基线 + 工厂）。

提供：
    * :func:`keep_all` —— 基线 D（不下采样）
    * :func:`keep_all_with_focal_loss` —— 基线 B（focal loss，meta flag）
    * :func:`random_downsample_majority` —— 基线 C（随机下采样 majority）
    * :func:`apply_imbalance` —— 工厂函数（接 :class:`ImbalanceMethod` enum）

段内合并（基线 A）见 :mod:`.features`（基于 raw close + atr14，独立 requirement）。
``dummy_baseline`` 见 :mod:`.evaluator`（基于 ``EvalReport``），不放在本模块。

标签语义（v2 修正）：
    * build_training_matrix 之后，``y == 0`` 全是**显式 0**（未打 0 已被 is_labeled
      过滤掉）；
    * ``y ∈ {+1, -1}`` 是 minority；
    * ``y == 0`` 视为 majority。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from .models import ImbalanceMethod


# ============================================================
# 4 基线 + 1 dummy baseline
# ============================================================


def keep_all(
    X: pd.DataFrame, y: np.ndarray, is_labeled: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """基线 D：不下采样，全保留。

    Args:
        X, y, is_labeled: 训练数据。

    Returns:
        ``(X, y, is_labeled, meta)``，meta 含 ``method='none'``。
    """
    meta = {
        "method": "none",
        "orig_n": int(len(y)),
        "resampled_n": int(len(y)),
        "ratio": 1.0,
        "use_focal": False,
        "random_state": None,
    }
    return X, y, is_labeled, meta


def keep_all_with_focal_loss(
    X: pd.DataFrame, y: np.ndarray, is_labeled: np.ndarray, gamma: float = 2.0
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """基线 B：Focal Loss（不下采样，仅标记 ``use_focal=True``）。

    ``train_lightgbm`` 据 ``meta['use_focal']`` 切换 objective 为
    自定义 multiclass focal loss。

    Args:
        X, y, is_labeled: 训练数据。
        gamma: focal loss gamma 参数（默认 2.0）。

    Returns:
        ``(X, y, is_labeled, meta)``，meta 含 ``use_focal=True, gamma=2.0``。
    """
    meta = {
        "method": "focal",
        "orig_n": int(len(y)),
        "resampled_n": int(len(y)),
        "ratio": 1.0,
        "use_focal": True,
        "gamma": float(gamma),
        "random_state": None,
    }
    return X, y, is_labeled, meta


def random_downsample_majority(
    X: pd.DataFrame, y: np.ndarray, is_labeled: np.ndarray,
    ratio: float = 2.0, random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """基线 C：随机下采样 majority（y=0）到 ``ratio * len(±1 total)``。

    标签语义：
        * ``±1`` 为 minority（保留**全部**）；
        * ``0`` 为 majority（按 ratio 抽）；
        * 抽样后 y 分布近似 ``ratio:1:1``（0: +1: -1 之和）。
    """
    y = np.asarray(y, dtype=np.int64)
    is_labeled = np.asarray(is_labeled, dtype=bool)
    rng = np.random.default_rng(random_state)

    minority_mask = (y == 1) | (y == -1)
    majority_mask = y == 0
    n_minority = int(minority_mask.sum())
    n_majority_target = int(ratio * n_minority)
    n_majority_available = int(majority_mask.sum())
    n_majority_keep = min(n_majority_target, n_majority_available)

    # 从 majority 抽
    majority_indices = np.where(majority_mask)[0]
    if n_majority_keep < len(majority_indices):
        chosen = rng.choice(majority_indices, size=n_majority_keep, replace=False)
    else:
        chosen = majority_indices
    # minority 全保留
    keep_idx = np.concatenate([np.where(minority_mask)[0], chosen])
    keep_idx = np.sort(keep_idx)

    X_new = X.iloc[keep_idx].reset_index(drop=True)
    y_new = y[keep_idx]
    is_lab_new = is_labeled[keep_idx]

    meta = {
        "method": "random_downsample",
        "orig_n": int(len(y)),
        "resampled_n": int(len(keep_idx)),
        "ratio": float(ratio),
        "use_focal": False,
        "random_state": int(random_state),
    }
    return X_new, y_new, is_lab_new, meta


# ============================================================
# 工厂函数
# ============================================================


def apply_imbalance(
    method: ImbalanceMethod,
    X: pd.DataFrame,
    y: np.ndarray,
    is_labeled: np.ndarray,
    **kwargs,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """工厂函数：按 ``method`` 路由到对应基线。

    Args:
        method: :class:`ImbalanceMethod` enum（v2 修正：从 str 改为 enum）。
        X, y, is_labeled: 训练数据。
        **kwargs: 透传给具体基线函数（gamma / ratio / random_state 等）。

    Returns:
        ``(X, y, is_labeled, meta)``。

    Raises:
        ValueError: method 不在 4 个枚举值内。
    """
    if method == ImbalanceMethod.NONE:
        return keep_all(X, y, is_labeled)
    if method == ImbalanceMethod.FOCAL:
        gamma = kwargs.get("gamma", 2.0)
        return keep_all_with_focal_loss(X, y, is_labeled, gamma=gamma)
    if method == ImbalanceMethod.RANDOM_DOWNSAMPLE:
        ratio = kwargs.get("ratio", 2.0)
        random_state = kwargs.get("random_state", 42)
        return random_downsample_majority(X, y, is_labeled, ratio=ratio, random_state=random_state)
    if method == ImbalanceMethod.SEGMENT_MERGE:
        # 段内合并在 build_training_matrix 之前调用（基于 raw data），不在本工厂
        raise ValueError(
            "ImbalanceMethod.SEGMENT_MERGE 应在 build_training_matrix 之前调用 "
            "downsample_segment_merge；本工厂仅处理 build 后的不平衡。"
        )
    raise ValueError(f"未知 ImbalanceMethod: {method}")
