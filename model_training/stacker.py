"""stacker —— Stacking 第二层（仅横向 concat + valid_mask 过滤）。

提供：
    * :func:`stack` —— OOF 概率 + 原始特征 → LR 第二层 meta-learner。

设计要点（与 spec.md "Stacking" Requirement 严格对齐）：
    * **不**做 ``oof_proba * oof_features`` 之类乘积特征；
    * **不**做 logit 变换；
    * **仅**横向 ``np.concatenate([oof_proba, oof_features], axis=1)``；
    * **仅**用 ``valid_mask==True`` 的行训练（v2 关键修正：头部无 OOF 行被剔除）；
    * ``valid_mask`` 全 False 时抛 ``ValueError``，避免 LR 在空集上训出空模型；
    * meta 默认 ``"logistic"``（sklearn LogisticRegression 二次学习）。

依赖：scikit-learn（用户手动 pip，**不**在本项目内自动装）。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def stack(
    oof_proba: np.ndarray,
    y_train: np.ndarray,
    oof_features: np.ndarray,
    valid_mask: np.ndarray,
    meta: str = "logistic",
    random_state: int = 42,
):
    """Stacking 第二层：OOF 概率 + 原始特征 → meta-learner。

    Args:
        oof_proba: ``ndarray[n, n_classes]``，无效位置含 NaN。
        y_train: ``ndarray[n]``，标签 ``∈ {+1, 0, -1}``。
        oof_features: ``ndarray[n, d]``，原始训练特征（与 oof_proba 行数一致）。
        valid_mask: ``ndarray[n] of bool``，True 行进入 meta 训练。
        meta: meta-learner 类型，目前仅支持 ``"logistic"``。
        random_state: 随机种子（透传给 LR）。

    Returns:
        训练好的 meta-learner estimator（带 ``predict_proba``）。

    Raises:
        ValueError: ``valid_mask`` 全 False，或形状不一致，或 meta 不支持。
    """
    # ---------------- 形状校验 ----------------
    n = oof_proba.shape[0]
    if oof_features.shape[0] != n:
        raise ValueError(
            f"oof_proba 形状 {oof_proba.shape[0]} 与 oof_features 形状 "
            f"{oof_features.shape[0]} 不一致"
        )
    if len(y_train) != n:
        raise ValueError(
            f"oof_proba 形状 {oof_proba.shape[0]} 与 y_train 形状 {len(y_train)} 不一致"
        )
    if valid_mask.shape != (n,):
        raise ValueError(
            f"valid_mask 形状 {valid_mask.shape} 应为 ({n},)"
        )
    if valid_mask.dtype != bool:
        raise ValueError(f"valid_mask dtype 应为 bool，实际 {valid_mask.dtype}")
    if not valid_mask.any():
        raise ValueError(
            "valid_mask 全 False：OOF 无任何有效样本，无法训练 meta-learner"
        )
    if meta != "logistic":
        raise ValueError(
            f"不支持的 meta 类型: {meta!r}；目前仅 'logistic'"
        )

    # ---------------- 过滤 + 横向 concat ----------------
    y_used = y_train[valid_mask]
    X_meta = np.concatenate(
        [oof_proba[valid_mask], oof_features[valid_mask]],
        axis=1,
    )

    # ---------------- 训练 meta-learner ----------------
    # 兼容 sklearn 1.2+：multi_class 参数已弃用，1.2+ 默认 multinomial
    estimator = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    estimator.fit(X_meta, y_used)
    return estimator
