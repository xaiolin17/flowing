"""trainer —— LightGBM 三分类 + label_map 双向 + focal loss + OOF。

提供：
    * :func:`label_map` —— 双向 label 映射 ``{+1,0,-1} ↔ {0,1,2}``；
    * :func:`focal_loss_multiclass` —— LightGBM 自定义多分类 focal loss objective；
    * :func:`train_lightgbm` —— 主训练函数（延迟导入 lightgbm）；
    * :func:`out_of_fold_predict` —— 内层 walk-forward-CV → OOF + valid_mask。

设计要点（与 spec.md "训练" Requirement 严格对齐）：
    * **延迟导入** lightgbm：仅在 :func:`train_lightgbm` 函数体内 import；
      缺包时给清晰 ImportError 提示（**不**自动 pip）；
    * 随机种子固定：``seed=42, deterministic=True``；
    * 复现 caveat：LightGBM 3.x + 多线程 + 多折下，``deterministic=True``
      **不一定**完全可复现（多折累积随机性可能引入微差）；
    * OOF 头部 ``initial_train_inner`` 样本无 OOF，``valid_oof_mask`` 标记；
    * OOF 无效位置填 NaN（**不**用 0 占位，避免 LR 学到"无效=0 类"）。

依赖：lightgbm（用户手动 pip，**不**在仓库内自动装）。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Label mapping
# ============================================================

# 标准映射：+1 → 0, 0 → 1, -1 → 2
LABEL_MAP_DEFAULT: Dict[int, int] = {1: 0, 0: 1, -1: 2}
INVERSE_MAP_DEFAULT: Dict[int, int] = {0: 1, 1: 0, 2: -1}


def label_map(
    y: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """把 ``y ∈ {+1, 0, -1}`` 映射到 ``{0, 1, 2}``，返回双向字典。

    Args:
        y: 标签数组，取值 ``{+1, 0, -1}``。

    Returns:
        ``(mapped_y, label_map, inverse_map)``：
            * mapped_y: ndarray，相同长度，取值 ``{0, 1, 2}``；
            * label_map: ``{+1: 0, 0: 1, -1: 2}``；
            * inverse_map: ``{0: +1, 1: 0, 2: -1}``。
    """
    y = np.asarray(y, dtype=np.int64)
    # 用 int64 最小值作 sentinel（避免与合法标签 -1 重叠，
    # 防止 y 含 {-2, -1, 0, 1, 2} 等其他值时被 -1 误判为 invalid）
    mapped = np.full_like(y, np.iinfo(np.int64).min)
    for k, v in LABEL_MAP_DEFAULT.items():
        mapped[y == k] = v
    if (mapped == np.iinfo(np.int64).min).any():
        bad = np.unique(y[mapped == np.iinfo(np.int64).min])
        raise ValueError(f"y 含未定义标签: {bad.tolist()}")
    return mapped, dict(LABEL_MAP_DEFAULT), dict(INVERSE_MAP_DEFAULT)


# ============================================================
# Focal loss objective（LightGBM 自定义 multiclass focal loss）
# ============================================================


def focal_loss_multiclass(
    gamma: float = 2.0,
    num_class: int = 3,
):
    """构造 multiclass focal loss objective（LightGBM 自定义）。

    公式（per sample, per class）:
        ``p_i = softmax(z_i)``
        ``L_i = -alpha * (1 - p_yi)^gamma * log(p_yi)``
        （``alpha`` 默认 1.0；本实现简化为 alpha=1）

    Returns:
        可调用对象，签名 ``(y_pred, dataset) -> (grad, hess)``，
        直接传给 ``lgb.train(params={'objective': focal_loss_multiclass(2.0)})``。
    """
    def _focal_obj(y_pred, dataset):
        # y_pred: ndarray[n, num_class]，raw scores
        # dataset: lgb.Dataset，含 label
        y_true = dataset.get_label().astype(np.int64)
        n = len(y_true)
        z = y_pred.reshape(n, num_class)
        # softmax
        z_max = z.max(axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)
        # one-hot
        y_onehot = np.zeros((n, num_class))
        y_onehot[np.arange(n), y_true] = 1.0
        # focal: p_t = sum(y_onehot * p, axis=1)
        p_t = (y_onehot * p).sum(axis=1)
        # focal factor: (1 - p_t)^gamma
        focal_factor = np.power(np.clip(1.0 - p_t, 0.0, 1.0), gamma)
        # grad/hess
        # 简化梯度：dL/dz_i = (p_i - y_i) * focal_factor (per sample, per class)
        # 当 i = y_true: dL/dz_y = (p_y - 1) * focal_factor
        # 当 i != y_true: dL/dz_i = p_i * focal_factor
        # 这是 softmax CE 梯度的 focal 变种
        grad = (p - y_onehot) * focal_factor[:, None]
        # 简化 Hessian：|p_i * (1 - p_i) * focal_factor| 近似
        hess = p * (1.0 - p) * focal_factor[:, None]
        # 保证非负（hess 要求非负）
        hess = np.maximum(hess, 1e-7)
        return grad.flatten(), hess.flatten()
    return _focal_obj


# ============================================================
# LightGBM 训练（延迟导入）
# ============================================================


def _default_params(num_class: int = 3, random_state: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "num_iterations": 200,
        "seed": random_state,
        "deterministic": True,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    random_state: int = 42,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> Tuple[object, Dict[int, int], Dict[int, int], dict]:
    """训练 LightGBM 三分类。

    关键不变量：
        * 延迟导入 lightgbm（v2 修正）；
        * 缺包抛 ImportError 含"请先手动 pip install lightgbm"；
        * ``use_focal=True`` 时切换 objective 为 :func:`focal_loss_multiclass`；
        * 有 X_val 时启用 ``early_stopping_rounds=20``；
        * label_map 双向正确。

    Args:
        X_train: 训练特征 DataFrame。
        y_train: 训练标签 ``∈ {+1, 0, -1}``。
        X_val: 验证特征（用于 early stopping，可选）。
        y_val: 验证标签。
        params: 自定义 LightGBM params；None 用默认。
        random_state: 随机种子。
        use_focal: True 时切换 focal loss objective。
        focal_gamma: focal loss gamma。

    Returns:
        ``(booster, label_map, inverse_map, history)``。

    Raises:
        ImportError: lightgbm 未装。
    """
    # 关键：延迟导入
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm 未安装；请先手动 pip install lightgbm"
        ) from e

    # label_map
    y_mapped, lmap, inv_map = label_map(y_train)
    num_class = 3

    p = dict(params) if params is not None else _default_params(num_class, random_state)
    p["seed"] = random_state
    if use_focal:
        p["objective"] = focal_loss_multiclass(gamma=focal_gamma, num_class=num_class)
        p.pop("metric", None)  # focal 自定义 objective 不直接用 multiclass metric

    # 构造 Dataset
    dtrain = lgb.Dataset(X_train, label=y_mapped)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = []
    if X_val is not None and y_val is not None and len(X_val) > 0:
        y_val_mapped, _, _ = label_map(y_val)
        dval = lgb.Dataset(X_val, label=y_val_mapped, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    callbacks.append(lgb.log_evaluation(period=0))  # 静默

    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=p.get("num_iterations", 200),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    history = {
        "best_iteration": getattr(booster, "best_iteration", None),
        "best_score": getattr(booster, "best_score", None),
    }
    return booster, lmap, inv_map, history


# ============================================================
# OOF 预测（out_of_fold_predict）
# ============================================================


def out_of_fold_predict(
    model_fn: Callable,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF 预测：内层 walk-forward-CV → 每样本来自"不包含它"的折。

    返回 ``(oof_proba, valid_oof_mask)``：
        * ``oof_proba``: ``ndarray[n, n_classes]``，无效位置填 NaN；
        * ``valid_oof_mask``: ``ndarray[n] of bool``，标记哪些行有 OOF。

    关键（v2 修正）：
        * 头部 ``initial_train_inner`` 样本**没有 OOF**（永远在 train 里）；
        * 后续 ``stack`` 据 ``valid_mask`` 过滤训练样本。

    Args:
        model_fn: 训练函数 ``(X_train, y_train) -> fitted model with predict_proba``。
        X: 特征 DataFrame。
        y: 标签数组（传给 model_fn 但**不**用于评估）。
        splits: 内层 walk-forward 切分（来自 :func:`inner_walk_forward_splits`）。
        random_state: 随机种子（透传给 model_fn）。

    Returns:
        ``(oof_proba, valid_oof_mask)``。
    """
    n = len(X)
    # 检测 n_classes
    n_classes = 3
    oof_proba = np.full((n, n_classes), np.nan)
    valid_oof_mask = np.zeros(n, dtype=bool)

    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        m = model_fn(X.iloc[train_idx], y[train_idx])
        proba = m.predict_proba(X.iloc[test_idx])
        # NaN 行不写 proba / valid（避免 stacker 训练时遇 NaN 学到 0 类）
        # 头部 pad（transformer OOF）的 NaN 自然被过滤
        if proba.ndim == 2 and np.isnan(proba).any():
            valid = ~np.isnan(proba).any(axis=1)
            oof_proba[test_idx[valid]] = proba[valid]
            valid_oof_mask[test_idx[valid]] = True
        else:
            oof_proba[test_idx] = proba
            valid_oof_mask[test_idx] = True
    return oof_proba, valid_oof_mask
