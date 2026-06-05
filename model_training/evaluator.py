"""evaluator —— 评估报告（指标 + 简易回测 + dummy baseline + 警告）。

提供：
    * :func:`compute_next_returns` —— 次日简单收益率（末行 NaN）；
    * :func:`evaluate` —— 指标 + 简易回测 → :class:`~model_training.models.EvalReport`；
    * :func:`dummy_baseline` —— DummyClassifier(most_frequent) 基线；
    * :func:`compare_with_dummy` —— f1_delta + is_meaningful + warning；
    * :func:`serialize_eval` —— EvalReport + compare → JSON 落盘。

设计要点（与 spec.md "评估报告" Requirement 严格对齐）：
    * **末行 NaN**：``next_returns`` 最后一行 NaN，``evaluate`` 正确处理
      （mask NaN 后再算指标 / 回测）；
    * **per_class_f1**（v2 新增）：除 precision/recall 外补 F1；
    * **warning**（v2 新增）：``f1_delta < 0.05`` 时
      ``compare_with_dummy`` 返回 dict 含 ``warning`` 字段，提示"模型未显著
      优于多数类 baseline"；
    * **dummy_baseline** 用 ``strategy='most_frequent'``，y_train 拟合，y_test 预测；
    * **serialize_eval** 输出 JSON 字段与 :class:`EvalReport` 一一对应。

依赖：scikit-learn（用户手动 pip，**不**在本项目内自动装）。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .models import EvalReport


# ============================================================
# 次日简单收益率
# ============================================================

def compute_next_returns(close: pd.Series) -> np.ndarray:
    """``next_returns[i] = (close[i+1] - close[i]) / close[i]``。

    末行 NaN（v2 文档化）：``close[i+1]`` 不存在，最后一根无法计算。
    """
    close = pd.Series(close).astype(float)
    ret = close.pct_change().shift(-1)
    return ret.to_numpy(dtype=float)


# ============================================================
# 指标 + 简易回测
# ============================================================

def _signal_from_proba(
    y_pred_proba: np.ndarray,
    inverse_map: Dict[int, int],
) -> np.ndarray:
    """``argmax`` → ``inverse_map`` → 交易信号 ``{+1, 0, -1}``。"""
    argmax = np.argmax(y_pred_proba, axis=1)
    return np.array([inverse_map[int(a)] for a in argmax], dtype=np.int64)


def _mask_valid(y_pred_proba: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    """生成 mask：剔除 NaN 位置（末行 next_returns NaN + OOF 头部 NaN）。"""
    mask = ~np.isnan(next_returns)
    if np.isnan(y_pred_proba).any():
        mask &= ~np.isnan(y_pred_proba).any(axis=1)
    return mask


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    inverse_map: Dict[int, int],
    next_returns: np.ndarray,
    meta: Optional[dict] = None,
) -> EvalReport:
    """评估一次预测：accuracy / f1 / per_class / multi-class AUC / 简易回测。

    Args:
        y_true: ``ndarray[n]``，真实标签 ``∈ {+1, 0, -1}``。
        y_pred_proba: ``ndarray[n, n_classes]``，预测概率。
        inverse_map: ``{0: +1, 1: 0, 2: -1}``。
        next_returns: ``ndarray[n]``，次日收益率（末行 NaN）。
        meta: 附加元数据（如标签映射、模型版本等）。

    Returns:
        :class:`EvalReport` 实例。

    关键（v2）：
        * 末行 next_returns NaN 正确 mask 掉；
        * per_class_f1（v2 新增）；
        * multi-class AUC 用 ``multi_class='ovr'``；
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)

    # mask：剔除 NaN 行
    mask = _mask_valid(y_pred_proba, next_returns)
    if mask.sum() != len(y_true):
        y_true = y_true[mask]
        y_pred_proba = y_pred_proba[mask]
        next_returns = next_returns[mask]

    n = len(y_true)
    if n == 0:
        # 极端：全 NaN
        return EvalReport(
            accuracy=0.0,
            f1_macro=0.0,
            per_class_precision=[0.0, 0.0, 0.0],
            per_class_recall=[0.0, 0.0, 0.0],
            per_class_f1=[0.0, 0.0, 0.0],
            auc=0.5,
            backtest_return=0.0,
            n_samples=0,
            meta=meta or {},
        )

    # signal
    signal = _signal_from_proba(y_pred_proba, inverse_map)

    # 指标
    y_pred = signal  # signal ∈ {+1, 0, -1}，与 y_true 一致
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 0, -1], zero_division=0
    )

    # multi-class AUC（ovr）
    try:
        auc = roc_auc_score(
            np.eye(3)[(y_true + 1).astype(int)],  # 0/1/2
            y_pred_proba,
            multi_class="ovr",
            labels=[0, 1, 2],
        )
    except ValueError:
        # 极端 case：某类缺失
        auc = 0.5

    # 简易回测：signal * next_returns
    strat = signal.astype(float) * next_returns
    backtest_return = float(np.nansum(strat))

    return EvalReport(
        accuracy=float(acc),
        f1_macro=float(f1m),
        per_class_precision=[float(p) for p in prec],
        per_class_recall=[float(r) for r in rec],
        per_class_f1=[float(f) for f in f1c],
        auc=float(auc),
        backtest_return=backtest_return,
        n_samples=n,
        meta=meta or {},
    )


# ============================================================
# Dummy baseline
# ============================================================

def dummy_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    next_returns_test: np.ndarray,
) -> EvalReport:
    """``DummyClassifier(strategy='most_frequent')`` 训练 + 预测 + 评估。

    Args:
        y_train: ``ndarray``，训练标签。
        y_test: ``ndarray``，测试标签。
        next_returns_test: ``ndarray``，测试期次日收益率（末行 NaN）。

    Returns:
        :class:`EvalReport`（多数类 baseline）。
    """
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    next_returns_test = np.asarray(next_returns_test, dtype=float)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    # 用 sklearn 真 proba：多数类 = 1.0，其他类 = 0.0（行和 = 1.0）
    # 三分类时不再是 1/3，AUC/f1_delta 计算正确
    proba = dummy.predict_proba(np.zeros((len(y_test), 1)))
    # sklearn DummyClassifier.classes_ 是 sorted 后的 unique 标签（[-1, 0, 1]），
    # 与内部 label_map 索引顺序 [+1, 0, -1] 不一致 → 必须按 label_map 顺序重排
    # 这样 ``argmax + inverse_map`` 才能映射回正确的信号
    classes_list = dummy.classes_.tolist()
    majority = int(classes_list[0]) if len(classes_list) > 0 else 0
    # target 顺序 = label_map: {+1: 0, 0: 1, -1: 2}
    target_order = [1, 0, -1]
    if classes_list != target_order:
        proba_full = np.zeros((len(y_test), 3), dtype=float)
        for i, c in enumerate(classes_list):
            col = {1: 0, 0: 1, -1: 2}[int(c)]
            proba_full[:, col] = proba[:, i]
        proba = proba_full

    return evaluate(
        y_true=y_test,
        y_pred_proba=proba,
        inverse_map={0: 1, 1: 0, 2: -1},
        next_returns=next_returns_test,
        meta={"baseline": "most_frequent", "majority_class": int(majority)},
    )


# ============================================================
# 与 dummy baseline 对比
# ============================================================

def compare_with_dummy(
    eval_real: EvalReport,
    eval_dummy: EvalReport,
    f1_delta_threshold: float = 0.05,
) -> Dict[str, Any]:
    """对比真实模型与 dummy baseline。

    Returns:
        dict，含：
            * ``f1_delta``: ``eval_real.f1_macro - eval_dummy.f1_macro``；
            * ``accuracy_delta``: ``eval_real.accuracy - eval_dummy.accuracy``；
            * ``is_meaningful``: ``f1_delta > threshold``；
            * ``warning``（可选）: ``is_meaningful=False`` 时含此字段。
    """
    f1_delta = eval_real.f1_macro - eval_dummy.f1_macro
    acc_delta = eval_real.accuracy - eval_dummy.accuracy
    is_meaningful = bool(f1_delta > f1_delta_threshold)

    result: Dict[str, Any] = {
        "f1_delta": float(f1_delta),
        "accuracy_delta": float(acc_delta),
        "is_meaningful": is_meaningful,
        "f1_delta_threshold": f1_delta_threshold,
    }
    if not is_meaningful:
        result["warning"] = (
            f"模型未显著优于多数类 baseline（F1 delta < {f1_delta_threshold}）"
        )
    return result


# ============================================================
# JSON 序列化
# ============================================================

def serialize_eval(
    report: EvalReport,
    compare_dict: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """EvalReport + compare_dict → JSON 落盘（或返回 dict）。

    JSON 字段与 :class:`EvalReport` 字段一一对应（v2 完整字段）：
        * accuracy
        * f1_macro
        * per_class_precision
        * per_class_recall
        * per_class_f1（v2 新增）
        * auc
        * backtest_return
        * n_samples
        * meta
        * compare（可选）

    Args:
        report: EvalReport 实例。
        compare_dict: ``compare_with_dummy`` 输出（可选）。
        path: 落盘路径。None 时仅返回 dict。

    Returns:
        dict 形式的可序列化报告。
    """
    payload: Dict[str, Any] = {
        "accuracy": report.accuracy,
        "f1_macro": report.f1_macro,
        "per_class_precision": report.per_class_precision,
        "per_class_recall": report.per_class_recall,
        "per_class_f1": report.per_class_f1,
        "auc": report.auc,
        "backtest_return": report.backtest_return,
        "n_samples": report.n_samples,
        "meta": report.meta,
    }
    if compare_dict is not None:
        payload["compare"] = compare_dict
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return payload


# ============================================================
# 便捷：聚合多折评估
# ============================================================

def aggregate_eval(reports: List[EvalReport]) -> EvalReport:
    """多折评估 → 聚合（均值）。

    Args:
        reports: 多份 EvalReport。

    Returns:
        一份聚合 EvalReport（accuracy/f1 等取均值，per_class 取均值）。
    """
    if not reports:
        raise ValueError("reports 为空")
    n = len(reports)
    n_samples_total = sum(r.n_samples for r in reports)
    if n_samples_total == 0:
        return reports[0]
    return EvalReport(
        accuracy=float(np.mean([r.accuracy for r in reports])),
        f1_macro=float(np.mean([r.f1_macro for r in reports])),
        per_class_precision=[
            float(np.mean([r.per_class_precision[i] for r in reports]))
            for i in range(3)
        ],
        per_class_recall=[
            float(np.mean([r.per_class_recall[i] for r in reports]))
            for i in range(3)
        ],
        per_class_f1=[
            float(np.mean([r.per_class_f1[i] for r in reports]))
            for i in range(3)
        ],
        auc=float(np.mean([r.auc for r in reports])),
        backtest_return=float(np.sum([r.backtest_return for r in reports])),
        n_samples=int(n_samples_total),
        meta={"n_folds": n, "aggregated": True},
    )
