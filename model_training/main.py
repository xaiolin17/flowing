"""model_training CLI —— 阶段 B 一键训练入口。

支持：
    * 外层 walk-forward 评估（n_splits 折）；
    * 内层 OOF（inner_n_splits 折）→ Stacking；
    * 4 种不平衡处理（segment_merge / focal / random_downsample / none）；
    * baseline-only 模式（强制 ``--stacking none``，**不**训练 stacker）；
    * artifacts 落盘（v2 完整 6 文件：model.pkl / lightgbm_booster.txt /
      oof_proba.csv / eval_report.json / feature_spec.json / meta.json）。
    * 阶段 C 扩展（Transformer 模型）：transformer 路径下用 ``transformer.pt``
      替代 ``lightgbm_booster.txt``；artifacts 共 7 文件。

设计要点（与 spec.md "CLI" Requirement 严格对齐）：
    * **优先级规则**（v2）：``--baseline-only True`` 强制 ``--stacking "none"`` 并打印提示；
    * **test 折走 build**（v2 关键修正）：test 折**所有 K 线**（无论 is_labeled）走
      ``build_training_matrix`` → 变化率 → 预测；
    * **仅评估 is_labeled==True**（v2 关键修正）：test 折评估时仅用
      ``is_labeled==True`` 子集（"沉默多数"被剔除）；
    * **LightGBM 持久化**（v2 修正）：用 ``booster.save_model()`` 存 ``.txt``，**不**用 pickle；
    * **sklearn 持久化**（含 stacker）：用 pickle 存 ``model.pkl``；
    * 顶部 docstring 写明：``lightgbm`` / ``shap`` / ``optuna`` 需用户手动 pip。

用法::

    # 端到端（默认 segment_merge + logistic stacking）
    python -m model_training --dataset-id 1

    # 仅 baseline（无 stacking）
    python -m model_training --dataset-id 1 --baseline-only

    # focal loss + 5 折 + 3 内层
    python -m model_training --dataset-id 1 --imbalance focal --n-splits 5 --inner-n-splits 3

    # 随机下采样 + 无 stacking
    python -m model_training --dataset-id 1 --imbalance random_downsample --stacking none
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 顶层不 import lightgbm（延迟到 train_lightgbm 函数体内）
from . import ARTIFACTS_DIR, __version__
from .data_loader import load_dataset
from .evaluator import (aggregate_eval, compare_with_dummy,
                        compute_next_returns, dummy_baseline, evaluate,
                        serialize_eval)
from .features import build_training_matrix, downsample_segment_merge
from .imbalance import apply_imbalance
from .models import DEFAULT_FACTOR_SPEC, FeatureSpec, ImbalanceMethod
from .splitter import inner_walk_forward_splits, walk_forward_splits
from .stacker import stack
from .trainer import out_of_fold_predict, train_lightgbm

# ============================================================
# artifacts 目录结构（v2 完整 6 文件）
# ============================================================

def _make_artifacts_dir(
    root: Path,
    dataset_id: int,
    imbalance: str,
    stacking: str,
) -> Path:
    """``{root}/dataset_{id}_{YYYYMMDD_HHMMSS}_{imbalance}_{stacking}/``"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"dataset_{dataset_id}_{ts}_{imbalance}_{stacking}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_feature_spec(spec: FeatureSpec, path: Path) -> None:
    """FeatureSpec → JSON 落盘（v2 新增）。"""
    payload = {
        "timeframe": spec.timeframe,
        "fib_lags": list(spec.fib_lags),
        "candle_windows": list(spec.candle_windows),
        "ma_windows": list(spec.ma_windows),
        "ema_windows": list(spec.ema_windows),
        "rsi_windows": list(spec.rsi_windows),
        "macd_params": list(spec.macd_params),
        "boll_params": list(spec.boll_params),
        "atr_window": spec.atr_window,
        "kdj_params": list(spec.kdj_params),
        "cci_window": spec.cci_window,
        "wr_window": spec.wr_window,
        "vma_windows": list(spec.vma_windows),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_max_fib_lag(
    value,
    n_samples: int,
    train_size_frac: float,
    n_splits: int,
    fib_lags: Tuple[int, ...],
) -> Optional[int]:
    """解析 ``--max-fib-lag`` 值：支持 ``None`` / ``"auto"`` / 显式 int。

    算法（auto）：
        * ``test_size = floor((n - floor(n*train_size_frac)) / n_splits)``；
        * ``safe_max_lag = max(1, test_size - 1)``（留 1 行余量给
          ``build_training_matrix`` 的 dropna）；
        * 从 ``fib_lags`` 中选 ``<= safe_max_lag`` 的最大 lag。

    Args:
        value: 用户传入值（``None`` / ``"auto"`` / int）。
        n_samples: 样本总数（``len(factor_df)``）。
        train_size_frac: 外层 walk-forward 初始 train 比例。
        n_splits: 外层 walk-forward 折数。
        fib_lags: FeatureSpec 默认 fib_lags 列表。

    Returns:
        解析后的 max_lag。``None`` 表示用 spec 默认全集。
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        # 复刻 walk_forward_splits 公式（splitter.py）
        initial = int(n_samples * train_size_frac)  # 等价 math.floor for float
        test_size = int((n_samples - initial) / n_splits)
        # 留 1 行 buffer：build_training_matrix 走 max_lag 行的 logret/delta 后会 dropna
        safe_max_lag = max(1, test_size - 1)
        kept = tuple(n for n in fib_lags if n <= safe_max_lag)
        if not kept:
            return min(fib_lags)  # 极端兜底
        return max(kept)
    # 显式 int
    return int(value)


def _save_meta(
    path: Path,
    *,
    dataset_id: int,
    imbalance: str,
    stacking: str,
    n_splits: int,
    inner_n_splits: int,
    random_state: int,
    baseline_only: bool,
    model_type: str = "lightgbm",
    extra: Optional[dict] = None,
) -> None:
    """``meta.json``（v2 新增：版本号、随机种子、参数 + 阶段 C model_type）。"""
    payload = {
        "version": __version__,
        "dataset_id": int(dataset_id),
        "model_type": str(model_type),
        "imbalance": str(imbalance),
        "stacking": str(stacking),
        "n_splits": int(n_splits),
        "inner_n_splits": int(inner_n_splits),
        "random_state": int(random_state),
        "baseline_only": bool(baseline_only),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


# ============================================================
# 单折训练 + 评估
# ============================================================


class _TransformerOOFWrapper:
    """Transformer OOF 包装类：对齐 LightGBM OOF 接口。

    设计：
        * ``wrapper_inner.predict_proba`` 输出 ``(n - seq_len + 1, 3)``，与
          ``n_in`` 差 ``seq_len - 1`` 行；
        * **头部 pad** 用 NaN（让 ``out_of_fold_predict`` 内部对 NaN 行
          不写 valid_mask，stacker 训练时这些行被过滤）；
        * **边界**（n_in < seq_len）：返 ``(n_in, 3)`` 全 NaN，避免 stacker
          学到 "1/3 占位" 的虚假模式（DLClassifierWrapper.predict_proba
          在此情形返 1/3 占位，包装层改写为 NaN）。
    """

    def __init__(self, w):
        self._w = w

    def predict_proba(self, X):
        n_in = len(X)
        seq_len = self._w.cfg.seq_len
        # 边界：n_in < seq_len → 无法构造序列 → 返 NaN
        if n_in < seq_len:
            return np.full((n_in, 3), np.nan, dtype=np.float32)
        proba = self._w.predict_proba(X)
        if len(proba) < n_in:
            # 头部 pad：transformer 输出缺前 seq_len-1 行
            pad = np.full(
                (n_in - len(proba), proba.shape[1]),
                np.nan, dtype=np.float32,
            )
            proba = np.concatenate([pad, proba], axis=0)
        return proba



def _train_one_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    is_labeled_train: np.ndarray,
    X_test_changes: pd.DataFrame,
    y_test_built: np.ndarray,
    is_labeled_test_built: np.ndarray,
    next_returns_test: np.ndarray,
    *,
    imbalance: str,
    stacking: str,
    random_state: int,
    inner_n_splits: int,
    train_size_frac: float,
    model_type: str = "lightgbm",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """单折：训练 → 评估 → 可选 stacking。

    Args:
        model_type: ``"lightgbm"`` / ``"transformer"``，默认 lightgbm。
        model_kwargs: 透传给底层 train 函数的 kwargs。

    Returns:
        dict：``eval_real / eval_dummy / compare / oof_proba / model / stacker / inverse_map``。
        * ``model``：LightGBM 时为 booster，Transformer 时为 ``DLClassifierWrapper``。
    """
    model_kwargs = model_kwargs or {}

    # 1) imbalance（在 build 之后，除 segment_merge 外——segment_merge
    #    已在 main loop 的 build 之前基于 raw close + atr14 调用，X_train
    #    已是段内合并后的样本；这里视作 "none"）
    use_focal = False
    imb_for_factory = (
        ImbalanceMethod.NONE if imbalance == "segment_merge"
        else ImbalanceMethod(imbalance)
    )
    X_used, y_used, is_lab_used, imb_meta = apply_imbalance(
        imb_for_factory, X_train, y_train, is_labeled_train,
        **(
            {"random_state": random_state}
            if imb_for_factory == ImbalanceMethod.RANDOM_DOWNSAMPLE
            else {}
        ),
    )
    if imbalance == "focal":
        use_focal = True

    # 2) 训练（按 model_type 分发）
    if model_type == "lightgbm":
        # 延迟导入 lightgbm
        model, lmap, inv_map, history = train_lightgbm(
            X_used, y_used,
            use_focal=use_focal, random_state=random_state,
        )
        # 边界：test 折 build 后 0 行（小数据集 + segment_merge 极端情况）
        if len(X_test_changes) == 0:
            test_proba = np.zeros((0, 3), dtype=np.float32)
        else:
            test_proba = model.predict(X_test_changes)
        # LightGBM 不滑动窗口，对齐变量 = 原始
        is_labeled_test_aligned = is_labeled_test_built
        y_test_aligned = y_test_built
        next_ret_aligned = next_returns_test
    elif model_type == "transformer":
        # 延迟导入 dl 子包
        from .dl.trainer import train_transformer

        # transformer 走 numpy 而非 DataFrame（内部用 numpy 构造序列）
        # X_used / X_test_changes 都是变化率 DataFrame，columns 已是
        # 白名单通过；to_numpy() 转 float32 防类型问题
        X_used_np = X_used.to_numpy(dtype=np.float32)
        X_test_np = X_test_changes.to_numpy(dtype=np.float32)
        wrapper, lmap, inv_map, history = train_transformer(
            X_used_np, y_used,
            X_val=None, y_val=None,
            use_focal=use_focal,
            random_state=random_state,
            **model_kwargs,
        )
        model = wrapper  # 包装类对齐 LightGBM booster 接口
        test_proba_full = wrapper.predict_proba(X_test_np)
        # transformer 输出长度 = n - seq_len + 1（序列构造滑动窗口）
        # 与 X_test_changes 长度差 seq_len-1，需要对齐
        # 工程做法：用 build_training_matrix 输出截取末尾 seq_len-1 行
        # 与 transformer 输出对齐（transformer 输出对应"每个序列的最后一根"）
        seq_len = wrapper.cfg.seq_len
        # 边界：predict_proba 内部对 n < seq_len 返回均匀 (n, 3) 占位
        # 评估时用 test_proba 长度对齐 is_labeled_test_built
        n_proba = len(test_proba_full)
        # 截取 is_labeled_test_built 末尾 n_proba 行
        is_labeled_test_aligned = is_labeled_test_built[-n_proba:]
        y_test_aligned = y_test_built[-n_proba:]
        next_ret_aligned = next_returns_test[-n_proba:]
        test_proba = test_proba_full
    else:
        raise ValueError(f"不支持的 model_type: {model_type!r}")

    # 3) test 折预测（已在上方分类型处理）

    # 4) 评估（仅 is_labeled==True；用 _aligned 变量兼容 transformer 滑动窗口）
    eval_real = evaluate(
        y_true=y_test_aligned[is_labeled_test_aligned],
        y_pred_proba=test_proba[is_labeled_test_aligned],
        inverse_map=inv_map,
        next_returns=next_ret_aligned[is_labeled_test_aligned],
        meta={"imbalance": imbalance, "use_focal": use_focal, "imb_meta": imb_meta,
              "model_type": model_type},
    )

    # 5) dummy baseline
    y_train_used_for_dummy = y_used
    eval_dummy = dummy_baseline(
        y_train=y_train_used_for_dummy,
        y_test=y_test_aligned[is_labeled_test_aligned],
        next_returns_test=next_ret_aligned[is_labeled_test_aligned],
    )

    # 6) compare
    compare = compare_with_dummy(eval_real, eval_dummy, f1_delta_threshold=0.05)

    # 7) 可选 stacking
    oof_proba = None
    valid_oof_mask = None
    stacker = None
    if stacking == "logistic":
        if model_type == "lightgbm":
            def _oof_fn(Xt, yt):
                booster_inner, _, _, _ = train_lightgbm(
                    Xt, yt, use_focal=use_focal, random_state=random_state,
                )
                class _ProbaWrapper:
                    def __init__(self, b): self._b = b
                    def predict_proba(self, X): return self._b.predict(X)
                return _ProbaWrapper(booster_inner)
        else:  # transformer
            from .dl.trainer import train_transformer
            def _oof_fn(Xt, yt):
                # Xt 是 DataFrame（out_of_fold_predict 内部 X.iloc[slice]）
                # 但 train_transformer 需要 numpy → 转
                if hasattr(Xt, "to_numpy"):
                    Xt_np = Xt.to_numpy(dtype=np.float32)
                else:
                    Xt_np = np.asarray(Xt, dtype=np.float32)
                wrapper_inner, _, _, _ = train_transformer(
                    Xt_np, yt,
                    use_focal=use_focal, random_state=random_state,
                    **model_kwargs,
                )
                # OOF 内部 out_of_fold_predict 调 m.predict_proba(X.iloc[test_idx])
                # transformer predict_proba 输出 (n - seq_len + 1, 3)
                # 与 test_idx 长度 (n_test) 差 seq_len-1 → 用 _TransformerOOFWrapper
                # 处理头部 NaN pad + n<seq_len 边界
                return _TransformerOOFWrapper(wrapper_inner)

        inner_splits = inner_walk_forward_splits(
            n_train=len(X_used), n_inner_splits=inner_n_splits,
            train_size_frac=train_size_frac,
        )
        oof_proba, valid_oof_mask = out_of_fold_predict(
            model_fn=_oof_fn,
            X=X_used, y=y_used, splits=inner_splits, random_state=random_state,
        )
        if valid_oof_mask.any():
            stacker = stack(
                oof_proba=oof_proba,
                y_train=y_used,
                oof_features=X_used.values,
                valid_mask=valid_oof_mask,
                meta="logistic", random_state=random_state,
            )
            # stacker 二层预测 test 折（oof_features 用 train 折原始变化率）
            if model_type == "transformer":
                # transformer 路径 test_proba 形状 = (n_test - seq_len + 1, 3)
                # 每个 proba[i] 对应 X_test_changes 的索引 i+seq_len-1（即**末段**）
                # 取末 n_test_seq 行才能与 test_proba 行序对齐；用前段会错位
                # → stacker 学到错误关联，f1 偏低
                n_test_seq = len(test_proba)
                test_proba2 = stacker.predict_proba(
                    np.concatenate(
                        [test_proba, X_test_changes.iloc[-n_test_seq:].values],
                        axis=1,
                    )
                )
            else:
                test_proba2 = stacker.predict_proba(
                    np.concatenate([test_proba, X_test_changes.values], axis=1)
                )
            eval_stacker = evaluate(
                y_true=y_test_aligned[is_labeled_test_aligned],
                y_pred_proba=test_proba2[is_labeled_test_aligned],
                inverse_map=inv_map,
                next_returns=next_ret_aligned[is_labeled_test_aligned],
                meta={"imbalance": imbalance, "use_focal": use_focal,
                      "stacker": "logistic", "imb_meta": imb_meta,
                      "model_type": model_type},
            )
            # 用 stacker 评估覆盖 eval_real
            eval_real = eval_stacker

    return {
        "eval_real": eval_real,
        "eval_dummy": eval_dummy,
        "compare": compare,
        "oof_proba": oof_proba,
        "valid_oof_mask": valid_oof_mask,
        "model": model,
        "stacker": stacker,
        "imbalance_meta": imb_meta,
        "inverse_map": inv_map,
    }


# ============================================================
# 主流程
# ============================================================


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 主入口。

    Returns:
        进程退出码（0 = 成功，1 = 异常）。
    """
    parser = argparse.ArgumentParser(
        prog="python -m model_training",
        description="阶段 B 训练 CLI",
    )
    parser.add_argument("--dataset-id", type=int, default=None,
                        help="dataset_id（来自 data_labeling SQLite；--synthetic 模式可省略）")
    parser.add_argument("--imbalance", type=str, default="segment_merge",
                        choices=["none", "segment_merge", "focal", "random_downsample"],
                        help="不平衡处理方法（默认 segment_merge）")
    parser.add_argument("--n-splits", type=int, default=5, help="外层 walk-forward 折数")
    parser.add_argument("--inner-n-splits", type=int, default=3, help="内层 OOF 折数")
    parser.add_argument("--stacking", type=str, default="logistic",
                        choices=["logistic", "none"],
                        help="Stacking 第二层（默认 logistic）")
    parser.add_argument("--baseline-only", action="store_true",
                        help="仅训练 + 评估 baseline（不 stacking）")
    parser.add_argument("--random-state", type=int, default=42, help="全局随机种子")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR,
                        help="artifacts 根目录")
    parser.add_argument("--train-size-frac", type=float, default=0.6,
                        help="外层 walk-forward 初始 train 比例")
    parser.add_argument("--inner-train-size-frac", type=float, default=0.5,
                        help="内层 walk-forward 初始 train 比例")
    parser.add_argument("--max-fib-lag", type=str, default=None,
                        help="最大 fib lag（None=用 spec 默认；数字=硬上限；"
                             "auto=按 test_size 自适应降级，避免 n 小时 test 折 build 为空）")
    parser.add_argument("--max-candle-window", type=int, default=None,
                        help="最大虚拟 K 线窗口 N（None=用 spec 默认 fib 9 元组；"
                             "数字=硬上限，保留 <= N 的 candle_windows）")
    parser.add_argument("--min-meaningful-folds", type=int, default=0,
                        help="聚合多折时最少有效 fold 数（Section 3 新增）。"
                             "0=不过滤（默认）；>0=要求 >=N 个 is_meaningful=True fold，"
                             "不足时 is_meaningful_aggregate=False + warning")
    parser.add_argument("--registry-db", type=Path, default=None,
                        help="模型注册表 SQLite DB 路径（Section 4 新增）。"
                             "None=不写入注册表；"
                             "默认 model_training/model_registry.db")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别（Section 5 新增；默认 INFO）")

    # ---- 阶段 C：模型层扩展 ----
    parser.add_argument("--model", type=str, default="lightgbm",
                        choices=["lightgbm", "transformer"],
                        help="模型选择（阶段 C 加 transformer）")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="设备（None=自动检测；transformer only）")
    parser.add_argument("--quick-epochs", type=int, default=5,
                        help="快速模式 epoch 数（transformer only，默认 5）")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer d_model")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Transformer 多头注意力头数（必须整除 d_model）")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Transformer encoder 层数")
    parser.add_argument("--dim-ff", type=int, default=256,
                        help="Transformer FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Transformer 序列窗口")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Transformer batch size")

    # ---- 合成数据 ----
    parser.add_argument("--synthetic", action="store_true",
                        help="用合成 K 线训练（避免 dataset-id 必填）；自动写临时 SQLite "
                             "+ 进程退出时自动清理临时目录；label_count = 10% × n_rows")
    parser.add_argument("--synthetic-n-rows", type=int, default=10000,
                        help="合成 K 线数（默认 10000）")
    parser.add_argument("--synthetic-seed", type=int, default=42,
                        help="合成数据随机种子")
    args = parser.parse_args(argv)

    # Section 5：结构化日志配置（顶部）
    try:
        from .logging import setup_logging
    except ImportError:
        from model_training.logging import setup_logging
    setup_logging(level=args.log_level)

    # 优先级规则：--baseline-only True → 强制 --stacking none
    if args.baseline_only:
        if args.stacking != "none":
            print("[INFO] baseline-only 强制 --stacking none")
        args.stacking = "none"

    print(f"[INFO] model_training v{__version__}")
    print(f"[INFO] model={args.model} dataset_id={args.dataset_id} "
          f"imbalance={args.imbalance} stacking={args.stacking} "
          f"n_splits={args.n_splits} inner_n_splits={args.inner_n_splits}")

    # 0) --synthetic 模式：生成合成数据到临时 DB
    if args.synthetic:
        from tools.seed_data import seed_to_sqlite
        tmp_dir = tempfile.TemporaryDirectory(prefix="dl_synth_")
        tmp_db = Path(tmp_dir.name) / "synth.db"
        # label_count = 10% 采样（避免超过 30% 上限）
        synth_labels = max(1, int(args.synthetic_n_rows * 0.1))
        print(f"[INFO] --synthetic 模式：生成 {args.synthetic_n_rows} 根 K 线，"
              f"{synth_labels} 标签 → {tmp_db}")
        ds = seed_to_sqlite(
            n_rows=args.synthetic_n_rows,
            label_count=synth_labels,
            seed=args.synthetic_seed,
            db_path=tmp_db,
        )
        args.dataset_id = ds.id
        args._synthetic_db_path = tmp_db  # 供后续覆盖 DEFAULT_DB_PATH
        print(f"[INFO] synthetic dataset_id={ds.id}, db={tmp_db}")

    # dataset_id 必填检查（--synthetic 已在上面赋值）
    if args.dataset_id is None:
        print("[ERROR] 必须指定 --dataset-id 或 --synthetic")
        return 1

    # 1) 加载数据
    from data_labeling.db import DEFAULT_DB_PATH

    from .data_loader import list_datasets

    # --synthetic 用临时 DB（覆盖默认）
    db_to_use = getattr(args, "_synthetic_db_path", None) or DEFAULT_DB_PATH

    datasets = list_datasets(db_to_use)
    if not any(d.id == args.dataset_id for d in datasets):
        print(f"[ERROR] dataset_id={args.dataset_id} 不存在（DB: {db_to_use}）")
        return 1
    X_raw, y_all, is_labeled_all = load_dataset(args.dataset_id, db_to_use)
    print(f"[INFO] 加载数据: {len(X_raw)} K 线, "
          f"is_labeled={int(is_labeled_all.sum())} ({is_labeled_all.mean()*100:.1f}%)")

    # 2) 因子
    factor_spec = DEFAULT_FACTOR_SPEC
    # 解析 --max-fib-lag（支持 "auto" / int 字符串 / None）
    n_raw = len(X_raw)
    max_fib_lag_value: Optional[object] = args.max_fib_lag
    if isinstance(max_fib_lag_value, str) and max_fib_lag_value.strip().isdigit():
        max_fib_lag_value = int(max_fib_lag_value)
    resolved_max_lag = resolve_max_fib_lag(
        max_fib_lag_value,
        n_samples=n_raw,
        train_size_frac=args.train_size_frac,
        n_splits=args.n_splits,
        fib_lags=factor_spec.fib_lags,
    )
    if resolved_max_lag is not None:
        from dataclasses import replace

        # 自适应：保留 <= resolved_max_lag 的 lags
        kept_lags = tuple(n for n in factor_spec.fib_lags if n <= resolved_max_lag)
        if not kept_lags:
            kept_lags = (1, 2, 3, 5, 8)
        factor_spec = replace(factor_spec, fib_lags=kept_lags)
        print(f"[INFO] 自适应 fib_lags（max={resolved_max_lag}）: {kept_lags}")

    # 解析 --max-candle-window（Section 1 新增；与 --max-fib-lag 独立，不联动）
    if args.max_candle_window is not None:
        from dataclasses import replace as _dc_replace_cw

        max_cw = int(args.max_candle_window)
        kept_cw = tuple(n for n in factor_spec.candle_windows if n <= max_cw)
        if not kept_cw:
            kept_cw = (1, 2, 3)
        factor_spec = _dc_replace_cw(factor_spec, candle_windows=kept_cw)
        print(f"[INFO] 自适应 candle_windows（max={max_cw}）: {kept_cw}")

    # 延迟 import factors（内部会 import pandas / numpy）
    from .factors import compute_factors
    factor_df = compute_factors(factor_spec, X_raw)
    print(f"[INFO] 因子: {factor_df.shape[1]} 列")

    # 3) next_returns（用 raw close，**不**用因子 close）
    next_ret_full = compute_next_returns(X_raw["close"])

    # 4) 外层 walk-forward
    outer_splits = walk_forward_splits(
        n_samples=len(factor_df),
        n_splits=args.n_splits,
        train_size_frac=args.train_size_frac,
        expanding=True,
    )
    print(f"[INFO] 外层切分: {len(outer_splits)} 折")

    # 5) artifacts 目录
    art_root = _make_artifacts_dir(
        args.artifacts_dir, args.dataset_id, args.imbalance, args.stacking,
    )
    print(f"[INFO] artifacts 目录: {art_root}")

    # 6) 逐折训练 + 评估
    all_real: list = []
    all_dummy: list = []
    all_compare: list[dict[str, Any]] = []
    oof_proba_per_fold: list[np.ndarray] = []
    valid_mask_per_fold: list[np.ndarray] = []
    last_model = None
    last_stacker = None
    last_inv_map = None

    # 阶段 C：构造 model_kwargs
    if args.model == "transformer":
        from dataclasses import replace as _dc_replace

        from .dl.models import TransformerConfig

        # 验证 n_heads 整除 d_model
        if args.d_model % args.n_heads != 0:
            print(f"[ERROR] d_model={args.d_model} 必须能被 n_heads={args.n_heads} 整除")
            return 1
        # quick 模式：max_epochs=quick-epochs, patience=2
        # 正常模式：max_epochs=30, patience=5
        transformer_cfg = TransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_epochs=args.quick_epochs,
            patience=2,
            device=args.device,
            random_state=args.random_state,
        )
        model_kwargs = {"cfg": transformer_cfg}
        print(f"[INFO] transformer config: d_model={args.d_model} n_heads={args.n_heads} "
              f"n_layers={args.n_layers} dim_ff={args.dim_ff} dropout={args.dropout} "
              f"seq_len={args.seq_len} batch_size={args.batch_size} "
              f"max_epochs={args.quick_epochs}")
    else:
        model_kwargs = {}

    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\n=== Fold {fold_idx + 1}/{len(outer_splits)} "
              f"train={len(train_idx)} test={len(test_idx)} ===")

        # ----- 段内合并（基于 raw close + atr14，独立于 build） -----
        if args.imbalance == "segment_merge":
            close_train = X_raw["close"].iloc[train_idx]
            atr_train = factor_df["atr14"].iloc[train_idx]
            kept_indices, sm_meta = downsample_segment_merge(
                close=close_train,
                atr=atr_train,
                is_labeled=is_labeled_all[train_idx],
                y=y_all[train_idx],
            )
            print(f"  [seg-merge] orig={sm_meta['orig_n']} "
                  f"kept={sm_meta['resampled_n']} segments={sm_meta['n_segments']}")
        else:
            kept_indices = np.arange(len(train_idx), dtype=np.int64)
            sm_meta = {"method": "none", "orig_n": int(len(train_idx)),
                       "resampled_n": int(len(train_idx))}

        # ----- build 训练矩阵 -----
        X_train_changes, y_train_built, is_labeled_train_built, _ = (
            build_training_matrix(
                factor_df=factor_df.iloc[train_idx].iloc[kept_indices],
                label_series=y_all[train_idx][kept_indices],
                is_labeled=is_labeled_all[train_idx][kept_indices],
                timeframe=factor_spec.timeframe,
                kept_indices=None,  # 已在上面手工切片
                fib_lags=factor_spec.fib_lags,
            )
        )
        # 上面的 build_training_matrix 内部还会再做"变化率 + 白名单 + is_labeled 过滤 + dropna"
        print(f"  [build] X_train={X_train_changes.shape} "
              f"is_labeled={int(is_labeled_train_built.sum())}")

        # ----- test 折走 build（v2 关键修正：所有 K 线都走） -----
        X_test_changes, y_test_built, is_labeled_test_built, valid_index_test = (
            build_training_matrix(
                factor_df=factor_df.iloc[test_idx],
                label_series=y_all[test_idx],
                is_labeled=is_labeled_all[test_idx],
                timeframe=factor_spec.timeframe,
                kept_indices=None,
                fib_lags=factor_spec.fib_lags,
            )
        )
        # test 折的 next_returns 严格对齐：dropna 删头部，
        # 直接 [:N] 截尾会与 is_labeled 错位 → 用 build 返回的
        # valid_index 反推 X_raw 位置
        next_ret_test_raw = next_ret_full[test_idx]
        test_index = X_raw.index[test_idx]
        pos_in_test = test_index.get_indexer(valid_index_test)
        if (pos_in_test == -1).any():
            raise RuntimeError(
                f"build 后的 valid_index 不全在 test_idx 范围内 "
                f"(n_invalid={int((pos_in_test == -1).sum())})"
            )
        next_ret_test_aligned = next_ret_test_raw[pos_in_test]
        assert len(next_ret_test_aligned) == len(X_test_changes), (
            f"next_ret 对齐后长度 {len(next_ret_test_aligned)} "
            f"≠ X_test_changes 长度 {len(X_test_changes)}"
        )
        # 严格对齐已在上文完成（valid_index 反推）。

        # ----- 边界：test 折走 build 后为空 -----
        # 触发条件：test 折长度 < max_fib_lag（默认 987），导致 compute_changes
        # 的滞后列整列 NaN，build 内部 dropna 删光 → X_test_changes 长度为 0。
        # 数据生成建议：总 K 线数 ≥ max_fib_lag + test_size，避免此边界。
        # 工程做法：跳过该 fold（不评估、不聚合），记 warning，**不**抛错。
        if len(X_test_changes) == 0 or int(is_labeled_test_built.sum()) == 0:
            print(f"  [WARN] Fold {fold_idx + 1} test 折 build 后为空 "
                  f"(n_test={len(test_idx)}, max_fib_lag={max(factor_spec.fib_lags)}); "
                  "跳过该 fold。建议增大 n_splits 后的总 K 线数或降低 max_fib_lag。")
            continue


        # ----- 单折训练 + 评估 -----
        result = _train_one_fold(
            X_train=X_train_changes,
            y_train=y_train_built,
            is_labeled_train=is_labeled_train_built,
            X_test_changes=X_test_changes,
            y_test_built=y_test_built,
            is_labeled_test_built=is_labeled_test_built,
            next_returns_test=next_ret_test_aligned,
            imbalance=args.imbalance,
            stacking=args.stacking,
            random_state=args.random_state,
            inner_n_splits=args.inner_n_splits,
            train_size_frac=args.inner_train_size_frac,
            model_type=args.model,
            model_kwargs=model_kwargs,
        )
        all_real.append(result["eval_real"])
        all_dummy.append(result["eval_dummy"])
        all_compare.append(result["compare"])
        if result["oof_proba"] is not None:
            oof_proba_per_fold.append(result["oof_proba"])
            valid_mask_per_fold.append(result["valid_oof_mask"])
        last_model = result["model"]
        last_stacker = result["stacker"]
        last_inv_map = result["inverse_map"]
        ev = result["eval_real"]
        cm = result["compare"]
        print(f"  [eval] acc={ev.accuracy:.3f} f1={ev.f1_macro:.3f} "
              f"auc={ev.auc:.3f} f1_delta={cm['f1_delta']:.3f} "
              f"meaningful={cm['is_meaningful']}")

    # 7) 聚合
    # 边界：所有 fold 都因 test 折为空被跳过时，all_real / all_dummy 为空
    # → aggregate_eval([]) 会抛 ValueError。提前打印明确错误并返回 1。
    if not all_real:
        print(
            f"[ERROR] 全部 {len(outer_splits)} 个 fold 都被跳过（test 折 build 后为空）。\n"
            "  建议：\n"
            "  1) 增大 --synthetic-n-rows（≥ 2000 + max_fib_lag），或\n"
            "  2) 降低 --max-fib-lag（如 144），或\n"
            "  3) 减少 --n-splits（让 test 折更大）。",
            file=sys.stderr,
        )
        return 1
    agg_real = aggregate_eval(
        all_real,
        min_meaningful_folds=args.min_meaningful_folds,
        compare_dicts=all_compare if args.min_meaningful_folds > 0 else None,
    )
    agg_dummy = aggregate_eval(all_dummy)
    # Section 3：聚合层 warning
    if agg_real.meta.get("warning"):
        print(f"  [WARN] {agg_real.meta['warning']}")
    n_invalid = agg_real.n_total_folds - agg_real.n_valid_folds
    if n_invalid > 0:
        print(f"  [WARN] {n_invalid}/{agg_real.n_total_folds} folds is_meaningful=False")
    print(f"\n=== Aggregate ({len(outer_splits)} folds) ===")
    print(f"  acc={agg_real.accuracy:.3f} f1={agg_real.f1_macro:.3f} "
          f"auc={agg_real.auc:.3f} backtest={agg_real.backtest_return:.4f}")
    print(f"  dummy acc={agg_dummy.accuracy:.3f} f1={agg_dummy.f1_macro:.3f}")
    print(f"  f1_delta = {agg_real.f1_macro - agg_dummy.f1_macro:+.3f}")

    # 8) 落盘 artifacts
    # 8.1 model.pkl
    if last_stacker is not None:
        with open(art_root / "model.pkl", "wb") as f:
            pickle.dump(last_stacker, f)

    # 8.2 model checkpoint（按 model_type 分发）
    if last_model is not None:
        if args.model == "lightgbm":
            # LightGBM 用 booster.save_model() 存 .txt（v2 修正）
            last_model.save_model(str(art_root / "lightgbm_booster.txt"))
        elif args.model == "transformer":
            # Transformer 用 wrapper.save() 存 state_dict + meta
            last_model.save(str(art_root / "transformer.pt"))

    # 8.3 oof_proba.csv（非 baseline-only；用 CSV 通用，pyarrow 选装）
    if not args.baseline_only and oof_proba_per_fold:
        rows = []
        for fi, (op, vm) in enumerate(zip(oof_proba_per_fold, valid_mask_per_fold)):
            for si in range(op.shape[0]):
                rows.append([fi, si, op[si, 0], op[si, 1], op[si, 2], bool(vm[si])])
        oof_df = pd.DataFrame(rows, columns=["fold", "sample", "p0", "p1", "p2", "valid"])
        oof_df.to_csv(art_root / "oof_proba.csv", index=False)

    # 8.4 eval_report.json
    compare_agg = compare_with_dummy(agg_real, agg_dummy, f1_delta_threshold=0.05)
    serialize_eval(agg_real, compare_agg, path=art_root / "eval_report.json")

    # 8.5 feature_spec.json
    _save_feature_spec(factor_spec, art_root / "feature_spec.json")

    # 8.6 meta.json
    _save_meta(
        art_root / "meta.json",
        dataset_id=args.dataset_id,
        model_type=args.model,
        imbalance=args.imbalance,
        stacking=args.stacking,
        n_splits=args.n_splits,
        inner_n_splits=args.inner_n_splits,
        random_state=args.random_state,
        baseline_only=args.baseline_only,
        extra={"n_folds": len(outer_splits), "elapsed_sec": time.time() - t0},
    )

    print(f"\n[INFO] artifacts 已落盘到 {art_root}")
    # Section 5：结构化日志（关键事件）
    try:
        from .logging import get_logger
    except ImportError:
        from model_training.logging import get_logger
    _log = get_logger("model_training.main")
    _log.info("artifacts saved", extra={
        "artifact_dir": str(art_root),
        "f1_macro": float(agg_real.f1_macro),
        "n_features": int(agg_real.n_samples),
    })
    print(f"[INFO] 耗时 {time.time() - t0:.1f}s")

    # 8.7 模型注册表（Section 4 新增；可选）
    if args.registry_db is not None:
        try:
            from .registry import RunRecord, register_run
        except ImportError:
            from model_training.registry import RunRecord, register_run
        import datetime as _dt_reg
        # 从 meta.json 读 model_signature（若有）
        import json as _json_reg
        sig = ""
        meta_json_path = art_root / "meta.json"
        if meta_json_path.exists():
            try:
                with open(meta_json_path) as _f:
                    sig = _json_reg.load(_f).get("model_signature", "")
            except Exception:
                sig = ""
        rec = RunRecord(
            dataset_id=args.dataset_id,
            model_type=args.model,
            imbalance=args.imbalance,
            stacking=args.stacking,
            timestamp=_dt_reg.datetime.now(_dt_reg.timezone.utc).isoformat(),
            f1_macro=float(agg_real.f1_macro),
            f1_delta=float(compare_agg.get("f1_delta", 0.0)),
            is_meaningful=bool(compare_agg.get("is_meaningful", False)),
            artifact_dir=str(art_root),
            model_signature=sig,
        )
        try:
            register_run(args.registry_db, rec)
            print(f"[INFO] 已注册到 {args.registry_db}")
        except Exception as _e:  # noqa: BLE001
            print(f"[WARN] 注册到 DB 失败: {_e}", file=__import__("sys").stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
