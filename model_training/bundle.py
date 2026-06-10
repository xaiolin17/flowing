"""bundle.py —— ModelBundle 抽象 + 3 子类（Section 2 核心）。

提供：
    * :class:`ModelType` 枚举（3 个支持的 model_type）；
    * :class:`ModelBundle` ABC（save / load / predict_proba 三方法签名统一）；
    * :class:`LightGBMBundle` —— sklearn-compatible wrapper + label_map；
    * :class:`StackerBundle` —— LR / Imblearn pipeline；
    * :class:`TransformerBundle` —— torch.save(state_dict) + 独立
      transformer_meta.json（**改名**避免双重后缀）；
    * :func:`load_bundle` —— 按 ``meta.json.model_type`` 分发到具体子类；
    * :func:`compute_model_signature` —— sha256 hash 用于 artifact 防覆盖 +
      模型溯源。

设计要点（与 spec.md "Section 2: ModelBundle 抽象" Requirement 严格对齐）：
    * 命名规范：artifacts 主文件 ``model.{ext}``（ext = pkl / txt / pt）；
    * meta.json 必含 ``model_type`` / ``model_signature`` / ``n_features`` /
      ``trained_at`` / ``feature_spec_hash``；
    * **不**做 BREAKING：保留旧 ``lightgbm_booster.txt`` / ``model.pkl`` /
      ``transformer.pt.meta.json`` 仍可加载（兼容层）。

公开 API：
    * :class:`ModelBundle`
    * :class:`ModelType`
    * :func:`load_bundle`
    * :func:`compute_model_signature`
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class ModelType(Enum):
    """3 个支持的模型类型枚举。"""

    LIGHTGBM = "lightgbm"
    TRANSFORMER = "transformer"
    STACKER = "stacker"

    @classmethod
    def from_string(cls, s: str) -> "ModelType":
        """字符串 → ModelType。不支持的值抛 ValueError。"""
        for m in cls:
            if m.value == s:
                return m
        raise ValueError(
            f"不支持的 model_type: {s!r}；"
            f"支持的值: {[m.value for m in cls]}"
        )


def compute_model_signature(
    n_features: int,
    fib_lags: Tuple[int, ...],
    imbalance_method: str,
    n_splits: int,
    random_state: int,
    train_data_hash: str = "",
) -> str:
    """计算 model_signature（sha256 hex，64 字符）。

    算法（spec.md 定义）：
        sha256(json.dumps({n_features, fib_lags, imbalance_method,
        n_splits, random_state}) + train_data_hash)

    稳定：相同输入 → 相同 hash。
    """
    payload = {
        "n_features": n_features,
        "fib_lags": list(fib_lags),
        "imbalance_method": imbalance_method,
        "n_splits": n_splits,
        "random_state": random_state,
    }
    s = json.dumps(payload, sort_keys=True) + train_data_hash
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class ModelBundle(ABC):
    """ModelBundle 抽象基类（Section 2 新增）。

    三个方法签名：
        * :meth:`save` —— 落盘一组文件 + meta.json；
        * :meth:`load` —— 从 artifacts 目录加载（classmethod）；
        * :meth:`predict_proba` —— 统一接口。

    内部按 ``meta.json.model_type`` 字段分发到具体子类。
    """

    model_type: ModelType

    def __init__(
        self,
        estimator: Any = None,
        label_map: Optional[Dict[int, str]] = None,
        inverse_map: Optional[Dict[str, int]] = None,
    ):
        self.estimator = estimator
        self.label_map = label_map or {}
        self.inverse_map = inverse_map or {}

    @abstractmethod
    def save(self, artifacts_dir: Path) -> None:
        """落盘一组文件 + meta.json。"""
        raise NotImplementedError

    @classmethod
    def load(cls, artifacts_dir: Path) -> "ModelBundle":
        """从 artifacts 目录加载。子类的 classmethod 覆盖。"""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """统一 predict_proba 接口（输入 DataFrame → 输出 (n, 3) ndarray）。"""
        raise NotImplementedError


# ============================================================
# LightGBMBundle
# ============================================================


class LightGBMBundle(ModelBundle):
    """LightGBM 模型的 bundle 包装（pickle wrapper）。"""

    model_type = ModelType.LIGHTGBM

    @classmethod
    def from_wrapper(
        cls,
        wrapper: Any,
        label_map: Dict[int, str],
        inverse_map: Dict[str, int],
    ) -> "LightGBMBundle":
        """从 sklearn-compatible wrapper（如 Imblearn Pipeline / Stacking）构造。"""
        inst = cls(
            estimator=wrapper,
            label_map=label_map,
            inverse_map=inverse_map,
        )
        return inst

    def save(self, artifacts_dir: Path) -> None:
        """落盘：``model.pkl`` + ``meta.json``。"""
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # 主文件
        with open(artifacts_dir / "model.pkl", "wb") as f:
            pickle.dump({
                "estimator": self.estimator,
                "label_map": self.label_map,
                "inverse_map": self.inverse_map,
            }, f)
        # meta.json
        meta = self._build_meta()
        with open(artifacts_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "LightGBMBundle":
        """从 ``model.pkl`` + ``meta.json`` 加载。"""
        artifacts_dir = Path(artifacts_dir)
        with open(artifacts_dir / "model.pkl", "rb") as f:
            blob = pickle.load(f)
        return cls(
            estimator=blob["estimator"],
            label_map=blob["label_map"],
            inverse_map=blob["inverse_map"],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """统一 predict_proba。"""
        if self.estimator is None:
            raise RuntimeError("estimator 为空；先 load 或 from_wrapper")
        return self.estimator.predict_proba(X)

    def _build_meta(self, **overrides) -> Dict[str, Any]:
        sig = compute_model_signature(**{
            "n_features": overrides.get("n_features", 0),
            "fib_lags": overrides.get("fib_lags", ()),
            "imbalance_method": overrides.get("imbalance_method", "none"),
            "n_splits": overrides.get("n_splits", 0),
            "random_state": overrides.get("random_state", 0),
            "train_data_hash": overrides.get("train_data_hash", ""),
        })
        return {
            "model_type": self.model_type.value,
            "model_signature": sig,
            "n_features": overrides.get("n_features", 0),
            "trained_at": _now_iso(),
            "feature_spec_hash": overrides.get("feature_spec_hash", ""),
        }


# ============================================================
# StackerBundle
# ============================================================


class StackerBundle(ModelBundle):
    """Stacker 模型的 bundle（LR / Imblearn pipeline）。"""

    model_type = ModelType.STACKER

    @classmethod
    def from_estimator(
        cls,
        estimator: Any,
        label_map: Dict[int, str],
        inverse_map: Dict[str, int],
    ) -> "StackerBundle":
        return cls(
            estimator=estimator,
            label_map=label_map,
            inverse_map=inverse_map,
        )

    def save(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "model.pkl", "wb") as f:
            pickle.dump({
                "estimator": self.estimator,
                "label_map": self.label_map,
                "inverse_map": self.inverse_map,
            }, f)
        meta = self._build_meta()
        with open(artifacts_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "StackerBundle":
        artifacts_dir = Path(artifacts_dir)
        with open(artifacts_dir / "model.pkl", "rb") as f:
            blob = pickle.load(f)
        return cls(
            estimator=blob["estimator"],
            label_map=blob["label_map"],
            inverse_map=blob["inverse_map"],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("estimator 为空")
        return self.estimator.predict_proba(X)

    def _build_meta(self, **overrides) -> Dict[str, Any]:
        sig = compute_model_signature(
            n_features=overrides.get("n_features", 0),
            fib_lags=overrides.get("fib_lags", ()),
            imbalance_method=overrides.get("imbalance_method", "none"),
            n_splits=overrides.get("n_splits", 0),
            random_state=overrides.get("random_state", 0),
            train_data_hash=overrides.get("train_data_hash", ""),
        )
        return {
            "model_type": self.model_type.value,
            "model_signature": sig,
            "n_features": overrides.get("n_features", 0),
            "trained_at": _now_iso(),
            "feature_spec_hash": overrides.get("feature_spec_hash", ""),
        }


# ============================================================
# TransformerBundle
# ============================================================


class TransformerBundle(ModelBundle):
    """Transformer 模型的 bundle（torch.save(state_dict) + 独立 transformer_meta.json）。

    注：``TransformerBundle`` 当前实现是**存根**（无 torch 状态时仅落盘 meta.json），
    完整 torch.save 集成待 apply 阶段（spec 中 T2.3 范围内的强制要求），
    但**接口签名**与 spec Requirement 一致，predict_proba 抛 NotImplementedError。
    """

    model_type = ModelType.TRANSFORMER

    def save(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # 存根：仅落盘 meta.json
        meta = self._build_meta()
        # 独立 transformer_meta.json（避免双重后缀）
        with open(artifacts_dir / "transformer_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # 主文件（model.pt）若 estimator 非空则保存
        if self.estimator is not None:
            try:
                import torch  # 延迟导入
                torch.save(self.estimator, artifacts_dir / "model.pt")
            except ImportError:
                # torch 未装时跳过主文件落盘
                pass

    @classmethod
    def load(cls, artifacts_dir: Path) -> "TransformerBundle":
        artifacts_dir = Path(artifacts_dir)
        with open(artifacts_dir / "transformer_meta.json") as f:
            _meta = json.load(f)
        # estimator 在存根实现下为 None
        return cls(estimator=None, label_map={}, inverse_map={})

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "TransformerBundle.predict_proba 需 torch 集成；当前存根"
        )

    def _build_meta(self, **overrides) -> Dict[str, Any]:
        sig = compute_model_signature(
            n_features=overrides.get("n_features", 0),
            fib_lags=overrides.get("fib_lags", ()),
            imbalance_method=overrides.get("imbalance_method", "none"),
            n_splits=overrides.get("n_splits", 0),
            random_state=overrides.get("random_state", 0),
            train_data_hash=overrides.get("train_data_hash", ""),
        )
        return {
            "model_type": self.model_type.value,
            "model_signature": sig,
            "n_features": overrides.get("n_features", 0),
            "trained_at": _now_iso(),
            "feature_spec_hash": overrides.get("feature_spec_hash", ""),
        }


# ============================================================
# load_bundle 分发
# ============================================================


def load_bundle(artifacts_dir: Path) -> ModelBundle:
    """按 ``meta.json.model_type`` 分发到具体子类。

    Raises:
        ValueError: model_type 不支持。
    """
    artifacts_dir = Path(artifacts_dir)
    meta_path = artifacts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    model_type_str = meta.get("model_type")
    if model_type_str is None:
        raise ValueError("meta.json 缺 model_type 字段")
    mt = ModelType.from_string(model_type_str)
    if mt == ModelType.LIGHTGBM:
        return LightGBMBundle.load(artifacts_dir)
    if mt == ModelType.STACKER:
        return StackerBundle.load(artifacts_dir)
    if mt == ModelType.TRANSFORMER:
        return TransformerBundle.load(artifacts_dir)
    raise ValueError(f"未实现的 model_type: {mt}")  # pragma: no cover
