"""trainer.py —— Transformer 训练循环（阶段 C）。

提供：
    * :func:`_make_sequences` —— 1D 变化率 → 2D 滑动窗口 ``(n-T+1, T, F)``；
    * :class:`SequenceStandardScaler` —— 跨折 fit/transform 状态；
    * :func:`_focal_loss` —— 多分类 focal loss（与阶段 B trainer 共享语义）；
    * :class:`DLClassifierWrapper` —— 对齐 LightGBM 接口（``fit / predict_proba / save / load``）；
    * :func:`train_transformer` —— 高层 API，返回 ``(wrapper, lmap, inv, history)``。

设计要点（与 spec.md "训练循环" / "与阶段 B pipeline 集成" 严格对齐）：
    * 设备自动检测：``cuda`` 可用时自动用，否则 ``cpu``；
    * 随机种子：``torch + numpy + random`` 三处；
    * **防泄露关键**：scaler fit **只在 train 折**；test 折 transform 用 train mean/std；
    * 延迟 import torch，缺包抛 ImportError；
    * 接口对齐：``predict_proba(X)`` 返回 ``(n, 3)``，与 LightGBM 完全一致；
    * 错误处理：缺 torch / X 含 NaN / seq_len 过大 / loss=NaN。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 顶部不 import torch（延迟到函数体内）
from .models import TransformerConfig, build_model

# ============================================================
# 设备
# ============================================================

def select_device(device: Optional[str] = None) -> str:
    """设备选择。

    Args:
        device: ``"cpu"`` / ``"cuda"`` / ``None``。
            * ``None`` → 自动检测（CUDA 可用用 CUDA，否则 CPU）
            * 显式值 → 直接用（**不**做检测）

    Returns:
        设备字符串（``"cpu"`` / ``"cuda"``）。
    """
    if device is None:
        # 延迟 import torch
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device 必须是 'cpu' 或 'cuda'，实际 {device!r}")
    return device


# ============================================================
# 随机种子
# ============================================================

def set_seed(seed: int) -> None:
    """设置 torch + numpy + random 三处种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # 缺 torch 不报错（set_seed 是 helper）


# ============================================================
# 序列构造（1D → 2D 滑动窗口）
# ============================================================

def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """滑动窗口：``X[i:i+seq_len]`` → ``y[i+seq_len-1]``。

    Args:
        X: ``ndarray[n, F]``，变化率矩阵。
        y: ``ndarray[n]``，label。
        seq_len: 窗口长度。

    Returns:
        ``(X_seq, y_seq)``：
            * X_seq: ``ndarray[n - seq_len + 1, seq_len, F]``（float32）
            * y_seq: ``ndarray[n - seq_len + 1]``（int64）

    Raises:
        ValueError: ``seq_len > n`` 或 ``X`` 含 NaN/Inf。
    """
    n = len(X)
    # 兼容 DataFrame / ndarray 输入
    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    if seq_len > n:
        raise ValueError(f"seq_len={seq_len} > n_samples={n}")
    if np.isnan(X).any():
        raise ValueError("X 含 NaN；build_training_matrix 已 dropna，请检查上游")
    if np.isinf(X).any():
        raise ValueError("X 含 Inf；请检查上游")
    if seq_len < 1:
        raise ValueError(f"seq_len={seq_len} 必须 >= 1")

    n_seq = n - seq_len + 1
    X_seq = np.empty((n_seq, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X[i:i + seq_len]
    y_seq = y[seq_len - 1:].astype(np.int64)
    return X_seq, y_seq


# ============================================================
# 标准化（跨折隔离）
# ============================================================

class SequenceStandardScaler:
    """序列级 StandardScaler：fit in train, transform in test。

    关键：fit **只在 train 折**调用；test 折 transform 用 train 折的 mean/std。
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X_seq: np.ndarray) -> "SequenceStandardScaler":
        """``X_seq: ndarray[n, T, F]`` → 跨时间维 + 批次维统计 ``(F,)``。"""
        # reshape to (n*T, F) 然后 mean/std on axis=0
        flat = X_seq.reshape(-1, X_seq.shape[-1])
        self.mean_ = flat.mean(axis=0).astype(np.float32)
        self.std_ = flat.std(axis=0).astype(np.float32)
        # 防除零
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X_seq: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("必须先 fit 才能 transform")
        return ((X_seq - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X_seq: np.ndarray) -> np.ndarray:
        return self.fit(X_seq).transform(X_seq)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "SequenceStandardScaler":
        s = cls()
        s.mean_ = np.array(d["mean"], dtype=np.float32)
        s.std_ = np.array(d["std"], dtype=np.float32)
        return s


# ============================================================
# Focal loss（多分类，gamma 共享阶段 B 语义）
# ============================================================

def _focal_loss(
    logits: "torch.Tensor",
    target: "torch.Tensor",
    gamma: float,
    num_classes: int = 3,
) -> "torch.Tensor":
    """多分类 focal loss。

    Args:
        logits: ``(B, C)``。
        target: ``(B,)``，类别索引 ``∈ [0, C)``。
        gamma: 聚焦参数（0=退化为 CE）。
        num_classes: 类别数。
    """
    import torch
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
    probs = log_probs.exp()  # (B, C)

    # one-hot
    y_onehot = F.one_hot(target, num_classes=num_classes).float()  # (B, C)
    pt = (probs * y_onehot).sum(dim=-1)  # (B,)，真实类概率
    log_pt = (log_probs * y_onehot).sum(dim=-1)  # (B,)

    if gamma > 0:
        focal_factor = (1.0 - pt) ** gamma
        loss = -(focal_factor * log_pt).mean()
    else:
        loss = -log_pt.mean()  # CE
    return loss


# ============================================================
# DL wrapper
# ============================================================

class DLClassifierWrapper:
    """对齐 LightGBM 接口的 Transformer 包装类。

    接口：
        * :meth:`fit(X, y, X_val, y_val, max_epochs, batch_size, patience)` —— 训练；
        * :meth:`predict_proba(X)` —— 推理，返回 ``(n, 3)``；
        * :meth:`save(path)` / :meth:`load(path)` —— 持久化（state_dict + config + scaler）。
    """

    def __init__(
        self,
        n_features: int,
        cfg: Optional[TransformerConfig] = None,
    ):
        self.n_features = n_features
        self.cfg = cfg or TransformerConfig()
        self.device_ = select_device(self.cfg.device)

        # 延迟 import torch
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch 未安装；请先手动 pip install torch"
            ) from e

        self.torch = torch
        set_seed(self.cfg.random_state)

        self.model_ = build_model(n_features, self.cfg).to(self.device_)
        self.scaler_ = SequenceStandardScaler()
        self.history_: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": -1,
        }
        self.fitted_ = False

    # ----- 训练 -----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        max_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        patience: Optional[int] = None,
        verbose: bool = False,
    ) -> "DLClassifierWrapper":
        """训练。

        Args:
            X: ``ndarray[n, F]`` 训练特征。
            y: ``ndarray[n]`` 训练标签。
            X_val: ``ndarray[n_val, F]`` 验证特征（可选）。
            y_val: ``ndarray[n_val]`` 验证标签（可选）。
            max_epochs: 覆盖 cfg。
            batch_size: 覆盖 cfg。
            patience: 覆盖 cfg。
            verbose: 打印 epoch 进度。

        Returns:
            self（链式 API）。
        """
        max_epochs = max_epochs or self.cfg.max_epochs
        batch_size = batch_size or self.cfg.batch_size
        patience = patience or self.cfg.patience

        torch = self.torch
        # 1) 序列构造
        X_seq, y_seq = _make_sequences(X, y, self.cfg.seq_len)
        # 2) 标准化 fit in train
        X_seq = self.scaler_.fit_transform(X_seq)
        # 3) DataLoader
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq), torch.from_numpy(y_seq),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        )
        # 4) Val (optional)
        val_loader = None
        if X_val is not None and y_val is not None:
            Xv_seq, yv_seq = _make_sequences(X_val, y_val, self.cfg.seq_len)
            Xv_seq = self.scaler_.transform(Xv_seq)
            val_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(Xv_seq), torch.from_numpy(yv_seq),
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            )

        # 5) Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )

        # 6) Train loop
        best_val = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(max_epochs):
            self.model_.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = _focal_loss(
                    logits, yb,
                    gamma=self.cfg.focal_gamma,
                    num_classes=self.cfg.num_classes,
                )
                if torch.isnan(loss):
                    raise RuntimeError(
                        "训练 loss=NaN；请降低 lr 或检查数据"
                    )
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()
            train_loss = float(np.mean(train_losses))
            self.history_["train_loss"].append(train_loss)

            # 7) Validation
            val_loss = float("nan")
            if val_loader is not None:
                self.model_.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        logits = self.model_(xb)
                        loss = _focal_loss(
                            logits, yb,
                            gamma=self.cfg.focal_gamma,
                            num_classes=self.cfg.num_classes,
                        )
                        val_losses.append(loss.item())
                val_loss = float(np.mean(val_losses))
                self.history_["val_loss"].append(val_loss)

                # best checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                    no_improve = 0
                    self.history_["best_epoch"] = epoch
                else:
                    no_improve += 1

            if verbose:
                print(
                    f"epoch {epoch + 1}/{max_epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                )

            # 8) Early stopping
            if val_loader is not None and no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # 9) 恢复 best
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.fitted_ = True
        return self

    # ----- 推理 -----

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """推理：``(n, F)`` → ``(n, 3)`` proba。

        Raises:
            RuntimeError: 未训练。
            ValueError: 形状不匹配。
        """
        if not self.fitted_:
            raise RuntimeError("必须先 fit 才能 predict_proba")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X 特征数 {X.shape[1]} != 模型期望 {self.n_features}"
            )

        torch = self.torch
        n = len(X)
        # 边界：测试折长度 < seq_len（walk-forward 末尾或小数据集），
        # 序列不足 ⇒ 返回均匀占位 proba（后续评估/对齐会处理 None）
        if n < self.cfg.seq_len:
            return np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
        X_seq, _ = _make_sequences(X, np.zeros(n, dtype=np.int64), self.cfg.seq_len)
        X_seq = self.scaler_.transform(X_seq)
        X_t = torch.from_numpy(X_seq).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ----- 持久化 -----

    def save(self, path: str) -> None:
        """保存 state_dict + config + scaler 到 ``path``。

        文件：``transformer.pt``（torch zip）+ ``transformer_meta.json``（cfg + scaler）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # state_dict
        torch = self.torch
        torch.save(self.model_.state_dict(), path)
        # meta
        meta = {
            "config": asdict(self.cfg),
            "n_features": self.n_features,
            "scaler": self.scaler_.to_dict(),
            "history": self.history_,
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "DLClassifierWrapper":
        """从 ``path`` 加载（与 :meth:`save` 配对）。"""
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = TransformerConfig(**meta["config"])
        wrapper = cls(n_features=meta["n_features"], cfg=cfg)
        wrapper.scaler_ = SequenceStandardScaler.from_dict(meta["scaler"])
        wrapper.history_ = meta["history"]
        wrapper.model_.load_state_dict(wrapper.torch.load(path, map_location=wrapper.device_))
        wrapper.fitted_ = True
        return wrapper


# ============================================================
# 高层 API（与 train_lightgbm 对齐）
# ============================================================

def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    cfg: Optional[TransformerConfig] = None,
    n_features: Optional[int] = None,
    random_state: int = 42,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[DLClassifierWrapper, Dict[int, int], Dict[int, int], Dict[str, Any]]:
    """高层训练 API，对齐 :func:`model_training.trainer.train_lightgbm` 签名。

    Args:
        X_train: ``ndarray[n, F]`` 训练特征（变化率）。
        y_train: ``ndarray[n]`` 训练标签 ``∈ {+1, 0, -1}``。
        X_val/y_val: 可选验证集。
        cfg: 自定义配置。``None`` → 用默认 + ``random_state/focal_gamma/max_epochs/...`` 覆盖。
        n_features: ``None`` → 用 ``X_train.shape[1]``。
        random_state: 随机种子。
        use_focal: ``True`` → focal loss（gamma=focal_gamma）。
        focal_gamma: focal 聚焦参数。
        max_epochs/batch_size/patience: 覆盖 cfg。

    Returns:
        ``(wrapper, label_map, inverse_map, history)``。

    Raises:
        ImportError: 缺 torch。
    """
    # 共享阶段 B 的 label_map（统一单源；避免重复实现）
    from model_training.trainer import label_map

    # label_map
    y_mapped, lmap, inv = label_map(y_train)

    # config 合并
    n_features = n_features or X_train.shape[1]
    overrides: Dict[str, Any] = {
        "random_state": random_state,
        "focal_gamma": focal_gamma if use_focal else 0.0,
    }
    if max_epochs is not None:
        overrides["max_epochs"] = max_epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if patience is not None:
        overrides["patience"] = patience
    if cfg is None:
        cfg = TransformerConfig(**overrides)
    else:
        # 覆盖 cfg（保持 frozen）
        from dataclasses import replace
        cfg = replace(cfg, **overrides)

    # 训练
    wrapper = DLClassifierWrapper(n_features=n_features, cfg=cfg)
    # y_val 必须独立映射到 {0, 1, 2}（不能用 y_mapped；那是 y_train 映射后的）
    if y_val is not None:
        y_val_mapped, _, _ = label_map(y_val)
    else:
        y_val_mapped = None
    wrapper.fit(
        X_train, y_mapped, X_val,
        y_val_mapped,
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        verbose=verbose,
    )

    history = {
        "best_iteration": wrapper.history_.get("best_epoch", -1),
        "best_val_loss": min(wrapper.history_["val_loss"]) if wrapper.history_["val_loss"] else None,
        "train_loss_final": wrapper.history_["train_loss"][-1] if wrapper.history_["train_loss"] else None,
        "n_features": n_features,
    }
    return wrapper, lmap, inv, history
