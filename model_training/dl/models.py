"""models.py —— Transformer 三分类模型（阶段 C）。

提供：
    * :class:`TransformerConfig` —— 不可变 dataclass，模型 + 训练超参数；
    * :class:`SimpleTimeSeriesTransformer` —— 简化版时序 Transformer
      （input projection + learnable pos enc + encoder + global pool + linear）。

设计要点（与 spec.md "模型设计" 严格对齐）：
    * 单一模型族（不 LSTM/不 TCN）；
    * 默认配置 d_model=64 / n_heads=4 / n_layers=2（CPU 友好）；
    * 6GB VRAM 容量充足（实际峰值 < 500MB）；
    * 关键防御：n_heads 必须整除 d_model（构造时校验）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# 配置
# ============================================================

@dataclass(frozen=True)
class TransformerConfig:
    """Transformer + 训练超参数（frozen=True，构造后不可变）。

    字段：
        * 架构：d_model / n_heads / n_layers / dim_ff / dropout / max_seq_len；
        * 训练：seq_len / lr / weight_decay / batch_size / max_epochs / patience；
        * 损失：focal_gamma（0=CE，>0=Focal）；
        * 设备：device（None=自动检测）；
        * 随机：random_state。
    """
    # 架构
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512
    num_classes: int = 3

    # 训练
    seq_len: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5

    # 损失
    focal_gamma: float = 0.0  # 0=CrossEntropy, >0=Focal

    # 设备与种子
    device: Optional[str] = None  # None=自动检测
    random_state: int = 42

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} 必须能被 n_heads={self.n_heads} 整除"
            )
        if self.seq_len < 1:
            raise ValueError(f"seq_len={self.seq_len} 必须 >= 1")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout={self.dropout} 必须在 [0, 1]")
        if self.max_seq_len < self.seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} 应 >= seq_len={self.seq_len}"
            )


# ============================================================
# 模型
# ============================================================

class SimpleTimeSeriesTransformer(nn.Module):
    """简化版时序 Transformer 三分类。

    架构（与 spec.md "3.1 架构" 一致）::

        Input (B, T, F)
            ↓
        input_proj: Linear(F → d_model)
            ↓
        + pos_enc: (1, max_seq_len, d_model)  # 可学习位置编码
            ↓
        TransformerEncoder × n_layers
            ↓
        Global average pooling (over time)
            ↓
        LayerNorm + Linear(d_model → num_classes)
            ↓
        Output (B, num_classes) logits
    """

    def __init__(
        self,
        n_features: int,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_classes: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # 1) Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # 2) Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # 3) TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4) Head: LayerNorm + Linear
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward：``(B, T, F)`` → ``(B, num_classes)`` logits。

        Args:
            x: 输入序列，shape ``(batch, seq_len, n_features)``。

        Returns:
            logits，shape ``(batch, num_classes)``（**未** softmax，调用方决定）。
        """
        B, T, _ = x.shape
        if T > self.pos_enc.shape[1]:
            raise ValueError(
                f"序列长度 T={T} 超过 max_seq_len={self.pos_enc.shape[1]}"
            )

        # 1) Input projection
        h = self.input_proj(x)  # (B, T, d_model)

        # 2) + positional encoding
        h = h + self.pos_enc[:, :T, :]

        # 3) Encoder
        h = self.encoder(h)  # (B, T, d_model)

        # 4) Global average pooling + head
        h = h.mean(dim=1)  # (B, d_model)
        logits = self.head(h)  # (B, num_classes)
        return logits

    def num_parameters(self) -> int:
        """模型参数量（用于 sanity check）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 工厂
# ============================================================

def build_model(n_features: int, cfg: TransformerConfig) -> SimpleTimeSeriesTransformer:
    """从 :class:`TransformerConfig` 构造模型。

    便捷工厂，避免重复传参。
    """
    return SimpleTimeSeriesTransformer(
        n_features=n_features,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        num_classes=cfg.num_classes,
    )
