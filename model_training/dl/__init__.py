"""深度学习模型子包（阶段 C）。

模块（**顶部不** import torch，延迟到函数体内）：
    * :mod:`.models` —— ``SimpleTimeSeriesTransformer`` nn.Module + ``TransformerConfig``。
    * :mod:`.trainer` —— ``DLClassifierWrapper``（对齐 LightGBM 接口）+ ``train_transformer``。

设计要点（与 spec.md "范围与非目标" / "训练循环" 严格对齐）：
    * 只 Transformer 模型（不 LSTM/TCN/CNN/N-BEATS）；
    * 自动设备检测（CPU 默认，CUDA 可用时自动用）；
    * 延迟导入 torch，缺包抛 ImportError；
    * 与阶段 B pipeline 接口对齐：``fit / predict_proba / save / load``。
"""
from __future__ import annotations

__all__ = ["models", "trainer"]
