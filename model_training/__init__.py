"""model_training —— 阶段 B：特征工程 + 数据倾斜 + 监督学习模型训练。

本包提供：
    * ``FeatureSpec`` / ``SplitSpec`` / ``EvalReport`` / ``ImbalanceMethod``
      四个领域模型（见 :mod:`.models`）；
    * ``data_loader`` —— 从 :mod:`data_labeling` SQLite 拉数据（含 is_labeled mask）；
    * ``factors`` —— 自实现技术指标（FeatureSpec 驱动，29 列）；
    * ``features`` —— 变化率（fib_lags + logret/delta 三分法）+ 训练矩阵白名单选列 +
      段内合并入口；
    * ``imbalance`` —— 4 基线 + 1 dummy baseline + ImbalanceMethod 工厂；
    * ``splitter`` —— walk-forward 切分（外层 + 内层 + time_aware）；
    * ``trainer`` —— LightGBM 三分类 + label_map 双向 + focal loss + OOF；
    * ``stacker`` —— OOF → LR 第二层（仅横向 concat）；
    * ``evaluator`` —— 指标 + 简易回测 + dummy baseline 对比；
    * ``main`` —— 一键训练 CLI。

依赖说明（重要）
================
本包**不**在 ``pip install`` 时自动拉取 lightgbm / shap / optuna —— 遵守
「不在本项目内执行 pip 下载」约束。请用户自行在虚拟环境中执行::

    pip install lightgbm shap optuna

包设计原则：
    * **顶部不 import 任何重依赖**（如 lightgbm）：仅在 ``trainer.train_lightgbm``
      函数体内 import，保证 ``import model_training`` 不依赖 lightgbm；
    * **只读消费** ``data_labeling``：通过 DAO 读数据，**不**写回 SQLite；
    * **训练产物**（model / OOF / 报告）落 :data:`ARTIFACTS_DIR`，**不**进 SQLite。
"""

from __future__ import annotations

from pathlib import Path

# 包的对外版本号。后续小版本升级（添加新字段、新 DAO）时同步更新。
__version__ = "0.1.0"

# artifacts 落盘目录（gitignore 屏蔽）
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"

# 延迟到调用方触发 import 顶部导出，避免 model_training 包被外部模块 import 时
# 把 dataclass 也强拉进来（外部只需 ARTIFACTS_DIR / __version__ 时可走 importlib）。
from .models import (DEFAULT_FACTOR_SPEC, EvalReport,  # noqa: E402
                     FeatureSpec, ImbalanceMethod, SplitSpec)

__all__ = [
    "ARTIFACTS_DIR",
    "DEFAULT_FACTOR_SPEC",
    "EvalReport",
    "FeatureSpec",
    "ImbalanceMethod",
    "SplitSpec",
    "__version__",
]
