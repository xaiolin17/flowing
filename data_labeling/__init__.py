"""data_labeling —— K 线可视化打标签 + SQLite 持久化（阶段 A）。

本包提供:
    * ``Dataset`` / ``Candle`` / ``Label`` 三个数据模型（见 :mod:`.models`）；
    * SQLite + DAO 访问层（见 :mod:`.db`）；
    * ``make_dataset_name`` / ``import_from_mt5`` 两个高层入口（见 :mod:`.persistence`）；
    * 基于 PyQt5 + pyqtgraph 的可视化打标签 GUI（见 :mod:`.app`）。

依赖说明（重要）
================
本包**不**在 ``pip install`` 时自动拉取 PyQt5 —— 遵守「不在本项目内执行 pip 下载」约束。
请用户自行在虚拟环境中执行::

    pip install PyQt5 pyqtgraph

DAO / persistence / models 不依赖 PyQt5，可在 CI / 单元测试中无 GUI 环境下跑通。
GUI 模块（``app.py`` / ``main.py``）只在使用 ``python data_labeling/main.py`` 时才 import PyQt5。
"""

# 包的对外版本号。后续小版本升级（添加新字段、新 DAO）时同步更新。
__version__ = "0.1.0"

# 延迟到调用方触发 import 顶部导出，避免 DAO 单测被 PyQt5 拖入 sys.path。
from .models import Dataset, Candle, Label  # noqa: E402

__all__ = [
    "Dataset",
    "Candle",
    "Label",
    "__version__",
]
