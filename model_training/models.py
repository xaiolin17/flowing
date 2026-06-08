"""models —— model_training 包的 dataclass 定义。

4 个领域对象都用 ``@dataclass(frozen=True)`` 修饰：
    * 不可变，避免下游误改内存对象后与配置脱节；
    * 字段顺序与 spec.md "ADDED Requirements" 严格对齐。

设计要点：
    * ``FeatureSpec`` **真做配置**（v2 关键修正）：factors.py 接收本类，
      所有窗口/参数从字段读，**不**写死；
    * ``SplitSpec`` 含 ``train_idx_range`` / ``test_idx_range`` / ``time_range_train`` /
      ``time_range_test`` 四组元数据，便于报告 + 反查；
    * ``EvalReport`` 字段与 ``serialize_eval`` 输出的 JSON 字段一一对应；
    * ``ImbalanceMethod`` 是 ``str`` enum，可直接序列化/反序列化（CLI 解析时
      str → enum）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from typing_extensions import TypeAlias

# ============================================================
# FeatureSpec —— 因子 + 变化率特征规格
# ============================================================

# 时间周期 → 斐波那契 lags（M1 / M5 / M15 / H1 / D1）
# 详见 spec.md §9；其他 TF fallback 到 M1
DEFAULT_FIB_LAGS: Tuple[int, ...] = (
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
)


@dataclass(frozen=True)
class FeatureSpec:
    """技术指标窗口/参数配置（v2 真做配置）。

    ``compute_factors`` 接收本类，**所有**指标窗口从本类的字段读，**不**写死。
    默认值见 :data:`DEFAULT_FACTOR_SPEC`（与 spec.md §9 严格一致）。

    Attributes:
        timeframe: K 线周期字符串（"M1" / "M5" / "M15" / "H1" / "D1" 等）。
        fib_lags: 变化率 lag 列表（默认 M1 全集）。
        ma_windows: 简单移动平均窗口列表（如 (5, 10, 20, 60)）。
        ema_windows: 指数移动平均窗口列表。
        rsi_windows: RSI 窗口列表（如 (6, 12, 24)）。
        macd_params: MACD 参数 (fast, slow, signal) 如 (12, 26, 9)。
        boll_params: BOLL 参数 (n, k) 如 (20, 2)。
        atr_window: ATR 窗口（默认 14）。
        kdj_params: KDJ 参数 (n, m1, m2) 如 (9, 3, 3)。
        cci_window: CCI 窗口（默认 14）。
        wr_window: WR 窗口（默认 14）。
        vma_windows: 成交量 MA 窗口列表（如 (5, 10, 20)）。
    """

    timeframe: str
    fib_lags: Tuple[int, ...] = DEFAULT_FIB_LAGS
    ma_windows: Tuple[int, ...] = (5, 10, 20, 60)
    ema_windows: Tuple[int, ...] = (5, 10, 20, 60)
    rsi_windows: Tuple[int, ...] = (6, 12, 24)
    macd_params: Tuple[int, int, int] = (12, 26, 9)
    boll_params: Tuple[int, float] = (20, 2.0)
    atr_window: int = 14
    kdj_params: Tuple[int, int, int] = (9, 3, 3)
    cci_window: int = 14
    wr_window: int = 14
    vma_windows: Tuple[int, ...] = (5, 10, 20)


# 默认 FeatureSpec：M1 + 全集窗口（与 spec.md §9 严格一致）
DEFAULT_FACTOR_SPEC: FeatureSpec = FeatureSpec(timeframe="M1")


# ============================================================
# SplitSpec —— 时间序列切分元数据
# ============================================================

@dataclass(frozen=True)
class SplitSpec:
    """单个 (train, test) 切分的元数据。

    Attributes:
        label: 切分标识（如 "outer_fold_3" / "inner_fold_1"）。
        train_idx_range: 训练集在原始数据中的 ``(start, end)`` 索引范围（含左不含右）。
        test_idx_range: 测试集索引范围。
        time_range_train: 训练集时间范围 ``(start_iso, end_iso)``。
        time_range_test: 测试集时间范围。
        n_train: 训练样本数。
        n_test: 测试样本数。
        meta: 附加元数据（initial_train / test_size / expanding 等）。
    """

    label: str
    train_idx_range: Tuple[int, int]
    test_idx_range: Tuple[int, int]
    time_range_train: Tuple[str, str]
    time_range_test: Tuple[str, str]
    n_train: int
    n_test: int
    meta: dict = field(default_factory=dict)


# ============================================================
# EvalReport —— 评估报告
# ============================================================

@dataclass(frozen=True)
class EvalReport:
    """单次评估的指标集合。

    字段与 ``evaluator.serialize_eval`` 输出的 JSON 字段一一对应。
    详见 spec.md "评估报告字段"。

    Attributes:
        accuracy: 准确率。
        f1_macro: macro F1 分数。
        per_class_precision: 每类精度，长度 = 类别数。
        per_class_recall: 每类召回率。
        per_class_f1: 每类 F1（v2 新增）。
        auc: multi-class AUC（``multi_class='ovr'``）。
        backtest_return: 简易回测累计收益。
        n_samples: 评估样本数。
        meta: 附加元数据（标签映射、回测参数等）。
    """

    accuracy: float
    f1_macro: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    auc: float
    backtest_return: float
    n_samples: int
    meta: dict = field(default_factory=dict)


# ============================================================
# ImbalanceMethod —— 不平衡处理方法（str enum）
# ============================================================


class ImbalanceMethod(str, Enum):
    """数据倾斜处理方法标识。

    用 ``str`` 混入以便：
        * JSON 序列化时直接 ``json.dumps(method)``（不带 ``ImbalanceMethod.`` 前缀）；
        * CLI 解析时 ``ImbalanceMethod(args.imbalance)`` 自动报错（无效值）。

    Values:
        NONE: 不处理（保留全部样本作基线 D）。
        SEGMENT_MERGE: 段内合并（基线 A），基于 raw close + atr14。
        FOCAL: Focal Loss（基线 B），不下采样，标记 ``use_focal=True``。
        RANDOM_DOWNSAMPLE: 随机下采样 majority（基线 C）。
    """

    NONE = "none"
    SEGMENT_MERGE = "segment_merge"
    FOCAL = "focal"
    RANDOM_DOWNSAMPLE = "random_downsample"


# ============================================================
# 便捷类型别名（仅文档用途，不参与运行时）
# ============================================================

LabelMap: TypeAlias = dict[int, int]
InverseLabelMap: TypeAlias = dict[int, int]
