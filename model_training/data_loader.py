"""data_loader —— 从 :mod:`data_labeling` SQLite 拉数据（只读消费）。

提供：
    * :func:`list_datasets` —— 列出所有 dataset；
    * :func:`load_dataset` —— 拉一个 dataset 的 raw OHLCV + 标签 + **is_labeled mask**。

设计要点（与 spec.md "数据加载" Requirement 严格对齐）：
    * 返回 **三元组** ``(X_raw, y, is_labeled)``，**不**是 ``(X, y)``；
    * ``is_labeled[i] == True`` ⟺ i 时刻在 ``labels`` 表存在**真实行**
      （通过 set 比较显式生成，**不**靠 y 值推断）；
    * 未打标签的 K 线 y=0 + ``is_labeled=False``；显式标 0 的 K 线 y=0 + ``is_labeled=True``；
    * 内部缓存 ``_cache: dict[(ds_id, str(db_path))] -> (df, y, is_labeled)``，
      同进程内连续调用不发 SQL；
    * **只读消费** ``data_labeling``：本模块**不**导入 ``data_labeling.persistence``
      的写路径。

防混淆语义（v2 关键）：
    * y=0 **不一定**代表"未打"——也可能是用户显式标"不操作"；
    * 必须用 ``is_labeled`` mask 区分：未打 = 沉默多数（应被剔除），
      显式 0 = 真实信号（应保留为"不操作"类）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_labeling.db import CandleDAO, DatasetDAO, LabelDAO  # noqa: E402
from data_labeling.models import Dataset  # noqa: E402


# 进程内缓存：键 (ds_id, str(db_path)) -> (X_raw, y, is_labeled)
_cache: Dict[Tuple[int, str], Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}


def list_datasets(db_path: Optional[Path] = None) -> List[Dataset]:
    """列出所有 dataset（按 created_at DESC）。

    包装 :class:`data_labeling.DatasetDAO.list_all`。
    """
    return DatasetDAO(db_path).list_all()


def _candles_to_df(candles) -> pd.DataFrame:
    """把 Candle 列表转成 ``pd.DataFrame``，索引为 time（datetime）。

    列固定为 ``[open, high, low, close, tick_volume]``，dtype 全部 numeric。
    """
    df = pd.DataFrame(
        [
            {
                "time": c.time,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "tick_volume": int(c.tick_volume),
            }
            for c in candles
        ]
    )
    if df.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "tick_volume"]
        ).set_index(pd.DatetimeIndex([], name="time"))
    df = df.set_index("time").sort_index()
    return df


def load_dataset(
    dataset_id: int, db_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """拉一个 dataset 的 raw OHLCV + 标签 + is_labeled mask。

    流程：
        1. 缓存命中 → 直接返回；
        2. 调 :class:`CandleDAO.list_by_dataset` 拉 candles（按 time ASC）；
        3. 调 :class:`LabelDAO.list_by_dataset` 拉 labels；
        4. 构造 ``{time -> value}`` 字典；
        5. **左连接**到 candles：未打填 0，得到 y；
        6. **显式**生成 is_labeled（set 比较，**不**靠 y 值推断）；
        7. 缓存。

    Args:
        dataset_id: dataset 主键。
        db_path: SQLite 路径；None 用 :data:`data_labeling.db.DEFAULT_DB_PATH`。

    Returns:
        ``(X_raw, y, is_labeled)`` 三元组：
            * X_raw: ``pd.DataFrame``，索引 time，列 ``[open, high, low, close, tick_volume]``；
            * y: ``ndarray[N] of int64``，取值 ``{+1, 0, -1}``；
            * is_labeled: ``ndarray[N] of bool``，标记打过标签的位置。

    Raises:
        ValueError: dataset_id 不存在。
    """
    key = (dataset_id, str(db_path) if db_path is not None else "")
    if key in _cache:
        return _cache[key]

    # 1) 验证 dataset 存在
    ds = DatasetDAO(db_path).get_by_id(dataset_id)
    if ds is None:
        raise ValueError(f"dataset_id={dataset_id} 不存在")

    # 2) 拉 candles
    candles = CandleDAO(db_path).list_by_dataset(dataset_id)
    df = _candles_to_df(candles)

    # 3) 拉 labels
    labels = LabelDAO(db_path).list_by_dataset(dataset_id)
    label_dict: Dict[pd.Timestamp, int] = {pd.Timestamp(la.time): int(la.value) for la in labels}

    # 4) 构造 y + is_labeled
    n = len(df)
    y = np.zeros(n, dtype=np.int64)
    is_labeled = np.zeros(n, dtype=bool)
    if n > 0:
        # 用索引对齐：loc[time] 在 label_dict 中则标记 + 填值
        # 为保证精确对齐，遍历索引（一次性）
        idx = df.index
        # 用 dict 的 get + 时间戳精确匹配
        for i, ts in enumerate(idx):
            ts_key = pd.Timestamp(ts)
            if ts_key in label_dict:
                y[i] = label_dict[ts_key]
                is_labeled[i] = True

    # 5) 缓存
    _cache[key] = (df, y, is_labeled)
    return df, y, is_labeled


def clear_cache() -> None:
    """清空进程内缓存（测试用）。"""
    _cache.clear()
