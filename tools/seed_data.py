"""seed_data —— 构造一个真实形状的 XAUUSD M1 dataset + 伪 labels。

用途：
    * 阶段 B 实施 / e2e 验证时快速生成可用的 dataset_id（无需启动 GUI）；
    * 复现：每次运行 np.random.default_rng(seed) 固定，确保可重入；
    * 构造的 OHLCV 用对数随机游走 + 合理波动率（模拟 XAUUSD 价格范围）；
    * 伪 labels 用结构化生成（局部趋势 + 显式 0 + ±1 + 未打段）。

不依赖 mt5_data / MetaTrader5 SDK（用户机器可能没启动 MT5）。
仅依赖 numpy + pandas + data_labeling DAO（阶段 A 阶段 B 共享基础设施）。

典型用法（PowerShell）::

    python tools/seed_data.py --n-rows 1500 --label-count 150 --seed 42
    python tools/seed_data.py --n-rows 3000 --label-count 200 --db-path tests/_tmp/test_seed.db
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# 把项目根加到 sys.path，使 import data_labeling.* 能用
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_labeling.db import CandleDAO, DatasetDAO, LabelDAO, init_db  # noqa: E402
from data_labeling.models import Candle, Dataset  # noqa: E402
from data_labeling.persistence import make_dataset_name  # noqa: E402


def synth_ohlcv(n: int, seed: int, start_price: float = 2050.0) -> pd.DataFrame:
    """合成 n 根 M1 OHLCV。

    形状：
        * ``log P_t = log P_{t-1} + N(0, sigma)`` 的对数随机游走；
        * ``open_t = close_{t-1}``（开盘紧接前收）；
        * ``high_t = max(open, close) + |N(0, sigma/2)|``；
        * ``low_t = min(open, close) - |N(0, sigma/2)|``；
        * ``tick_volume_t = Poisson(120) + 50``。

    与 fetch_rates 返回的 DataFrame 格式严格一致：索引 time，列
    ``[open, high, low, close, tick_volume]``。

    Args:
        n: 蜡烛条数。
        seed: 随机种子。
        start_price: 起始价格（XAUUSD 模拟 2050 美元/盎司附近）。

    Returns:
        pandas DataFrame，索引 time（DST-naive UTC），列如上。
    """
    rng = np.random.default_rng(seed)
    sigma = 0.0008  # ~ 0.08% / minute（XAUUSD M1 的合理波动）

    # 对数随机游走
    log_returns = rng.normal(0.0, sigma, n)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    closes = np.exp(log_prices)

    # 开盘 = 前收（第一根用 start_price）
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    # high / low = max/min(o, c) ± 半 sigma
    intra = np.abs(rng.normal(0.0, sigma / 2, n))
    highs = np.maximum(opens, closes) + intra
    lows = np.minimum(opens, closes) - intra

    # tick_volume：泊松 + base
    volumes = rng.poisson(120, n) + 50

    # 时间索引：M1
    times = pd.date_range(
        "2026-05-01 00:00:00", periods=n, freq="1min"
    )

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes.astype(int),
        },
        index=times,
    )
    df.index.name = "time"
    return df


def make_pseudo_labels(
    n: int,
    label_count: int,
    seed: int,
    ratios: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> List[tuple[int, int]]:
    """生成伪 labels 列表。

    标签语义：
        * 0  = 显式标记"不操作"；
        * +1 = 买（看多）；
        * -1 = 卖（看空）。

    分布：
        * ``ratios`` 控制 0 / +1 / -1 的相对比例，默认 4:4:2（显式 0 与 ±1 各半，
          ±1 中 +1 略多于 -1 模拟轻微多头偏好）；
        * **不**给所有 K 线打标签（label_count < n），剩余"未打"的 K 线
          ``is_labeled == False``，模拟用户忽略的"沉默多数"。

    Args:
        n: 蜡烛总数。
        label_count: 拟打标签数（不超过 n 的 30%，避免过度标注）。
        seed: 随机种子。

    Returns:
        ``[(time_index, value), ...]`` 列表，``time_index ∈ [0, n)``。
    """
    if label_count < 0 or label_count > n:
        raise ValueError(f"label_count={label_count} 越界 [0, {n}]")
    if label_count > int(n * 0.3):
        raise ValueError(
            f"label_count={label_count} 超过 n * 30%={int(n * 0.3)}，"
            "避免过度标注"
        )

    rng = np.random.default_rng(seed + 1)
    indices = rng.choice(n, size=label_count, replace=False)
    indices.sort()  # 时间顺序

    # 按 ratios 分配 0 / +1 / -1
    p0, p_pos, p_neg = ratios
    total = p0 + p_pos + p_neg
    p0, p_pos, p_neg = p0 / total, p_pos / total, p_neg / total

    labels: List[tuple[int, int]] = []
    for idx in indices:
        u = rng.random()
        if u < p0:
            value = 0
        elif u < p0 + p_pos:
            value = 1
        else:
            value = -1
        labels.append((int(idx), value))
    return labels


def seed_to_sqlite(
    n_rows: int,
    label_count: int,
    seed: int,
    db_path: Path,
    symbol: str = "XAUUSDm",
    timeframe: str = "M1",
) -> Dataset:
    """构造 + 落库 + 返回 Dataset。

    步骤：
        1. 构造 OHLCV；
        2. 初始化 DB；
        3. 创建 dataset 行（自动生成 name）；
        4. 批量写 candles；
        5. 写伪 labels（upsert）。

    Returns:
        新创建的 ``Dataset`` 实例（带 id）。
    """
    # 1) 构造数据
    df = synth_ohlcv(n_rows, seed)
    labels = make_pseudo_labels(n_rows, label_count, seed)

    # 2) DB
    init_db(db_path)

    # 3) Dataset
    name = make_dataset_name(
        symbol, timeframe,
        df.index[0].to_pydatetime(),
        df.index[-1].to_pydatetime(),
    )
    ds_dao = DatasetDAO(db_path)
    try:
        ds = ds_dao.create(
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=df.index[0].to_pydatetime(),
            date_to=df.index[-1].to_pydatetime(),
            row_count=len(df),
            notes=f"seed_data(n={n_rows}, labels={label_count}, seed={seed})",
        )
    except ValueError as e:
        # 重名：删旧的，重建
        print(f"[WARN] {e}; 清理后重建", file=sys.stderr)
        old = ds_dao.get_by_name(name)
        if old is not None:
            ds_dao.delete(old.id)
        ds = ds_dao.create(
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=df.index[0].to_pydatetime(),
            date_to=df.index[-1].to_pydatetime(),
            row_count=len(df),
            notes=f"seed_data(n={n_rows}, labels={label_count}, seed={seed})",
        )

    # 4) Candles
    candles = [
        Candle(
            dataset_id=ds.id,
            time=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            tick_volume=int(row["tick_volume"]),
        )
        for ts, row in df.iterrows()
    ]
    CandleDAO(db_path).bulk_insert(candles)

    # 5) Labels
    label_dao = LabelDAO(db_path)
    labeled_at = datetime.utcnow()
    for idx, value in labels:
        ts = df.index[idx].to_pydatetime() if hasattr(df.index[idx], "to_pydatetime") else df.index[idx]
        label_dao.upsert(
            dataset_id=ds.id,
            time=ts,
            value=value,
            note="seed_data pseudo",
            labeled_at=labeled_at,
        )

    return ds


def main() -> int:
    p = argparse.ArgumentParser(description="Seed synthetic XAUUSD M1 dataset + pseudo labels")
    p.add_argument("--n-rows", type=int, default=1500, help="蜡烛条数（默认 1500）")
    p.add_argument("--label-count", type=int, default=150, help="伪标签数（默认 150，约 10%）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data_labeling" / "data.db",
        help="SQLite 路径（默认 data_labeling/data.db）",
    )
    args = p.parse_args()

    ds = seed_to_sqlite(
        n_rows=args.n_rows,
        label_count=args.label_count,
        seed=args.seed,
        db_path=args.db_path,
    )
    print(f"[OK] seed_data 写入成功")
    print(f"     dataset_id = {ds.id}")
    print(f"     name       = {ds.name}")
    print(f"     db_path    = {args.db_path}")
    print(f"     n_rows     = {args.n_rows}")
    print(f"     labels     = {args.label_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
