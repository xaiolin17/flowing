"""db —— SQLite 连接 + DAO 访问层。

本模块**不**依赖 PyQt5，可在任何 Python 环境下 import 与单测。

约定：
    * 数据库文件路径由调用方传入，缺省 ``data_labeling/data.db``；
    * ``init_db`` 幂等：可重复调用，仅在表不存在时建表；
    * DAO 实例短生命周期：每次操作都新开 connection（SQLite 单文件足够快）；
    * 所有 datetime 进出 DB 走 ISO 8601 字符串（``T`` 分隔，naive UTC），与
      ``fetch_rates`` 返回的索引类型保持一致。

Schema（与 spec.md "ADDED Requirements" 一致）：

    datasets(id, name UNIQUE, symbol, timeframe, date_from, date_to,
             row_count, created_at, notes)
    candles(dataset_id, time, open, high, low, close, tick_volume,
            PK(dataset_id, time))
    labels(dataset_id, time, label, labeled_at, note,
           PK(dataset_id, time))
    factors(dataset_id, time, name, value,
            PK(dataset_id, time, name))  -- 阶段 B 使用，阶段 A 留空
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from .models import Candle, Dataset, Label


# ---- 默认 DB 路径：data_labeling/data.db ----
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data.db"


# ---- DDL ----
_DDL_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    date_from   TEXT    NOT NULL,
    date_to     TEXT,
    row_count   INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    notes       TEXT
);
"""

_DDL_CANDLES = """
CREATE TABLE IF NOT EXISTS candles (
    dataset_id  INTEGER NOT NULL,
    time        TEXT    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    tick_volume INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, time),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
"""

_DDL_LABELS = """
CREATE TABLE IF NOT EXISTS labels (
    dataset_id  INTEGER NOT NULL,
    time        TEXT    NOT NULL,
    label       INTEGER NOT NULL,
    labeled_at  TEXT    NOT NULL,
    note        TEXT,
    PRIMARY KEY (dataset_id, time),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
"""

_DDL_FACTORS = """
CREATE TABLE IF NOT EXISTS factors (
    dataset_id  INTEGER NOT NULL,
    time        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    value       REAL    NOT NULL,
    PRIMARY KEY (dataset_id, time, name),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
"""

# 索引：candles 按 time 范围查得多，建索引提升扫描速度。
_DDL_CANDLES_TIME_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(dataset_id, time);"
)
_DDL_LABELS_TIME_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_labels_time ON labels(dataset_id, time);"
)


def _connect(db_path: Path) -> sqlite3.Connection:
    """打开一个 SQLite 连接并启用必要 PRAGMA。"""
    conn = sqlite3.connect(str(db_path))
    # 让 SELECT 结果可以按列名访问（row["name"]）
    conn.row_factory = sqlite3.Row
    # 启用外键约束（默认 OFF；不开启则 ON DELETE CASCADE 不会触发）
    conn.execute("PRAGMA foreign_keys = ON;")
    # WAL 模式提升并发读写（GUI 标 + 后台训练读可同时进行）
    conn.execute("PRAGMA journal_mode = WAL;")
    # 同步策略：NORMAL 即可（WAL 模式下不会丢数据）
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """获取一个 SQLite 连接的 context manager。

    用法::

        with get_connection() as conn:
            conn.execute(...)

    Args:
        db_path: 数据库文件路径；None 时用 :data:`DEFAULT_DB_PATH`。
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    # 父目录不存在时建出来（首次运行场景）
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> Path:
    """初始化数据库（建表 + 索引）。幂等，可重复调用。

    Returns:
        实际使用的数据库文件绝对路径。
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    with get_connection(path) as conn:
        conn.executescript(
            _DDL_DATASETS + _DDL_CANDLES + _DDL_LABELS + _DDL_FACTORS
        )
        conn.execute(_DDL_CANDLES_TIME_IDX)
        conn.execute(_DDL_LABELS_TIME_IDX)
    return path


# ---- 时间字段序列化辅助 ----

def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    """datetime -> ISO 8601 字符串。None 透传。"""
    if dt is None:
        return None
    # naive UTC：直接 isoformat()，避免出现 "+00:00" 后缀
    return dt.replace(tzinfo=None).isoformat()


# ============================================================
# DatasetDAO
# ============================================================

class DatasetDAO:
    """``datasets`` 表的访问层。

    使用方式::

        dao = DatasetDAO(db_path)
        ds = dao.create(name=..., symbol=..., ...)
        rows = dao.list_all()
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def create(
        self,
        name: str,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: Optional[datetime],
        row_count: int,
        notes: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Dataset:
        """插入一行 dataset 并返回带 id 的实例。

        Raises:
            ValueError: name 已存在（UNIQUE 冲突）。
        """
        created = created_at or datetime.utcnow()
        with get_connection(self.db_path) as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO datasets
                        (name, symbol, timeframe, date_from, date_to,
                         row_count, created_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        symbol,
                        timeframe,
                        _to_iso(date_from),
                        _to_iso(date_to),
                        int(row_count),
                        _to_iso(created),
                        notes,
                    ),
                )
            except sqlite3.IntegrityError as e:
                # UNIQUE 冲突 => 重名
                raise ValueError(f"dataset name '{name}' already exists") from e
            new_id = int(cur.lastrowid)
        return Dataset(
            id=new_id,
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            row_count=row_count,
            created_at=created,
            notes=notes,
        )

    def list_all(self) -> List[Dataset]:
        """按 created_at DESC 返回所有 dataset。"""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC"
            ).fetchall()
        return [Dataset.from_row(r) for r in rows]

    def get_by_name(self, name: str) -> Optional[Dataset]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE name = ?", (name,)
            ).fetchone()
        return Dataset.from_row(row) if row else None

    def get_by_id(self, ds_id: int) -> Optional[Dataset]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ?", (ds_id,)
            ).fetchone()
        return Dataset.from_row(row) if row else None

    def delete(self, ds_id: int) -> int:
        """删除一个 dataset，关联 candles / labels / factors 由 CASCADE 清理。

        Returns:
            被影响的 dataset 行数（0 或 1）。
        """
        with get_connection(self.db_path) as conn:
            cur = conn.execute("DELETE FROM datasets WHERE id = ?", (ds_id,))
            return cur.rowcount


# ============================================================
# CandleDAO
# ============================================================

class CandleDAO:
    """``candles`` 表的访问层。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def bulk_insert(self, candles: Sequence[Candle]) -> int:
        """批量插入 candles。用 ``executemany`` 一次性写入。

        Args:
            candles: Candle dataclass 列表，必须 ``dataset_id`` 一致。

        Returns:
            写入的行数。
        """
        if not candles:
            return 0
        rows = [
            (
                c.dataset_id,
                _to_iso(c.time),
                float(c.open),
                float(c.high),
                float(c.low),
                float(c.close),
                int(c.tick_volume),
            )
            for c in candles
        ]
        with get_connection(self.db_path) as conn:
            # INSERT OR IGNORE：相同 (dataset_id, time) 重复不会爆错，便于断点续传。
            conn.executemany(
                """
                INSERT OR IGNORE INTO candles
                    (dataset_id, time, open, high, low, close, tick_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_by_dataset(self, dataset_id: int) -> List[Candle]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT dataset_id, time, open, high, low, close, tick_volume
                FROM candles
                WHERE dataset_id = ?
                ORDER BY time ASC
                """,
                (dataset_id,),
            ).fetchall()
        return [Candle.from_row(r) for r in rows]


# ============================================================
# LabelDAO
# ============================================================

class LabelDAO:
    """``labels`` 表的访问层。

    标签值：
        +1 = 买；-1 = 卖；0 = 显式标记"不操作"。
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def upsert(
        self,
        dataset_id: int,
        time: datetime,
        value: int,
        note: Optional[str] = None,
        labeled_at: Optional[datetime] = None,
    ) -> Label:
        """插入或更新一个标签。"""
        if value not in (1, -1, 0):
            raise ValueError(f"label value 必须是 +1 / -1 / 0，得到 {value!r}")
        ts = labeled_at or datetime.utcnow()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO labels (dataset_id, time, label, labeled_at, note)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, time) DO UPDATE SET
                    label = excluded.label,
                    labeled_at = excluded.labeled_at,
                    note = excluded.note
                """,
                (
                    dataset_id,
                    _to_iso(time),
                    int(value),
                    _to_iso(ts),
                    note,
                ),
            )
        return Label(
            dataset_id=dataset_id,
            time=time,
            value=value,
            labeled_at=ts,
            note=note,
        )

    def delete(self, dataset_id: int, time: datetime) -> int:
        """删除一条标签。"""
        with get_connection(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM labels WHERE dataset_id = ? AND time = ?",
                (dataset_id, _to_iso(time)),
            )
            return cur.rowcount

    def list_by_dataset(self, dataset_id: int) -> List[Label]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT dataset_id, time, label, labeled_at, note
                FROM labels
                WHERE dataset_id = ?
                ORDER BY time ASC
                """,
                (dataset_id,),
            ).fetchall()
        return [Label.from_row(r) for r in rows]
