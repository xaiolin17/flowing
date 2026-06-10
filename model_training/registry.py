"""registry.py —— MLflow-lite 模型注册表（Section 4 核心）。

提供：
    * :class:`RunRecord` —— 一条训练 run 的元数据 dataclass；
    * :func:`init_db` —— 首次启动时建表；
    * :func:`register_run` —— 训练完成后写入一条 run 记录；
    * :func:`list_runs` —— 按字段过滤 + 排序读出 run 列表；
    * :func:`main` —— CLI 入口（``list --top N --by f1_delta``）。

设计要点（与 spec.md "Section 4: 模型注册表" Requirement 严格对齐）：
    * 本地 SQLite（stdlib sqlite3，无 pip）；
    * 训练完成后 main.py 调 :func:`register_run`；
    * CLI ``--registry-db PATH`` 默认 ``model_training/model_registry.db``；
    * schema 字段：dataset_id, model_type, imbalance, stacking, timestamp,
      f1_macro, f1_delta, is_meaningful, artifact_dir, model_signature。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


# ============================================================
# RunRecord
# ============================================================


@dataclass
class RunRecord:
    """一条训练 run 的元数据。

    字段与 SQLite ``runs`` 表一一对应。
    """

    dataset_id: str
    model_type: str
    imbalance: str
    stacking: str
    timestamp: str
    f1_macro: float
    f1_delta: float
    is_meaningful: bool
    artifact_dir: str
    model_signature: str


# ============================================================
# Schema / init_db
# ============================================================


RUNS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      TEXT    NOT NULL,
    model_type      TEXT    NOT NULL,
    imbalance       TEXT    NOT NULL,
    stacking        TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    f1_macro        REAL    NOT NULL,
    f1_delta        REAL    NOT NULL,
    is_meaningful   INTEGER NOT NULL,  -- 0/1
    artifact_dir    TEXT    NOT NULL,
    model_signature TEXT    NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    """首次启动时建表。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(RUNS_TABLE_SCHEMA)
        conn.commit()


# ============================================================
# register_run / list_runs
# ============================================================


def register_run(db_path: Path, rec: RunRecord) -> None:
    """写入一条 run 记录到 DB。"""
    db_path = Path(db_path)
    init_db(db_path)  # 幂等：建表
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                dataset_id, model_type, imbalance, stacking, timestamp,
                f1_macro, f1_delta, is_meaningful, artifact_dir, model_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.dataset_id, rec.model_type, rec.imbalance, rec.stacking,
                rec.timestamp, rec.f1_macro, rec.f1_delta,
                1 if rec.is_meaningful else 0,
                rec.artifact_dir, rec.model_signature,
            ),
        )
        conn.commit()


def list_runs(
    db_path: Path,
    top_n: Optional[int] = None,
    by: str = "f1_delta",
    model_type: Optional[str] = None,
) -> List[RunRecord]:
    """按字段过滤 + 排序读出 run 列表。

    Args:
        db_path: SQLite DB 路径。
        top_n: 限制返回数量（None = 全部）。
        by: 排序字段（默认 f1_delta 倒序）；支持 f1_macro / f1_delta / timestamp。
        model_type: 过滤 model_type（None = 不过滤）。
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    # 排序字段白名单（防 SQL 注入）
    valid_by = {"f1_macro", "f1_delta", "timestamp", "id"}
    if by not in valid_by:
        by = "f1_delta"
    sql = (
        f"SELECT dataset_id, model_type, imbalance, stacking, timestamp, "
        f"f1_macro, f1_delta, is_meaningful, artifact_dir, model_signature "
        f"FROM runs"
    )
    args: list = []
    if model_type is not None:
        sql += " WHERE model_type = ?"
        args.append(model_type)
    sql += f" ORDER BY {by} DESC"
    if top_n is not None:
        sql += " LIMIT ?"
        args.append(int(top_n))

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(sql, args)
        rows = cur.fetchall()
    out: List[RunRecord] = []
    for row in rows:
        out.append(RunRecord(
            dataset_id=row[0],
            model_type=row[1],
            imbalance=row[2],
            stacking=row[3],
            timestamp=row[4],
            f1_macro=row[5],
            f1_delta=row[6],
            is_meaningful=bool(row[7]),
            artifact_dir=row[8],
            model_signature=row[9],
        ))
    return out


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """``python -m model_training.registry`` CLI 入口。

    子命令：
        * ``list`` —— 列出训练 run（``--top N`` + ``--by FIELD`` +
          ``--model-type LIGHTGBM``）；
    """
    parser = argparse.ArgumentParser(
        description="MLflow-lite 模型注册表（Section 4）",
        prog="python -m model_training.registry",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出训练 run")
    p_list.add_argument(
        "--db", type=Path,
        default=Path("model_training/model_registry.db"),
        help="SQLite DB 路径（默认 model_training/model_registry.db）",
    )
    p_list.add_argument(
        "--top", type=int, default=None,
        help="限制返回数量（默认全部）",
    )
    p_list.add_argument(
        "--by", type=str, default="f1_delta",
        help="排序字段（f1_delta / f1_macro / timestamp，默认 f1_delta 倒序）",
    )
    p_list.add_argument(
        "--model-type", type=str, default=None,
        help="过滤 model_type（默认不过滤）",
    )

    args = parser.parse_args()

    if args.command == "list":
        runs = list_runs(
            db_path=args.db, top_n=args.top, by=args.by,
            model_type=args.model_type,
        )
        # 表格化输出
        if not runs:
            print("（无记录）")
            return 0
        print(f"{'dataset_id':<20} {'model_type':<12} {'f1_macro':<10} "
              f"{'f1_delta':<10} {'meaningful':<11} {'artifact_dir'}")
        print("-" * 100)
        for r in runs:
            print(
                f"{r.dataset_id:<20} {r.model_type:<12} "
                f"{r.f1_macro:<10.4f} {r.f1_delta:<10.4f} "
                f"{str(r.is_meaningful):<11} {r.artifact_dir}"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
