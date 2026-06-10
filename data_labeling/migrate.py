"""migrate.py —— data_labeling 数据库迁移工具（Section 6 核心）。

提供：
    * :func:`init_migrations_table` —— 创建 ``_migrations`` 表；
    * :func:`applied_migrations` —— 返回已应用迁移列表；
    * :func:`apply_pending` —— 应用所有未跑迁移；
    * :func:`run_migrations` —— 启动时自动跑（主入口）；

设计要点（与 spec.md "Section 6: 数据迁移工具" Requirement 严格对齐）：
    * 每个迁移一个 ``NNN_description.sql`` 文件，按编号升序应用；
    * ``_migrations`` 表记录已应用迁移（name + applied_at）；
    * 启动时自动跑未应用迁移；
    * **不引入新 pip**：仅 stdlib ``sqlite3`` + ``pathlib``。

公开 API：
    * :func:`run_migrations`
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import List, Tuple

# ---- 默认 migrations 目录 + DB 路径 ----
DEFAULT_MIGRATIONS_DIR: Path = Path(__file__).resolve().parent / "migrations"
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent / "data.db"


_MIGRATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);
"""


def init_migrations_table(db_path: Path) -> None:
    """创建 ``_migrations`` 表（幂等）。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_MIGRATIONS_TABLE_DDL)
        conn.commit()


def applied_migrations(db_path: Path) -> List[str]:
    """返回已应用迁移的名称列表（按名称升序）。"""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        try:
            cur = conn.execute("SELECT name FROM _migrations ORDER BY name ASC")
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            # _migrations 表还不存在
            return []


def _list_migration_files(migrations_dir: Path) -> List[Path]:
    """列出所有 ``NNN_*.sql`` 迁移文件，按编号升序。"""
    if not migrations_dir.exists():
        return []
    return sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))


def apply_pending(
    db_path: Path,
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR,
) -> List[Tuple[str, str]]:
    """应用所有未跑迁移。

    Returns:
        列表 of (name, status)；status ∈ {"applied", "skipped"}。
    """
    db_path = Path(db_path)
    migrations_dir = Path(migrations_dir)
    init_migrations_table(db_path)
    already = set(applied_migrations(db_path))
    out: List[Tuple[str, str]] = []
    for sql_path in _list_migration_files(migrations_dir):
        name = sql_path.name
        if name in already:
            out.append((name, "skipped"))
            continue
        sql = sql_path.read_text(encoding="utf-8")
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (name, _dt.datetime.now(_dt.timezone.utc).isoformat()),
            )
            conn.commit()
        out.append((name, "applied"))
    return out


def run_migrations(
    db_path: Path = DEFAULT_DB_PATH,
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR,
) -> List[Tuple[str, str]]:
    """启动时自动跑未应用迁移（主入口）。"""
    return apply_pending(db_path, migrations_dir)


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """``python -m data_labeling.migrate`` CLI 入口。"""
    import argparse
    import sys as _sys

    parser = argparse.ArgumentParser(
        description="data_labeling 数据库迁移（Section 6 G）",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help="数据库路径（默认 data_labeling/data.db）",
    )
    parser.add_argument(
        "--migrations-dir", type=Path, default=DEFAULT_MIGRATIONS_DIR,
        help="migrations 目录（默认 data_labeling/migrations）",
    )
    args = parser.parse_args()

    results = run_migrations(args.db, args.migrations_dir)
    for name, status in results:
        print(f"  {status:<8} {name}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
