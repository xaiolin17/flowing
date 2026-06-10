"""migrate_artifacts.py —— artifacts 命名迁移脚本（Section 2 BREAKING 配套）。

背景：spec v1.1（model-training-followup-improvements A 项）规定
artifacts 主文件统一 ``model.{ext}``（ext = pkl / txt / pt），不再保留
``lightgbm_booster.txt`` 这种类型名开头的命名。

用法：
    python -m model_training.migrate_artifacts \\
        --artifacts-dir model_training/artifacts \\
        [--dry-run]

行为：
    * 扫描 ``<artifacts-dir>/<run_dir>`` 下的 ``meta.json`` 读 model_type；
    * 若含旧命名（``lightgbm_booster.txt`` / ``transformer.pt.meta.json``），
      复制到新命名（``model.txt`` / ``transformer_meta.json``）；
    * 旧文件**保留**（不在原地删除，避免破坏引用；用户用 --delete-old
      显式删除）；
    * 生成 ``migration_report.json`` 记录每个目录的迁移动作。

不引入新 pip：仅 stdlib + pathlib + json。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# 旧 → 新 命名映射
LEGACY_TO_NEW: Dict[str, str] = {
    "lightgbm_booster.txt": "model.txt",
    "transformer.pt.meta.json": "transformer_meta.json",
}


def _read_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 读 {meta_path} 失败: {e}", file=sys.stderr)
        return None


def migrate_one(
    run_dir: Path,
    delete_old: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """迁移一个 run 目录。返回该目录的迁移动作记录。"""
    actions: List[str] = []
    meta = _read_meta(run_dir)
    if meta is None:
        return {"run_dir": str(run_dir), "skipped": True,
                "reason": "no meta.json"}

    for old_name, new_name in LEGACY_TO_NEW.items():
        old_path = run_dir / old_name
        new_path = run_dir / new_name
        if old_path.exists() and not new_path.exists():
            if dry_run:
                actions.append(f"DRY-RUN copy {old_name} -> {new_name}")
            else:
                shutil.copy2(old_path, new_path)
                actions.append(f"copied {old_name} -> {new_name}")
                if delete_old:
                    old_path.unlink()
                    actions.append(f"deleted {old_name}")
        elif old_path.exists() and new_path.exists():
            actions.append(f"skip (both exist) {old_name} / {new_name}")
        else:
            actions.append(f"skip (no legacy) {old_name}")

    return {
        "run_dir": str(run_dir),
        "model_type": meta.get("model_type"),
        "actions": actions,
    }


def migrate_all(
    artifacts_dir: Path,
    delete_old: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """迁移整个 artifacts 目录树。"""
    if not artifacts_dir.exists():
        return {"error": f"artifacts_dir 不存在: {artifacts_dir}"}
    report: List[Dict[str, Any]] = []
    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        report.append(migrate_one(run_dir, delete_old=delete_old,
                                  dry_run=dry_run))
    summary = {
        "artifacts_dir": str(artifacts_dir),
        "n_runs": len(report),
        "dry_run": dry_run,
        "delete_old": delete_old,
        "report": report,
    }
    # 落盘迁移报告
    out_path = artifacts_dir / "migration_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="artifacts 命名迁移（lightgbm_booster.txt → model.txt 等）"
    )
    parser.add_argument(
        "--artifacts-dir", type=Path,
        default=Path("model_training/artifacts"),
        help="artifacts 根目录（默认 model_training/artifacts）",
    )
    parser.add_argument(
        "--delete-old", action="store_true",
        help="迁移成功后删除旧文件（默认保留以避免破坏引用）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印不真改",
    )
    args = parser.parse_args()

    summary = migrate_all(
        artifacts_dir=args.artifacts_dir,
        delete_old=args.delete_old,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
