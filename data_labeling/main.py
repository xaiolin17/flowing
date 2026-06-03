"""data_labeling 包的 GUI 入口。

运行方式（任选其一）::

    python -m data_labeling
    python data_labeling/main.py

前置依赖（必须先手动安装）::

    pip install PyQt5 pyqtgraph

    * PyQt5        —— Qt for Python（窗口/事件/控件）
    * pyqtgraph    —— 高性能科学绘图（K 线、坐标轴）

不通过 ``pip install -r requirements.txt`` 自动拉取，遵循本项目"不在仓库内
执行 pip"约束。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    # 解析可选参数：--db / -d 指定 SQLite 路径
    db_path: Path | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("--db", "-d") and i + 1 < len(argv):
            db_path = Path(argv[i + 1])
            i += 2
            continue
        if arg in ("-h", "--help"):
            print(__doc__)
            return 0
        print(f"未知参数: {arg}\n{__doc__}")
            return 2
        i += 1

    # 延迟 import GUI 依赖；缺失时给出可读报错
    try:
        from .app import run
    except ImportError as e:
        # 显式提示用户先装 PyQt5 + pyqtgraph
        print(
            "[data_labeling] 启动失败：缺少 GUI 依赖。\n"
            "请先执行: pip install PyQt5 pyqtgraph\n"
            f"原始错误: {e}",
            file=sys.stderr,
        )
        return 1
    run(db_path=db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
