"""``python -m model_training`` 入口。

把 main() 转发到 :mod:`model_training.main`。
"""
from .main import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
