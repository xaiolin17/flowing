"""example —— mt5_data 包的最小可运行示例。

使用方式：
    1. 启动 MetaTrader 5 桌面终端；
    2. 配置登录凭据（三选一，**不要把真实账号写入仓库**）：
        a) 复制 ``mt5_credentials.example.json`` 为 ``%USERPROFILE%\\.mt5_credentials.json``
           并填入真实值（该路径在 git 仓库外）；
        b) 设置环境变量 ``MT5_LOGIN`` / ``MT5_PASSWORD`` / ``MT5_SERVER``；
        c) 先在 MetaTrader 5 桌面终端里手动登录并勾选“保存密码”，本示例不传账号参数即可。
    3. 在本目录执行 ``python example.py``。

期望输出：打印 EURUSD 在 2020-01-27 13:00 到 2020-01-28 13:00 之间的 1 分钟 K 线。
"""

from datetime import datetime

from mt5_data import (
    Credentials,
    fetch_rates,
    initialize,
    load_credentials,
    shutdown,
)


def main() -> None:
    # Step 1: 加载凭据。
    # load_credentials() 会按以下顺序自动查找：
    #   1) 环境变量 MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
    #   2) %USERPROFILE%\.mt5_credentials.json
    #   3) ./mt5_credentials.json（已被 .gitignore 屏蔽）
    # 找不到时返回 None；此时 initialize() 会复用终端已登录的账户。
    creds: Credentials | None = load_credentials()

    # Step 2: 显式连接（也可省略，fetch_rates 内部会自动重试一次）。
    # 传 creds 时会调用 mt5.login() 切换账户；不传则复用终端当前账户。
    if not initialize(
        login=creds.login if creds else None,
        password=creds.password if creds else None,
        server=creds.server if creds else None,
    ):
        # initialize() 失败时已经把 mt5.last_error() 打印到 stdout，这里直接退出。
        return

    try:
        # Step 3: 拉取 EURUSD 的 1 分钟 K 线。
        # 区间端点用本地时间传入；mt5 会按交易服务器时区解析。
        # 时长 = 24 小时 = 理论 1441 根（与用户文档示例数据规模一致）。
        df = fetch_rates(
            symbol="EURUSD",
            timeframe="M1",
            date_from=datetime(2020, 1, 27, 13, 0, 0),
            date_to=datetime(2020, 1, 28, 13, 0, 0),
        )
        print(f"EURUSD M1 区间内共 {len(df)} 根 K 线：")
        print(df.head())
    finally:
        # Step 4: 无论成功失败都断开连接，避免 mt5 进程残留。
        shutdown()


# 作为脚本直接运行时的入口；被 import 时不会触发。
if __name__ == "__main__":
    main()
