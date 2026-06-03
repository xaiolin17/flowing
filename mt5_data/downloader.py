"""downloader —— 数据下载主入口。

本模块负责把 mt5 终端返回的原始 K 线数据转换为 ``pandas.DataFrame`` 供业务层使用。
约定：
    * 返回的 DataFrame 以 ``time``（UTC ``datetime``）为索引，按时间升序；
    * 任何下载失败均抛 ``RuntimeError`` 或 ``ValueError``，**不**静默返回空 DataFrame；
    * 业务层明确不需要的 ``spread`` / ``real_volume`` 两列会在下载时直接 drop；
    * 空值（NaN、空字符串、"NA"、"NULL" 等）会按相邻前后均值线性插值填充。
"""

from datetime import datetime, timezone
from typing import List, Optional, Union

import pandas as pd

import MetaTrader5 as mt5

from .connection import initialize
from .timeframes import parse_timeframe


# 在下载后统一 drop 的列：Exness Trial 账户下 spread 恒为 280、real_volume 恒为 0，
# 都没有业务信号意义；放在模块顶层常量便于后续扩展（比如新增 noise 列只需改这里）。
_DROPPED_COLUMNS = ("spread", "real_volume")


# 防御性归一为 NaN 的占位符。
# mt5 通常只返回数值，但部分 broker 历史数据可能混入字符串占位符。
# 当前实现选择「对 object 列做 pd.to_numeric(errors='coerce')」自动吸收任何非数字字符串，
# 比维护显式列表更通用；此处保留常量仅为文档参考，未来如需更精确控制可改用显式 replace。
_PLACEHOLDER_NULLS = ("", "NA", "N/A", "NULL", "null", "NaN", "nan", "None")


def _ensure_initialized() -> None:
    """确保 mt5 终端已连接；未连接时尝试初始化一次。

    供 ``fetch_rates`` 内部调用，使调用方不必显式 ``initialize`` 也能使用。
    """
    if not initialize():
        # initialize() 内部已打印 last_error()，这里再补一句便于排障。
        raise RuntimeError("mt5 终端未连接，且 initialize() 失败；请检查终端是否启动。")


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """对数值列做空值检测与插值；非数值列保持不变。

    策略（按用户要求）：
        1. 把空字符串 / "NA" / "NULL" / "NaN" / "None" 等占位符归一为 NaN；
        2. 用 ``interpolate(method='linear', limit_direction='both')`` 按相邻前后均值
           填充（pandas 线性插值 = (前值 + 后值) / 2，对单点 NaN 等价于「均值」）；
        3. 整列全空等 interpolate 无能为力的极端情况，用 ``bfill().ffill()`` 兜底；
        4. 仍残留 NaN 抛 RuntimeError，让上层明确知道数据源有问题。

    Args:
        df: 清洗前的 DataFrame。

    Returns:
        清洗后的 DataFrame（不修改原对象）。

    Raises:
        RuntimeError: 存在整列均为空的数值列时抛出。
    """
    # Step 1: 把 object 列尝试强转 numeric。
    # 这样 "NA" / "" / "NULL" / "NaN" / 任何异常字符串都会变成 NaN，
    # 同时让该列进入 numeric 列表从而被后续 interpolate 覆盖。
    # 比维护显式占位符列表更通用：broker 偶发的奇怪字符串也能吸收。
    # 仅对 object 列操作，不影响已经是 numeric 的列（避免误伤）。
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df = df.copy()
        for c in obj_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 2: 数值列插值。time 已在 fetch_rates 里转 datetime，不会出现在 numeric 列表中。
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df

    total_nans = int(df[numeric_cols].isna().sum().sum())
    if total_nans == 0:
        return df  # 最常见情况：数据干净，直接返回，避免无谓 IO。

    affected_cols: List[str] = [c for c in numeric_cols if df[c].isna().any()]
    # 打印一行 summary，让调用方能感知到清洗发生过（数据问题可追溯）。
    print(
        f"[mt5_data] fetch_rates 检测到 {total_nans} 个空值 "
        f"（涉及列: {affected_cols}），按相邻前后均值线性插值填充"
    )

    # Step 3: 线性插值。limit_direction='both' 让开头 / 结尾的 NaN 也能取到邻居。
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # Step 4: 兜底 —— 整列全空时 interpolate 无法填，用最近有效值 bfill/ffill。
    remaining = int(df[numeric_cols].isna().sum().sum())
    if remaining > 0:
        df[numeric_cols] = df[numeric_cols].bfill().ffill()
        final_nans = int(df[numeric_cols].isna().sum().sum())
        if final_nans > 0:
            # 整列都没有有效值，插值无解 —— 抛错让上层明确数据源问题，
            # 比"静默返回一个含 NaN 的 DataFrame"更安全。
            raise RuntimeError(
                f"fetch_rates: 列 {affected_cols} 整列均为空，无法插值填充；"
                "请检查 broker 数据源或调整 date_from / date_to 区间。"
            )

    return df


def fetch_rates(
    symbol: str,
    timeframe: Union[str, int],
    date_from: datetime,
    date_to: Optional[datetime] = None,
) -> pd.DataFrame:
    """下载指定品种在 [date_from, date_to] 区间内的 K 线数据。

    Args:
        symbol: 交易品种代码（如 ``"EURUSD"``、``"XAUUSD"``）。
        timeframe: 周期，支持字符串别名（``"M1"`` / ``"H1"`` / ``"D1"`` 等），
            也可直接传 ``mt5.TIMEFRAME_*`` 整型。
        date_from: 区间起始时间（含）。
        date_to: 区间结束时间（含）。为 ``None`` 时取到最新可用 K 线。

    Returns:
        以 ``time`` 为索引的 ``pandas.DataFrame``，列固定为
        ``[open, high, low, close, tick_volume]``。
        ``time`` 列为 ``datetime64[ns]``（UTC，无时区）。
        ``spread`` / ``real_volume`` 两列已在下载后 drop（Exness 恒为 280/0，无信号意义）。
        空值处理：NaN / 空字符串 / "NA" / "NULL" 等占位符会按相邻前后均值线性插值；
        整列全空时 bfill/ffill 兜底；仍残留则抛 RuntimeError。

    Raises:
        RuntimeError: 终端未连接 / 数据区间无效 / mt5 内部错误 / 整列全空无法插值。
        ValueError: 品种无法加入 MarketWatch（broker 未提供 / 名称拼写错误）。
    """
    # Step 1: 确保终端连接可用；连接失败时主动抛错。
    _ensure_initialized()

    # Step 2: 解析周期字符串为 mt5 原生常量。
    # 解析失败会在内部抛 ValueError，向上自然透传。
    tf = parse_timeframe(timeframe)

    # Step 3: 把品种加入 MarketWatch。
    # mt5 的 MarketWatch 是惰性的：未启用的品种直接 copy_rates_range 会失败。
    # symbol_select 第二个参数 True 表示"加入 MarketWatch"。
    if not mt5.symbol_select(symbol, True):
        raise ValueError(
            f"无法在 MarketWatch 中启用品种 '{symbol}'：{mt5.last_error()}"
        )

    # Step 4: 拉取区间 K 线。
    # 官方文档：copy_rates_range 在区间为空、未来时间或终端无数据时返回 None。
    # 注意：mt5 5.0.x 在 Windows 上不接受 ``date_to=None``，会返回错误码 -2
    # "Invalid arguments"。因此在 None 时用「当前 UTC 时间」作为右端点，
    # 与 mt5 数据时间戳（UTC epoch）保持同一时区基准，避免本地时区干扰。
    if date_to is None:
        date_to = datetime.now(timezone.utc).replace(tzinfo=None)
    raw = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    if raw is None or len(raw) == 0:
        # 主动抛错而非返回空 DF：上层做回测时，空 DF 容易掩盖"拉错了品种/区间"等问题。
        raise RuntimeError(
            f"copy_rates_range('{symbol}', {tf}, {date_from}, {date_to}) 返回空："
            f"{mt5.last_error()}"
        )

    # Step 5: numpy 结构化数组 -> DataFrame。
    # mt5 返回的 time 字段是秒级 epoch（UTC），需显式转换。
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Step 5.5: 业务侧明确不需要的列统一 drop。
    #   * spread：在 Exness Trial 账户恒为 280，无信号意义；
    #   * real_volume：Exness 不提供真实成交量，恒为 0。
    # 在此集中过滤，避免上层每次重复；如未来某些 broker 提供有意义的值，
    # 改为「仅在值恒为 0 时 drop」或在调用方按需 reindex 即可。
    df = df.drop(columns=list(_DROPPED_COLUMNS), errors="ignore")

    # Step 5.6: 空值检测 + 线性插值。
    # 业务侧需要连续 OHLC；mt5 偶发 NaN / broker 偶发空字符串占位符，
    # 在此统一归一后按相邻前后均值插值。详见 _fill_missing_values 文档。
    df = _fill_missing_values(df)

    # Step 6: 设 time 为索引并按时间升序。
    # 排序是防御性的：个别 broker 历史数据可能存在轻微乱序。
    df = df.set_index("time").sort_index()

    return df
