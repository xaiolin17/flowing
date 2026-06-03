"""timeframes —— 周期字符串别名与解析器。

mt5 终端的周期以 ``mt5.TIMEFRAME_*`` 整型常量表示。
本模块提供业务层字符串别名（如 ``"M1"`` / ``"H1"`` / ``"D1"``）到这些整型常量的双向映射，
并对常见书写差异（大小写、单位 ``min``、``1h`` vs ``H1``）做归一化，提升调用方容错度。
"""

import re
from typing import Union

import MetaTrader5 as mt5


# 业务层字符串别名 -> mt5 原生常量。
# 键必须保持大写，格式为「单位字母 + 数字」（M30、H1、D1）。
# 新增/废弃时同步检查 mt5_data/__init__.py 是否对外导出。
TIMEFRAME_ALIASES = {
    "M1":  mt5.TIMEFRAME_M1,   # 1 分钟
    "M2":  mt5.TIMEFRAME_M2,   # 2 分钟
    "M3":  mt5.TIMEFRAME_M3,   # 3 分钟
    "M4":  mt5.TIMEFRAME_M4,   # 4 分钟
    "M5":  mt5.TIMEFRAME_M5,   # 5 分钟
    "M6":  mt5.TIMEFRAME_M6,   # 6 分钟
    "M10": mt5.TIMEFRAME_M10,  # 10 分钟
    "M12": mt5.TIMEFRAME_M12,  # 12 分钟
    "M15": mt5.TIMEFRAME_M15,  # 15 分钟
    "M20": mt5.TIMEFRAME_M20,  # 20 分钟
    "M30": mt5.TIMEFRAME_M30,  # 30 分钟
    "H1":  mt5.TIMEFRAME_H1,   # 1 小时
    "H2":  mt5.TIMEFRAME_H2,   # 2 小时
    "H3":  mt5.TIMEFRAME_H3,   # 3 小时
    "H4":  mt5.TIMEFRAME_H4,   # 4 小时
    "H6":  mt5.TIMEFRAME_H6,   # 6 小时
    "H8":  mt5.TIMEFRAME_H8,   # 8 小时
    "H12": mt5.TIMEFRAME_H12,  # 12 小时
    "D1":  mt5.TIMEFRAME_D1,   # 1 天
    "W1":  mt5.TIMEFRAME_W1,   # 1 周
    "MN1": mt5.TIMEFRAME_MN1,  # 1 月
}


# 归一化用：把各种书写形式（"1h" / "H1" / "1hour" / "60min"）统一为 "H1" / "M60" 等。
_UNIT_NORMALIZE = [
    # (匹配模式, 输出单位字符) —— 按顺序匹配，命中后停止。
    (re.compile(r"^M(IN)?$"),     "M"),   # m / min -> M
    (re.compile(r"^H(OUR)?$"),    "H"),   # h / hour -> H
    (re.compile(r"^D(AY)?$"),     "D"),   # d / day -> D
    (re.compile(r"^W(EEK)?$"),    "W"),   # w / week -> W
    (re.compile(r"^MN$|^MONTH$"), "MN"),  # mn / month -> MN
]


def _normalize_timeframe_string(raw: str) -> str:
    """将任意书写形式归一化为「单位 + 数字」格式（如 "M30" / "H1"）。

    支持：
        * 前缀单位："M1" / "H1" / "D1" / "W1" / "MN1"
        * 后缀单位 + min/hour 关键字："30min" / "1h" / "60min" / "1hour"
    """
    s = raw.strip()
    if not s:
        return s

    upper = s.upper()

    # 形式 A：以前缀单位开头（"M30" / "H1" / "D1" / "W1" / "MN1"）。
    # 直接尝试在别名表中命中；命中失败时再走形式 B。
    if upper in TIMEFRAME_ALIASES:
        return upper

    # 形式 B：数字 + 单位关键字（"30min" / "1h" / "60min" / "1hour"）。
    m = re.match(r"^(\d+)\s*([A-Za-z]+)$", s)
    if m:
        number, unit_raw = m.group(1), m.group(2).upper()
        for pattern, unit_symbol in _UNIT_NORMALIZE:
            if pattern.match(unit_raw):
                return f"{unit_symbol}{number}"

    return upper  # 无法归一化，返回原大写形式交由查表失败时报错。


def parse_timeframe(value: Union[str, int]) -> int:
    """将业务层传入的周期值解析为 mt5 原生整型常量。

    Args:
        value: 字符串别名（如 ``"M1"``、``"30min"``、``"1h"``）或已知的
            ``mt5.TIMEFRAME_*`` 整型常量。

    Returns:
        对应的 ``mt5.TIMEFRAME_*`` 整型。

    Raises:
        ValueError: 当字符串无法匹配任何已知别名时抛出，错误信息列出全部可用别名，
            便于调用方定位拼写错误。
    """
    # 允许业务层直接传 mt5.TIMEFRAME_* 整型，避免常见路径上做无谓的查表往返。
    if isinstance(value, int):
        return value

    # 字符串归一化：兼容 "M30" / "30min" / "1h" / "1hour" 等多种书写形式。
    if isinstance(value, str):
        normalized = _normalize_timeframe_string(value)
        if normalized in TIMEFRAME_ALIASES:
            return TIMEFRAME_ALIASES[normalized]
        # 故意不静默回退到默认周期，避免把拼写错误隐藏成"拉到了错误粒度的数据"。
        raise ValueError(
            f"未识别的周期 '{value}'（已归一化为 '{normalized}'）。"
            f"可用别名: {sorted(TIMEFRAME_ALIASES.keys())}"
        )

    raise TypeError(f"period 必须是 str 或 int，收到 {type(value).__name__}")
