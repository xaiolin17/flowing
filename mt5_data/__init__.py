"""mt5_data —— MetaTrader5 数据下载封装包。

提供以 ``pandas.DataFrame`` 形式返回的 K 线/柱形图下载能力，
仅做内存中数据装配，不负责文件或数据库落盘。
"""

# 连接生命周期管理：initialize / shutdown
# 凭据加载：load_credentials / Credentials（支持从环境变量或仓库外的 JSON 文件读取）
from .connection import Credentials, initialize, load_credentials, shutdown
# 数据下载主入口：fetch_rates
from .downloader import fetch_rates
# 周期字符串解析与别名表
from .timeframes import TIMEFRAME_ALIASES, parse_timeframe

# 对外稳定 API。新增功能时需同步在此登记；废弃符号请保留一段时间再移除。
__all__ = [
    "fetch_rates",
    "initialize",
    "shutdown",
    "load_credentials",
    "Credentials",
    "parse_timeframe",
    "TIMEFRAME_ALIASES",
]
