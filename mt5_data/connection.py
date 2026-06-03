"""connection —— MetaTrader 5 终端的连接生命周期管理。

提供 ``initialize`` / ``shutdown`` 两个对外函数，分别负责建立与断开与 mt5 终端的连接。
``initialize`` 设计为幂等：重复调用不会重复连接；``shutdown`` 设计为可重复释放。

凭据管理（``load_credentials``）支持从环境变量或本地 JSON 文件读取登录信息，
真实凭据应放在 git 仓库外（推荐 ``%USERPROFILE%\\.mt5_credentials.json``），
仓库内仅保留 ``mt5_credentials.example.json`` 作为模板。
"""

import json
import os
from pathlib import Path
from typing import NamedTuple, Optional

import MetaTrader5 as mt5


# 本地跟踪当前是否已成功连接到 mt5 终端。
# mt5 包未公开查询接口，因此我们在模块层维护状态；只读访问。
_initialized: bool = False


# 凭据查找顺序（前者优先）：
#   1. 函数显式参数
#   2. 环境变量 MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
#   3. 用户主目录下的 .mt5_credentials.json
#   4. 当前工作目录下的 mt5_credentials.json（开发用，git 会忽略）
_DEFAULT_CREDENTIAL_PATHS = [
    Path.home() / ".mt5_credentials.json",          # %USERPROFILE%\.mt5_credentials.json
    Path.cwd() / "mt5_credentials.json",             # 仓库根（已被 .gitignore 屏蔽）
]


class Credentials(NamedTuple):
    """MT5 登录凭据三元组。

    Attributes:
        login: 交易账户登录号。
        password: 交易账户密码。
        server: 交易服务器名（如 ``"Exness-MT5Trial5"``）。
    """
    login: int
    password: str
    server: str


def load_credentials(
    config_path: Optional[os.PathLike] = None,
    require_all: bool = True,
) -> Optional[Credentials]:
    """从环境变量或本地 JSON 文件加载 MT5 登录凭据。

    查找顺序：
        1. 显式传入的 ``config_path``；
        2. 环境变量 ``MT5_LOGIN`` / ``MT5_PASSWORD`` / ``MT5_SERVER``；
        3. ``Path.home() / ".mt5_credentials.json"``；
        4. ``Path.cwd() / "mt5_credentials.json"``（开发用，已被 .gitignore 屏蔽）。

    Args:
        config_path: 显式指定的凭据 JSON 文件路径。为 ``None`` 时按上述顺序自动查找。
        require_all: 是否要求 ``login / password / server`` 三个字段全部存在。
            为 ``False`` 时仅返回已有字段（用于部分覆盖环境变量）。

    Returns:
        找到凭据时返回 :class:`Credentials`；所有来源都找不到或字段不完整时返回 ``None``。

    Note:
        真实凭据文件应放在 git 仓库外；仓库内仅有 ``mt5_credentials.example.json`` 模板。
    """
    # 情况 1：显式指定路径 —— 最高优先级。
    # 设计约定：显式路径找不到时直接返回 None，**不**回退到环境变量。
    # 这样调用方可以准确知道"指定文件里有没有凭据"，避免意外命中环境变量。
    if config_path is not None:
        return _read_credentials_file(Path(config_path))

    # 情况 2：从环境变量 + 默认候选文件中合并读取。
    env_login = os.environ.get("MT5_LOGIN")
    env_password = os.environ.get("MT5_PASSWORD")
    env_server = os.environ.get("MT5_SERVER")

    file_creds: Optional[Credentials] = None
    for candidate in _DEFAULT_CREDENTIAL_PATHS:
        if candidate.exists():
            file_creds = _read_credentials_file(candidate)
            if file_creds is not None:
                break

    # 环境变量优先于文件值（便于临时切换测试账户）。
    login = env_login if env_login is not None else (file_creds.login if file_creds else None)
    password = env_password if env_password is not None else (file_creds.password if file_creds else None)
    server = env_server if env_server is not None else (file_creds.server if file_creds else None)

    # 字段完整性校验。
    if login is None or password is None or server is None:
        if require_all:
            return None
        # 部分返回（仅用于诊断场景）。
        return None

    try:
        return Credentials(login=int(login), password=password, server=server)
    except (TypeError, ValueError):
        return None


def _read_credentials_file(path: Path) -> Optional[Credentials]:
    """读取单个 JSON 凭据文件，容错处理。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # 忽略以 ``_`` 开头的注释字段（用于 example 模板的说明）。
    fields = {k: v for k, v in data.items() if not k.startswith("_")}
    try:
        return Credentials(
            login=int(fields["login"]),
            password=str(fields["password"]),
            server=str(fields["server"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def initialize(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    path: Optional[str] = None,
) -> bool:
    """建立与 MetaTrader 5 终端的连接，可选地切换到指定账户。

    Args:
        login: 交易账户登录号。为 ``None`` 时复用终端当前已登录的账户。
        password: 交易账户密码。仅在 ``login`` 提供时使用。
        server: 交易服务器（如 ``"MetaQuotes-Demo"``）。仅在 ``login`` 提供时使用。
        path: 可选的 mt5 终端可执行文件绝对路径；为 ``None`` 时复用本机已安装的终端。

    Returns:
        ``True`` 表示连接就绪；``False`` 表示初始化或登录失败（错误已打印到 stdout）。

    Raises:
        不主动抛异常；所有失败路径以返回值 + 日志方式呈现，方便上层控制流。
    """
    global _initialized

    # 已连接时直接返回成功，避免重复 init 造成 mt5 内部状态错乱。
    if _initialized:
        return True

    # 初始化 mt5 终端。
    # 注意：mt5 5.0.x 在 Windows 上不接受显式 ``path=None``，会返回错误码 -2
    # "Invalid 'path' argument"。所以只在我们有具体路径时才传 path 关键字。
    # 不传 path 时 mt5 会自动连接本机默认安装的 MetaTrader 5。
    # 该 API 在终端未启动/路径错误时返回 False，mt5.last_error() 含具体原因。
    init_kwargs = {"path": path} if path is not None else {}
    if not mt5.initialize(**init_kwargs):
        print(f"[mt5_data] initialize() failed: {mt5.last_error()}")
        return False

    # 仅当业务层显式提供登录参数时才尝试切换账户；
    # 不传则保持终端当前已登录账户（与官方文档示例一致）。
    if login is not None or password is not None or server is not None:
        if not mt5.login(login=login, password=password, server=server):
            print(f"[mt5_data] login() failed: {mt5.last_error()}")
            mt5.shutdown()  # 登录失败时回滚初始化，避免半连接状态。
            return False

    _initialized = True
    return True


def shutdown() -> None:
    """断开与 MetaTrader 5 终端的连接。

    幂等：多次调用安全；未初始化时调用是空操作。
    不抛异常：即使 mt5.shutdown 内部报错，也吞掉以保证资源清理路径稳定。
    """
    global _initialized
    if not _initialized:
        return
    try:
        mt5.shutdown()
    except Exception as exc:  # noqa: BLE001 —— shutdown 必须稳，吞掉所有异常。
        print(f"[mt5_data] shutdown() raised: {exc!r}")
    finally:
        _initialized = False
