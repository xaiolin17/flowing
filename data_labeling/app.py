"""app —— PyQt5 + pyqtgraph 可视化打标签主窗口。

依赖（用户需自行 pip install，本模块不强制 import）:
    pip install PyQt5 pyqtgraph

本模块是 GUI 入口，结构：
    * :class:`MainWindow` —— QMainWindow 顶层容器；
    * :class:`CandlestickWidget` —— 自实现 K 线图 + 鼠标点击事件；
    * :class:`LabelButtonBar` —— "买/卖/清除" 按钮条；
    * :func:`run` —— 启动 QApplication 的工厂。

注意：
    * 所有 PyQt5 / pyqtgraph import 都放进 ``MainWindow`` 内部或 ``run`` 内部，
      避免 DAO 单测被 GUI 库污染；
    * K 线点击用 ``scene().items()`` 取最近点的策略，确保小 K 线也容易点中；
    * 标签写库走 :class:`data_labeling.db.LabelDAO`，无延迟批量写。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

# PyQt5 / pyqtgraph 延迟 import —— 见模块顶部 docstring。
# 测试或 DAO 路径不应强制触发它们。
PyQt5QtCore = PyQt5QtGui = PyQt5QtWidgets = None  # type: ignore
pg = None  # type: ignore


def _ensure_gui_deps():
    """延迟加载 GUI 依赖；缺失时给清晰报错。"""
    global PyQt5QtCore, PyQt5QtGui, PyQt5QtWidgets, pg
    if PyQt5QtWidgets is not None:
        return
    try:
        import pyqtgraph as _pg  # type: ignore
        from PyQt5 import QtCore as _QtCore  # type: ignore
        from PyQt5 import QtGui as _QtGui  # type: ignore
        from PyQt5 import QtWidgets as _QtWidgets  # type: ignore
    except ImportError as e:
        raise ImportError(
            "启动 GUI 需要 PyQt5 + pyqtgraph；"
            "请先在虚拟环境中执行: pip install PyQt5 pyqtgraph"
        ) from e
    PyQt5QtCore = _QtCore
    PyQt5QtGui = _QtGui
    PyQt5QtWidgets = _QtWidgets
    pg = _pg


# 标签颜色（绿涨/红跌/灰未标）
_BULL_COLOR = "#26a69a"   # 涨：青绿
_BEAR_COLOR = "#ef5350"   # 跌：红色
_HIGHLIGHT_COLOR = "#ffc107"  # 选中高亮：琥珀
_BUY_MARKER_COLOR = "#1e88e5"   # 买 ▲：蓝
_SELL_MARKER_COLOR = "#fb8c00"  # 卖 ▼：橙


class _BarChartPlaceholder:
    """占位：避免在 _ensure_gui_deps 调用前被分析工具误读。"""


# 顶层立刻触发依赖检查：类定义要 PyQt5QtCore / pg 可见。
# app.py 本身是 GUI 入口，import 时缺 PyQt5/pyqtgraph 应直接抛错。
# DAO 单测走 db.py / persistence.py，不 import 本模块，无副作用。
_ensure_gui_deps()


class CandlestickWidget(PyQt5QtCore.QObject):
    """自实现 K 线图组件。

    公开方法：
        * :meth:`set_data` 装载 candles + labels；
        * :meth:`clear` 清空图；
        * 信号 ``candle_clicked`` 在鼠标点击 K 线时发出（参数：时间戳）。

    内部用 pyqtgraph 的 :class:`BarGraphItem`（画 open-close 矩形）
    + :class:`PlotCurveItem`（画 high-low 影线）组合实现 K 线。

    继承 :class:`QObject` 是为了让 ``pyqtSignal`` 可以 ``.connect``；
    ``pyqtSignal`` 在 QObject 子类之外是 metaclass，调用 connect 会抛
    ``AttributeError``（pyqtgraph 0.14.x 严格了）。
    """

    # 信号：candle_clicked(datetime)
    candle_clicked = PyQt5QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        _ensure_gui_deps()
        self._pg = pg
        # 把 plot 设为白底黑字，符合交易员看盘习惯
        self.plot = pg.PlotWidget(parent=parent)
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # X 轴用整数索引（numpy.arange），避免时间轴 hover 慢
        self.plot.getPlotItem().getAxis("bottom").setStyle(
            tickFont=PyQt5QtGui.QFont("Arial", 8)
        )
        axis = pg.DateAxisItem(orientation="bottom")
        self.plot.setAxisItems({"bottom": axis})

        # 当前数据
        self._candles: list = []  # list of (time, o, h, l, c, v)
        self._labels: dict = {}   # time -> label value

        # Graphics items（用名字引用，便于重绘时 remove）
        self._bar_item: Optional[object] = None
        self._bear_bar_item: Optional[object] = None
        self._wick_item: Optional[object] = None
        self._highlight_item: Optional[object] = None
        self._buy_markers: Optional[object] = None
        self._sell_markers: Optional[object] = None

        # 鼠标点击：pyqtgraph 用信号更稳
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # ---- 公共 API ----

    def clear(self):
        self._candles = []
        self._labels = {}
        for attr in ("_bar_item", "_bear_bar_item", "_wick_item",
                     "_highlight_item", "_buy_markers", "_sell_markers"):
            item = getattr(self, attr, None)
            if item is not None:
                self.plot.removeItem(item)
                setattr(self, attr, None)

    def set_data(self, candles, labels):
        """装载 candles + labels（labels 是 dict[time, value]）。"""
        self.clear()
        self._candles = list(candles)
        self._labels = dict(labels)
        if not self._candles:
            return
        self._draw_candles()
        self._draw_label_markers()

    # ---- 绘制 ----

    def _draw_candles(self):
        # 用整数索引做 X 坐标，再用自定义 tickStrings 把整数转回 datetime 字符串
        n = len(self._candles)
        x = list(range(n))
        opens = [c[1] for c in self._candles]
        highs = [c[2] for c in self._candles]
        lows = [c[3] for c in self._candles]
        closes = [c[4] for c in self._candles]

        # BarGraphItem：width=0.6 表示实体宽度
        # 涨/跌分别画：用两段拼接
        # pyqtgraph 0.14.x BarGraphItem 期待 x + y0 + height 三个独立数组，
        # 不再支持旧版 (bottom, height) tuple 形式。
        bull_x, bull_y0, bull_h = [], [], []
        bear_x, bear_y0, bear_h = [], [], []
        wick_x, wick_y = [], []
        for i, (o, h, l, cl) in enumerate(zip(opens, highs, lows, closes)):
            bottom = min(o, cl)
            height = abs(cl - o) if o != cl else 1e-9
            if cl >= o:
                bull_x.append(i)
                bull_y0.append(bottom)
                bull_h.append(height)
            else:
                bear_x.append(i)
                bear_y0.append(bottom)
                bear_h.append(height)
            # 影线（high-low）
            wick_x.append([i, i])
            wick_y.append([l, h])

        if bull_x:
            self._bar_item = self._pg.BarGraphItem(
                x=bull_x, y0=bull_y0, height=bull_h, width=0.6,
                brush=self._pg.mkBrush(_BULL_COLOR),
                pen=self._pg.mkPen(_BULL_COLOR),
            )
            self.plot.addItem(self._bar_item)
        if bear_x:
            # 第二个 BarGraphItem（叠加绘制）
            bear_item = self._pg.BarGraphItem(
                x=bear_x, y0=bear_y0, height=bear_h, width=0.6,
                brush=self._pg.mkBrush(_BEAR_COLOR),
                pen=self._pg.mkPen(_BEAR_COLOR),
            )
            self.plot.addItem(bear_item)
            # 保留引用以便 clear 时清理
            self._bear_bar_item = bear_item

        # 影线（一根 line 包含 N 段）
        import numpy as np  # 局部 import，避免 GUI 启动时无谓依赖
        wick_x_arr = np.array(wick_x).flatten()
        wick_y_arr = np.array(wick_y).flatten()
        self._wick_item = self._pg.PlotCurveItem(
            x=wick_x_arr, y=wick_y_arr,
            pen=self._pg.mkPen("#666666", width=1),
            connect="pairs",
        )
        self.plot.addItem(self._wick_item)

        # 把 X 轴 ticks 替换为时间字符串（每 N 根一个）
        self._install_time_ticks(n)

    def _install_time_ticks(self, n: int):
        # 每 max(1, n // 8) 根一个 tick
        stride = max(1, n // 8)
        ticks = [
            (i, self._candles[i][0].strftime("%m-%d %H:%M"))
            for i in range(0, n, stride)
        ]
        axis = self.plot.getPlotItem().getAxis("bottom")
        # pg.DateAxisItem 支持 dict ticks
        # 不同 pg 版本 API 略有差异；这里走 setTicks 通用接口
        try:
            axis.setTicks([ticks])
        except Exception:
            pass  # 不阻塞主流程

    def _draw_label_markers(self):
        if not self._labels:
            return
        # 用 ScatterPlotItem 画三角
        n = len(self._candles)
        time_to_idx = {c[0]: i for i, c in enumerate(self._candles)}

        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        # 取每根 K 线的最高/最低价做 marker 位置
        for i, c in enumerate(self._candles):
            t = c[0]
            if t in self._labels:
                v = self._labels[t]
                if v == 1:
                    buy_x.append(i)
                    buy_y.append(c[2])  # high
                elif v == -1:
                    sell_x.append(i)
                    sell_y.append(c[3])  # low

        if buy_x:
            self._buy_markers = self._pg.ScatterPlotItem(
                x=buy_x, y=buy_y,
                symbol="t1", size=14,
                brush=self._pg.mkBrush(_BUY_MARKER_COLOR),
                pen=self._pg.mkPen("#000000", width=1),
            )
            self.plot.addItem(self._buy_markers)
        if sell_x:
            self._sell_markers = self._pg.ScatterPlotItem(
                x=sell_x, y=sell_y,
                symbol="t", size=14,
                brush=self._pg.mkBrush(_SELL_MARKER_COLOR),
                pen=self._pg.mkPen("#000000", width=1),
            )
            self.plot.addItem(self._sell_markers)

    # ---- 鼠标事件 ----

    def _on_mouse_clicked(self, ev):
        # 只处理左键
        if not ev.button() == self._pg.QtCore.Qt.LeftButton:
            return
        if not self._candles:
            return
        # 把鼠标位置映射到数据坐标
        vb = self.plot.getPlotItem().getViewBox()
        scene_pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        x_click, y_click = mouse_point.x(), mouse_point.y()

        # 找 X 坐标最接近的 K 线
        n = len(self._candles)
        idx = int(round(x_click))
        if idx < 0:
            idx = 0
        elif idx >= n:
            idx = n - 1
        # 同时也支持「鼠标 X 不准、点到 K 线实体附近」的情况：
        # 如果 click 落在 [low, high] 范围内，认为点中。
        candle = self._candles[idx]
        c_low, c_high = candle[3], candle[2]
        if not (c_low - (c_high - c_low) * 0.5 <= y_click <= c_high + (c_high - c_low) * 0.5):
            # 简单兜底：只要找到 X 最近的，就接受点击
            pass
        # 高亮
        self._highlight(idx)
        # 发信号（带 datetime）
        self.candle_clicked.emit(candle[0])

    def _highlight(self, idx: int):
        if self._highlight_item is not None:
            self.plot.removeItem(self._highlight_item)
        candle = self._candles[idx]
        c_low, c_high = candle[3], candle[2]
        # 一根竖向 Line
        self._highlight_item = self._pg.InfiniteLine(
            pos=idx,
            angle=90,
            pen=self._pg.mkPen(_HIGHLIGHT_COLOR, width=2, style=self._pg.QtCore.Qt.DashLine),
        )
        self.plot.addItem(self._highlight_item)


class LabelButtonBar:
    """「买 / 卖 / 清除」按钮条（水平布局）。"""

    clicked_buy = None
    clicked_sell = None
    clicked_clear = None

    def __init__(self, parent=None):
        _ensure_gui_deps()
        QtCore = PyQt5QtCore
        self.widget = PyQt5QtWidgets.QWidget(parent)
        layout = PyQt5QtWidgets.QHBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_buy = PyQt5QtWidgets.QPushButton("买 (B)")
        self.btn_sell = PyQt5QtWidgets.QPushButton("卖 (S)")
        self.btn_clear = PyQt5QtWidgets.QPushButton("清除 (C)")
        self.btn_buy.setStyleSheet("background:#1e88e5;color:white;font-weight:bold;")
        self.btn_sell.setStyleSheet("background:#fb8c00;color:white;font-weight:bold;")
        self.btn_clear.setStyleSheet("background:#9e9e9e;color:white;")

        layout.addWidget(self.btn_buy)
        layout.addWidget(self.btn_sell)
        layout.addWidget(self.btn_clear)
        layout.addStretch(1)

        self.clicked_buy = self.btn_buy.clicked
        self.clicked_sell = self.btn_sell.clicked
        self.clicked_clear = self.btn_clear.clicked


class MainWindow:
    """主窗口工厂（不直接继承 QMainWindow，简化类型注解）。"""

    def __init__(self, db_path: Optional[Path] = None):
        _ensure_gui_deps()
        # 局部 import 的别名，方便下面用
        QtCore = PyQt5QtCore
        QtGui = PyQt5QtGui
        QtWidgets = PyQt5QtWidgets

        # 确保 DB 已建
        from .db import (DEFAULT_DB_PATH, CandleDAO, DatasetDAO, LabelDAO,
                         init_db)
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        init_db(db_path)
        self._db_path = db_path
        self._dataset_dao = DatasetDAO(db_path)
        self._candle_dao = CandleDAO(db_path)
        self._label_dao = LabelDAO(db_path)

        # QMainWindow
        self.qmain = QtWidgets.QMainWindow()
        self.qmain.setWindowTitle("Flowing —— K 线打标签")
        self.qmain.resize(1280, 720)

        central = QtWidgets.QWidget()
        self.qmain.setCentralWidget(central)
        hlayout = QtWidgets.QHBoxLayout(central)

        # 左侧：数据集列表
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("数据集"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_dataset_selected)
        left_layout.addWidget(self.list_widget)

        # 列表操作按钮
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("刷新")
        self.btn_delete = QtWidgets.QPushButton("删除")
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_delete)
        left_layout.addLayout(btn_row)

        # 右侧：K 线 + 按钮条
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        # 工具栏（导入）
        toolbar_layout = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("导入新数据")
        self.lbl_status = QtWidgets.QLabel("就绪")
        toolbar_layout.addWidget(self.btn_import)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.lbl_status)
        right_layout.addLayout(toolbar_layout)

        # K 线图
        self.candle_widget = CandlestickWidget(parent=right)
        right_layout.addWidget(self.candle_widget.plot, stretch=1)

        # 标签按钮条
        self.label_bar = LabelButtonBar(parent=right)
        right_layout.addWidget(self.label_bar.widget)

        hlayout.addWidget(left, stretch=1)
        hlayout.addWidget(right, stretch=4)

        # 信号绑定
        self.btn_refresh.clicked.connect(self._refresh_dataset_list)
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.candle_widget.candle_clicked.connect(self._on_candle_clicked)
        self.label_bar.clicked_buy.connect(lambda: self._apply_label(1))
        self.label_bar.clicked_sell.connect(lambda: self._apply_label(-1))
        self.label_bar.clicked_clear.connect(lambda: self._apply_label(0))

        # 初始状态：未选 dataset
        self._current_dataset_id: Optional[int] = None
        self._current_clicked_time: Optional[datetime] = None

        self._refresh_dataset_list()

    # ---- 数据集列表 ----

    def _refresh_dataset_list(self):
        self.list_widget.clear()
        for ds in self._dataset_dao.list_all():
            item = PyQt5QtWidgets.QListWidgetItem(
                f"{ds.name}  ({ds.row_count} 根)  {ds.created_at:%Y-%m-%d %H:%M}"
            )
            item.setData(PyQt5QtCore.Qt.UserRole, ds.id)
            self.list_widget.addItem(item)
        self.lbl_status.setText(f"已加载 {self.list_widget.count()} 个数据集")

    def _on_dataset_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        ds_id = items[0].data(PyQt5QtCore.Qt.UserRole)
        self._load_dataset(ds_id)

    def _load_dataset(self, ds_id: int):
        candles = self._candle_dao.list_by_dataset(ds_id)
        labels_rows = self._label_dao.list_by_dataset(ds_id)
        # 转为 widget 需要的格式
        candle_tuples = [
            (c.time, c.open, c.high, c.low, c.close, c.tick_volume)
            for c in candles
        ]
        labels_dict = {row.time: row.value for row in labels_rows}
        self.candle_widget.set_data(candle_tuples, labels_dict)
        self._current_dataset_id = ds_id
        self.lbl_status.setText(
            f"数据集 {ds_id}: {len(candle_tuples)} 根 K 线，{len(labels_dict)} 条标签"
        )

    # ---- 删除 ----

    def _on_delete_clicked(self):
        items = self.list_widget.selectedItems()
        if not items:
            self.lbl_status.setText("请先在左侧选中要删除的数据集")
            return
        ds_id = items[0].data(PyQt5QtCore.Qt.UserRole)
        ret = PyQt5QtWidgets.QMessageBox.question(
            self.qmain,
            "确认删除",
            f"确认删除数据集 id={ds_id}？\n（关联 candles / labels 将一起清除）",
        )
        if ret != PyQt5QtWidgets.QMessageBox.Yes:
            return
        self._dataset_dao.delete(ds_id)
        if self._current_dataset_id == ds_id:
            self.candle_widget.clear()
            self._current_dataset_id = None
        self._refresh_dataset_list()
        self.lbl_status.setText(f"已删除数据集 {ds_id}")

    # ---- 导入 ----

    def _on_import_clicked(self):
        QtWidgets = PyQt5QtWidgets
        symbol, ok = QtWidgets.QInputDialog.getText(
            self.qmain, "导入新数据", "品种代码（如 XAUUSDm）:"
        )
        if not ok or not symbol.strip():
            return
        timeframe, ok = QtWidgets.QInputDialog.getText(
            self.qmain, "导入新数据", "周期（M1 / H1 / D1 等）:"
        )
        if not ok or not timeframe.strip():
            return
        date_from_str, ok = QtWidgets.QInputDialog.getText(
            self.qmain, "导入新数据", "起始日期 (YYYY-MM-DD):"
        )
        if not ok or not date_from_str.strip():
            return
        date_to_str, ok = QtWidgets.QInputDialog.getText(
            self.qmain, "导入新数据", "结束日期 (YYYY-MM-DD，可空):"
        )
        if not ok:
            return
        try:
            date_from = datetime.strptime(date_from_str.strip(), "%Y-%m-%d")
            date_to = (
                datetime.strptime(date_to_str.strip(), "%Y-%m-%d")
                if date_to_str.strip() else None
            )
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self.qmain, "日期格式错误", str(e))
            return

        try:
            from .persistence import import_from_mt5
            ds = import_from_mt5(
                symbol=symbol.strip(),
                timeframe=timeframe.strip(),
                date_from=date_from,
                date_to=date_to,
                db_path=self._db_path,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.qmain, "导入失败", str(e))
            return

        self._refresh_dataset_list()
        self.lbl_status.setText(f"已导入 {ds.name}（{ds.row_count} 根）")

    # ---- K 线点击 + 打标签 ----

    def _on_candle_clicked(self, ts: datetime):
        self._current_clicked_time = ts
        self.lbl_status.setText(f"已选中 K 线 @ {ts} —— 请选择买 / 卖 / 清除")

    def _apply_label(self, value: int):
        if self._current_dataset_id is None or self._current_clicked_time is None:
            self.lbl_status.setText("请先选中数据集并点击 K 线")
            return
        if value == 0:
            self._label_dao.delete(self._current_dataset_id, self._current_clicked_time)
        else:
            self._label_dao.upsert(
                dataset_id=self._current_dataset_id,
                time=self._current_clicked_time,
                value=value,
            )
        # 重新加载当前 dataset（重绘三角）
        self._load_dataset(self._current_dataset_id)


def run(db_path: Optional[Path] = None):
    """启动 GUI 入口。"""
    _ensure_gui_deps()
    QtWidgets = PyQt5QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(db_path=db_path)
    win.qmain.show()
    app.exec_()
    return win
