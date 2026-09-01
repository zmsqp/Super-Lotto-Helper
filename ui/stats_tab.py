# -*- coding: utf-8 -*-
"""开奖列表与前/后区号值表。"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from services.stats import (
    BACK_N,
    FRONT_N,
    P_BACK,
    P_FRONT,
    format_nums,
    next_draw_meta,
    number_table,
    odd_even,
    odd_even_counts,
    pattern_stats,
    slice_draws,
    zone3_label,
    zone_shape_counts,
)

WINDOW_OPTIONS = [
    ("近 30 期", 30),
    ("近 50 期", 50),
    ("近 100 期", 100),
    ("近 200 期", 200),
    ("全量", 0),
]

NUM_HEADERS = ["号码", "出现次数", "频率", "期望", "期望差", "当前遗漏", "近30", "近100", "EWMA"]


class StatsTab(QWidget):
    """开奖统计页。"""

    refresh_requested = pyqtSignal()

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._draws_asc: list[dict] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        top = QGroupBox("最新开奖")
        top_l = QVBoxLayout(top)
        head = QHBoxLayout()
        self.latest_label = QLabel("尚未加载")
        self.latest_label.setObjectName("latestBalls")
        self.source_label = QLabel("数据源：—")
        self.source_label.setObjectName("subtitleLabel")
        self.refresh_btn = QPushButton("刷新开奖")
        self.refresh_btn.setObjectName("primaryButton")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        head.addWidget(self.latest_label, 1)
        head.addWidget(self.source_label)
        head.addWidget(self.refresh_btn)
        top_l.addLayout(head)
        self.next_label = QLabel("")
        self.next_label.setObjectName("subtitleLabel")
        top_l.addWidget(self.next_label)
        root.addWidget(top)

        split = QSplitter(Qt.Horizontal)

        left = QGroupBox("历史开奖")
        left_l = QVBoxLayout(left)
        jump = QHBoxLayout()
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("输入期号定位，如 26099")
        jump_btn = QPushButton("定位")
        jump_btn.clicked.connect(self._jump_period)
        jump.addWidget(self.jump_edit)
        jump.addWidget(jump_btn)
        left_l.addLayout(jump)
        self.draw_table = QTableWidget(0, 4)
        self.draw_table.setHorizontalHeaderLabels(["期号", "日期", "前区", "后区"])
        self.draw_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.draw_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.draw_table.setAlternatingRowColors(True)
        self.draw_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.draw_table.verticalHeader().setVisible(False)
        left_l.addWidget(self.draw_table)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("统计窗口"))
        self.window_combo = QComboBox()
        for text, val in WINDOW_OPTIONS:
            self.window_combo.addItem(text, val)
        self.window_combo.setCurrentIndex(2)
        self.window_combo.currentIndexChanged.connect(self._rebuild_stats)
        ctrl.addWidget(self.window_combo)
        ctrl.addStretch(1)
        right_l.addLayout(ctrl)

        self.pattern_label = QLabel("")
        self.pattern_label.setWordWrap(True)
        self.pattern_label.setObjectName("subtitleLabel")
        right_l.addWidget(self.pattern_label)

        self.front_table = self._make_num_table()
        self.back_table = self._make_num_table()
        fg = QGroupBox("前区号值 01–35")
        fl = QVBoxLayout(fg)
        fl.addWidget(self.front_table)
        bg = QGroupBox("后区号值 01–12")
        bl = QVBoxLayout(bg)
        bl.addWidget(self.back_table)
        right_l.addWidget(fg, 3)
        right_l.addWidget(bg, 2)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        root.addWidget(split, 1)

    @staticmethod
    def _make_num_table() -> QTableWidget:
        t = QTableWidget(0, len(NUM_HEADERS))
        t.setHorizontalHeaderLabels(NUM_HEADERS)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.setSortingEnabled(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return t

    def set_refreshing(self, busy: bool) -> None:
        self.refresh_btn.setEnabled(not busy)
        self.refresh_btn.setText("刷新中..." if busy else "刷新开奖")

    def reload(self) -> None:
        self._draws_asc = self.store.list_draws_asc()
        latest = self.store.latest_draw()
        source = self.store.get_meta("last_source", "—")
        when = self.store.get_meta("last_refresh", "")
        src_name = {"sporttery": "体彩接口", "500.com": "500.com", "csv": "本地 CSV"}.get(source, source)
        self.source_label.setText(f"数据源：{src_name}" + (f"  ·  {when}" if when else ""))
        if latest:
            f = format_nums(latest["front"])
            b = format_nums(latest["back"])
            self.latest_label.setText(f"{latest['period']}  {latest['date']}    {f}  +  {b}")
            nxt = next_draw_meta(latest["date"], latest["period"])
            self.next_label.setText(
                f"共 {len(self._draws_asc)} 期 · 下一期预计 {nxt['period']}  {nxt['date']}（{nxt['weekday']}）"
            )
        else:
            self.latest_label.setText("暂无开奖数据")
            self.next_label.setText("")
        self._fill_history()
        self._rebuild_stats()

    def _fill_history(self) -> None:
        rows = list(reversed(self._draws_asc))
        self.draw_table.setSortingEnabled(False)
        self.draw_table.setRowCount(len(rows))
        for i, d in enumerate(rows):
            vals = [d["period"], d["date"], format_nums(d["front"]), format_nums(d["back"])]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.draw_table.setItem(i, c, item)
        self.draw_table.setSortingEnabled(False)

    def _jump_period(self) -> None:
        target = self.jump_edit.text().strip()
        if not target:
            return
        for r in range(self.draw_table.rowCount()):
            item = self.draw_table.item(r, 0)
            if item and item.text() == target:
                self.draw_table.selectRow(r)
                self.draw_table.scrollToItem(item)
                return

    def _rebuild_stats(self) -> None:
        window = int(self.window_combo.currentData() or 0)
        use = slice_draws(self._draws_asc, window)
        if not use:
            self.pattern_label.setText("无数据")
            self._fill_num_table(self.front_table, [])
            self._fill_num_table(self.back_table, [])
            return
        pstats = pattern_stats(use)
        oe = odd_even_counts(use, "front")
        zs = zone_shape_counts(use)
        oe_txt = "　".join(f"{k}={v}" for k, v in oe[:4])
        z_txt = "　".join(f"{k}={v}" for k, v in zs[:6])
        last = use[-1]
        self.pattern_label.setText(
            f"窗口 {len(use)} 期　前区和值均值 {pstats['front_sum_mean']:.1f} "
            f"（σ {pstats['front_sum_std']:.1f}）　"
            f"最近一期形态 {zone3_label(last['front'])} / 奇偶 {odd_even(last['front'])[0]}:{odd_even(last['front'])[1]}\n"
            f"奇偶频次：{oe_txt}\n"
            f"三区形态：{z_txt}"
        )
        self._fill_num_table(self.front_table, number_table(use, "front", FRONT_N, P_FRONT))
        self._fill_num_table(self.back_table, number_table(use, "back", BACK_N, P_BACK))

    def _fill_num_table(self, table: QTableWidget, rows: list[dict]) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        keys = ["num", "all", "rate", "exp", "diff", "omit", "r30", "r100", "ewma"]
        for i, row in enumerate(rows):
            for c, k in enumerate(keys):
                val = row[k]
                item = QTableWidgetItem()
                if k == "num":
                    item.setText(str(val))
                elif isinstance(val, float):
                    item.setData(Qt.DisplayRole, float(val))
                else:
                    item.setData(Qt.DisplayRole, int(val) if isinstance(val, int) else val)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, c, item)
        table.setSortingEnabled(True)
