# -*- coding: utf-8 -*-
"""本期推测：标注 + 232 / 奇偶 / 冷热遗漏生成推荐。"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.algorithms import SINGLE_SHAPES, generate_recommendations
from services.stats import format_nums, next_draw_meta, pattern_stats, slice_draws, sum_band, zone_shape_counts
from ui.number_board import NumberBoard

WINDOW_OPTIONS = [
    ("近 30 期", 30),
    ("近 50 期", 50),
    ("近 100 期", 100),
    ("近 200 期", 200),
    ("全量", 0),
]
MODE_OPTIONS = [
    ("综合 blend", "blend"),
    ("热号", "hot"),
    ("冷号", "cold"),
    ("遗漏", "omit"),
    ("EWMA20", "ewma20"),
    ("EWMA50", "ewma50"),
]
FRONT_OE = [("3:2", 3), ("2:3", 2), ("不限", -1)]
BACK_OE = [("1:1", 1), ("不限", -1)]


class PredictTab(QWidget):
    """本期推测页。"""

    session_saved = pyqtSignal()

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.target_period = ""
        self.target_meta: dict = {}
        self.last_result: dict = {}
        self._silent = False
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        left = QVBoxLayout()
        head = QHBoxLayout()
        self.period_label = QLabel("目标期号：—")
        self.period_label.setObjectName("titleLabel")
        head.addWidget(self.period_label, 1)
        self.save_btn = QPushButton("保存本期推测")
        self.save_btn.setObjectName("primaryButton")
        self.clear_btn = QPushButton("清空标注")
        self.restore_btn = QPushButton("从上次同期标注恢复")
        self.save_btn.clicked.connect(self._save)
        self.clear_btn.clicked.connect(self._clear_marks)
        self.restore_btn.clicked.connect(self._restore)
        head.addWidget(self.save_btn)
        head.addWidget(self.clear_btn)
        head.addWidget(self.restore_btn)
        left.addLayout(head)

        board_box = QGroupBox("号码标注")
        bl = QVBoxLayout(board_box)
        self.board = NumberBoard()
        self.board.marks_changed.connect(self._persist_marks)
        bl.addWidget(self.board)
        left.addWidget(board_box, 1)

        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("备注"))
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("可选：思路、胆码说明等")
        note_row.addWidget(self.note_edit, 1)
        left.addLayout(note_row)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        algo = QGroupBox("缩号算法（基于当前统计窗口，尊重标注）")
        al = QVBoxLayout(algo)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("窗口"))
        self.window_combo = QComboBox()
        for text, val in WINDOW_OPTIONS:
            self.window_combo.addItem(text, val)
        self.window_combo.setCurrentIndex(2)
        row1.addWidget(self.window_combo)
        row1.addWidget(QLabel("打分"))
        self.mode_combo = QComboBox()
        for text, val in MODE_OPTIONS:
            self.mode_combo.addItem(text, val)
        row1.addWidget(self.mode_combo)
        row1.addStretch(1)
        al.addLayout(row1)

        self.zone_rank_label = QLabel("三区形态排行：—")
        self.zone_rank_label.setWordWrap(True)
        self.zone_rank_label.setObjectName("subtitleLabel")
        al.addWidget(self.zone_rank_label)
        self.sum_label = QLabel("前区和值：—")
        self.sum_label.setWordWrap(True)
        self.sum_label.setObjectName("subtitleLabel")
        al.addWidget(self.sum_label)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("单式形态"))
        self.shape_combo = QComboBox()
        for key in SINGLE_SHAPES:
            a, b, c = SINGLE_SHAPES[key]
            self.shape_combo.addItem(f"{key}（{a}+{b}+{c}）", key)
        row2.addWidget(self.shape_combo)
        self.duplex_check = QCheckBox("232 复式池（一区2+二区3+三区2，共7码）")
        self.duplex_check.setChecked(True)
        row2.addWidget(self.duplex_check)
        self.sum_check = QCheckBox("约束和值到均值±σ")
        self.sum_check.setChecked(True)
        row2.addWidget(self.sum_check)
        row2.addStretch(1)
        al.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("前区奇偶"))
        self.front_oe = QComboBox()
        for text, val in FRONT_OE:
            self.front_oe.addItem(text, val)
        row3.addWidget(self.front_oe)
        row3.addWidget(QLabel("后区奇偶"))
        self.back_oe = QComboBox()
        for text, val in BACK_OE:
            self.back_oe.addItem(text, val)
        row3.addWidget(self.back_oe)
        row3.addWidget(QLabel("注数"))
        self.ticket_spin = QSpinBox()
        self.ticket_spin.setRange(3, 5)
        self.ticket_spin.setValue(5)
        row3.addWidget(self.ticket_spin)
        row3.addStretch(1)
        al.addLayout(row3)

        self.gen_btn = QPushButton("生成推荐")
        self.gen_btn.setObjectName("primaryButton")
        self.gen_btn.setMinimumHeight(36)
        self.gen_btn.clicked.connect(self._generate)
        al.addWidget(self.gen_btn)
        hint = QLabel("历史拟合不能改变一等奖理论概率。排除号不进推荐，关注号加权靠前。")
        hint.setObjectName("subtitleLabel")
        hint.setWordWrap(True)
        al.addWidget(hint)
        right.addWidget(algo)

        result = QGroupBox("推荐结果")
        rl = QVBoxLayout(result)
        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)
        rl.addWidget(self.result_edit)
        right.addWidget(result, 1)
        root.addLayout(right, 2)

    def reload(self) -> None:
        latest = self.store.latest_draw()
        if latest:
            meta = next_draw_meta(latest["date"], latest["period"])
            self.set_target_period(meta["period"], meta)
        else:
            self.period_label.setText("目标期号：—（请先导入开奖）")
        self._refresh_zone_rank()

    def set_target_period(self, period: str, meta: dict | None = None) -> None:
        self.target_period = period
        self.target_meta = meta or {}
        extra = ""
        if meta:
            extra = f"　预计 {meta.get('date', '')}（{meta.get('weekday', '')}）"
        self.period_label.setText(f"目标期号：{period}{extra}")
        self._silent = True
        self.board.set_marks(self.store.load_marks(period))
        self._silent = False
        self._refresh_zone_rank()

    def _refresh_zone_rank(self) -> None:
        draws = self.store.list_draws_asc()
        window = int(self.window_combo.currentData() or 0)
        use = slice_draws(draws, window)
        if not use:
            self.zone_rank_label.setText("三区形态排行：无数据")
            self.sum_label.setText("前区和值：无数据")
            return
        ranks = zone_shape_counts(use)
        top = "　".join(f"{k} {v}次" for k, v in ranks[:8])
        self.zone_rank_label.setText(f"近窗三区形态排行（单式 5 码）：{top}")
        pstats = pattern_stats(use)
        lo, hi, mean = sum_band(pstats, 1.0)
        self.sum_label.setText(
            f"前区和值：近窗均值 {mean:.1f}（σ {pstats['front_sum_std']:.1f}），"
            f"常用区间 {lo}–{hi}；后区和均值 {pstats['back_sum_mean']:.1f}"
        )

    def _persist_marks(self) -> None:
        if self._silent or not self.target_period:
            return
        self.store.save_marks(self.target_period, self.board.get_marks())

    def _clear_marks(self) -> None:
        self.board.clear()
        if self.target_period:
            self.store.clear_marks(self.target_period)

    def _restore(self) -> None:
        src_period = self.store.latest_marks_period()
        session = self.store.latest_session()
        marks = None
        if src_period:
            marks = self.store.load_marks(src_period)
        if (not marks or self._marks_empty(marks)) and session:
            marks = session.get("marks")
            src_period = session.get("target_period")
        if not marks or self._marks_empty(marks):
            QMessageBox.information(self, "提示", "没有可恢复的标注。")
            return
        self._silent = True
        self.board.set_marks(marks)
        self._silent = False
        self._persist_marks()
        QMessageBox.information(self, "已恢复", f"已载入期号 {src_period} 的标注。")

    @staticmethod
    def _marks_empty(marks: dict) -> bool:
        for area in ("front", "back"):
            m = marks.get(area) or {}
            if m.get("focus") or m.get("exclude"):
                return False
        return True

    def _generate(self) -> None:
        draws = self.store.list_draws_asc()
        if len(draws) < 10:
            QMessageBox.warning(self, "提示", "开奖数据不足，无法生成推荐。")
            return
        self._refresh_zone_rank()
        marks = self.board.get_marks()
        result = generate_recommendations(
            draws,
            marks,
            window=int(self.window_combo.currentData() or 0),
            mode=self.mode_combo.currentData(),
            shape_key=self.shape_combo.currentData(),
            use_duplex_232=self.duplex_check.isChecked(),
            front_odd=int(self.front_oe.currentData()),
            back_odd=int(self.back_oe.currentData()),
            ticket_count=self.ticket_spin.value(),
            constrain_sum=self.sum_check.isChecked(),
        )
        if result.get("error"):
            QMessageBox.warning(self, "无法生成", result["error"])
            return
        self.last_result = result
        self.board.set_dans(result.get("front_dans") or {}, result.get("back_dans") or {})
        lines = []
        lines.append(f"目标期 {self.target_period}　{result['pool_note']}")
        lines.append(
            f"前区和值常用区间 {result.get('sum_lo')}–{result.get('sum_hi')}"
            f"（均值 {result.get('sum_mean')}）"
        )
        lines.append(f"前区{_fmt_dans(result.get('front_dans') or {})}")
        lines.append(f"后区{_fmt_dans(result.get('back_dans') or {})}")
        lines.append(f"232 复式池：{format_nums(result['duplex_232'])}")
        lines.append(f"前区关注池：{format_nums(result['front_pool'])}")
        lines.append(f"后区关注池：{format_nums(result['back_pool'])}")
        excl_f = marks["front"]["exclude"]
        excl_b = marks["back"]["exclude"]
        if excl_f:
            lines.append(f"已排除前区：{format_nums(excl_f)}")
        if excl_b:
            lines.append(f"已排除后区：{format_nums(excl_b)}")
        lines.append("")
        for t in result["tickets"]:
            lines.append(
                f"{t['name']}  {format_nums(t['front'])}  +  {format_nums(t['back'])}"
                f"    形态{t['shape']}  前区{t['odd_even']}  后区{t['back_oe']}"
                f"    前区和{t.get('front_sum', sum(t['front']))}（{t.get('sum_flag', '')}）"
                f"  后区和{t.get('back_sum', sum(t['back']))}"
            )
        self.result_edit.setPlainText("\n".join(lines))

    def _save(self) -> None:
        if not self.target_period:
            QMessageBox.warning(self, "提示", "没有目标期号。")
            return
        if not self.last_result.get("tickets"):
            self._generate()
            if not self.last_result.get("tickets"):
                return
        marks = self.board.get_marks()
        sid = self.store.save_session(
            self.target_period,
            marks,
            self.last_result.get("params") or {},
            self.last_result.get("tickets") or [],
            {
                "front_pool": self.last_result.get("front_pool") or [],
                "back_pool": self.last_result.get("back_pool") or [],
                "duplex_232": self.last_result.get("duplex_232") or [],
                "front_dans": self.last_result.get("front_dans") or {},
                "back_dans": self.last_result.get("back_dans") or {},
            },
            self.note_edit.text().strip(),
        )
        self._persist_marks()
        self.session_saved.emit()
        QMessageBox.information(self, "已保存", f"已写入日志 #{sid}，目标期 {self.target_period}。")

    def apply_session(self, session: dict) -> None:
        """从日志载入标注与结果。"""
        period = session.get("target_period") or self.target_period
        if period:
            self.target_period = period
            self.period_label.setText(f"目标期号：{period}（来自日志）")
        self._silent = True
        self.board.set_marks(session.get("marks") or {})
        self.board.set_dans((session.get("pool") or {}).get("front_dans") or {}, (session.get("pool") or {}).get("back_dans") or {})
        self._silent = False
        self.note_edit.setText(session.get("note") or "")
        params = session.get("params") or {}
        self._apply_params(params)
        tickets = session.get("tickets") or []
        pool = session.get("pool") or {}
        self.last_result = {
            "tickets": tickets,
            "params": params,
            "front_pool": pool.get("front_pool") or [],
            "back_pool": pool.get("back_pool") or [],
            "duplex_232": pool.get("duplex_232") or [],
            "front_dans": pool.get("front_dans") or {},
            "back_dans": pool.get("back_dans") or {},
            "pool_note": "日志载入",
        }
        lines = [f"已载入期号 {period} 的历史推测"]
        if pool.get("front_dans"):
            lines.append(f"前区{_fmt_dans(pool['front_dans'])}")
        if pool.get("back_dans"):
            lines.append(f"后区{_fmt_dans(pool['back_dans'])}")
        if pool.get("duplex_232"):
            lines.append(f"232 复式池：{format_nums(pool['duplex_232'])}")
        for t in tickets:
            lines.append(
                f"{t.get('name', '')}  {format_nums(t.get('front') or [])}  +  {format_nums(t.get('back') or [])}"
            )
        self.result_edit.setPlainText("\n".join(lines))
        self._persist_marks()

    def _apply_params(self, params: dict) -> None:
        def set_combo(combo: QComboBox, value) -> None:
            for i in range(combo.count()):
                if combo.itemData(i) == value:
                    combo.setCurrentIndex(i)
                    return

        set_combo(self.window_combo, params.get("window"))
        set_combo(self.mode_combo, params.get("mode"))
        set_combo(self.shape_combo, params.get("shape_key"))
        set_combo(self.front_oe, params.get("front_odd"))
        set_combo(self.back_oe, params.get("back_odd"))
        if "use_duplex_232" in params:
            self.duplex_check.setChecked(bool(params["use_duplex_232"]))
        if "constrain_sum" in params:
            self.sum_check.setChecked(bool(params["constrain_sum"]))
        if params.get("ticket_count"):
            self.ticket_spin.setValue(int(params["ticket_count"]))


def _fmt_dans(dans: dict) -> str:
    """金/银/铜胆一行显示。"""
    gold = format_nums(dans.get("gold") or []) or "—"
    silver = format_nums(dans.get("silver") or []) or "—"
    bronze = format_nums(dans.get("bronze") or []) or "—"
    return f"金胆 {gold}　银胆 {silver}　铜胆 {bronze}"
