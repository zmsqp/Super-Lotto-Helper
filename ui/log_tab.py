# -*- coding: utf-8 -*-
"""推测日志检索与开奖后命中对照。"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.algorithms import compare_tickets
from services.stats import format_nums


class LogTab(QWidget):
    """日志检索页。"""

    load_session = pyqtSignal(dict)
    export_requested = pyqtSignal()
    import_requested = pyqtSignal()

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._rows: list[dict] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        filt = QGroupBox("检索")
        fl = QHBoxLayout(filt)
        self.period_edit = QLineEdit()
        self.period_edit.setPlaceholderText("期号")
        self.date_from = QLineEdit()
        self.date_from.setPlaceholderText("起始日期 YYYY-MM-DD")
        self.date_to = QLineEdit()
        self.date_to.setPlaceholderText("结束日期 YYYY-MM-DD")
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("备注关键字")
        self.only_drawn = QCheckBox("仅看已开奖")
        search_btn = QPushButton("查询")
        search_btn.setObjectName("primaryButton")
        search_btn.clicked.connect(self.reload)
        export_btn = QPushButton("导出备份")
        import_btn = QPushButton("导入备份")
        export_btn.clicked.connect(self.export_requested.emit)
        import_btn.clicked.connect(self.import_requested.emit)
        fl.addWidget(QLabel("期号"))
        fl.addWidget(self.period_edit)
        fl.addWidget(self.date_from)
        fl.addWidget(self.date_to)
        fl.addWidget(self.keyword_edit, 1)
        fl.addWidget(self.only_drawn)
        fl.addWidget(search_btn)
        fl.addWidget(export_btn)
        fl.addWidget(import_btn)
        root.addWidget(filt)

        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "目标期号", "保存时间", "推荐摘要", "备注", "命中"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_detail)
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        btns = QHBoxLayout()
        self.compare_btn = QPushButton("对照开奖")
        self.compare_btn.setObjectName("primaryButton")
        self.load_btn = QPushButton("载入到推测页")
        self.delete_btn = QPushButton("删除日志")
        self.compare_btn.clicked.connect(self._compare)
        self.load_btn.clicked.connect(self._load_to_predict)
        self.delete_btn.clicked.connect(self._delete)
        btns.addWidget(self.compare_btn)
        btns.addWidget(self.load_btn)
        btns.addWidget(self.delete_btn)
        btns.addStretch(1)
        rl.addLayout(btns)
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        rl.addWidget(self.detail)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

    def reload(self) -> None:
        self._rows = self.store.search_sessions(
            period=self.period_edit.text(),
            date_from=self.date_from.text(),
            date_to=self.date_to.text(),
            keyword=self.keyword_edit.text(),
            only_drawn=self.only_drawn.isChecked(),
        )
        self.table.setRowCount(len(self._rows))
        for i, s in enumerate(self._rows):
            summary = ""
            tickets = s.get("tickets") or []
            if tickets:
                t0 = tickets[0]
                summary = f"{format_nums(t0.get('front') or [])} + {format_nums(t0.get('back') or [])}"
                if len(tickets) > 1:
                    summary += f" 等{len(tickets)}注"
            hit = "—"
            if s.get("hit_front") is not None:
                hit = f"前{s['hit_front']} 后{s['hit_back']}"
            vals = [str(s["id"]), s["target_period"], s["saved_at"], summary, s.get("note") or "", hit]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.UserRole, s["id"] if c == 0 else None)
                self.table.setItem(i, c, item)
        if self._rows:
            self.table.selectRow(0)
        else:
            self.detail.clear()

    def _current_session(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._rows):
            return None
        return self._rows[idx]

    def _show_detail(self) -> None:
        s = self._current_session()
        if not s:
            self.detail.clear()
            return
        lines = [
            f"日志 #{s['id']}　目标期 {s['target_period']}",
            f"保存时间 {s['saved_at']}",
            f"备注：{s.get('note') or '（无）'}",
        ]
        params = s.get("params") or {}
        if params:
            lines.append(
                f"参数：窗口{params.get('window')}  模式{params.get('mode')}  "
                f"形态{params.get('shape_key')}  232复式={params.get('use_duplex_232')}  "
                f"前区奇偶奇数{params.get('front_odd')}  后区{params.get('back_odd')}"
            )
        marks = s.get("marks") or {}
        ff = (marks.get("front") or {}).get("focus") or []
        fe = (marks.get("front") or {}).get("exclude") or []
        bf = (marks.get("back") or {}).get("focus") or []
        be = (marks.get("back") or {}).get("exclude") or []
        if ff or fe or bf or be:
            lines.append(f"关注前区 {format_nums(ff) or '无'}　排除前区 {format_nums(fe) or '无'}")
            lines.append(f"关注后区 {format_nums(bf) or '无'}　排除后区 {format_nums(be) or '无'}")
        pool = s.get("pool") or {}
        if pool.get("front_dans") or pool.get("back_dans"):
            fd = pool.get("front_dans") or {}
            bd = pool.get("back_dans") or {}
            lines.append(
                f"前区金胆 {format_nums(fd.get('gold') or []) or '—'}　"
                f"银胆 {format_nums(fd.get('silver') or []) or '—'}　"
                f"铜胆 {format_nums(fd.get('bronze') or []) or '—'}"
            )
            lines.append(
                f"后区金胆 {format_nums(bd.get('gold') or []) or '—'}　"
                f"银胆 {format_nums(bd.get('silver') or []) or '—'}　"
                f"铜胆 {format_nums(bd.get('bronze') or []) or '—'}"
            )
        if pool.get("duplex_232"):
            lines.append(f"232 池：{format_nums(pool['duplex_232'])}")
        lines.append("")
        for t in s.get("tickets") or []:
            extra = ""
            if t.get("hit_front") is not None:
                extra = f"  → 命中前{t['hit_front']}后{t['hit_back']}"
            lines.append(
                f"{t.get('name', '')}  {format_nums(t.get('front') or [])}  +  "
                f"{format_nums(t.get('back') or [])}{extra}"
            )
        if s.get("hit_front") is not None:
            lines.append(f"\n主推对照：前区 {s['hit_front']}　后区 {s['hit_back']}　于 {s.get('compared_at')}")
        draw = self.store.get_draw(s["target_period"])
        if draw:
            lines.append(
                f"开奖号码：{format_nums(draw['front'])}  +  {format_nums(draw['back'])}  （{draw['date']}）"
            )
        else:
            lines.append("该期尚未录入开奖。")
        self.detail.setPlainText("\n".join(lines))

    def _compare(self) -> None:
        s = self._current_session()
        if not s:
            QMessageBox.information(self, "提示", "请先选择一条日志。")
            return
        draw = self.store.get_draw(s["target_period"])
        if not draw:
            QMessageBox.information(self, "尚未开奖", f"期号 {s['target_period']} 还没有开奖记录。")
            return
        tickets, hf, hb = compare_tickets(s.get("tickets") or [], draw)
        self.store.update_session_hits(s["id"], hf, hb, tickets)
        self.reload()
        # 重新选中同一条
        for i, row in enumerate(self._rows):
            if row["id"] == s["id"]:
                self.table.selectRow(i)
                break
        lines = [f"{s['target_period']} 开奖 {format_nums(draw['front'])} + {format_nums(draw['back'])}", ""]
        for t in tickets:
            lines.append(
                f"{t.get('name', '')}  {format_nums(t.get('front') or [])} + {format_nums(t.get('back') or [])}"
                f"  → 前区{t['hit_front']} 后区{t['hit_back']}"
            )
        QMessageBox.information(self, "对照结果", "\n".join(lines))

    def _load_to_predict(self) -> None:
        s = self._current_session()
        if not s:
            QMessageBox.information(self, "提示", "请先选择一条日志。")
            return
        self.load_session.emit(s)

    def _delete(self) -> None:
        s = self._current_session()
        if not s:
            QMessageBox.information(self, "提示", "请先选择一条日志。")
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除日志 #{s['id']}（目标期 {s['target_period']}）？\n此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if not self.store.delete_session(s["id"]):
            QMessageBox.warning(self, "删除失败", "没有找到这条日志，可能已被删掉。")
            return
        self.detail.clear()
        self.reload()
