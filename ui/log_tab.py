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

from services.algorithms import ODD_EVEN_ANY, ODD_EVEN_INFER, SHAPE_INFER, comparison_summary
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
        for session in self._rows:
            draw = self.store.get_draw(session["target_period"])
            if not draw:
                continue
            comparison = comparison_summary(session.get("tickets") or [], draw, session.get("pool") or {})
            session["comparison"] = comparison
            needs_update = (
                session.get("hit_front") != comparison["best_front"]
                or session.get("hit_back") != comparison["best_back"]
                or any("grade" not in ticket for ticket in session.get("tickets") or [])
                or any(
                    old.get("grade") != new.get("grade")
                    for old, new in zip(session.get("tickets") or [], comparison["tickets"])
                )
            )
            if needs_update:
                self.store.update_session_hits(
                    session["id"], comparison["best_front"], comparison["best_back"], comparison["tickets"]
                )
                session["tickets"] = comparison["tickets"]
                session["hit_front"] = comparison["best_front"]
                session["hit_back"] = comparison["best_back"]
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
                comp = s.get("comparison") or {}
                grade = f" {comp['best_grade']}等奖" if comp.get("best_grade") else " 未中奖"
                hit = (
                    f"最佳前{s['hit_front']}后{s['hit_back']}{grade}｜"
                    f"池前{comp.get('front_pool_hits', 0)}后{comp.get('back_pool_hits', 0)}"
                )
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
                f"{_format_structure_param(params)}  "
                f"{_format_odd_even_param(params)}"
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
            label = params.get("duplex_label") or ""
            prefix = f"复式池 {label}：" if label else "复式池："
            lines.append(prefix + format_nums(pool["duplex_232"]))
        lines.append("")
        for t in s.get("tickets") or []:
            extra = ""
            if t.get("hit_front") is not None:
                grade = f"，{t['grade']}等奖" if t.get("grade") else "，未中奖"
                extra = f"  → 命中前{t['hit_front']}后{t['hit_back']}{grade}"
            lines.append(
                f"{t.get('name', '')}  {format_nums(t.get('front') or [])}  +  "
                f"{format_nums(t.get('back') or [])}{extra}"
            )
        if s.get("hit_front") is not None:
            comp = s.get("comparison") or {}
            best_grade = f"{comp['best_grade']}等奖" if comp.get("best_grade") else "未中奖"
            lines.append(
                f"\n五注最佳同注：前区{s['hit_front']}　后区{s['hit_back']}　{best_grade}"
            )
            lines.append(
                f"候选池覆盖：前区{comp.get('front_pool_hits', 0)}/{comp.get('front_pool_size', 0)}　"
                f"后区{comp.get('back_pool_hits', 0)}/{comp.get('back_pool_size', 0)}"
            )
            lines.append("公平单注命中均值：前区0.714　后区0.333；池覆盖不等于单注命中。")
        model = (s.get("pool") or {}).get("model") or {}
        front_ref = model.get("front_reference") or {}
        if front_ref:
            lines.append(
                f"保存时样本外参考：前区{front_ref.get('pool_size')}号池覆盖2+ "
                f"{front_ref.get('coverage_rate', 0):.1%}，公平基准{front_ref.get('fair_rate', 0):.1%}。"
            )
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
        comparison = comparison_summary(s.get("tickets") or [], draw, s.get("pool") or {})
        tickets = comparison["tickets"]
        self.store.update_session_hits(s["id"], comparison["best_front"], comparison["best_back"], tickets)
        self.reload()
        # 重新选中同一条
        for i, row in enumerate(self._rows):
            if row["id"] == s["id"]:
                self.table.selectRow(i)
                break
        lines = [f"{s['target_period']} 开奖 {format_nums(draw['front'])} + {format_nums(draw['back'])}", ""]
        for t in tickets:
            ticket_grade = f"{t['grade']}等奖" if t.get("grade") else "未中奖"
            lines.append(
                f"{t.get('name', '')}  {format_nums(t.get('front') or [])} + {format_nums(t.get('back') or [])}"
                f"  → 前区{t['hit_front']} 后区{t['hit_back']} {ticket_grade}"
            )
        grade = f"{comparison['best_grade']}等奖" if comparison.get("best_grade") else "未中奖"
        lines.append(
            f"\n五注最佳同注：前{comparison['best_front']}后{comparison['best_back']}，{grade}；"
            f"候选池覆盖前{comparison['front_pool_hits']}后{comparison['back_pool_hits']}。"
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


def _format_structure_param(params: dict) -> str:
    """参数行写出形态、复式池、和值是历史推理还是手动，并带上分配。"""
    shape_key = params.get("shape_key")
    if shape_key == SHAPE_INFER or params.get("shape_infer"):
        plan = params.get("shape_plan") or []
        shape_text = "形态历史推理 " + ("、".join(str(item) for item in plan) if plan else "—")
    else:
        shape_text = f"形态{shape_key}"
    if not params.get("use_duplex_232"):
        duplex_text = "复式池关"
    elif params.get("duplex_infer") or params.get("duplex_label"):
        duplex_text = f"复式池历史推理 {params.get('duplex_label') or params.get('duplex_shape') or '—'}"
    else:
        duplex_text = "232复式=开"
    if not params.get("constrain_sum"):
        sum_text = "和值不参考"
    elif params.get("sum_infer") or params.get("sum_plan"):
        plan = params.get("sum_plan") or []
        sum_text = "和值历史推理 " + ("、".join(str(item) for item in plan) if plan else "—")
    else:
        sum_text = "和值约束=开"
    return f"{shape_text}  {duplex_text}  {sum_text}"


def _format_odd_even_param(params: dict) -> str:
    """参数行：历史推理写出各注分配，不限和硬约束仍显示原来的含义。"""

    def one(side: str) -> str:
        name = "前区" if side == "front" else "后区"
        odd = params.get(f"{side}_odd")
        plan = params.get(f"{side}_odd_plan") or []
        if odd == ODD_EVEN_INFER:
            alloc = "、".join(str(item) for item in plan) if plan else "—"
            return f"{name}历史推理 {alloc}"
        if odd == ODD_EVEN_ANY:
            return f"{name}不限"
        return f"{name}奇偶奇数{odd}"

    return f"{one('front')}  {one('back')}"
