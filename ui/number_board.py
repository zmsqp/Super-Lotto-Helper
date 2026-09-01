# -*- coding: utf-8 -*-
"""号码球：单击循环 默认 → 重点关注 → 排除 → 默认。"""
from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

STATE_DEFAULT = 0
STATE_FOCUS = 1
STATE_EXCLUDE = 2
STATE_NAMES = ("默认", "重点关注", "排除")
DAN_NAMES = {"gold": "金胆", "silver": "银胆", "bronze": "铜胆"}
DAN_RING = {"gold": "#d4af37", "silver": "#7f8c8d", "bronze": "#c87d3a"}

# 关注黄、排除灰、前区默认红、后区默认蓝
_COLORS = {
    ("front", STATE_DEFAULT): ("#c0392b", "#ffffff"),
    ("back", STATE_DEFAULT): ("#2471a3", "#ffffff"),
    ("front", STATE_FOCUS): ("#f1c40f", "#1a1a1a"),
    ("back", STATE_FOCUS): ("#f1c40f", "#1a1a1a"),
    ("front", STATE_EXCLUDE): ("#95a5a6", "#ffffff"),
    ("back", STATE_EXCLUDE): ("#95a5a6", "#ffffff"),
}


def _odd_even_text(num: int) -> str:
    return "奇数" if num % 2 else "偶数"


def _zone_text(num: int, area: str) -> str:
    """前区三区 / 后区大小，便于悬停对照。"""
    if area == "front":
        if num <= 12:
            return "一区 01–12（小号）"
        if num <= 24:
            return "二区 13–24（中号）"
        return "三区 25–35（大号）"
    if num <= 6:
        return "后区小号 01–06"
    return "后区大号 07–12"


class BallButton(QPushButton):
    """单个号码球。自绘圆与数字，避免全局按钮内边距把两位数字裁切。"""

    state_changed = pyqtSignal(int, str, int)  # num, area, state

    def __init__(self, num: int, area: str, parent=None) -> None:
        super().__init__(parent)
        self.num = num
        self.area = area
        self.state = STATE_DEFAULT
        self.dan_role: str | None = None
        self.setFixedSize(42, 42)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        # 透明底，圆和字都在 paintEvent 里画，不受全局 QPushButton padding 影响
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; margin: 0px; }")
        self.clicked.connect(self.cycle)
        self._apply()

    def cycle(self) -> None:
        self.state = (self.state + 1) % 3
        self._apply()
        self.state_changed.emit(self.num, self.area, self.state)

    def set_state(self, state: int, emit: bool = False) -> None:
        self.state = int(state)
        self._apply()
        if emit:
            self.state_changed.emit(self.num, self.area, self.state)

    def set_dan_role(self, role: str | None) -> None:
        """算法叠加的金/银/铜胆，不影响单击标注。"""
        self.dan_role = role
        self._apply()

    def event(self, e: QEvent) -> bool:
        if e.type() in (QEvent.HoverEnter, QEvent.HoverLeave):
            self.update()
        return super().event(e)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        bg, fg = _COLORS[(self.area, self.state)]
        d = min(self.width(), self.height()) - 4
        x = (self.width() - d) // 2
        y = (self.height() - d) // 2
        painter.setBrush(QColor(bg))
        if self.dan_role:
            painter.setPen(QPen(QColor(DAN_RING[self.dan_role]), 3))
        elif self.underMouse():
            painter.setPen(QPen(QColor("#17202a"), 2))
        else:
            painter.setPen(QPen(QColor(bg), 1))
        painter.drawEllipse(x, y, d, d)
        if self.dan_role:
            painter.setPen(QColor(DAN_RING[self.dan_role]))
            badge = QFont("Microsoft YaHei", 7)
            badge.setBold(True)
            painter.setFont(badge)
            painter.drawText(self.rect().adjusted(0, 1, 0, 0), Qt.AlignHCenter | Qt.AlignTop, DAN_NAMES[self.dan_role][0])
        painter.setPen(QColor(fg))
        font = QFont("Microsoft YaHei", 11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 4, 0, 0), Qt.AlignCenter, f"{self.num:02d}")

    def _apply(self) -> None:
        area_name = "前区" if self.area == "front" else "后区"
        odd = _odd_even_text(self.num)
        zone = _zone_text(self.num, self.area)
        state_name = STATE_NAMES[self.state] if 0 <= self.state < 3 else "默认"
        dan_line = f"胆码：{DAN_NAMES[self.dan_role]}\n" if self.dan_role else ""
        self.setToolTip(
            f"{area_name} {self.num:02d}\n"
            f"奇偶：{odd}\n"
            f"分区：{zone}\n"
            f"{dan_line}"
            f"当前标注：{state_name}\n"
            f"单击切换：默认 → 重点关注 → 排除"
        )
        self.update()


class NumberBoard(QWidget):
    """前区 35 + 后区 12 号码盘。"""

    marks_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.front_balls: dict[int, BallButton] = {}
        self.back_balls: dict[int, BallButton] = {}
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        legend = QLabel("单击循环：默认 → 关注 → 排除。悬停看奇偶。生成推荐后，球边金色/银色/铜色圈为金胆/银胆/铜胆。")
        legend.setObjectName("subtitleLabel")
        root.addWidget(legend)

        front_wrap = QWidget()
        fg = QGridLayout(front_wrap)
        fg.setSpacing(6)
        for i in range(1, 36):
            btn = BallButton(i, "front")
            btn.state_changed.connect(self._on_ball)
            self.front_balls[i] = btn
            r, c = divmod(i - 1, 7)
            fg.addWidget(btn, r, c)
        root.addWidget(QLabel("前区 01–35"))
        root.addWidget(front_wrap)

        back_wrap = QWidget()
        bg = QGridLayout(back_wrap)
        bg.setSpacing(6)
        for i in range(1, 13):
            btn = BallButton(i, "back")
            btn.state_changed.connect(self._on_ball)
            self.back_balls[i] = btn
            r, c = divmod(i - 1, 6)
            bg.addWidget(btn, r, c)
        root.addWidget(QLabel("后区 01–12"))
        root.addWidget(back_wrap)

        hint = QHBoxLayout()
        hint.addWidget(self._chip("#c0392b", "前区默认"))
        hint.addWidget(self._chip("#2471a3", "后区默认"))
        hint.addWidget(self._chip("#f1c40f", "关注", "#1a1a1a"))
        hint.addWidget(self._chip("#95a5a6", "排除"))
        hint.addWidget(self._chip("#d4af37", "金胆", "#1a1a1a"))
        hint.addWidget(self._chip("#7f8c8d", "银胆"))
        hint.addWidget(self._chip("#c87d3a", "铜胆"))
        hint.addStretch(1)
        root.addLayout(hint)

    @staticmethod
    def _chip(color: str, text: str, fg: str = "#ffffff") -> QLabel:
        lab = QLabel(f" {text} ")
        lab.setStyleSheet(
            f"background:{color}; color:{fg}; border-radius:8px; padding:2px 8px; font-weight:700;"
        )
        return lab

    def _on_ball(self, _num: int, _area: str, _state: int) -> None:
        self.marks_changed.emit()

    def get_marks(self) -> dict:
        marks = {"front": {"focus": [], "exclude": []}, "back": {"focus": [], "exclude": []}}
        for area, balls in (("front", self.front_balls), ("back", self.back_balls)):
            for num, btn in balls.items():
                if btn.state == STATE_FOCUS:
                    marks[area]["focus"].append(num)
                elif btn.state == STATE_EXCLUDE:
                    marks[area]["exclude"].append(num)
        return marks

    def set_marks(self, marks: dict) -> None:
        self.clear(emit=False)
        for area, balls in (("front", self.front_balls), ("back", self.back_balls)):
            area_marks = marks.get(area) or {}
            for num in area_marks.get("focus") or []:
                if int(num) in balls:
                    balls[int(num)].set_state(STATE_FOCUS)
            for num in area_marks.get("exclude") or []:
                if int(num) in balls:
                    balls[int(num)].set_state(STATE_EXCLUDE)

    def set_dans(self, front: dict | None = None, back: dict | None = None) -> None:
        """把算法给出的金/银/铜胆画到号码球上。"""
        role_of: dict[tuple[str, int], str] = {}
        for area, data in (("front", front or {}), ("back", back or {})):
            for role in ("gold", "silver", "bronze"):
                for num in data.get(role) or []:
                    role_of[(area, int(num))] = role
        for num, btn in self.front_balls.items():
            btn.set_dan_role(role_of.get(("front", num)))
        for num, btn in self.back_balls.items():
            btn.set_dan_role(role_of.get(("back", num)))

    def clear(self, emit: bool = True) -> None:
        self.set_dans({}, {})
        for balls in (self.front_balls, self.back_balls):
            for btn in balls.values():
                btn.set_state(STATE_DEFAULT)
        if emit:
            self.marks_changed.emit()
