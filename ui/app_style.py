# -*- coding: utf-8 -*-
"""对齐 pdfcj Fusion + 微软雅黑 + GroupBox 的全局样式。"""

APP_STYLESHEET = """
QWidget {
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #17202a;
}
QLabel#titleLabel {
    font-size: 22px;
    font-weight: 700;
    color: #17202a;
}
QLabel#subtitleLabel {
    color: #5f6b76;
}
QLabel#disclaimerLabel {
    color: #7f8c8d;
    font-size: 12px;
    padding: 6px 4px 2px 4px;
}
QLabel#latestBalls {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
}
QGroupBox {
    border: 1px solid #d8dee6;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #34495e;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox, QDateEdit {
    border: 1px solid #cfd6df;
    border-radius: 4px;
    padding: 6px;
    background: #ffffff;
}
QPushButton {
    border: 1px solid #9fb3c8;
    border-radius: 4px;
    padding: 8px 14px;
    background: #f4f8fb;
}
QPushButton:hover {
    background: #e9f2f8;
}
QPushButton:disabled {
    color: #8a96a3;
    background: #eef1f4;
}
QPushButton#primaryButton {
    background: #2d8fdd;
    color: #ffffff;
    border: 1px solid #2478b8;
    font-weight: 700;
}
QPushButton#primaryButton:hover {
    background: #3b9be8;
}
QTabWidget::pane {
    border: 1px solid #d8dee6;
    border-radius: 6px;
    top: -1px;
    background: #ffffff;
}
QTabBar::tab {
    background: #eef2f6;
    color: #34495e;
    border: 1px solid #d8dee6;
    border-bottom: none;
    padding: 9px 18px;
    margin-right: 4px;
    min-width: 96px;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1a5276;
}
QTableWidget {
    gridline-color: #e5e8eb;
    selection-background-color: #d6eaf8;
    selection-color: #1a252f;
    alternate-background-color: #f8fafc;
}
QHeaderView::section {
    background: #eef2f6;
    padding: 6px;
    border: 1px solid #d8dee6;
    font-weight: 700;
}
QProgressBar {
    border: 1px solid #cfd6df;
    border-radius: 4px;
    text-align: center;
    height: 22px;
}
QProgressBar::chunk {
    background-color: #2d8fdd;
    border-radius: 3px;
}
"""
