# -*- coding: utf-8 -*-
"""三页签主窗：开奖统计 / 本期推测 / 日志检索。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from services.fetcher import fetch_incremental
from services.stats import format_nums, next_draw_meta
from services.store import Store, portable_dir
from ui.app_style import APP_STYLESHEET
from ui.log_tab import LogTab
from ui.predict_tab import PredictTab
from ui.stats_tab import StatsTab


class RefreshWorker(QThread):
    """后台抓取，避免卡住界面。"""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, last_period: str, parent=None) -> None:
        super().__init__(parent)
        self.last_period = last_period

    def run(self) -> None:
        try:
            result = fetch_incremental(self.last_period or None)
            if result.get("error"):
                self.failed.emit(result["error"])
                return
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 后台线程需兜住所有异常
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """大乐透统计与记录主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.store = Store()
        self._worker: RefreshWorker | None = None
        self.setWindowTitle("大乐透统计与记录 v2")
        self.resize(1280, 840)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_menu()
        self._build()
        self._bootstrap()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 8)
        root.setSpacing(8)

        title = QLabel("大乐透统计与记录 v2")
        title.setObjectName("titleLabel")
        subtitle = QLabel("开奖增量、逐期事前参考、覆盖组合优化与推测版本留档。历史拟合不能改变一等奖理论概率。")
        subtitle.setObjectName("subtitleLabel")
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.stats_tab = StatsTab(self.store)
        self.predict_tab = PredictTab(self.store)
        self.log_tab = LogTab(self.store)
        self.tabs.addTab(self.stats_tab, "开奖统计")
        self.tabs.addTab(self.predict_tab, "本期推测")
        self.tabs.addTab(self.log_tab, "日志检索")
        root.addWidget(self.tabs, 1)

        tip = QLabel("仅供个人统计与记录，不构成投注建议。  " + self._data_path_text())
        tip.setObjectName("disclaimerLabel")
        tip.setAlignment(Qt.AlignCenter)
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.stats_tab.refresh_requested.connect(self._start_refresh)
        self.predict_tab.session_saved.connect(self.log_tab.reload)
        self.log_tab.load_session.connect(self._load_session)
        self.log_tab.export_requested.connect(self._export_backup)
        self.log_tab.import_requested.connect(self._import_backup)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("文件")
        act_export = QAction("导出备份…", self)
        act_import = QAction("导入备份…", self)
        act_folder = QAction("打开数据目录", self)
        act_export.triggered.connect(self._export_backup)
        act_import.triggered.connect(self._import_backup)
        act_folder.triggered.connect(self._open_data_dir)
        file_menu.addAction(act_export)
        file_menu.addAction(act_import)
        file_menu.addSeparator()
        file_menu.addAction(act_folder)

    def _data_path_text(self) -> str:
        return f"数据目录：{self.store.data_dir}（跟软件放一起，不删程序就不会丢）"

    def _bootstrap(self) -> None:
        info = self.store.ensure_seed()
        self.stats_tab.reload()
        self.predict_tab.reload()
        self.log_tab.reload()
        if info["seeded"]:
            QMessageBox.information(
                self,
                "已导入历史",
                f"首次启动已导入本地 CSV，共 {info['total']} 期。之后请用「刷新开奖」增量更新。",
            )

    def _start_refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        latest = self.store.latest_draw()
        last_period = latest["period"] if latest else ""
        self.stats_tab.set_refreshing(True)
        self._worker = RefreshWorker(last_period, self)
        self._worker.finished_ok.connect(self._on_refresh_ok)
        self._worker.failed.connect(self._on_refresh_fail)
        self._worker.finished.connect(lambda: self.stats_tab.set_refreshing(False))
        self._worker.start()

    def _on_refresh_ok(self, payload: dict) -> None:
        source = payload.get("source") or ""
        draws = payload.get("draws") or []
        new_ones = self.store.upsert_draws(draws)
        self.store.set_meta("last_source", source)
        self.store.set_meta("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.stats_tab.reload()
        src_name = "体彩接口" if source == "sporttery" else source
        if not new_ones:
            QMessageBox.information(self, "已是最新", f"数据源 {src_name}，本地已是最新开奖，无需合并。")
            return
        latest = self.store.latest_draw()
        nxt = next_draw_meta(latest["date"], latest["period"]) if latest else {}
        lines = [f"数据源 {src_name}，新合并 {len(new_ones)} 期："]
        for d in new_ones[-8:]:
            lines.append(f"{d['period']}  {d['date']}  {format_nums(d['front'])} + {format_nums(d['back'])}")
        if nxt:
            lines.append(f"\n下一期 {nxt['period']} 预计 {nxt['date']}（{nxt['weekday']}），已打开空推测页。")
        QMessageBox.information(self, "发现新开奖", "\n".join(lines))
        if nxt:
            self.predict_tab.set_target_period(nxt["period"], nxt)
            self.tabs.setCurrentWidget(self.predict_tab)
        else:
            self.predict_tab.reload()
        self.log_tab.reload()

    def _on_refresh_fail(self, message: str) -> None:
        QMessageBox.warning(self, "刷新失败", message)

    def _load_session(self, session: dict) -> None:
        self.predict_tab.apply_session(session)
        self.tabs.setCurrentWidget(self.predict_tab)

    def _reload_all(self) -> None:
        self.stats_tab.reload()
        self.predict_tab.reload()
        self.log_tab.reload()

    def _export_backup(self) -> None:
        default = portable_dir() / f"大乐透备份_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dltbak.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出备份",
            str(default),
            "备份文件 (*.dltbak.json);;JSON (*.json);;所有文件 (*.*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() not in (".json",):
            out = out.with_suffix(".dltbak.json")
        try:
            info = self.store.export_backup(out)
        except Exception as exc:  # noqa: BLE001 导出失败直接提示
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QMessageBox.information(
            self,
            "已导出",
            f"已保存到：\n{info['path']}\n\n开奖 {info['draws']} 期，标注 {info['marks']} 条，日志 {info['sessions']} 条。",
        )

    def _import_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入备份",
            str(portable_dir()),
            "备份文件 (*.dltbak.json *.json);;所有文件 (*.*)",
        )
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "确认导入",
            "将把备份合并进当前数据库：\n开奖按期号去重，日志按时间去重，不会覆盖已有不同记录。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            info = self.store.import_backup(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self._reload_all()
        QMessageBox.information(
            self,
            "已导入",
            f"新开奖 {info['draws_new']} 期，标注 {info['marks']} 条，新日志 {info['sessions_new']} 条。\n"
            f"当前共 {info['draws_total']} 期开奖、{info['sessions_total']} 条日志。",
        )

    def _open_data_dir(self) -> None:
        self.store.data_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.data_dir)))

    def closeEvent(self, event) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        super().closeEvent(event)
