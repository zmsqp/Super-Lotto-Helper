# -*- coding: utf-8 -*-
"""打包 Windows 单文件 EXE。数据写在 EXE 旁边的 data 目录，不会打进临时目录。"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent
SEED_CSV = ROOT / "assets" / "dlt_history.csv"
if not SEED_CSV.exists():
    SEED_CSV = ROOT.parent / "1" / "dlt_history.csv"
DIST = ROOT / "dist_v2_1"
EXE_NAME = "大乐透统计与记录_v2.1"


def copy_database(source: Path, target: Path) -> None:
    """使用 SQLite 在线备份，源程序运行时也能得到一致的数据库副本。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as src, sqlite3.connect(str(target)) as dst:
        src.backup(dst)


def main() -> None:
    if not SEED_CSV.exists():
        raise FileNotFoundError(f"未找到种子开奖 CSV：{SEED_CSV}")

    PyInstaller.__main__.run(
        [
            "-y",
            str(ROOT / "main.py"),
            "--onefile",
            "--windowed",
            "--clean",
            "-n",
            EXE_NAME,
            "--hidden-import",
            "PyQt5.sip",
            f"--add-data={SEED_CSV}{os.pathsep}assets",
            "--distpath",
            str(DIST),
            "--workpath",
            str(ROOT / "build"),
            "--specpath",
            str(ROOT / "build"),
        ]
    )
    # 独立版本目录不覆盖旧程序，首次打包时继承上一版用户数据库和推测日志。
    dst_db_dir = DIST / "data"
    target = dst_db_dir / "dlt.db"
    sources = [ROOT / "dist_v2" / "data" / "dlt.db", ROOT / "data" / "dlt.db"]
    source = next((item for item in sources if item.exists()), None)
    if source is not None and not target.exists():
        copy_database(source, target)
    exe = DIST / f"{EXE_NAME}.exe"
    print(f"已生成：{exe}")
    print(f"数据目录：{dst_db_dir}（已存在的数据库不会被打包覆盖）")


if __name__ == "__main__":
    main()
