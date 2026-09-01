# -*- coding: utf-8 -*-
"""打包 Windows 单文件 EXE。数据写在 EXE 旁边的 data 目录，不会打进临时目录。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent
SEED_CSV = ROOT / "assets" / "dlt_history.csv"
if not SEED_CSV.exists():
    SEED_CSV = ROOT.parent / "1" / "dlt_history.csv"
DIST = ROOT / "dist"
EXE_NAME = "大乐透统计与记录"


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
    # 把现有本地库拷到 EXE 旁边，避免打包后看起来像“日志丢了”
    src_db_dir = ROOT / "data"
    dst_db_dir = DIST / "data"
    if src_db_dir.exists():
        dst_db_dir.mkdir(parents=True, exist_ok=True)
        for item in src_db_dir.iterdir():
            target = dst_db_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
    exe = DIST / f"{EXE_NAME}.exe"
    print(f"已生成：{exe}")
    print(f"数据目录：{dst_db_dir}（请与 EXE 放在同一文件夹）")


if __name__ == "__main__":
    main()
