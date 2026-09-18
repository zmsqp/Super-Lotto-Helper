# -*- coding: utf-8 -*-
"""SQLite 本地库：开奖、标注、推测日志、元数据。"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

BACKUP_VERSION = 2


def is_frozen() -> bool:
    """是否已打包成 EXE。"""
    return bool(getattr(sys, "frozen", False))


def portable_dir() -> Path:
    """程序目录：EXE 所在文件夹，或源码根目录。日志跟软件放一起。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """只读资源目录（打包后在临时解包目录）。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", portable_dir()))
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """可写数据目录：程序旁的 data，不删软件就不会丢。"""
    return portable_dir() / "data"


def default_db_path() -> Path:
    return data_dir() / "dlt.db"


def seed_csv_path() -> Path:
    """首次导入用的历史 CSV：打包资源优先，其次仓库里的 1/dlt_history.csv。"""
    bundled = resource_dir() / "assets" / "dlt_history.csv"
    if bundled.exists():
        return bundled
    return Path(__file__).resolve().parent.parent.parent / "1" / "dlt_history.csv"


def market_seed_path() -> Path:
    bundled = resource_dir() / "assets" / "dlt_market_history.json"
    if bundled.exists():
        return bundled
    return Path(__file__).resolve().parent.parent / "assets" / "dlt_market_history.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Store:
    """大乐透本地存储。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.data_dir = data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS draws (
                period TEXT PRIMARY KEY,
                draw_date TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS marks (
                target_period TEXT NOT NULL,
                area TEXT NOT NULL,
                num INTEGER NOT NULL,
                mark TEXT NOT NULL,
                PRIMARY KEY (target_period, area, num)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_period TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                marks_json TEXT NOT NULL,
                params_json TEXT NOT NULL,
                tickets_json TEXT NOT NULL,
                pool_json TEXT DEFAULT '{}',
                note TEXT DEFAULT '',
                hit_front INTEGER,
                hit_back INTEGER,
                compared_at TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS draw_market (
                period TEXT PRIMARY KEY,
                sales INTEGER NOT NULL DEFAULT 0,
                pool_balance REAL NOT NULL DEFAULT 0,
                prize_counts_json TEXT NOT NULL DEFAULT '{}',
                prize_amounts_json TEXT NOT NULL DEFAULT '{}',
                fetched_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(period) REFERENCES draws(period)
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_period ON sessions(target_period);
            CREATE INDEX IF NOT EXISTS idx_sessions_saved ON sessions(saved_at);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def count_draws(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM draws").fetchone()
        return int(row["n"])

    def latest_draw(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM draws ORDER BY period DESC LIMIT 1"
        ).fetchone()
        return self._row_to_draw(row) if row else None

    def get_draw(self, period: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM draws WHERE period=?", (period,)).fetchone()
        return self._row_to_draw(row) if row else None

    def list_draws(self, newest_first: bool = True) -> list[dict]:
        order = "DESC" if newest_first else "ASC"
        rows = self.conn.execute(f"SELECT * FROM draws ORDER BY period {order}").fetchall()
        return [self._row_to_draw(r) for r in rows]

    def list_draws_asc(self) -> list[dict]:
        """统计用：旧到新。"""
        return self.list_draws(newest_first=False)

    def upsert_draws(self, draws: list[dict]) -> list[dict]:
        """按期号合并，返回本次新插入的期。"""
        if not draws:
            return []
        existing = {r["period"] for r in self.conn.execute("SELECT period FROM draws")}
        new_ones = []
        for d in draws:
            period = str(d["period"]).strip()
            if period not in existing:
                self.conn.execute(
                    "INSERT INTO draws(period, draw_date, front, back) VALUES(?,?,?,?)",
                    (
                        period,
                        d["date"],
                        json.dumps(list(d["front"]), ensure_ascii=False),
                        json.dumps(list(d["back"]), ensure_ascii=False),
                    ),
                )
                existing.add(period)
                new_ones.append(
                    {
                        "period": period,
                        "date": d["date"],
                        "front": list(d["front"]),
                        "back": list(d["back"]),
                    }
                )
            if d.get("sales") or d.get("prize_counts"):
                self._upsert_market_row(d)
        self.conn.commit()
        new_ones.sort(key=lambda r: r["period"])
        return new_ones

    def _upsert_market_row(self, row: dict) -> None:
        period = str(row.get("period") or "").strip()
        if not period:
            return
        self.conn.execute(
            """
            INSERT INTO draw_market(
                period, sales, pool_balance, prize_counts_json, prize_amounts_json, fetched_at
            ) VALUES(?,?,?,?,?,?)
            ON CONFLICT(period) DO UPDATE SET
                sales=excluded.sales,
                pool_balance=excluded.pool_balance,
                prize_counts_json=excluded.prize_counts_json,
                prize_amounts_json=excluded.prize_amounts_json,
                fetched_at=excluded.fetched_at
            """,
            (
                period,
                int(row.get("sales") or 0),
                float(row.get("pool_balance") or 0),
                json.dumps(row.get("prize_counts") or {}, ensure_ascii=False),
                json.dumps(row.get("prize_amounts") or {}, ensure_ascii=False),
                _now(),
            ),
        )

    def upsert_market_rows(self, rows: list[dict]) -> int:
        existing_periods = {
            row["period"] for row in self.conn.execute("SELECT period FROM draws")
        }
        saved = 0
        for row in rows:
            if str(row.get("period") or "") not in existing_periods:
                continue
            self._upsert_market_row(row)
            saved += 1
        self.conn.commit()
        return saved

    def list_market_asc(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT m.*, d.draw_date, d.front, d.back
            FROM draw_market m JOIN draws d ON d.period=m.period
            ORDER BY m.period ASC
            """
        ).fetchall()
        return [
            {
                "period": row["period"],
                "date": row["draw_date"],
                "front": json.loads(row["front"]),
                "back": json.loads(row["back"]),
                "sales": int(row["sales"]),
                "pool_balance": float(row["pool_balance"]),
                "prize_counts": json.loads(row["prize_counts_json"] or "{}"),
                "prize_amounts": json.loads(row["prize_amounts_json"] or "{}"),
            }
            for row in rows
        ]

    def import_market_seed(self, path: Optional[Path] = None) -> int:
        source = Path(path) if path else market_seed_path()
        if not source.exists():
            return 0
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        return self.upsert_market_rows(rows or [])

    def ensure_market_seed(self) -> dict:
        existing = int(self.conn.execute("SELECT COUNT(*) FROM draw_market").fetchone()[0])
        imported = self.import_market_seed()
        total = int(self.conn.execute("SELECT COUNT(*) FROM draw_market").fetchone()[0])
        return {"imported": max(total - existing, 0), "total": total}

    def import_csv(self, path: Optional[Path] = None) -> int:
        """导入种子 CSV，返回新插入期数。"""
        csv_path = Path(path) if path else seed_csv_path()
        if not csv_path.exists():
            return 0
        rows = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    period = str(row["期号"]).strip()
                    date = str(row["开奖日期"]).strip()
                    front = [int(row[f"前区{i}"]) for i in range(1, 6)]
                    back = [int(row[f"后区{i}"]) for i in range(1, 3)]
                except (KeyError, ValueError, TypeError):
                    continue
                if len(front) != 5 or len(back) != 2:
                    continue
                rows.append({"period": period, "date": date, "front": front, "back": back})
        return len(self.upsert_draws(rows))

    def ensure_seed(self) -> dict:
        """库空时导入 CSV。返回 {imported, total, seeded}。"""
        total = self.count_draws()
        if total > 0:
            return {"imported": 0, "total": total, "seeded": False}
        imported = self.import_csv()
        self.set_meta("last_source", "csv")
        self.set_meta("last_refresh", _now())
        return {"imported": imported, "total": self.count_draws(), "seeded": True}

    def save_marks(self, target_period: str, marks: dict) -> None:
        """整期覆盖标注。marks: {front:{focus:[], exclude:[]}, back:{...}}。"""
        self.conn.execute("DELETE FROM marks WHERE target_period=?", (target_period,))
        for area in ("front", "back"):
            area_marks = marks.get(area) or {}
            for num in area_marks.get("focus") or []:
                self.conn.execute(
                    "INSERT INTO marks(target_period, area, num, mark) VALUES(?,?,?,?)",
                    (target_period, area, int(num), "focus"),
                )
            for num in area_marks.get("exclude") or []:
                self.conn.execute(
                    "INSERT INTO marks(target_period, area, num, mark) VALUES(?,?,?,?)",
                    (target_period, area, int(num), "exclude"),
                )
        self.conn.commit()

    def load_marks(self, target_period: str) -> dict:
        marks = {"front": {"focus": [], "exclude": []}, "back": {"focus": [], "exclude": []}}
        rows = self.conn.execute(
            "SELECT area, num, mark FROM marks WHERE target_period=?",
            (target_period,),
        ).fetchall()
        for r in rows:
            area = r["area"]
            kind = r["mark"]
            if area in marks and kind in marks[area]:
                marks[area][kind].append(int(r["num"]))
        for area in marks:
            for kind in marks[area]:
                marks[area][kind].sort()
        return marks

    def clear_marks(self, target_period: str) -> None:
        self.conn.execute("DELETE FROM marks WHERE target_period=?", (target_period,))
        self.conn.commit()

    def latest_marks_period(self) -> Optional[str]:
        """最近一次有标注的目标期号。"""
        row = self.conn.execute(
            "SELECT target_period FROM marks ORDER BY target_period DESC LIMIT 1"
        ).fetchone()
        if row:
            return row["target_period"]
        row = self.conn.execute(
            "SELECT target_period FROM sessions ORDER BY saved_at DESC LIMIT 1"
        ).fetchone()
        return row["target_period"] if row else None

    def save_session(
        self,
        target_period: str,
        marks: dict,
        params: dict,
        tickets: list,
        pool: dict,
        note: str = "",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO sessions(target_period, saved_at, marks_json, params_json, tickets_json, pool_json, note)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                target_period,
                _now(),
                json.dumps(marks, ensure_ascii=False),
                json.dumps(params, ensure_ascii=False),
                json.dumps(tickets, ensure_ascii=False),
                json.dumps(pool, ensure_ascii=False),
                note or "",
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_session(self, session_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def latest_session(self) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM sessions ORDER BY saved_at DESC LIMIT 1").fetchone()
        return self._row_to_session(row) if row else None

    def search_sessions(
        self,
        period: str = "",
        date_from: str = "",
        date_to: str = "",
        keyword: str = "",
        only_drawn: bool = False,
    ) -> list[dict]:
        sql = "SELECT s.* FROM sessions s WHERE 1=1"
        args: list[Any] = []
        if period.strip():
            sql += " AND s.target_period LIKE ?"
            args.append(f"%{period.strip()}%")
        if date_from.strip():
            sql += " AND s.saved_at >= ?"
            args.append(date_from.strip())
        if date_to.strip():
            sql += " AND s.saved_at <= ?"
            # 含当天：若只给日期则补到日末
            end = date_to.strip()
            if len(end) == 10:
                end = end + " 23:59:59"
            args.append(end)
        if keyword.strip():
            sql += " AND s.note LIKE ?"
            args.append(f"%{keyword.strip()}%")
        if only_drawn:
            sql += " AND EXISTS (SELECT 1 FROM draws d WHERE d.period = s.target_period)"
        sql += " ORDER BY s.saved_at DESC"
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_session(r) for r in rows]

    def list_all_sessions(self) -> list[dict]:
        """导出用：全部推测日志。"""
        rows = self.conn.execute("SELECT * FROM sessions ORDER BY id").fetchall()
        return [self._row_to_session(r) for r in rows]

    def dump_marks_rows(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT target_period, area, num, mark FROM marks ORDER BY target_period, area, num"
        ).fetchall()
        return [
            {
                "target_period": r["target_period"],
                "area": r["area"],
                "num": int(r["num"]),
                "mark": r["mark"],
            }
            for r in rows
        ]

    def dump_meta(self) -> dict:
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def export_backup(self, path: Path) -> dict:
        """导出开奖、标注、日志到 JSON 备份。"""
        payload = {
            "app": "dlt_studio",
            "version": BACKUP_VERSION,
            "exported_at": _now(),
            "draws": self.list_draws_asc(),
            "market": self.list_market_asc(),
            "marks": self.dump_marks_rows(),
            "sessions": self.list_all_sessions(),
            "meta": self.dump_meta(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "path": str(out),
            "draws": len(payload["draws"]),
            "market": len(payload["market"]),
            "marks": len(payload["marks"]),
            "sessions": len(payload["sessions"]),
        }

    def import_backup(self, path: Path) -> dict:
        """合并导入备份：开奖按期号去重，日志按期号+时间+备注去重。"""
        raw = Path(path).read_text(encoding="utf-8-sig")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("备份文件格式不对")
        if payload.get("app") not in (None, "dlt_studio"):
            raise ValueError("这不是本程序的备份文件")
        draws = payload.get("draws") or []
        new_draws = self.upsert_draws(draws)
        market_saved = self.upsert_market_rows(payload.get("market") or [])
        new_marks = 0
        for row in payload.get("marks") or []:
            try:
                self.conn.execute(
                    """
                    INSERT INTO marks(target_period, area, num, mark) VALUES(?,?,?,?)
                    ON CONFLICT(target_period, area, num) DO UPDATE SET mark=excluded.mark
                    """,
                    (
                        str(row["target_period"]),
                        str(row["area"]),
                        int(row["num"]),
                        str(row["mark"]),
                    ),
                )
                new_marks += 1
            except (KeyError, TypeError, ValueError):
                continue
        new_sessions = 0
        for s in payload.get("sessions") or []:
            if self._insert_session_from_backup(s):
                new_sessions += 1
        meta = payload.get("meta") or {}
        if isinstance(meta, dict):
            for key, value in meta.items():
                if key and value is not None:
                    self.set_meta(str(key), str(value))
        self.conn.commit()
        return {
            "draws_new": len(new_draws),
            "marks": new_marks,
            "sessions_new": new_sessions,
            "draws_total": self.count_draws(),
            "market": market_saved,
            "sessions_total": len(self.list_all_sessions()),
        }

    def _insert_session_from_backup(self, s: dict) -> bool:
        period = str(s.get("target_period") or "").strip()
        saved_at = str(s.get("saved_at") or "").strip()
        note = s.get("note") or ""
        if not period or not saved_at:
            return False
        exists = self.conn.execute(
            "SELECT id FROM sessions WHERE target_period=? AND saved_at=? AND IFNULL(note,'')=?",
            (period, saved_at, note),
        ).fetchone()
        if exists:
            return False
        self.conn.execute(
            """
            INSERT INTO sessions(
                target_period, saved_at, marks_json, params_json, tickets_json, pool_json,
                note, hit_front, hit_back, compared_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                period,
                saved_at,
                json.dumps(s.get("marks") or {}, ensure_ascii=False),
                json.dumps(s.get("params") or {}, ensure_ascii=False),
                json.dumps(s.get("tickets") or [], ensure_ascii=False),
                json.dumps(s.get("pool") or {}, ensure_ascii=False),
                note,
                s.get("hit_front"),
                s.get("hit_back"),
                s.get("compared_at"),
            ),
        )
        return True

    def delete_session(self, session_id: int) -> bool:
        """删除一条推测日志。"""
        cur = self.conn.execute("DELETE FROM sessions WHERE id=?", (int(session_id),))
        self.conn.commit()
        return cur.rowcount > 0

    def update_session_hits(
        self, session_id: int, hit_front: int, hit_back: int, tickets: Optional[list] = None
    ) -> None:
        if tickets is not None:
            self.conn.execute(
                "UPDATE sessions SET hit_front=?, hit_back=?, compared_at=?, tickets_json=? WHERE id=?",
                (hit_front, hit_back, _now(), json.dumps(tickets, ensure_ascii=False), session_id),
            )
        else:
            self.conn.execute(
                "UPDATE sessions SET hit_front=?, hit_back=?, compared_at=? WHERE id=?",
                (hit_front, hit_back, _now(), session_id),
            )
        self.conn.commit()

    @staticmethod
    def _row_to_draw(row: sqlite3.Row) -> dict:
        return {
            "period": row["period"],
            "date": row["draw_date"],
            "front": json.loads(row["front"]),
            "back": json.loads(row["back"]),
        }

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "target_period": row["target_period"],
            "saved_at": row["saved_at"],
            "marks": json.loads(row["marks_json"] or "{}"),
            "params": json.loads(row["params_json"] or "{}"),
            "tickets": json.loads(row["tickets_json"] or "[]"),
            "pool": json.loads(row["pool_json"] or "{}"),
            "note": row["note"] or "",
            "hit_front": row["hit_front"],
            "hit_back": row["hit_back"],
            "compared_at": row["compared_at"],
        }
