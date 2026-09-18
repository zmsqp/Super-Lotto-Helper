# -*- coding: utf-8 -*-
"""刷新官方销量与奖级注数种子文件。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.fetcher import fetch_sporttery_market_history  # noqa: E402


def main() -> None:
    rows = fetch_sporttery_market_history(max_pages=20)
    if len(rows) < 500:
        raise RuntimeError(f"官方销量历史仅抓到{len(rows)}期，拒绝覆盖现有种子")
    payload = {
        "source": "中国体彩网官方历史开奖接口",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
    }
    output = ROOT / "assets" / "dlt_market_history.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"saved={len(rows)} range={rows[0]['period']}..{rows[-1]['period']} path={output}")


if __name__ == "__main__":
    main()
