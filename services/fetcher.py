# -*- coding: utf-8 -*-
"""开奖增量抓取：体彩 API 优先，失败回退 500.com。"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

SPORTTERY_URL_TMPL = (
    "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
    "?gameNo=85&provinceId=0&pageSize={page_size}&isVerify=1&pageNo={page_no}"
)
SOURCE_500_TMPL = "https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _http_get(url: str, extra_headers: Optional[dict] = None, timeout: int = 20) -> bytes:
    headers = {"User-Agent": _UA, "Accept": "*/*"}
    if extra_headers:
        headers.update(extra_headers)
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_sporttery_json(text: str) -> list[dict]:
    """解析体彩 getHistoryPageListV1 返回。"""
    payload = json.loads(text)
    value = payload.get("value") or {}
    items = value.get("list") or []
    rows = []
    for it in items:
        period = str(it.get("lotteryDrawNum") or "").strip()
        date = str(it.get("lotteryDrawTime") or "").strip()[:10]
        result = str(it.get("lotteryDrawResult") or "").strip()
        nums = [int(x) for x in re.split(r"[\s,]+", result) if x.isdigit()]
        if not period or len(period) < 5 or len(nums) < 7:
            continue
        front = sorted(nums[:5])
        back = sorted(nums[5:7])
        if min(front) < 1 or max(front) > 35 or min(back) < 1 or max(back) > 12:
            continue
        if not date:
            continue
        prize_counts = {}
        prize_amounts = {}
        for prize in it.get("prizeLevelList") or []:
            group = str(prize.get("group") or "").strip()
            if not group:
                continue
            try:
                prize_counts[group] = int(str(prize.get("stakeCount") or "0").replace(",", ""))
            except ValueError:
                prize_counts[group] = 0
            try:
                prize_amounts[group] = int(
                    str(prize.get("stakeAmountFormat") or "0").replace(",", "")
                )
            except ValueError:
                prize_amounts[group] = 0
        try:
            sales = int(str(it.get("totalSaleAmount") or "0").replace(",", ""))
        except ValueError:
            sales = 0
        try:
            pool_balance = float(str(it.get("poolBalanceAfterdraw") or "0").replace(",", ""))
        except ValueError:
            pool_balance = 0.0
        rows.append(
            {
                "period": period,
                "date": date,
                "front": front,
                "back": back,
                "sales": sales,
                "pool_balance": pool_balance,
                "prize_counts": prize_counts,
                "prize_amounts": prize_amounts,
            }
        )
    rows.sort(key=lambda r: (r["date"], r["period"]))
    return _dedupe(rows)


def parse_draws_500(html: str) -> list[dict]:
    """解析 500.com 历史表（与 analyze_dlt.parse_draws 一致）。"""
    m = re.search(r'<tbody[^>]*id="tdata"[\s\S]*?</tbody>', html, re.I)
    if not m:
        raise RuntimeError("未找到开奖表 tdata")
    rows = []
    for tr in re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", m.group(0), re.I):
        tds = [re.sub(r"<[^>]+>", "", td).strip() for td in re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr, re.I)]
        if len(tds) < 9:
            continue
        if tds[1].isdigit() and len(tds[1]) >= 5:
            period, nums = tds[1], tds[2:]
        elif tds[0].isdigit() and len(tds[0]) >= 5:
            period, nums = tds[0], tds[1:]
        else:
            continue
        try:
            front = sorted(int(x) for x in nums[0:5])
            back = sorted(int(x) for x in nums[5:7])
        except ValueError:
            continue
        if len(front) != 5 or len(back) != 2:
            continue
        if min(front) < 1 or max(front) > 35 or min(back) < 1 or max(back) > 12:
            continue
        date = nums[-1] if nums else ""
        rows.append({"period": period, "date": date, "front": front, "back": back})
    rows.sort(key=lambda r: (r["date"], r["period"]))
    return _dedupe(rows)


def _dedupe(rows: list[dict]) -> list[dict]:
    uniq = []
    seen = set()
    for r in rows:
        if r["period"] in seen:
            continue
        seen.add(r["period"])
        uniq.append(r)
    return uniq


def fetch_sporttery(page_size: int = 100, page_no: int = 1) -> list[dict]:
    raw = _http_get(
        SPORTTERY_URL_TMPL.format(page_size=page_size, page_no=page_no),
        extra_headers={
            "Referer": "https://www.lottery.gov.cn/",
            "Accept": "application/json, text/plain, */*",
        },
    )
    text = raw.decode("utf-8", errors="replace")
    rows = parse_sporttery_json(text)
    if not rows:
        raise RuntimeError("体彩接口未解析到开奖记录")
    return rows


def fetch_sporttery_market_history(max_pages: int = 20) -> list[dict]:
    """抓取官方仍提供销量与奖级注数的完整历史段。"""
    rows = []
    empty_market_pages = 0
    for page_no in range(1, max_pages + 1):
        page = fetch_sporttery(page_size=100, page_no=page_no)
        market_page = [row for row in page if row.get("sales") and row.get("prize_counts")]
        if not market_page:
            empty_market_pages += 1
            if empty_market_pages >= 1:
                break
            continue
        empty_market_pages = 0
        rows.extend(market_page)
    rows.sort(key=lambda row: (row["date"], row["period"]))
    return _dedupe(rows)


def fetch_500(start: str, end: str) -> list[dict]:
    url = SOURCE_500_TMPL.format(start=start, end=end)
    raw = _http_get(url)
    text = None
    for enc in ("gb18030", "gbk", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    rows = parse_draws_500(text)
    if not rows:
        raise RuntimeError("500.com 未解析到开奖记录")
    return rows


def _period_range(last_period: Optional[str]) -> tuple[str, str]:
    """构造 500.com 增量查询区间。"""
    if last_period and last_period.isdigit():
        p = int(last_period)
        start = f"{max(p - 40, 7001):05d}"
        end = f"{p + 80:05d}"
        return start, end
    return "07001", "26120"


def fetch_incremental(last_period: Optional[str] = None) -> dict:
    """
    后台线程调用：只抓取，不写库。
    返回 {source, draws, error}。
    """
    errors = []
    try:
        draws = fetch_sporttery()
        return {"source": "sporttery", "draws": draws, "error": ""}
    except (URLError, HTTPError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"体彩接口：{exc}")
    start, end = _period_range(last_period)
    try:
        draws = fetch_500(start, end)
        return {"source": "500.com", "draws": draws, "error": ""}
    except (URLError, HTTPError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        errors.append(f"500.com：{exc}")
    return {"source": "", "draws": [], "error": "；".join(errors)}
