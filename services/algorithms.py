# -*- coding: utf-8 -*-
"""232 三区、奇偶、冷热遗漏缩号与推荐生成。"""
from __future__ import annotations

from itertools import combinations

from services.stats import (
    BACK_K,
    BACK_N,
    FRONT_K,
    FRONT_N,
    combo_quality,
    odd_even,
    pattern_stats,
    score_vector,
    slice_draws,
    sum_band,
    sum_comment,
    zone3,
    zone3_label,
)

# 前区三区：01-12 / 13-24 / 25-35
ZONE_RANGES = ((1, 12), (13, 24), (25, 35))
# 单式 5 码主流形态
SINGLE_SHAPES = {
    "221": (2, 2, 1),
    "212": (2, 1, 2),
    "122": (1, 2, 2),
    "131": (1, 3, 1),
    "113": (1, 1, 3),
}
# 232 复式池：一区 2 + 二区 3 + 三区 2
DUPLEX_232 = (2, 3, 2)
FOCUS_BOOST = 2.0


def zone_index(num: int) -> int:
    """号码所属三区：0/1/2。"""
    if num <= 12:
        return 0
    if num <= 24:
        return 1
    return 2


def split_zones(nums: list[int]) -> tuple[list[int], list[int], list[int]]:
    buckets: list[list[int]] = [[], [], []]
    for n in nums:
        buckets[zone_index(n)].append(n)
    return buckets[0], buckets[1], buckets[2]


def apply_marks(scores: list[float], nmax: int, focus: list[int], exclude: list[int]) -> list[float]:
    """排除号打到 -inf，关注号按分数跨度加权，确保排序靠前。"""
    out = list(scores)
    excl = set(int(x) for x in exclude)
    foc = set(int(x) for x in focus)
    finite = [out[i] for i in range(1, nmax + 1)]
    span = (max(finite) - min(finite)) if finite else 1.0
    boost = max(span * 0.6, FOCUS_BOOST)
    for i in range(1, nmax + 1):
        if i in excl:
            out[i] = float("-inf")
        elif i in foc:
            out[i] += boost
    return out


def pick_dans(
    scores: list[float],
    nmax: int,
    exclude: list[int],
    focus: list[int],
    gold_n: int = 1,
    silver_n: int = 2,
    bronze_n: int = 2,
) -> dict:
    """
    按分数与关注号分层：金胆 / 银胆 / 铜胆。
    用户标注的关注号优先进入金、银。
    """
    eligible = [i for i in eligible_nums(nmax, exclude) if scores[i] != float("-inf")]
    foc = set(int(x) for x in focus)
    eligible.sort(key=lambda n: (n in foc, scores[n], -n), reverse=True)
    gold = eligible[:gold_n]
    silver = eligible[gold_n : gold_n + silver_n]
    bronze = eligible[gold_n + silver_n : gold_n + silver_n + bronze_n]
    return {"gold": gold, "silver": silver, "bronze": bronze}


def eligible_nums(nmax: int, exclude: list[int]) -> list[int]:
    excl = set(int(x) for x in exclude)
    return [i for i in range(1, nmax + 1) if i not in excl]


def ranked_in_zone(scores: list[float], zone_nums: list[int], k: int) -> list[int]:
    """区内按分数取前 k，保持分数降序。"""
    cands = [n for n in zone_nums if scores[n] != float("-inf")]
    cands.sort(key=lambda n: (scores[n], -n), reverse=True)
    return cands[: max(0, k)]


def duplex_232_pool(front_scores: list[float], exclude: list[int]) -> list[int]:
    """232 复式关注池：三区各取 2/3/2，共 7 码。"""
    eligible = eligible_nums(FRONT_N, exclude)
    z1, z2, z3 = split_zones(eligible)
    picks = (
        ranked_in_zone(front_scores, z1, DUPLEX_232[0])
        + ranked_in_zone(front_scores, z2, DUPLEX_232[1])
        + ranked_in_zone(front_scores, z3, DUPLEX_232[2])
    )
    return sorted(picks)


def _shape_ok(front: list[int], shape: tuple[int, int, int]) -> bool:
    return zone3(front) == shape


def _oe_ok(nums: list[int], odd_n: int) -> bool:
    return odd_even(nums)[0] == odd_n


def _focus_count(nums: list[int], focus: list[int]) -> int:
    return len(set(nums) & set(focus))


def _enumerate_fronts(
    z1: list[int],
    z2: list[int],
    z3: list[int],
    shape: tuple[int, int, int],
    odd_n: int,
    focus: list[int],
    stats: dict,
    back_ref: list[int],
    limit: int = 80,
    sum_lo: int | None = None,
    sum_hi: int | None = None,
) -> list[list[int]]:
    """按三区个数枚举前区 5 码，再按和值区间、关注数与形态分排序。"""
    a, b, c = shape
    if len(z1) < a or len(z2) < b or len(z3) < c:
        return []
    scored = []
    for c1 in combinations(z1, a):
        for c2 in combinations(z2, b):
            for c3 in combinations(z3, c):
                front = sorted(c1 + c2 + c3)
                if odd_n >= 0 and not _oe_ok(front, odd_n):
                    continue
                s = sum(front)
                in_band = 1
                if sum_lo is not None and sum_hi is not None:
                    in_band = 1 if sum_lo <= s <= sum_hi else 0
                fc = _focus_count(front, focus)
                q = combo_quality(front, back_ref, stats)
                scored.append((in_band, fc, q, front))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    uniq = []
    seen = set()
    for _band, fc, q, front in scored:
        key = tuple(front)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(front)
        if len(uniq) >= limit:
            break
    return uniq


def _pick_backs(
    back_scores: list[float],
    exclude: list[int],
    focus: list[int],
    odd_n: int,
    count: int,
) -> list[list[int]]:
    """后区 2 码，尽量 1:1，关注号优先。"""
    eligible = [i for i in eligible_nums(BACK_N, exclude) if back_scores[i] != float("-inf")]
    eligible.sort(key=lambda n: (n in set(focus), back_scores[n]), reverse=True)
    pool = eligible[: max(6, len(focus) + 3)]
    if len(pool) < 2:
        pool = eligible
    if len(pool) < 2:
        return []
    scored = []
    foc = set(focus)
    for cand in combinations(pool, 2):
        back = sorted(cand)
        if odd_n >= 0 and not _oe_ok(back, odd_n):
            continue
        fc = len(set(back) & foc)
        s = back_scores[back[0]] + back_scores[back[1]]
        scored.append((fc, s, back))
    if not scored and odd_n >= 0:
        return _pick_backs(back_scores, exclude, focus, -1, count)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = []
    seen = set()
    for _, _, back in scored:
        key = tuple(back)
        if key in seen:
            continue
        seen.add(key)
        out.append(back)
        if len(out) >= count:
            break
    return out


def generate_recommendations(
    draws: list[dict],
    marks: dict,
    window: int = 100,
    mode: str = "blend",
    shape_key: str = "221",
    use_duplex_232: bool = True,
    front_odd: int = 3,
    back_odd: int = 1,
    ticket_count: int = 5,
    constrain_sum: bool = True,
) -> dict:
    """
    生成关注池与 3～5 注单式。
    front_odd / back_odd 为奇数个数；负数表示不约束。
    """
    use = slice_draws(draws, window)
    stats = pattern_stats(use if len(use) >= 30 else draws)
    front_focus = list((marks.get("front") or {}).get("focus") or [])
    front_excl = list((marks.get("front") or {}).get("exclude") or [])
    back_focus = list((marks.get("back") or {}).get("focus") or [])
    back_excl = list((marks.get("back") or {}).get("exclude") or [])

    fs = apply_marks(score_vector(draws, "front", FRONT_N, mode, window), FRONT_N, front_focus, front_excl)
    bs = apply_marks(score_vector(draws, "back", BACK_N, mode, window), BACK_N, back_focus, back_excl)

    eligible_f = eligible_nums(FRONT_N, front_excl)
    if len(eligible_f) < FRONT_K:
        return {
            "front_pool": [],
            "back_pool": [],
            "duplex_232": [],
            "tickets": [],
            "params": {},
            "pool_note": "",
            "error": "排除号过多，前区可选票不足 5 个",
        }
    z1, z2, z3 = split_zones(eligible_f)
    duplex = duplex_232_pool(fs, front_excl)

    shape = SINGLE_SHAPES.get(shape_key, SINGLE_SHAPES["221"])
    back_ref = [i for i in range(1, BACK_N + 1) if bs[i] != float("-inf")][:2] or [1, 2]
    sum_lo, sum_hi, sum_mean = sum_band(stats, 1.0)
    enum_lo, enum_hi = (sum_lo, sum_hi) if constrain_sum else (None, None)

    front_dans = pick_dans(fs, FRONT_N, front_excl, front_focus, 1, 2, 2)
    back_dans = pick_dans(bs, BACK_N, back_excl, back_focus, 1, 1, 1)

    if use_duplex_232 and len(duplex) >= FRONT_K:
        src1, src2, src3 = split_zones(duplex)
        # 复式池不足目标形态时，用池内实际可组形态兜底
        fronts = _enumerate_fronts(
            src1, src2, src3, shape, front_odd, front_focus, stats, back_ref,
            sum_lo=enum_lo, sum_hi=enum_hi,
        )
        if len(fronts) < ticket_count:
            fronts = _enumerate_fronts(
                src1, src2, src3, shape, -1, front_focus, stats, back_ref,
                sum_lo=enum_lo, sum_hi=enum_hi,
            )
        if len(fronts) < ticket_count:
            extra = []
            for cand in combinations(sorted(duplex), FRONT_K):
                extra.append(sorted(cand))
            extra.sort(
                key=lambda f: (_focus_count(f, front_focus), combo_quality(f, back_ref, stats)),
                reverse=True,
            )
            seen = {tuple(x) for x in fronts}
            for f in extra:
                if tuple(f) in seen:
                    continue
                fronts.append(f)
                if len(fronts) >= ticket_count * 3:
                    break
        pool_note = "232复式池"
        front_pool = duplex
    else:
        # 单式形态：各区多取几个高分号再组合
        src1 = ranked_in_zone(fs, z1, max(shape[0] + 3, 4))
        src2 = ranked_in_zone(fs, z2, max(shape[1] + 3, 4))
        src3 = ranked_in_zone(fs, z3, max(shape[2] + 3, 4))
        fronts = _enumerate_fronts(
            src1, src2, src3, shape, front_odd, front_focus, stats, back_ref,
            sum_lo=enum_lo, sum_hi=enum_hi,
        )
        if len(fronts) < ticket_count:
            fronts = _enumerate_fronts(
                src1, src2, src3, shape, -1, front_focus, stats, back_ref,
                sum_lo=enum_lo, sum_hi=enum_hi,
            )
        pool_note = f"单式形态{shape_key}"
        front_pool = sorted(set(src1 + src2 + src3))

    backs = _pick_backs(bs, back_excl, back_focus, back_odd, max(ticket_count, 3))
    if not backs:
        backs = [[1, 2]]
    back_pool = sorted({n for b in backs for n in b})
    # 后区关注池再补几个高分
    extra_b = ranked_in_zone(bs, eligible_nums(BACK_N, back_excl), 4)
    back_pool = sorted(set(back_pool) | set(extra_b))

    tickets = []
    n = min(max(ticket_count, 3), 5)
    sum_std = float(stats.get("front_sum_std") or 1.0)
    for i in range(n):
        front = fronts[i] if i < len(fronts) else (fronts[0] if fronts else sorted(eligible_f[:5]))
        back = backs[i % len(backs)]
        fsum = sum(front)
        bsum = sum(back)
        tickets.append(
            {
                "name": f"第{i + 1}注",
                "front": list(front),
                "back": list(back),
                "shape": zone3_label(front),
                "odd_even": f"{odd_even(front)[0]}:{odd_even(front)[1]}",
                "back_oe": f"{odd_even(back)[0]}:{odd_even(back)[1]}",
                "front_sum": fsum,
                "back_sum": bsum,
                "sum_flag": sum_comment(fsum, sum_mean, sum_std),
                "note": pool_note,
            }
        )

    params = {
        "window": window,
        "mode": mode,
        "shape_key": shape_key,
        "use_duplex_232": use_duplex_232,
        "front_odd": front_odd,
        "back_odd": back_odd,
        "ticket_count": n,
        "constrain_sum": constrain_sum,
        "sum_lo": sum_lo,
        "sum_hi": sum_hi,
    }
    return {
        "front_pool": front_pool,
        "back_pool": back_pool,
        "duplex_232": duplex,
        "front_dans": front_dans,
        "back_dans": back_dans,
        "sum_mean": round(sum_mean, 1),
        "sum_lo": sum_lo,
        "sum_hi": sum_hi,
        "tickets": tickets,
        "params": params,
        "pool_note": pool_note,
    }


def compare_tickets(tickets: list[dict], draw: dict) -> tuple[list[dict], int, int]:
    """对照开奖，返回带命中的票、主推前区命中、主推后区命中。"""
    actual_f = set(draw["front"])
    actual_b = set(draw["back"])
    out = []
    for t in tickets:
        hf = len(set(t.get("front") or []) & actual_f)
        hb = len(set(t.get("back") or []) & actual_b)
        item = dict(t)
        item["hit_front"] = hf
        item["hit_back"] = hb
        out.append(item)
    main_f = out[0]["hit_front"] if out else 0
    main_b = out[0]["hit_back"] if out else 0
    return out, main_f, main_b
