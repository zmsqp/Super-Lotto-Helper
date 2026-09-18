# -*- coding: utf-8 -*-
"""时间序列号码评分、232 缩号、覆盖组合优化与开奖对照。"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Callable

from services.market import fit_crowd_model, load_market_validation
from services.stats import (
    BACK_K,
    BACK_N,
    FRONT_K,
    FRONT_N,
    combo_quality,
    fair_pool_coverage,
    odd_even,
    pattern_stats,
    prize_grade,
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
FOCUS_BOOST = 0.25
MODEL_VERSION = "ensemble-coverage-v2.2"
_REFERENCE_CACHE: dict[tuple, dict] = {}


def _rank01(values: list[float], nmax: int) -> list[float]:
    """转成 0..1 排名分，避免不同统计量的量纲互相吞没。"""
    order = list(range(1, nmax + 1))
    order.sort(key=lambda n: (values[n], -n), reverse=True)
    out = [0.0] * (nmax + 1)
    denominator = max(nmax - 1, 1)
    for rank, number in enumerate(order):
        out[number] = 1.0 - rank / denominator
    return out


def _frequency_rate(draws: list[dict], key: str, nmax: int, window: int, prior: float) -> list[float]:
    use = slice_draws(draws, window)
    picks = FRONT_K if key == "front" else BACK_K
    expected = picks / nmax
    counts = [0] * (nmax + 1)
    for draw in use:
        for number in draw[key]:
            counts[number] += 1
    return [0.0] + [
        (counts[number] + prior * expected) / (len(use) + prior)
        for number in range(1, nmax + 1)
    ]


def ensemble_score_vector(draws: list[dict], key: str, nmax: int) -> tuple[list[float], dict]:
    """多时间尺度贝叶斯收缩与趋势的确定性集成；只读取目标期以前数据。"""
    if not draws:
        return [0.0] * (nmax + 1), {"components": []}
    components: list[tuple[str, float, list[float]]] = []
    for window, prior, weight in ((30, 60.0, 0.12), (100, 100.0, 0.22), (300, 200.0, 0.24), (800, 400.0, 0.18), (0, 800.0, 0.10)):
        values = _frequency_rate(draws, key, nmax, window, prior)
        components.append((f"bayes_{window or 'all'}", weight, _rank01(values, nmax)))

    recent = _frequency_rate(draws, key, nmax, 30, 60.0)
    medium = _frequency_rate(draws, key, nmax, 120, 120.0)
    trend = [0.0] + [recent[n] - medium[n] for n in range(1, nmax + 1)]
    components.append(("dynamic_30_120", 0.08, _rank01(trend, nmax)))

    ewma = score_vector(draws, key, nmax, "ewma50", 0)
    components.append(("ewma50", 0.06, _rank01(ewma, nmax)))

    scores = [0.0] * (nmax + 1)
    for _name, weight, ranked in components:
        for number in range(1, nmax + 1):
            scores[number] += weight * ranked[number]
    return scores, {
        "version": MODEL_VERSION,
        "components": [{"name": name, "weight": weight} for name, weight, _ in components],
    }


def model_score_vector(draws: list[dict], key: str, nmax: int, mode: str, window: int) -> tuple[list[float], dict]:
    if mode == "ensemble":
        return ensemble_score_vector(draws, key, nmax)
    return score_vector(draws, key, nmax, mode, window), {"version": "legacy", "components": [{"name": mode, "weight": 1.0}]}


def _pair_residuals(draws: list[dict], key: str, nmax: int, window: int = 300) -> dict[tuple[int, int], float]:
    use = slice_draws(draws, window)
    picks = FRONT_K if key == "front" else BACK_K
    fair = picks * (picks - 1) / (nmax * (nmax - 1))
    counts: Counter = Counter()
    for draw in use:
        counts.update(combinations(sorted(draw[key]), 2))
    alpha = max(len(use), 1)
    out = {}
    for pair in combinations(range(1, nmax + 1), 2):
        posterior = (counts[pair] + alpha * fair) / (len(use) + alpha)
        out[pair] = max(-1.0, min(1.0, math.log(max(posterior, 1e-12) / fair)))
    return out


def _pair_score(numbers: list[int], residuals: dict[tuple[int, int], float]) -> float:
    return sum(residuals.get(tuple(sorted(pair)), 0.0) for pair in combinations(numbers, 2))


def _popularity_proxy(numbers: list[int]) -> float:
    """购彩人气启发式，只用于同分组合排序，不代表开奖概率。"""
    ordered = sorted(numbers)
    score = sum(0.45 for number in ordered if number <= 31)
    score += sum(0.75 for number in ordered if number in {6, 8, 9, 18, 28})
    score += sum(0.35 for number in ordered if number % 5 == 0)
    score += sum(1.0 for left, right in zip(ordered, ordered[1:]) if right - left == 1)
    score += sum(0.65 for left, right in combinations(ordered, 2) if left % 10 == right % 10)
    if len(ordered) >= 3 and len(set(b - a for a, b in zip(ordered, ordered[1:]))) == 1:
        score += 0.8
    return score


def walk_forward_reference(draws: list[dict], key: str, nmax: int, pool_size: int, horizon: int = 500) -> dict:
    """逐期事前评价集成号码池，不拿目标期开奖参与该期打分。"""
    cache_key = (
        draws[-1].get("period", "") if draws else "",
        len(draws),
        key,
        nmax,
        pool_size,
        horizon,
    )
    if cache_key in _REFERENCE_CACHE:
        return dict(_REFERENCE_CACHE[cache_key])
    picks = FRONT_K if key == "front" else BACK_K
    minimum = min(300, max(30, len(draws) // 3))
    start = max(minimum, len(draws) - horizon)
    hit_counts = []
    top_pick_hits = []
    for target in range(start, len(draws)):
        scores, _ = ensemble_score_vector(draws[:target], key, nmax)
        ranked = sorted(range(1, nmax + 1), key=lambda n: (scores[n], -n), reverse=True)
        actual = set(draws[target][key])
        hit_counts.append(len(set(ranked[:pool_size]) & actual))
        top_pick_hits.append(len(set(ranked[:picks]) & actual))
    at_least = 2
    covered = sum(value >= at_least for value in hit_counts)
    fair = fair_pool_coverage(pool_size, nmax, picks, at_least)
    result = {
        "draws": len(hit_counts),
        "pool_size": pool_size,
        "covered_2plus": covered,
        "coverage_rate": covered / len(hit_counts) if hit_counts else 0.0,
        "fair_rate": fair,
        "mean_pool_hits": sum(hit_counts) / len(hit_counts) if hit_counts else 0.0,
        "mean_top_pick_hits": sum(top_pick_hits) / len(top_pick_hits) if top_pick_hits else 0.0,
        "fair_top_pick_mean": picks * picks / nmax,
        "beats_fair": (covered / len(hit_counts) > fair) if hit_counts else False,
    }
    _REFERENCE_CACHE[cache_key] = dict(result)
    return result


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
    # 标注是偏好，不是预测证据；只做温和加权，避免用户点一个号就强行变成“金胆”。
    boost = max(span * 0.35, FOCUS_BOOST)
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
    pair_residuals: dict[tuple[int, int], float] | None = None,
    popularity_scorer: Callable[[list[int]], float] | None = None,
    popularity_weight: float = 0.0,
) -> list[list[int]]:
    """后区 2 码组合优化：模型分、轻量关系分和五注号码覆盖分开考虑。"""
    eligible = [i for i in eligible_nums(BACK_N, exclude) if back_scores[i] != float("-inf")]
    eligible.sort(key=lambda n: (back_scores[n], n in set(focus), -n), reverse=True)
    pool = eligible[: min(10, max(6, len(focus) + 5, count * 2))]
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
        relation = (pair_residuals or {}).get(tuple(back), 0.0)
        popularity = popularity_scorer(back) if popularity_scorer else 0.0
        scored.append((s + 0.12 * relation + 0.20 * fc, s, back, popularity))
    if not scored and odd_n >= 0:
        return _pick_backs(
            back_scores,
            exclude,
            focus,
            -1,
            count,
            pair_residuals,
            popularity_scorer,
            popularity_weight,
        )
    crowd_order = {
        tuple(row[2]): rank / max(len(scored) - 1, 1)
        for rank, row in enumerate(sorted(scored, key=lambda item: item[3]))
    }
    scored = [
        (
            row[0] - 0.8 * popularity_weight * crowd_order[tuple(row[2])],
            row[1],
            row[2],
            row[3],
        )
        for row in scored
    ]
    scored.sort(key=lambda x: (x[0], x[1], -x[3], tuple(-n for n in x[2])), reverse=True)
    out = []
    used: Counter = Counter()
    while scored and len(out) < count:
        best = max(
            scored,
            key=lambda row: (
                row[0] + 0.65 * sum(used[n] == 0 for n in row[2]) - 0.30 * sum(used[n] for n in row[2]),
                row[1],
                tuple(-n for n in row[2]),
            ),
        )
        out.append(best[2])
        used.update(best[2])
        scored.remove(best)
    return out


def _select_diverse_fronts(
    candidates: list[list[int]],
    scores: list[float],
    count: int,
    residuals: dict[tuple[int, int], float],
    preferred_shape: tuple[int, int, int] | None = None,
    preferred_odd: int = -1,
    tier_scores: list[float] | None = None,
    popularity_scorer: Callable[[list[int]], float] | None = None,
    popularity_weight: float = 0.0,
) -> list[tuple[list[int], str, float, float]]:
    """优先覆盖尚未覆盖的三元组，避免推荐列表只是同一注换一个号码。"""
    unique = []
    seen = set()
    for front in candidates:
        key = tuple(sorted(front))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    # 分层必须来自未叠加人工关注/排除的模型分，人工标注只影响组合选择。
    ranking_scores = tier_scores if tier_scores is not None else scores
    ranked_numbers = sorted(range(1, FRONT_N + 1), key=lambda n: (ranking_scores[n], -n), reverse=True)
    high = set(ranked_numbers[:10])
    low = set(ranked_numbers[10:25])
    extreme_cold = set(ranked_numbers[-10:])
    popularity_by_front = {
        tuple(front): (
            popularity_scorer(front) if popularity_scorer else _popularity_proxy(front)
        )
        for front in unique
    }
    popularity_rank = {
        key: rank / max(len(unique) - 1, 1)
        for rank, (key, _value) in enumerate(
            sorted(popularity_by_front.items(), key=lambda item: item[1])
        )
    }
    selected: list[list[int]] = []
    covered_triples: set[tuple[int, int, int]] = set()
    usage: Counter = Counter()
    result = []
    for index in range(count):
        experimental = count >= 5 and index >= count - 2
        best = None
        for front in unique:
            if front in selected:
                continue
            # 五注不能把任一号码全部绑死；至少留一注承担相反结果。
            if selected and any(usage[number] >= count - 1 for number in front):
                continue
            triples = set(combinations(front, 3))
            new_triples = len(triples - covered_triples)
            overlap = max((len(set(front) & set(old)) for old in selected), default=0)
            model_score = sum(scores[number] for number in front) + 0.08 * _pair_score(front, residuals)
            cold_count = len(set(front) & extreme_cold)
            tier_pattern = (
                len(set(front) & high),
                len(set(front) & low),
                cold_count,
            )
            popularity = popularity_by_front[tuple(front)]
            crowd_penalty = popularity_rank[tuple(front)]
            utility = model_score + 0.10 * new_triples - 0.28 * overlap
            utility += 0.16 * sum(usage[number] == 0 for number in front)
            # 人气层只在开奖评分接近时轻量择优，不把低人气解释成更易开奖。
            utility -= popularity_weight * crowd_penalty
            if preferred_shape is not None and zone3(front) == preferred_shape:
                utility += 0.24
            if preferred_odd < 0 or odd_even(front)[0] == preferred_odd:
                utility += 0.12
            if experimental:
                # 重点检验 3高+1低+1极冷；这是实验结构，不是假定有效的规律。
                distance = sum(abs(actual - expected) for actual, expected in zip(tier_pattern, (3, 1, 1)))
                structure_bonus = 0.55 if tier_pattern == (3, 1, 1) else -0.08 * distance
                utility += structure_bonus + 0.20 * min(cold_count, 2) - 0.08 * crowd_penalty
            key = (utility, model_score, -popularity, tuple(-number for number in front))
            if best is None or key > best[0]:
                best = (key, front, model_score, popularity)
        if best is None:
            # 极端筛选条件下再放宽，确保仍能生成请求的注数。
            remaining = [front for front in unique if front not in selected]
            if not remaining:
                break
            front = remaining[0]
            popularity = popularity_by_front.get(
                tuple(front), popularity_scorer(front) if popularity_scorer else _popularity_proxy(front)
            )
            best = ((0.0,), front, sum(scores[number] for number in front), popularity)
        _, front, model_score, popularity = best
        selected.append(front)
        covered_triples.update(combinations(front, 3))
        usage.update(front)
        selected_pattern = (
            len(set(front) & high),
            len(set(front) & low),
            len(set(front) & extreme_cold),
        )
        if experimental and selected_pattern == (3, 1, 1):
            role = "3高+1低+1极冷实验"
        elif experimental and selected_pattern[2] > 0:
            role = "低人气冷门实验（结构放宽）"
        elif experimental:
            role = "覆盖优化（冷门池不足）"
        else:
            role = "覆盖优化"
        result.append((front, role, model_score, popularity))
    return result


def generate_recommendations(
    draws: list[dict],
    marks: dict,
    window: int = 100,
    mode: str = "ensemble",
    shape_key: str = "221",
    use_duplex_232: bool = True,
    front_odd: int = 3,
    back_odd: int = 1,
    ticket_count: int = 5,
    constrain_sum: bool = True,
    market_rows: list[dict] | None = None,
    crowd_weight: float = 0.20,
    include_reference: bool = True,
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

    raw_fs, front_model = model_score_vector(draws, "front", FRONT_N, mode, window)
    raw_bs, back_model = model_score_vector(draws, "back", BACK_N, mode, window)
    fs = apply_marks(raw_fs, FRONT_N, front_focus, front_excl)
    bs = apply_marks(raw_bs, BACK_N, back_focus, back_excl)

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
    back_ref = sorted(
        (i for i in range(1, BACK_N + 1) if bs[i] != float("-inf")),
        key=lambda n: (bs[n], -n),
        reverse=True,
    )[:2] or [1, 2]
    crowd_model = fit_crowd_model(draws, market_rows or []) if market_rows else None
    effective_crowd_weight = max(0.0, float(crowd_weight)) if crowd_model else 0.0
    front_popularity_scorer = (
        (lambda front: crowd_model.score(front, back_ref, draws)) if crowd_model else None
    )
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
        constraints_relaxed = []
        if len(fronts) < ticket_count:
            constraints_relaxed.append("前区奇偶")
            fronts = _enumerate_fronts(
                src1, src2, src3, shape, -1, front_focus, stats, back_ref,
                sum_lo=enum_lo, sum_hi=enum_hi,
            )
        # 232 池内始终纳入全部 5 码子集；选定形态作为偏好而非硬锁，避免某一区两码绑死五注。
        extra = [sorted(cand) for cand in combinations(sorted(duplex), FRONT_K)]
        extra.sort(
            key=lambda f: (_focus_count(f, front_focus), combo_quality(f, back_ref, stats)),
            reverse=True,
        )
        seen = {tuple(x) for x in fronts}
        for f in extra:
            if tuple(f) in seen:
                continue
            fronts.append(f)
            seen.add(tuple(f))
        pool_note = "232复式池"
        front_pool = duplex
    else:
        constraints_relaxed = []
        # 单式形态：各区多取几个高分号再组合
        src1 = ranked_in_zone(fs, z1, max(shape[0] + 3, 4))
        src2 = ranked_in_zone(fs, z2, max(shape[1] + 3, 4))
        src3 = ranked_in_zone(fs, z3, max(shape[2] + 3, 4))
        fronts = _enumerate_fronts(
            src1, src2, src3, shape, front_odd, front_focus, stats, back_ref,
            sum_lo=enum_lo, sum_hi=enum_hi,
        )
        if len(fronts) < ticket_count:
            constraints_relaxed.append("前区奇偶")
            fronts = _enumerate_fronts(
                src1, src2, src3, shape, -1, front_focus, stats, back_ref,
                sum_lo=enum_lo, sum_hi=enum_hi,
            )
        pool_note = f"单式形态{shape_key}"
        front_pool = sorted(set(src1 + src2 + src3))

    front_pairs = _pair_residuals(draws, "front", FRONT_N)
    back_pairs = _pair_residuals(draws, "back", BACK_N)
    n = min(max(ticket_count, 3), 5)
    selected_fronts = _select_diverse_fronts(
        fronts,
        fs,
        n,
        front_pairs,
        shape,
        front_odd,
        tier_scores=raw_fs,
        popularity_scorer=front_popularity_scorer,
        popularity_weight=effective_crowd_weight,
    )
    front_anchor = sorted(eligible_f, key=lambda number: (fs[number], -number), reverse=True)[:5]
    back_popularity_scorer = (
        (lambda back: crowd_model.score(front_anchor, back, draws)) if crowd_model else None
    )
    backs = _pick_backs(
        bs,
        back_excl,
        back_focus,
        back_odd,
        n,
        back_pairs,
        back_popularity_scorer,
        effective_crowd_weight,
    )
    if not backs:
        backs = [[1, 2]]
    back_pool = sorted({n for b in backs for n in b})
    # 后区关注池再补几个高分
    extra_b = ranked_in_zone(bs, eligible_nums(BACK_N, back_excl), 4)
    back_pool = sorted(set(back_pool) | set(extra_b))

    tickets = []
    sum_std = float(stats.get("front_sum_std") or 1.0)
    tier_order = sorted(
        range(1, FRONT_N + 1),
        key=lambda number: (raw_fs[number], -number),
        reverse=True,
    )
    high_tier = set(tier_order[:10])
    low_tier = set(tier_order[10:25])
    cold_tier = set(tier_order[25:])
    for i in range(n):
        if i < len(selected_fronts):
            front, role, draw_score, popularity = selected_fronts[i]
        else:
            front = fronts[i] if i < len(fronts) else (fronts[0] if fronts else sorted(eligible_f[:5]))
            role, draw_score, popularity = "覆盖优化", sum(fs[number] for number in front), _popularity_proxy(front)
        back = backs[i % len(backs)]
        fsum = sum(front)
        bsum = sum(back)
        tier_pattern = [
            len(set(front) & high_tier),
            len(set(front) & low_tier),
            len(set(front) & cold_tier),
        ]
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
                "role": role,
                "draw_score": round(float(draw_score), 4),
                "popularity_proxy": round(float(popularity), 3),
                "tier_pattern": tier_pattern,
            }
        )

    if crowd_model and tickets:
        crowd_values = [
            crowd_model.expected_first_winners_per_100m(
                ticket["front"], ticket["back"], draws
            )
            for ticket in tickets
        ]
        order = sorted(range(len(tickets)), key=lambda index: crowd_values[index])
        labels = ["中"] * len(tickets)
        low_count = max(1, len(tickets) // 3)
        high_count = max(1, len(tickets) // 3)
        for index in order[:low_count]:
            labels[index] = "低"
        for index in order[-high_count:]:
            labels[index] = "高"
        for index, ticket in enumerate(tickets):
            ticket["crowd_score"] = round(crowd_values[index], 3)
            ticket["crowd_level"] = labels[index]
            ticket["crowd_basis"] = "官方销量与一等奖注数"
            ticket["popularity_proxy"] = round(crowd_values[index], 3)
    else:
        for ticket in tickets:
            ticket["crowd_score"] = ticket.get("popularity_proxy", 0.0)
            ticket["crowd_level"] = "启发式"
            ticket["crowd_basis"] = "号码形态代理"

    if any(zone3(ticket["front"]) != shape for ticket in tickets):
        constraints_relaxed.append("部分单式形态（用于减少五注绑号）")
    if front_odd >= 0 and any(odd_even(ticket["front"])[0] != front_odd for ticket in tickets):
        if "前区奇偶" not in constraints_relaxed:
            constraints_relaxed.append("前区奇偶")

    front_reference = (
        walk_forward_reference(draws, "front", FRONT_N, len(front_pool))
        if include_reference
        else {}
    )
    back_reference = (
        walk_forward_reference(draws, "back", BACK_N, len(back_pool))
        if include_reference
        else {}
    )
    front_order = sorted(
        range(1, FRONT_N + 1),
        key=lambda number: (raw_fs[number], -number),
        reverse=True,
    )
    back_order = sorted(
        range(1, BACK_N + 1),
        key=lambda number: (raw_bs[number], -number),
        reverse=True,
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
        "model_version": MODEL_VERSION if mode == "ensemble" else "legacy",
        "data_cutoff_period": draws[-1].get("period", ""),
        "data_cutoff_date": draws[-1].get("date", ""),
        "constraints_relaxed": constraints_relaxed,
        "crowd_weight": effective_crowd_weight,
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
        "model": {
            "version": MODEL_VERSION if mode == "ensemble" else "legacy",
            "front": front_model,
            "back": back_model,
            "front_ranking": front_order,
            "back_ranking": back_order,
            "front_reference": front_reference,
            "back_reference": back_reference,
            "crowd": (
                {**crowd_model.detail(), "validation": load_market_validation()}
                if crowd_model
                else {
                    "version": "heuristic-only",
                    "samples": 0,
                    "reason": "官方销量样本不足",
                }
            ),
            "warning": "回测覆盖率不是单号概率；低人气分不改变开奖概率。",
        },
    }


def compare_tickets(tickets: list[dict], draw: dict) -> tuple[list[dict], int, int]:
    """对照开奖，返回逐注结果及五注中表现最佳的同一注命中。"""
    actual_f = set(draw["front"])
    actual_b = set(draw["back"])
    out = []
    for t in tickets:
        hf = len(set(t.get("front") or []) & actual_f)
        hb = len(set(t.get("back") or []) & actual_b)
        item = dict(t)
        item["hit_front"] = hf
        item["hit_back"] = hb
        item["grade"] = prize_grade(hf, hb)
        out.append(item)
    if not out:
        return out, 0, 0

    def quality(item: dict) -> tuple:
        grade = item.get("grade")
        return (grade is not None, -(grade or 99), item["hit_front"], item["hit_back"])

    best = max(out, key=quality)
    return out, int(best["hit_front"]), int(best["hit_back"])


def comparison_summary(tickets: list[dict], draw: dict, pool: dict | None = None) -> dict:
    """供日志页使用的组合级对照；池命中与单注命中明确分开。"""
    compared, best_front, best_back = compare_tickets(tickets, draw)
    front_union = set((pool or {}).get("front_pool") or [])
    back_union = set((pool or {}).get("back_pool") or [])
    if not front_union:
        front_union = set().union(*(set(ticket.get("front") or []) for ticket in compared)) if compared else set()
    if not back_union:
        back_union = set().union(*(set(ticket.get("back") or []) for ticket in compared)) if compared else set()
    prizes = [ticket for ticket in compared if ticket.get("grade") is not None]
    best_grade = min((ticket["grade"] for ticket in prizes), default=None)
    return {
        "tickets": compared,
        "best_front": best_front,
        "best_back": best_back,
        "best_grade": best_grade,
        "winning_tickets": len(prizes),
        "front_pool_hits": len(front_union & set(draw["front"])),
        "back_pool_hits": len(back_union & set(draw["back"])),
        "front_pool_size": len(front_union),
        "back_pool_size": len(back_union),
        "fair_single_mean_front": FRONT_K * FRONT_K / FRONT_N,
        "fair_single_mean_back": BACK_K * BACK_K / BACK_N,
    }
