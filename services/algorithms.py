# -*- coding: utf-8 -*-
"""时间序列号码评分、奇偶/形态/和值历史推理、7 码复式池、覆盖组合优化与开奖对照。"""
from __future__ import annotations

import math
import random
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
# 232 复式池的历史默认结构；勾选复式后实际个数改由历史推理决定。
DUPLEX_232 = (2, 3, 2)
FOCUS_BOOST = 0.25
MODEL_VERSION = "ensemble-coverage-v2.2"
_REFERENCE_CACHE: dict[tuple, dict] = {}

# 与号码集成共用的多窗口贝叶斯收缩：(窗口, 先验强度, 权重)。窗口 0 表示全量。
_ENSEMBLE_BAYES = (
    (30, 60.0, 0.12),
    (100, 100.0, 0.22),
    (300, 200.0, 0.24),
    (800, 400.0, 0.18),
    (0, 800.0, 0.10),
)
_ENSEMBLE_TREND_WEIGHT = 0.08
_ENSEMBLE_EWMA_WEIGHT = 0.06
# 奇偶哨兵：-2 按历史推理分配各注，-1 不限，>=0 为奇数个数硬约束。
ODD_EVEN_INFER = -2
ODD_EVEN_ANY = -1
# 单式形态哨兵：infer 按历史三区形态分配各注；其它键仍是手动形态。
SHAPE_INFER = "infer"
# 次高分低于最高分的该比例时视为可忽略，不再强行拆开最后一注。
_OE_RUNNER_UP_MIN_RATIO = 0.12
# 命中该注目标奇偶或目标形态的加分。高于同池号码分差，使分配跟得上。
_OE_TARGET_BONUS = 5.0
# 和值命中加分低于形态命中，避免两项抵消后五注又收成同一个形态。
_SUM_TARGET_BONUS = 3.0
# 前区和值分箱宽度。目标取箱中心，落在半宽内视为命中该注目标。
_SUM_MIN = sum(range(1, FRONT_K + 1))
_SUM_MAX = sum(range(FRONT_N - FRONT_K + 1, FRONT_N + 1))
_SUM_BIN = 5
_SUM_MATCH = _SUM_BIN // 2
# 公平基准用二项分布：前区 35 个号 18 奇，后区 12 个号 6 奇。
_POOL_ODD_RATE = {"front": 18 / 35, "back": 6 / 12}


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
    for window, prior, weight in _ENSEMBLE_BAYES:
        values = _frequency_rate(draws, key, nmax, window, prior)
        components.append((f"bayes_{window or 'all'}", weight, _rank01(values, nmax)))

    recent = _frequency_rate(draws, key, nmax, 30, 60.0)
    medium = _frequency_rate(draws, key, nmax, 120, 120.0)
    trend = [0.0] + [recent[n] - medium[n] for n in range(1, nmax + 1)]
    components.append(("dynamic_30_120", _ENSEMBLE_TREND_WEIGHT, _rank01(trend, nmax)))

    ewma = score_vector(draws, key, nmax, "ewma50", 0)
    components.append(("ewma50", _ENSEMBLE_EWMA_WEIGHT, _rank01(ewma, nmax)))

    scores = [0.0] * (nmax + 1)
    for _name, weight, ranked in components:
        for number in range(1, nmax + 1):
            scores[number] += weight * ranked[number]
    return scores, {
        "version": MODEL_VERSION,
        "components": [{"name": name, "weight": weight} for name, weight, _ in components],
    }


def _binomial_fair(k: int, odd_rate: float) -> list[float]:
    """奇数个数 0..k 的二项分布，用作频率收缩的公平基准。"""
    return [
        math.comb(k, odd_n) * (odd_rate ** odd_n) * ((1.0 - odd_rate) ** (k - odd_n))
        for odd_n in range(k + 1)
    ]


def _rank_values(values: list[float]) -> list[float]:
    """转成 0..1 排名分。并列时较小下标靠前，与号码排名里较小号码优先一致。"""
    order = list(range(len(values)))
    order.sort(key=lambda index: (values[index], -index), reverse=True)
    out = [0.0] * len(values)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        out[index] = 1.0 - rank / denominator
    return out


def _odd_rate(draws: list[dict], key: str, k: int, window: int, prior: float, fair: list[float]) -> list[float]:
    """各奇数个数的贝叶斯收缩频率。每一期恰好落入一个奇数个数。"""
    use = slice_draws(draws, window)
    counts = [0] * (k + 1)
    for draw in use:
        odd_n = odd_even(draw[key])[0]
        if 0 <= odd_n <= k:
            counts[odd_n] += 1
    total = len(use)
    return [(counts[odd_n] + prior * fair[odd_n]) / (total + prior) for odd_n in range(k + 1)]


def _ewma_odd(draws: list[dict], key: str, k: int, span: int = 50) -> list[float]:
    """奇数个数的指数加权出现得分，口径对齐号码模型的 ewma50。"""
    alpha = 2.0 / (span + 1.0)
    scores = [0.0] * (k + 1)
    for draw in draws:
        odd_n = odd_even(draw[key])[0]
        for index in range(k + 1):
            appeared = 1.0 if index == odd_n else 0.0
            scores[index] = alpha * appeared + (1.0 - alpha) * scores[index]
    return scores


def _sample_odd_targets(scores: list[float], ticket_count: int, rng: random.Random) -> list[int]:
    """按分数加权抽出每注目标；全相同且次高分不可忽略时，最后一注改到次高分。"""
    weights = [max(float(score), 0.0) for score in scores]
    if sum(weights) <= 0:
        weights = [1.0] * len(scores)
    targets = [rng.choices(range(len(scores)), weights=weights, k=1)[0] for _ in range(ticket_count)]
    if ticket_count < 2 or len(scores) < 2 or len(set(targets)) > 1:
        return targets
    ranked = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)
    top, second = ranked[0], ranked[1]
    leader = float(scores[top])
    runner = float(scores[second])
    negligible = runner <= 0 or (leader > 0 and runner < leader * _OE_RUNNER_UP_MIN_RATIO)
    if negligible:
        return targets
    # 抽成同一个比例时拆开最后一注；若抽中的本来就是次高，则改到最高，避免仍然锁死。
    if targets[-1] != second:
        targets[-1] = second
    elif top != second:
        targets[-1] = top
    return targets


def infer_odd_even(draws: list[dict], key: str, ticket_count: int, window: int = 0) -> dict:
    """
    按与号码评分相同的多窗口集成，给奇数个数打分并分配各注目标。
    前区奇数个数 0–5，后区 0–2。同一批开奖、同一最近期号，分配结果不变。
    """
    if key not in ("front", "back"):
        raise ValueError(key)
    k = FRONT_K if key == "front" else BACK_K
    fair = _binomial_fair(k, _POOL_ODD_RATE[key])
    components: list[tuple[str, float, list[float]]] = []
    if draws:
        for bayes_window, prior, weight in _ENSEMBLE_BAYES:
            rates = _odd_rate(draws, key, k, bayes_window, prior, fair)
            components.append((f"bayes_{bayes_window or 'all'}", weight, _rank_values(rates)))
        recent = _odd_rate(draws, key, k, 30, 60.0, fair)
        medium = _odd_rate(draws, key, k, 120, 120.0, fair)
        trend = [recent[odd_n] - medium[odd_n] for odd_n in range(k + 1)]
        components.append(("dynamic_30_120", _ENSEMBLE_TREND_WEIGHT, _rank_values(trend)))
        ewma = _ewma_odd(draws, key, k, 50)
        components.append(("ewma50", _ENSEMBLE_EWMA_WEIGHT, _rank_values(ewma)))
    else:
        for bayes_window, _prior, weight in _ENSEMBLE_BAYES:
            components.append((f"bayes_{bayes_window or 'all'}", weight, _rank_values(list(fair))))
        components.append(("dynamic_30_120", _ENSEMBLE_TREND_WEIGHT, _rank_values([0.0] * (k + 1))))
        components.append(("ewma50", _ENSEMBLE_EWMA_WEIGHT, _rank_values([0.0] * (k + 1))))

    scores = [0.0] * (k + 1)
    for _name, weight, ranked in components:
        for odd_n in range(k + 1):
            scores[odd_n] += weight * ranked[odd_n]

    view_counts = [0] * (k + 1)
    for draw in slice_draws(draws, window):
        odd_n = odd_even(draw[key])[0]
        if 0 <= odd_n <= k:
            view_counts[odd_n] += 1

    period = str(draws[-1].get("period", "")) if draws else ""
    salt = "front-oe" if key == "front" else "back-oe"
    rng = random.Random(f"{period}:{salt}")
    targets = _sample_odd_targets(scores, max(int(ticket_count), 1), rng)
    ranking = [
        {
            "ratio": f"{odd_n}:{k - odd_n}",
            "odd": odd_n,
            "score": round(float(scores[odd_n]), 6),
            "window_count": view_counts[odd_n],
        }
        for odd_n in range(k + 1)
    ]
    ranking.sort(key=lambda row: (row["score"], -row["odd"]), reverse=True)
    return {
        "key": key,
        "seed_period": period,
        "salt": salt,
        "ranking": ranking,
        "targets": targets,
        "target_ratios": [f"{odd_n}:{k - odd_n}" for odd_n in targets],
        "window": window,
        "components": [{"name": name, "weight": weight} for name, weight, _ranked in components],
    }


def _build_duplex_shapes() -> list[tuple[int, int, int]]:
    """三区之和为 7、每区 1–4 的复式池结构，含 232、322、223、331 等。"""
    catalog = []
    for first in range(1, 5):
        for second in range(1, 5):
            third = 7 - first - second
            if 1 <= third <= 4:
                catalog.append((first, second, third))
    return catalog


DUPLEX_SHAPES = _build_duplex_shapes()


def _shrink_rate(counts: list[int], total: int, prior: float, fair: list[float]) -> list[float]:
    """贝叶斯收缩频率。prior 是等效期数，fair 是各类别的公平基准。"""
    denom = total + prior
    return [(counts[index] + prior * fair[index]) / denom for index in range(len(fair))]


def _combine_ranked(components: list[tuple[str, float, list[float]]]) -> list[float]:
    width = len(components[0][2]) if components else 0
    scores = [0.0] * width
    for _name, weight, ranked in components:
        for index, value in enumerate(ranked):
            scores[index] += weight * value
    return scores


def _ensemble_category_scores(
    draws: list[dict],
    fair: list[float],
    counts_of: Callable[[int], tuple[list[int], int]],
    ewma_of: Callable[[], list[float]],
) -> tuple[list[float], list[tuple[str, float, list[float]]]]:
    """与奇偶推理相同的多窗口集成：贝叶斯收缩、30 对 120 趋势、ewma50。"""
    width = len(fair)
    components: list[tuple[str, float, list[float]]] = []
    if draws:
        for bayes_window, prior, weight in _ENSEMBLE_BAYES:
            counts, total = counts_of(bayes_window)
            rates = _shrink_rate(counts, total, prior, fair)
            components.append((f"bayes_{bayes_window or 'all'}", weight, _rank_values(rates)))
        recent_counts, recent_total = counts_of(30)
        medium_counts, medium_total = counts_of(120)
        recent = _shrink_rate(recent_counts, recent_total, 60.0, fair)
        medium = _shrink_rate(medium_counts, medium_total, 120.0, fair)
        trend = [recent[index] - medium[index] for index in range(width)]
        components.append(("dynamic_30_120", _ENSEMBLE_TREND_WEIGHT, _rank_values(trend)))
        components.append(("ewma50", _ENSEMBLE_EWMA_WEIGHT, _rank_values(ewma_of())))
    else:
        for bayes_window, _prior, weight in _ENSEMBLE_BAYES:
            components.append((f"bayes_{bayes_window or 'all'}", weight, _rank_values(list(fair))))
        components.append(("dynamic_30_120", _ENSEMBLE_TREND_WEIGHT, _rank_values([0.0] * width)))
        components.append(("ewma50", _ENSEMBLE_EWMA_WEIGHT, _rank_values([0.0] * width)))
    return _combine_ranked(components), components


def _component_meta(components: list[tuple[str, float, list[float]]]) -> list[dict]:
    return [{"name": name, "weight": weight} for name, weight, _ranked in components]


def _allocate_by_score(
    labels: list,
    scores: list[float],
    ticket_count: int,
    period: str,
    salt: str,
    allowed: set | None = None,
) -> list:
    """用最近一期期号做随机种子，按分数加权分配。allowed 给定时只在其中抽取。"""
    rng = random.Random(f"{period}:{salt}")
    pairs = list(zip(labels, scores))
    if allowed is not None:
        filtered = [(label, score) for label, score in pairs if label in allowed]
        if filtered:
            pairs = filtered
    use_labels = [label for label, _score in pairs]
    use_scores = [score for _label, score in pairs]
    picked = _sample_odd_targets(use_scores, max(int(ticket_count), 1), rng)
    return [use_labels[index] for index in picked]


def _parse_shape_key(label: str) -> tuple[int, int, int]:
    return int(label[0]), int(label[1]), int(label[2])


def _parts_key(parts: tuple[int, int, int]) -> str:
    return f"{int(parts[0])}{int(parts[1])}{int(parts[2])}"


def _parts_label(parts: tuple[int, int, int]) -> str:
    return f"{int(parts[0])}+{int(parts[1])}+{int(parts[2])}"


def _iter_valid_fronts(draws: list[dict], window: int):
    """只产出前区恰好 5 码的开奖，供形态与和值计数。"""
    for draw in slice_draws(draws, window):
        front = list(draw.get("front") or [])
        if len(front) == FRONT_K:
            yield front


def _single_shape_catalog(draws: list[dict]) -> list[str]:
    """至少覆盖 SINGLE_SHAPES，并纳入历史里出现过、三区之和为 5 的形态。"""
    found = set(SINGLE_SHAPES)
    for front in _iter_valid_fronts(draws, 0):
        parts = zone3(front)
        if sum(parts) == FRONT_K:
            found.add(_parts_key(parts))
    return sorted(found)


def _fair_shape_prior(labels: list[str]) -> list[float]:
    """按三区容量的组合数给单式形态一个公平基准，再在候选形态上归一。"""
    denom = math.comb(FRONT_N, FRONT_K)
    raw = []
    for label in labels:
        first, second, third = _parse_shape_key(label)
        if (
            first + second + third != FRONT_K
            or first > 12
            or second > 12
            or third > 11
            or min(first, second, third) < 0
        ):
            raw.append(0.0)
            continue
        raw.append(math.comb(12, first) * math.comb(12, second) * math.comb(11, third) / denom)
    total = sum(raw)
    if total <= 0:
        return [1.0 / max(len(labels), 1)] * len(labels)
    return [value / total for value in raw]


def infer_front_shapes(
    draws: list[dict],
    ticket_count: int,
    window: int = 0,
    allowed: set[str] | None = None,
) -> dict:
    """
    给前区三区形态打分，并按注分配目标。
    同一批开奖、同一最近期号，分配不变。allowed 用来避开本期候选组不出的形态。
    """
    labels = _single_shape_catalog(draws)
    fair = _fair_shape_prior(labels)
    index_of = {label: index for index, label in enumerate(labels)}

    def counts_of(bayes_window: int) -> tuple[list[int], int]:
        counts = [0] * len(labels)
        total = 0
        for front in _iter_valid_fronts(draws, bayes_window):
            index = index_of.get(_parts_key(zone3(front)))
            if index is None:
                continue
            counts[index] += 1
            total += 1
        return counts, total

    def ewma_of() -> list[float]:
        alpha = 2.0 / 51.0
        scores = [0.0] * len(labels)
        for front in _iter_valid_fronts(draws, 0):
            hit = index_of.get(_parts_key(zone3(front)))
            for index in range(len(labels)):
                appeared = 1.0 if index == hit else 0.0
                scores[index] = alpha * appeared + (1.0 - alpha) * scores[index]
        return scores

    scores, components = _ensemble_category_scores(draws, fair, counts_of, ewma_of)
    view_counts = counts_of(window)[0] if draws else [0] * len(labels)
    period = str(draws[-1].get("period", "")) if draws else ""
    targets = _allocate_by_score(labels, scores, ticket_count, period, "front-shape", allowed)
    ranking = [
        {
            "shape": label,
            "parts": list(_parse_shape_key(label)),
            "label": _parts_label(_parse_shape_key(label)),
            "score": round(float(scores[index]), 6),
            "window_count": view_counts[index],
        }
        for index, label in enumerate(labels)
    ]
    ranking.sort(key=lambda row: (row["score"], row["shape"]), reverse=True)
    return {
        "seed_period": period,
        "salt": "front-shape",
        "ranking": ranking,
        "targets": targets,
        "window": window,
        "components": _component_meta(components),
    }


def _nearest_duplex(parts: tuple[int, int, int], catalog: list[tuple[int, int, int]]) -> int:
    """把一期 5 码的三区个数映射到比例最接近的 7 码池结构。"""
    best_index = 0
    best_key: tuple[float, int] | None = None
    for index, (first, second, third) in enumerate(catalog):
        distance = (
            abs(first / 7.0 - parts[0] / FRONT_K)
            + abs(second / 7.0 - parts[1] / FRONT_K)
            + abs(third / 7.0 - parts[2] / FRONT_K)
        )
        key = (distance, index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def _fair_duplex_prior(catalog: list[tuple[int, int, int]]) -> list[float]:
    ways = [
        math.comb(12, first) * math.comb(12, second) * math.comb(11, third)
        for first, second, third in catalog
    ]
    total = sum(ways) or 1
    return [value / total for value in ways]


def infer_duplex_shape(draws: list[dict], window: int = 0) -> dict:
    """推理一个本期共享的 7 码复式池结构。池不按注拆开，只抽一次。"""
    catalog = DUPLEX_SHAPES
    fair = _fair_duplex_prior(catalog)

    def counts_of(bayes_window: int) -> tuple[list[int], int]:
        counts = [0] * len(catalog)
        total = 0
        for front in _iter_valid_fronts(draws, bayes_window):
            counts[_nearest_duplex(zone3(front), catalog)] += 1
            total += 1
        return counts, total

    def ewma_of() -> list[float]:
        alpha = 2.0 / 51.0
        scores = [0.0] * len(catalog)
        for front in _iter_valid_fronts(draws, 0):
            hit = _nearest_duplex(zone3(front), catalog)
            for index in range(len(catalog)):
                appeared = 1.0 if index == hit else 0.0
                scores[index] = alpha * appeared + (1.0 - alpha) * scores[index]
        return scores

    scores, components = _ensemble_category_scores(draws, fair, counts_of, ewma_of)
    view_counts = counts_of(window)[0] if draws else [0] * len(catalog)
    period = str(draws[-1].get("period", "")) if draws else ""
    labels = [_parts_key(parts) for parts in catalog]
    chosen = _allocate_by_score(labels, scores, 1, period, "duplex-pool")[0]
    chosen_index = labels.index(chosen)
    parts = catalog[chosen_index]
    ranking = [
        {
            "shape": labels[index],
            "label": _parts_label(catalog[index]),
            "parts": list(catalog[index]),
            "score": round(float(scores[index]), 6),
            "window_count": view_counts[index],
        }
        for index in range(len(catalog))
    ]
    ranking.sort(key=lambda row: (row["score"], row["shape"]), reverse=True)
    return {
        "seed_period": period,
        "salt": "duplex-pool",
        "shape": chosen,
        "label": _parts_label(parts),
        "parts": list(parts),
        "score": round(float(scores[chosen_index]), 6),
        "ranking": ranking,
        "window": window,
        "components": _component_meta(components),
    }


def _sum_bin_count() -> int:
    return (_SUM_MAX - _SUM_MIN) // _SUM_BIN + 1


def _sum_to_bin(total: int) -> int:
    bounded = min(_SUM_MAX, max(_SUM_MIN, int(total)))
    return (bounded - _SUM_MIN) // _SUM_BIN


def _bin_center(index: int) -> int:
    low = _SUM_MIN + index * _SUM_BIN
    high = min(_SUM_MAX, low + _SUM_BIN - 1)
    return (low + high) // 2


_FAIR_SUM_BINS: list[float] | None = None


def _fair_sum_bins() -> list[float]:
    """5 个不重复前区号码之和的分箱公平基准，体现和值自然波动。"""
    global _FAIR_SUM_BINS
    if _FAIR_SUM_BINS is not None:
        return list(_FAIR_SUM_BINS)
    width = _sum_bin_count()
    ways = [0] * width
    table = [[0] * (_SUM_MAX + 1) for _ in range(FRONT_K + 1)]
    table[0][0] = 1
    for number in range(1, FRONT_N + 1):
        for taken in range(FRONT_K, 0, -1):
            for total in range(_SUM_MAX, number - 1, -1):
                table[taken][total] += table[taken - 1][total - number]
    for total in range(_SUM_MIN, _SUM_MAX + 1):
        ways[_sum_to_bin(total)] += table[FRONT_K][total]
    denom = sum(ways) or 1
    _FAIR_SUM_BINS = [count / denom for count in ways]
    return list(_FAIR_SUM_BINS)


def infer_front_sums(draws: list[dict], ticket_count: int, window: int = 0) -> dict:
    """按历史前区和值分布给每注一个目标和值。近窗均值和波动体现在多窗口频率里。"""
    fair = _fair_sum_bins()
    width = len(fair)
    centers = [_bin_center(index) for index in range(width)]

    def counts_of(bayes_window: int) -> tuple[list[int], int]:
        counts = [0] * width
        total = 0
        for front in _iter_valid_fronts(draws, bayes_window):
            counts[_sum_to_bin(sum(front))] += 1
            total += 1
        return counts, total

    def ewma_of() -> list[float]:
        alpha = 2.0 / 51.0
        scores = [0.0] * width
        for front in _iter_valid_fronts(draws, 0):
            hit = _sum_to_bin(sum(front))
            for index in range(width):
                appeared = 1.0 if index == hit else 0.0
                scores[index] = alpha * appeared + (1.0 - alpha) * scores[index]
        return scores

    scores, components = _ensemble_category_scores(draws, fair, counts_of, ewma_of)
    view_counts = counts_of(window)[0] if draws else [0] * width
    period = str(draws[-1].get("period", "")) if draws else ""
    targets = _allocate_by_score(centers, scores, ticket_count, period, "front-sum")
    ranking = [
        {
            "sum": centers[index],
            "score": round(float(scores[index]), 6),
            "window_count": view_counts[index],
        }
        for index in range(width)
    ]
    ranking.sort(key=lambda row: (row["score"], -row["sum"]), reverse=True)
    return {
        "seed_period": period,
        "salt": "front-sum",
        "ranking": ranking,
        "targets": [int(value) for value in targets],
        "window": window,
        "bin_width": _SUM_BIN,
        "components": _component_meta(components),
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


def duplex_232_pool(
    front_scores: list[float],
    exclude: list[int],
    shape: tuple[int, int, int] | None = None,
) -> list[int]:
    """7 码复式关注池。未传入结构时仍按 232；传入时按推理出的三区个数取号。"""
    quota = shape or DUPLEX_232
    eligible = eligible_nums(FRONT_N, exclude)
    z1, z2, z3 = split_zones(eligible)
    picks = (
        ranked_in_zone(front_scores, z1, quota[0])
        + ranked_in_zone(front_scores, z2, quota[1])
        + ranked_in_zone(front_scores, z3, quota[2])
    )
    return sorted(picks)


def _shape_ok(front: list[int], shape: tuple[int, int, int]) -> bool:
    return zone3(front) == shape


def _oe_ok(nums: list[int], odd_n: int) -> bool:
    return odd_even(nums)[0] == odd_n


def _focus_count(nums: list[int], focus: list[int]) -> int:
    return len(set(nums) & set(focus))


def _quota_fronts_by_sum(scored: list[tuple], sum_targets: list[int], limit: int) -> list[list[int]]:
    """每个目标和值留一批最接近的组合，其余再按原排序补足。"""
    picked: list[list[int]] = []
    seen = set()
    per_target = 12
    for target in sum_targets:
        taken = 0
        nearest = sorted(
            scored,
            key=lambda row: (abs(sum(row[3]) - int(target)), -row[0], -row[1], -row[2]),
        )
        for row in nearest:
            front = row[3]
            key = tuple(front)
            if key in seen:
                continue
            distance = abs(sum(front) - int(target))
            if taken > 0 and distance > _SUM_BIN * 2:
                break
            seen.add(key)
            picked.append(list(front))
            taken += 1
            if taken >= per_target:
                break
    cap = max(limit, per_target * max(len(set(sum_targets)), 1))
    for row in scored:
        front = row[3]
        key = tuple(front)
        if key in seen:
            continue
        seen.add(key)
        picked.append(list(front))
        if len(picked) >= cap:
            break
    return picked


def _zone_sources(
    scores: list[float],
    zones: tuple[list[int], list[int], list[int]],
    shapes: list[tuple[int, int, int]],
) -> tuple[list[int], list[int], list[int]]:
    """按要覆盖的形态准备各区候选号，避免只够拼出一种形态。"""
    need = [0, 0, 0]
    for shape in shapes:
        for index in range(3):
            need[index] = max(need[index], int(shape[index]))
    sources = []
    for index in range(3):
        width = min(len(zones[index]), max(need[index] + 1, 4))
        sources.append(ranked_in_zone(scores, zones[index], width))
    return sources[0], sources[1], sources[2]


def _enumerate_many_shapes(
    z1: list[int],
    z2: list[int],
    z3: list[int],
    shapes: list[tuple[int, int, int]],
    odd_n: int,
    focus: list[int],
    stats: dict,
    back_ref: list[int],
    sum_targets: list[int] | None,
) -> list[list[int]]:
    """推理模式下把多个三区形态的组合并到同一候选池。"""
    merged: list[list[int]] = []
    seen = set()
    for shape in shapes:
        part = _enumerate_fronts(
            z1,
            z2,
            z3,
            shape,
            odd_n,
            focus,
            stats,
            back_ref,
            limit=24,
            sum_targets=sum_targets,
        )
        for front in part:
            key = tuple(front)
            if key in seen:
                continue
            seen.add(key)
            merged.append(front)
    return merged


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
    sum_targets: list[int] | None = None,
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
    if sum_targets:
        # 每个目标和值都留一批接近的组合，避免 80 个高分组合全挤在同一个和值附近。
        return _quota_fronts_by_sum(scored, sum_targets, limit)
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
    odd_targets: list[int] | None = None,
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
        target_odd = odd_targets[len(out)] if odd_targets is not None and len(out) < len(odd_targets) else None

        def _back_key(row, target_odd=target_odd):
            # 推理模式下按该注目标奇偶加分；没有对应组合时退回覆盖最优。
            bonus = _OE_TARGET_BONUS if target_odd is not None and odd_even(row[2])[0] == target_odd else 0.0
            return (
                row[0] + bonus + 0.65 * sum(used[n] == 0 for n in row[2]) - 0.30 * sum(used[n] for n in row[2]),
                row[1],
                tuple(-n for n in row[2]),
            )

        best = max(scored, key=_back_key)
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
    odd_targets: list[int] | None = None,
    shape_targets: list[str] | None = None,
    sum_targets: list[int] | None = None,
) -> list[tuple[list[int], str, float, float] | None]:
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
    # 少数派目标先选。反锁死放在最后一注，若留到最后才选，容易被覆盖约束挤掉。
    pick_order = list(range(count))
    odd_tally = Counter(odd_targets or [])
    shape_tally = Counter(shape_targets or [])
    sum_tally = Counter(sum_targets or [])

    def _rarity(index: int) -> tuple:
        # 少数派目标先选，避免留到最后被覆盖约束挤掉。
        odd_rank = odd_tally[odd_targets[index]] if odd_targets and index < len(odd_targets) else 0
        shape_rank = shape_tally[shape_targets[index]] if shape_targets and index < len(shape_targets) else 0
        sum_rank = sum_tally[sum_targets[index]] if sum_targets and index < len(sum_targets) else 0
        return (shape_rank, odd_rank, sum_rank, -index)

    if odd_targets or shape_targets or sum_targets:
        pick_order.sort(key=_rarity)
    slots: list = [None] * count
    for index in pick_order:
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
            if shape_targets is not None and index < len(shape_targets):
                if zone3_label(front) == shape_targets[index]:
                    utility += _OE_TARGET_BONUS
            elif preferred_shape is not None and zone3(front) == preferred_shape:
                utility += 0.24
            if sum_targets is not None and index < len(sum_targets):
                distance = abs(sum(front) - int(sum_targets[index]))
                if distance <= _SUM_MATCH:
                    utility += _SUM_TARGET_BONUS
                else:
                    utility += _SUM_TARGET_BONUS * max(0.0, 1.0 - (distance - _SUM_MATCH) / (_SUM_BIN * 3))
            target_odd = odd_targets[index] if odd_targets is not None and index < len(odd_targets) else None
            if target_odd is not None:
                if odd_even(front)[0] == target_odd:
                    utility += _OE_TARGET_BONUS
            elif preferred_odd < 0 or odd_even(front)[0] == preferred_odd:
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
            target_odd = odd_targets[index] if odd_targets is not None and index < len(odd_targets) else None
            target_shape = shape_targets[index] if shape_targets is not None and index < len(shape_targets) else None
            target_sum = sum_targets[index] if sum_targets is not None and index < len(sum_targets) else None
            pool = remaining
            if target_shape is not None:
                matched = [front for front in pool if zone3_label(front) == target_shape]
                if matched:
                    pool = matched
            if target_sum is not None:
                matched = [front for front in pool if abs(sum(front) - int(target_sum)) <= _SUM_MATCH]
                if matched:
                    pool = matched
            if target_odd is not None:
                matched = [front for front in pool if odd_even(front)[0] == target_odd]
                if matched:
                    pool = matched
            front = max(
                pool,
                key=lambda item: (sum(scores[number] for number in item), tuple(-number for number in item)),
            )
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
        slots[index] = (front, role, model_score, popularity)
    return slots


def _mark_odd_shortfall(
    relaxed: list[str],
    tickets: list[dict],
    targets: list[int] | None,
    area: str,
    key: str,
) -> None:
    """目标奇偶在候选里凑不满时记一笔，票仍然照出。"""
    if not targets:
        return
    for index, ticket in enumerate(tickets):
        if index >= len(targets):
            break
        if odd_even(ticket[key])[0] != targets[index]:
            text = f"{area}奇偶推理（未凑满，已回退）"
            if text not in relaxed:
                relaxed.append(text)
            return


def generate_recommendations(
    draws: list[dict],
    marks: dict,
    window: int = 100,
    mode: str = "ensemble",
    shape_key: str = SHAPE_INFER,
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
    front_odd / back_odd：-2 历史推理，-1 不限，>=0 为奇数个数硬约束。
    shape_key：infer 为历史推理；其它键为手动单式形态。
    use_duplex_232 为真时使用 7 码复式池，区个数按历史推理，不再写死 232。
    constrain_sum 为真时按历史和值给每注一个目标，不再把五注卡在同一个均值±σ。
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
    # 均值±σ只作为结果里的历史参照，勾选和值时不再用它硬卡全部注。
    sum_lo, sum_hi, sum_mean = sum_band(stats, 1.0)

    front_dans = pick_dans(fs, FRONT_N, front_excl, front_focus, 1, 2, 2)
    back_dans = pick_dans(bs, BACK_N, back_excl, back_focus, 1, 1, 1)

    n = min(max(ticket_count, 3), 5)
    front_infer = infer_odd_even(draws, "front", n, window) if front_odd == ODD_EVEN_INFER else None
    back_infer = infer_odd_even(draws, "back", n, window) if back_odd == ODD_EVEN_INFER else None
    front_targets = list(front_infer["targets"]) if front_infer else None
    back_targets = list(back_infer["targets"]) if back_infer else None
    # 推理和不限都不按单一奇偶硬过滤；只有 >=0 才是硬约束。
    enum_front_odd = front_odd if front_odd >= 0 else ODD_EVEN_ANY
    enum_back_odd = back_odd if back_odd >= 0 else ODD_EVEN_ANY
    if shape_key != SHAPE_INFER and shape_key not in SINGLE_SHAPES:
        shape_key = "221"
    manual_shape = None if shape_key == SHAPE_INFER else SINGLE_SHAPES[shape_key]
    sum_infer = infer_front_sums(draws, n, window) if constrain_sum else None
    sum_targets = list(sum_infer["targets"]) if sum_infer else None

    constraints_relaxed: list[str] = []
    duplex_infer = infer_duplex_shape(draws, window) if use_duplex_232 else None
    duplex_parts = tuple(duplex_infer["parts"]) if duplex_infer else None
    duplex = duplex_232_pool(fs, front_excl, duplex_parts) if duplex_parts else []
    if use_duplex_232 and len(duplex) >= FRONT_K and duplex_infer is not None:
        # 池内全部 5 码子集都可入选；形态只作偏好，避免某一区个数把五注绑死。
        fronts = [sorted(cand) for cand in combinations(sorted(duplex), FRONT_K)]
        actual_parts = zone3(duplex)
        duplex_infer["actual_parts"] = list(actual_parts)
        duplex_infer["actual_shape"] = _parts_key(actual_parts)
        duplex_infer["actual_label"] = _parts_label(actual_parts)
        if actual_parts != duplex_parts:
            constraints_relaxed.append("复式池分区（未凑满，已回退）")
        pool_note = f"复式池{duplex_infer['actual_label']}"
        front_pool = list(duplex)
    else:
        if use_duplex_232:
            duplex_infer = None
            duplex = []
        if manual_shape is None:
            catalog = _single_shape_catalog(draws)
            shape_tuples = [_parse_shape_key(label) for label in catalog]
            src1, src2, src3 = _zone_sources(fs, (z1, z2, z3), shape_tuples)
            fronts = _enumerate_many_shapes(
                src1, src2, src3, shape_tuples, enum_front_odd, front_focus, stats, back_ref, sum_targets,
            )
            if front_odd >= 0 and len(fronts) < n:
                constraints_relaxed.append("前区奇偶")
                fronts = _enumerate_many_shapes(
                    src1, src2, src3, shape_tuples, ODD_EVEN_ANY, front_focus, stats, back_ref, sum_targets,
                )
            pool_note = "形态历史推理"
        else:
            src1 = ranked_in_zone(fs, z1, max(manual_shape[0] + 3, 4))
            src2 = ranked_in_zone(fs, z2, max(manual_shape[1] + 3, 4))
            src3 = ranked_in_zone(fs, z3, max(manual_shape[2] + 3, 4))
            fronts = _enumerate_fronts(
                src1, src2, src3, manual_shape, enum_front_odd, front_focus, stats, back_ref,
                sum_targets=sum_targets,
            )
            if front_odd >= 0 and len(fronts) < n:
                constraints_relaxed.append("前区奇偶")
                fronts = _enumerate_fronts(
                    src1, src2, src3, manual_shape, ODD_EVEN_ANY, front_focus, stats, back_ref,
                    sum_targets=sum_targets,
                )
            pool_note = f"单式形态{shape_key}"
        front_pool = sorted(set(src1 + src2 + src3))

    shape_infer = None
    shape_targets = None
    if manual_shape is None:
        allowed = {zone3_label(front) for front in fronts}
        shape_infer = infer_front_shapes(draws, n, window, allowed)
        shape_targets = list(shape_infer["targets"])

    front_pairs = _pair_residuals(draws, "front", FRONT_N)
    back_pairs = _pair_residuals(draws, "back", BACK_N)
    selected_fronts = _select_diverse_fronts(
        fronts,
        fs,
        n,
        front_pairs,
        manual_shape,
        enum_front_odd,
        tier_scores=raw_fs,
        popularity_scorer=front_popularity_scorer,
        popularity_weight=effective_crowd_weight,
        odd_targets=front_targets,
        shape_targets=shape_targets,
        sum_targets=sum_targets,
    )
    front_anchor = sorted(eligible_f, key=lambda number: (fs[number], -number), reverse=True)[:5]
    back_popularity_scorer = (
        (lambda back: crowd_model.score(front_anchor, back, draws)) if crowd_model else None
    )
    backs = _pick_backs(
        bs,
        back_excl,
        back_focus,
        enum_back_odd,
        n,
        back_pairs,
        back_popularity_scorer,
        effective_crowd_weight,
        odd_targets=back_targets,
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
        chosen = selected_fronts[i] if i < len(selected_fronts) else None
        if chosen is not None:
            front, role, draw_score, popularity = chosen
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
                "target_shape": shape_targets[i] if shape_targets and i < len(shape_targets) else "",
                "target_sum": int(sum_targets[i]) if sum_targets and i < len(sum_targets) else None,
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

    if manual_shape is not None and any(zone3(ticket["front"]) != manual_shape for ticket in tickets):
        constraints_relaxed.append("部分单式形态（用于减少五注绑号）")
    if shape_targets and any(
        zone3_label(ticket["front"]) != shape_targets[index]
        for index, ticket in enumerate(tickets)
        if index < len(shape_targets)
    ):
        constraints_relaxed.append("前区形态推理（未凑满，已回退）")
    if sum_targets and any(
        abs(int(ticket["front_sum"]) - int(sum_targets[index])) > _SUM_MATCH
        for index, ticket in enumerate(tickets)
        if index < len(sum_targets)
    ):
        constraints_relaxed.append("前区和值推理（未凑满，已回退）")
    if front_odd >= 0 and any(odd_even(ticket["front"])[0] != front_odd for ticket in tickets):
        if "前区奇偶" not in constraints_relaxed:
            constraints_relaxed.append("前区奇偶")
    _mark_odd_shortfall(constraints_relaxed, tickets, front_targets, "前区", "front")
    _mark_odd_shortfall(constraints_relaxed, tickets, back_targets, "后区", "back")

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
        "shape_plan": list(shape_infer["targets"]) if shape_infer else [],
        "shape_infer": shape_infer,
        "use_duplex_232": use_duplex_232,
        "duplex_shape": (duplex_infer or {}).get("actual_shape") or (duplex_infer or {}).get("shape") or "",
        "duplex_label": (duplex_infer or {}).get("actual_label") or (duplex_infer or {}).get("label") or "",
        "duplex_infer": duplex_infer,
        "front_odd": front_odd,
        "back_odd": back_odd,
        "front_odd_plan": list(front_infer["target_ratios"]) if front_infer else [],
        "back_odd_plan": list(back_infer["target_ratios"]) if back_infer else [],
        "front_odd_infer": front_infer,
        "back_odd_infer": back_infer,
        "ticket_count": n,
        "constrain_sum": constrain_sum,
        "sum_plan": list(sum_infer["targets"]) if sum_infer else [],
        "sum_infer": sum_infer,
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
