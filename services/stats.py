# -*- coding: utf-8 -*-
"""从 analyze_dlt 抽复的纯统计函数（不含全量回测）。"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from datetime import datetime, timedelta

FRONT_N, BACK_N = 35, 12
FRONT_K, BACK_K = 5, 2
P_FRONT = FRONT_K / FRONT_N
P_BACK = BACK_K / BACK_N


def slice_draws(draws: list[dict], window: int) -> list[dict]:
    """按窗口截取最近若干期；window<=0 表示全量。"""
    if window and window > 0 and len(draws) > window:
        return draws[-window:]
    return draws


def freq_count(draws: list[dict], key: str, nmax: int) -> list[int]:
    """统计每个号码的出现次数，下标 1..nmax 有效。"""
    c = [0] * (nmax + 1)
    for d in draws:
        for x in d[key]:
            c[x] += 1
    return c


def current_omission(draws: list[dict], key: str, nmax: int) -> list[int]:
    """当前遗漏：距上次出现的期数。从未出现则为总期数。"""
    last = [-1] * (nmax + 1)
    for i, d in enumerate(draws):
        for x in d[key]:
            last[x] = i
    n = len(draws)
    return [n - 1 - last[i] if last[i] >= 0 else n for i in range(nmax + 1)]


def ewma_scores(draws: list[dict], key: str, nmax: int, span: int) -> list[float]:
    """指数加权出现得分。"""
    alpha = 2.0 / (span + 1.0)
    s = [0.0] * (nmax + 1)
    for d in draws:
        appeared = set(d[key])
        for i in range(1, nmax + 1):
            x = 1.0 if i in appeared else 0.0
            s[i] = alpha * x + (1.0 - alpha) * s[i]
    return s


def odd_even(nums: list[int]) -> tuple[int, int]:
    """返回 (奇数个数, 偶数个数)。"""
    odd = sum(1 for x in nums if x % 2)
    return odd, len(nums) - odd


def zone3(front: list[int]) -> tuple[int, int, int]:
    """前区三区个数：01-12 / 13-24 / 25-35。"""
    a = sum(1 for x in front if x <= 12)
    b = sum(1 for x in front if 13 <= x <= 24)
    c = sum(1 for x in front if x >= 25)
    return a, b, c


def zone3_label(front: list[int]) -> str:
    """三区形态字符串，如 221。"""
    a, b, c = zone3(front)
    return f"{a}{b}{c}"


def consec_pairs(nums: list[int]) -> int:
    """相邻连号对数。"""
    s = sorted(nums)
    return sum(1 for i in range(len(s) - 1) if s[i + 1] - s[i] == 1)


def pick_top(scores: list[float], k: int, nmax: int, reverse: bool = True) -> list[int]:
    """按分数取前 k 个号码，返回升序号码列表。"""
    idx = list(range(1, nmax + 1))
    idx.sort(key=lambda i: (scores[i], -i), reverse=reverse)
    return sorted(idx[:k])


def hits(pred: list[int], actual: list[int]) -> int:
    """命中个数。"""
    return len(set(pred) & set(actual))


def _zscore(arr: list[float]) -> list[float]:
    vals = arr[1:]
    mu = statistics.fmean(vals) if vals else 0.0
    sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
    if sd == 0:
        sd = 1.0
    return [0.0] + [(x - mu) / sd for x in vals]


def score_vector(draws: list[dict], key: str, nmax: int, mode: str, window: int) -> list[float]:
    """为每个号码打分（下标 1..nmax）。GUI 用此函数，不跑回测。"""
    use = slice_draws(draws, window)
    if not use:
        return [0.0] * (nmax + 1)
    k = FRONT_K if key == "front" else BACK_K
    if mode == "hot":
        c = freq_count(use, key, nmax)
        return [float(x) for x in c]
    if mode == "cold":
        c = freq_count(use, key, nmax)
        return [0.0] + [-float(c[i]) for i in range(1, nmax + 1)]
    if mode == "omit":
        om = current_omission(use, key, nmax)
        return [float(x) for x in om]
    if mode == "ewma50":
        return ewma_scores(use, key, nmax, 50)
    if mode == "ewma20":
        return ewma_scores(use, key, nmax, 20)
    if mode == "under":
        c = freq_count(use, key, nmax)
        exp = len(use) * (k / nmax)
        return [0.0] + [(exp - c[i]) / math.sqrt(max(exp, 1.0)) for i in range(1, nmax + 1)]
    if mode == "blend":
        c = freq_count(use, key, nmax)
        om = current_omission(use, key, nmax)
        ew = ewma_scores(use, key, nmax, 40)
        zc = _zscore([float(x) for x in c])
        zo = _zscore([float(x) for x in om])
        ze = _zscore(ew)
        return [0.0] + [0.45 * zc[i] + 0.25 * zo[i] + 0.30 * ze[i] for i in range(1, nmax + 1)]
    raise ValueError(mode)


def rank_pick(
    hist: list[dict],
    key: str,
    nmax: int,
    k: int,
    mode: str,
    window: int,
) -> list[int]:
    """按模式取最高分的 k 个号码。"""
    scores = score_vector(hist, key, nmax, mode, window)
    return pick_top(scores, k, nmax, True)


def combo_quality(front: list[int], back: list[int], stats: dict) -> float:
    """单式与历史形态均值的贴近程度，越大越好。"""
    s = sum(front)
    odd, _ = odd_even(front)
    a, b, c = zone3(front)
    cp = consec_pairs(front)
    span = max(front) - min(front)
    bs = sum(back)
    bo, _ = odd_even(back)
    score = 0.0
    score -= abs(s - stats["front_sum_mean"]) / max(stats["front_sum_std"], 1.0)
    score -= abs(span - stats["front_span_mean"]) / max(stats["front_span_std"], 1.0)
    score -= abs(odd - stats["front_odd_mean"])
    score -= abs((a, b, c).count(0)) * 0.4
    score -= abs(cp - stats["front_consec_mean"])
    score -= abs(bs - stats["back_sum_mean"]) / max(stats["back_sum_std"], 1.0)
    score -= abs(bo - stats["back_odd_mean"]) * 0.3
    return score


def pattern_stats(draws: list[dict]) -> dict:
    """和值、跨度、奇偶、连号等形态均值。"""
    if not draws:
        return {
            "front_sum_mean": 0.0,
            "front_sum_std": 1.0,
            "front_span_mean": 0.0,
            "front_span_std": 1.0,
            "front_odd_mean": 0.0,
            "front_consec_mean": 0.0,
            "back_sum_mean": 0.0,
            "back_sum_std": 1.0,
            "back_odd_mean": 0.0,
        }
    fsums = [sum(d["front"]) for d in draws]
    fspan = [max(d["front"]) - min(d["front"]) for d in draws]
    fodd = [odd_even(d["front"])[0] for d in draws]
    fcon = [consec_pairs(d["front"]) for d in draws]
    bsums = [sum(d["back"]) for d in draws]
    bodd = [odd_even(d["back"])[0] for d in draws]
    return {
        "front_sum_mean": statistics.fmean(fsums),
        "front_sum_std": statistics.pstdev(fsums) if len(fsums) > 1 else 1.0,
        "front_span_mean": statistics.fmean(fspan),
        "front_span_std": statistics.pstdev(fspan) if len(fspan) > 1 else 1.0,
        "front_odd_mean": statistics.fmean(fodd),
        "front_consec_mean": statistics.fmean(fcon),
        "back_sum_mean": statistics.fmean(bsums),
        "back_sum_std": statistics.pstdev(bsums) if len(bsums) > 1 else 1.0,
        "back_odd_mean": statistics.fmean(bodd),
    }


def binom_p_two_sided(k: int, n: int, p: float) -> float:
    """二项分布双侧近似 p 值。"""
    if n == 0:
        return 1.0
    mu = n * p
    var = n * p * (1.0 - p)
    if var <= 0:
        return 1.0
    z = abs(k - mu) / math.sqrt(var)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def number_table(draws: list[dict], key: str, nmax: int, p: float) -> list[dict]:
    """号值表：次数、频率、期望差、遗漏、近 30/100、EWMA。"""
    n = len(draws)
    if n == 0:
        return []
    c = freq_count(draws, key, nmax)
    om = current_omission(draws, key, nmax)
    ew = ewma_scores(draws, key, nmax, 40)
    exp = n * p
    recent100 = freq_count(draws[-100:], key, nmax)
    recent30 = freq_count(draws[-30:], key, nmax)
    rows = []
    for i in range(1, nmax + 1):
        rows.append(
            {
                "num": f"{i:02d}",
                "all": c[i],
                "rate": round(c[i] / n, 4),
                "exp": round(exp, 1),
                "diff": round(c[i] - exp, 2),
                "pval": round(binom_p_two_sided(c[i], n, p), 4),
                "omit": om[i],
                "ewma": round(ew[i], 4),
                "r100": recent100[i],
                "r30": recent30[i],
            }
        )
    return rows


def next_draw_meta(last_date: str, last_period: str) -> dict:
    """由最近一期推下一期（周一 / 周三 / 周六）。"""
    last = datetime.strptime(last_date, "%Y-%m-%d")
    d = last
    for _ in range(8):
        d = d + timedelta(days=1)
        if d.weekday() in (0, 2, 5):
            break
    try:
        nxt_period = str(int(last_period) + 1)
    except ValueError:
        nxt_period = ""
    week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return {
        "period": nxt_period,
        "date": d.strftime("%Y-%m-%d"),
        "weekday": week[d.weekday()],
    }


def odd_even_counts(draws: list[dict], key: str = "front") -> list[tuple[str, int]]:
    """奇偶比频次，按奇数个数分组。"""
    c = Counter(odd_even(d[key])[0] for d in draws)
    k = FRONT_K if key == "front" else BACK_K
    rows = []
    for odd_n in range(k + 1):
        even_n = k - odd_n
        rows.append((f"{odd_n}:{even_n}", c.get(odd_n, 0)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def zone_shape_counts(draws: list[dict]) -> list[tuple[str, int]]:
    """前区三区形态频次，高到低。"""
    c = Counter(zone3_label(d["front"]) for d in draws)
    return c.most_common()


def format_nums(nums: list[int]) -> str:
    """号码格式化为两位空格分隔。"""
    return " ".join(f"{x:02d}" for x in nums)


def sum_band(stats: dict, sigma: float = 1.0) -> tuple[int, int, float]:
    """前区和值常用区间：均值 ± sigma·标准差。"""
    mean = float(stats.get("front_sum_mean") or 0.0)
    std = max(float(stats.get("front_sum_std") or 1.0), 1.0)
    lo = int(round(mean - sigma * std))
    hi = int(round(mean + sigma * std))
    return lo, hi, mean


def sum_comment(value: int, mean: float, std: float) -> str:
    """和值相对历史均值的位置说明。"""
    sd = max(std, 1.0)
    z = (value - mean) / sd
    if abs(z) <= 0.5:
        return "适中"
    if z < -1.0:
        return "偏小"
    if z > 1.0:
        return "偏大"
    if z < 0:
        return "略小"
    return "略大"
