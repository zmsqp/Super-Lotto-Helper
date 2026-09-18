# -*- coding: utf-8 -*-
"""基于官方销量与一等奖注数校准公众选号集中度。

该模型只估计中奖后可能有多少同奖者，不改变任何组合的开奖概率。
所有动态特征都只读取目标期以前的开奖，避免未来数据泄漏。
"""
from __future__ import annotations

import math
import random
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


MARKET_MODEL_VERSION = "official-sales-crowd-v1"
FEATURE_NAMES = (
    "birthday_front",
    "lucky_front",
    "round_front",
    "front_consecutive",
    "front_same_tail",
    "front_sum_center",
    "front_odd_balance",
    "front_zone_balance",
    "front_recent_heat",
    "front_repeat",
    "back_lucky",
    "back_consecutive",
    "back_sum_center",
    "back_recent_heat",
    "back_repeat",
    "cross_same_tail",
)
LUCKY_FRONT = {6, 8, 9, 18, 28}
LUCKY_BACK = {6, 7, 8, 9}
_MODEL_CACHE: dict[tuple, CrowdModel | None] = {}


def load_market_validation() -> dict:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    path = root / "assets" / "dlt_market_validation.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}


def _safe_int(value, default=0) -> int:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _first_winners(row: dict) -> int:
    if row.get("first_count") is not None:
        return _safe_int(row.get("first_count"))
    counts = row.get("prize_counts") or {}
    return _safe_int(counts.get("101") or counts.get(101))


def _recent_rates(history: list[dict]) -> tuple[list[float], list[float]]:
    recent = history[-30:]
    if not recent:
        return [1.0] * 36, [1.0] * 13
    front_counts = [0] * 36
    back_counts = [0] * 13
    for draw in recent:
        for number in draw.get("front") or []:
            front_counts[int(number)] += 1
        for number in draw.get("back") or []:
            back_counts[int(number)] += 1
    front_expected = len(recent) * 5 / 35
    back_expected = len(recent) * 2 / 12
    return (
        [0.0] + [front_counts[n] / front_expected for n in range(1, 36)],
        [0.0] + [back_counts[n] / back_expected for n in range(1, 13)],
    )


def combo_features(front: Iterable[int], back: Iterable[int], history: list[dict]) -> list[float]:
    """返回可解释的人气特征；值越大通常越接近常见人工选号习惯。"""
    f = sorted(int(number) for number in front)
    b = sorted(int(number) for number in back)
    if len(f) != 5 or len(b) != 2:
        raise ValueError("大乐透单式必须是前区5个、后区2个号码")
    front_heat, back_heat = _recent_rates(history)
    previous_front = set(history[-1].get("front") or []) if history else set()
    previous_back = set(history[-1].get("back") or []) if history else set()
    zones = (
        sum(number <= 12 for number in f),
        sum(13 <= number <= 24 for number in f),
        sum(number >= 25 for number in f),
    )
    same_tail = sum(left % 10 == right % 10 for left, right in combinations(f, 2))
    cross_tail = sum(left % 10 == right % 10 for left in f for right in b)
    return [
        sum(number <= 31 for number in f) / 5,
        sum(number in LUCKY_FRONT for number in f) / 5,
        sum(number % 5 == 0 for number in f) / 5,
        sum(right - left == 1 for left, right in zip(f, f[1:])) / 4,
        same_tail / 3,
        1 - min(abs(sum(f) - 90) / 60, 1),
        1 - abs(sum(number % 2 for number in f) - 2.5) / 2.5,
        1 - min((max(zones) - min(zones)) / 5, 1),
        sum(front_heat[number] for number in f) / 5,
        len(set(f) & previous_front) / 5,
        sum(number in LUCKY_BACK for number in b) / 2,
        float(b[1] - b[0] == 1),
        1 - min(abs(sum(b) - 13) / 11, 1),
        sum(back_heat[number] for number in b) / 2,
        len(set(b) & previous_back) / 2,
        cross_tail / 4,
    ]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """带主元选择的高斯消元；矩阵很小，不引入额外数值依赖。"""
    n = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) < 1e-15:
                continue
            augmented[row] = [
                augmented[row][col] - factor * augmented[column][col]
                for col in range(n + 1)
            ]
    return [augmented[row][-1] for row in range(n)]


@dataclass(frozen=True)
class CrowdModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    samples: int
    first_period: str
    last_period: str

    def predict_features(self, features: list[float]) -> float:
        standardized = [
            (features[index] - self.means[index]) / self.scales[index]
            for index in range(len(FEATURE_NAMES))
        ]
        return self.coefficients[0] + sum(
            self.coefficients[index + 1] * value for index, value in enumerate(standardized)
        )

    def score(self, front: Iterable[int], back: Iterable[int], history: list[dict]) -> float:
        return self.predict_features(combo_features(front, back, history))

    def expected_first_winners_per_100m(
        self, front: Iterable[int], back: Iterable[int], history: list[dict]
    ) -> float:
        return max(0.0, math.expm1(self.score(front, back, history)))

    def detail(self) -> dict:
        ranked = sorted(
            zip(FEATURE_NAMES, self.coefficients[1:]), key=lambda item: abs(item[1]), reverse=True
        )
        return {
            "version": MARKET_MODEL_VERSION,
            "samples": self.samples,
            "period_range": [self.first_period, self.last_period],
            "strongest_features": [
                {"name": name, "coefficient": round(coefficient, 5)}
                for name, coefficient in ranked[:6]
            ],
        }


def _fit_xy(xs: list[list[float]], ys: list[float], ridge: float = 8.0) -> CrowdModel | None:
    if len(xs) < 80:
        return None
    width = len(FEATURE_NAMES)
    means = [sum(row[col] for row in xs) / len(xs) for col in range(width)]
    scales = []
    for col in range(width):
        variance = sum((row[col] - means[col]) ** 2 for row in xs) / len(xs)
        scales.append(max(math.sqrt(variance), 1e-6))
    design = [[1.0] + [(row[col] - means[col]) / scales[col] for col in range(width)] for row in xs]
    size = width + 1
    gram = [[0.0] * size for _ in range(size)]
    target = [0.0] * size
    for row, y in zip(design, ys):
        for left in range(size):
            target[left] += row[left] * y
            for right in range(size):
                gram[left][right] += row[left] * row[right]
    for index in range(1, size):
        gram[index][index] += ridge
    coefficients = _solve(gram, target)
    return CrowdModel(tuple(means), tuple(scales), tuple(coefficients), len(xs), "", "")


def build_market_examples(draws: list[dict], market_rows: list[dict]) -> list[dict]:
    market_by_period = {str(row.get("period")): row for row in market_rows}
    examples = []
    for index, draw in enumerate(draws):
        row = market_by_period.get(str(draw.get("period")))
        if not row:
            continue
        sales = _safe_int(row.get("sales"))
        if sales <= 0:
            continue
        first_count = _first_winners(row)
        density = first_count * 100_000_000 / sales
        examples.append(
            {
                "draw_index": index,
                "period": str(draw.get("period")),
                "features": combo_features(draw["front"], draw["back"], draws[:index]),
                "target": math.log1p(density),
                "first_count": first_count,
                "sales": sales,
                "density": density,
            }
        )
    return examples


def fit_crowd_model(draws: list[dict], market_rows: list[dict]) -> CrowdModel | None:
    cache_key = (
        str(draws[-1].get("period")) if draws else "",
        len(draws),
        str(market_rows[-1].get("period")) if market_rows else "",
        len(market_rows),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    examples = build_market_examples(draws, market_rows)
    model = _fit_xy([row["features"] for row in examples], [row["target"] for row in examples])
    if model is None:
        _MODEL_CACHE[cache_key] = None
        return None
    result = CrowdModel(
        model.means,
        model.scales,
        model.coefficients,
        len(examples),
        examples[0]["period"],
        examples[-1]["period"],
    )
    _MODEL_CACHE[cache_key] = result
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def validate_crowd_model(
    draws: list[dict], market_rows: list[dict], horizon: int = 300, random_combos: int = 40
) -> dict:
    """逐期事前验证人气预测，并检验开奖号是否异常偏向低人气。"""
    examples = build_market_examples(draws, market_rows)
    if len(examples) < 160:
        return {"available": False, "samples": len(examples), "reason": "官方销量样本不足160期"}
    start = max(120, len(examples) - horizon)
    predictions: list[float] = []
    actuals: list[float] = []
    baselines: list[float] = []
    percentiles: list[float] = []
    for target in range(start, len(examples)):
        train = examples[:target]
        model = _fit_xy(
            [row["features"] for row in train], [row["target"] for row in train]
        )
        if model is None:
            continue
        current = examples[target]
        prediction = model.predict_features(current["features"])
        predictions.append(prediction)
        actuals.append(current["target"])
        baselines.append(sum(row["target"] for row in train) / len(train))

        draw_index = current["draw_index"]
        history = draws[:draw_index]
        rng = random.Random(int(current["period"]) if current["period"].isdigit() else target)
        random_scores = []
        for _ in range(random_combos):
            front = sorted(rng.sample(range(1, 36), 5))
            back = sorted(rng.sample(range(1, 13), 2))
            random_scores.append(model.score(front, back, history))
        percentile = sum(score <= prediction for score in random_scores) / len(random_scores)
        percentiles.append(percentile)

    model_mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)
    baseline_mae = sum(abs(p - a) for p, a in zip(baselines, actuals)) / len(actuals)
    mean_percentile = sum(percentiles) / len(percentiles)
    percentile_se = math.sqrt(1 / (12 * len(percentiles)))
    z_score = (mean_percentile - 0.5) / percentile_se if percentile_se else 0.0
    control_p = math.erfc(abs(z_score) / math.sqrt(2))
    order = sorted(range(len(predictions)), key=lambda index: predictions[index])
    quartile = max(1, len(order) // 4)
    low_actual = sum(actuals[index] for index in order[:quartile]) / quartile
    high_actual = sum(actuals[index] for index in order[-quartile:]) / quartile
    correlation = _pearson(predictions, actuals)
    deployable = correlation > 0.03 and model_mae < baseline_mae
    return {
        "available": True,
        "version": MARKET_MODEL_VERSION,
        "samples": len(predictions),
        "correlation": round(correlation, 5),
        "model_mae": round(model_mae, 5),
        "baseline_mae": round(baseline_mae, 5),
        "low_quartile_actual": round(low_actual, 5),
        "high_quartile_actual": round(high_actual, 5),
        "actual_draw_mean_popularity_percentile": round(mean_percentile, 5),
        "control_hypothesis_two_sided_p": round(control_p, 6),
        "control_hypothesis_supported": bool(mean_percentile < 0.47 and control_p < 0.05),
        "deployable": deployable,
        "conclusion": (
            "样本外可用于降低潜在分奖人数" if deployable else "样本外预测力不足，仅保留启发式人气提示"
        ),
    }
