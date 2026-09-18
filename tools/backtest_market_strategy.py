# -*- coding: utf-8 -*-
"""严格时间序列回测官方销量人气层，不使用目标期开奖调参。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.algorithms import generate_recommendations  # noqa: E402
from services.market import fit_crowd_model, validate_crowd_model  # noqa: E402
from services.stats import prize_grade  # noqa: E402
from services.store import Store  # noqa: E402

WEIGHTS = (0.0, 0.05, 0.10, 0.20)
MARKS = {"front": {"focus": [], "exclude": []}, "back": {"focus": [], "exclude": []}}


def _outcome(tickets: list[dict], actual: dict) -> dict:
    grades = []
    for ticket in tickets:
        front_hits = len(set(ticket["front"]) & set(actual["front"]))
        back_hits = len(set(ticket["back"]) & set(actual["back"]))
        grade = prize_grade(front_hits, back_hits)
        if grade is not None:
            grades.append(grade)
    best = min(grades) if grades else None
    return {
        "any_prize": bool(grades),
        "grade_2_7": bool(best is not None and 2 <= best <= 7),
        "grade_2_5": bool(best is not None and 2 <= best <= 5),
        "best_grade": best,
    }


def _mean_crowd(tickets: list[dict], model, history: list[dict]) -> float:
    if model is None or not tickets:
        return 0.0
    return sum(
        model.expected_first_winners_per_100m(ticket["front"], ticket["back"], history)
        for ticket in tickets
    ) / len(tickets)


def _generate(history: list[dict], market: list[dict] | None, weight: float) -> dict:
    return generate_recommendations(
        history,
        MARKS,
        mode="ensemble",
        shape_key="212",
        use_duplex_232=False,
        front_odd=-1,
        back_odd=-1,
        ticket_count=5,
        constrain_sum=True,
        market_rows=market,
        crowd_weight=weight,
        include_reference=False,
    )


def evaluate(draws: list[dict], market_rows: list[dict], indexes: list[int], weights: tuple) -> dict:
    by_weight = {str(weight): [] for weight in weights}
    baseline = []
    for sequence, target_index in enumerate(indexes, 1):
        history = draws[:target_index]
        actual = draws[target_index]
        market_train = [
            row for row in market_rows if str(row["period"]) < str(actual["period"])
        ]
        crowd_model = fit_crowd_model(history, market_train)

        base_result = _generate(history, None, 0.0)
        base_outcome = _outcome(base_result["tickets"], actual)
        base_outcome.update(
            {
                "period": actual["period"],
                "mean_crowd": _mean_crowd(base_result["tickets"], crowd_model, history),
            }
        )
        baseline.append(base_outcome)

        for weight in weights:
            result = _generate(history, market_train, weight)
            outcome = _outcome(result["tickets"], actual)
            outcome.update(
                {
                    "period": actual["period"],
                    "mean_crowd": _mean_crowd(result["tickets"], crowd_model, history),
                }
            )
            by_weight[str(weight)].append(outcome)
        if sequence % 50 == 0:
            print(f"progress={sequence}/{len(indexes)} period={actual['period']}", flush=True)
    return {"baseline": baseline, "weights": by_weight}


def _summary(rows: list[dict]) -> dict:
    count = len(rows)
    rolling = []
    for fold in range(8):
        start = round(fold * count / 8)
        end = round((fold + 1) * count / 8)
        part = rows[start:end]
        rolling.append(sum(row["grade_2_7"] for row in part) / len(part) if part else 0.0)
    return {
        "draws": count,
        "any_prize_hits": sum(row["any_prize"] for row in rows),
        "any_prize_rate": sum(row["any_prize"] for row in rows) / count,
        "grade_2_7_hits": sum(row["grade_2_7"] for row in rows),
        "grade_2_7_rate": sum(row["grade_2_7"] for row in rows) / count,
        "grade_2_5_hits": sum(row["grade_2_5"] for row in rows),
        "grade_2_5_rate": sum(row["grade_2_5"] for row in rows) / count,
        "mean_crowd": sum(row["mean_crowd"] for row in rows) / count,
        "rolling_8_grade_2_7": rolling,
    }


def _paired(candidate: list[dict], baseline: list[dict]) -> dict:
    candidate_only = sum(
        left["grade_2_7"] and not right["grade_2_7"]
        for left, right in zip(candidate, baseline)
    )
    baseline_only = sum(
        right["grade_2_7"] and not left["grade_2_7"]
        for left, right in zip(candidate, baseline)
    )
    discordant = candidate_only + baseline_only
    if discordant:
        p_value = sum(
            math.comb(discordant, value) for value in range(candidate_only, discordant + 1)
        ) / (2**discordant)
    else:
        p_value = 1.0
    return {
        "candidate_only_wins": candidate_only,
        "baseline_only_wins": baseline_only,
        "one_sided_p": p_value,
    }


def _official_summary(rows: list[dict]) -> dict:
    sales = [int(row.get("sales") or 0) for row in rows]
    first = [int((row.get("prize_counts") or {}).get("101") or 0) for row in rows]
    lower_groups = ("301", "501", "601", "701", "801", "901")
    lower = [sum(int((row.get("prize_counts") or {}).get(group) or 0) for group in lower_groups) for row in rows]
    return {
        "draws": len(rows),
        "mean_sales": sum(sales) / len(sales),
        "mean_first_prize_stakes": sum(first) / len(first),
        "zero_first_prize_rate": sum(value == 0 for value in first) / len(first),
        "mean_grade_2_7_stakes": sum(lower) / len(lower),
    }


def main() -> None:
    db_candidates = [ROOT / "dist_v2_1" / "data" / "dlt.db", ROOT / "data" / "dlt.db"]
    db_path = next(path for path in db_candidates if path.exists())
    store = Store(db_path)
    draws = store.list_draws_asc()
    store.close()
    market_rows = json.loads(
        (ROOT / "assets" / "dlt_market_history.json").read_text(encoding="utf-8")
    )["rows"]
    market_periods = {str(row["period"]) for row in market_rows}
    eligible = [
        index for index, draw in enumerate(draws) if str(draw["period"]) in market_periods and index >= 500
    ]
    if len(eligible) < 700:
        raise RuntimeError(f"可回测市场期数不足700：{len(eligible)}")
    development_indexes = eligible[-700:-500]
    test_indexes = eligible[-500:]

    development = evaluate(draws, market_rows, development_indexes, WEIGHTS)
    development_summary = {
        weight: _summary(rows) for weight, rows in development["weights"].items()
    }
    best_development_rate = max(
        development_summary[str(weight)]["grade_2_7_rate"] for weight in WEIGHTS
    )
    eligible_weights = [
        weight
        for weight in WEIGHTS
        if development_summary[str(weight)]["grade_2_7_rate"]
        >= best_development_rate - 0.01 - 1e-12
    ]
    selected_weight = min(
        eligible_weights,
        key=lambda weight: (development_summary[str(weight)]["mean_crowd"], weight),
    )
    locked_weights = tuple(dict.fromkeys((0.0, selected_weight)))
    locked = evaluate(draws, market_rows, test_indexes, locked_weights)
    baseline_summary = _summary(locked["baseline"])
    neutral_summary = _summary(locked["weights"]["0.0"])
    selected_rows = locked["weights"][str(selected_weight)]
    selected_summary = _summary(selected_rows)
    validation = validate_crowd_model(draws, market_rows, horizon=300, random_combos=40)
    crowd_reduction = (
        1 - selected_summary["mean_crowd"] / baseline_summary["mean_crowd"]
        if baseline_summary["mean_crowd"]
        else 0.0
    )
    result = {
        "model_version": "ensemble-coverage-v2.2+official-sales-crowd-v1",
        "data": {
            "draws": len(draws),
            "market_rows": len(market_rows),
            "market_range": [market_rows[0]["period"], market_rows[-1]["period"]],
            "development_range": [draws[development_indexes[0]]["period"], draws[development_indexes[-1]]["period"]],
            "locked_test_range": [draws[test_indexes[0]]["period"], draws[test_indexes[-1]]["period"]],
        },
        "official_market": _official_summary(market_rows),
        "crowd_model_validation": validation,
        "development": development_summary,
        "selection_rule": "开发段2–7等奖率距最优不超过1个百分点时，选择人气风险最低权重",
        "selected_weight": selected_weight,
        "locked_test": {
            "legacy_baseline": baseline_summary,
            "official_model_weight_0": neutral_summary,
            "payout_aware": selected_summary,
            "paired_vs_legacy": _paired(selected_rows, locked["baseline"]),
            "crowd_reduction_vs_legacy": crowd_reduction,
        },
        "conclusion": {
            "win_rate_improved": selected_summary["grade_2_7_rate"] > baseline_summary["grade_2_7_rate"],
            "win_rate_significantly_improved": bool(
                selected_summary["grade_2_7_rate"] > baseline_summary["grade_2_7_rate"]
                and _paired(selected_rows, locked["baseline"])["one_sided_p"] < 0.05
            ),
            "crowd_risk_reduced": crowd_reduction > 0,
            "control_hypothesis_supported": validation.get("control_hypothesis_supported", False),
            "deployment_rule": "低人气仅作为同等开奖评分的轻量择优，不改变开奖概率",
        },
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "market_model_backtest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_asset = {
        "version": result["model_version"],
        "market_rows": len(market_rows),
        "locked_draws": selected_summary["draws"],
        "locked_range": result["data"]["locked_test_range"],
        "selected_weight": selected_weight,
        "baseline_rate": baseline_summary["grade_2_7_rate"],
        "payout_aware_rate": selected_summary["grade_2_7_rate"],
        "grade_2_5_hits": selected_summary["grade_2_5_hits"],
        "paired_p": result["locked_test"]["paired_vs_legacy"]["one_sided_p"],
        "crowd_reduction": crowd_reduction,
        "control_hypothesis_supported": validation.get("control_hypothesis_supported", False),
        "control_hypothesis_p": validation.get("control_hypothesis_two_sided_p", 1.0),
    }
    (ROOT / "assets" / "dlt_market_validation.json").write_text(
        json.dumps(validation_asset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 大乐透官方销量人气层回测",
        "",
        f"- 官方市场数据：{len(market_rows)}期（{market_rows[0]['period']}–{market_rows[-1]['period']}）。",
        f"- 开发段：{result['data']['development_range'][0]}–{result['data']['development_range'][1]}，锁定测试：{result['data']['locked_test_range'][0]}–{result['data']['locked_test_range'][1]}。",
        f"- 选定人气权重：{selected_weight:.2f}。",
        f"- 原模型2–7等奖率：{baseline_summary['grade_2_7_rate']:.2%}；新模型：{selected_summary['grade_2_7_rate']:.2%}。",
        f"- 新模型相对原模型的人气风险变化：{-crowd_reduction:.2%}（负数表示降低）。",
        f"- 配对单侧检验p={result['locked_test']['paired_vs_legacy']['one_sided_p']:.4f}。",
        f"- 实际开奖号人气百分位：{validation.get('actual_draw_mean_popularity_percentile', 0):.2%}，控制假设p={validation.get('control_hypothesis_two_sided_p', 1):.4f}。",
        "",
        "结论：低人气层只服务于降低潜在分奖人数。若中奖率没有显著提升，不将其宣传为开奖预测优势。",
    ]
    (reports / "market_model_backtest.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
