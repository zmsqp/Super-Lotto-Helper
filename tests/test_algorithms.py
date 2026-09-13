# -*- coding: utf-8 -*-
import math
import unittest

from services.algorithms import (
    _select_diverse_fronts,
    compare_tickets,
    comparison_summary,
    ensemble_score_vector,
    generate_recommendations,
)
from services.stats import fair_pool_coverage, prize_grade


class PrizeTests(unittest.TestCase):
    def test_nine_prize_includes_back_pair_only(self):
        self.assertEqual(prize_grade(0, 2), 9)
        self.assertEqual(prize_grade(3, 0), 9)
        self.assertIsNone(prize_grade(1, 1))

    def test_comparison_uses_best_ticket_not_first_ticket(self):
        tickets = [
            {"name": "第1注", "front": [1, 2, 3, 4, 5], "back": [1, 2]},
            {"name": "第2注", "front": [6, 7, 8, 20, 21], "back": [3, 4]},
        ]
        draw = {"front": [6, 7, 8, 30, 31], "back": [9, 10]}
        compared, front_hits, back_hits = compare_tickets(tickets, draw)
        self.assertEqual((front_hits, back_hits), (3, 0))
        self.assertEqual(compared[1]["grade"], 9)

    def test_pool_and_ticket_hits_are_separate(self):
        tickets = [
            {"front": [1, 6, 11, 16, 21], "back": [1, 6]},
            {"front": [2, 7, 12, 17, 22], "back": [2, 7]},
        ]
        draw = {"front": [1, 7, 30, 31, 32], "back": [1, 7]}
        summary = comparison_summary(tickets, draw)
        self.assertEqual(summary["best_front"], 1)
        self.assertEqual(summary["best_back"], 1)
        self.assertEqual(summary["front_pool_hits"], 2)
        self.assertEqual(summary["back_pool_hits"], 2)


class BaselineTests(unittest.TestCase):
    def test_front_seven_pool_baseline(self):
        expected = 1 - (math.comb(28, 5) + 7 * math.comb(28, 4)) / math.comb(35, 5)
        self.assertAlmostEqual(fair_pool_coverage(7, 35, 5, 2), expected)

    def test_back_ten_pool_baseline(self):
        self.assertAlmostEqual(fair_pool_coverage(10, 12, 2, 2), 45 / 66)


class ModelTests(unittest.TestCase):
    @staticmethod
    def sample_draws(count=80):
        draws = []
        for index in range(count):
            front = sorted({(index * 3 + offset * 7) % 35 + 1 for offset in range(5)})
            while len(front) < 5:
                front.append(max(front) % 35 + 1)
                front = sorted(set(front))
            back = sorted({index % 12 + 1, (index + 5) % 12 + 1})
            draws.append({"period": str(index), "date": "2026-01-01", "front": front, "back": back})
        return draws

    def test_ensemble_is_deterministic_and_normalized(self):
        scores_a, detail_a = ensemble_score_vector(self.sample_draws(), "front", 35)
        scores_b, detail_b = ensemble_score_vector(self.sample_draws(), "front", 35)
        self.assertEqual(scores_a, scores_b)
        self.assertEqual(detail_a, detail_b)
        self.assertGreaterEqual(min(scores_a[1:]), 0.0)
        self.assertLessEqual(max(scores_a[1:]), 1.0)

    def test_portfolio_selection_is_unique_and_diverse(self):
        pool = [1, 2, 13, 14, 25, 26, 27]
        candidates = [list(item) for item in __import__("itertools").combinations(pool, 5)]
        scores = [0.0] + [1.0 - number / 100 for number in range(1, 36)]
        selected = _select_diverse_fronts(candidates, scores, 5, {})
        fronts = [tuple(row[0]) for row in selected]
        self.assertEqual(len(fronts), 5)
        self.assertEqual(len(set(fronts)), 5)
        self.assertGreaterEqual(len(set().union(*(set(front) for front in fronts))), 7)
        self.assertLess(max(sum(number in front for front in fronts) for number in pool), 5)

    def test_ticket_tiers_ignore_manual_focus_boost(self):
        draws = self.sample_draws()
        raw_scores, _ = ensemble_score_vector(draws, "front", 35)
        ranking = sorted(range(1, 36), key=lambda n: (raw_scores[n], -n), reverse=True)
        high, low, cold = set(ranking[:10]), set(ranking[10:25]), set(ranking[25:])
        focus = sorted(cold)[:7]
        result = generate_recommendations(
            draws,
            {"front": {"focus": focus, "exclude": [1, 2]}, "back": {"focus": [], "exclude": []}},
            ticket_count=5,
            constrain_sum=False,
        )
        self.assertEqual(result["model"]["front_ranking"], ranking)
        for ticket in result["tickets"]:
            front = set(ticket["front"])
            self.assertEqual(
                ticket["tier_pattern"],
                [len(front & high), len(front & low), len(front & cold)],
            )


if __name__ == "__main__":
    unittest.main()
