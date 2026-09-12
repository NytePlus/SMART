import importlib.util
import random
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_metric_statistics.py"
SPEC = importlib.util.spec_from_file_location("paired_metric_statistics", SCRIPT)
assert SPEC and SPEC.loader
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


class PairedMetricStatisticsTests(unittest.TestCase):
    def test_wer_turnover_preserves_accounting_identity(self):
        reference = ["a", "b", "c"]
        old = STATS._alignment_error_events(reference, ["a", "x", "c", "z"])
        new = STATS._alignment_error_events(reference, ["a", "b", "y"])
        repaired, introduced = STATS.event_turnover(old, new)

        self.assertEqual(sum(old.values()), 2)
        self.assertEqual(sum(new.values()), 1)
        self.assertEqual((repaired, introduced), (2, 1))
        self.assertEqual(sum(new.values()), sum(old.values()) - repaired + introduced)

    def test_semantic_turnover_matches_overlapping_reference_spans(self):
        repaired, introduced = STATS.semantic_turnover(
            [(10, 20), (10, 20), (30, 35)],
            [(12, 18), (40, 45)],
        )

        self.assertEqual((repaired, introduced), (2, 1))

    def test_statistics_reports_repaired_and_new_rates(self):
        result = STATS.paired_statistics(
            old_num=[2, 1],
            new_num=[1, 2],
            repaired_num=[1, 0],
            introduced_num=[0, 1],
            denominator=[10, 10],
            rng=random.Random(1),
            samples=100,
        )

        self.assertEqual(result["old"], 15.0)
        self.assertEqual(result["after"], 15.0)
        self.assertEqual(result["repaired"], 5.0)
        self.assertEqual(result["new"], 5.0)
        self.assertEqual(result["repaired_error_units"], 1)
        self.assertEqual(result["new_error_units"], 1)


if __name__ == "__main__":
    unittest.main()
