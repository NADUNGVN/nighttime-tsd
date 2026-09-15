import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_precision_head_paired import (  # noqa: E402
    CONTRASTS,
    LABELS,
    MODELS,
    build_samples,
    canonical_sha256,
    paired_contrast_summary,
    require_equal,
)


class PrecisionHeadPairedAnalysisTests(unittest.TestCase):
    def test_locked_sample_plan_is_sorted_and_retains_duplicates(self):
        samples = build_samples(4, 3, 7)
        self.assertEqual(samples.shape, (3, 4))
        for row in samples:
            self.assertTrue(np.all(row[:-1] <= row[1:]))
            self.assertTrue(np.any(np.bincount(row, minlength=4) > 1))

    def test_identical_arms_have_zero_paired_delta_for_all_ten_contrasts(self):
        draws = np.tile(np.arange(5, dtype=float)[:, None, None, None] / 10.0, (1, len(MODELS), len(LABELS), 2))
        points = draws[0]
        summary = paired_contrast_summary(draws, points)
        self.assertEqual(len(summary), 10)
        self.assertEqual(set(summary), {name for name, _, _ in CONTRASTS})
        for contrast in summary.values():
            for size in contrast.values():
                for metric in size.values():
                    self.assertEqual(metric["point_delta_pp"], 0.0)
                    self.assertEqual(metric["ci95_percentile_pp"], [0.0, 0.0])

    def test_all_fixed_contrasts_are_present_in_declared_order(self):
        self.assertEqual(
            [name for name, _, _ in CONTRASTS],
            [
                "bbox_minus_baseline", "classification_minus_baseline", "both_minus_baseline",
                "bbox_minus_classification", "both_minus_bbox", "both_minus_classification",
                "baseline_minus_fp16", "bbox_minus_fp16", "classification_minus_fp16", "both_minus_fp16",
            ],
        )

    def test_changed_input_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            require_equal("changed", "expected", "capture input")

    def test_missing_canonical_input_is_rejected(self):
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            canonical_sha256(repo, repo / "results/measurement_audit_v1/does_not_exist.json")


if __name__ == "__main__":
    unittest.main()
