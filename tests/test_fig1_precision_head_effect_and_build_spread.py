from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.fig1_precision_head_effect_and_build_spread import (
    FIGURE_ID,
    generate,
    load_inputs,
)


REPO = Path(__file__).resolve().parents[1]


class FigureOneTests(unittest.TestCase):
    def test_committed_rows_match_the_figure_contract(self):
        contrasts, builds = load_inputs(REPO)
        self.assertEqual(contrasts["size"].tolist(), ["all", "xs", "s", "m", "l", "xl"])
        self.assertEqual(builds["size"].tolist(), ["all", "xs", "s", "m", "l", "xl"])
        self.assertTrue((contrasts["ci95_low_pp"] > 0).all())
        self.assertTrue((contrasts["valid_resamples"] == 1000).all())
        self.assertTrue((contrasts["undefined_resamples"] == 0).all())
        self.assertTrue((builds["builds"] == 3).all())

    def test_primary_effect_matches_the_source_table(self):
        contrasts, _ = load_inputs(REPO)
        full = contrasts.iloc[0]
        self.assertAlmostEqual(float(full["point_delta_pp"]), 8.506713, places=6)
        self.assertAlmostEqual(float(full["ci95_low_pp"]), 7.710367, places=6)
        self.assertAlmostEqual(float(full["ci95_high_pp"]), 8.999865, places=6)

    def test_bounded_export_writes_review_labeled_three_formats(self):
        with tempfile.TemporaryDirectory(prefix="fig1-paper-core-") as td:
            out_dir = Path(td)
            manifest = generate(REPO, out_dir)
            self.assertEqual(manifest["status"], "rendered_draft_review_required")
            self.assertEqual(manifest["checks"]["canonical_icarus_critique_gate"], "unavailable_not_run")
            self.assertEqual(set(manifest["outputs"]), {"pdf", "svg", "png"})
            for item in manifest["outputs"].values():
                p = Path(item["path"])
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 1000)
            svg = (out_dir / f"{FIGURE_ID}.svg").read_text(encoding="utf-8")
            self.assertIn("<text", svg)
            self.assertIn("no CI", svg)
            self.assertTrue(all(line == line.rstrip() for line in svg.splitlines()))
            emitted = json.loads((out_dir / f"{FIGURE_ID}_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(emitted["checks"]["release_label"], "rendered_draft_review_required")
            with (out_dir / "figure_manifest.csv").open(encoding="utf-8") as f:
                self.assertIn(FIGURE_ID, f.read())

    def test_three_format_exports_are_byte_stable(self):
        with tempfile.TemporaryDirectory(prefix="fig1-repeatability-") as td:
            out_dir = Path(td)
            first = generate(REPO, out_dir)
            hashes_first = {k: v["sha256"] for k, v in first["outputs"].items()}
            second = generate(REPO, out_dir)
            hashes_second = {k: v["sha256"] for k, v in second["outputs"].items()}
            self.assertEqual(hashes_first, hashes_second)


if __name__ == "__main__":
    unittest.main()
