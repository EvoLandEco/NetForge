import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from temporal_sbm.sweep import expand_generation_grid, pick_best_primary_setting


class SweepTests(unittest.TestCase):
    def test_expand_generation_grid_builds_cross_product(self):
        settings = expand_generation_grid(
            {
                "samplers": ["micro", "canonical_ml"],
                "rewires": ["none", "configuration"],
            }
        )

        self.assertEqual(
            [setting.label for setting in settings],
            [
                "micro__rewire_none",
                "micro__rewire_configuration",
                "canonical_ml__rewire_none",
                "canonical_ml__rewire_configuration",
            ],
        )

    def test_pick_best_primary_setting_ranks_by_overlap_then_weight_then_novelty(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            summary = pd.DataFrame(
                {
                    "sample_label": [
                        "micro__rewire_none",
                        "canonical_ml__rewire_none",
                        "micro__rewire_configuration",
                    ],
                    "mean_snapshot_edge_jaccard": [0.70, 0.70, 0.95],
                    "weight_total_correlation": [0.90, 0.95, 0.99],
                    "mean_synthetic_novel_edge_rate": [0.10, 0.12, 0.20],
                }
            )
            summary.to_csv(diagnostics_dir / "setting_posterior_summary.csv", index=False)

            output_path = diagnostics_dir / "best_primary_setting.txt"
            best = pick_best_primary_setting(run_dir, output_path)
            written = output_path.read_text().strip()

        self.assertEqual(best, "canonical_ml__rewire_none")
        self.assertEqual(written, "canonical_ml__rewire_none")


if __name__ == "__main__":
    unittest.main()
