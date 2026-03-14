import json
import os
import unittest
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from temporal_sbm.simulation import (
    HybridPanelPack,
    HybridSimulationConfig,
    _build_seed_pool,
    _filter_sample_manifests,
    _write_region_geo_html,
    aggregate_posterior_reports,
    write_report,
    write_scenario_comparison_report,
    write_scientific_validation_report,
)

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "netforge-test-mpl"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


class SimulationTests(unittest.TestCase):
    def test_filter_sample_manifests_can_keep_requested_setting(self):
        sample_manifests = [
            {"setting_label": "alpha", "sample_label": "alpha__sample_0000"},
            {"setting_label": "beta", "sample_label": "beta__sample_0000"},
        ]

        filtered = _filter_sample_manifests(sample_manifests, setting_labels=["beta"])

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["setting_label"], "beta")

    def test_build_seed_pool_uses_common_day0_and_all_farms_modes(self):
        pack = HybridPanelPack(
            label="observed",
            node_universe=(0, 1, 2, 3),
            ts_values=(10, 11),
            src=(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            dst=(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            weight=(np.array([], dtype=float), np.array([], dtype=float)),
            active_by_ts=(
                np.array([True, False, True, False], dtype=bool),
                np.array([False, True, True, False], dtype=bool),
            ),
            is_farm=np.array([True, True, True, False], dtype=bool),
            is_region=np.array([False, False, False, True], dtype=bool),
        )

        common_day0_pool = _build_seed_pool(
            observed_pack=pack,
            synthetic_day0_activity_mask=np.array([True, True, False, False], dtype=bool),
            config=HybridSimulationConfig(seed_pool_mode="common_day0"),
        )
        all_farms_pool = _build_seed_pool(
            observed_pack=pack,
            synthetic_day0_activity_mask=np.array([True, True, False, False], dtype=bool),
            config=HybridSimulationConfig(seed_scope="all_farms"),
        )

        self.assertEqual(common_day0_pool.tolist(), [0])
        self.assertEqual(all_farms_pool.tolist(), [0, 1, 2])

    def test_aggregate_posterior_reports_keeps_rewire_none_as_posterior_predictive(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            per_snapshot_a = tmp_path / "run_a_per_snapshot.csv"
            per_snapshot_b = tmp_path / "run_b_per_snapshot.csv"
            pd.DataFrame(
                {
                    "day_index": [0],
                    "ts": [10],
                    "original_farm_prevalence": [2.0],
                    "synthetic_farm_prevalence": [2.0],
                }
            ).to_csv(per_snapshot_a, index=False)
            pd.DataFrame(
                {
                    "day_index": [0],
                    "ts": [10],
                    "original_farm_prevalence": [2.0],
                    "synthetic_farm_prevalence": [3.0],
                }
            ).to_csv(per_snapshot_b, index=False)

            reports = [
                {
                    "sample_label": "maxent_micro__rewire_none__sample_0000",
                    "summary": {
                        "sample_label": "maxent_micro__rewire_none__sample_0000",
                        "setting_label": "maxent_micro__rewire_none",
                        "sample_class": "posterior_predictive",
                        "farm_prevalence_curve_correlation": 0.9,
                        "farm_incidence_curve_correlation": 0.8,
                        "farm_attack_rate_wasserstein": 0.1,
                        "farm_peak_prevalence_wasserstein": 0.2,
                        "farm_duration_wasserstein": 0.3,
                    },
                    "outputs": {"per_snapshot_csv": str(per_snapshot_a)},
                },
                {
                    "sample_label": "maxent_micro__rewire_none__sample_0001",
                    "summary": {
                        "sample_label": "maxent_micro__rewire_none__sample_0001",
                        "setting_label": "maxent_micro__rewire_none",
                        "sample_class": "posterior_predictive",
                        "farm_prevalence_curve_correlation": 0.85,
                        "farm_incidence_curve_correlation": 0.75,
                        "farm_attack_rate_wasserstein": 0.12,
                        "farm_peak_prevalence_wasserstein": 0.22,
                        "farm_duration_wasserstein": 0.35,
                    },
                    "outputs": {"per_snapshot_csv": str(per_snapshot_b)},
                },
            ]

            aggregated = aggregate_posterior_reports(
                reports,
                output_dir=tmp_path,
                setting_label="maxent_micro__rewire_none",
            )

            self.assertEqual(aggregated["sample_class"], "posterior_predictive")

    def test_write_scientific_validation_report_reads_custom_simulation_dir(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            simulation_dir = run_dir / "custom_simulation"
            run_dir.mkdir(parents=True, exist_ok=True)
            simulation_dir.mkdir(parents=True, exist_ok=True)

            manifest = {"dataset": "CR35", "directed": True}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pd.DataFrame(
                {
                    "sample_label": ["maxent_micro__rewire_none"],
                    "sample_class": ["posterior_predictive"],
                    "farm_prevalence_curve_correlation": [0.9],
                    "farm_incidence_curve_correlation": [0.8],
                    "farm_attack_rate_wasserstein": [0.1],
                    "farm_peak_prevalence_wasserstein": [0.2],
                    "farm_duration_wasserstein": [0.3],
                }
            ).to_csv(simulation_dir / "setting_posterior_summary.csv", index=False)

            report_path = write_scientific_validation_report(
                run_dir,
                simulation_dir=simulation_dir,
                output_path=simulation_dir / "scientific_validation_report.html",
            )

            self.assertTrue(report_path.exists())
            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("How to read this figure", report_html)
            self.assertIn("How to read this table", report_html)

    def test_write_scientific_validation_report_includes_run_details_when_both_summary_csvs_exist(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            simulation_dir = run_dir / "custom_simulation"
            run_dir.mkdir(parents=True, exist_ok=True)
            simulation_dir.mkdir(parents=True, exist_ok=True)

            manifest = {"dataset": "CR35", "directed": True}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pd.DataFrame(
                {
                    "sample_label": ["maxent_micro__rewire_none"],
                    "sample_class": ["posterior_predictive"],
                    "posterior_num_runs": [3],
                    "farm_prevalence_curve_correlation": [0.9],
                    "farm_incidence_curve_correlation": [0.8],
                    "farm_attack_rate_wasserstein": [0.1],
                    "farm_peak_prevalence_wasserstein": [0.2],
                    "farm_duration_wasserstein": [0.3],
                }
            ).to_csv(simulation_dir / "setting_posterior_summary.csv", index=False)
            pd.DataFrame(
                {
                    "sample_label": [
                        "maxent_micro__rewire_none__sample_0000",
                        "maxent_micro__rewire_none__sample_0001",
                        "maxent_micro__rewire_none__sample_0002",
                    ],
                    "sample_class": ["posterior_predictive"] * 3,
                    "farm_prevalence_curve_correlation": [0.88, 0.86, 0.83],
                    "farm_incidence_curve_correlation": [0.79, 0.77, 0.76],
                    "farm_attack_rate_wasserstein": [0.11, 0.12, 0.14],
                    "farm_peak_prevalence_wasserstein": [0.21, 0.23, 0.25],
                    "farm_duration_wasserstein": [0.31, 0.35, 0.38],
                }
            ).to_csv(simulation_dir / "all_samples_summary.csv", index=False)
            for sample_label in [
                "maxent_micro__rewire_none",
                "maxent_micro__rewire_none__sample_0000",
                "maxent_micro__rewire_none__sample_0001",
                "maxent_micro__rewire_none__sample_0002",
            ]:
                (simulation_dir / f"{sample_label}_region_geo_compare.html").write_text(
                    "<html><body>placeholder</body></html>",
                    encoding="utf-8",
                )

            report_path = write_scientific_validation_report(
                run_dir,
                simulation_dir=simulation_dir,
                output_path=simulation_dir / "scientific_validation_report.html",
            )

            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("<h3>Settings</h3><div class='metric-value'>1</div>", report_html)
            self.assertIn("<h3>Run details</h3><div class='metric-value'>3</div>", report_html)
            self.assertIn("Best curve fit: maxent_micro__rewire_none", report_html)
            self.assertIn("Run detail: maxent_micro__rewire_none__sample_0000", report_html)
            self.assertIn("Run detail: maxent_micro__rewire_none__sample_0002", report_html)

    def test_write_scientific_validation_report_uses_responsive_table_and_figure_cards(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            simulation_dir = run_dir / "custom_simulation"
            run_dir.mkdir(parents=True, exist_ok=True)
            simulation_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "manifest.json").write_text(json.dumps({"dataset": "CR35", "directed": True}), encoding="utf-8")
            pd.DataFrame(
                {
                    "sample_label": ["maxent_micro__rewire_none"],
                    "sample_class": ["posterior_predictive"],
                    "posterior_num_runs": [3],
                    "farm_prevalence_curve_correlation": [0.9],
                    "farm_incidence_curve_correlation": [0.8],
                    "farm_attack_rate_wasserstein": [0.1],
                    "farm_peak_prevalence_wasserstein": [0.2],
                    "farm_duration_wasserstein": [0.3],
                    "region_reservoir_spatial_correlation_mean": [0.7],
                }
            ).to_csv(simulation_dir / "setting_posterior_summary.csv", index=False)

            report_path = write_scientific_validation_report(
                run_dir,
                simulation_dir=simulation_dir,
                output_path=simulation_dir / "scientific_validation_report.html",
            )

            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn(".summary-card, .figure-card, .artifact-card, .table-card { min-width: 0; }", report_html)
            self.assertIn(".figure-frame { min-width: 0; overflow: hidden;", report_html)
            self.assertIn(".table-wrap { max-width: 100%; overflow: auto;", report_html)
            self.assertIn("<div class='table-card'><div class='table-wrap'><table class='report-table'>", report_html)

    def test_write_region_geo_html_uses_stacked_metric_sections_and_scale_switch(self):
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "region_geo_compare.html"
            payload = {
                "dataset": "CR35",
                "sample_label": "maxent_micro__rewire_none",
                "focal_corop": "CR35",
                "calendar": [{"ts": 1, "label": "2018-01-01"}],
                "metrics": ["reservoir_pressure", "import_pressure", "export_pressure"],
                "observed": [{"ts": 1, "corop": "CR35", "reservoir_pressure": 0.000321, "import_pressure": 0.000042, "export_pressure": 0.000017}],
                "synthetic": [{"ts": 1, "corop": "CR35", "reservoir_pressure": 0.000654, "import_pressure": 0.000055, "export_pressure": 0.000022}],
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"statcode": "CR35"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[4.0, 52.0], [4.1, 52.0], [4.1, 52.1], [4.0, 52.1], [4.0, 52.0]]],
                            },
                        }
                    ],
                },
            }

            written_path = _write_region_geo_html(payload, output_path)

            self.assertEqual(written_path, output_path)
            html_text = output_path.read_text(encoding="utf-8")
            self.assertIn("Color mapping", html_text)
            self.assertIn('option value="sqrt" selected', html_text)
            self.assertIn("Reservoir pressure", html_text)
            self.assertIn("Synthetic", html_text)
            self.assertIn("Delta", html_text)
            self.assertIn("synthetic minus observed", html_text)
            self.assertIn("scaleModeSelect", html_text)
            self.assertIn("model-scale hazard accumulators", html_text)
            self.assertIn("toSuperscriptExponent", html_text)
            self.assertTrue(output_path.with_name("region_geo_compare_payload.js").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_report_creates_delta_and_distribution_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            per_snapshot = pd.DataFrame(
                {
                    "day_index": [0, 1],
                    "ts": [10, 11],
                    "original_farm_prevalence": [1.0, 2.0],
                    "synthetic_farm_prevalence": [1.0, 3.0],
                    "original_farm_incidence": [1.0, 1.0],
                    "synthetic_farm_incidence": [1.0, 2.0],
                    "original_farm_incidence_mean": [0.8, 1.1],
                    "synthetic_farm_incidence_mean": [0.9, 1.6],
                    "original_farm_incidence_q05": [0.0, 0.0],
                    "synthetic_farm_incidence_q05": [0.0, 0.0],
                    "original_farm_incidence_q95": [1.0, 2.0],
                    "synthetic_farm_incidence_q95": [1.0, 3.0],
                    "original_farm_infection_event_probability": [0.35, 0.45],
                    "synthetic_farm_infection_event_probability": [0.40, 0.65],
                    "original_farm_cumulative_incidence": [1.0, 2.0],
                    "synthetic_farm_cumulative_incidence": [1.0, 3.0],
                    "original_reservoir_total": [0.5, 0.8],
                    "synthetic_reservoir_total": [0.4, 1.0],
                    "original_reservoir_max": [0.2, 0.3],
                    "synthetic_reservoir_max": [0.2, 0.4],
                    "original_reservoir_positive_regions": [1.0, 2.0],
                    "synthetic_reservoir_positive_regions": [1.0, 3.0],
                    "farm_prevalence_delta": [0.0, 1.0],
                    "farm_incidence_delta": [0.0, 1.0],
                    "farm_cumulative_incidence_delta": [0.0, 1.0],
                    "reservoir_total_delta": [-0.1, 0.2],
                }
            )
            summary = {
                "farm_prevalence_curve_correlation": 0.9,
                "farm_incidence_curve_correlation": 0.8,
                "farm_cumulative_incidence_curve_correlation": 0.85,
                "reservoir_total_curve_correlation": 0.7,
                "reservoir_max_curve_correlation": 0.6,
                "reservoir_positive_regions_curve_correlation": 0.75,
                "farm_attack_rate_wasserstein": 0.1,
                "farm_peak_prevalence_wasserstein": 0.2,
                "farm_peak_day_wasserstein": 0.3,
                "farm_duration_wasserstein": 0.4,
                "simulation_config": {"model": "SEIR"},
            }
            observed_outcomes = pd.DataFrame(
                {
                    "farm_attack_rate": [0.1, 0.2, 0.3],
                    "farm_peak_prevalence": [2.0, 3.0, 4.0],
                    "farm_peak_day_index": [1.0, 2.0, 3.0],
                    "farm_duration_days": [4.0, 5.0, 6.0],
                }
            )
            synthetic_outcomes = pd.DataFrame(
                {
                    "farm_attack_rate": [0.2, 0.3, 0.4],
                    "farm_peak_prevalence": [3.0, 4.0, 5.0],
                    "farm_peak_day_index": [2.0, 3.0, 4.0],
                    "farm_duration_days": [5.0, 6.0, 7.0],
                }
            )
            outcome_summary = pd.DataFrame(
                {
                    "metric": [
                        "farm_attack_rate",
                        "farm_peak_prevalence",
                        "farm_peak_day_index",
                        "farm_duration_days",
                        "farm_cumulative_incidence",
                        "farm_prevalence_auc",
                        "reservoir_total_auc",
                        "reservoir_max_peak",
                    ],
                    "wasserstein_distance": [0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.3, 0.2],
                    "original_median": [0.2, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 0.4],
                    "synthetic_median": [0.3, 4.0, 3.0, 6.0, 3.0, 4.0, 1.5, 0.5],
                }
            )

            outputs = write_report(
                per_snapshot,
                summary,
                output_dir,
                "sample_a",
                detailed_outputs={
                    "observed_outcomes": observed_outcomes,
                    "synthetic_outcomes": synthetic_outcomes,
                    "outcome_distribution_summary": outcome_summary,
                },
            )

            self.assertTrue(Path(outputs["dashboard_png"]).exists())
            self.assertTrue(Path(outputs["delta_png"]).exists())
            self.assertTrue(Path(outputs["distribution_png"]).exists())
            self.assertTrue(Path(outputs["parity_png"]).exists())
            report_md = Path(outputs["report_md"]).read_text(encoding="utf-8")
            self.assertIn("## How to read the figures", report_md)
            self.assertIn("## How to read the tables", report_md)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_scenario_comparison_report_creates_html(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            scenario_dir = run_dir / "scenario_outputs"
            scenario_output = scenario_dir / "baseline"
            run_dir.mkdir(parents=True, exist_ok=True)
            scenario_output.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps({"dataset": "CR35"}), encoding="utf-8")
            summary_rows = pd.DataFrame(
                {
                    "scenario_name": ["baseline"],
                    "scenario_description": ["Default scenario"],
                    "sample_label": ["baseline"],
                    "selected_setting_label": ["maxent_micro__rewire_none"],
                    "selected_sample_label": ["maxent_micro__rewire_none"],
                    "output_dir": [str(scenario_output)],
                    "report_path": [str(scenario_output / "scientific_validation_report.html")],
                    "farm_prevalence_curve_correlation": [0.9],
                    "farm_incidence_curve_correlation": [0.8],
                    "farm_cumulative_incidence_curve_correlation": [0.85],
                    "reservoir_total_curve_correlation": [0.7],
                    "farm_attack_rate_wasserstein": [0.1],
                    "farm_peak_prevalence_wasserstein": [0.2],
                    "farm_duration_wasserstein": [0.3],
                    "farm_prevalence_interval_coverage": [1.0],
                    "farm_incidence_interval_coverage": [1.0],
                    "farm_cumulative_incidence_interval_coverage": [0.9],
                    "reservoir_total_interval_coverage": [0.8],
                    "farm_attack_rate_observed_median_in_pooled_synthetic_90pct": [1.0],
                    "farm_attack_rate_observed_median_pooled_tail_area": [0.4],
                    "farm_peak_prevalence_observed_median_in_pooled_synthetic_90pct": [1.0],
                    "farm_peak_prevalence_observed_median_pooled_tail_area": [0.3],
                    "farm_peak_day_observed_median_in_pooled_synthetic_90pct": [1.0],
                    "farm_peak_day_observed_median_pooled_tail_area": [0.5],
                    "farm_duration_observed_median_in_pooled_synthetic_90pct": [0.0],
                    "farm_duration_observed_median_pooled_tail_area": [0.1],
                    "farm_attack_rate_network_uncertainty_share": [0.2],
                    "farm_peak_prevalence_network_uncertainty_share": [0.3],
                    "farm_peak_day_network_uncertainty_share": [0.4],
                    "farm_duration_network_uncertainty_share": [0.5],
                    "farm_cumulative_incidence_network_uncertainty_share": [0.6],
                }
            )

            report_path = write_scenario_comparison_report(run_dir, scenario_dir, summary_rows)

            self.assertTrue(report_path.exists())
            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("baseline", report_html)
            self.assertIn("How to read this figure", report_html)
            self.assertIn("How to read this table", report_html)
            self.assertIn("Daily calibration", report_html)
            self.assertIn("Scalar calibration and uncertainty", report_html)


if __name__ == "__main__":
    unittest.main()
