import os
import unittest
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from temporal_sbm.diagnostics import (
    _build_hybrid_network_summary_table,
    _write_daily_network_snapshot_assets,
    _complete_entity_time_series,
    _compute_pi_mass_time_series,
    _load_sweep_summary_rows,
    _merge_entity_time_series,
    aggregate_posterior_reports,
    canonicalise_edge_frame,
    compare_panels,
    compare_panels_detailed,
    load_node_blocks,
    _summarise_metric_time_series,
    summary_payload_to_row,
    write_all_samples_overview,
    write_log_visual_summary,
    write_report,
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


class DiagnosticsTests(unittest.TestCase):
    def test_compute_pi_mass_time_series_marks_singleton_lazy_component_as_missing(self):
        frame = pd.DataFrame(
            {
                "u": [0],
                "i": [0],
                "ts": [10],
                "weight": [1.0],
            }
        )

        result = _compute_pi_mass_time_series(
            frame,
            node_universe=[0, 1],
            node_types={0: "Farm", 1: "Region"},
            directed=True,
            weight_col="weight",
            mode="largest_component_lazy",
        )

        self.assertEqual(float(result.loc[0, "lic_size"]), 1.0)
        self.assertTrue(pd.isna(result.loc[0, "pi_mass__farm"]))
        self.assertTrue(pd.isna(result.loc[0, "pi_mass__region"]))
        self.assertTrue(pd.isna(result.loc[0, "pi_gini"]))

    def test_summarise_metric_time_series_can_ignore_missing_pi_mass_days(self):
        original = pd.DataFrame({"ts": [10, 11, 12], "pi_mass__farm": [float("nan"), 0.2, 0.4]})
        synthetic = pd.DataFrame({"ts": [10, 11, 12], "pi_mass__farm": [float("nan"), 0.1, 0.5]})

        merged = _merge_entity_time_series(original, synthetic, ["ts"], fill_value=None)
        summary = _summarise_metric_time_series(merged, ["pi_mass__farm"], treat_missing_as_zero=False)

        self.assertTrue(pd.isna(merged.loc[0, "pi_mass__farm_delta"]))
        self.assertAlmostEqual(float(summary.loc[0, "original_mean"]), 0.3, places=6)
        self.assertAlmostEqual(float(summary.loc[0, "synthetic_mean"]), 0.3, places=6)
        self.assertAlmostEqual(float(summary.loc[0, "correlation"]), 1.0, places=6)

    def test_compare_panels_detailed_tracks_active_counts_in_pi_mass_summary(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 2],
                "i": [1, 2, 3],
                "ts": [10, 10, 11],
                "weight": [1.0, 2.0, 3.0],
            }
        )
        synthetic = original.copy()

        comparison = compare_panels_detailed(
            original_df=original,
            synthetic_df=synthetic,
            directed=True,
            weight_col="weight",
            node_types={0: "Farm", 1: "Region", 2: "Farm", 3: "Region"},
        )

        pi_summary = comparison["details"]["pi_mass_summary"]
        self.assertIn("active_node_count", pi_summary["metric"].tolist())
        self.assertIn("active_farm_count", pi_summary["metric"].tolist())
        self.assertIn("active_region_count", pi_summary["metric"].tolist())
        self.assertAlmostEqual(
            float(pi_summary.loc[pi_summary["metric"] == "active_node_count", "correlation"].iloc[0]),
            1.0,
            places=6,
        )
        self.assertIn("lic_active_node_count_correlation", comparison["summary"])
        self.assertIn("lic_active_farm_count_correlation", comparison["summary"])
        self.assertIn("lic_active_region_count_correlation", comparison["summary"])

    def test_complete_entity_time_series_fills_missing_snapshot_rows_with_zero(self):
        merged = pd.DataFrame(
            {
                "ts": [10, 11],
                "node_id": [1, 2],
                "original_out_edge_count": [2.0, 3.0],
                "synthetic_out_edge_count": [1.0, 4.0],
                "out_edge_count_delta": [-1.0, 1.0],
            }
        )

        completed = _complete_entity_time_series(merged, entity_keys=["node_id"])

        self.assertEqual(completed["ts"].tolist(), [10, 10, 11, 11])
        self.assertEqual(completed["node_id"].tolist(), [1, 2, 1, 2])
        node_one = completed.loc[completed["node_id"] == 1].sort_values("ts")
        self.assertEqual(node_one["original_out_edge_count"].tolist(), [2.0, 0.0])
        self.assertEqual(node_one["synthetic_out_edge_count"].tolist(), [1.0, 0.0])

    def test_canonicalise_undirected_edges(self):
        frame = pd.DataFrame(
            {
                "src": [2, 1, 2],
                "dst": [1, 2, 1],
                "time": [5, 5, 6],
            }
        )
        result = canonicalise_edge_frame(frame, directed=False, src_col="src", dst_col="dst", ts_col="time")
        self.assertEqual(result.to_dict(orient="records"), [{"u": 1, "i": 2, "ts": 5}, {"u": 1, "i": 2, "ts": 6}])

    def test_compare_panels_identical(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
            }
        )
        synthetic = original.copy()
        per_snapshot, summary = compare_panels(original, synthetic, directed=True)
        self.assertTrue((per_snapshot["edge_jaccard"] == 1.0).all())
        self.assertAlmostEqual(summary["mean_snapshot_edge_jaccard"], 1.0)
        self.assertAlmostEqual(summary["unique_edge_jaccard"], 1.0)

    def test_compare_panels_detailed_returns_block_and_node_outputs(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [2.0, 3.0, 5.0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [1.0, 4.0, 6.0],
            }
        )

        comparison = compare_panels_detailed(
            original_df=original,
            synthetic_df=synthetic,
            directed=False,
            weight_col="weight",
            node_blocks={0: 10, 1: 10, 2: 20},
            node_types={0: "Type 0", 1: "Type 1", 2: "Type 0"},
        )

        self.assertIn("details", comparison)
        self.assertFalse(comparison["details"]["block_pair_summary"].empty)
        self.assertFalse(comparison["details"]["block_activity_summary"].empty)
        self.assertFalse(comparison["details"]["node_activity_summary"].empty)
        self.assertFalse(comparison["details"]["tea_summary"].empty)
        self.assertFalse(comparison["details"]["tna_summary"].empty)
        self.assertFalse(comparison["details"]["pi_mass_summary"].empty)
        self.assertFalse(comparison["details"]["magnetic_laplacian_summary"].empty)
        self.assertIn("tea_new_ratio_correlation", comparison["summary"])
        self.assertIn("pi_mass_mean_correlation", comparison["summary"])
        self.assertIn("magnetic_spectrum_mean_correlation", comparison["summary"])

    def test_compare_panels_detailed_completes_node_activity_over_full_snapshot_span(self):
        original = pd.DataFrame(
            {
                "u": [0, 2],
                "i": [1, 3],
                "ts": [10, 11],
                "weight": [2.0, 3.0],
            }
        )
        synthetic = original.copy()

        comparison = compare_panels_detailed(
            original_df=original,
            synthetic_df=synthetic,
            directed=True,
            weight_col="weight",
            node_blocks={0: 10, 1: 10, 2: 20, 3: 20},
            node_types={0: "Farm", 1: "Farm", 2: "Region", 3: "Region"},
        )

        node_summary = comparison["details"]["node_activity_summary"]
        node_zero = node_summary.loc[node_summary["node_id"] == 0].iloc[0]
        self.assertEqual(int(node_zero["snapshot_count"]), 2)
        node_per_snapshot = comparison["details"]["node_activity_per_snapshot"]
        self.assertEqual(node_per_snapshot.loc[node_per_snapshot["node_id"] == 0, "ts"].tolist(), [10, 11])
        self.assertEqual(
            node_per_snapshot.loc[node_per_snapshot["node_id"] == 0, "original_out_edge_count"].tolist(),
            [1.0, 0.0],
        )

    def test_build_hybrid_network_summary_table_compares_observed_to_best_setting_mean(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            generated_root = run_dir / "generated" / "best_setting"
            for sample_idx, weight in enumerate((5.0, 7.0)):
                sample_dir = generated_root / f"sample_{sample_idx:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "u": [0],
                        "i": [1],
                        "ts": [10],
                        "weight": [weight],
                    }
                ).to_csv(sample_dir / "synthetic_edges.csv", index=False)

            input_edges = pd.DataFrame(
                {
                    "u": [0],
                    "i": [1],
                    "ts": [10],
                    "weight": [3.0],
                }
            )
            hybrid_node_frame = pd.DataFrame(
                {
                    "node_id": [0, 1],
                    "type_label": ["Farm", "Region"],
                }
            )

            table = _build_hybrid_network_summary_table(
                run_dir=run_dir,
                manifest={"weight_model": {"input_column": "weight"}},
                input_edges=input_edges,
                hybrid_node_frame=hybrid_node_frame,
                directed=True,
                best_setting_label="best_setting",
                best_setting_run_labels=["best_setting__sample_0000", "best_setting__sample_0001"],
            )

            self.assertIn("Observed network", table.columns)
            self.assertIn("Best setting mean", table.columns)
            weight_row = table.loc[table["Hybrid summary"] == "F→R weight"].iloc[0]
            self.assertEqual(int(weight_row["Observed network"]), 3)
            self.assertEqual(int(weight_row["Best setting mean"]), 6)

    def test_load_node_blocks_reads_block_ids(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "node_attributes.csv"
            path.write_text("node_id,x,y,block_id\n1,0,0,7\n2,1,1,8\n")

            mapping = load_node_blocks(path)

        self.assertEqual(mapping, {1: 7, 2: 8})

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_report_creates_visual_artifacts(self):
        original = pd.DataFrame({"u": [0, 1], "i": [1, 2], "ts": [10, 11]})
        synthetic = original.copy()
        per_snapshot, summary = compare_panels(original, synthetic, directed=True)

        with TemporaryDirectory() as tmpdir:
            outputs = write_report(per_snapshot, summary, Path(tmpdir), "sample_a")

            self.assertTrue(Path(outputs["per_snapshot_csv"]).exists())
            self.assertTrue(Path(outputs["summary_json"]).exists())
            self.assertTrue(Path(outputs["report_md"]).exists())
            self.assertTrue(Path(outputs["dashboard_png"]).exists())
            self.assertTrue(Path(outputs["parity_png"]).exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_report_handles_undirected_panel_with_reciprocity_columns(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [2.0, 3.0, 5.0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [1.0, 4.0, 6.0],
            }
        )
        comparison = compare_panels_detailed(
            original_df=original,
            synthetic_df=synthetic,
            directed=False,
            weight_col="weight",
            node_blocks={0: 10, 1: 10, 2: 20},
            node_types={0: "Farm", 1: "Region", 2: "Farm"},
        )

        self.assertIn("original_reciprocity", comparison["per_snapshot"].columns)
        self.assertNotIn("original_active_source_node_count", comparison["per_snapshot"].columns)

        with TemporaryDirectory() as tmpdir:
            outputs = write_report(
                comparison["per_snapshot"],
                comparison["summary"],
                Path(tmpdir),
                "sample_undirected",
                directed=False,
            )

            self.assertTrue(Path(outputs["dashboard_png"]).exists())
            self.assertTrue(Path(outputs["parity_png"]).exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_report_creates_detailed_diagnostic_artifacts(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [2.0, 3.0, 5.0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [1.0, 4.0, 6.0],
            }
        )
        comparison = compare_panels_detailed(
            original_df=original,
            synthetic_df=synthetic,
            directed=False,
            weight_col="weight",
            node_blocks={0: 10, 1: 10, 2: 20},
            node_types={0: "Type 0", 1: "Type 1", 2: "Type 0"},
        )

        with TemporaryDirectory() as tmpdir:
            outputs = write_report(
                comparison["per_snapshot"],
                comparison["summary"],
                Path(tmpdir),
                "sample_b",
                detailed_diagnostics=comparison["details"],
                directed=False,
                diagnostic_top_k=4,
            )

            self.assertTrue(Path(outputs["block_pair_summary"]).exists())
            self.assertTrue(Path(outputs["block_pair_edge_plot"]).exists())
            self.assertTrue(Path(outputs["node_activity_summary"]).exists())
            self.assertTrue(Path(outputs["node_activity_weight_plot"]).exists())
            self.assertTrue(Path(outputs["tea_plot"]).exists())
            self.assertTrue(Path(outputs["tna_plot"]).exists())
            self.assertTrue(Path(outputs["pi_mass_plot"]).exists())
            self.assertTrue(Path(outputs["magnetic_laplacian_plot"]).exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_log_visual_summary_creates_dashboard(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "fit.log"
            log_path.write_text(
                "\n".join(
                    [
                        "12:00:00 | DEBUG | Starting fit command | args={}",
                        "12:00:02 | DEBUG | Built layered graph | vertices=5 | edges=4 | layers=2",
                        "12:00:05 | INFO | Fitted layered nested SBM in 0:00:05 | run dir: /tmp/demo",
                    ]
                )
                + "\n"
            )

            outputs = write_log_visual_summary(log_path, root / "logs", label="fit_test")

            self.assertTrue(Path(outputs["summary_json"]).exists())
            self.assertTrue(Path(outputs["report_md"]).exists())
            self.assertTrue(Path(outputs["dashboard_png"]).exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_write_all_samples_overview_creates_png(self):
        summary_rows = pd.DataFrame(
            [
                {
                    "sample_label": "sample_0000",
                    "mean_snapshot_edge_jaccard": 0.1,
                    "mean_snapshot_node_jaccard": 0.2,
                    "mean_synthetic_novel_edge_rate": 0.8,
                    "edge_count_correlation": 0.9,
                    "weight_total_correlation": 0.95,
                }
            ]
        )
        with TemporaryDirectory() as tmpdir:
            output_path = write_all_samples_overview(summary_rows, Path(tmpdir))
            self.assertIsNotNone(output_path)
            self.assertTrue(Path(output_path).exists())

    def test_summary_payload_to_row_keeps_posterior_summary_fields(self):
        row = summary_payload_to_row(
            "maxent_micro__rewire_none",
            {
                "mean_snapshot_edge_jaccard": 0.31,
                "mean_snapshot_edge_jaccard_q05": 0.25,
                "mean_snapshot_edge_jaccard_q95": 0.38,
                "posterior_num_runs": 4,
                "posterior_run_labels": ["a", "b", "c", "d"],
            },
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["posterior_num_runs"], 4)
        self.assertEqual(row["mean_snapshot_edge_jaccard_q05"], 0.25)
        self.assertNotIn("posterior_run_labels", row)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed in test environment")
    def test_aggregate_posterior_reports_writes_setting_level_artifacts(self):
        original = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [2.0, 3.0, 5.0],
            }
        )
        synthetic_a = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [1.0, 4.0, 6.0],
            }
        )
        synthetic_b = pd.DataFrame(
            {
                "u": [0, 1, 0],
                "i": [1, 2, 2],
                "ts": [10, 10, 11],
                "weight": [3.0, 2.0, 4.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            diagnostics_dir = Path(tmpdir)
            reports = []
            for sample_index, synthetic in enumerate((synthetic_a, synthetic_b)):
                comparison = compare_panels_detailed(
                    original_df=original,
                    synthetic_df=synthetic,
                    directed=False,
                    weight_col="weight",
                    node_blocks={0: 10, 1: 10, 2: 20},
                    node_types={0: "Farm", 1: "Region", 2: "Farm"},
                )
                sample_label = f"maxent_micro__rewire_none__sample_{sample_index:04d}"
                outputs = write_report(
                    comparison["per_snapshot"],
                    comparison["summary"],
                    diagnostics_dir,
                    sample_label,
                    detailed_diagnostics=comparison["details"],
                    directed=False,
                    diagnostic_top_k=4,
                )
                reports.append(
                    {
                        "sample_label": sample_label,
                        "summary": comparison["summary"],
                        "outputs": outputs,
                    }
                )

            aggregate = aggregate_posterior_reports(
                reports,
                output_dir=diagnostics_dir,
                setting_label="maxent_micro__rewire_none",
                directed=False,
                diagnostic_top_k=4,
            )

            self.assertEqual(aggregate["summary"]["posterior_num_runs"], 2)
            aggregate_per_snapshot = pd.read_csv(diagnostics_dir / "maxent_micro__rewire_none_per_snapshot.csv")
            self.assertIn("synthetic_edge_count_q05", aggregate_per_snapshot.columns)
            self.assertTrue((diagnostics_dir / "maxent_micro__rewire_none_summary.json").exists())
            self.assertTrue((diagnostics_dir / "maxent_micro__rewire_none_dashboard.png").exists())

    def test_write_scientific_validation_report_creates_html(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "input_edges_filtered.csv").write_text("u,i,ts,weight\n0,1,10,2\n1,2,11,3\n")
            (run_dir / "manifest.json").write_text(
                '{"dataset":"demo","filtered_input_edges_path":"%s","fit_covariates":["dist_km"],"input_summary":{"node_count":3,"edge_count":2},"weight_model":{"candidate_label":"weight:demo"},"directed":false}'  # noqa: E501
                % (run_dir / "input_edges_filtered.csv")
            )
            (diagnostics_dir / "novelty_grid_summary.csv").write_text(
                "sample_label,mean_snapshot_edge_jaccard,mean_snapshot_node_jaccard,mean_synthetic_novel_edge_rate,edge_count_correlation,weight_total_correlation,mean_abs_edge_count_delta,mean_abs_weight_total_delta,tea_new_ratio_correlation,tna_new_ratio_correlation,pi_mass_mean_correlation,magnetic_spectrum_mean_correlation,magnetic_spectrum_mean_abs_delta\n"
                "sample_a__rewire_none,0.2,0.9,0.5,0.95,0.96,1.0,2.0,0.8,0.7,0.6,0.5,0.1\n"
                "sample_b__rewire_configuration,0.3,0.8,0.6,0.92,0.93,2.0,4.0,0.7,0.6,0.5,0.4,0.2\n"
            )
            for label in ("sample_a__rewire_none", "sample_b__rewire_configuration"):
                (diagnostics_dir / f"{label}_summary.json").write_text(
                    '{"mean_snapshot_edge_jaccard":0.2,"mean_snapshot_node_jaccard":0.9,"mean_synthetic_novel_edge_rate":0.5,"edge_count_correlation":0.95,"weight_total_correlation":0.96,"mean_abs_edge_count_delta":1.0,"mean_abs_weight_total_delta":2.0,"tea_new_ratio_correlation":0.8,"tna_new_ratio_correlation":0.7,"pi_mass_mean_correlation":0.6,"magnetic_spectrum_mean_correlation":0.5,"magnetic_spectrum_mean_abs_delta":0.1}'  # noqa: E501
                )
                (diagnostics_dir / f"{label}_block_pair_summary.csv").write_text(
                    "block_u,block_v,edge_count_correlation,weight_total_correlation\n1,1,0.9,0.8\n"
                )
                (diagnostics_dir / f"{label}_node_activity_summary.csv").write_text(
                    "node_id,original_total_incident_weight_total,incident_edge_count_correlation,incident_weight_total_correlation,block_id\n1,10,0.8,0.7,1\n"
                )
                (diagnostics_dir / f"{label}_tea_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\nnew_ratio,0.8,0.1\npersist_ratio,0.9,0.2\n"
                )
                (diagnostics_dir / f"{label}_tna_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\nnew_ratio,0.7,0.1\npersist_ratio,0.8,0.2\n"
                )
                (diagnostics_dir / f"{label}_pi_mass_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\npi_mass__type_0,0.6,0.1\nlic_share_active,0.7,0.2\n"
                )
                (diagnostics_dir / f"{label}_magnetic_laplacian_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\neig_1,0.5,0.1\neig_2,0.4,0.2\n"
                )

            output_path = write_scientific_validation_report(run_dir)

            self.assertTrue(Path(output_path).exists())
            html_text = Path(output_path).read_text()
            self.assertIn("Validation Report", html_text)
            self.assertIn("Configuration rewiring", html_text)
            self.assertIn("table-search-input", html_text)
            self.assertIn("How to read this", html_text)
            self.assertIn("TEA new corr", html_text)
            self.assertIn("Pi-mass corr", html_text)
            self.assertIn("Mag spectrum corr", html_text)

    def test_write_scientific_validation_report_mentions_posterior_medians_when_available(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "input_edges_filtered.csv").write_text("u,i,ts,weight\n0,1,10,2\n1,2,11,3\n")
            (run_dir / "manifest.json").write_text(
                '{"dataset":"demo","filtered_input_edges_path":"%s","fit_covariates":["dist_km"],"input_summary":{"node_count":3,"edge_count":2},"weight_model":{"candidate_label":"weight:demo"},"directed":false}'  # noqa: E501
                % (run_dir / "input_edges_filtered.csv")
            )
            (diagnostics_dir / "all_samples_summary.csv").write_text(
                "sample_label,sample_class,sample_mode,rewire_mode,posterior_num_runs,mean_snapshot_edge_jaccard,mean_snapshot_edge_jaccard_q05,mean_snapshot_edge_jaccard_q95,mean_snapshot_node_jaccard,mean_synthetic_novel_edge_rate,mean_synthetic_novel_edge_rate_q05,mean_synthetic_novel_edge_rate_q95,edge_count_correlation,weight_total_correlation\n"
                "maxent_micro__rewire_none,posterior_predictive,maxent_micro,none,3,0.32,0.28,0.36,0.88,0.41,0.38,0.47,0.98,0.97\n"
            )
            (diagnostics_dir / "maxent_micro__rewire_none_summary.json").write_text(
                '{"posterior_num_runs":3,"mean_snapshot_edge_jaccard":0.32,"mean_snapshot_edge_jaccard_q05":0.28,"mean_snapshot_edge_jaccard_q95":0.36,"mean_snapshot_node_jaccard":0.88,"mean_synthetic_novel_edge_rate":0.41,"mean_synthetic_novel_edge_rate_q05":0.38,"mean_synthetic_novel_edge_rate_q95":0.47,"edge_count_correlation":0.98,"weight_total_correlation":0.97}'  # noqa: E501
            )

            output_path = write_scientific_validation_report(run_dir)

            html_text = Path(output_path).read_text()
            self.assertIn("Posterior median across 3 draws", html_text)
            self.assertIn("Posterior medians with 5th-95th percentile intervals", html_text)

    def test_write_daily_network_snapshot_assets_creates_payload_and_viewer(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            observed_path = run_dir / "input_edges_filtered.csv"
            observed_path.write_text("u,i,ts,trade\n0,1,10,2\n1,2,11,3\n")
            node_attributes_path = run_dir / "node_attributes.csv"
            node_attributes_path.write_text(
                "node_id,x,y,total_animals,type,type_label,corop,ubn,block_id\n"
                "0,100,200,12,0,Farm,CR35,1001,0\n"
                "1,120,220,18,1,Region,CR35,,0\n"
                "2,140,240,25,0,Farm,CR35,1002,1\n"
            )

            for sample_idx in range(2):
                sample_dir = run_dir / "generated" / "maxent_micro__rewire_none" / f"sample_{sample_idx:04d}" / "snapshots"
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / "snapshot_10.csv").write_text("u,i,ts,snapshot,trade\n0,1,10,0,4\n")
                (sample_dir / "snapshot_11.csv").write_text("u,i,ts,snapshot,trade\n1,2,11,1,5\n")

            (diagnostics_dir / "maxent_micro__rewire_none_summary.json").write_text(
                '{"posterior_run_labels":["maxent_micro__rewire_none__sample_0000","maxent_micro__rewire_none__sample_0001"]}'
            )
            summary_rows = pd.DataFrame(
                [
                    {
                        "sample_label": "maxent_micro__rewire_none",
                        "sample_class": "posterior_predictive",
                        "mean_snapshot_edge_jaccard": 0.32,
                        "weight_total_correlation": 0.97,
                        "mean_synthetic_novel_edge_rate": 0.41,
                    }
                ]
            )

            manifest = {
                "filtered_input_edges_path": str(observed_path),
                "node_attributes_path": str(node_attributes_path),
                "weight_model": {"output_column": "trade"},
                "directed": True,
            }

            def fake_render(*_args, **kwargs):
                output_dir = Path(kwargs["output_dir"])
                ts_value = int(kwargs["ts_value"])
                output_dir.mkdir(parents=True, exist_ok=True)
                forced = output_dir / f"snapshot_{ts_value}_forced.pdf"
                geographic = output_dir / f"snapshot_{ts_value}_geographic.pdf"
                forced.write_text("forced")
                geographic.write_text("geographic")
                return {"forced": forced, "geographic": geographic}

            with patch("temporal_sbm.diagnostics._load_graph_tool_module", return_value=object()):
                with patch("temporal_sbm.diagnostics._render_daily_snapshot_pdfs", side_effect=fake_render):
                    outputs = _write_daily_network_snapshot_assets(
                        run_dir,
                        diagnostics_dir,
                        manifest,
                        summary_rows,
                        ["maxent_micro__rewire_none"],
                    )

            self.assertTrue(Path(outputs["network_compare_html"]).exists())
            self.assertTrue(Path(outputs["network_payload_js"]).exists())
            payload_text = Path(outputs["network_payload_js"]).read_text()
            self.assertIn("snapshot_10_forced.pdf", payload_text)
            self.assertIn("Run 1", payload_text)
            html_text = Path(outputs["network_compare_html"]).read_text()
            self.assertIn("Left setting", html_text)
            self.assertIn("Open PDF in a new tab", html_text)

    def test_write_scientific_validation_report_adds_magnetic_run_switcher(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "input_edges_filtered.csv").write_text("u,i,ts,weight\n0,1,10,2\n1,2,11,3\n")
            (run_dir / "manifest.json").write_text(
                '{"dataset":"demo","filtered_input_edges_path":"%s","fit_covariates":["dist_km"],"input_summary":{"node_count":3,"edge_count":2},"weight_model":{"candidate_label":"weight:demo"},"directed":true}'  # noqa: E501
                % (run_dir / "input_edges_filtered.csv")
            )
            (diagnostics_dir / "all_samples_summary.csv").write_text(
                "sample_label,sample_class,sample_mode,rewire_mode,posterior_num_runs,mean_snapshot_edge_jaccard,mean_snapshot_edge_jaccard_q05,mean_snapshot_edge_jaccard_q95,mean_snapshot_node_jaccard,mean_synthetic_novel_edge_rate,mean_synthetic_novel_edge_rate_q05,mean_synthetic_novel_edge_rate_q95,edge_count_correlation,weight_total_correlation,reciprocity_correlation\n"
                "maxent_micro__rewire_none,posterior_predictive,maxent_micro,none,3,0.32,0.28,0.36,0.88,0.41,0.38,0.47,0.98,0.97,0.9\n"
            )
            (diagnostics_dir / "maxent_micro__rewire_none_summary.json").write_text(
                '{"posterior_num_runs":3,"posterior_run_labels":["maxent_micro__rewire_none__sample_0000","maxent_micro__rewire_none__sample_0001","maxent_micro__rewire_none__sample_0002"],"mean_snapshot_edge_jaccard":0.32,"mean_snapshot_node_jaccard":0.88,"mean_synthetic_novel_edge_rate":0.41,"edge_count_correlation":0.98,"weight_total_correlation":0.97,"reciprocity_correlation":0.9}'  # noqa: E501
            )
            for suffix in (
                "magnetic_laplacian.png",
                "magnetic_laplacian_diff.png",
                "dashboard.png",
            ):
                (diagnostics_dir / f"maxent_micro__rewire_none_{suffix}").write_text("x")
            for run_idx in range(3):
                for suffix in ("magnetic_laplacian.png", "magnetic_laplacian_diff.png"):
                    (diagnostics_dir / f"maxent_micro__rewire_none__sample_{run_idx:04d}_{suffix}").write_text("x")

            output_path = write_scientific_validation_report(run_dir)

            html_text = Path(output_path).read_text()
            self.assertIn("Posterior summary", html_text)
            self.assertIn("Run 1", html_text)
            self.assertIn("figure-switcher-select", html_text)

    def test_write_scientific_validation_report_embeds_daily_network_viewer(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "input_edges_filtered.csv").write_text("u,i,ts,weight\n0,1,10,2\n1,2,11,3\n")
            (run_dir / "manifest.json").write_text(
                '{"dataset":"demo","filtered_input_edges_path":"%s","fit_covariates":["dist_km"],"input_summary":{"node_count":3,"edge_count":2},"weight_model":{"candidate_label":"weight:demo"},"directed":true}'  # noqa: E501
                % (run_dir / "input_edges_filtered.csv")
            )
            (diagnostics_dir / "all_samples_summary.csv").write_text(
                "sample_label,sample_class,sample_mode,rewire_mode,posterior_num_runs,mean_snapshot_edge_jaccard,mean_snapshot_node_jaccard,mean_synthetic_novel_edge_rate,edge_count_correlation,weight_total_correlation,reciprocity_correlation\n"
                "maxent_micro__rewire_none,posterior_predictive,maxent_micro,none,2,0.32,0.88,0.41,0.98,0.97,0.9\n"
            )
            (diagnostics_dir / "maxent_micro__rewire_none_summary.json").write_text(
                '{"posterior_num_runs":2,"posterior_run_labels":["maxent_micro__rewire_none__sample_0000","maxent_micro__rewire_none__sample_0001"],"mean_snapshot_edge_jaccard":0.32,"mean_snapshot_node_jaccard":0.88,"mean_synthetic_novel_edge_rate":0.41,"edge_count_correlation":0.98,"weight_total_correlation":0.97,"reciprocity_correlation":0.9}'  # noqa: E501
            )
            viewer_path = diagnostics_dir / "daily_network_compare.html"
            viewer_path.write_text("<html>viewer</html>")

            with patch(
                "temporal_sbm.diagnostics._write_daily_network_snapshot_assets",
                return_value={"network_compare_html": viewer_path},
            ):
                output_path = write_scientific_validation_report(run_dir)

            html_text = Path(output_path).read_text()
            self.assertIn("Daily Network Snapshots", html_text)
            self.assertIn("Interactive daily network comparison in forced and geographic layouts", html_text)

    def test_load_sweep_summary_rows_prefers_setting_summary_csv_over_run_jsons(self):
        with TemporaryDirectory() as tmpdir:
            diagnostics_dir = Path(tmpdir)
            (diagnostics_dir / "setting_posterior_summary.csv").write_text(
                "sample_label,posterior_num_runs,mean_snapshot_edge_jaccard,weight_total_correlation\n"
                "maxent_micro__rewire_none,3,0.32,0.97\n"
            )
            (diagnostics_dir / "maxent_micro__rewire_none_summary.json").write_text(
                '{"posterior_num_runs":3,"mean_snapshot_edge_jaccard":0.32,"weight_total_correlation":0.97}'
            )
            for idx, value in enumerate((0.31, 0.33)):
                (diagnostics_dir / f"maxent_micro__rewire_none__sample_{idx:04d}_summary.json").write_text(
                    '{"mean_snapshot_edge_jaccard":%s,"weight_total_correlation":0.97}' % value
                )

            summary_rows = _load_sweep_summary_rows(diagnostics_dir)

            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows.loc[0, "sample_label"], "maxent_micro__rewire_none")
            self.assertEqual(int(summary_rows.loc[0, "posterior_num_runs"]), 3)


    def test_write_scientific_validation_report_directed_mentions_directional_metrics(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            diagnostics_dir = run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "input_edges_filtered.csv").write_text("u,i,ts,weight\n0,1,10,2\n1,0,11,3\n")
            (run_dir / "manifest.json").write_text(
                '{"dataset":"demo","filtered_input_edges_path":"%s","fit_covariates":["dist_km"],"input_summary":{"node_count":2,"edge_count":2},"weight_model":{"candidate_label":"weight:demo"},"directed":true}'  # noqa: E501
                % (run_dir / "input_edges_filtered.csv")
            )
            (diagnostics_dir / "all_samples_summary.csv").write_text(
                "sample_label,mean_snapshot_edge_jaccard,mean_snapshot_node_jaccard,mean_synthetic_novel_edge_rate,edge_count_correlation,weight_total_correlation,reciprocity_correlation,mean_abs_edge_count_delta,mean_abs_weight_total_delta,mean_abs_reciprocity_delta,tea_new_ratio_correlation,tna_new_ratio_correlation,pi_mass_mean_correlation,magnetic_spectrum_mean_correlation,magnetic_spectrum_mean_abs_delta\n"
                "maxent_micro__rewire_none,0.2,0.9,0.5,0.95,0.96,0.88,1.0,2.0,0.1,0.8,0.7,0.6,0.5,0.1\n"
                "canonical_maxent__rewire_configuration,0.3,0.8,0.6,0.92,0.93,0.82,2.0,4.0,0.2,0.7,0.6,0.5,0.4,0.2\n"
            )
            for label in ("maxent_micro__rewire_none", "canonical_maxent__rewire_configuration"):
                (diagnostics_dir / f"{label}_summary.json").write_text(
                    '{"mean_snapshot_edge_jaccard":0.2,"mean_snapshot_node_jaccard":0.9,"mean_synthetic_novel_edge_rate":0.5,"edge_count_correlation":0.95,"reciprocity_correlation":0.88,"mean_abs_reciprocity_delta":0.1,"weight_total_correlation":0.96,"mean_abs_edge_count_delta":1.0,"mean_abs_weight_total_delta":2.0,"tea_new_ratio_correlation":0.8,"tna_new_ratio_correlation":0.7,"pi_mass_mean_correlation":0.6,"magnetic_spectrum_mean_correlation":0.5,"magnetic_spectrum_mean_abs_delta":0.1}'  # noqa: E501
                )
                (diagnostics_dir / f"{label}_block_pair_summary.csv").write_text(
                    "block_u,block_v,edge_count_correlation,weight_total_correlation\n1,2,0.9,0.8\n"
                )
                (diagnostics_dir / f"{label}_node_activity_summary.csv").write_text(
                    "node_id,original_total_incident_weight_total,incident_edge_count_correlation,out_edge_count_correlation,in_edge_count_correlation,incident_weight_total_correlation,out_weight_total_correlation,in_weight_total_correlation,block_id\n1,10,0.8,0.9,0.7,0.7,0.8,0.6,1\n"
                )
                (diagnostics_dir / f"{label}_tea_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\nnew_ratio,0.8,0.1\npersist_ratio,0.9,0.2\n"
                )
                (diagnostics_dir / f"{label}_tna_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\nnew_ratio,0.7,0.1\npersist_ratio,0.8,0.2\n"
                )
                (diagnostics_dir / f"{label}_pi_mass_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\npi_mass__type_0,0.6,0.1\nlic_share_active,0.7,0.2\n"
                )
                (diagnostics_dir / f"{label}_magnetic_laplacian_summary.csv").write_text(
                    "metric,correlation,mean_abs_delta\neig_1,0.5,0.1\neig_2,0.4,0.2\n"
                )

            output_path = write_scientific_validation_report(run_dir)

            html_text = Path(output_path).read_text()
            self.assertIn("Directed Temporal SBM Validation Report", html_text)
            self.assertIn("Directed simple graph", html_text)
            self.assertIn("Reciprocity corr", html_text)
            self.assertIn("B1->B2", html_text)
            self.assertIn("How to read this", html_text)
            self.assertIn("TEA summary", html_text)
            self.assertIn("Magnetic spectrum summary", html_text)


if __name__ == "__main__":
    unittest.main()
