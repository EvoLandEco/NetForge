import unittest
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from temporal_sbm.pipeline import (
    _build_joint_metadata_links,
    _align_external_weight_values,
    _detect_transformed_external_weight_companion,
    _fit_includes_edge_weight_covariate,
    _merge_generated_sample_records,
    _merge_generated_setting_records,
    _node_blocks_from_state,
    _standalone_weight_model,
    _stored_weight_reference_blocks,
    _state_summary_text,
    _select_covariate_specs,
    _weight_generation_mode_name,
    CovariateSpec,
    extract_node_block_map,
    prepare_data,
    resolve_input_paths,
    _sample_kwargs,
    _split_layered_entropy_args,
    fit_command,
)


class PipelineWeightAlignmentTests(unittest.TestCase):
    def test_fit_command_keeps_weight_candidates_when_joint_metadata_is_active(self):
        prepared = argparse.Namespace(
            metadata_summary={"enabled": True},
            weight_column="trade",
            original_edges=pd.DataFrame(),
        )
        args = argparse.Namespace(
            output_dir="/tmp/netforge-fit",
            directed=True,
            fit_covariates=None,
            exclude_weight_from_fit=False,
            weight_generation_mode="legacy",
        )
        weight_model = {
            "input_column": "trade",
            "output_column": "trade",
            "graph_property": "_edge_weight_int",
            "rec_type": "discrete-geometric",
            "transform": "none",
            "candidate_label": "trade:discrete-geometric/none",
        }

        with (
            mock.patch("temporal_sbm.pipeline.prepare_data", return_value=prepared),
            mock.patch("temporal_sbm.pipeline._validate_weight_generation_configuration"),
            mock.patch("temporal_sbm.pipeline.build_layered_graph", return_value=object()),
            mock.patch("temporal_sbm.pipeline._build_weight_candidates", return_value=[object()]) as build_weights,
            mock.patch("temporal_sbm.pipeline._fit_with_weight_candidates", return_value=("state", weight_model, [])),
            mock.patch("temporal_sbm.pipeline.attach_partition_maps"),
            mock.patch("temporal_sbm.pipeline.write_fit_artifacts", return_value={"run_dir": "/tmp/netforge-fit"}),
        ):
            manifest = fit_command(args)

        build_weights.assert_called_once()
        self.assertEqual(manifest["run_dir"], "/tmp/netforge-fit")

    def test_stored_weight_reference_blocks_uses_saved_node_blocks(self):
        class DummyGraph:
            def num_vertices(self) -> int:
                return 3

            def vertex(self, index: int) -> int:
                return index

        class DummyBase:
            g = DummyGraph()

        node_id_prop = {0: 10, 1: 11, 2: 12}
        blocks = np.array([4, 5, 6], dtype=np.int64)
        weight_generator_model = {
            "node_blocks": {
                "10": 40,
                "11": 50,
                "12": 60,
            }
        }

        stored = _stored_weight_reference_blocks(
            base=DummyBase(),
            blocks=blocks,
            node_id_prop=node_id_prop,
            weight_generator_model=weight_generator_model,
        )

        self.assertTrue(np.array_equal(stored, np.array([40, 50, 60], dtype=np.int64)))

    def test_align_external_weight_values_uses_idx_with_padding(self):
        frame = pd.DataFrame({"idx": [1, 2, 3]})
        values = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
        aligned, strategy = _align_external_weight_values(frame, values)
        self.assertEqual(strategy, "idx_with_padding")
        self.assertTrue(np.array_equal(aligned, np.array([10.0, 20.0, 30.0], dtype=float)))

    def test_align_external_weight_values_skips_leading_padding_without_idx(self):
        frame = pd.DataFrame({"u": [0, 1, 2]})
        values = np.array([0.0, 5.0, 6.0, 7.0], dtype=float)
        aligned, strategy = _align_external_weight_values(frame, values)
        self.assertEqual(strategy, "row_order_skip_first")
        self.assertTrue(np.array_equal(aligned, np.array([5.0, 6.0, 7.0], dtype=float)))

    def test_split_layered_entropy_args_removes_serialized_entropy_args(self):
        layered_type = type("LayeredBlockState", (), {})
        payload = {
            "base_type": layered_type,
            "state_args": {
                "entropy_args": {"multigraph": False},
                "foo": "bar",
            },
        }

        clean_payload, entropy_args = _split_layered_entropy_args(payload)

        self.assertEqual(clean_payload["state_args"], {"foo": "bar"})
        self.assertEqual(entropy_args, {"multigraph": False})

    def test_split_layered_entropy_args_keeps_non_layered_payloads(self):
        block_type = type("BlockState", (), {})
        payload = {
            "base_type": block_type,
            "state_args": {
                "entropy_args": {"multigraph": False},
                "foo": "bar",
            },
        }

        clean_payload, entropy_args = _split_layered_entropy_args(payload)

        self.assertEqual(clean_payload, payload)
        self.assertIsNone(entropy_args)

    def test_detect_transformed_external_weight_companion_detects_log1p_raw_sibling(self):
        frame = pd.DataFrame({"idx": [1, 2, 3]})
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "ml_demo_weight_raw.npy"
            log_path = tmp_path / "ml_demo_weight.npy"
            raw = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
            np.save(raw_path, raw)
            np.save(log_path, np.log1p(raw))

            aligned, _ = _align_external_weight_values(frame, np.load(log_path))
            detected = _detect_transformed_external_weight_companion(frame, log_path, aligned)

        self.assertIsNotNone(detected)
        transform_name, detected_raw_path, raw_alignment = detected
        self.assertEqual(transform_name, "log1p")
        self.assertEqual(detected_raw_path, raw_path)
        self.assertEqual(raw_alignment, "idx_with_padding")

    def test_detect_transformed_external_weight_companion_ignores_raw_file(self):
        frame = pd.DataFrame({"idx": [1, 2, 3]})
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "ml_demo_weight_raw.npy"
            raw = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
            np.save(raw_path, raw)

            aligned, _ = _align_external_weight_values(frame, np.load(raw_path))
            detected = _detect_transformed_external_weight_companion(frame, raw_path, aligned)

        self.assertIsNone(detected)

    def test_sample_kwargs_respects_explicit_sample_params(self):
        class Args:
            sample_canonical = True
            sample_max_ent = False
            sample_n_iter = 123
            sample_params = False

        kwargs = _sample_kwargs(Args())

        self.assertEqual(kwargs["canonical"], True)
        self.assertEqual(kwargs["max_ent"], False)
        self.assertEqual(kwargs["n_iter"], 123)
        self.assertEqual(kwargs["sample_params"], False)

    def test_merge_generated_sample_records_keeps_prior_settings(self):
        existing = [
            {
                "sample_manifest_path": "/tmp/generated/micro/sample_0000/sample_manifest.json",
                "setting_label": "micro__rewire_none",
                "sample_index": 0,
            }
        ]
        new_records = [
            {
                "sample_manifest_path": "/tmp/generated/canonical/sample_0000/sample_manifest.json",
                "setting_label": "canonical_ml__rewire_none",
                "sample_index": 0,
            }
        ]

        merged = _merge_generated_sample_records(existing, new_records)

        self.assertEqual(
            [record["setting_label"] for record in merged],
            ["canonical_ml__rewire_none", "micro__rewire_none"],
        )

    def test_merge_generated_setting_records_replaces_same_setting_manifest(self):
        existing = [
            {
                "setting_manifest_path": "/tmp/generated/micro/setting_manifest.json",
                "setting_label": "micro__rewire_none",
                "num_samples_requested": 2,
            }
        ]
        replacement = {
            "setting_manifest_path": "/tmp/generated/micro/setting_manifest.json",
            "setting_label": "micro__rewire_none",
            "num_samples_requested": 3,
        }

        merged = _merge_generated_setting_records(existing, replacement)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["num_samples_requested"], 3)

    def test_weight_generation_mode_name_defaults_to_parametric(self):
        self.assertEqual(_weight_generation_mode_name(argparse.Namespace()), "parametric")

    def test_fit_includes_edge_weight_covariate_follows_flag(self):
        self.assertTrue(_fit_includes_edge_weight_covariate(argparse.Namespace(exclude_weight_from_fit=False)))
        self.assertFalse(_fit_includes_edge_weight_covariate(argparse.Namespace(exclude_weight_from_fit=True)))

    def test_standalone_weight_model_tracks_weight_column(self):
        prepared = argparse.Namespace(weight_column="trade")

        model = _standalone_weight_model(prepared)

        self.assertEqual(
            model,
            {
                "input_column": "trade",
                "output_column": "trade",
                "candidate_label": "separate_parametric_generator",
                "fit_as_edge_covariate": False,
            },
        )

    def test_build_joint_metadata_links_keeps_scalar_columns_when_node_map_overlaps(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            node_map_csv = tmp_path / "node_map.csv"
            pd.DataFrame(
                {
                    "node_id": [0, 1],
                    "type": ["Farm", "Region"],
                    "corop": ["CR35", "CR35"],
                    "num_farms": [99.0, 88.0],
                    "total_animals": [999.0, 888.0],
                }
            ).to_csv(node_map_csv, index=False)

            args = argparse.Namespace(
                joint_metadata_model=True,
                metadata_fields=["corop", "num_farms_bin", "total_animals_bin"],
                metadata_numeric_bins=2,
                metadata_grid_km=50.0,
                metadata_ft_top_k=3,
            )

            links, summary = _build_joint_metadata_links(
                compact_to_original=np.array([0, 1], dtype=np.int64),
                node_features=np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [10.0, 20.0, 1.0, 20.0],
                        [30.0, 40.0, 2.0, 40.0],
                    ],
                    dtype=float,
                ),
                node_feature_columns=["xco", "yco", "num_farms", "total_animals"],
                centroid_x_index=0,
                centroid_y_index=1,
                active_compact_mask=np.array([True, True]),
                node_map_csv=node_map_csv,
                args=args,
            )

        self.assertFalse(links.empty)
        self.assertEqual(summary["num_links"], len(links))
        self.assertIn("corop", summary["fields_used"])
        self.assertIn("num_farms_bin", summary["fields_used"])
        self.assertIn("total_animals_bin", summary["fields_used"])

    def test_select_covariate_specs_accepts_none_sentinel(self):
        available = [
            CovariateSpec(name="dist_km", graph_property="edge_dist_km", rec_type="real-normal"),
            CovariateSpec(name="mass_grav", graph_property="edge_mass_grav", rec_type="real-normal"),
        ]

        selected = _select_covariate_specs(available, ["none"])

        self.assertEqual(selected, [])

    def test_state_summary_text_uses_nonempty_block_count(self):
        class DummyProp:
            def __init__(self, values):
                self.a = np.asarray(values, dtype=np.int64)

        class DummyGraph:
            def num_vertices(self) -> int:
                return 847

            def num_edges(self) -> int:
                return 31137

        class DummyBase:
            def __init__(self):
                self.g = DummyGraph()

            def get_nonempty_B(self) -> int:
                return 3

            def get_nonoverlap_blocks(self):
                return DummyProp(np.arange(147, dtype=np.int64))

        class DummyNestedState:
            def __init__(self):
                self._base = DummyBase()

            def get_levels(self):
                return [self._base] + [object()] * 10

            def entropy(self) -> float:
                return 389740.873832

        summary = _state_summary_text(DummyNestedState())

        self.assertIn("levels=11", summary)
        self.assertIn("blocks=3", summary)
        self.assertIn("vertices=847", summary)
        self.assertIn("edges=31137", summary)

    def test_node_blocks_from_state_prefers_nested_partition_vector(self):
        class DummyProp:
            def __init__(self, values):
                self.a = np.asarray(values, dtype=np.int64)

        class DummyGraph:
            def num_vertices(self) -> int:
                return 4

        class DummyBase:
            def __init__(self):
                self.g = DummyGraph()

            def get_nonoverlap_blocks(self):
                return DummyProp([100, 101, 102, 103])

        class DummyNestedState:
            def __init__(self):
                self._base = DummyBase()

            def get_levels(self):
                return [self._base]

            def get_bs(self):
                return [np.array([7, 7, 9, 9], dtype=np.int64)]

        blocks = _node_blocks_from_state(DummyNestedState())

        self.assertTrue(np.array_equal(blocks, np.array([7, 7, 9, 9], dtype=np.int64)))

    def test_extract_node_block_map_skips_metadata_vertices(self):
        class DummyVertex:
            def __init__(self, index: int):
                self.index = index

            def __int__(self) -> int:
                return self.index

        class DummyProp:
            def __init__(self, values):
                self.values = list(values)

            def __getitem__(self, vertex: DummyVertex) -> int:
                return self.values[int(vertex)]

        class DummyGraph:
            def __init__(self):
                self.vp = {
                    "node_id": DummyProp([10, -1, 11]),
                    "sbm_b": DummyProp([4, 8, 4]),
                    "is_metadata_tag": DummyProp([0, 1, 0]),
                }

            def vertices(self):
                return [DummyVertex(0), DummyVertex(1), DummyVertex(2)]

        mapping = extract_node_block_map(DummyGraph())

        self.assertEqual(mapping, {10: 4, 11: 4})


class PipelineInputFormatTests(unittest.TestCase):
    def _args(self, data_root: Path, dataset: str) -> argparse.Namespace:
        return argparse.Namespace(
            data_root=str(data_root),
            dataset=dataset,
            edges_csv=None,
            weight_npy=None,
            node_features_npy=None,
            node_schema_json=None,
            node_map_csv=None,
            src_col="u",
            dst_col="i",
            ts_col="ts",
            weight_col="trade",
            directed=True,
            tz="Europe/Amsterdam",
            ts_format="ordinal",
            ts_unit="s",
            holiday_country="NL",
            ts_start=None,
            ts_end=None,
            date_start=None,
            date_end=None,
            duplicate_policy="collapse",
            self_loop_policy="drop",
        )

    def test_resolve_input_paths_uses_clean_default_names(self):
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset_dir = data_root / "DEMO"
            dataset_dir.mkdir()
            (dataset_dir / "edges.csv").write_text("u,i,ts,trade\n0,1,1,2.0\n")
            np.save(dataset_dir / "node_features.npy", np.array([[1.0, 2.0, 1.0, 3.0], [3.0, 4.0, 1.0, 5.0]]))
            (dataset_dir / "node_schema.json").write_text(
                '{"node_feature_columns_in_order": ["xco", "yco", "num_farms", "total_animals"], "node_row_offset": 0}\n'
            )
            (dataset_dir / "node_map.csv").write_text("node_id,type\n0,Farm\n1,Region\n")

            paths = resolve_input_paths(self._args(data_root, "DEMO"))

        self.assertEqual(paths.edges_csv.name, "edges.csv")
        self.assertEqual(paths.node_features_npy.name, "node_features.npy")
        self.assertEqual(paths.node_schema_json.name, "node_schema.json")
        self.assertEqual(paths.node_map_csv.name, "node_map.csv")

    def test_prepare_data_does_not_scan_legacy_file_names(self):
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset_dir = data_root / "DEMO"
            dataset_dir.mkdir()
            pd.DataFrame({"u": [0], "i": [0], "ts": [737425], "trade": [2.0]}).to_csv(
                dataset_dir / "ml_DEMO.csv",
                index=False,
            )
            np.save(dataset_dir / "ml_DEMO_node.npy", np.array([[10.0, 20.0, 1.0, 30.0]], dtype=float))
            (dataset_dir / "node_feature_columns.json").write_text(
                '{"columns_in_order": ["xco", "yco", "num_farms", "total_animals"]}\n'
            )
            (dataset_dir / "node_map_DEMO.csv").write_text("node_id,type\n0,Farm\n")

            with self.assertRaises(FileNotFoundError):
                prepare_data(self._args(data_root, "DEMO"))

    def test_prepare_data_accepts_unpadded_node_arrays(self):
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset_dir = data_root / "DEMO"
            dataset_dir.mkdir()
            pd.DataFrame(
                {
                    "u": [0, 1],
                    "i": [1, 0],
                    "ts": [737425, 737425],
                    "trade": [2.0, 1.0],
                }
            ).to_csv(dataset_dir / "edges.csv", index=False)
            np.save(
                dataset_dir / "node_features.npy",
                np.array(
                    [
                        [10.0, 20.0, 1.0, 30.0],
                        [30.0, 40.0, 2.0, 60.0],
                    ],
                    dtype=float,
                ),
            )
            (dataset_dir / "node_schema.json").write_text(
                '{"node_feature_columns_in_order": ["xco", "yco", "num_farms", "total_animals"], "node_row_offset": 0}\n'
            )
            (dataset_dir / "node_map.csv").write_text("node_id,type\n0,Farm\n1,Region\n")

            prepared = prepare_data(self._args(data_root, "DEMO"))

        self.assertEqual(prepared.node_features.shape, (3, 4))
        self.assertTrue(np.allclose(prepared.node_features[0], 0.0))
        self.assertTrue(np.allclose(prepared.node_features[1], np.array([10.0, 20.0, 1.0, 30.0])))
        self.assertEqual(prepared.compact_to_original.tolist(), [0, 1])

    def test_prepare_data_rejects_padded_node_arrays(self):
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset_dir = data_root / "DEMO"
            dataset_dir.mkdir()
            pd.DataFrame(
                {
                    "u": [0],
                    "i": [1],
                    "ts": [737425],
                    "trade": [2.0],
                }
            ).to_csv(dataset_dir / "edges.csv", index=False)
            np.save(
                dataset_dir / "node_features.npy",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [10.0, 20.0, 1.0, 30.0],
                        [30.0, 40.0, 2.0, 60.0],
                    ],
                    dtype=float,
                ),
            )
            (dataset_dir / "node_schema.json").write_text(
                '{"node_feature_columns_in_order": ["xco", "yco", "num_farms", "total_animals"]}\n'
            )
            (dataset_dir / "node_map.csv").write_text("node_id,type\n0,Farm\n1,Region\n")

            with self.assertRaisesRegex(ValueError, "Expected 2 rows, observed 3"):
                prepare_data(self._args(data_root, "DEMO"))

    def test_prepare_data_requires_node_map_csv(self):
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            dataset_dir = data_root / "DEMO"
            dataset_dir.mkdir()
            pd.DataFrame({"u": [0], "i": [0], "ts": [737425], "trade": [2.0]}).to_csv(
                dataset_dir / "edges.csv",
                index=False,
            )
            np.save(dataset_dir / "node_features.npy", np.array([[10.0, 20.0, 1.0, 30.0]], dtype=float))
            (dataset_dir / "node_schema.json").write_text(
                '{"node_feature_columns_in_order": ["xco", "yco", "num_farms", "total_animals"], "node_row_offset": 0}\n'
            )

            with self.assertRaises(FileNotFoundError):
                prepare_data(self._args(data_root, "DEMO"))


if __name__ == "__main__":
    unittest.main()
