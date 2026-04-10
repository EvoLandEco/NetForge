import json
from datetime import date
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "examples" / "toy_nl" / "processed_data" / "TOY_NL"


class ToyDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "examples" / "toy_nl" / "build_toy_nl_dataset.py")],
            check=True,
            cwd=ROOT,
        )
        cls.edge_frame = pd.read_csv(DATASET_DIR / "edges.csv")
        cls.node_matrix = np.load(DATASET_DIR / "node_features.npy")
        cls.node_map = pd.read_csv(DATASET_DIR / "node_map.csv")
        cls.schema = json.loads((DATASET_DIR / "node_schema.json").read_text())
        cls.manifest = json.loads((DATASET_DIR / "dataset_manifest.json").read_text())

    def test_required_files_align(self) -> None:
        feature_columns = self.schema["node_feature_columns_in_order"]
        self.assertEqual(self.node_matrix.shape[1], len(feature_columns))
        self.assertEqual(self.node_matrix.shape[0], int(self.node_map["node_id"].max()) + 1)
        self.assertEqual(self.schema["node_row_offset"], 0)
        self.assertEqual(self.manifest["edge_file"], "edges.csv")
        self.assertEqual(self.manifest["node_features_file"], "node_features.npy")
        self.assertEqual(self.manifest["node_schema_file"], "node_schema.json")
        self.assertEqual(self.manifest["node_map_file"], "node_map.csv")
        self.assertTrue({"u", "i", "ts", "trade"}.issubset(self.edge_frame.columns))
        max_node_id = int(self.node_map["node_id"].max())
        self.assertTrue((self.edge_frame["u"] >= 0).all())
        self.assertTrue((self.edge_frame["i"] >= 0).all())
        self.assertTrue((self.edge_frame["u"] <= max_node_id).all())
        self.assertTrue((self.edge_frame["i"] <= max_node_id).all())

    def test_default_metadata_inputs_are_present(self) -> None:
        feature_columns = set(self.schema["node_feature_columns_in_order"])
        self.assertTrue({"xco", "yco", "num_farms", "total_animals", "count_ft_cattle", "count_ft_pig"}.issubset(feature_columns))
        self.assertTrue(
            {
                "node_id",
                "type",
                "corop",
                "coord_source",
                "priority",
                "CR_code",
                "trade_species",
                "diersoort",
                "diergroep",
                "diergroeplang",
                "BtypNL",
                "bedrtype",
            }.issubset(self.node_map.columns)
        )

    def test_manifest_describes_default_metadata_layer(self) -> None:
        metadata = self.manifest["joint_metadata_model"]
        self.assertTrue(metadata["enabled_by_default"])
        self.assertEqual(metadata["layer_name"], "__metadata__")
        self.assertEqual(
            metadata["metadata_fields"],
            [
                "corop",
                "coord_source",
                "priority",
                "CR_code",
                "num_farms_bin",
                "total_animals_bin",
                "centroid_grid",
                "trade_species",
                "diersoort",
                "diergroep",
                "diergroeplang",
                "BtypNL",
                "bedrtype",
            ],
        )
        self.assertEqual(
            metadata["node_map_fields"],
            [
                "corop",
                "coord_source",
                "priority",
                "CR_code",
                "trade_species",
                "diersoort",
                "diergroep",
                "diergroeplang",
                "BtypNL",
                "bedrtype",
            ],
        )

    def test_text_metadata_fields_include_multi_value_tokens(self) -> None:
        self.assertTrue(self.node_map["trade_species"].astype(str).str.contains(r"\|", regex=True).any())
        self.assertTrue(self.node_map["diersoort"].astype(str).str.contains(";", regex=False).any())

    def test_timestamps_match_ordinal_dates(self) -> None:
        ordinal_dates = pd.to_datetime(self.edge_frame["date"]).dt.date.map(date.toordinal)
        self.assertTrue((self.edge_frame["ts"] == ordinal_dates).all())

    def test_weekends_and_holidays_have_lower_activity(self) -> None:
        daily_trade = self.edge_frame.groupby(
            ["date", "is_weekend", "is_public_holiday"],
            as_index=False,
        )["trade"].sum()
        weekend_mean = daily_trade.loc[daily_trade["is_weekend"], "trade"].mean()
        weekday_mean = daily_trade.loc[~daily_trade["is_weekend"], "trade"].mean()
        holiday_mean = daily_trade.loc[daily_trade["is_public_holiday"], "trade"].mean()
        non_holiday_mean = daily_trade.loc[~daily_trade["is_public_holiday"], "trade"].mean()

        self.assertLess(weekend_mean, weekday_mean)
        self.assertLess(holiday_mean, non_holiday_mean)

    def test_trade_drops_with_distance(self) -> None:
        pair_summary = self.edge_frame.groupby(["u", "i"], as_index=False).agg(
            trade=("trade", "mean"),
            distance_km=("distance_km", "mean"),
        )
        correlation = pair_summary["trade"].corr(pair_summary["distance_km"])
        self.assertLess(correlation, -0.25)


if __name__ == "__main__":
    unittest.main()
