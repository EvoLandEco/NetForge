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
        if not (DATASET_DIR / "edges.csv").exists():
            subprocess.run(
                [sys.executable, str(ROOT / "examples" / "toy_nl" / "build_toy_nl_dataset.py")],
                check=True,
                cwd=ROOT,
            )
        cls.edge_frame = pd.read_csv(DATASET_DIR / "edges.csv")
        cls.node_matrix = np.load(DATASET_DIR / "node_features.npy")
        cls.node_map = pd.read_csv(DATASET_DIR / "node_map.csv")
        cls.schema = json.loads((DATASET_DIR / "node_schema.json").read_text())

    def test_required_files_align(self) -> None:
        feature_columns = self.schema["node_feature_columns_in_order"]
        self.assertEqual(self.node_matrix.shape[1], len(feature_columns))
        self.assertEqual(self.node_matrix.shape[0], int(self.node_map["node_id"].max()) + 1)
        self.assertEqual(self.schema["node_row_offset"], 0)
        self.assertTrue({"u", "i", "ts", "trade"}.issubset(self.edge_frame.columns))
        max_node_id = int(self.node_map["node_id"].max())
        self.assertTrue((self.edge_frame["u"] >= 0).all())
        self.assertTrue((self.edge_frame["i"] >= 0).all())
        self.assertTrue((self.edge_frame["u"] <= max_node_id).all())
        self.assertTrue((self.edge_frame["i"] <= max_node_id).all())

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
