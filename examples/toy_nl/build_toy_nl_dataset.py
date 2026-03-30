"""Create the Dutch toy dataset used in the docs and tutorial."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = "TOY_NL"
OUT_DIR = ROOT / "examples" / "toy_nl" / "processed_data" / DATASET
SCHEMA_PATH = OUT_DIR / "node_schema.json"
NODE_PATH = OUT_DIR / "node_features.npy"
EDGE_PATH = OUT_DIR / "edges.csv"
NODE_MAP_PATH = OUT_DIR / "node_map.csv"
MANIFEST_PATH = OUT_DIR / "dataset_manifest.json"


@dataclass(frozen=True)
class CoropSeed:
    code: str
    name: str
    center_x: float
    center_y: float
    offsets: tuple[tuple[float, float], ...]
    cattle: tuple[int, ...]
    pigs: tuple[int, ...]


COROPS: tuple[CoropSeed, ...] = (
    CoropSeed(
        code="CR17",
        name="Utrecht",
        center_x=140_711.0,
        center_y=458_661.7,
        offsets=((-3_200.0, -1_800.0), (2_100.0, 1_300.0), (1_400.0, -2_700.0)),
        cattle=(180, 260, 210),
        pigs=(120, 170, 140),
    ),
    CoropSeed(
        code="CR23",
        name="Groot-Amsterdam",
        center_x=124_989.6,
        center_y=488_798.3,
        offsets=((-2_600.0, 1_400.0), (3_100.0, -1_200.0), (1_000.0, 2_400.0)),
        cattle=(240, 220, 280),
        pigs=(180, 160, 210),
    ),
    CoropSeed(
        code="CR24",
        name="Het Gooi en Vechtstreek",
        center_x=136_605.7,
        center_y=477_595.9,
        offsets=((-1_900.0, -1_300.0), (2_200.0, 900.0), (-1_100.0, 2_600.0)),
        cattle=(150, 170, 210),
        pigs=(110, 130, 160),
    ),
    CoropSeed(
        code="CR26",
        name="Agglomeratie 's-Gravenhage",
        center_x=85_103.3,
        center_y=453_820.5,
        offsets=((2_200.0, 1_600.0), (-2_500.0, -1_400.0), (1_400.0, -2_600.0)),
        cattle=(230, 210, 260),
        pigs=(170, 150, 190),
    ),
)

PUBLIC_HOLIDAYS = {
    date(2019, 12, 25),
    date(2019, 12, 26),
    date(2020, 1, 1),
}

DEFAULT_METADATA_FIELDS = (
    "corop",
    "num_farms_bin",
    "total_animals_bin",
    "centroid_grid",
    "ft_tokens",
)


def build_nodes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    node_id = 0
    for seed in COROPS:
        for farm_index, (offset_x, offset_y) in enumerate(seed.offsets, start=1):
            cattle = seed.cattle[farm_index - 1]
            pigs = seed.pigs[farm_index - 1]
            rows.append(
                {
                    "node_id": node_id,
                    "node_label": f"{seed.code}_farm_{farm_index}",
                    "type": "Farm",
                    "ubn": f"TOYNL{node_id:04d}",
                    "corop": seed.code,
                    "corop_name": seed.name,
                    "xco": seed.center_x + offset_x,
                    "yco": seed.center_y + offset_y,
                    "num_farms": 1.0,
                    "total_animals": float(cattle + pigs),
                    "count_ft_cattle": 1.0,
                    "count_ft_pig": 1.0,
                    "count_animal_cattle": float(cattle),
                    "count_animal_pig": float(pigs),
                }
            )
            node_id += 1
    return pd.DataFrame(rows)


def day_scale(day: date) -> float:
    scale = 1.0
    if day.weekday() >= 5:
        scale *= 0.38
    if day in PUBLIC_HOLIDAYS:
        scale *= 0.28
    return scale


def build_edges(node_frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260312)
    date_start = date(2019, 12, 16)
    date_end = date(2020, 1, 12)

    node_lookup = node_frame.set_index("node_id").to_dict(orient="index")
    sender_scale = {node_id: 0.92 + 0.03 * (node_id % 4) for node_id in node_lookup}
    receiver_scale = {node_id: 0.95 + 0.02 * (node_id % 3) for node_id in node_lookup}

    rows: list[dict[str, object]] = []
    current = date_start
    while current <= date_end:
        calendar_scale = day_scale(current)
        is_weekend = current.weekday() >= 5
        is_public_holiday = current in PUBLIC_HOLIDAYS
        ordinal = current.toordinal()

        for source_id, source_row in node_lookup.items():
            for target_id, target_row in node_lookup.items():
                if source_id == target_id:
                    continue

                dx = float(source_row["xco"]) - float(target_row["xco"])
                dy = float(source_row["yco"]) - float(target_row["yco"])
                distance_m = math.hypot(dx, dy)
                distance_km = distance_m / 1_000.0
                mass_term = math.sqrt(float(source_row["total_animals"]) * float(target_row["total_animals"]))
                same_corop_scale = 1.1 if source_row["corop"] == target_row["corop"] else 1.0
                gravity_mean = (
                    0.0024
                    * mass_term
                    * math.exp(-distance_m / 45_000.0)
                    * calendar_scale
                    * same_corop_scale
                    * sender_scale[source_id]
                    * receiver_scale[target_id]
                )
                trade = int(rng.poisson(gravity_mean))
                if trade <= 0:
                    continue

                rows.append(
                    {
                        "u": source_id,
                        "i": target_id,
                        "ts": ordinal,
                        "date": current.isoformat(),
                        "trade": trade,
                        "distance_km": round(distance_km, 3),
                        "gravity_mean": round(gravity_mean, 6),
                        "is_weekend": is_weekend,
                        "is_public_holiday": is_public_holiday,
                    }
                )
        current += timedelta(days=1)

    edge_frame = pd.DataFrame(rows).sort_values(["ts", "u", "i"]).reset_index(drop=True)
    return edge_frame


def write_dataset(node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = {
        "node_feature_columns_in_order": [
            "xco",
            "yco",
            "num_farms",
            "total_animals",
            "count_ft_cattle",
            "count_ft_pig",
            "count_animal_cattle",
            "count_animal_pig",
        ],
        "node_row_offset": 0,
        "indexing": "row n matches node_id n",
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")

    feature_columns = schema["node_feature_columns_in_order"]
    node_matrix = node_frame[feature_columns].to_numpy(dtype=float)
    np.save(NODE_PATH, node_matrix)

    edge_frame.to_csv(EDGE_PATH, index=False)
    node_frame[["node_id", "node_label", "type", "ubn", "corop", "corop_name"]].to_csv(
        NODE_MAP_PATH,
        index=False,
    )
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "format": "netforge_dataset_v1",
                "dataset": DATASET,
                "edge_file": EDGE_PATH.name,
                "node_features_file": NODE_PATH.name,
                "node_schema_file": SCHEMA_PATH.name,
                "node_map_file": NODE_MAP_PATH.name,
                "weight_column": "trade",
                "node_row_offset": 0,
                "joint_metadata_model": {
                    "enabled_by_default": True,
                    "layer_name": "__metadata__",
                    "metadata_fields": list(DEFAULT_METADATA_FIELDS),
                    "node_map_fields": ["corop"],
                    "node_feature_columns": [
                        "xco",
                        "yco",
                        "num_farms",
                        "total_animals",
                        "count_ft_cattle",
                        "count_ft_pig",
                    ],
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    node_frame = build_nodes()
    edge_frame = build_edges(node_frame)
    write_dataset(node_frame, edge_frame)
    print(f"Wrote {len(node_frame)} nodes to {NODE_PATH}")
    print(f"Wrote {len(edge_frame)} edges to {EDGE_PATH}")
    print(f"Wrote node map to {NODE_MAP_PATH}")
    print(f"Wrote schema to {SCHEMA_PATH}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
