
"""Transmission simulation and validation for hybrid temporal panels.

The simulation layer applies the same epidemic process to the observed panel and to
synthetic panels, then compares the resulting farm and region outcomes.

Model structure:
- Farm nodes carry compartment states.
- Region nodes carry a continuous import-pressure reservoir.
- The four hybrid edge channels use separate transmission multipliers.
- Farm outcomes are the main validation targets.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

STANDARD_EDGE_COLUMNS = ("u", "i", "ts")
SAMPLE_SUFFIX_PATTERN = re.compile(r"__sample_(?P<index>\d{4})$")
PLOT_COLORS = {
    "original": "#4e79a7",
    "synthetic": "#f28e2b",
    "novel": "#59a14f",
    "delta": "#e15759",
    "accent": "#76b7b2",
    "neutral": "#bab0ab",
    "grid": "#dce3ea",
    "grid_strong": "#c6d1dd",
    "text": "#22313f",
    "muted": "#5f6f7f",
    "panel": "#ffffff",
    "panel_soft": "#f7fafc",
    "figure": "#eef3f8",
    "farm": "#35c9c3",
    "region": "#ef8f7d",
}
SCENARIO_RANK_COLUMNS = [
    "farm_prevalence_curve_correlation",
    "farm_incidence_curve_correlation",
    "farm_attack_rate_wasserstein",
    "farm_peak_prevalence_wasserstein",
    "farm_duration_wasserstein",
    "region_reservoir_spatial_correlation_mean",
    "farm_attack_probability_correlation",
]
SCENARIO_RANK_ASCENDING = [False, False, True, True, True, False, False]
REGION_DAILY_METRICS = ("reservoir_pressure", "import_pressure", "export_pressure")
HYBRID_CHANNEL_DAILY_METRICS = ("farm_hazard_ff", "farm_hazard_rf", "region_pressure_fr", "region_pressure_rr")
TRAJECTORY_DISTANCE_METRICS = (
    "farm_prevalence",
    "farm_incidence",
    "farm_cumulative_incidence",
    "reservoir_total",
    "reservoir_max",
)
REGION_SPATIAL_SUMMARY_COLUMNS = [
    "region_reservoir_spatial_correlation_mean",
    "region_import_spatial_correlation_mean",
    "region_export_spatial_correlation_mean",
    "region_reservoir_temporal_correlation_mean",
    "region_import_temporal_correlation_mean",
    "region_export_temporal_correlation_mean",
]


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    description: str
    overrides: dict[str, object]


# Utilities
def _save_json(payload: dict, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def _load_json_if_exists(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(subvalue) for subvalue in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(subvalue) for subvalue in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _resolve_corop_geojson_path(manifest: dict, override: Optional[str] = None) -> Optional[Path]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(str(override)).expanduser().resolve())
    if manifest.get("corop_geojson"):
        candidates.append(Path(str(manifest["corop_geojson"])).expanduser().resolve())
    if manifest.get("dataset_dir"):
        dataset_dir = Path(str(manifest["dataset_dir"])).expanduser().resolve()
        try:
            candidates.append(dataset_dir.parents[2] / "public" / "nl_corop.geojson")
        except Exception:
            pass
        candidates.append(dataset_dir / "nl_corop.geojson")
    if manifest.get("data_root"):
        try:
            candidates.append(Path(str(manifest["data_root"])).expanduser().resolve().parent / "public" / "nl_corop.geojson")
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ts_display_label(ts_value: int | float) -> str:
    try:
        return pd.Timestamp.fromordinal(int(round(float(ts_value)))).strftime("%Y-%m-%d")
    except Exception:
        return str(int(round(float(ts_value))))


def _calendar_records(ts_values: Iterable[float]) -> list[dict[str, object]]:
    values = np.asarray(list(ts_values), dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    unique_ts = sorted({int(round(float(value))) for value in values})
    return [
        {"ts": int(ts_value), "label": _ts_display_label(ts_value), "offset": int(index)}
        for index, ts_value in enumerate(unique_ts)
    ]


def _safe_corop_label(value: object, fallback: str) -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _safe_correlation(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if x.size == 1:
        return 1.0 if np.isclose(x[0], y[0]) else 0.0
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 and y_std <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _metric_text(value: object, *, digits: int = 4) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "nan"
    return f"{float(numeric):.{digits}f}"


def _frame_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.full(len(frame), np.nan), index=frame.index, dtype=float)


def _frame_metric_values(frame: pd.DataFrame, metric_name: str) -> np.ndarray:
    if frame.empty or metric_name not in frame.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(frame[metric_name], errors="coerce").dropna().to_numpy(dtype=float)


def _wasserstein_distance_1d(observed: np.ndarray, synthetic: np.ndarray) -> float:
    x = np.sort(np.asarray(observed, dtype=float)[np.isfinite(observed)])
    y = np.sort(np.asarray(synthetic, dtype=float)[np.isfinite(synthetic)])
    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0:
        return np.nan
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(x, y))
    except Exception:
        if len(x) == len(y):
            return float(np.mean(np.abs(x - y)))
        q_count = max(len(x), len(y))
        q = (np.arange(q_count, dtype=float) + 0.5) / q_count
        xq = np.quantile(x, q)
        yq = np.quantile(y, q)
        return float(np.mean(np.abs(xq - yq)))


def _energy_distance_1d(observed: np.ndarray, synthetic: np.ndarray) -> float:
    x = np.asarray(observed, dtype=float)
    y = np.asarray(synthetic, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0:
        return np.nan
    cross = float(np.mean(np.abs(x[:, None] - y[None, :])))
    within_x = float(np.mean(np.abs(x[:, None] - x[None, :]))) if len(x) > 1 else 0.0
    within_y = float(np.mean(np.abs(y[:, None] - y[None, :]))) if len(y) > 1 else 0.0
    return float(max(0.0, 2.0 * cross - within_x - within_y))


def _safe_divide(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return np.nan
    return float(numerator) / denom


def _coerce_2d_finite_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        arr = np.reshape(arr, (arr.shape[0], int(np.prod(arr.shape[1:]))))
    finite_rows = np.all(np.isfinite(arr), axis=1)
    return arr[finite_rows]


def _ensemble_energy_distance(observed_samples: np.ndarray, synthetic_samples: np.ndarray) -> float:
    x = _coerce_2d_finite_rows(np.asarray(observed_samples, dtype=float))
    y = _coerce_2d_finite_rows(np.asarray(synthetic_samples, dtype=float))
    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0:
        return np.nan
    cross = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    within_x = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2) if len(x) > 1 else np.zeros((1, 1), dtype=float)
    within_y = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2) if len(y) > 1 else np.zeros((1, 1), dtype=float)
    score = 2.0 * float(np.mean(cross)) - float(np.mean(within_x)) - float(np.mean(within_y))
    return float(max(0.0, score))


def _ensemble_variogram_distance(observed_samples: np.ndarray, synthetic_samples: np.ndarray, *, p: float = 1.0) -> float:
    x = _coerce_2d_finite_rows(np.asarray(observed_samples, dtype=float))
    y = _coerce_2d_finite_rows(np.asarray(synthetic_samples, dtype=float))
    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0 or x.shape[1] != y.shape[1]:
        return np.nan
    dimension = int(x.shape[1])
    if dimension <= 1:
        return 0.0
    xv = np.mean(np.abs(x[:, :, None] - x[:, None, :]) ** float(p), axis=0)
    yv = np.mean(np.abs(y[:, :, None] - y[:, None, :]) ** float(p), axis=0)
    upper = np.triu_indices(dimension, k=1)
    diff = xv[upper] - yv[upper]
    return float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else 0.0


def _best_lagged_correlation(
    original_values: Iterable[float],
    synthetic_values: Iterable[float],
    *,
    max_lag: int = 7,
) -> dict[str, float]:
    x = np.asarray(list(original_values), dtype=float)
    y = np.asarray(list(synthetic_values), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return {"best_lag_days": np.nan, "best_lag_correlation": np.nan}
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: len(y) + lag]
        elif lag > 0:
            xs = x[: len(x) - lag]
            ys = y[lag:]
        else:
            xs = x
            ys = y
        if len(xs) < 2 or len(ys) < 2:
            continue
        corr = _safe_correlation(xs, ys)
        if (corr > best_corr + 1e-12) or (abs(corr - best_corr) <= 1e-12 and abs(lag) < abs(best_lag)):
            best_corr = corr
            best_lag = lag
    if not np.isfinite(best_corr):
        return {"best_lag_days": np.nan, "best_lag_correlation": np.nan}
    return {"best_lag_days": float(best_lag), "best_lag_correlation": float(best_corr)}


def _top_fraction_overlap(
    original_values: np.ndarray,
    synthetic_values: np.ndarray,
    *,
    fraction: float = 0.10,
    min_k: int = 10,
) -> float:
    original = np.asarray(original_values, dtype=float)
    synthetic = np.asarray(synthetic_values, dtype=float)
    if original.size == 0 or synthetic.size == 0 or original.size != synthetic.size:
        return np.nan
    size = min(int(original.size), max(int(min_k), int(math.ceil(float(fraction) * int(original.size)))))
    if size <= 0:
        return np.nan
    original_rank = set(np.argsort(-original)[:size].tolist())
    synthetic_rank = set(np.argsort(-synthetic)[:size].tolist())
    return float(len(original_rank & synthetic_rank) / size)


def _global_moran_i(values: np.ndarray, coords: np.ndarray, *, k: int = 8) -> float:
    x = np.asarray(values, dtype=float)
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return np.nan
    valid = np.isfinite(x) & np.all(np.isfinite(xy), axis=1)
    x = x[valid]
    xy = xy[valid]
    n = int(len(x))
    if n < 3:
        return np.nan
    centered = x - float(np.mean(x))
    denominator = float(np.sum(centered ** 2))
    if denominator <= 1e-12:
        return 0.0
    neighbour_count = min(max(1, int(k)), n - 1)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        _, neighbour_idx = tree.query(xy, k=neighbour_count + 1)
        neighbour_idx = np.asarray(neighbour_idx, dtype=int)
        if neighbour_idx.ndim == 1:
            neighbour_idx = neighbour_idx[:, None]
        neighbours = neighbour_idx[:, 1:]
    except Exception:
        dist = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        neighbours = np.argsort(dist, axis=1)[:, :neighbour_count]
    numerator = 0.0
    total_weight = 0.0
    for i in range(n):
        neigh = np.asarray(neighbours[i], dtype=int)
        neigh = neigh[(neigh >= 0) & (neigh < n) & (neigh != i)]
        if len(neigh) == 0:
            continue
        weight = 1.0 / float(len(neigh))
        numerator += float(np.sum(weight * centered[i] * centered[neigh]))
        total_weight += weight * len(neigh)
    if total_weight <= 0:
        return np.nan
    return float((n / total_weight) * (numerator / denominator))


def _format_node_type_label(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "Unknown"
        lowered = text.lower()
        if lowered in {"farm", "farms"}:
            return "Farm"
        if lowered in {"region", "regions", "supernode", "regional supernode"}:
            return "Region"
        if lowered in {"f"}:
            return "Farm"
        if lowered in {"r"}:
            return "Region"
        if lowered in {"type 0", "type0"}:
            return "Farm"
        if lowered in {"type 1", "type1"}:
            return "Region"
        return text.replace("_", " ").title()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if float(numeric).is_integer():
        integer = int(numeric)
        if integer == 0:
            return "Farm"
        if integer == 1:
            return "Region"
        return f"Type {integer}"
    return str(value)


def _setting_label_from_sample_label(sample_label: str) -> str:
    return SAMPLE_SUFFIX_PATTERN.sub("", str(sample_label))


def _sample_index_from_label(sample_label: str) -> Optional[int]:
    match = SAMPLE_SUFFIX_PATTERN.search(str(sample_label))
    if not match:
        return None
    try:
        return int(match.group("index"))
    except Exception:
        return None


def _parse_sample_label_parts(sample_label: str) -> tuple[str, str]:
    clean_label = _setting_label_from_sample_label(sample_label)
    if "__rewire_" not in clean_label:
        return clean_label, "none"
    sample_mode, rewire_mode = clean_label.split("__rewire_", 1)
    return sample_mode, rewire_mode


def _display_sampler_name(sample_mode: str) -> str:
    mapping = {
        "micro": "Microcanonical SBM",
        "canonical_posterior": "Canonical posterior SBM",
        "canonical_ml": "Canonical maximum-likelihood SBM",
        "maxent_micro": "Max-entropy microcanonical SBM",
        "canonical_maxent": "Canonical max-entropy SBM",
    }
    return mapping.get(sample_mode, sample_mode.replace("_", " ").replace("-", " ").title())


def _display_rewire_name(rewire_mode: str) -> str:
    canonical = rewire_mode.replace("_", "-")
    mapping = {
        "none": "No rewiring",
        "configuration": "Configuration rewiring",
        "constrained-configuration": "Constrained configuration rewiring",
        "blockmodel-micro": "Blockmodel micro rewiring",
    }
    return mapping.get(canonical, canonical.replace("-", " ").title())


def _default_sample_class(sample_label: str) -> str:
    _, rewire_mode = _parse_sample_label_parts(sample_label)
    return "posterior_predictive" if rewire_mode == "none" else "sensitivity_analysis"


def _setting_display_payload(sample_label: str) -> dict[str, str]:
    sample_mode, rewire_mode = _parse_sample_label_parts(sample_label)
    sampler_name = _display_sampler_name(sample_mode)
    rewire_name = _display_rewire_name(rewire_mode)
    short_label = sampler_name if rewire_name == "No rewiring" else f"{sampler_name} + {rewire_name}"
    return {
        "sample_label": sample_label,
        "sample_mode": sample_mode,
        "rewire_mode": rewire_mode,
        "sampler_name": sampler_name,
        "rewire_name": rewire_name,
        "short_label": short_label,
    }


def _log_edge_frame_debug(
    label: str,
    frame: pd.DataFrame,
    directed: bool,
    weight_col: Optional[str] = None,
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    node_count = 0
    if {"u", "i"}.issubset(frame.columns):
        node_count = int(len(set(frame["u"].tolist()) | set(frame["i"].tolist())))
    layer_count = int(frame["ts"].nunique()) if "ts" in frame.columns else 0
    LOGGER.debug(
        "%s | rows=%s | directed=%s | nodes=%s | layers=%s | columns=%s",
        label,
        len(frame),
        directed,
        node_count,
        layer_count,
        frame.columns.tolist(),
    )
    if weight_col and weight_col in frame.columns:
        arr = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        LOGGER.debug(
            "%s | weight_col=%s | count=%s | mean=%.6f | median=%.6f | max=%.6f",
            label, weight_col, len(arr), float(arr.mean()) if len(arr) else 0.0,
            float(np.median(arr)) if len(arr) else 0.0, float(arr.max()) if len(arr) else 0.0
        )


def canonicalise_edge_frame(
    df: pd.DataFrame,
    directed: bool,
    src_col: str = "u",
    dst_col: str = "i",
    ts_col: str = "ts",
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    required = [src_col, dst_col, ts_col]
    if weight_col:
        required.append(weight_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    keep_columns = [src_col, dst_col, ts_col] + ([weight_col] if weight_col else [])
    frame = df[keep_columns].copy()
    for column in (src_col, dst_col, ts_col):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    if weight_col:
        frame[weight_col] = pd.to_numeric(frame[weight_col], errors="raise").astype(float)
    frame = frame.rename(columns={src_col: "u", dst_col: "i", ts_col: "ts"})
    if not directed:
        uv = np.sort(frame[["u", "i"]].to_numpy(dtype=np.int64, copy=False), axis=1)
        frame["u"] = uv[:, 0]
        frame["i"] = uv[:, 1]
    if weight_col:
        frame = (
            frame.groupby(["u", "i", "ts"], as_index=False, sort=False)[weight_col]
            .sum()
            .reset_index(drop=True)
        )
    else:
        frame = frame.drop_duplicates(STANDARD_EDGE_COLUMNS).reset_index(drop=True)
    _log_edge_frame_debug("Canonicalised edge frame", frame, directed=directed, weight_col=weight_col)
    return frame


def _resolve_node_map_path(manifest: dict) -> Optional[Path]:
    if manifest.get("node_map_csv"):
        candidate = Path(str(manifest["node_map_csv"])).expanduser().resolve()
        return candidate if candidate.exists() else None
    if manifest.get("dataset_dir"):
        candidate = Path(str(manifest["dataset_dir"])).expanduser().resolve() / "node_map.csv"
        return candidate if candidate.exists() else None
    return None


def _load_hybrid_node_frame(run_dir: Path, manifest: dict) -> pd.DataFrame:
    node_attributes_path = Path(str(manifest.get("node_attributes_path") or run_dir / "node_attributes.csv"))
    if not node_attributes_path.exists():
        return pd.DataFrame(columns=["node_id", "x", "y", "type_label", "corop", "ubn", "block_id", "num_farms", "total_animals"])
    node_frame = pd.read_csv(node_attributes_path)
    if "node_id" not in node_frame.columns:
        return pd.DataFrame(columns=["node_id", "x", "y", "type_label", "corop", "ubn", "block_id", "num_farms", "total_animals"])
    node_frame["node_id"] = pd.to_numeric(node_frame["node_id"], errors="coerce").astype("Int64")
    node_frame = node_frame.dropna(subset=["node_id"]).copy()
    node_frame["node_id"] = node_frame["node_id"].astype(int)

    node_map_path = _resolve_node_map_path(manifest)
    if node_map_path is not None and node_map_path.exists():
        node_map = pd.read_csv(node_map_path)
        if "node_id" in node_map.columns:
            keep_columns = ["node_id"]
            for column in ("type", "ubn", "corop"):
                if column in node_map.columns:
                    keep_columns.append(column)
            hybrid_map = node_map[keep_columns].drop_duplicates(subset=["node_id"]).copy()
            if "type" in hybrid_map.columns:
                hybrid_map = hybrid_map.rename(columns={"type": "node_map_type"})
            if "ubn" in hybrid_map.columns:
                hybrid_map = hybrid_map.rename(columns={"ubn": "node_map_ubn"})
            if "corop" in hybrid_map.columns:
                hybrid_map = hybrid_map.rename(columns={"corop": "node_map_corop"})
            node_frame = node_frame.merge(hybrid_map, on="node_id", how="left")

    if "type_label" in node_frame.columns:
        if "node_map_type" in node_frame.columns:
            node_frame["type_label"] = node_frame["type_label"].fillna(node_frame["node_map_type"])
    elif "node_map_type" in node_frame.columns:
        node_frame["type_label"] = node_frame["node_map_type"]
    elif "type" in node_frame.columns:
        node_frame["type_label"] = node_frame["type"]
    else:
        node_frame["type_label"] = "Unknown"
    node_frame["type_label"] = node_frame["type_label"].map(_format_node_type_label)
    corop_sources: list[pd.Series] = []
    if "corop" in node_frame.columns:
        corop_sources.append(node_frame["corop"])
    if "node_map_corop" in node_frame.columns:
        corop_sources.append(node_frame["node_map_corop"])
    if corop_sources:
        corop_series = corop_sources[0].copy()
        for source in corop_sources[1:]:
            corop_series = corop_series.where(corop_series.notna() & (corop_series.astype(str).str.strip() != ""), source)
        node_frame["corop"] = corop_series.fillna("").astype(str)
    else:
        node_frame["corop"] = ""

    ubn_sources: list[pd.Series] = []
    if "ubn" in node_frame.columns:
        ubn_sources.append(node_frame["ubn"])
    if "node_map_ubn" in node_frame.columns:
        ubn_sources.append(node_frame["node_map_ubn"])
    if ubn_sources:
        ubn_series = ubn_sources[0].copy()
        for source in ubn_sources[1:]:
            ubn_series = ubn_series.where(ubn_series.notna() & (ubn_series.astype(str).str.strip() != ""), source)
        node_frame["ubn"] = ubn_series
    else:
        node_frame["ubn"] = np.nan
    if "block_id" not in node_frame.columns:
        node_frame["block_id"] = -1
    return node_frame.sort_values("node_id").reset_index(drop=True)


def _build_node_type_map(node_frame: pd.DataFrame) -> dict[int, str]:
    if node_frame.empty or "node_id" not in node_frame.columns:
        return {}
    return {
        int(node_id): _format_node_type_label(type_value)
        for node_id, type_value in zip(
            pd.to_numeric(node_frame["node_id"], errors="coerce").fillna(-1).astype(int),
            node_frame["type_label"] if "type_label" in node_frame.columns else ["Unknown"] * len(node_frame),
        )
        if int(node_id) >= 0
    }


def _focal_corop_from_node_frame(node_frame: pd.DataFrame) -> str:
    if node_frame.empty or "type_label" not in node_frame.columns or "corop" not in node_frame.columns:
        return ""
    farm_corops = node_frame.loc[node_frame["type_label"].astype(str) == "Farm", "corop"].astype(str).str.strip()
    farm_corops = farm_corops.loc[farm_corops != ""]
    if farm_corops.empty:
        return ""
    mode = farm_corops.mode()
    return str(mode.iloc[0]) if len(mode) else str(farm_corops.iloc[0])


def _build_region_node_frame(
    node_frame: pd.DataFrame,
    *,
    node_universe: list[int],
    node_types: dict[int, str],
) -> pd.DataFrame:
    if not node_universe:
        return pd.DataFrame(columns=["region_order", "region_node_id", "corop", "x", "y", "display_label"])

    lookup = node_frame.copy()
    if "node_id" not in lookup.columns:
        lookup = pd.DataFrame(columns=["node_id", "corop", "x", "y", "ubn"])
    else:
        lookup = lookup.drop_duplicates(subset=["node_id"]).set_index("node_id", drop=False)

    rows: list[dict[str, object]] = []
    for node_id in node_universe:
        if node_types.get(int(node_id), "Unknown") != "Region":
            continue
        fallback = f"region_{int(node_id)}"
        if int(node_id) in lookup.index:
            source = lookup.loc[int(node_id)]
            corop = _safe_corop_label(source.get("corop"), fallback)
            x_value = pd.to_numeric(pd.Series([source.get("x")]), errors="coerce").iloc[0] if "x" in lookup.columns else np.nan
            y_value = pd.to_numeric(pd.Series([source.get("y")]), errors="coerce").iloc[0] if "y" in lookup.columns else np.nan
        else:
            corop = fallback
            x_value = np.nan
            y_value = np.nan
        rows.append(
            {
                "region_order": int(len(rows)),
                "region_node_id": int(node_id),
                "corop": str(corop),
                "x": np.nan if pd.isna(x_value) else float(x_value),
                "y": np.nan if pd.isna(y_value) else float(y_value),
                "display_label": str(corop),
            }
        )
    return pd.DataFrame(rows)


def load_run_manifest(run_dir: Path) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    payload["manifest_path"] = str(manifest_path)
    return payload


def _discover_sample_manifests(run_dir: Path, manifest: dict) -> list[dict]:
    discovered: list[dict] = []

    def _normalise_payload(payload: dict, manifest_path: Path) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        sample_dir = Path(str(payload.get("sample_dir") or manifest_path.parent)).expanduser().resolve()
        synthetic_path = Path(str(payload.get("synthetic_edges_csv") or sample_dir / "synthetic_edges.csv")).expanduser().resolve()
        if not synthetic_path.exists():
            return None
        sample_index = payload.get("sample_index")
        try:
            sample_index_int = int(sample_index) if sample_index is not None else None
        except Exception:
            sample_index_int = None
        parent_label = manifest_path.parent.parent.name if manifest_path.parent.parent.name != "generated" else ""
        sample_label = str(payload.get("sample_label") or manifest_path.parent.name)
        setting_label = str(payload.get("setting_label") or parent_label or _setting_label_from_sample_label(sample_label))
        if sample_index_int is not None and not sample_label.endswith(f"{sample_index_int:04d}"):
            sample_label = f"{setting_label}__sample_{sample_index_int:04d}"
        elif sample_label.startswith("sample_") and setting_label and sample_label != setting_label:
            sample_label = f"{setting_label}__{sample_label}"
        sample_settings = payload.get("sample_settings") or {}
        is_sensitivity = bool(sample_settings.get("is_sensitivity_analysis")) or str(sample_settings.get("rewire_model", "none")) != "none"
        output = dict(payload)
        output["sample_manifest_path"] = str(manifest_path)
        output["sample_dir"] = str(sample_dir)
        output["synthetic_edges_csv"] = str(synthetic_path)
        output["setting_label"] = setting_label
        output["sample_label"] = sample_label
        output["sample_class"] = str(output.get("sample_class") or ("sensitivity_analysis" if is_sensitivity else "posterior_predictive"))
        output["sample_index"] = sample_index_int
        return output

    for record in manifest.get("generated_samples", []):
        if not isinstance(record, dict):
            continue
        manifest_path = Path(str(record.get("sample_manifest_path") or Path(str(record.get("sample_dir", run_dir))) / "sample_manifest.json"))
        payload = _load_json_if_exists(manifest_path) or record
        normalised = _normalise_payload(payload, manifest_path)
        if normalised is not None:
            discovered.append(normalised)

    manifest_paths = sorted(set(
        list((run_dir / "generated").glob("sample_*/sample_manifest.json")) +
        list((run_dir / "generated").glob("*/sample_*/sample_manifest.json"))
    ))
    for manifest_path in manifest_paths:
        payload = _load_json_if_exists(manifest_path)
        normalised = _normalise_payload(payload or {}, manifest_path)
        if normalised is not None:
            discovered.append(normalised)

    if not discovered:
        raise FileNotFoundError(f"No generated sample manifests were found under {run_dir / 'generated'}")

    frame = pd.DataFrame(discovered).drop_duplicates(subset=["sample_manifest_path"]).reset_index(drop=True)
    relabelled: list[dict] = []
    for setting_label, setting_frame in frame.groupby("setting_label", sort=True):
        setting_frame = setting_frame.sort_values(["sample_index", "sample_manifest_path"], na_position="last").reset_index(drop=True)
        for idx, row in setting_frame.iterrows():
            payload = row.to_dict()
            payload["sample_label"] = setting_label if len(setting_frame) == 1 else f"{setting_label}__sample_{idx:04d}"
            relabelled.append(payload)
    return relabelled


def _filter_sample_manifests(
    sample_manifests: list[dict],
    *,
    setting_labels: Optional[list[str]] = None,
    sample_label_pattern: Optional[str] = None,
) -> list[dict]:
    if not sample_manifests:
        return []

    filtered = list(sample_manifests)
    requested_settings = [str(label).strip() for label in (setting_labels or []) if str(label).strip()]
    if requested_settings:
        requested = set(requested_settings)
        filtered = [payload for payload in filtered if str(payload.get("setting_label") or "") in requested]
        if not filtered:
            raise ValueError(
                "No generated samples matched the requested setting labels: "
                + ", ".join(sorted(requested))
            )

    if sample_label_pattern:
        pattern = re.compile(str(sample_label_pattern))
        filtered = [
            payload for payload in filtered
            if pattern.search(str(payload.get("sample_label") or "")) or pattern.search(str(payload.get("setting_label") or ""))
        ]
        if not filtered:
            raise ValueError(f"No generated samples matched the sample-label pattern: {sample_label_pattern}")

    return filtered


# Simulation settings
@dataclass(frozen=True)
class HybridSimulationConfig:
    model: str = "SEIR"
    beta_ff: float = 0.20
    beta_fr: float = 0.04
    beta_rf: float = 0.16
    beta_rr: float = 0.02
    sigma: float = 0.35
    gamma: float = 0.10
    num_replicates: int = 256
    seed: int = 42
    initial_seed_count: int = 3
    weight_mode: str = "log1p"  # binary | linear | sqrt | log1p
    weight_scale: Optional[float] = None
    tail_days: int = 30
    seed_scope: str = "farm_only"  # farm_only | all_farms
    seed_pool_mode: str = "observed_day0"  # observed_day0 | common_day0 | overall
    require_day0_activity: bool = True
    farm_susceptibility: float = 1.0
    farm_infectiousness: float = 1.0
    reservoir_decay: float = 0.70
    reservoir_background: float = 0.0
    reservoir_clip: float = 20.0
    farm_daily_import_prob: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_config(config: HybridSimulationConfig) -> None:
    if config.model not in {"SIR", "SEIR", "SIS"}:
        raise ValueError("model must be one of SIR, SEIR, SIS")
    if config.weight_mode not in {"binary", "linear", "sqrt", "log1p"}:
        raise ValueError("weight_mode must be one of binary, linear, sqrt, log1p")
    if config.seed_scope not in {"farm_only", "all_farms"}:
        raise ValueError("seed_scope must be farm_only or all_farms")
    if config.seed_pool_mode not in {"observed_day0", "common_day0", "overall"}:
        raise ValueError("seed_pool_mode must be observed_day0, common_day0, or overall")
    for name, value in {
        "beta_ff": config.beta_ff,
        "beta_fr": config.beta_fr,
        "beta_rf": config.beta_rf,
        "beta_rr": config.beta_rr,
        "sigma": config.sigma,
        "gamma": config.gamma,
        "farm_susceptibility": config.farm_susceptibility,
        "farm_infectiousness": config.farm_infectiousness,
        "reservoir_decay": config.reservoir_decay,
        "reservoir_background": config.reservoir_background,
    }.items():
        if float(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if config.initial_seed_count <= 0 or config.num_replicates <= 0:
        raise ValueError("initial_seed_count and num_replicates must be positive")
    if config.tail_days < 0:
        raise ValueError("tail_days must be nonnegative")
    if not (0.0 <= float(config.farm_daily_import_prob) < 1.0):
        raise ValueError("farm_daily_import_prob must lie in [0, 1)")


# Panel packing
def _weight_transform(raw_weights: np.ndarray, mode: str) -> np.ndarray:
    weights = np.asarray(raw_weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if mode == "binary":
        return np.where(weights > 0, 1.0, 0.0)
    if mode == "linear":
        return weights
    if mode == "sqrt":
        return np.sqrt(weights)
    if mode == "log1p":
        return np.log1p(weights)
    raise ValueError(f"Unsupported weight mode: {mode}")


def _auto_weight_scale(observed_edges: pd.DataFrame, weight_col: Optional[str], mode: str) -> float:
    if not weight_col or weight_col not in observed_edges.columns or mode == "binary":
        return 1.0
    transformed = _weight_transform(pd.to_numeric(observed_edges[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), mode)
    transformed = transformed[np.isfinite(transformed) & (transformed > 0)]
    if len(transformed) == 0:
        return 1.0
    scale = float(np.median(transformed))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


@dataclass(frozen=True)
class HybridPanelPack:
    label: str
    node_universe: tuple[int, ...]
    ts_values: tuple[int, ...]
    src: tuple[np.ndarray, ...]
    dst: tuple[np.ndarray, ...]
    weight: tuple[np.ndarray, ...]
    active_by_ts: tuple[np.ndarray, ...]
    is_farm: np.ndarray
    is_region: np.ndarray


def _log_panel_pack_debug(pack: HybridPanelPack) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    edge_counts = np.asarray([len(src) for src in pack.src], dtype=float)
    weight_totals = np.asarray([float(np.sum(weight)) for weight in pack.weight], dtype=float)
    active_counts = np.asarray([float(np.sum(active)) for active in pack.active_by_ts], dtype=float)
    LOGGER.debug(
        "Packed hybrid panel | label=%s | nodes=%s | farms=%s | regions=%s | snapshots=%s | total_edges=%s | mean_edges=%.2f | mean_weight_total=%.4f | mean_active_nodes=%.2f",
        pack.label,
        len(pack.node_universe),
        int(np.sum(pack.is_farm)),
        int(np.sum(pack.is_region)),
        len(pack.ts_values),
        int(np.sum(edge_counts)),
        float(edge_counts.mean()) if len(edge_counts) else 0.0,
        float(weight_totals.mean()) if len(weight_totals) else 0.0,
        float(active_counts.mean()) if len(active_counts) else 0.0,
    )


def _log_seed_pool_debug(
    *,
    seed_pool: np.ndarray,
    observed_pack: HybridPanelPack,
    synthetic_day0_activity_mask: Optional[np.ndarray],
    config: "HybridSimulationConfig",
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    observed_day0 = np.asarray(observed_pack.active_by_ts[0], dtype=bool) if observed_pack.active_by_ts else np.zeros(len(observed_pack.node_universe), dtype=bool)
    observed_day0_farms = int(np.sum(observed_day0 & np.asarray(observed_pack.is_farm, dtype=bool)))
    synthetic_day0_farms = (
        int(np.sum(np.asarray(synthetic_day0_activity_mask, dtype=bool) & np.asarray(observed_pack.is_farm, dtype=bool)))
        if synthetic_day0_activity_mask is not None else 0
    )
    LOGGER.debug(
        "Seed-pool summary | seed_scope=%s | seed_pool_mode=%s | require_day0_activity=%s | eligible_farms=%s | observed_day0_farms=%s | synthetic_day0_farms=%s | initial_seed_count=%s",
        config.seed_scope,
        config.seed_pool_mode,
        config.require_day0_activity,
        len(seed_pool),
        observed_day0_farms,
        synthetic_day0_farms,
        config.initial_seed_count,
    )
    if len(seed_pool):
        preview = ", ".join(str(int(node_id)) for node_id in seed_pool[: min(len(seed_pool), 8)])
        LOGGER.debug("Seed-pool preview | first_nodes=%s", preview)


def _prepare_snapshot_arrays(
    frame: pd.DataFrame,
    *,
    ts_values: list[int],
    node_index: dict[int, int],
    weight_col: Optional[str],
    weight_mode: str,
    weight_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []
    active_list: list[np.ndarray] = []

    grouped = {int(ts): snap for ts, snap in frame.groupby("ts", sort=True)}
    node_count = len(node_index)
    for ts_value in ts_values:
        snapshot = grouped.get(int(ts_value))
        if snapshot is None or snapshot.empty:
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)
            weight = np.array([], dtype=float)
        else:
            src = np.asarray([node_index[int(v)] for v in snapshot["u"].to_numpy(dtype=np.int64, copy=False)], dtype=np.int64)
            dst = np.asarray([node_index[int(v)] for v in snapshot["i"].to_numpy(dtype=np.int64, copy=False)], dtype=np.int64)
            if weight_col and weight_col in snapshot.columns:
                raw = pd.to_numeric(snapshot[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                raw = np.ones(len(src), dtype=float)
            weight = _weight_transform(raw, weight_mode)
            if weight_scale > 0:
                weight = weight / float(weight_scale)
        active = np.zeros(node_count, dtype=bool)
        if len(src):
            active[src] = True
            active[dst] = True
        src_list.append(src)
        dst_list.append(dst)
        weight_list.append(weight)
        active_list.append(active)
    return src_list, dst_list, weight_list, active_list


def _build_hybrid_panel_pack(
    frame: pd.DataFrame,
    *,
    label: str,
    node_universe: list[int],
    ts_values: list[int],
    weight_col: Optional[str],
    weight_mode: str,
    weight_scale: float,
    node_types: dict[int, str],
) -> HybridPanelPack:
    node_index = {int(node_id): idx for idx, node_id in enumerate(node_universe)}
    src_list, dst_list, weight_list, active_list = _prepare_snapshot_arrays(
        frame,
        ts_values=ts_values,
        node_index=node_index,
        weight_col=weight_col,
        weight_mode=weight_mode,
        weight_scale=weight_scale,
    )
    type_labels = np.asarray([node_types.get(int(node_id), "Unknown") for node_id in node_universe], dtype=object)
    is_farm = type_labels == "Farm"
    is_region = type_labels == "Region"
    pack = HybridPanelPack(
        label=label,
        node_universe=tuple(int(node_id) for node_id in node_universe),
        ts_values=tuple(int(ts) for ts in ts_values),
        src=tuple(src_list),
        dst=tuple(dst_list),
        weight=tuple(weight_list),
        active_by_ts=tuple(active_list),
        is_farm=is_farm,
        is_region=is_region,
    )
    _log_panel_pack_debug(pack)
    return pack


def _build_seed_pool(
    *,
    observed_pack: HybridPanelPack,
    synthetic_day0_activity_mask: Optional[np.ndarray],
    config: HybridSimulationConfig,
) -> np.ndarray:
    node_universe = np.asarray(observed_pack.node_universe, dtype=np.int64)
    farm_mask = np.asarray(observed_pack.is_farm, dtype=bool)
    if config.seed_scope == "all_farms":
        activity_mask = np.ones(len(node_universe), dtype=bool)
    elif config.seed_pool_mode == "overall" or not config.require_day0_activity:
        activity_mask = np.ones(len(node_universe), dtype=bool)
    elif config.seed_pool_mode == "common_day0":
        synthetic_mask = (
            np.asarray(synthetic_day0_activity_mask, dtype=bool)
            if synthetic_day0_activity_mask is not None
            else np.ones(len(node_universe), dtype=bool)
        )
        activity_mask = np.asarray(observed_pack.active_by_ts[0], dtype=bool) & synthetic_mask
    else:
        activity_mask = np.asarray(observed_pack.active_by_ts[0], dtype=bool)
    pool = node_universe[farm_mask & activity_mask]
    if len(pool) == 0:
        pool = node_universe[farm_mask]
    if len(pool) == 0:
        raise ValueError("No eligible farm nodes are available for seeding.")
    _log_seed_pool_debug(
        seed_pool=pool,
        observed_pack=observed_pack,
        synthetic_day0_activity_mask=synthetic_day0_activity_mask,
        config=config,
    )
    return np.asarray(pool, dtype=np.int64)


def _build_initial_seed_sets(
    *,
    observed_pack: HybridPanelPack,
    synthetic_day0_activity_mask: Optional[np.ndarray],
    config: HybridSimulationConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    master_rng = np.random.default_rng(int(config.seed))
    seed_pool = _build_seed_pool(
        observed_pack=observed_pack,
        synthetic_day0_activity_mask=synthetic_day0_activity_mask,
        config=config,
    )
    size = min(int(config.initial_seed_count), len(seed_pool))
    seed_sets = []
    for _ in range(int(config.num_replicates)):
        seed_sets.append(np.asarray(master_rng.choice(seed_pool, size=size, replace=False), dtype=np.int64))
    run_seeds = master_rng.integers(0, np.iinfo(np.int32).max, size=int(config.num_replicates), dtype=np.int64)
    return run_seeds, seed_sets


def _build_common_day0_activity_mask(
    *,
    synthetic_frames: list[pd.DataFrame],
    node_universe: list[int],
    day0_ts: int,
) -> Optional[np.ndarray]:
    if not synthetic_frames:
        return None
    common_nodes: Optional[set[int]] = None
    for frame in synthetic_frames:
        snapshot = frame.loc[frame["ts"].astype(int) == int(day0_ts), ["u", "i"]]
        active_nodes = set(snapshot["u"].astype(int).tolist()) | set(snapshot["i"].astype(int).tolist())
        common_nodes = active_nodes if common_nodes is None else common_nodes & active_nodes
    if common_nodes is None:
        return None
    return np.asarray([int(node_id) in common_nodes for node_id in node_universe], dtype=bool)


# Hybrid transmission model
def _simulate_single_hybrid_outbreak(
    pack: HybridPanelPack,
    *,
    run_seed: int,
    initial_seed_nodes: np.ndarray,
    config: HybridSimulationConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(run_seed))
    node_index = {int(node_id): idx for idx, node_id in enumerate(pack.node_universe)}
    initial_indices = np.asarray([node_index[int(node_id)] for node_id in initial_seed_nodes if int(node_id) in node_index], dtype=np.int64)
    if initial_indices.size == 0:
        raise ValueError("No initial seed nodes were found in the current panel.")

    N = len(pack.node_universe)
    is_farm = np.asarray(pack.is_farm, dtype=bool)
    is_region = np.asarray(pack.is_region, dtype=bool)
    region_positions = np.flatnonzero(is_region)
    region_count = int(len(region_positions))
    region_index_by_node = np.full(N, -1, dtype=np.int64)
    if region_count:
        region_index_by_node[region_positions] = np.arange(region_count, dtype=np.int64)

    # Region nodes carry reservoir pressure only.
    state = np.zeros(N, dtype=np.int8)  # S=0, E=1, I=2, R=3
    state[initial_indices] = 2
    ever_infected_farm = np.zeros(N, dtype=bool)
    ever_infected_farm[initial_indices] = True
    farm_first_infection_day = np.full(N, np.nan, dtype=float)

    reservoir_pressure = np.zeros(N, dtype=float)
    beta_ff = float(config.beta_ff)
    beta_fr = float(config.beta_fr)
    beta_rf = float(config.beta_rf)
    beta_rr = float(config.beta_rr)
    sigma = float(config.sigma)
    gamma = float(config.gamma)
    farm_susc = float(config.farm_susceptibility)
    farm_inf = float(config.farm_infectiousness)
    reservoir_decay = float(config.reservoir_decay)
    reservoir_background = float(config.reservoir_background)
    reservoir_clip = max(1e-9, float(config.reservoir_clip))
    farm_import_hazard = -math.log(max(1e-12, 1.0 - float(config.farm_daily_import_prob))) if config.farm_daily_import_prob > 0 else 0.0

    T = len(pack.ts_values)
    max_days = T + int(config.tail_days)

    farm_incidence = np.zeros(T, dtype=float)
    farm_prevalence = np.zeros(T, dtype=float)
    farm_cumulative_incidence = np.zeros(T, dtype=float)
    reservoir_total = np.zeros(T, dtype=float)
    reservoir_max = np.zeros(T, dtype=float)
    reservoir_positive_regions = np.zeros(T, dtype=float)
    farm_hazard_ff = np.zeros(T, dtype=float)
    farm_hazard_rf = np.zeros(T, dtype=float)
    region_pressure_fr = np.zeros(T, dtype=float)
    region_pressure_rr = np.zeros(T, dtype=float)
    region_reservoir = np.zeros((region_count, T), dtype=float)
    region_import_pressure = np.zeros((region_count, T), dtype=float)
    region_export_pressure = np.zeros((region_count, T), dtype=float)

    first_new_farm_infection_day = None
    extinction_day = None

    for day_index in range(max_days):
        if day_index < T:
            src = np.asarray(pack.src[day_index], dtype=np.int64)
            dst = np.asarray(pack.dst[day_index], dtype=np.int64)
            weight = np.asarray(pack.weight[day_index], dtype=float)
        else:
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)
            weight = np.array([], dtype=float)

        infectious_farms = (state == 2) & is_farm
        susceptible_farms = (state == 0) & is_farm

        ff_mask = is_farm[src] & is_farm[dst] if len(src) else np.array([], dtype=bool)
        fr_mask = is_farm[src] & is_region[dst] if len(src) else np.array([], dtype=bool)
        rf_mask = is_region[src] & is_farm[dst] if len(src) else np.array([], dtype=bool)
        rr_mask = is_region[src] & is_region[dst] if len(src) else np.array([], dtype=bool)

        farm_hazards = np.zeros(N, dtype=float)
        import_pressure_day = np.zeros(region_count, dtype=float)
        export_pressure_day = np.zeros(region_count, dtype=float)
        ff_hazard_total = 0.0
        rf_hazard_total = 0.0
        fr_pressure_total = 0.0
        rr_pressure_total = 0.0

        if len(src):
            ff_active = ff_mask & infectious_farms[src] & susceptible_farms[dst]
            if ff_active.any():
                contrib = beta_ff * weight[ff_active] * farm_inf * farm_susc
                farm_hazards += np.bincount(dst[ff_active], weights=contrib, minlength=N).astype(float)
                ff_hazard_total = float(np.sum(contrib))

            if rf_mask.any():
                effective_pressure = np.clip(reservoir_pressure[src[rf_mask]], a_min=0.0, a_max=reservoir_clip)
                contrib = beta_rf * weight[rf_mask] * np.log1p(effective_pressure) * farm_susc
                susceptible_edges = susceptible_farms[dst[rf_mask]]
                if susceptible_edges.any():
                    active_dst = dst[rf_mask][susceptible_edges]
                    active_src = src[rf_mask][susceptible_edges]
                    active_contrib = contrib[susceptible_edges]
                    farm_hazards += np.bincount(active_dst, weights=active_contrib, minlength=N).astype(float)
                    rf_hazard_total = float(np.sum(active_contrib))
                    if region_count:
                        region_ids = region_index_by_node[active_src]
                        valid = region_ids >= 0
                        if valid.any():
                            import_pressure_day += np.bincount(region_ids[valid], weights=active_contrib[valid], minlength=region_count).astype(float)

        if farm_import_hazard > 0:
            farm_hazards[susceptible_farms] += farm_import_hazard

        newly_exposed = np.zeros(N, dtype=bool)
        susceptible_idx = np.flatnonzero(susceptible_farms)
        if len(susceptible_idx):
            p_infect = 1.0 - np.exp(-np.clip(farm_hazards[susceptible_idx], a_min=0.0, a_max=20.0))
            newly_exposed[susceptible_idx] = rng.random(len(susceptible_idx)) < p_infect

        next_reservoir = reservoir_decay * reservoir_pressure
        if reservoir_background > 0:
            next_reservoir[is_region] += reservoir_background

        if len(src):
            fr_active = fr_mask & infectious_farms[src]
            if fr_active.any():
                contrib = beta_fr * weight[fr_active] * farm_inf
                active_dst = dst[fr_active]
                next_reservoir += np.bincount(active_dst, weights=contrib, minlength=N).astype(float)
                fr_pressure_total = float(np.sum(contrib))
                if region_count:
                    region_ids = region_index_by_node[active_dst]
                    valid = region_ids >= 0
                    if valid.any():
                        export_pressure_day += np.bincount(region_ids[valid], weights=contrib[valid], minlength=region_count).astype(float)

            if rr_mask.any():
                source_pressure = np.clip(reservoir_pressure[src[rr_mask]], a_min=0.0, a_max=reservoir_clip)
                contrib = beta_rr * weight[rr_mask] * np.log1p(source_pressure)
                next_reservoir += np.bincount(dst[rr_mask], weights=contrib, minlength=N).astype(float)
                rr_pressure_total = float(np.sum(contrib))

        next_reservoir[~is_region] = 0.0
        reservoir_pressure = np.clip(next_reservoir, a_min=0.0, a_max=reservoir_clip)

        next_state = state.copy()
        if config.model == "SEIR":
            exposed = (state == 1) & is_farm
            infected = infectious_farms
            if exposed.any():
                next_state[exposed & (rng.random(N) < sigma)] = 2
            if infected.any():
                next_state[infected & (rng.random(N) < gamma)] = 3
            next_state[newly_exposed] = 1
        elif config.model == "SIR":
            infected = infectious_farms
            if infected.any():
                next_state[infected & (rng.random(N) < gamma)] = 3
            next_state[newly_exposed] = 2
        elif config.model == "SIS":
            infected = infectious_farms
            if infected.any():
                next_state[infected & (rng.random(N) < gamma)] = 0
            next_state[newly_exposed] = 2
        else:
            raise ValueError(f"Unsupported model: {config.model}")

        state = next_state
        new_farm_infections = newly_exposed & is_farm
        ever_infected_farm |= new_farm_infections
        first_time_farms = new_farm_infections & ~np.isfinite(farm_first_infection_day)
        farm_first_infection_day[first_time_farms] = float(day_index)

        if first_new_farm_infection_day is None and bool(new_farm_infections.any()):
            first_new_farm_infection_day = int(day_index)

        active_farm_disease = bool(np.any(((state == 1) | (state == 2)) & is_farm))
        if extinction_day is None and not active_farm_disease and day_index >= 0:
            extinction_day = int(day_index)
            if day_index >= T:
                break

        if day_index < T:
            farm_incidence[day_index] = float(np.sum(new_farm_infections))
            farm_prevalence[day_index] = float(np.sum((state == 2) & is_farm))
            farm_cumulative_incidence[day_index] = (
                float(farm_cumulative_incidence[day_index - 1]) + farm_incidence[day_index]
                if day_index > 0 else farm_incidence[day_index]
            )
            farm_hazard_ff[day_index] = float(ff_hazard_total)
            farm_hazard_rf[day_index] = float(rf_hazard_total)
            region_pressure_fr[day_index] = float(fr_pressure_total)
            region_pressure_rr[day_index] = float(rr_pressure_total)
            region_pressures = reservoir_pressure[is_region]
            reservoir_total[day_index] = float(np.sum(region_pressures))
            reservoir_max[day_index] = float(np.max(region_pressures)) if len(region_pressures) else 0.0
            reservoir_positive_regions[day_index] = float(np.sum(region_pressures > 0))
            if region_count:
                region_reservoir[:, day_index] = region_pressures
                region_import_pressure[:, day_index] = import_pressure_day
                region_export_pressure[:, day_index] = export_pressure_day

    if extinction_day is None:
        extinction_day = int(max_days - 1)

    peak_idx = int(np.argmax(farm_prevalence)) if len(farm_prevalence) else 0
    farm_count = max(int(np.sum(is_farm)), 1)
    farm_hazard_total_auc = float(np.sum(farm_hazard_ff) + np.sum(farm_hazard_rf))
    region_pressure_total_auc = float(np.sum(region_pressure_fr) + np.sum(region_pressure_rr))
    scalar = {
        "seed_count_farm": float(len(initial_indices)),
        "farm_attack_count": float(np.sum(ever_infected_farm & is_farm)),
        "farm_attack_rate": float(np.sum(ever_infected_farm & is_farm) / farm_count),
        "farm_cumulative_incidence": float(np.sum(farm_incidence)),
        "farm_peak_prevalence": float(np.max(farm_prevalence)) if len(farm_prevalence) else 0.0,
        "farm_peak_prevalence_fraction": float(np.max(farm_prevalence) / farm_count) if len(farm_prevalence) else 0.0,
        "farm_peak_day_index": float(peak_idx),
        "farm_peak_ts": float(pack.ts_values[peak_idx]) if len(pack.ts_values) else np.nan,
        "farm_duration_days": float(extinction_day + 1),
        "farm_first_infection_day_index": float(first_new_farm_infection_day) if first_new_farm_infection_day is not None else np.nan,
        "farm_prevalence_auc": float(np.sum(farm_prevalence)),
        "reservoir_total_auc": float(np.sum(reservoir_total)),
        "reservoir_max_peak": float(np.max(reservoir_max)) if len(reservoir_max) else 0.0,
        "farm_hazard_ff_auc": float(np.sum(farm_hazard_ff)),
        "farm_hazard_rf_auc": float(np.sum(farm_hazard_rf)),
        "farm_hazard_total_auc": farm_hazard_total_auc,
        "farm_hazard_rf_share": _safe_divide(float(np.sum(farm_hazard_rf)), farm_hazard_total_auc),
        "region_pressure_fr_auc": float(np.sum(region_pressure_fr)),
        "region_pressure_rr_auc": float(np.sum(region_pressure_rr)),
        "region_pressure_total_auc": region_pressure_total_auc,
        "region_pressure_rr_share": _safe_divide(float(np.sum(region_pressure_rr)), region_pressure_total_auc),
    }
    daily = {
        "farm_incidence": farm_incidence,
        "farm_prevalence": farm_prevalence,
        "farm_cumulative_incidence": farm_cumulative_incidence,
        "reservoir_total": reservoir_total,
        "reservoir_max": reservoir_max,
        "reservoir_positive_regions": reservoir_positive_regions,
        "farm_hazard_ff": farm_hazard_ff,
        "farm_hazard_rf": farm_hazard_rf,
        "region_pressure_fr": region_pressure_fr,
        "region_pressure_rr": region_pressure_rr,
    }
    region_daily = {
        "reservoir_pressure": region_reservoir,
        "import_pressure": region_import_pressure,
        "export_pressure": region_export_pressure,
    }
    farm_daily = {
        "farm_ever_infected": ever_infected_farm.astype(float),
        "farm_first_infection_day": farm_first_infection_day.astype(float),
    }
    return scalar, daily, region_daily, farm_daily



def _summarise_daily_arrays(
    *,
    ts_values: tuple[int, ...],
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    incidence_matrix = arrays.get("farm_incidence")
    for day_index, ts_value in enumerate(ts_values):
        row: dict[str, float] = {"day_index": int(day_index), "ts": int(ts_value)}
        for metric_name, matrix in arrays.items():
            values = np.asarray(matrix[:, day_index], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                row[metric_name] = float(np.median(finite))
                row[f"{metric_name}_q05"] = float(np.quantile(finite, 0.05))
                row[f"{metric_name}_q95"] = float(np.quantile(finite, 0.95))
                row[f"{metric_name}_mean"] = float(np.mean(finite))
                row[f"{metric_name}_std"] = float(np.std(finite, ddof=0))
            else:
                row[metric_name] = np.nan
                row[f"{metric_name}_q05"] = np.nan
                row[f"{metric_name}_q95"] = np.nan
                row[f"{metric_name}_mean"] = np.nan
                row[f"{metric_name}_std"] = np.nan
        if incidence_matrix is not None:
            incidence_values = np.asarray(incidence_matrix[:, day_index], dtype=float)
            finite_incidence = incidence_values[np.isfinite(incidence_values)]
            if len(finite_incidence):
                event_values = (finite_incidence > 0.0).astype(float)
                row["farm_infection_event_probability"] = float(np.mean(event_values))
                row["farm_infection_event_probability_std"] = float(np.std(event_values, ddof=0))
            else:
                row["farm_infection_event_probability"] = np.nan
                row["farm_infection_event_probability_std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_region_daily_arrays(
    *,
    ts_values: tuple[int, ...],
    region_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    columns = ["day_index", "ts", "region_order", "region_node_id", "corop", "x", "y", "display_label"]
    if region_frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for region_index, region_row in region_frame.reset_index(drop=True).iterrows():
        for day_index, ts_value in enumerate(ts_values):
            row: dict[str, object] = {
                "day_index": int(day_index),
                "ts": int(ts_value),
                "region_order": int(region_row.get("region_order", region_index)),
                "region_node_id": int(region_row.get("region_node_id", -1)),
                "corop": str(region_row.get("corop", "")),
                "x": pd.to_numeric(pd.Series([region_row.get("x")]), errors="coerce").iloc[0],
                "y": pd.to_numeric(pd.Series([region_row.get("y")]), errors="coerce").iloc[0],
                "display_label": str(region_row.get("display_label", region_row.get("corop", ""))),
            }
            for metric_name, matrix in arrays.items():
                values = np.asarray(matrix[:, region_index, day_index], dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite):
                    row[metric_name] = float(np.median(finite))
                    row[f"{metric_name}_q05"] = float(np.quantile(finite, 0.05))
                    row[f"{metric_name}_q95"] = float(np.quantile(finite, 0.95))
                    row[f"{metric_name}_mean"] = float(np.mean(finite))
                    row[f"{metric_name}_std"] = float(np.std(finite, ddof=0))
                else:
                    row[metric_name] = np.nan
                    row[f"{metric_name}_q05"] = np.nan
                    row[f"{metric_name}_q95"] = np.nan
                    row[f"{metric_name}_mean"] = np.nan
                    row[f"{metric_name}_std"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _summarise_farm_node_arrays(
    *,
    pack: HybridPanelPack,
    node_frame: Optional[pd.DataFrame],
    ever_infected_matrix: np.ndarray,
    first_infection_day_matrix: np.ndarray,
    seed_matrix: np.ndarray,
) -> pd.DataFrame:
    farm_positions = np.flatnonzero(np.asarray(pack.is_farm, dtype=bool))
    if len(farm_positions) == 0:
        return pd.DataFrame(columns=[
            "node_id", "ubn", "corop", "x", "y", "display_label",
            "seed_probability", "ever_infected_probability", "network_generated_attack_probability",
            "first_infection_day_mean", "first_infection_day_median",
            "first_infection_day_q05", "first_infection_day_q95",
        ])

    ever = np.asarray(ever_infected_matrix, dtype=float)
    first = np.asarray(first_infection_day_matrix, dtype=float)
    seeds = np.asarray(seed_matrix, dtype=bool)
    node_ids = np.asarray(pack.node_universe, dtype=np.int64)[farm_positions]

    if node_frame is not None and not node_frame.empty and "node_id" in node_frame.columns:
        lookup = node_frame.drop_duplicates(subset=["node_id"]).set_index("node_id", drop=False)
    else:
        lookup = pd.DataFrame(columns=["node_id", "ubn", "corop", "x", "y"])

    rows: list[dict[str, object]] = []
    for farm_index, node_id in enumerate(node_ids):
        ever_values = np.asarray(ever[:, farm_index], dtype=bool)
        first_values = np.asarray(first[:, farm_index], dtype=float)
        seed_values = np.asarray(seeds[:, farm_index], dtype=bool)
        nonseed_values = ~seed_values
        network_infected = ever_values & nonseed_values
        finite_first = first_values[np.isfinite(first_values) & network_infected]
        if int(node_id) in getattr(lookup, 'index', []):
            source = lookup.loc[int(node_id)]
            ubn = source.get("ubn") if "ubn" in lookup.columns else np.nan
            corop = _safe_corop_label(source.get("corop"), f"farm_{int(node_id)}") if "corop" in lookup.columns else f"farm_{int(node_id)}"
            x_value = pd.to_numeric(pd.Series([source.get("x")]), errors="coerce").iloc[0] if "x" in lookup.columns else np.nan
            y_value = pd.to_numeric(pd.Series([source.get("y")]), errors="coerce").iloc[0] if "y" in lookup.columns else np.nan
        else:
            ubn = np.nan
            corop = f"farm_{int(node_id)}"
            x_value = np.nan
            y_value = np.nan
        display_label = str(ubn) if not pd.isna(ubn) and str(ubn).strip() else f"farm_{int(node_id)}"
        row = {
            "node_id": int(node_id),
            "ubn": ubn,
            "corop": str(corop),
            "x": np.nan if pd.isna(x_value) else float(x_value),
            "y": np.nan if pd.isna(y_value) else float(y_value),
            "display_label": display_label,
            "seed_probability": float(np.mean(seed_values)) if len(seed_values) else np.nan,
            "ever_infected_probability": float(np.mean(ever_values)) if len(ever_values) else np.nan,
            "network_generated_attack_probability": float(np.mean(network_infected) / np.mean(nonseed_values)) if np.any(nonseed_values) else np.nan,
        }
        if len(finite_first):
            row["first_infection_day_mean"] = float(np.mean(finite_first))
            row["first_infection_day_median"] = float(np.median(finite_first))
            row["first_infection_day_q05"] = float(np.quantile(finite_first, 0.05))
            row["first_infection_day_q95"] = float(np.quantile(finite_first, 0.95))
        else:
            row["first_infection_day_mean"] = np.nan
            row["first_infection_day_median"] = np.nan
            row["first_infection_day_q05"] = np.nan
            row["first_infection_day_q95"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["corop", "node_id"]).reset_index(drop=True)


def _merge_farm_node_summaries(
    observed_farm_summary: pd.DataFrame,
    synthetic_farm_summary: pd.DataFrame,
) -> pd.DataFrame:
    if observed_farm_summary.empty and synthetic_farm_summary.empty:
        return pd.DataFrame(columns=["node_id", "ubn", "corop", "x", "y", "display_label"])
    observed_prefixed = observed_farm_summary.rename(
        columns={column: f"original_{column}" for column in observed_farm_summary.columns if column != "node_id"}
    )
    synthetic_prefixed = synthetic_farm_summary.rename(
        columns={column: f"synthetic_{column}" for column in synthetic_farm_summary.columns if column != "node_id"}
    )
    merged = observed_prefixed.merge(synthetic_prefixed, on=["node_id"], how="outer").sort_values("node_id").reset_index(drop=True)
    for meta in ["ubn", "corop", "x", "y", "display_label"]:
        original_col = f"original_{meta}"
        synthetic_col = f"synthetic_{meta}"
        if original_col in merged.columns or synthetic_col in merged.columns:
            original_series = merged[original_col] if original_col in merged.columns else pd.Series([np.nan] * len(merged), index=merged.index)
            synthetic_series = merged[synthetic_col] if synthetic_col in merged.columns else pd.Series([np.nan] * len(merged), index=merged.index)
            merged[meta] = original_series.where(original_series.notna(), synthetic_series)
    for metric_name in ["network_generated_attack_probability", "ever_infected_probability", "first_infection_day_median", "first_infection_day_mean"]:
        original_col = f"original_{metric_name}"
        synthetic_col = f"synthetic_{metric_name}"
        if original_col in merged.columns and synthetic_col in merged.columns:
            merged[f"{metric_name}_delta"] = pd.to_numeric(merged[synthetic_col], errors="coerce") - pd.to_numeric(merged[original_col], errors="coerce")
    return merged


def _summarise_farm_corop_fit(farm_merged: pd.DataFrame) -> pd.DataFrame:
    if farm_merged.empty or "corop" not in farm_merged.columns:
        return pd.DataFrame(columns=[
            "corop",
            "original_network_generated_attack_probability",
            "synthetic_network_generated_attack_probability",
            "network_generated_attack_probability_delta",
            "original_first_infection_day_median",
            "synthetic_first_infection_day_median",
            "first_infection_day_median_delta",
            "farm_count",
        ])
    rows: list[dict[str, object]] = []
    for corop, group in farm_merged.groupby("corop", dropna=False, sort=True):
        original_attack = pd.to_numeric(group.get("original_network_generated_attack_probability", np.nan), errors="coerce")
        synthetic_attack = pd.to_numeric(group.get("synthetic_network_generated_attack_probability", np.nan), errors="coerce")
        original_first = pd.to_numeric(group.get("original_first_infection_day_median", np.nan), errors="coerce")
        synthetic_first = pd.to_numeric(group.get("synthetic_first_infection_day_median", np.nan), errors="coerce")
        rows.append({
            "corop": str(corop),
            "original_network_generated_attack_probability": float(original_attack.mean()) if original_attack.notna().any() else np.nan,
            "synthetic_network_generated_attack_probability": float(synthetic_attack.mean()) if synthetic_attack.notna().any() else np.nan,
            "network_generated_attack_probability_delta": float((synthetic_attack - original_attack).mean()) if original_attack.notna().any() and synthetic_attack.notna().any() else np.nan,
            "original_first_infection_day_median": float(original_first.mean()) if original_first.notna().any() else np.nan,
            "synthetic_first_infection_day_median": float(synthetic_first.mean()) if synthetic_first.notna().any() else np.nan,
            "first_infection_day_median_delta": float((synthetic_first - original_first).mean()) if original_first.notna().any() and synthetic_first.notna().any() else np.nan,
            "farm_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def _summarise_trajectory_distribution_distances(
    observed_daily_arrays: dict[str, np.ndarray],
    synthetic_daily_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_name in TRAJECTORY_DISTANCE_METRICS:
        observed = observed_daily_arrays.get(metric_name)
        synthetic = synthetic_daily_arrays.get(metric_name)
        if observed is None or synthetic is None:
            continue
        rows.append({
            "metric": metric_name,
            "trajectory_energy_distance": _ensemble_energy_distance(observed, synthetic),
            "trajectory_variogram_distance": _ensemble_variogram_distance(observed, synthetic),
            "time_steps": int(np.asarray(observed, dtype=float).shape[1]) if np.asarray(observed, dtype=float).ndim >= 2 else 1,
        })
    return pd.DataFrame(rows)


def _summarise_region_field_distances(
    *,
    ts_values: tuple[int, ...],
    observed_region_arrays: dict[str, np.ndarray],
    synthetic_region_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    if not observed_region_arrays or not synthetic_region_arrays:
        return pd.DataFrame(columns=["day_index", "ts"])
    rows: list[dict[str, object]] = []
    for day_index, ts_value in enumerate(ts_values):
        row: dict[str, object] = {"day_index": int(day_index), "ts": int(ts_value)}
        for metric_name in REGION_DAILY_METRICS:
            observed = observed_region_arrays.get(metric_name)
            synthetic = synthetic_region_arrays.get(metric_name)
            if observed is None or synthetic is None:
                row[f"{metric_name}_energy_distance"] = np.nan
                row[f"{metric_name}_variogram_distance"] = np.nan
                continue
            row[f"{metric_name}_energy_distance"] = _ensemble_energy_distance(observed[:, :, day_index], synthetic[:, :, day_index])
            row[f"{metric_name}_variogram_distance"] = _ensemble_variogram_distance(observed[:, :, day_index], synthetic[:, :, day_index])
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_panel(
    pack: HybridPanelPack,
    *,
    run_seeds: np.ndarray,
    initial_seed_sets: list[np.ndarray],
    config: HybridSimulationConfig,
    region_frame: Optional[pd.DataFrame] = None,
    node_frame: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    LOGGER.debug(
        "Simulating panel | label=%s | model=%s | replicates=%s | snapshots=%s",
        pack.label,
        config.model,
        len(run_seeds),
        len(pack.ts_values),
    )
    scalar_rows = []
    daily_store: dict[str, list[np.ndarray]] = defaultdict(list)
    region_store: dict[str, list[np.ndarray]] = defaultdict(list)
    farm_store: dict[str, list[np.ndarray]] = defaultdict(list)
    farm_positions = np.flatnonzero(np.asarray(pack.is_farm, dtype=bool))
    node_index = {int(node_id): idx for idx, node_id in enumerate(pack.node_universe)}
    farm_index = {int(pack.node_universe[position]): idx for idx, position in enumerate(farm_positions)}
    seed_store: list[np.ndarray] = []
    progress_stride = max(1, len(run_seeds) // 4)
    for replicate_index, (run_seed, seed_nodes) in enumerate(zip(run_seeds, initial_seed_sets)):
        scalar, daily, region_daily, farm_daily = _simulate_single_hybrid_outbreak(
            pack,
            run_seed=int(run_seed),
            initial_seed_nodes=np.asarray(seed_nodes, dtype=np.int64),
            config=config,
        )
        scalar["replicate_index"] = float(replicate_index)
        scalar_rows.append(scalar)
        for metric_name, values in daily.items():
            daily_store[metric_name].append(np.asarray(values, dtype=float))
        for metric_name, values in region_daily.items():
            region_store[metric_name].append(np.asarray(values, dtype=float))
        for metric_name, values in farm_daily.items():
            farm_store[metric_name].append(np.asarray(values, dtype=float)[farm_positions])
        seed_indicator = np.zeros(len(farm_positions), dtype=bool)
        for node_id in np.asarray(seed_nodes, dtype=np.int64):
            position = farm_index.get(int(node_id))
            if position is not None:
                seed_indicator[position] = True
        seed_store.append(seed_indicator)
        if LOGGER.isEnabledFor(logging.DEBUG) and (
            replicate_index == 0
            or (replicate_index + 1) % progress_stride == 0
            or replicate_index + 1 == len(run_seeds)
        ):
            LOGGER.debug(
                "Simulation progress | label=%s | completed=%s/%s",
                pack.label,
                replicate_index + 1,
                len(run_seeds),
            )
    scalar_frame = pd.DataFrame(scalar_rows)
    daily_arrays = {metric_name: np.vstack(value_list) for metric_name, value_list in daily_store.items()}
    daily_summary = _summarise_daily_arrays(ts_values=pack.ts_values, arrays=daily_arrays)
    if region_frame is not None and not region_frame.empty and region_store:
        region_arrays = {metric_name: np.stack(value_list, axis=0) for metric_name, value_list in region_store.items()}
        region_daily_summary = _summarise_region_daily_arrays(ts_values=pack.ts_values, region_frame=region_frame, arrays=region_arrays)
    else:
        region_arrays = {}
        region_daily_summary = pd.DataFrame(columns=["day_index", "ts", "region_order", "region_node_id", "corop", "x", "y", "display_label"])
    if farm_store:
        farm_ever = np.vstack(farm_store.get("farm_ever_infected", [])) if farm_store.get("farm_ever_infected") else np.empty((0, len(farm_positions)), dtype=float)
        farm_first = np.vstack(farm_store.get("farm_first_infection_day", [])) if farm_store.get("farm_first_infection_day") else np.empty((0, len(farm_positions)), dtype=float)
        seed_matrix = np.vstack(seed_store) if seed_store else np.empty((0, len(farm_positions)), dtype=bool)
        farm_summary = _summarise_farm_node_arrays(
            pack=pack,
            node_frame=node_frame,
            ever_infected_matrix=farm_ever,
            first_infection_day_matrix=farm_first,
            seed_matrix=seed_matrix,
        )
    else:
        farm_summary = pd.DataFrame(columns=["node_id", "ubn", "corop", "x", "y", "display_label"])
    diagnostics = {
        "daily_arrays": daily_arrays,
        "region_arrays": region_arrays,
        "farm_summary": farm_summary,
    }
    if LOGGER.isEnabledFor(logging.DEBUG) and not scalar_frame.empty:
        LOGGER.debug(
            "Simulation panel summary | label=%s | attack_rate_mean=%s | peak_prev_mean=%s | duration_mean=%s",
            pack.label,
            _metric_text(scalar_frame.get("farm_attack_rate", pd.Series(dtype=float)).mean()),
            _metric_text(scalar_frame.get("farm_peak_prevalence", pd.Series(dtype=float)).mean()),
            _metric_text(scalar_frame.get("farm_duration_days", pd.Series(dtype=float)).mean()),
        )
    return scalar_frame, daily_summary, region_daily_summary, diagnostics


# Comparison and summary helpers
# Comparison and summary helpers
def _merge_daily_summaries(observed_daily: pd.DataFrame, synthetic_daily: pd.DataFrame) -> pd.DataFrame:
    merged = observed_daily.rename(
        columns={column: f"original_{column}" for column in observed_daily.columns if column not in {"day_index", "ts"}}
    ).merge(
        synthetic_daily.rename(
            columns={column: f"synthetic_{column}" for column in synthetic_daily.columns if column not in {"day_index", "ts"}}
        ),
        on=["day_index", "ts"],
        how="outer",
    ).sort_values(["day_index", "ts"]).reset_index(drop=True)

    metric_names = sorted(
        {
            column.removeprefix("original_")
            for column in merged.columns
            if column.startswith("original_") and not column.endswith(("_q05", "_q95", "_mean", "_std"))
        }
    )
    for metric_name in metric_names:
        original_col = f"original_{metric_name}"
        synthetic_col = f"synthetic_{metric_name}"
        if original_col in merged.columns and synthetic_col in merged.columns:
            merged[f"{metric_name}_delta"] = (
                pd.to_numeric(merged[synthetic_col], errors="coerce") -
                pd.to_numeric(merged[original_col], errors="coerce")
            )
    return merged


def _summarise_outcome_distribution_comparison(
    observed_scalar: pd.DataFrame,
    synthetic_scalar: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    rows = []
    for metric_name in metric_names:
        observed = pd.to_numeric(observed_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        synthetic = pd.to_numeric(synthetic_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        if len(observed) == 0 and len(synthetic) == 0:
            continue
        rows.append(
            {
                "metric": metric_name,
                "original_mean": float(np.mean(observed)) if len(observed) else np.nan,
                "synthetic_mean": float(np.mean(synthetic)) if len(synthetic) else np.nan,
                "original_median": float(np.median(observed)) if len(observed) else np.nan,
                "synthetic_median": float(np.median(synthetic)) if len(synthetic) else np.nan,
                "mean_delta": float(np.mean(synthetic) - np.mean(observed)) if len(observed) and len(synthetic) else np.nan,
                "median_delta": float(np.median(synthetic) - np.median(observed)) if len(observed) and len(synthetic) else np.nan,
                "wasserstein_distance": _wasserstein_distance_1d(observed, synthetic),
                "correlation": _safe_correlation(observed, synthetic) if len(observed) == len(synthetic) else np.nan,
                "replicate_count_original": int(len(observed)),
                "replicate_count_synthetic": int(len(synthetic)),
            }
        )
    return pd.DataFrame(rows)


def _metric_lookup(summary: pd.DataFrame, metric_name: str, field_name: str) -> Optional[float]:
    if summary.empty or "metric" not in summary.columns:
        return None
    subset = summary.loc[summary["metric"].astype(str) == str(metric_name), field_name]
    if subset.empty:
        return None
    value = pd.to_numeric(subset, errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _calibration_lookup(summary: pd.DataFrame, metric_name: str, field_name: str) -> Optional[float]:
    if summary.empty or "metric" not in summary.columns or field_name not in summary.columns:
        return None
    subset = summary.loc[summary["metric"].astype(str) == str(metric_name), field_name]
    if subset.empty:
        return None
    value = pd.to_numeric(subset, errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _summarise_daily_interval_calibration(
    per_snapshot: pd.DataFrame,
    metric_name: str,
) -> dict[str, object]:
    observed_col = f"original_{metric_name}"
    synthetic_col = f"synthetic_{metric_name}"
    lower_col = f"synthetic_{metric_name}_q05"
    upper_col = f"synthetic_{metric_name}_q95"

    if any(column not in per_snapshot.columns for column in (observed_col, synthetic_col, lower_col, upper_col)):
        return {
            "metric": metric_name,
            "metric_type": "daily",
            "valid_count": 0,
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
            "median_interval_width": np.nan,
            "mean_abs_exceedance": np.nan,
            "max_abs_exceedance": np.nan,
            "mean_abs_delta_to_synthetic_center": np.nan,
        }

    observed = pd.to_numeric(per_snapshot[observed_col], errors="coerce")
    synthetic = pd.to_numeric(per_snapshot[synthetic_col], errors="coerce")
    lower = pd.to_numeric(per_snapshot[lower_col], errors="coerce")
    upper = pd.to_numeric(per_snapshot[upper_col], errors="coerce")

    valid = observed.notna() & synthetic.notna() & lower.notna() & upper.notna()
    if not bool(valid.any()):
        return {
            "metric": metric_name,
            "metric_type": "daily",
            "valid_count": 0,
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
            "median_interval_width": np.nan,
            "mean_abs_exceedance": np.nan,
            "max_abs_exceedance": np.nan,
            "mean_abs_delta_to_synthetic_center": np.nan,
        }

    obs = observed.loc[valid].to_numpy(dtype=float)
    syn = synthetic.loc[valid].to_numpy(dtype=float)
    lo = lower.loc[valid].to_numpy(dtype=float)
    hi = upper.loc[valid].to_numpy(dtype=float)

    inside = (obs >= lo) & (obs <= hi)
    exceedance = np.where(obs < lo, lo - obs, np.where(obs > hi, obs - hi, 0.0))
    width = hi - lo

    return {
        "metric": metric_name,
        "metric_type": "daily",
        "valid_count": int(len(obs)),
        "interval_coverage": float(np.mean(inside)),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "mean_abs_exceedance": float(np.mean(exceedance)),
        "max_abs_exceedance": float(np.max(exceedance)) if len(exceedance) else np.nan,
        "mean_abs_delta_to_synthetic_center": float(np.mean(np.abs(obs - syn))),
    }


def _summarise_daily_mean_comparison(
    per_snapshot: pd.DataFrame,
    metric_name: str,
) -> dict[str, object]:
    observed_col = f"original_{metric_name}_mean"
    synthetic_col = f"synthetic_{metric_name}_mean"

    if observed_col not in per_snapshot.columns or synthetic_col not in per_snapshot.columns:
        return {
            "metric": metric_name,
            "metric_type": "daily_mean",
            "valid_count": 0,
            "observed_curve_mean": np.nan,
            "synthetic_curve_mean": np.nan,
            "curve_correlation": np.nan,
            "mean_delta": np.nan,
            "mean_abs_delta": np.nan,
            "rmse": np.nan,
            "max_abs_delta": np.nan,
        }

    observed = pd.to_numeric(per_snapshot[observed_col], errors="coerce")
    synthetic = pd.to_numeric(per_snapshot[synthetic_col], errors="coerce")
    valid = observed.notna() & synthetic.notna()
    if not bool(valid.any()):
        return {
            "metric": metric_name,
            "metric_type": "daily_mean",
            "valid_count": 0,
            "observed_curve_mean": np.nan,
            "synthetic_curve_mean": np.nan,
            "curve_correlation": np.nan,
            "mean_delta": np.nan,
            "mean_abs_delta": np.nan,
            "rmse": np.nan,
            "max_abs_delta": np.nan,
        }

    obs = observed.loc[valid].to_numpy(dtype=float)
    syn = synthetic.loc[valid].to_numpy(dtype=float)
    delta = syn - obs
    return {
        "metric": metric_name,
        "metric_type": "daily_mean",
        "valid_count": int(len(obs)),
        "observed_curve_mean": float(np.mean(obs)),
        "synthetic_curve_mean": float(np.mean(syn)),
        "curve_correlation": _safe_correlation(obs, syn),
        "mean_delta": float(np.mean(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else np.nan,
    }


def _summarise_scalar_calibration(
    observed_scalar: pd.DataFrame,
    synthetic_scalar: pd.DataFrame,
    metric_name: str,
) -> dict[str, object]:
    observed = pd.to_numeric(observed_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
    synthetic = pd.to_numeric(synthetic_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)

    if len(observed) == 0 or len(synthetic) == 0:
        return {
            "metric": metric_name,
            "metric_type": "scalar",
            "observed_mean": np.nan,
            "observed_median": np.nan,
            "synthetic_mean": np.nan,
            "synthetic_median": np.nan,
            "synthetic_q05": np.nan,
            "synthetic_q95": np.nan,
            "synthetic_interval_width": np.nan,
            "observed_median_in_synthetic_90pct": np.nan,
            "observed_mean_in_synthetic_90pct": np.nan,
            "observed_median_percentile": np.nan,
            "observed_median_tail_area": np.nan,
            "observed_mean_percentile": np.nan,
            "observed_mean_tail_area": np.nan,
            "abs_gap_to_interval_median": np.nan,
            "abs_gap_to_interval_mean": np.nan,
        }

    observed_mean = float(np.mean(observed))
    observed_median = float(np.median(observed))
    synthetic_mean = float(np.mean(synthetic))
    synthetic_median = float(np.median(synthetic))
    synthetic_q05 = float(np.quantile(synthetic, 0.05))
    synthetic_q95 = float(np.quantile(synthetic, 0.95))
    interval_width = float(synthetic_q95 - synthetic_q05)

    median_percentile = float(np.mean(synthetic <= observed_median))
    mean_percentile = float(np.mean(synthetic <= observed_mean))
    median_tail = float(min(1.0, 2.0 * min(median_percentile, 1.0 - median_percentile)))
    mean_tail = float(min(1.0, 2.0 * min(mean_percentile, 1.0 - mean_percentile)))

    median_gap = 0.0 if synthetic_q05 <= observed_median <= synthetic_q95 else (
        synthetic_q05 - observed_median if observed_median < synthetic_q05 else observed_median - synthetic_q95
    )
    mean_gap = 0.0 if synthetic_q05 <= observed_mean <= synthetic_q95 else (
        synthetic_q05 - observed_mean if observed_mean < synthetic_q05 else observed_mean - synthetic_q95
    )

    return {
        "metric": metric_name,
        "metric_type": "scalar",
        "observed_mean": observed_mean,
        "observed_median": observed_median,
        "synthetic_mean": synthetic_mean,
        "synthetic_median": synthetic_median,
        "synthetic_q05": synthetic_q05,
        "synthetic_q95": synthetic_q95,
        "synthetic_interval_width": interval_width,
        "observed_median_in_synthetic_90pct": float(synthetic_q05 <= observed_median <= synthetic_q95),
        "observed_mean_in_synthetic_90pct": float(synthetic_q05 <= observed_mean <= synthetic_q95),
        "observed_median_percentile": median_percentile,
        "observed_median_tail_area": median_tail,
        "observed_mean_percentile": mean_percentile,
        "observed_mean_tail_area": mean_tail,
        "abs_gap_to_interval_median": float(abs(median_gap)),
        "abs_gap_to_interval_mean": float(abs(mean_gap)),
    }




def _merge_region_daily_summaries(
    observed_region_daily: pd.DataFrame,
    synthetic_region_daily: pd.DataFrame,
) -> pd.DataFrame:
    keys = [column for column in ["day_index", "ts", "region_order", "region_node_id", "corop", "display_label"] if column in observed_region_daily.columns or column in synthetic_region_daily.columns]
    observed_prefixed = observed_region_daily.rename(
        columns={column: f"original_{column}" for column in observed_region_daily.columns if column not in keys}
    )
    synthetic_prefixed = synthetic_region_daily.rename(
        columns={column: f"synthetic_{column}" for column in synthetic_region_daily.columns if column not in keys}
    )
    merged = observed_prefixed.merge(synthetic_prefixed, on=keys, how="outer").sort_values(keys).reset_index(drop=True)
    metric_names = sorted(
        {
            column.removeprefix("original_")
            for column in merged.columns
            if column.startswith("original_") and not column.endswith(("_q05", "_q95", "_mean", "_std"))
        }
    )
    for metric_name in metric_names:
        original_col = f"original_{metric_name}"
        synthetic_col = f"synthetic_{metric_name}"
        if original_col in merged.columns and synthetic_col in merged.columns:
            merged[f"{metric_name}_delta"] = (
                pd.to_numeric(merged[synthetic_col], errors="coerce") -
                pd.to_numeric(merged[original_col], errors="coerce")
            )
    return merged


def _top_k_overlap(original_values: np.ndarray, synthetic_values: np.ndarray, k: int = 3) -> float:
    original = np.asarray(original_values, dtype=float)
    synthetic = np.asarray(synthetic_values, dtype=float)
    if original.size == 0 or synthetic.size == 0:
        return np.nan
    size = min(int(k), int(original.size), int(synthetic.size))
    if size <= 0:
        return np.nan
    original_rank = set(np.argsort(-original)[:size].tolist())
    synthetic_rank = set(np.argsort(-synthetic)[:size].tolist())
    return float(len(original_rank & synthetic_rank) / size)


def _normalised_share_mae(original_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    original = np.clip(np.asarray(original_values, dtype=float), a_min=0.0, a_max=None)
    synthetic = np.clip(np.asarray(synthetic_values, dtype=float), a_min=0.0, a_max=None)
    original_total = float(original.sum())
    synthetic_total = float(synthetic.sum())
    if original_total <= 0 and synthetic_total <= 0:
        return 0.0
    if original_total <= 0 or synthetic_total <= 0:
        return np.nan
    return float(np.mean(np.abs(original / original_total - synthetic / synthetic_total)))


def _summarise_region_spatial_fit(region_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial_columns = ["day_index", "ts"]
    temporal_columns = ["region_node_id", "corop", "display_label"]
    if region_merged.empty:
        return pd.DataFrame(columns=spatial_columns), pd.DataFrame(columns=temporal_columns)

    spatial_rows: list[dict[str, object]] = []
    for (day_index, ts_value), group in region_merged.groupby(["day_index", "ts"], sort=True):
        row: dict[str, object] = {"day_index": int(day_index), "ts": int(ts_value)}
        for metric_name in REGION_DAILY_METRICS:
            original = pd.to_numeric(group.get(f"original_{metric_name}", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            synthetic = pd.to_numeric(group.get(f"synthetic_{metric_name}", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            row[f"{metric_name}_spatial_correlation"] = _safe_correlation(original, synthetic) if len(original) else np.nan
            row[f"{metric_name}_mean_abs_delta"] = float(np.mean(np.abs(synthetic - original))) if len(original) else np.nan
            row[f"{metric_name}_share_mae"] = _normalised_share_mae(original, synthetic) if len(original) else np.nan
            row[f"{metric_name}_hotspot_overlap_top3"] = _top_k_overlap(original, synthetic, k=3) if len(original) else np.nan
        spatial_rows.append(row)

    temporal_rows: list[dict[str, object]] = []
    for key_values, group in region_merged.groupby(["region_node_id", "corop", "display_label"], sort=True):
        region_node_id, corop, display_label = key_values
        row = {
            "region_node_id": int(region_node_id),
            "corop": str(corop),
            "display_label": str(display_label),
        }
        for metric_name in REGION_DAILY_METRICS:
            original = pd.to_numeric(group.get(f"original_{metric_name}", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            synthetic = pd.to_numeric(group.get(f"synthetic_{metric_name}", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            row[f"{metric_name}_temporal_correlation"] = _safe_correlation(original, synthetic) if len(original) else np.nan
            row[f"original_total_{metric_name}"] = float(np.sum(original)) if len(original) else np.nan
            row[f"synthetic_total_{metric_name}"] = float(np.sum(synthetic)) if len(synthetic) else np.nan
            row[f"mean_abs_{metric_name}_delta"] = float(np.mean(np.abs(synthetic - original))) if len(original) else np.nan
            if len(original):
                original_peak_idx = int(np.argmax(original))
                synthetic_peak_idx = int(np.argmax(synthetic))
                try:
                    original_peak_ts = int(group.iloc[original_peak_idx]["ts"])
                    synthetic_peak_ts = int(group.iloc[synthetic_peak_idx]["ts"])
                    row[f"{metric_name}_peak_day_abs_delta"] = abs(synthetic_peak_ts - original_peak_ts)
                except Exception:
                    row[f"{metric_name}_peak_day_abs_delta"] = np.nan
            else:
                row[f"{metric_name}_peak_day_abs_delta"] = np.nan
        temporal_rows.append(row)

    return pd.DataFrame(spatial_rows), pd.DataFrame(temporal_rows)


def _prepare_region_geo_payload(
    *,
    manifest: dict,
    sample_label: str,
    observed_region_daily: pd.DataFrame,
    synthetic_region_daily: pd.DataFrame,
    corop_geojson_path: Optional[Path],
    focal_corop: str,
) -> Optional[dict[str, object]]:
    if corop_geojson_path is None or not corop_geojson_path.exists():
        return None
    if observed_region_daily.empty and synthetic_region_daily.empty:
        return None
    try:
        geojson_payload = json.loads(Path(corop_geojson_path).read_text())
    except Exception:
        return None

    payload = {
        "dataset": str(manifest.get("dataset") or "dataset"),
        "sample_label": str(sample_label),
        "focal_corop": str(focal_corop),
        "calendar": _calendar_records(observed_region_daily["ts"] if "ts" in observed_region_daily.columns else synthetic_region_daily.get("ts", [])),
        "metrics": list(REGION_DAILY_METRICS),
        "observed": _json_ready(observed_region_daily.to_dict(orient="records")),
        "synthetic": _json_ready(synthetic_region_daily.to_dict(orient="records")),
        "geojson": geojson_payload,
    }
    return payload


def _write_region_geo_html(payload: dict[str, object], output_path: Path) -> Optional[Path]:
    if not payload:
        return None
    output_path = Path(output_path)
    payload_js_path = output_path.with_name(f"{output_path.stem}_payload.js")
    payload_js_path.write_text("window.__TEMPORAL_SBM_REGION_PAYLOAD__ = " + json.dumps(_json_ready(payload), separators=(",", ":")) + ";")
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid Region Epidemic Compare</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src=\"""" + payload_js_path.name + """\"></script>
  <style>
    :root { --bg:#f5f9fc; --panel:#ffffff; --line:#d8e1ea; --text:#20303f; --muted:#607286; --shadow:rgba(32,48,63,0.08); }
    body { margin:0; background:var(--bg); color:var(--text); font-family:"Avenir Next","Segoe UI",sans-serif; }
    .app { padding:18px; }
    .banner { margin:0 0 14px; padding:12px 14px; border-radius:14px; border:1px solid #ead6aa; background:#fff7e6; color:#7a5b14; font-size:13px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:18px; }
    .toolbar label { font-size:13px; color:var(--muted); display:flex; gap:8px; align-items:center; }
    .toolbar select { padding:8px 10px; border:1px solid var(--line); border-radius:10px; background:#fbfdff; color:var(--text); }
    .toolbar input[type="range"] { width:320px; }
    .note { margin:0 0 18px; padding:12px 14px; border-radius:14px; border:1px solid var(--line); background:#fbfdff; color:var(--muted); font-size:13px; line-height:1.45; }
    .metric-stack { display:grid; gap:18px; }
    .metric-section { display:grid; gap:12px; }
    .section-head { display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:10px; }
    .section-head h2 { margin:0; font-size:17px; }
    .section-head span { font-size:12px; color:var(--muted); }
    .grid { display:grid; gap:14px; grid-template-columns:repeat(3, minmax(260px, 1fr)); }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:14px; box-shadow:0 10px 24px var(--shadow); }
    .panel h3 { margin:0 0 8px; font-size:15px; }
    svg { width:100%; height:330px; display:block; }
    .stats { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; color:var(--muted); margin-top:10px; }
    .tooltip { position:fixed; pointer-events:none; background:#fff; border:1px solid var(--line); border-radius:12px; padding:8px 10px; font-size:12px; box-shadow:0 8px 20px rgba(0,0,0,0.12); opacity:0; }
    @media (max-width: 980px) {
      .grid { grid-template-columns:1fr; }
      svg { height:360px; }
    }
  </style>
</head>
<body>
<div class="app">
  <div class="banner" id="statusBanner" hidden></div>
  <div class="toolbar">
    <label>Day <input type="range" id="timeSlider" min="0" max="0" step="1" value="0"></label>
    <label>Color mapping
      <select id="scaleModeSelect">
        <option value="sqrt" selected>sqrt</option>
        <option value="linear">linear</option>
        <option value="log">log</option>
      </select>
    </label>
    <span id="timeLabel" style="font-size:13px;color:var(--muted)"></span>
  </div>
  <div class="note">
    These pressures are model-scale hazard accumulators, not probabilities. Reservoir pressure is the latent regional reservoir state after daily decay and incoming seeding. Import pressure is the same-day R→F hazard mass that reaches susceptible farms from each region. Export pressure is the same-day F→R seeding mass pushed from infectious farms into each region reservoir.
  </div>
  <div class="metric-stack" id="metricRows"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const statusBanner = document.getElementById('statusBanner');
function setStatus(message) {
  if (!statusBanner) return;
  statusBanner.hidden = !message;
  statusBanner.textContent = message || '';
}
if (typeof window.d3 === 'undefined') {
  setStatus('The interactive map could not load D3. Check network access to the D3 CDN and reload this page.');
  throw new Error('D3 failed to load for the region comparison viewer.');
}
const payload = window.__TEMPORAL_SBM_REGION_PAYLOAD__;
if (!payload || !payload.geojson) {
  setStatus('The region comparison payload is missing or could not be parsed.');
  throw new Error('Missing region comparison payload.');
}
const metrics = payload.metrics || [];
const observed = payload.observed || [];
const synthetic = payload.synthetic || [];
const calendar = payload.calendar || [];
const geojson = payload.geojson;
const timeSlider = document.getElementById('timeSlider');
const scaleModeSelect = document.getElementById('scaleModeSelect');
const timeLabel = document.getElementById('timeLabel');
const metricRows = document.getElementById('metricRows');
const tooltip = d3.select('#tooltip');
timeSlider.max = String(Math.max(0, calendar.length - 1));
timeSlider.disabled = calendar.length <= 1;
const featureKey = feature => {
  const props = feature && feature.properties ? feature.properties : {};
  return String(props.statcode || props.CR_code || props.corop || props.code || props.id || feature.id || '').trim();
};
const metricLabel = metric => ({
  reservoir_pressure: 'Reservoir pressure',
  import_pressure: 'Import pressure',
  export_pressure: 'Export pressure',
}[metric] || metric.replaceAll('_', ' '));
const observedIndex = new Map();
const syntheticIndex = new Map();
for (const row of observed) {
  const key = `${row.ts}||${String(row.corop || '').trim()}`;
  observedIndex.set(key, row);
}
for (const row of synthetic) {
  const key = `${row.ts}||${String(row.corop || '').trim()}`;
  syntheticIndex.set(key, row);
}
function valuesFor(ts, metric) {
  const obs = new Map();
  const syn = new Map();
  for (const feature of geojson.features || []) {
    const corop = featureKey(feature);
    const key = `${ts}||${corop}`;
    const obsRow = observedIndex.get(key);
    const synRow = syntheticIndex.get(key);
    obs.set(corop, obsRow && obsRow[metric] != null ? +obsRow[metric] : NaN);
    syn.set(corop, synRow && synRow[metric] != null ? +synRow[metric] : NaN);
  }
  return {obs, syn};
}
function deltaValuesFor(obs, syn) {
  const delta = new Map();
  for (const feature of geojson.features || []) {
    const corop = featureKey(feature);
    const a = obs.get(corop);
    const b = syn.get(corop);
    delta.set(corop, Number.isFinite(a) && Number.isFinite(b) ? (b - a) : NaN);
  }
  return delta;
}
function geometryUsesProjectedCoordinates(collection) {
  let maxAbs = 0;
  function scanCoordinates(coords) {
    if (!Array.isArray(coords) || coords.length === 0) return;
    if (typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      maxAbs = Math.max(maxAbs, Math.abs(+coords[0]), Math.abs(+coords[1]));
      return;
    }
    for (const item of coords) scanCoordinates(item);
  }
  for (const feature of collection.features || []) {
    if (feature && feature.geometry) scanCoordinates(feature.geometry.coordinates);
  }
  return maxAbs > 180;
}
const useProjectedCoordinates = geometryUsesProjectedCoordinates(geojson);
function buildProjection(width, height) {
  if (useProjectedCoordinates) {
    return d3.geoIdentity().reflectY(true).fitSize([width, height], geojson);
  }
  return d3.geoMercator().fitSize([width, height], geojson);
}
function transformValue(value, mode, signed) {
  if (!Number.isFinite(value)) return NaN;
  const magnitude = Math.abs(+value);
  if (mode === 'sqrt') {
    const mapped = Math.sqrt(magnitude);
    return signed && value < 0 ? -mapped : mapped;
  }
  if (mode === 'log') {
    const mapped = Math.log1p(magnitude);
    return signed && value < 0 ? -mapped : mapped;
  }
  return +value;
}
function toSuperscriptExponent(exponent) {
  const digits = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
  return String(exponent).split('').map(char => digits[char] || char).join('');
}
function formatValue(value) {
  if (!Number.isFinite(value)) return 'n/a';
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 1e-2) || abs >= 1e4) {
    const exponent = Math.floor(Math.log10(abs));
    const mantissa = value / Math.pow(10, exponent);
    return `${mantissa.toFixed(2)} × 10${toSuperscriptExponent(exponent)}`;
  }
  if (abs >= 100) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(4);
  if (abs >= 0.01) return value.toFixed(5);
  return value.toFixed(6);
}
function sequentialScale(values, mode) {
  const transformed = values
    .filter(value => Number.isFinite(value) && value >= 0)
    .map(value => transformValue(value, mode, false));
  const bound = transformed.length ? d3.max(transformed) : 1;
  const scale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, bound > 0 ? bound : 1]);
  return value => Number.isFinite(value) ? scale(transformValue(Math.max(0, +value), mode, false)) : '#f1f5f9';
}
function divergingScale(values, mode) {
  const transformed = values
    .filter(Number.isFinite)
    .map(value => transformValue(value, mode, true));
  const bound = transformed.length ? d3.max(transformed.map(value => Math.abs(value))) : 1;
  const scale = d3.scaleDiverging(d3.interpolateRdBu).domain([bound > 0 ? bound : 1, 0, -(bound > 0 ? bound : 1)]);
  return value => Number.isFinite(value) ? scale(transformValue(value, mode, true)) : '#f1f5f9';
}
function statsText(values, isDelta) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return 'no finite values';
  const meanText = formatValue(d3.mean(finite));
  if (isDelta) {
    return `mean ${meanText} · min ${formatValue(d3.min(finite))} · max ${formatValue(d3.max(finite))}`;
  }
  return `mean ${meanText} · max ${formatValue(d3.max(finite))}`;
}
function renderMap(svgId, accessor, titleStatsId, colorFn, isDelta) {
  const svg = d3.select(svgId);
  svg.selectAll('*').remove();
  const width = svg.node().clientWidth || 320;
  const height = svg.node().clientHeight || 330;
  const projection = buildProjection(width, height);
  const path = d3.geoPath(projection);
  const g = svg.append('g');
  const values = [];
  for (const feature of geojson.features || []) {
    const value = accessor(feature);
    if (Number.isFinite(value)) values.push(value);
  }
  g.selectAll('path')
    .data(geojson.features || [])
    .join('path')
    .attr('d', path)
    .attr('fill', feature => {
      const value = accessor(feature);
      return Number.isFinite(value) ? colorFn(value) : '#f1f5f9';
    })
    .attr('stroke', '#c9d5e2')
    .attr('stroke-width', 0.8)
    .on('mousemove', function(event, feature) {
      const corop = featureKey(feature);
      const value = accessor(feature);
      tooltip.style('opacity', 1)
        .style('left', `${event.clientX + 14}px`)
        .style('top', `${event.clientY + 14}px`)
        .html(`<strong>${corop}</strong><br>${formatValue(value)}`);
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));
  const statTarget = document.getElementById(titleStatsId);
  statTarget.textContent = statsText(values, isDelta);
}
const rowConfigs = [];
for (const metric of metrics) {
  const slug = `metric-${metric.replaceAll('_', '-')}`;
  const section = document.createElement('section');
  section.className = 'metric-section';
  section.innerHTML = `
    <div class="section-head">
      <h2>${metricLabel(metric)}</h2>
      <span>Observed, synthetic, and synthetic minus observed</span>
    </div>
    <div class="grid">
      <div class="panel"><h3>Observed</h3><svg id="${slug}-observed"></svg><div class="stats" id="${slug}-observed-stats"></div></div>
      <div class="panel"><h3>Synthetic</h3><svg id="${slug}-synthetic"></svg><div class="stats" id="${slug}-synthetic-stats"></div></div>
      <div class="panel"><h3>Delta</h3><svg id="${slug}-delta"></svg><div class="stats" id="${slug}-delta-stats"></div></div>
    </div>
  `;
  metricRows.appendChild(section);
  rowConfigs.push({
    metric,
    observedSvg: `#${slug}-observed`,
    syntheticSvg: `#${slug}-synthetic`,
    deltaSvg: `#${slug}-delta`,
    observedStats: `${slug}-observed-stats`,
    syntheticStats: `${slug}-synthetic-stats`,
    deltaStats: `${slug}-delta-stats`,
  });
}
function update() {
  const idx = +timeSlider.value;
  const calendarEntry = calendar[idx] || {};
  const ts = +calendarEntry.ts;
  const scaleMode = scaleModeSelect.value || 'sqrt';
  timeLabel.textContent = `${calendarEntry.label || ts} · color ${scaleMode}`;
  let hasMatchedValues = false;
  for (const row of rowConfigs) {
    const {obs, syn} = valuesFor(ts, row.metric);
    const delta = deltaValuesFor(obs, syn);
    const seqValues = Array.from(obs.values()).concat(Array.from(syn.values())).filter(Number.isFinite);
    const deltaSeries = Array.from(delta.values()).filter(Number.isFinite);
    if (seqValues.length || deltaSeries.length) {
      hasMatchedValues = true;
    }
    const seqColor = sequentialScale(seqValues, scaleMode);
    const deltaColor = divergingScale(deltaSeries, scaleMode);
    renderMap(row.observedSvg, feature => obs.get(featureKey(feature)), row.observedStats, seqColor, false);
    renderMap(row.syntheticSvg, feature => syn.get(featureKey(feature)), row.syntheticStats, seqColor, false);
    renderMap(row.deltaSvg, feature => delta.get(featureKey(feature)), row.deltaStats, deltaColor, true);
  }
  setStatus(hasMatchedValues ? '' : 'No COROP values matched the map for the selected day.');
}
timeSlider.addEventListener('input', update);
scaleModeSelect.addEventListener('change', update);
update();
</script>
</body>
</html>""",
        encoding="utf-8",
    )
    return output_path


def _write_region_spatial_overview(
    region_spatial_per_snapshot: pd.DataFrame,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if region_spatial_per_snapshot.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    ts_values = pd.to_numeric(region_spatial_per_snapshot["ts"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(14.6, 12.2), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Regional spatial fit: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    metric_specs = [
        ("reservoir_pressure", "Reservoir pressure"),
        ("import_pressure", "Import pressure"),
        ("export_pressure", "Export pressure"),
    ]
    panel_specs = [
        ("spatial_correlation", "Daily spatial correlation across COROPs", "Correlation", (-0.05, 1.05)),
        ("mean_abs_delta", "Daily mean absolute delta across COROPs", "Mean absolute delta", None),
        ("share_mae", "Daily share-allocation MAE across COROPs", "MAE on normalized shares", None),
        ("hotspot_overlap_top3", "Daily top-3 hotspot overlap across COROPs", "Overlap", (-0.05, 1.05)),
        ("energy_distance", "Daily regional-field energy distance", "Energy distance", None),
        ("variogram_distance", "Daily regional-field variogram distance", "Variogram distance", None),
    ]
    for ax, (suffix, title, ylabel, ylim) in zip(np.atleast_1d(axes).ravel(), panel_specs):
        for metric_name, label in metric_specs:
            col = f"{metric_name}_{suffix}"
            if col in region_spatial_per_snapshot.columns:
                ax.plot(ts_values, pd.to_numeric(region_spatial_per_snapshot[col], errors="coerce"), marker="o", linewidth=2.1, markersize=4.2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        _style_legend(ax.legend())
    for axis in np.atleast_1d(axes).ravel():
        _set_timestamp_ticks(axis, ts_values, show_calendar_bands=False)
    output_path = Path(output_dir) / f"{sample_label}_region_spatial_overview.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path



def _compare_simulation_outputs(
    observed_scalar: pd.DataFrame,
    observed_daily: pd.DataFrame,
    observed_region_daily: pd.DataFrame,
    synthetic_scalar: pd.DataFrame,
    synthetic_daily: pd.DataFrame,
    synthetic_region_daily: pd.DataFrame,
    *,
    observed_diagnostics: Optional[dict[str, object]] = None,
    synthetic_diagnostics: Optional[dict[str, object]] = None,
    config: HybridSimulationConfig,
    sample_label: str,
    setting_label: str,
    sample_class: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    observed_diagnostics = observed_diagnostics or {}
    synthetic_diagnostics = synthetic_diagnostics or {}
    observed_daily_arrays = dict(observed_diagnostics.get("daily_arrays") or {})
    synthetic_daily_arrays = dict(synthetic_diagnostics.get("daily_arrays") or {})
    observed_region_arrays = dict(observed_diagnostics.get("region_arrays") or {})
    synthetic_region_arrays = dict(synthetic_diagnostics.get("region_arrays") or {})
    observed_farm_summary_obj = observed_diagnostics.get("farm_summary")
    synthetic_farm_summary_obj = synthetic_diagnostics.get("farm_summary")
    observed_farm_summary = observed_farm_summary_obj.copy() if isinstance(observed_farm_summary_obj, pd.DataFrame) else pd.DataFrame()
    synthetic_farm_summary = synthetic_farm_summary_obj.copy() if isinstance(synthetic_farm_summary_obj, pd.DataFrame) else pd.DataFrame()

    per_snapshot = _merge_daily_summaries(observed_daily, synthetic_daily)
    outcome_metrics = [
        "farm_attack_rate",
        "farm_attack_count",
        "farm_cumulative_incidence",
        "farm_peak_prevalence",
        "farm_peak_prevalence_fraction",
        "farm_peak_day_index",
        "farm_duration_days",
        "farm_first_infection_day_index",
        "farm_prevalence_auc",
        "reservoir_total_auc",
        "reservoir_max_peak",
        "farm_hazard_ff_auc",
        "farm_hazard_rf_auc",
        "farm_hazard_rf_share",
        "region_pressure_fr_auc",
        "region_pressure_rr_auc",
        "region_pressure_rr_share",
    ]
    outcome_summary = _summarise_outcome_distribution_comparison(observed_scalar, synthetic_scalar, outcome_metrics)

    farm_prevalence_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_farm_prevalence").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_farm_prevalence").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    farm_incidence_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_farm_incidence").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_farm_incidence").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    farm_cumulative_incidence_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_farm_cumulative_incidence").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_farm_cumulative_incidence").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    reservoir_total_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_reservoir_total").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_reservoir_total").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    reservoir_max_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_reservoir_max").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_reservoir_max").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    reservoir_positive_regions_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_reservoir_positive_regions").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_reservoir_positive_regions").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) else 0.0
    farm_hazard_ff_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_farm_hazard_ff").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_farm_hazard_ff").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) and "original_farm_hazard_ff" in per_snapshot.columns and "synthetic_farm_hazard_ff" in per_snapshot.columns else np.nan
    farm_hazard_rf_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_farm_hazard_rf").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_farm_hazard_rf").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) and "original_farm_hazard_rf" in per_snapshot.columns and "synthetic_farm_hazard_rf" in per_snapshot.columns else np.nan
    region_pressure_fr_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_region_pressure_fr").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_region_pressure_fr").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) and "original_region_pressure_fr" in per_snapshot.columns and "synthetic_region_pressure_fr" in per_snapshot.columns else np.nan
    region_pressure_rr_corr = _safe_correlation(
        _frame_numeric_series(per_snapshot, "original_region_pressure_rr").fillna(0.0).to_numpy(dtype=float),
        _frame_numeric_series(per_snapshot, "synthetic_region_pressure_rr").fillna(0.0).to_numpy(dtype=float),
    ) if len(per_snapshot) and "original_region_pressure_rr" in per_snapshot.columns and "synthetic_region_pressure_rr" in per_snapshot.columns else np.nan

    lag_diagnostics = pd.DataFrame([
        {"metric": "farm_prevalence", **_best_lagged_correlation(_frame_numeric_series(per_snapshot, "original_farm_prevalence").fillna(0.0), _frame_numeric_series(per_snapshot, "synthetic_farm_prevalence").fillna(0.0))},
        {"metric": "farm_incidence", **_best_lagged_correlation(_frame_numeric_series(per_snapshot, "original_farm_incidence").fillna(0.0), _frame_numeric_series(per_snapshot, "synthetic_farm_incidence").fillna(0.0))},
        {"metric": "reservoir_total", **_best_lagged_correlation(_frame_numeric_series(per_snapshot, "original_reservoir_total").fillna(0.0), _frame_numeric_series(per_snapshot, "synthetic_reservoir_total").fillna(0.0))},
    ]) if len(per_snapshot) else pd.DataFrame(columns=["metric", "best_lag_days", "best_lag_correlation"])

    daily_calibration = pd.DataFrame([
        _summarise_daily_interval_calibration(per_snapshot, metric_name)
        for metric_name in [
            "farm_prevalence",
            "farm_incidence",
            "farm_cumulative_incidence",
            "reservoir_total",
            "reservoir_max",
            "reservoir_positive_regions",
        ]
    ])
    daily_mean_comparison = pd.DataFrame([
        _summarise_daily_mean_comparison(per_snapshot, metric_name)
        for metric_name in [
            "farm_prevalence",
            "farm_incidence",
            "farm_cumulative_incidence",
            "reservoir_total",
            "reservoir_max",
            "reservoir_positive_regions",
        ]
    ])
    scalar_calibration = pd.DataFrame([
        _summarise_scalar_calibration(observed_scalar, synthetic_scalar, metric_name)
        for metric_name in [
            "farm_attack_rate",
            "farm_attack_count",
            "farm_cumulative_incidence",
            "farm_peak_prevalence",
            "farm_peak_prevalence_fraction",
            "farm_peak_day_index",
            "farm_duration_days",
            "farm_first_infection_day_index",
            "farm_prevalence_auc",
            "reservoir_total_auc",
            "reservoir_max_peak",
            "farm_hazard_rf_share",
            "region_pressure_rr_share",
        ]
    ])

    trajectory_distribution_summary = _summarise_trajectory_distribution_distances(observed_daily_arrays, synthetic_daily_arrays)

    region_merged = _merge_region_daily_summaries(observed_region_daily, synthetic_region_daily)
    region_spatial_per_snapshot, region_temporal_summary = _summarise_region_spatial_fit(region_merged)
    region_field_scores = _summarise_region_field_distances(
        ts_values=tuple(per_snapshot["ts"].tolist()) if "ts" in per_snapshot.columns else tuple(),
        observed_region_arrays=observed_region_arrays,
        synthetic_region_arrays=synthetic_region_arrays,
    )
    if not region_field_scores.empty:
        if region_spatial_per_snapshot.empty:
            region_spatial_per_snapshot = region_field_scores.copy()
        else:
            region_spatial_per_snapshot = region_spatial_per_snapshot.merge(region_field_scores, on=["day_index", "ts"], how="outer")

    farm_spatial_summary = _merge_farm_node_summaries(observed_farm_summary, synthetic_farm_summary)
    farm_corop_summary = _summarise_farm_corop_fit(farm_spatial_summary)

    farm_attack_probability_correlation = np.nan
    farm_attack_probability_mae = np.nan
    farm_attack_probability_wasserstein = np.nan
    farm_attack_probability_top10_overlap = np.nan
    farm_attack_probability_moran_original = np.nan
    farm_attack_probability_moran_synthetic = np.nan
    farm_attack_probability_moran_abs_delta = np.nan
    farm_first_infection_day_correlation = np.nan
    farm_first_infection_day_mae = np.nan
    farm_corop_attack_probability_correlation = np.nan
    farm_corop_attack_probability_mae = np.nan

    if not farm_spatial_summary.empty:
        original_attack = pd.to_numeric(farm_spatial_summary.get("original_network_generated_attack_probability", np.nan), errors="coerce")
        synthetic_attack = pd.to_numeric(farm_spatial_summary.get("synthetic_network_generated_attack_probability", np.nan), errors="coerce")
        valid_attack = original_attack.notna() & synthetic_attack.notna()
        if bool(valid_attack.any()):
            obs_attack = original_attack.loc[valid_attack].to_numpy(dtype=float)
            syn_attack = synthetic_attack.loc[valid_attack].to_numpy(dtype=float)
            farm_attack_probability_correlation = _safe_correlation(obs_attack, syn_attack)
            farm_attack_probability_mae = float(np.mean(np.abs(syn_attack - obs_attack)))
            farm_attack_probability_wasserstein = _wasserstein_distance_1d(obs_attack, syn_attack)
            farm_attack_probability_top10_overlap = _top_fraction_overlap(obs_attack, syn_attack, fraction=0.10, min_k=10)
            coords = farm_spatial_summary.loc[valid_attack, ["x", "y"]].to_numpy(dtype=float) if {"x", "y"}.issubset(farm_spatial_summary.columns) else np.empty((0, 2), dtype=float)
            farm_attack_probability_moran_original = _global_moran_i(obs_attack, coords)
            farm_attack_probability_moran_synthetic = _global_moran_i(syn_attack, coords)
            if np.isfinite(farm_attack_probability_moran_original) and np.isfinite(farm_attack_probability_moran_synthetic):
                farm_attack_probability_moran_abs_delta = float(abs(farm_attack_probability_moran_synthetic - farm_attack_probability_moran_original))
        original_first = pd.to_numeric(farm_spatial_summary.get("original_first_infection_day_median", np.nan), errors="coerce")
        synthetic_first = pd.to_numeric(farm_spatial_summary.get("synthetic_first_infection_day_median", np.nan), errors="coerce")
        valid_first = original_first.notna() & synthetic_first.notna()
        if bool(valid_first.any()):
            obs_first = original_first.loc[valid_first].to_numpy(dtype=float)
            syn_first = synthetic_first.loc[valid_first].to_numpy(dtype=float)
            farm_first_infection_day_correlation = _safe_correlation(obs_first, syn_first)
            farm_first_infection_day_mae = float(np.mean(np.abs(syn_first - obs_first)))
    if not farm_corop_summary.empty:
        original_corop_attack = pd.to_numeric(farm_corop_summary.get("original_network_generated_attack_probability", np.nan), errors="coerce")
        synthetic_corop_attack = pd.to_numeric(farm_corop_summary.get("synthetic_network_generated_attack_probability", np.nan), errors="coerce")
        valid_corop_attack = original_corop_attack.notna() & synthetic_corop_attack.notna()
        if bool(valid_corop_attack.any()):
            farm_corop_attack_probability_correlation = _safe_correlation(original_corop_attack.loc[valid_corop_attack], synthetic_corop_attack.loc[valid_corop_attack])
            farm_corop_attack_probability_mae = float(np.mean(np.abs(synthetic_corop_attack.loc[valid_corop_attack] - original_corop_attack.loc[valid_corop_attack])))

    summary: dict[str, object] = {
        "sample_label": sample_label,
        "setting_label": setting_label,
        "sample_class": sample_class,
        "simulation_model": config.model,
        "num_replicates": int(config.num_replicates),
        "network_horizon": int(len(per_snapshot)),
        "weight_mode": config.weight_mode,
        "weight_scale": float(config.weight_scale) if config.weight_scale is not None else None,
        "farm_prevalence_curve_correlation": float(farm_prevalence_corr),
        "farm_incidence_curve_correlation": float(farm_incidence_corr),
        "farm_cumulative_incidence_curve_correlation": float(farm_cumulative_incidence_corr),
        "reservoir_total_curve_correlation": float(reservoir_total_corr),
        "reservoir_max_curve_correlation": float(reservoir_max_corr),
        "reservoir_positive_regions_curve_correlation": float(reservoir_positive_regions_corr),
        "farm_hazard_ff_curve_correlation": float(farm_hazard_ff_corr) if pd.notna(farm_hazard_ff_corr) else np.nan,
        "farm_hazard_rf_curve_correlation": float(farm_hazard_rf_corr) if pd.notna(farm_hazard_rf_corr) else np.nan,
        "region_pressure_fr_curve_correlation": float(region_pressure_fr_corr) if pd.notna(region_pressure_fr_corr) else np.nan,
        "region_pressure_rr_curve_correlation": float(region_pressure_rr_corr) if pd.notna(region_pressure_rr_corr) else np.nan,
        "mean_abs_farm_prevalence_delta": float(pd.to_numeric(per_snapshot.get("farm_prevalence_delta", np.nan), errors="coerce").abs().mean()) if "farm_prevalence_delta" in per_snapshot.columns else np.nan,
        "mean_abs_farm_incidence_delta": float(pd.to_numeric(per_snapshot.get("farm_incidence_delta", np.nan), errors="coerce").abs().mean()) if "farm_incidence_delta" in per_snapshot.columns else np.nan,
        "mean_abs_farm_cumulative_incidence_delta": float(pd.to_numeric(per_snapshot.get("farm_cumulative_incidence_delta", np.nan), errors="coerce").abs().mean()) if "farm_cumulative_incidence_delta" in per_snapshot.columns else np.nan,
        "mean_abs_reservoir_total_delta": float(pd.to_numeric(per_snapshot.get("reservoir_total_delta", np.nan), errors="coerce").abs().mean()) if "reservoir_total_delta" in per_snapshot.columns else np.nan,
        "mean_abs_reservoir_max_delta": float(pd.to_numeric(per_snapshot.get("reservoir_max_delta", np.nan), errors="coerce").abs().mean()) if "reservoir_max_delta" in per_snapshot.columns else np.nan,
        "mean_abs_farm_hazard_ff_delta": float(pd.to_numeric(per_snapshot.get("farm_hazard_ff_delta", np.nan), errors="coerce").abs().mean()) if "farm_hazard_ff_delta" in per_snapshot.columns else np.nan,
        "mean_abs_farm_hazard_rf_delta": float(pd.to_numeric(per_snapshot.get("farm_hazard_rf_delta", np.nan), errors="coerce").abs().mean()) if "farm_hazard_rf_delta" in per_snapshot.columns else np.nan,
        "mean_abs_region_pressure_fr_delta": float(pd.to_numeric(per_snapshot.get("region_pressure_fr_delta", np.nan), errors="coerce").abs().mean()) if "region_pressure_fr_delta" in per_snapshot.columns else np.nan,
        "mean_abs_region_pressure_rr_delta": float(pd.to_numeric(per_snapshot.get("region_pressure_rr_delta", np.nan), errors="coerce").abs().mean()) if "region_pressure_rr_delta" in per_snapshot.columns else np.nan,
        "farm_attack_rate_wasserstein": _metric_lookup(outcome_summary, "farm_attack_rate", "wasserstein_distance"),
        "farm_cumulative_incidence_wasserstein": _metric_lookup(outcome_summary, "farm_cumulative_incidence", "wasserstein_distance"),
        "farm_peak_prevalence_wasserstein": _metric_lookup(outcome_summary, "farm_peak_prevalence", "wasserstein_distance"),
        "farm_peak_day_wasserstein": _metric_lookup(outcome_summary, "farm_peak_day_index", "wasserstein_distance"),
        "farm_duration_wasserstein": _metric_lookup(outcome_summary, "farm_duration_days", "wasserstein_distance"),
        "farm_first_infection_day_wasserstein": _metric_lookup(outcome_summary, "farm_first_infection_day_index", "wasserstein_distance"),
        "farm_prevalence_auc_wasserstein": _metric_lookup(outcome_summary, "farm_prevalence_auc", "wasserstein_distance"),
        "reservoir_total_auc_wasserstein": _metric_lookup(outcome_summary, "reservoir_total_auc", "wasserstein_distance"),
        "reservoir_max_peak_wasserstein": _metric_lookup(outcome_summary, "reservoir_max_peak", "wasserstein_distance"),
        "farm_hazard_rf_share_wasserstein": _metric_lookup(outcome_summary, "farm_hazard_rf_share", "wasserstein_distance"),
        "region_pressure_rr_share_wasserstein": _metric_lookup(outcome_summary, "region_pressure_rr_share", "wasserstein_distance"),
        "observed_farm_attack_rate_mean": _metric_lookup(outcome_summary, "farm_attack_rate", "original_mean"),
        "synthetic_farm_attack_rate_mean": _metric_lookup(outcome_summary, "farm_attack_rate", "synthetic_mean"),
        "observed_farm_peak_prevalence_mean": _metric_lookup(outcome_summary, "farm_peak_prevalence", "original_mean"),
        "synthetic_farm_peak_prevalence_mean": _metric_lookup(outcome_summary, "farm_peak_prevalence", "synthetic_mean"),
        "farm_prevalence_interval_coverage": _calibration_lookup(daily_calibration, "farm_prevalence", "interval_coverage"),
        "farm_incidence_interval_coverage": _calibration_lookup(daily_calibration, "farm_incidence", "interval_coverage"),
        "farm_cumulative_incidence_interval_coverage": _calibration_lookup(daily_calibration, "farm_cumulative_incidence", "interval_coverage"),
        "reservoir_total_interval_coverage": _calibration_lookup(daily_calibration, "reservoir_total", "interval_coverage"),
        "farm_prevalence_interval_width_mean": _calibration_lookup(daily_calibration, "farm_prevalence", "mean_interval_width"),
        "farm_incidence_interval_width_mean": _calibration_lookup(daily_calibration, "farm_incidence", "mean_interval_width"),
        "farm_prevalence_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "farm_prevalence", "curve_correlation"),
        "farm_incidence_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "farm_incidence", "curve_correlation"),
        "farm_cumulative_incidence_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "farm_cumulative_incidence", "curve_correlation"),
        "reservoir_total_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "reservoir_total", "curve_correlation"),
        "reservoir_max_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "reservoir_max", "curve_correlation"),
        "reservoir_positive_regions_mean_curve_correlation": _calibration_lookup(daily_mean_comparison, "reservoir_positive_regions", "curve_correlation"),
        "farm_prevalence_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "farm_prevalence", "mean_abs_delta"),
        "farm_incidence_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "farm_incidence", "mean_abs_delta"),
        "farm_cumulative_incidence_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "farm_cumulative_incidence", "mean_abs_delta"),
        "reservoir_total_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "reservoir_total", "mean_abs_delta"),
        "reservoir_max_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "reservoir_max", "mean_abs_delta"),
        "reservoir_positive_regions_mean_curve_mae": _calibration_lookup(daily_mean_comparison, "reservoir_positive_regions", "mean_abs_delta"),
        "farm_attack_rate_observed_median_in_synthetic_90pct": _calibration_lookup(scalar_calibration, "farm_attack_rate", "observed_median_in_synthetic_90pct"),
        "farm_attack_rate_observed_median_tail_area": _calibration_lookup(scalar_calibration, "farm_attack_rate", "observed_median_tail_area"),
        "farm_attack_rate_synthetic_interval_width": _calibration_lookup(scalar_calibration, "farm_attack_rate", "synthetic_interval_width"),
        "farm_peak_prevalence_observed_median_in_synthetic_90pct": _calibration_lookup(scalar_calibration, "farm_peak_prevalence", "observed_median_in_synthetic_90pct"),
        "farm_peak_prevalence_observed_median_tail_area": _calibration_lookup(scalar_calibration, "farm_peak_prevalence", "observed_median_tail_area"),
        "farm_duration_observed_median_in_synthetic_90pct": _calibration_lookup(scalar_calibration, "farm_duration_days", "observed_median_in_synthetic_90pct"),
        "farm_duration_observed_median_tail_area": _calibration_lookup(scalar_calibration, "farm_duration_days", "observed_median_tail_area"),
        "farm_prevalence_best_lag_correlation": _calibration_lookup(lag_diagnostics, "farm_prevalence", "best_lag_correlation"),
        "farm_prevalence_best_lag_days": _calibration_lookup(lag_diagnostics, "farm_prevalence", "best_lag_days"),
        "farm_incidence_best_lag_correlation": _calibration_lookup(lag_diagnostics, "farm_incidence", "best_lag_correlation"),
        "farm_incidence_best_lag_days": _calibration_lookup(lag_diagnostics, "farm_incidence", "best_lag_days"),
        "reservoir_total_best_lag_correlation": _calibration_lookup(lag_diagnostics, "reservoir_total", "best_lag_correlation"),
        "reservoir_total_best_lag_days": _calibration_lookup(lag_diagnostics, "reservoir_total", "best_lag_days"),
        "farm_prevalence_trajectory_energy_distance": _metric_lookup(trajectory_distribution_summary, "farm_prevalence", "trajectory_energy_distance"),
        "farm_incidence_trajectory_energy_distance": _metric_lookup(trajectory_distribution_summary, "farm_incidence", "trajectory_energy_distance"),
        "farm_cumulative_incidence_trajectory_energy_distance": _metric_lookup(trajectory_distribution_summary, "farm_cumulative_incidence", "trajectory_energy_distance"),
        "reservoir_total_trajectory_energy_distance": _metric_lookup(trajectory_distribution_summary, "reservoir_total", "trajectory_energy_distance"),
        "reservoir_max_trajectory_energy_distance": _metric_lookup(trajectory_distribution_summary, "reservoir_max", "trajectory_energy_distance"),
        "farm_attack_probability_correlation": farm_attack_probability_correlation,
        "farm_attack_probability_mae": farm_attack_probability_mae,
        "farm_attack_probability_wasserstein": farm_attack_probability_wasserstein,
        "farm_attack_probability_top10_overlap": farm_attack_probability_top10_overlap,
        "farm_attack_probability_moran_i_original": farm_attack_probability_moran_original,
        "farm_attack_probability_moran_i_synthetic": farm_attack_probability_moran_synthetic,
        "farm_attack_probability_moran_i_abs_delta": farm_attack_probability_moran_abs_delta,
        "farm_first_infection_day_correlation": farm_first_infection_day_correlation,
        "farm_first_infection_day_mae": farm_first_infection_day_mae,
        "farm_corop_attack_probability_correlation": farm_corop_attack_probability_correlation,
        "farm_corop_attack_probability_mae": farm_corop_attack_probability_mae,
    }

    if not region_spatial_per_snapshot.empty:
        summary["region_reservoir_spatial_correlation_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("reservoir_pressure_spatial_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_import_spatial_correlation_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("import_pressure_spatial_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_export_spatial_correlation_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("export_pressure_spatial_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_reservoir_hotspot_overlap_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("reservoir_pressure_hotspot_overlap_top3", np.nan), errors="coerce").dropna().mean())
        summary["region_import_hotspot_overlap_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("import_pressure_hotspot_overlap_top3", np.nan), errors="coerce").dropna().mean())
        summary["region_export_hotspot_overlap_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("export_pressure_hotspot_overlap_top3", np.nan), errors="coerce").dropna().mean())
        summary["region_reservoir_share_mae_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("reservoir_pressure_share_mae", np.nan), errors="coerce").dropna().mean())
        summary["region_reservoir_field_energy_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("reservoir_pressure_energy_distance", np.nan), errors="coerce").dropna().mean()) if "reservoir_pressure_energy_distance" in region_spatial_per_snapshot.columns else np.nan
        summary["region_import_field_energy_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("import_pressure_energy_distance", np.nan), errors="coerce").dropna().mean()) if "import_pressure_energy_distance" in region_spatial_per_snapshot.columns else np.nan
        summary["region_export_field_energy_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("export_pressure_energy_distance", np.nan), errors="coerce").dropna().mean()) if "export_pressure_energy_distance" in region_spatial_per_snapshot.columns else np.nan
        summary["region_reservoir_field_variogram_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("reservoir_pressure_variogram_distance", np.nan), errors="coerce").dropna().mean()) if "reservoir_pressure_variogram_distance" in region_spatial_per_snapshot.columns else np.nan
        summary["region_import_field_variogram_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("import_pressure_variogram_distance", np.nan), errors="coerce").dropna().mean()) if "import_pressure_variogram_distance" in region_spatial_per_snapshot.columns else np.nan
        summary["region_export_field_variogram_distance_mean"] = float(pd.to_numeric(region_spatial_per_snapshot.get("export_pressure_variogram_distance", np.nan), errors="coerce").dropna().mean()) if "export_pressure_variogram_distance" in region_spatial_per_snapshot.columns else np.nan
    if not region_temporal_summary.empty:
        summary["region_reservoir_temporal_correlation_mean"] = float(pd.to_numeric(region_temporal_summary.get("reservoir_pressure_temporal_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_import_temporal_correlation_mean"] = float(pd.to_numeric(region_temporal_summary.get("import_pressure_temporal_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_export_temporal_correlation_mean"] = float(pd.to_numeric(region_temporal_summary.get("export_pressure_temporal_correlation", np.nan), errors="coerce").dropna().mean())
        summary["region_reservoir_peak_day_abs_delta_mean"] = float(pd.to_numeric(region_temporal_summary.get("reservoir_pressure_peak_day_abs_delta", np.nan), errors="coerce").dropna().mean())

    detailed = {
        "observed_outcomes": observed_scalar,
        "synthetic_outcomes": synthetic_scalar,
        "observed_daily": observed_daily,
        "synthetic_daily": synthetic_daily,
        "outcome_distribution_summary": outcome_summary,
        "daily_calibration": daily_calibration,
        "daily_mean_comparison": daily_mean_comparison,
        "scalar_calibration": scalar_calibration,
        "trajectory_distribution_summary": trajectory_distribution_summary,
        "lag_diagnostics": lag_diagnostics,
        "observed_region_daily": observed_region_daily,
        "synthetic_region_daily": synthetic_region_daily,
        "region_spatial_per_snapshot": region_spatial_per_snapshot,
        "region_temporal_summary": region_temporal_summary,
        "region_field_scores": region_field_scores,
        "observed_farm_summary": observed_farm_summary,
        "synthetic_farm_summary": synthetic_farm_summary,
        "farm_spatial_summary": farm_spatial_summary,
        "farm_corop_summary": farm_corop_summary,
    }
    LOGGER.info(
        "Simulation comparison summary | sample=%s | prev_corr=%s | inc_corr=%s | attack_w1=%s | peak_w1=%s | duration_w1=%s | farm_space_corr=%s | region_res_corr=%s",
        sample_label,
        _metric_text(summary.get("farm_prevalence_curve_correlation")),
        _metric_text(summary.get("farm_incidence_curve_correlation")),
        _metric_text(summary.get("farm_attack_rate_wasserstein")),
        _metric_text(summary.get("farm_peak_prevalence_wasserstein")),
        _metric_text(summary.get("farm_duration_wasserstein")),
        _metric_text(summary.get("farm_attack_probability_correlation")),
        _metric_text(summary.get("region_reservoir_spatial_correlation_mean")),
    )
    return per_snapshot, summary, detailed


# Posterior aggregation
# Posterior aggregation
DETAIL_GROUP_KEYS: dict[str, list[str]] = {
    "per_snapshot": ["day_index", "ts"],
    "outcome_distribution_summary": ["metric"],
    "observed_daily": ["day_index", "ts"],
    "synthetic_daily": ["day_index", "ts"],
    "daily_calibration": ["metric"],
    "daily_mean_comparison": ["metric"],
    "scalar_calibration": ["metric"],
    "trajectory_distribution_summary": ["metric"],
    "lag_diagnostics": ["metric"],
    "observed_region_daily": ["day_index", "ts", "region_order", "region_node_id", "corop", "display_label"],
    "synthetic_region_daily": ["day_index", "ts", "region_order", "region_node_id", "corop", "display_label"],
    "region_spatial_per_snapshot": ["day_index", "ts"],
    "region_temporal_summary": ["region_node_id", "corop", "display_label"],
    "region_field_scores": ["day_index", "ts"],
    "observed_farm_summary": ["node_id"],
    "synthetic_farm_summary": ["node_id"],
    "farm_spatial_summary": ["node_id"],
    "farm_corop_summary": ["corop"],
    "uncertainty_decomposition": ["metric"],
}


def _aggregate_grouped_numeric_frames(
    frames: list[pd.DataFrame],
    *,
    group_keys: list[str],
    run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=group_keys)
    stacked_frames = []
    for idx, frame in enumerate(frames):
        if frame.empty:
            continue
        working = frame.copy()
        label = run_labels[idx] if run_labels and idx < len(run_labels) else f"run_{idx:04d}"
        working["__run_label"] = str(label)
        stacked_frames.append(working)
    if not stacked_frames:
        return pd.DataFrame(columns=group_keys)
    stacked = pd.concat(stacked_frames, ignore_index=True)
    value_columns = [column for column in stacked.columns if column not in set(group_keys) | {"__run_label"}]
    rows = []
    for group_values, group_frame in stacked.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {key: value for key, value in zip(group_keys, group_values)}
        row["posterior_num_runs"] = int(group_frame["__run_label"].nunique())
        for column in value_columns:
            series = group_frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
            if len(finite):
                row[column] = float(finite.median())
                if row["posterior_num_runs"] > 1:
                    row[f"{column}_q05"] = float(finite.quantile(0.05))
                    row[f"{column}_q95"] = float(finite.quantile(0.95))
                    row[f"{column}_mean"] = float(finite.mean())
                    row[f"{column}_std"] = float(finite.std(ddof=0))
            else:
                non_null = series.dropna()
                row[column] = non_null.iloc[0] if len(non_null) else None
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(group_keys).reset_index(drop=True)
    return out


def _aggregate_summary_payloads(
    setting_label: str,
    summaries: list[dict[str, object]],
    *,
    run_labels: Optional[list[str]] = None,
) -> dict[str, object]:
    if not summaries:
        raise ValueError(f"No summaries were provided for posterior aggregation of setting '{setting_label}'.")
    frame = pd.DataFrame(summaries)
    payload: dict[str, object] = {
        "posterior_num_runs": int(len(summaries)),
        "posterior_setting_label": setting_label,
    }
    if run_labels:
        payload["posterior_run_labels"] = [str(label) for label in run_labels]
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
        if len(finite):
            payload[column] = float(finite.median())
            if len(summaries) > 1:
                payload[f"{column}_q05"] = float(finite.quantile(0.05))
                payload[f"{column}_q95"] = float(finite.quantile(0.95))
                payload[f"{column}_mean"] = float(finite.mean())
                payload[f"{column}_std"] = float(finite.std(ddof=0))
        else:
            non_null = series.dropna()
            if len(non_null):
                payload[column] = non_null.iloc[0]
    return payload


# Plotting
def _load_matplotlib():
    runtime_root = Path(tempfile.gettempdir()) / "temporal_sbm_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    mpl_root = runtime_root / "matplotlib"
    cache_root = runtime_root / "cache"
    mpl_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    current_mpl = os.getenv("MPLCONFIGDIR")
    if not current_mpl or not os.access(current_mpl, os.W_OK):
        os.environ["MPLCONFIGDIR"] = str(mpl_root)
    current_xdg = os.getenv("XDG_CACHE_HOME")
    if not current_xdg or not os.access(current_xdg, os.W_OK):
        os.environ["XDG_CACHE_HOME"] = str(cache_root)
    try:
        import matplotlib
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is not installed; skipping plot generation.")
        return None
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _style_axis(ax, *, grid_axis: str = "both") -> None:
    ax.set_facecolor(PLOT_COLORS["panel"])
    ax.grid(axis=grid_axis, color=PLOT_COLORS["grid"], alpha=0.85, linewidth=0.85)
    ax.tick_params(colors=PLOT_COLORS["muted"], labelsize=9)
    ax.title.set_color(PLOT_COLORS["text"])
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_color(PLOT_COLORS["text"])
    ax.yaxis.label.set_color(PLOT_COLORS["text"])
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(PLOT_COLORS["grid_strong"])
        ax.spines[spine_name].set_linewidth(0.9)


def _style_figure(fig, axes: Iterable[object]) -> None:
    fig.patch.set_facecolor(PLOT_COLORS["figure"])
    for ax in np.atleast_1d(axes).ravel():
        if hasattr(ax, "plot"):
            _style_axis(ax)


def _style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("#ffffff")
    frame.set_edgecolor(PLOT_COLORS["grid_strong"])
    frame.set_linewidth(0.8)
    frame.set_alpha(0.96)
    for text_item in legend.get_texts():
        text_item.set_color(PLOT_COLORS["text"])
    if legend.get_title() is not None:
        legend.get_title().set_color(PLOT_COLORS["text"])


def _set_timestamp_ticks(ax, ts_values: np.ndarray, *, show_calendar_bands: bool = False) -> None:
    records = _calendar_records(ts_values)
    if not records:
        return
    tick_values = np.asarray([record["ts"] for record in records], dtype=float)
    if len(tick_values) > 14:
        index_values = np.linspace(0, len(tick_values) - 1, num=14)
        tick_values = tick_values[np.unique(np.round(index_values).astype(int))]
    tick_labels = [_ts_display_label(value) for value in tick_values]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")


def _save_figure(fig, output_path: Path, *, dpi: int = 180) -> None:
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")


def _plot_line_with_band(
    ax,
    *,
    x: np.ndarray,
    values: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(values, dtype=float)
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        finite = np.isfinite(lo) & np.isfinite(hi)
        if finite.any():
            ax.fill_between(x_values[finite], lo[finite], hi[finite], color=color, alpha=0.14, linewidth=0.0, zorder=1)
    ax.plot(x_values, y_values, color=color, linewidth=2.15, marker="o", markersize=3.6, linestyle=linestyle, label=label, zorder=3)


def _prefixed_numeric_array_if_present(
    frame: pd.DataFrame,
    prefix: str,
    column: Optional[str],
) -> Optional[np.ndarray]:
    if not column:
        return None
    full_column = f"{prefix}_{column}"
    if full_column not in frame.columns:
        return None
    return _frame_numeric_series(frame, full_column).to_numpy(dtype=float)


def _plot_metric_panel(
    ax,
    *,
    per_snapshot: pd.DataFrame,
    value_column: str,
    title: str,
    y_label: str,
    summary_value: Optional[object] = None,
    lower_column: Optional[str] = None,
    upper_column: Optional[str] = None,
    observed_label: str = "Observed median",
    synthetic_label: str = "Synthetic median",
) -> None:
    _plot_line_with_band(
        ax,
        x=per_snapshot["day_index"].to_numpy(dtype=float),
        values=_frame_numeric_series(per_snapshot, f"original_{value_column}").to_numpy(dtype=float),
        lower=_prefixed_numeric_array_if_present(per_snapshot, "original", lower_column),
        upper=_prefixed_numeric_array_if_present(per_snapshot, "original", upper_column),
        label=observed_label,
        color=PLOT_COLORS["original"],
    )
    _plot_line_with_band(
        ax,
        x=per_snapshot["day_index"].to_numpy(dtype=float),
        values=_frame_numeric_series(per_snapshot, f"synthetic_{value_column}").to_numpy(dtype=float),
        lower=_prefixed_numeric_array_if_present(per_snapshot, "synthetic", lower_column),
        upper=_prefixed_numeric_array_if_present(per_snapshot, "synthetic", upper_column),
        label=synthetic_label,
        color=PLOT_COLORS["synthetic"],
        linestyle="--",
    )
    title_text = title if summary_value is None else f"{title} (corr={_metric_text(summary_value, digits=2)})"
    ax.set_title(title_text)
    ax.set_xlabel("Day index")
    ax.set_ylabel(y_label)
    _style_legend(ax.legend(loc="best"))


def _plot_dashboard_scorecard(ax, summary: dict) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(False)
    ax.set_facecolor(PLOT_COLORS["panel_soft"])
    ax.set_title("Scenario scorecard")
    lines = [
        f"Model: {summary.get('simulation_model', 'unknown')}",
        f"Replicates: {int(summary.get('num_replicates', 0) or 0)}",
        f"Prevalence corr: {_metric_text(summary.get('farm_prevalence_curve_correlation'))}",
        f"Incidence median corr: {_metric_text(summary.get('farm_incidence_curve_correlation'))}",
        f"Attack W1: {_metric_text(summary.get('farm_attack_rate_wasserstein'))}",
        f"Peak W1: {_metric_text(summary.get('farm_peak_prevalence_wasserstein'))}",
        f"Duration W1: {_metric_text(summary.get('farm_duration_wasserstein'))}",
    ]
    y_position = 0.88
    for line in lines:
        ax.text(
            0.04,
            y_position,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color=PLOT_COLORS["text"],
        )
        y_position -= 0.12


def _write_sample_dashboard(
    per_snapshot: pd.DataFrame,
    summary: dict,
    outcome_summary: pd.DataFrame,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if per_snapshot.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    fig, axes = plt.subplots(4, 2, figsize=(15.4, 17.4), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Hybrid transmission validation: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    _plot_metric_panel(
        axes[0, 0],
        per_snapshot=per_snapshot,
        value_column="farm_prevalence",
        title="Farm prevalence",
        y_label="Infectious farms",
        summary_value=summary.get("farm_prevalence_curve_correlation"),
        lower_column="farm_prevalence_q05",
        upper_column="farm_prevalence_q95",
    )
    _plot_metric_panel(
        axes[0, 1],
        per_snapshot=per_snapshot,
        value_column="farm_incidence_mean",
        title="Farm incidence mean",
        y_label="Newly infected farms",
        lower_column="farm_incidence_q05",
        upper_column="farm_incidence_q95",
        observed_label="Observed mean",
        synthetic_label="Synthetic mean",
    )
    _plot_metric_panel(
        axes[1, 0],
        per_snapshot=per_snapshot,
        value_column="farm_infection_event_probability",
        title="Farm infection-event probability",
        y_label="Share of runs with >=1 new infection",
        lower_column="farm_infection_event_probability_q05",
        upper_column="farm_infection_event_probability_q95",
        observed_label="Observed share",
        synthetic_label="Synthetic share",
    )
    axes[1, 0].set_ylim(-0.02, 1.02)
    _plot_metric_panel(
        axes[1, 1],
        per_snapshot=per_snapshot,
        value_column="farm_cumulative_incidence",
        title="Farm cumulative incidence",
        y_label="Cumulative infected farms",
        summary_value=summary.get("farm_cumulative_incidence_curve_correlation"),
        lower_column="farm_cumulative_incidence_q05",
        upper_column="farm_cumulative_incidence_q95",
    )
    _plot_metric_panel(
        axes[2, 0],
        per_snapshot=per_snapshot,
        value_column="reservoir_total",
        title="Reservoir total",
        y_label="Reservoir pressure total",
        summary_value=summary.get("reservoir_total_curve_correlation"),
        lower_column="reservoir_total_q05",
        upper_column="reservoir_total_q95",
    )
    _plot_metric_panel(
        axes[2, 1],
        per_snapshot=per_snapshot,
        value_column="reservoir_max",
        title="Reservoir max",
        y_label="Largest regional pressure",
        summary_value=summary.get("reservoir_max_curve_correlation"),
        lower_column="reservoir_max_q05",
        upper_column="reservoir_max_q95",
    )
    _plot_metric_panel(
        axes[3, 0],
        per_snapshot=per_snapshot,
        value_column="reservoir_positive_regions",
        title="Positive reservoir regions",
        y_label="Regions above zero pressure",
        summary_value=summary.get("reservoir_positive_regions_curve_correlation"),
        lower_column="reservoir_positive_regions_q05",
        upper_column="reservoir_positive_regions_q95",
    )
    _plot_dashboard_scorecard(axes[3, 1], summary)

    output_path = output_dir / f"{sample_label}_dashboard.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_sample_parity_plot(
    outcome_summary: pd.DataFrame,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if outcome_summary.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    plot_metrics = outcome_summary.loc[
        outcome_summary["metric"].isin(["farm_attack_rate", "farm_peak_prevalence", "farm_peak_day_index", "farm_duration_days"])
    ].copy()
    if plot_metrics.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle(f"Farm-outcome parity: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    for ax, (_, row) in zip(axes_array, plot_metrics.iterrows()):
        original = float(pd.to_numeric(pd.Series([row["original_median"]]), errors="coerce").fillna(0.0).iloc[0])
        synthetic = float(pd.to_numeric(pd.Series([row["synthetic_median"]]), errors="coerce").fillna(0.0).iloc[0])
        limit = max(original, synthetic, 1e-6)
        ax.scatter([original], [synthetic], s=80, color=PLOT_COLORS["accent"], edgecolors="white", linewidths=0.8)
        ax.plot([0.0, limit * 1.05], [0.0, limit * 1.05], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        ax.set_title(row["metric"].replace("_", " ").title())
        ax.set_xlabel("Observed median")
        ax.set_ylabel("Synthetic median")
        ax.set_xlim(0.0, limit * 1.08)
        ax.set_ylim(0.0, limit * 1.08)
    for ax in axes_array[len(plot_metrics):]:
        ax.axis("off")

    output_path = output_dir / f"{sample_label}_parity.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_curve_delta_plot(
    per_snapshot: pd.DataFrame,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if per_snapshot.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    metric_specs = [
        ("farm_prevalence_delta", "Farm prevalence delta"),
        ("farm_incidence_delta", "Farm incidence delta"),
        ("farm_cumulative_incidence_delta", "Farm cumulative-incidence delta"),
        ("reservoir_total_delta", "Reservoir total delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle(f"Observed vs synthetic median trajectory deltas: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    x_values = per_snapshot["day_index"].to_numpy(dtype=float)
    for ax, (column, title) in zip(axes_array, metric_specs):
        values = _frame_numeric_series(per_snapshot, column).fillna(0.0).to_numpy(dtype=float)
        ax.axhline(0.0, color=PLOT_COLORS["grid_strong"], linewidth=1.0, zorder=1)
        ax.plot(x_values, values, color=PLOT_COLORS["delta"], linewidth=2.0, marker="o", markersize=3.2, zorder=3)
        positive = np.where(values >= 0.0, values, 0.0)
        negative = np.where(values <= 0.0, values, 0.0)
        ax.fill_between(x_values, 0.0, positive, color=PLOT_COLORS["delta"], alpha=0.16)
        ax.fill_between(x_values, 0.0, negative, color=PLOT_COLORS["accent"], alpha=0.14)
        ax.set_title(f"{title} (mean abs={_metric_text(np.mean(np.abs(values)), digits=3)})")
        ax.set_xlabel("Day index")
        ax.set_ylabel("Synthetic - observed")
    output_path = output_dir / f"{sample_label}_delta.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_daily_mean_comparison_plot(
    per_snapshot: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if per_snapshot.empty:
        return None
    metric_specs = [
        ("farm_prevalence_mean", "Farm prevalence mean", "Infectious farms", "farm_prevalence_mean_curve_correlation"),
        ("farm_incidence_mean", "Farm incidence mean", "Newly infected farms", "farm_incidence_mean_curve_correlation"),
        ("farm_cumulative_incidence_mean", "Farm cumulative-incidence mean", "Cumulative infected farms", "farm_cumulative_incidence_mean_curve_correlation"),
        ("reservoir_total_mean", "Reservoir total mean", "Reservoir pressure total", "reservoir_total_mean_curve_correlation"),
        ("reservoir_max_mean", "Reservoir max mean", "Largest regional pressure", "reservoir_max_mean_curve_correlation"),
        ("reservoir_positive_regions_mean", "Positive reservoir regions mean", "Regions above zero pressure", "reservoir_positive_regions_mean_curve_correlation"),
    ]
    available_specs = [
        spec
        for spec in metric_specs
        if f"original_{spec[0]}" in per_snapshot.columns and f"synthetic_{spec[0]}" in per_snapshot.columns
    ]
    if not available_specs:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    ncols = 2
    nrows = int(math.ceil(len(available_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.6, max(5.2, 4.5 * nrows)), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle(f"Observed vs synthetic daily mean trajectories: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    for ax, (value_column, title, y_label, summary_key) in zip(axes_array, available_specs):
        base_name = value_column[: -len("_mean")]
        _plot_metric_panel(
            ax,
            per_snapshot=per_snapshot,
            value_column=value_column,
            title=title,
            y_label=y_label,
            summary_value=summary.get(summary_key),
            lower_column=f"{base_name}_q05",
            upper_column=f"{base_name}_q95",
            observed_label="Observed mean",
            synthetic_label="Synthetic mean",
        )
    for ax in axes_array[len(available_specs):]:
        ax.axis("off")

    output_path = output_dir / f"{sample_label}_daily_mean_compare.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_channel_diagnostics_plot(
    per_snapshot: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if per_snapshot.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Hybrid-channel diagnostics: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    panel_specs = [
        ("farm_hazard_ff", "Farm→Farm hazard", "Hazard mass", summary.get("farm_hazard_ff_curve_correlation")),
        ("farm_hazard_rf", "Region→Farm hazard", "Hazard mass", summary.get("farm_hazard_rf_curve_correlation")),
        ("region_pressure_fr", "Farm→Region seeding", "Pressure mass", summary.get("region_pressure_fr_curve_correlation")),
        ("region_pressure_rr", "Region→Region diffusion", "Pressure mass", summary.get("region_pressure_rr_curve_correlation")),
    ]
    for ax, (value_column, title, ylabel, score) in zip(np.atleast_1d(axes).ravel(), panel_specs):
        _plot_metric_panel(
            ax,
            per_snapshot=per_snapshot,
            value_column=value_column,
            title=title,
            y_label=ylabel,
            summary_value=score,
            lower_column=f"{value_column}_q05",
            upper_column=f"{value_column}_q95",
        )
    output_path = output_dir / f"{sample_label}_channel_diagnostics.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_farm_spatial_overview(
    farm_spatial_summary: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if farm_spatial_summary.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    x = pd.to_numeric(farm_spatial_summary.get("x", np.nan), errors="coerce")
    y = pd.to_numeric(farm_spatial_summary.get("y", np.nan), errors="coerce")
    original = pd.to_numeric(farm_spatial_summary.get("original_network_generated_attack_probability", np.nan), errors="coerce")
    synthetic = pd.to_numeric(farm_spatial_summary.get("synthetic_network_generated_attack_probability", np.nan), errors="coerce")
    delta = pd.to_numeric(farm_spatial_summary.get("network_generated_attack_probability_delta", np.nan), errors="coerce")
    valid_xy = x.notna() & y.notna()
    if not bool(valid_xy.any()):
        return None
    x_values = x.loc[valid_xy].to_numpy(dtype=float)
    y_values = y.loc[valid_xy].to_numpy(dtype=float)
    original_values = original.loc[valid_xy].fillna(0.0).to_numpy(dtype=float)
    synthetic_values = synthetic.loc[valid_xy].fillna(0.0).to_numpy(dtype=float)
    delta_values = delta.loc[valid_xy].fillna(0.0).to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 11.4), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Farm-level spatial validation: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    vmax = max(float(np.nanmax(original_values)) if len(original_values) else 0.0, float(np.nanmax(synthetic_values)) if len(synthetic_values) else 0.0, 1e-6)
    delta_vmax = max(float(np.nanmax(np.abs(delta_values))) if len(delta_values) else 0.0, 1e-6)

    ax = axes[0, 0]
    observed_plot = ax.scatter(x_values, y_values, c=original_values, s=20, linewidths=0.0, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title("Observed network-generated attack probability")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(observed_plot, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[0, 1]
    synthetic_plot = ax.scatter(x_values, y_values, c=synthetic_values, s=20, linewidths=0.0, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title("Synthetic network-generated attack probability")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(synthetic_plot, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 0]
    delta_plot = ax.scatter(x_values, y_values, c=delta_values, s=20, linewidths=0.0, cmap="coolwarm", vmin=-delta_vmax, vmax=delta_vmax)
    ax.set_title("Synthetic minus observed attack probability")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(delta_plot, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 1]
    valid_scatter = np.isfinite(original_values) & np.isfinite(synthetic_values)
    obs_scatter = original_values[valid_scatter]
    syn_scatter = synthetic_values[valid_scatter]
    if len(obs_scatter):
        ax.scatter(obs_scatter, syn_scatter, s=16, alpha=0.55, color=PLOT_COLORS["accent"], edgecolors="none")
        upper = max(float(np.max(obs_scatter)), float(np.max(syn_scatter)), 1e-6)
        ax.plot([0.0, upper], [0.0, upper], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        ax.set_xlim(0.0, upper * 1.03)
        ax.set_ylim(0.0, upper * 1.03)
    ax.set_title("Farm-level parity")
    ax.set_xlabel("Observed attack probability")
    ax.set_ylabel("Synthetic attack probability")
    annotation = "\n".join([
        f"corr = {_metric_text(summary.get('farm_attack_probability_correlation'), digits=3)}",
        f"MAE = {_metric_text(summary.get('farm_attack_probability_mae'), digits=3)}",
        f"top-10% overlap = {_metric_text(summary.get('farm_attack_probability_top10_overlap'), digits=3)}",
        f"|Δ Moran's I| = {_metric_text(summary.get('farm_attack_probability_moran_i_abs_delta'), digits=3)}",
        f"first-day corr = {_metric_text(summary.get('farm_first_infection_day_correlation'), digits=3)}",
    ])
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=PLOT_COLORS["text"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": PLOT_COLORS["grid_strong"], "alpha": 0.94},
    )
    output_path = output_dir / f"{sample_label}_farm_spatial_overview.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_outcome_distribution_plot(
    observed_scalar: pd.DataFrame,
    synthetic_scalar: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if observed_scalar.empty and synthetic_scalar.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    metric_specs = [
        ("farm_attack_rate", "Farm attack rate"),
        ("farm_peak_prevalence", "Farm peak prevalence"),
        ("farm_peak_day_index", "Farm peak day"),
        ("farm_duration_days", "Farm duration"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 12.0), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle(f"Outcome distributions: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    for ax, (metric_name, title) in zip(axes_array[:4], metric_specs):
        observed = pd.to_numeric(observed_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        synthetic = pd.to_numeric(synthetic_scalar.get(metric_name, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        if len(observed) == 0 and len(synthetic) == 0:
            ax.text(0.5, 0.5, "No replicate data", ha="center", va="center", transform=ax.transAxes)
            continue
        box = ax.boxplot(
            [observed if len(observed) else np.array([np.nan]), synthetic if len(synthetic) else np.array([np.nan])],
            patch_artist=True,
            widths=0.55,
        )
        for patch, color in zip(box["boxes"], [PLOT_COLORS["original"], PLOT_COLORS["synthetic"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color(PLOT_COLORS["text"])
            median.set_linewidth(1.8)
        ax.set_xticks([1, 2], ["Observed", "Synthetic"])
        ax.set_title(f"{title} (W1={_metric_text(_metric_lookup(outcome_summary, metric_name, 'wasserstein_distance'), digits=3)})")
        ax.set_ylabel(title)

    primary_subset = outcome_summary.loc[
        outcome_summary["metric"].isin([metric_name for metric_name, _ in metric_specs])
    ].copy() if "metric" in outcome_summary.columns else pd.DataFrame()
    mismatch_ax = axes_array[4]
    if len(primary_subset):
        positions = np.arange(len(primary_subset), dtype=float)
        values = pd.to_numeric(primary_subset["wasserstein_distance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mismatch_ax.barh(positions, values, color=PLOT_COLORS["accent"], alpha=0.92)
        mismatch_ax.set_yticks(positions, [metric.replace("_", " ").title() for metric in primary_subset["metric"].tolist()])
        mismatch_ax.set_xlabel("Wasserstein distance")
        mismatch_ax.set_title("Primary mismatch summary")
        mismatch_ax.invert_yaxis()
    else:
        mismatch_ax.text(0.5, 0.5, "No mismatch summary available", ha="center", va="center", transform=mismatch_ax.transAxes)

    secondary_ax = axes_array[5]
    secondary_metrics = [
        ("farm_cumulative_incidence", "Cum. incidence"),
        ("farm_prevalence_auc", "Prevalence AUC"),
        ("reservoir_total_auc", "Reservoir AUC"),
        ("reservoir_max_peak", "Reservoir peak"),
    ]
    if len(outcome_summary) and "metric" in outcome_summary.columns:
        positions = np.arange(len(secondary_metrics), dtype=float)
        values = np.asarray(
            [_metric_lookup(outcome_summary, metric_name, "wasserstein_distance") or np.nan for metric_name, _ in secondary_metrics],
            dtype=float,
        )
        clean_values = np.nan_to_num(values, nan=0.0)
        secondary_ax.barh(positions, clean_values, color=PLOT_COLORS["region"], alpha=0.88)
        secondary_ax.set_yticks(positions, [label for _, label in secondary_metrics])
        secondary_ax.set_xlabel("Wasserstein distance")
        secondary_ax.set_title("Secondary mismatch summary")
        secondary_ax.invert_yaxis()
    else:
        secondary_ax.text(0.5, 0.5, "No secondary summary available", ha="center", va="center", transform=secondary_ax.transAxes)

    output_path = output_dir / f"{sample_label}_distribution.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def write_all_samples_overview(summary_rows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_rows.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    frame = summary_rows.copy()
    label_source = frame["sample_label"] if "sample_label" in frame.columns else frame.index.astype(str)
    display_labels = [
        textwrap.fill(_setting_display_payload(str(label))["short_label"], width=24, break_long_words=False, break_on_hyphens=False)
        for label in label_source.tolist()
    ]
    positions = np.arange(len(frame), dtype=float)

    metrics = [
        ("farm_prevalence_curve_correlation", "Farm prevalence corr"),
        ("farm_incidence_curve_correlation", "Farm incidence corr"),
        ("farm_attack_probability_correlation", "Farm risk corr"),
        ("region_reservoir_spatial_correlation_mean", "Region spatial corr"),
        ("farm_attack_rate_wasserstein", "Farm attack-rate W1"),
        ("farm_peak_prevalence_wasserstein", "Farm peak-prev W1"),
        ("farm_peak_day_wasserstein", "Farm peak-day W1"),
        ("farm_duration_wasserstein", "Farm duration W1"),
    ]
    metrics = [(column, title) for column, title in metrics if column in frame.columns]
    if not metrics:
        return None

    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    fig_height = max(4.2 * nrows, 0.50 * len(frame) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, fig_height), constrained_layout=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle("Hybrid transmission validation across generated samples", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    last_metric_axis = axes_array[len(metrics) - 1]
    for ax, (column, title) in zip(axes_array, metrics):
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.hlines(positions, 0.0, values, color=PLOT_COLORS["grid_strong"], linewidth=2.0, zorder=1)
        ax.scatter(values, positions, s=68, color=PLOT_COLORS["original"], edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_yticks(positions)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        if "correlation" in column:
            if ax is last_metric_axis:
                ax.set_xlim(-0.58, 1.05)
                ax.axvline(0.0, color=PLOT_COLORS["grid_strong"], linewidth=1.0, zorder=0)
            else:
                ax.set_xlim(0.0, 1.05)
        else:
            if ax is last_metric_axis:
                left = -0.58
                right = max(1.0, float(np.nanmax(values)) * 1.1)
                ax.set_xlim(left, right)
                ax.axvline(0.0, color=PLOT_COLORS["grid_strong"], linewidth=1.0, zorder=0)
            else:
                ax.set_xlim(0.0, max(1.0, float(np.nanmax(values)) * 1.1))
        ax.grid(axis="x", color=PLOT_COLORS["grid"], alpha=0.85)
        ax.set_axisbelow(True)
        if ax is last_metric_axis:
            for label_y, label_text in zip(positions, display_labels):
                ax.text(-0.54, label_y - 0.22, label_text, ha="left", va="bottom", fontsize=8.3, color=PLOT_COLORS["text"], linespacing=1.05, zorder=4)
    for ax in axes_array[len(metrics):]:
        ax.axis("off")

    output_path = Path(output_dir) / "all_samples_overview.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


# Report writing
def _simulation_figure_reading_notes(*, include_scenario_overview: bool = False) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    if include_scenario_overview:
        notes.extend(
            [
                (
                    "Scenario comparison panels",
                    "The overview figure collects the main fit metrics. Higher correlations are better, while lower Wasserstein distances mean the synthetic epidemics stay closer to the observed panel.",
                ),
                (
                    "Tradeoff plot",
                    "Upper-left scenarios are the closest overall fit. Points farther right or lower down are more useful when you want a stress case that separates settings rather than a close reconstruction.",
                ),
                (
                    "Metric heatmap",
                    "Read across a row to see whether one scenario is consistently strong or only wins on one metric family. Darker cells mean a better within-metric rank.",
                ),
            ]
        )
    notes.extend(
        [
            (
                "Trajectory dashboard",
                "Most panels compare observed and synthetic center lines with 5th to 95th percentile bands. Close line shape and overlapping bands mean the simulator is matching both timing and spread.",
            ),
            (
                "Farm incidence panel",
                "This panel uses the daywise mean rather than the median because sparse outbreaks often leave the daily median at zero for long stretches. Read it together with the uncertainty band, not as a single deterministic curve.",
            ),
            (
                "Infection-event probability panel",
                "This panel shows the share of replicates with at least one new farm infection on each day. It is the quickest way to spot timing shifts when raw incidence counts are mostly zero.",
            ),
            (
                "Delta view",
                "Delta curves are synthetic minus observed medians. Long runs above zero mean the synthetic process overshoots the observed panel, while long runs below zero mean it misses sustained activity.",
            ),
            (
                "Daily mean view",
                "The daily mean figure shows the average trajectory across replicates. It is useful when the daily median stays at zero and hides timing or magnitude shifts that are still visible in the mean.",
            ),
            (
                "Distribution and parity views",
                "The distribution figure shows the full replicate spread for scalar endpoints such as attack rate, peak size, peak day, and duration. The parity figure condenses each endpoint to observed versus synthetic medians for fast scanning.",
            ),
            (
                "Hybrid-channel diagnostics",
                "These panels separate farm→farm, region→farm, farm→region, and region→region flow so you can see whether the hybrid coupling itself is preserved rather than only the final epidemic totals.",
            ),
            (
                "Farm spatial view",
                "This figure checks whether the same farms are consistently high-risk. Read the two maps with the parity panel and Moran's I delta together: good alignment means the synthetic network preserves both hotspot location and overall spatial clustering.",
            ),
            (
                "Regional field distances",
                "Energy distance summarizes daily mismatch of the whole regional field, while variogram distance is more sensitive to dependence and clustering structure across COROPs.",
            ),
        ]
    )
    return notes


def _simulation_table_reading_notes(*, include_scenario_scorecard: bool = False) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    if include_scenario_scorecard:
        notes.append(
            (
                "Scenario scorecard",
                "Read each row as one scenario profile. Close-fit scenarios should have high correlations, low Wasserstein distances, solid coverage, and modest endpoint gaps. Separation scenarios are allowed to move away from that corner if they reveal differences between observed and synthetic networks.",
            )
        )
    notes.extend(
        [
            (
                "Correlations and Wasserstein distances",
                "Correlation columns reward matching time ordering and curve shape, so higher is better. Wasserstein distances measure outcome gaps, so lower is better.",
            ),
            (
                "Daily 90% coverage",
                "Coverage near 0.9 means the observed daily path usually sits inside the synthetic 90% band. Much lower values point to bias or bands that are too narrow, while values pinned near 1 can also mean the bands are too wide.",
            ),
            (
                "Daily mean comparison",
                "These columns compare the mean trajectory across replicates for each day. Correlation rewards matching timing and shape, while mean absolute delta shows the average size of the daily gap.",
            ),
            (
                "Scalar in-band flags and tail area",
                "An in-band value of 1 means the observed median falls inside the synthetic 90% interval. Tail area close to 1 means the observed median sits near the center of the synthetic distribution, while values near 0 mean it lands in the tails.",
            ),
            (
                "Network uncertainty share",
                "This splits predictive variance into between-panel network variation and within-panel epidemic randomness. Values near 1 mean panel-to-panel network differences dominate, while values near 0 mean epidemic stochasticity dominates.",
            ),
            (
                "Energy and variogram distances",
                "Lower is better. Energy distance measures ensemble-to-ensemble mismatch of whole trajectories or spatial fields. Variogram distance puts more emphasis on getting dependence and clustering structure right.",
            ),
            (
                "Lag diagnostics and farm spatial metrics",
                "Best-lag correlation tells you whether the synthetic epidemic is shape-correct but shifted in time. Farm-level attack-probability correlation, hotspot overlap, and Moran's I delta tell you whether the detailed spatial risk pattern is preserved.",
            ),
        ]
    )
    return notes


def _simulation_notes_markdown(title: str, notes: list[tuple[str, str]]) -> list[str]:
    lines = ["", f"## {title}", ""]
    for label, text in notes:
        lines.append(f"- **{label}**: {text}")
    lines.append("")
    return lines


def _simulation_notes_text(notes: list[tuple[str, str]]) -> str:
    return " ".join(f"{label}: {text}" for label, text in notes)


def _render_explanation_toggle(
    *,
    control_id: str,
    text: Optional[str],
    button_label: str = "How to read this",
) -> str:
    if not text:
        return ""
    escaped_id = html.escape(control_id)
    escaped_text = html.escape(text)
    escaped_label = html.escape(button_label)
    return (
        "<div class='explain-widget'>"
        f"<button type='button' class='explain-button' data-explain-target='{escaped_id}' aria-controls='{escaped_id}' aria-expanded='false'>{escaped_label}</button>"
        f"<div id='{escaped_id}' class='explain-panel' hidden><p>{escaped_text}</p></div>"
        "</div>"
    )


def _render_section_heading(
    title: str,
    *,
    control_id: str,
    explain_text: Optional[str] = None,
    button_label: str = "How to read this",
) -> str:
    explain_html = _render_explanation_toggle(
        control_id=control_id,
        text=explain_text,
        button_label=button_label,
    )
    return (
        "<div class='section-heading'>"
        f"{explain_html}"
        f"<h2>{html.escape(title)}</h2>"
        "</div>"
    )


def write_report(
    per_snapshot: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    sample_label: str,
    *,
    detailed_outputs: Optional[dict[str, pd.DataFrame]] = None,
    node_frame: Optional[pd.DataFrame] = None,
    manifest: Optional[dict] = None,
    corop_geojson_path: Optional[Path] = None,
    focal_corop: str = "",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(
        "Writing simulation report | output_dir=%s | sample_label=%s | snapshot_rows=%s",
        output_dir,
        sample_label,
        len(per_snapshot),
    )
    csv_path = output_dir / f"{sample_label}_per_snapshot.csv"
    json_path = output_dir / f"{sample_label}_summary.json"
    md_path = output_dir / f"{sample_label}_report.md"

    detailed_outputs = detailed_outputs or {}
    outcome_summary = detailed_outputs.get("outcome_distribution_summary", pd.DataFrame())
    observed_scalar = detailed_outputs.get("observed_outcomes", pd.DataFrame())
    synthetic_scalar = detailed_outputs.get("synthetic_outcomes", pd.DataFrame())
    daily_calibration = detailed_outputs.get("daily_calibration", pd.DataFrame())
    daily_mean_comparison = detailed_outputs.get("daily_mean_comparison", pd.DataFrame())
    scalar_calibration = detailed_outputs.get("scalar_calibration", pd.DataFrame())
    uncertainty_decomposition = detailed_outputs.get("uncertainty_decomposition", pd.DataFrame())
    observed_region_daily = detailed_outputs.get("observed_region_daily", pd.DataFrame())
    synthetic_region_daily = detailed_outputs.get("synthetic_region_daily", pd.DataFrame())
    region_spatial_per_snapshot = detailed_outputs.get("region_spatial_per_snapshot", pd.DataFrame())
    region_temporal_summary = detailed_outputs.get("region_temporal_summary", pd.DataFrame())
    trajectory_distribution_summary = detailed_outputs.get("trajectory_distribution_summary", pd.DataFrame())
    lag_diagnostics = detailed_outputs.get("lag_diagnostics", pd.DataFrame())
    region_field_scores = detailed_outputs.get("region_field_scores", pd.DataFrame())
    observed_farm_summary = detailed_outputs.get("observed_farm_summary", pd.DataFrame())
    synthetic_farm_summary = detailed_outputs.get("synthetic_farm_summary", pd.DataFrame())
    farm_spatial_summary = detailed_outputs.get("farm_spatial_summary", pd.DataFrame())
    farm_corop_summary = detailed_outputs.get("farm_corop_summary", pd.DataFrame())
    dashboard_path = _write_sample_dashboard(per_snapshot, summary, outcome_summary, output_dir, sample_label)
    delta_path = _write_curve_delta_plot(per_snapshot, output_dir, sample_label)
    daily_mean_path = _write_daily_mean_comparison_plot(per_snapshot, summary, output_dir, sample_label)
    distribution_path = _write_outcome_distribution_plot(observed_scalar, synthetic_scalar, outcome_summary, output_dir, sample_label)
    parity_path = _write_sample_parity_plot(outcome_summary, output_dir, sample_label)
    channel_path = _write_channel_diagnostics_plot(per_snapshot, summary, output_dir, sample_label)
    farm_spatial_path = _write_farm_spatial_overview(farm_spatial_summary, summary, output_dir, sample_label)
    region_spatial_path = _write_region_spatial_overview(region_spatial_per_snapshot, output_dir, sample_label)

    region_geo_html_path = None
    if manifest is not None:
        region_payload = _prepare_region_geo_payload(
            manifest=manifest,
            sample_label=sample_label,
            observed_region_daily=observed_region_daily,
            synthetic_region_daily=synthetic_region_daily,
            corop_geojson_path=corop_geojson_path,
            focal_corop=focal_corop,
        )
        if region_payload is not None:
            region_geo_html_path = _write_region_geo_html(region_payload, output_dir / f"{sample_label}_region_geo_compare.html")

    per_snapshot.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    detail_paths: dict[str, str] = {}
    for key, frame in detailed_outputs.items():
        path = output_dir / f"{sample_label}_{key}.csv"
        frame.to_csv(path, index=False)
        detail_paths[key] = str(path)

    posterior_runs = int(pd.to_numeric(pd.Series([summary.get("posterior_num_runs", 1)]), errors="coerce").fillna(1).iloc[0])
    summary_scope_text = (
        f"This summary covers posterior discrepancy across {posterior_runs} generated panels from the same setting."
        if posterior_runs > 1
        else "This summary compares one generated panel against the observed hybrid temporal network."
    )
    lines = [
        f"# Hybrid transmission validation for {sample_label}",
        "",
        summary_scope_text,
        "",
        "## Configuration",
        "",
    ]
    for key, value in sorted((summary.get("simulation_config") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        _simulation_notes_markdown(
            "How to read the figures",
            _simulation_figure_reading_notes(),
        )
    )
    lines.extend(
        _simulation_notes_markdown(
            "How to read the tables",
            _simulation_table_reading_notes(),
        )
    )
    lines.extend(
        [
            "## Primary farm-outcome summary",
            "",
            f"- Replicates per panel: {int(summary.get('num_replicates', 0))}",
            f"- Farm prevalence-curve correlation: {float(summary.get('farm_prevalence_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm incidence-curve correlation: {float(summary.get('farm_incidence_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm prevalence mean-curve correlation: {float(summary.get('farm_prevalence_mean_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm incidence mean-curve correlation: {float(summary.get('farm_incidence_mean_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm attack-rate Wasserstein distance: {float(summary.get('farm_attack_rate_wasserstein', 0.0) or 0.0):.4f}",
            f"- Farm peak-prevalence Wasserstein distance: {float(summary.get('farm_peak_prevalence_wasserstein', 0.0) or 0.0):.4f}",
            f"- Farm peak-day Wasserstein distance: {float(summary.get('farm_peak_day_wasserstein', 0.0) or 0.0):.4f}",
            f"- Farm duration Wasserstein distance: {float(summary.get('farm_duration_wasserstein', 0.0) or 0.0):.4f}",
            f"- Mean absolute farm-prevalence delta: {float(summary.get('mean_abs_farm_prevalence_delta', 0.0) or 0.0):.4f}",
            f"- Mean absolute farm-incidence delta: {float(summary.get('mean_abs_farm_incidence_delta', 0.0) or 0.0):.4f}",
            f"- Mean absolute farm-prevalence daily-mean gap: {float(summary.get('farm_prevalence_mean_curve_mae', 0.0) or 0.0):.4f}",
            f"- Mean absolute farm-incidence daily-mean gap: {float(summary.get('farm_incidence_mean_curve_mae', 0.0) or 0.0):.4f}",
            "",
            "## Secondary checks",
            "",
            f"- Farm cumulative-incidence correlation: {float(summary.get('farm_cumulative_incidence_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm cumulative-incidence mean-curve correlation: {float(summary.get('farm_cumulative_incidence_mean_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Reservoir-total correlation: {float(summary.get('reservoir_total_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Reservoir-total mean-curve correlation: {float(summary.get('reservoir_total_mean_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Reservoir-max correlation: {float(summary.get('reservoir_max_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Reservoir-max mean-curve correlation: {float(summary.get('reservoir_max_mean_curve_correlation', 0.0) or 0.0):.4f}",
            f"- Farm cumulative-incidence Wasserstein distance: {float(summary.get('farm_cumulative_incidence_wasserstein', 0.0) or 0.0):.4f}",
            f"- Farm prevalence AUC Wasserstein distance: {float(summary.get('farm_prevalence_auc_wasserstein', 0.0) or 0.0):.4f}",
            f"- Reservoir-total AUC Wasserstein distance: {float(summary.get('reservoir_total_auc_wasserstein', 0.0) or 0.0):.4f}",
            f"- First-infection-day Wasserstein distance: {float(summary.get('farm_first_infection_day_wasserstein', 0.0) or 0.0):.4f}",
        ]
    )

    if not region_spatial_per_snapshot.empty or not region_temporal_summary.empty:
        lines.extend([
            "",
            "## Geospatial and spatio-temporal validation",
            "",
            "These summaries compare observed and synthetic region-level epidemic fields over COROP supernodes. In the current hybrid setup, the most informative mapped quantities are regional reservoir pressure, region-to-farm import pressure, and farm-to-region export seeding.",
            "",
            f"- Mean daily reservoir-pattern correlation across COROPs: {_metric_text(summary.get('region_reservoir_spatial_correlation_mean'))}",
            f"- Mean daily import-pressure correlation across COROPs: {_metric_text(summary.get('region_import_spatial_correlation_mean'))}",
            f"- Mean daily export-seeding correlation across COROPs: {_metric_text(summary.get('region_export_spatial_correlation_mean'))}",
            f"- Mean daily reservoir hotspot overlap (top 3 COROPs): {_metric_text(summary.get('region_reservoir_hotspot_overlap_mean'), digits=3)}",
            f"- Mean per-COROP reservoir temporal correlation: {_metric_text(summary.get('region_reservoir_temporal_correlation_mean'))}",
            f"- Mean per-COROP reservoir peak-day absolute shift: {_metric_text(summary.get('region_reservoir_peak_day_abs_delta_mean'))}",
            f"- Mean regional-field energy distance (reservoir): {_metric_text(summary.get('region_reservoir_field_energy_distance_mean'))}",
            f"- Mean regional-field variogram distance (reservoir): {_metric_text(summary.get('region_reservoir_field_variogram_distance_mean'))}",
        ])

    if not farm_spatial_summary.empty:
        lines.extend([
            "",
            "## Farm-level spatial validation",
            "",
            "These diagnostics check whether the synthetic network reproduces the farm-by-farm geographic pattern of network-generated infection risk rather than only the aggregate epidemic curve.",
            "",
            f"- Farm attack-probability correlation: {_metric_text(summary.get('farm_attack_probability_correlation'))}",
            f"- Farm attack-probability MAE: {_metric_text(summary.get('farm_attack_probability_mae'))}",
            f"- Farm attack-probability Wasserstein distance: {_metric_text(summary.get('farm_attack_probability_wasserstein'))}",
            f"- Farm top-10% hotspot overlap: {_metric_text(summary.get('farm_attack_probability_top10_overlap'), digits=3)}",
            f"- Farm Moran's I absolute delta: {_metric_text(summary.get('farm_attack_probability_moran_i_abs_delta'))}",
            f"- Farm first-infection-day correlation: {_metric_text(summary.get('farm_first_infection_day_correlation'))}",
            f"- COROP-mean farm attack-probability correlation: {_metric_text(summary.get('farm_corop_attack_probability_correlation'))}",
        ])

    if any(key in per_snapshot.columns for key in ["farm_hazard_ff_delta", "farm_hazard_rf_delta", "region_pressure_fr_delta", "region_pressure_rr_delta"]):
        lines.extend([
            "",
            "## Hybrid-channel diagnostics",
            "",
            "These checks quantify whether the synthetic panel preserves how much hazard moves through the farm→farm, region→farm, farm→region, and region→region channels in the hybrid network.",
            "",
            f"- Farm→farm hazard correlation: {_metric_text(summary.get('farm_hazard_ff_curve_correlation'))}",
            f"- Region→farm hazard correlation: {_metric_text(summary.get('farm_hazard_rf_curve_correlation'))}",
            f"- Farm→region seeding correlation: {_metric_text(summary.get('region_pressure_fr_curve_correlation'))}",
            f"- Region→region diffusion correlation: {_metric_text(summary.get('region_pressure_rr_curve_correlation'))}",
            f"- Region→farm hazard-share Wasserstein distance: {_metric_text(summary.get('farm_hazard_rf_share_wasserstein'))}",
            f"- Region→region pressure-share Wasserstein distance: {_metric_text(summary.get('region_pressure_rr_share_wasserstein'))}",
        ])

    if not trajectory_distribution_summary.empty or not lag_diagnostics.empty:
        lines.extend([
            "",
            "## Distribution-level temporal diagnostics",
            "",
            "These metrics compare the full replicate distribution of trajectories rather than only their medians. Energy distance measures overall ensemble mismatch; the lag diagnostics show whether synthetic curves line up better after a small phase shift.",
            "",
            f"- Farm prevalence trajectory energy distance: {_metric_text(summary.get('farm_prevalence_trajectory_energy_distance'))}",
            f"- Farm incidence trajectory energy distance: {_metric_text(summary.get('farm_incidence_trajectory_energy_distance'))}",
            f"- Reservoir-total trajectory energy distance: {_metric_text(summary.get('reservoir_total_trajectory_energy_distance'))}",
            f"- Best lagged farm-prevalence correlation: {_metric_text(summary.get('farm_prevalence_best_lag_correlation'))} at lag {_metric_text(summary.get('farm_prevalence_best_lag_days'), digits=0)} days",
            f"- Best lagged reservoir-total correlation: {_metric_text(summary.get('reservoir_total_best_lag_correlation'))} at lag {_metric_text(summary.get('reservoir_total_best_lag_days'), digits=0)} days",
        ])

    if not daily_mean_comparison.empty:
        lines.extend([
            "",
            "## Daily mean comparison",
            "",
            "These metrics compare the mean trajectory across replicates for each day. They are useful when daily medians stay flat for long stretches and hide smaller timing shifts.",
            "",
            "| Metric | Mean-curve corr | Mean abs daily gap | RMSE | Max abs daily gap |",
            "|---|---:|---:|---:|---:|",
        ])
        for _, row in daily_mean_comparison.iterrows():
            metric = str(row.get("metric", ""))
            corr = _metric_text(row.get("curve_correlation"), digits=3)
            mean_abs_delta = _metric_text(row.get("mean_abs_delta"), digits=3)
            rmse = _metric_text(row.get("rmse"), digits=3)
            max_abs_delta = _metric_text(row.get("max_abs_delta"), digits=3)
            lines.append(f"| {metric} | {corr} | {mean_abs_delta} | {rmse} | {max_abs_delta} |")

    if not daily_calibration.empty or not scalar_calibration.empty:
        lines.extend([
            "",
            "## Posterior-predictive calibration",
            "",
        ])
        if not daily_calibration.empty:
            lines.append("### Daily-curve coverage against synthetic 90% intervals")
            lines.append("")
            lines.append("| Metric | Coverage | Mean interval width | Mean absolute exceedance |")
            lines.append("|---|---:|---:|---:|")
            for _, row in daily_calibration.iterrows():
                metric = str(row.get("metric", ""))
                coverage = _metric_text(row.get("interval_coverage"), digits=3)
                width = _metric_text(row.get("mean_interval_width"), digits=3)
                exceed = _metric_text(row.get("mean_abs_exceedance"), digits=3)
                lines.append(f"| {metric} | {coverage} | {width} | {exceed} |")
        if not scalar_calibration.empty:
            lines.extend([
                "",
                "### Scalar-outcome calibration against synthetic 90% intervals",
                "",
                "| Metric | Observed median in synthetic 90% interval | Tail area | Interval width |",
                "|---|---:|---:|---:|",
            ])
            for _, row in scalar_calibration.iterrows():
                metric = str(row.get("metric", ""))
                in_interval = _metric_text(row.get("observed_median_in_synthetic_90pct"), digits=0)
                tail = _metric_text(row.get("observed_median_tail_area"), digits=3)
                width = _metric_text(row.get("synthetic_interval_width"), digits=3)
                lines.append(f"| {metric} | {in_interval} | {tail} | {width} |")

    if not uncertainty_decomposition.empty:
        lines.extend([
            "",
            "## Uncertainty decomposition across synthetic panels",
            "",
            "The table below separates between-panel variability (network uncertainty) from within-panel epidemic stochasticity using the repeated synthetic samples available for this setting.",
            "",
            "| Metric | Network uncertainty share | Epidemic stochasticity share | Observed median in pooled synthetic 90% interval | Pooled tail area |",
            "|---|---:|---:|---:|---:|",
        ])
        for _, row in uncertainty_decomposition.iterrows():
            metric = str(row.get("metric", ""))
            network_share = _metric_text(row.get("network_uncertainty_share"), digits=3)
            epidemic_share = _metric_text(row.get("epidemic_stochasticity_share"), digits=3)
            in_interval = _metric_text(row.get("observed_median_in_pooled_synthetic_90pct"), digits=0)
            tail = _metric_text(row.get("observed_median_pooled_tail_area"), digits=3)
            lines.append(f"| {metric} | {network_share} | {epidemic_share} | {in_interval} | {tail} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-day summary: `{csv_path.name}`",
            f"- Summary JSON: `{json_path.name}`",
        ]
    )
    if dashboard_path is not None:
        lines.append(f"- Dashboard PNG: `{Path(dashboard_path).name}`")
    if delta_path is not None:
        lines.append(f"- Delta PNG: `{Path(delta_path).name}`")
    if daily_mean_path is not None:
        lines.append(f"- Daily mean-comparison PNG: `{Path(daily_mean_path).name}`")
    if distribution_path is not None:
        lines.append(f"- Distribution PNG: `{Path(distribution_path).name}`")
    if parity_path is not None:
        lines.append(f"- Parity PNG: `{Path(parity_path).name}`")
    if channel_path is not None:
        lines.append(f"- Channel-diagnostics PNG: `{Path(channel_path).name}`")
    if farm_spatial_path is not None:
        lines.append(f"- Farm spatial-overview PNG: `{Path(farm_spatial_path).name}`")
    if region_spatial_path is not None:
        lines.append(f"- Regional spatial-overview PNG: `{Path(region_spatial_path).name}`")
    if region_geo_html_path is not None:
        lines.append(f"- Interactive COROP geo comparison: `{Path(region_geo_html_path).name}`")
    for key in sorted(detail_paths):
        lines.append(f"- {key}: `{Path(detail_paths[key]).name}`")
    md_path.write_text("\n".join(lines) + "\n")

    payload = {
        "per_snapshot_csv": str(csv_path),
        "summary_json": str(json_path),
        "report_md": str(md_path),
    }
    if dashboard_path is not None:
        payload["dashboard_png"] = str(dashboard_path)
    if delta_path is not None:
        payload["delta_png"] = str(delta_path)
    if daily_mean_path is not None:
        payload["daily_mean_png"] = str(daily_mean_path)
    if distribution_path is not None:
        payload["distribution_png"] = str(distribution_path)
    if parity_path is not None:
        payload["parity_png"] = str(parity_path)
    if channel_path is not None:
        payload["channel_png"] = str(channel_path)
    if farm_spatial_path is not None:
        payload["farm_spatial_png"] = str(farm_spatial_path)
    if region_spatial_path is not None:
        payload["region_spatial_png"] = str(region_spatial_path)
    if region_geo_html_path is not None:
        payload["region_geo_html"] = str(region_geo_html_path)
        payload["region_geo_payload_js"] = str(region_geo_html_path.with_name(f"{region_geo_html_path.stem}_payload.js"))
    payload.update(detail_paths)
    LOGGER.debug(
        "Simulation artifacts written | per_snapshot_csv=%s | summary_json=%s | report_md=%s | dashboard_png=%s | delta_png=%s | daily_mean_png=%s | distribution_png=%s | parity_png=%s | region_spatial_png=%s | region_geo_html=%s",
        csv_path,
        json_path,
        md_path,
        dashboard_path,
        delta_path,
        daily_mean_path,
        distribution_path,
        parity_path,
        region_spatial_path,
        region_geo_html_path,
    )
    return payload


def _summary_payload_to_row(label: str, payload: dict[str, object]) -> Optional[dict[str, object]]:
    if "farm_prevalence_curve_correlation" not in payload:
        return None
    keys = [
        "farm_prevalence_curve_correlation",
        "farm_incidence_curve_correlation",
        "farm_cumulative_incidence_curve_correlation",
        "reservoir_total_curve_correlation",
        "reservoir_max_curve_correlation",
        "reservoir_positive_regions_curve_correlation",
        "farm_hazard_ff_curve_correlation",
        "farm_hazard_rf_curve_correlation",
        "region_pressure_fr_curve_correlation",
        "region_pressure_rr_curve_correlation",
        "mean_abs_farm_prevalence_delta",
        "mean_abs_farm_incidence_delta",
        "mean_abs_farm_cumulative_incidence_delta",
        "mean_abs_reservoir_total_delta",
        "mean_abs_reservoir_max_delta",
        "mean_abs_farm_hazard_ff_delta",
        "mean_abs_farm_hazard_rf_delta",
        "mean_abs_region_pressure_fr_delta",
        "mean_abs_region_pressure_rr_delta",
        "farm_attack_rate_wasserstein",
        "farm_cumulative_incidence_wasserstein",
        "farm_peak_prevalence_wasserstein",
        "farm_peak_day_wasserstein",
        "farm_duration_wasserstein",
        "farm_first_infection_day_wasserstein",
        "farm_prevalence_auc_wasserstein",
        "reservoir_total_auc_wasserstein",
        "reservoir_max_peak_wasserstein",
        "farm_hazard_rf_share_wasserstein",
        "region_pressure_rr_share_wasserstein",
        "farm_prevalence_interval_coverage",
        "farm_incidence_interval_coverage",
        "farm_cumulative_incidence_interval_coverage",
        "reservoir_total_interval_coverage",
        "farm_prevalence_interval_width_mean",
        "farm_incidence_interval_width_mean",
        "farm_prevalence_mean_curve_correlation",
        "farm_incidence_mean_curve_correlation",
        "farm_cumulative_incidence_mean_curve_correlation",
        "reservoir_total_mean_curve_correlation",
        "reservoir_max_mean_curve_correlation",
        "reservoir_positive_regions_mean_curve_correlation",
        "farm_prevalence_mean_curve_mae",
        "farm_incidence_mean_curve_mae",
        "farm_cumulative_incidence_mean_curve_mae",
        "reservoir_total_mean_curve_mae",
        "reservoir_max_mean_curve_mae",
        "reservoir_positive_regions_mean_curve_mae",
        "farm_attack_rate_observed_median_in_synthetic_90pct",
        "farm_attack_rate_observed_median_tail_area",
        "farm_peak_prevalence_observed_median_in_synthetic_90pct",
        "farm_peak_prevalence_observed_median_tail_area",
        "farm_duration_observed_median_in_synthetic_90pct",
        "farm_duration_observed_median_tail_area",
        "farm_prevalence_best_lag_correlation",
        "farm_prevalence_best_lag_days",
        "farm_incidence_best_lag_correlation",
        "farm_incidence_best_lag_days",
        "reservoir_total_best_lag_correlation",
        "reservoir_total_best_lag_days",
        "farm_prevalence_trajectory_energy_distance",
        "farm_incidence_trajectory_energy_distance",
        "farm_cumulative_incidence_trajectory_energy_distance",
        "reservoir_total_trajectory_energy_distance",
        "reservoir_max_trajectory_energy_distance",
        "farm_attack_probability_correlation",
        "farm_attack_probability_mae",
        "farm_attack_probability_wasserstein",
        "farm_attack_probability_top10_overlap",
        "farm_attack_probability_moran_i_original",
        "farm_attack_probability_moran_i_synthetic",
        "farm_attack_probability_moran_i_abs_delta",
        "farm_first_infection_day_correlation",
        "farm_first_infection_day_mae",
        "farm_corop_attack_probability_correlation",
        "farm_corop_attack_probability_mae",
        "farm_attack_rate_network_uncertainty_share",
        "farm_attack_rate_epidemic_stochasticity_share",
        "farm_attack_rate_observed_median_in_pooled_synthetic_90pct",
        "farm_attack_rate_observed_median_pooled_tail_area",
        "farm_peak_prevalence_network_uncertainty_share",
        "farm_peak_prevalence_epidemic_stochasticity_share",
        "farm_peak_prevalence_observed_median_in_pooled_synthetic_90pct",
        "farm_peak_prevalence_observed_median_pooled_tail_area",
        "farm_duration_network_uncertainty_share",
        "farm_duration_epidemic_stochasticity_share",
        "farm_duration_observed_median_in_pooled_synthetic_90pct",
        "farm_duration_observed_median_pooled_tail_area",
        "farm_peak_day_network_uncertainty_share",
        "farm_peak_day_epidemic_stochasticity_share",
        "farm_peak_day_observed_median_in_pooled_synthetic_90pct",
        "farm_peak_day_observed_median_pooled_tail_area",
        "farm_cumulative_incidence_network_uncertainty_share",
        "farm_cumulative_incidence_epidemic_stochasticity_share",
        "farm_cumulative_incidence_observed_median_in_pooled_synthetic_90pct",
        "farm_cumulative_incidence_observed_median_pooled_tail_area",
        "region_reservoir_spatial_correlation_mean",
        "region_import_spatial_correlation_mean",
        "region_export_spatial_correlation_mean",
        "region_reservoir_temporal_correlation_mean",
        "region_import_temporal_correlation_mean",
        "region_export_temporal_correlation_mean",
        "region_reservoir_hotspot_overlap_mean",
        "region_import_hotspot_overlap_mean",
        "region_export_hotspot_overlap_mean",
        "region_reservoir_share_mae_mean",
        "region_reservoir_peak_day_abs_delta_mean",
        "region_reservoir_field_energy_distance_mean",
        "region_import_field_energy_distance_mean",
        "region_export_field_energy_distance_mean",
        "region_reservoir_field_variogram_distance_mean",
        "region_import_field_variogram_distance_mean",
        "region_export_field_variogram_distance_mean",
        "num_replicates",
        "posterior_num_runs",
    ]
    row = {"sample_label": label}
    for key in keys:
        if key in payload:
            row[key] = payload.get(key)
    if "sample_class" in payload:
        row["sample_class"] = payload.get("sample_class")
    return row


def _load_sweep_summary_rows(simulation_dir: Path) -> pd.DataFrame:
    simulation_dir = Path(simulation_dir)
    summary_frames: list[pd.DataFrame] = []
    for candidate in [simulation_dir / "setting_posterior_summary.csv", simulation_dir / "all_samples_summary.csv"]:
        if not candidate.exists():
            continue
        candidate_frame = pd.read_csv(candidate)
        if len(candidate_frame):
            summary_frames.append(candidate_frame)
    summary_json_rows: list[dict[str, object]] = []
    for path in sorted(simulation_dir.glob("*_summary.json")):
        label = path.name[: -len("_summary.json")]
        payload = _load_json_if_exists(path)
        if not isinstance(payload, dict):
            continue
        row = _summary_payload_to_row(label, payload)
        if row is not None:
            summary_json_rows.append(row)
    summary_json_frame = pd.DataFrame(summary_json_rows)
    if len(summary_json_frame):
        summary_frames.append(summary_json_frame)
    if summary_frames:
        summary_frame = pd.concat(summary_frames, ignore_index=True, sort=False)
    else:
        summary_frame = pd.DataFrame()
    if summary_frame.empty:
        raise FileNotFoundError(f"No summary CSV or per-setting summary JSON available under {simulation_dir}")
    if "sample_label" in summary_frame.columns:
        summary_frame = summary_frame.drop_duplicates(subset=["sample_label"]).reset_index(drop=True)
    return summary_frame





def _scenario_switch_items_from_summary_rows(
    summary_rows: pd.DataFrame,
    *,
    scenario_root: Path,
    current_report_path: Path,
) -> list[dict[str, object]]:
    if summary_rows.empty or "scenario_name" not in summary_rows.columns:
        return []
    scenario_root = Path(scenario_root).expanduser().resolve()
    current_report_path = Path(current_report_path).expanduser().resolve()
    start_dir = current_report_path.parent
    items: list[dict[str, object]] = []
    for _, row in summary_rows.iterrows():
        scenario_name = str(row.get("scenario_name") or "").strip()
        if not scenario_name:
            continue
        report_value = str(row.get("report_path") or "").strip()
        if report_value:
            report_path = Path(report_value).expanduser()
            report_path = report_path if report_path.is_absolute() else (scenario_root / report_path)
        else:
            report_path = scenario_root / scenario_name / "scientific_validation_report.html"
        report_path = report_path.resolve()
        fallback_path = (scenario_root / scenario_name / "scientific_validation_report.html").resolve()
        if not report_path.exists() and fallback_path.exists():
            report_path = fallback_path
        href = os.path.relpath(report_path, start=start_dir).replace(os.sep, "/")
        item: dict[str, object] = {
            "scenario_name": scenario_name,
            "scenario_description": str(row.get("scenario_description") or "").strip(),
            "model": str(row.get("model") or "").strip(),
            "selected_setting_label": str(row.get("selected_setting_label") or row.get("selected_sample_label") or "").strip(),
            "href": href,
        }
        for key in (
            "farm_prevalence_curve_correlation",
            "farm_incidence_curve_correlation",
            "farm_attack_rate_wasserstein",
            "farm_peak_prevalence_wasserstein",
            "farm_duration_wasserstein",
            "farm_attack_probability_correlation",
            "region_reservoir_spatial_correlation_mean",
            "farm_prevalence_interval_coverage",
            "farm_attack_rate_network_uncertainty_share",
        ):
            if key in row.index:
                value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
                item[key] = None if pd.isna(value) else float(value)
        items.append(item)
    return items


def _build_scenario_switch_payload(
    summary_rows: pd.DataFrame,
    *,
    scenario_root: Path,
    current_report_path: Path,
    current_scenario_name: Optional[str] = None,
    default_scenario_name: Optional[str] = None,
    comparison_report_path: Optional[Path] = None,
) -> Optional[dict[str, object]]:
    items = _scenario_switch_items_from_summary_rows(
        summary_rows,
        scenario_root=Path(scenario_root),
        current_report_path=Path(current_report_path),
    )
    if len(items) <= 1:
        return None
    scenario_names = [str(item.get("scenario_name")) for item in items]
    current_name = str(current_scenario_name).strip() if current_scenario_name else ""
    if current_name not in scenario_names:
        inferred_name = Path(current_report_path).expanduser().resolve().parent.name
        current_name = inferred_name if inferred_name in scenario_names else ""
    default_name = str(default_scenario_name).strip() if default_scenario_name else ""
    if default_name not in scenario_names:
        default_name = current_name or scenario_names[0]
    comparison_href = None
    if comparison_report_path is None:
        comparison_report_path = Path(scenario_root).expanduser().resolve() / "scientific_validation_report.html"
    comparison_report_path = Path(comparison_report_path).expanduser().resolve()
    if comparison_report_path != Path(current_report_path).expanduser().resolve():
        comparison_href = os.path.relpath(comparison_report_path, start=Path(current_report_path).expanduser().resolve().parent).replace(os.sep, "/")
    return {
        "items": items,
        "current_scenario_name": current_name or None,
        "default_scenario_name": default_name,
        "comparison_href": comparison_href,
    }


def _auto_discover_scenario_switch_payload(
    *,
    run_dir: Path,
    simulation_dir: Path,
    current_report_path: Path,
) -> Optional[dict[str, object]]:
    run_dir = Path(run_dir).expanduser().resolve()
    simulation_dir = Path(simulation_dir).expanduser().resolve()
    current_report_path = Path(current_report_path).expanduser().resolve()
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {}
    candidate_roots: list[Path] = []
    simulation_scenarios = manifest.get("simulation_scenarios") or {}
    scenario_root_value = simulation_scenarios.get("output_dir")
    if scenario_root_value:
        candidate_roots.append(Path(str(scenario_root_value)).expanduser().resolve())
    candidate_roots.extend([simulation_dir, simulation_dir.parent])
    seen_roots: set[Path] = set()
    for candidate_root in candidate_roots:
        if candidate_root in seen_roots:
            continue
        seen_roots.add(candidate_root)
        summary_csv = candidate_root / "scenario_summary.csv"
        if not summary_csv.exists():
            continue
        try:
            summary_rows = pd.read_csv(summary_csv)
        except Exception:
            continue
        default_name = str(simulation_scenarios.get("best_scenario") or "").strip() or None
        current_name = simulation_dir.name if "scenario_name" in summary_rows.columns and simulation_dir.name in summary_rows["scenario_name"].astype(str).tolist() else None
        payload = _build_scenario_switch_payload(
            summary_rows,
            scenario_root=candidate_root,
            current_report_path=current_report_path,
            current_scenario_name=current_name,
            default_scenario_name=default_name,
            comparison_report_path=candidate_root / "scientific_validation_report.html",
        )
        if payload is not None:
            return payload
    return None


def _scenario_navigation_css() -> str:
    return """
        .scenario-switch-card, .scenario-browser-card, .scenario-iframe-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 18px;
          box-shadow: var(--shadow);
        }
        .scenario-switch-controls {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-items: end;
          margin-top: 12px;
        }
        .scenario-select-label {
          display: flex;
          flex-direction: column;
          gap: 6px;
          color: var(--muted);
          font-size: 13px;
          min-width: min(320px, 100%);
        }
        .scenario-select {
          min-width: min(320px, 100%);
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--line);
          background: #fbfdff;
          color: var(--text);
          font: inherit;
        }
        .scenario-link-button {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 10px 14px;
          border-radius: 10px;
          border: 1px solid rgba(42, 111, 187, 0.22);
          background: var(--blue-soft);
          color: var(--blue);
          text-decoration: none;
          font-weight: 700;
          min-height: 42px;
        }
        .scenario-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 14px;
        }
        .scenario-chip {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: #fbfdff;
          color: var(--text);
          text-decoration: none;
          font-size: 13px;
          font-weight: 600;
          transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }
        .scenario-chip:hover {
          background: var(--blue-soft);
          border-color: rgba(42, 111, 187, 0.24);
        }
        .scenario-chip.is-active {
          background: var(--blue-soft);
          border-color: rgba(42, 111, 187, 0.32);
          color: var(--blue);
        }
        .scenario-current-summary {
          margin: 14px 0 0;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.6;
        }
        .scenario-browser-grid {
          display: grid;
          gap: 16px;
        }
        .scenario-iframe-card {
          padding: 12px;
        }
        .scenario-report-frame {
          width: 100%;
          height: 86vh;
          min-height: 980px;
          border: 1px solid var(--line);
          border-radius: 14px;
          background: #ffffff;
        }
        @media (max-width: 980px) {
          .scenario-switch-controls {
            flex-direction: column;
            align-items: stretch;
          }
          .scenario-link-button {
            width: 100%;
          }
          .scenario-report-frame {
            height: 78vh;
            min-height: 720px;
          }
        }
    """


def _scenario_summary_sentence(item: dict[str, object]) -> str:
    segments: list[str] = []
    model = str(item.get("model") or "").strip()
    selected_setting = str(item.get("selected_setting_label") or "").strip()
    description = str(item.get("scenario_description") or "").strip()
    header_bits: list[str] = []
    if model:
        header_bits.append(f"Model {model}")
    if selected_setting:
        header_bits.append(f"Selected setting {selected_setting}")
    if description:
        header_bits.append(description.rstrip("."))
    if header_bits:
        segments.append(" · ".join(header_bits))
    metric_bits: list[str] = []
    for label, key in (
        ("prev corr", "farm_prevalence_curve_correlation"),
        ("inc corr", "farm_incidence_curve_correlation"),
        ("attack W1", "farm_attack_rate_wasserstein"),
        ("duration W1", "farm_duration_wasserstein"),
        ("farm risk corr", "farm_attack_probability_correlation"),
        ("region spatial corr", "region_reservoir_spatial_correlation_mean"),
    ):
        value = pd.to_numeric(pd.Series([item.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value):
            metric_bits.append(f"{label} {float(value):.3f}")
    if metric_bits:
        segments.append(" · ".join(metric_bits))
    return " — ".join(segments)


def _render_scenario_navigation_section(payload: Optional[dict[str, object]]) -> str:
    if not payload:
        return ""
    items = list(payload.get("items") or [])
    if len(items) <= 1:
        return ""
    current_name = str(payload.get("current_scenario_name") or payload.get("default_scenario_name") or items[0].get("scenario_name"))
    current_item = next((item for item in items if str(item.get("scenario_name")) == current_name), items[0])
    options_html = "".join(
        f"<option value='{html.escape(str(item.get('href') or ''))}'{' selected' if str(item.get('scenario_name')) == current_name else ''}>{html.escape(str(item.get('scenario_name') or ''))}</option>"
        for item in items
    )
    chips_html = "".join(
        f"<a class='scenario-chip{' is-active' if str(item.get('scenario_name')) == current_name else ''}' href='{html.escape(str(item.get('href') or ''))}'>{html.escape(str(item.get('scenario_name') or ''))}</a>"
        for item in items
    )
    comparison_href = str(payload.get("comparison_href") or "").strip()
    comparison_link_html = (
        f"<a class='scenario-link-button' href='{html.escape(comparison_href)}'>Open scenario comparison report</a>"
        if comparison_href else ""
    )
    summary_text = html.escape(_scenario_summary_sentence(current_item))
    return (
        "<section class='section' id='scenario-switch'>"
        "<div class='scenario-switch-card'>"
        "<h2>Scenario switch</h2>"
        "<p class='subtitle'>This report is one member of a multi-scenario sweep. Use the switch below to jump to the full report for any scenario while keeping the same tables and visualizations.</p>"
        "<div class='scenario-switch-controls'>"
        "<label class='scenario-select-label' for='scenario_switch_select'>"
        "<span>Scenario</span>"
        f"<select id='scenario_switch_select' class='scenario-select' onchange='if(this.value) window.location.href=this.value'>{options_html}</select>"
        "</label>"
        f"{comparison_link_html}"
        "</div>"
        f"<div class='scenario-chip-row'>{chips_html}</div>"
        f"<p class='scenario-current-summary'>{summary_text}</p>"
        "</div>"
        "</section>"
    )


def _render_scenario_browser_section(
    payload: Optional[dict[str, object]],
    *,
    widget_id: str = "scenario_browser",
) -> str:
    if not payload:
        return ""
    items = list(payload.get("items") or [])
    if len(items) <= 1:
        return ""
    default_name = str(payload.get("default_scenario_name") or items[0].get("scenario_name"))
    default_item = next((item for item in items if str(item.get("scenario_name")) == default_name), items[0])
    options_html = "".join(
        f"<option value='{html.escape(str(item.get('scenario_name') or ''))}'{' selected' if str(item.get('scenario_name')) == default_name else ''}>{html.escape(str(item.get('scenario_name') or ''))}</option>"
        for item in items
    )
    chips_html = "".join(
        f"<a class='scenario-chip{' is-active' if str(item.get('scenario_name')) == default_name else ''}' href='{html.escape(str(item.get('href') or ''))}' data-scenario-browser-chip='{html.escape(str(item.get('scenario_name') or ''))}'>{html.escape(str(item.get('scenario_name') or ''))}</a>"
        for item in items
    )
    summary_text = html.escape(_scenario_summary_sentence(default_item))
    items_json = json.dumps(_json_ready(items), ensure_ascii=False).replace("</", "<\\/")
    default_href = html.escape(str(default_item.get("href") or ""))
    return (
        "<section class='section' id='scenario-browser'>"
        "<div class='scenario-browser-grid'>"
        "<div class='scenario-browser-card'>"
        "<h2>Scenario browser</h2>"
        "<p class='subtitle'>Use this switch to load the full report for any scenario directly inside this page. Each scenario opens the same detailed tables and visualizations that are available in its standalone validation report.</p>"
        "<div class='scenario-switch-controls'>"
        f"<label class='scenario-select-label' for='{html.escape(widget_id)}_select'>"
        "<span>Scenario</span>"
        f"<select id='{html.escape(widget_id)}_select' class='scenario-select'>{options_html}</select>"
        "</label>"
        f"<a id='{html.escape(widget_id)}_open' class='scenario-link-button' href='{default_href}' target='_blank' rel='noopener'>Open selected scenario in a new tab</a>"
        "</div>"
        f"<div class='scenario-chip-row'>{chips_html}</div>"
        f"<p id='{html.escape(widget_id)}_summary' class='scenario-current-summary'>{summary_text}</p>"
        "</div>"
        "<div class='scenario-iframe-card'>"
        f"<iframe id='{html.escape(widget_id)}_frame' class='scenario-report-frame' src='{default_href}' title='Scenario validation report' loading='lazy'></iframe>"
        "</div>"
        f"<script type='application/json' id='{html.escape(widget_id)}_data'>{items_json}</script>"
        "</div>"
        "</section>"
    )


def _scenario_browser_script(widget_id: str = "scenario_browser") -> str:
    escaped_widget_id = json.dumps(str(widget_id))
    return """
            (function() {
              const widgetId = """ + escaped_widget_id + """;
              const dataNode = document.getElementById(widgetId + "_data");
              const select = document.getElementById(widgetId + "_select");
              const frame = document.getElementById(widgetId + "_frame");
              const openLink = document.getElementById(widgetId + "_open");
              const summary = document.getElementById(widgetId + "_summary");
              if (!dataNode || !select || !frame || !openLink || !summary) return;
              let items = [];
              try {
                items = JSON.parse(dataNode.textContent || "[]");
              } catch (error) {
                return;
              }
              if (!Array.isArray(items) || !items.length) return;
              const chips = Array.from(document.querySelectorAll("[data-scenario-browser-chip]"));
              const formatItem = (item) => {
                const parts = [];
                if (item.model) parts.push(`Model ${item.model}`);
                if (item.selected_setting_label) parts.push(`Selected setting ${item.selected_setting_label}`);
                if (item.scenario_description) parts.push(String(item.scenario_description).replace(/\\.$/, ""));
                const metrics = [];
                const metricSpecs = [
                  ["prev corr", "farm_prevalence_curve_correlation"],
                  ["inc corr", "farm_incidence_curve_correlation"],
                  ["attack W1", "farm_attack_rate_wasserstein"],
                  ["duration W1", "farm_duration_wasserstein"],
                  ["farm risk corr", "farm_attack_probability_correlation"],
                  ["region spatial corr", "region_reservoir_spatial_correlation_mean"],
                ];
                metricSpecs.forEach(([label, key]) => {
                  const value = Number(item[key]);
                  if (Number.isFinite(value)) metrics.push(`${label} ${value.toFixed(3)}`);
                });
                return [parts.join(" · "), metrics.join(" · ")].filter(Boolean).join(" — ");
              };
              const activate = (scenarioName) => {
                const nextItem = items.find((item) => String(item.scenario_name) === String(scenarioName)) || items[0];
                if (!nextItem) return;
                select.value = String(nextItem.scenario_name);
                if (nextItem.href) {
                  frame.src = String(nextItem.href);
                  openLink.href = String(nextItem.href);
                }
                summary.textContent = formatItem(nextItem);
                chips.forEach((chip) => {
                  const active = chip.dataset.scenarioBrowserChip === String(nextItem.scenario_name);
                  chip.classList.toggle("is-active", active);
                });
              };
              select.addEventListener("change", () => activate(select.value));
              chips.forEach((chip) => {
                chip.addEventListener("click", (event) => {
                  event.preventDefault();
                  activate(chip.dataset.scenarioBrowserChip);
                });
              });
              activate(select.value || String(items[0].scenario_name));
            })();
    """

def write_scientific_validation_report(
    run_dir: Path,
    *,
    simulation_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    scenario_switch_payload: Optional[dict[str, object]] = None,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    simulation_dir = Path(simulation_dir).expanduser().resolve() if simulation_dir is not None else run_dir / "simulation"
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {}
    summary_rows = _load_sweep_summary_rows(simulation_dir)

    dataset_name = str(manifest.get("dataset", "Dataset"))
    directed_label = "Directed " if bool(manifest.get("directed", False)) else ""
    title = title or f"{dataset_name} {directed_label}Hybrid Transmission Reality-check Report"
    output_path = Path(output_path) if output_path is not None else simulation_dir / "scientific_validation_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_switch_payload = scenario_switch_payload or _auto_discover_scenario_switch_payload(
        run_dir=run_dir,
        simulation_dir=simulation_dir,
        current_report_path=output_path,
    )
    scenario_navigation_html = _render_scenario_navigation_section(scenario_switch_payload)

    sample_labels = summary_rows["sample_label"].astype(str) if "sample_label" in summary_rows.columns else pd.Series([""] * len(summary_rows))
    posterior_runs_source = summary_rows["posterior_num_runs"] if "posterior_num_runs" in summary_rows.columns else pd.Series(1.0, index=summary_rows.index)
    posterior_runs = pd.to_numeric(posterior_runs_source, errors="coerce").fillna(1.0)
    aggregate_mask = (posterior_runs > 1.0) | ~sample_labels.str.contains("__sample_", regex=False)
    aggregate_rows = summary_rows.loc[aggregate_mask].reset_index(drop=True)
    detail_rows = summary_rows.loc[~aggregate_mask].reset_index(drop=True)

    headline_source = aggregate_rows if not aggregate_rows.empty else summary_rows.reset_index(drop=True)
    overview_path = write_all_samples_overview(headline_source, simulation_dir)
    if "sample_class" in headline_source.columns:
        headline_rows = headline_source.loc[headline_source["sample_class"].astype(str) == "posterior_predictive"].reset_index(drop=True)
        if headline_rows.empty:
            headline_rows = headline_source.reset_index(drop=True)
    else:
        headline_rows = headline_source.reset_index(drop=True)

    def slugify(value: object) -> str:
        text_value = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
        return text_value or "item"

    def safe_best_row(frame: pd.DataFrame, sort_spec: list[tuple[str, bool]]) -> pd.Series:
        if frame.empty:
            raise ValueError("Cannot select a best row from an empty summary frame.")
        available = [(column, ascending) for column, ascending in sort_spec if column in frame.columns]
        if not available:
            return frame.iloc[0]
        ranked = frame.sort_values(
            [column for column, _ in available],
            ascending=[ascending for _, ascending in available],
            na_position="last",
        ).reset_index(drop=True)
        return ranked.iloc[0]

    def sort_frame_by_available(frame: pd.DataFrame, sort_spec: list[tuple[str, bool]]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        available = [(column, ascending) for column, ascending in sort_spec if column in frame.columns]
        if not available:
            if "sample_label" in frame.columns:
                return frame.sort_values("sample_label", na_position="last").reset_index(drop=True)
            return frame.reset_index(drop=True)
        return frame.sort_values(
            [column for column, _ in available],
            ascending=[ascending for _, ascending in available],
            na_position="last",
        ).reset_index(drop=True)

    best_curve = safe_best_row(
        headline_rows,
        [
            ("farm_prevalence_curve_correlation", False),
            ("farm_incidence_curve_correlation", False),
            ("farm_attack_rate_wasserstein", True),
        ],
    )
    lowest_attack = safe_best_row(
        headline_rows,
        [
            ("farm_attack_rate_wasserstein", True),
            ("farm_peak_prevalence_wasserstein", True),
            ("farm_duration_wasserstein", True),
        ],
    )
    best_farm_spatial = safe_best_row(
        headline_rows,
        [
            ("farm_attack_probability_correlation", False),
            ("farm_attack_probability_mae", True),
            ("farm_attack_probability_top10_overlap", False),
        ],
    )
    best_regional = safe_best_row(
        headline_rows,
        [
            ("region_reservoir_spatial_correlation_mean", False),
            ("region_reservoir_temporal_correlation_mean", False),
            ("region_reservoir_field_energy_distance_mean", True),
        ],
    )

    def figure_tag(path: Optional[Path], caption: str, *, title_text: Optional[str] = None) -> str:
        if path is None or not Path(path).exists():
            return ""
        heading_html = f"<h3>{html.escape(title_text)}</h3>" if title_text else ""
        filename = html.escape(Path(path).name)
        overlay_title = html.escape(title_text or caption)
        overlay_caption = html.escape(caption)
        return (
            "<figure class='figure-card'>"
            f"{heading_html}"
            "<div class='figure-frame'>"
            f"<button type='button' class='figure-popout-button' data-image-src='{filename}' data-image-title='{overlay_title}' data-image-caption='{overlay_caption}' aria-label='Open full-size figure'>"
            f"<img class='report-zoomable-image' src='{filename}' alt='{overlay_caption}' loading='lazy' decoding='async'>"
            "</button>"
            "</div>"
            f"<figcaption>{overlay_caption}</figcaption>"
            "</figure>"
        )

    def link_figure_card(
        path: Optional[Path],
        title_text: str,
        description: str,
        *,
        link_label: str = "Open interactive viewer",
    ) -> str:
        if path is None or not Path(path).exists():
            return ""
        filename = html.escape(Path(path).name)
        return (
            "<figure class='figure-card link-card'>"
            "<div class='link-card-inner'>"
            f"<h3>{html.escape(title_text)}</h3>"
            f"<p class='table-note'>{html.escape(description)}</p>"
            f"<p class='card-link'><a href='{filename}' target='_blank' rel='noopener'>{html.escape(link_label)}</a></p>"
            "</div>"
            "</figure>"
        )

    def artifact_link(path: Optional[Path], label: str) -> str:
        if path is None or not Path(path).exists():
            return ""
        filename = html.escape(Path(path).name)
        text_value = html.escape(label)
        return f"<li><a href='{filename}' target='_blank' rel='noopener'>{text_value}</a></li>"

    def artifact_path(sample_label: object, suffix: str, extension: str = "png") -> Optional[Path]:
        candidate = simulation_dir / f"{sample_label}_{suffix}.{extension}"
        return candidate if candidate.exists() else None

    def render_dataframe_table(frame: pd.DataFrame, *, max_rows: Optional[int] = None) -> str:
        if frame.empty:
            return ""
        display_frame = frame.copy()
        if max_rows is not None and len(display_frame) > max_rows:
            display_frame = display_frame.head(max_rows).copy()
        for column in display_frame.columns:
            numeric = pd.to_numeric(display_frame[column], errors="coerce")
            if numeric.notna().any():
                finite = numeric.dropna().to_numpy(dtype=float)
                if len(finite) and np.all(np.isclose(finite, np.round(finite))):
                    display_frame[column] = [str(int(round(float(value)))) if pd.notna(value) else "" for value in numeric]
                else:
                    display_frame[column] = [f"{float(value):.4f}" if pd.notna(value) else "" for value in numeric]
        return display_frame.to_html(index=False, classes=["report-table"], border=0, escape=False)

    def table_card_from_csv(
        path: Optional[Path],
        title_text: str,
        *,
        note: Optional[str] = None,
        max_rows: Optional[int] = None,
        transform=None,
    ) -> str:
        if path is None or not Path(path).exists():
            return ""
        try:
            frame = pd.read_csv(path)
        except Exception:
            return ""
        if transform is not None:
            try:
                frame = transform(frame)
            except Exception:
                pass
        table_html = render_dataframe_table(frame, max_rows=max_rows)
        if not table_html:
            return ""
        note_html = f"<p class='table-note'>{html.escape(note)}</p>" if note else ""
        truncated_html = ""
        if max_rows is not None and len(frame) > max_rows:
            truncated_html = (
                f"<p class='table-note'>Showing the first {int(max_rows)} rows. "
                "The full CSV is linked in the artifact list.</p>"
            )
        return (
            "<div class='table-card'>"
            f"<h3>{html.escape(title_text)}</h3>"
            f"{note_html}"
            "<div class='table-wrap'>"
            f"{table_html}"
            "</div>"
            f"{truncated_html}"
            "</div>"
        )

    def table_card_from_html(
        table_html: str,
        *,
        note: Optional[str] = None,
        title_text: Optional[str] = None,
    ) -> str:
        if not table_html:
            return ""
        note_html = f"<p class='table-note'>{html.escape(note)}</p>" if note else ""
        title_html = f"<h3>{html.escape(title_text)}</h3>" if title_text else ""
        return (
            "<div class='table-card'>"
            f"{title_html}"
            f"{note_html}"
            "<div class='table-wrap'>"
            f"{table_html}"
            "</div>"
            "</div>"
        )

    def kv_table_card(title_text: str, entries: list[tuple[str, str]], *, note: Optional[str] = None) -> str:
        clean_entries = [(str(label), str(value)) for label, value in entries if str(value).strip() and str(value).strip().lower() != "n/a"]
        if not clean_entries:
            return ""
        rows_html = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
            for label, value in clean_entries
        )
        return table_card_from_html(
            "<table class='report-table compact-table'><tbody>" + rows_html + "</tbody></table>",
            note=note,
            title_text=title_text,
        )

    def status_icon(present: bool) -> str:
        return "<span class='status-ok'>✓</span>" if present else "<span class='status-missing'>—</span>"

    def fmt_from_row(row: pd.Series, key: str, digits: int = 3) -> str:
        if key not in row.index:
            return "n/a"
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    def row_mean_text(frame: pd.DataFrame, columns: list[str]) -> str:
        available = [column for column in columns if column in frame.columns]
        if not available:
            return "n/a"
        numeric = frame[available].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean(axis=1, skipna=True).dropna()
        if means.empty:
            return "n/a"
        return f"{float(means.median()):.3f}"

    def section_id_for_sample(sample_label: object) -> str:
        return f"setting-{slugify(sample_label)}"

    def transform_farm_spatial(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "network_generated_attack_probability_delta" in working.columns:
            delta_series = pd.to_numeric(working["network_generated_attack_probability_delta"], errors="coerce")
            working["_abs_delta"] = delta_series.abs()
            working = working.sort_values("_abs_delta", ascending=False, na_position="last")
        keep_columns = [
            "node_id",
            "ubn",
            "corop",
            "x",
            "y",
            "original_network_generated_attack_probability",
            "synthetic_network_generated_attack_probability",
            "network_generated_attack_probability_delta",
            "original_first_infection_day_median",
            "synthetic_first_infection_day_median",
            "first_infection_day_median_delta",
        ]
        keep_columns = [column for column in keep_columns if column in working.columns]
        if keep_columns:
            working = working[keep_columns]
        return working.drop(columns=["_abs_delta"], errors="ignore").reset_index(drop=True)

    def transform_farm_corop(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if {"original_network_generated_attack_probability", "synthetic_network_generated_attack_probability"}.issubset(working.columns):
            original = pd.to_numeric(working["original_network_generated_attack_probability"], errors="coerce")
            synthetic = pd.to_numeric(working["synthetic_network_generated_attack_probability"], errors="coerce")
            working["_abs_delta"] = (synthetic - original).abs()
            working = working.sort_values(["_abs_delta", "original_network_generated_attack_probability"], ascending=[False, False], na_position="last")
        return working.drop(columns=["_abs_delta"], errors="ignore").reset_index(drop=True)

    def transform_region_field(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        keep_columns = [
            "day_index",
            "ts",
            "reservoir_pressure_energy_distance",
            "import_pressure_energy_distance",
            "export_pressure_energy_distance",
            "reservoir_pressure_variogram_distance",
            "import_pressure_variogram_distance",
            "export_pressure_variogram_distance",
        ]
        keep_columns = [column for column in keep_columns if column in working.columns]
        if keep_columns:
            working = working[keep_columns]
        sort_columns = [column for column in ["day_index", "ts"] if column in working.columns]
        if sort_columns:
            working = working.sort_values(sort_columns)
        return working.reset_index(drop=True)

    def transform_region_temporal(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        sort_columns = []
        ascending = []
        if "reservoir_pressure_temporal_correlation" in working.columns:
            sort_columns.append("reservoir_pressure_temporal_correlation")
            ascending.append(False)
        if "import_pressure_temporal_correlation" in working.columns:
            sort_columns.append("import_pressure_temporal_correlation")
            ascending.append(False)
        if sort_columns:
            working = working.sort_values(sort_columns, ascending=ascending, na_position="last")
        keep_columns = [
            "region_node_id",
            "corop",
            "display_label",
            "reservoir_pressure_temporal_correlation",
            "import_pressure_temporal_correlation",
            "export_pressure_temporal_correlation",
            "reservoir_pressure_peak_day_abs_delta",
            "import_pressure_peak_day_abs_delta",
            "export_pressure_peak_day_abs_delta",
        ]
        keep_columns = [column for column in keep_columns if column in working.columns]
        if keep_columns:
            working = working[keep_columns]
        return working.reset_index(drop=True)

    def coverage_matrix_html(frame: pd.DataFrame) -> tuple[str, Optional[float]]:
        if frame.empty:
            return "", None
        expected_assets = [
            ("dashboard", "Dashboard", "png"),
            ("delta", "Delta", "png"),
            ("daily_mean_compare", "Daily mean fig", "png"),
            ("distribution", "Distribution fig", "png"),
            ("parity", "Parity", "png"),
            ("channel_diagnostics", "Channel", "png"),
            ("farm_spatial_overview", "Farm spatial fig", "png"),
            ("region_spatial_overview", "Region spatial fig", "png"),
            ("region_geo_compare", "Interactive map", "html"),
            ("outcome_distribution_summary", "Outcome table", "csv"),
            ("trajectory_distribution_summary", "Trajectory table", "csv"),
            ("daily_mean_comparison", "Daily mean table", "csv"),
            ("daily_calibration", "Daily calib", "csv"),
            ("scalar_calibration", "Scalar calib", "csv"),
            ("uncertainty_decomposition", "Uncertainty", "csv"),
            ("lag_diagnostics", "Lag table", "csv"),
            ("farm_spatial_summary", "Farm spatial table", "csv"),
            ("farm_corop_summary", "Farm COROP table", "csv"),
            ("region_field_scores", "Region field table", "csv"),
            ("region_temporal_summary", "Region temporal table", "csv"),
        ]
        headers = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in expected_assets)
        body_rows = []
        completeness_values: list[float] = []
        for _, row in frame.iterrows():
            sample_label = str(row.get("sample_label", ""))
            cells = [f"<td>{html.escape(sample_label)}</td>"]
            present_count = 0
            for suffix, _, extension in expected_assets:
                present = artifact_path(sample_label, suffix, extension) is not None
                present_count += int(present)
                cells.append(f"<td class='status-cell'>{status_icon(present)}</td>")
            completeness = float(present_count / max(len(expected_assets), 1))
            completeness_values.append(completeness)
            cells.append(f"<td>{completeness * 100.0:.0f}%</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        table_html = (
            "<table class='report-table coverage-table'>"
            "<thead><tr><th>Setting</th>"
            + headers
            + "<th>Completeness</th></tr></thead>"
            + "<tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
        )
        median_completeness = float(np.median(completeness_values)) if completeness_values else None
        return table_html, median_completeness

    def setting_visual_section(row: pd.Series, role_label: str) -> str:
        sample_label = str(row["sample_label"])
        section_id = section_id_for_sample(sample_label)
        dashboard_path = artifact_path(sample_label, "dashboard")
        delta_path = artifact_path(sample_label, "delta")
        daily_mean_path = artifact_path(sample_label, "daily_mean_compare")
        distribution_path = artifact_path(sample_label, "distribution")
        parity_path = artifact_path(sample_label, "parity")
        channel_path = artifact_path(sample_label, "channel_diagnostics")
        farm_spatial_path = artifact_path(sample_label, "farm_spatial_overview")
        region_spatial_path = artifact_path(sample_label, "region_spatial_overview")
        region_geo_path = artifact_path(sample_label, "region_geo_compare", extension="html")
        report_md_path = artifact_path(sample_label, "report", extension="md")
        summary_json_path = artifact_path(sample_label, "summary", extension="json")
        per_snapshot_csv_path = artifact_path(sample_label, "per_snapshot", extension="csv")
        outcome_distribution_csv_path = artifact_path(sample_label, "outcome_distribution_summary", extension="csv")
        daily_mean_comparison_path = artifact_path(sample_label, "daily_mean_comparison", extension="csv")
        daily_calibration_path = artifact_path(sample_label, "daily_calibration", extension="csv")
        scalar_calibration_path = artifact_path(sample_label, "scalar_calibration", extension="csv")
        uncertainty_decomposition_path = artifact_path(sample_label, "uncertainty_decomposition", extension="csv")
        observed_daily_path = artifact_path(sample_label, "observed_daily", extension="csv")
        synthetic_daily_path = artifact_path(sample_label, "synthetic_daily", extension="csv")
        observed_outcomes_path = artifact_path(sample_label, "observed_outcomes", extension="csv")
        synthetic_outcomes_path = artifact_path(sample_label, "synthetic_outcomes", extension="csv")
        observed_region_daily_path = artifact_path(sample_label, "observed_region_daily", extension="csv")
        synthetic_region_daily_path = artifact_path(sample_label, "synthetic_region_daily", extension="csv")
        region_spatial_csv_path = artifact_path(sample_label, "region_spatial_per_snapshot", extension="csv")
        region_temporal_csv_path = artifact_path(sample_label, "region_temporal_summary", extension="csv")
        trajectory_distribution_csv_path = artifact_path(sample_label, "trajectory_distribution_summary", extension="csv")
        lag_diagnostics_path = artifact_path(sample_label, "lag_diagnostics", extension="csv")
        region_field_scores_path = artifact_path(sample_label, "region_field_scores", extension="csv")
        observed_farm_summary_path = artifact_path(sample_label, "observed_farm_summary", extension="csv")
        synthetic_farm_summary_path = artifact_path(sample_label, "synthetic_farm_summary", extension="csv")
        farm_spatial_csv_path = artifact_path(sample_label, "farm_spatial_summary", extension="csv")
        farm_corop_csv_path = artifact_path(sample_label, "farm_corop_summary", extension="csv")
        figures = [
            figure_tag(dashboard_path, f"{sample_label} trajectory dashboard", title_text="Trajectory dashboard"),
            figure_tag(delta_path, f"{sample_label} trajectory median deltas", title_text="Trajectory median deltas"),
            figure_tag(daily_mean_path, f"{sample_label} daily mean trajectories", title_text="Daily mean trajectories"),
            figure_tag(distribution_path, f"{sample_label} replicate outcome distributions", title_text="Replicate outcome distributions"),
            figure_tag(parity_path, f"{sample_label} median parity checks", title_text="Median parity checks"),
            figure_tag(channel_path, f"{sample_label} hybrid-channel diagnostics", title_text="Hybrid-channel diagnostics"),
            figure_tag(farm_spatial_path, f"{sample_label} farm-level spatial validation", title_text="Farm-level spatial validation"),
            figure_tag(region_spatial_path, f"{sample_label} regional spatial-fit overview", title_text="Regional spatial-fit overview"),
            link_figure_card(
                region_geo_path,
                "Interactive COROP viewer",
                "Open the linked HTML viewer for day-by-day observed, synthetic, and delta maps of reservoir, import, and export pressure.",
            ),
        ]
        figures_html = "".join(fig for fig in figures if fig)
        artifact_specs = [
            (report_md_path, "Markdown detail report"),
            (summary_json_path, "Summary JSON"),
            (per_snapshot_csv_path, "Per-day summary CSV"),
            (outcome_distribution_csv_path, "Outcome distribution summary CSV"),
            (daily_mean_comparison_path, "Daily mean-comparison CSV"),
            (daily_calibration_path, "Daily calibration CSV"),
            (scalar_calibration_path, "Scalar calibration CSV"),
            (uncertainty_decomposition_path, "Uncertainty decomposition CSV"),
            (observed_daily_path, "Observed daily CSV"),
            (synthetic_daily_path, "Synthetic daily CSV"),
            (observed_outcomes_path, "Observed outcomes CSV"),
            (synthetic_outcomes_path, "Synthetic outcomes CSV"),
            (trajectory_distribution_csv_path, "Trajectory-distribution summary CSV"),
            (lag_diagnostics_path, "Lag diagnostics CSV"),
            (observed_farm_summary_path, "Observed farm summary CSV"),
            (synthetic_farm_summary_path, "Synthetic farm summary CSV"),
            (farm_spatial_csv_path, "Farm spatial summary CSV"),
            (farm_corop_csv_path, "Farm COROP summary CSV"),
            (region_spatial_csv_path, "Regional spatial-fit summary CSV"),
            (region_temporal_csv_path, "Regional temporal-fit summary CSV"),
            (region_field_scores_path, "Regional field-score CSV"),
            (observed_region_daily_path, "Observed regional daily CSV"),
            (synthetic_region_daily_path, "Synthetic regional daily CSV"),
            (region_geo_path, "Interactive COROP geo comparison"),
        ]
        artifact_items = [artifact_link(path, label) for path, label in artifact_specs]
        regional_links = "".join(item for item in artifact_items if item)
        if not figures_html and not regional_links:
            return ""
        regional_links_html = ""
        if regional_links:
            regional_links_html = (
                "<div class='artifact-card'>"
                "<h3>Artifacts</h3>"
                "<ul class='artifact-list'>"
                + regional_links
                + "</ul>"
                "</div>"
            )

        pill_entries = [
            f"Prev corr {fmt_from_row(row, 'farm_prevalence_curve_correlation')}",
            f"Prev mean corr {fmt_from_row(row, 'farm_prevalence_mean_curve_correlation')}",
            f"Attack W1 {fmt_from_row(row, 'farm_attack_rate_wasserstein')}",
            f"Farm risk corr {fmt_from_row(row, 'farm_attack_probability_correlation')}",
            f"Region spatial corr {fmt_from_row(row, 'region_reservoir_spatial_correlation_mean')}",
        ]
        metric_pills_html = "".join(f"<span class='metric-pill'>{html.escape(entry)}</span>" for entry in pill_entries if "n/a" not in entry)

        channel_summary_entries = [
            ("Farm→Farm hazard correlation", fmt_from_row(row, "farm_hazard_ff_curve_correlation")),
            ("Region→Farm hazard correlation", fmt_from_row(row, "farm_hazard_rf_curve_correlation")),
            ("Farm→Region pressure correlation", fmt_from_row(row, "region_pressure_fr_curve_correlation")),
            ("Region→Region pressure correlation", fmt_from_row(row, "region_pressure_rr_curve_correlation")),
            ("Region→Farm share Wasserstein", fmt_from_row(row, "farm_hazard_rf_share_wasserstein")),
            ("Region→Region share Wasserstein", fmt_from_row(row, "region_pressure_rr_share_wasserstein")),
        ]
        spatial_summary_entries = [
            ("Farm attack-probability correlation", fmt_from_row(row, "farm_attack_probability_correlation")),
            ("Farm attack-probability MAE", fmt_from_row(row, "farm_attack_probability_mae")),
            ("Farm hotspot overlap (top 10%)", fmt_from_row(row, "farm_attack_probability_top10_overlap")),
            ("Farm Moran's I absolute delta", fmt_from_row(row, "farm_attack_probability_moran_i_abs_delta")),
            ("Farm first-infection-day correlation", fmt_from_row(row, "farm_first_infection_day_correlation")),
            ("COROP farm-risk correlation", fmt_from_row(row, "farm_corop_attack_probability_correlation")),
            ("Reservoir field energy distance", fmt_from_row(row, "region_reservoir_field_energy_distance_mean")),
            ("Reservoir field variogram distance", fmt_from_row(row, "region_reservoir_field_variogram_distance_mean")),
        ]

        detail_tables = "".join(
            card
            for card in [
                kv_table_card(
                    "Hybrid-channel summary",
                    channel_summary_entries,
                    note="These row-level metrics summarize whether the synthetic network preserves the relative contribution of the four hybrid transmission channels, not just the final epidemic totals.",
                ),
                kv_table_card(
                    "Spatial and field summary",
                    spatial_summary_entries,
                    note="These row-level metrics summarize detailed farm-space agreement and the regional field-distance diagnostics that are otherwise easy to miss in the figure gallery.",
                ),
                table_card_from_csv(
                    outcome_distribution_csv_path,
                    "Outcome distribution summary",
                    note="Observed and synthetic scalar endpoints compared with Wasserstein distance and median location.",
                ),
                table_card_from_csv(
                    daily_mean_comparison_path,
                    "Daily mean comparison",
                    note="Mean trajectory comparison across days for the main epidemic metrics.",
                ),
                table_card_from_csv(
                    daily_calibration_path,
                    "Daily calibration",
                    note="Daily 90% interval coverage and interval width for the main epidemic trajectories.",
                ),
                table_card_from_csv(
                    scalar_calibration_path,
                    "Scalar calibration",
                    note="Observed scalar medians checked against synthetic 90% intervals.",
                ),
                table_card_from_csv(
                    uncertainty_decomposition_path,
                    "Uncertainty decomposition",
                    note="Between-panel network uncertainty separated from within-panel epidemic stochasticity.",
                ),
                table_card_from_csv(
                    trajectory_distribution_csv_path,
                    "Trajectory distribution distances",
                    note="Energy and variogram distances for full replicate trajectory ensembles.",
                ),
                table_card_from_csv(
                    lag_diagnostics_path,
                    "Lag diagnostics",
                    note="Best-lag correlations show whether the synthetic epidemic is shape-correct but phase-shifted.",
                ),
                table_card_from_csv(
                    farm_spatial_csv_path,
                    "Farm spatial summary",
                    note="Farm-level network-generated attack probabilities and first-infection timing, sorted to surface the largest mismatches first.",
                    max_rows=30,
                    transform=transform_farm_spatial,
                ),
                table_card_from_csv(
                    farm_corop_csv_path,
                    "Farm risk by COROP",
                    note="Farm-level network-generated attack probabilities aggregated to the COROP resolution.",
                    max_rows=20,
                    transform=transform_farm_corop,
                ),
                table_card_from_csv(
                    region_field_scores_path,
                    "Regional field distances",
                    note="Daily energy and variogram distances for the reservoir, import, and export COROP fields.",
                    max_rows=35,
                    transform=transform_region_field,
                ),
                table_card_from_csv(
                    region_temporal_csv_path,
                    "Regional temporal fit",
                    note="Per-COROP temporal fit summary through time for reservoir, import, and export pressure.",
                    max_rows=40,
                    transform=transform_region_temporal,
                ),
            ]
            if card
        )
        return (
            f"<section class='section gallery-section' id='{html.escape(section_id)}'>"
            + _render_section_heading(
                f"{role_label}: {sample_label}",
                control_id=f"{slugify(sample_label)}_visual_explain",
                explain_text=_simulation_notes_text(_simulation_figure_reading_notes()),
                button_label="How to read this figure",
            )
            + f"<p class='subtitle'>Farm prevalence corr {fmt_from_row(row, 'farm_prevalence_curve_correlation')}, "
            + f"attack-rate W1 {fmt_from_row(row, 'farm_attack_rate_wasserstein')}, "
            + f"duration W1 {fmt_from_row(row, 'farm_duration_wasserstein')}, "
            + f"regional reservoir spatial corr {fmt_from_row(row, 'region_reservoir_spatial_correlation_mean')}, "
            + f"farm risk corr {fmt_from_row(row, 'farm_attack_probability_correlation')}. "
            + "These figures and tables surface temporal fit, hybrid-channel behavior, detailed farm-space agreement, and region-level spatial agreement in one place.</p>"
            + (f"<div class='metric-pill-row'>{metric_pills_html}</div>" if metric_pills_html else "")
            + regional_links_html
            + "<div class='figure-grid figure-grid-two'>"
            + f"{figures_html}"
            + "</div>"
            + ("<div class='detail-stack'>" + detail_tables + "</div>" if detail_tables else "")
            + "<p class='back-link'><a href='#top'>Back to top</a></p>"
            + "</section>"
        )

    def row_role_label(row: pd.Series) -> str:
        sample_label = str(row.get("sample_label", ""))
        posterior_runs_local = pd.to_numeric(pd.Series([row.get("posterior_num_runs", 1)]), errors="coerce").fillna(1).iloc[0]
        if sample_label == str(best_curve["sample_label"]):
            return "Best curve fit"
        if sample_label == str(best_farm_spatial["sample_label"]):
            return "Best farm-space fit"
        if sample_label == str(best_regional["sample_label"]):
            return "Best regional fit"
        if sample_label == str(lowest_attack["sample_label"]):
            return "Lowest attack distance"
        if float(posterior_runs_local) > 1 or "__sample_" not in sample_label:
            return "Posterior aggregate"
        return "Run detail"

    def gallery_rows(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.copy()
        prevalence = pd.to_numeric(ordered.get("farm_prevalence_curve_correlation", np.nan), errors="coerce").fillna(-np.inf)
        attack = pd.to_numeric(ordered.get("farm_attack_rate_wasserstein", np.nan), errors="coerce").fillna(np.inf)
        posterior_runs_local = pd.to_numeric(
            ordered["posterior_num_runs"] if "posterior_num_runs" in ordered.columns else pd.Series(1.0, index=ordered.index),
            errors="coerce",
        ).fillna(1.0)
        ordered["_gallery_is_aggregate"] = ((posterior_runs_local > 1.0) | ~ordered["sample_label"].astype(str).str.contains("__sample_", regex=False)).astype(int)
        ordered["_gallery_prevalence"] = prevalence
        ordered["_gallery_attack"] = attack
        ordered = ordered.sort_values(
            ["_gallery_is_aggregate", "_gallery_prevalence", "_gallery_attack", "sample_label"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        return ordered.drop(columns=["_gallery_is_aggregate", "_gallery_prevalence", "_gallery_attack"])

    def column_median(frame: pd.DataFrame, column: str) -> str:
        if column not in frame.columns:
            return "n/a"
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return "n/a"
        return f"{float(values.median()):.3f}"

    coverage_table_html, median_completeness_value = coverage_matrix_html(gallery_rows(headline_source))
    median_prevalence_coverage = column_median(headline_rows, "farm_prevalence_interval_coverage")
    median_attack_network_share = column_median(headline_rows, "farm_attack_rate_network_uncertainty_share")
    median_region_reservoir_spatial = column_median(headline_rows, "region_reservoir_spatial_correlation_mean")
    median_farm_spatial_corr = column_median(headline_rows, "farm_attack_probability_correlation")
    median_hybrid_corr = row_mean_text(
        headline_rows,
        [
            "farm_hazard_ff_curve_correlation",
            "farm_hazard_rf_curve_correlation",
            "region_pressure_fr_curve_correlation",
            "region_pressure_rr_curve_correlation",
        ],
    )
    median_completeness = f"{median_completeness_value * 100.0:.0f}%" if median_completeness_value is not None else "n/a"

    selected_columns = [
        ("sample_label", "Setting"),
        ("farm_prevalence_curve_correlation", "Farm prevalence corr"),
        ("farm_incidence_curve_correlation", "Farm incidence corr"),
        ("farm_prevalence_mean_curve_correlation", "Farm prevalence mean corr"),
        ("farm_incidence_mean_curve_correlation", "Farm incidence mean corr"),
        ("farm_cumulative_incidence_curve_correlation", "Cum. incidence corr"),
        ("reservoir_total_curve_correlation", "Reservoir corr"),
        ("farm_attack_probability_correlation", "Farm risk corr"),
        ("region_reservoir_spatial_correlation_mean", "Region reservoir spatial corr"),
        ("region_reservoir_temporal_correlation_mean", "Region reservoir temporal corr"),
        ("farm_attack_rate_wasserstein", "Farm attack-rate W1"),
        ("farm_peak_prevalence_wasserstein", "Farm peak-prev W1"),
        ("farm_duration_wasserstein", "Farm duration W1"),
        ("farm_prevalence_interval_coverage", "Prev. 90% coverage"),
        ("farm_attack_rate_network_uncertainty_share", "Attack network share"),
    ]
    available_columns = [(key, label) for key, label in selected_columns if key in summary_rows.columns or key == "sample_label"]
    table_header = "".join(f"<th>{html.escape(label)}</th>" for _, label in available_columns)
    table_rows = []
    for _, row in sort_frame_by_available(
        headline_rows,
        [
            ("farm_prevalence_curve_correlation", False),
            ("farm_attack_probability_correlation", False),
            ("farm_attack_rate_wasserstein", True),
        ],
    ).head(12).iterrows():
        cells = []
        for key, _ in available_columns:
            if key == "sample_label":
                cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
            else:
                cells.append(f"<td>{fmt_from_row(row, key)}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    hybrid_columns = [
        ("sample_label", "Setting"),
        ("farm_hazard_ff_curve_correlation", "F→F corr"),
        ("farm_hazard_rf_curve_correlation", "R→F corr"),
        ("region_pressure_fr_curve_correlation", "F→R corr"),
        ("region_pressure_rr_curve_correlation", "R→R corr"),
        ("farm_hazard_rf_share_wasserstein", "R→F share W1"),
        ("region_pressure_rr_share_wasserstein", "R→R share W1"),
    ]
    available_hybrid = [(key, label) for key, label in hybrid_columns if key in summary_rows.columns or key == "sample_label"]
    hybrid_rows = []
    if len(available_hybrid) > 1:
        hybrid_sorted = sort_frame_by_available(
            headline_rows,
            [
                ("farm_hazard_ff_curve_correlation", False),
                ("farm_hazard_rf_curve_correlation", False),
                ("region_pressure_fr_curve_correlation", False),
                ("region_pressure_rr_curve_correlation", False),
            ],
        ).head(12)
        for _, row in hybrid_sorted.iterrows():
            cells = []
            for key, _ in available_hybrid:
                if key == "sample_label":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            hybrid_rows.append("<tr>" + "".join(cells) + "</tr>")

    farm_space_columns = [
        ("sample_label", "Setting"),
        ("farm_attack_probability_correlation", "Farm risk corr"),
        ("farm_attack_probability_mae", "Farm risk MAE"),
        ("farm_attack_probability_top10_overlap", "Farm hotspot overlap"),
        ("farm_attack_probability_moran_i_abs_delta", "Farm Moran Δ"),
        ("farm_corop_attack_probability_correlation", "COROP farm-risk corr"),
        ("farm_first_infection_day_correlation", "First-infection corr"),
    ]
    available_farm_space = [(key, label) for key, label in farm_space_columns if key in summary_rows.columns or key == "sample_label"]
    farm_space_rows = []
    if len(available_farm_space) > 1:
        farm_space_sorted = sort_frame_by_available(
            headline_rows,
            [
                ("farm_attack_probability_correlation", False),
                ("farm_attack_probability_top10_overlap", False),
                ("farm_corop_attack_probability_correlation", False),
                ("farm_attack_probability_mae", True),
            ],
        ).head(12)
        for _, row in farm_space_sorted.iterrows():
            cells = []
            for key, _ in available_farm_space:
                if key == "sample_label":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            farm_space_rows.append("<tr>" + "".join(cells) + "</tr>")

    uncertainty_columns = [
        ("sample_label", "Setting"),
        ("farm_prevalence_interval_coverage", "Prev. interval coverage"),
        ("farm_attack_rate_observed_median_in_synthetic_90pct", "Attack in 90% band"),
        ("farm_attack_rate_observed_median_tail_area", "Attack tail area"),
        ("farm_attack_rate_network_uncertainty_share", "Attack network share"),
        ("farm_peak_prevalence_network_uncertainty_share", "Peak network share"),
        ("farm_duration_network_uncertainty_share", "Duration network share"),
    ]
    available_uncertainty = [(key, label) for key, label in uncertainty_columns if key in summary_rows.columns or key == "sample_label"]
    uncertainty_header = "".join(f"<th>{html.escape(label)}</th>" for _, label in available_uncertainty)
    uncertainty_rows = []
    if len(available_uncertainty) > 1:
        for _, row in sort_frame_by_available(
            headline_rows,
            [
                ("farm_prevalence_interval_coverage", False),
                ("farm_attack_rate_network_uncertainty_share", False),
                ("farm_attack_rate_wasserstein", True),
            ],
        ).head(12).iterrows():
            cells = []
            for key, _ in available_uncertainty:
                if key == "sample_label":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            uncertainty_rows.append("<tr>" + "".join(cells) + "</tr>")

    regional_columns = [
        ("sample_label", "Setting"),
        ("region_reservoir_spatial_correlation_mean", "Reservoir spatial corr"),
        ("region_import_spatial_correlation_mean", "Import spatial corr"),
        ("region_export_spatial_correlation_mean", "Export spatial corr"),
        ("region_reservoir_temporal_correlation_mean", "Reservoir temporal corr"),
        ("region_import_temporal_correlation_mean", "Import temporal corr"),
        ("region_export_temporal_correlation_mean", "Export temporal corr"),
        ("region_reservoir_hotspot_overlap_mean", "Reservoir hotspot overlap"),
        ("region_reservoir_field_energy_distance_mean", "Reservoir field ED"),
        ("region_reservoir_field_variogram_distance_mean", "Reservoir field VD"),
    ]
    available_regional = [(key, label) for key, label in regional_columns if key in summary_rows.columns or key == "sample_label"]
    regional_header = "".join(f"<th>{html.escape(label)}</th>" for _, label in available_regional)
    regional_rows = []
    if len(available_regional) > 1:
        regional_sorted = sort_frame_by_available(
            headline_rows,
            [
                ("region_reservoir_spatial_correlation_mean", False),
                ("region_reservoir_temporal_correlation_mean", False),
                ("region_import_spatial_correlation_mean", False),
                ("region_reservoir_field_energy_distance_mean", True),
            ],
        ).head(12)
        for _, row in regional_sorted.iterrows():
            cells = []
            for key, _ in available_regional:
                if key == "sample_label":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            regional_rows.append("<tr>" + "".join(cells) + "</tr>")

    headline_rows_html = []
    headline_specs = [
        ("Best curve fit", best_curve),
        ("Best farm-space fit", best_farm_spatial),
        ("Best regional fit", best_regional),
        ("Lowest attack distance", lowest_attack),
    ]
    for role, row in headline_specs:
        headline_rows_html.append(
            "<tr>"
            + f"<td>{html.escape(role)}</td>"
            + f"<td>{html.escape(str(row.get('sample_label')))}</td>"
            + f"<td>{fmt_from_row(row, 'farm_prevalence_curve_correlation')}</td>"
            + f"<td>{fmt_from_row(row, 'farm_attack_probability_correlation')}</td>"
            + f"<td>{fmt_from_row(row, 'region_reservoir_spatial_correlation_mean')}</td>"
            + f"<td>{fmt_from_row(row, 'farm_hazard_rf_curve_correlation')}</td>"
            + f"<td>{fmt_from_row(row, 'farm_attack_rate_wasserstein')}</td>"
            + f"<td>{fmt_from_row(row, 'farm_duration_wasserstein')}</td>"
            + f"<td>{fmt_from_row(row, 'region_reservoir_field_energy_distance_mean')}</td>"
            + f"<td>{fmt_from_row(row, 'farm_prevalence_interval_coverage')}</td>"
            + "</tr>"
        )

    top_jump_links = [
        ("overview", "Overview"),
        ("report-coverage", "Report coverage"),
        ("selected-settings", "Selected settings"),
        ("hybrid-channel-fit", "Hybrid channels"),
        ("farm-space-fit", "Farm-space fit"),
        ("uncertainty-calibration", "Uncertainty"),
        ("regional-fit-summary", "Regional fit"),
        ("headline-comparisons", "Headline comparisons"),
        ("full-baseline-gallery", "Gallery"),
    ]
    jumpbar_html = "".join(
        f"<a class='jump-chip' href='#{html.escape(anchor)}'>{html.escape(label)}</a>"
        for anchor, label in top_jump_links
    )
    setting_jumpbar_html = "".join(
        f"<a class='jump-chip jump-chip-soft' href='#{html.escape(section_id_for_sample(row.get('sample_label')))}'>{html.escape(_setting_display_payload(str(row.get('sample_label')))['short_label'])}</a>"
        for _, row in gallery_rows(summary_rows).iterrows()
    )

    gallery_sections = "".join(
        setting_visual_section(row, row_role_label(row))
        for _, row in gallery_rows(summary_rows).iterrows()
    )

    html_parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        """
        :root {
          --bg: #f5f7fb;
          --panel: #ffffff;
          --panel-soft: #f7fafc;
          --text: #22313f;
          --muted: #5f6f7f;
          --line: #dce3ea;
          --line-strong: #c8d4e0;
          --blue: #2a6fbb;
          --blue-soft: #edf4fb;
          --shadow: 0 10px 28px rgba(34, 49, 63, 0.06);
        }
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body {
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          margin: 0;
          background: linear-gradient(180deg, #f6f8fb 0%, #eef3f8 100%);
          color: var(--text);
        }
        a { color: var(--blue); }
        .page { max-width: 1480px; margin: 0 auto; padding: 28px 20px 64px; }
        .hero { margin-bottom: 24px; }
        .eyebrow {
          color: var(--blue);
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-weight: 700;
          font-size: 12px;
          margin-bottom: 10px;
        }
        h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 34px); }
        .subtitle { color: var(--muted); max-width: 1040px; line-height: 1.6; }
        .summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 14px;
          margin-top: 18px;
        }
        .summary-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 16px 18px;
          box-shadow: var(--shadow);
          min-width: 0;
        }
        .summary-card h3 {
          margin: 0 0 6px;
          font-size: 12px;
          color: var(--muted);
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .metric-value {
          font-size: clamp(22px, 3vw, 34px);
          font-weight: 700;
          line-height: 1.05;
          overflow-wrap: anywhere;
        }
        .jumpbar {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 16px;
        }
        .jump-chip {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(42, 111, 187, 0.18);
          background: #ffffff;
          color: var(--blue);
          text-decoration: none;
          font-size: 13px;
          font-weight: 600;
          box-shadow: 0 6px 14px rgba(42, 111, 187, 0.06);
        }
        .jump-chip:hover { background: var(--blue-soft); }
        .jump-chip-soft {
          background: #fbfdff;
          color: var(--text);
          border-color: var(--line);
        }
        .section { margin-top: 26px; scroll-margin-top: 18px; }
        .gallery-section { padding-top: 4px; }
        .section-heading {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 10px;
          margin-bottom: 12px;
        }
        .section-heading h2 { margin: 0; }
        .figure-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
          gap: 16px;
        }
        .figure-grid-two { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
        .summary-card, .figure-card, .artifact-card, .table-card { min-width: 0; }
        .figure-grid > *, .detail-stack > * { min-width: 0; }
        .figure-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 18px;
          box-shadow: var(--shadow);
        }
        .figure-card h3 { margin: 0 0 10px; font-size: 16px; }
        .figure-frame { min-width: 0; overflow: hidden; padding: 10px; border-radius: 14px; background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%); border: 1px solid var(--line); }
        .figure-card img {
          width: 100%;
          max-width: 100%;
          height: auto;
          border-radius: 12px;
          display: block;
          background: var(--panel-soft);
        }
        .figure-popout-button {
          width: 100%;
          padding: 0;
          border: 0;
          background: transparent;
          cursor: zoom-in;
        }
        .figure-popout-button:focus-visible {
          outline: 2px solid var(--blue);
          outline-offset: 4px;
          border-radius: 16px;
        }
        .figure-popout-button img { transition: transform 140ms ease; }
        .figure-popout-button:hover img { transform: scale(1.01); }
        .figure-card figcaption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.5; }
        .link-card .link-card-inner {
          min-height: 100%;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }
        .card-link { margin: 14px 0 0; }
        .card-link a {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 10px 14px;
          border-radius: 10px;
          border: 1px solid rgba(42, 111, 187, 0.22);
          background: var(--blue-soft);
          text-decoration: none;
          font-weight: 700;
        }
        .report-table {
          width: 100%;
          min-width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          background: var(--panel);
          font-size: 14px;
        }
        .report-table th, .report-table td {
          padding: 10px 12px;
          border-bottom: 1px solid var(--line);
          text-align: left;
          font-size: 14px;
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
          vertical-align: top;
        }
        .dataframe.report-table {
          width: max-content;
          min-width: 100%;
        }
        .dataframe.report-table th, .dataframe.report-table td {
          white-space: nowrap;
          overflow-wrap: normal;
          word-break: normal;
        }
        .report-table thead th {
          background: #eef4f8;
          font-weight: 700;
          position: sticky;
          top: 0;
          z-index: 2;
        }
        .report-table tbody tr:nth-child(even) td { background: #fbfdff; }
        .report-table tr:last-child td { border-bottom: 0; }
        .compact-table th { width: 64%; background: #f7fafc; position: static; }
        .compact-table th, .compact-table td {
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
        }
        .status-cell { text-align: center; }
        .status-ok {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 24px;
          min-height: 24px;
          border-radius: 999px;
          background: #e7f6ee;
          color: #1f8a4c;
          font-weight: 800;
        }
        .status-missing {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 24px;
          min-height: 24px;
          border-radius: 999px;
          background: #fff1f0;
          color: #c4483a;
          font-weight: 800;
        }
        .table-wrap { max-width: 100%; overflow: auto; max-height: 560px; border: 1px solid var(--line); border-radius: 16px; background: var(--panel-soft); -webkit-overflow-scrolling: touch; }
        .artifact-card {
          margin-bottom: 16px;
          padding: 14px 16px;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: var(--panel);
          box-shadow: var(--shadow);
        }
        .artifact-card h3 { margin: 0 0 10px; font-size: 14px; }
        .artifact-list {
          margin: 0;
          padding-left: 18px;
          color: var(--muted);
          column-width: 240px;
          column-gap: 18px;
        }
        .artifact-list li { margin: 6px 0; break-inside: avoid; }
        .detail-stack { display: grid; gap: 16px; margin-top: 16px; }
        .table-card {
          padding: 16px 18px;
          border-radius: 18px;
          border: 1px solid var(--line);
          background: var(--panel);
          box-shadow: var(--shadow);
        }
        .table-card h3 { margin: 0 0 10px; font-size: 16px; }
        .table-note { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.55; }
        .gallery-stack { display: grid; gap: 22px; }
        .metric-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin: 14px 0 14px;
        }
        .metric-pill {
          padding: 7px 10px;
          border-radius: 999px;
          background: #eef4f8;
          color: var(--text);
          font-size: 13px;
          font-weight: 600;
        }
        .figure-overlay[hidden] { display: none; }
        .figure-overlay {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }
        .figure-overlay-backdrop {
          position: absolute;
          inset: 0;
          background: rgba(17, 25, 33, 0.78);
          backdrop-filter: blur(3px);
        }
        .figure-overlay-panel {
          position: relative;
          z-index: 1;
          width: min(1240px, 100%);
          max-height: calc(100vh - 40px);
          display: grid;
          grid-template-rows: auto minmax(0, 1fr) auto;
          gap: 12px;
          padding: 18px;
          border-radius: 20px;
          border: 1px solid rgba(220, 227, 234, 0.45);
          background: rgba(247, 250, 252, 0.98);
          box-shadow: 0 24px 60px rgba(17, 25, 33, 0.28);
        }
        .figure-overlay-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }
        .figure-overlay-title {
          font-size: 16px;
          font-weight: 700;
          color: var(--text);
        }
        .figure-overlay-actions {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .overlay-button {
          appearance: none;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 42px;
          min-height: 38px;
          padding: 8px 12px;
          border-radius: 10px;
          border: 1px solid rgba(42, 111, 187, 0.18);
          background: #ffffff;
          color: var(--text);
          text-decoration: none;
          font: inherit;
          font-size: 14px;
          font-weight: 700;
          cursor: pointer;
        }
        .overlay-button:hover { background: var(--blue-soft); }
        .overlay-button-primary {
          background: var(--blue-soft);
          color: var(--blue);
        }
        .figure-overlay-stage {
          min-height: 0;
          overflow: auto;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%);
          display: flex;
          align-items: flex-start;
          justify-content: center;
          padding: 18px;
        }
        .figure-overlay-stage img {
          display: block;
          max-width: 100%;
          height: auto;
          transform-origin: top center;
          transition: transform 120ms ease;
          box-shadow: 0 14px 36px rgba(34, 49, 63, 0.16);
          border-radius: 12px;
          background: #ffffff;
        }
        .figure-overlay-caption {
          margin: 0;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.55;
        }
        body.overlay-open { overflow: hidden; }
        .back-link { margin: 14px 0 0; }
        .back-link a {
          text-decoration: none;
          font-size: 13px;
          font-weight: 600;
        }
        .explain-widget { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
        .explain-button {
          appearance: none;
          border: 1px solid rgba(42,111,187,0.24);
          background: linear-gradient(180deg, #fafdff 0%, #edf4f9 100%);
          color: var(--blue);
          border-radius: 999px;
          padding: 8px 12px;
          font: inherit;
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.02em;
          cursor: pointer;
          box-shadow: 0 6px 14px rgba(42,111,187,0.08);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .explain-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 10px 20px rgba(42,111,187,0.12);
          border-color: rgba(42,111,187,0.36);
        }
        .explain-button[aria-expanded="true"] {
          background: linear-gradient(180deg, #edf4f9 0%, #e2edf6 100%);
        }
        .explain-panel {
          width: 100%;
          padding: 12px 14px;
          border-radius: 14px;
          border: 1px solid var(--line);
          background: #f7fafc;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.6;
        }
        .explain-panel p { margin: 0; color: var(--muted); }
        .muted { color: var(--muted); }
        @media (max-width: 980px) {
          .figure-grid-two { grid-template-columns: 1fr; }
          .artifact-list { column-width: auto; }
        }
        @media (max-width: 640px) {
          .page { padding-left: 14px; padding-right: 14px; }
          .jump-chip { width: 100%; justify-content: flex-start; }
          .metric-pill-row { gap: 8px; }
          .figure-overlay { padding: 12px; }
          .figure-overlay-panel { max-height: calc(100vh - 24px); padding: 14px; }
          .figure-overlay-stage { padding: 12px; }
          .overlay-button { flex: 1 1 auto; }
        }
        """,
        _scenario_navigation_css(),
        "</style>",
        "</head>",
        "<body>",
        "<main class='page' id='top'>",
        "<section class='hero'>",
        "<div class='eyebrow'>Hybrid transmission validation</div>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='subtitle'>This report compares farm-focused epidemic outcomes on the observed temporal panel with outcomes on synthetic panels drawn from the fitted temporal block model. In addition to the aggregate epidemic trajectories, it now surfaces hybrid-channel diagnostics, detailed farm-space risk agreement, regional spatial and temporal fit, and a coverage matrix that checks whether the expected figures and key tables are actually present in the final report bundle.</p>",
        "<div class='summary-grid'>",
        "<div class='summary-card'><h3>Settings</h3><div class='metric-value'>" + str(int(len(headline_rows))) + "</div></div>",
        "<div class='summary-card'><h3>Run details</h3><div class='metric-value'>" + str(int(len(detail_rows))) + "</div></div>",
        "<div class='summary-card'><h3>Best farm prevalence corr</h3><div class='metric-value'>" + f"{float(pd.to_numeric(headline_rows['farm_prevalence_curve_correlation'], errors='coerce').max()):.3f}" + "</div></div>",
        "<div class='summary-card'><h3>Lowest farm attack-rate W1</h3><div class='metric-value'>" + f"{float(pd.to_numeric(headline_rows['farm_attack_rate_wasserstein'], errors='coerce').min()):.3f}" + "</div></div>",
        "<div class='summary-card'><h3>Median prev. coverage</h3><div class='metric-value'>" + html.escape(median_prevalence_coverage) + "</div></div>",
        "<div class='summary-card'><h3>Median region spatial corr</h3><div class='metric-value'>" + html.escape(median_region_reservoir_spatial) + "</div></div>",
        "<div class='summary-card'><h3>Median farm risk corr</h3><div class='metric-value'>" + html.escape(median_farm_spatial_corr) + "</div></div>",
        "<div class='summary-card'><h3>Median hybrid channel corr</h3><div class='metric-value'>" + html.escape(median_hybrid_corr) + "</div></div>",
        "<div class='summary-card'><h3>Median report completeness</h3><div class='metric-value'>" + html.escape(median_completeness) + "</div></div>",
        "</div>",
        "<div class='jumpbar'>",
        jumpbar_html,
        "</div>",
        "<div class='jumpbar'>",
        setting_jumpbar_html,
        "</div>",
        "</section>",
        scenario_navigation_html,
        "<section class='section' id='overview'>",
        _render_section_heading(
            "Overview",
            control_id="overview_section_explain",
            explain_text="This figure ranks settings on the main transmission-fit metrics across farm curves, farm-space risk, and regional-space agreement. Correlations are higher-is-better, while Wasserstein and field distances are lower-is-better. Use it as the first pass before opening the per-setting dashboards.",
            button_label="How to read this figure",
        ),
        "<div class='figure-grid'>",
        figure_tag(overview_path, "Across-setting overview of temporal, farm-space, and region-space transmission validation metrics. Correlations are higher-is-better; Wasserstein distances are lower-is-better.", title_text="Across-setting overview"),
        "</div>",
        "</section>",
    ]

    if coverage_table_html:
        html_parts.extend([
            "<section class='section' id='report-coverage'>",
            _render_section_heading(
                "Report coverage matrix",
                control_id="report_coverage_explain",
                explain_text="This table is a completeness check for the report bundle itself. A row is stronger when it has all core figures plus the key tables that support the new hybrid-channel, farm-space, and regional-field diagnostics.",
                button_label="How to read this table",
            ),
            "<p class='subtitle'>This matrix checks whether each setting-level report actually contains the expected visualizations and key detail tables. It helps catch cases where a metric was computed but never surfaced in the final HTML or linked artifact bundle.</p>",
            table_card_from_html(coverage_table_html),
            "</section>",
        ])

    html_parts.extend([
        "<section class='section' id='selected-settings'>",
        _render_section_heading(
            "Selected settings",
            control_id="selected_settings_explain",
            explain_text=_simulation_notes_text(_simulation_table_reading_notes()),
            button_label="How to read this table",
        ),
        table_card_from_html(
            "<table class='report-table'>"
            + f"<thead><tr>{table_header}</tr></thead>"
            + "<tbody>"
            + "".join(table_rows)
            + "</tbody></table>"
        ),
        "</section>",
    ])

    if hybrid_rows:
        html_parts.extend([
            "<section class='section' id='hybrid-channel-fit'>",
            _render_section_heading(
                "Hybrid-channel fit summary",
                control_id="hybrid_channel_fit_explain",
                explain_text="These columns make the coupling structure explicit. Higher correlations mean the synthetic panel preserves the same day-to-day flow pattern for the four hybrid channels, while lower share Wasserstein distances mean the relative contribution of those channels is closer to the observed panel.",
                button_label="How to read this table",
            ),
            "<p class='subtitle'>This section surfaces the channel-level diagnostics directly in the report so the hybrid network is evaluated on its own terms rather than only through farm-level epidemic totals.</p>",
            table_card_from_html(
                "<table class='report-table'>"
                + "<thead><tr>"
                + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_hybrid)
                + "</tr></thead><tbody>"
                + "".join(hybrid_rows)
                + "</tbody></table>"
            ),
            "</section>",
        ])

    if farm_space_rows:
        html_parts.extend([
            "<section class='section' id='farm-space-fit'>",
            _render_section_heading(
                "Farm-space fit summary",
                control_id="farm_space_fit_explain",
                explain_text="These metrics summarize whether the detailed farm-risk pattern is preserved. Correlation and hotspot overlap should be high, while MAE and Moran's I delta should be low if the synthetic panel preserves both local hotspots and overall clustering.",
                button_label="How to read this table",
            ),
            "<p class='subtitle'>This farm-space scorecard makes the detailed-resolution evaluation visible at the top level instead of leaving it buried in per-setting CSV files.</p>",
            table_card_from_html(
                "<table class='report-table'>"
                + "<thead><tr>"
                + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_farm_space)
                + "</tr></thead><tbody>"
                + "".join(farm_space_rows)
                + "</tbody></table>"
            ),
            "</section>",
        ])

    if uncertainty_rows:
        html_parts.extend([
            "<section class='section' id='uncertainty-calibration'>",
            _render_section_heading(
                "Uncertainty and calibration",
                control_id="uncertainty_calibration_explain",
                explain_text=_simulation_notes_text(_simulation_table_reading_notes()),
                button_label="How to read this table",
            ),
            "<p class='subtitle'>These columns make uncertainty explicit rather than implicit by showing whether the observed panel falls inside synthetic predictive intervals and, for repeated synthetic settings, how much predictive variance is attributable to between-panel network variation rather than within-panel epidemic stochasticity.</p>",
            table_card_from_html(
                "<table class='report-table'>"
                + f"<thead><tr>{uncertainty_header}</tr></thead>"
                + "<tbody>"
                + "".join(uncertainty_rows)
                + "</tbody></table>"
            ),
            "</section>",
        ])

    if regional_rows:
        html_parts.extend([
            "<section class='section' id='regional-fit-summary'>",
            _render_section_heading(
                "Regional fit summary",
                control_id="regional_fit_summary_explain",
                explain_text="These metrics summarize how well each setting matches the observed COROP-level reservoir, import, and export patterns. Spatial correlations compare regions within each day, temporal correlations compare each COROP through time, hotspot overlap checks whether the same high-intensity regions are highlighted, and field distances summarize whole-map mismatch.",
                button_label="How to read this table",
            ),
            "<p class='subtitle'>The regional table brings the COROP-level simulation outputs into the main report so the spatial and temporal fit can be reviewed next to the farm-level epidemic summary.</p>",
            table_card_from_html(
                "<table class='report-table'>"
                + f"<thead><tr>{regional_header}</tr></thead>"
                + "<tbody>"
                + "".join(regional_rows)
                + "</tbody></table>"
            ),
            "</section>",
        ])

    html_parts.extend([
        "<section class='section' id='headline-comparisons'>",
        _render_section_heading(
            "Headline comparisons",
            control_id="headline_comparisons_explain",
            explain_text="These rows are the quick shortlist across fit domains. Read them across farm trajectories, farm-space risk, regional-space fit, hybrid coupling, and calibration instead of focusing on one number. Higher correlations and coverage are better, while lower Wasserstein and field distances mean tighter agreement.",
            button_label="How to read this table",
        ),
        table_card_from_html(
            "<table class='report-table'>"
            + "<thead><tr><th>Role</th><th>Setting</th><th>Farm prevalence corr</th><th>Farm risk corr</th><th>Region spatial corr</th><th>R→F corr</th><th>Attack W1</th><th>Duration W1</th><th>Reservoir field ED</th><th>Prev. 90% coverage</th></tr></thead>"
            + "<tbody>"
            + "".join(headline_rows_html)
            + "</tbody></table>"
        ),
        "</section>",
        "<section class='section' id='full-baseline-gallery'>",
        _render_section_heading(
            "Full baseline gallery",
            control_id="full_baseline_gallery_explain",
            explain_text="This gallery embeds every generated baseline row. Use the aggregate row for the setting-level view, then inspect each run detail to check how stable the temporal fit, hybrid coupling, farm-space agreement, and regional diagnostics are across the saved panels.",
            button_label="How to read this figure",
        ),
        "<div class='gallery-stack'>",
        gallery_sections,
        "</div>",
        "</section>",
        "<div class='figure-overlay' id='figure_overlay' hidden aria-hidden='true'>",
        "<div class='figure-overlay-backdrop' data-overlay-close='true'></div>",
        "<div class='figure-overlay-panel' role='dialog' aria-modal='true' aria-labelledby='figure_overlay_title'>",
        "<div class='figure-overlay-toolbar'>",
        "<div class='figure-overlay-title' id='figure_overlay_title'>Figure</div>",
        "<div class='figure-overlay-actions'>",
        "<button type='button' class='overlay-button' id='figure_overlay_zoom_out' aria-label='Zoom out'>-</button>",
        "<button type='button' class='overlay-button' id='figure_overlay_zoom_in' aria-label='Zoom in'>+</button>",
        "<a class='overlay-button overlay-button-primary' id='figure_overlay_save' href='#' download>Save</a>",
        "<button type='button' class='overlay-button' id='figure_overlay_close' data-overlay-close='true'>Close</button>",
        "</div>",
        "</div>",
        "<div class='figure-overlay-stage'>",
        "<img id='figure_overlay_image' alt=''>",
        "</div>",
        "<p class='figure-overlay-caption' id='figure_overlay_caption'></p>",
        "</div>",
        "</div>",
        "<script>",
        """
        document.querySelectorAll('.explain-button').forEach((button) => {
          button.addEventListener('click', () => {
            const panel = document.getElementById(button.dataset.explainTarget);
            if (!panel) return;
            const expanded = button.getAttribute('aria-expanded') === 'true';
            button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
            panel.hidden = expanded;
          });
        });

        const overlay = document.getElementById('figure_overlay');
        const overlayImage = document.getElementById('figure_overlay_image');
        const overlayTitle = document.getElementById('figure_overlay_title');
        const overlayCaption = document.getElementById('figure_overlay_caption');
        const overlaySave = document.getElementById('figure_overlay_save');
        const zoomInButton = document.getElementById('figure_overlay_zoom_in');
        const zoomOutButton = document.getElementById('figure_overlay_zoom_out');
        let overlayScale = 1;

        function clampScale(nextScale) {
          return Math.max(0.5, Math.min(4, nextScale));
        }

        function applyOverlayScale(nextScale) {
          overlayScale = clampScale(nextScale);
          overlayImage.style.transform = `scale(${overlayScale})`;
        }

        function openFigureOverlay(trigger) {
          const imageSrc = trigger.dataset.imageSrc;
          if (!imageSrc) return;
          overlayImage.src = imageSrc;
          overlayImage.alt = trigger.dataset.imageCaption || trigger.dataset.imageTitle || 'Figure';
          overlayTitle.textContent = trigger.dataset.imageTitle || 'Figure';
          overlayCaption.textContent = trigger.dataset.imageCaption || '';
          overlaySave.href = imageSrc;
          overlaySave.download = imageSrc.split('/').pop() || 'figure.png';
          applyOverlayScale(1);
          overlay.hidden = false;
          overlay.setAttribute('aria-hidden', 'false');
          document.body.classList.add('overlay-open');
        }

        function closeFigureOverlay() {
          overlay.hidden = true;
          overlay.setAttribute('aria-hidden', 'true');
          overlayImage.removeAttribute('src');
          overlaySave.href = '#';
          document.body.classList.remove('overlay-open');
        }

        document.querySelectorAll('.figure-popout-button').forEach((button) => {
          button.addEventListener('click', () => openFigureOverlay(button));
        });

        zoomInButton.addEventListener('click', () => applyOverlayScale(overlayScale + 0.25));
        zoomOutButton.addEventListener('click', () => applyOverlayScale(overlayScale - 0.25));

        overlay.querySelectorAll('[data-overlay-close="true"]').forEach((element) => {
          element.addEventListener('click', closeFigureOverlay);
        });

        document.addEventListener('keydown', (event) => {
          if (overlay.hidden) return;
          if (event.key === 'Escape') {
            closeFigureOverlay();
          } else if (event.key === '+' || event.key === '=') {
            event.preventDefault();
            applyOverlayScale(overlayScale + 0.25);
          } else if (event.key === '-') {
            event.preventDefault();
            applyOverlayScale(overlayScale - 0.25);
          }
        });
        """,
        "</script>",
        "</main></body></html>",
    ])
    output_path.write_text("\n".join(part for part in html_parts if part), encoding="utf-8")
    return output_path


# Orchestration
def _build_config(args: argparse.Namespace, observed_edges: pd.DataFrame, weight_col: Optional[str]) -> HybridSimulationConfig:
    weight_scale = None if args.weight_scale in {None, "auto"} else float(args.weight_scale)
    config = HybridSimulationConfig(
        model=str(args.model).upper(),
        beta_ff=float(args.beta_ff),
        beta_fr=float(args.beta_fr),
        beta_rf=float(args.beta_rf),
        beta_rr=float(args.beta_rr),
        sigma=float(args.sigma),
        gamma=float(args.gamma),
        num_replicates=int(args.num_replicates),
        seed=int(args.seed),
        initial_seed_count=int(args.initial_seed_count),
        weight_mode=str(args.weight_mode),
        weight_scale=weight_scale,
        tail_days=int(args.tail_days),
        seed_scope=str(args.seed_scope),
        seed_pool_mode=str(args.seed_pool_mode),
        require_day0_activity=bool(args.require_day0_activity),
        farm_susceptibility=float(args.farm_susceptibility),
        farm_infectiousness=float(args.farm_infectiousness),
        reservoir_decay=float(args.reservoir_decay),
        reservoir_background=float(args.reservoir_background),
        reservoir_clip=float(args.reservoir_clip),
        farm_daily_import_prob=float(args.farm_daily_import_prob),
    )
    _validate_config(config)
    if config.weight_scale is None:
        auto_scale = _auto_weight_scale(observed_edges, weight_col, config.weight_mode)
        config = HybridSimulationConfig(**{**config.__dict__, "weight_scale": float(auto_scale)})
    return config


def _run_reality_check_for_sample(
    *,
    observed_scalar: pd.DataFrame,
    observed_daily: pd.DataFrame,
    observed_region_daily: pd.DataFrame,
    observed_diagnostics: dict[str, object],
    observed_pack: HybridPanelPack,
    synthetic_manifest: dict,
    synthetic_edges: pd.DataFrame,
    run_seeds: np.ndarray,
    initial_seed_sets: list[np.ndarray],
    config: HybridSimulationConfig,
    weight_col: Optional[str],
    node_types: dict[int, str],
    region_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    manifest: dict,
    focal_corop: str,
    corop_geojson_path: Optional[Path],
    output_dir: Path,
) -> dict[str, object]:
    sample_label = str(synthetic_manifest["sample_label"])
    setting_label = str(synthetic_manifest.get("setting_label") or _setting_label_from_sample_label(sample_label))
    sample_class = str(synthetic_manifest.get("sample_class") or _default_sample_class(setting_label))

    synthetic = synthetic_edges.copy()
    synthetic_pack = _build_hybrid_panel_pack(
        synthetic,
        label=sample_label,
        node_universe=list(observed_pack.node_universe),
        ts_values=list(observed_pack.ts_values),
        weight_col=weight_col,
        weight_mode=config.weight_mode,
        weight_scale=float(config.weight_scale or 1.0),
        node_types=node_types,
    )

    synthetic_scalar, synthetic_daily, synthetic_region_daily, synthetic_diagnostics = simulate_panel(
        synthetic_pack,
        run_seeds=run_seeds,
        initial_seed_sets=initial_seed_sets,
        config=config,
        region_frame=region_frame,
        node_frame=node_frame,
    )
    per_snapshot, summary, detailed = _compare_simulation_outputs(
        observed_scalar,
        observed_daily,
        observed_region_daily,
        synthetic_scalar,
        synthetic_daily,
        synthetic_region_daily,
        observed_diagnostics=observed_diagnostics,
        synthetic_diagnostics=synthetic_diagnostics,
        config=config,
        sample_label=sample_label,
        setting_label=setting_label,
        sample_class=sample_class,
    )
    summary["simulation_config"] = config.to_dict()
    outputs = write_report(
        per_snapshot=per_snapshot,
        summary=summary,
        output_dir=output_dir,
        sample_label=sample_label,
        detailed_outputs=detailed,
        node_frame=node_frame,
        manifest=manifest,
        corop_geojson_path=corop_geojson_path,
        focal_corop=focal_corop,
    )
    return {
        "sample_label": sample_label,
        "setting_label": setting_label,
        "sample_class": sample_class,
        "summary": summary,
        "outputs": outputs,
    }


def _compute_uncertainty_decomposition(
    reports: list[dict[str, object]],
    *,
    metric_names: list[str],
) -> pd.DataFrame:
    if not reports:
        return pd.DataFrame(columns=["metric"])

    observed_path = None
    for report in reports:
        outputs = dict(report.get("outputs", {}))
        observed_path = outputs.get("observed_outcomes") or observed_path
        if observed_path:
            break
    observed_frame = pd.read_csv(Path(str(observed_path))) if observed_path and Path(str(observed_path)).exists() else pd.DataFrame()

    sample_frames: list[pd.DataFrame] = []
    for report in reports:
        outputs = dict(report.get("outputs", {}))
        synthetic_path = outputs.get("synthetic_outcomes")
        if synthetic_path and Path(str(synthetic_path)).exists():
            sample_frames.append(pd.read_csv(Path(str(synthetic_path))))

    rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        observed = _frame_metric_values(observed_frame, metric_name)
        pooled_values: list[np.ndarray] = []
        sample_means: list[float] = []
        within_vars: list[float] = []

        for frame in sample_frames:
            values = _frame_metric_values(frame, metric_name)
            if len(values) == 0:
                continue
            pooled_values.append(values)
            sample_means.append(float(np.mean(values)))
            within_vars.append(float(np.var(values, ddof=0)))

        pooled = np.concatenate(pooled_values) if pooled_values else np.asarray([], dtype=float)
        between_var = float(np.var(np.asarray(sample_means, dtype=float), ddof=0)) if len(sample_means) > 1 else 0.0
        within_var = float(np.mean(within_vars)) if within_vars else np.nan
        total_var = between_var + within_var if np.isfinite(within_var) else np.nan
        network_share = float(between_var / total_var) if np.isfinite(total_var) and total_var > 0 else np.nan
        epidemic_share = float(within_var / total_var) if np.isfinite(total_var) and total_var > 0 else np.nan

        if len(observed) and len(pooled):
            observed_median = float(np.median(observed))
            observed_mean = float(np.mean(observed))
            pooled_q05 = float(np.quantile(pooled, 0.05))
            pooled_q95 = float(np.quantile(pooled, 0.95))
            pooled_percentile = float(np.mean(pooled <= observed_median))
            pooled_tail = float(min(1.0, 2.0 * min(pooled_percentile, 1.0 - pooled_percentile)))
            observed_median_in_pooled = float(pooled_q05 <= observed_median <= pooled_q95)
        else:
            observed_median = np.nan
            observed_mean = np.nan
            pooled_q05 = np.nan
            pooled_q95 = np.nan
            pooled_percentile = np.nan
            pooled_tail = np.nan
            observed_median_in_pooled = np.nan

        rows.append({
            "metric": metric_name,
            "sample_count": int(len(sample_means)),
            "pooled_replicate_count": int(len(pooled)),
            "observed_mean": observed_mean,
            "observed_median": observed_median,
            "pooled_synthetic_mean": float(np.mean(pooled)) if len(pooled) else np.nan,
            "pooled_synthetic_median": float(np.median(pooled)) if len(pooled) else np.nan,
            "pooled_synthetic_q05": pooled_q05,
            "pooled_synthetic_q95": pooled_q95,
            "between_sample_variance": between_var,
            "within_sample_variance_mean": within_var,
            "total_predictive_variance": total_var,
            "network_uncertainty_share": network_share,
            "epidemic_stochasticity_share": epidemic_share,
            "sample_mean_min": float(np.min(sample_means)) if sample_means else np.nan,
            "sample_mean_max": float(np.max(sample_means)) if sample_means else np.nan,
            "sample_mean_q05": float(np.quantile(sample_means, 0.05)) if sample_means else np.nan,
            "sample_mean_q95": float(np.quantile(sample_means, 0.95)) if sample_means else np.nan,
            "observed_median_in_pooled_synthetic_90pct": observed_median_in_pooled,
            "observed_median_pooled_percentile": pooled_percentile,
            "observed_median_pooled_tail_area": pooled_tail,
        })

    return pd.DataFrame(rows)


def aggregate_posterior_reports(
    reports: list[dict[str, object]],
    *,
    output_dir: Path,
    setting_label: str,
    node_frame: Optional[pd.DataFrame] = None,
    manifest: Optional[dict] = None,
    corop_geojson_path: Optional[Path] = None,
    focal_corop: str = "",
) -> dict[str, object]:
    if not reports:
        raise ValueError(f"No reports were provided for posterior aggregation of setting '{setting_label}'.")
    run_labels = [str(report.get("sample_label")) for report in reports]
    per_snapshot_frames = [pd.read_csv(Path(report["outputs"]["per_snapshot_csv"])) for report in reports]
    per_snapshot = _aggregate_grouped_numeric_frames(per_snapshot_frames, group_keys=["day_index", "ts"], run_labels=run_labels)
    summary = _aggregate_summary_payloads(setting_label, [dict(report["summary"]) for report in reports], run_labels=run_labels)

    detailed_outputs: dict[str, pd.DataFrame] = {}
    for detail_key, group_keys in DETAIL_GROUP_KEYS.items():
        if detail_key == "uncertainty_decomposition":
            continue
        frames: list[pd.DataFrame] = []
        for report in reports:
            outputs = dict(report.get("outputs", {}))
            detail_path = outputs.get(detail_key)
            if not detail_path:
                continue
            p = Path(str(detail_path))
            if p.exists():
                frames.append(pd.read_csv(p))
        if frames:
            detailed_outputs[detail_key] = _aggregate_grouped_numeric_frames(frames, group_keys=group_keys, run_labels=run_labels)

    observed_scalar_path = None
    for report in reports:
        outputs = dict(report.get("outputs", {}))
        observed_scalar_path = outputs.get("observed_outcomes") or observed_scalar_path
        if observed_scalar_path and Path(str(observed_scalar_path)).exists():
            break
    if observed_scalar_path and Path(str(observed_scalar_path)).exists():
        detailed_outputs["observed_outcomes"] = pd.read_csv(Path(str(observed_scalar_path)))

    synthetic_scalar_frames: list[pd.DataFrame] = []
    for report in reports:
        outputs = dict(report.get("outputs", {}))
        synthetic_scalar_path = outputs.get("synthetic_outcomes")
        if not synthetic_scalar_path:
            continue
        synthetic_scalar_file = Path(str(synthetic_scalar_path))
        if not synthetic_scalar_file.exists():
            continue
        frame = pd.read_csv(synthetic_scalar_file)
        frame["posterior_run_label"] = str(report.get("sample_label"))
        synthetic_scalar_frames.append(frame)
    if synthetic_scalar_frames:
        detailed_outputs["synthetic_outcomes"] = pd.concat(synthetic_scalar_frames, ignore_index=True, sort=False)

    uncertainty_decomposition = _compute_uncertainty_decomposition(
        reports,
        metric_names=[
            "farm_attack_rate",
            "farm_peak_prevalence",
            "farm_peak_day_index",
            "farm_duration_days",
            "farm_cumulative_incidence",
            "reservoir_total_auc",
        ],
    )
    if not uncertainty_decomposition.empty:
        detailed_outputs["uncertainty_decomposition"] = uncertainty_decomposition
        for metric_name, summary_prefix in [
            ("farm_attack_rate", "farm_attack_rate"),
            ("farm_peak_prevalence", "farm_peak_prevalence"),
            ("farm_duration_days", "farm_duration"),
            ("farm_peak_day_index", "farm_peak_day"),
            ("farm_cumulative_incidence", "farm_cumulative_incidence"),
        ]:
            summary[f"{summary_prefix}_network_uncertainty_share"] = _metric_lookup(uncertainty_decomposition, metric_name, "network_uncertainty_share")
            summary[f"{summary_prefix}_epidemic_stochasticity_share"] = _metric_lookup(uncertainty_decomposition, metric_name, "epidemic_stochasticity_share")
            summary[f"{summary_prefix}_observed_median_in_pooled_synthetic_90pct"] = _metric_lookup(uncertainty_decomposition, metric_name, "observed_median_in_pooled_synthetic_90pct")
            summary[f"{summary_prefix}_observed_median_pooled_tail_area"] = _metric_lookup(uncertainty_decomposition, metric_name, "observed_median_pooled_tail_area")

    outputs = write_report(
        per_snapshot=per_snapshot,
        summary=summary,
        output_dir=output_dir,
        sample_label=setting_label,
        detailed_outputs=detailed_outputs or None,
        node_frame=node_frame,
        manifest=manifest,
        corop_geojson_path=corop_geojson_path,
        focal_corop=focal_corop,
    )
    return {
        "sample_label": setting_label,
        "setting_label": setting_label,
        "sample_class": _default_sample_class(setting_label),
        "summary": summary,
        "outputs": outputs,
    }


def _major_scenarios() -> list[SimulationScenario]:
    return [
        SimulationScenario("baseline_seir_log1p", "Default SEIR with log1p weights.", {}),
        SimulationScenario("sir_log1p", "SIR farm dynamics with log1p weights.", {"model": "SIR"}),
        SimulationScenario("sis_log1p", "SIS farm dynamics with log1p weights.", {"model": "SIS"}),
        SimulationScenario("seir_binary_weights", "SEIR with binary contact intensity.", {"model": "SEIR", "weight_mode": "binary"}),
        SimulationScenario("seir_sqrt_weights", "SEIR with square-root weight compression.", {"model": "SEIR", "weight_mode": "sqrt"}),
        SimulationScenario("seir_linear_weights", "SEIR with raw linear weights.", {"model": "SEIR", "weight_mode": "linear"}),
        SimulationScenario("seir_all_farms_seed_scope", "SEIR seeded from the full farm pool.", {"model": "SEIR", "seed_scope": "all_farms"}),
        SimulationScenario("seir_common_day0_pool", "SEIR seeded from farms active on day 0 in both observed and synthetic panels.", {"model": "SEIR", "seed_pool_mode": "common_day0"}),
        SimulationScenario("seir_overall_pool", "SEIR seeded from farms active anywhere in the panel.", {"model": "SEIR", "seed_pool_mode": "overall"}),
    ]


def _namespace_with_overrides(args: argparse.Namespace, overrides: dict[str, object]) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _sort_by_available_scenario_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    available = [(column, ascending) for column, ascending in zip(SCENARIO_RANK_COLUMNS, SCENARIO_RANK_ASCENDING) if column in frame.columns]
    if not available:
        return frame.copy()
    columns = [column for column, _ in available]
    ascending = [flag for _, flag in available]
    return frame.sort_values(columns, ascending=ascending, na_position="last")


def _select_best_summary_row(summary_rows: pd.DataFrame) -> pd.Series:
    if summary_rows.empty:
        raise ValueError("Cannot select a best summary row from an empty frame.")
    sample_class_series = summary_rows["sample_class"] if "sample_class" in summary_rows.columns else pd.Series(["posterior_predictive"] * len(summary_rows))
    primary_rows = summary_rows.loc[sample_class_series.astype(str) == "posterior_predictive"].reset_index(drop=True)
    candidate_rows = primary_rows if not primary_rows.empty else summary_rows.reset_index(drop=True)
    ranked = _sort_by_available_scenario_rank_columns(candidate_rows).reset_index(drop=True)
    return ranked.iloc[0]


def _scenario_summary_row(
    *,
    scenario: SimulationScenario,
    summary_row: pd.Series,
    result: dict[str, object],
    config: HybridSimulationConfig,
) -> dict[str, object]:
    row = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "sample_label": scenario.name,
        "selected_setting_label": summary_row.get("sample_label"),
        "model": config.model,
        "weight_mode_config": config.weight_mode,
        "seed_scope": config.seed_scope,
        "seed_pool_mode": config.seed_pool_mode,
        "num_replicates": config.num_replicates,
        "report_path": result.get("report_path"),
        "output_dir": result.get("output_dir"),
    }
    for column in summary_row.index:
        target_column = "selected_sample_label" if column == "sample_label" else column
        row[target_column] = summary_row[column]
    return row


def _write_scenario_summary(
    rows: list[dict[str, object]],
    *,
    output_dir: Path,
) -> dict[str, Optional[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(rows)
    if summary_frame.empty:
        raise ValueError("Cannot write a scenario summary without any rows.")
    summary_frame = _sort_by_available_scenario_rank_columns(summary_frame).reset_index(drop=True)
    csv_path = output_dir / "scenario_summary.csv"
    md_path = output_dir / "scenario_summary.md"
    summary_frame.to_csv(csv_path, index=False)

    best = summary_frame.iloc[0]
    lines = [
        "# Simulation scenario summary",
        "",
        f"- Best scenario by curve fit then outcome distance: `{best['scenario_name']}`",
        f"- Best selected setting: `{best.get('selected_setting_label')}`",
        f"- Farm prevalence correlation: {_metric_text(best.get('farm_prevalence_curve_correlation'))}",
        f"- Farm incidence correlation: {_metric_text(best.get('farm_incidence_curve_correlation'))}",
        f"- Farm attack-rate Wasserstein: {_metric_text(best.get('farm_attack_rate_wasserstein'))}",
        f"- Farm peak-prevalence Wasserstein: {_metric_text(best.get('farm_peak_prevalence_wasserstein'))}",
        f"- Farm duration Wasserstein: {_metric_text(best.get('farm_duration_wasserstein'))}",
        "",
        "## Scenario ranking",
        "",
    ]
    for _, row in summary_frame.iterrows():
        lines.append(
            "- "
            + f"`{row['scenario_name']}` | prev corr {_metric_text(row.get('farm_prevalence_curve_correlation'))}"
            + f" | inc corr {_metric_text(row.get('farm_incidence_curve_correlation'))}"
            + f" | attack W1 {_metric_text(row.get('farm_attack_rate_wasserstein'))}"
            + f" | peak W1 {_metric_text(row.get('farm_peak_prevalence_wasserstein'))}"
            + f" | duration W1 {_metric_text(row.get('farm_duration_wasserstein'))}"
        )
    lines.extend(
        [
            "",
            f"- CSV summary: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    overview_path = write_all_samples_overview(summary_frame, output_dir)
    LOGGER.info(
        "Wrote simulation scenario summary | output_dir=%s | scenarios=%s | best=%s",
        output_dir,
        len(summary_frame),
        best["scenario_name"],
    )
    payload = {
        "scenario_summary_csv": str(csv_path),
        "scenario_summary_md": str(md_path),
        "best_scenario": str(best["scenario_name"]),
    }
    if overview_path is not None:
        payload["overview_path"] = str(overview_path)
    return payload


def _scenario_family(scenario_name: str) -> str:
    if "linear" in scenario_name or "sqrt" in scenario_name or "binary" in scenario_name:
        return "Weight transform"
    if scenario_name.startswith("sir") or scenario_name.startswith("sis") or scenario_name.startswith("baseline"):
        return "Farm model"
    return "Seed policy"


def _write_scenario_tradeoff_plot(summary_rows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_rows.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    frame = summary_rows.copy().reset_index(drop=True)
    frame["scenario_family"] = frame["scenario_name"].astype(str).map(_scenario_family)
    family_colors = {
        "Farm model": PLOT_COLORS["original"],
        "Weight transform": PLOT_COLORS["synthetic"],
        "Seed policy": PLOT_COLORS["accent"],
    }
    x_values = pd.to_numeric(frame["farm_duration_wasserstein"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_values = pd.to_numeric(frame["farm_prevalence_curve_correlation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    peak_values = pd.to_numeric(frame["farm_peak_prevalence_wasserstein"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    size_scale = 240.0 / max(float(np.nanmax(peak_values)), 0.02)
    sizes = np.clip(80.0 + peak_values * size_scale, 80.0, 380.0)

    fig, ax = plt.subplots(figsize=(11.5, 7.8), constrained_layout=True)
    _style_figure(fig, [ax])
    fig.suptitle("Scenario fit versus separation", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    for family_name, family_frame in frame.groupby("scenario_family", sort=False):
        idx = family_frame.index.to_numpy(dtype=int)
        ax.scatter(
            x_values[idx],
            y_values[idx],
            s=sizes[idx],
            color=family_colors.get(family_name, PLOT_COLORS["accent"]),
            alpha=0.86,
            edgecolors="white",
            linewidths=1.0,
            label=family_name,
        )
        for row_index in idx:
            ax.text(
                x_values[row_index] + 0.02,
                y_values[row_index],
                str(frame.loc[row_index, "scenario_name"]).replace("_", "\n"),
                fontsize=8.2,
                color=PLOT_COLORS["text"],
                va="center",
            )
    ax.set_xlabel("Farm duration W1")
    ax.set_ylabel("Farm prevalence correlation")
    ax.set_title("Upper-left is tighter fit; larger markers mean larger peak-prevalence mismatch")
    _style_legend(ax.legend(loc="lower right"))
    output_path = output_dir / "scenario_tradeoff.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_scenario_metric_heatmap(summary_rows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_rows.empty:
        return None
    plt = _load_matplotlib()
    if plt is None:
        return None
    frame = summary_rows.copy().reset_index(drop=True)
    metric_specs = [
        ("farm_prevalence_curve_correlation", "Prev corr", True),
        ("farm_incidence_curve_correlation", "Inc corr", True),
        ("farm_cumulative_incidence_curve_correlation", "Cum corr", True),
        ("reservoir_total_curve_correlation", "Reservoir corr", True),
        ("farm_attack_probability_correlation", "Farm risk corr", True),
        ("region_reservoir_spatial_correlation_mean", "Region spatial corr", True),
        ("region_reservoir_field_energy_distance_mean", "Region ED", False),
        ("farm_attack_rate_wasserstein", "Attack W1", False),
        ("farm_peak_prevalence_wasserstein", "Peak W1", False),
        ("farm_duration_wasserstein", "Duration W1", False),
    ]
    available_specs = [spec for spec in metric_specs if spec[0] in frame.columns]
    if not available_specs:
        return None
    score_columns = []
    annotation_columns = []
    for column, _, higher_is_better in available_specs:
        series = pd.to_numeric(frame[column], errors="coerce")
        score = series.rank(method="average", pct=True, ascending=higher_is_better).fillna(0.0)
        score_columns.append(score.to_numpy(dtype=float))
        annotation_columns.append(series.fillna(np.nan).to_numpy(dtype=float))
    score_matrix = np.column_stack(score_columns)
    annotation_matrix = np.column_stack(annotation_columns)

    fig_height = max(5.6, 0.55 * len(frame) + 3.0)
    fig, ax = plt.subplots(figsize=(12.8, fig_height), constrained_layout=True)
    _style_figure(fig, [ax])
    fig.suptitle("Scenario metric matrix", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])
    image = ax.imshow(score_matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(available_specs)), [label for _, label, _ in available_specs])
    ax.set_yticks(np.arange(len(frame)), frame["scenario_name"].astype(str).tolist())
    for row_index in range(score_matrix.shape[0]):
        for column_index in range(score_matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                _metric_text(annotation_matrix[row_index, column_index], digits=3),
                ha="center",
                va="center",
                fontsize=8.2,
                color=PLOT_COLORS["text"],
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    colorbar.ax.set_ylabel("Within-metric rank score", color=PLOT_COLORS["text"])
    output_path = output_dir / "scenario_metric_matrix.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _scenario_interpretation(row: pd.Series) -> str:
    prevalence_corr = float(pd.to_numeric(pd.Series([row.get("farm_prevalence_curve_correlation")]), errors="coerce").fillna(0.0).iloc[0])
    attack_w1 = float(pd.to_numeric(pd.Series([row.get("farm_attack_rate_wasserstein")]), errors="coerce").fillna(0.0).iloc[0])
    peak_w1 = float(pd.to_numeric(pd.Series([row.get("farm_peak_prevalence_wasserstein")]), errors="coerce").fillna(0.0).iloc[0])
    duration_w1 = float(pd.to_numeric(pd.Series([row.get("farm_duration_wasserstein")]), errors="coerce").fillna(0.0).iloc[0])
    if prevalence_corr >= 0.995 and peak_w1 <= 0.02 and duration_w1 <= 0.30:
        return "This is the closest match to the observed panel. Use it as the reference scenario."
    if prevalence_corr < 0.60 or duration_w1 >= 5.0:
        return "This is the largest departure from the observed panel. Use it when clear separation between scenarios is needed."
    if prevalence_corr < 0.98 or duration_w1 >= 0.80:
        return "This scenario creates moderate separation from the observed panel while keeping the trajectories interpretable."
    if attack_w1 <= 0.001 and peak_w1 <= 0.05:
        return "This remains a close-fit scenario, though not as tight as the strongest seed-pool variants."
    return "This scenario sits near the middle of the sweep. Read it with the delta and distribution figures before drawing conclusions."


def write_scenario_comparison_report(
    run_dir: Path,
    scenario_dir: Path,
    summary_rows: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    scenario_dir = Path(scenario_dir).expanduser().resolve()
    output_path = Path(output_path) if output_path is not None else scenario_dir / "scientific_validation_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {}
    dataset_name = str(manifest.get("dataset", "Dataset"))
    summary_rows = _sort_by_available_scenario_rank_columns(summary_rows).reset_index(drop=True)
    best_row = summary_rows.iloc[0]
    stress_row = summary_rows.sort_values(
        ["farm_duration_wasserstein", "farm_peak_prevalence_wasserstein", "farm_attack_rate_wasserstein"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    overview_path = write_all_samples_overview(summary_rows, scenario_dir)
    tradeoff_path = _write_scenario_tradeoff_plot(summary_rows, scenario_dir)
    heatmap_path = _write_scenario_metric_heatmap(summary_rows, scenario_dir)
    scenario_browser_payload = _build_scenario_switch_payload(
        summary_rows,
        scenario_root=scenario_dir,
        current_report_path=output_path,
        default_scenario_name=str(best_row["scenario_name"]),
        comparison_report_path=output_path,
    )
    scenario_browser_section_html = _render_scenario_browser_section(
        scenario_browser_payload,
        widget_id="scenario_browser",
    )
    scenario_browser_script_html = _scenario_browser_script("scenario_browser") if scenario_browser_section_html else ""

    def fmt_from_row(row: pd.Series, key: str, *, digits: int = 3) -> str:
        return _metric_text(row.get(key), digits=digits)

    def column_median(column: str) -> str:
        if column not in summary_rows.columns:
            return "n/a"
        values = pd.to_numeric(summary_rows[column], errors="coerce").dropna()
        if values.empty:
            return "n/a"
        return f"{float(values.median()):.3f}"

    def relative_asset(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.exists():
            return None
        try:
            return candidate.relative_to(scenario_dir).as_posix()
        except Exception:
            return candidate.name

    def figure_tag(relative_path: Optional[str], caption: str) -> str:
        if not relative_path:
            return ""
        return (
            "<figure class='figure-card'>"
            f"<div class='figure-frame'><img src='{html.escape(relative_path)}' alt='{html.escape(caption)}'></div>"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )

    def scenario_card(row: pd.Series, rank_index: int) -> str:
        scenario_name = str(row["scenario_name"])
        setting_label = str(row.get("selected_setting_label") or row.get("selected_sample_label") or "")
        scenario_path = Path(str(row["output_dir"]))
        dashboard_path = relative_asset(scenario_path / f"{setting_label}_dashboard.png")
        delta_path = relative_asset(scenario_path / f"{setting_label}_delta.png")
        daily_mean_path = relative_asset(scenario_path / f"{setting_label}_daily_mean_compare.png")
        distribution_path = relative_asset(scenario_path / f"{setting_label}_distribution.png")
        parity_path = relative_asset(scenario_path / f"{setting_label}_parity.png")
        channel_path = relative_asset(scenario_path / f"{setting_label}_channel_diagnostics.png")
        farm_spatial_path = relative_asset(scenario_path / f"{setting_label}_farm_spatial_overview.png")
        report_path = relative_asset(Path(str(row["report_path"])))
        figures = "".join(
            fig for fig in [
                figure_tag(dashboard_path, f"{scenario_name} trajectory dashboard"),
                figure_tag(delta_path, f"{scenario_name} trajectory median deltas"),
                figure_tag(daily_mean_path, f"{scenario_name} daily mean trajectories"),
                figure_tag(distribution_path, f"{scenario_name} replicate distributions"),
                figure_tag(parity_path, f"{scenario_name} parity summary"),
                figure_tag(channel_path, f"{scenario_name} hybrid-channel diagnostics"),
                figure_tag(farm_spatial_path, f"{scenario_name} farm-level spatial validation"),
            ]
            if fig
        )
        report_link = (
            f"<p class='card-link'><a href='{html.escape(report_path)}'>Open the full scenario report</a></p>"
            if report_path else ""
        )
        pills = [
            f"<span class='metric-pill'>Prev corr {fmt_from_row(row, 'farm_prevalence_curve_correlation')}</span>",
            f"<span class='metric-pill'>Prev mean corr {fmt_from_row(row, 'farm_prevalence_mean_curve_correlation')}</span>",
            f"<span class='metric-pill'>Attack W1 {fmt_from_row(row, 'farm_attack_rate_wasserstein')}</span>",
            f"<span class='metric-pill'>Peak W1 {fmt_from_row(row, 'farm_peak_prevalence_wasserstein')}</span>",
            f"<span class='metric-pill'>Duration W1 {fmt_from_row(row, 'farm_duration_wasserstein')}</span>",
            f"<span class='metric-pill'>Farm risk corr {fmt_from_row(row, 'farm_attack_probability_correlation')}</span>",
        ]
        if "farm_prevalence_interval_coverage" in row.index:
            pills.append(
                f"<span class='metric-pill'>Prev 90% cov {fmt_from_row(row, 'farm_prevalence_interval_coverage')}</span>"
            )
        if "farm_attack_rate_network_uncertainty_share" in row.index:
            pills.append(
                f"<span class='metric-pill'>Attack net share {fmt_from_row(row, 'farm_attack_rate_network_uncertainty_share')}</span>"
            )
        pill_html = "".join(pills)
        return (
            "<section class='scenario-card'>"
            f"<div class='scenario-head'><div class='scenario-rank'>#{rank_index}</div>"
            f"<div><h3>{html.escape(scenario_name)}</h3><p class='scenario-note'>{html.escape(str(row.get('scenario_description', '')))}</p></div></div>"
            f"<div class='metric-pill-row'>{pill_html}</div>"
            f"<p class='scenario-note'>{html.escape(_scenario_interpretation(row))}</p>"
            "<div class='figure-grid figure-grid-two'>"
            f"{figures}"
            "</div>"
            f"{report_link}"
            "</section>"
        )

    scenario_cards = "\n".join(scenario_card(row, idx + 1) for idx, (_, row) in enumerate(summary_rows.iterrows()))
    median_prevalence_coverage = column_median("farm_prevalence_interval_coverage")
    median_attack_network_share = column_median("farm_attack_rate_network_uncertainty_share")

    scorecard_columns = [
        ("scenario_name", "Scenario"),
        ("farm_prevalence_curve_correlation", "Prev corr"),
        ("farm_incidence_curve_correlation", "Inc corr"),
        ("farm_prevalence_mean_curve_correlation", "Prev mean corr"),
        ("farm_incidence_mean_curve_correlation", "Inc mean corr"),
        ("farm_cumulative_incidence_curve_correlation", "Cum corr"),
        ("reservoir_total_curve_correlation", "Reservoir corr"),
        ("farm_attack_probability_correlation", "Farm risk corr"),
        ("region_reservoir_field_energy_distance_mean", "Region ED"),
        ("farm_attack_rate_wasserstein", "Attack W1"),
        ("farm_peak_prevalence_wasserstein", "Peak W1"),
        ("farm_duration_wasserstein", "Duration W1"),
        ("farm_prevalence_interval_coverage", "Prev 90% cov"),
        ("farm_attack_rate_network_uncertainty_share", "Attack net share"),
    ]
    available_scorecard = [(key, label) for key, label in scorecard_columns if key in summary_rows.columns]

    daily_calibration_columns = [
        ("scenario_name", "Scenario"),
        ("farm_prevalence_interval_coverage", "Prev 90% cov"),
        ("farm_incidence_interval_coverage", "Inc 90% cov"),
        ("farm_cumulative_incidence_interval_coverage", "Cum 90% cov"),
        ("reservoir_total_interval_coverage", "Reservoir 90% cov"),
    ]
    available_daily = [(key, label) for key, label in daily_calibration_columns if key in summary_rows.columns]
    daily_mean_columns = [
        ("scenario_name", "Scenario"),
        ("farm_prevalence_mean_curve_correlation", "Prev mean corr"),
        ("farm_incidence_mean_curve_correlation", "Inc mean corr"),
        ("farm_cumulative_incidence_mean_curve_correlation", "Cum mean corr"),
        ("reservoir_total_mean_curve_correlation", "Reservoir mean corr"),
        ("farm_prevalence_mean_curve_mae", "Prev mean MAE"),
        ("farm_incidence_mean_curve_mae", "Inc mean MAE"),
        ("farm_cumulative_incidence_mean_curve_mae", "Cum mean MAE"),
        ("reservoir_total_mean_curve_mae", "Reservoir mean MAE"),
    ]
    available_daily_mean = [(key, label) for key, label in daily_mean_columns if key in summary_rows.columns]

    scalar_uncertainty_columns = [
        ("scenario_name", "Scenario"),
        ("farm_attack_rate_observed_median_in_pooled_synthetic_90pct", "Attack in 90% band"),
        ("farm_attack_rate_observed_median_pooled_tail_area", "Attack tail"),
        ("farm_peak_prevalence_observed_median_in_pooled_synthetic_90pct", "Peak in 90% band"),
        ("farm_peak_prevalence_observed_median_pooled_tail_area", "Peak tail"),
        ("farm_peak_day_observed_median_in_pooled_synthetic_90pct", "Peak day in 90% band"),
        ("farm_peak_day_observed_median_pooled_tail_area", "Peak day tail"),
        ("farm_duration_observed_median_in_pooled_synthetic_90pct", "Duration in 90% band"),
        ("farm_duration_observed_median_pooled_tail_area", "Duration tail"),
        ("farm_attack_rate_network_uncertainty_share", "Attack net share"),
        ("farm_peak_prevalence_network_uncertainty_share", "Peak net share"),
        ("farm_peak_day_network_uncertainty_share", "Peak day net share"),
        ("farm_duration_network_uncertainty_share", "Duration net share"),
        ("farm_cumulative_incidence_network_uncertainty_share", "Cum-inc net share"),
    ]
    available_scalar_uncertainty = [(key, label) for key, label in scalar_uncertainty_columns if key in summary_rows.columns]
    html_parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(dataset_name)} scenario comparison report</title>",
        "<style>",
        """
        :root {
          --bg: #eef3f8;
          --panel: #ffffff;
          --panel-soft: #f7fafc;
          --text: #22313f;
          --muted: #5f6f7f;
          --line: #dce3ea;
          --blue: #2a6fbb;
        }
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
        .page { max-width: 1340px; margin: 0 auto; padding: 30px 22px 60px; }
        .hero { margin-bottom: 24px; }
        .eyebrow { color: var(--blue); text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; font-size: 12px; margin-bottom: 10px; }
        h1 { margin: 0 0 8px; font-size: 32px; }
        h2 { margin: 0 0 12px; font-size: 22px; }
        h3 { margin: 0; font-size: 19px; }
        .subtitle { color: var(--muted); max-width: 980px; line-height: 1.6; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 14px; margin-top: 18px; }
        .summary-card, .scenario-card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; }
        .summary-card h3 { margin: 0 0 6px; font-size: 12px; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
        .metric-value { font-size: clamp(22px, 3vw, 34px); font-weight: 700; line-height: 1.05; overflow-wrap: anywhere; }
        .section { margin-top: 28px; }
        .section-heading { display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 12px; }
        .section-heading h2 { margin: 0; }
        .figure-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
        .figure-grid-two { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
        .figure-card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; }
        .figure-frame { padding: 10px; border-radius: 14px; background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%); border: 1px solid var(--line); }
        .figure-card img { width: 100%; border-radius: 12px; display: block; background: var(--panel-soft); }
        .figure-card figcaption { margin-top: 10px; color: var(--muted); font-size: 13px; }
        .scenario-stack { display: grid; gap: 18px; }
        .scenario-head { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 12px; }
        .scenario-rank { min-width: 36px; height: 36px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; background: #d9e8f6; color: var(--blue); font-weight: 700; }
        .scenario-note { color: var(--muted); line-height: 1.55; margin: 6px 0 0; }
        .metric-pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 12px; }
        .metric-pill { padding: 7px 10px; border-radius: 999px; background: #eef4f8; color: var(--text); font-size: 13px; }
        .report-table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
        .report-table th, .report-table td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; }
        .report-table th { background: #eef4f8; font-weight: 700; }
        .report-table tr:last-child td { border-bottom: 0; }
        .table-wrap { overflow-x: auto; }
        .explain-widget { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
        .explain-button {
          appearance: none;
          border: 1px solid rgba(42,111,187,0.24);
          background: linear-gradient(180deg, #fafdff 0%, #edf4f9 100%);
          color: var(--blue);
          border-radius: 999px;
          padding: 8px 12px;
          font: inherit;
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.02em;
          cursor: pointer;
          box-shadow: 0 6px 14px rgba(42,111,187,0.08);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .explain-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 10px 20px rgba(42,111,187,0.12);
          border-color: rgba(42,111,187,0.36);
        }
        .explain-button[aria-expanded="true"] { background: linear-gradient(180deg, #edf4f9 0%, #e2edf6 100%); }
        .explain-panel {
          width: 100%;
          padding: 12px 14px;
          border-radius: 14px;
          border: 1px solid var(--line);
          background: #f7fafc;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.6;
        }
        .explain-panel p { margin: 0; color: var(--muted); }
        .card-link { margin: 14px 0 0; }
        .card-link a { color: var(--blue); text-decoration: none; font-weight: 600; }
        @media (max-width: 1024px) { .summary-grid { grid-template-columns: 1fr 1fr; } .figure-grid-two { grid-template-columns: 1fr; } }
        @media (max-width: 640px) { .summary-grid { grid-template-columns: 1fr; } }
        """,
        _scenario_navigation_css(),
        "</style>",
        "</head>",
        "<body>",
        "<main class='page'>",
        "<section class='hero'>",
        "<div class='eyebrow'>Scenario comparison</div>",
        f"<h1>{html.escape(dataset_name)} hybrid transmission scenario report</h1>",
        "<p class='subtitle'>This report compares the main simulation scenarios run on the selected posterior setting from the diagnostics sweep. It brings the ranking table, the scenario-level figures, and the reading notes into one place so reviewers can compare close-fit settings against larger-deviation settings without leaving the report. It also reports posterior-predictive interval coverage and network-uncertainty shares in the same view.</p>",
        "<div class='summary-grid'>",
        "<div class='summary-card'><h3>Scenarios</h3><div class='metric-value'>" + str(int(len(summary_rows))) + "</div></div>",
        "<div class='summary-card'><h3>Best fit</h3><div class='metric-value'>" + html.escape(str(best_row["scenario_name"])) + "</div></div>",
        "<div class='summary-card'><h3>Strongest separator</h3><div class='metric-value'>" + html.escape(str(stress_row["scenario_name"])) + "</div></div>",
        "<div class='summary-card'><h3>Median prev. 90% cov</h3><div class='metric-value'>" + html.escape(median_prevalence_coverage) + "</div></div>",
        "<div class='summary-card'><h3>Median attack net share</h3><div class='metric-value'>" + html.escape(median_attack_network_share) + "</div></div>",
        "</div>",
        "</section>",
        scenario_browser_section_html,
        "<section class='section'>",
        _render_section_heading(
            "Scenario comparison",
            control_id="scenario_comparison_explain",
            explain_text=_simulation_notes_text(_simulation_figure_reading_notes(include_scenario_overview=True)),
            button_label="How to read this figure",
        ),
        "<div class='figure-grid figure-grid-two'>",
        figure_tag(relative_asset(overview_path), "Across-scenario summary of the main transmission metrics."),
        figure_tag(relative_asset(tradeoff_path), "Scenario fit versus separation. Upper-left is tighter fit; larger markers mean larger peak mismatch."),
        figure_tag(relative_asset(heatmap_path), "Scenario metric matrix with within-metric rank score coloring."),
        "</div>",
        "</section>",
        "<section class='section'>",
        _render_section_heading(
            "Scenario scorecard",
            control_id="scenario_scorecard_explain",
            explain_text=_simulation_notes_text(_simulation_table_reading_notes(include_scenario_scorecard=True)),
            button_label="How to read this table",
        ),
        "<div class='table-wrap'>",
        "<table class='report-table'>",
        "<thead><tr>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_scorecard) + "</tr></thead>",
        "<tbody>",
    ]
    for _, row in summary_rows.iterrows():
        cells = []
        for key, _ in available_scorecard:
            if key == "scenario_name":
                cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
            else:
                cells.append(f"<td>{fmt_from_row(row, key)}</td>")
        html_parts.append("<tr>" + "".join(cells) + "</tr>")
    html_parts.extend(
        [
            "</tbody></table>",
            "</div>",
            "</section>",
        ]
    )

    if len(available_daily) > 1:
        html_parts.extend(
            [
                "<section class='section'>",
                _render_section_heading(
                    "Daily calibration",
                    control_id="daily_calibration_explain",
                    explain_text="Coverage near 0.9 means the observed daily path usually sits inside the synthetic 90% band. Much lower values point to bias or bands that are too narrow, while values pinned near 1 can mean the bands are wide enough to hide real mismatch.",
                    button_label="How to read this table",
                ),
                "<p class='subtitle'>These columns report daywise posterior-predictive interval coverage for the main trajectory families. Values near 1 mean the observed daily curve usually falls inside the synthetic 90% interval.</p>",
                "<div class='table-wrap'>",
                "<table class='report-table'>",
                "<thead><tr>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_daily) + "</tr></thead>",
                "<tbody>",
            ]
        )
        for _, row in summary_rows.iterrows():
            cells = []
            for key, _ in available_daily:
                if key == "scenario_name":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            html_parts.append("<tr>" + "".join(cells) + "</tr>")
        html_parts.extend(
            [
                "</tbody></table>",
                "</div>",
                "</section>",
            ]
        )

    if len(available_daily_mean) > 1:
        html_parts.extend(
            [
                "<section class='section'>",
                _render_section_heading(
                    "Daily mean comparison",
                    control_id="daily_mean_comparison_explain",
                    explain_text="These columns compare the mean trajectory across replicates for each day. Higher correlations mean the timing and shape line up well, while lower mean absolute error means the daily gap stays small.",
                    button_label="How to read this table",
                ),
                "<p class='subtitle'>These columns summarize the daywise mean trajectory fit. They are helpful when the median stays flat at zero for long stretches and hides smaller shifts in timing or scale.</p>",
                "<div class='table-wrap'>",
                "<table class='report-table'>",
                "<thead><tr>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_daily_mean) + "</tr></thead>",
                "<tbody>",
            ]
        )
        for _, row in summary_rows.iterrows():
            cells = []
            for key, _ in available_daily_mean:
                if key == "scenario_name":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            html_parts.append("<tr>" + "".join(cells) + "</tr>")
        html_parts.extend(
            [
                "</tbody></table>",
                "</div>",
                "</section>",
            ]
        )

    if len(available_scalar_uncertainty) > 1:
        html_parts.extend(
            [
                "<section class='section'>",
                _render_section_heading(
                    "Scalar calibration and uncertainty",
                    control_id="scalar_uncertainty_explain",
                    explain_text="An in-band value of 1 means the observed median falls inside the pooled synthetic 90% interval. Tail area close to 1 means the observed median sits near the center of the synthetic distribution, while values near 0 place it in the tails. Network uncertainty share near 1 means panel-to-panel network variation dominates predictive variance.",
                    button_label="How to read this table",
                ),
                "<p class='subtitle'>These columns show whether the observed scalar endpoints fall inside pooled synthetic 90% intervals and how much predictive variance comes from between-panel network differences rather than within-panel epidemic stochasticity.</p>",
                "<div class='table-wrap'>",
                "<table class='report-table'>",
                "<thead><tr>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in available_scalar_uncertainty) + "</tr></thead>",
                "<tbody>",
            ]
        )
        for _, row in summary_rows.iterrows():
            cells = []
            for key, _ in available_scalar_uncertainty:
                if key == "scenario_name":
                    cells.append(f"<td>{html.escape(str(row.get(key)))}</td>")
                else:
                    cells.append(f"<td>{fmt_from_row(row, key)}</td>")
            html_parts.append("<tr>" + "".join(cells) + "</tr>")
        html_parts.extend(
            [
                "</tbody></table>",
                "</div>",
                "</section>",
            ]
        )

    html_parts.extend(
        [
            "<section class='section'>",
            _render_section_heading(
                "Scenario gallery",
                control_id="scenario_gallery_explain",
                explain_text=_simulation_notes_text(_simulation_figure_reading_notes()),
                button_label="How to read this figure",
            ),
            "<div class='scenario-stack'>",
            scenario_cards,
            "</div>",
            "</section>",
            "<script>",
            """
            document.querySelectorAll('.explain-button').forEach((button) => {
              button.addEventListener('click', () => {
                const panel = document.getElementById(button.dataset.explainTarget);
                if (!panel) return;
                const expanded = button.getAttribute('aria-expanded') === 'true';
                button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
                panel.hidden = expanded;
              });
            });
            """,
            scenario_browser_script_html,
            "</script>",
            "</main></body></html>",
        ]
    )
    output_path.write_text("\n".join(part for part in html_parts if part), encoding="utf-8")
    LOGGER.info("Wrote scenario comparison report to %s", output_path)
    return output_path


def _run_single_configuration(
    args: argparse.Namespace,
    *,
    output_dir: Optional[Path] = None,
    update_manifest: bool = True,
) -> dict[str, object]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "simulation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_run_manifest(run_dir)
    directed = bool(manifest.get("directed", True))
    if not directed:
        raise ValueError("The current hybrid simulation framework expects directed temporal panels.")
    observed_edges_path = Path(str(manifest.get("filtered_input_edges_path") or run_dir / "input_edges_filtered.csv"))
    if not observed_edges_path.exists():
        raise FileNotFoundError(f"Observed edge table not found: {observed_edges_path}")
    observed_raw = pd.read_csv(observed_edges_path)

    weight_model = manifest.get("weight_model") or {}
    inferred_weight_col = str(weight_model.get("output_column") or weight_model.get("input_column") or "").strip() or None
    weight_col = args.weight_col or inferred_weight_col
    if weight_col and weight_col not in observed_raw.columns:
        LOGGER.warning("Requested weight column %s is not present in the observed edge table; falling back to unweighted transmission.", weight_col)
        weight_col = None

    observed = canonicalise_edge_frame(observed_raw, directed=directed, weight_col=weight_col)
    node_frame = _load_hybrid_node_frame(run_dir, manifest)
    node_types = _build_node_type_map(node_frame)
    focal_corop = _focal_corop_from_node_frame(node_frame)
    corop_geojson_path = _resolve_corop_geojson_path(manifest, getattr(args, "corop_geojson", None))

    sample_manifests = _filter_sample_manifests(
        _discover_sample_manifests(run_dir, manifest),
        setting_labels=getattr(args, "setting_label", None),
        sample_label_pattern=getattr(args, "sample_label_pattern", None),
    )
    prepared_samples: list[tuple[dict[str, object], pd.DataFrame]] = []
    synthetic_node_ids: set[int] = set()
    synthetic_ts_values: set[int] = set()
    for sample_manifest in sample_manifests:
        synthetic_raw = pd.read_csv(Path(str(sample_manifest["synthetic_edges_csv"])))
        synthetic_frame = canonicalise_edge_frame(synthetic_raw, directed=directed, weight_col=weight_col)
        prepared_samples.append((sample_manifest, synthetic_frame))
        synthetic_node_ids.update(synthetic_frame["u"].astype(int).tolist())
        synthetic_node_ids.update(synthetic_frame["i"].astype(int).tolist())
        synthetic_ts_values.update(synthetic_frame["ts"].astype(int).tolist())

    node_universe = sorted(
        set(observed["u"].astype(int).tolist())
        | set(observed["i"].astype(int).tolist())
        | synthetic_node_ids
        | set(int(node_id) for node_id in node_types)
    )
    ts_values = sorted(set(observed["ts"].astype(int).tolist()) | synthetic_ts_values)
    if not ts_values:
        raise ValueError("The observed and synthetic panels contain no timestamps.")

    config = _build_config(args, observed_edges=observed, weight_col=weight_col)
    selected_settings = sorted({str(payload.get("setting_label") or payload.get("sample_label")) for payload in sample_manifests})
    LOGGER.info(
        "Running hybrid transmission validation | run_dir=%s | directed=%s | weight_col=%s | model=%s | replicates=%s | settings=%s | samples=%s | output_dir=%s",
        run_dir,
        directed,
        weight_col,
        config.model,
        config.num_replicates,
        len(selected_settings),
        len(sample_manifests),
        output_dir,
    )
    LOGGER.debug("Selected generated settings | labels=%s", selected_settings)

    observed_pack = _build_hybrid_panel_pack(
        observed,
        label="observed",
        node_universe=node_universe,
        ts_values=ts_values,
        weight_col=weight_col,
        weight_mode=config.weight_mode,
        weight_scale=float(config.weight_scale or 1.0),
        node_types=node_types,
    )
    region_frame = _build_region_node_frame(node_frame, node_universe=node_universe, node_types=node_types)
    synthetic_day0_activity_mask = _build_common_day0_activity_mask(
        synthetic_frames=[frame for _, frame in prepared_samples],
        node_universe=node_universe,
        day0_ts=int(ts_values[0]),
    )
    run_seeds, initial_seed_sets = _build_initial_seed_sets(
        observed_pack=observed_pack,
        synthetic_day0_activity_mask=synthetic_day0_activity_mask,
        config=config,
    )
    observed_scalar, observed_daily, observed_region_daily, observed_diagnostics = simulate_panel(
        observed_pack,
        run_seeds=run_seeds,
        initial_seed_sets=initial_seed_sets,
        config=config,
        region_frame=region_frame,
        node_frame=node_frame,
    )
    observed_scalar.to_csv(output_dir / "observed_outcomes.csv", index=False)
    observed_daily.to_csv(output_dir / "observed_daily.csv", index=False)
    observed_region_daily.to_csv(output_dir / "observed_region_daily.csv", index=False)

    reports = []
    for sample_manifest, synthetic_frame in prepared_samples:
        report = _run_reality_check_for_sample(
            observed_scalar=observed_scalar,
            observed_daily=observed_daily,
            observed_region_daily=observed_region_daily,
            observed_diagnostics=observed_diagnostics,
            observed_pack=observed_pack,
            synthetic_manifest=sample_manifest,
            synthetic_edges=synthetic_frame,
            run_seeds=run_seeds,
            initial_seed_sets=initial_seed_sets,
            config=config,
            weight_col=weight_col,
            node_types=node_types,
            region_frame=region_frame,
            node_frame=node_frame,
            manifest=manifest,
            focal_corop=focal_corop,
            corop_geojson_path=corop_geojson_path,
            output_dir=output_dir,
        )
        reports.append(report)

    run_summary_rows = pd.DataFrame(
        [_summary_payload_to_row(str(report["sample_label"]), dict(report["summary"])) for report in reports]
    ).dropna(how="all")
    if len(run_summary_rows):
        run_summary_rows.to_csv(output_dir / "all_samples_summary.csv", index=False)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for report in reports:
        grouped[str(report.get("setting_label") or report["sample_label"])].append(report)

    setting_reports = []
    for setting_label, setting_group in grouped.items():
        if len(setting_group) > 1:
            setting_reports.append(
                aggregate_posterior_reports(
                    setting_group,
                    output_dir=output_dir,
                    setting_label=setting_label,
                    node_frame=node_frame,
                    manifest=manifest,
                    corop_geojson_path=corop_geojson_path,
                    focal_corop=focal_corop,
                )
            )
        else:
            setting_reports.append(setting_group[0])

    setting_summary_rows = pd.DataFrame(
        [_summary_payload_to_row(str(report["sample_label"]), dict(report["summary"])) for report in setting_reports]
    ).dropna(how="all")
    if len(setting_summary_rows):
        setting_summary_rows.to_csv(output_dir / "setting_posterior_summary.csv", index=False)

    overview_path = write_all_samples_overview(setting_summary_rows if len(setting_summary_rows) else run_summary_rows, output_dir)
    report_path = write_scientific_validation_report(
        run_dir,
        simulation_dir=output_dir,
        output_path=output_dir / "scientific_validation_report.html",
    )

    if update_manifest:
        latest_manifest = load_run_manifest(run_dir)
        latest_manifest["simulation"] = {
            "output_dir": str(output_dir),
            "overview_path": str(overview_path) if overview_path is not None else None,
            "report_path": str(report_path),
            "config": config.to_dict(),
            "run_count": int(len(reports)),
            "setting_count": int(len(setting_reports)),
            "selected_setting_labels": selected_settings,
            "hybrid_mode": "farm_compartments_with_region_reservoirs",
            "primary_endpoints": [
                "farm_attack_rate",
                "farm_peak_prevalence",
                "farm_peak_day_index",
                "farm_duration_days",
            ],
            "region_outputs": list(REGION_DAILY_METRICS),
            "corop_geojson_path": str(corop_geojson_path) if corop_geojson_path is not None else None,
            "focal_corop": focal_corop,
        }
        _save_json(latest_manifest, Path(latest_manifest["manifest_path"]))

    return {
        "output_dir": str(output_dir),
        "overview_path": str(overview_path) if overview_path is not None else None,
        "report_path": str(report_path),
        "run_count": int(len(reports)),
        "setting_count": int(len(setting_reports)),
        "config": config,
        "run_summary_rows": run_summary_rows,
        "setting_summary_rows": setting_summary_rows,
    }


def run_command(args: argparse.Namespace) -> dict[str, object]:
    if str(getattr(args, "scenario_set", "single")) != "major":
        return _run_single_configuration(args)

    return run_scenario_set(args, _major_scenarios())


def run_scenario_set(args: argparse.Namespace, scenarios: list[SimulationScenario]) -> dict[str, object]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    scenario_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "simulation_scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)

    scenario_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_args = _namespace_with_overrides(args, scenario.overrides)
        scenario_output_dir = scenario_root / scenario.name
        LOGGER.info("Running simulation scenario | name=%s | output_dir=%s", scenario.name, scenario_output_dir)
        result = _run_single_configuration(
            scenario_args,
            output_dir=scenario_output_dir,
            update_manifest=False,
        )
        summary_rows = result["setting_summary_rows"]
        if not isinstance(summary_rows, pd.DataFrame) or summary_rows.empty:
            raise ValueError(f"Scenario {scenario.name} did not produce any setting summary rows.")
        best_row = _select_best_summary_row(summary_rows)
        config = result.get("config")
        if not isinstance(config, HybridSimulationConfig):
            raise TypeError(f"Scenario {scenario.name} returned an unexpected config payload.")
        scenario_rows.append(
            _scenario_summary_row(
                scenario=scenario,
                summary_row=best_row,
                result=result,
                config=config,
            )
        )

    summary_payload = _write_scenario_summary(scenario_rows, output_dir=scenario_root)
    summary_rows = pd.read_csv(Path(summary_payload["scenario_summary_csv"]))
    comparison_report_path = write_scenario_comparison_report(
        run_dir,
        scenario_root,
        summary_rows,
        output_path=scenario_root / "scientific_validation_report.html",
    )
    for _, row in summary_rows.iterrows():
        scenario_name = str(row.get("scenario_name") or "").strip()
        scenario_output_dir_value = str(row.get("output_dir") or "").strip()
        if not scenario_name or not scenario_output_dir_value:
            continue
        scenario_output_dir = Path(scenario_output_dir_value).expanduser().resolve()
        if not scenario_output_dir.exists():
            continue
        scenario_payload = _build_scenario_switch_payload(
            summary_rows,
            scenario_root=scenario_root,
            current_report_path=scenario_output_dir / "scientific_validation_report.html",
            current_scenario_name=scenario_name,
            default_scenario_name=summary_payload["best_scenario"],
            comparison_report_path=comparison_report_path,
        )
        write_scientific_validation_report(
            run_dir,
            simulation_dir=scenario_output_dir,
            output_path=scenario_output_dir / "scientific_validation_report.html",
            scenario_switch_payload=scenario_payload,
        )
    latest_manifest = load_run_manifest(run_dir)
    latest_manifest["simulation_scenarios"] = {
        "output_dir": str(scenario_root),
        "scenario_count": int(len(scenario_rows)),
        "best_scenario": summary_payload["best_scenario"],
        "selected_setting_labels": [str(label) for label in getattr(args, "setting_label", []) or []],
        "summary_csv": summary_payload["scenario_summary_csv"],
        "summary_md": summary_payload["scenario_summary_md"],
        "overview_path": summary_payload.get("overview_path"),
        "report_path": str(comparison_report_path),
    }
    _save_json(latest_manifest, Path(latest_manifest["manifest_path"]))
    return {
        "output_dir": str(scenario_root),
        "scenario_count": int(len(scenario_rows)),
        "best_scenario": summary_payload["best_scenario"],
        "scenario_summary_csv": summary_payload["scenario_summary_csv"],
        "scenario_summary_md": summary_payload["scenario_summary_md"],
        "overview_path": summary_payload.get("overview_path"),
        "report_path": str(comparison_report_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid transmission validation on observed and synthetic temporal panels."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing manifest.json and generated samples.")
    parser.add_argument("--output-dir", default=None, help="Output directory for simulation reports. Defaults to <run-dir>/simulation or <run-dir>/simulation_scenarios for --scenario-set major.")
    parser.add_argument("--weight-col", default=None, help="Edge-weight column to use. Defaults to the fitted weight-model output column when available.")
    parser.add_argument("--setting-label", action="append", default=None, help="Restrict the run to one setting label. Repeat the flag to keep multiple settings.")
    parser.add_argument("--sample-label-pattern", default=None, help="Regex used to keep only matching sample or setting labels.")
    parser.add_argument("--corop-geojson", default=None, help="Optional path to nl_corop.geojson used for region-level map outputs.")
    parser.add_argument("--scenario-set", default="single", choices=["single", "major"], help="Run one simulation configuration or the built-in major scenario sweep.")

    parser.add_argument("--model", default="SEIR", choices=["SIR", "SEIR", "SIS"], help="Farm-node compartmental model.")
    parser.add_argument("--beta-ff", type=float, default=0.20, help="Transmission multiplier for F->F edges.")
    parser.add_argument("--beta-fr", type=float, default=0.04, help="Contribution multiplier for F->R edges (farm infection exported into regional pressure).")
    parser.add_argument("--beta-rf", type=float, default=0.16, help="Transmission multiplier for R->F edges (regional pressure imported into farms).")
    parser.add_argument("--beta-rr", type=float, default=0.02, help="Propagation multiplier for R->R edges (regional pressure between regions).")
    parser.add_argument("--sigma", type=float, default=0.35, help="Daily E->I progression probability (SEIR only).")
    parser.add_argument("--gamma", type=float, default=0.10, help="Daily recovery probability.")
    parser.add_argument("--num-replicates", type=int, default=256, help="Number of stochastic epidemic replicates per panel.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--initial-seed-count", type=int, default=3, help="Number of initially infected farm seeds.")
    parser.add_argument("--weight-mode", default="log1p", choices=["binary", "linear", "sqrt", "log1p"], help="Transform applied to edge weights before channel hazards are computed.")
    parser.add_argument("--weight-scale", default="auto", help="Positive numeric scale applied after the weight transform, or 'auto' for the observed-panel median transformed weight.")
    parser.add_argument("--tail-days", type=int, default=30, help="Additional no-contact days used to let latent and infectious farm states resolve after the network horizon.")
    parser.add_argument("--seed-scope", default="farm_only", choices=["farm_only", "all_farms"], help="Eligible seed pool for initial infection seeding. 'all_farms' ignores the day-0 restriction.")
    parser.add_argument("--seed-pool-mode", default="observed_day0", choices=["observed_day0", "common_day0", "overall"], help="How to define the farm seed pool with respect to day-0 activity.")
    parser.add_argument("--require-day0-activity", dest="require_day0_activity", action="store_true", help="Restrict the farm seed pool to day-0 active nodes according to --seed-pool-mode.")
    parser.add_argument("--allow-non-day0-seeds", dest="require_day0_activity", action="store_false", help="Allow seed farms that are not active on day 0.")
    parser.add_argument("--farm-susceptibility", type=float, default=1.0, help="Susceptibility multiplier for farm nodes.")
    parser.add_argument("--farm-infectiousness", type=float, default=1.0, help="Infectiousness multiplier for farm nodes.")
    parser.add_argument("--reservoir-decay", type=float, default=0.70, help="Daily decay factor applied to regional pressure reservoirs.")
    parser.add_argument("--reservoir-background", type=float, default=0.0, help="Additive daily background import pressure applied to each region reservoir.")
    parser.add_argument("--reservoir-clip", type=float, default=20.0, help="Upper clip for region reservoir pressure before it enters hazards.")
    parser.add_argument("--farm-daily-import-prob", type=float, default=0.0, help="Exogenous daily infection probability added directly to susceptible farms.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.set_defaults(require_day0_activity=True)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_command(args)


if __name__ == "__main__":
    main()
