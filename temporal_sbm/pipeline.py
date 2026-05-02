"""Core NetForge data preparation, fitting, and sampling code."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
import re
import tempfile
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import holidays as pyhol
except Exception:
    pyhol = None


LOGGER = logging.getLogger(__name__)

ALL_EDGE_COVARIATES = (
    "dist_km",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
    "holiday_nl",
    "doy_sin",
    "doy_cos",
    "mass_grav",
    "anim_grav",
    "ft_cosine",
)

# The layered state already indexes each realised edge by snapshot, so
# calendar-derived annotations are left out of the default fit unless the
# caller asks for them.
LAYER_DERIVED_COVARIATES = (
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
    "holiday_nl",
    "doy_sin",
    "doy_cos",
)

# Default edge annotations used during fitting. They are attached to realised
# edges, not to the full dyad set.
DEFAULT_COVARIATES = (
    "dist_km",
    "mass_grav",
    "anim_grav",
    "ft_cosine",
)

DEFAULT_METADATA_FIELDS = (
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
)
DEFAULT_METADATA_GRID_KM = 50.0
DEFAULT_METADATA_NUMERIC_BINS = 5
DEFAULT_METADATA_FT_TOP_K = 3
METADATA_LAYER_NAME = "__metadata__"


WEIGHT_PROP_RAW = "_edge_weight_raw"
WEIGHT_PROP_INT = "_edge_weight_int"
WEIGHT_PROP_LOG = "_edge_weight_log"
WEIGHT_PROP_LOG1P = "_edge_weight_log1p"
GRAPH_TOOL_STATE_PAYLOAD_FORMAT = "temporal_sbm_nested_state_v2"
GRAPH_TOOL_LAYERED_ENTROPY_WARNING = r"unrecognized keyword arguments: \['entropy_args'\]"


@dataclass(frozen=True)
class CovariateSpec:
    name: str
    graph_property: str
    rec_type: str


@dataclass(frozen=True)
class WeightCandidate:
    input_column: str
    graph_property: str
    rec_type: str
    transform: str
    score_adjustment: float
    model_label: str


@dataclass
class WeightCellStats:
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        self.n += 1
        self.sum_x += value
        self.sum_x2 += value * value


def _require_graph_tool():
    _ensure_runtime_cache_env()
    gt = _import_graph_tool()

    try:
        gt.openmp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", os.cpu_count() or 1)))
    except Exception:
        pass
    return gt


def _import_graph_tool():
    with tempfile.TemporaryFile(mode="w+b") as stderr_buffer:
        original_stderr_fd = None
        try:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            original_stderr_fd = os.dup(2)
            os.dup2(stderr_buffer.fileno(), 2)
            import graph_tool.all as gt
        except ImportError as exc:
            raise RuntimeError(
                "graph-tool is required for the fit and generate stages. "
                "Install graph-tool in this environment or run only the report stage."
            ) from exc
        finally:
            if original_stderr_fd is not None:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)

        suppressed_markers = (
            "Gdk-WARNING",
            "Failed to initialize CVDisplayLink",
            "Connection Invalid error for service com.apple.hiservices-xpcservice.",
            "Error received in message reply handler: Connection invalid",
        )
        stderr_buffer.flush()
        stderr_buffer.seek(0)
        stderr_text = stderr_buffer.read().decode("utf-8", errors="replace")
    forwarded_lines = []
    suppressed_lines = []
    for line in stderr_text.splitlines():
        if any(marker in line for marker in suppressed_markers):
            suppressed_lines.append(line)
        else:
            forwarded_lines.append(line)

    if forwarded_lines:
        sys.stderr.write("\n".join(forwarded_lines) + "\n")
    if suppressed_lines:
        LOGGER.debug("Suppressed graph-tool import stderr lines | count=%s", len(suppressed_lines))
    return gt


def _ensure_runtime_cache_env() -> None:
    runtime_root = Path(tempfile.gettempdir()) / "temporal_sbm_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    cache_root = runtime_root / "cache"
    mpl_root = runtime_root / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_root.mkdir(parents=True, exist_ok=True)

    current_mpl = os.getenv("MPLCONFIGDIR")
    if not current_mpl or not os.access(current_mpl, os.W_OK):
        os.environ["MPLCONFIGDIR"] = str(mpl_root)

    current_xdg = os.getenv("XDG_CACHE_HOME")
    if not current_xdg or not os.access(current_xdg, os.W_OK):
        os.environ["XDG_CACHE_HOME"] = str(cache_root)

    os.environ.setdefault("MPLBACKEND", "Agg")


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _format_numeric_summary(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return "count=0"

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return f"count={array.size} | all_nonfinite"

    return (
        f"count={array.size} | min={float(finite.min()):.6f} | mean={float(finite.mean()):.6f} "
        f"| median={float(np.median(finite)):.6f} | p95={float(np.percentile(finite, 95)):.6f} "
        f"| max={float(finite.max()):.6f}"
    )


def _format_count_preview(counts: dict[Any, int], limit: int = 8) -> str:
    if not counts:
        return "none"
    items = sorted(counts.items(), key=lambda item: item[0])
    preview = ", ".join(f"{key}:{value}" for key, value in items[:limit])
    if len(items) > limit:
        preview += ", ..."
    return preview


def _log_edge_frame_debug(
    label: str,
    frame: pd.DataFrame,
    directed: bool,
    weight_col: Optional[str] = None,
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return

    columns = frame.columns.tolist()
    node_count = 0
    if {"u", "i"}.issubset(frame.columns):
        node_count = int(len(set(frame["u"].tolist()) | set(frame["i"].tolist())))
    layer_count = int(frame["ts"].nunique()) if "ts" in frame.columns else 0
    duplicate_count = int(frame.duplicated(["u", "i", "ts"]).sum()) if {"u", "i", "ts"}.issubset(frame.columns) else 0
    self_loop_count = int((frame["u"] == frame["i"]).sum()) if {"u", "i"}.issubset(frame.columns) else 0
    LOGGER.debug(
        "%s | rows=%s | columns=%s | directed=%s | nodes=%s | layers=%s | duplicates=%s | self_loops=%s",
        label,
        len(frame),
        columns,
        directed,
        node_count,
        layer_count,
        duplicate_count,
        self_loop_count,
    )

    if "ts" in frame.columns:
        layer_counts = frame["ts"].value_counts(sort=False).sort_index().to_dict()
        LOGGER.debug("%s | layer_counts=%s", label, _format_count_preview(layer_counts))

    if weight_col and weight_col in frame.columns:
        LOGGER.debug(
            "%s | weight_col=%s | weight_summary=%s",
            label,
            weight_col,
            _format_numeric_summary(frame[weight_col].to_numpy(dtype=float, copy=False)),
        )


def _log_covariate_specs(label: str, covariate_specs: Iterable[CovariateSpec]) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    parts = [
        f"{spec.name}[graph={spec.graph_property}, rec={spec.rec_type}]"
        for spec in covariate_specs
    ]
    LOGGER.debug("%s | covariates=%s", label, parts or ["none"])


def _looks_like_int_array(values: np.ndarray, atol: float = 1e-9) -> bool:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return False
    return bool(np.allclose(array, np.round(array), atol=atol, rtol=0.0))


def _transform_weight_values(values: np.ndarray, transform: str) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    if transform == "none":
        return array.astype(float, copy=False), 0.0
    if transform == "log":
        if np.any(array <= 0):
            raise ValueError("The 'log' edge-weight transform requires strictly positive weights.")
        transformed = np.log(array)
        return transformed, float(transformed.sum())
    if transform == "log1p":
        if np.any(array < 0):
            raise ValueError("The 'log1p' edge-weight transform requires nonnegative weights.")
        transformed = np.log1p(array)
        return transformed, float(transformed.sum())
    raise ValueError(f"Unsupported edge-weight transform: {transform}")


def _inverse_transform_weight_value(value: float, transform: str) -> float:
    if transform == "none":
        return float(value)
    if transform == "log":
        return float(np.exp(value))
    if transform == "log1p":
        return max(0.0, float(np.expm1(value)))
    raise ValueError(f"Unsupported edge-weight transform: {transform}")


def _parse_ts_ordinal(series_like: Iterable[Any]) -> pd.Series:
    series = pd.Series(series_like)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: pd.Timestamp.fromordinal(int(value)) if np.isfinite(value) else pd.NaT)


def _date_bound_to_ts(
    date_text: Optional[str],
    ts_format: str,
    ts_unit: str,
    tz: str,
    is_end: bool,
) -> Optional[int]:
    if date_text is None:
        return None

    timestamp = pd.Timestamp(date_text)
    if ts_format == "ordinal":
        if is_end:
            return int((timestamp + pd.Timedelta(days=1)).toordinal() - 1)
        return int(timestamp.toordinal())

    scale_ns = {
        "s": 1_000_000_000,
        "ms": 1_000_000,
        "us": 1_000,
        "ns": 1,
        "D": 86_400_000_000_000,
    }[ts_unit]

    localized = timestamp.tz_localize(tz) if timestamp.tzinfo is None else timestamp.tz_convert(tz)
    utc_value = localized.tz_convert("UTC")
    if is_end:
        utc_value = utc_value + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    if ts_unit == "D":
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return int((utc_value.floor("D") - epoch) / pd.Timedelta(days=1))
    return int(utc_value.value // scale_ns)


def add_calendar_columns(
    df: pd.DataFrame,
    ts_col: str,
    tz: str,
    ts_unit: str,
    ts_format: str = "ordinal",
    holiday_country: str = "NL",
) -> pd.DataFrame:
    if ts_format == "ordinal":
        dt = _parse_ts_ordinal(df[ts_col])
        if tz:
            dt = dt.dt.tz_localize(tz)
    else:
        dt = pd.to_datetime(df[ts_col].astype("int64"), unit=ts_unit, origin="unix", utc=True)
        if tz:
            dt = dt.dt.tz_convert(tz)

    frame = df.copy()
    frame["_dt"] = dt
    frame["_date"] = frame["_dt"].dt.date

    day_of_week = frame["_dt"].dt.dayofweek.to_numpy(dtype=int, copy=False)
    for index in range(7):
        frame[f"dow_{index}"] = (day_of_week == index).astype(np.int8)

    if pyhol is None:
        LOGGER.warning("python-holidays is not installed; holiday indicators will be zero.")
        frame["holiday_nl"] = 0
    else:
        years = sorted({value.year for value in frame["_dt"] if pd.notna(value)})
        holidays = pyhol.country_holidays(holiday_country, years=years)
        frame["holiday_nl"] = [1 if date in holidays else 0 for date in frame["_date"]]

    day_of_year = frame["_dt"].dt.dayofyear.astype(float).to_numpy()
    frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return frame


def _find_centroid_indices(columns: list[str]) -> tuple[int, int]:
    candidates = [
        ("xco", "yco"),
        ("centroid_x", "centroid_y"),
    ]
    for x_name, y_name in candidates:
        if x_name in columns and y_name in columns:
            return columns.index(x_name), columns.index(y_name)
    raise ValueError(
        "Node feature schema must contain either (xco, yco) or (centroid_x, centroid_y)."
    )


def _feature_index(columns: list[str], *names: str) -> Optional[int]:
    for name in names:
        if name in columns:
            return columns.index(name)
    return None


def _extract_node_scalars_from_arrays(
    node_features: np.ndarray,
    node_feature_columns: list[str],
    centroid_x_index: int,
    centroid_y_index: int,
) -> dict[str, np.ndarray]:
    features = node_features[1:]
    columns = node_feature_columns

    num_farms_index = _feature_index(columns, "num_farms")
    if num_farms_index is None:
        num_farms = np.ones(features.shape[0], dtype=float)
    else:
        num_farms = features[:, num_farms_index].astype(float)

    animals_index = _feature_index(columns, "herd_giab23_pigs", "total_animals")
    if animals_index is not None:
        total_animals = features[:, animals_index].astype(float)
    else:
        animal_indices = [index for index, name in enumerate(columns) if name.startswith("total_diergroep_")]
        total_animals = (
            features[:, animal_indices].astype(float).sum(axis=1) if animal_indices else np.zeros(features.shape[0], dtype=float)
        )

    ft_indices = [index for index, name in enumerate(columns) if name.startswith("count_ft_")]
    if ft_indices:
        ft_matrix = features[:, ft_indices].astype(float)
        ft_norm = np.linalg.norm(ft_matrix, axis=1)
    else:
        ft_matrix = np.zeros((features.shape[0], 0), dtype=float)
        ft_norm = np.zeros(features.shape[0], dtype=float)

    cx = features[:, centroid_x_index].astype(float)
    cy = features[:, centroid_y_index].astype(float)

    return {
        "cx": cx,
        "cy": cy,
        "num_farms": num_farms,
        "total_animals": total_animals,
        "ft_matrix": ft_matrix,
        "ft_norm": ft_norm,
    }


def _joint_metadata_model_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "joint_metadata_model", True))


def _parse_metadata_fields(args: argparse.Namespace) -> list[str]:
    raw_value = getattr(args, "metadata_fields", None)
    if raw_value is None:
        return list(DEFAULT_METADATA_FIELDS)
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw_value if str(part).strip()]
    if len(parts) == 1 and parts[0].lower() in {"none", "off", "false", "no"}:
        return []
    return parts


def _metadata_quantile_labels(values: Iterable[float], prefix: str, bins: int) -> pd.Series:
    series = pd.Series(values, dtype=float)
    out = pd.Series([None] * len(series), dtype=object)
    clean = series.replace([np.inf, -np.inf], np.nan)
    mask = clean.notna()
    if mask.sum() <= 0:
        return out
    unique_values = int(clean[mask].nunique())
    if unique_values <= 1:
        out.loc[mask] = f"{prefix}_q0"
        return out
    q = max(1, min(int(bins), int(mask.sum()), unique_values))
    try:
        ranked = clean[mask].rank(method="first")
        codes = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
        out.loc[mask] = [f"{prefix}_q{int(code)}" for code in codes.astype(int)]
    except Exception:
        out.loc[mask] = f"{prefix}_q0"
    return out


def _iter_metadata_tokens(value: object) -> Iterable[str]:
    if pd.isna(value):
        return ()

    text_value = str(value).strip()
    if not text_value:
        return ()

    lowered = text_value.lower()
    if lowered in {"nan", "none"}:
        return ()

    if "|" not in text_value and ";" not in text_value:
        return (text_value,)

    tokens: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[|;]", text_value):
        token = str(part).strip()
        if not token:
            continue
        token_lower = token.lower()
        if token_lower in {"nan", "none"} or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _build_joint_metadata_links(
    *,
    compact_to_original: np.ndarray,
    node_features: np.ndarray,
    node_feature_columns: list[str],
    centroid_x_index: int,
    centroid_y_index: int,
    active_compact_mask: np.ndarray,
    node_map_csv: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fields = _parse_metadata_fields(args)
    if not _joint_metadata_model_enabled(args) or not fields:
        empty = pd.DataFrame(columns=["u", "tag_key", "tag_kind", "tag_value"])
        return empty, {
            "enabled": False,
            "fields_requested": list(fields),
            "fields_used": [],
            "num_links": 0,
            "num_tags": 0,
            "field_link_counts": {},
            "field_tag_counts": {},
        }

    scalars = _extract_node_scalars_from_arrays(
        node_features=node_features,
        node_feature_columns=node_feature_columns,
        centroid_x_index=centroid_x_index,
        centroid_y_index=centroid_y_index,
    )
    node_frame = pd.DataFrame(
        {
            "compact_id": np.arange(len(compact_to_original), dtype=np.int64),
            "node_id": compact_to_original.astype(np.int64),
            "active_in_window": active_compact_mask.astype(np.int8),
            "num_farms": scalars["num_farms"].astype(float),
            "total_animals": scalars["total_animals"].astype(float),
            "cx": scalars["cx"].astype(float),
            "cy": scalars["cy"].astype(float),
        }
    )
    node_map = pd.read_csv(node_map_csv)
    merge_columns = ["node_id"] + [column for column in node_map.columns if column != "node_id" and column not in node_frame.columns]
    node_frame = node_frame.merge(node_map[merge_columns].drop_duplicates(subset=["node_id"]), on="node_id", how="left")

    numeric_bins = max(1, int(getattr(args, "metadata_numeric_bins", DEFAULT_METADATA_NUMERIC_BINS)))
    grid_km = max(1e-6, float(getattr(args, "metadata_grid_km", DEFAULT_METADATA_GRID_KM)))
    ft_top_k = max(1, int(getattr(args, "metadata_ft_top_k", DEFAULT_METADATA_FT_TOP_K)))
    grid_m = 1000.0 * grid_km

    records: list[dict[str, Any]] = []
    field_link_counts: Counter[str] = Counter()
    field_tag_values: dict[str, set[str]] = defaultdict(set)

    def add_token(kind: str, compact_id: int, value: object) -> None:
        for token_value in _iter_metadata_tokens(value):
            records.append(
                {
                    "u": int(compact_id),
                    "tag_key": f"{kind}::{token_value}",
                    "tag_kind": kind,
                    "tag_value": token_value,
                }
            )
            field_link_counts[kind] += 1
            field_tag_values[kind].add(token_value)

    for field in fields:
        if field == "corop" and "corop" in node_frame.columns:
            for row in node_frame.itertuples(index=False):
                add_token("corop", int(row.compact_id), getattr(row, "corop", None))
            continue

        if field == "type_label" and "type" in node_frame.columns:
            for row in node_frame.itertuples(index=False):
                add_token("type_label", int(row.compact_id), getattr(row, "type", None))
            continue

        if field == "num_farms_bin":
            labels = _metadata_quantile_labels(node_frame["num_farms"], "num_farms", numeric_bins)
            for compact_id, label in zip(node_frame["compact_id"], labels):
                add_token("num_farms_bin", int(compact_id), label)
            continue

        if field == "total_animals_bin":
            labels = _metadata_quantile_labels(node_frame["total_animals"], "total_animals", numeric_bins)
            for compact_id, label in zip(node_frame["compact_id"], labels):
                add_token("total_animals_bin", int(compact_id), label)
            continue

        if field == "centroid_grid":
            grid_x = np.floor(node_frame["cx"].astype(float).to_numpy() / grid_m).astype(np.int64)
            grid_y = np.floor(node_frame["cy"].astype(float).to_numpy() / grid_m).astype(np.int64)
            for compact_id, gx, gy in zip(node_frame["compact_id"], grid_x, grid_y):
                add_token("centroid_grid", int(compact_id), f"{int(gx)}_{int(gy)}")
            continue

        if field == "ft_tokens":
            ft_indices = [index for index, name in enumerate(node_feature_columns) if name.startswith("count_ft_")]
            features = node_features[1:]
            if ft_indices:
                ft_names = [node_feature_columns[index][len("count_ft_"):] for index in ft_indices]
                ft_matrix = features[:, ft_indices].astype(float)
                for compact_id, row_values in zip(node_frame["compact_id"].astype(int).tolist(), ft_matrix):
                    positive = np.flatnonzero(row_values > 0)
                    if positive.size == 0:
                        continue
                    order = positive[np.argsort(-row_values[positive], kind="stable")][:ft_top_k]
                    for idx in order:
                        add_token("ft_tokens", int(compact_id), ft_names[int(idx)])
            continue

        if field in node_frame.columns:
            column = node_frame[field]
            if pd.api.types.is_numeric_dtype(column) and column.nunique(dropna=True) > max(numeric_bins, 10):
                labels = _metadata_quantile_labels(column.astype(float), field, numeric_bins)
                for compact_id, label in zip(node_frame["compact_id"], labels):
                    add_token(field, int(compact_id), label)
            else:
                for compact_id, value in zip(node_frame["compact_id"], column):
                    add_token(field, int(compact_id), value)
            continue

        LOGGER.warning("Requested metadata field '%s' is unavailable and will be skipped.", field)

    links = pd.DataFrame(records, columns=["u", "tag_key", "tag_kind", "tag_value"])
    if not links.empty:
        links = links.drop_duplicates(["u", "tag_key"]).sort_values(["tag_kind", "tag_key", "u"]).reset_index(drop=True)

    summary = {
        "enabled": bool(not links.empty),
        "fields_requested": list(fields),
        "fields_used": sorted(field_link_counts.keys()),
        "num_links": int(len(links)),
        "num_tags": int(links["tag_key"].nunique()) if not links.empty else 0,
        "field_link_counts": {str(key): int(value) for key, value in sorted(field_link_counts.items())},
        "field_tag_counts": {str(key): int(len(value)) for key, value in sorted(field_tag_values.items())},
        "grid_km": float(grid_km),
        "numeric_bins": int(numeric_bins),
        "ft_top_k": int(ft_top_k),
    }
    return links, summary


@dataclass
class InputPaths:
    dataset: str
    dataset_dir: Path
    edges_csv: Path
    weight_npy: Optional[Path]
    node_features_npy: Path
    node_schema_json: Path
    node_map_csv: Path


@dataclass(frozen=True)
class NodeSchema:
    columns: list[str]
    node_row_offset: int


@dataclass
class PreparedData:
    input_paths: InputPaths
    original_edges: pd.DataFrame
    compact_edges: pd.DataFrame
    node_features: np.ndarray
    node_feature_columns: list[str]
    compact_to_original: np.ndarray
    original_to_compact: dict[int, int]
    centroid_x_index: int
    centroid_y_index: int
    layer_map: dict[int, int]
    node_type_by_compact: Optional[np.ndarray]
    active_compact_mask: np.ndarray
    no_compact: bool
    metadata_links: pd.DataFrame
    metadata_summary: dict[str, Any]
    weight_column: Optional[str]
    duplicate_edge_count: int
    self_loop_count: int


def resolve_input_paths(args: argparse.Namespace) -> InputPaths:
    dataset_dir = Path(args.data_root).expanduser().resolve() / args.dataset
    edges_csv = (
        Path(args.edges_csv).expanduser().resolve()
        if args.edges_csv
        else dataset_dir / "edges.csv"
    )
    weight_npy = Path(args.weight_npy).expanduser().resolve() if getattr(args, "weight_npy", None) else None
    node_features_npy = (
        Path(args.node_features_npy).expanduser().resolve()
        if args.node_features_npy
        else dataset_dir / "node_features.npy"
    )

    if args.node_schema_json:
        node_schema_json = Path(args.node_schema_json).expanduser().resolve()
    else:
        node_schema_json = dataset_dir / "node_schema.json"

    node_map_csv = Path(args.node_map_csv).expanduser().resolve() if args.node_map_csv else dataset_dir / "node_map.csv"

    return InputPaths(
        dataset=args.dataset,
        dataset_dir=dataset_dir,
        edges_csv=edges_csv,
        weight_npy=weight_npy,
        node_features_npy=node_features_npy,
        node_schema_json=node_schema_json,
        node_map_csv=node_map_csv,
    )


def _load_node_schema(path: Path) -> NodeSchema:
    payload = json.loads(path.read_text())
    columns = payload.get("node_feature_columns_in_order")
    if columns is None:
        raise ValueError(
            f"{path} must define 'node_feature_columns_in_order'."
        )
    node_row_offset = int(payload.get("node_row_offset", 0))
    if node_row_offset != 0:
        raise ValueError(
            f"{path} declares node_row_offset={node_row_offset}. NetForge requires row n to match node_id n."
        )
    return NodeSchema(columns=list(columns), node_row_offset=0)


def _validate_node_feature_rows(
    node_features: np.ndarray,
    *,
    min_endpoint: int,
    max_endpoint: int,
    node_map_max_id: int,
) -> None:
    if node_features.ndim != 2:
        raise ValueError("Node feature matrix must be two-dimensional.")
    if min_endpoint < 0:
        raise ValueError(f"Edge endpoints must be non-negative, but observed {min_endpoint}.")
    expected_rows = int(node_map_max_id) + 1
    if node_features.shape[0] != expected_rows:
        raise ValueError(
            "Node feature matrix must have one row per node_id in node_map.csv. "
            f"Expected {expected_rows} rows, observed {node_features.shape[0]}."
        )
    if max_endpoint > int(node_features.shape[0] - 1):
        raise ValueError(
            "Node feature matrix does not cover the node ids used by the edge table. "
            f"Observed endpoints [{min_endpoint}, {max_endpoint}] with feature shape {node_features.shape}."
        )


def _pad_node_feature_rows(node_features: np.ndarray) -> np.ndarray:
    padded = np.zeros((node_features.shape[0] + 1, node_features.shape[1]), dtype=node_features.dtype)
    padded[1:] = node_features
    return padded


def _default_weight_column_name(input_paths: InputPaths, weight_path: Path) -> str:
    stem = weight_path.stem
    prefix = f"ml_{input_paths.dataset}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem or "weight"


def _align_external_weight_values(edge_frame: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, str]:
    weights = np.asarray(values, dtype=float).reshape(-1)
    edge_count = len(edge_frame)
    if edge_count == 0:
        return weights[:0], "empty"

    if "idx" in edge_frame.columns:
        idx_series = pd.to_numeric(edge_frame["idx"], errors="coerce")
        if idx_series.notna().all():
            idx_values = idx_series.astype(np.int64).to_numpy()
            if len(weights) > 0 and idx_values.min() >= 1 and idx_values.max() < len(weights):
                return weights[idx_values], "idx_with_padding"
            if idx_values.min() >= 1 and idx_values.max() <= len(weights):
                return weights[idx_values - 1], "idx_zero_based_adjusted"

    if len(weights) == edge_count:
        return weights, "row_order"
    if len(weights) == edge_count + 1:
        return weights[1:], "row_order_skip_first"

    raise ValueError(
        "External weight vector length does not align with the edge CSV rows. "
        f"Observed weight length={len(weights)}, edge rows={edge_count}."
    )


def _detect_transformed_external_weight_companion(
    edge_frame: pd.DataFrame,
    weight_path: Path,
    aligned_weights: np.ndarray,
) -> Optional[tuple[str, Path, str]]:
    weight_path = Path(weight_path)
    if weight_path.suffix != ".npy" or weight_path.stem.endswith("_raw"):
        return None

    raw_candidate = weight_path.with_name(f"{weight_path.stem}_raw.npy")
    if not raw_candidate.exists():
        return None

    raw_values = np.load(raw_candidate)
    aligned_raw_weights, raw_alignment_strategy = _align_external_weight_values(edge_frame, raw_values)

    comparisons: list[tuple[str, np.ndarray]] = []
    if np.all(aligned_raw_weights > 0):
        comparisons.append(("log", np.log(aligned_raw_weights)))
    if np.all(aligned_raw_weights >= 0):
        comparisons.append(("log1p", np.log1p(aligned_raw_weights)))

    for transform_name, transformed_weights in comparisons:
        if np.allclose(aligned_weights, transformed_weights, rtol=1e-9, atol=1e-9):
            return transform_name, raw_candidate, raw_alignment_strategy
    return None


def _standardise_edge_columns(
    frame: pd.DataFrame,
    src_col: str,
    dst_col: str,
    ts_col: str,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    required = [src_col, dst_col, ts_col] + ([weight_col] if weight_col else [])
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Edge CSV is missing required columns: {missing}")

    keep_columns = [src_col, dst_col, ts_col] + ([weight_col] if weight_col else [])
    out = frame[keep_columns].copy()
    out["u"] = pd.to_numeric(out[src_col], errors="raise").astype(np.int64)
    out["i"] = pd.to_numeric(out[dst_col], errors="raise").astype(np.int64)
    out["ts"] = pd.to_numeric(out[ts_col], errors="raise").astype(np.int64)
    if weight_col:
        out[weight_col] = pd.to_numeric(out[weight_col], errors="raise").astype(float)
        if not np.isfinite(out[weight_col].to_numpy(dtype=float, copy=False)).all():
            raise ValueError(f"Edge weight column '{weight_col}' contains NaN or infinite values.")
        return out[["u", "i", "ts", weight_col]]
    return out[["u", "i", "ts"]]


def prepare_data(args: argparse.Namespace) -> PreparedData:
    t0 = pd.Timestamp.utcnow()
    input_paths = resolve_input_paths(args)
    LOGGER.debug(
        "Resolved input paths | dataset_dir=%s | edges_csv=%s | weight_npy=%s | node_features_npy=%s | node_schema_json=%s | node_map_csv=%s",
        input_paths.dataset_dir,
        input_paths.edges_csv,
        input_paths.weight_npy,
        input_paths.node_features_npy,
        input_paths.node_schema_json,
        input_paths.node_map_csv,
    )

    required_paths = [
        input_paths.edges_csv,
        input_paths.node_features_npy,
        input_paths.node_schema_json,
        input_paths.node_map_csv,
    ]
    if input_paths.weight_npy is not None:
        required_paths.append(input_paths.weight_npy)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    edge_frame = pd.read_csv(input_paths.edges_csv)
    LOGGER.debug(
        "Loaded edge CSV | path=%s | rows=%s | columns=%s",
        input_paths.edges_csv,
        len(edge_frame),
        edge_frame.columns.tolist(),
    )
    if input_paths.weight_npy is not None:
        weight_col = getattr(args, "weight_col", None) or _default_weight_column_name(input_paths, input_paths.weight_npy)
        if weight_col in edge_frame.columns:
            raise ValueError(
                f"External weight column name '{weight_col}' already exists in the edge CSV. "
                "Choose a different --weight-col when using --weight-npy."
            )
        raw_weights = np.load(input_paths.weight_npy)
        aligned_weights, alignment_strategy = _align_external_weight_values(edge_frame, raw_weights)
        transformed_companion = _detect_transformed_external_weight_companion(
            edge_frame=edge_frame,
            weight_path=input_paths.weight_npy,
            aligned_weights=aligned_weights,
        )
        if transformed_companion is not None:
            transform_name, raw_companion_path, raw_alignment_strategy = transformed_companion
            raise ValueError(
                "The supplied --weight-npy file appears to be a transformed companion of a raw additive weight file. "
                f"Detected {transform_name}({raw_companion_path.name}) == {input_paths.weight_npy.name} "
                f"(alignment={alignment_strategy}, raw_alignment={raw_alignment_strategy}). "
                f"Use '{raw_companion_path}' as --weight-npy instead, and let --weight-transform={transform_name} "
                "or --weight-transform=auto control the modeling scale."
            )
        edge_frame[weight_col] = aligned_weights.astype(float, copy=False)
        setattr(args, "weight_col", weight_col)
        LOGGER.debug(
            "Attached external edge weights | path=%s | weight_col=%s | alignment=%s | raw_length=%s | edge_rows=%s | summary=%s",
            input_paths.weight_npy,
            weight_col,
            alignment_strategy,
            len(np.asarray(raw_weights).reshape(-1)),
            len(edge_frame),
            _format_numeric_summary(aligned_weights),
        )
    edge_frame = _standardise_edge_columns(
        edge_frame,
        args.src_col,
        args.dst_col,
        args.ts_col,
        weight_col=getattr(args, "weight_col", None),
    )
    _log_edge_frame_debug(
        "Standardised edge frame",
        edge_frame,
        directed=bool(args.directed),
        weight_col=getattr(args, "weight_col", None),
    )
    raw_node_features = np.load(input_paths.node_features_npy)
    if raw_node_features.ndim != 2:
        raise ValueError("Node feature matrix must be two-dimensional.")

    node_schema = _load_node_schema(input_paths.node_schema_json)
    feature_columns = node_schema.columns
    if len(feature_columns) != raw_node_features.shape[1]:
        raise ValueError(
            "Node feature schema width does not match the .npy feature matrix: "
            f"{len(feature_columns)} != {raw_node_features.shape[1]}"
        )
    LOGGER.debug(
        "Loaded node features | raw_shape=%s | schema_columns=%s | declared_row_offset=%s",
        raw_node_features.shape,
        len(feature_columns),
        node_schema.node_row_offset,
    )

    centroid_x_index, centroid_y_index = _find_centroid_indices(feature_columns)
    LOGGER.debug(
        "Resolved centroid columns | x=%s | y=%s",
        feature_columns[centroid_x_index],
        feature_columns[centroid_y_index],
    )

    ts_start = args.ts_start
    ts_end = args.ts_end
    if args.date_start:
        ts_start = _date_bound_to_ts(args.date_start, args.ts_format, args.ts_unit, args.tz, is_end=False)
    if args.date_end:
        ts_end = _date_bound_to_ts(args.date_end, args.ts_format, args.ts_unit, args.tz, is_end=True)
    LOGGER.debug(
        "Timestamp filters | ts_start=%s | ts_end=%s | date_start=%s | date_end=%s | ts_format=%s | ts_unit=%s | tz=%s",
        ts_start,
        ts_end,
        args.date_start,
        args.date_end,
        args.ts_format,
        args.ts_unit,
        args.tz,
    )

    if ts_start is not None:
        edge_frame = edge_frame.loc[edge_frame["ts"] >= int(ts_start)].copy()
    if ts_end is not None:
        edge_frame = edge_frame.loc[edge_frame["ts"] <= int(ts_end)].copy()
    if edge_frame.empty:
        raise ValueError("No edges remain after applying the requested timestamp filters.")
    _log_edge_frame_debug(
        "Timestamp-filtered edge frame",
        edge_frame,
        directed=bool(args.directed),
        weight_col=getattr(args, "weight_col", None),
    )

    if not args.directed:
        uv = np.sort(edge_frame[["u", "i"]].to_numpy(dtype=np.int64, copy=False), axis=1)
        edge_frame["u"] = uv[:, 0]
        edge_frame["i"] = uv[:, 1]
        _log_edge_frame_debug(
            "Undirected-canonical edge frame",
            edge_frame,
            directed=False,
            weight_col=getattr(args, "weight_col", None),
        )

    self_loop_mask = edge_frame["u"] == edge_frame["i"]
    self_loop_count = int(self_loop_mask.sum())
    if self_loop_count:
        if args.self_loop_policy == "error":
            raise ValueError(
                f"Detected {self_loop_count} self-loops after filtering. "
                "Use --self-loop-policy drop to discard them."
            )
        if args.self_loop_policy == "drop":
            edge_frame = edge_frame.loc[~self_loop_mask].copy()

    duplicate_edge_count = int(edge_frame.duplicated(["u", "i", "ts"]).sum())
    if duplicate_edge_count:
        if args.duplicate_policy == "error":
            raise ValueError(
                f"Detected {duplicate_edge_count} duplicate (u, i, ts) rows. "
                "Use --duplicate-policy collapse to model simple graphs."
            )
        if args.duplicate_policy == "collapse":
            if getattr(args, "weight_col", None):
                edge_frame = (
                    edge_frame.groupby(["u", "i", "ts"], as_index=False, sort=False)[args.weight_col]
                    .sum()
                    .reset_index(drop=True)
                )
            else:
                edge_frame = edge_frame.drop_duplicates(["u", "i", "ts"]).reset_index(drop=True)
    LOGGER.debug(
        "Edge cleanup decisions | self_loop_policy=%s | self_loops_found=%s | duplicate_policy=%s | duplicates_found=%s",
        args.self_loop_policy,
        self_loop_count,
        args.duplicate_policy,
        duplicate_edge_count,
    )
    _log_edge_frame_debug(
        "Prepared input edge frame",
        edge_frame,
        directed=bool(args.directed),
        weight_col=getattr(args, "weight_col", None),
    )

    if edge_frame.empty:
        raise ValueError("No edges remain after applying self-loop and duplicate policies.")

    min_endpoint = int(edge_frame[["u", "i"]].to_numpy().min())
    max_endpoint = int(edge_frame[["u", "i"]].to_numpy().max())
    node_map_ids = pd.read_csv(input_paths.node_map_csv, usecols=["node_id"])
    if node_map_ids.empty:
        raise ValueError(f"{input_paths.node_map_csv} must contain at least one node_id row.")
    parsed_node_ids = pd.to_numeric(node_map_ids["node_id"], errors="coerce")
    if parsed_node_ids.isna().any():
        raise ValueError(f"{input_paths.node_map_csv} contains non-numeric node_id values.")
    node_map_max_id = int(parsed_node_ids.max())
    _validate_node_feature_rows(
        raw_node_features,
        min_endpoint=min_endpoint,
        max_endpoint=max_endpoint,
        node_map_max_id=node_map_max_id,
    )
    node_features = _pad_node_feature_rows(raw_node_features)
    max_node_id = int(node_features.shape[0] - 2)
    LOGGER.debug(
        "Normalised node features | internal_shape=%s | max_node_id=%s",
        node_features.shape,
        max_node_id,
    )

    active_original_ids = np.sort(np.unique(edge_frame[["u", "i"]].to_numpy().ravel()).astype(np.int64))
    no_compact = bool(getattr(args, "no_compact", False))
    if no_compact:
        used_original_ids = np.arange(max_node_id + 1, dtype=np.int64)
        original_to_compact = {int(node_id): int(node_id) for node_id in used_original_ids.tolist()}
        active_compact_mask = np.zeros(len(used_original_ids), dtype=bool)
        active_compact_mask[active_original_ids] = True
        compact_edges = edge_frame.copy()
        compact_features = node_features.copy()
        LOGGER.debug(
            "No-compaction mode | node_universe=%s | active_nodes=%s | inactive_nodes=%s | min_active_node_id=%s | max_active_node_id=%s",
            len(used_original_ids),
            int(active_compact_mask.sum()),
            int(len(used_original_ids) - int(active_compact_mask.sum())),
            int(active_original_ids.min()) if len(active_original_ids) else None,
            int(active_original_ids.max()) if len(active_original_ids) else None,
        )
    else:
        used_original_ids = active_original_ids
        original_to_compact = {int(node_id): index for index, node_id in enumerate(used_original_ids.tolist())}
        active_compact_mask = np.ones(len(used_original_ids), dtype=bool)
        compact_edges = edge_frame.copy()
        compact_edges["u"] = compact_edges["u"].map(original_to_compact).astype(np.int64)
        compact_edges["i"] = compact_edges["i"].map(original_to_compact).astype(np.int64)
        compact_features = np.zeros((len(used_original_ids) + 1, node_features.shape[1]), dtype=node_features.dtype)
        compact_features[1:] = node_features[1:][used_original_ids]
        LOGGER.debug(
            "Compact node mapping | unique_nodes=%s | min_node_id=%s | max_node_id=%s",
            len(used_original_ids),
            int(used_original_ids.min()) if len(used_original_ids) else None,
            int(used_original_ids.max()) if len(used_original_ids) else None,
        )

    compact_edges = add_calendar_columns(
        compact_edges,
        ts_col="ts",
        tz=args.tz,
        ts_unit=args.ts_unit,
        ts_format=args.ts_format,
        holiday_country=args.holiday_country,
    )

    layer_values = sorted(compact_edges["ts"].astype(np.int64).unique().tolist())
    layer_map = {int(ts): index for index, ts in enumerate(layer_values)}
    _log_edge_frame_debug(
        "Compact edge frame with calendar columns",
        compact_edges,
        directed=bool(args.directed),
        weight_col=getattr(args, "weight_col", None),
    )
    LOGGER.debug(
        "Layer map | layer_count=%s | preview=%s",
        len(layer_map),
        _format_count_preview({ts: layer_map[ts] for ts in layer_values}),
    )

    node_map = pd.read_csv(input_paths.node_map_csv)
    if not {"node_id", "type"}.issubset(node_map.columns):
        raise ValueError(f"{input_paths.node_map_csv} must contain 'node_id' and 'type' columns.")
    node_type_by_compact = np.zeros(len(used_original_ids), dtype=np.int32)
    node_map = node_map[node_map["node_id"].isin(used_original_ids)].copy()
    node_map["compact_id"] = node_map["node_id"].map(original_to_compact)
    node_map = node_map[node_map["compact_id"].notna()].copy()
    is_region = node_map["type"].astype(str).str.lower().eq("region")
    node_type_by_compact[node_map.loc[is_region, "compact_id"].astype(int).to_numpy()] = 1

    metadata_links, metadata_summary = _build_joint_metadata_links(
        compact_to_original=used_original_ids,
        node_features=compact_features,
        node_feature_columns=feature_columns,
        centroid_x_index=centroid_x_index,
        centroid_y_index=centroid_y_index,
        active_compact_mask=active_compact_mask,
        node_map_csv=input_paths.node_map_csv,
        args=args,
    )
    LOGGER.debug(
        "Loaded node types | mapped_rows=%s | region_nodes=%s | no_compact=%s | metadata_links=%s | metadata_tags=%s | metadata_fields=%s",
        len(node_map),
        int(is_region.sum()),
        no_compact,
        int(metadata_summary.get("num_links", 0)),
        int(metadata_summary.get("num_tags", 0)),
        metadata_summary.get("fields_used", []),
    )

    LOGGER.info(
        "Prepared data in %s | edges=%s | unique nodes=%s | layers=%s | metadata_links=%s | metadata_tags=%s",
        _fmt_duration((pd.Timestamp.utcnow() - t0).total_seconds()),
        len(edge_frame),
        len(used_original_ids),
        len(layer_map),
        int(metadata_summary.get("num_links", 0)),
        int(metadata_summary.get("num_tags", 0)),
    )

    return PreparedData(
        input_paths=input_paths,
        original_edges=edge_frame.reset_index(drop=True),
        compact_edges=compact_edges.reset_index(drop=True),
        node_features=compact_features,
        node_feature_columns=feature_columns,
        compact_to_original=used_original_ids,
        original_to_compact=original_to_compact,
        centroid_x_index=centroid_x_index,
        centroid_y_index=centroid_y_index,
        layer_map=layer_map,
        node_type_by_compact=node_type_by_compact,
        active_compact_mask=active_compact_mask,
        no_compact=no_compact,
        metadata_links=metadata_links,
        metadata_summary=metadata_summary,
        weight_column=getattr(args, "weight_col", None),
        duplicate_edge_count=duplicate_edge_count,
        self_loop_count=self_loop_count,
    )


def _extract_node_scalars(prepared: PreparedData) -> dict[str, np.ndarray]:
    return _extract_node_scalars_from_arrays(
        node_features=prepared.node_features,
        node_feature_columns=prepared.node_feature_columns,
        centroid_x_index=prepared.centroid_x_index,
        centroid_y_index=prepared.centroid_y_index,
    )


def build_layered_graph(prepared: PreparedData, directed: bool) -> Any:
    gt = _require_graph_tool()

    scalars = _extract_node_scalars(prepared)
    frame = prepared.compact_edges
    metadata_links = prepared.metadata_links if isinstance(prepared.metadata_links, pd.DataFrame) else pd.DataFrame()
    metadata_links = metadata_links.copy()
    metadata_enabled = bool(not metadata_links.empty)
    metadata_layer_id = int(len(prepared.layer_map)) if metadata_enabled else -1
    tag_keys = sorted(metadata_links["tag_key"].drop_duplicates().tolist()) if metadata_enabled else []
    data_vertex_count = len(prepared.compact_to_original)
    tag_vertex_by_key = {key: data_vertex_count + index for index, key in enumerate(tag_keys)}
    total_vertices = data_vertex_count + len(tag_keys)

    LOGGER.debug(
        "Building layered graph | directed=%s | data_nodes=%s | metadata_tags=%s | compact_edges=%s | active_nodes=%s | inactive_nodes=%s | node_type_labels=%s | ft_dimensions=%s | no_compact=%s | metadata_enabled=%s",
        directed,
        data_vertex_count,
        len(tag_keys),
        len(frame),
        int(prepared.active_compact_mask.sum()),
        int(len(prepared.active_compact_mask) - int(prepared.active_compact_mask.sum())),
        prepared.node_type_by_compact is not None,
        scalars["ft_matrix"].shape[1],
        prepared.no_compact,
        metadata_enabled,
    )
    graph = gt.Graph(directed=directed)
    graph.add_vertex(total_vertices)

    vertex_props = {
        "node_id": graph.new_vp("int"),
        "active_in_window": graph.new_vp("int"),
        "cx": graph.new_vp("double"),
        "cy": graph.new_vp("double"),
        "num_farms": graph.new_vp("double"),
        "total_animals": graph.new_vp("double"),
        "is_metadata_tag": graph.new_vp("int"),
        "partition_role": graph.new_vp("int"),
        "tag_label": graph.new_vp("string"),
        "tag_kind": graph.new_vp("string"),
        "tag_value": graph.new_vp("string"),
    }
    if prepared.node_type_by_compact is not None:
        vertex_props["type"] = graph.new_vp("int")

    # Data vertices occupy the first block of indices and keep the compact data-node order.
    for index, node_id in enumerate(prepared.compact_to_original.tolist()):
        vertex = graph.vertex(index)
        vertex_props["node_id"][vertex] = int(node_id)
        vertex_props["active_in_window"][vertex] = int(prepared.active_compact_mask[index])
        vertex_props["cx"][vertex] = float(scalars["cx"][index])
        vertex_props["cy"][vertex] = float(scalars["cy"][index])
        vertex_props["num_farms"][vertex] = float(scalars["num_farms"][index])
        vertex_props["total_animals"][vertex] = float(scalars["total_animals"][index])
        vertex_props["is_metadata_tag"][vertex] = 0
        role_value = int(prepared.node_type_by_compact[index]) if prepared.node_type_by_compact is not None else 0
        vertex_props["partition_role"][vertex] = role_value
        vertex_props["tag_label"][vertex] = ""
        vertex_props["tag_kind"][vertex] = ""
        vertex_props["tag_value"][vertex] = ""
        if "type" in vertex_props:
            vertex_props["type"][vertex] = int(prepared.node_type_by_compact[index])

    # Metadata-tag vertices are appended after the data-node block.
    tag_role_value = 2 if prepared.node_type_by_compact is not None else 1
    for tag_key, vertex_index in tag_vertex_by_key.items():
        vertex = graph.vertex(int(vertex_index))
        try:
            tag_kind, tag_value = str(tag_key).split("::", 1)
        except ValueError:
            tag_kind, tag_value = "metadata", str(tag_key)
        vertex_props["node_id"][vertex] = int(-(int(vertex_index) - data_vertex_count + 1))
        vertex_props["active_in_window"][vertex] = 0
        vertex_props["cx"][vertex] = 0.0
        vertex_props["cy"][vertex] = 0.0
        vertex_props["num_farms"][vertex] = 0.0
        vertex_props["total_animals"][vertex] = 0.0
        vertex_props["is_metadata_tag"][vertex] = 1
        vertex_props["partition_role"][vertex] = int(tag_role_value)
        vertex_props["tag_label"][vertex] = str(tag_key)
        vertex_props["tag_kind"][vertex] = str(tag_kind)
        vertex_props["tag_value"][vertex] = str(tag_value)
        if "type" in vertex_props:
            vertex_props["type"][vertex] = -1

    for name, prop in vertex_props.items():
        graph.vp[name] = prop

    edge_layer = graph.new_ep("int")
    edge_dist = graph.new_ep("double")
    edge_mass = graph.new_ep("double")
    edge_anim = graph.new_ep("double")
    edge_ftcos = graph.new_ep("double")
    edge_is_metadata = graph.new_ep("int")
    calendar_int = {name: graph.new_ep("int") for name in [f"dow_{index}" for index in range(7)] + ["holiday_nl"]}
    calendar_real = {name: graph.new_ep("double") for name in ["doy_sin", "doy_cos"]}

    weight_col = prepared.weight_column
    edge_weight_raw = edge_weight_log = edge_weight_log1p = edge_weight_int = None
    weight_values = None
    if weight_col:
        weight_values = frame[weight_col].to_numpy(dtype=float, copy=False)
        edge_weight_raw = graph.new_ep("double")
        if np.all(weight_values > 0):
            edge_weight_log = graph.new_ep("double")
        if np.all(weight_values >= 0):
            edge_weight_log1p = graph.new_ep("double")
        if np.all(weight_values >= 0) and _looks_like_int_array(weight_values):
            edge_weight_int = graph.new_ep("int64_t")
        LOGGER.debug(
            "Weight property compatibility | column=%s | raw=%s | log=%s | log1p=%s | int=%s | summary=%s",
            weight_col,
            edge_weight_raw is not None,
            edge_weight_log is not None,
            edge_weight_log1p is not None,
            edge_weight_int is not None,
            _format_numeric_summary(weight_values),
        )

    u_array = frame["u"].to_numpy(dtype=np.int64, copy=False)
    i_array = frame["i"].to_numpy(dtype=np.int64, copy=False)
    ts_array = frame["ts"].to_numpy(dtype=np.int64, copy=False)
    ft_matrix = scalars["ft_matrix"]
    ft_norm = scalars["ft_norm"]
    include_ft_cosine = bool(ft_matrix.shape[1] > 0 and np.any(ft_norm > 0))
    calendar_int_arrays = {
        name: frame[name].to_numpy(dtype=np.int64, copy=False) for name in calendar_int
    }
    calendar_real_arrays = {
        name: frame[name].to_numpy(dtype=float, copy=False) for name in calendar_real
    }

    for edge_index, (u_value, v_value, ts_value) in enumerate(zip(u_array, i_array, ts_array)):
        edge = graph.add_edge(int(u_value), int(v_value))
        edge_layer[edge] = int(prepared.layer_map[int(ts_value)])
        edge_is_metadata[edge] = 0

        dx = float(scalars["cx"][u_value] - scalars["cx"][v_value])
        dy = float(scalars["cy"][u_value] - scalars["cy"][v_value])
        edge_dist[edge] = (dx * dx + dy * dy) ** 0.5 / 1000.0

        edge_mass[edge] = float(
            np.log1p(max(0.0, scalars["num_farms"][u_value]) * max(0.0, scalars["num_farms"][v_value]))
        )
        edge_anim[edge] = float(
            np.log1p(max(0.0, scalars["total_animals"][u_value]) * max(0.0, scalars["total_animals"][v_value]))
        )

        if ft_matrix.shape[1] > 0 and ft_norm[u_value] > 0 and ft_norm[v_value] > 0:
            edge_ftcos[edge] = float(
                np.dot(ft_matrix[u_value], ft_matrix[v_value]) / (ft_norm[u_value] * ft_norm[v_value])
            )
        else:
            edge_ftcos[edge] = 0.0

        for name, prop in calendar_int.items():
            prop[edge] = int(calendar_int_arrays[name][edge_index])
        for name, prop in calendar_real.items():
            prop[edge] = float(calendar_real_arrays[name][edge_index])

        if weight_values is not None and edge_weight_raw is not None:
            weight_value = float(weight_values[edge_index])
            edge_weight_raw[edge] = weight_value
            if edge_weight_log is not None:
                edge_weight_log[edge] = float(np.log(weight_value))
            if edge_weight_log1p is not None:
                edge_weight_log1p[edge] = float(np.log1p(weight_value))
            if edge_weight_int is not None:
                edge_weight_int[edge] = int(round(weight_value))

    if metadata_enabled:
        def _set_metadata_edge_properties(edge: Any) -> None:
            edge_layer[edge] = int(metadata_layer_id)
            edge_is_metadata[edge] = 1
            edge_dist[edge] = 0.0
            edge_mass[edge] = 0.0
            edge_anim[edge] = 0.0
            edge_ftcos[edge] = 0.0
            for prop in calendar_int.values():
                prop[edge] = 0
            for prop in calendar_real.values():
                prop[edge] = 0.0
            if edge_weight_raw is not None:
                edge_weight_raw[edge] = 0.0
            if edge_weight_log is not None:
                edge_weight_log[edge] = 0.0
            if edge_weight_log1p is not None:
                edge_weight_log1p[edge] = 0.0
            if edge_weight_int is not None:
                edge_weight_int[edge] = 0

        bidirectional_metadata = bool(getattr(prepared, "no_compact", False) or directed)
        for row in metadata_links.itertuples(index=False):
            data_vertex = int(row.u)
            tag_vertex = int(tag_vertex_by_key[str(row.tag_key)])
            edge = graph.add_edge(data_vertex, tag_vertex)
            _set_metadata_edge_properties(edge)
            if directed and bidirectional_metadata:
                reverse_edge = graph.add_edge(tag_vertex, data_vertex)
                _set_metadata_edge_properties(reverse_edge)

    graph.ep["layer"] = edge_layer
    graph.ep["dist_km"] = edge_dist
    graph.ep["mass_grav"] = edge_mass
    graph.ep["anim_grav"] = edge_anim
    graph.ep["is_metadata"] = edge_is_metadata
    if include_ft_cosine:
        graph.ep["ft_cosine"] = edge_ftcos
    else:
        LOGGER.debug("Skipping ft_cosine covariate because no usable count_ft_* feature basis is available.")
    for name, prop in calendar_int.items():
        graph.ep[name] = prop
    for name, prop in calendar_real.items():
        graph.ep[name] = prop
    if edge_weight_raw is not None:
        graph.ep[WEIGHT_PROP_RAW] = edge_weight_raw
    if edge_weight_log is not None:
        graph.ep[WEIGHT_PROP_LOG] = edge_weight_log
    if edge_weight_log1p is not None:
        graph.ep[WEIGHT_PROP_LOG1P] = edge_weight_log1p
    if edge_weight_int is not None:
        graph.ep[WEIGHT_PROP_INT] = edge_weight_int
    graph.gp["num_layers"] = graph.new_gp("int", len(prepared.layer_map) + (1 if metadata_enabled else 0))
    graph.gp["num_trade_layers"] = graph.new_gp("int", len(prepared.layer_map))
    graph.gp["metadata_layer_id"] = graph.new_gp("int", int(metadata_layer_id))
    graph.gp["metadata_tag_count"] = graph.new_gp("int", int(len(tag_keys)))
    graph.gp["no_compact"] = graph.new_gp("bool", bool(prepared.no_compact))

    LOGGER.info(
        "Built layered graph | vertices=%s | data_vertices=%s | metadata_tags=%s | edges=%s | trade_layers=%s | metadata_layer=%s",
        graph.num_vertices(),
        data_vertex_count,
        len(tag_keys),
        graph.num_edges(),
        len(prepared.layer_map),
        metadata_layer_id,
    )
    LOGGER.debug(
        "Graph properties | vertex_props=%s | edge_props=%s",
        sorted(graph.vp.keys()),
        sorted(graph.ep.keys()),
    )
    return graph


def _covariate_rec_type(name: str) -> str:
    if name == "dist_km":
        return "real-exponential"
    if name in {"doy_sin", "doy_cos", "mass_grav", "anim_grav", "ft_cosine"}:
        return "real-normal"
    return "discrete-binomial"


def _available_covariate_specs(graph: Any) -> list[CovariateSpec]:
    return [
        CovariateSpec(name=name, graph_property=name, rec_type=_covariate_rec_type(name))
        for name in ALL_EDGE_COVARIATES
        if name in graph.ep
    ]


def _default_covariate_specs(graph: Any) -> list[CovariateSpec]:
    by_name = {spec.name: spec for spec in _available_covariate_specs(graph)}
    return [by_name[name] for name in DEFAULT_COVARIATES if name in by_name]


def _select_covariate_specs(
    available_covariate_specs: list[CovariateSpec],
    requested_names: Optional[Iterable[str]],
) -> list[CovariateSpec]:
    by_name = {spec.name: spec for spec in available_covariate_specs}
    if not requested_names:
        selected = [by_name[name] for name in DEFAULT_COVARIATES if name in by_name]
        if not selected:
            raise ValueError("None of the default fit covariates are available in the constructed graph.")
        return selected

    requested = list(dict.fromkeys(str(name) for name in requested_names))
    if requested == ["none"]:
        return []
    invalid = [name for name in requested if name not in ALL_EDGE_COVARIATES]
    if invalid:
        raise ValueError(
            f"Unknown fit covariates requested: {invalid}. "
            f"Valid choices are: {list(ALL_EDGE_COVARIATES)}"
        )

    selected = [by_name[name] for name in requested if name in by_name]
    if not selected:
        raise ValueError("None of the requested fit covariates are available in the constructed graph.")

    missing = [name for name in requested if name not in by_name]
    if missing:
        LOGGER.warning("Requested fit covariates are unavailable in this graph and will be skipped: %s", missing)
    LOGGER.debug("Selected fit covariates | requested=%s | selected=%s", requested, [spec.name for spec in selected])
    return selected


def _log_covariate_interpretation_notes(covariate_specs: Iterable[CovariateSpec]) -> None:
    names = [spec.name for spec in covariate_specs]
    if not names:
        return

    LOGGER.info(
        "Measured edge covariates are attached to realised edges only and are interpreted "
        "as edge annotations during inference, not as predictors evaluated over the full dyad set."
    )

    layer_derived = [name for name in names if name in LAYER_DERIVED_COVARIATES]
    if layer_derived:
        LOGGER.warning(
            "Layer-derived calendar covariates were requested for fitting (%s). Because the model "
            "already indexes time through the discrete layer assignment, these annotations can be "
            "redundant with the layer index.",
            layer_derived,
        )



def _build_weight_candidates(
    prepared: PreparedData,
    graph: Any,
    args: argparse.Namespace,
) -> list[WeightCandidate]:
    if not prepared.weight_column:
        return []

    input_column = prepared.weight_column
    raw_weights = prepared.compact_edges[input_column].to_numpy(dtype=float, copy=False)
    if raw_weights.size == 0:
        raise ValueError("No edge weights remain after filtering.")
    if not np.isfinite(raw_weights).all():
        raise ValueError(f"Edge weight column '{input_column}' contains NaN or infinite values.")

    model = str(getattr(args, "weight_model", "auto"))
    transform_arg = str(getattr(args, "weight_transform", "auto"))
    if model.startswith("discrete") and transform_arg not in {"auto", "none"}:
        raise ValueError("Discrete edge-weight models do not support transformed weights in this CLI.")

    is_nonnegative = bool(np.all(raw_weights >= 0))
    is_integer = bool(is_nonnegative and _looks_like_int_array(raw_weights))
    LOGGER.debug(
        "Weight support | column=%s | requested_model=%s | requested_transform=%s | nonnegative=%s | integer=%s | summary=%s",
        input_column,
        model,
        transform_arg,
        is_nonnegative,
        is_integer,
        _format_numeric_summary(raw_weights),
    )

    if model == "auto":
        if is_integer:
            candidate_pairs = [("discrete-poisson", "none"), ("discrete-geometric", "none")]
        elif is_nonnegative:
            candidate_pairs = [
                ("real-exponential", "none"),
                ("real-normal", "log" if np.all(raw_weights > 0) else "log1p"),
            ]
        else:
            candidate_pairs = [("real-normal", "none")]
    else:
        transform = transform_arg
        if transform == "auto":
            if model == "real-normal":
                if np.all(raw_weights > 0):
                    transform = "log"
                elif is_nonnegative:
                    transform = "log1p"
                else:
                    transform = "none"
            else:
                transform = "none"
        candidate_pairs = [(model, transform)]

    candidates: list[WeightCandidate] = []

    for rec_type, transform in candidate_pairs:
        if rec_type.startswith("discrete"):
            if not is_integer:
                raise ValueError(
                    f"Edge weight model '{rec_type}' requires nonnegative integer weights. "
                    f"Column '{input_column}' is not integer-valued."
                )
            graph_property = WEIGHT_PROP_INT
        elif transform == "none":
            graph_property = WEIGHT_PROP_RAW
        elif transform == "log":
            graph_property = WEIGHT_PROP_LOG
        elif transform == "log1p":
            graph_property = WEIGHT_PROP_LOG1P
        else:
            raise ValueError(f"Unsupported edge-weight transform: {transform}")

        if graph_property not in graph.ep:
            raise ValueError(
                f"Requested edge-weight specification rec_type={rec_type!r}, transform={transform!r} "
                "is incompatible with the observed weight support."
            )

        _, score_adjustment = _transform_weight_values(raw_weights, transform)
        candidates.append(
            WeightCandidate(
                input_column=input_column,
                graph_property=graph_property,
                rec_type=rec_type,
                transform=transform,
                score_adjustment=float(score_adjustment),
                model_label=f"{input_column}:{rec_type}/{transform}",
            )
        )
    LOGGER.debug(
        "Weight candidates | %s",
        [
            {
                "label": candidate.model_label,
                "graph_property": candidate.graph_property,
                "rec_type": candidate.rec_type,
                "transform": candidate.transform,
            }
            for candidate in candidates
        ],
    )
    return candidates


def _weight_generation_mode_name(args: argparse.Namespace) -> str:
    return str(getattr(args, "weight_generation_mode", "parametric")).strip().lower()


def _pure_generative_weight_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "weight_pure_generative", False))


def _validate_weight_generation_configuration(
    args: argparse.Namespace,
    *,
    has_weight_data: bool,
) -> None:
    weight_generation_mode = _weight_generation_mode_name(args)
    weight_partition_policy = str(getattr(args, "weight_parametric_partition_policy", "fixed")).strip().lower()
    pure_generative = _pure_generative_weight_mode(args)

    if weight_partition_policy not in {"fixed", "refit_on_refresh"}:
        raise ValueError(
            "Unknown weight_parametric_partition_policy. Use 'fixed' or 'refit_on_refresh'."
        )

    if pure_generative and weight_partition_policy == "refit_on_refresh":
        raise ValueError(
            "weight_pure_generative=True is incompatible with "
            "weight_parametric_partition_policy='refit_on_refresh' because that policy "
            "reuses observed weighted edges during generation."
        )

    if pure_generative and has_weight_data and weight_generation_mode not in {"parametric", "model", "generative"}:
        raise ValueError(
            "weight_pure_generative=True requires weight_generation_mode in {'parametric', 'model', 'generative'} "
            "so edge weights are generated only from fitted distributional models."
        )


def _fit_includes_edge_weight_covariate(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "exclude_weight_from_fit", False))


def _standalone_weight_model(prepared: PreparedData) -> Optional[dict[str, Any]]:
    if not prepared.weight_column:
        return None
    return {
        "input_column": prepared.weight_column,
        "output_column": prepared.weight_column,
        "candidate_label": "separate_parametric_generator",
        "fit_as_edge_covariate": False,
    }


def fit_nested_sbm(
    graph: Any,
    covariate_specs: Iterable[CovariateSpec],
    layered: bool,
    allow_mixed_node_types: bool,
    deg_corr: bool,
    overlap: bool,
    verbose: bool,
    refine_multiflip_rounds: int,
    refine_multiflip_niter: int,
    anneal_niter: int,
    anneal_beta_start: float,
    anneal_beta_stop: float,
) -> Any:
    gt = _require_graph_tool()
    covariate_specs = [spec for spec in covariate_specs if spec.graph_property in graph.ep]
    _log_covariate_specs("Starting nested SBM fit", covariate_specs)
    LOGGER.debug(
        "Fit options | layered=%s | allow_mixed_node_types=%s | deg_corr=%s | overlap=%s | fit_verbose=%s | refine_multiflip_rounds=%s | refine_multiflip_niter=%s | anneal_niter=%s | anneal_beta_start=%s | anneal_beta_stop=%s",
        layered,
        allow_mixed_node_types,
        deg_corr,
        overlap,
        verbose,
        refine_multiflip_rounds,
        refine_multiflip_niter,
        anneal_niter,
        anneal_beta_start,
        anneal_beta_stop,
    )

    rec_props = [graph.ep[spec.graph_property] for spec in covariate_specs]
    rec_types = [spec.rec_type for spec in covariate_specs]
    if allow_mixed_node_types:
        clabel = None
    elif "partition_role" in graph.vp:
        clabel = graph.vp["partition_role"]
    elif "type" in graph.vp:
        clabel = graph.vp["type"]
    else:
        clabel = None

    state = gt.minimize_nested_blockmodel_dl(
        graph,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            hentropy_args=dict(multigraph=False),
            state_args=dict(
                ec=graph.ep["layer"],
                layers=bool(layered),
                recs=rec_props,
                rec_types=rec_types,
                deg_corr=deg_corr,
                overlap=overlap,
                clabel=clabel,
            ),
        ),
        multilevel_mcmc_args=dict(
            verbose=bool(verbose),
        ),
    )

    if refine_multiflip_rounds > 0:
        for _ in range(int(refine_multiflip_rounds)):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=int(refine_multiflip_niter), verbose=bool(verbose))

    if anneal_niter > 0:
        gt.mcmc_anneal(
            state,
            beta_range=(float(anneal_beta_start), float(anneal_beta_stop)),
            niter=int(anneal_niter),
            mcmc_equilibrate_args=dict(force_niter=int(refine_multiflip_niter)),
            verbose=bool(verbose),
        )

    LOGGER.debug("Completed nested SBM fit | %s", _state_summary_text(state))
    return state


def _fit_with_weight_candidates(
    graph: Any,
    base_covariate_specs: list[CovariateSpec],
    weight_candidates: list[WeightCandidate],
    prepared: PreparedData,
    args: argparse.Namespace,
) -> tuple[Any, Optional[dict], list[str]]:
    if not weight_candidates:
        LOGGER.debug("No weight candidates selected; fitting topology-only model.")
        state = fit_nested_sbm(
            graph=graph,
            covariate_specs=base_covariate_specs,
            layered=bool(getattr(args, "layered", True)),
            allow_mixed_node_types=bool(getattr(args, "allow_mixed_node_types", False)),
            deg_corr=not args.no_deg_corr,
            overlap=bool(args.overlap),
            verbose=not args.fit_quiet,
            refine_multiflip_rounds=int(args.refine_multiflip_rounds),
            refine_multiflip_niter=int(args.refine_multiflip_niter),
            anneal_niter=int(args.anneal_niter),
            anneal_beta_start=float(args.anneal_beta_start),
            anneal_beta_stop=float(args.anneal_beta_stop),
        )
        return state, None, [spec.name for spec in base_covariate_specs]

    candidate_records: list[dict[str, Any]] = []
    best_state = None
    best_candidate: Optional[WeightCandidate] = None
    best_specs: list[CovariateSpec] = []
    best_score: Optional[float] = None

    for candidate in weight_candidates:
        LOGGER.info("Fitting edge-weight candidate %s", candidate.model_label)
        fit_specs = list(base_covariate_specs) + [
            CovariateSpec(
                name=candidate.model_label,
                graph_property=candidate.graph_property,
                rec_type=candidate.rec_type,
            )
        ]
        state = fit_nested_sbm(
            graph=graph,
            covariate_specs=fit_specs,
            layered=bool(getattr(args, "layered", True)),
            allow_mixed_node_types=bool(getattr(args, "allow_mixed_node_types", False)),
            deg_corr=not args.no_deg_corr,
            overlap=bool(args.overlap),
            verbose=not args.fit_quiet,
            refine_multiflip_rounds=int(args.refine_multiflip_rounds),
            refine_multiflip_niter=int(args.refine_multiflip_niter),
            anneal_niter=int(args.anneal_niter),
            anneal_beta_start=float(args.anneal_beta_start),
            anneal_beta_stop=float(args.anneal_beta_stop),
        )
        entropy = float(state.entropy())
        comparable_score = float(entropy + candidate.score_adjustment)
        record = {
            "candidate_label": candidate.model_label,
            "graph_property": candidate.graph_property,
            "rec_type": candidate.rec_type,
            "transform": candidate.transform,
            "entropy": entropy,
            "score_adjustment": float(candidate.score_adjustment),
            "comparable_score": comparable_score,
        }
        candidate_records.append(record)
        LOGGER.debug("Candidate result | %s", record)
        if best_score is None or comparable_score < best_score:
            best_score = comparable_score
            best_state = state
            best_candidate = candidate
            best_specs = fit_specs

    if best_state is None or best_candidate is None:
        raise RuntimeError("No valid edge-weight model candidate could be fitted.")

    raw_weights = prepared.compact_edges[prepared.weight_column].to_numpy(dtype=float, copy=False)
    weight_model = {
        "input_column": best_candidate.input_column,
        "output_column": best_candidate.input_column,
        "graph_property": best_candidate.graph_property,
        "rec_type": best_candidate.rec_type,
        "transform": best_candidate.transform,
        "candidate_label": best_candidate.model_label,
        "candidate_scores": candidate_records,
    }
    if best_candidate.rec_type == "discrete-binomial":
        observed_max = int(np.round(raw_weights.max())) if raw_weights.size else 1
        requested_max = getattr(args, "weight_binomial_max", None)
        weight_model["binomial_max"] = int(requested_max) if requested_max is not None else observed_max

    LOGGER.info(
        "Selected edge-weight model %s | comparable score=%.6f",
        best_candidate.model_label,
        float(best_score),
    )
    LOGGER.debug("Weight candidate ranking | %s", candidate_records)
    return best_state, weight_model, [spec.name for spec in best_specs]


def _base_state(nested_or_base: Any) -> Any:
    return nested_or_base.get_levels()[0] if hasattr(nested_or_base, "get_levels") else nested_or_base


def _node_blocks_from_state(nested_or_base: Any) -> np.ndarray:
    base = _base_state(nested_or_base)
    vertex_count = int(base.g.num_vertices()) if hasattr(base, "g") else -1

    if hasattr(nested_or_base, "get_bs"):
        try:
            block_levels = nested_or_base.get_bs()
            if block_levels:
                blocks = np.asarray(block_levels[0], dtype=np.int64).reshape(-1)
                if vertex_count < 0 or blocks.size == vertex_count:
                    return blocks
        except Exception:
            pass

    return np.asarray(base.get_nonoverlap_blocks().a, dtype=np.int64)


def _is_layered_base_type(base_type: Any) -> bool:
    return getattr(base_type, "__name__", None) == "LayeredBlockState"


def _split_layered_entropy_args(state_payload: dict[str, Any]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    payload = dict(state_payload)
    state_args = payload.get("state_args")
    if not isinstance(state_args, dict) or not _is_layered_base_type(payload.get("base_type")):
        return payload, None

    clean_state_args = dict(state_args)
    entropy_args = clean_state_args.pop("entropy_args", None)
    payload["state_args"] = clean_state_args
    if isinstance(entropy_args, dict) and entropy_args:
        return payload, dict(entropy_args)
    return payload, None


def _apply_layered_entropy_args(state: Any, entropy_args: Optional[dict[str, Any]]) -> None:
    if not entropy_args:
        return

    targets = [state, getattr(state, "agg_state", None)]
    layer_states = getattr(state, "layer_states", None)
    if layer_states is not None:
        try:
            targets.extend(list(layer_states))
        except Exception:
            pass

    for target in targets:
        if target is None or not hasattr(target, "update_entropy_args"):
            continue
        try:
            target.update_entropy_args(**entropy_args)
        except Exception:
            continue


def _serialise_graph_tool_state(state: Any) -> Any:
    if not hasattr(state, "get_levels") or not hasattr(state, "__getstate__"):
        return state

    state_payload = state.__getstate__()
    clean_state_payload, layered_base_entropy_args = _split_layered_entropy_args(state_payload)
    payload = {
        "format": GRAPH_TOOL_STATE_PAYLOAD_FORMAT,
        "state": clean_state_payload,
        "layered_base_entropy_args": layered_base_entropy_args,
    }
    LOGGER.debug(
        "Serialised graph-tool state | format=%s | layered_base_entropy_args=%s",
        GRAPH_TOOL_STATE_PAYLOAD_FORMAT,
        sorted(layered_base_entropy_args.keys()) if layered_base_entropy_args else [],
    )
    return payload


def _restore_graph_tool_state(payload: dict[str, Any]) -> Any:
    gt = _require_graph_tool()
    state = gt.NestedBlockState(**payload["state"])
    _apply_layered_entropy_args(_base_state(state), payload.get("layered_base_entropy_args"))
    LOGGER.debug(
        "Restored graph-tool state | format=%s | layered_base_entropy_args=%s",
        payload.get("format"),
        sorted((payload.get("layered_base_entropy_args") or {}).keys()),
    )
    return state


def _copy_graph_tool_state(state: Any) -> Any:
    payload = _serialise_graph_tool_state(state)
    if isinstance(payload, dict) and payload.get("format") == GRAPH_TOOL_STATE_PAYLOAD_FORMAT:
        return _restore_graph_tool_state(payload)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=GRAPH_TOOL_LAYERED_ENTROPY_WARNING)
        return state.copy()


def _state_summary_text(nested_state: Any) -> str:
    try:
        base = _base_state(nested_state)
        levels = len(nested_state.get_levels()) if hasattr(nested_state, "get_levels") else 1
        if hasattr(base, "get_nonempty_B"):
            try:
                block_count = int(base.get_nonempty_B())
            except Exception:
                block_count = -1
        else:
            block_count = -1
        if block_count < 0:
            blocks = np.asarray(base.get_nonoverlap_blocks().a, dtype=np.int64)
            block_count = int(np.unique(blocks).size)
        entropy = float(nested_state.entropy()) if hasattr(nested_state, "entropy") else float("nan")
        return (
            f"levels={levels} | blocks={block_count} | "
            f"vertices={int(base.g.num_vertices())} | edges={int(base.g.num_edges())} | entropy={entropy:.6f}"
        )
    except Exception as exc:
        return f"state_summary_unavailable ({exc})"


def attach_partition_maps(graph: Any, nested_state: Any) -> None:
    base = _base_state(nested_state)
    try:
        blocks = _node_blocks_from_state(nested_state)
        block_prop = graph.new_vp("int64_t")
        for index in range(int(graph.num_vertices())):
            block_prop[graph.vertex(index)] = int(blocks[index])
        graph.vp["sbm_b"] = graph.own_property(block_prop)
    except Exception:
        pass

    if getattr(base, "overlap", False):
        try:
            bv, bc_in, bc_out, bc_tot = base.get_overlap_blocks()
            graph.vp["sbm_bv"] = graph.own_property(bv.copy())
            graph.vp["sbm_bc_in"] = graph.own_property(bc_in.copy())
            graph.vp["sbm_bc_out"] = graph.own_property(bc_out.copy())
            graph.vp["sbm_bc_tot"] = graph.own_property(bc_tot.copy())
        except Exception:
            pass


def save_state(state: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(_serialise_graph_tool_state(state), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(path: Path) -> Any:
    with gzip.open(Path(path), "rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=GRAPH_TOOL_LAYERED_ENTROPY_WARNING)
            payload = pickle.load(handle)
    if isinstance(payload, dict) and payload.get("format") == GRAPH_TOOL_STATE_PAYLOAD_FORMAT:
        return _restore_graph_tool_state(payload)
    return payload


def map_graph_lid_to_state_lid(nested_state: Any) -> dict[int, int]:
    base = _base_state(nested_state)
    graph_layers = np.asarray(base.g.ep["layer"].a, dtype=np.int64)
    state_layers = np.asarray(base.ec.a, dtype=np.int64)
    mapping = {}
    for graph_lid in np.unique(graph_layers):
        candidates = state_layers[graph_layers == graph_lid]
        if candidates.size:
            mapping[int(graph_lid)] = int(np.bincount(candidates).argmax())
    return mapping


def extract_node_block_map(graph: Any) -> dict[int, int]:
    if "node_id" not in graph.vp or "sbm_b" not in graph.vp:
        return {}

    node_id_prop = graph.vp["node_id"]
    block_prop = graph.vp["sbm_b"]
    metadata_prop = graph.vp["is_metadata_tag"] if "is_metadata_tag" in graph.vp else None
    mapping: dict[int, int] = {}
    for vertex in graph.vertices():
        if metadata_prop is not None and bool(metadata_prop[vertex]):
            continue
        node_id = int(node_id_prop[vertex])
        if node_id < 0:
            continue
        mapping[node_id] = int(block_prop[vertex])
    return mapping


def load_node_block_map_from_graph_path(graph_path: Path) -> Optional[dict[int, int]]:
    graph_path = Path(graph_path)
    if not graph_path.exists():
        return None

    gt = _require_graph_tool()
    graph = gt.load_graph(str(graph_path))
    mapping = extract_node_block_map(graph)
    return mapping or None


def write_node_attributes(
    prepared: PreparedData,
    path: Path,
    node_blocks: Optional[dict[int, int]] = None,
) -> str:
    scalars = _extract_node_scalars(prepared)
    frame = pd.DataFrame(
        {
            "node_id": prepared.compact_to_original.astype(np.int64),
            "active_in_window": prepared.active_compact_mask.astype(np.int8),
            "x": scalars["cx"].astype(float),
            "y": scalars["cy"].astype(float),
            "num_farms": scalars["num_farms"].astype(float),
            "total_animals": scalars["total_animals"].astype(float),
        }
    )
    if prepared.node_type_by_compact is not None:
        frame["type"] = prepared.node_type_by_compact.astype(np.int64)
    node_map = pd.read_csv(prepared.input_paths.node_map_csv)
    keep_columns = ["node_id"]
    for column in ("type", "ubn", "corop"):
        if column in node_map.columns:
            keep_columns.append(column)
    metadata = node_map[keep_columns].drop_duplicates(subset=["node_id"]).copy()
    if "type" in metadata.columns:
        metadata = metadata.rename(columns={"type": "type_label"})
    frame = frame.merge(metadata, on="node_id", how="left")
    if node_blocks:
        frame["block_id"] = frame["node_id"].map(lambda node_id: int(node_blocks[int(node_id)]) if int(node_id) in node_blocks else -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _sample_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "self_loops": False,
        "multigraph": False,
    }
    if args.sample_max_ent or args.sample_canonical:
        kwargs["canonical"] = bool(args.sample_canonical)
        kwargs["max_ent"] = bool(args.sample_max_ent)
        kwargs["n_iter"] = int(args.sample_n_iter)
        if args.sample_canonical and not args.sample_max_ent:
            sample_params = getattr(args, "sample_params", None)
            kwargs["sample_params"] = True if sample_params is None else bool(sample_params)
    LOGGER.debug("Sample graph kwargs | %s", kwargs)
    return kwargs


def _generation_sample_mode_label(args: argparse.Namespace) -> str:
    if bool(getattr(args, "sample_canonical", False)) and bool(getattr(args, "sample_max_ent", False)):
        return "canonical_maxent"
    if bool(getattr(args, "sample_canonical", False)):
        sample_params = getattr(args, "sample_params", None)
        if sample_params is True:
            return "canonical_posterior"
        if sample_params is False:
            return "canonical_ml"
        return "canonical"
    if bool(getattr(args, "sample_max_ent", False)):
        return "maxent_micro"
    return "micro"


def _generation_setting_label(args: argparse.Namespace) -> str:
    output_subdir = getattr(args, "output_subdir", None)
    if output_subdir:
        return str(output_subdir)

    temporal_mode = _temporal_generator_mode_name(args)
    if temporal_mode == "operational":
        proposal_mode = _temporal_proposal_mode_name(args, temporal_mode=temporal_mode)
        if proposal_mode == "random":
            return "operational__proposal_random"
        rewire_model = str(getattr(args, "rewire_model", "none")).replace("-", "_")
        return f"operational__proposal_sbm__{_generation_sample_mode_label(args)}__rewire_{rewire_model}"

    rewire_model = str(getattr(args, "rewire_model", "none")).replace("-", "_")
    return f"independent_sbm__{_generation_sample_mode_label(args)}__rewire_{rewire_model}"


def _maybe_random_rewire_sample(
    sampled_graph: Any,
    layer_state: Any,
    blocks: np.ndarray,
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    rewire_model = str(getattr(args, "rewire_model", "none"))
    if rewire_model == "none":
        return None

    gt = _require_graph_tool()
    kwargs: dict[str, Any] = {
        "model": rewire_model,
        "n_iter": max(1, int(getattr(args, "rewire_n_iter", 10))),
        "edge_sweep": True,
        "parallel_edges": False,
        "self_loops": False,
        "persist": bool(getattr(args, "rewire_persist", False)),
    }

    if rewire_model in {"constrained-configuration", "blockmodel-micro"}:
        vmap = layer_state.g.vp["vmap"] if "vmap" in layer_state.g.vp else None
        if vmap is None:
            raise RuntimeError("Layer state is missing the vertex property 'vmap' required for block-aware rewiring.")
        block_membership = sampled_graph.new_vp("int64_t")
        for vertex in sampled_graph.vertices():
            base_index = int(vmap[layer_state.g.vertex(int(vertex))])
            block_membership[vertex] = int(blocks[base_index])
        kwargs["block_membership"] = block_membership

    rejection_count = int(gt.random_rewire(sampled_graph, **kwargs))
    summary = {
        "model": rewire_model,
        "n_iter": int(kwargs["n_iter"]),
        "persist": bool(kwargs["persist"]),
        "rejection_count": rejection_count,
    }
    LOGGER.debug("Applied random_rewire to sampled snapshot | %s", summary)
    return summary


def _posterior_partition_state(
    nested_state: Any,
    seed: int,
    args: argparse.Namespace,
) -> Any:
    gt = _require_graph_tool()
    gt.seed_rng(int(seed))
    state = _copy_graph_tool_state(nested_state)

    sweeps = max(0, int(getattr(args, "posterior_partition_sweeps", 0)))
    if sweeps <= 0:
        LOGGER.debug("Posterior partition refresh disabled for seed=%s", seed)
        return state

    LOGGER.debug(
        "Refreshing posterior partition | seed=%s | sweeps=%s | sweep_niter=%s | beta=%s",
        seed,
        sweeps,
        max(1, int(getattr(args, "posterior_partition_sweep_niter", 10))),
        float(getattr(args, "posterior_partition_beta", 1.0)),
    )
    gt.mcmc_equilibrate(
        state,
        force_niter=sweeps,
        mcmc_args=dict(
            niter=max(1, int(getattr(args, "posterior_partition_sweep_niter", 10))),
            beta=float(getattr(args, "posterior_partition_beta", 1.0)),
        ),
        verbose=False,
    )
    LOGGER.debug("Posterior partition refresh complete | %s", _state_summary_text(state))
    return state




WEIGHT_GENERATOR_PAYLOAD_FORMAT = "parametric_weight_generator_v2"
WEIGHT_GENERATOR_PAYLOAD_FORMAT_LEGACY = "parametric_weight_generator_v1"
TEMPORAL_GENERATOR_PAYLOAD_FORMAT = "temporal_generator_model_v3"


def _canonical_weight_channel(
    src_type: Optional[int],
    dst_type: Optional[int],
    directed: bool,
) -> tuple[Optional[int], Optional[int]]:
    if src_type is None or dst_type is None:
        return None, None
    src = int(src_type)
    dst = int(dst_type)
    if not directed and src > dst:
        src, dst = dst, src
    return src, dst


def _weight_channel_to_str(
    channel: tuple[Optional[int], Optional[int]],
) -> str:
    src_type, dst_type = channel
    return f"{'' if src_type is None else int(src_type)}|{'' if dst_type is None else int(dst_type)}"


def _weight_channel_from_str(
    text_value: str,
) -> tuple[Optional[int], Optional[int]]:
    parts = str(text_value).split("|")
    if len(parts) != 2:
        raise ValueError(f"Invalid serialised weight-channel key: {text_value!r}")
    return (
        None if parts[0] == "" else int(parts[0]),
        None if parts[1] == "" else int(parts[1]),
    )


def _weight_channel_label(
    src_type: Optional[int],
    dst_type: Optional[int],
) -> str:
    left = "*" if src_type is None else str(int(src_type))
    right = "*" if dst_type is None else str(int(dst_type))
    return f"{left}->{right}"


def _canonical_weight_key(
    ts_value: Optional[int],
    r: Optional[int],
    s: Optional[int],
    src_type: Optional[int],
    dst_type: Optional[int],
    directed: bool,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    src_type, dst_type = _canonical_weight_channel(src_type, dst_type, directed)
    ts_out = None if ts_value is None else int(ts_value)
    r_out = None if r is None else int(r)
    s_out = None if s is None else int(s)
    if not directed and r_out is not None and s_out is not None and r_out > s_out:
        r_out, s_out = s_out, r_out
    return ts_out, r_out, s_out, src_type, dst_type


def _weight_key_to_str(
    key: tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
) -> str:
    return "|".join("" if value is None else str(int(value)) for value in key)


def _weight_key_from_str(
    text_value: str,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    parts = str(text_value).split("|")
    if len(parts) != 5:
        raise ValueError(f"Invalid serialised weight-cell key: {text_value!r}")
    return tuple(None if part == "" else int(part) for part in parts)  # type: ignore[return-value]


def _weight_key_level(
    key: tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
) -> str:
    ts_value, r, s, src_type, dst_type = key
    if ts_value is None and r is None and s is None and src_type is None and dst_type is None:
        return "global"
    if ts_value is None and r is None and s is None and src_type is not None and dst_type is not None:
        return "channel"
    if ts_value is not None and r is None and s is None and src_type is None and dst_type is None:
        return "layer"
    if ts_value is None and r is not None and s is not None:
        return "block_pair"
    if ts_value is not None and r is None and s is None and src_type is not None and dst_type is not None:
        return "layer_channel"
    if ts_value is not None and r is not None and s is not None:
        return "exact"
    raise ValueError(f"Unrecognised weight-cell key: {key}")


def _weight_cell_keys(
    ts_value: int,
    r: int,
    s: int,
    *,
    src_type: Optional[int],
    dst_type: Optional[int],
    directed: bool,
) -> dict[str, tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]]:
    return {
        "exact": _canonical_weight_key(ts_value, r, s, src_type, dst_type, directed),
        "block_pair": _canonical_weight_key(None, r, s, src_type, dst_type, directed),
        "layer_channel": _canonical_weight_key(ts_value, None, None, src_type, dst_type, directed),
        "channel": _canonical_weight_key(None, None, None, src_type, dst_type, directed),
        "layer": _canonical_weight_key(ts_value, None, None, None, None, directed),
        "global": _canonical_weight_key(None, None, None, None, None, directed),
    }


def _weight_parent_keys(
    key: tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
    directed: bool,
) -> list[tuple[str, tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]]]:
    level = _weight_key_level(key)
    ts_value, r, s, src_type, dst_type = key
    global_key = _canonical_weight_key(None, None, None, None, None, directed)
    channel_key = _canonical_weight_key(None, None, None, src_type, dst_type, directed)
    layer_key = _canonical_weight_key(ts_value, None, None, None, None, directed)
    layer_channel_key = _canonical_weight_key(ts_value, None, None, src_type, dst_type, directed)
    block_pair_key = _canonical_weight_key(None, r, s, src_type, dst_type, directed)

    if level == "global":
        return []
    if level == "channel":
        return [("global", global_key)]
    if level == "layer":
        return [("global", global_key)]
    if level == "block_pair":
        return [("channel", channel_key), ("global", global_key)]
    if level == "layer_channel":
        return [("channel", channel_key), ("layer", layer_key), ("global", global_key)]
    if level == "exact":
        return [
            ("block_pair", block_pair_key),
            ("layer_channel", layer_channel_key),
            ("channel", channel_key),
            ("layer", layer_key),
            ("global", global_key),
        ]
    raise ValueError(f"Unsupported weight-cell level: {level}")


def _nb2_alpha_from_stats(stats: WeightCellStats) -> Optional[float]:
    if stats.n <= 1:
        return None
    mean_x = float(stats.sum_x / stats.n)
    if not np.isfinite(mean_x) or mean_x <= 1e-12:
        return 0.0
    var_x = float(stats.sum_x2 / stats.n - mean_x * mean_x)
    var_x = max(var_x, 0.0)
    alpha = (var_x - mean_x) / max(mean_x * mean_x, 1e-12)
    return float(max(alpha, 0.0))


def _sample_nb2(
    rng: np.random.Generator,
    *,
    mean: float,
    alpha: float,
) -> int:
    mean = max(float(mean), 0.0)
    alpha = max(float(alpha), 0.0)
    if mean <= 1e-12:
        return 0
    if alpha <= 1e-10:
        return int(rng.poisson(mean))

    shape = 1.0 / alpha
    if not np.isfinite(shape) or shape >= 1e8:
        return int(rng.poisson(mean))

    rate = float(rng.gamma(shape=shape, scale=mean / shape))
    return int(rng.poisson(max(rate, 0.0)))


def _select_parametric_weight_family(
    raw_weights: np.ndarray,
    requested: str,
) -> dict[str, Any]:
    values = np.asarray(raw_weights, dtype=float)
    if values.size == 0:
        raise ValueError("A parametric weight generator requires at least one observed weight.")

    requested = str(requested).strip().lower()
    is_nonnegative = bool(np.all(values >= 0))
    is_positive = bool(np.all(values > 0))
    is_integer = bool(is_nonnegative and _looks_like_int_array(values))

    if requested == "auto":
        if is_integer and is_positive:
            return {
                "family": "shifted-negbin",
                "shift": 1,
                "transform": "none",
                "support": "positive_integer",
            }
        if is_integer and is_nonnegative:
            return {
                "family": "negbin",
                "shift": 0,
                "transform": "none",
                "support": "nonnegative_integer",
            }
        if is_positive:
            return {
                "family": "lognormal",
                "shift": 0,
                "transform": "log",
                "support": "positive_real",
            }
        if is_nonnegative:
            return {
                "family": "lognormal",
                "shift": 0,
                "transform": "log1p",
                "support": "nonnegative_real",
            }
        raise ValueError("Parametric weight generation only supports nonnegative weights in this pipeline.")

    if requested in {"shifted-negbin", "shifted_nbinom", "shifted-negbinom", "shifted_nb"}:
        if not is_integer or not is_positive:
            raise ValueError("The shifted negative-binomial generator requires strictly positive integer weights.")
        return {
            "family": "shifted-negbin",
            "shift": 1,
            "transform": "none",
            "support": "positive_integer",
        }

    if requested in {"negbin", "negative-binomial", "negative_binomial", "nb"}:
        if not is_integer or not is_nonnegative:
            raise ValueError("The negative-binomial generator requires nonnegative integer weights.")
        return {
            "family": "negbin",
            "shift": 0,
            "transform": "none",
            "support": "nonnegative_integer",
        }

    if requested in {"lognormal", "log-normal"}:
        if not is_nonnegative:
            raise ValueError("The log-normal generator requires nonnegative weights.")
        return {
            "family": "lognormal",
            "shift": 0,
            "transform": "log" if is_positive else "log1p",
            "support": "positive_real" if is_positive else "nonnegative_real",
        }

    raise ValueError(
        f"Unknown parametric weight generator family: {requested!r}. "
        "Use auto, shifted-negbin, negbin, or lognormal."
    )


def _parametric_weight_shrinkage(args: argparse.Namespace) -> dict[str, float]:
    prior_strength = max(0.0, float(getattr(args, "weight_prior_strength", 5.0)))
    return {
        "block_pair": prior_strength,
        "layer_channel": prior_strength,
        "channel": 2.0 * prior_strength,
        "layer": 2.0 * prior_strength,
        "global": 4.0 * prior_strength,
    }


def _fit_parametric_weight_generator_model(
    observed_edges: pd.DataFrame,
    base: Any,
    weight_model: dict,
    directed: bool,
    args: argparse.Namespace,
    blocks: Optional[np.ndarray] = None,
) -> dict:
    weight_col = str(weight_model.get("output_column") or weight_model.get("input_column"))
    required = {"u", "i", "ts", weight_col}
    missing = required.difference(observed_edges.columns)
    if missing:
        raise ValueError(
            "Observed edge table is missing required columns for parametric weight fitting: "
            f"{sorted(missing)}"
        )

    frame = observed_edges[["u", "i", "ts", weight_col]].copy()
    frame["u"] = pd.to_numeric(frame["u"], errors="raise").astype(np.int64)
    frame["i"] = pd.to_numeric(frame["i"], errors="raise").astype(np.int64)
    frame["ts"] = pd.to_numeric(frame["ts"], errors="raise").astype(np.int64)
    frame[weight_col] = pd.to_numeric(frame[weight_col], errors="raise").astype(float)
    raw_weights = frame[weight_col].to_numpy(dtype=float, copy=False)
    if raw_weights.size == 0:
        raise ValueError("No observed weights are available for parametric fitting.")

    node_id_prop = base.g.vp["node_id"] if "node_id" in base.g.vp else None
    if node_id_prop is None:
        raise RuntimeError("Fitted graph is missing the vertex property 'node_id' required for weight fitting.")
    type_prop = base.g.vp["type"] if "type" in base.g.vp else None
    if blocks is None:
        blocks = _node_blocks_from_state(base)
    else:
        blocks = np.asarray(blocks, dtype=np.int64)

    node_id_to_block: dict[int, int] = {}
    node_id_to_type: dict[int, int] = {}
    for index in range(int(base.g.num_vertices())):
        vertex = base.g.vertex(index)
        node_id = int(node_id_prop[vertex])
        node_id_to_block[node_id] = int(blocks[index])
        if type_prop is not None:
            node_id_to_type[node_id] = int(type_prop[vertex])

    rows = list(frame.itertuples(index=False))
    channel_records: dict[
        tuple[Optional[int], Optional[int]],
        list[tuple[Any, float, Optional[int], Optional[int]]],
    ] = defaultdict(list)
    for row, raw_weight in zip(rows, raw_weights):
        src_type = node_id_to_type.get(int(row.u)) if node_id_to_type else None
        dst_type = node_id_to_type.get(int(row.i)) if node_id_to_type else None
        channel = _canonical_weight_channel(src_type, dst_type, bool(directed))
        channel_records[channel].append((row, float(raw_weight), src_type, dst_type))

    if not channel_records:
        raise RuntimeError("Failed to assign any observed weighted edges to a weight channel.")

    requested_family = str(getattr(args, "weight_parametric_family", "auto"))
    shrinkage = _parametric_weight_shrinkage(args)
    level_order = ["global", "channel", "layer", "block_pair", "layer_channel", "exact"]
    channel_models: dict[str, dict[str, Any]] = {}
    aggregate_level_counts: Counter[str] = Counter()

    for channel in sorted(channel_records, key=_weight_channel_to_str):
        records = channel_records[channel]
        channel_raw_weights = np.asarray([record[1] for record in records], dtype=float)
        family_spec = _select_parametric_weight_family(channel_raw_weights, requested=requested_family)

        if family_spec["family"] == "lognormal":
            if family_spec["transform"] == "log":
                model_values = np.log(channel_raw_weights)
            else:
                model_values = np.log1p(channel_raw_weights)
        else:
            shift = int(family_spec["shift"])
            model_values = np.round(channel_raw_weights).astype(np.int64) - shift
            if np.any(model_values < 0):
                raise ValueError(
                    "The selected parametric count family produced negative shifted weights. "
                    "Check the observed support or choose a different family."
                )
            model_values = model_values.astype(float)

        stats: dict[
            tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
            WeightCellStats,
        ] = defaultdict(WeightCellStats)

        for (row, _, src_type, dst_type), model_value in zip(records, model_values):
            u_block = node_id_to_block.get(int(row.u))
            v_block = node_id_to_block.get(int(row.i))
            if u_block is None or v_block is None:
                continue

            for key in _weight_cell_keys(
                int(row.ts),
                int(u_block),
                int(v_block),
                src_type=src_type,
                dst_type=dst_type,
                directed=bool(directed),
            ).values():
                stats[key].update(float(model_value))

        global_key = _canonical_weight_key(None, None, None, None, None, bool(directed))
        if global_key not in stats or stats[global_key].n <= 0:
            raise RuntimeError(
                "Failed to accumulate global sufficient statistics for parametric weights "
                f"in channel {_weight_channel_label(*channel)}."
            )

        fitted_params: dict[
            tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
            dict[str, Any],
        ] = {}
        for level in level_order:
            level_keys = sorted(
                [key for key in stats if _weight_key_level(key) == level],
                key=_weight_key_to_str,
            )
            for key in level_keys:
                cell_stats = stats[key]
                parent_keys = _weight_parent_keys(key, bool(directed))

                mean_num = float(cell_stats.sum_x)
                mean_den = float(cell_stats.n)
                for parent_name, parent_key in parent_keys:
                    parent_params = fitted_params.get(parent_key)
                    if parent_params is None or int(parent_params.get("n", 0)) <= 0:
                        continue
                    strength = min(float(shrinkage.get(parent_name, 0.0)), float(parent_params["n"]))
                    if strength <= 0.0:
                        continue
                    mean_num += strength * float(parent_params["mean"])
                    mean_den += strength
                mean_hat = mean_num / mean_den if mean_den > 0.0 else 0.0

                param_record: dict[str, Any] = {
                    "level": level,
                    "n": int(cell_stats.n),
                    "mean": float(mean_hat),
                }

                if family_spec["family"] == "lognormal":
                    raw_var = None
                    if cell_stats.n > 0:
                        raw_mean = float(cell_stats.sum_x / cell_stats.n)
                        raw_var = float(cell_stats.sum_x2 / cell_stats.n - raw_mean * raw_mean)
                        raw_var = max(raw_var, 1e-9)

                    var_num = 0.0
                    var_den = 0.0
                    if raw_var is not None and cell_stats.n > 1:
                        own_weight = float(cell_stats.n - 1)
                        var_num += own_weight * raw_var
                        var_den += own_weight

                    for parent_name, parent_key in parent_keys:
                        parent_params = fitted_params.get(parent_key)
                        if parent_params is None or "variance" not in parent_params or int(parent_params.get("n", 0)) <= 0:
                            continue
                        strength = min(float(shrinkage.get(parent_name, 0.0)), float(parent_params["n"]))
                        if strength <= 0.0:
                            continue
                        var_num += strength * float(parent_params["variance"])
                        var_den += strength

                    if var_den <= 0.0:
                        variance_hat = raw_var if raw_var is not None else 1.0
                    else:
                        variance_hat = var_num / var_den
                    param_record["variance"] = float(max(variance_hat, 1e-9))
                else:
                    raw_alpha = _nb2_alpha_from_stats(cell_stats)
                    alpha_num = 0.0
                    alpha_den = 0.0
                    if raw_alpha is not None and cell_stats.n > 1:
                        own_weight = float(cell_stats.n - 1)
                        alpha_num += own_weight * float(raw_alpha)
                        alpha_den += own_weight

                    for parent_name, parent_key in parent_keys:
                        parent_params = fitted_params.get(parent_key)
                        if parent_params is None or "alpha" not in parent_params or int(parent_params.get("n", 0)) <= 0:
                            continue
                        strength = min(float(shrinkage.get(parent_name, 0.0)), float(parent_params["n"]))
                        if strength <= 0.0:
                            continue
                        alpha_num += strength * float(parent_params["alpha"])
                        alpha_den += strength

                    alpha_hat = alpha_num / alpha_den if alpha_den > 0.0 else 0.0
                    param_record["alpha"] = float(max(alpha_hat, 0.0))
                    param_record["shift"] = int(family_spec["shift"])

                fitted_params[key] = param_record

        serialised_cells = {
            _weight_key_to_str(key): value
            for key, value in sorted(fitted_params.items(), key=lambda item: _weight_key_to_str(item[0]))
        }
        level_counts = Counter(value["level"] for value in serialised_cells.values())
        aggregate_level_counts.update(level_counts)
        channel_key = _weight_channel_to_str(channel)
        channel_models[channel_key] = {
            "channel": {
                "src_type": None if channel[0] is None else int(channel[0]),
                "dst_type": None if channel[1] is None else int(channel[1]),
                "label": _weight_channel_label(channel[0], channel[1]),
            },
            "family": family_spec["family"],
            "transform": family_spec["transform"],
            "shift": int(family_spec["shift"]),
            "support": family_spec["support"],
            "fallback_order": ["exact", "block_pair", "layer_channel", "channel", "layer", "global"],
            "cells": serialised_cells,
            "summary": {
                "channel_key": channel_key,
                "channel_label": _weight_channel_label(channel[0], channel[1]),
                "num_cells": int(len(serialised_cells)),
                "edge_count": int(len(records)),
                "weight_total": float(channel_raw_weights.sum()),
                "level_counts": {str(level): int(count) for level, count in sorted(level_counts.items())},
                "family": family_spec["family"],
                "support": family_spec["support"],
                "model_scale_summary": _format_numeric_summary(model_values),
                "raw_weight_summary": _format_numeric_summary(channel_raw_weights),
            },
        }

    if not channel_models:
        raise RuntimeError("Parametric weight fitting produced no channel-specific models.")

    sorted_channel_models = dict(sorted(channel_models.items(), key=lambda item: item[0]))
    families_by_channel = {
        channel_key: str(channel_model["family"])
        for channel_key, channel_model in sorted_channel_models.items()
    }
    channel_summaries = {
        channel_key: channel_model["summary"]
        for channel_key, channel_model in sorted_channel_models.items()
    }
    only_channel_model = next(iter(sorted_channel_models.values())) if len(sorted_channel_models) == 1 else None
    total_num_cells = int(sum(int(channel_model["summary"]["num_cells"]) for channel_model in sorted_channel_models.values()))

    model = {
        "format": WEIGHT_GENERATOR_PAYLOAD_FORMAT,
        "mode": "parametric",
        "output_column": weight_col,
        "directed": bool(directed),
        "family": str(only_channel_model["family"]) if only_channel_model is not None else "per_channel",
        "transform": str(only_channel_model["transform"]) if only_channel_model is not None else "per_channel",
        "shift": int(only_channel_model["shift"]) if only_channel_model is not None else 0,
        "support": str(only_channel_model["support"]) if only_channel_model is not None else "per_channel",
        "channel_models": sorted_channel_models,
        "node_blocks": {str(node_id): int(block_id) for node_id, block_id in sorted(node_id_to_block.items())},
        "node_types": {str(node_id): int(type_value) for node_id, type_value in sorted(node_id_to_type.items())},
        "shrinkage": {name: float(value) for name, value in shrinkage.items()},
        "reference_partition_source": "fitted_or_refreshed_state",
        "summary": {
            "num_channels": int(len(sorted_channel_models)),
            "num_cells": total_num_cells,
            "level_counts": {str(level): int(count) for level, count in sorted(aggregate_level_counts.items())},
            "weight_column": weight_col,
            "requested_family": requested_family,
            "families_by_channel": families_by_channel,
            "channel_summaries": channel_summaries,
            "raw_weight_summary": _format_numeric_summary(raw_weights),
        },
    }
    LOGGER.info(
        "Fitted parametric weight generator | column=%s | channels=%s | families=%s | cells=%s",
        weight_col,
        len(sorted_channel_models),
        families_by_channel,
        total_num_cells,
    )
    LOGGER.debug("Parametric weight generator summary | %s", model["summary"])
    return model


class ParametricWeightSampler:
    def __init__(
        self,
        weight_generator_model: dict,
        directed: bool,
        rng: np.random.Generator,
    ) -> None:
        payload_format = str(weight_generator_model.get("format"))
        if payload_format not in {WEIGHT_GENERATOR_PAYLOAD_FORMAT, WEIGHT_GENERATOR_PAYLOAD_FORMAT_LEGACY}:
            raise ValueError(
                "Unsupported parametric weight generator payload format: "
                f"{weight_generator_model.get('format')!r}"
            )
        self.weight_col = str(weight_generator_model["output_column"])
        self.directed = bool(directed)
        self.rng = rng
        self.payload_format = payload_format
        self.channel_models = self._load_channel_models(weight_generator_model)
        self.resolution_counts: Counter[str] = Counter()

    def _parse_channel_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "family": str(payload["family"]),
            "transform": str(payload.get("transform", "none")),
            "shift": int(payload.get("shift", 0)),
            "fallback_order": [str(value) for value in payload.get("fallback_order", [])],
            "cells": {
                _weight_key_from_str(key): value
                for key, value in payload.get("cells", {}).items()
            },
        }

    def _load_channel_models(self, payload: dict[str, Any]) -> dict[tuple[Optional[int], Optional[int]], dict[str, Any]]:
        channel_models_payload = payload.get("channel_models")
        if isinstance(channel_models_payload, dict) and channel_models_payload:
            parsed: dict[tuple[Optional[int], Optional[int]], dict[str, Any]] = {}
            for channel_key, channel_payload in sorted(channel_models_payload.items(), key=lambda item: str(item[0])):
                parsed[_weight_channel_from_str(str(channel_key))] = self._parse_channel_model(channel_payload)
            return parsed
        return {
            _canonical_weight_channel(None, None, self.directed): self._parse_channel_model(payload),
        }

    def _resolve_channel_model(
        self,
        *,
        src_type: Optional[int],
        dst_type: Optional[int],
    ) -> tuple[dict[str, Any], tuple[Optional[int], Optional[int]]]:
        channel = _canonical_weight_channel(src_type, dst_type, self.directed)
        params = self.channel_models.get(channel)
        if params is not None:
            return params, channel

        default_channel = _canonical_weight_channel(None, None, self.directed)
        if default_channel in self.channel_models and len(self.channel_models) == 1:
            return self.channel_models[default_channel], channel

        raise RuntimeError(
            "Parametric weight generator is missing a fitted channel model for "
            f"{_weight_channel_label(channel[0], channel[1])}."
        )

    def _resolve_params(
        self,
        channel_model: dict[str, Any],
        ts_value: int,
        r: int,
        s: int,
        *,
        src_type: Optional[int],
        dst_type: Optional[int],
    ) -> tuple[dict[str, Any], str]:
        keys = _weight_cell_keys(
            int(ts_value),
            int(r),
            int(s),
            src_type=src_type,
            dst_type=dst_type,
            directed=self.directed,
        )
        for label in channel_model["fallback_order"]:
            key = keys.get(label)
            if key is None:
                continue
            params = channel_model["cells"].get(key)
            if params is not None:
                return params, label

        global_key = _canonical_weight_key(None, None, None, None, None, self.directed)
        params = channel_model["cells"].get(global_key)
        if params is None:
            raise RuntimeError("Parametric weight generator is missing the global cell for its resolved channel model.")
        return params, "global"

    def sample(
        self,
        ts_value: int,
        r: int,
        s: int,
        *,
        src_type: Optional[int] = None,
        dst_type: Optional[int] = None,
    ) -> float | int:
        if not self.directed and r > s:
            r, s = s, r
        src_type, dst_type = _canonical_weight_channel(src_type, dst_type, self.directed)
        channel_model, requested_channel = self._resolve_channel_model(src_type=src_type, dst_type=dst_type)
        params, resolution = self._resolve_params(
            channel_model,
            int(ts_value),
            int(r),
            int(s),
            src_type=src_type,
            dst_type=dst_type,
        )
        channel_key = _weight_channel_to_str(requested_channel)
        self.resolution_counts[resolution] += 1
        self.resolution_counts[f"{channel_key}::{resolution}"] += 1

        mean_x = max(float(params.get("mean", 0.0)), 0.0)
        family = str(channel_model["family"])
        transform = str(channel_model.get("transform", "none"))
        shift = int(channel_model.get("shift", 0))
        if family in {"shifted-negbin", "negbin"}:
            alpha = max(float(params.get("alpha", 0.0)), 0.0)
            sampled = _sample_nb2(self.rng, mean=mean_x, alpha=alpha)
            return int(max(0, sampled + shift))

        if family == "lognormal":
            variance = max(float(params.get("variance", 1.0)), 1e-9)
            sampled_x = float(self.rng.normal(loc=mean_x, scale=math.sqrt(variance)))
            if transform == "log":
                return max(0.0, float(np.exp(sampled_x)))
            if transform == "log1p":
                return max(0.0, float(np.expm1(sampled_x)))
            raise ValueError(f"Unsupported parametric transform: {transform!r}")

        raise ValueError(f"Unsupported parametric weight family: {family!r}")


def _stored_weight_reference_blocks(
    base: Any,
    blocks: np.ndarray,
    node_id_prop: Any,
    weight_generator_model: Optional[dict],
) -> Optional[np.ndarray]:
    if not weight_generator_model:
        return None

    node_blocks = weight_generator_model.get("node_blocks")
    if not isinstance(node_blocks, dict) or not node_blocks:
        return None

    reference = np.asarray(blocks, dtype=np.int64).copy()
    for index in range(int(base.g.num_vertices())):
        node_id = int(node_id_prop[base.g.vertex(index)])
        reference[index] = int(node_blocks.get(str(node_id), reference[index]))
    return reference


def _aligned_blocks_for_weight_generation(
    gt: Any,
    base: Any,
    blocks: np.ndarray,
    node_id_prop: Any,
    weight_generator_model: Optional[dict],
) -> np.ndarray:
    reference = _stored_weight_reference_blocks(
        base=base,
        blocks=blocks,
        node_id_prop=node_id_prop,
        weight_generator_model=weight_generator_model,
    )
    if reference is None:
        return blocks

    try:
        aligned = np.asarray(gt.align_partition_labels(blocks, reference), dtype=np.int64)
        if aligned.shape == reference.shape:
            LOGGER.debug(
                "Aligned sampled partition labels to the stored weight-model labels | overlap=%.6f",
                float(np.mean(aligned == reference)) if aligned.size else 1.0,
            )
            return aligned
    except Exception as exc:
        LOGGER.debug("Block-label alignment for weight generation failed | %s", exc)
    return blocks


class EdgeWeightSampler:
    def __init__(
        self,
        observed_edges: pd.DataFrame,
        node_id_to_base: dict[int, int],
        blocks: np.ndarray,
        weight_model: dict,
        directed: bool,
        min_cell_count: int,
        rng: np.random.Generator,
        node_id_to_type: Optional[dict[int, int]] = None,
    ) -> None:
        self.weight_col = str(weight_model["output_column"])
        self.rec_type = str(weight_model["rec_type"])
        self.transform = str(weight_model.get("transform", "none"))
        self.directed = bool(directed)
        self.min_cell_count = max(1, int(min_cell_count))
        self.rng = rng
        self.binomial_max = weight_model.get("binomial_max")
        self.node_id_to_type = (
            {int(node_id): int(type_value) for node_id, type_value in node_id_to_type.items()}
            if node_id_to_type
            else None
        )
        # Keys are (ts, r, s, source_type, target_type). Block-pair terms use the
        # sampled partition, and the type terms keep the hybrid edge channel during
        # shrinkage and backoff.
        self.stats: dict[
            tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]],
            WeightCellStats,
        ] = defaultdict(WeightCellStats)
        self.resolution_counts: Counter[str] = Counter()

        required = {"u", "i", "ts", self.weight_col}
        missing = required.difference(observed_edges.columns)
        if missing:
            raise ValueError(f"Observed edge table is missing required columns for weight sampling: {sorted(missing)}")

        frame = observed_edges[["u", "i", "ts", self.weight_col]].copy()
        frame["u"] = pd.to_numeric(frame["u"], errors="raise").astype(np.int64)
        frame["i"] = pd.to_numeric(frame["i"], errors="raise").astype(np.int64)
        frame["ts"] = pd.to_numeric(frame["ts"], errors="raise").astype(np.int64)
        frame[self.weight_col] = pd.to_numeric(frame[self.weight_col], errors="raise").astype(float)

        raw_values = frame[self.weight_col].to_numpy(dtype=float, copy=False)
        transformed_values, _ = _transform_weight_values(raw_values, self.transform)

        self.global_mean = float(transformed_values.mean()) if transformed_values.size else 0.0
        self.global_var = float(transformed_values.var(ddof=0)) if transformed_values.size else 1.0
        if not np.isfinite(self.global_var) or self.global_var <= 1e-12:
            self.global_var = max(abs(self.global_mean), 1.0)

        if self.rec_type == "discrete-binomial":
            observed_max = int(np.round(raw_values.max())) if raw_values.size else 1
            if self.binomial_max is None:
                self.binomial_max = observed_max
            self.binomial_max = max(1, int(self.binomial_max))

        for row, transformed_value in zip(frame.itertuples(index=False), transformed_values):
            u_base = node_id_to_base.get(int(row.u))
            v_base = node_id_to_base.get(int(row.i))
            if u_base is None or v_base is None:
                continue
            r = int(blocks[u_base])
            s = int(blocks[v_base])
            if not self.directed and r > s:
                r, s = s, r

            src_type, dst_type = self._channel_from_node_ids(int(row.u), int(row.i))
            for key in self._candidate_keys(int(row.ts), r, s, src_type=src_type, dst_type=dst_type):
                self.stats[key].update(float(transformed_value))

        self.global_stats = self.stats.get((None, None, None, None, None), WeightCellStats())
        self.channel_stats: dict[tuple[int, int], WeightCellStats] = {}
        for (ts_value, r, s, src_type, dst_type), stats in self.stats.items():
            if ts_value is None and r is None and s is None and src_type is not None and dst_type is not None and stats.n > 0:
                self.channel_stats[(int(src_type), int(dst_type))] = stats

        exact_cells = sum(
            1
            for ts_value, r, s, src_type, dst_type in self.stats
            if ts_value is not None and r is not None and s is not None
        )
        sparse_exact_cells = sum(
            1
            for (ts_value, r, s, src_type, dst_type), stats in self.stats.items()
            if ts_value is not None and r is not None and s is not None and stats.n < self.min_cell_count
        )
        LOGGER.debug(
            "Initialized edge-weight sampler | weight_col=%s | rec_type=%s | transform=%s | directed=%s | "
            "min_cell_count=%s | exact_cells=%s | sparse_exact_cells=%s | channel_cells=%s | global_summary=%s",
            self.weight_col,
            self.rec_type,
            self.transform,
            self.directed,
            self.min_cell_count,
            exact_cells,
            sparse_exact_cells,
            len(self.channel_stats),
            _format_numeric_summary(raw_values),
        )

    def _canonical_channel(
        self,
        src_type: Optional[int],
        dst_type: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        if src_type is None or dst_type is None:
            return None, None
        src = int(src_type)
        dst = int(dst_type)
        if not self.directed and src > dst:
            src, dst = dst, src
        return src, dst

    def _channel_from_node_ids(self, u_node_id: int, v_node_id: int) -> tuple[Optional[int], Optional[int]]:
        if not self.node_id_to_type:
            return None, None
        return self._canonical_channel(
            self.node_id_to_type.get(int(u_node_id)),
            self.node_id_to_type.get(int(v_node_id)),
        )

    def _candidate_keys(
        self,
        ts_value: int,
        r: int,
        s: int,
        *,
        src_type: Optional[int] = None,
        dst_type: Optional[int] = None,
    ) -> list[tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]]:
        src_type, dst_type = self._canonical_channel(src_type, dst_type)
        keys: list[tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]] = [
            (int(ts_value), int(r), int(s), None, None),
            (None, int(r), int(s), None, None),
        ]
        if src_type is not None and dst_type is not None:
            keys.extend(
                [
                    (int(ts_value), None, None, int(src_type), int(dst_type)),
                    (None, None, None, int(src_type), int(dst_type)),
                ]
            )
        keys.extend(
            [
                (int(ts_value), None, None, None, None),
                (None, None, None, None, None),
            ]
        )
        return keys

    def _resolve_stats(
        self,
        ts_value: int,
        r: int,
        s: int,
        *,
        src_type: Optional[int] = None,
        dst_type: Optional[int] = None,
    ) -> tuple[WeightCellStats, str]:
        fallback = None
        fallback_label = "global"
        labels = ["exact", "block_pair"]
        src_type, dst_type = self._canonical_channel(src_type, dst_type)
        if src_type is not None and dst_type is not None:
            labels.extend(["layer_channel", "channel"])
        labels.extend(["layer", "global"])

        for label, key in zip(labels, self._candidate_keys(ts_value, r, s, src_type=src_type, dst_type=dst_type)):
            stats = self.stats.get(key)
            if stats is None or stats.n <= 0:
                continue
            if stats.n >= self.min_cell_count:
                return stats, label
            if fallback is None:
                fallback = stats
                fallback_label = f"{label}_sparse"
        return fallback or self.global_stats, fallback_label

    def _prior_moments(
        self,
        *,
        src_type: Optional[int] = None,
        dst_type: Optional[int] = None,
    ) -> tuple[float, float, str]:
        src_type, dst_type = self._canonical_channel(src_type, dst_type)
        if src_type is not None and dst_type is not None:
            channel_stats = self.channel_stats.get((int(src_type), int(dst_type)))
            if channel_stats is not None and channel_stats.n > 0:
                mean_x = float(channel_stats.sum_x / channel_stats.n)
                var_x = float(channel_stats.sum_x2 / channel_stats.n - mean_x * mean_x)
                if not np.isfinite(var_x) or var_x <= 1e-12:
                    var_x = max(abs(mean_x), 1.0)
                return mean_x, var_x, "channel"
        return self.global_mean, self.global_var, "global"

    def _sample_real_exponential(self, stats: WeightCellStats, *, prior_mean: float) -> float:
        mean_x = max(float(prior_mean), 1e-9)
        alpha0 = 1.0
        beta0 = 1.0 / mean_x
        alpha_n = alpha0 + stats.n
        beta_n = beta0 + max(stats.sum_x, 0.0)
        rate = float(self.rng.gamma(shape=alpha_n, scale=1.0 / max(beta_n, 1e-12)))
        return float(self.rng.exponential(scale=1.0 / max(rate, 1e-12)))

    def _sample_discrete_poisson(self, stats: WeightCellStats, *, prior_mean: float) -> int:
        mean_x = max(float(prior_mean), 1e-9)
        alpha0 = 1.0
        beta0 = alpha0 / mean_x
        alpha_n = alpha0 + max(stats.sum_x, 0.0)
        beta_n = beta0 + stats.n
        lam = float(self.rng.gamma(shape=alpha_n, scale=1.0 / max(beta_n, 1e-12)))
        return int(self.rng.poisson(max(lam, 0.0)))

    def _sample_discrete_geometric(self, stats: WeightCellStats, *, prior_mean: float) -> int:
        mean_x = max(float(prior_mean), 1e-9)
        prior_mean_p = 1.0 / (1.0 + mean_x)
        concentration = 2.0
        alpha0 = max(prior_mean_p * concentration, 1e-6)
        beta0 = max((1.0 - prior_mean_p) * concentration, 1e-6)
        alpha_n = alpha0 + stats.n
        beta_n = beta0 + max(stats.sum_x, 0.0)
        p = float(self.rng.beta(alpha_n, beta_n))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return int(self.rng.geometric(p) - 1)

    def _sample_discrete_binomial(self, stats: WeightCellStats, *, prior_mean: float) -> int:
        N = max(1, int(self.binomial_max or 1))
        mean_x = min(max(float(prior_mean), 0.0), float(N))
        prior_mean_p = min(max(mean_x / N, 1e-6), 1.0 - 1e-6)
        concentration = 2.0
        alpha0 = prior_mean_p * concentration
        beta0 = (1.0 - prior_mean_p) * concentration
        alpha_n = alpha0 + max(stats.sum_x, 0.0)
        beta_n = beta0 + max(stats.n * N - stats.sum_x, 0.0)
        p = float(self.rng.beta(alpha_n, beta_n))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return int(self.rng.binomial(N, p))

    def _sample_real_normal(self, stats: WeightCellStats, *, prior_mean: float, prior_var: float) -> float:
        mu0 = float(prior_mean)
        kappa0 = 1.0
        nu0 = 3.0
        sigma0_sq = max(float(prior_var), 1e-9)

        if stats.n <= 0:
            scale = math.sqrt(sigma0_sq * (kappa0 + 1.0) / kappa0)
            return float(mu0 + scale * self.rng.standard_t(df=max(nu0, 1.0)))

        xbar = stats.sum_x / stats.n
        ss = max(stats.sum_x2 - stats.n * (xbar ** 2), 0.0)
        kappa_n = kappa0 + stats.n
        mu_n = (kappa0 * mu0 + stats.n * xbar) / kappa_n
        nu_n = nu0 + stats.n
        sigma_n_sq = (nu0 * sigma0_sq + ss + (kappa0 * stats.n * (xbar - mu0) ** 2) / kappa_n) / nu_n
        scale = math.sqrt(max(sigma_n_sq, 1e-12) * (kappa_n + 1.0) / kappa_n)
        return float(mu_n + scale * self.rng.standard_t(df=max(nu_n, 1.0)))

    def sample(
        self,
        ts_value: int,
        r: int,
        s: int,
        *,
        src_type: Optional[int] = None,
        dst_type: Optional[int] = None,
    ) -> float | int:
        if not self.directed and r > s:
            r, s = s, r
        src_type, dst_type = self._canonical_channel(src_type, dst_type)
        stats, resolution = self._resolve_stats(
            int(ts_value),
            int(r),
            int(s),
            src_type=src_type,
            dst_type=dst_type,
        )
        self.resolution_counts[resolution] += 1
        prior_mean, prior_var, prior_label = self._prior_moments(src_type=src_type, dst_type=dst_type)
        self.resolution_counts[f"prior_{prior_label}"] += 1

        if self.rec_type == "real-exponential":
            transformed_value = self._sample_real_exponential(stats, prior_mean=prior_mean)
            return max(0.0, _inverse_transform_weight_value(transformed_value, self.transform))
        if self.rec_type == "real-normal":
            transformed_value = self._sample_real_normal(stats, prior_mean=prior_mean, prior_var=prior_var)
            value = _inverse_transform_weight_value(transformed_value, self.transform)
            if self.transform in {"log", "log1p"}:
                value = max(0.0, value)
            return float(value)
        if self.rec_type == "discrete-poisson":
            return int(max(0, self._sample_discrete_poisson(stats, prior_mean=prior_mean)))
        if self.rec_type == "discrete-geometric":
            return int(max(0, self._sample_discrete_geometric(stats, prior_mean=prior_mean)))
        if self.rec_type == "discrete-binomial":
            return int(max(0, self._sample_discrete_binomial(stats, prior_mean=prior_mean)))
        raise ValueError(f"Unsupported edge-weight rec_type: {self.rec_type}")




def _temporal_generator_mode_name(args: argparse.Namespace) -> str:
    raw_value = str(getattr(args, "temporal_generator_mode", "operational")).strip().lower()
    if raw_value not in {"operational", "independent_sbm"}:
        raise ValueError("Unknown temporal_generator_mode. Use 'operational' or 'independent_sbm'.")
    return raw_value


def _temporal_proposal_mode_name(
    args: argparse.Namespace,
    *,
    temporal_mode: Optional[str] = None,
) -> str:
    if temporal_mode is None:
        temporal_mode = _temporal_generator_mode_name(args)

    raw_value = str(getattr(args, "temporal_proposal_mode", "sbm")).strip().lower()
    aliases = {
        "uniform": "random",
        "uniform_random": "random",
        "random_uniform": "random",
        "randomized": "random",
        "layered_sbm": "sbm",
        "model": "sbm",
        "graph_tool": "sbm",
    }
    raw_value = aliases.get(raw_value, raw_value)
    if raw_value not in {"sbm", "random"}:
        raise ValueError("Unknown temporal_proposal_mode. Use 'sbm' or 'random'.")
    return raw_value


def _temporal_random_proposal_multiplier(args: argparse.Namespace) -> float:
    return max(1.0, float(getattr(args, "temporal_random_proposal_multiplier", 1.0)))


def _active_dyad_capacity(
    active_node_count: int,
    *,
    directed: bool,
) -> int:
    active_node_count = max(int(active_node_count), 0)
    if directed:
        return max(active_node_count * max(active_node_count - 1, 0), 0)
    return max((active_node_count * max(active_node_count - 1, 0)) // 2, 0)


def _random_proposal_edge_budget(
    desired_total: int,
    active_node_count: int,
    *,
    directed: bool,
    multiplier: float,
) -> int:
    capacity = _active_dyad_capacity(active_node_count, directed=directed)
    if capacity <= 0 or int(desired_total) <= 0:
        return 0
    requested = max(1, int(math.ceil(float(multiplier) * float(desired_total))))
    return min(int(capacity), int(requested))


def _sample_uniform_active_edge_candidates(
    active_nodes: set[int],
    *,
    directed: bool,
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    node_array = np.asarray(sorted({int(node_id) for node_id in active_nodes}), dtype=np.int64)
    node_count = int(node_array.size)
    if node_count <= 1 or int(sample_size) <= 0:
        return []

    capacity = _active_dyad_capacity(node_count, directed=directed)
    draw_count = min(int(sample_size), int(capacity))
    if draw_count <= 0:
        return []

    sampled_indices = np.asarray(
        rng.choice(int(capacity), size=int(draw_count), replace=False),
        dtype=np.int64,
    ).reshape(-1)

    if directed:
        u_index = sampled_indices // max(node_count - 1, 1)
        offset = sampled_indices % max(node_count - 1, 1)
        v_index = offset + (offset >= u_index)
    else:
        row_starts = np.zeros(max(node_count - 1, 1), dtype=np.int64)
        if node_count > 2:
            row_starts[1:] = np.cumsum(
                (node_count - 1) - np.arange(0, node_count - 2, dtype=np.int64)
            )
        u_index = np.searchsorted(row_starts, sampled_indices, side="right") - 1
        u_index = np.clip(u_index, 0, node_count - 2)
        offset = sampled_indices - row_starts[u_index]
        v_index = u_index + 1 + offset

    return [
        (int(node_array[int(u_idx)]), int(node_array[int(v_idx)]))
        for u_idx, v_idx in zip(u_index.tolist(), v_index.tolist())
    ]


def _temporal_generator_enabled(args: argparse.Namespace) -> bool:
    return _temporal_generator_mode_name(args) == "operational"


def _temporal_activity_level_name(
    args: argparse.Namespace,
    *,
    has_blocks: bool,
) -> str:
    raw_value = str(getattr(args, "temporal_activity_level", "block")).strip().lower()
    aliases = {
        "nodes": "node",
        "blocks": "block",
        "vertex": "node",
        "vertices": "node",
    }
    raw_value = aliases.get(raw_value, raw_value)
    if raw_value not in {"node", "block"}:
        raise ValueError("Unknown temporal_activity_level. Use 'node' or 'block'.")
    if raw_value == "block" and not has_blocks:
        raise ValueError("temporal_activity_level='block' requires fitted node block assignments.")
    return raw_value


def _temporal_group_mode_name(
    args: argparse.Namespace,
    *,
    has_blocks: bool,
    has_types: bool,
    activity_level: str,
) -> str:
    raw_value = str(getattr(args, "temporal_group_mode", "block_pair")).strip().lower()
    aliases = {
        "blocks": "block_pair",
        "block": "block_pair",
        "types": "type_pair",
        "type": "type_pair",
        "global_only": "global",
    }
    raw_value = aliases.get(raw_value, raw_value)
    if raw_value not in {"block_pair", "type_pair", "global"}:
        raise ValueError("Unknown temporal_group_mode. Use 'block_pair', 'type_pair', or 'global'.")
    if raw_value == "block_pair" and not has_blocks:
        raise ValueError("temporal_group_mode='block_pair' requires fitted node block assignments.")
    if raw_value == "type_pair" and not has_types:
        raise ValueError("temporal_group_mode='type_pair' requires node type attributes.")
    return raw_value


def _canonical_edge_pair(
    u_value: int,
    v_value: int,
    *,
    directed: bool,
) -> tuple[int, int]:
    u_out = int(u_value)
    v_out = int(v_value)
    if not directed and u_out > v_out:
        u_out, v_out = v_out, u_out
    return u_out, v_out


def _canonical_temporal_edge_frame(
    observed_edges: Optional[pd.DataFrame],
    *,
    directed: bool,
) -> pd.DataFrame:
    columns = ["u", "i", "ts"]
    if observed_edges is None or len(observed_edges) == 0:
        return pd.DataFrame(columns=columns)

    missing = [column for column in columns if column not in observed_edges.columns]
    if missing:
        raise ValueError(f"Observed edge table is missing required columns for temporal generation: {missing}")

    frame = observed_edges[columns].copy()
    frame["u"] = pd.to_numeric(frame["u"], errors="raise").astype(np.int64)
    frame["i"] = pd.to_numeric(frame["i"], errors="raise").astype(np.int64)
    frame["ts"] = pd.to_numeric(frame["ts"], errors="raise").astype(np.int64)
    if not directed:
        uv = np.sort(frame[["u", "i"]].to_numpy(dtype=np.int64, copy=False), axis=1)
        frame["u"] = uv[:, 0]
        frame["i"] = uv[:, 1]
    frame = frame.drop_duplicates(["u", "i", "ts"]).sort_values(["ts", "u", "i"]).reset_index(drop=True)
    return frame


def _canonical_group_pair(
    left_value: Any,
    right_value: Any,
    *,
    directed: bool,
) -> tuple[Any, Any]:
    left_out = left_value
    right_out = right_value
    if not directed:
        try:
            should_swap = bool(left_out > right_out)
        except Exception:
            should_swap = str(left_out) > str(right_out)
        if should_swap:
            left_out, right_out = right_out, left_out
    return left_out, right_out


def _edge_group_key(
    edge: tuple[int, int],
    *,
    node_id_to_block: dict[int, int],
    node_id_to_type: Optional[dict[int, int]],
    group_mode: str,
    directed: bool,
) -> tuple[Any, ...]:
    u_value, v_value = edge
    if group_mode == "block_pair":
        block_u = node_id_to_block.get(int(u_value))
        block_v = node_id_to_block.get(int(v_value))
        if block_u is not None and block_v is not None:
            left_value, right_value = _canonical_group_pair(int(block_u), int(block_v), directed=directed)
            return (left_value, right_value)
        if node_id_to_type:
            group_mode = "type_pair"
        else:
            group_mode = "global"

    if group_mode == "type_pair":
        if node_id_to_type:
            type_u = node_id_to_type.get(int(u_value))
            type_v = node_id_to_type.get(int(v_value))
            if type_u is not None and type_v is not None:
                left_value, right_value = _canonical_group_pair(int(type_u), int(type_v), directed=directed)
                return (left_value, right_value)
        group_mode = "global"

    if group_mode == "global":
        return ("__all__",)

    raise ValueError(f"Unsupported temporal grouping mode: {group_mode}")


def _serialise_temporal_group_key(group_key: tuple[Any, ...]) -> str:
    if not isinstance(group_key, tuple):
        return str(group_key)
    return "|".join("" if value is None else str(value) for value in group_key)


def _deserialise_temporal_group_key(text_value: str) -> tuple[Any, ...]:
    parts = str(text_value).split("|")
    if len(parts) == 1 and parts[0] == "__all__":
        return ("__all__",)
    parsed: list[Any] = []
    for part in parts:
        if part == "":
            parsed.append(None)
            continue
        try:
            parsed.append(int(part))
        except ValueError:
            parsed.append(part)
    return tuple(parsed)


def _activity_snapshot_sets(
    observed_edges: pd.DataFrame,
    timeline: list[int],
    *,
    level: str,
    node_id_to_block: dict[int, int],
) -> dict[int, set[int]]:
    snapshot_sets: dict[int, set[int]] = {int(ts_value): set() for ts_value in timeline}
    if observed_edges.empty:
        return snapshot_sets

    for ts_value, snapshot in observed_edges.groupby("ts", sort=True):
        ts_int = int(ts_value)
        if level == "node":
            active_entities = set(snapshot["u"].astype(np.int64).tolist()) | set(snapshot["i"].astype(np.int64).tolist())
        elif level == "block":
            active_entities = set()
            for edge in snapshot[["u", "i"]].itertuples(index=False, name=None):
                block_u = node_id_to_block.get(int(edge[0]))
                block_v = node_id_to_block.get(int(edge[1]))
                if block_u is not None:
                    active_entities.add(int(block_u))
                if block_v is not None:
                    active_entities.add(int(block_v))
        else:
            raise ValueError(f"Unsupported temporal activity level: {level}")
        snapshot_sets[ts_int] = active_entities
    return snapshot_sets



def _fit_temporal_activity_path(
    observed_edges: pd.DataFrame,
    timeline: list[int],
    *,
    level: str,
    entity_universe: Iterable[int],
    node_id_to_block: dict[int, int],
) -> dict[str, Any]:
    entity_list = sorted({int(entity) for entity in entity_universe})
    snapshot_sets = _activity_snapshot_sets(
        observed_edges,
        timeline,
        level=level,
        node_id_to_block=node_id_to_block,
    )
    return {
        "level": level,
        "timeline": [int(ts_value) for ts_value in timeline],
        "entities": entity_list,
        "observed_active_counts": {
            int(ts_value): int(len(snapshot_sets.get(int(ts_value), set())))
            for ts_value in timeline
        },
        "observed_snapshot_sets": {
            int(ts_value): set(int(entity) for entity in snapshot_sets.get(int(ts_value), set()))
            for ts_value in timeline
        },
    }


def _temporal_realized_activity_mode_name(
    args: argparse.Namespace,
    *,
    has_types: bool,
) -> str:
    raw_value = str(getattr(args, "temporal_realized_activity_mode", "total")).strip().lower()
    aliases = {
        "off": "none",
        "disabled": "none",
        "false": "none",
        "total_count": "total",
        "count": "total",
        "counts": "total",
        "type": "type_count",
        "types": "type_count",
    }
    raw_value = aliases.get(raw_value, raw_value)
    if raw_value not in {"none", "total", "type_count"}:
        raise ValueError(
            "Unknown temporal_realized_activity_mode. Use 'none', 'total', or 'type_count'."
        )
    if raw_value == "type_count" and not has_types:
        raise ValueError("temporal_realized_activity_mode='type_count' requires node type attributes.")
    return raw_value


def _count_nodes_by_type(
    node_ids: Iterable[int],
    node_id_to_type: Optional[dict[int, int]],
) -> dict[int, int]:
    if not node_id_to_type:
        return {}
    counts: Counter[int] = Counter()
    for node_id in node_ids:
        type_id = node_id_to_type.get(int(node_id))
        if type_id is None:
            continue
        counts[int(type_id)] += 1
    return {int(type_id): int(count) for type_id, count in counts.items()}


def _fit_temporal_realized_activity_targets(
    observed_edges: pd.DataFrame,
    timeline: list[int],
    *,
    node_id_to_type: Optional[dict[int, int]],
) -> dict[str, Any]:
    by_ts: dict[int, dict[str, Any]] = {}
    snapshot_nodes: dict[int, set[int]] = {int(ts_value): set() for ts_value in timeline}
    if not observed_edges.empty:
        for ts_value, snapshot in observed_edges.groupby("ts", sort=True):
            ts_int = int(ts_value)
            snapshot_nodes[ts_int] = set(snapshot["u"].astype(np.int64).tolist()) | set(
                snapshot["i"].astype(np.int64).tolist()
            )

    for ts_value in timeline:
        ts_int = int(ts_value)
        active_nodes = snapshot_nodes.get(ts_int, set())
        by_ts[ts_int] = {
            "total": int(len(active_nodes)),
            "by_type": _count_nodes_by_type(active_nodes, node_id_to_type),
        }

    type_values = (
        sorted({int(type_id) for type_id in node_id_to_type.values()})
        if node_id_to_type
        else []
    )
    return {
        "by_ts": by_ts,
        "types": type_values,
    }


def _activity_target_deficit(
    *,
    current_total: int,
    current_by_type: dict[int, int] | Counter[int],
    target_total: Optional[int],
    target_by_type: Optional[dict[int, int]],
    mode: str,
    total_weight: float = 1.0,
    type_weight: float = 1.0,
) -> float:
    if mode == "none":
        return 0.0

    deficit = 0.0
    if mode in {"total", "type_count"} and target_total is not None:
        deficit += float(total_weight) * max(0, int(target_total) - int(current_total))
    if mode == "type_count" and target_by_type:
        for type_id, target_count in target_by_type.items():
            deficit += float(type_weight) * max(
                0,
                int(target_count) - int(current_by_type.get(int(type_id), 0)),
            )
    return float(deficit)


def _activity_target_gain(
    *,
    current_total: int,
    current_by_type: dict[int, int] | Counter[int],
    add_total: int,
    add_by_type: Optional[dict[int, int]],
    target_total: Optional[int],
    target_by_type: Optional[dict[int, int]],
    mode: str,
    total_weight: float = 1.0,
    type_weight: float = 1.0,
) -> float:
    if mode == "none":
        return 0.0

    before = _activity_target_deficit(
        current_total=current_total,
        current_by_type=current_by_type,
        target_total=target_total,
        target_by_type=target_by_type,
        mode=mode,
        total_weight=total_weight,
        type_weight=type_weight,
    )
    updated_by_type: Counter[int] = Counter({int(key): int(value) for key, value in dict(current_by_type).items()})
    for type_id, count in (add_by_type or {}).items():
        updated_by_type[int(type_id)] += int(count)
    after = _activity_target_deficit(
        current_total=int(current_total) + int(add_total),
        current_by_type=updated_by_type,
        target_total=target_total,
        target_by_type=target_by_type,
        mode=mode,
        total_weight=total_weight,
        type_weight=type_weight,
    )
    return float(before - after)


def _copy_activity_snapshot_sets(
    snapshot_sets: dict[int, set[int]],
    timeline: list[int],
) -> dict[int, set[int]]:
    return {
        int(ts_value): set(int(entity) for entity in snapshot_sets.get(int(ts_value), set()))
        for ts_value in timeline
    }



def _activity_snapshot_match_metrics(
    sampled_states: dict[int, set[int]],
    target_snapshot_sets: dict[int, set[int]],
    timeline: list[int],
) -> dict[str, Any]:
    exact_daily_matches = 0
    overlap_entity_total = 0
    target_entity_total = 0
    union_entity_total = 0
    jaccard_sum = 0.0

    for ts_value in timeline:
        sampled = set(int(entity) for entity in sampled_states.get(int(ts_value), set()))
        target = set(int(entity) for entity in target_snapshot_sets.get(int(ts_value), set()))
        intersection_size = int(len(sampled & target))
        union_size = int(len(sampled | target))
        exact_daily_matches += int(sampled == target)
        overlap_entity_total += intersection_size
        target_entity_total += int(len(target))
        union_entity_total += union_size
        jaccard_sum += 1.0 if union_size <= 0 else float(intersection_size) / float(union_size)

    timeline_length = int(len(timeline))
    return {
        "timeline_length": timeline_length,
        "exact_daily_matches": int(exact_daily_matches),
        "exact_daily_match_fraction": float(exact_daily_matches / timeline_length) if timeline_length > 0 else 1.0,
        "overlap_entity_total": int(overlap_entity_total),
        "target_entity_total": int(target_entity_total),
        "union_entity_total": int(union_entity_total),
        "mean_daily_jaccard": float(jaccard_sum / timeline_length) if timeline_length > 0 else 1.0,
        "full_match": bool(timeline_length > 0 and exact_daily_matches == timeline_length),
    }



def _stored_temporal_activity_states(
    activity_path: dict[str, Any],
) -> tuple[dict[int, set[int]], dict[str, Any]]:
    timeline = [int(ts_value) for ts_value in activity_path.get("timeline", [])]
    activity_level = str(activity_path.get("level", "node"))
    target_snapshot_sets = {
        int(ts_value): {int(entity) for entity in snapshot}
        for ts_value, snapshot in dict(activity_path.get("observed_snapshot_sets", {})).items()
    }
    selected_states = _copy_activity_snapshot_sets(target_snapshot_sets, timeline)
    selected_metrics = _activity_snapshot_match_metrics(selected_states, target_snapshot_sets, timeline)
    LOGGER.debug(
        "Using stored temporal activity path | level=%s | timeline=%s",
        activity_level,
        len(timeline),
    )
    return selected_states, {
        "mode": "stored",
        "selected_source": "stored_activity_path",
        "has_snapshot_targets": bool(target_snapshot_sets),
        "exact_full_match_found": True,
        "exact_full_match_attempt": 0,
        "selected_metrics": dict(selected_metrics),
    }


def _activity_nodes_for_timestamp(

    ts_value: int,
    *,
    activity_level: str,
    activity_states: dict[int, set[int]],
    block_to_nodes: dict[int, set[int]],
    node_id_to_block: dict[int, int],
) -> tuple[set[int], set[int]]:
    current_entities = set(int(entity) for entity in activity_states.get(int(ts_value), set()))
    if activity_level == "node":
        active_nodes = current_entities
        active_blocks = {
            int(node_id_to_block[int(node_id)])
            for node_id in active_nodes
            if int(node_id) in node_id_to_block
        }
        return active_nodes, active_blocks

    active_blocks = current_entities
    active_nodes: set[int] = set()
    for block_id in active_blocks:
        active_nodes |= set(int(node_id) for node_id in block_to_nodes.get(int(block_id), set()))
    return active_nodes, active_blocks


def _fit_temporal_turnover_targets(
    observed_edges: pd.DataFrame,
    timeline: list[int],
    *,
    node_id_to_block: dict[int, int],
    node_id_to_type: Optional[dict[int, int]],
    group_mode: str,
    directed: bool,
) -> dict[str, Any]:
    by_ts: dict[int, dict[tuple[Any, ...], dict[str, int]]] = {}
    totals_by_ts: dict[int, dict[str, int]] = {}
    seen_edges: set[tuple[int, int]] = set()
    previous_edges: set[tuple[int, int]] = set()

    snapshot_lookup: dict[int, set[tuple[int, int]]] = {int(ts_value): set() for ts_value in timeline}
    for ts_value, snapshot in observed_edges.groupby("ts", sort=True):
        snapshot_edges = {
            _canonical_edge_pair(int(edge.u), int(edge.i), directed=directed)
            for edge in snapshot.itertuples(index=False)
        }
        snapshot_lookup[int(ts_value)] = snapshot_edges

    for ts_value in timeline:
        ts_int = int(ts_value)
        current_edges = snapshot_lookup.get(ts_int, set())
        group_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
            lambda: {"persist": 0, "reactivated": 0, "new": 0, "total": 0}
        )
        totals = {"persist": 0, "reactivated": 0, "new": 0, "total": 0}
        for edge in sorted(current_edges):
            if edge in previous_edges:
                category = "persist"
            elif edge in seen_edges:
                category = "reactivated"
            else:
                category = "new"
            group_key = _edge_group_key(
                edge,
                node_id_to_block=node_id_to_block,
                node_id_to_type=node_id_to_type,
                group_mode=group_mode,
                directed=directed,
            )
            group_counts[group_key][category] += 1
            group_counts[group_key]["total"] += 1
            totals[category] += 1
            totals["total"] += 1
        by_ts[ts_int] = {group_key: dict(counts) for group_key, counts in group_counts.items()}
        totals_by_ts[ts_int] = dict(totals)
        previous_edges = current_edges
        seen_edges |= current_edges

    return {
        "group_mode": group_mode,
        "by_ts": by_ts,
        "totals_by_ts": totals_by_ts,
    }


def _edge_is_activity_allowed(
    edge: tuple[int, int],
    *,
    activity_level: str,
    active_nodes: set[int],
    active_blocks: set[int],
    node_id_to_block: dict[int, int],
) -> bool:
    u_value, v_value = edge
    if activity_level == "node":
        return int(u_value) in active_nodes and int(v_value) in active_nodes
    block_u = node_id_to_block.get(int(u_value))
    block_v = node_id_to_block.get(int(v_value))
    return (
        block_u is not None
        and block_v is not None
        and int(block_u) in active_blocks
        and int(block_v) in active_blocks
    )


def _build_turnover_candidate_pools(
    *,
    proposal_counts: Counter[tuple[int, int]],
    previous_edges: set[tuple[int, int]],
    seen_edges: set[tuple[int, int]],
    activity_level: str,
    active_nodes: set[int],
    active_blocks: set[int],
    node_id_to_block: dict[int, int],
    node_id_to_type: Optional[dict[int, int]],
    group_mode: str,
    directed: bool,
) -> dict[tuple[Any, ...], dict[str, list[tuple[tuple[int, int], float]]]]:
    pools: dict[tuple[Any, ...], dict[str, list[tuple[tuple[int, int], float]]]] = defaultdict(
        lambda: {"persist": [], "reactivated": [], "new": []}
    )

    def _append(edge: tuple[int, int], category: str) -> None:
        if not _edge_is_activity_allowed(
            edge,
            activity_level=activity_level,
            active_nodes=active_nodes,
            active_blocks=active_blocks,
            node_id_to_block=node_id_to_block,
        ):
            return
        group_key = _edge_group_key(
            edge,
            node_id_to_block=node_id_to_block,
            node_id_to_type=node_id_to_type,
            group_mode=group_mode,
            directed=directed,
        )
        pools[group_key][category].append((edge, float(proposal_counts.get(edge, 0.0))))

    for edge in sorted(previous_edges):
        _append(edge, "persist")
    for edge in sorted(seen_edges - previous_edges):
        _append(edge, "reactivated")
    for edge in sorted(proposal_counts):
        if edge in seen_edges:
            continue
        _append(edge, "new")

    for group_key in list(pools):
        for category in ("persist", "reactivated", "new"):
            pools[group_key][category].sort(
                key=lambda item: (-float(item[1]), int(item[0][0]), int(item[0][1]))
            )
    return pools


def _turnover_pool_shortfalls(
    pools: dict[tuple[Any, ...], dict[str, list[tuple[tuple[int, int], float]]]],
    target_by_group: dict[tuple[Any, ...], dict[str, int]],
) -> dict[str, int]:
    shortfalls = {"persist": 0, "reactivated": 0, "new": 0}
    for group_key, target in target_by_group.items():
        group_pools = pools.get(group_key, {})
        for category in shortfalls:
            shortfalls[category] += max(
                0,
                int(target.get(category, 0)) - int(len(group_pools.get(category, []))),
            )
    return shortfalls


def _resolve_feasible_turnover_counts(
    desired: dict[str, int],
    capacities: dict[str, int],
) -> dict[str, int]:
    categories = ("persist", "reactivated", "new")
    desired_total = int(desired.get("total", sum(int(desired.get(category, 0)) for category in categories)))
    capacity_total = int(sum(int(capacities.get(category, 0)) for category in categories))
    target_total = min(desired_total, capacity_total)

    actual = {
        category: min(int(desired.get(category, 0)), int(capacities.get(category, 0)))
        for category in categories
    }
    assigned_total = int(sum(actual.values()))
    if assigned_total > target_total:
        for category in sorted(categories, key=lambda value: actual[value], reverse=True):
            while assigned_total > target_total and actual[category] > 0:
                actual[category] -= 1
                assigned_total -= 1

    needed = target_total - assigned_total
    if needed > 0:
        while needed > 0:
            spares = {
                category: int(capacities.get(category, 0)) - int(actual.get(category, 0))
                for category in categories
            }
            available = [(spare, category) for category, spare in spares.items() if spare > 0]
            if not available:
                break
            _, category = max(available, key=lambda item: (item[0], -categories.index(item[1])))
            actual[category] += 1
            needed -= 1

    actual["total"] = int(sum(actual.values()))
    return actual



def _edge_activity_delta(
    edge: tuple[int, int],
    *,
    active_nodes: set[int],
    node_id_to_type: Optional[dict[int, int]],
) -> tuple[list[int], dict[int, int]]:
    new_nodes: list[int] = []
    add_by_type: Counter[int] = Counter()
    seen_local: set[int] = set()
    for node_id in edge:
        node_int = int(node_id)
        if node_int in seen_local or node_int in active_nodes:
            continue
        seen_local.add(node_int)
        new_nodes.append(node_int)
        if node_id_to_type:
            type_id = node_id_to_type.get(node_int)
            if type_id is not None:
                add_by_type[int(type_id)] += 1
    return new_nodes, {int(type_id): int(count) for type_id, count in add_by_type.items()}


def _edge_activity_selection_key(
    edge: tuple[int, int],
    score: float,
    *,
    selected_active_nodes: set[int],
    selected_type_counts: Counter[int],
    target_total: Optional[int],
    target_by_type: Optional[dict[int, int]],
    activity_mode: str,
    activity_weight: float,
    node_id_to_type: Optional[dict[int, int]],
    category_bonus: float = 0.0,
) -> tuple[tuple[float, float, float, float, float, float], list[int], dict[int, int], float]:
    new_nodes, add_by_type = _edge_activity_delta(
        edge,
        active_nodes=selected_active_nodes,
        node_id_to_type=node_id_to_type,
    )
    gain = _activity_target_gain(
        current_total=len(selected_active_nodes),
        current_by_type=selected_type_counts,
        add_total=len(new_nodes),
        add_by_type=add_by_type,
        target_total=target_total,
        target_by_type=target_by_type,
        mode=activity_mode,
        total_weight=1.0,
        type_weight=1.0,
    )
    key = (
        float(gain),
        float(category_bonus),
        float(score) + float(activity_weight) * float(gain),
        float(len(new_nodes)),
        float(score),
        -float(edge[0]) - 1e-6 * float(edge[1]),
    )
    return key, new_nodes, add_by_type, float(gain)


def _select_edges_from_candidate_pool(
    candidates: list[tuple[tuple[int, int], float]],
    *,
    keep_count: int,
    selected_set: set[tuple[int, int]],
    selected_active_nodes: set[int],
    selected_type_counts: Counter[int],
    target_total: Optional[int],
    target_by_type: Optional[dict[int, int]],
    activity_mode: str,
    activity_weight: float,
    node_id_to_type: Optional[dict[int, int]],
    category_bonus: float = 0.0,
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    available = [(edge, float(score)) for edge, score in candidates if edge not in selected_set]
    while len(chosen) < int(keep_count) and available:
        best_index: Optional[int] = None
        best_key: Optional[tuple[float, float, float, float, float, float]] = None
        best_nodes: list[int] = []
        best_type_counts: dict[int, int] = {}
        for index, (edge, score) in enumerate(available):
            key, new_nodes, add_by_type, _gain = _edge_activity_selection_key(
                edge,
                float(score),
                selected_active_nodes=selected_active_nodes,
                selected_type_counts=selected_type_counts,
                target_total=target_total,
                target_by_type=target_by_type,
                activity_mode=activity_mode,
                activity_weight=activity_weight,
                node_id_to_type=node_id_to_type,
                category_bonus=category_bonus,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = int(index)
                best_nodes = new_nodes
                best_type_counts = add_by_type
        if best_index is None:
            break
        edge, _score = available.pop(best_index)
        if edge in selected_set:
            continue
        chosen.append(edge)
        selected_set.add(edge)
        for node_id in best_nodes:
            selected_active_nodes.add(int(node_id))
        for type_id, count in best_type_counts.items():
            selected_type_counts[int(type_id)] += int(count)
    return chosen


def _select_edges_from_turnover_pools(
    pools: dict[tuple[Any, ...], dict[str, list[tuple[tuple[int, int], float]]]],
    target_by_group: dict[tuple[Any, ...], dict[str, int]],
    *,
    desired_total: int,
    activity_target: Optional[dict[str, Any]] = None,
    activity_mode: str = "none",
    activity_weight: float = 0.0,
    node_id_to_type: Optional[dict[int, int]] = None,
) -> tuple[list[tuple[int, int]], dict[tuple[Any, ...], dict[str, int]], dict[tuple[Any, ...], dict[str, int]]]:
    categories = ("persist", "reactivated", "new")
    selected_edges: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    selected_active_nodes: set[int] = set()
    selected_type_counts: Counter[int] = Counter()
    achieved_by_group: dict[tuple[Any, ...], dict[str, int]] = {}
    capacities_by_group: dict[tuple[Any, ...], dict[str, int]] = {}

    target_total = None
    target_by_type = None
    if activity_target:
        if activity_target.get("total") is not None:
            target_total = int(activity_target.get("total", 0))
        target_by_type = {
            int(type_id): int(count)
            for type_id, count in dict(activity_target.get("by_type", {})).items()
        }

    all_groups = sorted(
        set(target_by_group) | set(pools),
        key=lambda group_key: _serialise_temporal_group_key(group_key),
    )

    category_order = {"persist": 0, "reactivated": 1, "new": 2}
    selection_plan: list[tuple[float, int, str, tuple[Any, ...], str, int]] = []

    for group_key in all_groups:
        group_pools = pools.get(group_key, {"persist": [], "reactivated": [], "new": []})
        capacities = {category: int(len(group_pools.get(category, []))) for category in categories}
        capacities_by_group[group_key] = dict(capacities)
        target = target_by_group.get(group_key, {"persist": 0, "reactivated": 0, "new": 0, "total": 0})
        achieved = _resolve_feasible_turnover_counts(target, capacities)
        achieved_by_group[group_key] = dict(achieved)
        for category in categories:
            keep_count = int(achieved.get(category, 0))
            if keep_count <= 0:
                continue
            pool_size = int(len(group_pools.get(category, [])))
            scarcity = float(pool_size) / float(max(keep_count, 1))
            selection_plan.append(
                (
                    scarcity,
                    int(category_order[category]),
                    _serialise_temporal_group_key(group_key),
                    group_key,
                    category,
                    keep_count,
                )
            )

    selection_plan.sort()

    for _scarcity, _category_order, _group_text, group_key, category, keep_count in selection_plan:
        group_pools = pools.get(group_key, {"persist": [], "reactivated": [], "new": []})
        chosen = _select_edges_from_candidate_pool(
            group_pools.get(category, []),
            keep_count=int(keep_count),
            selected_set=selected_set,
            selected_active_nodes=selected_active_nodes,
            selected_type_counts=selected_type_counts,
            target_total=target_total,
            target_by_type=target_by_type,
            activity_mode=activity_mode,
            activity_weight=float(activity_weight),
            node_id_to_type=node_id_to_type,
            category_bonus=0.0,
        )
        selected_edges.extend(chosen)

    if len(selected_edges) < int(desired_total):
        category_bonus = {"new": 2.0, "reactivated": 1.0, "persist": 0.0}
        remaining: list[tuple[tuple[int, int], float, tuple[Any, ...], str]] = []
        for group_key in all_groups:
            group_pools = pools.get(group_key, {"persist": [], "reactivated": [], "new": []})
            achieved = achieved_by_group.get(group_key, {"persist": 0, "reactivated": 0, "new": 0, "total": 0})
            for category in categories:
                start_index = int(achieved.get(category, 0))
                for edge, score in group_pools.get(category, [])[start_index:]:
                    if edge in selected_set:
                        continue
                    remaining.append((edge, float(score), group_key, category))

        while len(selected_edges) < int(desired_total) and remaining:
            best_index: Optional[int] = None
            best_key: Optional[tuple[float, float, float, float, float, float]] = None
            best_nodes: list[int] = []
            best_type_counts: dict[int, int] = {}
            best_group_key: Optional[tuple[Any, ...]] = None
            best_category: Optional[str] = None
            for index, (edge, score, group_key, category) in enumerate(remaining):
                key, new_nodes, add_by_type, _gain = _edge_activity_selection_key(
                    edge,
                    float(score),
                    selected_active_nodes=selected_active_nodes,
                    selected_type_counts=selected_type_counts,
                    target_total=target_total,
                    target_by_type=target_by_type,
                    activity_mode=activity_mode,
                    activity_weight=float(activity_weight),
                    node_id_to_type=node_id_to_type,
                    category_bonus=float(category_bonus[category]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = int(index)
                    best_nodes = new_nodes
                    best_type_counts = add_by_type
                    best_group_key = group_key
                    best_category = category
            if best_index is None or best_group_key is None or best_category is None:
                break
            edge, _score, group_key, category = remaining.pop(best_index)
            if edge in selected_set:
                continue
            selected_edges.append(edge)
            selected_set.add(edge)
            for node_id in best_nodes:
                selected_active_nodes.add(int(node_id))
            for type_id, count in best_type_counts.items():
                selected_type_counts[int(type_id)] += int(count)
            group_counts = achieved_by_group.setdefault(
                group_key,
                {"persist": 0, "reactivated": 0, "new": 0, "total": 0},
            )
            group_counts[category] = int(group_counts.get(category, 0)) + 1
            group_counts["total"] = int(group_counts.get("total", 0)) + 1

    for group_key, counts in achieved_by_group.items():
        counts["total"] = int(sum(int(counts.get(category, 0)) for category in categories))
    return selected_edges, achieved_by_group, capacities_by_group


def _aggregate_turnover_totals(
    counts_by_group: dict[tuple[Any, ...], dict[str, int]],
) -> dict[str, int]:
    totals = {"persist": 0, "reactivated": 0, "new": 0, "total": 0}
    for counts in counts_by_group.values():
        for category in ("persist", "reactivated", "new"):
            totals[category] += int(counts.get(category, 0))
            totals["total"] += int(counts.get(category, 0))
    return totals


def _build_snapshot_gt_graph(
    gt: Any,
    edges: list[tuple[int, int]],
    *,
    directed: bool,
    weight_col: Optional[str] = None,
    weight_values: Optional[list[float]] = None,
) -> Any:
    graph = gt.Graph(directed=directed)
    node_ids = sorted({int(node_id) for edge in edges for node_id in edge})
    graph.add_vertex(len(node_ids))
    node_id_prop = graph.new_vp("int64_t")
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    for node_id, index in node_index.items():
        node_id_prop[graph.vertex(index)] = int(node_id)
    graph.vp["node_id"] = node_id_prop

    edge_weight_prop = graph.new_ep("double") if weight_col is not None and weight_values is not None else None
    for edge_index, (u_value, v_value) in enumerate(edges):
        edge = graph.add_edge(graph.vertex(node_index[int(u_value)]), graph.vertex(node_index[int(v_value)]))
        if edge_weight_prop is not None and weight_values is not None:
            edge_weight_prop[edge] = float(weight_values[edge_index])
    if edge_weight_prop is not None and weight_col is not None:
        graph.ep[str(weight_col)] = edge_weight_prop
    return graph


def _proposal_round_settings(args: argparse.Namespace) -> tuple[int, int]:
    min_rounds = max(1, int(getattr(args, "temporal_proposal_rounds", 3)))
    max_rounds = max(min_rounds, int(getattr(args, "temporal_proposal_rounds_max", 12)))
    return min_rounds, max_rounds


def _build_generation_node_maps(
    base: Any,
    blocks: np.ndarray,
    node_id_prop: Any,
    type_prop: Any,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, set[int]]]:
    node_id_to_base: dict[int, int] = {}
    node_id_to_block: dict[int, int] = {}
    node_id_to_type: dict[int, int] = {}
    block_to_nodes: dict[int, set[int]] = defaultdict(set)

    metadata_prop = base.g.vp["is_metadata_tag"] if "is_metadata_tag" in base.g.vp else None
    for index in range(int(base.g.num_vertices())):
        vertex = base.g.vertex(index)
        if metadata_prop is not None and bool(metadata_prop[vertex]):
            continue
        node_id = int(node_id_prop[vertex])
        if node_id < 0:
            continue
        node_id_to_base[node_id] = int(index)
        node_id_to_block[node_id] = int(blocks[index])
        block_to_nodes[int(blocks[index])].add(node_id)
        if type_prop is not None:
            node_id_to_type[node_id] = int(type_prop[vertex])
    return node_id_to_base, node_id_to_block, node_id_to_type, block_to_nodes



def _serialise_temporal_activity_path_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": str(payload.get("level", "node")),
        "timeline": [int(ts_value) for ts_value in payload.get("timeline", [])],
        "entities": [int(entity) for entity in payload.get("entities", [])],
        "observed_active_counts": {
            str(int(ts_value)): int(count)
            for ts_value, count in dict(payload.get("observed_active_counts", {})).items()
        },
        "observed_snapshot_sets": {
            str(int(ts_value)): sorted(int(entity) for entity in snapshot)
            for ts_value, snapshot in dict(payload.get("observed_snapshot_sets", {})).items()
        },
    }


def _deserialise_temporal_activity_path_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": str(payload.get("level", "node")),
        "timeline": [int(ts_value) for ts_value in payload.get("timeline", [])],
        "entities": [int(entity) for entity in payload.get("entities", [])],
        "observed_active_counts": {
            int(ts_value): int(count)
            for ts_value, count in dict(payload.get("observed_active_counts", {})).items()
        },
        "observed_snapshot_sets": {
            int(ts_value): {int(entity) for entity in snapshot}
            for ts_value, snapshot in dict(payload.get("observed_snapshot_sets", {})).items()
        },
    }


def _serialise_temporal_realized_activity_targets(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "by_ts": {
            str(int(ts_value)): {
                "total": int(targets.get("total", 0)),
                "by_type": {
                    str(int(type_id)): int(count)
                    for type_id, count in dict(targets.get("by_type", {})).items()
                },
            }
            for ts_value, targets in dict(payload.get("by_ts", {})).items()
        },
        "types": [int(type_id) for type_id in payload.get("types", [])],
    }


def _deserialise_temporal_realized_activity_targets(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "by_ts": {
            int(ts_value): {
                "total": int(targets.get("total", 0)),
                "by_type": {
                    int(type_id): int(count)
                    for type_id, count in dict(targets.get("by_type", {})).items()
                },
            }
            for ts_value, targets in dict(payload.get("by_ts", {})).items()
        },
        "types": [int(type_id) for type_id in payload.get("types", [])],
    }


def _serialise_temporal_turnover_target_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_mode": str(payload.get("group_mode", "global")),
        "by_ts": {
            str(int(ts_value)): {
                _serialise_temporal_group_key(group_key): {
                    "persist": int(counts.get("persist", 0)),
                    "reactivated": int(counts.get("reactivated", 0)),
                    "new": int(counts.get("new", 0)),
                    "total": int(counts.get("total", 0)),
                }
                for group_key, counts in dict(group_targets).items()
            }
            for ts_value, group_targets in dict(payload.get("by_ts", {})).items()
        },
        "totals_by_ts": {
            str(int(ts_value)): {
                "persist": int(counts.get("persist", 0)),
                "reactivated": int(counts.get("reactivated", 0)),
                "new": int(counts.get("new", 0)),
                "total": int(counts.get("total", 0)),
            }
            for ts_value, counts in dict(payload.get("totals_by_ts", {})).items()
        },
    }


def _deserialise_temporal_turnover_target_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_mode": str(payload.get("group_mode", "global")),
        "by_ts": {
            int(ts_value): {
                _deserialise_temporal_group_key(group_key): {
                    "persist": int(counts.get("persist", 0)),
                    "reactivated": int(counts.get("reactivated", 0)),
                    "new": int(counts.get("new", 0)),
                    "total": int(counts.get("total", 0)),
                }
                for group_key, counts in dict(group_targets).items()
            }
            for ts_value, group_targets in dict(payload.get("by_ts", {})).items()
        },
        "totals_by_ts": {
            int(ts_value): {
                "persist": int(counts.get("persist", 0)),
                "reactivated": int(counts.get("reactivated", 0)),
                "new": int(counts.get("new", 0)),
                "total": int(counts.get("total", 0)),
            }
            for ts_value, counts in dict(payload.get("totals_by_ts", {})).items()
        },
    }


def _fit_temporal_generator_model(
    prepared: PreparedData,
    nested_state: Any,
    *,
    directed: bool,
) -> dict[str, Any]:
    base = _base_state(nested_state)
    node_id_prop = base.g.vp["node_id"] if "node_id" in base.g.vp else None
    if node_id_prop is None:
        raise RuntimeError("Fitted graph is missing the vertex property 'node_id' required for temporal model export.")

    blocks = _node_blocks_from_state(nested_state)
    type_prop = base.g.vp["type"] if "type" in base.g.vp else None
    _node_id_to_base, node_id_to_block, node_id_to_type, _block_to_nodes = _build_generation_node_maps(
        base,
        blocks,
        node_id_prop,
        type_prop,
    )

    timeline = [int(ts_value) for ts_value, _ in sorted(prepared.layer_map.items(), key=lambda item: item[1])]
    observed_frame = _canonical_temporal_edge_frame(prepared.original_edges, directed=directed)

    node_snapshot_sets = _activity_snapshot_sets(
        observed_frame,
        timeline,
        level="node",
        node_id_to_block=node_id_to_block,
    )
    block_snapshot_sets = _activity_snapshot_sets(
        observed_frame,
        timeline,
        level="block",
        node_id_to_block=node_id_to_block,
    )

    activity_paths = {
        "node": _serialise_temporal_activity_path_payload(
            _fit_temporal_activity_path(
                observed_frame,
                timeline,
                level="node",
                entity_universe=sorted(node_id_to_block.keys()),
                node_id_to_block=node_id_to_block,
            )
        ),
        "block": _serialise_temporal_activity_path_payload(
            _fit_temporal_activity_path(
                observed_frame,
                timeline,
                level="block",
                entity_universe=sorted(set(node_id_to_block.values())),
                node_id_to_block=node_id_to_block,
            )
        ),
    }

    realized_activity_targets = _serialise_temporal_realized_activity_targets(
        _fit_temporal_realized_activity_targets(
            observed_frame,
            timeline,
            node_id_to_type=node_id_to_type or None,
        )
    )

    turnover_targets: dict[str, Any] = {
        "global": _serialise_temporal_turnover_target_payload(
            _fit_temporal_turnover_targets(
                observed_frame,
                timeline,
                node_id_to_block=node_id_to_block,
                node_id_to_type=node_id_to_type or None,
                group_mode="global",
                directed=directed,
            )
        ),
        "block_pair": _serialise_temporal_turnover_target_payload(
            _fit_temporal_turnover_targets(
                observed_frame,
                timeline,
                node_id_to_block=node_id_to_block,
                node_id_to_type=node_id_to_type or None,
                group_mode="block_pair",
                directed=directed,
            )
        ),
    }
    if node_id_to_type:
        turnover_targets["type_pair"] = _serialise_temporal_turnover_target_payload(
            _fit_temporal_turnover_targets(
                observed_frame,
                timeline,
                node_id_to_block=node_id_to_block,
                node_id_to_type=node_id_to_type or None,
                group_mode="type_pair",
                directed=directed,
            )
        )

    return {
        "format": TEMPORAL_GENERATOR_PAYLOAD_FORMAT,
        "mode": "operational",
        "directed": bool(directed),
        "timeline": [int(ts_value) for ts_value in timeline],
        "partition_source": "fitted_state",
        "supports_posterior_refresh": False,
        "activity_paths": activity_paths,
        "realized_activity_targets": realized_activity_targets,
        "turnover_targets": turnover_targets,
        "summary": {
            "timeline_length": int(len(timeline)),
            "node_activity_entity_count": int(len(activity_paths["node"].get("entities", []))),
            "block_activity_entity_count": int(len(activity_paths["block"].get("entities", []))),
            "available_turnover_group_modes": sorted(turnover_targets.keys()),
            "has_node_types": bool(node_id_to_type),
            "fitted_block_count": int(len(set(node_id_to_block.values()))),
            "fitted_node_count": int(len(node_id_to_block)),
            "stores_activity_paths": True,
            "node_snapshot_target_days": int(len(node_snapshot_sets)),
            "block_snapshot_target_days": int(len(block_snapshot_sets)),
            "full_generative_topology_supported": True,
            "posterior_refresh_requires_observed_edges": True,
        },
    }


def _prepare_temporal_targets_for_generation(
    *,
    timeline: list[int],
    observed_edges: Optional[pd.DataFrame],
    temporal_generator_model: Optional[dict[str, Any]],
    directed: bool,
    activity_level: str,
    group_mode: str,
    node_id_to_block: dict[int, int],
    node_id_to_type: Optional[dict[int, int]],
    posterior_partition_sweeps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if temporal_generator_model is not None and int(posterior_partition_sweeps) <= 0:
        payload_format = str(temporal_generator_model.get("format"))
        if payload_format != TEMPORAL_GENERATOR_PAYLOAD_FORMAT:
            raise ValueError(
                "Unsupported temporal generator payload format: "
                f"{temporal_generator_model.get('format')!r}"
            )
        stored_timeline = [int(ts_value) for ts_value in temporal_generator_model.get("timeline", [])]
        if stored_timeline != [int(ts_value) for ts_value in timeline]:
            raise ValueError(
                "The saved temporal generator timeline does not match the fitted layer map. "
                "Regenerate the run artifacts or use the matching model bundle."
            )
        activity_payload = dict((temporal_generator_model.get("activity_paths") or {}).get(activity_level, {}))
        if not activity_payload:
            raise ValueError(
                f"The saved temporal generator model does not include activity paths for level {activity_level!r}."
            )
        turnover_payload = dict((temporal_generator_model.get("turnover_targets") or {}).get(group_mode, {}))
        if not turnover_payload:
            raise ValueError(
                f"The saved temporal generator model does not include turnover targets for mode {group_mode!r}."
            )
        return (
            _deserialise_temporal_realized_activity_targets(
                dict(temporal_generator_model.get("realized_activity_targets", {}))
            ),
            _deserialise_temporal_activity_path_payload(activity_payload),
            _deserialise_temporal_turnover_target_payload(turnover_payload),
            "stored_operational_temporal_model",
        )

    if observed_edges is None:
        if int(posterior_partition_sweeps) > 0:
            raise ValueError(
                "temporal_generator_mode='operational' with posterior_partition_sweeps > 0 "
                "requires the filtered observed edge panel, because turnover quotas must be "
                "recomputed on the refreshed partition."
            )
        raise ValueError(
            "temporal_generator_mode='operational' requires observed_edges or a saved temporal "
            "generator model in the fitted run artifacts."
        )

    observed_frame = _canonical_temporal_edge_frame(observed_edges, directed=directed)
    entity_universe = (
        sorted(node_id_to_block.values()) if activity_level == "block" else sorted(node_id_to_block.keys())
    )
    return (
        _fit_temporal_realized_activity_targets(
            observed_frame,
            timeline,
            node_id_to_type=node_id_to_type or None,
        ),
        _fit_temporal_activity_path(
            observed_frame,
            timeline,
            level=activity_level,
            entity_universe=entity_universe,
            node_id_to_block=node_id_to_block,
        ),
        _fit_temporal_turnover_targets(
            observed_frame,
            timeline,
            node_id_to_block=node_id_to_block,
            node_id_to_type=node_id_to_type or None,
            group_mode=group_mode,
            directed=directed,
        ),
        "observed_refit_posterior_refresh" if int(posterior_partition_sweeps) > 0 else "observed_refit",
    )


def _serialise_temporal_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialised: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        if "target_by_group" in out:
            out["target_by_group"] = {
                _serialise_temporal_group_key(group_key): _json_ready(group_counts)
                for group_key, group_counts in out["target_by_group"].items()
            }
        if "achieved_by_group" in out:
            out["achieved_by_group"] = {
                _serialise_temporal_group_key(group_key): _json_ready(group_counts)
                for group_key, group_counts in out["achieved_by_group"].items()
            }
        if "capacity_by_group" in out:
            out["capacity_by_group"] = {
                _serialise_temporal_group_key(group_key): _json_ready(group_counts)
                for group_key, group_counts in out["capacity_by_group"].items()
            }
        serialised.append(_json_ready(out))
    return serialised


def _sample_synthetic_panel_operational_turnover(
    graph: Any,
    nested_state: Any,
    layer_map: dict[int, int],
    output_dir: Path,
    directed: bool,
    seed: int,
    args: argparse.Namespace,
    observed_edges: Optional[pd.DataFrame] = None,
    weight_model: Optional[dict] = None,
    weight_generator_model: Optional[dict] = None,
    temporal_generator_model: Optional[dict[str, Any]] = None,
) -> dict:
    gt = _require_graph_tool()
    gt.seed_rng(int(seed))
    weight_partition_policy = str(getattr(args, "weight_parametric_partition_policy", "fixed")).strip().lower()
    pure_generative_weights = _pure_generative_weight_mode(args)
    if pure_generative_weights and weight_partition_policy == "refit_on_refresh":
        raise ValueError(
            "weight_pure_generative=True is incompatible with weight_parametric_partition_policy='refit_on_refresh'."
        )

    LOGGER.debug(
        "Sampling operational temporal panel | output_dir=%s | directed=%s | seed=%s | layer_count=%s | temporal_mode=%s | proposal_mode=%s",
        output_dir,
        directed,
        seed,
        len(layer_map),
        _temporal_generator_mode_name(args),
        _temporal_proposal_mode_name(args, temporal_mode=_temporal_generator_mode_name(args)),
    )

    sampled_state = _posterior_partition_state(nested_state, seed=seed, args=args)
    base = _base_state(sampled_state)
    lid_to_state = map_graph_lid_to_state_lid(sampled_state)
    LOGGER.debug("Sampled state ready for operational temporal generation | %s", _state_summary_text(sampled_state))

    node_id_prop = base.g.vp["node_id"] if "node_id" in base.g.vp else None
    if node_id_prop is None:
        raise RuntimeError("Fitted graph is missing the vertex property 'node_id'.")

    blocks = _node_blocks_from_state(sampled_state)
    type_prop = base.g.vp["type"] if "type" in base.g.vp else None
    node_id_to_base, node_id_to_block, node_id_to_type, block_to_nodes = _build_generation_node_maps(
        base,
        blocks,
        node_id_prop,
        type_prop,
    )

    active_weight_generator = weight_generator_model
    weight_generation_mode = "none"
    if weight_model is not None and observed_edges is not None and weight_partition_policy == "refit_on_refresh":
        active_weight_generator = _fit_parametric_weight_generator_model(
            observed_edges=observed_edges,
            base=base,
            weight_model=weight_model,
            directed=directed,
            args=args,
            blocks=blocks,
        )
        weight_generation_mode = "parametric_refit"
    elif active_weight_generator is not None:
        weight_generation_mode = "parametric_fixed"

    if pure_generative_weights and weight_model is not None and active_weight_generator is None:
        raise ValueError(
            "weight_pure_generative=True requires a fitted parametric weight generator. "
            "Generation cannot proceed with empirical weight backoff."
        )

    weight_blocks = blocks
    if active_weight_generator is not None and weight_partition_policy != "refit_on_refresh":
        if pure_generative_weights:
            stored_weight_blocks = _stored_weight_reference_blocks(
                base=base,
                blocks=blocks,
                node_id_prop=node_id_prop,
                weight_generator_model=active_weight_generator,
            )
            if stored_weight_blocks is None:
                raise ValueError(
                    "weight_pure_generative=True requires node_blocks in the saved parametric weight generator."
                )
            weight_blocks = stored_weight_blocks
            LOGGER.debug(
                "Using stored node blocks from the saved parametric weight generator | unique_blocks=%s",
                int(np.unique(weight_blocks).size),
            )
        else:
            weight_blocks = _aligned_blocks_for_weight_generation(
                gt=gt,
                base=base,
                blocks=blocks,
                node_id_prop=node_id_prop,
                weight_generator_model=active_weight_generator,
            )

    partition_records = []
    include_weight_blocks = active_weight_generator is not None and not np.array_equal(weight_blocks, blocks)
    active_prop = base.g.vp["active_in_window"] if "active_in_window" in base.g.vp else None
    metadata_prop = base.g.vp["is_metadata_tag"] if "is_metadata_tag" in base.g.vp else None
    for index in range(int(base.g.num_vertices())):
        vertex = base.g.vertex(index)
        if metadata_prop is not None and bool(metadata_prop[vertex]):
            continue
        node_id = int(node_id_prop[vertex])
        if node_id < 0:
            continue
        record = {
            "node_id": node_id,
            "block_id": int(blocks[index]),
        }
        if active_prop is not None:
            record["active_in_window"] = int(active_prop[vertex])
        if include_weight_blocks:
            record["weight_block_id"] = int(weight_blocks[index])
        partition_records.append(record)

    partition_frame = pd.DataFrame(partition_records).sort_values(
        ["node_id", "block_id"] + (["weight_block_id"] if include_weight_blocks else [])
    ).reset_index(drop=True)
    partition_path = Path(output_dir) / "sample_node_partition.csv"
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_frame.to_csv(partition_path, index=False)

    weight_col = None
    weight_sampler = None
    if active_weight_generator is not None:
        weight_col = str(active_weight_generator["output_column"])
        weight_sampler = ParametricWeightSampler(
            weight_generator_model=active_weight_generator,
            directed=directed,
            rng=np.random.default_rng(int(seed)),
        )
    elif weight_model and observed_edges is not None:
        if pure_generative_weights:
            raise ValueError(
                "weight_pure_generative=True forbids empirical weight backoff. "
                "Fit and load a parametric weight generator instead."
            )
        weight_col = str(weight_model.get("output_column") or weight_model.get("input_column"))
        weight_sampler = EdgeWeightSampler(
            observed_edges=observed_edges,
            node_id_to_base=node_id_to_base,
            blocks=blocks,
            weight_model=weight_model,
            directed=directed,
            min_cell_count=max(1, int(getattr(args, "weight_min_cell_count", 3))),
            rng=np.random.default_rng(int(seed)),
            node_id_to_type=node_id_to_type or None,
        )
        weight_generation_mode = "empirical_backoff"

    output_dir = Path(output_dir)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    timeline = [int(ts_value) for ts_value, _ in sorted(layer_map.items(), key=lambda item: item[1])]
    activity_level = _temporal_activity_level_name(args, has_blocks=bool(node_id_to_block))
    group_mode = _temporal_group_mode_name(
        args,
        has_blocks=bool(node_id_to_block),
        has_types=bool(node_id_to_type),
        activity_level=activity_level,
    )
    realized_activity_mode = _temporal_realized_activity_mode_name(
        args,
        has_types=bool(node_id_to_type),
    )
    realized_activity_weight = max(
        0.0,
        float(getattr(args, "temporal_realized_activity_weight", 2.0)),
    )
    posterior_partition_sweeps = max(0, int(getattr(args, "posterior_partition_sweeps", 0)))
    observed_realized_activity_targets, activity_path, turnover_targets, temporal_target_source = _prepare_temporal_targets_for_generation(
        timeline=timeline,
        observed_edges=observed_edges,
        temporal_generator_model=temporal_generator_model,
        directed=directed,
        activity_level=activity_level,
        group_mode=group_mode,
        node_id_to_block=node_id_to_block,
        node_id_to_type=node_id_to_type or None,
        posterior_partition_sweeps=posterior_partition_sweeps,
    )
    activity_states, activity_sampling_summary = _stored_temporal_activity_states(activity_path)
    activity_snapshot_targets = {
        int(ts_value): set(int(entity) for entity in snapshot)
        for ts_value, snapshot in dict(activity_path.get("observed_snapshot_sets", {})).items()
    }

    sample_records: list[dict[str, Any]] = []
    record_columns = ["u", "i", "ts", "snapshot"] + ([weight_col] if weight_col else [])
    rewire_summaries: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    sample_kwargs = _sample_kwargs(args)
    proposal_rounds_min, proposal_rounds_max = _proposal_round_settings(args)
    proposal_mode = _temporal_proposal_mode_name(
        args,
        temporal_mode=_temporal_generator_mode_name(args),
    )
    random_proposal_multiplier = _temporal_random_proposal_multiplier(args)
    proposal_rng = np.random.default_rng(int(seed) + 15485863)

    if proposal_mode == "random" and str(getattr(args, "rewire_model", "none")).strip().lower() != "none":
        raise ValueError(
            "Temporal random proposals do not support rewire_model because no SBM proposal graph is sampled. "
            "Set rewire_model='none' for temporal_proposal_mode='random'."
        )

    previous_edges: set[tuple[int, int]] = set()
    seen_edges: set[tuple[int, int]] = set()

    for ts_value, graph_lid in sorted(layer_map.items(), key=lambda item: item[1]):
        ts_int = int(ts_value)
        state_lid = lid_to_state[int(graph_lid)]
        layer_state = base.layer_states[state_lid]
        active_nodes, active_blocks = _activity_nodes_for_timestamp(
            ts_int,
            activity_level=activity_level,
            activity_states=activity_states,
            block_to_nodes=block_to_nodes,
            node_id_to_block=node_id_to_block,
        )
        target_by_group = turnover_targets["by_ts"].get(ts_int, {})
        desired_totals = turnover_targets["totals_by_ts"].get(
            ts_int,
            {"persist": 0, "reactivated": 0, "new": 0, "total": 0},
        )
        desired_total = int(desired_totals.get("total", 0))
        observed_activity_target = observed_realized_activity_targets.get("by_ts", {}).get(
            ts_int,
            {"total": 0, "by_type": {}},
        )
        target_activity_entities = set(int(entity) for entity in activity_snapshot_targets.get(ts_int, set()))

        proposal_counts: Counter[tuple[int, int]] = Counter()
        rounds_used = 0
        shortfalls = {"persist": 0, "reactivated": 0, "new": 0}

        if desired_total > 0 and active_nodes:
            while rounds_used < proposal_rounds_max:
                rounds_used += 1
                if proposal_mode == "sbm":
                    sampled_graph = layer_state.sample_graph(**sample_kwargs)
                    rewire_summary = _maybe_random_rewire_sample(sampled_graph, layer_state, blocks, args)
                    if rewire_summary is not None:
                        rewire_summary = dict(
                            rewire_summary,
                            ts=ts_int,
                            graph_lid=int(graph_lid),
                            state_lid=int(state_lid),
                            proposal_round=int(rounds_used),
                        )
                        rewire_summaries.append(rewire_summary)

                    vmap = layer_state.g.vp["vmap"] if "vmap" in layer_state.g.vp else None
                    if vmap is None:
                        raise RuntimeError("Layer state is missing the vertex property 'vmap'.")

                    accepted_edge_count = 0
                    for edge in sampled_graph.edges():
                        u_layer = int(edge.source())
                        v_layer = int(edge.target())
                        u_base = int(vmap[layer_state.g.vertex(u_layer)])
                        v_base = int(vmap[layer_state.g.vertex(v_layer)])
                        u_original = int(node_id_prop[base.g.vertex(u_base)])
                        v_original = int(node_id_prop[base.g.vertex(v_base)])
                        if u_original < 0 or v_original < 0:
                            continue
                        edge_key = _canonical_edge_pair(u_original, v_original, directed=directed)
                        if not _edge_is_activity_allowed(
                            edge_key,
                            activity_level=activity_level,
                            active_nodes=active_nodes,
                            active_blocks=active_blocks,
                            node_id_to_block=node_id_to_block,
                        ):
                            continue
                        proposal_counts[edge_key] += 1
                        accepted_edge_count += 1
                else:
                    proposal_budget = _random_proposal_edge_budget(
                        desired_total,
                        len(active_nodes),
                        directed=directed,
                        multiplier=float(random_proposal_multiplier),
                    )
                    proposed_edges = _sample_uniform_active_edge_candidates(
                        active_nodes,
                        directed=directed,
                        sample_size=proposal_budget,
                        rng=proposal_rng,
                    )
                    accepted_edge_count = int(len(proposed_edges))
                    for edge_key in proposed_edges:
                        proposal_counts[edge_key] += 1

                LOGGER.debug(
                    "Temporal proposal round | ts=%s | round=%s/%s | proposal_mode=%s | accepted_edges=%s | candidate_edges=%s",
                    ts_int,
                    rounds_used,
                    proposal_rounds_max,
                    proposal_mode,
                    accepted_edge_count,
                    len(proposal_counts),
                )

                pools = _build_turnover_candidate_pools(
                    proposal_counts=proposal_counts,
                    previous_edges=previous_edges,
                    seen_edges=seen_edges,
                    activity_level=activity_level,
                    active_nodes=active_nodes,
                    active_blocks=active_blocks,
                    node_id_to_block=node_id_to_block,
                    node_id_to_type=node_id_to_type or None,
                    group_mode=group_mode,
                    directed=directed,
                )
                shortfalls = _turnover_pool_shortfalls(pools, target_by_group)
                if rounds_used >= proposal_rounds_min and int(shortfalls.get("new", 0)) <= 0:
                    break

        pools = _build_turnover_candidate_pools(
            proposal_counts=proposal_counts,
            previous_edges=previous_edges,
            seen_edges=seen_edges,
            activity_level=activity_level,
            active_nodes=active_nodes,
            active_blocks=active_blocks,
            node_id_to_block=node_id_to_block,
            node_id_to_type=node_id_to_type or None,
            group_mode=group_mode,
            directed=directed,
        )
        selected_edges, achieved_by_group, capacities_by_group = _select_edges_from_turnover_pools(
            pools,
            target_by_group,
            desired_total=desired_total,
            activity_target=observed_activity_target,
            activity_mode=realized_activity_mode,
            activity_weight=float(realized_activity_weight),
            node_id_to_type=node_id_to_type or None,
        )
        selected_edges = sorted(selected_edges)
        achieved_totals = _aggregate_turnover_totals(achieved_by_group)

        records: list[dict[str, Any]] = []
        weight_values_for_graph: list[float] = []
        for edge_key in selected_edges:
            u_original, v_original = edge_key
            record: dict[str, Any] = {
                "u": int(u_original),
                "i": int(v_original),
                "ts": ts_int,
                "snapshot": int(graph_lid),
            }
            if weight_sampler is not None and weight_col is not None:
                u_base = int(node_id_to_base[int(u_original)])
                v_base = int(node_id_to_base[int(v_original)])
                src_type = node_id_to_type.get(int(u_original)) if node_id_to_type else None
                dst_type = node_id_to_type.get(int(v_original)) if node_id_to_type else None
                if weight_generation_mode.startswith("parametric"):
                    weight_value = weight_sampler.sample(
                        ts_value=ts_int,
                        r=int(weight_blocks[u_base]),
                        s=int(weight_blocks[v_base]),
                        src_type=src_type,
                        dst_type=dst_type,
                    )
                else:
                    weight_value = weight_sampler.sample(
                        ts_value=ts_int,
                        r=int(blocks[u_base]),
                        s=int(blocks[v_base]),
                        src_type=src_type,
                        dst_type=dst_type,
                    )
                record[weight_col] = weight_value
                weight_values_for_graph.append(float(weight_value))
            if not directed and record["u"] > record["i"]:
                record["u"], record["i"] = record["i"], record["u"]
            records.append(record)
            sample_records.append(record)

        snapshot_frame = pd.DataFrame.from_records(records, columns=record_columns)
        snapshot_path = snapshot_dir / f"snapshot_{ts_int}.csv"
        snapshot_frame.to_csv(snapshot_path, index=False)
        realized_active_nodes = {int(node_id) for edge in selected_edges for node_id in edge}
        eligible_type_counts = _count_nodes_by_type(active_nodes, node_id_to_type or None)
        realized_type_counts = _count_nodes_by_type(realized_active_nodes, node_id_to_type or None)
        observed_type_targets = {
            int(type_id): int(count)
            for type_id, count in dict(observed_activity_target.get("by_type", {})).items()
        }
        realized_type_shortfalls = {
            int(type_id): max(0, int(target_count) - int(realized_type_counts.get(int(type_id), 0)))
            for type_id, target_count in observed_type_targets.items()
        }
        LOGGER.debug(
            "Temporal synthetic snapshot | ts=%s | graph_lid=%s | state_lid=%s | proposal_mode=%s | desired_edges=%s | achieved_edges=%s | eligible_active_nodes=%s | realized_active_nodes=%s%s",
            ts_int,
            graph_lid,
            state_lid,
            proposal_mode,
            desired_total,
            len(snapshot_frame),
            len(active_nodes),
            len(realized_active_nodes),
            (
                f" | weight_total={float(snapshot_frame[weight_col].sum()):.6f}"
                if weight_col and weight_col in snapshot_frame.columns and len(snapshot_frame)
                else ""
            ),
        )
        if args.save_graph_tool_snapshots:
            output_graph = _build_snapshot_gt_graph(
                gt,
                selected_edges,
                directed=directed,
                weight_col=weight_col,
                weight_values=weight_values_for_graph if weight_col else None,
            )
            output_graph.save(str(snapshot_dir / f"snapshot_{ts_int}.gt"))

        previous_edges = set(selected_edges)
        seen_edges |= previous_edges

        temporal_rows.append(
            {
                "ts": ts_int,
                "proposal_mode": proposal_mode,
                "proposal_rounds_used": int(rounds_used),
                "proposal_candidate_edge_count": int(len(proposal_counts)),
                "observed_active_entity_count": int(activity_path.get("observed_active_counts", {}).get(ts_int, 0)),
                "activity_snapshot_target_entity_count": int(len(target_activity_entities)),
                "sampled_active_entity_count": int(len(activity_states.get(ts_int, set()))),
                "activity_snapshot_exact_match": (
                    None
                    if not activity_snapshot_targets
                    else bool(activity_states.get(ts_int, set()) == target_activity_entities)
                ),
                "observed_active_node_count": int(observed_activity_target.get("total", 0)),
                "sampled_active_node_count": int(len(active_nodes)),
                "eligible_active_node_count": int(len(active_nodes)),
                "realized_active_node_count": int(len(realized_active_nodes)),
                "observed_active_node_count_by_type": observed_type_targets,
                "eligible_active_node_count_by_type": eligible_type_counts,
                "realized_active_node_count_by_type": realized_type_counts,
                "realized_active_node_shortfall": max(
                    0,
                    int(observed_activity_target.get("total", 0)) - int(len(realized_active_nodes)),
                ),
                "realized_active_node_shortfall_by_type": realized_type_shortfalls,
                "target_persist_count": int(desired_totals.get("persist", 0)),
                "target_reactivated_count": int(desired_totals.get("reactivated", 0)),
                "target_new_count": int(desired_totals.get("new", 0)),
                "target_total_count": int(desired_totals.get("total", 0)),
                "achieved_persist_count": int(achieved_totals.get("persist", 0)),
                "achieved_reactivated_count": int(achieved_totals.get("reactivated", 0)),
                "achieved_new_count": int(achieved_totals.get("new", 0)),
                "achieved_total_count": int(achieved_totals.get("total", 0)),
                "new_shortfall_after_proposals": int(shortfalls.get("new", 0)),
                "target_by_group": target_by_group,
                "achieved_by_group": achieved_by_group,
                "capacity_by_group": capacities_by_group,
            }
        )

    panel_frame = pd.DataFrame.from_records(sample_records, columns=record_columns)
    pre_dedup_count = int(len(panel_frame))
    panel_frame = panel_frame.drop_duplicates(["u", "i", "ts", "snapshot"]).reset_index(drop=True)
    panel_path = output_dir / "synthetic_edges.csv"
    panel_frame.to_csv(panel_path, index=False)

    if weight_sampler is not None and hasattr(weight_sampler, "resolution_counts"):
        LOGGER.debug("Weight sampling resolution counts | %s", dict(weight_sampler.resolution_counts))
    LOGGER.debug(
        "Temporal synthetic panel summary | rows_before_dedup=%s | rows_after_dedup=%s%s",
        pre_dedup_count,
        len(panel_frame),
        (
            f" | weight_total={float(panel_frame[weight_col].sum()):.6f}"
            if weight_col and weight_col in panel_frame.columns and len(panel_frame)
            else ""
        ),
    )

    mean_edge_shortfall = (
        float(
            np.mean(
                [
                    int(row["target_total_count"]) - int(row["achieved_total_count"])
                    for row in temporal_rows
                ]
            )
        )
        if temporal_rows
        else 0.0
    )
    mean_active_node_shortfall = (
        float(
            np.mean(
                [
                    max(
                        0,
                        int(row.get("observed_active_node_count", 0))
                        - int(row.get("realized_active_node_count", 0)),
                    )
                    for row in temporal_rows
                ]
            )
        )
        if temporal_rows
        else 0.0
    )
    temporal_summary_path = output_dir / "temporal_generation_summary.json"
    temporal_summary_payload = {
        "mode": _temporal_generator_mode_name(args),
        "proposal_mode": proposal_mode,
        "target_source": temporal_target_source,
        "activity_level": activity_level,
        "group_mode": group_mode,
        "realized_activity_mode": realized_activity_mode,
        "realized_activity_weight": float(realized_activity_weight),
        "proposal_rounds_min": int(proposal_rounds_min),
        "proposal_rounds_max": int(proposal_rounds_max),
        "random_proposal_multiplier": float(random_proposal_multiplier),
        "activity_path": {
            "entity_count": int(len(activity_path.get("entities", []))),
            "stores_snapshot_targets": bool(activity_snapshot_targets),
        },
        "activity_path_summary": _json_ready(activity_sampling_summary),
        "realized_activity_targets": {
            "types": _json_ready(observed_realized_activity_targets.get("types", [])),
            "mean_active_node_shortfall": float(mean_active_node_shortfall),
        },
        "per_snapshot": _serialise_temporal_generation_rows(temporal_rows),
    }
    save_json(temporal_summary_payload, temporal_summary_path)

    rewire_model = str(getattr(args, "rewire_model", "none"))
    sample_manifest_path = output_dir / "sample_manifest.json"
    setting_dir = output_dir.parent
    sample_class = (
        "proposal_ablation"
        if proposal_mode == "random"
        else ("sensitivity_analysis" if rewire_model != "none" else "posterior_predictive")
    )
    payload = {
        "sample_dir": str(output_dir),
        "sample_manifest_path": str(sample_manifest_path),
        "synthetic_edges_csv": str(panel_path),
        "snapshot_dir": str(snapshot_dir),
        "setting_label": _generation_setting_label(args),
        "setting_dir": str(setting_dir),
        "setting_manifest_path": str(setting_dir / "setting_manifest.json"),
        "node_partition_path": str(partition_path),
        "partition_source": "posterior_refresh" if int(getattr(args, "posterior_partition_sweeps", 0)) > 0 else "fitted_state",
        "sample_class": sample_class,
        "sample_seed": int(seed),
        "edge_count": int(len(panel_frame)),
        "generation_args": _serialise_generation_args(args),
        "sample_settings": {
            "sample_mode": _generation_sample_mode_label(args),
            "sample_canonical": bool(getattr(args, "sample_canonical", False)),
            "sample_max_ent": bool(getattr(args, "sample_max_ent", False)),
            "sample_n_iter": int(getattr(args, "sample_n_iter", 20000)),
            "sample_params": None if getattr(args, "sample_params", None) is None else bool(getattr(args, "sample_params")),
            "rewire_model": rewire_model,
            "rewire_n_iter": int(getattr(args, "rewire_n_iter", 10)),
            "rewire_persist": bool(getattr(args, "rewire_persist", False)),
            "is_sensitivity_analysis": bool(rewire_model != "none"),
            "weight_sampler_channel_aware": bool(type_prop is not None and weight_sampler is not None),
            "weight_generation_mode": weight_generation_mode,
            "weight_parametric_partition_policy": weight_partition_policy,
            "weight_pure_generative": pure_generative_weights,
            "temporal_generator_mode": _temporal_generator_mode_name(args),
            "temporal_proposal_mode": proposal_mode,
            "temporal_target_source": temporal_target_source,
            "temporal_activity_level": activity_level,
            "temporal_group_mode": group_mode,
            "temporal_realized_activity_mode": realized_activity_mode,
            "temporal_realized_activity_weight": float(realized_activity_weight),
            "temporal_activity_path_mode": str(activity_sampling_summary.get("mode", "stored")),
            "temporal_activity_path_source": str(activity_sampling_summary.get("selected_source", "stored_activity_path")),
            "temporal_proposal_rounds": int(proposal_rounds_min),
            "temporal_proposal_rounds_max": int(proposal_rounds_max),
            "temporal_random_proposal_multiplier": float(random_proposal_multiplier),
        },
        "temporal_generation_summary_path": str(temporal_summary_path),
        "temporal_generation_summary": {
            "mode": _temporal_generator_mode_name(args),
            "proposal_mode": proposal_mode,
            "activity_level": activity_level,
            "group_mode": group_mode,
            "realized_activity_mode": realized_activity_mode,
            "activity_path_summary": _json_ready(activity_sampling_summary),
            "proposal_rounds_min": int(proposal_rounds_min),
            "proposal_rounds_max": int(proposal_rounds_max),
            "random_proposal_multiplier": float(random_proposal_multiplier),
            "mean_edge_shortfall": float(mean_edge_shortfall),
            "mean_active_node_shortfall": float(mean_active_node_shortfall),
        },
    }
    if weight_col and weight_col in panel_frame.columns:
        payload["weight_column"] = weight_col
        payload["weight_total"] = float(panel_frame[weight_col].sum()) if len(panel_frame) else 0.0
    if active_weight_generator is not None:
        payload["weight_generator_summary"] = _json_ready(active_weight_generator.get("summary", {}))
        if active_weight_generator.get("family") is not None:
            payload["weight_generator_family"] = str(active_weight_generator.get("family"))
        families_by_channel = active_weight_generator.get("summary", {}).get("families_by_channel")
        if isinstance(families_by_channel, dict) and families_by_channel:
            payload["weight_generator_families_by_channel"] = _json_ready(families_by_channel)
    if rewire_summaries:
        payload["rewire_summaries"] = rewire_summaries
    if weight_sampler is not None and hasattr(weight_sampler, "resolution_counts"):
        payload["weight_sampling_resolution_counts"] = {
            str(key): int(value) for key, value in sorted(weight_sampler.resolution_counts.items())
        }
    save_json(payload, sample_manifest_path)
    return payload


def _sample_synthetic_panel_independent_layers(
    graph: Any,
    nested_state: Any,
    layer_map: dict[int, int],
    output_dir: Path,
    directed: bool,
    seed: int,
    args: argparse.Namespace,
    observed_edges: Optional[pd.DataFrame] = None,
    weight_model: Optional[dict] = None,
    weight_generator_model: Optional[dict] = None,
) -> dict:
    gt = _require_graph_tool()
    gt.seed_rng(int(seed))
    weight_partition_policy = str(getattr(args, "weight_parametric_partition_policy", "fixed")).strip().lower()
    pure_generative_weights = _pure_generative_weight_mode(args)
    if pure_generative_weights and weight_partition_policy == "refit_on_refresh":
        raise ValueError(
            "weight_pure_generative=True is incompatible with weight_parametric_partition_policy='refit_on_refresh'."
        )
    LOGGER.debug(
        "Sampling synthetic panel | output_dir=%s | directed=%s | seed=%s | layer_count=%s | "
        "weight_model=%s | weight_generator=%s | weight_partition_policy=%s | weight_pure_generative=%s",
        output_dir,
        directed,
        seed,
        len(layer_map),
        weight_model,
        None if weight_generator_model is None else weight_generator_model.get("summary"),
        weight_partition_policy,
        pure_generative_weights,
    )

    sampled_state = _posterior_partition_state(nested_state, seed=seed, args=args)
    base = _base_state(sampled_state)
    lid_to_state = map_graph_lid_to_state_lid(sampled_state)
    LOGGER.debug("Sampled state ready | %s", _state_summary_text(sampled_state))

    node_id_prop = base.g.vp["node_id"] if "node_id" in base.g.vp else None
    if node_id_prop is None:
        raise RuntimeError("Fitted graph is missing the vertex property 'node_id'.")

    blocks = _node_blocks_from_state(sampled_state)
    active_weight_generator = weight_generator_model
    weight_generation_mode = "none"

    if weight_model is not None and observed_edges is not None and weight_partition_policy == "refit_on_refresh":
        active_weight_generator = _fit_parametric_weight_generator_model(
            observed_edges=observed_edges,
            base=base,
            weight_model=weight_model,
            directed=directed,
            args=args,
            blocks=blocks,
        )
        weight_generation_mode = "parametric_refit"
    elif active_weight_generator is not None:
        weight_generation_mode = "parametric_fixed"

    if pure_generative_weights and weight_model is not None and active_weight_generator is None:
        raise ValueError(
            "weight_pure_generative=True requires a fitted parametric weight generator. "
            "Generation cannot proceed with empirical weight backoff."
        )

    weight_blocks = blocks
    if active_weight_generator is not None and weight_partition_policy != "refit_on_refresh":
        if pure_generative_weights:
            stored_weight_blocks = _stored_weight_reference_blocks(
                base=base,
                blocks=blocks,
                node_id_prop=node_id_prop,
                weight_generator_model=active_weight_generator,
            )
            if stored_weight_blocks is None:
                raise ValueError(
                    "weight_pure_generative=True requires node_blocks in the saved parametric weight generator."
                )
            weight_blocks = stored_weight_blocks
            LOGGER.debug(
                "Using stored node blocks from the saved parametric weight generator | unique_blocks=%s",
                int(np.unique(weight_blocks).size),
            )
        else:
            weight_blocks = _aligned_blocks_for_weight_generation(
                gt=gt,
                base=base,
                blocks=blocks,
                node_id_prop=node_id_prop,
                weight_generator_model=active_weight_generator,
            )

    partition_records = []
    include_weight_blocks = active_weight_generator is not None and not np.array_equal(weight_blocks, blocks)
    active_prop = base.g.vp["active_in_window"] if "active_in_window" in base.g.vp else None
    metadata_prop = base.g.vp["is_metadata_tag"] if "is_metadata_tag" in base.g.vp else None
    for index in range(int(base.g.num_vertices())):
        vertex = base.g.vertex(index)
        if metadata_prop is not None and bool(metadata_prop[vertex]):
            continue
        node_id = int(node_id_prop[vertex])
        if node_id < 0:
            continue
        record = {
            "node_id": node_id,
            "block_id": int(blocks[index]),
        }
        if active_prop is not None:
            record["active_in_window"] = int(active_prop[vertex])
        if include_weight_blocks:
            record["weight_block_id"] = int(weight_blocks[index])
        partition_records.append(record)

    partition_frame = pd.DataFrame(partition_records).sort_values(
        ["node_id", "block_id"] + (["weight_block_id"] if include_weight_blocks else [])
    ).reset_index(drop=True)
    partition_path = Path(output_dir) / "sample_node_partition.csv"
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_frame.to_csv(partition_path, index=False)

    weight_col = None
    weight_sampler = None
    type_prop = base.g.vp["type"] if "type" in base.g.vp else None
    if active_weight_generator is not None:
        weight_col = str(active_weight_generator["output_column"])
        weight_sampler = ParametricWeightSampler(
            weight_generator_model=active_weight_generator,
            directed=directed,
            rng=np.random.default_rng(int(seed)),
        )
    elif weight_model and observed_edges is not None:
        if pure_generative_weights:
            raise ValueError(
                "weight_pure_generative=True forbids empirical weight backoff. "
                "Fit and load a parametric weight generator instead."
            )
        weight_col = str(weight_model.get("output_column") or weight_model.get("input_column"))
        node_id_to_base = {
            int(node_id_prop[base.g.vertex(index)]): int(index)
            for index in range(int(base.g.num_vertices()))
        }
        node_id_to_type = None
        if type_prop is not None:
            node_id_to_type = {
                int(node_id_prop[base.g.vertex(index)]): int(type_prop[base.g.vertex(index)])
                for index in range(int(base.g.num_vertices()))
            }
        weight_sampler = EdgeWeightSampler(
            observed_edges=observed_edges,
            node_id_to_base=node_id_to_base,
            blocks=blocks,
            weight_model=weight_model,
            directed=directed,
            min_cell_count=max(1, int(getattr(args, "weight_min_cell_count", 3))),
            rng=np.random.default_rng(int(seed)),
            node_id_to_type=node_id_to_type,
        )
        weight_generation_mode = "empirical_backoff"

    output_dir = Path(output_dir)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    sample_records: list[dict[str, Any]] = []
    sample_kwargs = _sample_kwargs(args)
    record_columns = ["u", "i", "ts", "snapshot"] + ([weight_col] if weight_col else [])
    rewire_summaries: list[dict[str, Any]] = []

    for ts_value, graph_lid in sorted(layer_map.items(), key=lambda item: item[1]):
        state_lid = lid_to_state[int(graph_lid)]
        layer_state = base.layer_states[state_lid]
        sampled_graph = layer_state.sample_graph(**sample_kwargs)
        rewire_summary = _maybe_random_rewire_sample(sampled_graph, layer_state, blocks, args)
        if rewire_summary is not None:
            rewire_summary = dict(rewire_summary, ts=int(ts_value), graph_lid=int(graph_lid), state_lid=int(state_lid))
            rewire_summaries.append(rewire_summary)

        vmap = layer_state.g.vp["vmap"] if "vmap" in layer_state.g.vp else None
        if vmap is None:
            raise RuntimeError("Layer state is missing the vertex property 'vmap'.")

        records = []
        for edge in sampled_graph.edges():
            u_layer = int(edge.source())
            v_layer = int(edge.target())
            u_base = int(vmap[layer_state.g.vertex(u_layer)])
            v_base = int(vmap[layer_state.g.vertex(v_layer)])
            u_original = int(node_id_prop[base.g.vertex(u_base)])
            v_original = int(node_id_prop[base.g.vertex(v_base)])

            record: dict[str, Any] = {
                "u": u_original,
                "i": v_original,
                "ts": int(ts_value),
                "snapshot": int(graph_lid),
            }
            if weight_sampler is not None and weight_col:
                src_type = int(type_prop[base.g.vertex(u_base)]) if type_prop is not None else None
                dst_type = int(type_prop[base.g.vertex(v_base)]) if type_prop is not None else None
                if weight_generation_mode.startswith("parametric"):
                    record[weight_col] = weight_sampler.sample(
                        ts_value=int(ts_value),
                        r=int(weight_blocks[u_base]),
                        s=int(weight_blocks[v_base]),
                        src_type=src_type,
                        dst_type=dst_type,
                    )
                else:
                    record[weight_col] = weight_sampler.sample(
                        ts_value=int(ts_value),
                        r=int(blocks[u_base]),
                        s=int(blocks[v_base]),
                        src_type=src_type,
                        dst_type=dst_type,
                    )
            if not directed and record["u"] > record["i"]:
                record["u"], record["i"] = record["i"], record["u"]
            records.append(record)
            sample_records.append(record)

        snapshot_frame = pd.DataFrame.from_records(records, columns=record_columns)
        snapshot_path = snapshot_dir / f"snapshot_{int(ts_value)}.csv"
        snapshot_frame.to_csv(snapshot_path, index=False)
        LOGGER.debug(
            "Sampled snapshot | ts=%s | graph_lid=%s | state_lid=%s | edges=%s | unique_nodes=%s%s",
            ts_value,
            graph_lid,
            state_lid,
            len(snapshot_frame),
            int(len(set(snapshot_frame["u"].tolist()) | set(snapshot_frame["i"].tolist()))) if len(snapshot_frame) else 0,
            (
                f" | weight_total={float(snapshot_frame[weight_col].sum()):.6f}"
                if weight_col and weight_col in snapshot_frame.columns and len(snapshot_frame)
                else ""
            ),
        )
        if args.save_graph_tool_snapshots:
            sampled_graph.save(str(snapshot_dir / f"snapshot_{int(ts_value)}.gt"))

    panel_frame = pd.DataFrame.from_records(sample_records, columns=record_columns)
    pre_dedup_count = int(len(panel_frame))
    panel_frame = panel_frame.drop_duplicates(["u", "i", "ts", "snapshot"]).reset_index(drop=True)
    panel_path = output_dir / "synthetic_edges.csv"
    panel_frame.to_csv(panel_path, index=False)

    if weight_sampler is not None and hasattr(weight_sampler, "resolution_counts"):
        LOGGER.debug("Weight sampling resolution counts | %s", dict(weight_sampler.resolution_counts))
    LOGGER.debug(
        "Synthetic panel summary | rows_before_dedup=%s | rows_after_dedup=%s%s",
        pre_dedup_count,
        len(panel_frame),
        (
            f" | weight_total={float(panel_frame[weight_col].sum()):.6f}"
            if weight_col and weight_col in panel_frame.columns and len(panel_frame)
            else ""
        ),
    )

    rewire_model = str(getattr(args, "rewire_model", "none"))
    sample_manifest_path = output_dir / "sample_manifest.json"
    setting_dir = output_dir.parent
    payload = {
        "sample_dir": str(output_dir),
        "sample_manifest_path": str(sample_manifest_path),
        "synthetic_edges_csv": str(panel_path),
        "snapshot_dir": str(snapshot_dir),
        "setting_label": _generation_setting_label(args),
        "setting_dir": str(setting_dir),
        "setting_manifest_path": str(setting_dir / "setting_manifest.json"),
        "node_partition_path": str(partition_path),
        "partition_source": "posterior_refresh" if int(getattr(args, "posterior_partition_sweeps", 0)) > 0 else "fitted_state",
        "sample_class": "sensitivity_analysis" if rewire_model != "none" else "posterior_predictive",
        "sample_seed": int(seed),
        "edge_count": int(len(panel_frame)),
        "generation_args": _serialise_generation_args(args),
        "sample_settings": {
            "sample_mode": _generation_sample_mode_label(args),
            "sample_canonical": bool(getattr(args, "sample_canonical", False)),
            "sample_max_ent": bool(getattr(args, "sample_max_ent", False)),
            "sample_n_iter": int(getattr(args, "sample_n_iter", 20000)),
            "sample_params": None if getattr(args, "sample_params", None) is None else bool(getattr(args, "sample_params")),
            "rewire_model": rewire_model,
            "rewire_n_iter": int(getattr(args, "rewire_n_iter", 10)),
            "rewire_persist": bool(getattr(args, "rewire_persist", False)),
            "is_sensitivity_analysis": bool(rewire_model != "none"),
            "weight_sampler_channel_aware": bool(type_prop is not None and weight_sampler is not None),
            "weight_generation_mode": weight_generation_mode,
            "weight_parametric_partition_policy": weight_partition_policy,
            "weight_pure_generative": pure_generative_weights,
        },
    }
    if weight_col and weight_col in panel_frame.columns:
        payload["weight_column"] = weight_col
        payload["weight_total"] = float(panel_frame[weight_col].sum()) if len(panel_frame) else 0.0
    if active_weight_generator is not None:
        payload["weight_generator_summary"] = _json_ready(active_weight_generator.get("summary", {}))
        if active_weight_generator.get("family") is not None:
            payload["weight_generator_family"] = str(active_weight_generator.get("family"))
        families_by_channel = active_weight_generator.get("summary", {}).get("families_by_channel")
        if isinstance(families_by_channel, dict) and families_by_channel:
            payload["weight_generator_families_by_channel"] = _json_ready(families_by_channel)
    if rewire_summaries:
        payload["rewire_summaries"] = rewire_summaries
    if weight_sampler is not None and hasattr(weight_sampler, "resolution_counts"):
        payload["weight_sampling_resolution_counts"] = {
            str(key): int(value) for key, value in sorted(weight_sampler.resolution_counts.items())
        }
    save_json(payload, sample_manifest_path)
    return payload





def sample_synthetic_panel(
    graph: Any,
    nested_state: Any,
    layer_map: dict[int, int],
    output_dir: Path,
    directed: bool,
    seed: int,
    args: argparse.Namespace,
    observed_edges: Optional[pd.DataFrame] = None,
    weight_model: Optional[dict] = None,
    weight_generator_model: Optional[dict] = None,
    temporal_generator_model: Optional[dict[str, Any]] = None,
) -> dict:
    temporal_mode = _temporal_generator_mode_name(args)
    if temporal_mode == "independent_sbm":
        return _sample_synthetic_panel_independent_layers(
            graph=graph,
            nested_state=nested_state,
            layer_map=layer_map,
            output_dir=output_dir,
            directed=directed,
            seed=seed,
            args=args,
            observed_edges=observed_edges,
            weight_model=weight_model,
            weight_generator_model=weight_generator_model,
        )
    if observed_edges is None and temporal_generator_model is None:
        raise ValueError(
            f"temporal_generator_mode={temporal_mode!r} requires observed_edges or a saved temporal generator model "
            "so the activity and turnover targets can be prepared."
        )
    return _sample_synthetic_panel_operational_turnover(
        graph=graph,
        nested_state=nested_state,
        layer_map=layer_map,
        output_dir=output_dir,
        directed=directed,
        seed=seed,
        args=args,
        observed_edges=observed_edges,
        weight_model=weight_model,
        weight_generator_model=weight_generator_model,
        temporal_generator_model=temporal_generator_model,
    )

def save_json(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
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


def _serialise_generation_args(args: argparse.Namespace) -> dict[str, object]:
    return {str(key): _json_ready(value) for key, value in vars(args).items()}


def _merge_generated_sample_records(existing: list[dict], new_records: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in existing:
        key = str(
            record.get("sample_manifest_path")
            or record.get("synthetic_edges_csv")
            or record.get("sample_label")
            or len(merged)
        )
        merged[key] = record
    for record in new_records:
        key = str(
            record.get("sample_manifest_path")
            or record.get("synthetic_edges_csv")
            or record.get("sample_label")
            or len(merged)
        )
        merged[key] = record
    return sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("setting_label") or ""),
            int(record.get("sample_index", 0)),
            str(record.get("sample_manifest_path") or ""),
        ),
    )


def _merge_generated_setting_records(existing: list[dict], new_record: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in existing:
        key = str(
            record.get("setting_manifest_path")
            or record.get("setting_dir")
            or record.get("setting_label")
            or len(merged)
        )
        merged[key] = record
    key = str(
        new_record.get("setting_manifest_path")
        or new_record.get("setting_dir")
        or new_record.get("setting_label")
    )
    merged[key] = new_record
    return sorted(merged.values(), key=lambda record: str(record.get("setting_label") or ""))




def write_fit_artifacts(
    prepared: PreparedData,
    graph: Any,
    nested_state: Any,
    output_dir: Path,
    args: argparse.Namespace,
    fit_covariates: list[str],
    weight_model: Optional[dict] = None,
    weight_generator_model: Optional[dict] = None,
    temporal_generator_model: Optional[dict[str, Any]] = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Writing fit artifacts under %s", output_dir)

    graph_path = output_dir / "layered_model.gt"
    state_path = output_dir / "nested_state.pkl.gz"
    layer_map_path = output_dir / "layer_map.json"
    filtered_edges_path = output_dir / "input_edges_filtered.csv"
    node_attributes_path = output_dir / "node_attributes.csv"
    weight_generator_path = output_dir / "weight_generator_model.json" if weight_generator_model is not None else None
    temporal_generator_path = output_dir / "temporal_generator_model.json" if temporal_generator_model is not None else None

    graph.save(str(graph_path))
    save_state(nested_state, state_path)
    save_json({str(ts): int(lid) for ts, lid in prepared.layer_map.items()}, layer_map_path)
    prepared.original_edges.to_csv(filtered_edges_path, index=False)
    write_node_attributes(prepared, node_attributes_path, node_blocks=extract_node_block_map(graph))
    if weight_generator_model is not None and weight_generator_path is not None:
        save_json(weight_generator_model, weight_generator_path)
    if temporal_generator_model is not None and temporal_generator_path is not None:
        save_json(temporal_generator_model, temporal_generator_path)

    input_summary = {
        "edge_count": int(len(prepared.original_edges)),
        "node_count": int(len(prepared.compact_to_original)),
        "active_node_count": int(prepared.active_compact_mask.sum()),
        "inactive_node_count": int(len(prepared.compact_to_original) - int(prepared.active_compact_mask.sum())),
        "layer_count": int(len(prepared.layer_map)),
        "duplicate_edge_count": int(prepared.duplicate_edge_count),
        "self_loop_count": int(prepared.self_loop_count),
        "metadata_link_count": int(prepared.metadata_summary.get("num_links", 0)),
        "metadata_tag_count": int(prepared.metadata_summary.get("num_tags", 0)),
        "metadata_fields_used": list(prepared.metadata_summary.get("fields_used", [])),
    }
    if prepared.weight_column:
        input_summary["weight_column"] = prepared.weight_column
        input_summary["weight_total"] = float(prepared.original_edges[prepared.weight_column].sum())
        input_summary["weight_mean"] = float(prepared.original_edges[prepared.weight_column].mean())
    if prepared.input_paths.weight_npy is not None:
        input_summary["weight_npy"] = str(prepared.input_paths.weight_npy)

    manifest = {
        "dataset": prepared.input_paths.dataset,
        "data_root": str(prepared.input_paths.dataset_dir.parent),
        "dataset_dir": str(prepared.input_paths.dataset_dir),
        "run_dir": str(output_dir),
        "graph_path": str(graph_path),
        "state_path": str(state_path),
        "layer_map_path": str(layer_map_path),
        "filtered_input_edges_path": str(filtered_edges_path),
        "node_attributes_path": str(node_attributes_path),
        "node_map_csv": str(prepared.input_paths.node_map_csv),
        "directed": bool(args.directed),
        "no_compact": bool(prepared.no_compact),
        "node_universe_scope": "full_node_universe" if prepared.no_compact else "active_subgraph",
        "metadata_model": {
            "enabled": bool(prepared.metadata_summary.get("enabled")),
            "implementation": "joint_data_metadata_multilayer",
            "paper": "Hric-Peixoto-Fortunato-2016",
            "metadata_layer_name": METADATA_LAYER_NAME,
            "summary": _json_ready(prepared.metadata_summary),
        },
        "ts_format": args.ts_format,
        "ts_unit": args.ts_unit,
        "tz": args.tz,
        "holiday_country": args.holiday_country,
        "duplicate_policy": args.duplicate_policy,
        "self_loop_policy": args.self_loop_policy,
        "weight_npy": str(prepared.input_paths.weight_npy) if prepared.input_paths.weight_npy is not None else None,
        "fit_covariates": list(fit_covariates),
        "available_edge_covariates": list(ALL_EDGE_COVARIATES),
        "default_fit_covariates": list(DEFAULT_COVARIATES),
        "layer_derived_covariates": list(LAYER_DERIVED_COVARIATES),
        "edge_covariate_scope": "realized_edges_only",
        "rewired_samples_are_sensitivity_analyses": True,
        "weight_model": weight_model,
        "weight_generator_path": str(weight_generator_path) if weight_generator_path is not None else None,
        "weight_generator_summary": weight_generator_model.get("summary") if weight_generator_model is not None else None,
        "temporal_generator_path": str(temporal_generator_path) if temporal_generator_path is not None else None,
        "temporal_generator_summary": temporal_generator_model.get("summary") if temporal_generator_model is not None else None,
        "temporal_generation_mode": (
            "stored_operational_temporal_model"
            if temporal_generator_model is not None
            else None
        ),
        "weight_generation_mode": (
            "parametric"
            if weight_generator_model is not None
            else ("empirical_backoff" if weight_model is not None else None)
        ),
        "fit_options": {
            "layered": bool(getattr(args, "layered", True)),
            "allow_mixed_node_types": bool(getattr(args, "allow_mixed_node_types", False)),
            "deg_corr": not args.no_deg_corr,
            "overlap": bool(args.overlap),
            "exclude_weight_from_fit": bool(getattr(args, "exclude_weight_from_fit", False)),
            "weight_pure_generative": _pure_generative_weight_mode(args),
            "weight_parametric_partition_policy": str(getattr(args, "weight_parametric_partition_policy", "fixed")).strip().lower(),
            "refine_multiflip_rounds": int(args.refine_multiflip_rounds),
            "refine_multiflip_niter": int(args.refine_multiflip_niter),
            "anneal_niter": int(args.anneal_niter),
            "anneal_beta_start": float(args.anneal_beta_start),
            "anneal_beta_stop": float(args.anneal_beta_stop),
        },
        "input_summary": input_summary,
        "generated_samples": [],
        "diagnostics": [],
    }
    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    LOGGER.debug(
        "Fit artifacts written | graph_path=%s | state_path=%s | layer_map_path=%s | filtered_edges_path=%s | "
        "node_attributes_path=%s | weight_generator_path=%s | temporal_generator_path=%s | manifest_path=%s",
        graph_path,
        state_path,
        layer_map_path,
        filtered_edges_path,
        node_attributes_path,
        weight_generator_path,
        temporal_generator_path,
        manifest_path,
    )
    return manifest


def load_manifest(run_dir: Path) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = load_json(manifest_path)
    payload["manifest_path"] = str(manifest_path)
    return payload


def update_manifest(run_dir: Path, **updates: Any) -> dict:
    manifest = load_manifest(run_dir)
    manifest.update(updates)
    save_json(manifest, Path(manifest["manifest_path"]))
    return manifest



def fit_command(args: argparse.Namespace) -> dict:
    LOGGER.debug("Starting fit command | args=%s", vars(args))
    prepared = prepare_data(args)
    _validate_weight_generation_configuration(args, has_weight_data=bool(prepared.weight_column))
    graph = build_layered_graph(prepared, directed=bool(args.directed))
    joint_metadata_active = bool(prepared.metadata_summary.get("enabled"))
    requested_fit_covariates = getattr(args, "fit_covariates", None)
    if joint_metadata_active and requested_fit_covariates is None:
        base_covariate_specs = []
        LOGGER.info(
            "Joint data-metadata model active: using topology-only fitting by default, matching Hric–Peixoto–Fortunato (2016). "
            "Set fit_covariates explicitly to add trade-edge covariates on top of the metadata layer."
        )
    else:
        base_covariate_specs = _select_covariate_specs(_available_covariate_specs(graph), requested_fit_covariates)
    weight_generation_mode = _weight_generation_mode_name(args)
    include_weight_in_fit = _fit_includes_edge_weight_covariate(args)
    if include_weight_in_fit:
        weight_candidates = _build_weight_candidates(prepared, graph, args)
    else:
        LOGGER.info("Edge weights are excluded from SBM fitting and will be handled by the weight generator.")
        if prepared.weight_column and weight_generation_mode in {"legacy", "empirical", "empirical_backoff"}:
            raise ValueError(
                "Edge weights cannot be excluded from SBM fitting when legacy weight generation is requested."
            )
        weight_candidates = []
    _log_covariate_specs("Default fit covariates", base_covariate_specs)
    _log_covariate_interpretation_notes(base_covariate_specs)
    t0 = pd.Timestamp.utcnow()

    nested_state, weight_model, fit_covariates = _fit_with_weight_candidates(
        graph=graph,
        base_covariate_specs=base_covariate_specs,
        weight_candidates=weight_candidates,
        prepared=prepared,
        args=args,
    )
    if weight_model is None and prepared.weight_column and weight_generation_mode in {"parametric", "model", "generative"}:
        weight_model = _standalone_weight_model(prepared)
    attach_partition_maps(graph, nested_state)

    weight_generator_model = None
    if weight_model is not None and weight_generation_mode in {"parametric", "model", "generative"}:
        weight_generator_model = _fit_parametric_weight_generator_model(
            observed_edges=prepared.original_edges,
            base=_base_state(nested_state),
            weight_model=weight_model,
            directed=bool(args.directed),
            args=args,
        )
    elif weight_model is not None and weight_generation_mode not in {"legacy", "empirical", "empirical_backoff", "none"}:
        raise ValueError(
            "Unknown weight_generation_mode. Use 'parametric' to save a fully fitted generator or "
            "'legacy' to keep the empirical backoff sampler."
        )

    if _pure_generative_weight_mode(args) and prepared.weight_column and weight_generator_model is None:
        raise ValueError(
            "weight_pure_generative=True requires fitting and saving a parametric weight generator."
        )

    temporal_generator_model = _fit_temporal_generator_model(
        prepared,
        nested_state,
        directed=bool(args.directed),
    )

    manifest = write_fit_artifacts(
        prepared,
        graph,
        nested_state,
        Path(args.output_dir).expanduser().resolve(),
        args,
        fit_covariates=fit_covariates,
        weight_model=weight_model,
        weight_generator_model=weight_generator_model,
        temporal_generator_model=temporal_generator_model,
    )
    LOGGER.info(
        "Fitted nested SBM in %s | run dir: %s",
        _fmt_duration((pd.Timestamp.utcnow() - t0).total_seconds()),
        manifest["run_dir"],
    )
    LOGGER.debug("Fit manifest summary | %s", manifest)
    return manifest




def generate_command(args: argparse.Namespace) -> list[dict]:
    LOGGER.debug("Starting generate command | args=%s", vars(args))
    manifest = load_manifest(Path(args.run_dir))
    _validate_weight_generation_configuration(args, has_weight_data=bool(manifest.get("weight_model")))
    gt = _require_graph_tool()
    graph = gt.load_graph(manifest["graph_path"])
    nested_state = load_state(Path(manifest["state_path"]))
    layer_map = {int(ts): int(lid) for ts, lid in load_json(Path(manifest["layer_map_path"])).items()}
    LOGGER.debug(
        "Loaded generation artifacts | run_dir=%s | graph_path=%s | state_path=%s | layers=%s",
        manifest["run_dir"],
        manifest["graph_path"],
        manifest["state_path"],
        len(layer_map),
    )

    weight_model = manifest.get("weight_model")
    weight_generator_model = None
    temporal_generator_model = None
    observed_edges = None
    weight_partition_policy = str(getattr(args, "weight_parametric_partition_policy", "fixed")).strip().lower()
    pure_generative_weights = _pure_generative_weight_mode(args)

    if manifest.get("weight_generator_path"):
        weight_generator_model = load_json(Path(manifest["weight_generator_path"]))
        LOGGER.debug(
            "Loaded saved parametric weight generator | path=%s | summary=%s",
            manifest["weight_generator_path"],
            None if weight_generator_model is None else weight_generator_model.get("summary"),
        )

    if manifest.get("temporal_generator_path"):
        temporal_generator_model = load_json(Path(manifest["temporal_generator_path"]))
        LOGGER.debug(
            "Loaded saved temporal generator model | path=%s | summary=%s",
            manifest["temporal_generator_path"],
            None if temporal_generator_model is None else temporal_generator_model.get("summary"),
        )

    if pure_generative_weights and weight_model and weight_generator_model is None:
        raise ValueError(
            "weight_pure_generative=True requires a saved parametric weight generator in the fitted run artifacts."
        )

    temporal_generator_enabled = _temporal_generator_enabled(args)
    posterior_partition_sweeps = max(0, int(getattr(args, "posterior_partition_sweeps", 0)))
    temporal_requires_observed_edges = bool(
        temporal_generator_enabled
        and (
            temporal_generator_model is None
            or posterior_partition_sweeps > 0
        )
    )
    weight_requires_observed_edges = bool(
        weight_model
        and not pure_generative_weights
        and (weight_generator_model is None or weight_partition_policy == "refit_on_refresh")
    )
    needs_observed_edges = bool(temporal_requires_observed_edges or weight_requires_observed_edges)
    if needs_observed_edges:
        filtered_input_edges_path = manifest.get("filtered_input_edges_path")
        if not filtered_input_edges_path:
            if temporal_requires_observed_edges and posterior_partition_sweeps > 0:
                raise ValueError(
                    "Generation with posterior_partition_sweeps > 0 requires the filtered observed edge panel. "
                    "The saved temporal generator model is tied to the fitted partition and is not used on the "
                    "posterior-refresh research branch."
                )
            if temporal_requires_observed_edges:
                raise ValueError(
                    "Generation requested the filtered observed edge panel for temporal target fitting, but the run "
                    "manifest does not expose filtered_input_edges_path."
                )
            raise ValueError(
                "Generation requested access to the filtered observed edge panel for weight generation, but the run "
                "manifest does not expose filtered_input_edges_path."
            )
        observed_edges = pd.read_csv(filtered_input_edges_path)
        _log_edge_frame_debug(
            "Observed edge frame for generation",
            observed_edges,
            directed=bool(manifest["directed"]),
            weight_col=(
                (weight_model.get("output_column") or weight_model.get("input_column"))
                if isinstance(weight_model, dict)
                else None
            ),
        )

    generated_root = Path(manifest["run_dir"]) / "generated"
    if getattr(args, "output_subdir", None):
        generated_root = generated_root / str(args.output_subdir)
    generated_root.mkdir(parents=True, exist_ok=True)

    sample_records = []
    for sample_index in range(int(args.num_samples)):
        sample_dir = generated_root / f"sample_{sample_index:04d}"
        LOGGER.debug("Generating sample | sample_index=%s | sample_dir=%s", sample_index, sample_dir)
        sample_manifest = sample_synthetic_panel(
            graph=graph,
            nested_state=nested_state,
            layer_map=layer_map,
            output_dir=sample_dir,
            directed=bool(manifest["directed"]),
            seed=int(args.seed) + sample_index,
            args=args,
            observed_edges=observed_edges,
            weight_model=weight_model,
            weight_generator_model=weight_generator_model,
            temporal_generator_model=temporal_generator_model,
        )
        sample_manifest["sample_index"] = int(sample_index)
        sample_manifest["sample_label"] = f"sample_{sample_index:04d}"
        save_json(sample_manifest, Path(sample_manifest["sample_manifest_path"]))
        sample_records.append(sample_manifest)

    setting_manifest_path = generated_root / "setting_manifest.json"
    setting_manifest = {
        "setting_label": _generation_setting_label(args),
        "setting_dir": str(generated_root),
        "setting_manifest_path": str(setting_manifest_path),
        "num_samples_requested": int(args.num_samples),
        "generation_args": _serialise_generation_args(args),
        "sample_manifest_paths": [str(record["sample_manifest_path"]) for record in sample_records],
        "sample_labels": [str(record["sample_label"]) for record in sample_records],
        "sample_indices": [int(record["sample_index"]) for record in sample_records],
    }
    save_json(setting_manifest, setting_manifest_path)

    manifest["generated_samples"] = _merge_generated_sample_records(
        list(manifest.get("generated_samples", [])),
        sample_records,
    )
    manifest["generated_settings"] = _merge_generated_setting_records(
        list(manifest.get("generated_settings", [])),
        setting_manifest,
    )
    save_json(manifest, Path(manifest["manifest_path"]))
    LOGGER.info("Generated %s synthetic sample(s) under %s", len(sample_records), generated_root)
    LOGGER.debug("Generated sample manifests | %s", sample_records)
    return sample_records
