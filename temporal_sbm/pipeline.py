"""Core NetForge data preparation, fitting, and sampling code."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
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

    used_original_ids = np.sort(np.unique(edge_frame[["u", "i"]].to_numpy().ravel()).astype(np.int64))
    original_to_compact = {int(node_id): index for index, node_id in enumerate(used_original_ids.tolist())}
    LOGGER.debug(
        "Compact node mapping | unique_nodes=%s | min_node_id=%s | max_node_id=%s",
        len(used_original_ids),
        int(used_original_ids.min()) if len(used_original_ids) else None,
        int(used_original_ids.max()) if len(used_original_ids) else None,
    )

    compact_edges = edge_frame.copy()
    compact_edges["u"] = compact_edges["u"].map(original_to_compact).astype(np.int64)
    compact_edges["i"] = compact_edges["i"].map(original_to_compact).astype(np.int64)
    compact_edges = add_calendar_columns(
        compact_edges,
        ts_col="ts",
        tz=args.tz,
        ts_unit=args.ts_unit,
        ts_format=args.ts_format,
        holiday_country=args.holiday_country,
    )

    compact_features = np.zeros((len(used_original_ids) + 1, node_features.shape[1]), dtype=node_features.dtype)
    compact_features[1:] = node_features[1:][used_original_ids]

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
    is_region = node_map["type"].astype(str).str.lower().eq("region")
    node_type_by_compact[node_map.loc[is_region, "compact_id"].astype(int).to_numpy()] = 1
    LOGGER.debug(
        "Loaded node types | mapped_rows=%s | region_nodes=%s",
        len(node_map),
        int(is_region.sum()),
    )

    LOGGER.info(
        "Prepared data in %s | edges=%s | unique nodes=%s | layers=%s",
        _fmt_duration((pd.Timestamp.utcnow() - t0).total_seconds()),
        len(edge_frame),
        len(used_original_ids),
        len(layer_map),
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
        weight_column=getattr(args, "weight_col", None),
        duplicate_edge_count=duplicate_edge_count,
        self_loop_count=self_loop_count,
    )


def _extract_node_scalars(prepared: PreparedData) -> dict[str, np.ndarray]:
    features = prepared.node_features[1:]
    columns = prepared.node_feature_columns

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

    cx = features[:, prepared.centroid_x_index].astype(float)
    cy = features[:, prepared.centroid_y_index].astype(float)

    return {
        "cx": cx,
        "cy": cy,
        "num_farms": num_farms,
        "total_animals": total_animals,
        "ft_matrix": ft_matrix,
        "ft_norm": ft_norm,
    }


def build_layered_graph(prepared: PreparedData, directed: bool) -> Any:
    gt = _require_graph_tool()

    scalars = _extract_node_scalars(prepared)
    frame = prepared.compact_edges
    LOGGER.debug(
        "Building layered graph | directed=%s | compact_nodes=%s | compact_edges=%s | node_type_labels=%s | ft_dimensions=%s",
        directed,
        len(prepared.compact_to_original),
        len(frame),
        prepared.node_type_by_compact is not None,
        scalars["ft_matrix"].shape[1],
    )
    graph = gt.Graph(directed=directed)
    graph.add_vertex(len(prepared.compact_to_original))

    vertex_props = {
        "node_id": graph.new_vp("int"),
        "cx": graph.new_vp("double"),
        "cy": graph.new_vp("double"),
        "num_farms": graph.new_vp("double"),
        "total_animals": graph.new_vp("double"),
    }
    if prepared.node_type_by_compact is not None:
        vertex_props["type"] = graph.new_vp("int")

    for index, node_id in enumerate(prepared.compact_to_original.tolist()):
        vertex = graph.vertex(index)
        vertex_props["node_id"][vertex] = int(node_id)
        vertex_props["cx"][vertex] = float(scalars["cx"][index])
        vertex_props["cy"][vertex] = float(scalars["cy"][index])
        vertex_props["num_farms"][vertex] = float(scalars["num_farms"][index])
        vertex_props["total_animals"][vertex] = float(scalars["total_animals"][index])
        if "type" in vertex_props:
            vertex_props["type"][vertex] = int(prepared.node_type_by_compact[index])

    for name, prop in vertex_props.items():
        graph.vp[name] = prop

    edge_layer = graph.new_ep("int")
    edge_dist = graph.new_ep("double")
    edge_mass = graph.new_ep("double")
    edge_anim = graph.new_ep("double")
    edge_ftcos = graph.new_ep("double")
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

    graph.ep["layer"] = edge_layer
    graph.ep["dist_km"] = edge_dist
    graph.ep["mass_grav"] = edge_mass
    graph.ep["anim_grav"] = edge_anim
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
    graph.gp["num_layers"] = graph.new_gp("int", len(prepared.layer_map))

    LOGGER.info(
        "Built layered graph | vertices=%s | edges=%s | layers=%s",
        graph.num_vertices(),
        graph.num_edges(),
        len(prepared.layer_map),
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


def fit_nested_sbm(
    graph: Any,
    covariate_specs: Iterable[CovariateSpec],
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
    if not covariate_specs:
        raise ValueError("No valid covariates remain for fitting.")
    _log_covariate_specs("Starting nested SBM fit", covariate_specs)
    LOGGER.debug(
        "Fit options | deg_corr=%s | overlap=%s | fit_verbose=%s | refine_multiflip_rounds=%s | refine_multiflip_niter=%s | anneal_niter=%s | anneal_beta_start=%s | anneal_beta_stop=%s",
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
    clabel = graph.vp["type"] if "type" in graph.vp else None

    state = gt.minimize_nested_blockmodel_dl(
        graph,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            hentropy_args=dict(multigraph=False),
            state_args=dict(
                ec=graph.ep["layer"],
                layers=True,
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
        blocks = np.asarray(base.get_nonoverlap_blocks().a, dtype=np.int64)
        entropy = float(nested_state.entropy()) if hasattr(nested_state, "entropy") else float("nan")
        return (
            f"levels={levels} | blocks={int(np.unique(blocks).size)} | "
            f"vertices={int(base.g.num_vertices())} | edges={int(base.g.num_edges())} | entropy={entropy:.6f}"
        )
    except Exception as exc:
        return f"state_summary_unavailable ({exc})"


def attach_partition_maps(graph: Any, nested_state: Any) -> None:
    base = _base_state(nested_state)
    try:
        graph.vp["sbm_b"] = graph.own_property(base.get_nonoverlap_blocks().copy())
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
    mapping: dict[int, int] = {}
    for vertex in graph.vertices():
        mapping[int(node_id_prop[vertex])] = int(block_prop[vertex])
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
    rewire_model = str(getattr(args, "rewire_model", "none")).replace("-", "_")
    return f"{_generation_sample_mode_label(args)}__rewire_{rewire_model}"


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
) -> dict:
    gt = _require_graph_tool()
    gt.seed_rng(int(seed))
    LOGGER.debug(
        "Sampling synthetic panel | output_dir=%s | directed=%s | seed=%s | layer_count=%s | weight_model=%s",
        output_dir,
        directed,
        seed,
        len(layer_map),
        weight_model,
    )

    sampled_state = _posterior_partition_state(nested_state, seed=seed, args=args)
    base = _base_state(sampled_state)
    lid_to_state = map_graph_lid_to_state_lid(sampled_state)
    LOGGER.debug("Sampled state ready | %s", _state_summary_text(sampled_state))
    node_id_prop = base.g.vp["node_id"] if "node_id" in base.g.vp else None
    if node_id_prop is None:
        raise RuntimeError("Fitted graph is missing the vertex property 'node_id'.")

    blocks = np.asarray(base.get_nonoverlap_blocks().a, dtype=np.int64)
    partition_records = [
        {
            "node_id": int(node_id_prop[base.g.vertex(index)]),
            "block_id": int(blocks[index]),
        }
        for index in range(int(base.g.num_vertices()))
    ]
    partition_frame = pd.DataFrame(partition_records).sort_values(["node_id", "block_id"]).reset_index(drop=True)
    partition_path = Path(output_dir) / "sample_node_partition.csv"
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_frame.to_csv(partition_path, index=False)

    weight_col = None
    weight_sampler = None
    type_prop = base.g.vp["type"] if "type" in base.g.vp else None
    if weight_model and observed_edges is not None:
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
    if weight_sampler is not None:
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
        },
    }
    if weight_col and weight_col in panel_frame.columns:
        payload["weight_column"] = weight_col
        payload["weight_total"] = float(panel_frame[weight_col].sum()) if len(panel_frame) else 0.0
    if rewire_summaries:
        payload["rewire_summaries"] = rewire_summaries
    save_json(payload, sample_manifest_path)
    return payload


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
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Writing fit artifacts under %s", output_dir)

    graph_path = output_dir / "layered_model.gt"
    state_path = output_dir / "nested_state.pkl.gz"
    layer_map_path = output_dir / "layer_map.json"
    filtered_edges_path = output_dir / "input_edges_filtered.csv"
    node_attributes_path = output_dir / "node_attributes.csv"

    graph.save(str(graph_path))
    save_state(nested_state, state_path)
    save_json({str(ts): int(lid) for ts, lid in prepared.layer_map.items()}, layer_map_path)
    prepared.original_edges.to_csv(filtered_edges_path, index=False)
    write_node_attributes(prepared, node_attributes_path, node_blocks=extract_node_block_map(graph))

    input_summary = {
        "edge_count": int(len(prepared.original_edges)),
        "node_count": int(len(prepared.compact_to_original)),
        "layer_count": int(len(prepared.layer_map)),
        "duplicate_edge_count": int(prepared.duplicate_edge_count),
        "self_loop_count": int(prepared.self_loop_count),
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
        "fit_options": {
            "deg_corr": not args.no_deg_corr,
            "overlap": bool(args.overlap),
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
        "Fit artifacts written | graph_path=%s | state_path=%s | layer_map_path=%s | filtered_edges_path=%s | node_attributes_path=%s | manifest_path=%s",
        graph_path,
        state_path,
        layer_map_path,
        filtered_edges_path,
        node_attributes_path,
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
    graph = build_layered_graph(prepared, directed=bool(args.directed))
    base_covariate_specs = _select_covariate_specs(_available_covariate_specs(graph), getattr(args, "fit_covariates", None))
    weight_candidates = _build_weight_candidates(prepared, graph, args)
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
    attach_partition_maps(graph, nested_state)

    manifest = write_fit_artifacts(
        prepared,
        graph,
        nested_state,
        Path(args.output_dir).expanduser().resolve(),
        args,
        fit_covariates=fit_covariates,
        weight_model=weight_model,
    )
    LOGGER.info(
        "Fitted layered nested SBM in %s | run dir: %s",
        _fmt_duration((pd.Timestamp.utcnow() - t0).total_seconds()),
        manifest["run_dir"],
    )
    LOGGER.debug("Fit manifest summary | %s", manifest)
    return manifest


def generate_command(args: argparse.Namespace) -> list[dict]:
    LOGGER.debug("Starting generate command | args=%s", vars(args))
    manifest = load_manifest(Path(args.run_dir))
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
    observed_edges = None
    if weight_model:
        observed_edges = pd.read_csv(manifest["filtered_input_edges_path"])
        _log_edge_frame_debug(
            "Observed edge frame for weighted generation",
            observed_edges,
            directed=bool(manifest["directed"]),
            weight_col=weight_model.get("output_column") or weight_model.get("input_column"),
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
