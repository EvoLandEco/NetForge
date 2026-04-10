"""Diagnostics for observed and synthetic temporal panels."""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import holidays as pyhol
except Exception:
    pyhol = None


LOGGER = logging.getLogger(__name__)

try:
    NL_HOLIDAYS = pyhol.Netherlands() if pyhol is not None else None
except Exception:
    NL_HOLIDAYS = None

STANDARD_EDGE_COLUMNS = ("u", "i", "ts")
LOG_LINE_PATTERN = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[A-Z]+) \| (?P<message>.*)$")
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
}
ADVANCED_TEMPORAL_COLORS = {
    "persist": "#90a4b4",
    "reactivated": "#b7d4dc",
    "new": "#f28e2b",
    "churn": "#e15759",
}
ADVANCED_TEMPORAL_THEMES = {
    "tea": {
        "persist": "#9aa5b1",
        "reactivated": "#cbd5e1",
        "new": "#f28e2b",
        "churn": "#f87171",
    },
    "tna": {
        "persist": "#9aa5b1",
        "reactivated": "#cbd5e1",
        "new": "#2a9d8f",
        "churn": "#f87171",
    },
}
PI_MASS_TYPE_COLORS = {
    "Farm": "#35c9c3",
    "Region": "#ef8f7d",
}
MAGNETIC_EIGENVALUE_COUNT = 12
MAGNETIC_CHARGE = 0.25
LAZY_WALK_ALPHA = 0.10
PAGERANK_TELEPORT_ALPHA = 0.15
MAX_TIME_TICK_LABELS = 14
TIME_SERIES_CORRELATION_METHOD = "spearman_rank"
TIME_SERIES_CORRELATION_LABEL = "Spearman rank correlation"
SAMPLE_SUFFIX_PATTERN = re.compile(r"__sample_(?P<index>\d{4})$")
POSTERIOR_DETAIL_GROUP_KEYS: dict[str, list[str]] = {
    "block_pair_per_snapshot": ["ts", "block_u", "block_v"],
    "block_pair_summary": ["block_u", "block_v"],
    "block_activity_per_snapshot": ["ts", "block_id"],
    "block_activity_summary": ["block_id"],
    "node_activity_per_snapshot": ["ts", "node_id"],
    "node_activity_summary": ["node_id", "block_id"],
    "edge_type_per_snapshot": ["ts", "source_type", "target_type"],
    "edge_type_summary": ["source_type", "target_type"],
    "tea_per_snapshot": ["ts"],
    "tea_summary": ["metric"],
    "tea_type_pair_per_snapshot": ["ts", "source_type", "target_type"],
    "tea_type_pair_summary": ["source_type", "target_type"],
    "tna_per_snapshot": ["ts"],
    "tna_summary": ["metric"],
    "tna_type_per_snapshot": ["ts", "type_label"],
    "tna_type_summary": ["type_label"],
    "pi_mass_per_snapshot": ["ts"],
    "pi_mass_summary": ["metric"],
    "pi_mass_closed_per_snapshot": ["ts"],
    "pi_mass_closed_summary": ["metric"],
    "pi_mass_pagerank_per_snapshot": ["ts"],
    "pi_mass_pagerank_summary": ["metric"],
    "temporal_reachability_per_snapshot": ["ts"],
    "temporal_reachability_summary": ["metric"],
    "temporal_reachability_source_summary": ["node_id"],
    "magnetic_laplacian_per_snapshot": ["ts"],
    "magnetic_laplacian_summary": ["metric"],
    "magnetic_spectral_distance_per_snapshot": ["ts"],
    "magnetic_spectral_distance_summary": ["metric"],
}


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


def _posterior_run_display_label(sample_label: str) -> str:
    run_index = _sample_index_from_label(sample_label)
    if run_index is not None:
        return f"Run {run_index + 1}"
    return str(sample_label)


def _posterior_run_count(payload: dict[str, Any] | pd.Series | None) -> int:
    if payload is None:
        return 1
    try:
        value = payload.get("posterior_num_runs", 1)  # type: ignore[union-attr]
    except Exception:
        value = 1
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _posterior_interval_from_mapping(payload: dict[str, Any] | pd.Series, key: str) -> tuple[Optional[float], Optional[float]]:
    q05_key = f"{key}_q05"
    q95_key = f"{key}_q95"
    try:
        lower = payload.get(q05_key)  # type: ignore[union-attr]
        upper = payload.get(q95_key)  # type: ignore[union-attr]
    except Exception:
        return None, None
    lower_value = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
    upper_value = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
    return (
        None if pd.isna(lower_value) else float(lower_value),
        None if pd.isna(upper_value) else float(upper_value),
    )


def _posterior_interval_from_frame(frame: pd.DataFrame, key: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    lower_column = f"{key}_q05"
    upper_column = f"{key}_q95"
    if lower_column not in frame.columns or upper_column not in frame.columns:
        return None, None
    lower = pd.to_numeric(frame[lower_column], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame[upper_column], errors="coerce").to_numpy(dtype=float)
    return lower, upper


def _posterior_band_label(label: str, run_count: int) -> str:
    return f"{label} median ({run_count} draws)" if run_count > 1 else label


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
    for text in legend.get_texts():
        text.set_color(PLOT_COLORS["text"])
    if legend.get_title() is not None:
        legend.get_title().set_color(PLOT_COLORS["text"])


def _plot_line_with_band(
    ax,
    *,
    ts_values: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    linewidth: float,
    marker_size: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    band_alpha: float = 0.14,
    zorder: int = 3,
) -> None:
    value_array = np.asarray(values, dtype=float)
    lower_array = None if lower is None else np.asarray(lower, dtype=float)
    upper_array = None if upper is None else np.asarray(upper, dtype=float)
    if lower_array is not None and upper_array is not None:
        finite_mask = np.isfinite(lower_array) & np.isfinite(upper_array)
        if finite_mask.any():
            ax.fill_between(
                ts_values[finite_mask],
                lower_array[finite_mask],
                upper_array[finite_mask],
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=max(1, zorder - 2),
            )
    ax.plot(
        ts_values,
        value_array,
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=marker_size,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def _set_timestamp_ticks(
    ax,
    ts_values: np.ndarray,
    *,
    show_calendar_bands: bool = True,
) -> None:
    records = _calendar_records(ts_values)
    if not records:
        return

    if show_calendar_bands:
        _add_calendar_bands(ax, ts_values)
    tick_values = np.asarray([record["ts"] for record in records], dtype=float)
    if len(tick_values) > MAX_TIME_TICK_LABELS:
        index_values = np.linspace(0, len(tick_values) - 1, num=MAX_TIME_TICK_LABELS)
        tick_values = tick_values[np.unique(np.round(index_values).astype(int))]
    tick_labels = [_calendar_label_for_ts(int(value), records) for value in tick_values]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")


def _ts_to_date(ts_value: int | float) -> date:
    return date.fromordinal(int(round(float(ts_value))))


def _calendar_category(ts_value: int | float) -> str:
    current_date = _ts_to_date(ts_value)
    if NL_HOLIDAYS is not None and current_date in NL_HOLIDAYS:
        return "holiday"
    return "weekend" if current_date.weekday() >= 5 else "weekday"


def _calendar_records(ts_values: Iterable[float]) -> list[dict[str, object]]:
    values = np.asarray(list(ts_values), dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return []

    unique_ts = sorted({int(round(float(value))) for value in values})
    records: list[dict[str, object]] = []
    for ts_value in unique_ts:
        current_date = _ts_to_date(ts_value)
        records.append(
            {
                "ts": int(ts_value),
                "date": current_date.isoformat(),
                "offset": int(ts_value - unique_ts[0]),
                "category": _calendar_category(ts_value),
                "label": current_date.isoformat(),
            }
        )
    return records


def _calendar_label_for_ts(ts_value: int, records: list[dict[str, object]]) -> str:
    for record in records:
        if int(record["ts"]) == int(ts_value):
            return str(record["label"])
    current_date = _ts_to_date(ts_value)
    return current_date.isoformat()


def _add_calendar_bands(ax, ts_values: Iterable[float]) -> None:
    xlim = ax.get_xlim()
    should_restore_xlim = np.all(np.isfinite(xlim))
    for record in _calendar_records(ts_values):
        category = str(record["category"])
        if category == "holiday":
            ax.axvspan(int(record["ts"]) - 0.5, int(record["ts"]) + 0.5, facecolor="#fde2e2", alpha=0.65, zorder=0, linewidth=0)
        elif category == "weekend":
            ax.axvspan(int(record["ts"]) - 0.5, int(record["ts"]) + 0.5, facecolor="#e6f0ff", alpha=0.45, zorder=0, linewidth=0)
    if should_restore_xlim:
        ax.set_xlim(*xlim)


def _save_figure(fig, output_path: Path, *, dpi: int = 180) -> None:
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")


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
        LOGGER.debug(
            "%s | weight_col=%s | weight_summary=%s",
            label,
            weight_col,
            _format_numeric_summary(frame[weight_col].to_numpy(dtype=float, copy=False)),
        )


def canonicalise_edge_frame(
    df: pd.DataFrame,
    directed: bool,
    src_col: str = "u",
    dst_col: str = "i",
    ts_col: str = "ts",
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    LOGGER.debug(
        "Canonicalising edge frame | directed=%s | src_col=%s | dst_col=%s | ts_col=%s | weight_col=%s | input_rows=%s",
        directed,
        src_col,
        dst_col,
        ts_col,
        weight_col,
        len(df),
    )
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
        _log_edge_frame_debug("Canonicalised weighted edge frame", frame, directed=directed, weight_col=weight_col)
        return frame
    frame = frame.drop_duplicates(STANDARD_EDGE_COLUMNS).reset_index(drop=True)
    _log_edge_frame_debug("Canonicalised edge frame", frame, directed=directed, weight_col=None)
    return frame


def load_node_coordinates(path: Optional[Path]) -> Optional[Dict[int, Tuple[float, float]]]:
    if path is None:
        LOGGER.debug("No node coordinate path provided.")
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        LOGGER.debug("Node coordinate file does not exist: %s", csv_path)
        return None

    df = pd.read_csv(csv_path)
    required = {"node_id", "x", "y"}
    if not required.issubset(df.columns):
        LOGGER.debug("Node coordinate file missing required columns | path=%s | columns=%s", csv_path, df.columns.tolist())
        return None

    out: Dict[int, Tuple[float, float]] = {}
    for row in df.itertuples(index=False):
        try:
            out[int(row.node_id)] = (float(row.x), float(row.y))
        except (TypeError, ValueError):
            continue
    LOGGER.debug("Loaded node coordinates | path=%s | node_count=%s", csv_path, len(out))
    return out or None


def load_node_blocks(path: Optional[Path]) -> Optional[Dict[int, int]]:
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"node_id", "block_id"}
    if not required.issubset(df.columns):
        LOGGER.debug("Node attribute file missing block_id | path=%s | columns=%s", csv_path, df.columns.tolist())
        return None

    out: Dict[int, int] = {}
    for row in df.itertuples(index=False):
        try:
            node_id = int(row.node_id)
            block_id = int(row.block_id)
        except (TypeError, ValueError):
            continue
        if block_id >= 0:
            out[node_id] = block_id
    LOGGER.debug("Loaded node blocks | path=%s | node_count=%s", csv_path, len(out))
    return out or None


def resolve_node_blocks(
    node_blocks: Optional[Dict[int, int]] = None,
    *,
    node_blocks_path: Optional[Path] = None,
    sample_manifest: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[int, int]]:
    if node_blocks:
        return node_blocks
    if sample_manifest:
        candidate = sample_manifest.get("node_partition_path") or sample_manifest.get("partition_path")
        if candidate:
            loaded = load_node_blocks(Path(str(candidate)))
            if loaded:
                return loaded
    if node_blocks_path is not None:
        loaded = load_node_blocks(Path(node_blocks_path))
        if loaded:
            return loaded
    return None


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
        if lowered in {"region", "regions"}:
            return "Region"
        if lowered in {"f", "farm node"}:
            return "Farm"
        if lowered in {"r", "region node", "supernode"}:
            return "Region"
        if lowered in {"type 0", "type0"}:
            return "Farm"
        if lowered in {"type 1", "type1"}:
            return "Region"
        if re.fullmatch(r"type\s*[01]", lowered):
            return "Farm" if lowered.endswith("0") else "Region"
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


def load_node_types(path: Optional[Path]) -> Optional[Dict[int, str]]:
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"node_id", "type"}
    if not required.issubset(df.columns):
        LOGGER.debug("Node attribute file missing type column | path=%s | columns=%s", csv_path, df.columns.tolist())
        return None

    out: Dict[int, str] = {}
    for row in df.itertuples(index=False):
        try:
            node_id = int(row.node_id)
        except (TypeError, ValueError):
            continue
        out[node_id] = _format_node_type_label(getattr(row, "type"))
    LOGGER.debug(
        "Loaded node types | path=%s | node_count=%s | type_values=%s",
        csv_path,
        len(out),
        sorted(set(out.values())),
    )
    return out or None


def _sorted_type_labels(node_types: Optional[Dict[int, str]]) -> list[str]:
    if not node_types:
        return []
    return sorted(set(node_types.values()), key=lambda value: (value == "Unknown", value))


def _safe_metric_key(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    return text or "unknown"


def _type_short_label(label: object) -> str:
    text = _format_node_type_label(label)
    lowered = text.lower()
    if lowered == "farm":
        return "F"
    if lowered == "region":
        return "R"
    if lowered == "unknown":
        return "U"
    return text


def _display_type_pair_label(source_type: object, target_type: object, *, directed: bool = True) -> str:
    arrow = "→" if directed else "–"
    return f"{_type_short_label(source_type)}{arrow}{_type_short_label(target_type)}"


def _snapshot_edge_sets(df: pd.DataFrame) -> dict[int, set[Tuple[int, int]]]:
    if df.empty:
        return {}
    edge_sets: dict[int, set[Tuple[int, int]]] = {}
    for ts_value, snapshot in df.groupby("ts", sort=True):
        edge_sets[int(ts_value)] = set(snapshot[["u", "i"]].itertuples(index=False, name=None))
    return edge_sets


def _snapshot_node_sets(df: pd.DataFrame) -> dict[int, set[int]]:
    if df.empty:
        return {}
    node_sets: dict[int, set[int]] = {}
    for ts_value, snapshot in df.groupby("ts", sort=True):
        node_sets[int(ts_value)] = _node_set(snapshot)
    return node_sets


def _summarise_transition_counts(snapshot_sets: dict[int, set[Any]]) -> pd.DataFrame:
    columns = [
        "ts",
        "new_count",
        "reactivated_count",
        "persist_count",
        "ceased_prev_count",
        "repeated_count",
        "total_count",
        "new_ratio",
        "persist_ratio",
        "reactivated_ratio",
        "churn_ratio",
    ]
    if not snapshot_sets:
        return pd.DataFrame(columns=columns)

    seen: set[Any] = set()
    prev: set[Any] = set()
    rows: list[dict[str, float]] = []
    for ts_value in sorted(snapshot_sets):
        current = snapshot_sets[ts_value]
        persist_count = int(len(current & prev))
        new_count = int(sum(item not in seen for item in current))
        reactivated_count = int(len(current) - persist_count - new_count)
        ceased_prev_count = int(len(prev - current))
        total_count = int(len(current))
        rows.append(
            {
                "ts": int(ts_value),
                "new_count": new_count,
                "reactivated_count": reactivated_count,
                "persist_count": persist_count,
                "ceased_prev_count": ceased_prev_count,
                "repeated_count": persist_count + reactivated_count,
                "total_count": total_count,
                "new_ratio": float(new_count / total_count) if total_count else 0.0,
                "persist_ratio": float(persist_count / total_count) if total_count else 0.0,
                "reactivated_ratio": float(reactivated_count / total_count) if total_count else 0.0,
                "churn_ratio": float(ceased_prev_count / len(prev)) if prev else 0.0,
            }
        )
        seen |= current
        prev = current
    return pd.DataFrame(rows, columns=columns)


def _compute_tea_counts(df: pd.DataFrame) -> pd.DataFrame:
    return _summarise_transition_counts(_snapshot_edge_sets(df))


def _compute_tna_counts(df: pd.DataFrame) -> pd.DataFrame:
    return _summarise_transition_counts(_snapshot_node_sets(df))


def _type_pair_denominator(
    source_count: int,
    target_count: int,
    *,
    same_type: bool,
    directed: bool,
) -> int:
    if directed:
        if same_type:
            return max(source_count * max(source_count - 1, 0), 0)
        return max(source_count * target_count, 0)
    if same_type:
        return max((source_count * max(source_count - 1, 0)) // 2, 0)
    return max(source_count * target_count, 0)


def _compute_tea_type_pair_time_series(
    df: pd.DataFrame,
    *,
    node_types: Optional[Dict[int, str]],
    directed: bool,
) -> pd.DataFrame:
    columns = [
        "ts",
        "source_type",
        "target_type",
        "new_count",
        "reactivated_count",
        "persist_count",
        "ceased_prev_count",
        "repeated_count",
        "total_count",
        "new_ratio",
        "persist_ratio",
        "reactivated_ratio",
        "churn_ratio",
    ]
    if df.empty or not node_types:
        return pd.DataFrame(columns=columns)

    type_labels = _sorted_type_labels(node_types)
    if not type_labels:
        return pd.DataFrame(columns=columns)

    snapshot_sets: dict[int, dict[tuple[str, str], set[Tuple[int, int]]]] = {}
    for ts_value, snapshot in df.groupby("ts", sort=True):
        current: dict[tuple[str, str], set[Tuple[int, int]]] = defaultdict(set)
        for row in snapshot.itertuples(index=False):
            source_type = node_types.get(int(row.u), "Unknown")
            target_type = node_types.get(int(row.i), "Unknown")
            if not directed and source_type > target_type:
                source_type, target_type = target_type, source_type
            current[(source_type, target_type)].add((int(row.u), int(row.i)))
        snapshot_sets[int(ts_value)] = current

    group_keys: list[tuple[str, str]] = []
    for source_type in type_labels:
        target_iterable = type_labels if directed else [label for label in type_labels if label >= source_type]
        for target_type in target_iterable:
            group_keys.append((source_type, target_type))

    seen: dict[tuple[str, str], set[Tuple[int, int]]] = {key: set() for key in group_keys}
    previous: dict[tuple[str, str], set[Tuple[int, int]]] = {key: set() for key in group_keys}
    rows: list[dict[str, object]] = []
    for ts_value in sorted(snapshot_sets):
        current_lookup = snapshot_sets[ts_value]
        for source_type, target_type in group_keys:
            key = (source_type, target_type)
            current_edges = current_lookup.get(key, set())
            previous_edges = previous.get(key, set())
            seen_edges = seen.get(key, set())
            persist_count = int(len(current_edges & previous_edges))
            new_count = int(sum(edge not in seen_edges for edge in current_edges))
            reactivated_count = int(len(current_edges) - persist_count - new_count)
            ceased_prev_count = int(len(previous_edges - current_edges))
            total_count = int(len(current_edges))
            rows.append(
                {
                    "ts": int(ts_value),
                    "source_type": source_type,
                    "target_type": target_type,
                    "new_count": new_count,
                    "reactivated_count": reactivated_count,
                    "persist_count": persist_count,
                    "ceased_prev_count": ceased_prev_count,
                    "repeated_count": persist_count + reactivated_count,
                    "total_count": total_count,
                    "new_ratio": float(new_count / total_count) if total_count else 0.0,
                    "persist_ratio": float(persist_count / total_count) if total_count else 0.0,
                    "reactivated_ratio": float(reactivated_count / total_count) if total_count else 0.0,
                    "churn_ratio": float(ceased_prev_count / len(previous_edges)) if previous_edges else 0.0,
                }
            )
            seen[key] = seen_edges | current_edges
            previous[key] = current_edges
    return pd.DataFrame(rows, columns=columns)


def _compute_tna_type_time_series(
    df: pd.DataFrame,
    *,
    node_types: Optional[Dict[int, str]],
) -> pd.DataFrame:
    columns = [
        "ts",
        "type_label",
        "new_count",
        "reactivated_count",
        "persist_count",
        "ceased_prev_count",
        "repeated_count",
        "total_count",
        "new_ratio",
        "persist_ratio",
        "reactivated_ratio",
        "churn_ratio",
    ]
    if df.empty or not node_types:
        return pd.DataFrame(columns=columns)

    type_labels = _sorted_type_labels(node_types)
    if not type_labels:
        return pd.DataFrame(columns=columns)
    snapshot_sets = _snapshot_node_sets(df)
    seen: dict[str, set[int]] = {label: set() for label in type_labels}
    previous: dict[str, set[int]] = {label: set() for label in type_labels}
    rows: list[dict[str, object]] = []
    for ts_value in sorted(snapshot_sets):
        active_now = snapshot_sets[ts_value]
        current_by_type = {
            type_label: {node_id for node_id in active_now if node_types.get(node_id) == type_label}
            for type_label in type_labels
        }
        for type_label in type_labels:
            current_nodes = current_by_type.get(type_label, set())
            previous_nodes = previous.get(type_label, set())
            seen_nodes = seen.get(type_label, set())
            persist_count = int(len(current_nodes & previous_nodes))
            new_count = int(sum(node_id not in seen_nodes for node_id in current_nodes))
            reactivated_count = int(len(current_nodes) - persist_count - new_count)
            ceased_prev_count = int(len(previous_nodes - current_nodes))
            total_count = int(len(current_nodes))
            rows.append(
                {
                    "ts": int(ts_value),
                    "type_label": type_label,
                    "new_count": new_count,
                    "reactivated_count": reactivated_count,
                    "persist_count": persist_count,
                    "ceased_prev_count": ceased_prev_count,
                    "repeated_count": persist_count + reactivated_count,
                    "total_count": total_count,
                    "new_ratio": float(new_count / total_count) if total_count else 0.0,
                    "persist_ratio": float(persist_count / total_count) if total_count else 0.0,
                    "reactivated_ratio": float(reactivated_count / total_count) if total_count else 0.0,
                    "churn_ratio": float(ceased_prev_count / len(previous_nodes)) if previous_nodes else 0.0,
                }
            )
            seen[type_label] = seen_nodes | current_nodes
            previous[type_label] = current_nodes
    return pd.DataFrame(rows, columns=columns)


def _compute_edge_type_time_series(
    df: pd.DataFrame,
    *,
    node_types: Optional[Dict[int, str]],
    directed: bool,
    weight_col: Optional[str],
) -> pd.DataFrame:
    columns = [
        "ts",
        "source_type",
        "target_type",
        "edge_count",
        "edge_share",
        "weight_total",
        "weight_share",
    ]
    if df.empty or not node_types:
        return pd.DataFrame(columns=columns)

    type_labels = _sorted_type_labels(node_types)
    if not type_labels:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for ts_value, snapshot in df.groupby("ts", sort=True):
        working = snapshot.copy()
        working["source_type"] = working["u"].map(lambda node_id: node_types.get(int(node_id), "Unknown"))
        working["target_type"] = working["i"].map(lambda node_id: node_types.get(int(node_id), "Unknown"))
        if not directed:
            ordered = working[["source_type", "target_type"]].apply(lambda row: sorted(row.tolist()), axis=1, result_type="expand")
            working["source_type"] = ordered[0]
            working["target_type"] = ordered[1]

        if weight_col and weight_col in working.columns:
            grouped = working.groupby(["source_type", "target_type"], as_index=False, sort=False).agg(
                edge_count=("u", "size"),
                weight_total=(weight_col, "sum"),
            )
        else:
            grouped = working.groupby(["source_type", "target_type"], as_index=False, sort=False).agg(edge_count=("u", "size"))
            grouped["weight_total"] = np.nan

        total_edges = max(int(len(working)), 1)
        total_weight = float(pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0).sum()) if weight_col and weight_col in working.columns else np.nan
        pair_lookup = {
            (str(row.source_type), str(row.target_type)): {
                "edge_count": int(row.edge_count),
                "weight_total": float(row.weight_total) if pd.notna(row.weight_total) else np.nan,
            }
            for row in grouped.itertuples(index=False)
        }
        for source_type in type_labels:
            target_iterable = type_labels if directed else [label for label in type_labels if label >= source_type]
            for target_type in target_iterable:
                payload = pair_lookup.get((source_type, target_type), {})
                edge_count = int(payload.get("edge_count", 0))
                weight_total = float(payload.get("weight_total", 0.0)) if np.isfinite(payload.get("weight_total", 0.0)) else np.nan
                rows.append(
                    {
                        "ts": int(ts_value),
                        "source_type": source_type,
                        "target_type": target_type,
                        "edge_count": edge_count,
                        "edge_share": float(edge_count / total_edges),
                        "weight_total": weight_total,
                        "weight_share": float(weight_total / total_weight) if np.isfinite(total_weight) and total_weight > 0 and np.isfinite(weight_total) else np.nan,
                    }
                )

    return pd.DataFrame(rows, columns=columns)



def _build_snapshot_out_bitsets(
    df: pd.DataFrame,
    node_index: Dict[int, int],
    *,
    directed: bool,
    weight_col: Optional[str] = None,
) -> list[tuple[int, list[int]]]:
    if df.empty:
        return []

    node_count = int(len(node_index))
    snapshots: list[tuple[int, list[int]]] = []
    for ts_value, snapshot in df.groupby("ts", sort=True):
        out_bits = [0] * node_count
        if weight_col and weight_col in snapshot.columns:
            edge_rows = snapshot[["u", "i", weight_col]].itertuples(index=False, name=None)
        else:
            edge_rows = snapshot[["u", "i"]].itertuples(index=False, name=None)
        for edge in edge_rows:
            if len(edge) == 3:
                u_value, v_value, weight_value = edge
                try:
                    weight_numeric = float(weight_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(weight_numeric) or weight_numeric <= 0:
                    continue
            else:
                u_value, v_value = edge
            u_index = node_index[int(u_value)]
            v_index = node_index[int(v_value)]
            out_bits[u_index] |= 1 << v_index
            if not directed and u_index != v_index:
                out_bits[v_index] |= 1 << u_index
        snapshots.append((int(ts_value), out_bits))
    return snapshots


def _forward_reach_counts_from_target_sources(
    target_sources: list[int],
    node_count: int,
) -> np.ndarray:
    counts = np.zeros(node_count, dtype=np.int64)
    for source_bits in target_sources:
        remaining = int(source_bits)
        while remaining:
            least_significant = remaining & -remaining
            source_index = int(least_significant.bit_length() - 1)
            counts[source_index] += 1
            remaining ^= least_significant
    if node_count:
        counts -= 1
    return counts


def _static_reachability_counts(
    df: pd.DataFrame,
    *,
    node_universe: list[int],
    directed: bool,
    weight_col: Optional[str] = None,
) -> np.ndarray:
    node_count = int(len(node_universe))
    if node_count == 0:
        return np.zeros(0, dtype=np.int64)

    node_index = {node_id: index for index, node_id in enumerate(node_universe)}
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    if not df.empty:
        if weight_col and weight_col in df.columns:
            edge_rows = df[["u", "i", weight_col]].itertuples(index=False, name=None)
        else:
            edge_rows = df[["u", "i"]].itertuples(index=False, name=None)
        for edge in edge_rows:
            if len(edge) == 3:
                u_value, v_value, weight_value = edge
                try:
                    weight_numeric = float(weight_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(weight_numeric) or weight_numeric <= 0:
                    continue
            else:
                u_value, v_value = edge
            u_index = node_index[int(u_value)]
            v_index = node_index[int(v_value)]
            adjacency[u_index].append(v_index)
            if not directed and u_index != v_index:
                adjacency[v_index].append(u_index)

    counts = np.zeros(node_count, dtype=np.int64)
    for source_index in range(node_count):
        seen = np.zeros(node_count, dtype=bool)
        stack = [source_index]
        seen[source_index] = True
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        counts[source_index] = int(seen.sum()) - 1
    return counts


def _compute_temporal_reachability_diagnostics(
    df: pd.DataFrame,
    *,
    node_universe: list[int],
    directed: bool,
    weight_col: Optional[str] = None,
    node_types: Optional[Dict[int, str]] = None,
) -> dict[str, Any]:
    per_snapshot_columns = [
        "ts",
        "reachable_pair_count",
        "reachability_ratio",
        "new_reachable_pair_count",
        "temporal_efficiency",
        "mean_arrival_time_reached",
    ]
    source_columns = [
        "node_id",
        "forward_reach_count",
        "forward_reach_ratio",
        "static_forward_reach_count",
        "static_forward_reach_ratio",
    ]
    if node_types:
        source_columns.append("type_label")

    node_count = int(len(node_universe))
    if node_count == 0:
        return {
            "per_snapshot": pd.DataFrame(columns=per_snapshot_columns),
            "source_summary": pd.DataFrame(columns=source_columns),
            "global_summary": {
                "reachability_ratio": 0.0,
                "reachable_pair_count": 0,
                "temporal_efficiency": 0.0,
                "mean_arrival_time_reached": np.nan,
                "static_reachability_ratio": 0.0,
                "causal_fidelity": 1.0,
                "mean_forward_reach_ratio": 0.0,
                "p95_forward_reach_ratio": 0.0,
                "max_forward_reach_ratio": 0.0,
            },
        }

    node_index = {node_id: index for index, node_id in enumerate(node_universe)}
    snapshots = _build_snapshot_out_bitsets(df, node_index, directed=directed, weight_col=weight_col)
    ordered_pair_denominator = max(node_count * max(node_count - 1, 0), 1)
    static_counts = _static_reachability_counts(
        df,
        node_universe=node_universe,
        directed=directed,
        weight_col=weight_col,
    )
    static_reachability_ratio = float(static_counts.sum() / ordered_pair_denominator) if node_count > 1 else 0.0

    if not snapshots:
        source_summary = pd.DataFrame(
            {
                "node_id": node_universe,
                "forward_reach_count": np.zeros(node_count, dtype=np.int64),
                "forward_reach_ratio": np.zeros(node_count, dtype=float),
                "static_forward_reach_count": static_counts,
                "static_forward_reach_ratio": static_counts / max(node_count - 1, 1),
            }
        )
        if node_types:
            source_summary["type_label"] = source_summary["node_id"].map(lambda node_id: node_types.get(int(node_id), "Unknown"))
        source_summary = source_summary[source_columns]
        causal_fidelity = 1.0 if static_reachability_ratio <= 0 else 0.0
        return {
            "per_snapshot": pd.DataFrame(columns=per_snapshot_columns),
            "source_summary": source_summary,
            "global_summary": {
                "reachability_ratio": 0.0,
                "reachable_pair_count": 0,
                "temporal_efficiency": 0.0,
                "mean_arrival_time_reached": np.nan,
                "static_reachability_ratio": static_reachability_ratio,
                "causal_fidelity": float(causal_fidelity),
                "mean_forward_reach_ratio": 0.0,
                "p95_forward_reach_ratio": 0.0,
                "max_forward_reach_ratio": 0.0,
            },
        }

    first_ts = int(snapshots[0][0])
    target_sources = [1 << index for index in range(node_count)]
    arrival_time_weighted_sum = 0.0
    inverse_arrival_time_sum = 0.0
    rows: list[dict[str, float]] = []

    for ts_value, out_bits in snapshots:
        previous_sources = list(target_sources)
        updated_sources = list(previous_sources)
        new_reachable_pair_count = 0
        for u_index, neighbour_bits in enumerate(out_bits):
            if neighbour_bits == 0:
                continue
            source_bits = previous_sources[u_index]
            if source_bits == 0:
                continue
            remaining = int(neighbour_bits)
            while remaining:
                least_significant = remaining & -remaining
                v_index = int(least_significant.bit_length() - 1)
                before_bits = updated_sources[v_index]
                after_bits = before_bits | source_bits
                if after_bits != before_bits:
                    new_bits = after_bits ^ before_bits
                    new_reachable_pair_count += int(new_bits.bit_count())
                    updated_sources[v_index] = after_bits
                remaining ^= least_significant
        target_sources = updated_sources
        reachable_pair_count = int(sum(int(bits).bit_count() for bits in target_sources) - node_count)
        elapsed_days = int(ts_value - first_ts + 1)
        inverse_arrival_time_sum += float(new_reachable_pair_count / max(elapsed_days, 1))
        arrival_time_weighted_sum += float(new_reachable_pair_count * elapsed_days)
        rows.append(
            {
                "ts": int(ts_value),
                "reachable_pair_count": reachable_pair_count,
                "reachability_ratio": float(reachable_pair_count / ordered_pair_denominator) if node_count > 1 else 0.0,
                "new_reachable_pair_count": int(new_reachable_pair_count),
                "temporal_efficiency": float(inverse_arrival_time_sum / ordered_pair_denominator) if node_count > 1 else 0.0,
                "mean_arrival_time_reached": float(arrival_time_weighted_sum / reachable_pair_count) if reachable_pair_count > 0 else np.nan,
            }
        )

    forward_counts = _forward_reach_counts_from_target_sources(target_sources, node_count)
    reach_denominator = max(node_count - 1, 1)
    source_summary = pd.DataFrame(
        {
            "node_id": node_universe,
            "forward_reach_count": forward_counts,
            "forward_reach_ratio": forward_counts / reach_denominator,
            "static_forward_reach_count": static_counts,
            "static_forward_reach_ratio": static_counts / reach_denominator,
        }
    )
    if node_types:
        source_summary["type_label"] = source_summary["node_id"].map(lambda node_id: node_types.get(int(node_id), "Unknown"))
    source_summary = source_summary[source_columns].sort_values(
        ["forward_reach_ratio", "forward_reach_count", "node_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    final_reachability_ratio = float(forward_counts.sum() / ordered_pair_denominator) if node_count > 1 else 0.0
    if static_reachability_ratio <= 0:
        causal_fidelity = 1.0 if final_reachability_ratio <= 0 else np.nan
    else:
        causal_fidelity = float(final_reachability_ratio / static_reachability_ratio)

    global_summary = {
        "reachability_ratio": final_reachability_ratio,
        "reachable_pair_count": int(forward_counts.sum()),
        "temporal_efficiency": float(rows[-1]["temporal_efficiency"]) if rows else 0.0,
        "mean_arrival_time_reached": float(rows[-1]["mean_arrival_time_reached"]) if rows else np.nan,
        "static_reachability_ratio": static_reachability_ratio,
        "causal_fidelity": causal_fidelity,
        "mean_forward_reach_ratio": float(source_summary["forward_reach_ratio"].mean()) if len(source_summary) else 0.0,
        "p95_forward_reach_ratio": float(np.percentile(source_summary["forward_reach_ratio"], 95)) if len(source_summary) else 0.0,
        "max_forward_reach_ratio": float(source_summary["forward_reach_ratio"].max()) if len(source_summary) else 0.0,
    }
    return {
        "per_snapshot": pd.DataFrame(rows, columns=per_snapshot_columns),
        "source_summary": source_summary,
        "global_summary": global_summary,
    }



def _build_edge_index_lists(
    df: pd.DataFrame,
    node_index: Dict[int, int],
    *,
    weight_col: Optional[str] = None,
) -> tuple[list[int], list[int], np.ndarray]:
    rows = [node_index[int(value)] for value in df["u"].to_numpy(dtype=np.int64, copy=False)]
    cols = [node_index[int(value)] for value in df["i"].to_numpy(dtype=np.int64, copy=False)]
    if weight_col and weight_col in df.columns:
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (weights < 0).any():
            LOGGER.warning("Negative edge weights were encountered in a weighted diagnostic and were clipped to zero.")
            weights = np.clip(weights, a_min=0.0, a_max=None)
        return rows, cols, weights
    return rows, cols, np.ones(len(rows), dtype=float)



def _component_lists(
    rows: list[int],
    cols: list[int],
    node_count: int,
    *,
    directed: bool,
    weights: Optional[np.ndarray] = None,
) -> list[list[int]]:
    if node_count <= 0:
        return []
    if weights is None:
        weights = np.ones(len(rows), dtype=float)
    weights = np.asarray(weights, dtype=float)
    active_nodes = sorted({row for row, weight in zip(rows, weights) if weight > 0} | {col for col, weight in zip(cols, weights) if weight > 0})
    if not active_nodes:
        return []

    if directed:
        adjacency = [[] for _ in range(node_count)]
        for row, col, weight in zip(rows, cols, weights):
            if weight <= 0:
                continue
            adjacency[row].append(col)

        index = 0
        indices = [-1] * node_count
        lowlink = [0] * node_count
        on_stack = [False] * node_count
        stack: list[int] = []
        components: list[list[int]] = []

        def strongconnect(vertex: int) -> None:
            nonlocal index
            indices[vertex] = index
            lowlink[vertex] = index
            index += 1
            stack.append(vertex)
            on_stack[vertex] = True
            for neighbor in adjacency[vertex]:
                if indices[neighbor] == -1:
                    strongconnect(neighbor)
                    lowlink[vertex] = min(lowlink[vertex], lowlink[neighbor])
                elif on_stack[neighbor]:
                    lowlink[vertex] = min(lowlink[vertex], indices[neighbor])
            if lowlink[vertex] != indices[vertex]:
                return
            component: list[int] = []
            while stack:
                node = stack.pop()
                on_stack[node] = False
                component.append(node)
                if node == vertex:
                    break
            components.append(component)

        for vertex in active_nodes:
            if indices[vertex] == -1:
                strongconnect(vertex)
        return components

    adjacency = [set() for _ in range(node_count)]
    for row, col, weight in zip(rows, cols, weights):
        if weight <= 0:
            continue
        adjacency[row].add(col)
        adjacency[col].add(row)
    visited = [False] * node_count
    components: list[list[int]] = []
    for vertex in active_nodes:
        if visited[vertex]:
            continue
        stack = [vertex]
        visited[vertex] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def _largest_closed_component_nodes(
    rows: list[int],
    cols: list[int],
    node_count: int,
    *,
    directed: bool,
    weights: Optional[np.ndarray] = None,
) -> list[int]:
    components = _component_lists(rows, cols, node_count, directed=directed, weights=weights)
    if not components:
        return []
    if not directed:
        return sorted(max(components, key=len))

    if weights is None:
        weights = np.ones(len(rows), dtype=float)
    weights = np.asarray(weights, dtype=float)
    comp_index: dict[int, int] = {}
    for idx, component in enumerate(components):
        for node in component:
            comp_index[int(node)] = idx
    closed = [True] * len(components)
    for row, col, weight in zip(rows, cols, weights):
        if weight <= 0:
            continue
        source = comp_index.get(int(row))
        target = comp_index.get(int(col))
        if source is None or target is None:
            continue
        if source != target:
            closed[source] = False
    closed_components = [component for component, is_closed in zip(components, closed) if is_closed]
    target_components = closed_components or components
    return sorted(max(target_components, key=len))

def _largest_component_nodes(
    rows: list[int],
    cols: list[int],
    node_count: int,
    *,
    directed: bool,
    weights: Optional[np.ndarray] = None,
) -> list[int]:
    components = _component_lists(rows, cols, node_count, directed=directed, weights=weights)
    if not components:
        return []
    return sorted(max(components, key=len))



def _power_stationary_distribution(
    component_nodes: list[int],
    rows: list[int],
    cols: list[int],
    weights: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    size = len(component_nodes)
    if size == 0:
        return np.array([], dtype=float)
    if size == 1:
        return np.array([1.0], dtype=float)

    component_index = {node_id: index for index, node_id in enumerate(component_nodes)}
    outgoing: list[dict[int, float]] = [defaultdict(float) for _ in range(size)]
    out_strength = np.zeros(size, dtype=float)

    for row, col, weight in zip(rows, cols, np.asarray(weights, dtype=float)):
        if weight <= 0:
            continue
        row_index = component_index.get(int(row))
        col_index = component_index.get(int(col))
        if row_index is None or col_index is None:
            continue
        outgoing[row_index][col_index] += float(weight)
        out_strength[row_index] += float(weight)

    pi = np.full(size, 1.0 / size, dtype=float)
    scratch = np.zeros(size, dtype=float)
    for _ in range(10000):
        scratch.fill(0.0)
        for row_index, neighbour_weights in enumerate(outgoing):
            strength = float(out_strength[row_index])
            if strength > 0:
                scale = (1.0 - alpha) * pi[row_index] / strength
                for col_index, edge_weight in neighbour_weights.items():
                    scratch[col_index] += scale * edge_weight
            else:
                scratch[row_index] += (1.0 - alpha) * pi[row_index]
            scratch[row_index] += alpha * pi[row_index]
        normaliser = float(scratch.sum())
        if normaliser > 0:
            scratch /= normaliser
        if float(np.linalg.norm(scratch - pi, ord=1)) < 1e-10:
            pi = scratch.copy()
            break
        pi, scratch = scratch.copy(), pi
    pi = np.asarray(pi, dtype=float)
    normaliser = float(pi.sum())
    if normaliser > 0:
        pi /= normaliser
    return pi



def _teleporting_pagerank_distribution(
    support_nodes: list[int],
    rows: list[int],
    cols: list[int],
    weights: np.ndarray,
    *,
    teleport_alpha: float,
) -> np.ndarray:
    size = len(support_nodes)
    if size == 0:
        return np.array([], dtype=float)
    if size == 1:
        return np.array([1.0], dtype=float)

    support_index = {node_id: index for index, node_id in enumerate(support_nodes)}
    outgoing: list[dict[int, float]] = [defaultdict(float) for _ in range(size)]
    out_strength = np.zeros(size, dtype=float)

    for row, col, weight in zip(rows, cols, np.asarray(weights, dtype=float)):
        if weight <= 0:
            continue
        row_index = support_index.get(int(row))
        col_index = support_index.get(int(col))
        if row_index is None or col_index is None:
            continue
        outgoing[row_index][col_index] += float(weight)
        out_strength[row_index] += float(weight)

    teleport = np.full(size, 1.0 / size, dtype=float)
    pi = teleport.copy()
    scratch = np.zeros(size, dtype=float)
    damping = 1.0 - float(teleport_alpha)

    for _ in range(10000):
        scratch[:] = float(teleport_alpha) * teleport
        dangling_mass = 0.0
        for row_index, neighbour_weights in enumerate(outgoing):
            strength = float(out_strength[row_index])
            if strength > 0:
                scale = damping * pi[row_index] / strength
                for col_index, edge_weight in neighbour_weights.items():
                    scratch[col_index] += scale * edge_weight
            else:
                dangling_mass += damping * pi[row_index]
        if dangling_mass > 0:
            scratch += dangling_mass * teleport
        normaliser = float(scratch.sum())
        if normaliser > 0:
            scratch /= normaliser
        if float(np.linalg.norm(scratch - pi, ord=1)) < 1e-10:
            pi = scratch.copy()
            break
        pi, scratch = scratch.copy(), pi
    pi = np.asarray(pi, dtype=float)
    normaliser = float(pi.sum())
    if normaliser > 0:
        pi /= normaliser
    return pi
def _gini(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    total = float(finite.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(finite)
    index = np.arange(1, ordered.size + 1, dtype=float)
    return float(np.sum((2.0 * index - ordered.size - 1.0) * ordered) / (ordered.size * total))



def _compute_pi_mass_time_series(
    df: pd.DataFrame,
    *,
    node_universe: list[int],
    node_types: Optional[Dict[int, str]],
    directed: bool,
    weight_col: Optional[str] = None,
    alpha: float = LAZY_WALK_ALPHA,
    mode: str = "largest_component_lazy",
    teleport_alpha: float = PAGERANK_TELEPORT_ALPHA,
) -> pd.DataFrame:
    type_labels = _sorted_type_labels(node_types)
    columns = [
        "ts",
        "lic_size",
        "lic_share_total",
        "lic_share_active",
        "active_node_count",
        "active_farm_count",
        "active_region_count",
        "pi_gini",
    ]
    columns.extend(f"pi_mass__{_safe_metric_key(label)}" for label in type_labels)
    if df.empty or not node_universe:
        return pd.DataFrame(columns=columns)

    node_index = {node_id: index for index, node_id in enumerate(node_universe)}
    rows: list[dict[str, float]] = []

    def _active_type_counts(active_nodes: set[int]) -> tuple[float, float]:
        if not node_types:
            return np.nan, np.nan
        farm_count = 0
        region_count = 0
        for node_idx in active_nodes:
            label = str(node_types.get(node_universe[node_idx], "Unknown"))
            if label == "Farm":
                farm_count += 1
            elif label == "Region":
                region_count += 1
        return float(farm_count), float(region_count)

    def _empty_pi_mass_row(
        ts_value: int,
        *,
        lic_size: float,
        lic_share_total: float,
        lic_share_active: float,
        active_node_count: float,
        active_farm_count: float,
        active_region_count: float,
    ) -> dict[str, float]:
        row = {
            "ts": int(ts_value),
            "lic_size": float(lic_size),
            "lic_share_total": float(lic_share_total),
            "lic_share_active": float(lic_share_active),
            "active_node_count": float(active_node_count),
            "active_farm_count": float(active_farm_count) if np.isfinite(active_farm_count) else np.nan,
            "active_region_count": float(active_region_count) if np.isfinite(active_region_count) else np.nan,
            "pi_gini": np.nan,
        }
        for label in type_labels:
            row[f"pi_mass__{_safe_metric_key(label)}"] = np.nan
        return row

    for ts_value, snapshot in df.groupby("ts", sort=True):
        edge_rows, edge_cols, edge_weights = _build_edge_index_lists(snapshot, node_index, weight_col=weight_col)
        if not directed:
            walk_rows = edge_rows + edge_cols
            walk_cols = edge_cols + edge_rows
            walk_weights = np.concatenate([edge_weights, edge_weights]).astype(float, copy=False)
        else:
            walk_rows = edge_rows
            walk_cols = edge_cols
            walk_weights = np.asarray(edge_weights, dtype=float)

        positive_mask = walk_weights > 0
        if not positive_mask.any():
            rows.append(
                _empty_pi_mass_row(
                    int(ts_value),
                    lic_size=0.0,
                    lic_share_total=0.0,
                    lic_share_active=np.nan,
                    active_node_count=0.0,
                    active_farm_count=0.0 if node_types else np.nan,
                    active_region_count=0.0 if node_types else np.nan,
                )
            )
            continue

        active_nodes = {row for row, keep in zip(walk_rows, positive_mask) if keep} | {col for col, keep in zip(walk_cols, positive_mask) if keep}
        active_node_count = float(len(active_nodes))
        active_farm_count, active_region_count = _active_type_counts(active_nodes)
        if mode == "largest_component_lazy":
            support_nodes = _largest_component_nodes(
                walk_rows, walk_cols, len(node_universe), directed=directed, weights=walk_weights
            )
            if len(support_nodes) < 2:
                rows.append(
                    _empty_pi_mass_row(
                        int(ts_value),
                        lic_size=float(len(support_nodes)),
                        lic_share_total=float(len(support_nodes) / len(node_universe)) if node_universe else 0.0,
                        lic_share_active=float(len(support_nodes) / len(active_nodes)) if active_nodes else np.nan,
                        active_node_count=active_node_count,
                        active_farm_count=active_farm_count,
                        active_region_count=active_region_count,
                    )
                )
                continue
            pi = _power_stationary_distribution(support_nodes, walk_rows, walk_cols, walk_weights, alpha=alpha)
        elif mode == "largest_closed_class_lazy":
            support_nodes = _largest_closed_component_nodes(
                walk_rows, walk_cols, len(node_universe), directed=directed, weights=walk_weights
            )
            if len(support_nodes) < 2:
                rows.append(
                    _empty_pi_mass_row(
                        int(ts_value),
                        lic_size=float(len(support_nodes)),
                        lic_share_total=float(len(support_nodes) / len(node_universe)) if node_universe else 0.0,
                        lic_share_active=float(len(support_nodes) / len(active_nodes)) if active_nodes else np.nan,
                        active_node_count=active_node_count,
                        active_farm_count=active_farm_count,
                        active_region_count=active_region_count,
                    )
                )
                continue
            pi = _power_stationary_distribution(support_nodes, walk_rows, walk_cols, walk_weights, alpha=alpha)
        elif mode == "teleporting_pagerank":
            support_nodes = sorted(active_nodes)
            if not support_nodes:
                rows.append(
                    _empty_pi_mass_row(
                        int(ts_value),
                        lic_size=0.0,
                        lic_share_total=0.0,
                        lic_share_active=np.nan,
                        active_node_count=active_node_count,
                        active_farm_count=active_farm_count,
                        active_region_count=active_region_count,
                    )
                )
                continue
            pi = _teleporting_pagerank_distribution(
                support_nodes, walk_rows, walk_cols, walk_weights, teleport_alpha=teleport_alpha
            )
        else:
            raise ValueError(f"Unsupported pi-mass mode: {mode}")

        row = {
            "ts": int(ts_value),
            "lic_size": float(len(support_nodes)),
            "lic_share_total": float(len(support_nodes) / len(node_universe)) if node_universe else 0.0,
            "lic_share_active": float(len(support_nodes) / len(active_nodes)) if active_nodes else 0.0,
            "active_node_count": active_node_count,
            "active_farm_count": active_farm_count,
            "active_region_count": active_region_count,
            "pi_gini": _gini(pi),
        }
        if node_types:
            support_type_labels = [node_types.get(node_universe[index], "Unknown") for index in support_nodes]
            for label in type_labels:
                column_name = f"pi_mass__{_safe_metric_key(label)}"
                row[column_name] = float(sum(weight for weight, node_label in zip(pi, support_type_labels) if node_label == label)) if len(pi) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _load_scipy_sparse():
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ModuleNotFoundError:
        return None, None
    return sp, spla



def _magnetic_normalized_laplacian(
    rows: list[int],
    cols: list[int],
    node_count: int,
    *,
    charge: float,
    weights: Optional[np.ndarray] = None,
):
    sp, _ = _load_scipy_sparse()
    data = np.asarray(weights if weights is not None else np.ones(len(rows), dtype=float), dtype=float)
    if data.size != len(rows):
        raise ValueError("Weighted magnetic Laplacian requires one weight per edge.")

    if sp is None:
        adjacency = np.zeros((node_count, node_count), dtype=float)
        for row, col, value in zip(rows, cols, data):
            adjacency[int(row), int(col)] += float(value)
        adjacency_symmetric = 0.5 * (adjacency + adjacency.T)
        total_flow = adjacency + adjacency.T
        delta = adjacency - adjacency.T
        imbalance = np.zeros_like(adjacency_symmetric, dtype=float)
        nonzero_mask = total_flow > 0
        imbalance[nonzero_mask] = delta[nonzero_mask] / total_flow[nonzero_mask]
        phase = np.exp(1j * 2.0 * np.pi * charge * imbalance)
        hermitian = adjacency_symmetric.astype(np.complex128) * phase
        hermitian = 0.5 * (hermitian + hermitian.conj().T)
        degree = adjacency_symmetric.sum(axis=1)
        inv_sqrt = np.zeros(node_count, dtype=float)
        mask = degree > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
        laplacian = np.eye(node_count, dtype=np.complex128) - (inv_sqrt[:, None] * hermitian * inv_sqrt[None, :])
        laplacian[~mask, ~mask] = 0.0
        return laplacian

    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(node_count, node_count), dtype=np.float64)
    adjacency_symmetric = ((adjacency + adjacency.T) * 0.5).tocsr()
    delta = (adjacency - adjacency.T).tocsr()
    coo = adjacency_symmetric.tocoo()
    delta_values = delta[coo.row, coo.col].A1
    total_flow = 2.0 * coo.data.astype(float)
    imbalance = np.divide(
        delta_values,
        total_flow,
        out=np.zeros_like(delta_values, dtype=float),
        where=total_flow > 0,
    )
    phase = np.exp(1j * 2.0 * np.pi * charge * imbalance)
    hermitian = sp.coo_matrix(
        (coo.data.astype(np.complex128) * phase, (coo.row, coo.col)),
        shape=adjacency.shape,
    ).tocsr()
    hermitian = (hermitian + hermitian.getH()) * 0.5
    degree = np.asarray(adjacency_symmetric.sum(axis=1)).ravel()
    inv_sqrt = np.zeros(node_count, dtype=float)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    diagonal = sp.diags(inv_sqrt, format="csr")
    laplacian = sp.eye(node_count, format="csr", dtype=np.complex128) - (diagonal @ hermitian @ diagonal)
    if (~mask).any():
        diagonal_values = laplacian.diagonal()
        diagonal_values[~mask] = 0.0
        laplacian.setdiag(diagonal_values)
    return laplacian.tocsr()


def _smallest_magnetic_eigenvalues(
    laplacian,
    *,
    k: int,
) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=float)

    sp, spla = _load_scipy_sparse()
    if sp is not None and hasattr(laplacian, "shape"):
        node_count = int(laplacian.shape[0])
    else:
        node_count = int(np.asarray(laplacian).shape[0])
    if node_count <= 1:
        return np.zeros(min(k, node_count), dtype=float)
    k_eff = min(k, max(1, node_count - 1))

    if sp is not None and spla is not None and hasattr(laplacian, "tocsr") and node_count > k_eff + 1:
        try:
            values = spla.eigsh(
                laplacian,
                k=k_eff,
                which="SA",
                return_eigenvectors=False,
                tol=1e-7,
                maxiter=max(5000, node_count * 20),
            )
            return np.sort(np.real(values)).astype(float)
        except Exception as exc:
            LOGGER.debug("Falling back to dense magnetic eigendecomposition | nodes=%s | error=%s", node_count, exc)

    dense = laplacian.toarray() if hasattr(laplacian, "toarray") else np.asarray(laplacian)
    values = np.linalg.eigvalsh(np.asarray(dense, dtype=np.complex128))
    return np.sort(np.real(values))[:k_eff].astype(float)


def _smallest_magnetic_eigenpairs(
    laplacian,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=np.complex128)

    sp, spla = _load_scipy_sparse()
    if sp is not None and hasattr(laplacian, "shape"):
        node_count = int(laplacian.shape[0])
    else:
        node_count = int(np.asarray(laplacian).shape[0])
    if node_count <= 1:
        size = min(k, node_count)
        return np.zeros(size, dtype=float), np.eye(node_count, size, dtype=np.complex128)
    k_eff = min(k, max(1, node_count - 1))

    if sp is not None and spla is not None and hasattr(laplacian, "tocsr") and node_count > k_eff + 1:
        try:
            values, vectors = spla.eigsh(
                laplacian,
                k=k_eff,
                which="SA",
                return_eigenvectors=True,
                tol=1e-7,
                maxiter=max(5000, node_count * 20),
            )
            order = np.argsort(np.real(values))
            return np.real(values[order]).astype(float), np.asarray(vectors[:, order], dtype=np.complex128)
        except Exception as exc:
            LOGGER.debug("Falling back to dense magnetic eigendecomposition for eigenpairs | nodes=%s | error=%s", node_count, exc)

    dense = laplacian.toarray() if hasattr(laplacian, "toarray") else np.asarray(laplacian)
    values, vectors = np.linalg.eigh(np.asarray(dense, dtype=np.complex128))
    order = np.argsort(np.real(values))[:k_eff]
    return np.real(values[order]).astype(float), np.asarray(vectors[:, order], dtype=np.complex128)



def _compute_magnetic_spectrum_time_series(
    df: pd.DataFrame,
    *,
    node_universe: list[int],
    weight_col: Optional[str] = None,
    directed: bool = True,
    k: int = MAGNETIC_EIGENVALUE_COUNT,
    charge: float = MAGNETIC_CHARGE,
) -> pd.DataFrame:
    columns = ["ts"] + [f"eig_{index + 1}" for index in range(k)]
    if df.empty or not node_universe:
        return pd.DataFrame(columns=columns)

    node_index = {node_id: index for index, node_id in enumerate(node_universe)}
    rows_out: list[dict[str, float]] = []
    for ts_value, snapshot in df.groupby("ts", sort=True):
        edge_rows, edge_cols, edge_weights = _build_edge_index_lists(snapshot, node_index, weight_col=weight_col)
        if not directed:
            reverse_rows = list(edge_cols)
            reverse_cols = list(edge_rows)
            edge_rows = edge_rows + reverse_rows
            edge_cols = edge_cols + reverse_cols
            edge_weights = np.concatenate([edge_weights, edge_weights]).astype(float, copy=False)
        laplacian = _magnetic_normalized_laplacian(
            edge_rows,
            edge_cols,
            len(node_universe),
            charge=charge,
            weights=edge_weights,
        )
        eigenvalues = _smallest_magnetic_eigenvalues(laplacian, k=k)
        row = {"ts": int(ts_value)}
        for index in range(k):
            row[f"eig_{index + 1}"] = float(eigenvalues[index]) if index < len(eigenvalues) else np.nan
        rows_out.append(row)
    return pd.DataFrame(rows_out, columns=columns)


def _align_magnetic_subspace(previous_vectors: np.ndarray, current_vectors: np.ndarray) -> np.ndarray:
    if previous_vectors.size == 0 or current_vectors.size == 0:
        return current_vectors
    overlap = previous_vectors.conj().T @ current_vectors
    u_matrix, _, vh_matrix = np.linalg.svd(overlap, full_matrices=False)
    aligned = current_vectors @ (u_matrix @ vh_matrix)
    for column_index in range(min(previous_vectors.shape[1], aligned.shape[1])):
        numerator = np.vdot(previous_vectors[:, column_index], aligned[:, column_index])
        phase_shift = np.angle(numerator) if np.abs(numerator) > 0 else 0.0
        aligned[:, column_index] *= np.exp(-1j * phase_shift)
    return aligned



def _compute_magnetic_phase_time_series(
    df: pd.DataFrame,
    *,
    node_universe: list[int],
    weight_col: Optional[str] = None,
    directed: bool = True,
    k: int = 2,
    charge: float = MAGNETIC_CHARGE,
) -> dict[str, object]:
    if df.empty or not node_universe:
        return {"ts": np.array([], dtype=int), "eigenvalues": np.empty((0, 0), dtype=float), "phi": np.empty((0, 0, 0), dtype=float), "mag": np.empty((0, 0, 0), dtype=float), "mask": np.empty((0, 0), dtype=bool)}

    node_index = {node_id: index for index, node_id in enumerate(node_universe)}
    ts_values = sorted(pd.to_numeric(df["ts"], errors="coerce").dropna().astype(int).unique().tolist())
    eigenvalues = np.full((len(ts_values), k), np.nan, dtype=float)
    phases = np.full((len(ts_values), len(node_universe), k), np.nan, dtype=float)
    magnitudes = np.full((len(ts_values), len(node_universe), k), np.nan, dtype=float)
    masks = np.zeros((len(ts_values), len(node_universe)), dtype=bool)
    previous_vectors: Optional[np.ndarray] = None

    for time_index, ts_value in enumerate(ts_values):
        snapshot = df.loc[df["ts"] == ts_value]
        edge_rows, edge_cols, edge_weights = _build_edge_index_lists(snapshot, node_index, weight_col=weight_col)
        if not directed:
            reverse_rows = list(edge_cols)
            reverse_cols = list(edge_rows)
            edge_rows = edge_rows + reverse_rows
            edge_cols = edge_cols + reverse_cols
            edge_weights = np.concatenate([edge_weights, edge_weights]).astype(float, copy=False)
        degree = np.zeros(len(node_universe), dtype=float)
        for row_index, col_index, weight in zip(edge_rows, edge_cols, np.asarray(edge_weights, dtype=float)):
            degree[int(row_index)] += float(weight)
            degree[int(col_index)] += float(weight)
        mask = degree > 0
        masks[time_index] = mask
        laplacian = _magnetic_normalized_laplacian(
            edge_rows,
            edge_cols,
            len(node_universe),
            charge=charge,
            weights=edge_weights,
        )
        values, vectors = _smallest_magnetic_eigenpairs(laplacian, k=k)
        if previous_vectors is not None and vectors.size:
            vectors = _align_magnetic_subspace(previous_vectors, vectors)
        previous_vectors = vectors
        for mode_index in range(min(k, len(values))):
            eigenvalues[time_index, mode_index] = float(values[mode_index])
            magnitudes[time_index, :, mode_index] = np.abs(vectors[:, mode_index])
            current_phase = np.angle(vectors[:, mode_index])
            current_phase[~mask] = np.nan
            phases[time_index, :, mode_index] = current_phase

    return {
        "ts": np.asarray(ts_values, dtype=int),
        "eigenvalues": eigenvalues,
        "phi": phases,
        "mag": magnitudes,
        "mask": masks,
    }



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
        quantiles = (np.arange(q_count, dtype=float) + 0.5) / q_count
        xq = np.quantile(x, quantiles)
        yq = np.quantile(y, quantiles)
        return float(np.mean(np.abs(xq - yq)))


def _compute_magnetic_spectral_distance_time_series(
    original_spectrum: pd.DataFrame,
    synthetic_spectrum: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "ts",
        "spectral_wasserstein_distance",
        "spectral_mean_abs_delta",
        "spectral_rmse",
        "spectral_max_abs_delta",
        "mode_count_observed",
        "mode_count_synthetic",
    ]
    if original_spectrum.empty or synthetic_spectrum.empty:
        return pd.DataFrame(columns=columns)

    mode_columns = sorted(set(original_spectrum.columns).intersection(synthetic_spectrum.columns) - {"ts"})
    if not mode_columns:
        return pd.DataFrame(columns=columns)

    merged = original_spectrum.merge(
        synthetic_spectrum,
        on="ts",
        how="outer",
        suffixes=("_original", "_synthetic"),
    ).sort_values("ts").reset_index(drop=True)

    rows_out: list[dict[str, float]] = []
    for row in merged.itertuples(index=False):
        observed_values = np.asarray([getattr(row, f"{column}_original") for column in mode_columns], dtype=float)
        synthetic_values = np.asarray([getattr(row, f"{column}_synthetic") for column in mode_columns], dtype=float)
        observed_finite = observed_values[np.isfinite(observed_values)]
        synthetic_finite = synthetic_values[np.isfinite(synthetic_values)]
        aligned_mask = np.isfinite(observed_values) & np.isfinite(synthetic_values)
        if aligned_mask.any():
            deltas = np.abs(observed_values[aligned_mask] - synthetic_values[aligned_mask])
            mae = float(deltas.mean())
            rmse = float(np.sqrt(np.mean((observed_values[aligned_mask] - synthetic_values[aligned_mask]) ** 2)))
            max_abs = float(deltas.max())
        else:
            mae = np.nan
            rmse = np.nan
            max_abs = np.nan
        rows_out.append(
            {
                "ts": int(getattr(row, "ts")),
                "spectral_wasserstein_distance": _wasserstein_distance_1d(observed_finite, synthetic_finite),
                "spectral_mean_abs_delta": mae,
                "spectral_rmse": rmse,
                "spectral_max_abs_delta": max_abs,
                "mode_count_observed": int(len(observed_finite)),
                "mode_count_synthetic": int(len(synthetic_finite)),
            }
        )
    return pd.DataFrame(rows_out, columns=columns)


def _summarise_distance_time_series(
    frame: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    columns = ["metric", "mean", "median", "max", "snapshot_count"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows_out: list[dict[str, float]] = []
    for metric_name in metric_names:
        values = pd.to_numeric(frame.get(metric_name, np.nan), errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if len(finite):
            rows_out.append(
                {
                    "metric": metric_name,
                    "mean": float(finite.mean()),
                    "median": float(finite.median()),
                    "max": float(finite.max()),
                    "snapshot_count": int(len(finite)),
                }
            )
    return pd.DataFrame(rows_out, columns=columns)
def _summarise_metric_time_series(
    merged: pd.DataFrame,
    metric_names: list[str],
    *,
    label_column: str = "metric",
    treat_missing_as_zero: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if merged.empty:
        return pd.DataFrame(columns=[label_column, "original_mean", "synthetic_mean", "original_total", "synthetic_total", "mean_abs_delta", "max_abs_delta", "correlation", "snapshot_count"])

    def numeric_column_or_default(column_name: str, default: float = 0.0) -> pd.Series:
        if column_name in merged.columns:
            return pd.to_numeric(merged[column_name], errors="coerce")
        return pd.Series(np.full(len(merged), float(default), dtype=float), index=merged.index, dtype=float)

    for metric_name in metric_names:
        original_column = f"original_{metric_name}"
        synthetic_column = f"synthetic_{metric_name}"
        delta_column = f"{metric_name}_delta"
        original_values = numeric_column_or_default(original_column).to_numpy(dtype=float)
        synthetic_values = numeric_column_or_default(synthetic_column).to_numpy(dtype=float)
        delta_values = numeric_column_or_default(delta_column).to_numpy(dtype=float)
        if treat_missing_as_zero:
            original_values = np.nan_to_num(original_values, nan=0.0)
            synthetic_values = np.nan_to_num(synthetic_values, nan=0.0)
            delta_values = np.nan_to_num(delta_values, nan=0.0)
            original_mean = float(original_values.mean()) if len(original_values) else 0.0
            synthetic_mean = float(synthetic_values.mean()) if len(synthetic_values) else 0.0
            original_total = float(original_values.sum()) if len(original_values) else 0.0
            synthetic_total = float(synthetic_values.sum()) if len(synthetic_values) else 0.0
            mean_abs_delta = float(np.abs(delta_values).mean()) if len(delta_values) else 0.0
            max_abs_delta = float(np.abs(delta_values).max()) if len(delta_values) else 0.0
        else:
            original_mean = float(np.nanmean(original_values)) if np.isfinite(original_values).any() else np.nan
            synthetic_mean = float(np.nanmean(synthetic_values)) if np.isfinite(synthetic_values).any() else np.nan
            original_total = float(np.nansum(original_values)) if len(original_values) else 0.0
            synthetic_total = float(np.nansum(synthetic_values)) if len(synthetic_values) else 0.0
            mean_abs_delta = float(np.nanmean(np.abs(delta_values))) if np.isfinite(delta_values).any() else np.nan
            max_abs_delta = float(np.nanmax(np.abs(delta_values))) if np.isfinite(delta_values).any() else np.nan
        rows.append(
            {
                label_column: metric_name,
                "original_mean": original_mean,
                "synthetic_mean": synthetic_mean,
                "original_total": original_total,
                "synthetic_total": synthetic_total,
                "mean_abs_delta": mean_abs_delta,
                "max_abs_delta": max_abs_delta,
                "correlation": _safe_correlation(original_values, synthetic_values),
                "snapshot_count": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def _metric_lookup(summary: pd.DataFrame, metric_name: str, field_name: str) -> Optional[float]:
    if summary.empty or metric_name not in summary.get("metric", pd.Series(dtype=object)).tolist():
        return None
    subset = summary.loc[summary["metric"] == metric_name, field_name]
    if subset.empty:
        return None
    value = pd.to_numeric(subset, errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)



def _display_advanced_metric_name(metric_name: str) -> str:
    label = str(metric_name)
    if label.startswith("pi_mass__"):
        suffix = label.removeprefix("pi_mass__")
        if suffix in {"type_0", "farm"}:
            return "Pi-Mass (Farm)"
        if suffix in {"type_1", "region"}:
            return "Pi-Mass (Region)"
        return f"Pi-Mass ({suffix.replace('_', ' ').title()})"
    if label.startswith("eig_"):
        return f"Mode {label.removeprefix('eig_')}"
    mapping = {
        "new_count": "New count",
        "reactivated_count": "Reactivated count",
        "persist_count": "Persistent count",
        "ceased_prev_count": "Churn count",
        "repeated_count": "Repeated count",
        "total_count": "Total count",
        "new_ratio": "New ratio",
        "persist_ratio": "Persistence ratio",
        "reactivated_ratio": "Reactivated ratio",
        "churn_ratio": "Churn ratio",
        "birth_count": "Birth count",
        "birth_rate": "Birth rate",
        "active_count": "Active count",
        "new_rate": "New-node rate",
        "lic_size": "Support size",
        "lic_share_total": "Support share of node universe",
        "lic_share_active": "Support share of active nodes",
        "active_node_count": "Active nodes",
        "active_farm_count": "Active farm nodes",
        "active_region_count": "Active regional supernodes",
        "pi_gini": "Pi gini",
        "reachable_pair_count": "Reachable ordered pairs",
        "reachability_ratio": "Reachability ratio",
        "new_reachable_pair_count": "New reachable pairs",
        "temporal_efficiency": "Temporal efficiency",
        "mean_arrival_time_reached": "Mean arrival time",
        "static_reachability_ratio": "Static reachability ratio",
        "causal_fidelity": "Causal fidelity",
        "forward_reach_ratio": "Forward reachable fraction",
        "spectral_wasserstein_distance": "Spectral Wasserstein distance",
        "spectral_mean_abs_delta": "Spectral mean absolute delta",
        "spectral_rmse": "Spectral RMSE",
        "spectral_max_abs_delta": "Spectral max absolute delta",
    }
    return mapping.get(label, label.replace("_", " ").title())


def _density(node_count: int, edge_count: int, directed: bool) -> float:
    if node_count <= 1:
        return 0.0
    denom = node_count * (node_count - 1)
    if not directed:
        denom /= 2.0
    if denom <= 0:
        return 0.0
    return float(edge_count / denom)


def _safe_stats(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def _degree_metrics(df: pd.DataFrame, directed: bool) -> Dict[str, float]:
    if df.empty:
        base = {
            "active_node_count": 0,
            "edge_count": 0,
            "self_loop_count": 0,
        }
        base.update(_safe_stats([], "degree"))
        if directed:
            base["active_source_node_count"] = 0
            base["active_target_node_count"] = 0
            base.update(_safe_stats([], "in_degree"))
            base.update(_safe_stats([], "out_degree"))
        return base

    edges = list(df[["u", "i"]].itertuples(index=False, name=None))
    nodes = sorted(set(df["u"].tolist()) | set(df["i"].tolist()))
    node_set = set(nodes)

    degree = Counter({node: 0 for node in node_set})
    in_degree = Counter({node: 0 for node in node_set})
    out_degree = Counter({node: 0 for node in node_set})
    self_loop_count = 0

    for u, v in edges:
        if u == v:
            self_loop_count += 1
        if directed:
            out_degree[u] += 1
            in_degree[v] += 1
            degree[u] += 1
            degree[v] += 1
        else:
            degree[u] += 1
            degree[v] += 1

    metrics = {
        "active_node_count": int(len(node_set)),
        "edge_count": int(len(edges)),
        "self_loop_count": int(self_loop_count),
    }
    metrics.update(_safe_stats((degree[node] for node in nodes), "degree"))
    if directed:
        metrics["active_source_node_count"] = int(sum(out_degree[node] > 0 for node in nodes))
        metrics["active_target_node_count"] = int(sum(in_degree[node] > 0 for node in nodes))
        metrics.update(_safe_stats((in_degree[node] for node in nodes), "in_degree"))
        metrics.update(_safe_stats((out_degree[node] for node in nodes), "out_degree"))
    return metrics


def _edge_set(df: pd.DataFrame) -> set[Tuple[int, int]]:
    return set(df[["u", "i"]].itertuples(index=False, name=None))


def _node_set(df: pd.DataFrame) -> set[int]:
    if df.empty:
        return set()
    return set(pd.concat([df["u"], df["i"]], ignore_index=True).astype(np.int64).tolist())


def _prepare_aligned_numeric_series(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    return x, y


def _spearman_rank_correlation(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> float:
    x, y = _prepare_aligned_numeric_series(x_values, y_values)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size == 1:
        return 1.0 if np.isclose(x[0], y[0]) else 0.0

    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 and y_std <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0

    try:
        from scipy.stats import spearmanr

        corr = float(spearmanr(x, y, nan_policy="omit").statistic)
        if np.isfinite(corr):
            return corr
    except Exception:
        pass

    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    corr = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _safe_correlation(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    return _spearman_rank_correlation(x_values, y_values)


def _reciprocity(edges: set[Tuple[int, int]], directed: bool) -> float:
    if not directed:
        return 1.0
    if not edges:
        return 0.0
    non_loops = [edge for edge in edges if edge[0] != edge[1]]
    if not non_loops:
        return 0.0
    reciprocated = sum((v, u) in edges for u, v in non_loops)
    return float(reciprocated / len(non_loops))


def _distance_metrics(
    df: pd.DataFrame,
    node_coordinates: Optional[Dict[int, Tuple[float, float]]],
) -> Dict[str, float]:
    if not node_coordinates:
        return {
            "distance_km_mean": 0.0,
            "distance_km_std": 0.0,
            "distance_km_median": 0.0,
            "distance_km_p95": 0.0,
            "distance_km_coverage": 0.0,
        }

    distances = []
    covered = 0
    for row in df.itertuples(index=False):
        pair = (node_coordinates.get(int(row.u)), node_coordinates.get(int(row.i)))
        if pair[0] is None or pair[1] is None:
            continue
        covered += 1
        dx = pair[0][0] - pair[1][0]
        dy = pair[0][1] - pair[1][1]
        distances.append(((dx * dx + dy * dy) ** 0.5) / 1000.0)

    metrics = _safe_stats(distances, "distance_km")
    metrics["distance_km_coverage"] = float(covered / len(df)) if len(df) else 0.0
    return metrics


def _weight_metrics(
    df: pd.DataFrame,
    directed: bool,
    weight_col: Optional[str],
) -> Dict[str, float]:
    if not weight_col:
        return {}

    if df.empty or weight_col not in df.columns:
        base = {
            "weight_total": 0.0,
        }
        base.update(_safe_stats([], "weight"))
        base.update(_safe_stats([], "strength"))
        if directed:
            base.update(_safe_stats([], "in_strength"))
            base.update(_safe_stats([], "out_strength"))
        if not df.empty:
            base.update(_safe_stats([], "log1p_weight"))
        else:
            base.update(_safe_stats([], "log1p_weight"))
        return base

    weights = pd.to_numeric(df[weight_col], errors="raise").astype(float).to_numpy()
    metrics: Dict[str, float] = {
        "weight_total": float(weights.sum()),
    }
    metrics.update(_safe_stats(weights, "weight"))

    if np.all(weights >= 0):
        metrics.update(_safe_stats(np.log1p(weights), "log1p_weight"))
    else:
        metrics.update(_safe_stats([], "log1p_weight"))

    nodes = sorted(set(df["u"].tolist()) | set(df["i"].tolist()))
    node_set = set(nodes)
    strength = Counter({node: 0.0 for node in node_set})
    in_strength = Counter({node: 0.0 for node in node_set})
    out_strength = Counter({node: 0.0 for node in node_set})

    for row in df.itertuples(index=False):
        weight = float(getattr(row, weight_col))
        u = int(row.u)
        v = int(row.i)
        if directed:
            out_strength[u] += weight
            in_strength[v] += weight
            strength[u] += weight
            strength[v] += weight
        else:
            strength[u] += weight
            strength[v] += weight

    metrics.update(_safe_stats((strength[node] for node in nodes), "strength"))
    if directed:
        metrics.update(_safe_stats((in_strength[node] for node in nodes), "in_strength"))
        metrics.update(_safe_stats((out_strength[node] for node in nodes), "out_strength"))
    return metrics


def snapshot_metrics(
    df: pd.DataFrame,
    directed: bool,
    node_coordinates: Optional[Dict[int, Tuple[float, float]]] = None,
    edge_universe: Optional[set[Tuple[int, int]]] = None,
    weight_col: Optional[str] = None,
) -> Dict[str, float]:
    metrics = _degree_metrics(df, directed)
    edge_count = metrics["edge_count"]
    active_node_count = metrics["active_node_count"]
    metrics["density"] = _density(active_node_count, edge_count, directed)
    edges = _edge_set(df)
    metrics["reciprocity"] = _reciprocity(edges, directed)
    metrics["unique_edge_count"] = int(len(edges))
    metrics.update(_distance_metrics(df, node_coordinates))
    metrics.update(_weight_metrics(df, directed, weight_col=weight_col))
    if edge_universe is not None:
        novel_edges = len(edges - edge_universe)
        metrics["novel_edge_count"] = int(novel_edges)
        metrics["novel_edge_rate"] = float(novel_edges / len(edges)) if edges else 0.0
    LOGGER.debug(
        "Snapshot metrics | rows=%s | ts_values=%s | weight_col=%s | edge_count=%s | active_nodes=%s | density=%.6f",
        len(df),
        sorted(df["ts"].unique().tolist()) if "ts" in df.columns else [],
        weight_col,
        metrics["edge_count"],
        metrics["active_node_count"],
        metrics["density"],
    )
    return metrics


def compare_panels(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    directed: bool,
    node_coordinates: Optional[Dict[int, Tuple[float, float]]] = None,
    weight_col: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    effective_weight_col = None
    if weight_col and weight_col in original_df.columns and weight_col in synthetic_df.columns:
        effective_weight_col = weight_col
    LOGGER.debug(
        "Comparing panels | directed=%s | requested_weight_col=%s | effective_weight_col=%s | original_rows=%s | synthetic_rows=%s",
        directed,
        weight_col,
        effective_weight_col,
        len(original_df),
        len(synthetic_df),
    )

    original = canonicalise_edge_frame(original_df, directed=directed, weight_col=effective_weight_col)
    synthetic = canonicalise_edge_frame(synthetic_df, directed=directed, weight_col=effective_weight_col)
    _log_edge_frame_debug("Canonical original panel", original, directed=directed, weight_col=effective_weight_col)
    _log_edge_frame_debug("Canonical synthetic panel", synthetic, directed=directed, weight_col=effective_weight_col)

    observed_edge_universe = _edge_set(original)
    all_ts = sorted(set(original["ts"].unique().tolist()) | set(synthetic["ts"].unique().tolist()))
    LOGGER.debug("Panel comparison timeline | snapshots=%s | ts_values=%s", len(all_ts), all_ts)

    rows = []
    for ts_value in all_ts:
        orig_snapshot = original.loc[original["ts"] == ts_value].copy()
        syn_snapshot = synthetic.loc[synthetic["ts"] == ts_value].copy()

        orig_metrics = snapshot_metrics(
            orig_snapshot,
            directed=directed,
            node_coordinates=node_coordinates,
            edge_universe=observed_edge_universe,
            weight_col=effective_weight_col,
        )
        syn_metrics = snapshot_metrics(
            syn_snapshot,
            directed=directed,
            node_coordinates=node_coordinates,
            edge_universe=observed_edge_universe,
            weight_col=effective_weight_col,
        )

        orig_edges = _edge_set(orig_snapshot)
        syn_edges = _edge_set(syn_snapshot)
        orig_nodes = _node_set(orig_snapshot)
        syn_nodes = _node_set(syn_snapshot)

        edge_union = orig_edges | syn_edges
        node_union = orig_nodes | syn_nodes

        edge_jaccard = float(len(orig_edges & syn_edges) / len(edge_union)) if edge_union else 1.0
        node_jaccard = float(len(orig_nodes & syn_nodes) / len(node_union)) if node_union else 1.0

        row = {
            "ts": int(ts_value),
            "edge_jaccard": edge_jaccard,
            "node_jaccard": node_jaccard,
            "edge_overlap_count": int(len(orig_edges & syn_edges)),
            "original_unique_edge_count": int(len(orig_edges)),
            "synthetic_unique_edge_count": int(len(syn_edges)),
            "original_active_node_count": int(len(orig_nodes)),
            "synthetic_active_node_count": int(len(syn_nodes)),
            "edge_count_delta": int(syn_metrics["edge_count"] - orig_metrics["edge_count"]),
            "active_node_delta": int(syn_metrics["active_node_count"] - orig_metrics["active_node_count"]),
            "density_delta": float(syn_metrics["density"] - orig_metrics["density"]),
            "distance_km_mean_delta": float(
                syn_metrics["distance_km_mean"] - orig_metrics["distance_km_mean"]
            ),
        }
        if directed:
            row["reciprocity_delta"] = float(syn_metrics["reciprocity"] - orig_metrics["reciprocity"])
            row["source_node_delta"] = int(
                syn_metrics.get("active_source_node_count", 0) - orig_metrics.get("active_source_node_count", 0)
            )
            row["target_node_delta"] = int(
                syn_metrics.get("active_target_node_count", 0) - orig_metrics.get("active_target_node_count", 0)
            )
        if effective_weight_col:
            row["weight_total_delta"] = float(syn_metrics["weight_total"] - orig_metrics["weight_total"])
            row["weight_mean_delta"] = float(syn_metrics["weight_mean"] - orig_metrics["weight_mean"])
            row["log1p_weight_mean_delta"] = float(
                syn_metrics["log1p_weight_mean"] - orig_metrics["log1p_weight_mean"]
            )
        for key, value in orig_metrics.items():
            row[f"original_{key}"] = value
        for key, value in syn_metrics.items():
            row[f"synthetic_{key}"] = value
        rows.append(row)
        LOGGER.debug(
            "Snapshot comparison | ts=%s | edge_jaccard=%.6f | node_jaccard=%.6f | edge_count_delta=%s%s",
            ts_value,
            edge_jaccard,
            node_jaccard,
            row["edge_count_delta"],
            (
                f" | weight_total_delta={row['weight_total_delta']:.6f}"
                if effective_weight_col
                else ""
            ),
        )

    per_snapshot = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    overall_original = snapshot_metrics(
        original,
        directed=directed,
        node_coordinates=node_coordinates,
        edge_universe=observed_edge_universe,
        weight_col=effective_weight_col,
    )
    overall_synthetic = snapshot_metrics(
        synthetic,
        directed=directed,
        node_coordinates=node_coordinates,
        edge_universe=observed_edge_universe,
        weight_col=effective_weight_col,
    )

    edge_count_corr = 0.0
    if len(per_snapshot) >= 2:
        edge_count_corr = _safe_correlation(
            per_snapshot["original_edge_count"].to_numpy(dtype=float),
            per_snapshot["synthetic_edge_count"].to_numpy(dtype=float),
        )

    weight_total_corr = 0.0
    if effective_weight_col and len(per_snapshot) >= 2:
        weight_total_corr = _safe_correlation(
            per_snapshot["original_weight_total"].to_numpy(dtype=float),
            per_snapshot["synthetic_weight_total"].to_numpy(dtype=float),
        )

    summary = {
        "snapshot_count": int(len(per_snapshot)),
        "original_total_edges": int(len(original)),
        "synthetic_total_edges": int(len(synthetic)),
        "original_unique_edges": int(len(_edge_set(original))),
        "synthetic_unique_edges": int(len(_edge_set(synthetic))),
        "shared_unique_edges": int(len(_edge_set(original) & _edge_set(synthetic))),
        "unique_edge_jaccard": float(
            len(_edge_set(original) & _edge_set(synthetic))
            / len(_edge_set(original) | _edge_set(synthetic))
        )
        if (_edge_set(original) | _edge_set(synthetic))
        else 1.0,
        "mean_snapshot_edge_jaccard": float(per_snapshot["edge_jaccard"].mean()) if len(per_snapshot) else 0.0,
        "mean_snapshot_node_jaccard": float(per_snapshot["node_jaccard"].mean()) if len(per_snapshot) else 0.0,
        "mean_abs_edge_count_delta": float(per_snapshot["edge_count_delta"].abs().mean()) if len(per_snapshot) else 0.0,
        "mean_abs_active_node_delta": float(per_snapshot["active_node_delta"].abs().mean()) if len(per_snapshot) else 0.0,
        "mean_abs_density_delta": float(per_snapshot["density_delta"].abs().mean()) if len(per_snapshot) else 0.0,
        "mean_synthetic_novel_edge_rate": float(per_snapshot["synthetic_novel_edge_rate"].mean())
        if "synthetic_novel_edge_rate" in per_snapshot
        else 0.0,
        "time_series_correlation_method": TIME_SERIES_CORRELATION_METHOD,
        "edge_count_correlation": edge_count_corr,
        "original_overall": overall_original,
        "synthetic_overall": overall_synthetic,
    }
    if directed:
        summary["mean_abs_reciprocity_delta"] = float(per_snapshot["reciprocity_delta"].abs().mean()) if len(per_snapshot) else 0.0
        summary["mean_abs_source_node_delta"] = float(per_snapshot["source_node_delta"].abs().mean()) if len(per_snapshot) else 0.0
        summary["mean_abs_target_node_delta"] = float(per_snapshot["target_node_delta"].abs().mean()) if len(per_snapshot) else 0.0
        summary["reciprocity_correlation"] = (
            _safe_correlation(
                per_snapshot["original_reciprocity"].to_numpy(dtype=float),
                per_snapshot["synthetic_reciprocity"].to_numpy(dtype=float),
            )
            if len(per_snapshot) >= 2
            else 0.0
        )
    if effective_weight_col:
        summary["weight_column"] = effective_weight_col
        summary["original_total_weight"] = float(original[effective_weight_col].sum()) if len(original) else 0.0
        summary["synthetic_total_weight"] = float(synthetic[effective_weight_col].sum()) if len(synthetic) else 0.0
        summary["mean_abs_weight_total_delta"] = float(per_snapshot["weight_total_delta"].abs().mean()) if len(per_snapshot) else 0.0
        summary["mean_abs_weight_mean_delta"] = float(per_snapshot["weight_mean_delta"].abs().mean()) if len(per_snapshot) else 0.0
        summary["mean_abs_log1p_weight_mean_delta"] = (
            float(per_snapshot["log1p_weight_mean_delta"].abs().mean()) if len(per_snapshot) else 0.0
        )
        summary["weight_total_correlation"] = weight_total_corr
    LOGGER.debug("Panel comparison summary | %s", summary)
    return per_snapshot, summary


def _annotate_block_membership(df: pd.DataFrame, node_blocks: Dict[int, int], directed: bool) -> pd.DataFrame:
    frame = df.copy()
    frame["block_u"] = frame["u"].map(node_blocks)
    frame["block_v"] = frame["i"].map(node_blocks)
    missing_rows = int(frame["block_u"].isna().sum() + frame["block_v"].isna().sum())
    if missing_rows:
        LOGGER.debug("Dropping rows with missing block assignments | missing_entries=%s", missing_rows)
    frame = frame.dropna(subset=["block_u", "block_v"]).copy()
    if frame.empty:
        return frame

    frame["block_u"] = frame["block_u"].astype(np.int64)
    frame["block_v"] = frame["block_v"].astype(np.int64)
    if not directed:
        ordered = np.sort(frame[["block_u", "block_v"]].to_numpy(dtype=np.int64, copy=False), axis=1)
        frame["block_u"] = ordered[:, 0]
        frame["block_v"] = ordered[:, 1]
    return frame


def _aggregate_block_pair_time_series(
    df: pd.DataFrame,
    directed: bool,
    node_blocks: Dict[int, int],
    weight_col: Optional[str],
) -> pd.DataFrame:
    frame = _annotate_block_membership(df, node_blocks=node_blocks, directed=directed)
    if frame.empty:
        columns = ["ts", "block_u", "block_v", "edge_count"] + (["weight_total"] if weight_col else [])
        return pd.DataFrame(columns=columns)

    aggregations = {"edge_count": ("u", "size")}
    if weight_col:
        aggregations["weight_total"] = (weight_col, "sum")
    return (
        frame.groupby(["ts", "block_u", "block_v"], as_index=False, sort=True)
        .agg(**aggregations)
        .sort_values(["ts", "block_u", "block_v"])
        .reset_index(drop=True)
    )


def _aggregate_activity_time_series(
    df: pd.DataFrame,
    *,
    directed: bool,
    entity_col: str,
    source_col: str,
    target_col: str,
    weight_col: Optional[str],
) -> pd.DataFrame:
    if df.empty:
        columns = ["ts", entity_col, "incident_edge_count"]
        if directed:
            columns.extend(["out_edge_count", "in_edge_count"])
        if weight_col:
            columns.append("incident_weight_total")
            if directed:
                columns.extend(["out_weight_total", "in_weight_total"])
        return pd.DataFrame(columns=columns)

    common = {
        "ts": pd.to_numeric(df["ts"], errors="raise").astype(np.int64),
        entity_col: pd.to_numeric(df[source_col], errors="raise").astype(np.int64),
        "incident_edge_count": np.ones(len(df), dtype=np.int64),
    }
    source_rows = pd.DataFrame(common)
    target_rows = pd.DataFrame(
        {
            "ts": pd.to_numeric(df["ts"], errors="raise").astype(np.int64),
            entity_col: pd.to_numeric(df[target_col], errors="raise").astype(np.int64),
            "incident_edge_count": np.ones(len(df), dtype=np.int64),
        }
    )
    if directed:
        source_rows["out_edge_count"] = 1
        source_rows["in_edge_count"] = 0
        target_rows["out_edge_count"] = 0
        target_rows["in_edge_count"] = 1
    if weight_col:
        weights = pd.to_numeric(df[weight_col], errors="raise").astype(float)
        source_rows["incident_weight_total"] = weights
        target_rows["incident_weight_total"] = weights
        if directed:
            source_rows["out_weight_total"] = weights
            source_rows["in_weight_total"] = 0.0
            target_rows["out_weight_total"] = 0.0
            target_rows["in_weight_total"] = weights

    frame = pd.concat([source_rows, target_rows], ignore_index=True)
    numeric_columns = [column for column in frame.columns if column not in {"ts", entity_col}]
    return (
        frame.groupby(["ts", entity_col], as_index=False, sort=True)[numeric_columns]
        .sum()
        .sort_values(["ts", entity_col])
        .reset_index(drop=True)
    )


def _merge_entity_time_series(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    keys: list[str],
    *,
    fill_value: Optional[float] = 0.0,
) -> pd.DataFrame:
    original_prefixed = original.rename(columns={column: f"original_{column}" for column in original.columns if column not in keys})
    synthetic_prefixed = synthetic.rename(columns={column: f"synthetic_{column}" for column in synthetic.columns if column not in keys})
    merged = original_prefixed.merge(synthetic_prefixed, on=keys, how="outer")
    if fill_value is not None:
        merged = merged.fillna(fill_value)

    metric_names = sorted(
        {
            column.removeprefix("original_")
            for column in merged.columns
            if column.startswith("original_")
        }
        | {
            column.removeprefix("synthetic_")
            for column in merged.columns
            if column.startswith("synthetic_")
        }
    )
    for metric_name in metric_names:
        original_values = pd.to_numeric(merged.get(f"original_{metric_name}", np.nan), errors="coerce")
        synthetic_values = pd.to_numeric(merged.get(f"synthetic_{metric_name}", np.nan), errors="coerce")
        if fill_value is not None:
            original_values = original_values.fillna(float(fill_value))
            synthetic_values = synthetic_values.fillna(float(fill_value))
        merged[f"{metric_name}_delta"] = synthetic_values.to_numpy(dtype=float) - original_values.to_numpy(dtype=float)
    return merged.sort_values(keys).reset_index(drop=True)


def _complete_entity_time_series(
    merged: pd.DataFrame,
    *,
    entity_keys: list[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    if merged.empty or "ts" not in merged.columns:
        return merged

    ts_values = np.sort(pd.to_numeric(merged["ts"], errors="coerce").dropna().unique())
    if ts_values.size == 0:
        return merged.sort_values(["ts"] + entity_keys).reset_index(drop=True)

    entity_frame = merged[entity_keys].drop_duplicates().reset_index(drop=True)
    if entity_frame.empty:
        return merged.sort_values(["ts"] + entity_keys).reset_index(drop=True)

    ts_frame = pd.DataFrame({"ts": ts_values.astype(int, copy=False)})
    ts_frame["__join_key"] = 1
    entity_frame = entity_frame.copy()
    entity_frame["__join_key"] = 1
    complete_index = (
        ts_frame.merge(entity_frame, on="__join_key", how="inner")
        .drop(columns="__join_key")
        .sort_values(["ts"] + entity_keys)
        .reset_index(drop=True)
    )

    completed = complete_index.merge(merged, on=["ts"] + entity_keys, how="left", sort=True)
    value_columns = [column for column in completed.columns if column not in {"ts", *entity_keys}]
    for column in value_columns:
        numeric = pd.to_numeric(completed[column], errors="coerce")
        if numeric.notna().any() or completed[column].isna().all():
            completed[column] = numeric.fillna(float(fill_value))
    return completed.sort_values(["ts"] + entity_keys).reset_index(drop=True)


def _summarise_entity_time_series(
    merged: pd.DataFrame,
    entity_keys: list[str],
    metric_names: list[str],
) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=entity_keys)

    rows = []
    for entity_values, entity_frame in merged.groupby(entity_keys, sort=True):
        if not isinstance(entity_values, tuple):
            entity_values = (entity_values,)
        row = {key: value for key, value in zip(entity_keys, entity_values)}
        for metric_name in metric_names:
            original_col = f"original_{metric_name}"
            synthetic_col = f"synthetic_{metric_name}"
            delta_col = f"{metric_name}_delta"
            row[f"original_total_{metric_name}"] = float(entity_frame[original_col].sum())
            row[f"synthetic_total_{metric_name}"] = float(entity_frame[synthetic_col].sum())
            row[f"mean_abs_{metric_name}_delta"] = float(entity_frame[delta_col].abs().mean())
            row[f"max_abs_{metric_name}_delta"] = float(entity_frame[delta_col].abs().max())
            row[f"{metric_name}_correlation"] = _safe_correlation(entity_frame[original_col], entity_frame[synthetic_col])
        row["snapshot_count"] = int(len(entity_frame))
        active_metric = metric_names[0]
        active_mask = (
            pd.to_numeric(entity_frame[f"original_{active_metric}"], errors="coerce").fillna(0.0) > 0
        ) | (
            pd.to_numeric(entity_frame[f"synthetic_{active_metric}"], errors="coerce").fillna(0.0) > 0
        )
        row["active_snapshot_count"] = int(active_mask.sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    rank_columns = [column for column in ("original_total_incident_weight_total", "original_total_weight_total", "original_total_incident_edge_count", "original_total_edge_count") if column in summary.columns]
    if rank_columns:
        summary = summary.sort_values(rank_columns + entity_keys, ascending=[False] * len(rank_columns) + [True] * len(entity_keys))
    return summary.reset_index(drop=True)



def compare_panels_detailed(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    directed: bool,
    node_coordinates: Optional[Dict[int, Tuple[float, float]]] = None,
    weight_col: Optional[str] = None,
    node_blocks: Optional[Dict[int, int]] = None,
    node_types: Optional[Dict[int, str]] = None,
    skip_spectral_metrics: bool = False,
) -> dict:
    per_snapshot, summary = compare_panels(
        original_df=original_df,
        synthetic_df=synthetic_df,
        directed=directed,
        node_coordinates=node_coordinates,
        weight_col=weight_col,
    )

    effective_weight_col = None
    if weight_col and weight_col in original_df.columns and weight_col in synthetic_df.columns:
        effective_weight_col = weight_col
    original = canonicalise_edge_frame(original_df, directed=directed, weight_col=effective_weight_col)
    synthetic = canonicalise_edge_frame(synthetic_df, directed=directed, weight_col=effective_weight_col)
    node_universe = sorted(
        set(original["u"].tolist())
        | set(original["i"].tolist())
        | set(synthetic["u"].tolist())
        | set(synthetic["i"].tolist())
        | (set(node_types) if node_types else set())
    )

    details: dict[str, pd.DataFrame] = {
        "per_snapshot": per_snapshot,
    }

    tea_original = _compute_tea_counts(original)
    tea_synthetic = _compute_tea_counts(synthetic)
    tea_merged = _merge_entity_time_series(tea_original, tea_synthetic, ["ts"])
    tea_metrics = [
        "new_count",
        "reactivated_count",
        "persist_count",
        "ceased_prev_count",
        "repeated_count",
        "total_count",
        "new_ratio",
        "persist_ratio",
        "reactivated_ratio",
        "churn_ratio",
    ]
    tea_summary = _summarise_metric_time_series(tea_merged, tea_metrics)
    details.update(
        {
            "tea_per_snapshot": tea_merged,
            "tea_summary": tea_summary,
        }
    )
    summary["tea_new_ratio_correlation"] = _metric_lookup(tea_summary, "new_ratio", "correlation") or 0.0
    summary["tea_persist_ratio_correlation"] = _metric_lookup(tea_summary, "persist_ratio", "correlation") or 0.0
    summary["tea_reactivated_ratio_correlation"] = _metric_lookup(tea_summary, "reactivated_ratio", "correlation") or 0.0

    tna_original = _compute_tna_counts(original)
    tna_synthetic = _compute_tna_counts(synthetic)
    tna_merged = _merge_entity_time_series(tna_original, tna_synthetic, ["ts"])
    tna_metrics = [
        "new_count",
        "reactivated_count",
        "persist_count",
        "ceased_prev_count",
        "repeated_count",
        "total_count",
        "new_ratio",
        "persist_ratio",
        "reactivated_ratio",
        "churn_ratio",
    ]
    tna_summary = _summarise_metric_time_series(tna_merged, tna_metrics)
    details.update(
        {
            "tna_per_snapshot": tna_merged,
            "tna_summary": tna_summary,
        }
    )
    summary["tna_new_ratio_correlation"] = _metric_lookup(tna_summary, "new_ratio", "correlation") or 0.0
    summary["tna_persist_ratio_correlation"] = _metric_lookup(tna_summary, "persist_ratio", "correlation") or 0.0
    summary["tna_reactivated_ratio_correlation"] = _metric_lookup(tna_summary, "reactivated_ratio", "correlation") or 0.0

    tea_type_pair_original = _compute_tea_type_pair_time_series(original, node_types=node_types, directed=directed)
    tea_type_pair_synthetic = _compute_tea_type_pair_time_series(synthetic, node_types=node_types, directed=directed)
    tea_type_pair_merged = _merge_entity_time_series(
        tea_type_pair_original,
        tea_type_pair_synthetic,
        ["ts", "source_type", "target_type"],
    )
    tea_type_pair_summary = _summarise_entity_time_series(
        tea_type_pair_merged,
        ["source_type", "target_type"],
        ["new_count", "reactivated_count", "persist_count", "ceased_prev_count", "total_count", "new_ratio", "persist_ratio", "reactivated_ratio", "churn_ratio"],
    )
    if "new_ratio_correlation" in tea_type_pair_summary.columns:
        tea_type_pair_summary["birth_rate_correlation"] = tea_type_pair_summary["new_ratio_correlation"]
    if "mean_abs_new_ratio_delta" in tea_type_pair_summary.columns:
        tea_type_pair_summary["mean_abs_birth_rate_delta"] = tea_type_pair_summary["mean_abs_new_ratio_delta"]
    if not tea_type_pair_merged.empty:
        details["tea_type_pair_per_snapshot"] = tea_type_pair_merged
        details["tea_type_pair_summary"] = tea_type_pair_summary
        if "new_ratio_correlation" in tea_type_pair_summary.columns and len(tea_type_pair_summary):
            value = float(tea_type_pair_summary["new_ratio_correlation"].mean())
            summary["tea_type_pair_new_ratio_correlation"] = value
            summary["tea_type_pair_birth_rate_correlation"] = value

    tna_type_original = _compute_tna_type_time_series(original, node_types=node_types)
    tna_type_synthetic = _compute_tna_type_time_series(synthetic, node_types=node_types)
    tna_type_merged = _merge_entity_time_series(tna_type_original, tna_type_synthetic, ["ts", "type_label"])
    tna_type_summary = _summarise_entity_time_series(
        tna_type_merged,
        ["type_label"],
        ["new_count", "reactivated_count", "persist_count", "ceased_prev_count", "total_count", "new_ratio", "persist_ratio", "reactivated_ratio", "churn_ratio"],
    )
    if "new_ratio_correlation" in tna_type_summary.columns:
        tna_type_summary["new_rate_correlation"] = tna_type_summary["new_ratio_correlation"]
    if "mean_abs_new_ratio_delta" in tna_type_summary.columns:
        tna_type_summary["mean_abs_new_rate_delta"] = tna_type_summary["mean_abs_new_ratio_delta"]
    if not tna_type_merged.empty:
        details["tna_type_per_snapshot"] = tna_type_merged
        details["tna_type_summary"] = tna_type_summary
        if "new_ratio_correlation" in tna_type_summary.columns and len(tna_type_summary):
            value = float(tna_type_summary["new_ratio_correlation"].mean())
            summary["tna_type_new_ratio_correlation"] = value
            summary["tna_type_new_rate_correlation"] = value

    edge_type_original = _compute_edge_type_time_series(
        original,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
    )
    edge_type_synthetic = _compute_edge_type_time_series(
        synthetic,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
    )
    edge_type_merged = _merge_entity_time_series(edge_type_original, edge_type_synthetic, ["ts", "source_type", "target_type"])
    edge_type_metrics = ["edge_count", "edge_share"]
    if effective_weight_col:
        edge_type_metrics.extend(["weight_total", "weight_share"])
    edge_type_summary = _summarise_entity_time_series(
        edge_type_merged,
        ["source_type", "target_type"],
        edge_type_metrics,
    )
    if not edge_type_merged.empty:
        details["edge_type_per_snapshot"] = edge_type_merged
        details["edge_type_summary"] = edge_type_summary
        if "edge_share_correlation" in edge_type_summary.columns and len(edge_type_summary):
            summary["edge_type_share_correlation"] = float(pd.to_numeric(edge_type_summary["edge_share_correlation"], errors="coerce").fillna(0.0).mean())
        if "weight_share_correlation" in edge_type_summary.columns and len(edge_type_summary):
            summary["edge_type_weight_share_correlation"] = float(pd.to_numeric(edge_type_summary["weight_share_correlation"], errors="coerce").fillna(0.0).mean())

    reach_original = _compute_temporal_reachability_diagnostics(
        original,
        node_universe=node_universe,
        directed=directed,
        weight_col=effective_weight_col,
        node_types=node_types,
    )
    reach_synthetic = _compute_temporal_reachability_diagnostics(
        synthetic,
        node_universe=node_universe,
        directed=directed,
        weight_col=effective_weight_col,
        node_types=node_types,
    )
    reach_merged = _merge_entity_time_series(
        reach_original["per_snapshot"],
        reach_synthetic["per_snapshot"],
        ["ts"],
        fill_value=None,
    )
    reach_metrics = [
        "reachable_pair_count",
        "reachability_ratio",
        "new_reachable_pair_count",
        "temporal_efficiency",
        "mean_arrival_time_reached",
    ]
    reach_summary = _summarise_metric_time_series(
        reach_merged,
        reach_metrics,
        treat_missing_as_zero=False,
    )
    source_summary = pd.DataFrame({"node_id": node_universe})
    if node_types:
        source_summary["type_label"] = source_summary["node_id"].map(lambda node_id: node_types.get(int(node_id), "Unknown"))
    source_summary = source_summary.merge(
        reach_original["source_summary"].rename(
            columns={
                "forward_reach_count": "original_forward_reach_count",
                "forward_reach_ratio": "original_forward_reach_ratio",
                "static_forward_reach_count": "original_static_forward_reach_count",
                "static_forward_reach_ratio": "original_static_forward_reach_ratio",
            }
        ),
        on=[column for column in ("node_id", "type_label") if column in source_summary.columns],
        how="left",
    ).merge(
        reach_synthetic["source_summary"].rename(
            columns={
                "forward_reach_count": "synthetic_forward_reach_count",
                "forward_reach_ratio": "synthetic_forward_reach_ratio",
                "static_forward_reach_count": "synthetic_static_forward_reach_count",
                "static_forward_reach_ratio": "synthetic_static_forward_reach_ratio",
            }
        ),
        on=[column for column in ("node_id", "type_label") if column in source_summary.columns],
        how="left",
    )
    for metric_name in ("forward_reach_count", "forward_reach_ratio", "static_forward_reach_count", "static_forward_reach_ratio"):
        original_column = f"original_{metric_name}"
        synthetic_column = f"synthetic_{metric_name}"
        if original_column in source_summary.columns and synthetic_column in source_summary.columns:
            source_summary[f"{metric_name}_delta"] = pd.to_numeric(
                source_summary[synthetic_column],
                errors="coerce",
            ) - pd.to_numeric(source_summary[original_column], errors="coerce")
    if len(source_summary):
        sort_columns = [column for column in ("original_forward_reach_ratio", "original_forward_reach_count", "node_id") if column in source_summary.columns]
        ascending = [False, False, True][: len(sort_columns)]
        source_summary = source_summary.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    if not reach_merged.empty:
        details["temporal_reachability_per_snapshot"] = reach_merged
        details["temporal_reachability_summary"] = reach_summary
        details["temporal_reachability_source_summary"] = source_summary
        summary["temporal_reachability_ratio_correlation"] = _metric_lookup(reach_summary, "reachability_ratio", "correlation") or 0.0
        summary["temporal_efficiency_correlation"] = _metric_lookup(reach_summary, "temporal_efficiency", "correlation") or 0.0
        summary["temporal_new_reachable_pair_count_correlation"] = _metric_lookup(reach_summary, "new_reachable_pair_count", "correlation") or 0.0
        summary["temporal_mean_arrival_time_correlation"] = _metric_lookup(reach_summary, "mean_arrival_time_reached", "correlation") or 0.0
        if "original_forward_reach_ratio" in source_summary.columns and "synthetic_forward_reach_ratio" in source_summary.columns:
            summary["temporal_forward_reach_node_correlation"] = _safe_correlation(
                source_summary["original_forward_reach_ratio"],
                source_summary["synthetic_forward_reach_ratio"],
            )
        else:
            summary["temporal_forward_reach_node_correlation"] = 0.0
        summary["original_temporal_reachability_ratio"] = float(reach_original["global_summary"].get("reachability_ratio", 0.0))
        summary["synthetic_temporal_reachability_ratio"] = float(reach_synthetic["global_summary"].get("reachability_ratio", 0.0))
        summary["original_temporal_efficiency"] = float(reach_original["global_summary"].get("temporal_efficiency", 0.0))
        summary["synthetic_temporal_efficiency"] = float(reach_synthetic["global_summary"].get("temporal_efficiency", 0.0))
        summary["original_static_reachability_ratio"] = float(reach_original["global_summary"].get("static_reachability_ratio", 0.0))
        summary["synthetic_static_reachability_ratio"] = float(reach_synthetic["global_summary"].get("static_reachability_ratio", 0.0))
        summary["original_causal_fidelity"] = float(reach_original["global_summary"].get("causal_fidelity", np.nan))
        summary["synthetic_causal_fidelity"] = float(reach_synthetic["global_summary"].get("causal_fidelity", np.nan))
        summary["original_mean_arrival_time_reached"] = float(reach_original["global_summary"].get("mean_arrival_time_reached", np.nan))
        summary["synthetic_mean_arrival_time_reached"] = float(reach_synthetic["global_summary"].get("mean_arrival_time_reached", np.nan))
        summary["mean_forward_reach_ratio_original"] = float(reach_original["global_summary"].get("mean_forward_reach_ratio", 0.0))
        summary["mean_forward_reach_ratio_synthetic"] = float(reach_synthetic["global_summary"].get("mean_forward_reach_ratio", 0.0))
        summary["max_forward_reach_ratio_original"] = float(reach_original["global_summary"].get("max_forward_reach_ratio", 0.0))
        summary["max_forward_reach_ratio_synthetic"] = float(reach_synthetic["global_summary"].get("max_forward_reach_ratio", 0.0))

    # Pi-Mass variants: lazy walk on the largest SCC, lazy walk on the largest
    # closed SCC, and teleporting PageRank on the active snapshot.
    pi_original = _compute_pi_mass_time_series(
        original,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="largest_component_lazy",
    )
    pi_synthetic = _compute_pi_mass_time_series(
        synthetic,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="largest_component_lazy",
    )
    pi_merged = _merge_entity_time_series(pi_original, pi_synthetic, ["ts"], fill_value=None)
    pi_metrics = [column for column in pi_original.columns if column != "ts"]
    pi_summary = _summarise_metric_time_series(pi_merged, pi_metrics, treat_missing_as_zero=False)
    if not pi_merged.empty:
        details["pi_mass_per_snapshot"] = pi_merged
        details["pi_mass_summary"] = pi_summary
        pi_mass_rows = pi_summary.loc[pi_summary["metric"].astype(str).str.startswith("pi_mass__")]
        if len(pi_mass_rows):
            summary["pi_mass_mean_correlation"] = float(pd.to_numeric(pi_mass_rows["correlation"], errors="coerce").fillna(0.0).mean())
        summary["pi_gini_correlation"] = _metric_lookup(pi_summary, "pi_gini", "correlation") or 0.0
        summary["lic_share_active_correlation"] = _metric_lookup(pi_summary, "lic_share_active", "correlation") or 0.0
        summary["lic_size_correlation"] = _metric_lookup(pi_summary, "lic_size", "correlation") or 0.0
        summary["lic_active_node_count_correlation"] = _metric_lookup(pi_summary, "active_node_count", "correlation") or 0.0
        summary["lic_active_farm_count_correlation"] = _metric_lookup(pi_summary, "active_farm_count", "correlation") or 0.0
        summary["lic_active_region_count_correlation"] = _metric_lookup(pi_summary, "active_region_count", "correlation") or 0.0

    pi_closed_original = _compute_pi_mass_time_series(
        original,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="largest_closed_class_lazy",
    )
    pi_closed_synthetic = _compute_pi_mass_time_series(
        synthetic,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="largest_closed_class_lazy",
    )
    pi_closed_merged = _merge_entity_time_series(pi_closed_original, pi_closed_synthetic, ["ts"], fill_value=None)
    pi_closed_metrics = [column for column in pi_closed_original.columns if column != "ts"]
    pi_closed_summary = _summarise_metric_time_series(pi_closed_merged, pi_closed_metrics, treat_missing_as_zero=False)
    if not pi_closed_merged.empty:
        details["pi_mass_closed_per_snapshot"] = pi_closed_merged
        details["pi_mass_closed_summary"] = pi_closed_summary
        pi_mass_rows = pi_closed_summary.loc[pi_closed_summary["metric"].astype(str).str.startswith("pi_mass__")]
        if len(pi_mass_rows):
            summary["pi_mass_closed_mean_correlation"] = float(pd.to_numeric(pi_mass_rows["correlation"], errors="coerce").fillna(0.0).mean())
        summary["pi_mass_closed_gini_correlation"] = _metric_lookup(pi_closed_summary, "pi_gini", "correlation") or 0.0
        summary["closed_class_share_active_correlation"] = _metric_lookup(pi_closed_summary, "lic_share_active", "correlation") or 0.0
        summary["closed_class_size_correlation"] = _metric_lookup(pi_closed_summary, "lic_size", "correlation") or 0.0
        summary["closed_class_active_node_count_correlation"] = _metric_lookup(pi_closed_summary, "active_node_count", "correlation") or 0.0
        summary["closed_class_active_farm_count_correlation"] = _metric_lookup(pi_closed_summary, "active_farm_count", "correlation") or 0.0
        summary["closed_class_active_region_count_correlation"] = _metric_lookup(pi_closed_summary, "active_region_count", "correlation") or 0.0

    pi_pagerank_original = _compute_pi_mass_time_series(
        original,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="teleporting_pagerank",
    )
    pi_pagerank_synthetic = _compute_pi_mass_time_series(
        synthetic,
        node_universe=node_universe,
        node_types=node_types,
        directed=directed,
        weight_col=effective_weight_col,
        mode="teleporting_pagerank",
    )
    pi_pagerank_merged = _merge_entity_time_series(pi_pagerank_original, pi_pagerank_synthetic, ["ts"], fill_value=None)
    pi_pagerank_metrics = [column for column in pi_pagerank_original.columns if column != "ts"]
    pi_pagerank_summary = _summarise_metric_time_series(pi_pagerank_merged, pi_pagerank_metrics, treat_missing_as_zero=False)
    if not pi_pagerank_merged.empty:
        details["pi_mass_pagerank_per_snapshot"] = pi_pagerank_merged
        details["pi_mass_pagerank_summary"] = pi_pagerank_summary
        pi_mass_rows = pi_pagerank_summary.loc[pi_pagerank_summary["metric"].astype(str).str.startswith("pi_mass__")]
        if len(pi_mass_rows):
            summary["pi_mass_pagerank_mean_correlation"] = float(pd.to_numeric(pi_mass_rows["correlation"], errors="coerce").fillna(0.0).mean())
        summary["pi_mass_pagerank_gini_correlation"] = _metric_lookup(pi_pagerank_summary, "pi_gini", "correlation") or 0.0
        summary["pagerank_support_share_total_correlation"] = _metric_lookup(pi_pagerank_summary, "lic_share_total", "correlation") or 0.0
        summary["pagerank_support_size_correlation"] = _metric_lookup(pi_pagerank_summary, "lic_size", "correlation") or 0.0
        summary["pagerank_active_node_count_correlation"] = _metric_lookup(pi_pagerank_summary, "active_node_count", "correlation") or 0.0
        summary["pagerank_active_farm_count_correlation"] = _metric_lookup(pi_pagerank_summary, "active_farm_count", "correlation") or 0.0
        summary["pagerank_active_region_count_correlation"] = _metric_lookup(pi_pagerank_summary, "active_region_count", "correlation") or 0.0

    if skip_spectral_metrics:
        LOGGER.info("Skipping magnetic spectral diagnostics for this comparison.")
    else:
        magnetic_original = _compute_magnetic_spectrum_time_series(
            original,
            node_universe=node_universe,
            weight_col=effective_weight_col,
            directed=directed,
        )
        magnetic_synthetic = _compute_magnetic_spectrum_time_series(
            synthetic,
            node_universe=node_universe,
            weight_col=effective_weight_col,
            directed=directed,
        )
        magnetic_merged = _merge_entity_time_series(magnetic_original, magnetic_synthetic, ["ts"])
        magnetic_metrics = [column for column in magnetic_original.columns if column != "ts"]
        magnetic_summary = _summarise_metric_time_series(magnetic_merged, magnetic_metrics)
        if not magnetic_merged.empty:
            details["magnetic_laplacian_per_snapshot"] = magnetic_merged
            details["magnetic_laplacian_summary"] = magnetic_summary
            if len(magnetic_summary):
                summary["magnetic_spectrum_mean_correlation"] = float(pd.to_numeric(magnetic_summary["correlation"], errors="coerce").fillna(0.0).mean())
                summary["magnetic_spectrum_mean_abs_delta"] = float(pd.to_numeric(magnetic_summary["mean_abs_delta"], errors="coerce").fillna(0.0).mean())

        magnetic_distance = _compute_magnetic_spectral_distance_time_series(magnetic_original, magnetic_synthetic)
        magnetic_distance_summary = _summarise_distance_time_series(
            magnetic_distance,
            ["spectral_wasserstein_distance", "spectral_mean_abs_delta", "spectral_rmse", "spectral_max_abs_delta"],
        )
        if not magnetic_distance.empty:
            details["magnetic_spectral_distance_per_snapshot"] = magnetic_distance
            details["magnetic_spectral_distance_summary"] = magnetic_distance_summary
            summary["magnetic_spectral_wasserstein_mean"] = float(pd.to_numeric(magnetic_distance["spectral_wasserstein_distance"], errors="coerce").dropna().mean()) if len(magnetic_distance) else 0.0
            summary["magnetic_spectral_wasserstein_median"] = float(pd.to_numeric(magnetic_distance["spectral_wasserstein_distance"], errors="coerce").dropna().median()) if len(magnetic_distance) else 0.0
            summary["magnetic_spectral_wasserstein_max"] = float(pd.to_numeric(magnetic_distance["spectral_wasserstein_distance"], errors="coerce").dropna().max()) if len(magnetic_distance) else 0.0

    details["summary"] = pd.DataFrame([summary])
    if not node_blocks:
        LOGGER.debug("Skipping detailed block/node diagnostics because no node block mapping is available.")
        return {"per_snapshot": per_snapshot, "summary": summary, "details": details}

    block_pair_original = _aggregate_block_pair_time_series(original, directed=directed, node_blocks=node_blocks, weight_col=effective_weight_col)
    block_pair_synthetic = _aggregate_block_pair_time_series(synthetic, directed=directed, node_blocks=node_blocks, weight_col=effective_weight_col)
    block_pair_merged = _merge_entity_time_series(block_pair_original, block_pair_synthetic, ["ts", "block_u", "block_v"])
    block_pair_merged = _complete_entity_time_series(block_pair_merged, entity_keys=["block_u", "block_v"])
    block_pair_metric_names = ["edge_count"] + (["weight_total"] if effective_weight_col else [])
    block_pair_summary = _summarise_entity_time_series(block_pair_merged, ["block_u", "block_v"], block_pair_metric_names)

    original_with_blocks = _annotate_block_membership(original, node_blocks=node_blocks, directed=directed)
    synthetic_with_blocks = _annotate_block_membership(synthetic, node_blocks=node_blocks, directed=directed)
    block_activity_original = _aggregate_activity_time_series(
        original_with_blocks,
        directed=directed,
        entity_col="block_id",
        source_col="block_u",
        target_col="block_v",
        weight_col=effective_weight_col,
    )
    block_activity_synthetic = _aggregate_activity_time_series(
        synthetic_with_blocks,
        directed=directed,
        entity_col="block_id",
        source_col="block_u",
        target_col="block_v",
        weight_col=effective_weight_col,
    )
    block_activity_merged = _merge_entity_time_series(block_activity_original, block_activity_synthetic, ["ts", "block_id"])
    block_activity_merged = _complete_entity_time_series(block_activity_merged, entity_keys=["block_id"])
    block_activity_metric_names = ["incident_edge_count"] + (["out_edge_count", "in_edge_count"] if directed else []) + (["incident_weight_total"] if effective_weight_col else []) + (["out_weight_total", "in_weight_total"] if directed and effective_weight_col else [])
    block_activity_summary = _summarise_entity_time_series(block_activity_merged, ["block_id"], block_activity_metric_names)

    node_activity_original = _aggregate_activity_time_series(
        original,
        directed=directed,
        entity_col="node_id",
        source_col="u",
        target_col="i",
        weight_col=effective_weight_col,
    )
    node_activity_synthetic = _aggregate_activity_time_series(
        synthetic,
        directed=directed,
        entity_col="node_id",
        source_col="u",
        target_col="i",
        weight_col=effective_weight_col,
    )
    node_activity_merged = _merge_entity_time_series(node_activity_original, node_activity_synthetic, ["ts", "node_id"])
    node_activity_merged = _complete_entity_time_series(node_activity_merged, entity_keys=["node_id"])
    node_activity_metric_names = ["incident_edge_count"] + (["out_edge_count", "in_edge_count"] if directed else []) + (["incident_weight_total"] if effective_weight_col else []) + (["out_weight_total", "in_weight_total"] if directed and effective_weight_col else [])
    node_activity_summary = _summarise_entity_time_series(node_activity_merged, ["node_id"], node_activity_metric_names)
    node_activity_summary["block_id"] = node_activity_summary["node_id"].map(node_blocks).fillna(-1).astype(np.int64)

    details.update(
        {
            "block_pair_per_snapshot": block_pair_merged,
            "block_pair_summary": block_pair_summary,
            "block_activity_per_snapshot": block_activity_merged,
            "block_activity_summary": block_activity_summary,
            "node_activity_per_snapshot": node_activity_merged,
            "node_activity_summary": node_activity_summary,
        }
    )
    details["summary"] = pd.DataFrame([summary])
    LOGGER.debug(
        "Detailed panel diagnostics | block_pairs=%s | blocks=%s | nodes=%s",
        len(block_pair_summary),
        len(block_activity_summary),
        len(node_activity_summary),
    )
    return {"per_snapshot": per_snapshot, "summary": summary, "details": details}


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


def _write_sample_dashboard(
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
    ts_values = per_snapshot["ts"].to_numpy(dtype=float)
    directed = {
        "original_reciprocity",
        "synthetic_reciprocity",
        "original_active_source_node_count",
        "synthetic_active_source_node_count",
        "original_active_target_node_count",
        "synthetic_active_target_node_count",
    }.issubset(per_snapshot.columns)
    has_weight = "weight_column" in summary and "original_weight_total" in per_snapshot.columns
    posterior_runs = _posterior_run_count(summary)
    nrows = 3 if directed else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 12 if directed else 9), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Diagnostics Dashboard: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    ax = axes[0, 0]
    original_edge = per_snapshot["original_edge_count"].to_numpy(dtype=float)
    synthetic_edge = per_snapshot["synthetic_edge_count"].to_numpy(dtype=float)
    synthetic_edge_lower, synthetic_edge_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_edge_count")
    ax.fill_between(ts_values, original_edge, synthetic_edge, color=PLOT_COLORS["accent"], alpha=0.12, zorder=1)
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=original_edge,
        label="Original",
        color=PLOT_COLORS["original"],
        linewidth=2.35,
        marker_size=4.5,
        zorder=3,
    )
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=synthetic_edge,
        label=_posterior_band_label("Synthetic", posterior_runs),
        color=PLOT_COLORS["synthetic"],
        linewidth=2.35,
        marker_size=4.5,
        lower=synthetic_edge_lower,
        upper=synthetic_edge_upper,
        zorder=4,
    )
    ax.set_title("Snapshot Edge Counts")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Edges")
    _set_timestamp_ticks(ax, ts_values)
    _style_legend(ax.legend())

    ax = axes[0, 1]
    if has_weight:
        original_weight = per_snapshot["original_weight_total"].to_numpy(dtype=float)
        synthetic_weight = per_snapshot["synthetic_weight_total"].to_numpy(dtype=float)
        synthetic_weight_lower, synthetic_weight_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_weight_total")
        ax.fill_between(ts_values, original_weight, synthetic_weight, color=PLOT_COLORS["accent"], alpha=0.12, zorder=1)
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=original_weight,
            label="Original",
            color=PLOT_COLORS["original"],
            linewidth=2.35,
            marker_size=4.5,
            zorder=3,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=synthetic_weight,
            label=_posterior_band_label("Synthetic", posterior_runs),
            color=PLOT_COLORS["synthetic"],
            linewidth=2.35,
            marker_size=4.5,
            lower=synthetic_weight_lower,
            upper=synthetic_weight_upper,
            zorder=4,
        )
        ax.set_title(f"Snapshot Total Weight ({summary['weight_column']})")
        ax.set_ylabel("Total weight")
    elif directed:
        ax.plot(ts_values, per_snapshot["original_active_source_node_count"], marker="o", markersize=4.5, linewidth=2.2, color=PLOT_COLORS["original"], label="Original senders", zorder=3)
        ax.plot(ts_values, per_snapshot["synthetic_active_source_node_count"], marker="o", markersize=4.5, linewidth=2.2, color=PLOT_COLORS["synthetic"], label="Synthetic senders", zorder=4)
        ax.plot(ts_values, per_snapshot["original_active_target_node_count"], marker="o", markersize=4.5, linewidth=1.8, linestyle="--", color=PLOT_COLORS["original"], alpha=0.75, label="Original receivers", zorder=3)
        ax.plot(ts_values, per_snapshot["synthetic_active_target_node_count"], marker="o", markersize=4.5, linewidth=1.8, linestyle="--", color=PLOT_COLORS["synthetic"], alpha=0.75, label="Synthetic receivers", zorder=4)
        ax.set_title("Active Senders and Receivers")
        ax.set_ylabel("Nodes")
    else:
        original_nodes = per_snapshot["original_active_node_count"].to_numpy(dtype=float)
        synthetic_nodes = per_snapshot["synthetic_active_node_count"].to_numpy(dtype=float)
        synthetic_nodes_lower, synthetic_nodes_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_active_node_count")
        ax.fill_between(ts_values, original_nodes, synthetic_nodes, color=PLOT_COLORS["accent"], alpha=0.12, zorder=1)
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=original_nodes,
            label="Original",
            color=PLOT_COLORS["original"],
            linewidth=2.35,
            marker_size=4.5,
            zorder=3,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=synthetic_nodes,
            label=_posterior_band_label("Synthetic", posterior_runs),
            color=PLOT_COLORS["synthetic"],
            linewidth=2.35,
            marker_size=4.5,
            lower=synthetic_nodes_lower,
            upper=synthetic_nodes_upper,
            zorder=4,
        )
        ax.set_title("Snapshot Active Nodes")
        ax.set_ylabel("Active nodes")
    ax.set_xlabel("Timestamp")
    _set_timestamp_ticks(ax, ts_values)
    _style_legend(ax.legend())

    ax = axes[1, 0]
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=per_snapshot["edge_jaccard"].to_numpy(dtype=float),
        label="Edge Jaccard",
        color=PLOT_COLORS["original"],
        linewidth=2.3,
        marker_size=4.5,
        lower=_posterior_interval_from_frame(per_snapshot, "edge_jaccard")[0],
        upper=_posterior_interval_from_frame(per_snapshot, "edge_jaccard")[1],
    )
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=per_snapshot["node_jaccard"].to_numpy(dtype=float),
        label="Node Jaccard",
        color=PLOT_COLORS["accent"],
        linewidth=2.3,
        marker_size=4.5,
        lower=_posterior_interval_from_frame(per_snapshot, "node_jaccard")[0],
        upper=_posterior_interval_from_frame(per_snapshot, "node_jaccard")[1],
    )
    if "synthetic_novel_edge_rate" in per_snapshot.columns:
        novelty_lower, novelty_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_novel_edge_rate")
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["synthetic_novel_edge_rate"].to_numpy(dtype=float),
            label="Novel edge rate",
            color=PLOT_COLORS["novel"],
            linewidth=2.3,
            marker_size=4.5,
            lower=novelty_lower,
            upper=novelty_upper,
        )
    ax.set_title("Overlap and Novelty")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    _set_timestamp_ticks(ax, ts_values)
    _style_legend(ax.legend())

    ax = axes[1, 1]
    if directed:
        reciprocity_lower, reciprocity_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_reciprocity")
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["original_reciprocity"].to_numpy(dtype=float),
            label="Original",
            color=PLOT_COLORS["original"],
            linewidth=2.25,
            marker_size=4.5,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["synthetic_reciprocity"].to_numpy(dtype=float),
            label=_posterior_band_label("Synthetic", posterior_runs),
            color=PLOT_COLORS["synthetic"],
            linewidth=2.25,
            marker_size=4.5,
            lower=reciprocity_lower,
            upper=reciprocity_upper,
        )
        ax.set_title(
            "Reciprocity Through Time"
            + (
                f" (corr={float(summary.get('reciprocity_correlation', 0.0)):.3f})"
                if "reciprocity_correlation" in summary
                else ""
            )
        )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Reciprocity")
        ax.set_ylim(0.0, 1.05)
        _set_timestamp_ticks(ax, ts_values)
        _style_legend(ax.legend())
        ax = axes[2, 0]

    ax.axhline(0.0, color=PLOT_COLORS["neutral"], linewidth=1, alpha=0.8)
    edge_delta = per_snapshot["edge_count_delta"].to_numpy(dtype=float)
    edge_delta_lower, edge_delta_upper = _posterior_interval_from_frame(per_snapshot, "edge_count_delta")
    ax.bar(ts_values, edge_delta, width=0.66, alpha=0.78, color=PLOT_COLORS["accent"], label="Edge-count delta")
    if edge_delta_lower is not None and edge_delta_upper is not None:
        ax.vlines(ts_values, edge_delta_lower, edge_delta_upper, color=PLOT_COLORS["text"], linewidth=1.0, alpha=0.55, zorder=4)
    if "weight_total_delta" in per_snapshot.columns:
        twin = ax.twinx()
        _style_axis(twin, grid_axis="x")
        weight_delta_lower, weight_delta_upper = _posterior_interval_from_frame(per_snapshot, "weight_total_delta")
        _plot_line_with_band(
            twin,
            ts_values=ts_values,
            values=per_snapshot["weight_total_delta"].to_numpy(dtype=float),
            label="Weight-total delta",
            color=PLOT_COLORS["delta"],
            linewidth=2.15,
            marker_size=4.3,
            lower=weight_delta_lower,
            upper=weight_delta_upper,
        )
        twin.set_ylabel("Weight-total delta")
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        _style_legend(ax.legend(handles + twin_handles, labels + twin_labels, loc="upper right"))
    else:
        _style_legend(ax.legend(loc="upper right"))
    ax.set_title("Snapshot Deltas")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Edge-count delta")
    _set_timestamp_ticks(ax, ts_values)

    if directed:
        ax = axes[2, 1]
        sender_lower, sender_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_active_source_node_count")
        receiver_lower, receiver_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_active_target_node_count")
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["original_active_source_node_count"].to_numpy(dtype=float),
            label="Original senders",
            color=PLOT_COLORS["original"],
            linewidth=2.1,
            marker_size=4.2,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["synthetic_active_source_node_count"].to_numpy(dtype=float),
            label=_posterior_band_label("Synthetic senders", posterior_runs),
            color=PLOT_COLORS["synthetic"],
            linewidth=2.1,
            marker_size=4.2,
            lower=sender_lower,
            upper=sender_upper,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["original_active_target_node_count"].to_numpy(dtype=float),
            label="Original receivers",
            color=PLOT_COLORS["original"],
            linewidth=1.8,
            marker_size=4.0,
            linestyle="--",
            alpha=0.75,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=per_snapshot["synthetic_active_target_node_count"].to_numpy(dtype=float),
            label=_posterior_band_label("Synthetic receivers", posterior_runs),
            color=PLOT_COLORS["synthetic"],
            linewidth=1.8,
            marker_size=4.0,
            linestyle="--",
            alpha=0.75,
            lower=receiver_lower,
            upper=receiver_upper,
        )
        ax.set_title("Sender / Receiver Coverage")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Nodes")
        _set_timestamp_ticks(ax, ts_values)
        _style_legend(ax.legend(loc="best"))

    output_path = output_dir / f"{sample_label}_dashboard.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_sample_parity_plot(
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
    has_weight = "weight_column" in summary and "original_weight_total" in per_snapshot.columns
    directed = {
        "original_reciprocity",
        "synthetic_reciprocity",
        "original_active_source_node_count",
        "synthetic_active_source_node_count",
        "original_active_target_node_count",
        "synthetic_active_target_node_count",
    }.issubset(per_snapshot.columns)
    panel_count = 1 + int(has_weight) + (3 if directed else 0)
    if panel_count <= 2:
        fig, axes = plt.subplots(1, panel_count, figsize=(12 if panel_count == 2 else 6.5, 5), constrained_layout=True)
    elif panel_count <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), constrained_layout=True)
    _style_figure(fig, axes)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Parity Checks: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    edge_ax = axes[0]
    edge_x = per_snapshot["original_edge_count"].to_numpy(dtype=float)
    edge_y = per_snapshot["synthetic_edge_count"].to_numpy(dtype=float)
    edge_lower, edge_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_edge_count")
    edge_ax.scatter(edge_x, edge_y, alpha=0.92, s=62, color=PLOT_COLORS["original"], edgecolors="white", linewidths=0.8)
    if edge_lower is not None and edge_upper is not None:
        edge_ax.vlines(edge_x, edge_lower, edge_upper, color=PLOT_COLORS["grid_strong"], linewidth=1.0, alpha=0.8, zorder=1)
    edge_limit = max(float(edge_x.max(initial=0.0)), float(edge_y.max(initial=0.0)), 1.0)
    edge_ax.plot([0.0, edge_limit], [0.0, edge_limit], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
    edge_ax.set_title(f"Edge Counts (corr={summary['edge_count_correlation']:.3f})")
    edge_ax.set_xlabel("Original")
    edge_ax.set_ylabel("Synthetic")
    edge_ax.set_xlim(0.0, edge_limit * 1.03)
    edge_ax.set_ylim(0.0, edge_limit * 1.03)

    axis_index = 1
    if has_weight:
        weight_ax = axes[axis_index]
        weight_x = per_snapshot["original_weight_total"].to_numpy(dtype=float)
        weight_y = per_snapshot["synthetic_weight_total"].to_numpy(dtype=float)
        weight_lower, weight_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_weight_total")
        weight_ax.scatter(weight_x, weight_y, alpha=0.92, s=62, color=PLOT_COLORS["synthetic"], edgecolors="white", linewidths=0.8)
        if weight_lower is not None and weight_upper is not None:
            weight_ax.vlines(weight_x, weight_lower, weight_upper, color=PLOT_COLORS["grid_strong"], linewidth=1.0, alpha=0.8, zorder=1)
        weight_limit = max(float(weight_x.max(initial=0.0)), float(weight_y.max(initial=0.0)), 1.0)
        weight_ax.plot([0.0, weight_limit], [0.0, weight_limit], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        weight_ax.set_title(f"Weight Totals (corr={summary['weight_total_correlation']:.3f})")
        weight_ax.set_xlabel("Original")
        weight_ax.set_ylabel("Synthetic")
        weight_ax.set_xlim(0.0, weight_limit * 1.03)
        weight_ax.set_ylim(0.0, weight_limit * 1.03)
        axis_index += 1

    if directed:
        reciprocity_ax = axes[axis_index]
        reciprocity_x = per_snapshot["original_reciprocity"].to_numpy(dtype=float)
        reciprocity_y = per_snapshot["synthetic_reciprocity"].to_numpy(dtype=float)
        reciprocity_lower, reciprocity_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_reciprocity")
        reciprocity_ax.scatter(reciprocity_x, reciprocity_y, alpha=0.92, s=62, color=PLOT_COLORS["accent"], edgecolors="white", linewidths=0.8)
        if reciprocity_lower is not None and reciprocity_upper is not None:
            reciprocity_ax.vlines(reciprocity_x, reciprocity_lower, reciprocity_upper, color=PLOT_COLORS["grid_strong"], linewidth=1.0, alpha=0.8, zorder=1)
        reciprocity_ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        reciprocity_ax.set_title(f"Reciprocity (corr={float(summary.get('reciprocity_correlation', 0.0)):.3f})")
        reciprocity_ax.set_xlabel("Original")
        reciprocity_ax.set_ylabel("Synthetic")
        reciprocity_ax.set_xlim(0.0, 1.03)
        reciprocity_ax.set_ylim(0.0, 1.03)
        axis_index += 1

        sender_ax = axes[axis_index]
        sender_x = per_snapshot["original_active_source_node_count"].to_numpy(dtype=float)
        sender_y = per_snapshot["synthetic_active_source_node_count"].to_numpy(dtype=float)
        sender_lower, sender_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_active_source_node_count")
        sender_ax.scatter(sender_x, sender_y, alpha=0.92, s=62, color=PLOT_COLORS["novel"], edgecolors="white", linewidths=0.8)
        if sender_lower is not None and sender_upper is not None:
            sender_ax.vlines(sender_x, sender_lower, sender_upper, color=PLOT_COLORS["grid_strong"], linewidth=1.0, alpha=0.8, zorder=1)
        sender_limit = max(float(sender_x.max(initial=0.0)), float(sender_y.max(initial=0.0)), 1.0)
        sender_ax.plot([0.0, sender_limit], [0.0, sender_limit], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        sender_ax.set_title("Active Senders")
        sender_ax.set_xlabel("Original")
        sender_ax.set_ylabel("Synthetic")
        sender_ax.set_xlim(0.0, sender_limit * 1.03)
        sender_ax.set_ylim(0.0, sender_limit * 1.03)
        axis_index += 1

        receiver_ax = axes[axis_index]
        receiver_x = per_snapshot["original_active_target_node_count"].to_numpy(dtype=float)
        receiver_y = per_snapshot["synthetic_active_target_node_count"].to_numpy(dtype=float)
        receiver_lower, receiver_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_active_target_node_count")
        receiver_ax.scatter(receiver_x, receiver_y, alpha=0.92, s=62, color=PLOT_COLORS["delta"], edgecolors="white", linewidths=0.8)
        if receiver_lower is not None and receiver_upper is not None:
            receiver_ax.vlines(receiver_x, receiver_lower, receiver_upper, color=PLOT_COLORS["grid_strong"], linewidth=1.0, alpha=0.8, zorder=1)
        receiver_limit = max(float(receiver_x.max(initial=0.0)), float(receiver_y.max(initial=0.0)), 1.0)
        receiver_ax.plot([0.0, receiver_limit], [0.0, receiver_limit], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        receiver_ax.set_title("Active Receivers")
        receiver_ax.set_xlabel("Original")
        receiver_ax.set_ylabel("Synthetic")
        receiver_ax.set_xlim(0.0, receiver_limit * 1.03)
        receiver_ax.set_ylim(0.0, receiver_limit * 1.03)
        axis_index += 1

    for ax in axes[axis_index:]:
        ax.axis("off")

    output_path = output_dir / f"{sample_label}_parity.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def write_all_samples_overview(summary_rows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_rows.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None
    from matplotlib.transforms import blended_transform_factory

    summary_rows = summary_rows.copy()
    summary_rows["sample_mode"], summary_rows["rewire_mode"] = zip(*summary_rows["sample_label"].map(_parse_sample_label_parts))
    summary_rows = summary_rows.sort_values(
        ["mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    labels = summary_rows["sample_label"].astype(str).tolist()
    display_labels = [_overview_setting_label(label) for label in labels]
    positions = np.arange(len(summary_rows), dtype=float)

    metrics = [
        ("mean_snapshot_edge_jaccard", "Mean edge Jaccard"),
        ("mean_snapshot_node_jaccard", "Mean node Jaccard"),
        ("mean_synthetic_novel_edge_rate", "Novel edge rate"),
        ("edge_count_correlation", "Edge-count correlation"),
    ]
    if "weight_total_correlation" in summary_rows.columns:
        metrics.append(("weight_total_correlation", "Weight-total correlation"))
    if "reciprocity_correlation" in summary_rows.columns:
        metrics.append(("reciprocity_correlation", "Reciprocity correlation"))
    if "tea_new_ratio_correlation" in summary_rows.columns:
        metrics.append(("tea_new_ratio_correlation", "TEA new-ratio corr"))
    if "tna_new_ratio_correlation" in summary_rows.columns:
        metrics.append(("tna_new_ratio_correlation", "TNA new-ratio corr"))
    if "pi_mass_mean_correlation" in summary_rows.columns:
        metrics.append(("pi_mass_mean_correlation", "Pi-mass corr"))
    if "magnetic_spectrum_mean_correlation" in summary_rows.columns:
        metrics.append(("magnetic_spectrum_mean_correlation", "Mag spectrum corr"))

    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    fig_height = max(4.3 * nrows, 0.48 * len(summary_rows) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, fig_height), constrained_layout=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle("Diagnostics Across Generated Samples", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    for ax, (column, title) in zip(axes_array, metrics):
        values = pd.to_numeric(summary_rows[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        lower = pd.to_numeric(summary_rows.get(f"{column}_q05", pd.Series(np.nan, index=summary_rows.index)), errors="coerce").to_numpy(dtype=float)
        upper = pd.to_numeric(summary_rows.get(f"{column}_q95", pd.Series(np.nan, index=summary_rows.index)), errors="coerce").to_numpy(dtype=float)
        ax.hlines(positions, 0.0, values, color=PLOT_COLORS["grid_strong"], linewidth=2.0, zorder=1)
        if np.isfinite(lower).any() and np.isfinite(upper).any():
            ax.hlines(positions, lower, upper, color=PLOT_COLORS["accent"], linewidth=4.4, alpha=0.45, zorder=2)
        ax.scatter(values, positions, s=68, color=PLOT_COLORS["original"], edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_yticks(positions)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        if "correlation" in column or "rate" in column or "jaccard" in column:
            finite_candidates = [values[np.isfinite(values)]]
            if np.isfinite(lower).any():
                finite_candidates.append(lower[np.isfinite(lower)])
            if np.isfinite(upper).any():
                finite_candidates.append(upper[np.isfinite(upper)])
            finite_joined = np.concatenate([candidate.astype(float, copy=False) for candidate in finite_candidates if candidate.size])
            x_data_min = float(finite_joined.min()) if finite_joined.size else 0.0
            x_data_max = float(finite_joined.max()) if finite_joined.size else 1.0
            x_left = min(-0.92, x_data_min - 0.18)
            x_right = max(1.05, x_data_max + 0.05)
            ax.set_xlim(x_left, x_right)
            ax.axvline(0.0, color=PLOT_COLORS["grid_strong"], linewidth=1.0, zorder=0)
        ax.grid(axis="x", color=PLOT_COLORS["grid"], alpha=0.85)
        ax.set_axisbelow(True)
        label_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for label_y, label_text in zip(positions, display_labels):
            ax.text(
                0.02,
                label_y - 0.22,
                label_text,
                transform=label_transform,
                ha="left",
                va="bottom",
                fontsize=7.4,
                color=PLOT_COLORS["text"],
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.24,rounding_size=0.45",
                    "facecolor": "#f8fbff",
                    "edgecolor": PLOT_COLORS["grid_strong"],
                    "linewidth": 0.8,
                    "alpha": 0.97,
                },
            )

    for ax in axes_array[len(metrics):]:
        ax.axis("off")

    output_path = Path(output_dir) / "all_samples_overview.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _entity_label(row: pd.Series, entity_kind: str, directed: bool) -> str:
    if entity_kind == "block_pair":
        connector = "->" if directed else "-"
        return f"B{int(row['block_u'])}{connector}B{int(row['block_v'])}"
    if entity_kind == "block_activity":
        return f"B{int(row['block_id'])}"
    if entity_kind == "node_activity":
        if "block_id" in row and int(row["block_id"]) >= 0:
            return f"N{int(row['node_id'])} (B{int(row['block_id'])})"
        return f"N{int(row['node_id'])}"
    return str(entity_kind)


def _write_entity_metric_grid(
    per_snapshot: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
    entity_kind: str,
    directed: bool,
    top_k: int,
    original_metric: str,
    synthetic_metric: str,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    if per_snapshot.empty or summary.empty or top_k <= 0:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    top_summary = summary.head(top_k).copy().reset_index(drop=True)
    top_summary["entity_label"] = top_summary.apply(lambda row: _entity_label(row, entity_kind=entity_kind, directed=directed), axis=1)

    key_columns = {
        "block_pair": ["block_u", "block_v"],
        "block_activity": ["block_id"],
        "node_activity": ["node_id"],
    }[entity_kind]

    n_panels = len(top_summary)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), constrained_layout=True, sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    _style_figure(fig, axes_array)
    fig.suptitle(f"{title}: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

    for ax, (_, entity_row) in zip(axes_array, top_summary.iterrows()):
        mask = pd.Series(True, index=per_snapshot.index)
        for key_column in key_columns:
            mask &= per_snapshot[key_column] == entity_row[key_column]
        entity_frame = per_snapshot.loc[mask].sort_values("ts")
        ts_values = entity_frame["ts"].to_numpy(dtype=float)
        original_values = entity_frame[original_metric].to_numpy(dtype=float)
        synthetic_values = entity_frame[synthetic_metric].to_numpy(dtype=float)
        synthetic_lower, synthetic_upper = _posterior_interval_from_frame(entity_frame, synthetic_metric)
        ax.fill_between(ts_values, original_values, synthetic_values, color=PLOT_COLORS["accent"], alpha=0.1, zorder=1)
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=original_values,
            label="Original",
            color=PLOT_COLORS["original"],
            linewidth=2.05,
            marker_size=3.8,
            zorder=3,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=synthetic_values,
            label="Synthetic",
            color=PLOT_COLORS["synthetic"],
            linewidth=2.05,
            marker_size=3.8,
            lower=synthetic_lower,
            upper=synthetic_upper,
            zorder=4,
        )
        ax.set_title(entity_row["entity_label"])
        ax.set_ylabel(ylabel)
        if len(ts_values):
            _set_timestamp_ticks(ax, ts_values)
        metric_name = original_metric.removeprefix("original_")
        corr_column = f"{metric_name}_correlation"
        if corr_column in entity_row:
            ax.text(
                0.02,
                0.95,
                f"corr={float(entity_row[corr_column]):.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8.7,
                color=PLOT_COLORS["text"],
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": PLOT_COLORS["grid_strong"], "linewidth": 0.8},
            )

    for ax in axes_array[n_panels:]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        _style_legend(fig.legend(handles, labels, loc="upper right"))

    output_path = output_dir / f"{sample_label}_{entity_kind}_{original_metric.removeprefix('original_')}.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _temporal_transition_limits(merged: pd.DataFrame) -> tuple[float, float]:
    pos_max = 0.0
    churn_max = 0.0
    for prefix in ("original", "synthetic"):
        persist = pd.to_numeric(merged.get(f"{prefix}_persist_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reactivated = pd.to_numeric(merged.get(f"{prefix}_reactivated_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        new = pd.to_numeric(merged.get(f"{prefix}_new_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        churn = pd.to_numeric(merged.get(f"{prefix}_ceased_prev_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(persist):
            pos_max = max(pos_max, float(np.nanmax(persist + reactivated + new)))
        if len(churn):
            churn_max = max(churn_max, float(np.nanmax(churn)))
    pad = 0.08
    return -churn_max * (1.0 + pad), max(1.0, pos_max * (1.0 + pad))


def _transition_theme(figure_name: str) -> dict[str, str]:
    return ADVANCED_TEMPORAL_THEMES.get(str(figure_name).lower(), ADVANCED_TEMPORAL_COLORS)


def _smooth_line(values: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size <= 2 or sigma <= 0:
        return values
    try:
        from scipy.ndimage import gaussian_filter1d

        finite = np.isfinite(values)
        if finite.all():
            return gaussian_filter1d(values, sigma=sigma, mode="nearest")
        if finite.any():
            filled = values.copy()
            filled[~finite] = float(np.nanmean(values[finite]))
            smoothed = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
            smoothed[~finite] = np.nan
            return smoothed
        return values
    except Exception:
        kernel = np.array([1.0, 1.0, 1.0], dtype=float) / 3.0
        finite = np.isfinite(values)
        if not finite.any():
            return values
        filled = values.copy()
        filled[~finite] = float(np.nanmean(values[finite]))
        smoothed = np.convolve(filled, kernel, mode="same")
        smoothed[~finite] = np.nan
        return smoothed


def _plot_transition_stack_panel(
    ax,
    merged: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    ylabel: str,
    y_limits: tuple[float, float],
    theme: dict[str, str],
    show_legend: bool = False,
) -> None:
    ts_values = merged["ts"].to_numpy(dtype=float)
    persist = pd.to_numeric(merged.get(f"{prefix}_persist_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    reactivated = pd.to_numeric(merged.get(f"{prefix}_reactivated_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    new = pd.to_numeric(merged.get(f"{prefix}_new_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    churn = pd.to_numeric(merged.get(f"{prefix}_ceased_prev_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    base_zero = np.zeros(len(ts_values), dtype=float)
    base_one = base_zero + persist
    base_two = base_one + reactivated
    ax.bar(ts_values, persist, bottom=base_zero, width=0.9, color=theme["persist"], edgecolor="white", linewidth=0.4, label="Persist")
    ax.bar(ts_values, reactivated, bottom=base_one, width=0.9, color=theme["reactivated"], edgecolor="white", linewidth=0.4, label="Reactivated")
    ax.bar(ts_values, new, bottom=base_two, width=0.9, color=theme["new"], edgecolor="white", linewidth=0.4, label="New")
    if len(churn):
        ax.bar(ts_values, -churn, width=0.9, color=theme["churn"], edgecolor="white", linewidth=0.4, alpha=0.92, label="Churn")
    ax.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.65)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*y_limits)
    if len(ts_values):
        ax.set_xlim(float(np.nanmin(ts_values)) - 0.7, float(np.nanmax(ts_values)) + 0.7)
        _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="y")
    if show_legend:
        _style_legend(ax.legend(loc="upper right", ncol=4))


def _plot_ratio_comparison_panel(ax, merged: pd.DataFrame, *, title: str, theme: dict[str, str]) -> None:
    ts_values = merged["ts"].to_numpy(dtype=float)
    metric_specs = [
        ("new_ratio", "New", theme["new"]),
        ("persist_ratio", "Persist", theme["persist"]),
        ("reactivated_ratio", "Reactivated", theme["reactivated"]),
        ("churn_ratio", "Churn", theme["churn"]),
    ]
    for metric_name, label, color in metric_specs:
        original = pd.to_numeric(merged.get(f"original_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        synthetic = pd.to_numeric(merged.get(f"synthetic_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        synthetic_lower, synthetic_upper = _posterior_interval_from_frame(merged, f"synthetic_{metric_name}")
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=original,
            label=f"{label} (Obs)",
            color=color,
            linewidth=2.15,
            marker_size=3.8,
            zorder=3,
        )
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=synthetic,
            label=f"{label} (Syn)",
            color=color,
            linewidth=1.9,
            marker_size=3.6,
            linestyle="--",
            alpha=0.92,
            lower=synthetic_lower,
            upper=synthetic_upper,
            zorder=4,
        )
    ax.set_title(title, loc="left")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 1.05)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="upper right", ncol=4))


def _plot_type_fit_sidebar(
    ax,
    *,
    type_summary: Optional[pd.DataFrame],
    type_label_column: Optional[str],
    summary_df: pd.DataFrame,
    directed: bool,
) -> None:
    focus = None
    title = "Hybrid type fit"
    corr_column = "correlation"
    delta_column = None

    if type_summary is not None and not type_summary.empty:
        frame = type_summary.copy()
        if {"source_type", "target_type"}.issubset(frame.columns):
            frame["display_label"] = frame.apply(
                lambda row: _display_type_pair_label(row["source_type"], row["target_type"], directed=directed),
                axis=1,
            )
            if "new_ratio_correlation" in frame.columns:
                corr_column = "new_ratio_correlation"
                delta_column = "mean_abs_new_ratio_delta" if "mean_abs_new_ratio_delta" in frame.columns else None
            else:
                corr_column = "birth_rate_correlation" if "birth_rate_correlation" in frame.columns else "edge_share_correlation"
                delta_column = "mean_abs_birth_rate_delta" if "mean_abs_birth_rate_delta" in frame.columns else "mean_abs_edge_share_delta"
            title = "Hybrid channel fit"
        else:
            label_source = frame[type_label_column] if type_label_column and type_label_column in frame.columns else pd.Series(["Unknown"] * len(frame))
            frame["display_label"] = label_source.map(_format_node_type_label)
            if "new_ratio_correlation" in frame.columns:
                corr_column = "new_ratio_correlation"
                delta_column = "mean_abs_new_ratio_delta" if "mean_abs_new_ratio_delta" in frame.columns else None
            else:
                corr_column = "new_rate_correlation" if "new_rate_correlation" in frame.columns else "active_count_correlation"
                delta_column = "mean_abs_new_rate_delta" if "mean_abs_new_rate_delta" in frame.columns else "mean_abs_active_count_delta"
            title = "Node-type fit"
        if corr_column in frame.columns:
            focus = frame[["display_label", corr_column] + ([delta_column] if delta_column and delta_column in frame.columns else [])].copy()
            focus = focus.sort_values(corr_column, ascending=True).reset_index(drop=True)

    if focus is None or focus.empty:
        focus = summary_df.loc[
            summary_df["metric"].isin(["new_ratio", "persist_ratio", "reactivated_ratio", "churn_ratio"]),
            ["metric", "correlation", "mean_abs_delta"],
        ].copy()
        if focus.empty:
            ax.axis("off")
            return
        focus["display_label"] = focus["metric"].map(_display_advanced_metric_name)
        focus = focus.rename(columns={"mean_abs_delta": "delta_value"})
        corr_column = "correlation"
        delta_column = "delta_value"
        title = "Summary correlations"

    values = pd.to_numeric(focus[corr_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positions = np.arange(len(focus), dtype=float)
    ax.barh(positions, values, color=PLOT_COLORS["accent"], alpha=0.92)
    ax.set_yticks(positions, focus["display_label"].astype(str).tolist())
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Correlation")
    ax.set_title(title, loc="left")
    for position, (_, row) in zip(positions, focus.iterrows()):
        corr_value = float(pd.to_numeric(pd.Series([row[corr_column]]), errors="coerce").fillna(0.0).iloc[0])
        annotation = f"{corr_value:.2f}"
        if delta_column and delta_column in focus.columns:
            delta_value = pd.to_numeric(pd.Series([row[delta_column]]), errors="coerce").iloc[0]
            if pd.notna(delta_value):
                annotation = f"{annotation}  Δ={float(delta_value):.2g}"
        ax.text(min(corr_value + 0.02, 0.99), position, annotation, va="center", ha="left", fontsize=8.2, color=PLOT_COLORS["text"])
    _style_axis(ax, grid_axis="x")


def _write_temporal_dynamics_figure(
    merged: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
    figure_name: str,
    title: str,
    ylabel: str,
    directed: bool,
    type_summary: Optional[pd.DataFrame] = None,
    type_label_column: Optional[str] = None,
) -> Optional[Path]:
    if merged.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    theme = _transition_theme(figure_name)
    fig = plt.figure(figsize=(17.0, 11.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.78], height_ratios=[1, 1, 1])
    ax_observed = fig.add_subplot(gs[0, :5])
    ax_synthetic = fig.add_subplot(gs[1, :5])
    ax_ratio = fig.add_subplot(gs[2, :5])
    ax_side = fig.add_subplot(gs[:, 5])
    _style_figure(fig, [ax_observed, ax_synthetic, ax_ratio, ax_side])
    fig.suptitle(f"{title}: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])
    y_limits = _temporal_transition_limits(merged)
    _plot_transition_stack_panel(
        ax_observed,
        merged,
        prefix="original",
        title="Observed transitions",
        ylabel=ylabel,
        y_limits=y_limits,
        theme=theme,
        show_legend=True,
    )
    _plot_transition_stack_panel(
        ax_synthetic,
        merged,
        prefix="synthetic",
        title="Synthetic transitions",
        ylabel=ylabel,
        y_limits=y_limits,
        theme=theme,
    )
    _plot_ratio_comparison_panel(ax_ratio, merged, title="Ratio comparison", theme=theme)
    _plot_type_fit_sidebar(
        ax_side,
        type_summary=type_summary,
        type_label_column=type_label_column,
        summary_df=summary_df,
        directed=directed,
    )
    ax_observed.tick_params(axis="x", which="both", labelbottom=False)
    ax_synthetic.tick_params(axis="x", which="both", labelbottom=False)

    output_path = output_dir / f"{sample_label}_{figure_name}.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _plot_pi_mass_type_panel(ax, merged: pd.DataFrame, *, type_metrics: list[str]) -> None:
    ts_values = merged["ts"].to_numpy(dtype=float)
    no_data_y = 0.5
    no_data_color = "#9aa5b1"
    missing_mask = np.zeros(len(ts_values), dtype=bool)
    observed_missing_mask = np.ones(len(ts_values), dtype=bool)
    synthetic_missing_mask = np.ones(len(ts_values), dtype=bool)

    for metric_name in type_metrics:
        label = _display_advanced_metric_name(metric_name).removeprefix("Pi-Mass (").removesuffix(")")
        color = PI_MASS_TYPE_COLORS.get(label, PLOT_COLORS["accent"])
        observed = pd.to_numeric(merged.get(f"original_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        synthetic = pd.to_numeric(merged.get(f"synthetic_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        synthetic_lower, synthetic_upper = _posterior_interval_from_frame(merged, f"synthetic_{metric_name}")
        observed_missing_mask &= ~np.isfinite(observed)
        synthetic_missing_mask &= ~np.isfinite(synthetic)
        obs_mask = np.isfinite(observed)
        syn_mask = np.isfinite(synthetic)
        if obs_mask.any():
            ax.plot(ts_values, observed, color="#8f9aa6", linestyle="--", linewidth=1.2, alpha=0.45, zorder=1)
            ax.scatter(ts_values[obs_mask], observed[obs_mask], s=28, color=color, edgecolors="white", linewidths=0.6, zorder=3, label=f"{label} (Obs)")
        if syn_mask.any():
            if synthetic_lower is not None and synthetic_upper is not None:
                finite_band = syn_mask & np.isfinite(synthetic_lower) & np.isfinite(synthetic_upper)
                if finite_band.any():
                    ax.fill_between(
                        ts_values[finite_band],
                        synthetic_lower[finite_band],
                        synthetic_upper[finite_band],
                        color=color,
                        alpha=0.14,
                        linewidth=0.0,
                        zorder=2,
                    )
            ax.plot(ts_values, synthetic, color="#657381", linestyle=":", linewidth=1.15, alpha=0.45, zorder=1)
            ax.scatter(ts_values[syn_mask], synthetic[syn_mask], s=34, color=color, marker="s", edgecolors="#20303f", linewidths=0.4, zorder=4, label=f"{label} (Syn)")

    missing_mask = observed_missing_mask | synthetic_missing_mask
    if missing_mask.any():
        ax.scatter(
            ts_values[missing_mask],
            np.full(missing_mask.sum(), no_data_y),
            marker="x",
            s=34,
            color=no_data_color,
            alpha=0.9,
            linewidths=1.1,
            zorder=5,
            label="No data",
        )

    ax.set_title("Pi-Mass by hybrid node type", loc="left")
    ax.set_ylabel("Stationary mass")
    ax.set_ylim(-0.08, 1.08)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        _style_legend(ax.legend(unique.values(), unique.keys(), loc="upper right", ncol=3))


def _plot_series_compare_panel(
    ax,
    *,
    ts_values: np.ndarray,
    observed: np.ndarray,
    synthetic: np.ndarray,
    synthetic_lower: Optional[np.ndarray] = None,
    synthetic_upper: Optional[np.ndarray] = None,
    title: str,
    ylabel: str,
    observed_label: str = "Observed",
    synthetic_label: str = "Synthetic",
    observed_color: str = PLOT_COLORS["original"],
    synthetic_color: str = PLOT_COLORS["synthetic"],
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=np.asarray(observed, dtype=float),
        label=observed_label,
        color=observed_color,
        linewidth=2.15,
        marker_size=3.8,
        zorder=3,
    )
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=np.asarray(synthetic, dtype=float),
        label=synthetic_label,
        color=synthetic_color,
        linewidth=1.95,
        marker_size=3.6,
        linestyle="--",
        alpha=0.92,
        lower=synthetic_lower,
        upper=synthetic_upper,
        zorder=4,
    )
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if ymin is not None or ymax is not None:
        current = ax.get_ylim()
        ax.set_ylim(current[0] if ymin is None else ymin, current[1] if ymax is None else ymax)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))


def _plot_summary_corr_sidebar(
    ax,
    summary_df: pd.DataFrame,
    *,
    metric_order: list[str],
    title: str,
) -> None:
    focus = summary_df.loc[summary_df["metric"].isin(metric_order)].copy()
    if focus.empty:
        ax.axis("off")
        return
    focus["display_metric"] = focus["metric"].map(_display_advanced_metric_name)
    focus = focus.sort_values("correlation", ascending=True).reset_index(drop=True)
    values = pd.to_numeric(focus["correlation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lower = pd.to_numeric(focus["correlation_q05"], errors="coerce").to_numpy(dtype=float) if "correlation_q05" in focus.columns else np.full(len(focus), np.nan, dtype=float)
    upper = pd.to_numeric(focus["correlation_q95"], errors="coerce").to_numpy(dtype=float) if "correlation_q95" in focus.columns else np.full(len(focus), np.nan, dtype=float)
    positions = np.arange(len(focus), dtype=float)
    ax.barh(positions, values, color=PLOT_COLORS["accent"], alpha=0.92)
    if np.isfinite(lower).any() and np.isfinite(upper).any():
        ax.hlines(positions, lower, upper, color=PLOT_COLORS["text"], linewidth=1.1, alpha=0.55)
    ax.set_yticks(positions, focus["display_metric"].tolist())
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Correlation")
    ax.set_title(title, loc="left")
    for index, (position, value) in enumerate(zip(positions, values)):
        if np.isfinite(lower[index]) and np.isfinite(upper[index]):
            label = f"{value:.2f} [{lower[index]:.2f}, {upper[index]:.2f}]"
        else:
            label = f"{value:.2f}"
        ax.text(min(value + 0.02, 0.99), position, label, va="center", ha="left", fontsize=8.2, color=PLOT_COLORS["text"])
    _style_axis(ax, grid_axis="x")



def _write_pi_mass_figure(
    merged: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
    title_prefix: str = "Stationary Mass / LIC Diagnostics",
    support_title: str = "Largest strongly connected component size",
    share_title: str = "LIC concentration and dispersion",
    share_metric_key: str = "lic_share_active",
    filename_suffix: str = "pi_mass",
    summary_metric_tail: Optional[list[str]] = None,
) -> Optional[Path]:
    if merged.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    fig = plt.figure(figsize=(17.0, 14.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1, 0.78], height_ratios=[1.0, 1.0, 0.96, 1.0])
    ax_pi = fig.add_subplot(gs[0, :5])
    ax_support = fig.add_subplot(gs[1, :5])
    activity_gs = gs[2, :5].subgridspec(1, 3, wspace=0.14)
    ax_active_total = fig.add_subplot(activity_gs[0, 0])
    ax_active_farm = fig.add_subplot(activity_gs[0, 1])
    ax_active_region = fig.add_subplot(activity_gs[0, 2])
    ax_concentration = fig.add_subplot(gs[3, :5])
    ax_side = fig.add_subplot(gs[:, 5])
    _style_figure(fig, [ax_pi, ax_support, ax_active_total, ax_active_farm, ax_active_region, ax_concentration, ax_side])
    fig.suptitle(f"{title_prefix}: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])
    ts_values = merged["ts"].to_numpy(dtype=float)
    type_metrics = [row.metric for row in summary_df.itertuples(index=False) if str(row.metric).startswith("pi_mass__")]

    def _plot_activity_panel(ax, metric_name: str, title: str) -> None:
        observed = pd.to_numeric(merged.get(f"original_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        synthetic = pd.to_numeric(merged.get(f"synthetic_{metric_name}", np.nan), errors="coerce").to_numpy(dtype=float)
        lower, upper = _posterior_interval_from_frame(merged, f"synthetic_{metric_name}")
        if not (np.isfinite(observed).any() or np.isfinite(synthetic).any()):
            ax.text(0.5, 0.5, "No node-type metadata available", ha="center", va="center", transform=ax.transAxes, color=PLOT_COLORS["muted"])
            ax.set_title(title, loc="left")
            _style_axis(ax, grid_axis="both")
            return
        _plot_series_compare_panel(
            ax,
            ts_values=ts_values,
            observed=observed,
            synthetic=synthetic,
            synthetic_lower=lower,
            synthetic_upper=upper,
            title=title,
            ylabel="Nodes",
            ymin=0.0,
        )

    if type_metrics:
        _plot_pi_mass_type_panel(ax_pi, merged, type_metrics=type_metrics)
    else:
        ax_pi.text(0.5, 0.5, "No node-type metadata available", ha="center", va="center", transform=ax_pi.transAxes, color=PLOT_COLORS["muted"])
        ax_pi.set_title("Pi-Mass by hybrid node type", loc="left")
        _style_axis(ax_pi, grid_axis="both")

    _plot_series_compare_panel(
        ax_support,
        ts_values=ts_values,
        observed=pd.to_numeric(merged.get("original_lic_size", np.nan), errors="coerce").to_numpy(dtype=float),
        synthetic=pd.to_numeric(merged.get("synthetic_lic_size", np.nan), errors="coerce").to_numpy(dtype=float),
        synthetic_lower=_posterior_interval_from_frame(merged, "synthetic_lic_size")[0],
        synthetic_upper=_posterior_interval_from_frame(merged, "synthetic_lic_size")[1],
        title=support_title,
        ylabel="Nodes",
    )

    _plot_activity_panel(ax_active_total, "active_node_count", "Active nodes per day")
    _plot_activity_panel(ax_active_farm, "active_farm_count", "Active farm nodes per day")
    _plot_activity_panel(ax_active_region, "active_region_count", "Active regional supernodes per day")

    _plot_series_compare_panel(
        ax_concentration,
        ts_values=ts_values,
        observed=pd.to_numeric(merged.get(f"original_{share_metric_key}", np.nan), errors="coerce").to_numpy(dtype=float),
        synthetic=pd.to_numeric(merged.get(f"synthetic_{share_metric_key}", np.nan), errors="coerce").to_numpy(dtype=float),
        synthetic_lower=_posterior_interval_from_frame(merged, f"synthetic_{share_metric_key}")[0],
        synthetic_upper=_posterior_interval_from_frame(merged, f"synthetic_{share_metric_key}")[1],
        title=share_title,
        ylabel="Value",
        observed_label=f"{_display_advanced_metric_name(share_metric_key)} (Obs)",
        synthetic_label=f"{_display_advanced_metric_name(share_metric_key)} (Syn)",
        observed_color=PLOT_COLORS["accent"],
        synthetic_color=PLOT_COLORS["accent"],
        ymin=0.0,
        ymax=1.05,
    )
    ax_concentration.plot(
        ts_values,
        pd.to_numeric(merged.get("original_pi_gini", np.nan), errors="coerce").to_numpy(dtype=float),
        color=PLOT_COLORS["delta"],
        linewidth=2.0,
        marker="o",
        markersize=3.6,
        label="Pi gini (Obs)",
    )
    _plot_line_with_band(
        ax_concentration,
        ts_values=ts_values,
        values=pd.to_numeric(merged.get("synthetic_pi_gini", np.nan), errors="coerce").to_numpy(dtype=float),
        label="Pi gini (Syn)",
        color=PLOT_COLORS["delta"],
        linewidth=1.8,
        marker_size=3.4,
        linestyle="--",
        alpha=0.92,
        lower=_posterior_interval_from_frame(merged, "synthetic_pi_gini")[0],
        upper=_posterior_interval_from_frame(merged, "synthetic_pi_gini")[1],
    )
    _style_legend(ax_concentration.legend(loc="best"))

    tail = summary_metric_tail or ["lic_size", "active_node_count", "active_farm_count", "active_region_count", share_metric_key, "pi_gini"]
    _plot_summary_corr_sidebar(
        ax_side,
        summary_df,
        metric_order=type_metrics + tail,
        title="Summary correlations",
    )

    output_path = output_dir / f"{sample_label}_{filename_suffix}.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _base_magnetic_mode_names(merged: pd.DataFrame) -> list[str]:
    """Return base magnetic eigenvalue columns such as eig_1 and eig_2.

    Posterior summary fields such as eig_1_q05 and eig_1_mean are excluded.
    """
    base_pattern = re.compile(r"^original_eig_(\d+)$")
    indexed_modes: list[tuple[int, str]] = []
    for column in merged.columns:
        match = base_pattern.match(str(column))
        if not match:
            continue
        indexed_modes.append((int(match.group(1)), f"eig_{int(match.group(1))}"))
    indexed_modes.sort(key=lambda item: item[0])
    return [mode_name for _, mode_name in indexed_modes]


def _prepare_magnetic_matrices(merged: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    mode_suffixes = _base_magnetic_mode_names(merged)
    ts_values = merged["ts"].to_numpy(dtype=float)
    if not mode_suffixes:
        return ts_values, [], np.empty((0, len(ts_values)), dtype=float), np.empty((0, len(ts_values)), dtype=float)

    observed = np.vstack(
        [
            pd.to_numeric(merged[f"original_{metric_name}"], errors="coerce").to_numpy(dtype=float)
            for metric_name in mode_suffixes
        ]
    )
    synthetic = np.vstack(
        [
            pd.to_numeric(merged[f"synthetic_{metric_name}"], errors="coerce").to_numpy(dtype=float)
            for metric_name in mode_suffixes
        ]
    )
    return ts_values, mode_suffixes, observed, synthetic


def _finite_min_max(values: list[np.ndarray], *, symmetric: bool = False, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite_blocks = [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values if np.isfinite(value).any()]
    if not finite_blocks:
        return default
    pool = np.concatenate(finite_blocks)
    if symmetric:
        extent = float(np.nanpercentile(np.abs(pool), 99.0))
        extent = max(extent, 1e-6)
        return -extent, extent
    lo = float(np.nanmin(pool))
    hi = float(np.nanmax(pool))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return default
    return lo, hi


def _draw_heatmap_with_sideplots(
    ax,
    *,
    matrix: np.ndarray,
    ts_values: np.ndarray,
    panel_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    overlay_matrix: Optional[np.ndarray] = None,
    top_limits: Optional[tuple[float, float]] = None,
    right_limits: Optional[tuple[float, float]] = None,
    smooth_sigma: float = 1.0,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="18%", pad=0.1, sharex=ax)
    ax_right = divider.append_axes("right", size="14%", pad=0.1, sharey=ax)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[float(np.nanmin(ts_values)) - 0.5, float(np.nanmax(ts_values)) + 0.5, 0.5, matrix.shape[0] + 0.5],
        interpolation="nearest",
    )
    ax.set_title("  ")
    ax.set_ylabel(panel_label)
    ax.set_xlabel(None)
    ax.set_yticks(np.arange(1, matrix.shape[0] + 1), [str(index) for index in range(1, matrix.shape[0] + 1)])
    _set_timestamp_ticks(ax, ts_values, show_calendar_bands=False)
    _style_axis(ax, grid_axis="y")

    col_means = _smooth_line(np.nanmean(matrix, axis=0), sigma=smooth_sigma)
    row_means = _smooth_line(np.nanmean(matrix, axis=1), sigma=smooth_sigma)
    _add_calendar_bands(ax_top, ts_values)
    ax_top.plot(ts_values, col_means, linewidth=1.2, color=PLOT_COLORS["original"])
    if overlay_matrix is not None:
        ax_top.plot(ts_values, _smooth_line(np.nanmean(overlay_matrix, axis=0), sigma=smooth_sigma), linewidth=1.1, linestyle="--", alpha=0.6, color="gray")
    ax_top.set_xlim(float(np.nanmin(ts_values)), float(np.nanmax(ts_values)))
    if top_limits is not None:
        ax_top.set_ylim(*top_limits)
    ax_top.tick_params(axis="x", which="both", labelbottom=False, length=0)
    _style_axis(ax_top, grid_axis="y")

    y_index = np.arange(1, matrix.shape[0] + 1, dtype=float)
    ax_right.plot(row_means, y_index, linewidth=1.2, color=PLOT_COLORS["original"])
    if overlay_matrix is not None:
        ax_right.plot(_smooth_line(np.nanmean(overlay_matrix, axis=1), sigma=smooth_sigma), y_index, linewidth=1.1, linestyle="--", alpha=0.6, color="gray")
    ax_right.set_ylim(0.5, matrix.shape[0] + 0.5)
    if right_limits is not None:
        ax_right.set_xlim(*right_limits)
    ax_right.tick_params(axis="y", which="both", labelleft=False, length=0)
    _style_axis(ax_right, grid_axis="x")

    return image


def _write_edge_type_figure(
    per_snapshot: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
    directed: bool,
) -> Optional[Path]:
    if per_snapshot.empty or summary_df.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    fig = plt.figure(figsize=(17.0, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.78], height_ratios=[1, 1])
    ax_edge = fig.add_subplot(gs[0, :5])
    ax_weight = fig.add_subplot(gs[1, :5])
    ax_side = fig.add_subplot(gs[:, 5])
    _style_figure(fig, [ax_edge, ax_weight, ax_side])
    fig.suptitle(f"Hybrid Edge-Type Fit: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    pair_palette = {
        pair: color
        for pair, color in zip(
            ["F→F", "F→R", "R→F", "R→R", "F–F", "F–R", "R–R"],
            ["#35c9c3", "#5d8ad1", "#ef8f7d", "#7a6fd6", "#35c9c3", "#5d8ad1", "#ef8f7d"],
        )
    }
    frame = per_snapshot.copy()
    frame["edge_type"] = frame.apply(
        lambda row: _display_type_pair_label(row["source_type"], row["target_type"], directed=directed),
        axis=1,
    )
    ts_values = frame["ts"].to_numpy(dtype=float)
    for edge_type, group in frame.groupby("edge_type", sort=True):
        color = pair_palette.get(edge_type, PLOT_COLORS["accent"])
        group = group.sort_values("ts")
        ts_group = group["ts"].to_numpy(dtype=float)
        _plot_line_with_band(
            ax_edge,
            ts_values=ts_group,
            values=group["original_edge_share"].to_numpy(dtype=float),
            label=f"{edge_type} (Obs)",
            color=color,
            linewidth=2.0,
            marker_size=3.4,
        )
        _plot_line_with_band(
            ax_edge,
            ts_values=ts_group,
            values=group["synthetic_edge_share"].to_numpy(dtype=float),
            label=f"{edge_type} (Syn)",
            color=color,
            linewidth=1.8,
            marker_size=3.2,
            linestyle="--",
            alpha=0.9,
            lower=_posterior_interval_from_frame(group, "synthetic_edge_share")[0],
            upper=_posterior_interval_from_frame(group, "synthetic_edge_share")[1],
        )
        if {"original_weight_share", "synthetic_weight_share"}.issubset(group.columns):
            _plot_line_with_band(
                ax_weight,
                ts_values=ts_group,
                values=group["original_weight_share"].to_numpy(dtype=float),
                label=f"{edge_type} (Obs)",
                color=color,
                linewidth=2.0,
                marker_size=3.4,
            )
            _plot_line_with_band(
                ax_weight,
                ts_values=ts_group,
                values=group["synthetic_weight_share"].to_numpy(dtype=float),
                label=f"{edge_type} (Syn)",
                color=color,
                linewidth=1.8,
                marker_size=3.2,
                linestyle="--",
                alpha=0.9,
                lower=_posterior_interval_from_frame(group, "synthetic_weight_share")[0],
                upper=_posterior_interval_from_frame(group, "synthetic_weight_share")[1],
            )

    ax_edge.set_title("Edge-share trajectories", loc="left")
    ax_edge.set_ylabel("Share of edges")
    ax_edge.set_ylim(0.0, 1.02)
    _set_timestamp_ticks(ax_edge, ts_values)
    _style_axis(ax_edge, grid_axis="both")
    _style_legend(ax_edge.legend(loc="upper right", ncol=4))

    ax_weight.set_title("Weight-share trajectories", loc="left")
    ax_weight.set_ylabel("Share of total weight")
    ax_weight.set_ylim(0.0, 1.02)
    _set_timestamp_ticks(ax_weight, ts_values)
    _style_axis(ax_weight, grid_axis="both")
    _style_legend(ax_weight.legend(loc="upper right", ncol=4))

    focus = summary_df.copy()
    focus["edge_type"] = focus.apply(
        lambda row: _display_type_pair_label(row["source_type"], row["target_type"], directed=directed),
        axis=1,
    )
    focus = focus.sort_values("edge_share_correlation", ascending=True).reset_index(drop=True)
    values = pd.to_numeric(focus["edge_share_correlation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positions = np.arange(len(focus), dtype=float)
    ax_side.barh(positions, values, color=PLOT_COLORS["accent"], alpha=0.92)
    ax_side.set_yticks(positions, focus["edge_type"].tolist())
    ax_side.set_xlim(0.0, 1.02)
    ax_side.set_xlabel("Correlation")
    ax_side.set_title("Channel summary", loc="left")
    for position, value in zip(positions, values):
        ax_side.text(min(value + 0.02, 0.99), position, f"{value:.2f}", va="center", ha="left", fontsize=8.2, color=PLOT_COLORS["text"])
    _style_axis(ax_side, grid_axis="x")

    output_path = output_dir / f"{sample_label}_edge_type.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_temporal_reachability_figure(
    per_snapshot: pd.DataFrame,
    source_summary: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if per_snapshot.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Temporal Reachability / Transmission Potential: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    ts_values = per_snapshot["ts"].to_numpy(dtype=float)

    ax = axes[0, 0]
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=pd.to_numeric(per_snapshot.get("original_reachability_ratio", np.nan), errors="coerce").to_numpy(dtype=float),
        label="Observed",
        color=PLOT_COLORS["original"],
        linewidth=2.2,
        marker_size=4.0,
    )
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=pd.to_numeric(per_snapshot.get("synthetic_reachability_ratio", np.nan), errors="coerce").to_numpy(dtype=float),
        label="Synthetic",
        color=PLOT_COLORS["synthetic"],
        linewidth=2.0,
        marker_size=3.8,
        linestyle="--",
        alpha=0.92,
        lower=_posterior_interval_from_frame(per_snapshot, "synthetic_reachability_ratio")[0],
        upper=_posterior_interval_from_frame(per_snapshot, "synthetic_reachability_ratio")[1],
    )
    ax.set_title("Reachability ratio through time", loc="left")
    ax.set_ylabel("Fraction of ordered pairs")
    ax.set_ylim(0.0, 1.05)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))
    annotation_lines = []
    if summary.get("original_static_reachability_ratio") is not None and summary.get("synthetic_static_reachability_ratio") is not None:
        annotation_lines.append(
            f"Static reach: {float(summary.get('original_static_reachability_ratio', 0.0)):.3f} / {float(summary.get('synthetic_static_reachability_ratio', 0.0)):.3f}"
        )
    if summary.get("original_causal_fidelity") is not None and summary.get("synthetic_causal_fidelity") is not None:
        annotation_lines.append(
            f"Causal fidelity: {float(summary.get('original_causal_fidelity', np.nan)):.3f} / {float(summary.get('synthetic_causal_fidelity', np.nan)):.3f}"
        )
    if annotation_lines:
        ax.text(
            0.02,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8.7,
            color=PLOT_COLORS["text"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": PLOT_COLORS["grid_strong"], "linewidth": 0.8},
        )

    ax = axes[0, 1]
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=pd.to_numeric(per_snapshot.get("original_temporal_efficiency", np.nan), errors="coerce").to_numpy(dtype=float),
        label="Observed",
        color=PLOT_COLORS["accent"],
        linewidth=2.2,
        marker_size=4.0,
    )
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=pd.to_numeric(per_snapshot.get("synthetic_temporal_efficiency", np.nan), errors="coerce").to_numpy(dtype=float),
        label="Synthetic",
        color=PLOT_COLORS["delta"],
        linewidth=2.0,
        marker_size=3.8,
        linestyle="--",
        alpha=0.92,
        lower=_posterior_interval_from_frame(per_snapshot, "synthetic_temporal_efficiency")[0],
        upper=_posterior_interval_from_frame(per_snapshot, "synthetic_temporal_efficiency")[1],
    )
    ax.set_title("Temporal efficiency through time", loc="left")
    ax.set_ylabel("Average inverse arrival time")
    ax.set_ylim(bottom=0.0)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))
    if summary.get("original_mean_arrival_time_reached") is not None and summary.get("synthetic_mean_arrival_time_reached") is not None:
        ax.text(
            0.02,
            0.96,
            (
                f"Final mean arrival time: "
                f"{float(summary.get('original_mean_arrival_time_reached', np.nan)):.2f} / "
                f"{float(summary.get('synthetic_mean_arrival_time_reached', np.nan)):.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8.7,
            color=PLOT_COLORS["text"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": PLOT_COLORS["grid_strong"], "linewidth": 0.8},
        )

    ax = axes[1, 0]
    observed_new_pairs = pd.to_numeric(per_snapshot.get("original_new_reachable_pair_count", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    synthetic_new_pairs = pd.to_numeric(per_snapshot.get("synthetic_new_reachable_pair_count", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    synthetic_lower, synthetic_upper = _posterior_interval_from_frame(per_snapshot, "synthetic_new_reachable_pair_count")
    width = 0.36 if len(ts_values) else 0.36
    ax.bar(ts_values - width / 2.0, observed_new_pairs, width=width, color=PLOT_COLORS["original"], alpha=0.82, label="Observed")
    ax.bar(ts_values + width / 2.0, synthetic_new_pairs, width=width, color=PLOT_COLORS["synthetic"], alpha=0.74, label="Synthetic")
    if synthetic_lower is not None and synthetic_upper is not None:
        ax.vlines(ts_values + width / 2.0, synthetic_lower, synthetic_upper, color=PLOT_COLORS["text"], linewidth=1.0, alpha=0.5, zorder=4)
    ax.set_title("Newly reachable ordered pairs", loc="left")
    ax.set_ylabel("Pairs first reached")
    ax.set_ylim(bottom=0.0)
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))

    ax = axes[1, 1]
    if source_summary is None or source_summary.empty or "original_forward_reach_ratio" not in source_summary.columns or "synthetic_forward_reach_ratio" not in source_summary.columns:
        ax.axis("off")
    else:
        original_values = pd.to_numeric(source_summary["original_forward_reach_ratio"], errors="coerce").to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(source_summary["synthetic_forward_reach_ratio"], errors="coerce").to_numpy(dtype=float)
        lower, upper = _posterior_interval_from_frame(source_summary, "synthetic_forward_reach_ratio")
        ax.scatter(original_values, synthetic_values, s=48, color=PLOT_COLORS["accent"], edgecolors="white", linewidths=0.8, alpha=0.92)
        if lower is not None and upper is not None:
            ax.vlines(original_values, lower, upper, color=PLOT_COLORS["grid_strong"], linewidth=0.9, alpha=0.75, zorder=1)
        limit = max(
            float(np.nanmax(original_values)) if np.isfinite(original_values).any() else 0.0,
            float(np.nanmax(synthetic_values)) if np.isfinite(synthetic_values).any() else 0.0,
            1.0,
        )
        ax.plot([0.0, limit], [0.0, limit], linestyle="--", color=PLOT_COLORS["neutral"], linewidth=1.2)
        correlation_value = float(summary.get("temporal_forward_reach_node_correlation", 0.0) or 0.0)
        ax.set_title(f"Final forward reach by source (corr={correlation_value:.3f})", loc="left")
        ax.set_xlabel("Observed forward reachable fraction")
        ax.set_ylabel("Synthetic forward reachable fraction")
        ax.set_xlim(0.0, limit * 1.03)
        ax.set_ylim(0.0, limit * 1.03)
        _style_axis(ax, grid_axis="both")

    output_path = output_dir / f"{sample_label}_temporal_reachability.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path



def _write_magnetic_laplacian_figure(
    merged: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if merged.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    ts_values, mode_names, observed, synthetic = _prepare_magnetic_matrices(merged)
    if not mode_names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 5.8), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Magnetic Laplacian Eigenvalue Panels: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    vmin, vmax = _finite_min_max([observed, synthetic], default=(0.0, 1.0))
    top_min, top_max = _finite_min_max([np.nanmean(observed, axis=0), np.nanmean(synthetic, axis=0)], default=(0.0, 1.0))
    right_min, right_max = _finite_min_max([np.nanmean(observed, axis=1), np.nanmean(synthetic, axis=1)], default=(0.0, 1.0))
    top_pad = 0.05 * max(top_max - top_min, 1e-6)
    right_pad = 0.05 * max(right_max - right_min, 1e-6)
    image = _draw_heatmap_with_sideplots(
        axes[0],
        matrix=observed,
        ts_values=ts_values,
        panel_label="Observed",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        top_limits=(top_min - top_pad, top_max + top_pad),
        right_limits=(right_min - right_pad, right_max + right_pad),
    )
    _draw_heatmap_with_sideplots(
        axes[1],
        matrix=synthetic,
        ts_values=ts_values,
        panel_label="Synthetic",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        overlay_matrix=observed,
        top_limits=(top_min - top_pad, top_max + top_pad),
        right_limits=(right_min - right_pad, right_max + right_pad),
    )
    cbar = fig.colorbar(image, ax=np.atleast_1d(axes).ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("eigenvalue (L)")
    cbar.ax.tick_params(labelsize=8, colors=PLOT_COLORS["muted"])

    output_path = output_dir / f"{sample_label}_magnetic_laplacian.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_magnetic_laplacian_diff_figure(
    merged: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if merged.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    ts_values, mode_names, observed, synthetic = _prepare_magnetic_matrices(merged)
    if not mode_names:
        return None
    zero_baseline = observed - observed
    difference = synthetic - observed
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 5.8), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Magnetic Laplacian Difference Panels: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    vmin, vmax = _finite_min_max([difference], symmetric=True, default=(-1.0, 1.0))
    top_limits = _finite_min_max([np.nanmean(zero_baseline, axis=0), np.nanmean(difference, axis=0)], symmetric=True, default=(-1.0, 1.0))
    right_limits = _finite_min_max([np.nanmean(zero_baseline, axis=1), np.nanmean(difference, axis=1)], symmetric=True, default=(-1.0, 1.0))
    image = _draw_heatmap_with_sideplots(
        axes[0],
        matrix=zero_baseline,
        ts_values=ts_values,
        panel_label="Observed – observed",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        top_limits=top_limits,
        right_limits=right_limits,
    )
    _draw_heatmap_with_sideplots(
        axes[1],
        matrix=difference,
        ts_values=ts_values,
        panel_label="Synthetic – observed",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        top_limits=top_limits,
        right_limits=right_limits,
    )
    cbar = fig.colorbar(image, ax=np.atleast_1d(axes).ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Δ eigenvalue (synthetic − observed)")
    cbar.ax.tick_params(labelsize=8, colors=PLOT_COLORS["muted"])

    output_path = output_dir / f"{sample_label}_magnetic_laplacian_diff.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path



def _write_magnetic_spectral_distance_figure(
    distance_frame: pd.DataFrame,
    *,
    output_dir: Path,
    sample_label: str,
) -> Optional[Path]:
    if distance_frame.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    frame = distance_frame.sort_values("ts").reset_index(drop=True)
    ts_values = frame["ts"].to_numpy(dtype=float)
    wasserstein = pd.to_numeric(frame.get("spectral_wasserstein_distance", np.nan), errors="coerce").to_numpy(dtype=float)
    mae = pd.to_numeric(frame.get("spectral_mean_abs_delta", np.nan), errors="coerce").to_numpy(dtype=float)
    rmse = pd.to_numeric(frame.get("spectral_rmse", np.nan), errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle(f"Magnetic Spectral Distance Diagnostics: {sample_label}", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    ax = axes[0]
    _plot_line_with_band(
        ax,
        ts_values=ts_values,
        values=wasserstein,
        label="Wasserstein distance",
        color=PLOT_COLORS["original"],
        linewidth=2.25,
        marker_size=4.0,
        lower=_posterior_interval_from_frame(frame, "spectral_wasserstein_distance")[0],
        upper=_posterior_interval_from_frame(frame, "spectral_wasserstein_distance")[1],
    )
    ax.set_title("Per-snapshot spectral Wasserstein distance", loc="left")
    ax.set_ylabel("Distance")
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))

    ax = axes[1]
    if np.isfinite(mae).any():
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=mae,
            label="Mean abs delta",
            color=PLOT_COLORS["accent"],
            linewidth=2.05,
            marker_size=3.8,
            lower=_posterior_interval_from_frame(frame, "spectral_mean_abs_delta")[0],
            upper=_posterior_interval_from_frame(frame, "spectral_mean_abs_delta")[1],
        )
    if np.isfinite(rmse).any():
        _plot_line_with_band(
            ax,
            ts_values=ts_values,
            values=rmse,
            label="RMSE",
            color=PLOT_COLORS["delta"],
            linewidth=1.95,
            marker_size=3.8,
            linestyle="--",
            lower=_posterior_interval_from_frame(frame, "spectral_rmse")[0],
            upper=_posterior_interval_from_frame(frame, "spectral_rmse")[1],
        )
    ax.set_title("Per-snapshot aligned-mode errors", loc="left")
    ax.set_ylabel("Error")
    _set_timestamp_ticks(ax, ts_values)
    _style_axis(ax, grid_axis="both")
    _style_legend(ax.legend(loc="best"))

    output_path = output_dir / f"{sample_label}_magnetic_spectral_distance.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path

def _write_detailed_diagnostics_artifacts(
    details: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    sample_label: str,
    directed: bool,
    top_k: int,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    table_specs = {
        "block_pair_per_snapshot": f"{sample_label}_block_pair_per_snapshot.csv",
        "block_pair_summary": f"{sample_label}_block_pair_summary.csv",
        "block_activity_per_snapshot": f"{sample_label}_block_activity_per_snapshot.csv",
        "block_activity_summary": f"{sample_label}_block_activity_summary.csv",
        "node_activity_per_snapshot": f"{sample_label}_node_activity_per_snapshot.csv",
        "node_activity_summary": f"{sample_label}_node_activity_summary.csv",
        "edge_type_per_snapshot": f"{sample_label}_edge_type_per_snapshot.csv",
        "edge_type_summary": f"{sample_label}_edge_type_summary.csv",
        "tea_per_snapshot": f"{sample_label}_tea_per_snapshot.csv",
        "tea_summary": f"{sample_label}_tea_summary.csv",
        "tea_type_pair_per_snapshot": f"{sample_label}_tea_type_pair_per_snapshot.csv",
        "tea_type_pair_summary": f"{sample_label}_tea_type_pair_summary.csv",
        "tna_per_snapshot": f"{sample_label}_tna_per_snapshot.csv",
        "tna_summary": f"{sample_label}_tna_summary.csv",
        "tna_type_per_snapshot": f"{sample_label}_tna_type_per_snapshot.csv",
        "tna_type_summary": f"{sample_label}_tna_type_summary.csv",
        "pi_mass_per_snapshot": f"{sample_label}_pi_mass_per_snapshot.csv",
        "pi_mass_summary": f"{sample_label}_pi_mass_summary.csv",
        "pi_mass_closed_per_snapshot": f"{sample_label}_pi_mass_closed_per_snapshot.csv",
        "pi_mass_closed_summary": f"{sample_label}_pi_mass_closed_summary.csv",
        "pi_mass_pagerank_per_snapshot": f"{sample_label}_pi_mass_pagerank_per_snapshot.csv",
        "pi_mass_pagerank_summary": f"{sample_label}_pi_mass_pagerank_summary.csv",
        "temporal_reachability_per_snapshot": f"{sample_label}_temporal_reachability_per_snapshot.csv",
        "temporal_reachability_summary": f"{sample_label}_temporal_reachability_summary.csv",
        "temporal_reachability_source_summary": f"{sample_label}_temporal_reachability_source_summary.csv",
        "magnetic_laplacian_per_snapshot": f"{sample_label}_magnetic_laplacian_per_snapshot.csv",
        "magnetic_laplacian_summary": f"{sample_label}_magnetic_laplacian_summary.csv",
        "magnetic_spectral_distance_per_snapshot": f"{sample_label}_magnetic_spectral_distance_per_snapshot.csv",
        "magnetic_spectral_distance_summary": f"{sample_label}_magnetic_spectral_distance_summary.csv",
    }
    for key, filename in table_specs.items():
        frame = details.get(key)
        if frame is None or frame.empty:
            continue
        output_path = output_dir / filename
        frame.to_csv(output_path, index=False)
        outputs[key] = str(output_path)

    plot_specs = [
        ("block_pair_per_snapshot", "block_pair_summary", "block_pair", "original_edge_count", "synthetic_edge_count", "Block-Pair Edge Counts", "Edges"),
        ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_incident_edge_count", "synthetic_incident_edge_count", "Block Incident Edge Counts", "Incident edges"),
        ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_incident_edge_count", "synthetic_incident_edge_count", "Node Incident Edge Counts", "Incident edges"),
    ]
    if directed:
        plot_specs.extend(
            [
                ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_out_edge_count", "synthetic_out_edge_count", "Block Outgoing Edge Counts", "Outgoing edges"),
                ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_in_edge_count", "synthetic_in_edge_count", "Block Incoming Edge Counts", "Incoming edges"),
                ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_out_edge_count", "synthetic_out_edge_count", "Node Outgoing Edge Counts", "Outgoing edges"),
                ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_in_edge_count", "synthetic_in_edge_count", "Node Incoming Edge Counts", "Incoming edges"),
            ]
        )
    for per_snapshot_key, summary_key, entity_kind, original_metric, synthetic_metric, title, ylabel in plot_specs:
        per_snapshot = details.get(per_snapshot_key)
        summary = details.get(summary_key)
        if per_snapshot is None or summary is None:
            continue
        output_path = _write_entity_metric_grid(
            per_snapshot=per_snapshot,
            summary=summary,
            output_dir=output_dir,
            sample_label=sample_label,
            entity_kind=entity_kind,
            directed=directed,
            top_k=top_k,
            original_metric=original_metric,
            synthetic_metric=synthetic_metric,
            title=title,
            ylabel=ylabel,
        )
        if output_path is not None:
            metric_suffix = original_metric.removeprefix("original_")
            outputs[f"{entity_kind}_{metric_suffix}_plot"] = str(output_path)
            if entity_kind == "block_pair" and metric_suffix == "edge_count":
                outputs["block_pair_edge_plot"] = str(output_path)
            if entity_kind == "node_activity" and metric_suffix == "incident_edge_count":
                outputs["node_activity_edge_plot"] = str(output_path)

    weighted_plot_specs = [
        ("block_pair_per_snapshot", "block_pair_summary", "block_pair", "original_weight_total", "synthetic_weight_total", "Block-Pair Total Weights", "Total weight"),
        ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_incident_weight_total", "synthetic_incident_weight_total", "Block Incident Weights", "Incident weight"),
        ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_incident_weight_total", "synthetic_incident_weight_total", "Node Incident Weights", "Incident weight"),
    ]
    if directed:
        weighted_plot_specs.extend(
            [
                ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_out_weight_total", "synthetic_out_weight_total", "Block Outgoing Weights", "Outgoing weight"),
                ("block_activity_per_snapshot", "block_activity_summary", "block_activity", "original_in_weight_total", "synthetic_in_weight_total", "Block Incoming Weights", "Incoming weight"),
                ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_out_weight_total", "synthetic_out_weight_total", "Node Outgoing Weights", "Outgoing weight"),
                ("node_activity_per_snapshot", "node_activity_summary", "node_activity", "original_in_weight_total", "synthetic_in_weight_total", "Node Incoming Weights", "Incoming weight"),
            ]
        )
    for per_snapshot_key, summary_key, entity_kind, original_metric, synthetic_metric, title, ylabel in weighted_plot_specs:
        per_snapshot = details.get(per_snapshot_key)
        summary = details.get(summary_key)
        if per_snapshot is None or summary is None or original_metric not in per_snapshot.columns:
            continue
        output_path = _write_entity_metric_grid(
            per_snapshot=per_snapshot,
            summary=summary,
            output_dir=output_dir,
            sample_label=sample_label,
            entity_kind=entity_kind,
            directed=directed,
            top_k=top_k,
            original_metric=original_metric,
            synthetic_metric=synthetic_metric,
            title=title,
            ylabel=ylabel,
        )
        if output_path is not None:
            metric_suffix = original_metric.removeprefix("original_")
            outputs[f"{entity_kind}_{metric_suffix}_plot"] = str(output_path)
            if entity_kind == "block_pair" and metric_suffix == "weight_total":
                outputs["block_pair_weight_plot"] = str(output_path)
            if entity_kind == "node_activity" and metric_suffix == "incident_weight_total":
                outputs["node_activity_weight_plot"] = str(output_path)

    tea_plot_path = _write_temporal_dynamics_figure(
        details.get("tea_per_snapshot", pd.DataFrame()),
        details.get("tea_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        figure_name="tea",
        title="Temporal Edge Appearance (TEA)",
        ylabel="Edge count",
        directed=directed,
        type_summary=details.get("tea_type_pair_summary"),
        type_label_column="source_type",
    )
    if tea_plot_path is not None:
        outputs["tea_plot"] = str(tea_plot_path)

    tna_plot_path = _write_temporal_dynamics_figure(
        details.get("tna_per_snapshot", pd.DataFrame()),
        details.get("tna_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        figure_name="tna",
        title="Temporal Node Appearance (TNA)",
        ylabel="Node count",
        directed=directed,
        type_summary=details.get("tna_type_summary"),
        type_label_column="type_label",
    )
    if tna_plot_path is not None:
        outputs["tna_plot"] = str(tna_plot_path)

    edge_type_plot_path = _write_edge_type_figure(
        details.get("edge_type_per_snapshot", pd.DataFrame()),
        details.get("edge_type_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        directed=directed,
    )
    if edge_type_plot_path is not None:
        outputs["edge_type_plot"] = str(edge_type_plot_path)

    temporal_reachability_plot_path = _write_temporal_reachability_figure(
        details.get("temporal_reachability_per_snapshot", pd.DataFrame()),
        details.get("temporal_reachability_source_summary", pd.DataFrame()),
        details.get("summary", pd.DataFrame([{}])).iloc[0].to_dict() if isinstance(details.get("summary"), pd.DataFrame) and len(details.get("summary")) else {},
        output_dir=output_dir,
        sample_label=sample_label,
    )
    if temporal_reachability_plot_path is not None:
        outputs["temporal_reachability_plot"] = str(temporal_reachability_plot_path)

    pi_mass_plot_path = _write_pi_mass_figure(
        details.get("pi_mass_per_snapshot", pd.DataFrame()),
        details.get("pi_mass_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        title_prefix="Stationary Mass / Largest-SCC Lazy Walk",
        support_title="Largest strongly connected component size",
        share_title="LIC concentration and dispersion",
        share_metric_key="lic_share_active",
        filename_suffix="pi_mass",
        summary_metric_tail=["lic_size", "active_node_count", "active_farm_count", "active_region_count", "lic_share_active", "pi_gini"],
    )
    if pi_mass_plot_path is not None:
        outputs["pi_mass_plot"] = str(pi_mass_plot_path)

    pi_closed_plot_path = _write_pi_mass_figure(
        details.get("pi_mass_closed_per_snapshot", pd.DataFrame()),
        details.get("pi_mass_closed_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        title_prefix="Stationary Mass / Largest Closed-Class Lazy Walk",
        support_title="Largest closed strongly connected class size",
        share_title="Closed-class concentration and dispersion",
        share_metric_key="lic_share_active",
        filename_suffix="pi_mass_closed",
        summary_metric_tail=["lic_size", "active_node_count", "active_farm_count", "active_region_count", "lic_share_active", "pi_gini"],
    )
    if pi_closed_plot_path is not None:
        outputs["pi_mass_closed_plot"] = str(pi_closed_plot_path)

    pi_pagerank_plot_path = _write_pi_mass_figure(
        details.get("pi_mass_pagerank_per_snapshot", pd.DataFrame()),
        details.get("pi_mass_pagerank_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
        title_prefix="Teleporting PageRank / Whole Active Snapshot",
        support_title="Active snapshot size",
        share_title="Active-snapshot coverage and dispersion",
        share_metric_key="lic_share_total",
        filename_suffix="pi_mass_pagerank",
        summary_metric_tail=["lic_size", "active_node_count", "active_farm_count", "active_region_count", "lic_share_total", "pi_gini"],
    )
    if pi_pagerank_plot_path is not None:
        outputs["pi_mass_pagerank_plot"] = str(pi_pagerank_plot_path)

    magnetic_plot_path = _write_magnetic_laplacian_figure(
        details.get("magnetic_laplacian_per_snapshot", pd.DataFrame()),
        details.get("magnetic_laplacian_summary", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
    )
    if magnetic_plot_path is not None:
        outputs["magnetic_laplacian_plot"] = str(magnetic_plot_path)
    magnetic_diff_plot_path = _write_magnetic_laplacian_diff_figure(
        details.get("magnetic_laplacian_per_snapshot", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
    )
    if magnetic_diff_plot_path is not None:
        outputs["magnetic_laplacian_diff_plot"] = str(magnetic_diff_plot_path)

    magnetic_distance_plot_path = _write_magnetic_spectral_distance_figure(
        details.get("magnetic_spectral_distance_per_snapshot", pd.DataFrame()),
        output_dir=output_dir,
        sample_label=sample_label,
    )
    if magnetic_distance_plot_path is not None:
        outputs["magnetic_spectral_distance_plot"] = str(magnetic_distance_plot_path)

    return outputs


def _log_category(message: str) -> str:
    if any(token in message for token in ("Resolved input paths", "Loaded edge CSV", "Timestamp-filtered edge frame", "Prepared input edge frame", "Prepared data in", "Attached external edge weights", "Loaded node features")):
        return "prepare"
    if any(token in message for token in ("Building layered graph", "Built layered graph", "Graph properties", "Weight property compatibility")):
        return "graph"
    if any(token in message for token in ("Starting fit command", "Fitting edge-weight candidate", "Starting nested SBM fit", "Completed nested SBM fit", "Selected edge-weight model", "Fit artifacts written", "Fitted layered nested SBM")):
        return "fit"
    if any(token in message for token in ("Starting generate command", "Loaded generation artifacts", "Generating sample", "Sampling synthetic panel", "Posterior partition", "Sampled snapshot", "Synthetic panel summary", "Generated sample manifests", "Generated ")):
        return "generate"
    if any(token in message for token in ("Starting report stage", "Reporting sample", "Comparing panels", "Snapshot comparison", "Panel comparison summary", "Writing diagnostics report", "Diagnostics artifacts written", "Wrote diagnostics")):
        return "report"
    if any(token in message for token in ("Serialised graph-tool state", "Restored graph-tool state", "Writing fit artifacts", "Suppressed graph-tool import stderr lines")):
        return "artifacts"
    return "runtime"


def _parse_log_lines(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_seconds: Optional[int] = None

    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        match = LOG_LINE_PATTERN.match(line)
        if not match:
            continue
        hh, mm, ss = (int(part) for part in match.group("time").split(":"))
        total_seconds = hh * 3600 + mm * 60 + ss
        if base_seconds is None:
            base_seconds = total_seconds
        elapsed_seconds = total_seconds - base_seconds
        if elapsed_seconds < 0:
            elapsed_seconds += 24 * 3600
        message = match.group("message")
        rows.append(
            {
                "line_no": line_no,
                "time": match.group("time"),
                "level": match.group("level"),
                "message": message,
                "elapsed_seconds": float(elapsed_seconds),
                "category": _log_category(message),
            }
        )
    return pd.DataFrame(rows)


def write_log_visual_summary(
    log_path: Path,
    output_dir: Path,
    label: Optional[str] = None,
) -> dict:
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_label = label or log_path.stem
    parsed = _parse_log_lines(log_path)

    summary = {
        "log_path": str(log_path),
        "label": sample_label,
        "line_count": int(len(parsed)),
        "elapsed_seconds": float(parsed["elapsed_seconds"].max()) if len(parsed) else 0.0,
        "counts_by_level": parsed["level"].value_counts(sort=False).sort_index().to_dict() if len(parsed) else {},
        "counts_by_category": parsed["category"].value_counts(sort=False).sort_index().to_dict() if len(parsed) else {},
    }

    json_path = output_dir / f"{sample_label}_summary.json"
    md_path = output_dir / f"{sample_label}_report.md"
    png_path = output_dir / f"{sample_label}_dashboard.png"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        f"# Log Summary for {sample_label}",
        "",
        f"- Source log: `{log_path.name}`",
        f"- Parsed log lines: {summary['line_count']}",
        f"- Elapsed seconds: {summary['elapsed_seconds']:.1f}",
        f"- Levels: {summary['counts_by_level']}",
        f"- Categories: {summary['counts_by_category']}",
        "",
        f"- Dashboard: `{png_path.name}`",
        f"- JSON summary: `{json_path.name}`",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    if len(parsed):
        plt = _load_matplotlib()
        if plt is None:
            return {
                "log_path": str(log_path),
                "summary_json": str(json_path),
                "report_md": str(md_path),
            }
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        _style_figure(fig, axes)
        fig.suptitle(f"Run Log Dashboard: {sample_label}", fontsize=16, fontweight="bold", color=PLOT_COLORS["text"])

        counts_by_level = parsed["level"].value_counts(sort=False).sort_index()
        axes[0, 0].bar(counts_by_level.index.tolist(), counts_by_level.to_numpy(dtype=float), color=PLOT_COLORS["original"], alpha=0.9)
        axes[0, 0].set_title("Log Lines by Level")
        axes[0, 0].set_ylabel("Lines")

        counts_by_category = parsed["category"].value_counts(sort=False).sort_values(ascending=False)
        axes[0, 1].bar(counts_by_category.index.tolist(), counts_by_category.to_numpy(dtype=float), color=PLOT_COLORS["synthetic"], alpha=0.9)
        axes[0, 1].set_title("Log Lines by Category")
        axes[0, 1].tick_params(axis="x", rotation=20)
        axes[0, 1].set_ylabel("Lines")

        elapsed = parsed["elapsed_seconds"].to_numpy(dtype=float)
        bin_count = max(5, min(20, int(math.sqrt(len(parsed)))))
        axes[1, 0].hist(elapsed, bins=bin_count, color=PLOT_COLORS["novel"], alpha=0.88)
        axes[1, 0].set_title("Event Density Over Time")
        axes[1, 0].set_xlabel("Elapsed seconds")
        axes[1, 0].set_ylabel("Log lines")

        stage_order = ["prepare", "graph", "fit", "generate", "report", "artifacts", "runtime"]
        stage_spans = []
        for category in stage_order:
            subset = parsed.loc[parsed["category"] == category]
            if subset.empty:
                continue
            start = float(subset["elapsed_seconds"].min())
            end = float(subset["elapsed_seconds"].max())
            if end <= start:
                end = start + 1.0
            stage_spans.append((category, start, end - start))

        axes[1, 1].set_title("Category Time Spans")
        for idx, (category, start, duration) in enumerate(stage_spans):
            axes[1, 1].barh(idx, duration, left=start, height=0.62, alpha=0.88, color=PLOT_COLORS["accent"])
        axes[1, 1].set_yticks(np.arange(len(stage_spans)), [item[0] for item in stage_spans] if stage_spans else [])
        axes[1, 1].set_xlabel("Elapsed seconds")
        axes[1, 1].grid(axis="x", color=PLOT_COLORS["grid"], alpha=0.85)

        _save_figure(fig, png_path)
        plt.close(fig)

    return {
        "log_path": str(log_path),
        "summary_json": str(json_path),
        "report_md": str(md_path),
        "dashboard_png": str(png_path),
    }


def write_report(
    per_snapshot: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    sample_label: str,
    *,
    detailed_diagnostics: Optional[dict[str, pd.DataFrame]] = None,
    directed: bool = True,
    diagnostic_top_k: int = 12,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(
        "Writing diagnostics report | output_dir=%s | sample_label=%s | snapshot_rows=%s",
        output_dir,
        sample_label,
        len(per_snapshot),
    )

    csv_path = output_dir / f"{sample_label}_per_snapshot.csv"
    json_path = output_dir / f"{sample_label}_summary.json"
    md_path = output_dir / f"{sample_label}_report.md"
    dashboard_path = _write_sample_dashboard(per_snapshot, summary, output_dir, sample_label)
    parity_path = _write_sample_parity_plot(per_snapshot, summary, output_dir, sample_label)
    detailed_outputs = (
        _write_detailed_diagnostics_artifacts(
            detailed_diagnostics,
            output_dir=output_dir,
            sample_label=sample_label,
            directed=directed,
            top_k=max(1, int(diagnostic_top_k)),
        )
        if detailed_diagnostics
        else {}
    )

    per_snapshot.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    posterior_runs = _posterior_run_count(summary)
    summary_scope_text = (
        f"This summary covers posterior-predictive discrepancy across {posterior_runs} generated panels for the same setting."
        if posterior_runs > 1
        else "This summary compares one generated panel against the filtered input panel used for fitting."
    )

    lines = [
        f"# Diagnostics for {sample_label}",
        "",
        summary_scope_text,
        "",
        "## Quick Summary",
        "",
    ]
    if posterior_runs > 1:
        lines.append(f"- Posterior draws aggregated: {posterior_runs}")
        lines.append(f"- Time-series association metric: {TIME_SERIES_CORRELATION_LABEL}")
    lines.extend(
        [
        f"- Snapshots compared: {summary['snapshot_count']}",
        f"- Original total edges: {summary['original_total_edges']}",
        f"- Synthetic total edges: {summary['synthetic_total_edges']}",
        f"- Unique edge Jaccard: {summary['unique_edge_jaccard']:.4f}",
        f"- Mean snapshot edge Jaccard: {summary['mean_snapshot_edge_jaccard']:.4f}",
        f"- Mean snapshot node Jaccard: {summary['mean_snapshot_node_jaccard']:.4f}",
        f"- Mean absolute edge-count delta: {summary['mean_abs_edge_count_delta']:.4f}",
        f"- Mean absolute active-node delta: {summary['mean_abs_active_node_delta']:.4f}",
        f"- Mean synthetic novelty rate: {summary['mean_synthetic_novel_edge_rate']:.4f}",
        f"- Snapshot edge-count correlation: {summary['edge_count_correlation']:.4f}",
        ]
    )
    if posterior_runs > 1 and "edge_count_pooled_correlation" in summary:
        lines.append(f"- Snapshot edge-count correlation (all runs pooled): {float(summary['edge_count_pooled_correlation']):.4f}")
    if "weight_column" in summary:
        lines.extend(
            [
                f"- Original total weight ({summary['weight_column']}): {summary['original_total_weight']:.4f}",
                f"- Synthetic total weight ({summary['weight_column']}): {summary['synthetic_total_weight']:.4f}",
                f"- Mean absolute total-weight delta: {summary['mean_abs_weight_total_delta']:.4f}",
                f"- Snapshot total-weight correlation: {summary['weight_total_correlation']:.4f}",
            ]
        )
        if posterior_runs > 1 and "weight_total_pooled_correlation" in summary:
            lines.append(f"- Snapshot total-weight correlation (all runs pooled): {float(summary['weight_total_pooled_correlation']):.4f}")
    if "original_temporal_reachability_ratio" in summary and "synthetic_temporal_reachability_ratio" in summary:
        lines.extend(
            [
                f"- Final temporal reachability ratio (observed): {float(summary['original_temporal_reachability_ratio']):.4f}",
                f"- Final temporal reachability ratio (synthetic): {float(summary['synthetic_temporal_reachability_ratio']):.4f}",
                f"- Temporal reachability correlation: {float(summary.get('temporal_reachability_ratio_correlation', 0.0)):.4f}",
                f"- Temporal efficiency correlation: {float(summary.get('temporal_efficiency_correlation', 0.0)):.4f}",
                f"- Forward-reach node correlation: {float(summary.get('temporal_forward_reach_node_correlation', 0.0)):.4f}",
                f"- Causal fidelity (observed / synthetic): {float(summary.get('original_causal_fidelity', np.nan)):.4f} / {float(summary.get('synthetic_causal_fidelity', np.nan)):.4f}",
            ]
        )
        if posterior_runs > 1 and "temporal_reachability_ratio_pooled_correlation" in summary:
            lines.append(
                f"- Temporal reachability correlation (all runs pooled): {float(summary['temporal_reachability_ratio_pooled_correlation']):.4f}"
            )

    lines.extend(
        [
            "",
            "## Overall Network Metrics",
            "",
            f"- Original density: {summary['original_overall']['density']:.6f}",
            f"- Synthetic density: {summary['synthetic_overall']['density']:.6f}",
            f"- Original mean distance (km): {summary['original_overall']['distance_km_mean']:.4f}",
            f"- Synthetic mean distance (km): {summary['synthetic_overall']['distance_km_mean']:.4f}",
        ]
    )
    if "weight_column" in summary:
        lines.extend(
            [
                f"- Original mean weight: {summary['original_overall']['weight_mean']:.4f}",
                f"- Synthetic mean weight: {summary['synthetic_overall']['weight_mean']:.4f}",
                f"- Original mean log1p weight: {summary['original_overall']['log1p_weight_mean']:.4f}",
                f"- Synthetic mean log1p weight: {summary['synthetic_overall']['log1p_weight_mean']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Files Written",
            "",
            f"- Per-snapshot CSV: `{csv_path.name}`",
            f"- JSON summary: `{json_path.name}`",
        ]
    )
    if dashboard_path is not None:
        lines.append(f"- Dashboard plot: `{dashboard_path.name}`")
    if parity_path is not None:
        lines.append(f"- Parity plot: `{parity_path.name}`")
    if detailed_outputs:
        lines.extend(
            [
                "",
                "## Detailed Diagnostics",
                "",
                "These files break the goodness-of-fit check down by block pair, by block activity, by node activity, and by temporal reachability / transmission-potential summaries.",
            ]
        )
        for key in sorted(detailed_outputs):
            lines.append(f"- {key}: `{Path(detailed_outputs[key]).name}`")
    md_path.write_text("\n".join(lines) + "\n")
    LOGGER.debug(
        "Diagnostics artifacts written | per_snapshot_csv=%s | summary_json=%s | report_md=%s | dashboard_png=%s | parity_png=%s",
        csv_path,
        json_path,
        md_path,
        dashboard_path,
        parity_path,
    )

    payload = {
        "per_snapshot_csv": str(csv_path),
        "summary_json": str(json_path),
        "report_md": str(md_path),
    }
    if dashboard_path is not None:
        payload["dashboard_png"] = str(dashboard_path)
    if parity_path is not None:
        payload["parity_png"] = str(parity_path)
    payload.update(detailed_outputs)
    return payload


def _resolve_node_map_path(manifest: dict) -> Optional[Path]:
    if manifest.get("node_map_csv"):
        candidate = Path(str(manifest["node_map_csv"])).expanduser().resolve()
        return candidate if candidate.exists() else None
    if manifest.get("dataset_dir"):
        candidate = Path(str(manifest["dataset_dir"])).expanduser().resolve() / "node_map.csv"
        return candidate if candidate.exists() else None
    return None


def _resolve_corop_geojson_path(manifest: dict) -> Optional[Path]:
    candidates: list[Path] = []
    if manifest.get("dataset_dir"):
        dataset_dir = Path(str(manifest["dataset_dir"])).expanduser().resolve()
        parents = list(dataset_dir.parents)
        if len(parents) > 1:
            candidates.append(parents[1] / "examples" / "public" / "nl_corop.geojson")
            candidates.append(parents[1] / "public" / "nl_corop.geojson")
        if len(parents) > 2:
            candidates.append(parents[2] / "public" / "nl_corop.geojson")
    if manifest.get("weight_npy"):
        weight_path = Path(str(manifest["weight_npy"])).expanduser().resolve()
        parents = list(weight_path.parents)
        if len(parents) > 2:
            candidates.append(parents[2] / "examples" / "public" / "nl_corop.geojson")
            candidates.append(parents[2] / "public" / "nl_corop.geojson")
        if len(parents) > 3:
            candidates.append(parents[3] / "public" / "nl_corop.geojson")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_hybrid_node_frame(run_dir: Path, manifest: dict) -> pd.DataFrame:
    node_attributes_path = Path(str(manifest.get("node_attributes_path") or run_dir / "node_attributes.csv"))
    if not node_attributes_path.exists():
        return pd.DataFrame(columns=["node_id", "x", "y", "type_label", "corop", "ubn", "block_id"])

    node_frame = pd.read_csv(node_attributes_path)
    if "node_id" not in node_frame.columns:
        return pd.DataFrame(columns=["node_id", "x", "y", "type_label", "corop", "ubn", "block_id"])
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
            node_frame = node_frame.merge(hybrid_map, on="node_id", how="left")

    if "type_label" not in node_frame.columns and "node_map_type" in node_frame.columns:
        node_frame["type_label"] = node_frame["node_map_type"]
    elif "type_label" in node_frame.columns:
        node_frame["type_label"] = node_frame["type_label"].copy()
        if "node_map_type" in node_frame.columns:
            node_frame["type_label"] = node_frame["type_label"].fillna(node_frame["node_map_type"])
    elif "type" in node_frame.columns:
        node_frame["type_label"] = node_frame["type"]
    else:
        node_frame["type_label"] = "Unknown"

    node_frame["type_label"] = node_frame["type_label"].map(_format_node_type_label)
    if "corop" not in node_frame.columns:
        node_frame["corop"] = ""
    node_frame["corop"] = node_frame["corop"].fillna("").astype(str)
    if "ubn" not in node_frame.columns:
        node_frame["ubn"] = np.nan
    if "block_id" not in node_frame.columns:
        node_frame["block_id"] = -1
    node_frame["type_short"] = node_frame["type_label"].map(_type_short_label)
    node_frame["display_label"] = node_frame.apply(
        lambda row: (
            row["corop"]
            if row["type_label"] == "Region" and str(row["corop"]).strip()
            else (f"UBN {int(float(row['ubn']))}" if row["type_label"] == "Farm" and pd.notna(row["ubn"]) else f"{row['type_label']} {int(row['node_id'])}")
        ),
        axis=1,
    )
    return node_frame.sort_values("node_id").reset_index(drop=True)


def _load_report_edges_for_label(run_dir: Path, manifest: dict, label: str) -> Optional[pd.DataFrame]:
    if label == "observed":
        observed_path = Path(str(manifest.get("filtered_input_edges_path") or run_dir / "input_edges_filtered.csv"))
        if not observed_path.exists():
            return None
        return pd.read_csv(observed_path)
    sample_path = run_dir / "generated" / label / "sample_0000" / "synthetic_edges.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    fallback_paths = sorted((run_dir / "generated").glob(f"{label}/sample_*/synthetic_edges.csv"))
    if fallback_paths:
        return pd.read_csv(fallback_paths[0])
    return None


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _build_magnetic_phase_payload(
    run_dir: Path,
    diagnostics_dir: Path,
    manifest: dict,
    selected_labels: list[str],
) -> Optional[dict[str, object]]:
    node_frame = _load_hybrid_node_frame(run_dir, manifest)
    if node_frame.empty:
        return None

    treatment_specs = [("observed", "Observed panel")]
    for label in selected_labels:
        treatment_specs.append((label, _setting_display_payload(label)["short_label"]))

    edge_frames: dict[str, pd.DataFrame] = {}
    node_universe = set(node_frame["node_id"].astype(int).tolist())
    directed = bool(manifest.get("directed", False))
    weight_model = manifest.get("weight_model") if isinstance(manifest.get("weight_model"), dict) else None
    phase_weight_col = None
    if weight_model is not None:
        candidate = weight_model.get("output_column") or weight_model.get("input_column")
        if candidate:
            phase_weight_col = str(candidate)
    weight_model = manifest.get("weight_model") if isinstance(manifest.get("weight_model"), dict) else None
    phase_weight_col = None
    if weight_model is not None:
        candidate = weight_model.get("output_column") or weight_model.get("input_column")
        if candidate:
            phase_weight_col = str(candidate)
    for key, _ in treatment_specs:
        frame = _load_report_edges_for_label(run_dir, manifest, key)
        if frame is None or frame.empty:
            continue
        current_weight_col = phase_weight_col if phase_weight_col and phase_weight_col in frame.columns else None
        canonical = canonicalise_edge_frame(frame, directed=directed, weight_col=current_weight_col)
        edge_frames[key] = canonical
        node_universe |= set(canonical["u"].astype(int).tolist())
        node_universe |= set(canonical["i"].astype(int).tolist())
    if not edge_frames:
        return None

    node_universe_sorted = sorted(node_universe)
    node_frame = node_frame.set_index("node_id").reindex(node_universe_sorted).reset_index()
    node_frame["type_label"] = node_frame["type_label"].fillna("Unknown").map(_format_node_type_label)
    node_frame["type_short"] = node_frame["type_label"].map(_type_short_label)
    node_frame["corop"] = node_frame["corop"].fillna("").astype(str)
    node_frame["display_label"] = node_frame.apply(
        lambda row: (
            row["display_label"]
            if pd.notna(row.get("display_label"))
            else (row["corop"] if row["type_label"] == "Region" and str(row["corop"]).strip() else f"{row['type_label']} {int(row['node_id'])}")
        ),
        axis=1,
    )

    treatments: list[dict[str, object]] = []
    observed_series = None
    for key, display_name in treatment_specs:
        frame = edge_frames.get(key)
        if frame is None or frame.empty:
            continue
        current_weight_col = phase_weight_col if phase_weight_col and phase_weight_col in frame.columns else None
        series = _compute_magnetic_phase_time_series(frame, node_universe=node_universe_sorted, weight_col=current_weight_col, directed=directed, k=2)
        if key == "observed":
            observed_series = series
        treatments.append(
            {
                "key": key,
                "label": display_name,
                "sample_label": key,
                "series": _json_ready(series),
            }
        )
    if not treatments or observed_series is None:
        return None

    region_mask = node_frame["type_label"].eq("Region").to_numpy(dtype=bool)
    farm_mask = node_frame["type_label"].eq("Farm").to_numpy(dtype=bool)
    observed_mag = np.asarray(observed_series["mag"])
    if observed_mag.ndim == 3 and observed_mag.shape[2] > 0:
        region_scores = np.nanmean(observed_mag[:, region_mask, 0], axis=0) if region_mask.any() else np.array([], dtype=float)
        farm_scores = np.nanmean(observed_mag[:, farm_mask, 0], axis=0) if farm_mask.any() else np.array([], dtype=float)
    else:
        region_scores = np.array([], dtype=float)
        farm_scores = np.array([], dtype=float)

    region_indices = np.flatnonzero(region_mask)
    farm_indices = np.flatnonzero(farm_mask)
    region_ranked = region_indices[np.argsort(-region_scores)] if region_scores.size else region_indices
    farm_ranked = farm_indices[np.argsort(-farm_scores)] if farm_scores.size else farm_indices

    geojson_path = _resolve_corop_geojson_path(manifest)
    geojson_payload = None
    if geojson_path is not None and geojson_path.exists():
        geojson_payload = json.loads(geojson_path.read_text())

    payload = {
        "dataset": str(manifest.get("dataset") or "dataset"),
        "directed": directed,
        "focal_corop": str(node_frame.loc[node_frame["type_label"] == "Farm", "corop"].mode().iloc[0]) if len(node_frame.loc[node_frame["type_label"] == "Farm", "corop"].mode()) else "",
        "calendar": _calendar_records(observed_series["ts"] if observed_series is not None else []),
        "treatments": treatments,
        "nodes": _json_ready(
            node_frame[
                [column for column in ("node_id", "x", "y", "type_label", "type_short", "corop", "ubn", "block_id", "display_label") if column in node_frame.columns]
            ]
        .to_dict(orient="records")),
        "track_groups": {
            "regions": [int(index) for index in region_ranked.tolist()],
            "farms": [int(index) for index in farm_ranked[:20].tolist()],
        },
        "geojson": geojson_payload,
    }

    payload_js_path = diagnostics_dir / "hybrid_phase_payload.js"
    payload_js_path.write_text("window.__TEMPORAL_SBM_PHASE_PAYLOAD__ = " + json.dumps(_json_ready(payload), separators=(",", ":")) + ";")
    payload["payload_js_path"] = str(payload_js_path)
    return payload


def _write_phase_geo_html(payload: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid Magnetic Geo Phase Compare</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="hybrid_phase_payload.js"></script>
  <style>
    :root { --bg:#f5f9fc; --panel:#ffffff; --line:#d8e1ea; --text:#20303f; --muted:#607286; --shadow:rgba(32,48,63,0.08); }
    body { margin:0; background:var(--bg); color:var(--text); font-family:"Avenir Next","Segoe UI",sans-serif; }
    .app { padding:18px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:14px; }
    .toolbar label { font-size:13px; color:var(--muted); display:flex; gap:8px; align-items:center; }
    .toolbar input[type="range"] { width:280px; }
    .grid { display:grid; gap:14px; grid-template-columns:repeat(2, minmax(320px, 1fr)); }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:14px; box-shadow:0 10px 24px var(--shadow); }
    .panel h3 { margin:0 0 8px; font-size:15px; }
    svg { width:100%; height:420px; display:block; }
    .summary { margin-top:14px; display:grid; gap:14px; grid-template-columns:1fr; }
    .stats { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; color:var(--muted); margin-top:10px; }
    .tooltip { position:fixed; pointer-events:none; background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px 12px; box-shadow:0 10px 24px rgba(32,48,63,0.12); font-size:12px; opacity:0; max-width:240px; }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>Compare setting <select id="treatment-select"></select></label>
      <label>Snapshot <input id="ts-slider" type="range" min="0" max="0" value="0" /></label>
      <label>Eigen mode <select id="mode-select"><option value="0">Mode 1</option><option value="1">Mode 2</option></select></label>
      <label><input id="zoom-toggle" type="checkbox" /> Zoom to focal region</label>
      <label><input id="region-only" type="checkbox" /> Region supernodes only</label>
      <button id="play-button" type="button">Play</button>
      <div id="status" style="font-size:13px;color:var(--muted);"></div>
    </div>
    <div class="grid">
      <div class="panel"><h3>Observed panel</h3><svg id="observed-map"></svg><div class="stats" id="observed-stats"></div></div>
      <div class="panel"><h3 id="candidate-title">Synthetic panel</h3><svg id="candidate-map"></svg><div class="stats" id="candidate-stats"></div></div>
    </div>
    <div class="summary">
      <div class="panel"><h3>Interpretation</h3><div id="comparison-summary" style="font-size:13px;color:var(--muted);line-height:1.6;"></div></div>
    </div>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const payload = window.__TEMPORAL_SBM_PHASE_PAYLOAD__;
    const nodes = payload.nodes || [];
    const geojson = payload.geojson || {type:"FeatureCollection", features:[]};
    const treatments = payload.treatments || [];
    const observed = treatments.find((entry) => entry.key === "observed") || treatments[0];
    const candidates = treatments.filter((entry) => entry.key !== "observed");
    const snapshots = (observed?.series?.ts || []).map(Number);
    const calendar = payload.calendar || [];
    const calendarByTs = new Map(calendar.map((entry) => [Number(entry.ts), entry]));
    const slider = document.getElementById("ts-slider");
    const modeSelect = document.getElementById("mode-select");
    const zoomToggle = document.getElementById("zoom-toggle");
    const regionOnly = document.getElementById("region-only");
    const treatmentSelect = document.getElementById("treatment-select");
    const playButton = document.getElementById("play-button");
    const status = document.getElementById("status");
    const tooltip = d3.select("#tooltip");
    const summary = document.getElementById("comparison-summary");
    slider.max = Math.max(0, snapshots.length - 1);
    candidates.forEach((entry) => treatmentSelect.appendChild(new Option(entry.label, entry.key)));
    let timer = null;

    const color = d3.scaleLinear()
      .domain([-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI])
      .range(["#6a00a8", "#355f8d", "#f7f7f7", "#e67e22", "#b30000"]);

    const regionFeature = geojson.features.find((feature) => String(feature.properties?.statcode || "") === String(payload.focal_corop || ""));
    const fullProjection = d3.geoIdentity().reflectY(true).fitSize([620, 420], geojson);
    const zoomProjection = regionFeature
      ? d3.geoIdentity().reflectY(true).fitExtent([[14, 14], [606, 406]], {type:"FeatureCollection", features:[regionFeature]})
      : fullProjection;

    function getProjection() {
      return zoomToggle.checked ? zoomProjection : fullProjection;
    }

    function nodeRows(treatment, snapshotIndex, modeIndex) {
      const series = treatment?.series || {};
      const phases = series.phi?.[snapshotIndex] || [];
      const magnitudes = series.mag?.[snapshotIndex] || [];
      const masks = series.mask?.[snapshotIndex] || [];
      return nodes.map((node, nodeIndex) => ({
        ...node,
        phase: phases?.[nodeIndex]?.[modeIndex],
        magnitude: magnitudes?.[nodeIndex]?.[modeIndex],
        active: Boolean(masks?.[nodeIndex])
      })).filter((node) =>
        node.active &&
        Number.isFinite(node.phase) &&
        Number.isFinite(node.x) &&
        Number.isFinite(node.y) &&
        (!regionOnly.checked || node.type_label === "Region")
      );
    }

    function circularStats(rows) {
      const meanX = d3.mean(rows, (row) => Math.cos(row.phase)) || 0;
      const meanY = d3.mean(rows, (row) => Math.sin(row.phase)) || 0;
      return {
        count: rows.length,
        meanPhase: Math.atan2(meanY, meanX),
        resultant: Math.sqrt(meanX * meanX + meanY * meanY)
      };
    }

    function drawMap(svgId, rows, statId) {
      const svg = d3.select(svgId);
      const width = svg.node().clientWidth || 620;
      const height = svg.node().clientHeight || 420;
      const projection = getProjection();
      const geoPath = d3.geoPath(projection);
      svg.selectAll("*").remove();
      const root = svg.append("g");
      root.append("g")
        .selectAll("path")
        .data(geojson.features || [])
        .join("path")
        .attr("d", geoPath)
        .attr("fill", (feature) => String(feature.properties?.statcode || "") === String(payload.focal_corop || "") ? "#edf4fb" : "#fafcfe")
        .attr("stroke", "#9eb0c2")
        .attr("stroke-width", 0.7);

      const sizeScale = d3.scaleLinear().domain([0, d3.max(rows, (row) => Number(row.magnitude || 0)) || 1]).range([28, 96]);
      root.append("g")
        .selectAll("path.node")
        .data(rows, (d) => d.node_id)
        .join((enter) => enter.append("path")
            .attr("class", "node")
            .attr("opacity", 0)
            .attr("transform", (d) => `translate(${projection([d.x, d.y])})`)
            .attr("d", (d) => d3.symbol().type(d.type_label === "Region" ? d3.symbolTriangle : d3.symbolCircle).size(sizeScale(Number(d.magnitude || 0)))())
            .attr("fill", (d) => color(d.phase))
            .attr("stroke", "#22313f")
            .attr("stroke-width", 0.45)
            .call((sel) => sel.transition().duration(650).attr("opacity", (d) => 0.35 + 0.65 * Math.max(0, Math.min(1, Number(d.magnitude || 0))))),
          (update) => update.call((sel) => sel.transition().duration(650)
            .attr("transform", (d) => `translate(${projection([d.x, d.y])})`)
            .attr("d", (d) => d3.symbol().type(d.type_label === "Region" ? d3.symbolTriangle : d3.symbolCircle).size(sizeScale(Number(d.magnitude || 0)))())
            .attr("fill", (d) => color(d.phase))
            .attr("opacity", (d) => 0.35 + 0.65 * Math.max(0, Math.min(1, Number(d.magnitude || 0))))),
          (exit) => exit.call((sel) => sel.transition().duration(250).attr("opacity", 0).remove())
        )
        .on("mousemove", (event, d) => {
          tooltip.style("opacity", 1).style("left", `${event.clientX + 14}px`).style("top", `${event.clientY + 14}px`)
            .html(`<strong>${d.display_label}</strong><br/>${d.type_label}${d.corop ? ` · ${d.corop}` : ""}<br/>phase=${Number(d.phase).toFixed(3)}<br/>|u|=${Number(d.magnitude || 0).toFixed(3)}`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));

      const stats = circularStats(rows);
      document.getElementById(statId).innerHTML = `<span>active nodes: ${stats.count}</span><span>mean phase: ${Number(stats.meanPhase).toFixed(3)}</span><span>resultant length: ${Number(stats.resultant).toFixed(3)}</span>`;
    }

    function update() {
      const snapshotIndex = Number(slider.value || 0);
      const modeIndex = Number(modeSelect.value || 0);
      const candidate = candidates.find((entry) => entry.key === treatmentSelect.value) || candidates[0] || observed;
      const tsValue = snapshots[snapshotIndex];
      const calendarEntry = calendarByTs.get(tsValue);
      document.getElementById("candidate-title").textContent = candidate ? candidate.label : "Synthetic panel";
      status.textContent = snapshots.length
        ? `${calendarEntry?.date || `ts=${tsValue}`} · ${calendarEntry?.category || "snapshot"}`
        : "No snapshots";
      const observedRows = nodeRows(observed, snapshotIndex, modeIndex);
      const candidateRows = nodeRows(candidate, snapshotIndex, modeIndex);
      drawMap("#observed-map", observedRows, "observed-stats");
      drawMap("#candidate-map", candidateRows, "candidate-stats");
      const obsStats = circularStats(observedRows);
      const synStats = circularStats(candidateRows);
      const phaseGap = Math.atan2(Math.sin(synStats.meanPhase - obsStats.meanPhase), Math.cos(synStats.meanPhase - obsStats.meanPhase));
      summary.innerHTML = `Observed vs selected setting on the magnetic phase field for mode ${modeIndex + 1}. A small mean-phase gap and similar resultant lengths indicate that the synthetic panel is preserving the broad directional organization of the hybrid network at this snapshot. Current summary: mean-phase gap ${Number(phaseGap).toFixed(3)}, observed resultant ${Number(obsStats.resultant).toFixed(3)}, synthetic resultant ${Number(synStats.resultant).toFixed(3)}.`;
    }

    playButton.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playButton.textContent = "Play";
        return;
      }
      timer = setInterval(() => {
        slider.value = (Number(slider.value) + 1) % Math.max(1, snapshots.length);
        update();
      }, 1200);
      playButton.textContent = "Pause";
    });

    slider.addEventListener("input", update);
    modeSelect.addEventListener("change", update);
    zoomToggle.addEventListener("change", update);
    regionOnly.addEventListener("change", update);
    treatmentSelect.addEventListener("change", update);
    update();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _write_phase_tracks_html(payload: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid Node Phase Tracks</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="hybrid_phase_payload.js"></script>
  <style>
    body { margin:0; background:#f7fafc; color:#20303f; font-family:"Avenir Next","Segoe UI",sans-serif; }
    .app { padding:18px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:14px; }
    .toolbar label { font-size:13px; color:#5f7082; display:flex; gap:8px; align-items:center; }
    .panel { background:#fff; border:1px solid #d7e0ea; border-radius:18px; padding:14px; box-shadow:0 10px 24px rgba(32,48,63,0.08); }
    svg { width:100%; height:540px; display:block; }
    .legend { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px 12px; margin-top:12px; font-size:12px; color:#5f7082; }
    .legend-item { display:flex; align-items:center; gap:8px; }
    .legend-line { width:26px; height:0; border-top:3px solid currentColor; }
    .legend-line.dashed { border-top-style:dashed; }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>Compare setting <select id="treatment-select"></select></label>
      <label>Eigen mode <select id="mode-select"><option value="0">Mode 1</option><option value="1">Mode 2</option></select></label>
      <label>Node group <select id="group-select"><option value="regions">Regional supernodes</option><option value="farms">Top farm nodes</option></select></label>
      <label>Top nodes <select id="topk-select"><option value="8">8</option><option value="12" selected>12</option><option value="20">20</option></select></label>
    </div>
    <div class="panel">
      <svg id="tracks"></svg>
      <div class="legend" id="legend"></div>
    </div>
  </div>
  <script>
    const payload = window.__TEMPORAL_SBM_PHASE_PAYLOAD__;
    const nodes = payload.nodes || [];
    const treatments = payload.treatments || [];
    const observed = treatments.find((entry) => entry.key === "observed") || treatments[0];
    const candidates = treatments.filter((entry) => entry.key !== "observed");
    const calendar = payload.calendar || [];
    const calendarByTs = new Map(calendar.map((entry) => [Number(entry.ts), entry]));
    const tracksSvg = d3.select("#tracks");
    const legend = d3.select("#legend");
    const treatmentSelect = d3.select("#treatment-select");
    candidates.forEach((entry) => treatmentSelect.append("option").attr("value", entry.key).text(entry.label));
    const modeSelect = d3.select("#mode-select");
    const groupSelect = d3.select("#group-select");
    const topkSelect = d3.select("#topk-select");
    const color = d3.scaleOrdinal(d3.schemeTableau10);

    function unwrap(values) {
      const output = [];
      let offset = 0;
      for (let index = 0; index < values.length; index += 1) {
        const current = values[index];
        if (!Number.isFinite(current)) { output.push(NaN); continue; }
        if (index > 0 && Number.isFinite(values[index - 1])) {
          const delta = current - values[index - 1];
          if (delta > Math.PI) offset -= 2 * Math.PI;
          if (delta < -Math.PI) offset += 2 * Math.PI;
        }
        output.push(current + offset);
      }
      return output;
    }

    function seriesFor(treatment, nodeIndex, modeIndex) {
      const ts = (observed.series?.ts || []).map(Number);
      const phases = treatment?.series?.phi || [];
      return unwrap(ts.map((_, snapshotIndex) => phases?.[snapshotIndex]?.[nodeIndex]?.[modeIndex]));
    }

    function update() {
      const candidate = candidates.find((entry) => entry.key === treatmentSelect.property("value")) || candidates[0] || observed;
      const modeIndex = Number(modeSelect.property("value") || 0);
      const groupKey = groupSelect.property("value");
      const topK = Number(topkSelect.property("value") || 12);
      const nodeIndices = (payload.track_groups?.[groupKey] || []).slice(0, topK);
      const ts = (observed.series?.ts || []).map(Number);
      const selected = nodeIndices.map((index) => ({ node: nodes[index], index, observed: seriesFor(observed, index, modeIndex), synthetic: seriesFor(candidate, index, modeIndex) }));

      const width = tracksSvg.node().clientWidth;
      const height = tracksSvg.node().clientHeight;
      const margin = { top: 18, right: 18, bottom: 42, left: 56 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      tracksSvg.selectAll("*").remove();
      const root = tracksSvg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
      const flatValues = selected.flatMap((entry) => entry.observed.concat(entry.synthetic)).filter((value) => Number.isFinite(value));
      const yExtent = d3.extent(flatValues);
      const yPad = Math.max(1, ((yExtent[1] || 1) - (yExtent[0] || -1)) * 0.06);
      const x = d3.scaleLinear().domain(d3.extent(ts)).range([0, innerWidth]);
      const y = d3.scaleLinear().domain([(yExtent[0] || -1) - yPad, (yExtent[1] || 1) + yPad]).range([innerHeight, 0]);
      root.append("g")
        .selectAll("rect.day-band")
        .data(calendar.filter((entry) => entry.category !== "weekday"))
        .join("rect")
        .attr("x", (entry) => x(Number(entry.ts) - 0.5))
        .attr("y", 0)
        .attr("width", (entry) => Math.max(2, x(Number(entry.ts) + 0.5) - x(Number(entry.ts) - 0.5)))
        .attr("height", innerHeight)
        .attr("fill", (entry) => entry.category === "holiday" ? "#fde2e2" : "#e6f0ff")
        .attr("opacity", (entry) => entry.category === "holiday" ? 0.72 : 0.48);
      const maxTickLabels = 14;
      const tickValues = calendar.length <= maxTickLabels
        ? calendar.map((entry) => Number(entry.ts))
        : Array.from(
            new Set(
              d3.range(maxTickLabels).map((index) => {
                const position = index * (calendar.length - 1) / Math.max(1, maxTickLabels - 1);
                return Number(calendar[Math.round(position)].ts);
              })
            )
          );
      root.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(
          d3.axisBottom(x)
            .tickValues(tickValues)
            .tickFormat((value) => {
              const entry = calendarByTs.get(Number(value));
              return entry ? `${entry.date}` : d3.format("d")(value);
            })
        )
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-0.5em")
        .attr("dy", "0.7em")
        .attr("transform", "rotate(-32)");
      root.append("g").call(d3.axisLeft(y));
      root.append("text").attr("x", innerWidth / 2).attr("y", innerHeight + 34).attr("text-anchor", "middle").attr("fill", "#5f7082").text("Snapshot");
      root.append("text").attr("x", -innerHeight / 2).attr("y", -40).attr("transform", "rotate(-90)").attr("text-anchor", "middle").attr("fill", "#5f7082").text("Unwrapped phase");

      color.domain(selected.map((entry) => entry.node.display_label));
      const line = d3.line().defined((value) => Number.isFinite(value)).x((_, index) => x(ts[index])).y((value) => y(value));
      selected.forEach((entry) => {
        const hue = color(entry.node.display_label);
        root.append("path").datum(entry.observed).attr("fill", "none").attr("stroke", hue).attr("stroke-width", 2.1).attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("opacity", 0.92).attr("d", line);
        root.append("path").datum(entry.synthetic).attr("fill", "none").attr("stroke", hue).attr("stroke-width", 1.7).attr("stroke-dasharray", "7 5").attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("opacity", 0.9).attr("d", line);
      });

      legend.selectAll("*").remove();
      selected.forEach((entry) => {
        const item = legend.append("div").attr("class", "legend-item").style("color", color(entry.node.display_label));
        item.append("span").attr("class", "legend-line");
        item.append("span").text(`${entry.node.display_label} · observed`);
        item.append("span").attr("class", "legend-line dashed");
        item.append("span").text(candidate.label);
      });
    }

    treatmentSelect.on("change", update);
    modeSelect.on("change", update);
    groupSelect.on("change", update);
    topkSelect.on("change", update);
    update();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _write_phase_panels_html(payload: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid Magnetic Phase Panels</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="hybrid_phase_payload.js"></script>
  <style>
    body { margin:0; background:#f7fafc; color:#20303f; font-family:"Avenir Next","Segoe UI",sans-serif; }
    .app { padding:18px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:14px; }
    .toolbar label { font-size:13px; color:#5f7082; display:flex; gap:8px; align-items:center; }
    .grid { display:grid; gap:14px; grid-template-columns:repeat(2, minmax(280px,1fr)); }
    .panel { background:#fff; border:1px solid #d7e0ea; border-radius:18px; padding:12px; box-shadow:0 10px 24px rgba(32,48,63,0.08); }
    .panel h3 { margin:0 0 8px; font-size:14px; }
    svg { width:100%; height:300px; display:block; }
    .tooltip { position:fixed; pointer-events:none; background:#fff; border:1px solid #d7e0ea; border-radius:12px; padding:10px 12px; box-shadow:0 10px 24px rgba(32,48,63,0.12); font-size:12px; opacity:0; max-width:240px; }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>Compare setting <select id="treatment-select"></select></label>
      <label>Snapshot <input id="ts-slider" type="range" min="0" max="0" value="0" /></label>
      <button id="play-button" type="button">Play</button>
      <label><input id="region-only" type="checkbox" checked /> Region nodes only</label>
      <div id="status" style="font-size:13px;color:#5f7082;"></div>
    </div>
    <div class="grid">
      <div class="panel"><h3>Observed · Mode 1</h3><svg id="obs-m1"></svg></div>
      <div class="panel"><h3 id="cand-m1-title">Synthetic · Mode 1</h3><svg id="cand-m1"></svg></div>
      <div class="panel"><h3>Observed · Mode 2</h3><svg id="obs-m2"></svg></div>
      <div class="panel"><h3 id="cand-m2-title">Synthetic · Mode 2</h3><svg id="cand-m2"></svg></div>
    </div>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const payload = window.__TEMPORAL_SBM_PHASE_PAYLOAD__;
    const treatments = payload.treatments || [];
    const observed = treatments.find((entry) => entry.key === "observed") || treatments[0];
    const candidates = treatments.filter((entry) => entry.key !== "observed");
    const nodes = payload.nodes || [];
    const snapshots = (observed?.series?.ts || []).map(Number);
    const calendar = payload.calendar || [];
    const calendarByTs = new Map(calendar.map((entry) => [Number(entry.ts), entry]));
    const slider = document.getElementById("ts-slider");
    const playButton = document.getElementById("play-button");
    const regionOnly = document.getElementById("region-only");
    const status = document.getElementById("status");
    const tooltip = d3.select("#tooltip");
    const treatmentSelect = document.getElementById("treatment-select");
    candidates.forEach((entry) => treatmentSelect.appendChild(new Option(entry.label, entry.key)));
    slider.max = Math.max(0, snapshots.length - 1);
    let timer = null;
    const baseColor = d3.scaleLinear().domain([-Math.PI, 0, Math.PI]).range(["#355f8d", "#f7f7f7", "#e67e22"]);

    function nodeRows(treatment, snapshotIndex, modeIndex) {
      const series = treatment?.series || {};
      const phases = series.phi?.[snapshotIndex] || [];
      const magnitudes = series.mag?.[snapshotIndex] || [];
      const masks = series.mask?.[snapshotIndex] || [];
      return nodes.map((node, nodeIndex) => ({
        ...node,
        node_index: nodeIndex,
        phase: phases?.[nodeIndex]?.[modeIndex],
        magnitude: magnitudes?.[nodeIndex]?.[modeIndex],
        active: Boolean(masks?.[nodeIndex])
      })).filter((node) => node.active && Number.isFinite(node.phase) && (!regionOnly.checked || node.type_label === "Region"));
    }

    function circularDistance(a, b) {
      const delta = Math.atan2(Math.sin(a - b), Math.cos(a - b));
      return Math.sqrt(Math.max(0, 2 - 2 * Math.cos(delta)));
    }

    function drawPolar(svgId, rows, referenceRows) {
      const svg = d3.select(svgId);
      svg.selectAll("*").remove();
      const width = svg.node().clientWidth;
      const height = svg.node().clientHeight;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.34;
      const referenceRadius = radius * 0.74;
      const mismatchScale = radius * 0.24;
      const size = d3.scaleLinear().domain([0, d3.max(rows, (row) => Number(row.magnitude || 0)) || 1]).range([24, 84]);
      const referenceMap = new Map((referenceRows || []).map((row) => [row.node_id, row]));
      svg.append("circle").attr("cx", centerX).attr("cy", centerY).attr("r", referenceRadius).attr("fill", "none").attr("stroke", "#c7d2de").attr("stroke-width", 1.2);
      svg.append("circle").attr("cx", centerX).attr("cy", centerY).attr("r", referenceRadius + mismatchScale).attr("fill", "none").attr("stroke", "#e7edf4").attr("stroke-dasharray", "5 5").attr("stroke-width", 1.0);
      svg.append("line").attr("x1", centerX).attr("y1", centerY - (referenceRadius + mismatchScale)).attr("x2", centerX).attr("y2", centerY + (referenceRadius + mismatchScale)).attr("stroke", "#e0e7ef");
      svg.append("line").attr("x1", centerX - (referenceRadius + mismatchScale)).attr("y1", centerY).attr("x2", centerX + (referenceRadius + mismatchScale)).attr("y2", centerY).attr("stroke", "#e0e7ef");
      svg.append("g").selectAll("path.node").data(rows, (d) => `${d.node_id}`).join("path")
        .attr("transform", (d) => {
          const reference = referenceMap.get(d.node_id);
          const distance = reference ? circularDistance(d.phase, reference.phase) : 0;
          const localRadius = referenceRadius + mismatchScale * (distance / 2);
          const x = centerX + localRadius * Math.sin(d.phase);
          const y = centerY - localRadius * Math.cos(d.phase);
          return `translate(${x},${y})`;
        })
        .attr("d", (d) => d3.symbol().type(d.type_label === "Region" ? d3.symbolTriangle : d3.symbolCircle).size(size(Number(d.magnitude || 0)))())
        .attr("fill", (d) => baseColor(d.phase))
        .attr("stroke", "#22313f")
        .attr("stroke-width", 0.45)
        .attr("opacity", 0.88)
        .on("mousemove", (event, d) => {
          const reference = referenceMap.get(d.node_id);
          const distance = reference ? circularDistance(d.phase, reference.phase) : 0;
          const phaseGap = reference ? Math.atan2(Math.sin(d.phase - reference.phase), Math.cos(d.phase - reference.phase)) : 0;
          tooltip.style("opacity", 1).style("left", `${event.clientX + 14}px`).style("top", `${event.clientY + 14}px`)
            .html(`<strong>${d.display_label}</strong><br/>${d.type_label}${d.corop ? ` · ${d.corop}` : ""}<br/>phase=${Number(d.phase).toFixed(3)}<br/>|u|=${Number(d.magnitude || 0).toFixed(3)}<br/>phase gap=${Number(phaseGap).toFixed(3)}<br/>polar distance=${Number(distance).toFixed(3)}`);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));
    }

    function update() {
      const candidate = candidates.find((entry) => entry.key === treatmentSelect.value) || candidates[0] || observed;
      const snapshotIndex = Number(slider.value || 0);
      const tsValue = snapshots[snapshotIndex];
      const calendarEntry = calendarByTs.get(tsValue);
      status.textContent = snapshots.length
        ? `${calendarEntry?.date || `ts=${tsValue}`} · ${calendarEntry?.category || "snapshot"}`
        : "No snapshots";
      document.getElementById("cand-m1-title").textContent = `${candidate.label} · Mode 1`;
      document.getElementById("cand-m2-title").textContent = `${candidate.label} · Mode 2`;
      const observedMode1 = nodeRows(observed, snapshotIndex, 0);
      const observedMode2 = nodeRows(observed, snapshotIndex, 1);
      const candidateMode1 = nodeRows(candidate, snapshotIndex, 0);
      const candidateMode2 = nodeRows(candidate, snapshotIndex, 1);
      drawPolar("#obs-m1", observedMode1, observedMode1);
      drawPolar("#cand-m1", candidateMode1, observedMode1);
      drawPolar("#obs-m2", observedMode2, observedMode2);
      drawPolar("#cand-m2", candidateMode2, observedMode2);
    }

    playButton.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playButton.textContent = "Play";
        return;
      }
      timer = setInterval(() => {
        slider.value = (Number(slider.value) + 1) % Math.max(1, snapshots.length);
        update();
      }, 1200);
      playButton.textContent = "Pause";
    });

    slider.addEventListener("input", update);
    regionOnly.addEventListener("change", update);
    treatmentSelect.addEventListener("change", update);
    update();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _write_hybrid_phase_assets(
    run_dir: Path,
    diagnostics_dir: Path,
    manifest: dict,
    selected_labels: list[str],
) -> dict[str, Path]:
    payload = _build_magnetic_phase_payload(run_dir, diagnostics_dir, manifest, selected_labels)
    if payload is None:
        return {}
    outputs = {
        "phase_geo_html": diagnostics_dir / "hybrid_phase_geo_compare.html",
        "phase_tracks_html": diagnostics_dir / "hybrid_phase_tracks.html",
        "phase_panels_html": diagnostics_dir / "hybrid_phase_panels_compare.html",
    }
    _write_phase_geo_html(payload, outputs["phase_geo_html"])
    _write_phase_tracks_html(payload, outputs["phase_tracks_html"])
    _write_phase_panels_html(payload, outputs["phase_panels_html"])
    return outputs


def _load_graph_tool_module():
    try:
        import graph_tool.all as gt
    except ModuleNotFoundError:
        LOGGER.warning("graph-tool is not installed; skipping daily network snapshot assets.")
        return None
    except Exception as exc:
        LOGGER.warning("graph-tool could not be imported; skipping daily network snapshot assets | error=%s", exc)
        return None
    return gt


def _is_region_val(val):
    if isinstance(val, (int, np.integer, float)):
        try:
            return int(val) == 1
        except Exception:
            return False
    return str(val).strip().lower() in {"region", "reg", "adm2", "1", "2"}


def _pick_layout(gt, g, layout: str = "sfdp", pos_from=None):
    if pos_from is not None:
        return pos_from
    layout_name = (layout or "sfdp").lower()
    if layout_name == "sfdp":
        return gt.sfdp_layout(g)
    if layout_name == "fr":
        return gt.fruchterman_reingold_layout(g)
    if layout_name == "arf":
        return gt.arf_layout(g)
    if layout_name == "radial":
        vertices = list(g.vertices())
        if vertices:
            root = max(vertices, key=lambda vertex: int(vertex.out_degree() + vertex.in_degree()))
            return gt.radial_tree_layout(g, root)
        return gt.sfdp_layout(g)
    if layout_name == "random":
        return gt.random_layout(g)
    return gt.sfdp_layout(g)


def _pair_panels_pos(pos_left, vertices: list[object], gap_ratio: float = 0.25) -> tuple[float, tuple[float, float]]:
    xs = np.asarray([float(pos_left[vertex][0]) for vertex in vertices], dtype=float)
    width = float(xs.max() - xs.min()) if xs.size else 1.0
    gap = width * float(gap_ratio)
    return width + gap, (float(xs.min()) if xs.size else 0.0, float(xs.max()) if xs.size else 0.0)


def _snapshot_value_to_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(numeric) else numeric


def _build_snapshot_graph(
    gt,
    snapshot_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    *,
    directed: bool,
):
    graph = gt.Graph(directed=directed)
    graph.add_vertex(len(node_frame))

    node_id_prop = graph.new_vp("int64_t")
    type_prop = graph.new_vp("int")
    animal_prop = graph.new_vp("double")
    label_prop = graph.new_vp("string")
    x_prop = graph.new_vp("double")
    y_prop = graph.new_vp("double")

    for index, row in enumerate(node_frame.itertuples(index=False)):
        vertex = graph.vertex(index)
        node_id_prop[vertex] = int(row.node_id)
        raw_type = getattr(row, "type", 1 if str(getattr(row, "type_label", "")).strip() == "Region" else 0)
        type_prop[vertex] = 1 if _is_region_val(raw_type) else 0
        animal_prop[vertex] = _snapshot_value_to_float(getattr(row, "total_animals", 0.0), default=0.0)
        label_value = getattr(row, "display_label", None)
        if label_value is None or (isinstance(label_value, float) and pd.isna(label_value)):
            label_value = f"Node {int(row.node_id)}"
        label_prop[vertex] = str(label_value)
        x_prop[vertex] = _snapshot_value_to_float(getattr(row, "x", np.nan), default=np.nan)
        y_prop[vertex] = _snapshot_value_to_float(getattr(row, "y", np.nan), default=np.nan)

    graph.vp["node_id"] = node_id_prop
    graph.vp["type"] = type_prop
    graph.vp["total_animals"] = animal_prop
    graph.vp["display_label"] = label_prop
    graph.vp["name"] = label_prop
    graph.vp["cx"] = x_prop
    graph.vp["cy"] = y_prop

    local_index = {int(row.node_id): index for index, row in enumerate(node_frame.itertuples(index=False))}
    for edge in snapshot_frame.itertuples(index=False):
        source = local_index.get(int(edge.u))
        target = local_index.get(int(edge.i))
        if source is None or target is None:
            continue
        graph.add_edge(graph.vertex(source), graph.vertex(target))

    return graph, {
        "type": type_prop,
        "total_animals": animal_prop,
        "label": label_prop,
        "x": x_prop,
        "y": y_prop,
    }


def _build_pies_and_halos(g, v_type, v_anim_like=None):
    in_degree = np.fromiter((int(vertex.in_degree()) for vertex in g.vertices()), dtype=int, count=g.num_vertices())
    out_degree = np.fromiter((int(vertex.out_degree()) for vertex in g.vertices()), dtype=int, count=g.num_vertices())
    pies = g.new_vp("vector<double>")
    for index, vertex in enumerate(g.vertices()):
        total = int(in_degree[index] + out_degree[index])
        pies[vertex] = [in_degree[index] / total, out_degree[index] / total] if total else [0.5, 0.5]

    halo = g.new_vp("bool")
    halo.a = True
    halo_size = g.new_vp("double")
    if v_anim_like is not None and getattr(v_anim_like, "a", np.asarray([], dtype=float)).size:
        animals = np.clip(np.asarray(v_anim_like.a, dtype=float), a_min=0.0, a_max=None)
        max_animals = float(animals.max()) if animals.size else 0.0
        halo_size.a = 1.0 if max_animals <= 0 else 1.0 + 1.6 * (np.log1p(animals) / np.log1p(max_animals))
    else:
        halo_size.a = 1.0

    halo_color = g.new_vp("string")
    for vertex in g.vertices():
        halo_color[vertex] = "#FA8072" if _is_region_val(v_type[vertex]) else "#568203"
    return pies, halo, halo_size, halo_color


def _classify_snapshot_edges(g, v_type):
    edge_dash = g.new_ep("vector<double>")
    edge_color = g.new_ep("string")
    for edge in g.edges():
        source_is_region = _is_region_val(v_type[edge.source()]) if v_type is not None else False
        target_is_region = _is_region_val(v_type[edge.target()]) if v_type is not None else False
        if source_is_region and target_is_region:
            edge_dash[edge], edge_color[edge] = [3.0, 6.0], "#d62728"
        elif (not source_is_region) and (not target_is_region):
            edge_dash[edge], edge_color[edge] = [], "orange"
        else:
            edge_dash[edge], edge_color[edge] = [12.0, 6.0], "#1f77b4"
    return edge_dash, edge_color


def _copy_positions(gt, source_graph, source_pos, target_graph):
    copied = target_graph.new_vp("vector<double>")
    source_vertices = list(source_graph.vertices())
    target_vertices = list(target_graph.vertices())
    for index, target_vertex in enumerate(target_vertices):
        reference = source_vertices[index]
        copied[target_vertex] = [float(source_pos[reference][0]), float(source_pos[reference][1])]
    return copied


def _coordinate_positions(g, x_prop, y_prop):
    x_values = np.asarray(x_prop.a, dtype=float)
    y_values = np.asarray(y_prop.a, dtype=float)
    if not x_values.size or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return None
    pos = g.new_vp("vector<double>")
    for vertex in g.vertices():
        pos[vertex] = [float(x_prop[vertex]), float(-y_prop[vertex])]
    return pos


def _geojson_exterior_rings(geojson_payload: Optional[dict[str, object]]) -> list[list[tuple[float, float]]]:
    if not isinstance(geojson_payload, dict):
        return []

    def _project_ring(points: list[object]) -> list[tuple[float, float]]:
        ring: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x_value = float(point[0])
                y_value = -float(point[1])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                ring.append((x_value, y_value))
        return ring

    rings: list[list[tuple[float, float]]] = []
    for feature in geojson_payload.get("features", []) or []:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        geom_type = str(geometry.get("type", ""))
        coordinates = geometry.get("coordinates")
        if geom_type == "Polygon" and isinstance(coordinates, list) and coordinates:
            ring = _project_ring(coordinates[0])
            if len(ring) >= 3:
                rings.append(ring)
        elif geom_type == "MultiPolygon" and isinstance(coordinates, list):
            for polygon in coordinates:
                if not isinstance(polygon, list) or not polygon:
                    continue
                ring = _project_ring(polygon[0])
                if len(ring) >= 3:
                    rings.append(ring)
    return rings


def _draw_snapshot_pair_pdf(
    gt,
    g_left,
    g_right,
    pos_left,
    pos_right,
    pies_left,
    halo_left,
    halo_size_left,
    halo_color_left,
    label_left,
    pies_right,
    halo_right,
    halo_size_right,
    halo_color_right,
    label_right,
    edge_dash_left,
    edge_color_left,
    edge_dash_right,
    edge_color_right,
    output_path: Path,
    *,
    output_size: tuple[int, int],
    basemap_rings: Optional[list[list[tuple[float, float]]]] = None,
    panel_titles: tuple[str, str] = ("Observed", "Synthetic"),
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import cairo
    from graph_tool.draw.cairo_draw import adjust_default_sizes, cairo_draw as gt_cairo_draw, fit_to_view, fit_to_view_ink

    combined = gt.Graph(directed=g_left.is_directed())
    vertex_count = int(g_left.num_vertices())
    left_vertices = list(g_left.vertices())
    right_vertices = list(g_right.vertices())
    left_map = [combined.add_vertex() for _ in range(vertex_count)]
    right_map = [combined.add_vertex() for _ in range(vertex_count)]

    pos_prop = combined.new_vp("vector<double>")
    pie_prop = combined.new_vp("vector<double>")
    halo_prop = combined.new_vp("bool")
    halo_size_prop = combined.new_vp("double")
    halo_color_prop = combined.new_vp("string")
    edge_dash_prop = combined.new_ep("vector<double>")
    edge_color_prop = combined.new_ep("string")

    if basemap_rings:
        panel_points = [(float(pos_left[vertex][0]), float(pos_left[vertex][1])) for vertex in left_vertices]
        for ring in basemap_rings:
            panel_points.extend((float(point[0]), float(point[1])) for point in ring)
        xs = np.asarray([point[0] for point in panel_points], dtype=float)
        panel_width = float(xs.max() - xs.min()) if xs.size else 1.0
        dx = panel_width + panel_width * 0.30
    else:
        dx, _ = _pair_panels_pos(pos_left, left_vertices, gap_ratio=0.30)

    for index, source_vertex in enumerate(left_vertices):
        vertex = left_map[index]
        pos_prop[vertex] = [float(pos_left[source_vertex][0]), float(pos_left[source_vertex][1])]
        pie_prop[vertex] = pies_left[source_vertex]
        halo_prop[vertex] = bool(halo_left[source_vertex])
        halo_size_prop[vertex] = float(halo_size_left[source_vertex])
        halo_color_prop[vertex] = str(halo_color_left[source_vertex])

    for index, source_vertex in enumerate(right_vertices):
        vertex = right_map[index]
        pos_prop[vertex] = [float(pos_right[source_vertex][0]) + dx, float(pos_right[source_vertex][1])]
        pie_prop[vertex] = pies_right[source_vertex]
        halo_prop[vertex] = bool(halo_right[source_vertex])
        halo_size_prop[vertex] = float(halo_size_right[source_vertex])
        halo_color_prop[vertex] = str(halo_color_right[source_vertex])

    for edge in g_left.edges():
        combined_edge = combined.add_edge(left_map[int(edge.source())], left_map[int(edge.target())])
        edge_dash_prop[combined_edge] = edge_dash_left[edge]
        edge_color_prop[combined_edge] = edge_color_left[edge]

    for edge in g_right.edges():
        combined_edge = combined.add_edge(right_map[int(edge.source())], right_map[int(edge.target())])
        edge_dash_prop[combined_edge] = edge_dash_right[edge]
        edge_color_prop[combined_edge] = edge_color_right[edge]

    right_basemap = [[(x_value + dx, y_value) for x_value, y_value in ring] for ring in (basemap_rings or [])]
    viewport_size = [int(output_size[0]), max(int(output_size[1]) - 24, 1)]
    vprops = {
        "shape": "pie",
        "pie_fractions": pie_prop,
        "pie_colors": ["black", "lightgrey"],
        "halo": halo_prop,
        "halo_color": halo_color_prop,
        "halo_size": halo_size_prop,
        "size": 4,
    }
    eprops = {
        "dash_style": edge_dash_prop,
        "color": edge_color_prop,
        "pen_width": 0.7,
        "end_marker": "arrow",
        "marker_size": 3.0,
    }
    adjust_default_sizes(combined, viewport_size, vprops, eprops)

    if basemap_rings:
        basemap_points = [point for ring in basemap_rings for point in ring] + [point for ring in right_basemap for point in ring]
        graph_points = [(float(pos_prop[vertex][0]), float(pos_prop[vertex][1])) for vertex in combined.vertices()]
        points = basemap_points + graph_points
        xs = np.asarray([point[0] for point in points], dtype=float)
        ys = np.asarray([point[1] for point in points], dtype=float)
        min_x = float(xs.min()) if xs.size else -0.5
        max_x = float(xs.max()) if xs.size else 0.5
        min_y = float(ys.min()) if ys.size else -0.5
        max_y = float(ys.max()) if ys.size else 0.5
        x_value, y_value, zoom = fit_to_view((min_x, min_y, max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)), viewport_size, adjust_aspect=True, pad=0.9)
    else:
        x_value, y_value, zoom = fit_to_view_ink(combined, pos_prop, viewport_size, vprops, eprops, adjust_aspect=True, pad=0.9)

    panel_titles = tuple(panel_titles[:2]) if panel_titles else ("Observed", "Synthetic")
    if len(panel_titles) < 2:
        panel_titles = (panel_titles[0], "Synthetic")

    surface = cairo.PDFSurface(str(output_path), int(output_size[0]), int(output_size[1]))
    context = cairo.Context(surface)
    context.set_source_rgb(1.0, 1.0, 1.0)
    context.paint()
    context.translate(0.0, 24.0)
    context.scale(zoom, zoom)
    context.translate(-x_value, -y_value)

    def draw_basemap(rings: list[list[tuple[float, float]]]) -> None:
        if not rings:
            return
        context.save()
        context.set_line_join(cairo.LINE_JOIN_ROUND)
        context.set_line_cap(cairo.LINE_CAP_ROUND)
        for ring in rings:
            if len(ring) < 3:
                continue
            context.new_path()
            context.move_to(float(ring[0][0]), float(ring[0][1]))
            for point in ring[1:]:
                context.line_to(float(point[0]), float(point[1]))
            context.close_path()
            context.set_source_rgba(0.95, 0.97, 0.99, 0.92)
            context.fill_preserve()
            context.set_source_rgba(0.72, 0.78, 0.84, 0.85)
            context.set_line_width(0.6 / max(float(zoom), 1e-6))
            context.stroke()
        context.restore()

    draw_basemap(basemap_rings or [])
    draw_basemap(right_basemap)
    gt_cairo_draw(combined, pos_prop, context, vprops, eprops)

    context.identity_matrix()
    context.select_font_face("Avenir Next", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(13.0)
    context.set_source_rgb(0.14, 0.19, 0.25)
    context.move_to(16.0, 17.0)
    context.show_text(str(panel_titles[0]))
    context.move_to(float(output_size[0]) * 0.5 + 16.0, 17.0)
    context.show_text(str(panel_titles[1]))
    surface.finish()
    return output_path


def _empty_snapshot_frame(weight_col: Optional[str] = None) -> pd.DataFrame:
    columns = ["u", "i", "ts"] + ([str(weight_col)] if weight_col else [])
    return pd.DataFrame(columns=columns)


def _render_daily_snapshot_pdfs(
    gt,
    *,
    node_frame: pd.DataFrame,
    observed_snapshot: pd.DataFrame,
    synthetic_snapshot: pd.DataFrame,
    directed: bool,
    ts_value: int,
    output_dir: Path,
    output_size: tuple[int, int] = (800, 400),
    geographic_basemap_rings: Optional[list[list[tuple[float, float]]]] = None,
) -> dict[str, Path]:
    active_ids = sorted(
        set(pd.to_numeric(observed_snapshot.get("u", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        | set(pd.to_numeric(observed_snapshot.get("i", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        | set(pd.to_numeric(synthetic_snapshot.get("u", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        | set(pd.to_numeric(synthetic_snapshot.get("i", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    )
    if not active_ids:
        return {}

    node_slice = node_frame.loc[node_frame["node_id"].isin(active_ids)].drop_duplicates(subset=["node_id"]).copy()
    missing_ids = sorted(set(active_ids) - set(node_slice["node_id"].astype(int).tolist()))
    if missing_ids:
        filler = pd.DataFrame(
            {
                "node_id": missing_ids,
                "x": np.nan,
                "y": np.nan,
                "total_animals": 0.0,
                "type": 0,
                "type_label": "Unknown",
                "display_label": [f"Node {node_id}" for node_id in missing_ids],
            }
        )
        node_slice = pd.concat([node_slice, filler], ignore_index=True)
    node_slice = node_slice.sort_values("node_id").reset_index(drop=True)

    observed_graph, observed_props = _build_snapshot_graph(gt, observed_snapshot, node_slice, directed=directed)
    synthetic_graph, synthetic_props = _build_snapshot_graph(gt, synthetic_snapshot, node_slice, directed=directed)

    pies_left, halo_left, halo_size_left, halo_color_left = _build_pies_and_halos(
        observed_graph,
        observed_props["type"],
        observed_props["total_animals"],
    )
    pies_right, halo_right, halo_size_right, halo_color_right = _build_pies_and_halos(
        synthetic_graph,
        synthetic_props["type"],
        synthetic_props["total_animals"],
    )
    edge_dash_left, edge_color_left = _classify_snapshot_edges(observed_graph, observed_props["type"])
    edge_dash_right, edge_color_right = _classify_snapshot_edges(synthetic_graph, synthetic_props["type"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    forced_path = output_dir / f"snapshot_{int(ts_value)}_forced.pdf"
    if not forced_path.exists():
        pos_left_forced = _pick_layout(gt, observed_graph, layout="sfdp")
        pos_right_forced = _copy_positions(gt, observed_graph, pos_left_forced, synthetic_graph)
        _draw_snapshot_pair_pdf(
            gt,
            observed_graph,
            synthetic_graph,
            pos_left_forced,
            pos_right_forced,
            pies_left,
            halo_left,
            halo_size_left,
            halo_color_left,
            observed_props["label"],
            pies_right,
            halo_right,
            halo_size_right,
            halo_color_right,
            synthetic_props["label"],
            edge_dash_left,
            edge_color_left,
            edge_dash_right,
            edge_color_right,
            forced_path,
            output_size=output_size,
            panel_titles=("Observed", "Synthetic"),
        )
    outputs["forced"] = forced_path

    geographic_pos_left = _coordinate_positions(observed_graph, observed_props["x"], observed_props["y"])
    geographic_path = output_dir / f"snapshot_{int(ts_value)}_geographic.pdf"
    if geographic_pos_left is not None:
        if not geographic_path.exists():
            geographic_pos_right = _copy_positions(gt, observed_graph, geographic_pos_left, synthetic_graph)
            _draw_snapshot_pair_pdf(
                gt,
                observed_graph,
                synthetic_graph,
                geographic_pos_left,
                geographic_pos_right,
                pies_left,
                halo_left,
                halo_size_left,
                halo_color_left,
                observed_props["label"],
                pies_right,
                halo_right,
                halo_size_right,
                halo_color_right,
                synthetic_props["label"],
                edge_dash_left,
                edge_color_left,
                edge_dash_right,
                edge_color_right,
                geographic_path,
                output_size=output_size,
                basemap_rings=geographic_basemap_rings,
                panel_titles=("Observed", "Synthetic"),
            )
        outputs["geographic"] = geographic_path
    return outputs


def _sample_dir_from_labels(run_dir: Path, setting_label: str, run_label: str) -> Optional[Path]:
    run_dir = Path(run_dir)
    label = str(run_label)
    sample_index = _sample_index_from_label(label)
    if sample_index is not None:
        candidate = run_dir / "generated" / setting_label / f"sample_{sample_index:04d}"
        return candidate if candidate.exists() else None
    if label.startswith("sample_"):
        candidate = run_dir / "generated" / setting_label / label
        return candidate if candidate.exists() else None
    candidate = run_dir / "generated" / setting_label / "sample_0000"
    return candidate if candidate.exists() else None


def _run_labels_for_setting(run_dir: Path, diagnostics_dir: Path, setting_label: str) -> list[str]:
    summary_payload = _load_json_if_exists(diagnostics_dir / f"{setting_label}_summary.json") or {}
    run_labels = [str(value) for value in summary_payload.get("posterior_run_labels", []) or []]
    if run_labels:
        return run_labels

    sample_dirs = sorted((Path(run_dir) / "generated" / setting_label).glob("sample_*"))
    labels = []
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        if sample_name.startswith("sample_"):
            try:
                sample_index = int(sample_name.split("_", 1)[1])
            except Exception:
                continue
            labels.append(f"{setting_label}__sample_{sample_index:04d}")
    return labels


def _network_snapshot_setting_labels(
    summary_rows: pd.DataFrame,
    preferred_labels: list[str],
) -> list[str]:
    labels: list[str] = []
    available = set(summary_rows.get("sample_label", pd.Series(dtype=str)).astype(str).tolist())
    for label in preferred_labels:
        if label in available and label not in labels:
            labels.append(label)

    ranking = summary_rows.copy()
    if "sample_class" in ranking.columns:
        primary = ranking.loc[ranking["sample_class"].astype(str) == "posterior_predictive"].copy()
        if len(primary):
            ranking = primary
    sort_columns = [column for column in ("mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate") if column in ranking.columns]
    if sort_columns:
        ascending = [False, False, True][: len(sort_columns)]
        ranking = ranking.sort_values(sort_columns, ascending=ascending)
    for label in ranking.get("sample_label", pd.Series(dtype=str)).astype(str).tolist():
        if label not in labels:
            labels.append(label)
        if len(labels) >= 2:
            break
    return labels


def _load_sample_snapshots(
    sample_dir: Path,
    *,
    directed: bool,
    weight_col: Optional[str],
) -> dict[int, pd.DataFrame]:
    sample_dir = Path(sample_dir)
    snapshot_dir = sample_dir / "snapshots"
    snapshots: dict[int, pd.DataFrame] = {}

    if snapshot_dir.exists():
        for snapshot_path in sorted(snapshot_dir.glob("snapshot_*.csv")):
            match = re.search(r"snapshot_(\d+)\.csv$", snapshot_path.name)
            if not match:
                continue
            ts_value = int(match.group(1))
            frame = pd.read_csv(snapshot_path)
            current_weight_col = weight_col if weight_col and weight_col in frame.columns else None
            snapshots[ts_value] = canonicalise_edge_frame(frame, directed=directed, weight_col=current_weight_col)
        if snapshots:
            return snapshots

    synthetic_path = sample_dir / "synthetic_edges.csv"
    if not synthetic_path.exists():
        return snapshots
    synthetic_frame = pd.read_csv(synthetic_path)
    current_weight_col = weight_col if weight_col and weight_col in synthetic_frame.columns else None
    canonical = canonicalise_edge_frame(synthetic_frame, directed=directed, weight_col=current_weight_col)
    for ts_value, frame in canonical.groupby("ts", sort=True):
        snapshots[int(ts_value)] = frame.reset_index(drop=True)
    return snapshots


def _write_daily_network_viewer_html(output_path: Path) -> Path:
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Daily Network Snapshot Compare</title>
  <script src="daily_network_snapshot_payload.js"></script>
  <style>
    :root { --bg:#eef3f8; --panel:#ffffff; --line:#d7e0ea; --text:#20303f; --muted:#607286; --shadow:rgba(32,48,63,0.08); }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(180deg, #f8fbfd 0%, var(--bg) 100%); color:var(--text); font-family:"Avenir Next","Segoe UI",sans-serif; }
    .app { padding:18px; }
    .toolbar { display:grid; gap:12px; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); align-items:end; margin-bottom:16px; }
    .toolbar label { display:flex; flex-direction:column; gap:6px; font-size:13px; color:var(--muted); }
    .toolbar select, .toolbar input[type="range"], .toolbar button { width:100%; font:inherit; }
    .toolbar select, .toolbar input[type="range"] { padding:10px 12px; border:1px solid var(--line); border-radius:12px; background:#fbfdff; color:var(--text); }
    .toolbar button { padding:10px 14px; border:1px solid rgba(78,121,167,0.24); border-radius:12px; background:linear-gradient(180deg, #fafdff 0%, #edf4f9 100%); color:#365776; cursor:pointer; }
    .status { font-size:13px; color:var(--muted); margin-bottom:12px; }
    .grid { display:grid; gap:14px; grid-template-columns:repeat(2, minmax(320px, 1fr)); }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:14px; box-shadow:0 10px 24px var(--shadow); }
    .panel h3 { margin:0 0 10px; font-size:15px; }
    .panel-meta { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; color:var(--muted); margin-bottom:10px; }
    .viewer-frame { width:100%; height:760px; border:1px solid var(--line); border-radius:14px; background:#fff; }
    .viewer-empty { display:flex; align-items:center; justify-content:center; height:760px; border:1px dashed var(--line); border-radius:14px; color:var(--muted); background:#fbfdff; text-align:center; padding:24px; }
    .panel-actions { margin-top:10px; font-size:12px; color:var(--muted); }
    .panel-actions a { color:#365776; text-decoration:none; }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>Layout
        <select id="layout-select"></select>
      </label>
      <label>Snapshot
        <input id="day-slider" type="range" min="0" max="0" value="0" />
      </label>
      <label>Left setting
        <select id="left-setting"></select>
      </label>
      <label>Left run
        <select id="left-run"></select>
      </label>
      <label>Right setting
        <select id="right-setting"></select>
      </label>
      <label>Right run
        <select id="right-run"></select>
      </label>
      <label>Playback
        <button id="play-button" type="button">Play</button>
      </label>
    </div>
    <div class="status" id="status"></div>
    <div class="grid">
      <div class="panel">
        <h3 id="left-title">Left comparison</h3>
        <div class="panel-meta" id="left-meta"></div>
        <div id="left-shell"></div>
        <div class="panel-actions" id="left-actions"></div>
      </div>
      <div class="panel">
        <h3 id="right-title">Right comparison</h3>
        <div class="panel-meta" id="right-meta"></div>
        <div id="right-shell"></div>
        <div class="panel-actions" id="right-actions"></div>
      </div>
    </div>
  </div>
  <script>
    const payload = window.__TEMPORAL_SBM_DAILY_NETWORK_PAYLOAD__ || {};
    const settings = payload.settings || [];
    const calendar = payload.calendar || [];
    const layouts = payload.layouts || [];
    const calendarByTs = new Map(calendar.map((entry) => [String(entry.ts), entry]));
    const settingsByKey = new Map(settings.map((entry) => [entry.key, entry]));
    const leftSetting = document.getElementById("left-setting");
    const leftRun = document.getElementById("left-run");
    const rightSetting = document.getElementById("right-setting");
    const rightRun = document.getElementById("right-run");
    const layoutSelect = document.getElementById("layout-select");
    const daySlider = document.getElementById("day-slider");
    const playButton = document.getElementById("play-button");
    const status = document.getElementById("status");
    let timer = null;

    layouts.forEach((entry) => layoutSelect.appendChild(new Option(entry.label, entry.key)));
    settings.forEach((entry) => {
      leftSetting.appendChild(new Option(entry.label, entry.key));
      rightSetting.appendChild(new Option(entry.label, entry.key));
    });
    daySlider.max = Math.max(0, calendar.length - 1);

    function runsForSetting(settingKey) {
      return settingsByKey.get(settingKey)?.runs || [];
    }

    function fillRunSelect(settingSelect, runSelect, preferredKey) {
      const runs = runsForSetting(settingSelect.value);
      runSelect.innerHTML = "";
      runs.forEach((entry) => runSelect.appendChild(new Option(entry.label, entry.key)));
      if (!runs.length) {
        return;
      }
      const preferred = runs.find((entry) => entry.key === preferredKey);
      runSelect.value = preferred ? preferred.key : runs[0].key;
    }

    function runRecord(settingKey, runKey) {
      const setting = settingsByKey.get(settingKey);
      if (!setting) {
        return null;
      }
      return (setting.runs || []).find((entry) => entry.key === runKey) || setting.runs?.[0] || null;
    }

    function dayEntry(settingKey, runKey, tsValue) {
      const record = runRecord(settingKey, runKey);
      if (!record) {
        return null;
      }
      return record.days?.[String(tsValue)] || null;
    }

    function renderPanel(side, settingKey, runKey) {
      const tsValue = calendar[Number(daySlider.value || 0)]?.ts;
      const layoutKey = layoutSelect.value || layouts[0]?.key;
      const setting = settingsByKey.get(settingKey);
      const run = runRecord(settingKey, runKey);
      const entry = tsValue == null ? null : dayEntry(settingKey, runKey, tsValue);
      document.getElementById(`${side}-title`).textContent = `${setting?.label || "Setting"} · ${run?.label || "Run"}`;
      const meta = document.getElementById(`${side}-meta`);
      meta.innerHTML = "";
      [
        `layout ${layouts.find((item) => item.key === layoutKey)?.label || layoutKey}`,
        entry ? `observed edges ${entry.observed_edge_count}` : null,
        entry ? `synthetic edges ${entry.synthetic_edge_count}` : null,
        entry ? `active nodes ${entry.active_node_count}` : null
      ].filter(Boolean).forEach((text) => {
        const span = document.createElement("span");
        span.textContent = text;
        meta.appendChild(span);
      });

      const shell = document.getElementById(`${side}-shell`);
      const actions = document.getElementById(`${side}-actions`);
      const pdfPath = entry?.layouts?.[layoutKey];
      if (!pdfPath) {
        shell.innerHTML = `<div class="viewer-empty">No snapshot PDF is available for this selection.</div>`;
        actions.textContent = "";
        return;
      }
      shell.innerHTML = `<iframe class="viewer-frame" src="${pdfPath}#view=FitH" title="${side} snapshot viewer" loading="lazy"></iframe>`;
      actions.innerHTML = `<a href="${pdfPath}" target="_blank" rel="noopener">Open PDF in a new tab</a>`;
    }

    function updateStatus() {
      const entry = calendar[Number(daySlider.value || 0)];
      if (!entry) {
        status.textContent = "No snapshots available.";
        return;
      }
      const category = entry.category === "holiday" ? "public holiday" : entry.category;
      status.textContent = `${entry.date} · ${category}`;
    }

    function update() {
      updateStatus();
      renderPanel("left", leftSetting.value, leftRun.value);
      renderPanel("right", rightSetting.value, rightRun.value);
    }

    leftSetting.addEventListener("change", () => {
      fillRunSelect(leftSetting, leftRun, leftRun.value);
      update();
    });
    rightSetting.addEventListener("change", () => {
      fillRunSelect(rightSetting, rightRun, rightRun.value);
      update();
    });
    leftRun.addEventListener("change", update);
    rightRun.addEventListener("change", update);
    layoutSelect.addEventListener("change", update);
    daySlider.addEventListener("input", update);
    playButton.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playButton.textContent = "Play";
        return;
      }
      timer = setInterval(() => {
        daySlider.value = (Number(daySlider.value || 0) + 1) % Math.max(1, calendar.length);
        update();
      }, 1400);
      playButton.textContent = "Pause";
    });

    if (settings.length) {
      leftSetting.value = settings[0].key;
      rightSetting.value = settings[Math.min(1, settings.length - 1)].key;
      fillRunSelect(leftSetting, leftRun);
      fillRunSelect(rightSetting, rightRun);
      const leftRuns = runsForSetting(leftSetting.value);
      const rightRuns = runsForSetting(rightSetting.value);
      if (leftSetting.value === rightSetting.value && leftRuns.length > 1 && rightRuns.length > 1) {
        rightRun.value = rightRuns[Math.min(1, rightRuns.length - 1)].key;
      }
    }
    if (layouts.length) {
      layoutSelect.value = layouts[0].key;
    }
    update();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _write_daily_network_snapshot_assets(
    run_dir: Path,
    diagnostics_dir: Path,
    manifest: dict,
    summary_rows: pd.DataFrame,
    preferred_labels: list[str],
) -> dict[str, Path]:
    gt = _load_graph_tool_module()
    if gt is None:
        return {}

    node_frame = _load_hybrid_node_frame(run_dir, manifest)
    if node_frame.empty or "node_id" not in node_frame.columns:
        return {}

    directed = bool(manifest.get("directed", False))
    weight_model = manifest.get("weight_model") if isinstance(manifest.get("weight_model"), dict) else None
    weight_col = None
    if weight_model is not None:
        candidate = weight_model.get("output_column") or weight_model.get("input_column")
        if candidate:
            weight_col = str(candidate)

    observed_path = Path(str(manifest.get("filtered_input_edges_path") or run_dir / "input_edges_filtered.csv"))
    if not observed_path.exists():
        return {}
    observed_frame = pd.read_csv(observed_path)
    observed_weight_col = weight_col if weight_col and weight_col in observed_frame.columns else None
    observed_edges = canonicalise_edge_frame(observed_frame, directed=directed, weight_col=observed_weight_col)
    observed_by_ts = {
        int(ts_value): frame.reset_index(drop=True)
        for ts_value, frame in observed_edges.groupby("ts", sort=True)
    }
    if not observed_by_ts:
        return {}

    geojson_rings: list[list[tuple[float, float]]] = []
    geojson_path = _resolve_corop_geojson_path(manifest)
    if geojson_path is not None and geojson_path.exists():
        try:
            geojson_rings = _geojson_exterior_rings(json.loads(geojson_path.read_text()))
        except Exception as exc:
            LOGGER.warning("Failed to load geographic basemap for daily network snapshots | path=%s | error=%s", geojson_path, exc)
            geojson_rings = []

    setting_labels = _network_snapshot_setting_labels(summary_rows, preferred_labels)
    if not setting_labels:
        return {}

    snapshot_root = diagnostics_dir / "daily_network_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "calendar": _calendar_records(sorted(observed_by_ts)),
        "layouts": [
            {"key": "forced", "label": "Forced layout"},
            {"key": "geographic", "label": "Geographic layout"},
        ],
        "settings": [],
    }

    for setting_label in setting_labels:
        run_labels = _run_labels_for_setting(run_dir, diagnostics_dir, setting_label)
        runs_payload: list[dict[str, object]] = []
        for run_label in run_labels:
            sample_dir = _sample_dir_from_labels(run_dir, setting_label, run_label)
            if sample_dir is None:
                continue
            synthetic_by_ts = _load_sample_snapshots(sample_dir, directed=directed, weight_col=weight_col)
            if not synthetic_by_ts:
                continue

            run_payload = {
                "key": str(run_label),
                "label": _posterior_run_display_label(str(run_label)),
                "days": {},
            }
            output_dir = snapshot_root / setting_label / sample_dir.name
            for ts_value in sorted(set(observed_by_ts) | set(synthetic_by_ts)):
                observed_snapshot = observed_by_ts.get(ts_value, _empty_snapshot_frame(observed_weight_col))
                synthetic_snapshot = synthetic_by_ts.get(ts_value, _empty_snapshot_frame(observed_weight_col))
                pdf_paths = _render_daily_snapshot_pdfs(
                    gt,
                    node_frame=node_frame,
                    observed_snapshot=observed_snapshot,
                    synthetic_snapshot=synthetic_snapshot,
                    directed=directed,
                    ts_value=int(ts_value),
                    output_dir=output_dir,
                    geographic_basemap_rings=geojson_rings,
                )
                day_payload = {
                    "observed_edge_count": int(len(observed_snapshot)),
                    "synthetic_edge_count": int(len(synthetic_snapshot)),
                    "active_node_count": int(
                        len(
                            set(pd.to_numeric(observed_snapshot.get("u", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
                            | set(pd.to_numeric(observed_snapshot.get("i", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
                            | set(pd.to_numeric(synthetic_snapshot.get("u", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
                            | set(pd.to_numeric(synthetic_snapshot.get("i", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
                        )
                    ),
                    "layouts": {},
                }
                for layout_key, pdf_path in pdf_paths.items():
                    day_payload["layouts"][layout_key] = str(pdf_path.relative_to(diagnostics_dir).as_posix())
                run_payload["days"][str(int(ts_value))] = day_payload
            runs_payload.append(run_payload)

        if not runs_payload:
            continue
        payload["settings"].append(
            {
                "key": str(setting_label),
                "label": _setting_display_payload(setting_label)["short_label"],
                "runs": runs_payload,
            }
        )

    if not payload["settings"]:
        return {}

    payload_js_path = diagnostics_dir / "daily_network_snapshot_payload.js"
    payload_js_path.write_text(
        "window.__TEMPORAL_SBM_DAILY_NETWORK_PAYLOAD__ = "
        + json.dumps(_json_ready(payload), separators=(",", ":"))
        + ";",
        encoding="utf-8",
    )
    viewer_path = diagnostics_dir / "daily_network_compare.html"
    _write_daily_network_viewer_html(viewer_path)
    return {
        "network_compare_html": viewer_path,
        "network_payload_js": payload_js_path,
    }


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


def _compact_sampler_name(sample_mode: str) -> str:
    mapping = {
        "micro": "Micro SBM",
        "canonical_posterior": "Canonical posterior",
        "canonical_ml": "Canonical ML",
        "maxent_micro": "Max-ent micro",
        "canonical_maxent": "Canonical max-ent",
    }
    return mapping.get(sample_mode, sample_mode.replace("_", " ").replace("-", " ").title())


def _compact_rewire_name(rewire_mode: str) -> str:
    canonical = rewire_mode.replace("_", "-")
    mapping = {
        "none": "",
        "configuration": "config",
        "constrained-configuration": "constrained config",
        "blockmodel-micro": "blockmodel-micro",
    }
    return mapping.get(canonical, canonical.replace("-", " ").title())


def _overview_setting_label(sample_label: str) -> str:
    sample_mode, rewire_mode = _parse_sample_label_parts(sample_label)
    sampler_name = _compact_sampler_name(sample_mode)
    rewire_name = _compact_rewire_name(rewire_mode)
    if not rewire_name:
        return sampler_name
    return f"{sampler_name} + {rewire_name}"


def _heatmap_annotation_color(value: float, cmap) -> str:
    rgba = cmap(float(np.clip(value, 0.0, 1.0)))
    red, green, blue = float(rgba[0]), float(rgba[1]), float(rgba[2])
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    return "#ffffff" if luminance < 0.5 else PLOT_COLORS["text"]


def _sampler_ensemble_name(sample_mode: str) -> str:
    mapping = {
        "micro": "Microcanonical",
        "canonical_ml": "Canonical",
        "canonical_posterior": "Canonical",
        "maxent_micro": "Microcanonical",
        "canonical_maxent": "Canonical",
    }
    return mapping.get(sample_mode, sample_mode.replace("_", " ").replace("-", " ").title())


def _sampler_regime_name(sample_mode: str) -> str:
    mapping = {
        "micro": "Standard exact-count draw",
        "canonical_ml": "Maximum-likelihood block-rate draw",
        "canonical_posterior": "Posterior block-rate draw",
        "maxent_micro": "Maximum-entropy exact-count draw",
        "canonical_maxent": "Maximum-entropy block-rate draw",
    }
    return mapping.get(sample_mode, sample_mode.replace("_", " ").replace("-", " ").title())


def _rewire_step_name(rewire_mode: str) -> str:
    canonical = rewire_mode.replace("_", "-")
    mapping = {
        "none": "No post hoc rewiring",
        "configuration": "Configuration edge-swap step",
        "constrained-configuration": "Constrained configuration edge-swap step",
        "blockmodel-micro": "Blockmodel micro edge-swap step",
    }
    return mapping.get(canonical, canonical.replace("-", " ").title())


def _default_sample_class(sample_label: str) -> str:
    _, rewire_mode = _parse_sample_label_parts(sample_label)
    return "posterior_predictive" if rewire_mode == "none" else "sensitivity_analysis"


def _display_sample_class(sample_class: str) -> str:
    mapping = {
        "posterior_predictive": "Primary posterior-predictive",
        "sensitivity_analysis": "Sensitivity analysis",
    }
    return mapping.get(sample_class, sample_class.replace("_", " ").title())


def _load_sample_class_map(run_dir: Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    sample_class_map: dict[str, str] = {}
    for manifest_path in sorted(run_dir.glob("generated/*/sample_*/sample_manifest.json")):
        payload = _load_json_if_exists(manifest_path)
        if not isinstance(payload, dict):
            continue
        label = manifest_path.parent.parent.name
        sample_class = payload.get("sample_class")
        if label and sample_class:
            sample_class_map[str(label)] = str(sample_class)
    return sample_class_map


def _attach_sweep_metadata(summary_frame: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if summary_frame.empty or "sample_label" not in summary_frame.columns:
        return summary_frame
    frame = summary_frame.copy()
    sample_parts = frame["sample_label"].astype(str).map(_parse_sample_label_parts)
    frame["sample_mode"] = [sample_mode for sample_mode, _ in sample_parts]
    frame["rewire_mode"] = [rewire_mode for _, rewire_mode in sample_parts]
    sample_class_map = _load_sample_class_map(run_dir)
    frame["sample_class"] = frame["sample_label"].astype(str).map(
        lambda label: sample_class_map.get(label, _default_sample_class(label))
    )
    return frame


def _display_rec_type(rec_type: str) -> str:
    mapping = {
        "discrete-poisson": "Discrete Poisson",
        "discrete-geometric": "Discrete geometric",
        "discrete-binomial": "Discrete binomial",
        "real-exponential": "Exponential",
        "real-normal": "Normal",
    }
    return mapping.get(rec_type, rec_type.replace("-", " ").replace("_", " ").title())


def _display_transform_name(transform: str) -> str:
    mapping = {
        "none": "No transform",
        "log": "Log transform",
        "log1p": "Log1p transform",
    }
    return mapping.get(transform, transform.replace("_", " ").replace("-", " ").title())


def _setting_display_payload(sample_label: str) -> dict[str, str]:
    sample_mode, rewire_mode = _parse_sample_label_parts(sample_label)
    sampler_name = _display_sampler_name(sample_mode)
    rewire_name = _display_rewire_name(rewire_mode)
    if rewire_name == "No rewiring":
        short_label = sampler_name
    else:
        short_label = f"{sampler_name} + {rewire_name}"
    return {
        "sample_label": sample_label,
        "sample_mode": sample_mode,
        "rewire_mode": rewire_mode,
        "sampler_name": sampler_name,
        "rewire_name": rewire_name,
        "short_label": short_label,
    }


def _format_covariate_name(covariate: str) -> str:
    if ":" in covariate and "/" in covariate:
        column_name, model_spec = covariate.split(":", 1)
        rec_type, transform = model_spec.split("/", 1)
        return f"{column_name} ({_display_rec_type(rec_type)}; {_display_transform_name(transform)})"
    return covariate.replace("_", " ")


def _weight_model_display_payload(weight_model: Optional[dict], manifest: Optional[dict] = None) -> dict[str, str]:
    payload = weight_model or {}
    manifest = manifest or {}
    rec_type = str(payload.get("rec_type") or "none")
    transform = str(payload.get("transform") or "none")
    input_column = str(payload.get("input_column") or payload.get("output_column") or "n/a")
    source_name = Path(str(manifest.get("weight_npy", "embedded CSV"))).name if manifest.get("weight_npy") else "Embedded CSV"
    return {
        "title": _display_rec_type(rec_type),
        "transform": _display_transform_name(transform),
        "input_column": input_column,
        "source_name": source_name,
        "candidate_label": str(payload.get("candidate_label") or "n/a"),
    }


def _format_report_metric_value(row: pd.Series | dict[str, object], key: str, *, precision: int = 3) -> str:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "n/a"
    lower, upper = _posterior_interval_from_mapping(row, key)
    if lower is not None and upper is not None and _posterior_run_count(row) > 1:
        return f"{float(value):.{precision}f}<br><span class='setting-subtitle'>[{lower:.{precision}f}, {upper:.{precision}f}]</span>"
    return f"{float(value):.{precision}f}"


def _write_validation_frontier_plot(summary_rows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_rows.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    frame = summary_rows.copy()
    frame["sample_mode"], frame["rewire_mode"] = zip(*frame["sample_label"].map(_parse_sample_label_parts))
    rewire_palette = {
        "none": PLOT_COLORS["original"],
        "configuration": PLOT_COLORS["synthetic"],
        "constrained_configuration": PLOT_COLORS["novel"],
        "blockmodel_micro": PLOT_COLORS["delta"],
        "constrained-configuration": PLOT_COLORS["novel"],
        "blockmodel-micro": PLOT_COLORS["delta"],
    }
    mode_markers = {
        "micro": "o",
        "canonical_posterior": "s",
        "canonical_ml": "D",
        "maxent_micro": "^",
        "canonical_maxent": "P",
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    _style_figure(fig, [ax])
    fig.suptitle("Novelty / Overlap Frontier", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    for row in frame.itertuples(index=False):
        color = rewire_palette.get(str(row.rewire_mode), PLOT_COLORS["neutral"])
        marker = mode_markers.get(str(row.sample_mode), "o")
        weight_corr = float(getattr(row, "weight_total_correlation", 0.0) or 0.0)
        size = 90 + 220 * max(weight_corr, 0.0)
        x_value = float(row.mean_synthetic_novel_edge_rate)
        y_value = float(row.mean_snapshot_edge_jaccard)
        ax.scatter(
            x_value,
            y_value,
            s=size,
            color=color,
            marker=marker,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.9,
        )
        x_lower = getattr(row, "mean_synthetic_novel_edge_rate_q05", None)
        x_upper = getattr(row, "mean_synthetic_novel_edge_rate_q95", None)
        y_lower = getattr(row, "mean_snapshot_edge_jaccard_q05", None)
        y_upper = getattr(row, "mean_snapshot_edge_jaccard_q95", None)
        if all(value is not None for value in (x_lower, x_upper)) and np.isfinite([x_lower, x_upper]).all():
            ax.hlines(y_value, float(x_lower), float(x_upper), color=color, linewidth=1.4, alpha=0.55, zorder=2)
        if all(value is not None for value in (y_lower, y_upper)) and np.isfinite([y_lower, y_upper]).all():
            ax.vlines(x_value, float(y_lower), float(y_upper), color=color, linewidth=1.4, alpha=0.55, zorder=2)

    highlight = pd.concat(
        [
            frame.sort_values(
                ["mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate"],
                ascending=[False, False, True],
            ).head(1),
            frame.sort_values(
                ["mean_synthetic_novel_edge_rate", "mean_snapshot_edge_jaccard", "weight_total_correlation"],
                ascending=[True, False, False],
            ).head(1),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["sample_label"])
    for index, row in enumerate(highlight.itertuples(index=False)):
        x_value = float(row.mean_synthetic_novel_edge_rate)
        y_value = float(row.mean_snapshot_edge_jaccard)
        x_offset = -0.085 if index % 2 else 0.012
        y_offset = 0.012 if index % 2 == 0 else -0.018
        ax.text(
            x_value + x_offset,
            y_value + y_offset,
            str(row.sample_label),
            fontsize=8.5,
            color=PLOT_COLORS["text"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": PLOT_COLORS["grid_strong"], "linewidth": 0.9},
        )
        ax.plot(
            [x_value, x_value + x_offset * 0.82],
            [y_value, y_value + y_offset * 0.82],
            color=PLOT_COLORS["grid_strong"],
            linewidth=0.9,
            alpha=0.95,
        )

    ax.set_xlabel("Mean synthetic novel-edge rate (lower is better)")
    ax.set_ylabel("Mean snapshot edge Jaccard (higher is better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)

    rewire_handles = []
    rewire_labels = []
    for rewire_mode in sorted(frame["rewire_mode"].unique().tolist()):
        rewire_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=rewire_palette.get(str(rewire_mode), PLOT_COLORS["neutral"]),
                markeredgecolor="white",
                markersize=9,
            )
        )
        rewire_labels.append(str(rewire_mode).replace("_", "-"))
    mode_handles = []
    mode_labels = []
    for sample_mode in sorted(frame["sample_mode"].unique().tolist()):
        mode_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=mode_markers.get(str(sample_mode), "o"),
                color=PLOT_COLORS["text"],
                linestyle="None",
                markersize=8,
            )
        )
        mode_labels.append(str(sample_mode).replace("_", " "))
    legend_one = ax.legend(rewire_handles, rewire_labels, title="Rewire", loc="lower left")
    _style_legend(legend_one)
    ax.add_artist(legend_one)
    _style_legend(ax.legend(mode_handles, mode_labels, title="Sampler", loc="lower right"))

    output_path = Path(output_dir) / "validation_frontier.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _write_validation_metric_matrix(summary_rows: pd.DataFrame, output_dir: Path, top_n: int = 10) -> Optional[Path]:
    if summary_rows.empty:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    frame = summary_rows.sort_values(
        ["mean_snapshot_edge_jaccard", "mean_synthetic_novel_edge_rate", "weight_total_correlation"],
        ascending=[False, True, False],
    ).head(top_n).copy()
    metrics = [
        ("mean_snapshot_edge_jaccard", "Edge Jaccard", "higher"),
        ("mean_snapshot_node_jaccard", "Node Jaccard", "higher"),
        ("mean_synthetic_novel_edge_rate", "Novelty", "lower"),
        ("edge_count_correlation", "Edge Corr", "higher"),
    ]
    if "weight_total_correlation" in frame.columns:
        metrics.append(("weight_total_correlation", "Weight Corr", "higher"))
    if "reciprocity_correlation" in frame.columns:
        metrics.append(("reciprocity_correlation", "Reciprocity Corr", "higher"))
    if "tea_new_ratio_correlation" in frame.columns:
        metrics.append(("tea_new_ratio_correlation", "TEA New Corr", "higher"))
    if "tna_new_ratio_correlation" in frame.columns:
        metrics.append(("tna_new_ratio_correlation", "TNA New Corr", "higher"))
    if "pi_mass_mean_correlation" in frame.columns:
        metrics.append(("pi_mass_mean_correlation", "Pi-Mass Corr", "higher"))
    if "magnetic_spectrum_mean_correlation" in frame.columns:
        metrics.append(("magnetic_spectrum_mean_correlation", "Mag Corr", "higher"))

    display = np.zeros((len(frame), len(metrics)), dtype=float)
    quality = np.zeros_like(display)
    for column_index, (column_name, _, direction) in enumerate(metrics):
        values = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        display[:, column_index] = values
        quality[:, column_index] = values if direction == "higher" else 1.0 - values

    fig, ax = plt.subplots(figsize=(11.5, 0.58 * len(frame) + 2.8), constrained_layout=True)
    _style_figure(fig, [ax])
    fig.suptitle("Top Setting Scorecard", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(quality, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)), [metric[1] for metric in metrics])
    ax.set_yticks(np.arange(len(frame)), frame["sample_label"].tolist())
    ax.tick_params(axis="x", rotation=20)
    ax.grid(False)
    for row_index in range(len(frame)):
        for column_index in range(len(metrics)):
            value = display[row_index, column_index]
            metric_name = metrics[column_index][0]
            lower = frame.iloc[row_index].get(f"{metric_name}_q05")
            upper = frame.iloc[row_index].get(f"{metric_name}_q95")
            if pd.notna(lower) and pd.notna(upper):
                cell_text = f"{value:.3f}\n[{float(lower):.3f}, {float(upper):.3f}]"
                fontsize = 7.2
            else:
                cell_text = f"{value:.3f}"
                fontsize = 8.5
            ax.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=_heatmap_annotation_color(quality[row_index, column_index], cmap),
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("Quality scale", rotation=90, color=PLOT_COLORS["text"])

    output_path = Path(output_dir) / "validation_metric_matrix.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _format_html_table(df: pd.DataFrame, *, max_rows: Optional[int] = None, table_id: Optional[str] = None) -> str:
    frame = df.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: f"{float(value):.3f}")
    html_table = frame.to_html(index=False, classes=["report-table", "sortable-table"], border=0, escape=False, table_id=table_id)
    html_table = html_table.replace("<td><div class='setting-cell'>", "<td class='allow-wrap'><div class='setting-cell'>")
    return html_table


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


def _render_html_table_widget(
    df: pd.DataFrame,
    *,
    table_id: str,
    max_rows: Optional[int] = None,
    searchable: bool = True,
    explain_text: Optional[str] = None,
) -> str:
    toolbar_items = []
    if explain_text:
        toolbar_items.append(
            _render_explanation_toggle(
                control_id=f"{table_id}_explain",
                text=explain_text,
            )
        )
    if searchable:
        toolbar_items.append(
            "<div class='table-search-wrap'>"
            f"<label class='table-search-label' for='{html.escape(table_id)}_search'>Filter rows</label>"
            f"<input id='{html.escape(table_id)}_search' class='table-search-input' type='search' "
            f"data-table-target='{html.escape(table_id)}' placeholder='Search this table' />"
            "</div>"
        )
    toolbar_html = f"<div class='table-toolbar'>{''.join(toolbar_items)}</div>" if toolbar_items else ""
    return (
        "<div class='table-widget'>"
        f"{toolbar_html}"
        "<div class='table-scroll'>"
        f"{_format_html_table(df, max_rows=max_rows, table_id=table_id)}"
        "</div>"
        "</div>"
    )


def _load_json_if_exists(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_generated_edge_paths(
    run_dir: Path,
    *,
    setting_label: str,
    run_labels: Optional[list[str]] = None,
) -> list[Path]:
    generated_root = Path(run_dir) / "generated" / str(setting_label)
    paths: list[Path] = []
    labels = [str(label) for label in (run_labels or [])]
    if labels:
        for run_label in labels:
            sample_index = _sample_index_from_label(run_label)
            candidate_paths = []
            if sample_index is not None:
                candidate_paths.append(generated_root / f"sample_{sample_index:04d}" / "synthetic_edges.csv")
            candidate_paths.append(generated_root / run_label / "synthetic_edges.csv")
            for candidate in candidate_paths:
                if candidate.exists():
                    paths.append(candidate)
                    break
    else:
        for candidate in (
            generated_root / "sample_0000" / "synthetic_edges.csv",
            generated_root / "synthetic_edges.csv",
        ):
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def _hybrid_summary_metrics(
    edge_frame: pd.DataFrame,
    *,
    node_type_map: dict[int, str],
    directed: bool,
    weight_col: Optional[str],
) -> dict[str, int]:
    canonical = canonicalise_edge_frame(edge_frame, directed=directed, weight_col=weight_col)
    if canonical.empty:
        return {}

    safe_type_map = {int(node_id): _format_node_type_label(type_label) for node_id, type_label in node_type_map.items()}
    active_nodes = set(pd.to_numeric(canonical["u"], errors="coerce").dropna().astype(int).tolist()) | set(
        pd.to_numeric(canonical["i"], errors="coerce").dropna().astype(int).tolist()
    )
    metrics: dict[str, int] = {}
    metrics["Active nodes"] = int(len(active_nodes))
    type_counter = Counter(safe_type_map.get(node_id, "Unknown") for node_id in active_nodes)
    for type_label, count in sorted(type_counter.items(), key=lambda item: item[0]):
        metrics[f"Active {type_label} nodes"] = int(count)

    edge_type_frame = _compute_edge_type_time_series(
        canonical,
        node_types=safe_type_map,
        directed=directed,
        weight_col=weight_col,
    )
    if len(edge_type_frame):
        totals = (
            edge_type_frame.groupby(["source_type", "target_type"], as_index=False)[["edge_count", "weight_total"]]
            .sum()
            .sort_values(["source_type", "target_type"])
        )
        for row in totals.itertuples(index=False):
            pair_label = _display_type_pair_label(row.source_type, row.target_type, directed=directed)
            metrics[f"{pair_label} edges"] = int(np.rint(float(row.edge_count)))
            if weight_col is not None and pd.notna(row.weight_total):
                metrics[f"{pair_label} weight"] = int(np.rint(float(row.weight_total)))
    return metrics


def _build_hybrid_network_summary_table(
    *,
    run_dir: Path,
    manifest: dict[str, object],
    input_edges: pd.DataFrame,
    hybrid_node_frame: pd.DataFrame,
    directed: bool,
    best_setting_label: str,
    best_setting_run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    if input_edges.empty or hybrid_node_frame.empty:
        return pd.DataFrame()

    weight_col = None
    if isinstance(manifest.get("weight_model"), dict):
        weight_col = manifest.get("weight_model", {}).get("input_column")  # type: ignore[assignment]
    node_type_map = {
        int(row.node_id): _format_node_type_label(row.type_label)
        for row in hybrid_node_frame[["node_id", "type_label"]].itertuples(index=False)
    }
    observed_metrics = _hybrid_summary_metrics(
        input_edges,
        node_type_map=node_type_map,
        directed=directed,
        weight_col=weight_col if isinstance(weight_col, str) else None,
    )
    run_paths = _resolve_generated_edge_paths(
        run_dir,
        setting_label=best_setting_label,
        run_labels=best_setting_run_labels,
    )
    if not observed_metrics:
        return pd.DataFrame()

    synthetic_frames: list[dict[str, int]] = []
    for path in run_paths:
        try:
            synthetic_frame = pd.read_csv(path)
        except Exception:
            continue
        synthetic_metrics = _hybrid_summary_metrics(
            synthetic_frame,
            node_type_map=node_type_map,
            directed=directed,
            weight_col=weight_col if isinstance(weight_col, str) else None,
        )
        if synthetic_metrics:
            synthetic_frames.append(synthetic_metrics)

    row_order = list(observed_metrics.keys())
    for metrics in synthetic_frames:
        for key in metrics.keys():
            if key not in row_order:
                row_order.append(key)

    synthetic_mean: dict[str, Optional[int]] = {}
    for key in row_order:
        values = [metrics.get(key) for metrics in synthetic_frames if metrics.get(key) is not None]
        if values:
            synthetic_mean[key] = int(np.rint(float(np.mean(values))))
        else:
            synthetic_mean[key] = None

    rows: list[dict[str, object]] = []
    for key in row_order:
        observed_value = observed_metrics.get(key)
        synthetic_value = synthetic_mean.get(key)
        rows.append(
            {
                "Hybrid summary": key,
                "Observed network": int(np.rint(float(observed_value))) if observed_value is not None else None,
                "Best setting mean": synthetic_value,
            }
        )
    return pd.DataFrame(rows)



def _selected_setting_detail_row(label: str, diagnostics_dir: Path) -> dict[str, object]:
    summary_payload = _load_json_if_exists(diagnostics_dir / f"{label}_summary.json") or {}
    row = summary_payload_to_row(label, summary_payload) or {"sample_label": label}

    block_pair_summary_path = diagnostics_dir / f"{label}_block_pair_summary.csv"
    if block_pair_summary_path.exists():
        block_pair_df = pd.read_csv(block_pair_summary_path)
        if "weight_total_correlation" in block_pair_df.columns and len(block_pair_df):
            row["mean_block_pair_weight_corr"] = float(block_pair_df["weight_total_correlation"].mean())
            row["min_block_pair_weight_corr"] = float(block_pair_df["weight_total_correlation"].min())
        if "edge_count_correlation" in block_pair_df.columns and len(block_pair_df):
            row["mean_block_pair_edge_corr"] = float(block_pair_df["edge_count_correlation"].mean())

    node_activity_summary_path = diagnostics_dir / f"{label}_node_activity_summary.csv"
    if node_activity_summary_path.exists():
        node_df = pd.read_csv(node_activity_summary_path)
        rank_column = "original_total_incident_weight_total" if "original_total_incident_weight_total" in node_df.columns else "original_total_incident_edge_count"
        top_node_df = node_df.sort_values(rank_column, ascending=False).head(12)
        if "incident_edge_count_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_edge_corr"] = float(top_node_df["incident_edge_count_correlation"].median())
        if "out_edge_count_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_out_edge_corr"] = float(top_node_df["out_edge_count_correlation"].median())
        if "in_edge_count_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_in_edge_corr"] = float(top_node_df["in_edge_count_correlation"].median())
        if "incident_weight_total_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_weight_corr"] = float(top_node_df["incident_weight_total_correlation"].median())
        if "out_weight_total_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_out_weight_corr"] = float(top_node_df["out_weight_total_correlation"].median())
        if "in_weight_total_correlation" in top_node_df.columns and len(top_node_df):
            row["median_top12_node_in_weight_corr"] = float(top_node_df["in_weight_total_correlation"].median())
    return row


def _summarise_log_runs(logs_dir: Path) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(logs_dir.glob("*_summary.json")):
        payload = _load_json_if_exists(summary_path)
        if not payload:
            continue
        label = str(payload.get("label", summary_path.stem))
        command = label.split("_", 1)[0]
        counts_by_level = payload.get("counts_by_level", {}) or {}
        rows.append(
            {
                "command": command,
                "label": label,
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
                "line_count": int(payload.get("line_count", 0) or 0),
                "debug_lines": int(counts_by_level.get("DEBUG", 0) or 0),
                "info_lines": int(counts_by_level.get("INFO", 0) or 0),
                "warning_lines": int(counts_by_level.get("WARNING", 0) or 0),
                "error_lines": int(counts_by_level.get("ERROR", 0) or 0),
            }
        )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("command", as_index=False)
        .agg(
            run_count=("label", "count"),
            median_elapsed_seconds=("elapsed_seconds", "median"),
            max_elapsed_seconds=("elapsed_seconds", "max"),
            mean_line_count=("line_count", "mean"),
            total_warning_lines=("warning_lines", "sum"),
            total_error_lines=("error_lines", "sum"),
        )
        .sort_values("command")
        .reset_index(drop=True)
    )
    return summary


def _write_runtime_profile_plot(logs_dir: Path, output_dir: Path) -> Optional[Path]:
    logs_dir = Path(logs_dir)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(logs_dir.glob("*_summary.json")):
        payload = _load_json_if_exists(summary_path)
        if not payload:
            continue
        label = str(payload.get("label", summary_path.stem))
        rows.append(
            {
                "command": label.split("_", 1)[0],
                "label": label,
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
            }
        )
    if not rows:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    frame = pd.DataFrame(rows)
    command_order = [command for command in ("fit", "generate", "report") if command in frame["command"].unique().tolist()]
    command_palette = {
        "fit": PLOT_COLORS["original"],
        "generate": PLOT_COLORS["synthetic"],
        "report": PLOT_COLORS["accent"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle("Runtime Profile Across Validation Stages", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])

    left_ax, right_ax = axes
    for index, command in enumerate(command_order):
        subset = frame.loc[frame["command"] == command].reset_index(drop=True)
        x_values = np.full(len(subset), float(index))
        if len(subset) > 1:
            x_values = x_values + np.linspace(-0.12, 0.12, len(subset))
        left_ax.scatter(
            x_values,
            subset["elapsed_seconds"].to_numpy(dtype=float),
            s=60,
            color=command_palette.get(command, PLOT_COLORS["neutral"]),
            edgecolors="white",
            linewidths=0.8,
            alpha=0.92,
        )
    left_ax.set_xticks(np.arange(len(command_order)), [command.title() for command in command_order])
    left_ax.set_ylabel("Elapsed seconds")
    left_ax.set_title("Per-run durations")

    runtime_summary = _summarise_log_runs(logs_dir)
    ordered_summary = runtime_summary.set_index("command").reindex(command_order).reset_index()
    y_positions = np.arange(len(ordered_summary), dtype=float)
    right_ax.barh(
        y_positions,
        ordered_summary["median_elapsed_seconds"].to_numpy(dtype=float),
        color=[command_palette.get(command, PLOT_COLORS["neutral"]) for command in ordered_summary["command"]],
        alpha=0.9,
    )
    right_ax.set_yticks(y_positions, [command.title() for command in ordered_summary["command"]])
    right_ax.set_xlabel("Median elapsed seconds")
    right_ax.set_title("Median stage duration")
    right_ax.invert_yaxis()
    for y_position, row in zip(y_positions, ordered_summary.itertuples(index=False)):
        right_ax.text(
            float(row.median_elapsed_seconds) + max(0.2, float(row.median_elapsed_seconds) * 0.03),
            y_position,
            f"n={int(row.run_count)}",
            va="center",
            fontsize=9,
            color=PLOT_COLORS["muted"],
        )

    output_path = Path(output_dir) / "runtime_profile.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def _selected_setting_evidence_tables(label: str, diagnostics_dir: Path, *, directed: bool = False) -> dict[str, pd.DataFrame]:
    diagnostics_dir = Path(diagnostics_dir)
    outputs: dict[str, pd.DataFrame] = {}

    block_pair_path = diagnostics_dir / f"{label}_block_pair_summary.csv"
    if block_pair_path.exists():
        block_pair_df = pd.read_csv(block_pair_path)
        if len(block_pair_df):
            table = block_pair_df.copy()
            table["block_pair"] = table.apply(
                lambda row: f"B{int(row['block_u'])}{'->' if directed else '-'}B{int(row['block_v'])}",
                axis=1,
            )
            block_columns = [
                "block_pair",
                "edge_count_correlation",
                "weight_total_correlation",
                "mean_abs_edge_count_delta",
                "mean_abs_weight_total_delta",
            ]
            outputs["block_pairs"] = table[[column for column in block_columns if column in table.columns]].sort_values("block_pair")

    node_activity_path = diagnostics_dir / f"{label}_node_activity_summary.csv"
    if node_activity_path.exists():
        node_df = pd.read_csv(node_activity_path)
        if len(node_df):
            rank_column = "original_total_incident_weight_total" if "original_total_incident_weight_total" in node_df.columns else "original_total_incident_edge_count"
            node_columns = [
                "node_id",
                "block_id",
                "incident_edge_count_correlation",
                "out_edge_count_correlation",
                "in_edge_count_correlation",
                "incident_weight_total_correlation",
                "out_weight_total_correlation",
                "in_weight_total_correlation",
                "mean_abs_incident_weight_total_delta",
            ]
            outputs["nodes"] = node_df.sort_values(rank_column, ascending=False).head(8)[
                [column for column in node_columns if column in node_df.columns]
            ]

    edge_type_path = diagnostics_dir / f"{label}_edge_type_summary.csv"
    if edge_type_path.exists():
        edge_type_df = pd.read_csv(edge_type_path)
        if len(edge_type_df):
            table = edge_type_df.copy()
            table["edge_type"] = table.apply(
                lambda row: _display_type_pair_label(row["source_type"], row["target_type"], directed=directed),
                axis=1,
            )
            edge_columns = [
                "edge_type",
                "edge_count_correlation",
                "edge_share_correlation",
                "weight_total_correlation",
                "weight_share_correlation",
                "mean_abs_edge_count_delta",
                "mean_abs_weight_total_delta",
            ]
            outputs["edge_types"] = table[[column for column in edge_columns if column in table.columns]].sort_values("edge_type")

    advanced_specs = {
        "tea": diagnostics_dir / f"{label}_tea_summary.csv",
        "tna": diagnostics_dir / f"{label}_tna_summary.csv",
        "temporal_reachability": diagnostics_dir / f"{label}_temporal_reachability_summary.csv",
        "pi_mass": diagnostics_dir / f"{label}_pi_mass_summary.csv",
        "pi_mass_closed": diagnostics_dir / f"{label}_pi_mass_closed_summary.csv",
        "pi_mass_pagerank": diagnostics_dir / f"{label}_pi_mass_pagerank_summary.csv",
        "magnetic": diagnostics_dir / f"{label}_magnetic_laplacian_summary.csv",
        "magnetic_distance": diagnostics_dir / f"{label}_magnetic_spectral_distance_summary.csv",
    }
    for key, path in advanced_specs.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not len(frame):
            continue
        if "metric" in frame.columns:
            frame = frame.copy()
            frame["metric"] = frame["metric"].map(_display_advanced_metric_name)
        if {"source_type", "target_type"}.issubset(frame.columns):
            frame = frame.copy()
            frame["edge_type"] = frame.apply(
                lambda row: _display_type_pair_label(row["source_type"], row["target_type"], directed=directed),
                axis=1,
            )
            ordered_columns = ["edge_type"] + [column for column in frame.columns if column not in {"source_type", "target_type", "edge_type"}]
            frame = frame[ordered_columns]
        if "type_label" in frame.columns:
            frame = frame.copy()
            frame["type_label"] = frame["type_label"].map(_format_node_type_label)
        outputs[key] = frame

    source_path = diagnostics_dir / f"{label}_temporal_reachability_source_summary.csv"
    if source_path.exists():
        source_frame = pd.read_csv(source_path)
        if len(source_frame):
            ordered_columns = [
                column
                for column in (
                    "node_id",
                    "type_label",
                    "original_forward_reach_ratio",
                    "synthetic_forward_reach_ratio",
                    "forward_reach_ratio_delta",
                    "original_static_forward_reach_ratio",
                    "synthetic_static_forward_reach_ratio",
                    "static_forward_reach_ratio_delta",
                )
                if column in source_frame.columns
            ]
            outputs["reachability_sources"] = source_frame[ordered_columns].sort_values(
                [column for column in ("original_forward_reach_ratio", "node_id") if column in source_frame.columns],
                ascending=[False, True][: len([column for column in ("original_forward_reach_ratio", "node_id") if column in source_frame.columns])],
            ).head(12)
    return outputs


def _write_selected_setting_local_fit_plot(
    selected_labels: list[str],
    diagnostics_dir: Path,
    output_dir: Path,
    *,
    top_k_nodes: int = 12,
) -> Optional[Path]:
    if not selected_labels:
        return None

    plt = _load_matplotlib()
    if plt is None:
        return None

    selected_palette = [PLOT_COLORS["original"], PLOT_COLORS["synthetic"], PLOT_COLORS["accent"], PLOT_COLORS["novel"]]
    label_palette = {label: selected_palette[index % len(selected_palette)] for index, label in enumerate(selected_labels)}

    block_rows: list[dict[str, object]] = []
    node_metric_rows: dict[str, list[dict[str, object]]] = {
        "incident_edge_count_correlation": [],
        "out_edge_count_correlation": [],
        "in_edge_count_correlation": [],
        "incident_weight_total_correlation": [],
        "out_weight_total_correlation": [],
        "in_weight_total_correlation": [],
    }
    directed = False
    has_weight = False
    for label in selected_labels:
        block_pair_path = Path(diagnostics_dir) / f"{label}_block_pair_summary.csv"
        if block_pair_path.exists():
            block_pair_df = pd.read_csv(block_pair_path)
            for row in block_pair_df.itertuples(index=False):
                block_rows.append(
                    {
                        "sample_label": label,
                        "block_pair": f"B{int(row.block_u)}-B{int(row.block_v)}",
                        "weight_total_correlation": float(getattr(row, "weight_total_correlation", 0.0) or 0.0),
                    }
                )
        node_path = Path(diagnostics_dir) / f"{label}_node_activity_summary.csv"
        if node_path.exists():
            node_df = pd.read_csv(node_path)
            if len(node_df):
                rank_column = "original_total_incident_weight_total" if "original_total_incident_weight_total" in node_df.columns else "original_total_incident_edge_count"
                top_nodes = node_df.sort_values(rank_column, ascending=False).head(top_k_nodes)
                directed = directed or ("out_edge_count_correlation" in top_nodes.columns and "in_edge_count_correlation" in top_nodes.columns)
                has_weight = has_weight or ("incident_weight_total_correlation" in top_nodes.columns)
                for metric_name in node_metric_rows:
                    if metric_name not in top_nodes.columns:
                        continue
                    for value in pd.to_numeric(top_nodes[metric_name], errors="coerce").dropna().tolist():
                        node_metric_rows[metric_name].append({"sample_label": label, "value": float(value)})

    if not block_rows and not any(node_metric_rows.values()):
        return None

    metric_specs: list[tuple[str, str, str]] = [("block", "weight_total_correlation", "Block-pair weight fidelity")]
    if directed:
        metric_specs.extend(
            [
                ("node", "out_edge_count_correlation", "Top-node outgoing edge fidelity"),
                ("node", "in_edge_count_correlation", "Top-node incoming edge fidelity"),
            ]
        )
        if has_weight:
            metric_specs.extend(
                [
                    ("node", "out_weight_total_correlation", "Top-node outgoing weight fidelity"),
                    ("node", "in_weight_total_correlation", "Top-node incoming weight fidelity"),
                ]
            )
    else:
        metric_specs.append(("node", "incident_edge_count_correlation", "Top-node edge-count fidelity"))
        if has_weight:
            metric_specs.append(("node", "incident_weight_total_correlation", "Top-node weight fidelity"))

    ncols = 3
    nrows = int(math.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), constrained_layout=True)
    _style_figure(fig, axes)
    fig.suptitle("Selected-Setting Local Fit Summary", fontsize=17, fontweight="bold", color=PLOT_COLORS["text"])
    axes_array = np.atleast_1d(axes).ravel()

    label_display = [
        _setting_display_payload(label)["short_label"].replace(" + ", "\n+\n")
        for label in selected_labels
    ]
    block_df = pd.DataFrame(block_rows) if block_rows else pd.DataFrame()

    for ax, (metric_group, metric_name, title_text) in zip(axes_array, metric_specs):
        if metric_group == "block":
            if block_df.empty or metric_name not in block_df.columns:
                ax.axis("off")
                continue
            block_pairs = sorted(block_df["block_pair"].unique().tolist())
            block_pair_labels = [value.replace("-", "->") if directed else value for value in block_pairs]
            y_positions = np.arange(len(block_pairs), dtype=float)
            offsets = np.linspace(-0.18, 0.18, max(1, len(selected_labels)))
            for offset, label in zip(offsets, selected_labels):
                subset = block_df.loc[block_df["sample_label"] == label]
                values = []
                lower_values = []
                upper_values = []
                for block_pair in block_pairs:
                    matches = subset.loc[subset["block_pair"] == block_pair, metric_name]
                    values.append(float(matches.iloc[0]) if len(matches) else np.nan)
                    lower_matches = subset.loc[subset["block_pair"] == block_pair, f"{metric_name}_q05"] if f"{metric_name}_q05" in subset.columns else pd.Series(dtype=float)
                    upper_matches = subset.loc[subset["block_pair"] == block_pair, f"{metric_name}_q95"] if f"{metric_name}_q95" in subset.columns else pd.Series(dtype=float)
                    lower_values.append(float(lower_matches.iloc[0]) if len(lower_matches) else np.nan)
                    upper_values.append(float(upper_matches.iloc[0]) if len(upper_matches) else np.nan)
                ax.barh(
                    y_positions + offset,
                    values,
                    height=0.3,
                    color=label_palette[label],
                    alpha=0.9,
                    label=_setting_display_payload(label)["short_label"],
                )
                if np.isfinite(lower_values).any() and np.isfinite(upper_values).any():
                    ax.hlines(y_positions + offset, lower_values, upper_values, color=PLOT_COLORS["text"], linewidth=1.0, alpha=0.55, zorder=4)
            ax.set_yticks(y_positions, block_pair_labels)
            ax.set_xlim(0.0, 1.05)
            ax.set_xlabel("Correlation")
            ax.set_title(title_text)
            ax.invert_yaxis()
            _style_legend(ax.legend(fontsize=8, loc="lower right"))
            continue

        rows = node_metric_rows.get(metric_name, [])
        if not rows:
            ax.axis("off")
            continue
        frame = pd.DataFrame(rows)
        positions = np.arange(len(selected_labels), dtype=float)
        box_data = [frame.loc[frame["sample_label"] == label, "value"].to_numpy(dtype=float) for label in selected_labels]
        box = ax.boxplot(
            box_data,
            vert=False,
            patch_artist=True,
            positions=positions,
            widths=0.5,
            medianprops={"color": PLOT_COLORS["text"], "linewidth": 1.2},
            boxprops={"linewidth": 0.9, "edgecolor": PLOT_COLORS["grid_strong"]},
            whiskerprops={"linewidth": 0.9, "color": PLOT_COLORS["grid_strong"]},
            capprops={"linewidth": 0.9, "color": PLOT_COLORS["grid_strong"]},
        )
        for patch, label in zip(box["boxes"], selected_labels):
            patch.set_facecolor(label_palette[label])
            patch.set_alpha(0.32)
        for position, label in zip(positions, selected_labels):
            values = frame.loc[frame["sample_label"] == label, "value"].to_numpy(dtype=float)
            if not len(values):
                continue
            y_values = np.full(len(values), position) + np.linspace(-0.12, 0.12, len(values))
            ax.scatter(values, y_values, s=38, color=label_palette[label], edgecolors="white", linewidths=0.7, alpha=0.95, zorder=3)
        ax.set_yticks(positions, label_display)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("Correlation")
        ax.set_title(title_text)

    for ax in axes_array[len(metric_specs):]:
        ax.axis("off")

    output_path = Path(output_dir) / "selected_settings_local_fit.png"
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path

def summary_payload_to_row(
    label: str,
    payload: dict[str, object],
    *,
    extra: Optional[dict[str, object]] = None,
) -> Optional[dict[str, object]]:
    if "mean_snapshot_edge_jaccard" not in payload:
        return None
    row: dict[str, object] = {"sample_label": label}
    for key, value in payload.items():
        if key in {"posterior_run_labels"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    if extra:
        row.update(extra)
    return row


def _aggregate_grouped_numeric_frames(
    frames: list[pd.DataFrame],
    *,
    group_keys: list[str],
    run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=group_keys)

    tagged_frames: list[pd.DataFrame] = []
    run_labels = run_labels or [f"run_{index:04d}" for index in range(len(non_empty))]
    for index, frame in enumerate(non_empty):
        tagged = frame.copy()
        tagged["__run_label"] = str(run_labels[index] if index < len(run_labels) else f"run_{index:04d}")
        tagged_frames.append(tagged)
    combined = pd.concat(tagged_frames, ignore_index=True, sort=False)

    value_columns = [column for column in combined.columns if column not in set(group_keys) | {"__run_label"}]
    grouped = combined.groupby(group_keys, dropna=False, sort=True)
    output = grouped["__run_label"].nunique().rename("posterior_num_runs").reset_index()

    numeric_columns: list[str] = []
    for column in value_columns:
        numeric = pd.to_numeric(combined[column], errors="coerce")
        if numeric.notna().any():
            combined[column] = numeric.astype(float)
            numeric_columns.append(column)

    if numeric_columns:
        numeric_grouped = combined.groupby(group_keys, dropna=False, sort=True)[numeric_columns]
        output = output.merge(numeric_grouped.median().reset_index(), on=group_keys, how="left", sort=False)
        output = output.merge(
            numeric_grouped.quantile(0.05).add_suffix("_q05").reset_index(),
            on=group_keys,
            how="left",
            sort=False,
        )
        output = output.merge(
            numeric_grouped.quantile(0.95).add_suffix("_q95").reset_index(),
            on=group_keys,
            how="left",
            sort=False,
        )
        output = output.merge(
            numeric_grouped.mean().add_suffix("_mean").reset_index(),
            on=group_keys,
            how="left",
            sort=False,
        )
        output = output.merge(
            numeric_grouped.std(ddof=0).add_suffix("_std").reset_index(),
            on=group_keys,
            how="left",
            sort=False,
        )

    non_numeric_columns = [column for column in value_columns if column not in numeric_columns]
    if non_numeric_columns:
        non_numeric = grouped[non_numeric_columns].first().reset_index()
        output = output.merge(non_numeric, on=group_keys, how="left", sort=False)

    if len(output):
        single_run_mask = output["posterior_num_runs"] <= 1
        if single_run_mask.any():
            for column in numeric_columns:
                for suffix in ("_q05", "_q95", "_mean", "_std"):
                    derived_column = f"{column}{suffix}"
                    if derived_column in output.columns:
                        output.loc[single_run_mask, derived_column] = np.nan
        output = output.sort_values(group_keys).reset_index(drop=True)
    return output


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
            continue
        non_null = series.dropna()
        if len(non_null):
            payload[column] = non_null.iloc[0]
    return payload


def _tag_posterior_run_frames(
    frames: list[pd.DataFrame],
    *,
    run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    run_labels = run_labels or [f"run_{index:04d}" for index in range(len(non_empty))]
    tagged_frames: list[pd.DataFrame] = []
    for index, frame in enumerate(non_empty):
        tagged = frame.copy()
        tagged["__run_label"] = str(run_labels[index] if index < len(run_labels) else f"run_{index:04d}")
        tagged_frames.append(tagged)
    return pd.concat(tagged_frames, ignore_index=True, sort=False)


def _metric_names_from_original_columns(frame: pd.DataFrame) -> list[str]:
    metric_names: list[str] = []
    for column in frame.columns:
        if not str(column).startswith("original_"):
            continue
        metric_name = str(column).removeprefix("original_")
        if f"synthetic_{metric_name}" in frame.columns:
            metric_names.append(metric_name)
    return sorted(set(metric_names))


def _posterior_entity_correlation_details(
    frames: list[pd.DataFrame],
    *,
    entity_keys: list[str],
    metric_names: list[str],
    run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    combined = _tag_posterior_run_frames(frames, run_labels=run_labels)
    entity_keys = [key for key in entity_keys if key in combined.columns]
    if combined.empty or not entity_keys:
        return pd.DataFrame(columns=entity_keys)

    rows: list[dict[str, object]] = []
    for entity_values, entity_frame in combined.groupby(entity_keys, sort=True, dropna=False):
        if not isinstance(entity_values, tuple):
            entity_values = (entity_values,)
        row = {key: value for key, value in zip(entity_keys, entity_values)}
        for metric_name in metric_names:
            original_col = f"original_{metric_name}"
            synthetic_col = f"synthetic_{metric_name}"
            if original_col not in entity_frame.columns or synthetic_col not in entity_frame.columns:
                continue
            row[f"{metric_name}_pooled_correlation"] = _safe_correlation(entity_frame[original_col], entity_frame[synthetic_col])
            run_corrs = [
                _safe_correlation(run_frame[original_col], run_frame[synthetic_col])
                for _, run_frame in entity_frame.groupby("__run_label", sort=True)
            ]
            if run_corrs:
                run_series = pd.Series(run_corrs, dtype=float)
                row[f"{metric_name}_run_median_correlation"] = float(run_series.median())
                row[f"{metric_name}_run_mean_correlation"] = float(run_series.mean())
                if len(run_series) > 1:
                    row[f"{metric_name}_run_q05_correlation"] = float(run_series.quantile(0.05))
                    row[f"{metric_name}_run_q95_correlation"] = float(run_series.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def _posterior_metric_correlation_details(
    frames: list[pd.DataFrame],
    *,
    metric_names: list[str],
    run_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    combined = _tag_posterior_run_frames(frames, run_labels=run_labels)
    columns = ["metric", "pooled_correlation", "run_median_correlation", "run_mean_correlation", "run_q05_correlation", "run_q95_correlation"]
    if combined.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        original_col = f"original_{metric_name}"
        synthetic_col = f"synthetic_{metric_name}"
        if original_col not in combined.columns or synthetic_col not in combined.columns:
            continue
        row: dict[str, object] = {
            "metric": metric_name,
            "pooled_correlation": _safe_correlation(combined[original_col], combined[synthetic_col]),
        }
        run_corrs = [
            _safe_correlation(run_frame[original_col], run_frame[synthetic_col])
            for _, run_frame in combined.groupby("__run_label", sort=True)
        ]
        if run_corrs:
            run_series = pd.Series(run_corrs, dtype=float)
            row["run_median_correlation"] = float(run_series.median())
            row["run_mean_correlation"] = float(run_series.mean())
            if len(run_series) > 1:
                row["run_q05_correlation"] = float(run_series.quantile(0.05))
                row["run_q95_correlation"] = float(run_series.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _merge_posterior_correlation_details(
    summary_frame: pd.DataFrame,
    detail_frame: pd.DataFrame,
    *,
    on_keys: list[str],
) -> pd.DataFrame:
    if summary_frame.empty or detail_frame.empty:
        return summary_frame
    merge_keys = [key for key in on_keys if key in summary_frame.columns and key in detail_frame.columns]
    if not merge_keys:
        return summary_frame
    merged = summary_frame.merge(detail_frame, on=merge_keys, how="left", sort=False)
    return merged


def _recompute_posterior_detail_summaries(
    detailed_outputs: dict[str, pd.DataFrame],
    *,
    directed: bool,
) -> dict[str, pd.DataFrame]:
    recomputed = dict(detailed_outputs)

    def has_metric(frame: pd.DataFrame, metric_name: str) -> bool:
        return f"original_{metric_name}" in frame.columns and f"synthetic_{metric_name}" in frame.columns

    block_pair = recomputed.get("block_pair_per_snapshot")
    if block_pair is not None and not block_pair.empty:
        metrics = ["edge_count"]
        if has_metric(block_pair, "weight_total"):
            metrics.append("weight_total")
        recomputed["block_pair_summary"] = _summarise_entity_time_series(block_pair, ["block_u", "block_v"], metrics)

    block_activity = recomputed.get("block_activity_per_snapshot")
    if block_activity is not None and not block_activity.empty:
        metrics = ["incident_edge_count"]
        for metric_name in ("out_edge_count", "in_edge_count", "incident_weight_total", "out_weight_total", "in_weight_total"):
            if has_metric(block_activity, metric_name):
                metrics.append(metric_name)
        recomputed["block_activity_summary"] = _summarise_entity_time_series(block_activity, ["block_id"], metrics)

    node_activity = recomputed.get("node_activity_per_snapshot")
    if node_activity is not None and not node_activity.empty:
        entity_keys = ["node_id"] + (["block_id"] if "block_id" in node_activity.columns else [])
        metrics = ["incident_edge_count"]
        for metric_name in ("out_edge_count", "in_edge_count", "incident_weight_total", "out_weight_total", "in_weight_total"):
            if has_metric(node_activity, metric_name):
                metrics.append(metric_name)
        recomputed["node_activity_summary"] = _summarise_entity_time_series(node_activity, entity_keys, metrics)

    edge_type = recomputed.get("edge_type_per_snapshot")
    if edge_type is not None and not edge_type.empty:
        metrics = ["edge_count", "edge_share"]
        for metric_name in ("weight_total", "weight_share"):
            if has_metric(edge_type, metric_name):
                metrics.append(metric_name)
        recomputed["edge_type_summary"] = _summarise_entity_time_series(edge_type, ["source_type", "target_type"], metrics)

    tea = recomputed.get("tea_per_snapshot")
    if tea is not None and not tea.empty:
        tea_metrics = [
            "new_count",
            "reactivated_count",
            "persist_count",
            "ceased_prev_count",
            "repeated_count",
            "total_count",
            "new_ratio",
            "persist_ratio",
            "reactivated_ratio",
            "churn_ratio",
        ]
        recomputed["tea_summary"] = _summarise_metric_time_series(tea, tea_metrics)

    tea_type_pair = recomputed.get("tea_type_pair_per_snapshot")
    if tea_type_pair is not None and not tea_type_pair.empty:
        tea_type_pair_summary = _summarise_entity_time_series(
            tea_type_pair,
            ["source_type", "target_type"],
            ["new_count", "reactivated_count", "persist_count", "ceased_prev_count", "total_count", "new_ratio", "persist_ratio", "reactivated_ratio", "churn_ratio"],
        )
        if "new_ratio_correlation" in tea_type_pair_summary.columns:
            tea_type_pair_summary["birth_rate_correlation"] = tea_type_pair_summary["new_ratio_correlation"]
        if "mean_abs_new_ratio_delta" in tea_type_pair_summary.columns:
            tea_type_pair_summary["mean_abs_birth_rate_delta"] = tea_type_pair_summary["mean_abs_new_ratio_delta"]
        recomputed["tea_type_pair_summary"] = tea_type_pair_summary

    tna = recomputed.get("tna_per_snapshot")
    if tna is not None and not tna.empty:
        tna_metrics = [
            "new_count",
            "reactivated_count",
            "persist_count",
            "ceased_prev_count",
            "repeated_count",
            "total_count",
            "new_ratio",
            "persist_ratio",
            "reactivated_ratio",
            "churn_ratio",
        ]
        recomputed["tna_summary"] = _summarise_metric_time_series(tna, tna_metrics)

    tna_type = recomputed.get("tna_type_per_snapshot")
    if tna_type is not None and not tna_type.empty:
        tna_type_summary = _summarise_entity_time_series(
            tna_type,
            ["type_label"],
            ["new_count", "reactivated_count", "persist_count", "ceased_prev_count", "total_count", "new_ratio", "persist_ratio", "reactivated_ratio", "churn_ratio"],
        )
        if "new_ratio_correlation" in tna_type_summary.columns:
            tna_type_summary["new_rate_correlation"] = tna_type_summary["new_ratio_correlation"]
        if "mean_abs_new_ratio_delta" in tna_type_summary.columns:
            tna_type_summary["mean_abs_new_rate_delta"] = tna_type_summary["mean_abs_new_ratio_delta"]
        recomputed["tna_type_summary"] = tna_type_summary

    reachability = recomputed.get("temporal_reachability_per_snapshot")
    if reachability is not None and not reachability.empty:
        reachability_metrics = [
            "reachable_pair_count",
            "reachability_ratio",
            "new_reachable_pair_count",
            "temporal_efficiency",
            "mean_arrival_time_reached",
        ]
        recomputed["temporal_reachability_summary"] = _summarise_metric_time_series(
            reachability,
            reachability_metrics,
            treat_missing_as_zero=False,
        )

    for base_key in ("pi_mass", "pi_mass_closed", "pi_mass_pagerank", "magnetic_laplacian"):
        frame = recomputed.get(f"{base_key}_per_snapshot")
        if frame is None or frame.empty:
            continue
        metric_names = _metric_names_from_original_columns(frame)
        recomputed[f"{base_key}_summary"] = _summarise_metric_time_series(
            frame,
            metric_names,
            treat_missing_as_zero=not base_key.startswith("pi_mass"),
        )

    magnetic_distance = recomputed.get("magnetic_spectral_distance_per_snapshot")
    if magnetic_distance is not None and not magnetic_distance.empty:
        recomputed["magnetic_spectral_distance_summary"] = _summarise_distance_time_series(
            magnetic_distance,
            ["spectral_wasserstein_distance", "spectral_mean_abs_delta", "spectral_rmse", "spectral_max_abs_delta"],
        )

    return recomputed


def _update_summary_with_recomputed_diagnostics(
    summary: dict[str, object],
    *,
    per_snapshot: pd.DataFrame,
    detailed_outputs: dict[str, pd.DataFrame],
    run_level_per_snapshot: list[pd.DataFrame],
    run_level_details: dict[str, list[pd.DataFrame]],
) -> None:
    summary["time_series_correlation_method"] = TIME_SERIES_CORRELATION_METHOD
    if len(per_snapshot) >= 2:
        summary["edge_count_correlation"] = _safe_correlation(per_snapshot.get("original_edge_count", []), per_snapshot.get("synthetic_edge_count", []))
    summary["edge_count_pooled_correlation"] = _safe_correlation(
        pd.concat([frame.get("original_edge_count", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
        pd.concat([frame.get("synthetic_edge_count", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
    )
    if "original_weight_total" in per_snapshot.columns and "synthetic_weight_total" in per_snapshot.columns:
        summary["weight_total_correlation"] = _safe_correlation(per_snapshot["original_weight_total"], per_snapshot["synthetic_weight_total"])
        summary["weight_total_pooled_correlation"] = _safe_correlation(
            pd.concat([frame.get("original_weight_total", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
            pd.concat([frame.get("synthetic_weight_total", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
        )
    if "original_reciprocity" in per_snapshot.columns and "synthetic_reciprocity" in per_snapshot.columns:
        summary["reciprocity_correlation"] = _safe_correlation(per_snapshot["original_reciprocity"], per_snapshot["synthetic_reciprocity"])
        summary["reciprocity_pooled_correlation"] = _safe_correlation(
            pd.concat([frame.get("original_reciprocity", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
            pd.concat([frame.get("synthetic_reciprocity", pd.Series(dtype=float)) for frame in run_level_per_snapshot], ignore_index=True),
        )

    reachability_summary = detailed_outputs.get("temporal_reachability_summary", pd.DataFrame())
    if len(reachability_summary):
        summary["temporal_reachability_ratio_correlation"] = _metric_lookup(reachability_summary, "reachability_ratio", "correlation") or 0.0
        summary["temporal_efficiency_correlation"] = _metric_lookup(reachability_summary, "temporal_efficiency", "correlation") or 0.0
        summary["temporal_new_reachable_pair_count_correlation"] = _metric_lookup(reachability_summary, "new_reachable_pair_count", "correlation") or 0.0
        summary["temporal_mean_arrival_time_correlation"] = _metric_lookup(reachability_summary, "mean_arrival_time_reached", "correlation") or 0.0
        reachability_run_details = _posterior_metric_correlation_details(
            run_level_details.get("temporal_reachability_per_snapshot", []),
            metric_names=["reachability_ratio", "temporal_efficiency", "new_reachable_pair_count", "mean_arrival_time_reached"],
        )
        summary["temporal_reachability_ratio_pooled_correlation"] = _metric_lookup(reachability_run_details, "reachability_ratio", "pooled_correlation") or 0.0
        summary["temporal_efficiency_pooled_correlation"] = _metric_lookup(reachability_run_details, "temporal_efficiency", "pooled_correlation") or 0.0
        summary["temporal_new_reachable_pair_count_pooled_correlation"] = _metric_lookup(reachability_run_details, "new_reachable_pair_count", "pooled_correlation") or 0.0
        summary["temporal_mean_arrival_time_pooled_correlation"] = _metric_lookup(reachability_run_details, "mean_arrival_time_reached", "pooled_correlation") or 0.0
    source_frame = detailed_outputs.get("temporal_reachability_source_summary", pd.DataFrame())
    if len(source_frame) and "original_forward_reach_ratio" in source_frame.columns and "synthetic_forward_reach_ratio" in source_frame.columns:
        summary["temporal_forward_reach_node_correlation"] = _safe_correlation(
            source_frame["original_forward_reach_ratio"],
            source_frame["synthetic_forward_reach_ratio"],
        )

    tea_summary = detailed_outputs.get("tea_summary", pd.DataFrame())
    if len(tea_summary):
        summary["tea_new_ratio_correlation"] = _metric_lookup(tea_summary, "new_ratio", "correlation") or 0.0
        summary["tea_persist_ratio_correlation"] = _metric_lookup(tea_summary, "persist_ratio", "correlation") or 0.0
        summary["tea_reactivated_ratio_correlation"] = _metric_lookup(tea_summary, "reactivated_ratio", "correlation") or 0.0
        tea_run_details = _posterior_metric_correlation_details(run_level_details.get("tea_per_snapshot", []), metric_names=["new_ratio", "persist_ratio", "reactivated_ratio"])
        summary["tea_new_ratio_pooled_correlation"] = _metric_lookup(tea_run_details, "new_ratio", "pooled_correlation") or 0.0
        summary["tea_persist_ratio_pooled_correlation"] = _metric_lookup(tea_run_details, "persist_ratio", "pooled_correlation") or 0.0
        summary["tea_reactivated_ratio_pooled_correlation"] = _metric_lookup(tea_run_details, "reactivated_ratio", "pooled_correlation") or 0.0

    tna_summary = detailed_outputs.get("tna_summary", pd.DataFrame())
    if len(tna_summary):
        summary["tna_new_ratio_correlation"] = _metric_lookup(tna_summary, "new_ratio", "correlation") or 0.0
        summary["tna_persist_ratio_correlation"] = _metric_lookup(tna_summary, "persist_ratio", "correlation") or 0.0
        summary["tna_reactivated_ratio_correlation"] = _metric_lookup(tna_summary, "reactivated_ratio", "correlation") or 0.0
        tna_run_details = _posterior_metric_correlation_details(run_level_details.get("tna_per_snapshot", []), metric_names=["new_ratio", "persist_ratio", "reactivated_ratio"])
        summary["tna_new_ratio_pooled_correlation"] = _metric_lookup(tna_run_details, "new_ratio", "pooled_correlation") or 0.0
        summary["tna_persist_ratio_pooled_correlation"] = _metric_lookup(tna_run_details, "persist_ratio", "pooled_correlation") or 0.0
        summary["tna_reactivated_ratio_pooled_correlation"] = _metric_lookup(tna_run_details, "reactivated_ratio", "pooled_correlation") or 0.0

    edge_type_summary = detailed_outputs.get("edge_type_summary", pd.DataFrame())
    if len(edge_type_summary):
        if "edge_share_correlation" in edge_type_summary.columns:
            summary["edge_type_share_correlation"] = float(pd.to_numeric(edge_type_summary["edge_share_correlation"], errors="coerce").fillna(0.0).mean())
        if "weight_share_correlation" in edge_type_summary.columns:
            summary["edge_type_weight_share_correlation"] = float(pd.to_numeric(edge_type_summary["weight_share_correlation"], errors="coerce").fillna(0.0).mean())

    def set_pi_summary_fields(
        base_key: str,
        mean_key: str,
        gini_key: str,
        share_key: str,
        size_key: str,
        share_metric: str,
        extra_metric_keys: Optional[dict[str, str]] = None,
    ) -> None:
        frame = detailed_outputs.get(f"{base_key}_summary", pd.DataFrame())
        if frame.empty:
            return
        pi_rows = frame.loc[frame["metric"].astype(str).str.startswith("pi_mass__")]
        if len(pi_rows):
            summary[mean_key] = float(pd.to_numeric(pi_rows["correlation"], errors="coerce").fillna(0.0).mean())
        summary[gini_key] = _metric_lookup(frame, "pi_gini", "correlation") or 0.0
        summary[share_key] = _metric_lookup(frame, share_metric, "correlation") or 0.0
        summary[size_key] = _metric_lookup(frame, "lic_size", "correlation") or 0.0
        for metric_name, metric_key in (extra_metric_keys or {}).items():
            summary[metric_key] = _metric_lookup(frame, metric_name, "correlation") or 0.0
        run_frames = run_level_details.get(f"{base_key}_per_snapshot", [])
        run_detail = _posterior_metric_correlation_details(
            run_frames,
            metric_names=_metric_names_from_original_columns(run_frames[0]) if run_frames else [],
        )
        if len(run_detail):
            pooled_rows = run_detail.loc[run_detail["metric"].astype(str).str.startswith("pi_mass__")]
            if len(pooled_rows):
                summary[f"{mean_key.removesuffix('_correlation')}_pooled_correlation"] = float(pd.to_numeric(pooled_rows["pooled_correlation"], errors="coerce").fillna(0.0).mean())
            pooled_gini = _metric_lookup(run_detail, "pi_gini", "pooled_correlation")
            pooled_share = _metric_lookup(run_detail, share_metric, "pooled_correlation")
            pooled_size = _metric_lookup(run_detail, "lic_size", "pooled_correlation")
            if pooled_gini is not None:
                summary[f"{gini_key.removesuffix('_correlation')}_pooled_correlation"] = pooled_gini
            if pooled_share is not None:
                summary[f"{share_key.removesuffix('_correlation')}_pooled_correlation"] = pooled_share
            if pooled_size is not None:
                summary[f"{size_key.removesuffix('_correlation')}_pooled_correlation"] = pooled_size
            for metric_name, metric_key in (extra_metric_keys or {}).items():
                pooled_value = _metric_lookup(run_detail, metric_name, "pooled_correlation")
                if pooled_value is not None:
                    summary[f"{metric_key.removesuffix('_correlation')}_pooled_correlation"] = pooled_value

    set_pi_summary_fields(
        "pi_mass",
        "pi_mass_mean_correlation",
        "pi_gini_correlation",
        "lic_share_active_correlation",
        "lic_size_correlation",
        "lic_share_active",
        extra_metric_keys={
            "active_node_count": "lic_active_node_count_correlation",
            "active_farm_count": "lic_active_farm_count_correlation",
            "active_region_count": "lic_active_region_count_correlation",
        },
    )
    set_pi_summary_fields(
        "pi_mass_closed",
        "pi_mass_closed_mean_correlation",
        "pi_mass_closed_gini_correlation",
        "closed_class_share_active_correlation",
        "closed_class_size_correlation",
        "lic_share_active",
        extra_metric_keys={
            "active_node_count": "closed_class_active_node_count_correlation",
            "active_farm_count": "closed_class_active_farm_count_correlation",
            "active_region_count": "closed_class_active_region_count_correlation",
        },
    )
    set_pi_summary_fields(
        "pi_mass_pagerank",
        "pi_mass_pagerank_mean_correlation",
        "pi_mass_pagerank_gini_correlation",
        "pagerank_support_share_total_correlation",
        "pagerank_support_size_correlation",
        "lic_share_total",
        extra_metric_keys={
            "active_node_count": "pagerank_active_node_count_correlation",
            "active_farm_count": "pagerank_active_farm_count_correlation",
            "active_region_count": "pagerank_active_region_count_correlation",
        },
    )

    magnetic_summary = detailed_outputs.get("magnetic_laplacian_summary", pd.DataFrame())
    if len(magnetic_summary):
        summary["magnetic_spectrum_mean_correlation"] = float(pd.to_numeric(magnetic_summary["correlation"], errors="coerce").fillna(0.0).mean())
        summary["magnetic_spectrum_mean_abs_delta"] = float(pd.to_numeric(magnetic_summary["mean_abs_delta"], errors="coerce").fillna(0.0).mean())
        magnetic_run_detail = _posterior_metric_correlation_details(
            run_level_details.get("magnetic_laplacian_per_snapshot", []),
            metric_names=_metric_names_from_original_columns(run_level_details.get("magnetic_laplacian_per_snapshot", [pd.DataFrame()])[0]) if run_level_details.get("magnetic_laplacian_per_snapshot") else [],
        )
        if len(magnetic_run_detail):
            summary["magnetic_spectrum_pooled_correlation"] = float(pd.to_numeric(magnetic_run_detail["pooled_correlation"], errors="coerce").fillna(0.0).mean())

def aggregate_posterior_reports(
    reports: list[dict[str, object]],
    *,
    output_dir: Path,
    setting_label: str,
    directed: bool,
    diagnostic_top_k: int = 12,
    skip_detail_aggregation: bool = False,
    skip_spectral_metrics: bool = False,
) -> dict[str, object]:
    if not reports:
        raise ValueError(f"No reports were provided for posterior aggregation of setting '{setting_label}'.")

    run_labels = [str(report.get("sample_label")) for report in reports]
    per_snapshot_frames = [pd.read_csv(Path(report["outputs"]["per_snapshot_csv"])) for report in reports]
    per_snapshot = _aggregate_grouped_numeric_frames(per_snapshot_frames, group_keys=["ts"], run_labels=run_labels)
    summary = _aggregate_summary_payloads(setting_label, [dict(report["summary"]) for report in reports], run_labels=run_labels)

    detailed_outputs: dict[str, pd.DataFrame] = {}
    if skip_detail_aggregation:
        LOGGER.info(
            "Skipping posterior detail aggregation for setting summary | setting_label=%s",
            setting_label,
        )
    else:
        raw_detail_frames: dict[str, list[pd.DataFrame]] = {}
        for detail_key, group_keys in POSTERIOR_DETAIL_GROUP_KEYS.items():
            frames: list[pd.DataFrame] = []
            for report in reports:
                outputs = dict(report.get("outputs", {}))
                detail_path = outputs.get(detail_key)
                if not detail_path:
                    continue
                path = Path(str(detail_path))
                if path.exists():
                    frames.append(pd.read_csv(path))
            if frames:
                raw_detail_frames[detail_key] = frames
                detailed_outputs[detail_key] = _aggregate_grouped_numeric_frames(frames, group_keys=group_keys, run_labels=run_labels)

        detailed_outputs = _recompute_posterior_detail_summaries(
            detailed_outputs,
            directed=directed,
        )

        entity_summary_specs = [
            ("block_pair_per_snapshot", "block_pair_summary", ["block_u", "block_v"]),
            ("block_activity_per_snapshot", "block_activity_summary", ["block_id"]),
            ("node_activity_per_snapshot", "node_activity_summary", ["node_id", "block_id"]),
            ("edge_type_per_snapshot", "edge_type_summary", ["source_type", "target_type"]),
            ("tea_type_pair_per_snapshot", "tea_type_pair_summary", ["source_type", "target_type"]),
            ("tna_type_per_snapshot", "tna_type_summary", ["type_label"]),
        ]
        for per_snapshot_key, summary_key, entity_keys in entity_summary_specs:
            frames = raw_detail_frames.get(per_snapshot_key, [])
            summary_frame = detailed_outputs.get(summary_key)
            if not frames or summary_frame is None or summary_frame.empty:
                continue
            entity_metric_names = sorted(
                {
                    column.removesuffix("_correlation")
                    for column in summary_frame.columns
                    if str(column).endswith("_correlation") and not str(column).startswith("mean_abs_") and not str(column).startswith("max_abs_")
                }
            )
            detail_frame = _posterior_entity_correlation_details(
                frames,
                entity_keys=entity_keys,
                metric_names=entity_metric_names,
                run_labels=run_labels,
            )
            detailed_outputs[summary_key] = _merge_posterior_correlation_details(summary_frame, detail_frame, on_keys=entity_keys)

        metric_summary_specs = [
            ("tea_per_snapshot", "tea_summary"),
            ("tna_per_snapshot", "tna_summary"),
            ("temporal_reachability_per_snapshot", "temporal_reachability_summary"),
            ("pi_mass_per_snapshot", "pi_mass_summary"),
            ("pi_mass_closed_per_snapshot", "pi_mass_closed_summary"),
            ("pi_mass_pagerank_per_snapshot", "pi_mass_pagerank_summary"),
        ]
        if not skip_spectral_metrics:
            metric_summary_specs.append(("magnetic_laplacian_per_snapshot", "magnetic_laplacian_summary"))
        for per_snapshot_key, summary_key in metric_summary_specs:
            frames = raw_detail_frames.get(per_snapshot_key, [])
            summary_frame = detailed_outputs.get(summary_key)
            if not frames or summary_frame is None or summary_frame.empty or "metric" not in summary_frame.columns:
                continue
            metric_names = summary_frame["metric"].astype(str).tolist()
            detail_frame = _posterior_metric_correlation_details(
                frames,
                metric_names=metric_names,
                run_labels=run_labels,
            )
            detailed_outputs[summary_key] = _merge_posterior_correlation_details(summary_frame, detail_frame, on_keys=["metric"])

        if skip_spectral_metrics:
            detailed_outputs.pop("magnetic_laplacian_per_snapshot", None)
            detailed_outputs.pop("magnetic_laplacian_summary", None)
            detailed_outputs.pop("magnetic_spectral_distance_per_snapshot", None)
            detailed_outputs.pop("magnetic_spectral_distance_summary", None)
            raw_detail_frames.pop("magnetic_laplacian_per_snapshot", None)

        _update_summary_with_recomputed_diagnostics(
            summary,
            per_snapshot=per_snapshot,
            detailed_outputs=detailed_outputs,
            run_level_per_snapshot=per_snapshot_frames,
            run_level_details=raw_detail_frames,
        )

    report_paths = write_report(
        per_snapshot=per_snapshot,
        summary=summary,
        output_dir=output_dir,
        sample_label=setting_label,
        detailed_diagnostics=detailed_outputs or None,
        directed=directed,
        diagnostic_top_k=diagnostic_top_k,
    )
    return {
        "sample_label": setting_label,
        "setting_label": setting_label,
        "run_labels": run_labels,
        "summary": summary,
        "outputs": report_paths,
    }


def _summary_json_to_row(label: str, payload: dict[str, object]) -> Optional[dict[str, object]]:
    return summary_payload_to_row(label, payload)


def _load_sweep_summary_rows(diagnostics_dir: Path) -> pd.DataFrame:
    diagnostics_dir = Path(diagnostics_dir)
    summary_candidates = [
        diagnostics_dir / "setting_posterior_summary.csv",
        diagnostics_dir / "all_samples_summary.csv",
        diagnostics_dir / "novelty_grid_summary.csv",
    ]
    summary_frame = pd.DataFrame()
    for candidate in summary_candidates:
        if candidate.exists():
            candidate_frame = pd.read_csv(candidate)
            if len(candidate_frame):
                summary_frame = candidate_frame
                break

    summary_json_rows: list[dict[str, object]] = []
    for path in sorted(diagnostics_dir.glob("*_summary.json")):
        label = path.name[: -len("_summary.json")]
        payload = _load_json_if_exists(path)
        if not isinstance(payload, dict):
            continue
        row = _summary_json_to_row(label, payload)
        if row is not None:
            summary_json_rows.append(row)
    summary_json_frame = pd.DataFrame(summary_json_rows)

    if summary_frame.empty and len(summary_json_frame):
        summary_frame = summary_json_frame
    if summary_frame.empty:
        raise FileNotFoundError(f"No summary CSV or per-setting summary JSON available under {diagnostics_dir}")
    if "sample_label" in summary_frame.columns:
        summary_frame = summary_frame.drop_duplicates(subset=["sample_label"]).reset_index(drop=True)
    return summary_frame


def write_scientific_validation_report(
    run_dir: Path,
    *,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    skip_spectral_metrics: bool = False,
    include_daily_network_snapshots: bool = False,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    diagnostics_dir = run_dir / "diagnostics"
    manifest = _load_json_if_exists(run_dir / "manifest.json") or {}

    summary_rows = _attach_sweep_metadata(_load_sweep_summary_rows(diagnostics_dir), run_dir)
    summary_rows.to_csv(diagnostics_dir / "all_samples_summary.csv", index=False)
    if summary_rows.empty:
        raise ValueError(f"Summary rows are empty under {diagnostics_dir}")

    primary_rows = summary_rows.loc[summary_rows.get("sample_class", "posterior_predictive") == "posterior_predictive"].reset_index(drop=True)
    sensitivity_rows = summary_rows.loc[summary_rows.get("sample_class", "posterior_predictive") == "sensitivity_analysis"].reset_index(drop=True)
    headline_rows = primary_rows if len(primary_rows) else summary_rows
    posterior_mode = int(pd.to_numeric(summary_rows.get("posterior_num_runs", pd.Series([1] * len(summary_rows))), errors="coerce").fillna(1).max()) > 1

    dataset_name = str(manifest.get("dataset", "Dataset"))
    directed = bool(manifest.get("directed", False))
    weighted_label = "Weighted " if manifest.get("weight_model") else ""
    directed_label = "Directed " if directed else ""
    title = title or f"{dataset_name} {weighted_label}{directed_label}Temporal SBM Validation Report"
    output_path = Path(output_path) if output_path is not None else diagnostics_dir / "scientific_validation_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_edges_path = Path(manifest.get("filtered_input_edges_path", diagnostics_dir / "missing.csv"))
    input_edges = pd.read_csv(input_edges_path) if input_edges_path.exists() else pd.DataFrame()
    ts_min = int(input_edges["ts"].min()) if "ts" in input_edges.columns and len(input_edges) else None
    ts_max = int(input_edges["ts"].max()) if "ts" in input_edges.columns and len(input_edges) else None
    snapshot_count = int(input_edges["ts"].nunique()) if "ts" in input_edges.columns and len(input_edges) else 0

    frontier_path = _write_validation_frontier_plot(summary_rows, diagnostics_dir)
    metric_matrix_path = _write_validation_metric_matrix(summary_rows, diagnostics_dir)
    overview_path = write_all_samples_overview(summary_rows, diagnostics_dir)

    best_overlap = headline_rows.sort_values(
        ["mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    lowest_novelty = headline_rows.sort_values(
        ["mean_synthetic_novel_edge_rate", "mean_snapshot_edge_jaccard", "weight_total_correlation"],
        ascending=[True, False, False],
    ).iloc[0]
    best_sensitivity_overlap = None
    best_sensitivity_novelty = None
    if len(sensitivity_rows):
        best_sensitivity_overlap = sensitivity_rows.sort_values(
            ["mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        best_sensitivity_novelty = sensitivity_rows.sort_values(
            ["mean_synthetic_novel_edge_rate", "mean_snapshot_edge_jaccard", "weight_total_correlation"],
            ascending=[True, False, False],
        ).iloc[0]
    selected_labels = []
    for label in (str(best_overlap["sample_label"]), str(lowest_novelty["sample_label"])):
        if label not in selected_labels:
            selected_labels.append(label)
    selected_local_fit_path = _write_selected_setting_local_fit_plot(selected_labels, diagnostics_dir, diagnostics_dir)
    phase_assets = None
    if not skip_spectral_metrics:
        phase_assets = _write_hybrid_phase_assets(
            run_dir,
            diagnostics_dir,
            manifest,
            summary_rows["sample_label"].astype(str).tolist(),
        )
    network_assets = (
        _write_daily_network_snapshot_assets(
            run_dir,
            diagnostics_dir,
            manifest,
            summary_rows,
            selected_labels,
        )
        if include_daily_network_snapshots
        else {}
    )
    best_overlap_setting = _setting_display_payload(str(best_overlap["sample_label"]))
    lowest_novelty_setting = _setting_display_payload(str(lowest_novelty["sample_label"]))
    weight_model_display = _weight_model_display_payload(manifest.get("weight_model"), manifest)
    hybrid_node_frame = _load_hybrid_node_frame(run_dir, manifest)

    def setting_card_meta(setting_payload: dict[str, str]) -> str:
        return (
            f"<div class='setting-title'>{html.escape(setting_payload['sampler_name'])}</div>"
            f"<div class='setting-subtitle'>{html.escape(setting_payload['rewire_name'])}</div>"
        )

    def metric_card_meta(row: pd.Series, key: str) -> str:
        lower, upper = _posterior_interval_from_mapping(row, key)
        run_count = _posterior_run_count(row)
        if run_count > 1 and lower is not None and upper is not None:
            return f"<div class='setting-subtitle'>Posterior median across {run_count} draws [{lower:.3f}, {upper:.3f}]</div>"
        if run_count > 1:
            return f"<div class='setting-subtitle'>Posterior median across {run_count} draws</div>"
        return ""

    def setting_table_cell(sample_label: str) -> str:
        setting_payload = _setting_display_payload(sample_label)
        return (
            "<div class='setting-cell'>"
            f"<div class='setting-title'>{html.escape(setting_payload['sampler_name'])}</div>"
            f"<div class='setting-subtitle'>{html.escape(setting_payload['rewire_name'])}</div>"
            f"<code class='setting-code'>{html.escape(sample_label)}</code>"
            "</div>"
        )

    def weight_model_card_html() -> str:
        return (
            f"<div class='model-title'>{html.escape(weight_model_display['title'])}</div>"
            "<div class='badge-row'>"
            f"<span class='pill'>Column: {html.escape(weight_model_display['input_column'])}</span>"
            f"<span class='pill'>Transform: {html.escape(weight_model_display['transform'])}</span>"
            "</div>"
            f"<div class='setting-subtitle'>Source: {html.escape(weight_model_display['source_name'])}</div>"
        )

    runtime_summary = _summarise_log_runs(run_dir / "logs")
    runtime_profile_path = _write_runtime_profile_plot(run_dir / "logs", diagnostics_dir)
    failure_payload = None
    failure_path = diagnostics_dir / "novelty_grid_failures.json"
    if failure_path.exists():
        failure_payload = json.loads(failure_path.read_text())
    failure_count = len(failure_payload) if isinstance(failure_payload, (list, dict)) else 0

    constraint_note = "Structural constraint check summary was not available."
    novelty_report_path = diagnostics_dir / "novelty_grid_report.md"
    if novelty_report_path.exists():
        for line in novelty_report_path.read_text().splitlines():
            if line.lower().startswith("- structural constraint check:"):
                constraint_note = line.removeprefix("- ").strip()
                break

    top_settings = summary_rows.copy()
    top_setting_columns = [
        "sample_label",
        "sample_class",
        "sample_mode",
        "rewire_mode",
        "mean_snapshot_edge_jaccard",
        "mean_snapshot_node_jaccard",
        "mean_synthetic_novel_edge_rate",
        "edge_count_correlation",
        "weight_total_correlation",
        "reciprocity_correlation",
        "mean_abs_edge_count_delta",
        "mean_abs_weight_total_delta",
        "mean_abs_reciprocity_delta",
        "temporal_reachability_ratio_correlation",
        "temporal_efficiency_correlation",
        "temporal_forward_reach_node_correlation",
        "original_causal_fidelity",
        "synthetic_causal_fidelity",
        "tea_new_ratio_correlation",
        "edge_type_share_correlation",
        "tna_new_ratio_correlation",
        "pi_mass_mean_correlation",
        "pi_mass_closed_mean_correlation",
        "pi_mass_pagerank_mean_correlation",
        "magnetic_spectrum_mean_correlation",
        "magnetic_spectrum_mean_abs_delta",
        "magnetic_spectral_wasserstein_mean",
    ]
    top_settings = top_settings[[column for column in top_setting_columns if column in top_settings.columns]].sort_values(
        ["mean_snapshot_edge_jaccard", "weight_total_correlation", "mean_synthetic_novel_edge_rate"],
        ascending=[False, False, True],
    )

    selected_rows = pd.DataFrame([_selected_setting_detail_row(label, diagnostics_dir) for label in selected_labels])
    selected_display = selected_rows.rename(
        columns={
            "sample_label": "Setting",
            "posterior_num_runs": "Posterior draws",
            "mean_snapshot_edge_jaccard": "Edge Jaccard",
            "mean_snapshot_node_jaccard": "Node Jaccard",
            "mean_synthetic_novel_edge_rate": "Novel edge rate",
            "edge_count_correlation": "Edge-count corr",
            "reciprocity_correlation": "Reciprocity corr",
            "weight_total_correlation": "Weight-total corr",
            "mean_block_pair_weight_corr": "Mean block-pair weight corr",
            "min_block_pair_weight_corr": "Min block-pair weight corr",
            "mean_block_pair_edge_corr": "Mean block-pair edge corr",
            "median_top12_node_edge_corr": "Median top-node edge corr",
            "median_top12_node_out_edge_corr": "Median top-node out-edge corr",
            "median_top12_node_in_edge_corr": "Median top-node in-edge corr",
            "median_top12_node_weight_corr": "Median top-node weight corr",
            "median_top12_node_out_weight_corr": "Median top-node out-weight corr",
            "median_top12_node_in_weight_corr": "Median top-node in-weight corr",
            "mean_abs_edge_count_delta": "Mean abs edge delta",
            "mean_abs_reciprocity_delta": "Mean abs reciprocity delta",
            "mean_abs_weight_total_delta": "Mean abs weight delta",
            "temporal_reachability_ratio_correlation": "Temporal reach corr",
            "temporal_efficiency_correlation": "Temporal efficiency corr",
            "temporal_forward_reach_node_correlation": "Forward-reach node corr",
            "original_causal_fidelity": "Observed causal fidelity",
            "synthetic_causal_fidelity": "Synthetic causal fidelity",
            "tea_new_ratio_correlation": "TEA new corr",
            "tea_persist_ratio_correlation": "TEA persist corr",
            "tea_type_pair_birth_rate_correlation": "TEA type-pair corr",
            "tna_new_ratio_correlation": "TNA new corr",
            "tna_type_new_rate_correlation": "TNA type corr",
            "edge_type_share_correlation": "Hybrid edge-type corr",
            "edge_type_weight_share_correlation": "Hybrid edge-weight corr",
            "pi_mass_mean_correlation": "Pi-mass corr",
            "pi_mass_closed_mean_correlation": "Closed-class Pi corr",
            "pi_mass_pagerank_mean_correlation": "PageRank Pi corr",
            "pi_gini_correlation": "Pi-gini corr",
            "lic_share_active_correlation": "LIC share corr",
            "magnetic_spectrum_mean_correlation": "Mag spectrum corr",
            "magnetic_spectrum_mean_abs_delta": "Mag spectrum MAE",
            "magnetic_spectral_wasserstein_mean": "Mag W1",
        }
    )
    if "Setting" in selected_display.columns:
        selected_display["Setting"] = selected_rows["sample_label"].map(setting_table_cell)
    if posterior_mode:
        for source_column, display_column in [
            ("mean_snapshot_edge_jaccard", "Edge Jaccard"),
            ("mean_snapshot_node_jaccard", "Node Jaccard"),
            ("mean_synthetic_novel_edge_rate", "Novel edge rate"),
            ("edge_count_correlation", "Edge-count corr"),
            ("weight_total_correlation", "Weight-total corr"),
            ("reciprocity_correlation", "Reciprocity corr"),
            ("tea_new_ratio_correlation", "TEA new corr"),
            ("tna_new_ratio_correlation", "TNA new corr"),
            ("pi_mass_mean_correlation", "Pi-mass corr"),
            ("pi_mass_closed_mean_correlation", "Closed-class Pi corr"),
            ("pi_mass_pagerank_mean_correlation", "PageRank Pi corr"),
            ("magnetic_spectrum_mean_correlation", "Mag spectrum corr"),
            ("magnetic_spectral_wasserstein_mean", "Mag W1"),
        ]:
            if display_column in selected_display.columns and source_column in selected_rows.columns:
                selected_display[display_column] = selected_rows.apply(lambda row: _format_report_metric_value(row, source_column), axis=1)
    top_settings_display = pd.DataFrame(
        {
            "Setting": top_settings["sample_label"].map(setting_table_cell),
            "Class": top_settings["sample_class"].map(_display_sample_class),
            "Sampler code": top_settings["sample_mode"],
            "Ensemble": top_settings["sample_mode"].map(_sampler_ensemble_name),
            "Parameter regime": top_settings["sample_mode"].map(_sampler_regime_name),
            "Rewire code": top_settings["rewire_mode"],
            "Rewire step": top_settings["rewire_mode"].map(_rewire_step_name),
            "Edge Jaccard": top_settings.apply(lambda row: _format_report_metric_value(row, "mean_snapshot_edge_jaccard"), axis=1) if posterior_mode else top_settings["mean_snapshot_edge_jaccard"],
            "Node Jaccard": top_settings.apply(lambda row: _format_report_metric_value(row, "mean_snapshot_node_jaccard"), axis=1) if posterior_mode else top_settings["mean_snapshot_node_jaccard"],
            "Novel edge rate": top_settings.apply(lambda row: _format_report_metric_value(row, "mean_synthetic_novel_edge_rate"), axis=1) if posterior_mode else top_settings["mean_synthetic_novel_edge_rate"],
            "Temporal reachability corr": top_settings.apply(lambda row: _format_report_metric_value(row, "temporal_reachability_ratio_correlation"), axis=1) if posterior_mode and "temporal_reachability_ratio_correlation" in top_settings.columns else top_settings.get("temporal_reachability_ratio_correlation", pd.Series(np.nan, index=top_settings.index)),
            "Temporal efficiency corr": top_settings.apply(lambda row: _format_report_metric_value(row, "temporal_efficiency_correlation"), axis=1) if posterior_mode and "temporal_efficiency_correlation" in top_settings.columns else top_settings.get("temporal_efficiency_correlation", pd.Series(np.nan, index=top_settings.index)),
            "Forward-reach node corr": top_settings.apply(lambda row: _format_report_metric_value(row, "temporal_forward_reach_node_correlation"), axis=1) if posterior_mode and "temporal_forward_reach_node_correlation" in top_settings.columns else top_settings.get("temporal_forward_reach_node_correlation", pd.Series(np.nan, index=top_settings.index)),
            "Edge-count corr (median path)": top_settings.apply(lambda row: _format_report_metric_value(row, "edge_count_correlation"), axis=1) if posterior_mode else top_settings["edge_count_correlation"],
        }
    )
    if posterior_mode and "posterior_num_runs" in top_settings.columns:
        top_settings_display.insert(2, "Posterior draws", top_settings["posterior_num_runs"].astype(int))
    if "weight_total_correlation" in top_settings.columns:
        top_settings_display["Weight-total corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "weight_total_correlation"), axis=1) if posterior_mode else top_settings["weight_total_correlation"]
    if "reciprocity_correlation" in top_settings.columns:
        top_settings_display["Reciprocity corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "reciprocity_correlation"), axis=1) if posterior_mode else top_settings["reciprocity_correlation"]
    if posterior_mode and "edge_count_pooled_correlation" in top_settings.columns:
        top_settings_display["Edge-count corr (all runs)"] = top_settings["edge_count_pooled_correlation"]
    if posterior_mode and "temporal_reachability_ratio_pooled_correlation" in top_settings.columns:
        top_settings_display["Temporal reachability corr (all runs)"] = top_settings["temporal_reachability_ratio_pooled_correlation"]
    if posterior_mode and "temporal_efficiency_pooled_correlation" in top_settings.columns:
        top_settings_display["Temporal efficiency corr (all runs)"] = top_settings["temporal_efficiency_pooled_correlation"]
    if posterior_mode and "weight_total_pooled_correlation" in top_settings.columns:
        top_settings_display["Weight-total corr (all runs)"] = top_settings["weight_total_pooled_correlation"]
    if posterior_mode and "reciprocity_pooled_correlation" in top_settings.columns:
        top_settings_display["Reciprocity corr (all runs)"] = top_settings["reciprocity_pooled_correlation"]
    if "mean_abs_edge_count_delta" in top_settings.columns:
        top_settings_display["Mean abs edge delta"] = top_settings.apply(lambda row: _format_report_metric_value(row, "mean_abs_edge_count_delta"), axis=1) if posterior_mode else top_settings["mean_abs_edge_count_delta"]
    if "mean_abs_weight_total_delta" in top_settings.columns:
        top_settings_display["Mean abs weight delta"] = top_settings.apply(lambda row: _format_report_metric_value(row, "mean_abs_weight_total_delta"), axis=1) if posterior_mode else top_settings["mean_abs_weight_total_delta"]
    if "mean_abs_reciprocity_delta" in top_settings.columns:
        top_settings_display["Mean abs reciprocity delta"] = top_settings.apply(lambda row: _format_report_metric_value(row, "mean_abs_reciprocity_delta"), axis=1) if posterior_mode else top_settings["mean_abs_reciprocity_delta"]
    if "tea_new_ratio_correlation" in top_settings.columns:
        top_settings_display["TEA new corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "tea_new_ratio_correlation"), axis=1) if posterior_mode else top_settings["tea_new_ratio_correlation"]
    if posterior_mode and "tea_new_ratio_pooled_correlation" in top_settings.columns:
        top_settings_display["TEA new corr (all runs)"] = top_settings["tea_new_ratio_pooled_correlation"]
    if "edge_type_share_correlation" in top_settings.columns:
        top_settings_display["Hybrid edge-type corr"] = top_settings.apply(lambda row: _format_report_metric_value(row, "edge_type_share_correlation"), axis=1) if posterior_mode else top_settings["edge_type_share_correlation"]
    if "tna_new_ratio_correlation" in top_settings.columns:
        top_settings_display["TNA new corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "tna_new_ratio_correlation"), axis=1) if posterior_mode else top_settings["tna_new_ratio_correlation"]
    if posterior_mode and "tna_new_ratio_pooled_correlation" in top_settings.columns:
        top_settings_display["TNA new corr (all runs)"] = top_settings["tna_new_ratio_pooled_correlation"]
    if "pi_mass_mean_correlation" in top_settings.columns:
        top_settings_display["Pi-mass corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "pi_mass_mean_correlation"), axis=1) if posterior_mode else top_settings["pi_mass_mean_correlation"]
    if posterior_mode and "pi_mass_mean_pooled_correlation" in top_settings.columns:
        top_settings_display["Pi-mass corr (all runs)"] = top_settings["pi_mass_mean_pooled_correlation"]
    if "magnetic_spectrum_mean_correlation" in top_settings.columns:
        top_settings_display["Mag spectrum corr (median path)"] = top_settings.apply(lambda row: _format_report_metric_value(row, "magnetic_spectrum_mean_correlation"), axis=1) if posterior_mode else top_settings["magnetic_spectrum_mean_correlation"]
    if posterior_mode and "magnetic_spectrum_pooled_correlation" in top_settings.columns:
        top_settings_display["Mag spectrum corr (all runs)"] = top_settings["magnetic_spectrum_pooled_correlation"]
    if "magnetic_spectrum_mean_abs_delta" in top_settings.columns:
        top_settings_display["Mag spectrum MAE"] = top_settings.apply(lambda row: _format_report_metric_value(row, "magnetic_spectrum_mean_abs_delta"), axis=1) if posterior_mode else top_settings["magnetic_spectrum_mean_abs_delta"]

    best_setting_summary = _load_json_if_exists(diagnostics_dir / f"{str(best_overlap['sample_label'])}_summary.json") or {}
    best_setting_run_labels = [str(label) for label in best_setting_summary.get("posterior_run_labels", []) or []]
    hybrid_overview_table = _build_hybrid_network_summary_table(
        run_dir=run_dir,
        manifest=manifest,
        input_edges=input_edges,
        hybrid_node_frame=hybrid_node_frame,
        directed=directed,
        best_setting_label=str(best_overlap["sample_label"]),
        best_setting_run_labels=best_setting_run_labels,
    )

    methodology_table = pd.DataFrame(
        [
            ("Dataset", dataset_name),
            ("Slice", f"{ts_min} to {ts_max}" if ts_min is not None and ts_max is not None else "unknown"),
            ("Snapshots", snapshot_count),
            ("Nodes", manifest.get("input_summary", {}).get("node_count", "unknown")),
            ("Observed edges", manifest.get("input_summary", {}).get("edge_count", "unknown")),
            ("Duplicate rows collapsed", manifest.get("input_summary", {}).get("duplicate_edge_count", "unknown")),
            ("Network type", "Directed simple graph" if directed else "Undirected simple graph"),
            ("Weight source", Path(str(manifest.get("weight_npy", "embedded CSV"))).name if manifest.get("weight_npy") else "embedded CSV"),
            ("Weight model", f"{weight_model_display['title']} ({weight_model_display['transform']})"),
            ("Fit covariates", ", ".join(_format_covariate_name(name) for name in manifest.get("fit_covariates", []))),
            ("Hybrid network", "Focal farms plus contracted regional supernodes"),
            ("Sample classes", f"{len(primary_rows)} primary posterior-predictive, {len(sensitivity_rows)} sensitivity-analysis settings"),
            ("Posterior reporting", "Posterior medians with 5th-95th percentile intervals by setting" if posterior_mode else "Single generated panel per setting"),
            ("Time-series association", f"{TIME_SERIES_CORRELATION_LABEL} on aligned snapshot sequences"),
            ("Advanced diagnostics", "TEA, TNA, strict temporal reachability / transmission-potential diagnostics, hybrid edge-type composition, weighted Pi-Mass on the largest SCC, and weighted magnetic Laplacian eigenvalue / phase diagnostics"),
        ],
        columns=["Parameter", "Value"],
    )
    metric_rows = [
        ("Edge Jaccard", f"Per-snapshot overlap of realized {'directed' if directed else 'undirected'} edges; higher is better."),
        ("Node Jaccard", "Per-snapshot overlap of active nodes; higher is better."),
        ("Novel edge rate", "Share of synthetic edges absent from the observed panel; lower is better."),
        ("Temporal reachability corr", f"{TIME_SERIES_CORRELATION_LABEL} between observed and synthetic prefix reachability-ratio series under strict time-respecting paths on the discrete-time panel."),
        ("Temporal efficiency corr", f"{TIME_SERIES_CORRELATION_LABEL} between observed and synthetic prefix temporal-efficiency series, where efficiency is the average inverse earliest-arrival time and unreachable ordered pairs contribute zero."),
        ("Forward-reach node corr", "Spearman correlation between observed and synthetic source-level forward reachable fractions over the full observation window."),
        ("Causal fidelity", "Temporal reachability ratio divided by static aggregated reachability ratio. Values near one mean the aggregated network is not overstating causal transmission opportunity."),
        ("Edge-count corr", f"{TIME_SERIES_CORRELATION_LABEL} between observed and synthetic snapshot edge-count series. In posterior mode this is the observed-versus-posterior-median trajectory correlation."),
    ]
    if manifest.get("weight_model"):
        metric_rows.append(("Weight-total corr", f"{TIME_SERIES_CORRELATION_LABEL} between observed and synthetic snapshot total-weight series. In posterior mode this is the observed-versus-posterior-median trajectory correlation."))
    if directed:
        metric_rows.extend(
            [
                ("Reciprocity corr", f"{TIME_SERIES_CORRELATION_LABEL} between observed and synthetic reciprocity series."),
                ("Sender / receiver deltas", "Absolute differences in the number of active source and target nodes per snapshot; lower is better."),
                ("Top-node out/in corr", f"Median {TIME_SERIES_CORRELATION_LABEL.lower()} on outgoing and incoming activity for the most active nodes."),
                ("TEA new corr", f"{TIME_SERIES_CORRELATION_LABEL} of temporal edge-appearance novelty ratios between observed and synthetic panels."),
                ("Hybrid edge-type corr", f"Mean {TIME_SERIES_CORRELATION_LABEL.lower()} of edge-type shares through time across F→F, F→R, R→F, and R→R channels."),
                ("TNA new corr", f"{TIME_SERIES_CORRELATION_LABEL} of temporal node-appearance novelty ratios between observed and synthetic panels."),
                ("Pi-mass corr", f"Mean {TIME_SERIES_CORRELATION_LABEL.lower()} of weighted stationary-distribution mass assigned to node-type groups on the largest strongly connected component."),
                ("Mag spectrum corr", f"Mean {TIME_SERIES_CORRELATION_LABEL.lower()} of the smallest weighted magnetic Laplacian eigenvalue tracks through time."),
                ("Mag spectrum MAE", "Mean absolute error of the smallest magnetic Laplacian eigenvalues through time; lower is better."),
            ]
        )
    if posterior_mode:
        metric_rows.append(("All-runs corr", f"Additional {TIME_SERIES_CORRELATION_LABEL.lower()} computed after pooling all run-by-snapshot pairs for a setting. This is shown where repeated draws are available."))
    metric_definitions = pd.DataFrame(metric_rows, columns=["Metric", "Interpretation"])
    if len(runtime_summary):
        runtime_display = runtime_summary.rename(
            columns={
                "command": "Stage",
                "run_count": "Runs",
                "median_elapsed_seconds": "Median seconds",
                "max_elapsed_seconds": "Max seconds",
                "mean_line_count": "Mean log lines",
                "total_warning_lines": "Warnings",
                "total_error_lines": "Errors",
            }
        )
        runtime_display["Stage"] = runtime_display["Stage"].map(lambda value: str(value).title())
    else:
        runtime_display = pd.DataFrame()

    feasibility_points = [
        f"The weighted temporal SBM is operationally feasible on this {snapshot_count}-snapshot slice of {dataset_name}: all {len(summary_rows)} tested settings completed, with {failure_count} recorded sweep failures.",
        f"The primary posterior-predictive comparison covers {len(primary_rows)} settings without rewiring; {len(sensitivity_rows)} additional rewired settings are treated as sensitivity analyses rather than as draws from the fitted generative model.",
        (
            f"Panel-level fit is summarized as posterior medians and 5th-95th percentile intervals across repeated draws for each setting. "
            f"The best-overlap primary setting reached median edge Jaccard {float(best_overlap['mean_snapshot_edge_jaccard']):.3f}; "
            f"the lowest-novelty primary setting reached median novel-edge rate {float(lowest_novelty['mean_synthetic_novel_edge_rate']):.3f}."
            if posterior_mode
            else f"Panel-level fit is strongest on snapshot volumes and total weight trends within the primary posterior-predictive class. The best-overlap primary setting reached mean edge Jaccard {float(best_overlap['mean_snapshot_edge_jaccard']):.3f}; the lowest-novelty primary setting reached novel-edge rate {float(lowest_novelty['mean_synthetic_novel_edge_rate']):.3f}."
        ),
        "The added TEA and TNA diagnostics test whether the model reproduces when edges and active nodes appear for the first time, persist, reactivate, or churn between consecutive snapshots.",
        "The temporal reachability diagnostics track how many ordered node pairs become causally connected over the window, how quickly they do so, and whether the same source nodes can reach similarly large fractions of the network.",
        "Because this is a hybrid network, the report also checks whether activity is allocated correctly across the four edge channels F→F, F→R, R→F, and R→R rather than only matching totals after aggregation.",
        "The Pi-Mass and magnetic Laplacian diagnostics test higher-order directed structure using weighted flows: stationary-flow concentration on the largest strongly connected component, and orientation-sensitive weighted spectral structure over time.",
        "Exact link recovery is limited. Even the strongest settings generate many novel edges, so this workflow is more reliable for aggregate activity and weight-allocation patterns than for reproducing specific observed links.",
        "The fitted state on this slice has only two top-level blocks, so block-pair diagnostics are informative but coarse. Node-level weight traces are necessary to detect misallocation among the most active entities.",
    ]
    if directed:
        feasibility_points.insert(
            2,
            "Directed-network feasibility should be judged on more than overlap alone. Reciprocity, active sender coverage, active receiver coverage, and node-level in/out flow traces are required to detect directional misallocation.",
        )

    selected_sections = []
    for label in selected_labels:
        summary_json = _load_json_if_exists(diagnostics_dir / f"{label}_summary.json") or {}
        setting_payload = _setting_display_payload(label)
        section_figures: list[dict[str, object]] = []
        posterior_run_labels = [str(run_label) for run_label in summary_json.get("posterior_run_labels", []) or []]
        switchable_suffixes = {"magnetic_laplacian.png", "magnetic_laplacian_diff.png"}
        figure_suffixes = [
            "dashboard.png",
            "tea.png",
            "tna.png",
            "edge_type.png",
            "temporal_reachability.png",
            "pi_mass.png",
            "pi_mass_closed.png",
            "pi_mass_pagerank.png",
            "magnetic_laplacian.png",
            "magnetic_laplacian_diff.png",
            "magnetic_spectral_distance.png",
            "block_pair_weight_total.png",
        ]
        if directed:
            figure_suffixes.extend(
                [
                    "node_activity_out_edge_count.png",
                    "node_activity_in_edge_count.png",
                    "node_activity_out_weight_total.png",
                    "node_activity_in_weight_total.png",
                ]
            )
        else:
            figure_suffixes.extend(
                [
                    "node_activity_incident_edge_count.png",
                    "node_activity_incident_weight_total.png",
                ]
            )
        for suffix in figure_suffixes:
            figure_path = diagnostics_dir / f"{label}_{suffix}"
            if figure_path.exists():
                figure_spec: dict[str, object] = {"path": figure_path, "suffix": suffix}
                if suffix in switchable_suffixes and posterior_run_labels:
                    run_figures: list[dict[str, str]] = []
                    for run_label in posterior_run_labels:
                        run_path = diagnostics_dir / f"{run_label}_{suffix}"
                        if run_path.exists():
                            run_figures.append(
                                {
                                    "label": _posterior_run_display_label(run_label),
                                    "path": run_path.name,
                                }
                            )
                    if run_figures:
                        figure_spec["switcher_options"] = [
                            {"label": "Posterior summary", "path": figure_path.name},
                            *run_figures,
                        ]
                section_figures.append(figure_spec)
        narrative = []
        if label == str(best_overlap["sample_label"]):
            narrative.append(
                "This configuration is the strongest structural-overlap setting within the primary posterior-predictive class. It is therefore the main reference point for judging the fitted model rather than the rewiring sensitivity runs."
            )
        if label == str(lowest_novelty["sample_label"]):
            narrative.append(
                "This configuration has the lowest novel-edge rate within the primary posterior-predictive class. It is the clearest reference when the goal is to reduce synthetic edge invention without post-sampling rewiring."
            )
        if summary_json.get("mean_snapshot_node_jaccard", 1.0) < 0.9:
            narrative.append(
                f"It does not preserve the active-node set exactly through time (mean node Jaccard {float(summary_json['mean_snapshot_node_jaccard']):.3f}), so node-level plots should be interpreted as both activity and coverage diagnostics."
            )
        if directed and "reciprocity_correlation" in summary_json:
            narrative.append(
                f"Its directional fit is summarized by reciprocity correlation {float(summary_json['reciprocity_correlation']):.3f} and mean absolute reciprocity delta {float(summary_json.get('mean_abs_reciprocity_delta', 0.0)):.3f}."
            )
        if "tea_new_ratio_correlation" in summary_json:
            narrative.append(
                f"The edge-appearance evidence shows TEA new-ratio correlation {float(summary_json.get('tea_new_ratio_correlation', 0.0)):.3f} and TNA new-ratio correlation {float(summary_json.get('tna_new_ratio_correlation', 0.0)):.3f}."
            )
        if "edge_type_share_correlation" in summary_json:
            narrative.append(
                f"The hybrid channel allocation is summarized by mean edge-type share correlation {float(summary_json.get('edge_type_share_correlation', 0.0)):.3f}"
                + (
                    f" and weight-share correlation {float(summary_json.get('edge_type_weight_share_correlation', 0.0)):.3f}."
                    if summary_json.get("edge_type_weight_share_correlation") is not None
                    else "."
                )
            )
        if "pi_mass_mean_correlation" in summary_json:
            higher_order_bits = [
                f"weighted Pi-mass correlation {float(summary_json.get('pi_mass_mean_correlation', 0.0)):.3f}"
            ]
            if summary_json.get("pi_mass_closed_mean_correlation") is not None:
                higher_order_bits.append(
                    f"closed-class Pi correlation {float(summary_json.get('pi_mass_closed_mean_correlation', 0.0)):.3f}"
                )
            if summary_json.get("pi_mass_pagerank_mean_correlation") is not None:
                higher_order_bits.append(
                    f"PageRank Pi correlation {float(summary_json.get('pi_mass_pagerank_mean_correlation', 0.0)):.3f}"
                )
            if summary_json.get("magnetic_spectrum_mean_correlation") is not None:
                magnetic_text = f"weighted magnetic spectrum correlation {float(summary_json.get('magnetic_spectrum_mean_correlation', 0.0)):.3f}"
                if summary_json.get("magnetic_spectral_wasserstein_mean") is not None:
                    magnetic_text += f" and mean spectral Wasserstein distance {float(summary_json.get('magnetic_spectral_wasserstein_mean', 0.0)):.3f}"
                higher_order_bits.append(magnetic_text)
            narrative.append("Higher-order directed structure is summarized by " + ", ".join(higher_order_bits) + ".")
        selected_sections.append(
            {
                "label": label,
                "display": setting_payload,
                "narrative": " ".join(narrative),
                "figures": section_figures,
                "tables": _selected_setting_evidence_tables(label, diagnostics_dir, directed=directed),
            }
        )

    selected_lookup = {
        str(row["sample_label"]): row
        for row in selected_rows.to_dict(orient="records")
        if "sample_label" in row
    }
    best_detail = selected_lookup.get(str(best_overlap["sample_label"]), {})
    lowest_detail = selected_lookup.get(str(lowest_novelty["sample_label"]), {})
    best_overlap_class = str(best_overlap.get("sample_class", _default_sample_class(str(best_overlap["sample_label"]))))
    lowest_novelty_class = str(lowest_novelty.get("sample_class", _default_sample_class(str(lowest_novelty["sample_label"]))))

    def metric_median(column: str) -> float:
        comparison_frame = headline_rows if len(headline_rows) else summary_rows
        if column not in comparison_frame.columns:
            return 0.0
        series = pd.to_numeric(comparison_frame[column], errors="coerce").dropna()
        return float(series.median()) if len(series) else 0.0

    def float_or_none(payload: dict[str, object], key: str) -> Optional[float]:
        value = payload.get(key)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def figure_explanation(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        name = Path(path).name
        if name == "runtime_profile.png":
            return "The left panel shows individual fit, generate, and report runtimes; the right panel shows median runtime by stage. Use it to spot unstable or unusually slow stages."
        if name == "validation_frontier.png":
            return "Each point is one sampler and rewiring setting. Better settings are higher and farther left: more overlap and less novelty. Larger bubbles mean stronger snapshot weight-total correlation. Rewired points should be read as sensitivity analyses, not as primary posterior-predictive draws."
        if name == "validation_metric_matrix.png":
            return "Rows are the leading settings and columns are the main metrics. Darker cells are better after accounting for which metrics should be high versus low."
        if name == "all_samples_overview.png":
            return "Each panel ranks every tested setting on one metric. The final panel carries the in-figure setting labels in a left-side gutter so the lollipops can be compared without relying on axis tick text. Compare positions across panels to see whether a setting is consistently strong or only wins on one axis."
        if name == "selected_settings_local_fit.png":
            return "This combines local-fit evidence for the selected settings. Bar charts summarize block-pair fidelity, and boxplots summarize temporal correlations for the most active nodes."
        if name.endswith("_tea.png"):
            return "TEA tracks how edges persist, reactivate, appear for the first time, and disappear between consecutive snapshots. Matching stacks and correlated ratio traces indicate that the model reproduces temporal edge turnover, not just total edge counts. Blue background bands mark weekends and red bands mark Dutch public holidays."
        if name.endswith("_tna.png"):
            return "TNA tracks when nodes first become active, remain active, reactivate, or churn out. Use it to see whether the synthetic panel activates the right volume of nodes at the right times. Blue background bands mark weekends and red bands mark Dutch public holidays."
        if name.endswith("_edge_type.png"):
            return "This figure checks the hybrid-network channel allocation directly. Each color is one of the F→F, F→R, R→F, or R→R edge classes, and close observed versus synthetic trajectories mean the model is placing traffic in the right hybrid channels through time. Blue background bands mark weekends and red bands mark Dutch public holidays."
        if name.endswith("_temporal_reachability.png"):
            return "This figure measures transmission potential using strict time-respecting paths on the discrete-time panel. Reachability ratio shows how many ordered node pairs are causally connected, temporal efficiency summarizes how quickly those connections appear, newly reachable pairs show when the transmission envelope expands, and the source-level parity plot checks whether the same nodes can reach similarly large fractions of the network."
        if name.endswith("_pi_mass.png"):
            return "Pi-Mass summarizes how weighted stationary flow concentrates on node-type groups within the largest strongly connected component under a weighted lazy random walk. The LIC panels show whether directed flow is concentrated on the same part of the network as in the observed panel, and the active-node panels show whether the same number of farms and regional supernodes are engaged each day. Cross marks denote degenerate snapshots where Pi-Mass is undefined, and the traces break across those days. Blue background bands mark weekends and red bands mark Dutch public holidays."
        if name.endswith("_pi_mass_closed.png"):
            return "This variant recomputes Pi-Mass on the largest closed strongly connected class under the weighted lazy walk. Use it to see whether the model preserves stationary flow once leakage to other classes is excluded, while matching daily active-node counts inside the directed core. Cross marks denote degenerate snapshots where the closed-class Pi-Mass is undefined."
        if name.endswith("_pi_mass_pagerank.png"):
            return "This variant summarizes whole-snapshot teleporting PageRank by node type. It complements the closed-class lazy walk by providing a globally ergodic directed-flow summary over the active snapshot. Cross marks denote degenerate snapshots where the type-mass summary is undefined."
        if name.endswith("_magnetic_laplacian.png"):
            return "The magnetic Laplacian is a weighted, direction-sensitive spectrum. Similar observed and synthetic heatmaps, together with low per-snapshot spectral error, indicate that the model preserves cyclic and directional structure beyond edge totals."
        if name.endswith("_magnetic_laplacian_diff.png"):
            return "This difference view subtracts the observed spectrum from the synthetic one. Warm and cool bands show where directional spectral energy is being shifted across time or mode index."
        if name.endswith("_magnetic_spectral_distance.png"):
            return "This figure reports quantitative per-snapshot spectral distances for the magnetic Laplacian. Lower Wasserstein distance and lower aligned-mode errors indicate closer agreement between the observed and synthetic directed spectral fingerprints."
        if name.endswith("_dashboard.png"):
            return "This is the snapshot-by-snapshot fit diagnostic for one setting. Closer observed and synthetic traces, together with smaller deltas, indicate better temporal fit. Blue background bands mark weekends and red bands mark Dutch public holidays."
        if name.endswith("_block_pair_weight_total.png"):
            return "Each subplot tracks one block pair through time. Matching curves mean the model places weight in the right origin-destination block channel at the right times."
        if name.endswith("_block_pair_edge_count.png"):
            return "Each subplot tracks one block pair through time using edge counts. Matching traces mean the model reproduces when traffic occurs between those blocks."
        if name.endswith("_node_activity_out_edge_count.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves when that node sends edges through time."
        if name.endswith("_node_activity_in_edge_count.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves when that node receives edges through time."
        if name.endswith("_node_activity_out_weight_total.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves that node's outgoing traffic intensity through time."
        if name.endswith("_node_activity_in_weight_total.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves that node's incoming traffic intensity through time."
        if name.endswith("_node_activity_incident_edge_count.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves when that node is active through time."
        if name.endswith("_node_activity_incident_weight_total.png"):
            return "Each subplot is a high-activity node. Matching traces indicate the model preserves that node's total traffic intensity through time."
        if name.endswith("_block_activity_out_edge_count.png"):
            return "Each subplot is one block. Matching traces indicate the model preserves outgoing block activity over time."
        if name.endswith("_block_activity_in_edge_count.png"):
            return "Each subplot is one block. Matching traces indicate the model preserves incoming block activity over time."
        if name.endswith("_block_activity_out_weight_total.png"):
            return "Each subplot is one block. Matching traces indicate the model preserves outgoing block-level weight over time."
        if name.endswith("_block_activity_in_weight_total.png"):
            return "Each subplot is one block. Matching traces indicate the model preserves incoming block-level weight over time."
        return "Use this figure as a local goodness-of-fit diagnostic. Closer alignment between observed and synthetic structure means the setting is reproducing the relevant pattern more faithfully."

    table_help = {
        "methodology_table": "This table records the exact data slice, network type, weight source, and fitted covariates. Use it to confirm that the report matches the intended experiment.",
        "metric_definitions_table": "These definitions are the legend for the report. Higher is better for overlap and correlations; lower is better for novelty and absolute-delta metrics. The added TEA, TNA, Pi-Mass, and magnetic-spectrum metrics probe temporal turnover and higher-order directed structure.",
        "runtime_table": "This table aggregates the run logs by stage. Large time spreads or repeated warnings and errors are signs of execution instability.",
        "top_settings_table": "This is the compact comparison table for the sweep settings. The Class column separates primary posterior-predictive draws from rewiring sensitivity analyses. The sampler code, ensemble, and parameter-regime columns unpack how the topology was generated, while the rewiring code and rewiring-step columns show whether a post hoc edge-swap sensitivity step was applied.",
        "selected_settings_table": "This table condenses the selected primary posterior-predictive settings into one comparison: global panel metrics, local block-pair and node-level fit summaries, hybrid channel allocation, and the temporal / spectral goodness-of-fit evidence.",
    }

    worst_reciprocity_row = None
    reciprocity_frame = headline_rows if len(headline_rows) else summary_rows
    if directed and "reciprocity_correlation" in reciprocity_frame.columns and len(reciprocity_frame):
        worst_reciprocity_row = reciprocity_frame.sort_values(
            ["reciprocity_correlation", "mean_snapshot_edge_jaccard", "mean_synthetic_novel_edge_rate"],
            ascending=[True, False, True],
        ).iloc[0]

    conclusion_paragraphs: list[str] = []
    if str(best_overlap["sample_label"]) == str(lowest_novelty["sample_label"]):
        lead_sentence = (
            f"Within the primary posterior-predictive comparison, <strong>{html.escape(best_overlap_setting['short_label'])}</strong> is the leading setting on this slice. "
            f"It reaches mean edge Jaccard {float(best_overlap['mean_snapshot_edge_jaccard']):.3f} versus the median across the primary settings {metric_median('mean_snapshot_edge_jaccard'):.3f}, "
            f"while also reducing novel-edge rate to {float(best_overlap['mean_synthetic_novel_edge_rate']):.3f} versus the primary-setting median {metric_median('mean_synthetic_novel_edge_rate'):.3f}."
        )
    else:
        lead_sentence = (
            f"Within the primary posterior-predictive comparison, the frontier separates two operating points. <strong>{html.escape(best_overlap_setting['short_label'])}</strong> gives the highest structural overlap "
            f"(edge Jaccard {float(best_overlap['mean_snapshot_edge_jaccard']):.3f}), while <strong>{html.escape(lowest_novelty_setting['short_label'])}</strong> gives the lowest synthetic novelty "
            f"({float(lowest_novelty['mean_synthetic_novel_edge_rate']):.3f})."
        )
    if manifest.get("weight_model"):
        lead_sentence += f" The leading overlap setting keeps snapshot weight-total correlation at {float(best_overlap.get('weight_total_correlation', 0.0) or 0.0):.3f}"
        if directed and "reciprocity_correlation" in summary_rows.columns:
            lead_sentence += f" and reciprocity correlation at {float(best_overlap.get('reciprocity_correlation', 0.0) or 0.0):.3f}"
        lead_sentence += "."
    conclusion_paragraphs.append(lead_sentence)

    if best_sensitivity_overlap is not None and best_sensitivity_novelty is not None:
        best_sensitivity_overlap_payload = _setting_display_payload(str(best_sensitivity_overlap["sample_label"]))
        best_sensitivity_novelty_payload = _setting_display_payload(str(best_sensitivity_novelty["sample_label"]))
        if str(best_sensitivity_overlap["sample_label"]) == str(best_sensitivity_novelty["sample_label"]):
            sensitivity_sentence = (
                f"Rewiring-based sensitivity analysis can push the overlap-novelty frontier further: <strong>{html.escape(best_sensitivity_overlap_payload['short_label'])}</strong> reaches edge Jaccard "
                f"{float(best_sensitivity_overlap['mean_snapshot_edge_jaccard']):.3f} with novel-edge rate {float(best_sensitivity_overlap['mean_synthetic_novel_edge_rate']):.3f}. "
                "These runs are reported as robustness checks on secondary topological constraints, not as primary posterior-predictive evidence."
            )
        else:
            sensitivity_sentence = (
                f"Rewiring-based sensitivity analysis can alter the frontier further: <strong>{html.escape(best_sensitivity_overlap_payload['short_label'])}</strong> gives the strongest sensitivity overlap "
                f"(edge Jaccard {float(best_sensitivity_overlap['mean_snapshot_edge_jaccard']):.3f}), while <strong>{html.escape(best_sensitivity_novelty_payload['short_label'])}</strong> gives the lowest sensitivity novelty "
                f"({float(best_sensitivity_novelty['mean_synthetic_novel_edge_rate']):.3f}). These runs are robustness checks on post hoc topological constraints rather than primary posterior-predictive draws."
            )
        conclusion_paragraphs.append(sensitivity_sentence)

    local_fit_parts = []
    best_out_edge = float_or_none(best_detail, "median_top12_node_out_edge_corr")
    best_in_edge = float_or_none(best_detail, "median_top12_node_in_edge_corr")
    best_out_weight = float_or_none(best_detail, "median_top12_node_out_weight_corr")
    best_in_weight = float_or_none(best_detail, "median_top12_node_in_weight_corr")
    best_block_weight = float_or_none(best_detail, "mean_block_pair_weight_corr")
    best_block_weight_min = float_or_none(best_detail, "min_block_pair_weight_corr")
    best_source_delta = float_or_none(best_detail, "mean_abs_source_node_delta")
    best_target_delta = float_or_none(best_detail, "mean_abs_target_node_delta")
    best_recip_delta = float_or_none(best_detail, "mean_abs_reciprocity_delta")
    best_tea_new = float_or_none(best_detail, "tea_new_ratio_correlation")
    best_tna_new = float_or_none(best_detail, "tna_new_ratio_correlation")
    best_edge_type = float_or_none(best_detail, "edge_type_share_correlation")
    best_edge_type_weight = float_or_none(best_detail, "edge_type_weight_share_correlation")
    best_pi_mass = float_or_none(best_detail, "pi_mass_mean_correlation")
    best_pi_closed = float_or_none(best_detail, "pi_mass_closed_mean_correlation")
    best_pi_pagerank = float_or_none(best_detail, "pi_mass_pagerank_mean_correlation")
    best_pi_gini = float_or_none(best_detail, "pi_gini_correlation")
    best_lic_share = float_or_none(best_detail, "lic_share_active_correlation")
    best_mag_corr = float_or_none(best_detail, "magnetic_spectrum_mean_correlation")
    best_mag_mae = float_or_none(best_detail, "magnetic_spectrum_mean_abs_delta")
    best_mag_wasserstein = float_or_none(best_detail, "magnetic_spectral_wasserstein_mean")

    if directed and best_out_edge is not None and best_in_edge is not None:
        local_fit_parts.append(
            f"The selected-setting local-fit summary and top-node evidence show that directional timing is reproduced very closely for the leading setting: median top-node outgoing and incoming edge correlations are {best_out_edge:.3f} and {best_in_edge:.3f}"
            + (
                f", with mean absolute sender and receiver deltas of {best_source_delta:.1f} and {best_target_delta:.1f} active nodes per snapshot"
                if best_source_delta is not None and best_target_delta is not None
                else ""
            )
            + (
                f", and mean absolute reciprocity delta {best_recip_delta:.3f}"
                if best_recip_delta is not None
                else ""
            )
            + "."
        )
    if best_block_weight is not None and best_out_weight is not None and best_in_weight is not None:
        local_fit_parts.append(
            f"Weight allocation is weaker than topology at finer scale: mean block-pair weight correlation is {best_block_weight:.3f}"
            + (f" (minimum {best_block_weight_min:.3f})" if best_block_weight_min is not None else "")
            + f", while median top-node outgoing and incoming weight correlations are {best_out_weight:.3f} and {best_in_weight:.3f}."
        )
    if local_fit_parts:
        conclusion_paragraphs.append(" ".join(local_fit_parts))

    higher_order_parts = []
    if best_tea_new is not None and best_tna_new is not None:
        higher_order_parts.append(
            f"The TEA and TNA figures support the same ranking: the leading primary setting reaches TEA new-ratio correlation {best_tea_new:.3f} and TNA new-ratio correlation {best_tna_new:.3f}, indicating that consecutive-snapshot edge and node turnover is being reproduced rather than only the aggregate totals."
        )
    if best_edge_type is not None:
        higher_order_parts.append(
            f"The hybrid channel evidence is also supportive: mean F→F / F→R / R→F / R→R edge-share correlation is {best_edge_type:.3f}"
            + (
                f", and weight-share correlation is {best_edge_type_weight:.3f}."
                if best_edge_type_weight is not None
                else "."
            )
        )
    if best_pi_mass is not None and best_mag_corr is not None:
        higher_order_score = min(
            value
            for value in (
                best_pi_mass,
                best_pi_closed if best_pi_closed is not None else best_pi_mass,
                best_pi_pagerank if best_pi_pagerank is not None else best_pi_mass,
                best_mag_corr,
            )
        )
        if higher_order_score >= 0.85:
            prefix = "The higher-order directed diagnostics are supportive."
        elif higher_order_score >= 0.60:
            prefix = "The higher-order directed diagnostics are mixed but broadly supportive."
        else:
            prefix = "The higher-order directed diagnostics are the weakest part of the fit."
        sentence = (
            f"{prefix} Weighted Pi-mass correlation is {best_pi_mass:.3f}"
            + (f", closed-class Pi correlation is {best_pi_closed:.3f}" if best_pi_closed is not None else "")
            + (f", and teleporting-PageRank Pi correlation is {best_pi_pagerank:.3f}" if best_pi_pagerank is not None else "")
            + (f", with Pi-gini correlation {best_pi_gini:.3f}" if best_pi_gini is not None else "")
            + (f" and LIC-share correlation {best_lic_share:.3f}" if best_lic_share is not None else "")
            + f", while weighted magnetic-spectrum correlation is {best_mag_corr:.3f}"
            + (f" at mean absolute eigenvalue error {best_mag_mae:.3f}" if best_mag_mae is not None else "")
            + (f" and mean spectral Wasserstein distance {best_mag_wasserstein:.3f}" if best_mag_wasserstein is not None else "")
            + "."
        )
        higher_order_parts.append(sentence)
    if higher_order_parts:
        conclusion_paragraphs.append(" ".join(higher_order_parts))

    synthesis_strengths = []
    synthesis_limitations = []
    best_novelty = float_or_none(best_detail, "mean_synthetic_novel_edge_rate")
    best_weight_corr = float_or_none(best_detail, "weight_total_correlation")
    best_edge_jaccard = float_or_none(best_detail, "mean_snapshot_edge_jaccard")
    best_recip_corr = float_or_none(best_detail, "reciprocity_correlation")

    if best_edge_jaccard is not None and best_edge_jaccard >= metric_median("mean_snapshot_edge_jaccard"):
        synthesis_strengths.append(f"snapshot edge overlap is strong for this slice (edge Jaccard {best_edge_jaccard:.3f})")
    if best_weight_corr is not None and best_weight_corr >= 0.95:
        synthesis_strengths.append(f"total weight is tracked closely through time (weight-total correlation {best_weight_corr:.3f})")
    if directed and best_recip_corr is not None and best_recip_corr >= 0.90:
        synthesis_strengths.append(f"directed reciprocity is preserved well enough for operational comparison (reciprocity correlation {best_recip_corr:.3f})")
    if best_tea_new is not None and best_tna_new is not None and best_tea_new >= 0.90 and best_tna_new >= 0.90:
        synthesis_strengths.append(f"edge and node turnover are both well aligned with the observed panel (TEA/TNA new-ratio correlations {best_tea_new:.3f}/{best_tna_new:.3f})")

    if best_novelty is not None and best_novelty >= 0.40:
        synthesis_limitations.append(f"novel edges are substantial ({best_novelty:.3f} mean novel-edge rate)")
    if best_block_weight is not None and best_block_weight < 0.80:
        synthesis_limitations.append(f"block-pair weight allocation is only partially recovered (mean block-pair weight correlation {best_block_weight:.3f})")
    if best_edge_type_weight is not None and best_edge_type_weight < 0.80:
        synthesis_limitations.append(f"hybrid edge-channel weight shares are not yet fully stable ({best_edge_type_weight:.3f} weight-share correlation)")
    if best_pi_mass is not None and best_pi_mass < 0.80:
        synthesis_limitations.append(f"stationary-mass structure is only partially reproduced (Pi-mass correlation {best_pi_mass:.3f})")
    if best_mag_corr is not None and best_mag_corr < 0.90:
        synthesis_limitations.append(f"magnetic-spectrum agreement is good but not exact ({best_mag_corr:.3f} correlation)")

    if synthesis_strengths or synthesis_limitations:
        class_phrase = "primary posterior-predictive workflow" if best_overlap_class == "posterior_predictive" and lowest_novelty_class == "posterior_predictive" else "reported workflow"
        conclusion_sentence = f"Taken together, the evidence supports the feasibility of this directed weighted hybrid-network {class_phrase} on the current CR35 slice because "
        conclusion_sentence += "; ".join(synthesis_strengths) if synthesis_strengths else "the sweep identifies at least one usable operating point"
        if synthesis_limitations:
            conclusion_sentence += ". At the same time, the report does not support a claim of fully faithful directed network reproduction because " + "; ".join(synthesis_limitations) + "."
        else:
            conclusion_sentence += "."
        conclusion_paragraphs.append(conclusion_sentence)
        if posterior_mode:
            conclusion_paragraphs.append(
                "These conclusions are based on setting-level posterior medians and interval summaries across repeated draws, not on a single generated panel."
            )

    if directed and worst_reciprocity_row is not None:
        worst_payload = _setting_display_payload(str(worst_reciprocity_row["sample_label"]))
        worst_class = _display_sample_class(str(worst_reciprocity_row.get("sample_class", _default_sample_class(str(worst_reciprocity_row["sample_label"])))))
        conclusion_paragraphs.append(
            f"The directed diagnostics also identify a clear failure mode: <strong>{html.escape(worst_payload['short_label'])}</strong> ({html.escape(worst_class)}) drops reciprocity correlation to {float(worst_reciprocity_row['reciprocity_correlation']):.3f} while producing superficially plausible totals. This is why the report keeps reciprocity and node-level in/out evidence alongside overlap and weight summaries."
        )

    def figure_tag(path: Optional[Path], caption: str, explain_text: Optional[str] = None) -> str:
        if path is None or not Path(path).exists():
            return ""
        rel = html.escape(Path(path).name)
        explain_html = _render_explanation_toggle(
            control_id=f"{re.sub(r'[^a-z0-9]+', '_', Path(path).stem.lower()).strip('_')}_explain",
            text=explain_text,
        )
        return (
            "<figure class='figure-card full-width-figure'>"
            f"{explain_html}"
            "<div class='figure-frame'>"
            f"<img src='{rel}' alt='{html.escape(caption)}' />"
            "</div>"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )

    def figure_switcher_tag(
        path: Optional[Path],
        caption: str,
        explain_text: Optional[str] = None,
        *,
        options: Optional[list[dict[str, str]]] = None,
    ) -> str:
        if path is None or not Path(path).exists():
            return ""
        valid_options = [option for option in (options or []) if option.get("path")]
        if len(valid_options) <= 1:
            return figure_tag(path, caption, explain_text)
        stem_key = re.sub(r"[^a-z0-9]+", "_", Path(path).stem.lower()).strip("_")
        select_id = f"{stem_key}_switch"
        image_id = f"{stem_key}_image"
        explain_html = _render_explanation_toggle(
            control_id=f"{stem_key}_explain",
            text=explain_text,
        )
        option_html = "".join(
            f"<option value='{html.escape(option['path'])}'{' selected' if index == 0 else ''}>{html.escape(option['label'])}</option>"
            for index, option in enumerate(valid_options)
        )
        return (
            "<figure class='figure-card full-width-figure'>"
            f"{explain_html}"
            "<div class='figure-toolbar'>"
            f"<label class='figure-switcher' for='{html.escape(select_id)}'>Display"
            f"<select id='{html.escape(select_id)}' class='figure-switcher-select' onchange=\"document.getElementById('{html.escape(image_id)}').src=this.value;\">"
            f"{option_html}"
            "</select>"
            "</label>"
            "</div>"
            "<div class='figure-frame'>"
            f"<img id='{html.escape(image_id)}' src='{html.escape(valid_options[0]['path'])}' alt='{html.escape(caption)}' />"
            "</div>"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )

    def iframe_tag(path: Optional[Path], caption: str, explain_text: Optional[str] = None, *, height: int = 760) -> str:
        if path is None or not Path(path).exists():
            return ""
        rel = html.escape(Path(path).name)
        explain_html = _render_explanation_toggle(
            control_id=f"{re.sub(r'[^a-z0-9]+', '_', Path(path).stem.lower()).strip('_')}_explain",
            text=explain_text,
        )
        return (
            "<figure class='figure-card full-width-figure'>"
            f"{explain_html}"
            "<div class='figure-frame iframe-frame'>"
            f"<iframe src='{rel}' title='{html.escape(caption)}' loading='lazy' style='width:100%;height:{int(height)}px;border:0;border-radius:14px;background:#fff;'></iframe>"
            "</div>"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )

    subtitle = (
        f"Objective assessment of {weighted_label.lower()}{'directed ' if directed else ''}temporal SBM fit quality on {dataset_name}, with emphasis on structural overlap, "
        + ("directional realism, " if directed else "")
        + ("posterior-predictive discrepancy, " if posterior_mode else "")
        + "edge novelty, total-weight fidelity, execution stability, and finer diagnostics by block pair and node through time."
    )

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<meta name='viewport' content='width=device-width, initial-scale=1' />",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        """
        :root {
          --bg: #edf3f8;
          --panel: #ffffff;
          --panel-soft: #f6f9fc;
          --text: #22313f;
          --muted: #607286;
          --line: #d8e1ea;
          --blue: #4e79a7;
          --orange: #f28e2b;
          --green: #59a14f;
          --red: #e15759;
          --shadow: rgba(34,49,63,0.06);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          padding: 0;
          background:
            radial-gradient(circle at top left, rgba(78,121,167,0.11), transparent 24%),
            radial-gradient(circle at top right, rgba(118,183,178,0.10), transparent 28%),
            linear-gradient(180deg, #f7fafc 0%, var(--bg) 100%);
          color: var(--text);
          font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
          line-height: 1.55;
        }
        .page {
          max-width: 1380px;
          margin: 0 auto;
          padding: 40px 32px 72px;
        }
        .hero {
          background:
            radial-gradient(circle at top right, rgba(118,183,178,0.14), transparent 34%),
            linear-gradient(135deg, rgba(78,121,167,0.10), rgba(118,183,178,0.07));
          border: 1px solid rgba(78,121,167,0.16);
          border-radius: 24px;
          padding: 32px 34px;
          box-shadow: 0 20px 42px rgba(34,49,63,0.08);
        }
        .eyebrow {
          font-size: 12px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--blue);
          font-weight: 700;
        }
        h1 {
          margin: 8px 0 10px;
          font-size: 36px;
          line-height: 1.08;
        }
        .subtitle {
          margin: 0;
          color: var(--muted);
          max-width: 980px;
          font-size: 16px;
        }
        .summary-grid, .figure-grid, .two-col, .section-grid {
          display: grid;
          gap: 18px;
        }
        .summary-grid { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 24px; }
        .two-col { grid-template-columns: 1.1fr 0.9fr; margin-top: 26px; }
        .section-grid { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); margin-top: 18px; }
        .card, .figure-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 18px 20px 18px;
          box-shadow: 0 14px 30px var(--shadow);
          min-width: 0;
        }
        .summary-card {
          min-height: 190px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }
        .metric-value {
          font-size: 46px;
          font-weight: 700;
          margin: 10px 0 6px;
          line-height: 1;
          letter-spacing: -0.03em;
        }
        .metric-label {
          color: var(--muted);
          font-size: 13px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .model-title, .setting-title {
          font-size: 20px;
          font-weight: 700;
          line-height: 1.28;
          overflow-wrap: anywhere;
        }
        .setting-subtitle {
          margin-top: 6px;
          color: var(--muted);
          font-size: 14px;
          overflow-wrap: anywhere;
        }
        .setting-code {
          display: inline-block;
          margin-top: 8px;
          padding: 6px 9px;
          border-radius: 10px;
          background: var(--panel-soft);
          color: var(--muted);
          font-size: 12px;
          line-height: 1.35;
          white-space: normal;
          word-break: break-word;
        }
        .badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        h2 {
          margin: 34px 0 14px;
          font-size: 26px;
        }
        h3 {
          margin: 0 0 10px;
          font-size: 18px;
        }
        p, li { color: var(--text); }
        ul { padding-left: 20px; margin: 10px 0 0; }
        section > .card + .card,
        section > .card + .figure-grid,
        section > .figure-grid + .card {
          margin-top: 18px;
        }
        .table-widget {
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-width: 0;
        }
        .table-toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }
        .table-search-wrap {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-left: auto;
        }
        .explain-widget {
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: flex-start;
        }
        .explain-button {
          appearance: none;
          border: 1px solid rgba(78,121,167,0.24);
          background: linear-gradient(180deg, #fafdff 0%, #edf4f9 100%);
          color: var(--blue);
          border-radius: 999px;
          padding: 8px 12px;
          font: inherit;
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.02em;
          cursor: pointer;
          box-shadow: 0 6px 14px rgba(78,121,167,0.08);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .explain-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 10px 20px rgba(78,121,167,0.12);
          border-color: rgba(78,121,167,0.36);
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
        .explain-panel p {
          margin: 0;
          color: inherit;
        }
        .table-search-label {
          color: var(--muted);
          font-size: 13px;
          font-weight: 600;
        }
        .table-search-input {
          width: min(320px, 100%);
          padding: 10px 12px;
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #fbfdff;
          color: var(--text);
          font: inherit;
        }
        .table-search-input:focus {
          outline: none;
          border-color: rgba(78,121,167,0.45);
          box-shadow: 0 0 0 4px rgba(78,121,167,0.12);
        }
        .table-scroll {
          overflow: auto;
          max-width: 100%;
          border: 1px solid var(--line);
          border-radius: 16px;
          background: var(--panel-soft);
          -webkit-overflow-scrolling: touch;
        }
        .report-table {
          width: 100%;
          min-width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          font-size: 14px;
          table-layout: auto;
        }
        .report-table th, .report-table td {
          border-bottom: 1px solid var(--line);
          padding: 12px 14px;
          text-align: left;
          vertical-align: top;
          background: #ffffff;
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
          hyphens: auto;
          max-width: 18rem;
        }
        .report-table th {
          position: sticky;
          top: 0;
          z-index: 3;
          background: #eef4f8;
          color: var(--muted);
          font-size: 12px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          cursor: pointer;
          white-space: nowrap;
          overflow-wrap: normal;
          word-break: normal;
          hyphens: manual;
        }
        .report-table th:first-child,
        .report-table td:first-child {
          position: sticky;
          left: 0;
          z-index: 2;
          background: #ffffff;
        }
        .report-table td:first-child {
          white-space: nowrap;
          overflow-wrap: normal;
          word-break: normal;
          hyphens: manual;
        }
        .report-table td.allow-wrap {
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
          hyphens: auto;
        }
        .report-table th:first-child {
          z-index: 4;
          background: #eef4f8;
        }
        .report-table tr:last-child td { border-bottom: 0; }
        .report-table th.sorted-asc::after { content: " ▲"; color: var(--blue); }
        .report-table th.sorted-desc::after { content: " ▼"; color: var(--blue); }
        .setting-cell {
          min-width: 320px;
          white-space: normal;
        }
        .figure-grid {
          grid-template-columns: 1fr;
          margin-top: 16px;
        }
        .figure-card {
          padding: 18px 18px 16px;
        }
        .figure-card .explain-widget {
          margin-bottom: 12px;
        }
        .figure-toolbar {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          margin-bottom: 10px;
        }
        .figure-switcher {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          color: var(--muted);
          font-size: 13px;
          font-weight: 600;
        }
        .figure-switcher-select {
          min-width: 180px;
          padding: 9px 12px;
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #fbfdff;
          color: var(--text);
          font: inherit;
        }
        .figure-switcher-select:focus {
          outline: none;
          border-color: rgba(78,121,167,0.45);
          box-shadow: 0 0 0 4px rgba(78,121,167,0.12);
        }
        .figure-frame {
          padding: 16px;
          border-radius: 16px;
          background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%);
          border: 1px solid var(--line);
        }
        .iframe-frame {
          padding: 10px;
        }
        .figure-card img {
          width: 100%;
          border-radius: 14px;
          display: block;
          background: var(--panel-soft);
        }
        .figure-card figcaption {
          margin-top: 10px;
          color: var(--muted);
          font-size: 13px;
        }
        .section-note {
          color: var(--muted);
          margin-top: -6px;
          max-width: 980px;
        }
        .pill {
          display: inline-block;
          padding: 5px 11px;
          border-radius: 999px;
          background: #edf3f8;
          color: var(--blue);
          font-size: 12px;
          font-weight: 600;
        }
        @media (max-width: 980px) {
          .two-col { grid-template-columns: 1fr; }
          .page { padding: 28px 18px 52px; }
          h1 { font-size: 29px; }
          .summary-grid { grid-template-columns: 1fr; }
          .metric-value { font-size: 38px; }
          .table-toolbar { justify-content: stretch; }
          .table-search-wrap { width: 100%; margin-left: 0; }
          .table-search-input { width: 100%; }
          .explain-widget { width: 100%; }
          .explain-button { width: 100%; text-align: left; }
          .figure-toolbar { justify-content: stretch; }
          .figure-switcher { width: 100%; justify-content: space-between; }
          .figure-switcher-select { flex: 1 1 auto; min-width: 0; }
        }
        """,
        "</style>",
        "</head>",
        "<body>",
        "<main class='page'>",
        "<section class='hero'>",
        "<div class='eyebrow'>Temporal SBM Validation</div>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='subtitle'>{html.escape(subtitle)}</p>",
        "<div class='summary-grid'>",
        f"<div class='card summary-card'><div><div class='metric-label'>Primary Best Overlap</div><div class='metric-value'>{float(best_overlap['mean_snapshot_edge_jaccard']):.3f}</div>{metric_card_meta(best_overlap, 'mean_snapshot_edge_jaccard')}</div><div>{setting_card_meta(best_overlap_setting)}</div></div>",
        f"<div class='card summary-card'><div><div class='metric-label'>Primary Lowest Novelty</div><div class='metric-value'>{float(lowest_novelty['mean_synthetic_novel_edge_rate']):.3f}</div>{metric_card_meta(lowest_novelty, 'mean_synthetic_novel_edge_rate')}</div><div>{setting_card_meta(lowest_novelty_setting)}</div></div>",
        f"<div class='card summary-card'><div><div class='metric-label'>Settings Tested</div><div class='metric-value'>{len(summary_rows)}</div></div><div class='setting-subtitle'>{len(primary_rows)} primary, {len(sensitivity_rows)} sensitivity; {snapshot_count} consecutive snapshots</div><div class='setting-subtitle'>{'Posterior medians with 5th-95th percentile intervals' if posterior_mode else 'Single generated panel per setting'}</div></div>",
        (
            f"<div class='card summary-card'><div><div class='metric-label'>Weight Model</div></div><div>{weight_model_card_html()}</div></div>"
            if manifest.get("weight_model")
            else f"<div class='card summary-card'><div><div class='metric-label'>Network Type</div><div class='metric-value'>{'Directed' if directed else 'Undirected'}</div></div><div class='setting-subtitle'>Simple graph evaluation without self-loops or multiedges</div></div>"
        ),
        "</div>",
        "</section>",
        "<section>",
        "<h2>Methodology</h2>",
        "<div class='two-col'>",
        f"<div class='card'>{_render_html_table_widget(methodology_table, table_id='methodology_table', searchable=False, explain_text=table_help['methodology_table'])}</div>",
        "<div class='card'><h3>Assessment framing</h3><p class='section-note'>"
        + html.escape(constraint_note)
        + "</p><ul>"
        + "".join(f"<li>{html.escape(point)}</li>" for point in feasibility_points)
        + "</ul></div>",
        "</div>",
        "<div class='section-grid'>",
        f"<div class='card'><h3>Metric definitions</h3>{_render_html_table_widget(metric_definitions, table_id='metric_definitions_table', searchable=False, explain_text=table_help['metric_definitions_table'])}</div>",
        (
            f"<div class='card'><h3>Hybrid network summary</h3>{_render_html_table_widget(hybrid_overview_table, table_id='hybrid_overview_table', searchable=False, explain_text='This table compares the observed hybrid network against the posterior mean of the selected setting across its generated runs. Values are rounded to integers so you can compare active node counts and F→F, F→R, R→F, and R→R channel totals directly.')}</div>"
            if len(hybrid_overview_table)
            else ""
        ),
        "</div>",
        "</section>",
        "<section>",
        "<h2>Execution Stability</h2>",
        "<p class='section-note'>Verbose logs from the fit, generate, and report stages were aggregated to check runtime consistency and to confirm that the validation sweep completed without systemic warning or error cascades.</p>",
        "<div class='figure-grid'>",
        figure_tag(runtime_profile_path, "Runtime profile across fit, generate, and report stages.", figure_explanation(runtime_profile_path)),
        "</div>",
        f"<div class='card'>{_render_html_table_widget(runtime_display, table_id='runtime_table', searchable=False, explain_text=table_help['runtime_table']) if len(runtime_display) else '<p>No runtime summaries were available.</p>'}</div>",
        "</section>",
        "<section>",
        "<h2>Sweep Results</h2>",
        "<p class='section-note'>The setting sweep compares five sampling modes crossed with four rewiring modes on the same fitted model. "
        + ("Points and table entries represent posterior medians with 5th-95th percentile intervals for settings with repeated draws. " if posterior_mode else "")
        + "The frontier view emphasizes the trade-off between overlap and novelty; the scorecard view summarizes the leading settings across the main metrics"
        + (" and directed-network realism." if directed else ".")
        + "</p>",
        "<div class='figure-grid'>",
        figure_tag(frontier_path, "Novelty versus overlap across the 20 evaluated settings. Bubble size reflects weight-total correlation.", figure_explanation(frontier_path)),
        figure_tag(metric_matrix_path, "Metric scorecard for the leading settings, with novelty interpreted as a lower-is-better objective.", figure_explanation(metric_matrix_path)),
        figure_tag(overview_path, "Overview of the run-level metrics across all evaluated settings.", figure_explanation(overview_path)),
        "</div>",
        f"<div class='card'>{_render_html_table_widget(top_settings_display, table_id='top_settings_table', max_rows=8, explain_text=table_help['top_settings_table'])}</div>",
        "</section>",
        "<section>",
        "<h2>Selected Settings</h2>",
        "<p class='section-note'>Two settings were examined in more detail: the strongest-overlap configuration and the lowest-novelty configuration. This keeps the contrast between structural fit, local weight allocation, and synthetic edge invention explicit.</p>",
        "<div class='figure-grid'>",
        figure_tag(selected_local_fit_path, "Summary of block-pair weight fidelity and top-node temporal correlations for the selected settings.", figure_explanation(selected_local_fit_path)),
        "</div>",
        f"<div class='card'>{_render_html_table_widget(selected_display, table_id='selected_settings_table', explain_text=table_help['selected_settings_table'])}</div>",
        "</section>",
    ]

    if phase_assets:
        html_parts.extend(
            [
                "<section>",
                "<h2>Interactive Magnetic Phase Diagnostics</h2>",
                "<p class='section-note'>These interactive views complement the static magnetic-spectrum figures. They expose where directional phase structure sits geographically, how regional supernode phases drift through time, and how the observed panel compares against the selected synthetic settings on the unit circle.</p>",
                "<div class='figure-grid'>",
                iframe_tag(
                    phase_assets.get("phase_geo_html"),
                    "Interactive COROP basemap comparison for magnetic phase evolution through time.",
                    "Use the snapshot slider or autoplay to inspect how phase structure moves through the hybrid network. Color encodes phase, opacity tracks eigenvector magnitude, circles mark farms, and triangles mark regional supernodes.",
                    height=840,
                ),
                iframe_tag(
                    phase_assets.get("phase_tracks_html"),
                    "Interactive node-phase tracks for regional supernodes and top farm nodes.",
                    "This view unwraps phase through time. Stable parallel traces indicate coherent directional dynamics, while large separations or crossings indicate mode reallocation among nodes. Blue bands mark weekends and red bands mark Dutch public holidays.",
                    height=720,
                ),
                iframe_tag(
                    phase_assets.get("phase_panels_html"),
                    "Interactive magnetic phase panels comparing observed and selected settings on the unit circle.",
                    "Each panel places node phases on the unit circle for one eigen mode. The inner ring is the observed reference radius, and synthetic nodes move outward as their polar mismatch from the observed phase grows. Compare the observed panel against each selected setting at the same snapshot to see whether directional orientation and node-level phase agreement are preserved.",
                    height=760,
                ),
                "</div>",
                "</section>",
            ]
        )

    if network_assets:
        html_parts.extend(
            [
                "<section>",
                "<h2>Daily Network Snapshots</h2>",
                "<p class='section-note'>These paired PDFs compare observed and synthetic daily networks in two layouts. The viewer switches across the selected settings and their generated runs, while keeping the same day and layout on both sides so you can compare structural differences directly.</p>",
                "<div class='figure-grid'>",
                iframe_tag(
                    network_assets.get("network_compare_html"),
                    "Interactive daily network comparison in forced and geographic layouts.",
                    "Each panel is a pre-rendered PDF. The left half of the PDF is the observed network for one day, and the right half is the matching synthetic network. Use the two panel selectors to compare settings or runs on the same day.",
                    height=980,
                ),
                "</div>",
                "</section>",
            ]
        )

    for section in selected_sections:
        html_parts.extend(
            [
                "<section>",
                f"<h2>{html.escape(section['display']['short_label'])}</h2>",
                f"<p class='section-note'><code class='setting-code'>{html.escape(section['label'])}</code></p>",
                f"<p>{html.escape(section['narrative'])}</p>" if section["narrative"] else "",
                "<div class='figure-grid'>",
            ]
        )
        for figure_spec in section["figures"]:
            figure_path = Path(str(figure_spec["path"]))
            caption = f"{section['display']['short_label']} :: {figure_path.stem.split(section['label'] + '_', 1)[-1].replace('_', ' ').title()}"
            switcher_options = figure_spec.get("switcher_options")
            if isinstance(switcher_options, list) and switcher_options:
                html_parts.append(
                    figure_switcher_tag(
                        figure_path,
                        caption,
                        figure_explanation(figure_path),
                        options=[option for option in switcher_options if isinstance(option, dict)],
                    )
                )
            else:
                html_parts.append(figure_tag(figure_path, caption, figure_explanation(figure_path)))
        html_parts.append("</div>")
        if section["tables"].get("block_pairs") is not None:
            block_table_id = re.sub(r"[^a-z0-9]+", "_", section["label"].lower()).strip("_") + "_block_pairs"
            html_parts.append(
                f"<div class='card'><h3>Block-pair evidence</h3>{_render_html_table_widget(section['tables']['block_pairs'], table_id=block_table_id, explain_text='Each row is a block pair. High correlations and small absolute deltas mean the model reproduces the timing and scale of traffic between those blocks.')}</div>"
            )
        if section["tables"].get("nodes") is not None:
            node_table_id = re.sub(r"[^a-z0-9]+", "_", section["label"].lower()).strip("_") + "_nodes"
            html_parts.append(
                f"<div class='card'><h3>Top-node evidence</h3>{_render_html_table_widget(section['tables']['nodes'], table_id=node_table_id, explain_text='Each row is one of the most active nodes. Compare outgoing and incoming correlations to judge directional timing fit, and use the weight correlations to judge whether intensity is placed on the right nodes.')}</div>"
            )
        if section["tables"].get("edge_types") is not None:
            edge_type_table_id = re.sub(r"[^a-z0-9]+", "_", section["label"].lower()).strip("_") + "_edge_types"
            html_parts.append(
                f"<div class='card'><h3>Hybrid edge-type evidence</h3>{_render_html_table_widget(section['tables']['edge_types'], table_id=edge_type_table_id, explain_text='Each row is one hybrid edge channel. Use these correlations to see whether the model keeps activity on the right F→F, F→R, R→F, and R→R pathways through time.')}</div>"
            )
        if section["tables"].get("reachability_sources") is not None:
            reach_source_table_id = re.sub(r"[^a-z0-9]+", "_", section["label"].lower()).strip("_") + "_reachability_sources"
            html_parts.append(
                f"<div class='card'><h3>Transmission-source evidence</h3>{_render_html_table_widget(section['tables']['reachability_sources'], table_id=reach_source_table_id, explain_text='Each row is a source node. Compare the final forward reachable fractions to see whether the synthetic panel gives the same nodes similar causal reach across the full observation window.')}</div>"
            )
        advanced_table_specs = [
            ("tea", "TEA summary", "These rows summarize how closely the synthetic panel matches observed edge appearance, persistence, reactivation, and churn through time."),
            ("tna", "TNA summary", "These rows summarize whether nodes become active, persist, reactivate, and churn at the right times."),
            ("temporal_reachability", "Temporal reachability summary", "These rows summarize how closely the synthetic panel matches the observed transmission envelope under strict time-respecting paths: total reachable pairs, reachability ratio, new reachable pairs, temporal efficiency, and mean arrival time."),
            ("pi_mass", "Pi-Mass / LIC summary", "These rows summarize stationary flow concentration on the largest strongly connected component under the lazy walk, along with support size and daily active-node counts."),
            ("pi_mass_closed", "Pi-Mass / closed-class summary", "These rows summarize stationary flow concentration on the largest closed strongly connected class under the lazy walk, along with support size and daily active-node counts."),
            ("pi_mass_pagerank", "Pi-Mass / PageRank summary", "These rows summarize whole-snapshot teleporting PageRank mass by node type, along with active-snapshot size and daily active-node counts."),
            ("magnetic", "Magnetic spectrum summary", "Each row is one of the smallest magnetic Laplacian modes. Higher correlations and lower absolute deltas indicate better preservation of directed spectral structure."),
            ("magnetic_distance", "Magnetic spectral distances", "These rows summarize per-snapshot Wasserstein and aligned-mode errors between the observed and synthetic magnetic spectra; lower is better."),
        ]
        for table_key, table_title, explain_text in advanced_table_specs:
            table_frame = section["tables"].get(table_key)
            if table_frame is None:
                continue
            table_id = re.sub(r"[^a-z0-9]+", "_", f"{section['label']}_{table_key}".lower()).strip("_")
            html_parts.append(
                f"<div class='card'><h3>{html.escape(table_title)}</h3>{_render_html_table_widget(table_frame, table_id=table_id, explain_text=explain_text)}</div>"
            )
        html_parts.append("</section>")

    html_parts.extend(
        [
            "<section>",
            "<h2>Conclusion</h2>",
            "<div class='card'>",
            *[f"<p>{paragraph}</p>" for paragraph in conclusion_paragraphs],
            "</div>",
            "</section>",
            "</main>",
            "<script>",
            """
            document.addEventListener('DOMContentLoaded', () => {
              const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

              const toComparable = (value) => {
                const text = (value || '').replace(/,/g, '').trim();
                const numeric = Number.parseFloat(text);
                if (!Number.isNaN(numeric) && /[-+0-9.]/.test(text)) {
                  return { type: 'number', value: numeric };
                }
                return { type: 'text', value: text.toLowerCase() };
              };

              document.querySelectorAll('.table-search-input').forEach((input) => {
                input.addEventListener('input', () => {
                  const table = document.getElementById(input.dataset.tableTarget);
                  if (!table) return;
                  const query = input.value.trim().toLowerCase();
                  table.querySelectorAll('tbody tr').forEach((row) => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = !query || text.includes(query) ? '' : 'none';
                  });
                });
              });

              document.querySelectorAll('.explain-button').forEach((button) => {
                button.addEventListener('click', () => {
                  const panel = document.getElementById(button.dataset.explainTarget);
                  if (!panel) return;
                  const expanded = button.getAttribute('aria-expanded') === 'true';
                  button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
                  panel.hidden = expanded;
                });
              });

              document.querySelectorAll('.sortable-table').forEach((table) => {
                const headers = Array.from(table.querySelectorAll('thead th'));
                headers.forEach((header, index) => {
                  header.addEventListener('click', () => {
                    const tbody = table.querySelector('tbody');
                    if (!tbody) return;
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    const direction = header.classList.contains('sorted-asc') ? 'desc' : 'asc';
                    headers.forEach((cell) => cell.classList.remove('sorted-asc', 'sorted-desc'));
                    header.classList.add(direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
                    rows.sort((rowA, rowB) => {
                      const valueA = toComparable(rowA.children[index]?.textContent || '');
                      const valueB = toComparable(rowB.children[index]?.textContent || '');
                      let result = 0;
                      if (valueA.type === 'number' && valueB.type === 'number') {
                        result = valueA.value - valueB.value;
                      } else {
                        result = collator.compare(String(valueA.value), String(valueB.value));
                      }
                      return direction === 'asc' ? result : -result;
                    });
                    rows.forEach((row) => tbody.appendChild(row));
                  });
                });
              });
            });
            """,
            "</script>",
            "</body>",
            "</html>",
        ]
    )

    output_path.write_text("\n".join(part for part in html_parts if part != ""))
    LOGGER.info("Wrote scientific validation report to %s", output_path)
    return output_path
