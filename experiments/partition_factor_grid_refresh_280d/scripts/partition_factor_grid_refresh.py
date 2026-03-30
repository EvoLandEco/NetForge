#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_sbm.cli import add_fit_arguments, add_input_arguments
from temporal_sbm.pipeline import (
    _posterior_partition_state,
    attach_partition_maps,
    extract_node_block_map,
    fit_command,
    load_manifest,
    load_state,
)
from temporal_sbm.sweep import _build_namespace_from_config


LOGGER = logging.getLogger("partition_factor_grid_refresh")
EDGE_COVARIATES = ["dist_km", "mass_grav", "anim_grav"]


def _combo_label(
    *,
    joint_metadata_model: bool,
    edge_covariates: bool,
    exclude_weight_from_fit: bool,
    layered: bool,
) -> str:
    return (
        f"meta_{'on' if joint_metadata_model else 'off'}"
        f"__cov_{'on' if edge_covariates else 'off'}"
        f"__exw_{'on' if exclude_weight_from_fit else 'off'}"
        f"__layered_{'on' if layered else 'off'}"
    )


def _combo_short_label(
    *,
    joint_metadata_model: bool,
    edge_covariates: bool,
    exclude_weight_from_fit: bool,
    layered: bool,
) -> str:
    return (
        f"M{int(joint_metadata_model)} "
        f"C{int(edge_covariates)} "
        f"W{int(exclude_weight_from_fit)} "
        f"L{int(layered)}"
    )


def _combo_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    index = 0
    for joint_metadata_model in (False, True):
        for edge_covariates in (False, True):
            for exclude_weight_from_fit in (False, True):
                for layered in (False, True):
                    specs.append(
                        {
                            "index": index,
                            "label": _combo_label(
                                joint_metadata_model=joint_metadata_model,
                                edge_covariates=edge_covariates,
                                exclude_weight_from_fit=exclude_weight_from_fit,
                                layered=layered,
                            ),
                            "short_label": _combo_short_label(
                                joint_metadata_model=joint_metadata_model,
                                edge_covariates=edge_covariates,
                                exclude_weight_from_fit=exclude_weight_from_fit,
                                layered=layered,
                            ),
                            "joint_metadata_model": bool(joint_metadata_model),
                            "edge_covariates": bool(edge_covariates),
                            "exclude_weight_from_fit": bool(exclude_weight_from_fit),
                            "layered": bool(layered),
                        }
                    )
                    index += 1
    return specs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _edges_csv_path(base_fit: dict[str, Any]) -> Path:
    if base_fit.get("edges_csv"):
        return Path(str(base_fit["edges_csv"])).expanduser().resolve()
    data_root = Path(str(base_fit["data_root"])).expanduser().resolve()
    dataset = str(base_fit["dataset"])
    return data_root / dataset / "edges.csv"


def _dataset_ts_bounds(base_fit: dict[str, Any]) -> tuple[int, int]:
    edges_csv = _edges_csv_path(base_fit)
    ts_col = str(base_fit.get("ts_col", "ts"))
    frame = pd.read_csv(edges_csv, usecols=[ts_col])
    return int(frame[ts_col].min()), int(frame[ts_col].max())


def _default_output_root(days: int, replicates: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(".stress_runs") / f"cr35_partition_factor_grid_{days}d_rep{replicates}_{stamp}"


def _fit_values_for_combo(
    *,
    base_fit: dict[str, Any],
    combo_spec: dict[str, Any],
    combo_dir: Path,
    ts_start: int,
    ts_end: int,
) -> dict[str, Any]:
    fit_values = dict(base_fit)
    fit_values["output_dir"] = str((combo_dir / "fit").resolve())
    fit_values["ts_start"] = int(ts_start)
    fit_values["ts_end"] = int(ts_end)
    fit_values["joint_metadata_model"] = bool(combo_spec["joint_metadata_model"])
    fit_values["fit_covariates"] = list(EDGE_COVARIATES) if combo_spec["edge_covariates"] else ["none"]
    fit_values["exclude_weight_from_fit"] = bool(combo_spec["exclude_weight_from_fit"])
    fit_values["layered"] = bool(combo_spec["layered"])
    fit_values["fit_quiet"] = True
    return fit_values


def _type_labels(frame: pd.DataFrame) -> pd.Series:
    if "type_label" in frame.columns:
        labels = frame["type_label"].fillna("Unknown").astype(str)
        return labels
    if "type" in frame.columns:
        mapping = {0: "Farm", 1: "Region"}
        return frame["type"].map(mapping).fillna(frame["type"].astype(str))
    return pd.Series(["Unknown"] * len(frame), index=frame.index, dtype="object")


def _comb2(values: np.ndarray) -> np.ndarray:
    return values * (values - 1) / 2.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("Partition label arrays must have the same shape.")
    if labels_a.size <= 1:
        return 1.0

    _, inverse_a = np.unique(labels_a, return_inverse=True)
    _, inverse_b = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((inverse_a.max() + 1, inverse_b.max() + 1), dtype=np.int64)
    np.add.at(contingency, (inverse_a, inverse_b), 1)

    sum_index = float(_comb2(contingency).sum())
    sum_rows = float(_comb2(contingency.sum(axis=1)).sum())
    sum_cols = float(_comb2(contingency.sum(axis=0)).sum())
    total_pairs = float(labels_a.size * (labels_a.size - 1) / 2.0)
    if total_pairs <= 0:
        return 1.0
    expected = (sum_rows * sum_cols) / total_pairs
    max_index = 0.5 * (sum_rows + sum_cols)
    denominator = max_index - expected
    if abs(denominator) < 1e-12:
        return 1.0
    return float((sum_index - expected) / denominator)


def _partition_frame(
    *,
    node_frame: pd.DataFrame,
    node_block_map: dict[int, int],
) -> pd.DataFrame:
    frame = node_frame.copy()
    frame["block_id"] = frame["node_id"].map(node_block_map)
    if frame["block_id"].isna().any():
        missing = frame.loc[frame["block_id"].isna(), "node_id"].tolist()[:10]
        raise ValueError(f"Block map is missing node ids: {missing}")
    frame["block_id"] = frame["block_id"].astype(int)
    frame["type_label"] = _type_labels(frame)
    return frame


def _replicate_metrics(
    partition_frame: pd.DataFrame,
    *,
    replicate_index: int,
    seed: int,
    fit_labels: np.ndarray,
) -> dict[str, Any]:
    block_sizes = partition_frame.groupby("block_id").size()
    by_type = (
        partition_frame.groupby(["block_id", "type_label"]).size().unstack(fill_value=0)
        if len(partition_frame)
        else pd.DataFrame()
    )
    farm_present = by_type.get("Farm", pd.Series(dtype=int)).reindex(block_sizes.index, fill_value=0) > 0
    region_present = by_type.get("Region", pd.Series(dtype=int)).reindex(block_sizes.index, fill_value=0) > 0
    current_labels = partition_frame["block_id"].to_numpy(dtype=np.int64)
    active_frame = partition_frame.loc[partition_frame.get("active_in_window", 1).astype(int) == 1]

    return {
        "replicate_index": int(replicate_index),
        "seed": int(seed),
        "block_count": int(block_sizes.size),
        "active_block_count": int(active_frame["block_id"].nunique()),
        "pure_farm_block_count": int((farm_present & ~region_present).sum()),
        "pure_region_block_count": int((region_present & ~farm_present).sum()),
        "mixed_type_block_count": int((farm_present & region_present).sum()),
        "largest_block_size": int(block_sizes.max()),
        "largest_block_fraction": float(block_sizes.max() / len(partition_frame)),
        "singleton_block_count": int((block_sizes == 1).sum()),
        "ari_to_fit": float(adjusted_rand_index(fit_labels, current_labels)),
    }


def _write_partition_frame(partition_frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partition_frame.to_csv(path, index=False)


def plan_command(args: argparse.Namespace) -> int:
    base_payload = _load_json(Path(args.base_config))
    if "fit" not in base_payload:
        raise ValueError("Base config must contain a 'fit' object.")
    base_fit = dict(base_payload["fit"])
    base_generate = dict(base_payload.get("generate", {}))

    ts_start = int(args.start_ts if args.start_ts is not None else base_fit.get("ts_start"))
    if ts_start is None:
        raise ValueError("A start timestamp is required.")

    min_ts, max_ts = _dataset_ts_bounds(base_fit)
    ts_end = ts_start + int(args.days) - 1
    if ts_start < min_ts or ts_end > max_ts:
        raise ValueError(
            f"Requested slice [{ts_start}, {ts_end}] falls outside dataset bounds [{min_ts}, {max_ts}]."
        )

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _default_output_root(args.days, args.replicates).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    refresh_options = {
        "posterior_partition_sweeps": int(
            args.posterior_partition_sweeps
            if args.posterior_partition_sweeps is not None
            else base_generate.get("posterior_partition_sweeps", 10)
        ),
        "posterior_partition_sweep_niter": int(
            args.posterior_partition_sweep_niter
            if args.posterior_partition_sweep_niter is not None
            else base_generate.get("posterior_partition_sweep_niter", 5)
        ),
        "posterior_partition_beta": float(
            args.posterior_partition_beta
            if args.posterior_partition_beta is not None
            else base_generate.get("posterior_partition_beta", 1.0)
        ),
        "seed_base": int(args.seed if args.seed is not None else base_generate.get("seed", 2026)),
    }

    combo_specs = []
    for combo_spec in _combo_specs():
        combo_dir = output_root / "combos" / combo_spec["label"]
        fit_values = _fit_values_for_combo(
            base_fit=base_fit,
            combo_spec=combo_spec,
            combo_dir=combo_dir,
            ts_start=ts_start,
            ts_end=ts_end,
        )
        combo_payload = dict(combo_spec)
        combo_payload["combo_dir"] = str(combo_dir.resolve())
        combo_payload["fit_values"] = fit_values
        combo_specs.append(combo_payload)
        _save_json(combo_dir / "config.json", {"fit": fit_values})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(Path(args.base_config).expanduser().resolve()),
        "output_root": str(output_root),
        "days": int(args.days),
        "total_replicates": int(args.replicates),
        "ts_start": int(ts_start),
        "ts_end": int(ts_end),
        "dataset_bounds": {"ts_min": int(min_ts), "ts_max": int(max_ts)},
        "refresh_options": refresh_options,
        "base_fit": base_fit,
        "base_generate": base_generate,
        "combo_count": len(combo_specs),
    }

    _save_json(output_root / "run_manifest.json", manifest)
    _save_json(output_root / "combo_specs.json", combo_specs)
    (output_root / "combo_labels.txt").write_text("\n".join(spec["label"] for spec in combo_specs) + "\n")
    print(str(output_root))
    return 0


def _refresh_namespace(run_manifest: dict[str, Any]) -> argparse.Namespace:
    refresh_options = run_manifest["refresh_options"]
    return argparse.Namespace(
        posterior_partition_sweeps=int(refresh_options["posterior_partition_sweeps"]),
        posterior_partition_sweep_niter=int(refresh_options["posterior_partition_sweep_niter"]),
        posterior_partition_beta=float(refresh_options["posterior_partition_beta"]),
    )


def run_combo_command(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    run_manifest = _load_json(output_root / "run_manifest.json")
    combo_specs = _load_json(output_root / "combo_specs.json")
    combo_spec = next((spec for spec in combo_specs if spec["label"] == args.combo), None)
    if combo_spec is None:
        raise ValueError(f"Unknown combo label: {args.combo}")

    combo_dir = Path(combo_spec["combo_dir"]).expanduser().resolve()
    _configure_logging(combo_dir / "job.log")
    LOGGER.info("Running combo %s", combo_spec["label"])

    fit_values = dict(combo_spec["fit_values"])
    fit_run_dir = Path(fit_values["output_dir"]).expanduser().resolve()
    if (fit_run_dir / "manifest.json").exists() and (fit_run_dir / "node_attributes.csv").exists():
        manifest = load_manifest(fit_run_dir)
        LOGGER.info("Found existing fit artifacts under %s", fit_run_dir)
    else:
        fit_args = _build_namespace_from_config(fit_values, adders=[add_input_arguments, add_fit_arguments])
        manifest = fit_command(fit_args)

    node_frame = pd.read_csv(manifest["node_attributes_path"]).sort_values("node_id").reset_index(drop=True)
    graph_tool = __import__("graph_tool.all", fromlist=["load_graph"])
    graph = graph_tool.load_graph(str(manifest["graph_path"]))
    fitted_state = load_state(Path(manifest["state_path"]))
    refresh_args = _refresh_namespace(run_manifest)
    total_replicates = int(run_manifest["total_replicates"])
    seed_base = int(run_manifest["refresh_options"]["seed_base"]) + int(combo_spec["index"]) * 10000

    partitions_dir = combo_dir / "partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)
    replicate_metrics: list[dict[str, Any]] = []
    label_vectors: dict[int, np.ndarray] = {}

    def capture_partition(state: Any, replicate_index: int, seed: int) -> pd.DataFrame:
        attach_partition_maps(graph, state)
        node_block_map = extract_node_block_map(graph)
        partition_frame = _partition_frame(node_frame=node_frame, node_block_map=node_block_map)
        _write_partition_frame(partition_frame, partitions_dir / f"replicate_{replicate_index:03d}.csv")
        return partition_frame

    fitted_partition = capture_partition(fitted_state, replicate_index=0, seed=seed_base)
    fit_labels = fitted_partition["block_id"].to_numpy(dtype=np.int64)
    label_vectors[0] = fit_labels
    replicate_metrics.append(
        _replicate_metrics(
            fitted_partition,
            replicate_index=0,
            seed=seed_base,
            fit_labels=fit_labels,
        )
    )

    for replicate_index in range(1, total_replicates):
        seed = seed_base + replicate_index
        refreshed_state = _posterior_partition_state(fitted_state, seed=seed, args=refresh_args)
        partition_frame = capture_partition(refreshed_state, replicate_index=replicate_index, seed=seed)
        current_labels = partition_frame["block_id"].to_numpy(dtype=np.int64)
        label_vectors[replicate_index] = current_labels
        replicate_metrics.append(
            _replicate_metrics(
                partition_frame,
                replicate_index=replicate_index,
                seed=seed,
                fit_labels=fit_labels,
            )
        )

    pairwise_rows = []
    pairwise_scores: list[float] = []
    replicate_indices = sorted(label_vectors)
    for left_index, left in enumerate(replicate_indices):
        for right in replicate_indices[left_index + 1 :]:
            ari = adjusted_rand_index(label_vectors[left], label_vectors[right])
            pairwise_rows.append(
                {
                    "replicate_i": int(left),
                    "replicate_j": int(right),
                    "ari": float(ari),
                }
            )
            pairwise_scores.append(float(ari))

    metrics_frame = pd.DataFrame(replicate_metrics).sort_values("replicate_index").reset_index(drop=True)
    metrics_path = combo_dir / "replicate_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    pairwise_path = combo_dir / "replicate_pairwise_ari.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

    refresh_metrics = metrics_frame.loc[metrics_frame["replicate_index"] > 0].copy()
    metadata_enabled = bool(combo_spec["joint_metadata_model"])
    effective_exclude_weight_from_fit = bool(combo_spec["exclude_weight_from_fit"]) or metadata_enabled
    combo_summary = {
        "combo_label": combo_spec["label"],
        "short_label": combo_spec["short_label"],
        "combo_index": int(combo_spec["index"]),
        "requested_joint_metadata_model": bool(combo_spec["joint_metadata_model"]),
        "requested_edge_covariates": bool(combo_spec["edge_covariates"]),
        "requested_exclude_weight_from_fit": bool(combo_spec["exclude_weight_from_fit"]),
        "requested_layered": bool(combo_spec["layered"]),
        "effective_joint_metadata_model": metadata_enabled,
        "effective_fit_covariates": list(manifest.get("fit_covariates", [])),
        "effective_exclude_weight_from_fit": effective_exclude_weight_from_fit,
        "fit_run_dir": str(fit_run_dir),
        "metrics_path": str(metrics_path),
        "pairwise_ari_path": str(pairwise_path),
        "fit_block_count": int(metrics_frame.loc[metrics_frame["replicate_index"] == 0, "block_count"].iloc[0]),
        "block_count_mean": float(metrics_frame["block_count"].mean()),
        "block_count_std": float(metrics_frame["block_count"].std(ddof=0)),
        "block_count_min": int(metrics_frame["block_count"].min()),
        "block_count_max": int(metrics_frame["block_count"].max()),
        "refresh_block_count_mean": float(refresh_metrics["block_count"].mean()) if not refresh_metrics.empty else float("nan"),
        "refresh_block_count_std": float(refresh_metrics["block_count"].std(ddof=0)) if not refresh_metrics.empty else float("nan"),
        "refresh_ari_to_fit_mean": float(refresh_metrics["ari_to_fit"].mean()) if not refresh_metrics.empty else float("nan"),
        "refresh_ari_to_fit_std": float(refresh_metrics["ari_to_fit"].std(ddof=0)) if not refresh_metrics.empty else float("nan"),
        "pairwise_ari_mean": float(np.mean(pairwise_scores)) if pairwise_scores else float("nan"),
        "pairwise_ari_std": float(np.std(pairwise_scores, ddof=0)) if pairwise_scores else float("nan"),
        "mixed_type_block_count_mean": float(metrics_frame["mixed_type_block_count"].mean()),
        "pure_farm_block_count_mean": float(metrics_frame["pure_farm_block_count"].mean()),
        "pure_region_block_count_mean": float(metrics_frame["pure_region_block_count"].mean()),
        "largest_block_fraction_mean": float(metrics_frame["largest_block_fraction"].mean()),
        "total_replicates": int(total_replicates),
        "refresh_options": run_manifest["refresh_options"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(combo_dir / "combo_summary.json", combo_summary)
    LOGGER.info(
        "Finished combo %s | fit_blocks=%s | mean_blocks=%.3f | refresh_ari_to_fit_mean=%.3f",
        combo_spec["label"],
        combo_summary["fit_block_count"],
        combo_summary["block_count_mean"],
        combo_summary["refresh_ari_to_fit_mean"],
    )
    return 0


def _read_combo_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return payload


def _main_effects_table(summary_frame: pd.DataFrame, factor_column: str) -> pd.DataFrame:
    grouped = (
        summary_frame.groupby(factor_column, dropna=False)
        .agg(
            combo_count=("combo_label", "count"),
            fit_block_count_mean=("fit_block_count", "mean"),
            block_count_mean=("block_count_mean", "mean"),
            block_count_std_mean=("block_count_std", "mean"),
            refresh_ari_to_fit_mean=("refresh_ari_to_fit_mean", "mean"),
            pairwise_ari_mean=("pairwise_ari_mean", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "factor", factor_column)
    return grouped


def _plot_boxplot(summary_frame: pd.DataFrame, replicate_frame: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_frame.sort_values(
        [
            "requested_joint_metadata_model",
            "requested_edge_covariates",
            "requested_exclude_weight_from_fit",
            "requested_layered",
        ]
    )
    labels = ordered["short_label"].tolist()
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    series = []
    for combo_label in ordered["combo_label"]:
        values = replicate_frame.loc[replicate_frame["combo_label"] == combo_label, "block_count"].to_numpy(dtype=float)
        series.append(values)
    ax.boxplot(series, positions=positions, widths=0.6, patch_artist=True)
    for position, combo_label in zip(positions, ordered["combo_label"]):
        values = replicate_frame.loc[replicate_frame["combo_label"] == combo_label, "block_count"].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(position + jitter, values, s=25, alpha=0.8, color="#1f77b4")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Data-node block count")
    ax.set_title("Block count distribution across posterior partition refreshes")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap_facets(
    summary_frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    fmt: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    cmap = plt.cm.viridis
    valid_values = summary_frame[value_column].dropna().to_numpy(dtype=float)
    vmin = float(np.min(valid_values)) if valid_values.size else 0.0
    vmax = float(np.max(valid_values)) if valid_values.size else 1.0
    image = None
    for row_index, exclude_weight_from_fit in enumerate([False, True]):
        for col_index, layered in enumerate([False, True]):
            ax = axes[row_index, col_index]
            panel = summary_frame[
                (summary_frame["requested_exclude_weight_from_fit"] == exclude_weight_from_fit)
                & (summary_frame["requested_layered"] == layered)
            ]
            matrix = np.full((2, 2), np.nan, dtype=float)
            for metadata in [False, True]:
                for edge_covariates in [False, True]:
                    matched = panel[
                        (panel["requested_joint_metadata_model"] == metadata)
                        & (panel["requested_edge_covariates"] == edge_covariates)
                    ]
                    if not matched.empty:
                        matrix[int(metadata), int(edge_covariates)] = float(matched.iloc[0][value_column])
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["cov off", "cov on"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["meta off", "meta on"])
            ax.set_title(f"exclude_weight_from_fit={int(exclude_weight_from_fit)} | layered={int(layered)}")
            for meta_index in [0, 1]:
                for cov_index in [0, 1]:
                    value = matrix[meta_index, cov_index]
                    text = "NA" if math.isnan(value) else format(value, fmt)
                    ax.text(cov_index, meta_index, text, ha="center", va="center", color="white", fontsize=11)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.85)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_html_report(
    *,
    output_path: Path,
    run_manifest: dict[str, Any],
    summary_frame: pd.DataFrame,
    main_effects_frame: pd.DataFrame,
    effective_effects_frame: pd.DataFrame,
    plot_paths: dict[str, Path],
    missing_labels: list[str],
) -> None:
    summary_columns = [
        "short_label",
        "requested_joint_metadata_model",
        "requested_edge_covariates",
        "requested_exclude_weight_from_fit",
        "requested_layered",
        "effective_exclude_weight_from_fit",
        "fit_block_count",
        "block_count_mean",
        "block_count_std",
        "refresh_ari_to_fit_mean",
        "pairwise_ari_mean",
        "mixed_type_block_count_mean",
    ]
    summary_html = summary_frame[summary_columns].sort_values("short_label").to_html(index=False, float_format=lambda x: f"{x:.3f}")
    main_effects_html = main_effects_frame.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    effective_effects_html = effective_effects_frame.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    if missing_labels:
        missing_items = "".join(f"<li><code>{label}</code></li>" for label in sorted(missing_labels))
        missing_html = f"""
  <h2>Missing Combinations</h2>
  <p>
    This report is provisional. The following combinations were still running or missing outputs when the summary was written.
    The tables and figures below include only the completed combinations.
  </p>
  <ul>
    {missing_items}
  </ul>
"""
    else:
        missing_html = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Partition Factor Grid Refresh Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    p {{ max-width: 1100px; line-height: 1.5; }}
    table {{ border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 12px 0 24px 0; }}
    code {{ background: #f4f4f4; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Partition Factor Grid Refresh Report</h1>
  <p>
    Slice: <code>{run_manifest['ts_start']}</code> to <code>{run_manifest['ts_end']}</code>,
    days={run_manifest['days']}, total replicates={run_manifest['total_replicates']}.
    Each combination includes the fitted partition and {run_manifest['total_replicates'] - 1} posterior partition refreshes.
  </p>
  <p>
    One code path matters when reading the tables: with <code>joint_metadata_model=true</code>,
    the current pipeline forces edge weights out of SBM inference because the metadata layer is unweighted.
    The report keeps both the requested flags and the effective weight-fit flag visible.
  </p>
  {missing_html}
  <h2>Combination Summary</h2>
  {summary_html}
  <h2>Requested-Factor Main Effects</h2>
  {main_effects_html}
  <h2>Effective Exclude-Weight Main Effects</h2>
  {effective_effects_html}
  <h2>Figures</h2>
  <img src="{plot_paths['boxplot'].name}" alt="Block count boxplot">
  <img src="{plot_paths['mean_heatmap'].name}" alt="Mean block count heatmaps">
  <img src="{plot_paths['stability_heatmap'].name}" alt="Refresh stability heatmaps">
</body>
</html>
"""
    output_path.write_text(html)


def summarize_command(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    run_manifest = _load_json(output_root / "run_manifest.json")
    combo_specs = _load_json(output_root / "combo_specs.json")

    combo_summaries = []
    replicate_frames = []
    missing_labels = []
    for combo_spec in combo_specs:
        combo_dir = Path(combo_spec["combo_dir"]).expanduser().resolve()
        summary_path = combo_dir / "combo_summary.json"
        metrics_path = combo_dir / "replicate_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            if args.allow_incomplete:
                missing_labels.append(combo_spec["label"])
                continue
            raise FileNotFoundError(f"Missing combo summary or metrics for {combo_spec['label']}")
        combo_summary = _read_combo_summary(summary_path)
        combo_summaries.append(combo_summary)
        metrics_frame = pd.read_csv(metrics_path)
        metrics_frame["combo_label"] = combo_summary["combo_label"]
        metrics_frame["short_label"] = combo_summary["short_label"]
        metrics_frame["requested_joint_metadata_model"] = combo_summary["requested_joint_metadata_model"]
        metrics_frame["requested_edge_covariates"] = combo_summary["requested_edge_covariates"]
        metrics_frame["requested_exclude_weight_from_fit"] = combo_summary["requested_exclude_weight_from_fit"]
        metrics_frame["requested_layered"] = combo_summary["requested_layered"]
        replicate_frames.append(metrics_frame)

    if not combo_summaries:
        raise FileNotFoundError("No completed combination summaries were found.")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_frame = pd.DataFrame(combo_summaries).sort_values(
        [
            "requested_joint_metadata_model",
            "requested_edge_covariates",
            "requested_exclude_weight_from_fit",
            "requested_layered",
        ]
    ).reset_index(drop=True)
    replicate_frame = pd.concat(replicate_frames, ignore_index=True)

    summary_csv_path = summary_dir / "combination_summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)
    replicate_csv_path = summary_dir / "replicate_metrics.csv"
    replicate_frame.to_csv(replicate_csv_path, index=False)
    missing_csv_path = summary_dir / "missing_combinations.csv"
    pd.DataFrame({"combo_label": sorted(missing_labels)}).to_csv(missing_csv_path, index=False)

    main_effects_frames = []
    for factor in [
        "requested_joint_metadata_model",
        "requested_edge_covariates",
        "requested_exclude_weight_from_fit",
        "requested_layered",
    ]:
        main_effects_frames.append(_main_effects_table(summary_frame, factor))
    main_effects_frame = pd.concat(main_effects_frames, ignore_index=True)
    main_effects_path = summary_dir / "requested_main_effects.csv"
    main_effects_frame.to_csv(main_effects_path, index=False)

    effective_effects_frame = _main_effects_table(summary_frame, "effective_exclude_weight_from_fit")
    effective_effects_path = summary_dir / "effective_main_effects.csv"
    effective_effects_frame.to_csv(effective_effects_path, index=False)

    boxplot_path = summary_dir / "block_count_by_combination.png"
    _plot_boxplot(summary_frame, replicate_frame, boxplot_path)

    mean_heatmap_path = summary_dir / "mean_block_count_heatmaps.png"
    _plot_heatmap_facets(
        summary_frame,
        value_column="block_count_mean",
        title="Mean block count across requested factor combinations",
        output_path=mean_heatmap_path,
        fmt=".2f",
    )

    stability_heatmap_path = summary_dir / "refresh_stability_heatmaps.png"
    _plot_heatmap_facets(
        summary_frame,
        value_column="refresh_ari_to_fit_mean",
        title="Mean ARI to fitted partition across refreshes",
        output_path=stability_heatmap_path,
        fmt=".3f",
    )

    report_path = summary_dir / "partition_factor_grid_report.html"
    _write_html_report(
        output_path=report_path,
        run_manifest=run_manifest,
        summary_frame=summary_frame,
        main_effects_frame=main_effects_frame,
        effective_effects_frame=effective_effects_frame,
        plot_paths={
            "boxplot": boxplot_path,
            "mean_heatmap": mean_heatmap_path,
            "stability_heatmap": stability_heatmap_path,
        },
        missing_labels=missing_labels,
    )

    LOGGER.info("Summary written to %s", summary_dir)
    print(str(report_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and summarize a partition factor grid with posterior refresh replicates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Create a run manifest and per-combination configs.")
    plan_parser.add_argument("--base-config", required=True, help="Base JSON config used to seed the grid.")
    plan_parser.add_argument("--output-root", default=None, help="Output root for the batch run.")
    plan_parser.add_argument("--days", type=int, required=True, help="Inclusive slice length in days.")
    plan_parser.add_argument("--replicates", type=int, required=True, help="Total partitions per combination, including the fitted state.")
    plan_parser.add_argument("--start-ts", type=int, default=None, help="Override the slice start timestamp.")
    plan_parser.add_argument("--seed", type=int, default=None, help="Base seed for posterior refresh draws.")
    plan_parser.add_argument("--posterior-partition-sweeps", type=int, default=None, help="Refresh sweep count.")
    plan_parser.add_argument("--posterior-partition-sweep-niter", type=int, default=None, help="niter for each refresh sweep.")
    plan_parser.add_argument("--posterior-partition-beta", type=float, default=None, help="Refresh beta.")

    combo_parser = subparsers.add_parser("run-combo", help="Fit one factor combination and draw refresh replicates.")
    combo_parser.add_argument("--output-root", required=True, help="Batch run root created by the plan step.")
    combo_parser.add_argument("--combo", required=True, help="Combination label from combo_labels.txt.")

    summary_parser = subparsers.add_parser("summarize", help="Aggregate combination outputs into tables and figures.")
    summary_parser.add_argument("--output-root", required=True, help="Batch run root created by the plan step.")
    summary_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a provisional summary from the completed combinations and record the missing ones.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        _configure_logging()
        return plan_command(args)
    if args.command == "run-combo":
        return run_combo_command(args)
    if args.command == "summarize":
        _configure_logging()
        return summarize_command(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
