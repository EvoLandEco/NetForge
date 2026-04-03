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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_sbm.sweep import run_sweep_command


LOGGER = logging.getLogger("partition_factor_full_workflow")
EDGE_COVARIATES = ["dist_km", "mass_grav", "anim_grav"]
BEST_SCENARIO_SORT = [
    ("farm_prevalence_mean_curve_correlation", False),
    ("farm_incidence_mean_curve_correlation", False),
    ("farm_prevalence_curve_correlation", False),
    ("farm_incidence_curve_correlation", False),
    ("mean_abs_farm_prevalence_delta", True),
    ("mean_abs_farm_incidence_delta", True),
]

DIAGNOSTIC_METRICS = [
    ("unique_edge_jaccard_mean", "Unique-edge Jaccard", ".3f", "cividis"),
    ("mean_snapshot_edge_jaccard_mean", "Mean daily edge Jaccard", ".3f", "cividis"),
    ("reciprocity_correlation_mean", "Reciprocity correlation", ".3f", "cividis"),
    ("magnetic_spectrum_pooled_correlation", "Magnetic spectrum correlation", ".3f", "cividis"),
]
WEIGHT_METRICS = [
    ("weight_total_correlation_mean", "Total weight correlation", ".3f", "cividis"),
    ("edge_type_weight_share_correlation_mean", "Edge-type weight-share correlation", ".3f", "cividis"),
    ("mean_abs_weight_total_delta_mean", "Mean abs daily total-weight delta", ".1f", "magma_r"),
]
BEST_SCENARIO_METRICS = [
    ("best_farm_prevalence_mean_curve_correlation", "Best scenario prevalence mean corr.", ".3f", "cividis"),
    ("best_farm_incidence_mean_curve_correlation", "Best scenario incidence mean corr.", ".3f", "cividis"),
    ("best_farm_prevalence_curve_correlation", "Best scenario prevalence median corr.", ".3f", "cividis"),
    ("best_farm_incidence_curve_correlation", "Best scenario incidence median corr.", ".3f", "cividis"),
]
SCENARIO_EFFECT_METRICS = [
    "farm_prevalence_curve_correlation",
    "farm_incidence_curve_correlation",
    "farm_prevalence_mean_curve_correlation",
    "farm_incidence_mean_curve_correlation",
    "farm_attack_probability_correlation",
    "region_reservoir_spatial_correlation_mean",
]


def _combo_label(
    *,
    joint_metadata_model: bool,
    edge_covariates: bool,
    layered: bool,
) -> str:
    return (
        f"meta_{'on' if joint_metadata_model else 'off'}"
        f"__cov_{'on' if edge_covariates else 'off'}"
        f"__layered_{'on' if layered else 'off'}"
    )


def _combo_short_label(
    *,
    joint_metadata_model: bool,
    edge_covariates: bool,
    layered: bool,
) -> str:
    return (
        f"M{int(joint_metadata_model)} "
        f"C{int(edge_covariates)} "
        f"L{int(layered)}"
    )


def _combo_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    index = 0
    for joint_metadata_model in (False, True):
        for edge_covariates in (False, True):
            for layered in (False, True):
                specs.append(
                    {
                        "index": index,
                        "label": _combo_label(
                            joint_metadata_model=joint_metadata_model,
                            edge_covariates=edge_covariates,
                            layered=layered,
                        ),
                        "short_label": _combo_short_label(
                            joint_metadata_model=joint_metadata_model,
                            edge_covariates=edge_covariates,
                            layered=layered,
                        ),
                        "joint_metadata_model": bool(joint_metadata_model),
                        "edge_covariates": bool(edge_covariates),
                        "exclude_weight_from_fit": not bool(edge_covariates),
                        "layered": bool(layered),
                        "forced_weight_model": "discrete-geometric" if edge_covariates else None,
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


def _default_output_root(days: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(".stress_runs") / f"cr35_partition_factor_full_workflow_{days}d_{stamp}"


def _settings_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "settings"


def _fit_values_for_combo(
    *,
    base_fit: dict[str, Any],
    combo_spec: dict[str, Any],
    run_dir: Path,
    ts_start: int,
    ts_end: int,
) -> dict[str, Any]:
    fit_values = dict(base_fit)
    fit_values["output_dir"] = str(run_dir.resolve())
    fit_values["ts_start"] = int(ts_start)
    fit_values["ts_end"] = int(ts_end)
    fit_values["joint_metadata_model"] = bool(combo_spec["joint_metadata_model"])
    fit_values["fit_covariates"] = list(EDGE_COVARIATES) if combo_spec["edge_covariates"] else ["none"]
    fit_values["exclude_weight_from_fit"] = bool(combo_spec["exclude_weight_from_fit"])
    fit_values["layered"] = bool(combo_spec["layered"])
    fit_values["fit_quiet"] = True
    if combo_spec.get("forced_weight_model"):
        fit_values["weight_model"] = str(combo_spec["forced_weight_model"])
    return fit_values


def _complete_report_path(run_dir: Path) -> Path:
    return run_dir / "diagnostics" / "scientific_validation_report.html"


def _best_setting_path(run_dir: Path) -> Path:
    return run_dir / "diagnostics" / "best_primary_setting.txt"


def _scenario_summary_path(run_dir: Path, best_setting: str) -> Path:
    return run_dir / "simulation_scenarios" / best_setting / "scenario_summary.csv"


def _scenario_report_path(run_dir: Path, best_setting: str) -> Path:
    return run_dir / "simulation_scenarios" / best_setting / "scientific_validation_report.html"


def _combo_is_complete(run_dir: Path) -> bool:
    diagnostics_summary = run_dir / "diagnostics" / "setting_posterior_summary.csv"
    diagnostics_report = _complete_report_path(run_dir)
    best_setting_file = _best_setting_path(run_dir)
    if not diagnostics_summary.exists() or not diagnostics_report.exists() or not best_setting_file.exists():
        return False
    best_setting = best_setting_file.read_text().strip()
    if not best_setting:
        return False
    return _scenario_summary_path(run_dir, best_setting).exists() and _scenario_report_path(run_dir, best_setting).exists()


def plan_command(args: argparse.Namespace) -> int:
    settings_root = _settings_root_from_args(args.settings_root)
    settings_root.mkdir(parents=True, exist_ok=True)
    base_payload = _load_json(Path(args.base_config).expanduser().resolve())
    if "fit" not in base_payload:
        raise ValueError("Base config must contain a fit object.")

    base_fit = dict(base_payload["fit"])
    base_generate = dict(base_payload.get("generate") or {})
    base_grid = dict(base_payload.get("grid") or {"samplers": ["maxent_micro"], "rewires": ["none"]})
    base_report = dict(base_payload.get("report") or {})
    base_simulation = dict(base_payload.get("simulation") or {})

    ts_start = int(args.start_ts if args.start_ts is not None else base_fit.get("ts_start"))
    min_ts, max_ts = _dataset_ts_bounds(base_fit)
    ts_end = ts_start + int(args.days) - 1
    if ts_start < min_ts or ts_end > max_ts:
        raise ValueError(f"Requested slice [{ts_start}, {ts_end}] falls outside dataset bounds [{min_ts}, {max_ts}].")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else _default_output_root(int(args.days)).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.num_samples is not None:
        base_generate["num_samples"] = int(args.num_samples)
    if args.simulation_replicates is not None:
        scenario_specs = []
        for scenario in list(base_simulation.get("scenarios") or []):
            updated = dict(scenario)
            updated["num_replicates"] = int(args.simulation_replicates)
            scenario_specs.append(updated)
        base_simulation["scenarios"] = scenario_specs

    combo_specs = []
    for combo in _combo_specs():
        run_dir = output_root / "combos" / combo["label"]
        fit_values = _fit_values_for_combo(
            base_fit=base_fit,
            combo_spec=combo,
            run_dir=run_dir,
            ts_start=ts_start,
            ts_end=ts_end,
        )
        config_payload = {
            "fit": fit_values,
            "generate": base_generate,
            "grid": base_grid,
            "report": base_report,
            "simulation": base_simulation,
        }
        config_path = settings_root / "combos" / f"{combo['label']}.json"
        _save_json(config_path, config_payload)
        combo_specs.append(
            {
                **combo,
                "run_dir": str(run_dir.resolve()),
                "config_path": str(config_path.resolve()),
            }
        )

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(Path(args.base_config).expanduser().resolve()),
        "settings_root": str(settings_root.resolve()),
        "output_root": str(output_root.resolve()),
        "days": int(args.days),
        "ts_start": int(ts_start),
        "ts_end": int(ts_end),
        "dataset_bounds": {"ts_min": int(min_ts), "ts_max": int(max_ts)},
        "combo_count": len(combo_specs),
        "grid": base_grid,
        "generate": base_generate,
        "report": base_report,
        "simulation": base_simulation,
        "forced_weight_model_labels": [spec["label"] for spec in combo_specs if spec.get("forced_weight_model")],
    }

    _save_json(settings_root / "base_config.json", base_payload)
    _save_json(settings_root / "run_manifest.json", run_manifest)
    _save_json(settings_root / "combo_specs.json", combo_specs)
    (settings_root / "combo_labels.txt").write_text("\n".join(spec["label"] for spec in combo_specs) + "\n")
    print(str(output_root))
    return 0


def run_combo_command(args: argparse.Namespace) -> int:
    settings_root = _settings_root_from_args(args.settings_root)
    combo_specs = _load_json(settings_root / "combo_specs.json")
    combo_spec = next((spec for spec in combo_specs if spec["label"] == args.combo), None)
    if combo_spec is None:
        raise ValueError(f"Unknown combo label: {args.combo}")

    run_dir = Path(combo_spec["run_dir"]).expanduser().resolve()
    _configure_logging(run_dir / "logs" / "full_workflow_batch.log")

    if _combo_is_complete(run_dir):
        LOGGER.info("Skipping %s; full workflow artifacts are already present under %s", combo_spec["label"], run_dir)
        return 0

    sweep_args = argparse.Namespace(config=str(combo_spec["config_path"]), verbose=bool(args.verbose))
    LOGGER.info("Running full workflow | combo=%s | run_dir=%s", combo_spec["label"], run_dir)
    run_sweep_command(sweep_args)
    LOGGER.info("Finished full workflow | combo=%s", combo_spec["label"])
    return 0


def _float_value(row: pd.Series, column: str) -> float:
    if column not in row.index or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def _pick_best_scenario(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Scenario summary is empty.")
    sort_columns = [column for column, _ in BEST_SCENARIO_SORT if column in frame.columns]
    if not sort_columns:
        return frame.iloc[0]
    ascending = [ascending for column, ascending in BEST_SCENARIO_SORT if column in frame.columns]
    return frame.sort_values(sort_columns, ascending=ascending).iloc[0]


def _main_effects_table(frame: pd.DataFrame, factor_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in factor_columns:
        for level, group in frame.groupby(factor, dropna=False):
            for metric in metric_columns:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "factor": factor,
                        "level": level,
                        "metric": metric,
                        "combo_count": int(len(values)),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def _scenario_main_effects_table(frame: pd.DataFrame, factor_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_group in frame.groupby("scenario_name"):
        for factor in factor_columns:
            for level, group in scenario_group.groupby(factor, dropna=False):
                for metric in metric_columns:
                    if metric not in group.columns:
                        continue
                    values = pd.to_numeric(group[metric], errors="coerce").dropna()
                    if values.empty:
                        continue
                    rows.append(
                        {
                            "scenario_name": scenario_name,
                            "factor": factor,
                            "level": level,
                            "metric": metric,
                            "combo_count": int(len(values)),
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=0)),
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }
                    )
    return pd.DataFrame(rows)


def _heatmap_matrix(frame: pd.DataFrame, metric: str, layered: bool) -> np.ndarray:
    matrix = np.full((2, 2), np.nan, dtype=float)
    panel = frame[frame["requested_layered"] == layered]
    for metadata in (False, True):
        for edge_covariates in (False, True):
            matched = panel[
                (panel["requested_joint_metadata_model"] == metadata)
                & (panel["requested_edge_covariates"] == edge_covariates)
            ]
            if not matched.empty and metric in matched.columns:
                matrix[int(metadata), int(edge_covariates)] = float(matched.iloc[0][metric])
    return matrix


def _annotation_color(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "black"
    if abs(vmax - vmin) < 1e-12:
        return "white"
    midpoint = vmin + 0.55 * (vmax - vmin)
    return "white" if value >= midpoint else "black"


def _plot_metric_facets(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, str, str]],
    output_path: Path,
    title: str,
) -> None:
    panel_keys = [False, True]
    fig, axes = plt.subplots(len(metrics), len(panel_keys), figsize=(10.5, 3.9 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for row_index, (metric, metric_title, fmt, cmap) in enumerate(metrics):
        valid = pd.to_numeric(frame.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
        vmin = float(np.min(valid)) if valid.size else 0.0
        vmax = float(np.max(valid)) if valid.size else 1.0
        image = None
        for col_index, layered in enumerate(panel_keys):
            ax = axes[row_index, col_index]
            matrix = _heatmap_matrix(frame, metric, layered)
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["cov off", "cov on"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["meta off", "meta on"])
            ax.set_title(f"{metric_title}\nlayered={int(layered)}")
            for meta_index in (0, 1):
                for cov_index in (0, 1):
                    value = matrix[meta_index, cov_index]
                    text = "NA" if math.isnan(value) else format(value, fmt)
                    ax.text(
                        cov_index,
                        meta_index,
                        text,
                        ha="center",
                        va="center",
                        color=_annotation_color(value, vmin, vmax),
                        fontsize=10,
                    )
        if image is not None:
            fig.colorbar(image, ax=axes[row_index, :], shrink=0.88)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scenario_metric_facets(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    fmt: str = ".3f",
) -> None:
    scenario_names = sorted(frame["scenario_name"].dropna().astype(str).unique().tolist())
    panel_keys = [False, True]
    fig, axes = plt.subplots(len(scenario_names), len(panel_keys), figsize=(10.5, 3.9 * max(len(scenario_names), 1)), constrained_layout=True)
    if len(scenario_names) == 1:
        axes = np.array([axes])
    valid = pd.to_numeric(frame.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
    vmin = float(np.min(valid)) if valid.size else 0.0
    vmax = float(np.max(valid)) if valid.size else 1.0
    for row_index, scenario_name in enumerate(scenario_names):
        scenario_frame = frame[frame["scenario_name"] == scenario_name]
        image = None
        for col_index, layered in enumerate(panel_keys):
            ax = axes[row_index, col_index]
            matrix = _heatmap_matrix(scenario_frame, metric, layered)
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="cividis")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["cov off", "cov on"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["meta off", "meta on"])
            ax.set_title(f"{scenario_name}\nlayered={int(layered)}")
            for meta_index in (0, 1):
                for cov_index in (0, 1):
                    value = matrix[meta_index, cov_index]
                    text = "NA" if math.isnan(value) else format(value, fmt)
                    ax.text(
                        cov_index,
                        meta_index,
                        text,
                        ha="center",
                        va="center",
                        color=_annotation_color(value, vmin, vmax),
                        fontsize=10,
                    )
        if image is not None:
            fig.colorbar(image, ax=axes[row_index, :], shrink=0.88)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_scenario_counts(frame: pd.DataFrame, output_path: Path) -> None:
    counts = frame["best_scenario_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(counts.index.tolist(), counts.to_numpy(dtype=float), color=["#33658A", "#86BBD8"][: len(counts)])
    ax.set_ylabel("Combo count")
    ax.set_title("Best scenario counts across factor combinations")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_html_report(
    *,
    output_path: Path,
    run_manifest: dict[str, Any],
    combo_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    diagnostic_effects: pd.DataFrame,
    simulation_effects: pd.DataFrame,
    best_scenario_counts: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    combo_columns = [
        "short_label",
        "requested_joint_metadata_model",
        "requested_edge_covariates",
        "requested_layered",
        "forced_weight_model_override",
        "best_setting_label",
        "unique_edge_jaccard_mean",
        "mean_snapshot_edge_jaccard_mean",
        "weight_total_correlation_mean",
        "edge_type_weight_share_correlation_mean",
        "best_scenario_name",
        "best_farm_prevalence_mean_curve_correlation",
        "best_farm_incidence_mean_curve_correlation",
    ]
    combo_html = combo_summary[combo_columns].sort_values("short_label").to_html(index=False, float_format=lambda x: f"{x:.3f}")
    scenario_columns = [
        "short_label",
        "scenario_name",
        "farm_prevalence_curve_correlation",
        "farm_incidence_curve_correlation",
        "farm_prevalence_mean_curve_correlation",
        "farm_incidence_mean_curve_correlation",
        "farm_attack_probability_correlation",
        "region_reservoir_spatial_correlation_mean",
        "is_best_scenario",
    ]
    scenario_html = scenario_summary[scenario_columns].sort_values(["short_label", "scenario_name"]).to_html(index=False, float_format=lambda x: f"{x:.3f}")
    diagnostic_effects_html = diagnostic_effects.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    simulation_effects_html = simulation_effects.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    counts_html = best_scenario_counts.to_html(index=False)

    forced_labels = run_manifest.get("forced_weight_model_labels") or []
    if forced_labels:
        forced_html = "<ul>" + "".join(f"<li><code>{label}</code></li>" for label in forced_labels) + "</ul>"
    else:
        forced_html = "<p>None.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Partition Factor Full Workflow Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    p {{ max-width: 1100px; line-height: 1.5; }}
    table {{ border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 12px 0 24px 0; }}
    code {{ background: #f4f4f4; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Partition Factor Full Workflow Report</h1>
  <p>
    Slice: <code>{run_manifest['ts_start']}</code> to <code>{run_manifest['ts_end']}</code>,
    days={run_manifest['days']}, combos={run_manifest['combo_count']}.
    Each combo ran the full sweep workflow: fit, generation, diagnostics, best-setting selection, and simulation scenarios.
  </p>
  <p>
    This design uses three switches: <code>joint_metadata_model</code>, <code>edge_covariates</code>, and <code>layered</code>.
    With <code>cov on</code>, the fit uses <code>dist_km</code>, <code>mass_grav</code>, <code>anim_grav</code>, and the edge-weight covariate together.
    With <code>cov off</code>, all four stay out of the SBM.
  </p>
  <h2>Forced geometric runs</h2>
  <p>
    The weighted cells use <code>weight_model=discrete-geometric</code> because the Poisson candidate stalled on this 280 day slice.
  </p>
  {forced_html}
  <h2>Combo summary</h2>
  {combo_html}
  <h2>Scenario summary</h2>
  {scenario_html}
  <h2>Diagnostic main effects</h2>
  {diagnostic_effects_html}
  <h2>Simulation main effects</h2>
  {simulation_effects_html}
  <h2>Best scenario counts</h2>
  {counts_html}
  <h2>Figures</h2>
  <img src="{plot_paths['diagnostic_topology'].name}" alt="Topology diagnostics heatmaps">
  <img src="{plot_paths['diagnostic_weight'].name}" alt="Weight diagnostics heatmaps">
  <img src="{plot_paths['simulation_best'].name}" alt="Best scenario heatmaps">
  <img src="{plot_paths['simulation_prevalence'].name}" alt="Scenario prevalence heatmaps">
  <img src="{plot_paths['simulation_incidence'].name}" alt="Scenario incidence heatmaps">
  <img src="{plot_paths['best_scenario_counts'].name}" alt="Best scenario counts">
</body>
</html>
"""
    output_path.write_text(html)


def summarize_command(args: argparse.Namespace) -> int:
    settings_root = _settings_root_from_args(args.settings_root)
    run_manifest = _load_json(settings_root / "run_manifest.json")
    combo_specs = _load_json(settings_root / "combo_specs.json")

    combo_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for spec in combo_specs:
        run_dir = Path(spec["run_dir"]).expanduser().resolve()
        diagnostics_summary_path = run_dir / "diagnostics" / "setting_posterior_summary.csv"
        if not diagnostics_summary_path.exists():
            raise FileNotFoundError(f"Missing diagnostics summary for {spec['label']}: {diagnostics_summary_path}")
        best_setting_label = _best_setting_path(run_dir).read_text().strip()
        if not best_setting_label:
            raise ValueError(f"Best setting is empty for {spec['label']}")

        diagnostics_frame = pd.read_csv(diagnostics_summary_path)
        matched = diagnostics_frame[diagnostics_frame["sample_label"].astype(str) == best_setting_label]
        diagnostics_row = matched.iloc[0] if not matched.empty else diagnostics_frame.iloc[0]

        scenario_summary_path = _scenario_summary_path(run_dir, best_setting_label)
        if not scenario_summary_path.exists():
            raise FileNotFoundError(f"Missing scenario summary for {spec['label']}: {scenario_summary_path}")
        scenario_frame = pd.read_csv(scenario_summary_path)
        best_scenario_row = _pick_best_scenario(scenario_frame)
        manifest_payload = _load_json(run_dir / "manifest.json")
        manifest_fit_options = manifest_payload.get("fit_options") or {}
        effective_exclude_weight_from_fit = bool(
            manifest_fit_options.get("exclude_weight_from_fit", spec["exclude_weight_from_fit"])
        )

        for _, scenario_row in scenario_frame.iterrows():
            scenario_rows.append(
                {
                    "combo_label": spec["label"],
                    "short_label": spec["short_label"],
                    "requested_joint_metadata_model": bool(spec["joint_metadata_model"]),
                    "requested_edge_covariates": bool(spec["edge_covariates"]),
                    "requested_layered": bool(spec["layered"]),
                    "effective_exclude_weight_from_fit": effective_exclude_weight_from_fit,
                    "forced_weight_model_override": str(spec.get("forced_weight_model") or ""),
                    "best_setting_label": best_setting_label,
                    "run_dir": str(run_dir),
                    "scenario_name": str(scenario_row.get("scenario_name")),
                    "scenario_description": str(scenario_row.get("scenario_description", "")),
                    "report_path": str(scenario_row.get("report_path", "")),
                    "farm_prevalence_curve_correlation": _float_value(scenario_row, "farm_prevalence_curve_correlation"),
                    "farm_incidence_curve_correlation": _float_value(scenario_row, "farm_incidence_curve_correlation"),
                    "farm_prevalence_mean_curve_correlation": _float_value(scenario_row, "farm_prevalence_mean_curve_correlation"),
                    "farm_incidence_mean_curve_correlation": _float_value(scenario_row, "farm_incidence_mean_curve_correlation"),
                    "mean_abs_farm_prevalence_delta": _float_value(scenario_row, "mean_abs_farm_prevalence_delta"),
                    "mean_abs_farm_incidence_delta": _float_value(scenario_row, "mean_abs_farm_incidence_delta"),
                    "farm_attack_probability_correlation": _float_value(scenario_row, "farm_attack_probability_correlation"),
                    "region_reservoir_spatial_correlation_mean": _float_value(scenario_row, "region_reservoir_spatial_correlation_mean"),
                    "is_best_scenario": str(scenario_row.get("scenario_name")) == str(best_scenario_row.get("scenario_name")),
                }
            )

        combo_rows.append(
            {
                "combo_label": spec["label"],
                "short_label": spec["short_label"],
                "requested_joint_metadata_model": bool(spec["joint_metadata_model"]),
                "requested_edge_covariates": bool(spec["edge_covariates"]),
                "requested_layered": bool(spec["layered"]),
                "effective_exclude_weight_from_fit": effective_exclude_weight_from_fit,
                "forced_weight_model_override": str(spec.get("forced_weight_model") or ""),
                "run_dir": str(run_dir),
                "diagnostics_report_path": str(_complete_report_path(run_dir)),
                "best_setting_label": best_setting_label,
                "simulation_report_path": str(_scenario_report_path(run_dir, best_setting_label)),
                "unique_edge_jaccard_mean": _float_value(diagnostics_row, "unique_edge_jaccard_mean"),
                "mean_snapshot_edge_jaccard_mean": _float_value(diagnostics_row, "mean_snapshot_edge_jaccard_mean"),
                "mean_synthetic_novel_edge_rate_mean": _float_value(diagnostics_row, "mean_synthetic_novel_edge_rate_mean"),
                "mean_snapshot_node_jaccard_mean": _float_value(diagnostics_row, "mean_snapshot_node_jaccard_mean"),
                "reciprocity_correlation_mean": _float_value(diagnostics_row, "reciprocity_correlation_mean"),
                "magnetic_spectrum_pooled_correlation": _float_value(diagnostics_row, "magnetic_spectrum_pooled_correlation"),
                "weight_total_correlation_mean": _float_value(diagnostics_row, "weight_total_correlation_mean"),
                "edge_type_weight_share_correlation_mean": _float_value(diagnostics_row, "edge_type_weight_share_correlation_mean"),
                "mean_abs_weight_total_delta_mean": _float_value(diagnostics_row, "mean_abs_weight_total_delta_mean"),
                "best_scenario_name": str(best_scenario_row.get("scenario_name")),
                "best_farm_prevalence_curve_correlation": _float_value(best_scenario_row, "farm_prevalence_curve_correlation"),
                "best_farm_incidence_curve_correlation": _float_value(best_scenario_row, "farm_incidence_curve_correlation"),
                "best_farm_prevalence_mean_curve_correlation": _float_value(best_scenario_row, "farm_prevalence_mean_curve_correlation"),
                "best_farm_incidence_mean_curve_correlation": _float_value(best_scenario_row, "farm_incidence_mean_curve_correlation"),
                "best_farm_attack_probability_correlation": _float_value(best_scenario_row, "farm_attack_probability_correlation"),
                "best_region_reservoir_spatial_correlation_mean": _float_value(best_scenario_row, "region_reservoir_spatial_correlation_mean"),
            }
        )

    combo_summary = pd.DataFrame(combo_rows).sort_values(
        [
            "requested_joint_metadata_model",
            "requested_edge_covariates",
            "requested_layered",
        ]
    ).reset_index(drop=True)
    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["short_label", "scenario_name"]).reset_index(drop=True)

    output_root = Path(run_manifest["output_root"]).expanduser().resolve()
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    combo_summary.to_csv(summary_dir / "combo_summary.csv", index=False)
    scenario_summary.to_csv(summary_dir / "scenario_summary.csv", index=False)

    factor_columns = [
        "requested_joint_metadata_model",
        "requested_edge_covariates",
        "requested_layered",
    ]
    diagnostic_effects = _main_effects_table(
        combo_summary,
        factor_columns=factor_columns,
        metric_columns=[
            "unique_edge_jaccard_mean",
            "mean_snapshot_edge_jaccard_mean",
            "mean_synthetic_novel_edge_rate_mean",
            "reciprocity_correlation_mean",
            "magnetic_spectrum_pooled_correlation",
            "weight_total_correlation_mean",
            "edge_type_weight_share_correlation_mean",
            "mean_abs_weight_total_delta_mean",
        ],
    )
    diagnostic_effects.to_csv(summary_dir / "diagnostic_main_effects.csv", index=False)

    simulation_effects = _scenario_main_effects_table(
        scenario_summary,
        factor_columns=factor_columns,
        metric_columns=SCENARIO_EFFECT_METRICS,
    )
    simulation_effects.to_csv(summary_dir / "simulation_main_effects.csv", index=False)

    best_scenario_counts = (
        combo_summary["best_scenario_name"]
        .value_counts()
        .sort_index()
        .rename_axis("scenario_name")
        .reset_index(name="combo_count")
    )
    best_scenario_counts.to_csv(summary_dir / "best_scenario_counts.csv", index=False)

    topology_plot = summary_dir / "diagnostic_topology_heatmaps.png"
    weight_plot = summary_dir / "diagnostic_weight_heatmaps.png"
    best_sim_plot = summary_dir / "simulation_best_heatmaps.png"
    prevalence_plot = summary_dir / "simulation_prevalence_scenario_heatmaps.png"
    incidence_plot = summary_dir / "simulation_incidence_scenario_heatmaps.png"
    counts_plot = summary_dir / "best_scenario_counts.png"

    _plot_metric_facets(combo_summary, DIAGNOSTIC_METRICS, topology_plot, "Topology diagnostics by factor combination")
    _plot_metric_facets(combo_summary, WEIGHT_METRICS, weight_plot, "Weight diagnostics by factor combination")
    _plot_metric_facets(combo_summary, BEST_SCENARIO_METRICS, best_sim_plot, "Best-scenario simulation metrics by factor combination")
    _plot_scenario_metric_facets(
        scenario_summary,
        metric="farm_prevalence_mean_curve_correlation",
        title="Scenario prevalence mean correlation by factor combination",
        output_path=prevalence_plot,
    )
    _plot_scenario_metric_facets(
        scenario_summary,
        metric="farm_incidence_mean_curve_correlation",
        title="Scenario incidence mean correlation by factor combination",
        output_path=incidence_plot,
    )
    _plot_best_scenario_counts(combo_summary, counts_plot)

    report_path = summary_dir / "partition_factor_full_workflow_report.html"
    _write_html_report(
        output_path=report_path,
        run_manifest=run_manifest,
        combo_summary=combo_summary,
        scenario_summary=scenario_summary,
        diagnostic_effects=diagnostic_effects,
        simulation_effects=simulation_effects,
        best_scenario_counts=best_scenario_counts,
        plot_paths={
            "diagnostic_topology": topology_plot,
            "diagnostic_weight": weight_plot,
            "simulation_best": best_sim_plot,
            "simulation_prevalence": prevalence_plot,
            "simulation_incidence": incidence_plot,
            "best_scenario_counts": counts_plot,
        },
    )
    LOGGER.info("Summary written to %s", summary_dir)
    print(str(report_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, run, and summarize the partition-factor full-workflow experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write the 8 combo configs and run manifest.")
    plan_parser.add_argument("--base-config", required=True, help="Base sweep config used to seed the experiment.")
    plan_parser.add_argument("--settings-root", default=None, help="Archive folder for generated settings JSON files.")
    plan_parser.add_argument("--output-root", default=None, help="Run root for the combo workflows.")
    plan_parser.add_argument("--days", type=int, required=True, help="Inclusive slice length.")
    plan_parser.add_argument("--start-ts", type=int, default=None, help="Slice start timestamp. Defaults to the base config start.")
    plan_parser.add_argument("--num-samples", type=int, default=None, help="Override generate.num_samples.")
    plan_parser.add_argument("--simulation-replicates", type=int, default=None, help="Override simulation scenario replicate count.")

    combo_parser = subparsers.add_parser("run-combo", help="Run the full sweep workflow for one combo.")
    combo_parser.add_argument("--settings-root", default=None, help="Archive folder that contains combo_specs.json.")
    combo_parser.add_argument("--combo", required=True, help="Combo label from combo_labels.txt.")
    combo_parser.add_argument("--verbose", action="store_true", help="Pass verbose mode through to the sweep command.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize completed combo workflows.")
    summarize_parser.add_argument("--settings-root", default=None, help="Archive folder that contains the run manifest and combo specs.")

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
