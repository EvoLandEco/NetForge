"""NetForge command-line interface."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from temporal_sbm.diagnostics import (
    aggregate_posterior_reports,
    compare_panels_detailed,
    compare_panels,
    load_node_blocks,
    load_node_coordinates,
    load_node_types,
    summary_payload_to_row,
    write_scientific_validation_report,
    write_all_samples_overview,
    write_log_visual_summary,
    write_report,
)
from temporal_sbm.pipeline import (
    DEFAULT_COVARIATES,
    fit_command,
    generate_command,
    load_json,
    load_manifest,
    load_node_block_map_from_graph_path,
    save_json,
)


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool, log_path: Optional[Path] = None) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    package_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("temporal_sbm").setLevel(package_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def _default_output_dir(data_root: str, dataset: str) -> Path:
    return Path(data_root).expanduser().resolve() / dataset / "graph_tool_out" / "netforge"


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, help="Root directory containing dataset folders.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under --data-root.")
    parser.add_argument("--edges-csv", default=None, help="Use this edge CSV instead of the default dataset file.")
    parser.add_argument(
        "--weight-npy",
        default=None,
        help=(
            "Path to a .npy vector of raw additive edge weights aligned to the input edge rows. "
            "Supports a leading padding entry."
        ),
    )
    parser.add_argument("--node-features-npy", default=None, help="Use this node feature matrix instead of the default dataset file.")
    parser.add_argument("--node-schema-json", default=None, help="Use this feature schema JSON instead of the default dataset file.")
    parser.add_argument("--node-map-csv", default=None, help="Use this node metadata CSV instead of the default dataset file.")
    parser.add_argument("--src-col", default="u", help="Source column in the input edge CSV.")
    parser.add_argument("--dst-col", default="i", help="Target column in the input edge CSV.")
    parser.add_argument("--ts-col", default="ts", help="Timestamp / snapshot column in the input edge CSV.")
    parser.add_argument(
        "--weight-col",
        default=None,
        help="Raw additive edge-weight column in the input CSV, or the output column name when --weight-npy is used.",
    )
    parser.add_argument(
        "--weight-model",
        default="auto",
        choices=["auto", "real-exponential", "real-normal", "discrete-poisson", "discrete-geometric", "discrete-binomial"],
        help="Weighted-SBM family used for edge weights when --weight-col is provided.",
    )
    parser.add_argument(
        "--weight-transform",
        default="auto",
        choices=["auto", "none", "log", "log1p"],
        help="Transformation applied before fitting the edge-weight covariate.",
    )
    parser.add_argument(
        "--weight-binomial-max",
        type=int,
        default=None,
        help="Upper bound M for --weight-model=discrete-binomial. If unset, the observed maximum is used.",
    )
    parser.add_argument("--directed", action="store_true", help="Treat the network as directed.")
    parser.add_argument("--tz", default="Europe/Amsterdam", help="Timezone used for calendar feature engineering.")
    parser.add_argument("--ts-format", default="ordinal", choices=["ordinal", "unix"], help="Timestamp encoding in the input edge CSV.")
    parser.add_argument("--ts-unit", default="s", choices=["s", "ms", "us", "ns", "D"], help="Timestamp unit when --ts-format=unix.")
    parser.add_argument("--holiday-country", default="NL", help="Country code passed to python-holidays.")
    parser.add_argument("--ts-start", type=int, default=None, help="Inclusive lower timestamp bound after parsing.")
    parser.add_argument("--ts-end", type=int, default=None, help="Inclusive upper timestamp bound after parsing.")
    parser.add_argument("--date-start", default=None, help="Inclusive lower calendar date, interpreted using --ts-format / --ts-unit.")
    parser.add_argument("--date-end", default=None, help="Inclusive upper calendar date, interpreted using --ts-format / --ts-unit.")
    parser.add_argument(
        "--duplicate-policy",
        choices=["collapse", "error"],
        default="collapse",
        help="How to handle duplicate (u, i, ts) rows before fitting.",
    )
    parser.add_argument(
        "--self-loop-policy",
        choices=["drop", "error", "keep"],
        default="drop",
        help="How to handle self-loops before fitting.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Default: <data-root>/<dataset>/graph_tool_out/netforge.",
    )


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fit-covariates",
        nargs="+",
        default=None,
        help=(
            "Subset of built-in edge covariates to include during fitting. "
            f"Choices are: {', '.join(sorted(DEFAULT_COVARIATES))}."
        ),
    )
    parser.add_argument("--no-deg-corr", action="store_true", help="Disable degree correction in the fitted SBM.")
    parser.add_argument("--overlap", action="store_true", help="Fit an overlapping base partition.")
    parser.add_argument("--fit-quiet", action="store_true", help="Reduce graph-tool verbosity during fitting.")
    parser.add_argument("--refine-multiflip-rounds", type=int, default=0, help="Extra multiflip refinement rounds after the initial fit.")
    parser.add_argument("--refine-multiflip-niter", type=int, default=10, help="Iterations per multiflip refinement round.")
    parser.add_argument("--anneal-niter", type=int, default=0, help="If positive, run graph-tool annealing for this many iterations.")
    parser.add_argument("--anneal-beta-start", type=float, default=1.0, help="Starting beta when annealing is enabled.")
    parser.add_argument("--anneal-beta-stop", type=float, default=10.0, help="Stopping beta when annealing is enabled.")


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default=None, help="Existing fitted run directory for generation/report stages.")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under <run-dir>/generated for this generation batch.",
    )
    parser.add_argument("--num-samples", type=int, default=1, help="How many synthetic panels to draw.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for graph-tool sampling.")
    parser.add_argument("--sample-canonical", action="store_true", help="Use graph-tool canonical sampling for each layer.")
    parser.add_argument("--sample-max-ent", action="store_true", help="Use graph-tool max-entropy sampling for each layer.")
    parser.add_argument("--sample-n-iter", type=int, default=20000, help="n_iter used when canonical/max-ent sampling is enabled.")
    sample_params_group = parser.add_mutually_exclusive_group()
    sample_params_group.add_argument(
        "--sample-params",
        dest="sample_params",
        action="store_true",
        default=None,
        help="When canonical sampling is enabled, sample block-model count parameters from the posterior.",
    )
    sample_params_group.add_argument(
        "--no-sample-params",
        dest="sample_params",
        action="store_false",
        help="When canonical sampling is enabled, use maximum-likelihood count parameters instead of posterior sampling.",
    )
    parser.add_argument(
        "--posterior-partition-sweeps",
        type=int,
        default=25,
        help="Short MCMC walk on a copied fitted state before each synthetic panel. Set to 0 to disable.",
    )
    parser.add_argument(
        "--posterior-partition-sweep-niter",
        type=int,
        default=10,
        help="niter passed to each posterior-partition MCMC sweep.",
    )
    parser.add_argument(
        "--posterior-partition-beta",
        type=float,
        default=1.0,
        help="Inverse temperature beta used when sampling partitions from the posterior.",
    )
    parser.add_argument(
        "--weight-min-cell-count",
        type=int,
        default=3,
        help="Minimum observed edges in a layer/block-pair cell before backing off to a broader edge-weight sampler.",
    )
    parser.add_argument(
        "--rewire-model",
        default="none",
        choices=["none", "configuration", "constrained-configuration", "blockmodel-micro"],
        help="graph-tool random_rewire model applied after each sampled snapshot when rewiring is enabled.",
    )
    parser.add_argument(
        "--rewire-n-iter",
        type=int,
        default=10,
        help="n_iter passed to graph-tool random_rewire when --rewire-model is enabled.",
    )
    parser.add_argument(
        "--rewire-persist",
        action="store_true",
        help="Retry rejected rewiring moves until they succeed when --rewire-model is enabled.",
    )
    parser.add_argument("--save-graph-tool-snapshots", action="store_true", help="Save sampled per-snapshot .gt files in addition to CSV exports.")


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detailed-diagnostics",
        action="store_true",
        help="Write per-block-pair, per-block, and per-node time-series diagnostics and plots.",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=12,
        help="How many top block pairs, blocks, and nodes to include in detailed diagnostic plots.",
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Write a formatted HTML validation report for the run directory.",
    )
    parser.add_argument(
        "--html-report-path",
        default=None,
        help="Output path for --html-report. Default: <run-dir>/diagnostics/scientific_validation_report.html.",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging across the pipeline and diagnostics.")


def _normalise_argv(argv: Optional[Iterable[str]]) -> list[str]:
    argv = list(argv or [])
    commands = {"run", "fit", "generate", "report", "sweep"}
    if not argv:
        return ["run", *argv]
    if any(arg in {"-h", "--help"} for arg in argv) and not any(arg in commands for arg in argv):
        return argv
    if argv[0] not in commands:
        return ["run", *argv]
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NetForge fits a layered SBM to a temporal edge panel, generates synthetic snapshots, "
            "and writes comparison reports."
        )
    )
    add_runtime_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Fit, generate, and report in one command.")
    add_runtime_arguments(run_parser)
    add_input_arguments(run_parser)
    add_fit_arguments(run_parser)
    add_generation_arguments(run_parser)
    add_report_arguments(run_parser)

    fit_parser = subparsers.add_parser("fit", help="Validate input data and fit the layered SBM.")
    add_runtime_arguments(fit_parser)
    add_input_arguments(fit_parser)
    add_fit_arguments(fit_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic panels from a fitted run directory.")
    add_runtime_arguments(generate_parser)
    add_generation_arguments(generate_parser)

    report_parser = subparsers.add_parser("report", help="Compare generated networks against the fitted input panel.")
    add_runtime_arguments(report_parser)
    add_report_arguments(report_parser)
    report_parser.add_argument("--run-dir", required=True, help="Run directory created by the fit or run stage.")
    report_parser.add_argument(
        "--synthetic-edges-csv",
        default=None,
        help="Compare one specific synthetic edge CSV. If omitted, every saved sample in the run directory is reported.",
    )
    report_parser.add_argument("--sample-label", default=None, help="Label used in report filenames.")

    sweep_parser = subparsers.add_parser("sweep", help="Run a configured generation sweep, then report and simulate it.")
    add_runtime_arguments(sweep_parser)
    sweep_parser.add_argument("--config", required=True, help="Path to a JSON sweep configuration file.")
    return parser


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return _default_output_dir(args.data_root, args.dataset)


def _resolve_run_dir_for_logging(args: argparse.Namespace) -> Optional[Path]:
    if args.command in {"fit", "run"} and getattr(args, "output_dir", None):
        return Path(args.output_dir).expanduser().resolve()
    if args.command in {"generate", "report"} and getattr(args, "run_dir", None):
        return Path(args.run_dir).expanduser().resolve()
    return None


def _resolve_log_path(args: argparse.Namespace) -> Optional[Path]:
    run_dir = _resolve_run_dir_for_logging(args)
    if run_dir is None:
        return None
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "logs" / f"{args.command}_{timestamp}.log"


def _append_log_artifact(run_dir: Path, artifact: dict) -> None:
    try:
        manifest = load_manifest(run_dir)
    except Exception:
        return

    existing = [entry for entry in manifest.get("log_artifacts", []) if entry.get("log_path") != artifact.get("log_path")]
    existing.append(artifact)
    manifest["log_artifacts"] = existing
    save_json(manifest, Path(manifest["manifest_path"]))


def _finalize_log_artifacts(log_path: Optional[Path], run_dir: Optional[Path]) -> Optional[dict]:
    if log_path is None or run_dir is None or not Path(log_path).exists():
        return None

    artifact = write_log_visual_summary(
        log_path=Path(log_path),
        output_dir=Path(run_dir) / "logs",
        label=Path(log_path).stem,
    )
    _append_log_artifact(Path(run_dir), artifact)
    return artifact


def _report_one_sample(
    run_dir: Path,
    synthetic_edges_csv: Path,
    sample_label: str,
    *,
    detailed_diagnostics: bool = False,
    diagnostic_top_k: int = 12,
) -> dict:
    LOGGER.debug(
        "Reporting sample | run_dir=%s | synthetic_edges_csv=%s | sample_label=%s",
        run_dir,
        synthetic_edges_csv,
        sample_label,
    )
    manifest = load_manifest(run_dir)
    original_df = pd.read_csv(manifest["filtered_input_edges_path"])
    synthetic_df = pd.read_csv(synthetic_edges_csv)
    node_attributes_path = Path(manifest["node_attributes_path"])
    node_coordinates = load_node_coordinates(node_attributes_path)
    node_blocks = load_node_blocks(node_attributes_path)
    node_types = load_node_types(node_attributes_path)
    if node_blocks is None:
        try:
            node_blocks = load_node_block_map_from_graph_path(Path(manifest["graph_path"]))
            if node_blocks:
                LOGGER.debug("Recovered node block mapping from graph artifact | node_count=%s", len(node_blocks))
        except Exception as exc:
            LOGGER.debug("Failed to recover node block mapping from graph artifact | error=%s", exc)

    weight_model = manifest.get("weight_model") or {}
    weight_col = None
    if isinstance(weight_model, dict):
        weight_col = weight_model.get("output_column") or weight_model.get("input_column")

    detailed_outputs = None
    if detailed_diagnostics:
        comparison = compare_panels_detailed(
            original_df=original_df,
            synthetic_df=synthetic_df,
            directed=bool(manifest["directed"]),
            node_coordinates=node_coordinates,
            weight_col=weight_col,
            node_blocks=node_blocks,
            node_types=node_types,
        )
        per_snapshot = comparison["per_snapshot"]
        summary = comparison["summary"]
        detailed_outputs = comparison.get("details") or {}
    else:
        per_snapshot, summary = compare_panels(
            original_df=original_df,
            synthetic_df=synthetic_df,
            directed=bool(manifest["directed"]),
            node_coordinates=node_coordinates,
            weight_col=weight_col,
        )
    diagnostics_dir = Path(run_dir) / "diagnostics"
    report_paths = write_report(
        per_snapshot=per_snapshot,
        summary=summary,
        output_dir=diagnostics_dir,
        sample_label=sample_label,
        detailed_diagnostics=detailed_outputs,
        directed=bool(manifest["directed"]),
        diagnostic_top_k=diagnostic_top_k,
    )
    LOGGER.debug("Completed sample report | sample_label=%s | summary=%s", sample_label, summary)
    return {
        "sample_label": sample_label,
        "synthetic_edges_csv": str(synthetic_edges_csv),
        "summary": summary,
        "outputs": report_paths,
    }


def _sample_index_from_name(name: str, fallback: int = 0) -> int:
    if str(name).startswith("sample_"):
        try:
            return int(str(name).split("_", 1)[1])
        except Exception:
            return int(fallback)
    return int(fallback)


def _infer_sample_mode_from_settings(sample_settings: dict) -> str:
    if bool(sample_settings.get("sample_canonical")) and bool(sample_settings.get("sample_max_ent")):
        return "canonical_maxent"
    if bool(sample_settings.get("sample_canonical")):
        sample_params = sample_settings.get("sample_params")
        if sample_params is True:
            return "canonical_posterior"
        if sample_params is False:
            return "canonical_ml"
        return "canonical"
    if bool(sample_settings.get("sample_max_ent")):
        return "maxent_micro"
    return "micro"


def _infer_setting_label(sample_path: Path, generated_root: Path, sample: Optional[dict] = None) -> str:
    sample = sample or {}
    explicit = sample.get("setting_label")
    if explicit:
        return str(explicit)
    try:
        relative = sample_path.relative_to(generated_root)
        parts = relative.parts
        if len(parts) >= 3 and parts[-2].startswith("sample_"):
            return str(parts[-3])
    except ValueError:
        pass
    sample_settings = sample.get("sample_settings") or {}
    if sample_settings:
        rewire_model = str(sample_settings.get("rewire_model", "none")).replace("-", "_")
        return f"{_infer_sample_mode_from_settings(sample_settings)}__rewire_{rewire_model}"
    return str(sample.get("sample_label") or sample_path.parent.name)


def _discover_generated_samples(run_dir: Path, manifest: dict) -> list[dict[str, object]]:
    generated_root = run_dir / "generated"
    discovered: list[dict[str, object]] = []
    seen_paths: set[Path] = set()

    for sample in manifest.get("generated_samples", []):
        sample_path = Path(str(sample.get("synthetic_edges_csv", ""))).expanduser()
        if not sample_path.is_absolute():
            sample_path = (run_dir / sample_path).resolve()
        else:
            sample_path = sample_path.resolve()
        if not sample_path.exists() or sample_path in seen_paths:
            continue
        sample_index = int(sample.get("sample_index", _sample_index_from_name(sample.get("sample_label", ""), len(discovered))))
        setting_label = _infer_setting_label(sample_path, generated_root, sample)
        discovered.append(
            {
                "setting_label": setting_label,
                "sample_index": sample_index,
                "synthetic_edges_csv": sample_path,
                "sample_manifest_path": sample.get("sample_manifest_path"),
            }
        )
        seen_paths.add(sample_path)

    if generated_root.exists():
        for sample_path in sorted(generated_root.rglob("synthetic_edges.csv")):
            resolved_path = sample_path.resolve()
            if resolved_path in seen_paths:
                continue
            sample_manifest_path = sample_path.parent / "sample_manifest.json"
            sample_payload = load_json(sample_manifest_path) if sample_manifest_path.exists() else {}
            sample_index = int(sample_payload.get("sample_index", _sample_index_from_name(sample_path.parent.name, len(discovered))))
            setting_label = _infer_setting_label(resolved_path, generated_root, sample_payload)
            discovered.append(
                {
                    "setting_label": setting_label,
                    "sample_index": sample_index,
                    "synthetic_edges_csv": resolved_path,
                    "sample_manifest_path": str(sample_manifest_path) if sample_manifest_path.exists() else None,
                }
            )
            seen_paths.add(resolved_path)

    counts = Counter(str(record["setting_label"]) for record in discovered)
    for record in discovered:
        setting_label = str(record["setting_label"])
        sample_index = int(record["sample_index"])
        record["sample_label"] = setting_label if counts[setting_label] == 1 else f"{setting_label}__sample_{sample_index:04d}"

    discovered.sort(key=lambda item: (str(item["setting_label"]), int(item["sample_index"])))
    return discovered


def run_report_stage(args: argparse.Namespace) -> list[dict]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = load_manifest(run_dir)
    LOGGER.debug("Starting report stage | run_dir=%s | args=%s", run_dir, vars(args))
    synthetic_edges_csv = getattr(args, "synthetic_edges_csv", None)
    sample_label = getattr(args, "sample_label", None)

    if synthetic_edges_csv:
        resolved_label = sample_label or Path(synthetic_edges_csv).stem
        samples = [
            {
                "setting_label": resolved_label,
                "sample_label": resolved_label,
                "sample_index": 0,
                "synthetic_edges_csv": Path(synthetic_edges_csv).expanduser().resolve(),
                "sample_manifest_path": None,
            }
        ]
    else:
        samples = _discover_generated_samples(run_dir, manifest)
        if not samples:
            raise ValueError(f"No generated samples were found under {run_dir / 'generated'} and none are usable from the run manifest.")
    LOGGER.debug("Resolved report samples | %s", samples)

    run_reports: list[dict] = []
    for sample in samples:
        report = _report_one_sample(
            run_dir,
            Path(sample["synthetic_edges_csv"]),
            str(sample["sample_label"]),
            detailed_diagnostics=bool(getattr(args, "detailed_diagnostics", False)),
            diagnostic_top_k=max(1, int(getattr(args, "diagnostic_top_k", 12))),
        )
        report["setting_label"] = str(sample["setting_label"])
        report["sample_index"] = int(sample["sample_index"])
        run_reports.append(report)

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    run_summary_rows: list[dict[str, object]] = []
    for report in run_reports:
        row = summary_payload_to_row(
            str(report["sample_label"]),
            dict(report["summary"]),
            extra={
                "setting_label": str(report["setting_label"]),
                "sample_index": int(report["sample_index"]),
                "posterior_num_runs": 1,
            },
        )
        if row is not None:
            run_summary_rows.append(row)
    run_summary_frame = pd.DataFrame(run_summary_rows)
    if len(run_summary_frame):
        run_summary_frame.to_csv(diagnostics_dir / "all_sample_runs_summary.csv", index=False)

    setting_reports: list[dict] = []
    grouped_reports: dict[str, list[dict]] = {}
    for report in run_reports:
        grouped_reports.setdefault(str(report["setting_label"]), []).append(report)

    for setting_label, grouped in sorted(grouped_reports.items()):
        if len(grouped) == 1 and str(grouped[0]["sample_label"]) == setting_label:
            setting_reports.append(grouped[0])
            continue
        setting_reports.append(
            aggregate_posterior_reports(
                grouped,
                output_dir=diagnostics_dir,
                setting_label=setting_label,
                directed=bool(manifest["directed"]),
                diagnostic_top_k=max(1, int(getattr(args, "diagnostic_top_k", 12))),
            )
        )

    setting_summary_rows: list[dict[str, object]] = []
    for report in setting_reports:
        row = summary_payload_to_row(
            str(report["sample_label"]),
            dict(report["summary"]),
            extra={
                "setting_label": str(report.get("setting_label", report["sample_label"])),
            },
        )
        if row is not None:
            setting_summary_rows.append(row)
    summary_frame = pd.DataFrame(setting_summary_rows)
    summary_frame.to_csv(diagnostics_dir / "all_samples_summary.csv", index=False)
    if len(summary_frame):
        summary_frame.to_csv(diagnostics_dir / "setting_posterior_summary.csv", index=False)

    manifest["diagnostics_runs"] = run_reports
    manifest["diagnostics"] = setting_reports
    manifest["diagnostics_summary_mode"] = "posterior_setting_summary" if any(len(group) > 1 for group in grouped_reports.values()) else "single_run"

    overview_path = write_all_samples_overview(summary_frame, diagnostics_dir)
    if overview_path is not None:
        manifest["diagnostics_overview_png"] = str(overview_path)
    if bool(getattr(args, "html_report", False)):
        html_report_path = write_scientific_validation_report(
            run_dir=run_dir,
            output_path=Path(args.html_report_path).expanduser().resolve() if getattr(args, "html_report_path", None) else None,
        )
        manifest["scientific_validation_report_html"] = str(html_report_path)
    save_json(manifest, Path(manifest["manifest_path"]))
    LOGGER.info(
        "Wrote diagnostics for %s generated run(s) and %s setting summary row(s) under %s",
        len(run_reports),
        len(setting_reports),
        diagnostics_dir,
    )
    return setting_reports


def main(argv: Optional[Iterable[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _normalise_argv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"run", "fit"} and args.output_dir is None:
        args.output_dir = str(_resolve_output_dir(args))

    log_path = _resolve_log_path(args)
    _configure_logging(verbose=bool(args.verbose), log_path=log_path)
    LOGGER.debug("CLI invocation | argv=%s | command=%s | args=%s", argv, args.command, vars(args))
    if args.command in {"run", "fit"} and args.output_dir is not None:
        LOGGER.debug("Resolved default output_dir=%s", args.output_dir)

    if args.command == "fit":
        manifest = fit_command(args)
        _finalize_log_artifacts(log_path, Path(manifest["run_dir"]))
        return 0

    if args.command == "generate":
        if not args.run_dir:
            raise ValueError("--run-dir is required for the generate command.")
        generate_command(args)
        _finalize_log_artifacts(log_path, Path(args.run_dir).expanduser().resolve())
        return 0

    if args.command == "report":
        run_report_stage(args)
        _finalize_log_artifacts(log_path, Path(args.run_dir).expanduser().resolve())
        return 0

    if args.command == "sweep":
        from temporal_sbm.sweep import run_sweep_command

        result = run_sweep_command(args)
        run_dir = Path(str(result["run_dir"])).expanduser().resolve()
        _finalize_log_artifacts(log_path, run_dir)
        return 0

    if args.command == "run":
        manifest = fit_command(args)
        args.run_dir = manifest["run_dir"]
        generate_command(args)
        run_report_stage(args)
        _finalize_log_artifacts(log_path, Path(manifest["run_dir"]))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
