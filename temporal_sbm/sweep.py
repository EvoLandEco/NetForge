"""Sweep runner for fit, generation, reporting, and simulation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from temporal_sbm.pipeline import fit_command, generate_command, load_manifest, save_json
from temporal_sbm.simulation import SimulationScenario, run_scenario_set


LOGGER = logging.getLogger(__name__)

PRIMARY_SETTING_SORT_COLUMNS = [
    "mean_snapshot_edge_jaccard",
    "weight_total_correlation",
    "mean_synthetic_novel_edge_rate",
]
PRIMARY_SETTING_SORT_ASCENDING = [False, False, True]


@dataclass(frozen=True)
class SweepSetting:
    sampler: str
    rewire: str

    @property
    def label(self) -> str:
        return f"{self.sampler}__rewire_{self.rewire.replace('-', '_')}"


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep config must contain a JSON object: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def _argparse_action_map(adders: Iterable[Callable[[argparse.ArgumentParser], None]]) -> dict[str, list[argparse.Action]]:
    parser = argparse.ArgumentParser(add_help=False)
    for add_arguments in adders:
        add_arguments(parser)
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        actions.setdefault(action.dest, []).append(action)
    return actions


def _parser_action_map(parser: argparse.ArgumentParser) -> dict[str, list[argparse.Action]]:
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        actions.setdefault(action.dest, []).append(action)
    return actions


def _primary_option_string(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    if long_options:
        return max(long_options, key=len)
    if action.option_strings:
        return action.option_strings[0]
    raise ValueError(f"Argument action for '{action.dest}' does not expose any option strings.")


def _build_namespace_from_config(
    values: dict[str, Any],
    *,
    adders: Iterable[Callable[[argparse.ArgumentParser], None]],
) -> argparse.Namespace:
    action_map = _argparse_action_map(adders)
    unknown_keys = sorted(set(values) - set(action_map))
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    parser = argparse.ArgumentParser(add_help=False)
    for add_arguments in adders:
        add_arguments(parser)

    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        actions = action_map[key]
        if isinstance(value, bool):
            matched = False
            bool_only = all(
                isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
                for action in actions
            )
            for action in actions:
                if isinstance(action, argparse._StoreTrueAction) and value:
                    argv.append(_primary_option_string(action))
                    matched = True
                    break
                if isinstance(action, argparse._StoreFalseAction) and not value:
                    argv.append(_primary_option_string(action))
                    matched = True
                    break
            if matched or bool_only:
                continue
        action = actions[-1]
        option = _primary_option_string(action)
        if isinstance(action, argparse._AppendAction):
            if not isinstance(value, list):
                raise ValueError(f"Config key '{key}' must be a list.")
            for item in value:
                argv.extend([option, str(item)])
            continue
        if action.nargs in {"+", "*"}:
            if not isinstance(value, list):
                raise ValueError(f"Config key '{key}' must be a list.")
            argv.append(option)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([option, str(value)])

    return parser.parse_args(argv)


def _build_namespace_from_parser(values: dict[str, Any], parser: argparse.ArgumentParser) -> argparse.Namespace:
    action_map = _parser_action_map(parser)
    unknown_keys = sorted(set(values) - set(action_map))
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        actions = action_map[key]
        if isinstance(value, bool):
            matched = False
            bool_only = all(
                isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
                for action in actions
            )
            for action in actions:
                if isinstance(action, argparse._StoreTrueAction) and value:
                    argv.append(_primary_option_string(action))
                    matched = True
                    break
                if isinstance(action, argparse._StoreFalseAction) and not value:
                    argv.append(_primary_option_string(action))
                    matched = True
                    break
            if matched or bool_only:
                continue
        action = actions[-1]
        option = _primary_option_string(action)
        if isinstance(action, argparse._AppendAction):
            if not isinstance(value, list):
                raise ValueError(f"Config key '{key}' must be a list.")
            for item in value:
                argv.extend([option, str(item)])
            continue
        if action.nargs in {"+", "*"}:
            if not isinstance(value, list):
                raise ValueError(f"Config key '{key}' must be a list.")
            argv.append(option)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([option, str(value)])
    return parser.parse_args(argv)


def _sampler_overrides(sampler: str) -> dict[str, Any]:
    if sampler == "micro":
        return {
            "sample_canonical": False,
            "sample_max_ent": False,
            "sample_params": None,
        }
    if sampler == "maxent_micro":
        return {
            "sample_canonical": False,
            "sample_max_ent": True,
            "sample_params": None,
        }
    if sampler == "canonical_posterior":
        return {
            "sample_canonical": True,
            "sample_max_ent": False,
            "sample_params": True,
        }
    if sampler == "canonical_ml":
        return {
            "sample_canonical": True,
            "sample_max_ent": False,
            "sample_params": False,
        }
    if sampler == "canonical_maxent":
        return {
            "sample_canonical": True,
            "sample_max_ent": True,
            "sample_params": None,
        }
    raise ValueError(f"Unknown sampler: {sampler}")


def expand_generation_grid(grid_config: dict[str, Any]) -> list[SweepSetting]:
    samplers = [str(value).strip() for value in grid_config.get("samplers", []) if str(value).strip()]
    rewires = [str(value).strip() for value in grid_config.get("rewires", []) if str(value).strip()]
    if not samplers:
        raise ValueError("Sweep config grid.samplers must contain at least one sampler.")
    if not rewires:
        raise ValueError("Sweep config grid.rewires must contain at least one rewiring mode.")
    return [SweepSetting(sampler=sampler, rewire=rewire) for sampler in samplers for rewire in rewires]


def count_completed_samples(setting_dir: Path) -> int:
    if not setting_dir.exists():
        return 0
    return sum(1 for _ in setting_dir.glob("sample_*/sample_manifest.json"))


def pick_best_primary_setting(run_dir: Path, output_path: Optional[Path] = None) -> str:
    summary_path = run_dir / "diagnostics" / "setting_posterior_summary.csv"
    summary = pd.read_csv(summary_path)
    if "sample_label" not in summary.columns:
        raise ValueError(f"{summary_path} does not contain sample_label")

    primary = summary.loc[summary["sample_label"].astype(str).str.contains("__rewire_none", na=False)].copy()
    if primary.empty:
        raise ValueError("No primary posterior-predictive settings were found in the diagnostics summary.")

    ranked = primary.sort_values(PRIMARY_SETTING_SORT_COLUMNS, ascending=PRIMARY_SETTING_SORT_ASCENDING)
    best = str(ranked.iloc[0]["sample_label"])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(best + "\n")
    return best


def _save_sweep_record(
    *,
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    best_setting: str,
    simulation_result: dict[str, object],
) -> None:
    manifest = load_manifest(run_dir)
    sweep_record = {
        "config_path": str(config_path),
        "best_primary_setting": best_setting,
        "simulation": simulation_result,
    }
    manifest["sweep"] = sweep_record
    save_json(manifest, Path(manifest["manifest_path"]))

    copy_path = run_dir / "sweep_config.json"
    copied_config = {key: value for key, value in config.items() if key != "_config_path"}
    copy_path.write_text(json.dumps(copied_config, indent=2, sort_keys=True))


def run_sweep_command(args: argparse.Namespace) -> dict[str, object]:
    from temporal_sbm.cli import (
        add_fit_arguments,
        add_generation_arguments,
        add_input_arguments,
        add_report_arguments,
        run_report_stage,
    )
    from temporal_sbm.simulation import build_parser as build_simulation_parser

    config = load_sweep_config(args.config)
    config_path = Path(str(config["_config_path"]))

    fit_config = dict(config.get("fit") or {})
    if not fit_config:
        raise ValueError("Sweep config must include a fit section.")
    if "output_dir" not in fit_config or not str(fit_config["output_dir"]).strip():
        raise ValueError("Sweep config fit.output_dir is required.")

    run_dir = Path(str(fit_config["output_dir"])).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    fit_args = _build_namespace_from_config(
        fit_config,
        adders=[add_input_arguments, add_fit_arguments],
    )
    fit_args.output_dir = str(run_dir)
    fit_args.command = "fit"
    fit_args.verbose = bool(getattr(args, "verbose", False))

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        LOGGER.info("Fit already present at %s", run_dir)
    else:
        LOGGER.info("Starting fit | run_dir=%s", run_dir)
        fit_command(fit_args)
        LOGGER.info("Fit finished | run_dir=%s", run_dir)

    grid_config = dict(config.get("grid") or {})
    generate_config = dict(config.get("generate") or {})
    if not generate_config:
        raise ValueError("Sweep config must include a generate section.")
    for setting in expand_generation_grid(grid_config):
        setting_dir = run_dir / "generated" / setting.label
        expected_samples = int(generate_config.get("num_samples", 1))
        completed_samples = count_completed_samples(setting_dir)
        if completed_samples == expected_samples:
            LOGGER.info("Skipping %s; found %s completed samples", setting.label, completed_samples)
            continue

        setting_config = {
            **generate_config,
            **_sampler_overrides(setting.sampler),
            "rewire_model": setting.rewire,
            "output_subdir": setting.label,
            "run_dir": str(run_dir),
        }
        generate_args = _build_namespace_from_config(
            setting_config,
            adders=[add_generation_arguments],
        )
        generate_args.command = "generate"
        generate_args.verbose = bool(getattr(args, "verbose", False))
        LOGGER.info("Starting generation | setting=%s", setting.label)
        generate_command(generate_args)
        LOGGER.info("Finished generation | setting=%s", setting.label)

    report_config = dict(config.get("report") or {})
    report_config["run_dir"] = str(run_dir)
    report_args = _build_namespace_from_config(
        report_config,
        adders=[add_report_arguments, lambda parser: parser.add_argument("--run-dir", required=True)],
    )
    report_args.command = "report"
    report_args.verbose = bool(getattr(args, "verbose", False))
    LOGGER.info("Starting report | run_dir=%s", run_dir)
    run_report_stage(report_args)
    LOGGER.info("Report finished | run_dir=%s", run_dir)

    simulation_config = dict(config.get("simulation") or {})
    scenario_specs = list(simulation_config.get("scenarios") or [])
    if not scenario_specs:
        raise ValueError("Sweep config simulation.scenarios must contain at least one scenario.")

    selected_output_path = simulation_config.get("selected_setting_output_path")
    best_setting_output_path = (
        Path(str(selected_output_path)).expanduser().resolve()
        if selected_output_path
        else run_dir / "diagnostics" / "best_primary_setting.txt"
    )
    best_setting = pick_best_primary_setting(run_dir, best_setting_output_path)
    LOGGER.info("Best primary setting: %s", best_setting)

    simulation_base = dict(simulation_config.get("base_args") or {})
    simulation_output_dir = simulation_config.get("output_dir")
    if simulation_output_dir:
        resolved_simulation_output_dir = Path(str(simulation_output_dir)).expanduser().resolve()
    else:
        resolved_simulation_output_dir = run_dir / "simulation_scenarios" / best_setting
    simulation_base.update(
        {
            "run_dir": str(run_dir),
            "output_dir": str(resolved_simulation_output_dir),
            "setting_label": [best_setting],
        }
    )

    simulation_parser = build_simulation_parser()
    simulation_action_map = _parser_action_map(simulation_parser)
    unknown_simulation_keys = sorted(set(simulation_base) - set(simulation_action_map))
    if unknown_simulation_keys:
        raise ValueError(f"Unknown simulation base config keys: {', '.join(unknown_simulation_keys)}")

    simulation_args = _build_namespace_from_parser(simulation_base, simulation_parser)

    scenarios: list[SimulationScenario] = []
    for spec in scenario_specs:
        if not isinstance(spec, dict):
            raise ValueError("Each simulation scenario must be a JSON object.")
        name = str(spec.get("name") or "").strip()
        description = str(spec.get("description") or "").strip()
        if not name:
            raise ValueError("Each simulation scenario requires a name.")
        if not description:
            raise ValueError(f"Simulation scenario '{name}' requires a description.")
        overrides = {key: value for key, value in spec.items() if key not in {"name", "description"}}
        unknown_override_keys = sorted(set(overrides) - set(simulation_action_map))
        if unknown_override_keys:
            raise ValueError(
                f"Simulation scenario '{name}' uses unknown keys: {', '.join(unknown_override_keys)}"
            )
        scenarios.append(SimulationScenario(name=name, description=description, overrides=overrides))

    LOGGER.info(
        "Starting simulation scenarios | selected_setting=%s | scenario_count=%s",
        best_setting,
        len(scenarios),
    )
    simulation_result = run_scenario_set(simulation_args, scenarios)
    LOGGER.info("Simulation scenarios finished | output_dir=%s", simulation_result["output_dir"])

    _save_sweep_record(
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        best_setting=best_setting,
        simulation_result=simulation_result,
    )
    return {
        "run_dir": str(run_dir),
        "best_primary_setting": best_setting,
        "simulation": simulation_result,
    }
