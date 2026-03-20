import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from temporal_sbm.cli import _default_output_dir, _normalise_argv, _resolve_run_dir_for_logging, build_parser


class CliTests(unittest.TestCase):
    def test_default_output_dir_uses_netforge_leaf(self):
        output_dir = _default_output_dir("~/netforge-data", "CR35")

        self.assertEqual(output_dir.name, "netforge")
        self.assertEqual(output_dir, Path("~/netforge-data").expanduser().resolve() / "CR35" / "graph_tool_out" / "netforge")

    def test_normalise_argv_defaults_to_run_command(self):
        argv = _normalise_argv(["--data-root", "/tmp/data", "--dataset", "CR35"])

        self.assertEqual(argv[0], "run")
        self.assertEqual(argv[1:], ["--data-root", "/tmp/data", "--dataset", "CR35"])

    def test_normalise_argv_keeps_top_level_help_unchanged(self):
        self.assertEqual(_normalise_argv(["--help"]), ["--help"])

    def test_normalise_argv_keeps_explicit_subcommand(self):
        argv = _normalise_argv(["generate", "--run-dir", "/tmp/run"])

        self.assertEqual(argv, ["generate", "--run-dir", "/tmp/run"])

    def test_normalise_argv_keeps_explicit_sweep_subcommand(self):
        argv = _normalise_argv(["sweep", "--config", "/tmp/sweep.json"])

        self.assertEqual(argv, ["sweep", "--config", "/tmp/sweep.json"])

    def test_build_parser_accepts_run_arguments_after_normalisation(self):
        parser = build_parser()
        args = parser.parse_args(
            _normalise_argv(
                [
                    "--data-root",
                    "/tmp/data",
                    "--dataset",
                    "CR35",
                ]
            )
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.dataset, "CR35")
        self.assertEqual(args.output_dir, None)
        self.assertEqual(args.weight_model, "auto")

    def test_build_parser_accepts_sweep_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--config", "/tmp/sweep.json"])

        self.assertEqual(args.command, "sweep")
        self.assertEqual(args.config, "/tmp/sweep.json")

    def test_build_parser_exposes_parametric_weight_generation_arguments(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--run-dir",
                "/tmp/run",
                "--weight-generation-mode",
                "parametric",
                "--weight-parametric-partition-policy",
                "refit_on_refresh",
                "--weight-parametric-family",
                "lognormal",
                "--weight-prior-strength",
                "7.5",
            ]
        )

        self.assertEqual(args.weight_generation_mode, "parametric")
        self.assertEqual(args.weight_parametric_partition_policy, "refit_on_refresh")
        self.assertEqual(args.weight_parametric_family, "lognormal")
        self.assertEqual(args.weight_prior_strength, 7.5)

    def test_build_parser_accepts_exclude_weight_from_fit(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "fit",
                "--data-root",
                "/tmp/data",
                "--dataset",
                "CR35",
                "--exclude-weight-from-fit",
            ]
        )

        self.assertTrue(args.exclude_weight_from_fit)

    def test_resolve_run_dir_for_logging_reads_sweep_output_dir_from_config(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_dir = tmp_path / "run"
            config_path = tmp_path / "sweep.json"
            config_path.write_text('{"fit": {"output_dir": "' + str(run_dir) + '"}}', encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["sweep", "--config", str(config_path)])

            resolved = _resolve_run_dir_for_logging(args)

        self.assertEqual(resolved, run_dir.resolve())


if __name__ == "__main__":
    unittest.main()
