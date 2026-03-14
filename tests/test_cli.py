import unittest
from pathlib import Path

from temporal_sbm.cli import _default_output_dir, _normalise_argv, build_parser


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


if __name__ == "__main__":
    unittest.main()
