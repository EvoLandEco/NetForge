# Partition Factor Full Workflow 280d

This archive stores the scripts and settings for the 280 day CR35 factor experiment that runs the full workflow for every cell in the design.

The grid spans three switches:

- `joint_metadata_model`
- `edge_covariates`
- `layered`

The `edge_covariates` switch covers the full edge side of the SBM fit. With `cov on`, the fit uses `dist_km`, `mass_grav`, `anim_grav`, and the trade weight covariate together. With `cov off`, all four stay out of the fit.

Each of the 8 combinations runs the same workflow:

1. fit the SBM
2. generate synthetic panels
3. build diagnostics and HTML reports
4. run the simulation scenarios and write the simulation reports

## Contents

- `scripts/partition_factor_full_workflow.py`
  Helper that plans the grid, runs one cell, and builds the cross-combination summary.
- `scripts/run_cr35_partition_factor_full_workflow.sh`
  Shell entry point for the full batch run.
- `settings/base_config.json`
  Source sweep config used to seed the grid.
- `settings/run_manifest.json`
  Slice, workflow settings, and output root for the archived run.
- `settings/combo_specs.json`
  Full cell plan with the saved arguments for each cell.
- `settings/combos/*.json`
  Per-cell sweep settings.

## Notes

- Archived run root: `.stress_runs/cr35_partition_factor_full_workflow_280d_20260402_joint_cov_toggle`
- Slice: `ts_start=736695`, `ts_end=736974`
- Synthetic samples per cell: `10`
- Simulation replicates per scenario: `100`
- The weighted cells use `weight_model="discrete-geometric"` because the Poisson candidate stalled on this slice.

The run outputs stay under the archived run root in `.stress_runs/...`, and the cross-cell summary is written under its `summary/` directory.
