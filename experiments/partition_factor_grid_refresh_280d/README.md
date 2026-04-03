# Partition Factor Grid Refresh 280d

This archive holds the runner, helper, and settings for the CR35 partition refresh study on a 280 day slice.

The grid spans three switches:

- `joint_metadata_model`
- `edge_covariates`
- `layered`

The `edge_covariates` switch covers the full edge side of the SBM fit. With `cov on`, the fit uses `dist_km`, `mass_grav`, `anim_grav`, and the trade weight covariate together. With `cov off`, all four stay out of the fit.

The weighted cells use `trade:discrete-geometric/none` for the SBM weight model. This avoids the Poisson branch that stalls on the 280 day slice.

Each cell includes one fitted partition and nine posterior refreshes, for ten partitions per cell.

## Contents

- `scripts/partition_factor_grid_refresh.py`
  Batch helper that plans the grid, runs one cell, and writes the summary tables and figures.
- `scripts/run_cr35_partition_factor_grid_refresh.sh`
  Shell entry point for the batch run.
- `settings/base_config.json`
  Source fit config used to seed the grid.
- `settings/run_manifest.json`
  Slice, replicate count, and refresh settings for the archived run.
- `settings/combo_specs.json`
  Full cell plan with the saved fit arguments for each cell.
- `settings/combos/*/config.json`
  Per cell fit settings copied from the run directory.

## Notes

- Archived run root: `.stress_runs/cr35_partition_factor_grid_280d_rep10_20260402_joint_cov_toggle_refresh`
- Slice: `ts_start=736695`, `ts_end=736974`
- Replicates per cell: `10`

Summary artifacts live under the matching run root in `.stress_runs/.../summary`.
