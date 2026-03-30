# Partition Factor Grid Refresh 280d

This archive stores the scripts and settings for the 280 day partition factor experiment on CR35.

The run compares four factors:

- `joint_metadata_model`
- trade-edge covariates on or off
- `exclude_weight_from_fit`
- `layered`

Each factor combination was fit once and followed by 9 posterior partition refreshes, for 10 partitions per cell.

## Contents

- `scripts/partition_factor_grid_refresh.py`
  Batch helper that plans the grid, runs one cell, and builds the summary tables and figures.
- `scripts/run_cr35_partition_factor_grid_refresh.sh`
  Shell entry point for the full batch run.
- `settings/base_config.json`
  Source fit config used to seed the grid.
- `settings/run_manifest.json`
  Slice, replicate count, and refresh settings for the archived run.
- `settings/combo_specs.json`
  Full 16 cell plan with the saved fit arguments for each cell.
- `settings/combos/*/config.json`
  Per-cell fit settings copied from the run directory.

## Notes

- Archived run root: `.stress_runs/cr35_partition_factor_grid_280d_rep10_20260328_105445`
- Slice: `ts_start=736695`, `ts_end=736974`
- Replicates per cell: `10`
- The cell `meta_off__cov_on__exw_off__layered_off` was rerun with `weight_model="discrete-geometric"` after the Poisson candidate stalled on this slice.

The summary artifacts for the archived run remain under the original run root in `.stress_runs/.../summary`.
