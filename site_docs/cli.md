# CLI

NetForge uses `fit`, `generate`, and `report` as the main stages. For grid searches there is also `sweep`, which runs a configured generation grid, reports it, and can hand the selected setting to the simulation module.

## `netforge fit`

Use `fit` to validate the input data, build the layered graph, and fit the block model.

```bash
netforge fit \
  --data-root <data-root> \
  --dataset <dataset> \
  --directed \
  --weight-col trade \
  --weight-model auto \
  --weight-transform none
```

Common flags:

- `--date-start` and `--date-end` to select a calendar range
- `--ts-start` and `--ts-end` to select an ordinal or Unix range
- `--fit-covariates` to choose built in edge annotations such as `dist_km`, `mass_grav`, `anim_grav`, and `ft_cosine`
- `--joint-metadata-model` or `--no-joint-metadata-model` to include or skip metadata tag vertices
- `--metadata-fields` to choose which metadata become tag nodes
- `--metadata-numeric-bins` and `--metadata-grid-km` to control numeric-bin and centroid-grid tags
- `--metadata-ft-top-k` to cap how many `ft_tokens` are kept per node when `ft_tokens` is requested
- `--duplicate-policy` and `--self-loop-policy` to decide how repeated rows and self loops are handled before fitting
- `--no-compact` to keep the full node universe instead of compacting to active nodes in the selected time window
- `--output-dir` to write the run somewhere other than the default run directory

With the default fit settings, NetForge builds a joint data-metadata multilayer graph. Distinct metadata tokens become tag vertices in a layer named `__metadata__`. Text metadata fields split on `|` and `;`, so one cell can produce several tag links.

When the joint metadata model is active and `--fit-covariates` is not set, the fit is topology only over the trade and metadata edges. Set `--fit-covariates` when you want the realized-edge annotations in the SBM fit.

## `netforge generate`

Use `generate` on a fitted run directory to draw synthetic panels.

```bash
netforge generate \
  --run-dir <run-dir> \
  --num-samples 10 \
  --seed 2026 \
  --posterior-partition-sweeps 25 \
  --posterior-partition-sweep-niter 10
```

Common flags:

- `--sample-canonical` for canonical edge-count sampling
- `--sample-max-ent` for max-entropy edge-count sampling
- `--sample-params` or `--no-sample-params` to control parameter draws during canonical sampling
- `--output-subdir` to choose the setting directory under `<run-dir>/generated/`
- `--posterior-partition-sweeps`, `--posterior-partition-sweep-niter`, and `--posterior-partition-beta` to refresh the fitted partition before each draw
- `--weight-generation-mode`, `--weight-parametric-partition-policy`, and `--weight-pure-generative` to control how weighted generation uses the saved weight model
- `--temporal-generator-mode` to choose between the stored Markov turnover generator and the older independent-layer sampler
- `--temporal-activity-level`, `--temporal-group-mode`, and the temporal proposal flags to control turnover targets and proposal pools
- `--rewire-model` and `--rewire-n-iter` for rewiring sensitivity analyses after each sampled snapshot
- `--save-graph-tool-snapshots` to save `.gt` files for each generated snapshot

The default temporal generation path uses `--temporal-generator-mode markov_turnover`, so the default setting label is `markov_turnover__proposal_sbm__micro__rewire_none`. Each generation batch writes:

- `generated/<setting-label>/setting_manifest.json`
- `generated/<setting-label>/sample_####/synthetic_edges.csv`
- `generated/<setting-label>/sample_####/sample_manifest.json`

When the temporal generator is active, sample directories also include `temporal_generation_summary.json`.

## `netforge report`

Use `report` to compare saved synthetic panels against the observed panel in the same run directory.

```bash
netforge report \
  --run-dir <run-dir> \
  --detailed-diagnostics \
  --diagnostic-top-k 12 \
  --html-report
```

Common flags:

- `--synthetic-edges-csv` to report one saved sample
- `--sample-label` to control filenames
- `--html-report-path` to write the HTML report to a custom path
- `--include-daily-network-snapshots` to add the snapshot viewer to the HTML report
- `--skip-spectral-metrics` and `--skip-posterior-detail-aggregation` to trim longer report passes

Without `--synthetic-edges-csv`, `report` scans every saved sample under `generated/`, reports them one by one, and also writes setting summaries. Common outputs under `diagnostics/` include:

- `all_sample_runs_summary.csv`
- `all_samples_summary.csv`
- `setting_posterior_summary.csv`
- `scientific_validation_report.html` when `--html-report` is used

## `netforge sweep`

Use `sweep` with a JSON config when you want to fit once, generate several sampler and rewiring settings, report the whole grid, and select the best primary setting for simulation.

```bash
netforge sweep --config sweep.json
```

The sweep runner writes generation batches under `generated/<setting-label>/`, aggregates them through the report stage, and stores the chosen primary setting under `diagnostics/best_primary_setting.txt` unless the config points somewhere else.

## Help output

Every command exposes a full help page:

```bash
netforge --help
netforge fit --help
netforge generate --help
netforge report --help
netforge sweep --help
```
