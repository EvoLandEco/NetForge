# CLI

NetForge exposes three main commands through the `netforge` entry point:

- `fit`
- `generate`
- `report`

There is also a combined `run` command that executes those stages in sequence.

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
- `--fit-covariates` to choose built-in edge covariates
- `--joint-metadata-model` or `--no-joint-metadata-model` to include or skip metadata tag vertices
- `--metadata-fields` to choose which metadata become tag nodes
- `--metadata-numeric-bins`, `--metadata-grid-km`, and `--metadata-ft-top-k` to control tag construction
- `--output-dir` to write the run somewhere other than the default run directory

With the default fit settings, NetForge builds a joint data-metadata multilayer graph. Distinct metadata tokens become tag vertices in a layer named `__metadata__`.

## `netforge generate`

Use `generate` on a fitted run directory to draw synthetic panels.

```bash
netforge generate \
  --run-dir <run-dir> \
  --num-samples 10 \
  --seed 2026 \
  --posterior-partition-sweeps 10 \
  --posterior-partition-sweep-niter 5
```

Common flags:

- `--sample-canonical` for canonical edge-count sampling
- `--sample-max-ent` for max-entropy edge-count sampling
- `--sample-params` or `--no-sample-params` to control parameter draws during canonical sampling
- `--rewire-model` and `--rewire-n-iter` for post-sampling rewiring checks
- `--weight-min-cell-count` to control the backing-off threshold in the weight sampler

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

## `netforge run`

Use `run` when you want a single command that fits, generates, and reports.

```bash
netforge run \
  --data-root <data-root> \
  --dataset <dataset> \
  --directed \
  --weight-col trade \
  --weight-model auto \
  --num-samples 3 \
  --html-report
```

`run` accepts the same fit, generate, and report flags as the individual stages.

## Help output

Every command exposes a full help page:

```bash
netforge --help
netforge fit --help
netforge generate --help
netforge report --help
netforge run --help
```
