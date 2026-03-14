# Simulation

NetForge includes a transmission simulation module for fitted run directories that contain generated panels.

The simulation entry point is:

```bash
python -m temporal_sbm.simulation --help
```

## Inputs

The simulation module reads:

- a fitted run directory with generated samples
- the observed panel recorded in the run manifest
- one or more generated settings selected with `--setting-label`

## Baseline run

This command runs the baseline simulation for one generated setting:

```bash
python -m temporal_sbm.simulation \
  --run-dir <run-dir> \
  --setting-label maxent_micro__rewire_none \
  --beta-ff 0.20 \
  --beta-fr 0.04 \
  --beta-rf 0.16 \
  --beta-rr 0.02 \
  --sigma 0.35 \
  --gamma 0.10 \
  --num-replicates 256 \
  --seed 42 \
  --initial-seed-count 3 \
  --weight-mode log1p \
  --weight-scale auto \
  --tail-days 30 \
  --seed-scope farm_only \
  --seed-pool-mode observed_day0 \
  --require-day0-activity \
  --output-dir <run-dir>/simulation_best_maxent_micro__rewire_none_baseline \
  --log-level INFO
```

## Major scenario sweep

This command runs the built-in scenario set for the same generated setting:

```bash
python -m temporal_sbm.simulation \
  --run-dir <run-dir> \
  --setting-label maxent_micro__rewire_none \
  --scenario-set major \
  --beta-ff 0.20 \
  --beta-fr 0.04 \
  --beta-rf 0.16 \
  --beta-rr 0.02 \
  --sigma 0.35 \
  --gamma 0.10 \
  --num-replicates 256 \
  --seed 42 \
  --initial-seed-count 3 \
  --weight-mode log1p \
  --weight-scale auto \
  --tail-days 30 \
  --seed-scope farm_only \
  --seed-pool-mode observed_day0 \
  --require-day0-activity \
  --output-dir <run-dir>/simulation_scenarios_best_maxent_micro__rewire_none \
  --log-level INFO
```

## Output

Simulation runs write:

- per-snapshot comparison tables
- summary JSON files
- markdown reports
- dashboard and comparison figures
- aggregated reports when several synthetic panels are present for the same setting

The full argument list is available through `python -m temporal_sbm.simulation --help`.
