# creator.py

`creator.py` builds a simulated panel under the same broad assumptions used by the temporal generator:

- farm and region nodes carry metadata and numeric covariates
- blocks drive cross sectional structure
- block activity follows a stored activity path through time
- each snapshot is assembled from persist, reactivated, and new contacts
- weights are sampled after topology from a shifted negative binomial law

## Quick start

```bash
python creator.py \
  --output-root ./simulated_data \
  --dataset ideal_trade_sim \
  --profile medium \
  --seed 7
```

That writes:

- `simulated_data/ideal_trade_sim/edges.csv`
- `simulated_data/ideal_trade_sim/node_features.npy`
- `simulated_data/ideal_trade_sim/node_schema.json`
- `simulated_data/ideal_trade_sim/node_map.csv`
- `simulated_data/ideal_trade_sim/ml_ideal_trade_sim_headcount.npy`
- `simulated_data/ideal_trade_sim/manifest.json`
- `simulated_data/ideal_trade_sim/truth/...`

## Main scale presets

- `toy`: 120 farms, 8 regions, 60 days
- `small`: 600 farms, 18 regions, 140 days
- `medium`: 1800 farms, 36 regions, 280 days
- `large`: 4500 farms, 60 regions, 365 days

You can override any of those counts from the command line.

## Feeding the pipeline

The manifest records the exact paths and the timestamp settings. A typical fit call uses:

```bash
python pipeline_temporal_active_match.py fit \
  --data-root ./simulated_data \
  --dataset ideal_trade_sim \
  --weight-npy ./simulated_data/ideal_trade_sim/ml_ideal_trade_sim_headcount.npy \
  --ts-format ordinal \
  --ts-unit D
```

The sidecar weight file is indexed by the `idx` column in `edges.csv`.

## Truth files

`truth/` contains latent block parameters, per snapshot summaries, and per edge truth tables. Those files are there for recovery checks after fitting and posterior generation.
