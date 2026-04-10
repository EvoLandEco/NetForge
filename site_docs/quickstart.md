# Quickstart

This quickstart builds the Dutch toy dataset locally and runs the same three-stage workflow most NetForge users will use on real data.

## Toy dataset location

```text
examples/
  public/
    nl_corop.geojson
  toy_nl/
    build_toy_nl_dataset.py
    processed_data/
      TOY_NL/
        edges.csv
        node_features.npy
        node_map.csv
        node_schema.json
        dataset_manifest.json
```

Generate the source files first:

```bash
python examples/toy_nl/build_toy_nl_dataset.py
```

That command refreshes `examples/toy_nl/processed_data/TOY_NL/`.

## What the toy data encode

The toy panel covers `2019-12-16` through `2020-01-12` and includes 12 farm nodes across four Dutch COROP areas:

- `CR17` Utrecht
- `CR23` Groot-Amsterdam
- `CR24` Het Gooi en Vechtstreek
- `CR26` Agglomeratie 's-Gravenhage

The example is designed to show three signals clearly:

- larger farms exchange more weight
- longer routes exchange less weight
- weekends and Dutch public holidays carry less activity

The same files also feed the metadata layer used by the default fit:

- node-map columns such as `corop`, `coord_source`, `priority`, `CR_code`, `trade_species`, `diersoort`, `diergroep`, `diergroeplang`, `BtypNL`, and `bedrtype`
- `num_farms` and `total_animals` as quantile-bin tags
- coordinates as centroid-grid tags
- multi-value text fields, where `|` and `;` split one cell into several tag links

The matrix includes `count_ft_*` columns. They feed `ft_cosine`, and they also become `ft_tokens` when you add that field to `--metadata-fields`.

The basemap for geographic plots is stored at `examples/public/nl_corop.geojson`.

## Step 1: fit the model

```bash
netforge fit \
  --data-root examples/toy_nl/processed_data \
  --dataset TOY_NL \
  --directed \
  --weight-col trade \
  --weight-model discrete-poisson \
  --weight-transform none \
  --date-start 2019-12-16 \
  --date-end 2020-01-12
```

This fit keeps the daily trade layers and adds a `__metadata__` layer made of data-to-tag edges. The default toy run uses the richer node-map metadata set together with size-bin and centroid-grid tags. Use `--metadata-fields none` or `--no-joint-metadata-model` if you want a trade-only run.

Because `--fit-covariates` is not set here, the default fit is topology only across the trade edges and metadata links. Add `--fit-covariates dist_km mass_grav anim_grav ft_cosine` when you want the built in realized-edge annotations in the SBM fit.

By default, the run directory is:

```text
examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge/
```

## Step 2: generate synthetic panels

```bash
netforge generate \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --num-samples 3 \
  --seed 2026 \
  --posterior-partition-sweeps 25 \
  --posterior-partition-sweep-niter 10
```

This writes generated panels under `generated/markov_turnover__proposal_sbm__micro__rewire_none/` inside the run directory, with one `sample_####/` directory per draw.

## Step 3: write diagnostics

```bash
netforge report \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --detailed-diagnostics \
  --diagnostic-top-k 8 \
  --html-report
```

The report stage writes tables and figures under `diagnostics/`, including the HTML report and the cross-sample summary CSV files built from the saved generation batches.

## Before you swap in your own data

Read the [data format guide](data-format.md) before building a new dataset. NetForge expects one fixed dataset layout, and the node matrix uses direct indexing: row `n` must match `node_id == n`.
