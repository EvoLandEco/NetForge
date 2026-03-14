# NetForge

NetForge fits temporal stochastic block models to repeated network snapshots with `graph-tool`, samples synthetic panels on the same time grid, and writes diagnostics that compare the observed and generated networks. The command line entry point is `netforge`. The Python package path is `temporal_sbm`.

## What NetForge does

NetForge is built for datasets with repeated network snapshots, such as daily movement, trade, or contact panels. A standard workflow has three stages:

1. Fit a layered block model to the observed panel.
2. Generate synthetic panels from the fitted model.
3. Compare the observed and synthetic panels with reports, plots, and summary tables.

The repo also includes a transmission simulation module for run directories that contain generated panels.

## Installation

NetForge depends on `graph-tool`. Install `graph-tool` first from the official project:

- [graph-tool website](https://graph-tool.skewed.de/doc)
- [graph-tool installation guide](https://graph-tool.skewed.de/installation.html)

For many users, the cleanest setup is a fresh conda environment:

```bash
conda create --name gt -c conda-forge graph-tool
conda activate gt
pip install -e .
```

If `graph-tool` is already available in your environment, install NetForge with:

```bash
pip install -e .
```

## Quick start

The repo includes a Dutch toy dataset builder at `examples/toy_nl/build_toy_nl_dataset.py`. It writes the example inputs under `examples/toy_nl/processed_data/TOY_NL/`, which stays out of Git. The example uses the NL COROP basemap in `examples/public/nl_corop.geojson` and walks through the full NetForge workflow on a small panel.

Generate the toy dataset:

```bash
python examples/toy_nl/build_toy_nl_dataset.py
```

Fit the model:

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

Generate synthetic panels:

```bash
netforge generate \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --num-samples 3 \
  --seed 2026 \
  --posterior-partition-sweeps 25 \
  --posterior-partition-sweep-niter 10
```

Write diagnostics:

```bash
netforge report \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --detailed-diagnostics \
  --diagnostic-top-k 8 \
  --html-report
```

[`TUTORIAL.md`](TUTORIAL.md) walks through the toy data and the required input structure in more detail.

## Documentation site

The repository includes a GitHub Pages documentation setup built with MkDocs. The source files live under `site_docs/`, and the site configuration is in `mkdocs.yml`.

To preview the site locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```

To run the same strict docs build used in CI:

```bash
mkdocs build --strict
```

## Required dataset structure

NetForge reads each dataset from:

```text
<data-root>/<dataset>/
```

The loader requires these files:

```text
<data-root>/<dataset>/
  edges.csv
  node_features.npy
  node_schema.json
  node_map.csv
```

### `edges.csv`

The edge table must contain source, target, and timestamp columns. The default column names are:

- `u` for source node id
- `i` for target node id
- `ts` for snapshot index or timestamp

For weighted runs, the chosen weight column must also be present in `edges.csv`, unless you pass weights through a separate array path.

### `node_features.npy`

`node_features.npy` must be a two-dimensional numeric array. Row `n` must correspond to `node_id == n`. NetForge does not read padded node matrices.

### `node_schema.json`

The schema file must define the feature order used by `node_features.npy`:

```json
{
  "node_feature_columns_in_order": [
    "xco",
    "yco",
    "num_farms",
    "total_animals"
  ],
  "node_row_offset": 0
}
```

`node_row_offset` must be `0`. If you want spatial diagnostics, include either `xco` and `yco` or `centroid_x` and `centroid_y`.

### `node_map.csv`

The node map must contain `node_id` and `type`. You may add more metadata columns for labels, region codes, or other identifiers used in reports.

## Toy dataset

The generated Dutch toy dataset is small enough to inspect by hand, and it still carries the main patterns NetForge is meant to learn:

- distance decay in edge weights
- higher activity among larger farms
- lower activity on weekends
- lower activity on Dutch public holidays

The panel covers `2019-12-16` through `2020-01-12`. Regenerate it with [`examples/toy_nl/build_toy_nl_dataset.py`](examples/toy_nl/build_toy_nl_dataset.py).

The example directory also includes `dataset_manifest.json`. It documents the toy data, but it is not required by the NetForge loader.

## Output layout

`netforge fit` creates a run directory under:

```text
<data-root>/<dataset>/graph_tool_out/netforge/
```

That run directory is then used by `netforge generate`, `netforge report`, and the simulation tools. Common outputs include:

- `manifest.json` for run metadata
- `generated/` for synthetic samples
- `diagnostics/` for comparison tables, figures, and HTML reports
- simulation output directories when the simulation module is used

## Repository guide

- [`TUTORIAL.md`](TUTORIAL.md) gives a hands-on walkthrough with the toy dataset.
- [`examples/toy_nl/`](examples/toy_nl/) contains the toy data builder and the ignored output location.
- [`examples/public/nl_corop.geojson`](examples/public/nl_corop.geojson) contains the NL COROP basemap used by the example.
- [`tests/`](tests/) contains the unit tests.
