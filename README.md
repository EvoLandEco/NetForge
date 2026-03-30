# NetForge

NetForge fits temporal stochastic block models to repeated network snapshots with `graph-tool`, samples synthetic panels on the same time grid, and writes diagnostics that compare the observed and generated networks. It can fit the trade panel on its own or as a joint data-metadata multilayer graph with discrete metadata tag vertices. The command line entry point is `netforge`. The Python package path is `temporal_sbm`.

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

The repo includes a Dutch toy dataset builder at `examples/toy_nl/build_toy_nl_dataset.py`. It writes the example inputs under `examples/toy_nl/processed_data/TOY_NL/`, which stays out of Git. The example uses the NL COROP basemap in `examples/public/nl_corop.geojson` and is set up for the joint data-metadata fit: COROP labels live in `node_map.csv`, farm size inputs live in `node_features.npy`, and the default metadata layer turns them into tag nodes.

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
  --metadata-fields corop num_farms_bin total_animals_bin centroid_grid ft_tokens \
  --date-start 2019-12-16 \
  --date-end 2020-01-12
```

This fit builds the usual trade layers and a `__metadata__` layer. NetForge creates one metadata tag vertex per token and links each data node to the tags it carries. Use `--metadata-fields none` or `--no-joint-metadata-model` if you want a trade-only fit.

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

With the default metadata settings, NetForge also reads `num_farms`, `total_animals`, the coordinate pair, and any `count_ft_*` columns from this matrix to build metadata tag nodes.

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

The node map must contain `node_id` and `type`. The `type` column marks the partition role of each data node. Extra columns such as region codes, labels, or external ids can also feed the metadata layer when you list them in `--metadata-fields`. NetForge turns each distinct token into a metadata tag vertex and links matching data nodes to it in the `__metadata__` layer.

## Toy dataset

The generated Dutch toy dataset is small enough to inspect by hand, and it still carries the main patterns NetForge is meant to learn:

- distance decay in edge weights
- higher activity among larger farms
- lower activity on weekends
- lower activity on Dutch public holidays

It also includes the inputs for the default metadata layer:

- COROP codes in `node_map.csv`
- farm size columns in `node_features.npy`
- `count_ft_*` columns that become farm-type tokens

The panel covers `2019-12-16` through `2020-01-12`. Regenerate it with [`examples/toy_nl/build_toy_nl_dataset.py`](examples/toy_nl/build_toy_nl_dataset.py).

The example directory also includes `dataset_manifest.json`. It documents the toy data, but it is not required by the NetForge loader.

## Output layout

`netforge fit` creates a run directory under:

```text
<data-root>/<dataset>/graph_tool_out/netforge/
```

That run directory is then used by `netforge generate`, `netforge report`, and the simulation tools. Common outputs include:

- `manifest.json` for run metadata, including metadata-layer settings and tag counts
- `generated/` for synthetic samples
- `diagnostics/` for comparison tables, figures, and HTML reports
- simulation output directories when the simulation module is used

## Repository guide

- [`TUTORIAL.md`](TUTORIAL.md) gives a hands-on walkthrough with the toy dataset.
- [`examples/toy_nl/`](examples/toy_nl/) contains the toy data builder and the ignored output location.
- [`examples/public/nl_corop.geojson`](examples/public/nl_corop.geojson) contains the NL COROP basemap used by the example.
- [`tests/`](tests/) contains the unit tests.

## Acknowledgements

This project uses Codex for automated code review, version control work, and CI/CD support.
