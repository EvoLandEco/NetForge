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

The repo includes a Dutch toy dataset builder at `examples/toy_nl/build_toy_nl_dataset.py` and the example files under `examples/toy_nl/processed_data/TOY_NL/`. Running the builder refreshes that example data. The toy node map carries metadata columns such as `corop`, `coord_source`, `priority`, `CR_code`, `trade_species`, `diersoort`, `diergroep`, `diergroeplang`, `BtypNL`, and `bedrtype`. The node matrix carries the size and coordinate fields used for bin and grid tags. Positive `count_ft_*` columns are also present if you want `ft_tokens`.

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

This fit builds the usual trade layers and a `__metadata__` layer. By default the toy run tags nodes by region, coordinate source, priority, CR code, size bins, centroid-grid cell, species labels, and business type. NetForge creates one metadata tag vertex per token and links each data node to the tags it carries. Text fields split on `|` and `;`, so values such as `cattle|pig` or `RUNDVEE;VARKENS` become separate tag links. Use `--metadata-fields none` or `--no-joint-metadata-model` if you want a trade-only fit.

Because `--fit-covariates` is left unset here, the default fit is topology only across the trade edges and metadata links. Add `--fit-covariates dist_km mass_grav anim_grav ft_cosine` when you want the built in realized-edge annotations in the SBM fit. `ft_cosine` is available when the node matrix includes positive `count_ft_*` columns.

Generate synthetic panels:

```bash
netforge generate \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --num-samples 3 \
  --seed 2026 \
  --posterior-partition-sweeps 25 \
  --posterior-partition-sweep-niter 10
```

By default this generation batch is stored under `generated/markov_turnover__proposal_sbm__micro__rewire_none/`, with one `sample_####/` directory per draw.

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

Extra edge columns are fine for inspection. The built in fit covariates are derived inside the pipeline from timestamps and the node matrix rather than read from extra edge-table columns.

### `node_features.npy`

`node_features.npy` must be a two-dimensional numeric array. Row `n` must correspond to `node_id == n`. NetForge does not read padded node matrices.

The pipeline also derives several built in annotations from this matrix:

- coordinates feed `dist_km` and `centroid_grid`
- `num_farms` feeds `mass_grav` and `num_farms_bin`
- `total_animals` or `total_diergroep_*` feeds `anim_grav` and `total_animals_bin`
- positive `count_ft_*` columns feed `ft_cosine` and, when requested, `ft_tokens`

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

`node_row_offset` must be `0`. The schema must include either `xco` and `yco` or `centroid_x` and `centroid_y`. The fit uses those coordinates to build distance covariates and centroid-grid metadata tags.

### `node_map.csv`

The node map must contain `node_id` and `type`. The `type` column marks the partition role of each data node. Extra columns such as region codes, labels, external ids, or species fields can also feed the metadata layer when you list them in `--metadata-fields`. Text values may contain a single token or a `|` / `;` separated list of tokens. Numeric columns with many distinct values are binned into quantile labels. NetForge turns each distinct token into a metadata tag vertex and links matching data nodes to it in the `__metadata__` layer.

## Toy dataset

The generated Dutch toy dataset is small enough to inspect by hand, and it carries the main patterns NetForge is meant to learn:

- distance decay in edge weights
- higher activity among larger farms
- lower activity on weekends
- lower activity on Dutch public holidays

It also includes the inputs for the default metadata layer:

- region and business metadata in `node_map.csv`, including `corop`, `coord_source`, `priority`, `CR_code`, `trade_species`, `diersoort`, `diergroep`, `diergroeplang`, `BtypNL`, and `bedrtype`
- size and coordinate fields in `node_features.npy`, which become quantile-bin and centroid-grid tags
- multi-value text fields in `node_map.csv`, where `|` and `;` split one cell into several tag links

The toy node matrix includes `count_ft_*` columns. Add `ft_tokens` to `--metadata-fields` when you want farm-type token tags.

The panel covers `2019-12-16` through `2020-01-12`. Regenerate it with [`examples/toy_nl/build_toy_nl_dataset.py`](examples/toy_nl/build_toy_nl_dataset.py).

The example directory also includes `dataset_manifest.json`. It documents the toy data, but it is not required by the NetForge loader.

## Output layout

`netforge fit` creates a run directory under:

```text
<data-root>/<dataset>/graph_tool_out/netforge/
```

That run directory is then used by `netforge generate`, `netforge report`, and the simulation tools. Common outputs include:

- `manifest.json` for run metadata, including requested metadata fields, fields used, link counts, tag counts, and tag-construction settings
- `generated/<setting-label>/setting_manifest.json` for each generation batch
- `generated/<setting-label>/sample_####/` for each synthetic panel, including `synthetic_edges.csv`, `sample_manifest.json`, `node_partition.csv`, and temporal summaries when the temporal generator is active
- `diagnostics/` for comparison tables, figures, HTML reports, and cross-sample summary CSV files such as `all_sample_runs_summary.csv`, `all_samples_summary.csv`, and `setting_posterior_summary.csv`
- `logs/` for command logs and the log summary artifacts written by the CLI
- simulation output directories when the simulation module is used

## Repository guide

- [`TUTORIAL.md`](TUTORIAL.md) gives a hands-on walkthrough with the toy dataset.
- [`examples/toy_nl/`](examples/toy_nl/) contains the toy data builder and the example dataset files.
- [`examples/public/nl_corop.geojson`](examples/public/nl_corop.geojson) contains the NL COROP basemap used by the example.
- [`tests/`](tests/) contains the unit tests.

## Acknowledgements

This project uses Codex for automated code review, version control work, and CI/CD support.
