# NetForge Tutorial

This tutorial walks through the Dutch toy dataset builder, the metadata inputs it writes, and the file structure NetForge expects.

Before running the pipeline, install `graph-tool` from the official project site: [graph-tool.skewed.de](https://graph-tool.skewed.de/doc). The platform-specific install guide is [graph-tool installation instructions](https://graph-tool.skewed.de/installation.html).

## 1. Inspect the example layout

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

Generate the toy dataset first:

```bash
python examples/toy_nl/build_toy_nl_dataset.py
```

That command writes the files under [`examples/toy_nl/processed_data/TOY_NL/`](examples/toy_nl/processed_data/TOY_NL/). The directory is ignored, so the generated files stay local.

## 2. Match the required input structure

NetForge reads each dataset from:

```text
<data-root>/<dataset>/
```

For `--data-root examples/toy_nl/processed_data --dataset TOY_NL`, the loader reads these files:

- `edges.csv`
- `node_features.npy`
- `node_schema.json`
- `node_map.csv`

All four files are required.

### Edge table

The edge table must contain source, target, and timestamp columns. The default names are `u`, `i`, and `ts`. If you fit weights, the weight column must be present as well or supplied through `--weight-npy`.

This example keeps a few extra inspection columns. The pipeline ignores them unless you point a CLI flag at them.

```csv
u,i,ts,date,trade,distance_km,gravity_mean,is_weekend,is_public_holiday
0,1,737409,2019-12-16,2,6.14,0.738246,False,False
0,4,737409,2019-12-16,1,32.148,0.353966,False,False
1,0,737409,2019-12-16,1,6.14,0.746601,False,False
```

The toy data use ordinal timestamps, so `ts` is `date.toordinal()`. If your own data use Unix timestamps, switch to `--ts-format unix` and set `--ts-unit`.

### Node feature matrix

The node matrix must be a two-dimensional numeric array. Row `n` must match `node_id == n`. NetForge does not accept a leading padding row.

- Node id `0` is stored in row `0`
- Node id `11` is stored in row `11`
- With 12 nodes, the toy matrix has shape `(12, 8)`

The toy schema also supplies the inputs used by the default metadata layer:

- `num_farms` and `total_animals` for quantile-bin tags
- `xco` and `yco` for centroid-grid tags
- `count_ft_*` columns for farm-type token tags

### Node schema JSON

The schema JSON must list the node feature columns in the exact order used by `node_features.npy`. It also needs either `xco` and `yco` or `centroid_x` and `centroid_y`.

```json
{
  "node_feature_columns_in_order": [
    "xco",
    "yco",
    "num_farms",
    "total_animals",
    "count_ft_cattle",
    "count_ft_pig",
    "count_animal_cattle",
    "count_animal_pig"
  ],
  "node_row_offset": 0
}
```

### Node map CSV

The node map should contain `node_id` and `type`. In this example it also stores `ubn`, `corop`, and a readable label.

```csv
node_id,node_label,type,ubn,corop,corop_name
0,CR17_farm_1,Farm,TOYNL0000,CR17,Utrecht
1,CR17_farm_2,Farm,TOYNL0001,CR17,Utrecht
2,CR17_farm_3,Farm,TOYNL0002,CR17,Utrecht
```

`type` marks the partition role of each data node. Fields such as `corop` can also be named in `--metadata-fields`. NetForge then creates one metadata tag vertex per token and links the matching data nodes to it in the `__metadata__` layer.

For the NL map view, this repository keeps the basemap at [`examples/public/nl_corop.geojson`](examples/public/nl_corop.geojson). The bundled example looks for the basemap there.

## 3. See what the toy data encode

The toy dataset has 12 farm nodes in four COROP areas:

- `CR17` Utrecht
- `CR23` Groot-Amsterdam
- `CR24` Het Gooi en Vechtstreek
- `CR26` Agglomeratie 's-Gravenhage

The sample window runs from `2019-12-16` through `2020-01-12`.

Each daily directed edge weight is drawn from a Poisson mean built from:

```text
0.0024 * sqrt(total_animals_u * total_animals_i) * exp(-distance_m / 45000) * calendar_scale
```

`calendar_scale` is lower on weekends and lower again on the Dutch public holidays `2019-12-25`, `2019-12-26`, and `2020-01-01`.

This gives the example three clear trade signals:

- larger farms exchange more weight
- longer routes exchange less weight
- weekends and holidays carry less total activity

The same files also support the metadata layer used by the default fit:

- `corop` becomes a region tag
- `num_farms` and `total_animals` become quantile-bin tags
- coordinates become centroid-grid tags
- `count_ft_*` columns become farm-type token tags

## 4. Fit the model

Run the fit stage on the toy data:

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

This run fits the trade panel together with a metadata layer named `__metadata__`. Each distinct metadata token becomes its own tag vertex, and NetForge adds data-to-tag edges in that layer. Use `--metadata-fields none` or `--no-joint-metadata-model` to skip those vertices and fit the trade graph alone.

By default this writes the run to:

```text
examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge/
```

## 5. Generate synthetic panels

After fitting, draw a few synthetic samples:

```bash
netforge generate \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --num-samples 3 \
  --seed 2026 \
  --posterior-partition-sweeps 25 \
  --posterior-partition-sweep-niter 10
```

## 6. Write reports

Use the saved run directory to produce diagnostics:

```bash
netforge report \
  --run-dir examples/toy_nl/processed_data/TOY_NL/graph_tool_out/netforge \
  --detailed-diagnostics \
  --diagnostic-top-k 8 \
  --html-report
```

The report stage writes figures and tables under `diagnostics/`. With the bundled basemap in place, the HTML report can include the NL map view for the toy nodes.

## 7. Apply the same structure to your own data

To swap in a real dataset, keep the same contract:

- put the dataset under `<data-root>/<dataset>/`
- store edges in `edges.csv`
- store the node matrix in `node_features.npy`
- store the feature order in `node_schema.json`
- store node metadata in `node_map.csv`

If you plan to use the joint metadata fit, make sure the files also carry the metadata you want to turn into tags.

Start from the toy example and swap pieces one at a time.
