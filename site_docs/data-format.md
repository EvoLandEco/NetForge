# Data format

NetForge reads each dataset from a single directory:

```text
<data-root>/<dataset>/
```

The loader expects these files:

```text
<data-root>/<dataset>/
  edges.csv
  node_features.npy
  node_schema.json
  node_map.csv
```

`dataset_manifest.json` may be present for your own notes or for example data, but the loader does not require it.

## `edges.csv`

The edge table must contain:

- `u`: source node id
- `i`: target node id
- `ts`: snapshot index or timestamp

For weighted runs, the weight column you pass to `--weight-col` must also be present in `edges.csv`, unless you supply weights through `--weight-npy`.

Extra columns are fine. NetForge ignores them unless a command line flag points at them.

### Example

```csv
u,i,ts,date,trade,distance_km
0,1,737409,2019-12-16,2,6.14
0,4,737409,2019-12-16,1,32.148
1,0,737409,2019-12-16,1,6.14
```

## `node_features.npy`

`node_features.npy` must be a two-dimensional numeric array.

The indexing rule is strict:

- row `0` stores features for `node_id == 0`
- row `11` stores features for `node_id == 11`

If the dataset has `N` nodes, the matrix must have `N` rows. NetForge does not read padded node matrices.

With the default metadata settings, NetForge also reads `num_farms`, `total_animals`, the coordinate pair, and any `count_ft_*` columns from this matrix to build metadata tag nodes.

## `node_schema.json`

The schema file records the feature order used by `node_features.npy`.

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

`node_row_offset` must be `0`.

For geographic diagnostics, include either:

- `xco` and `yco`
- `centroid_x` and `centroid_y`

## `node_map.csv`

The node map must include:

- `node_id`
- `type`

You may also include labels, region codes, external ids, or any other metadata you want to keep next to the node ids.

### Example

```csv
node_id,node_label,type,ubn,corop,corop_name
0,CR17_farm_1,Farm,TOYNL0000,CR17,Utrecht
1,CR17_farm_2,Farm,TOYNL0001,CR17,Utrecht
2,CR17_farm_3,Farm,TOYNL0002,CR17,Utrecht
```

The `type` column marks the partition role of each data node. Extra metadata columns can also be named in `--metadata-fields` so NetForge can turn them into metadata tag vertices.

## Metadata tag layer

With the default fit settings, NetForge builds a joint data-metadata multilayer graph. It creates one metadata tag vertex per token and stores the data-to-tag edges in a layer named `__metadata__`.

The default metadata fields are:

- `corop`
- `num_farms_bin`
- `total_animals_bin`
- `centroid_grid`
- `ft_tokens`

Those tags come from these inputs:

- `corop` reads from `node_map.csv`
- `num_farms_bin` and `total_animals_bin` read from numeric columns in `node_features.npy`
- `centroid_grid` reads from `xco` and `yco` or `centroid_x` and `centroid_y`
- `ft_tokens` reads from positive `count_ft_*` columns in `node_features.npy`

Use `--metadata-fields` to choose a different set of metadata tags, `--metadata-fields none` to skip metadata tags while keeping the flag in the command, or `--no-joint-metadata-model` to fit the trade graph without the metadata layer.

## Timestamps

The default timestamp format is ordinal days. If your edge table uses Unix time, pass:

- `--ts-format unix`
- `--ts-unit s`, `ms`, `us`, `ns`, or `D`

You can slice the panel with:

- `--ts-start` and `--ts-end`
- `--date-start` and `--date-end`

## Build a new dataset safely

The bundled toy dataset is a good template. Before you create a larger dataset:

1. Copy the four required files into a new dataset directory.
2. Confirm that `node_features.npy` has one row per node id.
3. Check that `node_schema.json` lists columns in the same order as the matrix.
4. Run `netforge fit --help` and choose the timestamp and weight flags that match your input table.
