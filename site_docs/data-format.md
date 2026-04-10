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

Extra columns are fine. NetForge ignores them unless a command line flag points at them. The built in fit covariates are derived inside the pipeline from timestamps and the node matrix rather than read from extra edge-table columns.

When you use `--weight-npy`, the weight vector is aligned to the input edge rows and may include one leading padding entry.

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

The pipeline also derives several built in annotations from this matrix:

- coordinates feed `dist_km` and `centroid_grid`
- `num_farms` feeds `mass_grav` and `num_farms_bin`
- `total_animals` or `total_diergroep_*` feeds `anim_grav` and `total_animals_bin`
- positive `count_ft_*` columns feed `ft_cosine` and, when requested, `ft_tokens`

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

The schema must include either:

- `xco` and `yco`
- `centroid_x` and `centroid_y`

## `node_map.csv`

The node map must include:

- `node_id`
- `type`

You may also include labels, region codes, external ids, species fields, or any other metadata you want to keep next to the node ids.

### Example

```csv
node_id,node_label,type,ubn,corop,coord_source,priority,trade_species,BtypNL
0,CR17_farm_1,Farm,TOYNL0000,CR17,registry_point,standard,cattle,Melkvee
1,CR17_farm_2,Farm,TOYNL0001,CR17,survey_offset,high,cattle|pig,Gemengd
2,CR17_farm_3,Farm,TOYNL0002,CR17,survey_offset,medium,pig,Varkens
```

The `type` column marks the partition role of each data node. Extra metadata columns can also be named in `--metadata-fields` so NetForge can turn them into metadata tag vertices. Text fields split on `|` and `;`. Numeric fields with many distinct values are binned into quantile labels.

## Metadata tag layer

With the default fit settings, NetForge builds a joint data-metadata multilayer graph. It creates one metadata tag vertex per token and stores the data-to-tag edges in a layer named `__metadata__`.

When the joint metadata model is active and `--fit-covariates` is not set, the fit is topology only over the trade and metadata edges. Set `--fit-covariates` when you want the realized-edge annotations in the SBM fit.

The default metadata fields are:

- `corop`
- `coord_source`
- `priority`
- `CR_code`
- `num_farms_bin`
- `total_animals_bin`
- `centroid_grid`
- `trade_species`
- `diersoort`
- `diergroep`
- `diergroeplang`
- `BtypNL`
- `bedrtype`

Those tags come from these inputs:

- `corop`, `coord_source`, `priority`, `CR_code`, `trade_species`, `diersoort`, `diergroep`, `diergroeplang`, `BtypNL`, and `bedrtype` read from `node_map.csv`
- `num_farms_bin` and `total_animals_bin` read from numeric columns in `node_features.npy`
- `centroid_grid` reads from `xco` and `yco` or `centroid_x` and `centroid_y`

`trade_species`, `diersoort`, `diergroep`, and `diergroeplang` may contain a `|` or `;` separated list of tokens. NetForge splits those cells into one data-to-tag edge per token.

You can request `ft_tokens`. It reads positive `count_ft_*` columns in `node_features.npy`.

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
