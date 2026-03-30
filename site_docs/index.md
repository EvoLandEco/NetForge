# NetForge

NetForge fits temporal stochastic block models to repeated network snapshots, draws synthetic panels from the fitted model, and writes diagnostics that compare the observed and generated networks. It can also add a metadata layer with discrete tag vertices for a joint data-metadata fit.

These pages follow the workflow most people need first:

<div class="grid cards" markdown>

-   **Install**

    Set up `graph-tool`, install the package, and preview the docs locally.

    [Open the installation guide](installation.md)

-   **Quickstart**

    Build the toy dataset, then run the full workflow.

    [Open the quickstart](quickstart.md)

-   **Data format**

    See the exact file contract NetForge reads from disk.

    [Open the data format guide](data-format.md)

-   **API reference**

    Browse the public Python modules and command entry points.

    [Open the API reference](api/index.md)

</div>

## Workflow

NetForge has three main stages:

1. Fit a layered block model to the observed panel.
2. Generate one or more synthetic panels from the fitted run.
3. Write diagnostics that compare the observed and generated panels.

For run directories with generated panels, the repo also provides a transmission simulation module.

## Command line entry points

The main command line interface is `netforge`:

```bash
netforge fit ...
netforge generate ...
netforge report ...
```

The simulation entry point is exposed as a Python module:

```bash
python -m temporal_sbm.simulation ...
```

## Included example

The repository includes a Dutch toy dataset builder at `examples/toy_nl/build_toy_nl_dataset.py`. Run it to create `examples/toy_nl/processed_data/TOY_NL/` locally. The example uses the NL COROP basemap and encodes:

- distance decay in edge weights
- higher activity among larger farms
- lower activity on weekends
- lower activity on Dutch public holidays
- metadata tag inputs from COROP codes, size bins, grid cells, and `count_ft_*` columns

Use the [quickstart](quickstart.md) to run the full example.
