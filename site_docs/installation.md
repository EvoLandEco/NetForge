# Installation

## 1. Install graph-tool

NetForge depends on `graph-tool` for model fitting and network sampling. Install `graph-tool` first from the official project:

- [graph-tool website](https://graph-tool.skewed.de/doc)
- [graph-tool installation guide](https://graph-tool.skewed.de/installation.html)

For many users, a fresh conda environment is the shortest path:

```bash
conda create --name gt -c conda-forge graph-tool
conda activate gt
```

## 2. Install NetForge

From the repository root:

```bash
pip install -e .
```

The command line entry point is `netforge`. The Python package path is `temporal_sbm`.

## 3. Preview the documentation locally

The repository includes a MkDocs setup for local preview and GitHub Pages deployment. To preview the docs locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```

This starts a local preview server, usually at `http://127.0.0.1:8000/`.

If you want the same check used in CI, run:

```bash
mkdocs build --strict
```

## 4. Publish on GitHub Pages

The repository includes a GitHub Actions workflow at `.github/workflows/docs.yml`.

Pull requests build the docs site. Pushes to `main` and manual runs build the site and deploy it to GitHub Pages.

The workflow:

1. installs the docs dependencies
2. builds the site with `mkdocs build --strict`
3. uploads the rendered site as a GitHub Pages artifact
4. deploys the artifact with the official Pages deployment action

To turn the site on in GitHub:

1. Open the repository settings.
2. Open the **Pages** section.
3. Set the source to **GitHub Actions**.

After that, pushes to `main` will publish the site.
