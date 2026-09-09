# Notebooks

[Project README](../README.md)

Shared reference notebooks belong here. Keep personal experiments in `local/`,
whose contents are ignored by Git. The former `testing/test.ipynb` scratch file
has been moved there unchanged; it is an empty placeholder.

## Reference workflow

The reference notebook is generated from the source cells in
[`scripts/build_refactored_notebook.py`](../scripts/build_refactored_notebook.py).
It is not included in this checkout. Generate it from the repository root:

```bash
uv run --extra notebook python scripts/build_refactored_notebook.py
```

This writes `notebooks/07_refactored_forward_model.ipynb`, overwriting that file
if it already exists. It does not run a simulation. Open it in a notebook-capable
editor using the project's Python environment, or execute it in place:

```bash
uv run --extra notebook --extra detector python scripts/execute_refactored_notebook.py
```

Execution requires the [local observation and detector assets](../docs/DATA.md).
The notebook's full workflow also exports videos through FFmpeg. For its reduced
validation mode (three frames, no videos, output under `scratch/`):

```bash
FILAMENT_NOTEBOOK_FAST=1 uv run --extra notebook --extra detector \
  python scripts/execute_refactored_notebook.py
```

For pip environments, install `python -m pip install -e '.[notebook,detector]'`
and replace `uv run --extra notebook --extra detector python` with `python`.
