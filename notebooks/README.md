# Notebooks

[Project README](../README.md)

Shared reference notebooks belong here. Keep personal experiments in `local/`,
whose contents are ignored by Git.

## Reference workflow

After the standard pip installation, generate the reference notebook from the
repository root:

```bash
python scripts/build_refactored_notebook.py
```

This writes `notebooks/07_refactored_forward_model.ipynb`, overwriting that file
if it already exists. It does not run a simulation. Open it in a notebook-capable
editor using the project's Python environment, or execute it in place:

```bash
python scripts/execute_refactored_notebook.py
```

Execution uses the prepared background selected by `configs/default_experiment.toml`.
The full workflow exports videos through FFmpeg. For reduced validation (three
frames, no videos, output under `scratch/`):

```bash
FILAMENT_NOTEBOOK_FAST=1 python scripts/execute_refactored_notebook.py
```

The executor uses the `python3` Jupyter kernel. If your editor or Jupyter setup
points that kernel at another environment, register this project's environment
before execution:

```bash
python -m ipykernel install --user --name python3 --display-name "Python (FILOS)"
```
