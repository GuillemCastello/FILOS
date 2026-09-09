#!/usr/bin/env python3
"""Execute the refactored notebook in place with a reproducible kernel."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/07_refactored_forward_model.ipynb"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/filament-modelling-jupyter")

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

client = NotebookClient(
    notebook,
    timeout=1_200,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
)
client.execute()
nbformat.validate(notebook)
with NOTEBOOK.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)
print(NOTEBOOK)
