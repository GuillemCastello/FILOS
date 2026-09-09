# Scripts

[Project README](../README.md) · [Documentation](../docs/README.md)

Run commands from the repository root. Entry points remain together here because
several resolve the project root from their location, and the GUI launches its
worker by path.

## Interface and workers

| Script | Purpose |
| :--- | :--- |
| `experiment_gui.py` | Local Streamlit experiment interface |
| `run_experiment_worker.py` | Execute an immutable configuration snapshot; normally launched by the GUI |

```bash
uv run --extra gui --extra detector streamlit run scripts/experiment_gui.py
```

See [local inputs](../README.md#local-inputs) for observations, detector assets,
FFmpeg, and GPU selection.

## Regression checks

| Script | Coverage |
| :--- | :--- |
| `run_scientific_regression.py` | Calibrated opacity checksum and published anchors, static reference hashes, numerical checks, and optional dynamics/persistence |
| `run_experiment_regression.py` | Configuration, previews, both dynamics modes, detached workers, versioning, cloning, and retained failures |

```bash
uv run --extra detector python scripts/run_scientific_regression.py --include-dynamics
uv run --extra gui --extra detector python scripts/run_experiment_regression.py
```

These checks use local observation and detector assets. Scientific regression
expects the canonical background for its exact static hashes, and reads
calibration provenance under `processed/heinzel_table1_extension_final/`.
The GUI workflow regression also requires FFmpeg and FFprobe.

## Benchmarks

```bash
uv run --extra detector python scripts/benchmark_dynamics.py --frames 240

uv run --extra detector python scripts/benchmark_preview.py \
  --h5 FITS_files/20140101.h5 \
  --output scratch/preview_benchmark.json
```

`benchmark_dynamics.py` times simulation and both MP4 exports in automatically
removed temporary storage. `benchmark_preview.py` measures fixed-seed cold/warm
preview stages, cache behavior, scientific hashes, and process-tree memory.
See [performance controls](../docs/PERFORMANCE.md) for options.

## Scientific diagnostics

```bash
uv run python scripts/plot_luna_height_model.py
```

Exports the 1–100 Mm gravity, cut-off, period, and seismological-inversion table
and plots. Use `--help` to inspect output options.

## Notebook tooling

| Script | Purpose |
| :--- | :--- |
| `build_refactored_notebook.py` | Generate `notebooks/07_refactored_forward_model.ipynb` from source cells |
| `execute_refactored_notebook.py` | Execute that notebook top to bottom in place |

See the [notebook workflow](../notebooks/README.md) for dependencies and commands.
