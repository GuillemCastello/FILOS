# Scripts

[Project README](../README.md) · [Documentation](../docs/README.md)

Activate the virtual environment and run commands from the repository root.

## Normal use

| Script | Purpose |
| :--- | :--- |
| `experiment_gui.py` | Local Streamlit experiment interface |
| `run_experiment.py` | Run an experiment TOML through preview and production |
| `derive_segmentation_masks.py` | Derive detailed opacity masks for every saved frame and save a comparison plot |
| `prepare_backgrounds.py` | Prepare portable quiet-Sun sequences from full-disk observations |
| `run_experiment_worker.py` | Internal worker launched by the GUI or command-line runner |

```bash
python -m streamlit run scripts/experiment_gui.py
python scripts/run_experiment.py --config configs/readme_example.toml
python scripts/prepare_backgrounds.py
python scripts/derive_segmentation_masks.py /path/to/simulation.h5 --threshold 0.1
```

Ordinary runs only need prepared backgrounds and FFmpeg. Preparation uses source
observations and optionally the detector; see the [background guide](../docs/BACKGROUNDS.md).
Mask generation only needs an existing simulation; see the
[segmentation guide](../docs/SEGMENTATION.md) for outputs and GUI controls.

## Checks

```bash
python -m unittest discover -s tests -v
python scripts/run_scientific_regression.py --include-dynamics
python scripts/run_experiment_regression.py
```

| Check | Inputs and coverage |
| :--- | :--- |
| Background unit tests | Temporary synthetic files; portability, timing, validation, later-frame screening, and detector-free imports |
| Scientific regression | Included three-frame reference fixture and calibration archive; exact image hashes, numerical checks, and dynamics/persistence |
| Experiment regression | Local prepared library and FFmpeg/FFprobe; previews, caches, workers, both dynamics modes, versioning, cloning, and failures |

Scientific regression expects its historical fixture for exact image hashes.
Its `--dynamics-frames` value cannot exceed the fixture's three frames.

## Benchmarks and diagnostics

```bash
python scripts/benchmark_dynamics.py --frames 120
python scripts/benchmark_preview.py --output scratch/preview_benchmark.json
python scripts/plot_luna_height_model.py
```

The first two use prepared backgrounds; see [performance controls](../docs/PERFORMANCE.md).
The height-model script exports gravity, cut-off, period, and inversion tables and
figures to `scratch/luna_height_model/` by default.

## Notebook tooling

`build_refactored_notebook.py` generates the reference notebook from source cells.
`execute_refactored_notebook.py` executes it in place. See the
[notebook workflow](../notebooks/README.md).
