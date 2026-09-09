# Codebase structure

[Documentation index](README.md)

FILOS uses a flat Python package, terminal entry points, and separate directories
for configuration, research material, and local data.

## Python package

All application code lives in [`synthetic_filaments/`](../synthetic_filaments).
Its public API is collected in [`__init__.py`](../synthetic_filaments/__init__.py).

| Responsibility | Modules |
| :--- | :--- |
| Physical configuration and geometry | `config.py`, `geometry.py` |
| Plasma properties and calibrated opacity | `plasma.py`, `opacity_table.py` |
| Static synthesis and instrument response | `generator.py`, `render.py`, `degradation.py` |
| Background selection and local detection | `dynamic_background.py`, `detector.py` |
| Oscillation laws and time evolution | `oscillation.py`, `dynamics.py` |
| Analysis and visualization | `analysis.py`, `visualization.py` |
| Static/simulation persistence and video | `io.py`, `simulation_io.py`, `video.py` |
| Experiment configuration and jobs | `experiment_config.py`, `experiment_runner.py` |
| Shared paths, caches, and execution pools | `paths.py`, `cache.py`, `execution.py` |

The small reference assets in `synthetic_filaments/data/` are package data and
belong in version control. The package configuration includes its CSV and NPZ files.

## File placement

| Directory | What belongs here | Git policy |
| :--- | :--- | :--- |
| `scripts/` | Executable entry points and existing regression/benchmark tools | Include |
| `configs/` | Shared experiment presets | Include |
| `docs/` | Usage guides and documentation artwork | Include |
| `notebooks/` | Shared notebooks and their workflow instructions | Include |
| `notebooks/local/` | Personal exploration and unfinished notebooks | Ignore contents |
| `processed/` | Calibration provenance and reference products | Include |
| `FITS_files/` | Separately supplied observation files | Ignore data; include README |
| `FilamentSegmentator/` | Separately supplied detector assets | Ignore assets; include README |
| `simulations/` | Saved experiments, job records, datasets, and videos | Ignore |
| `scratch/` | Disposable benchmark and diagnostic output | Ignore |

`scratch/` and generated run directories are created by the workflows that use them.
Virtual environments, Python caches, and package build metadata are also ignored.

## Path conventions

Run terminal commands from the repository root. The package and several scripts
derive that root from their own file locations. In particular:

- The GUI launches `scripts/run_experiment_worker.py` by its repository-relative path.
- Default observations live in `FITS_files/20140101.h5`.
- The detector loads `FilamentSegmentator/models/detector_v1/`.
- The default profile is `configs/default_experiment.toml`.
- Notebook tooling targets `notebooks/07_refactored_forward_model.ipynb`.
- Scientific checks read the archive in `processed/heinzel_table1_extension_final/`.

Moving these entry points or inputs requires corresponding code changes. The
current layout keeps their expected locations intact.

## Organization changes

The calibration archive previously at the repository root now lives under
`processed/`, matching the scientific regression's existing lookup path. The
personal `testing/test.ipynb` file now lives at `notebooks/local/test.ipynb` and
remains ignored. Python source, experiment settings, packaged scientific assets,
and existing simulation outputs were preserved.
