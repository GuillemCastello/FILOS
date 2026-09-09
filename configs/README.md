# Experiment configuration

[`default_experiment.toml`](default_experiment.toml) supplies the starting values
for new GUI experiments and the generated reference notebook.

| Section | Controls |
| :--- | :--- |
| `experiment` | Name and description |
| `inputs` | HDF5 background path |
| `static` | Seed, geometry, plasma, opacity, and instrument response |
| `dynamic_background` | Crop selection, frame selection, and detector settings |
| `dynamics` | Oscillation mode, amplitudes, periods, damping, and cadence |
| `export` | Masks, contrast, and dataset compression |
| `video` | Frame rate and velocity visualization |

Save individual experiments through the GUI; their configurations live under
`simulations/experiments/`. Keep reusable shared presets here. See
[data and outputs](../docs/DATA.md) for required inputs and saved-run locations.
