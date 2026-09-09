# Experiment configuration

[`default_experiment.toml`](default_experiment.toml) supplies starting values for
new GUI experiments and the reference notebook.
[`readme_example.toml`](readme_example.toml) reproduces the README demonstration.

| Section | Controls |
| :--- | :--- |
| `experiment` | Name and description |
| `inputs` | Prepared background file or library directory |
| `static` | Seed, geometry, plasma, opacity, and instrument response |
| `dynamic_background` | Library selection seed, starting frame, and frame step |
| `dynamics` | Oscillation mode, amplitudes, periods, damping, and frame count |
| `export` | Masks, contrast, and dataset compression |
| `video` | Playback frame rate and velocity visualization |

Spatial dimensions and simulation cadence are read from the background file;
playback FPS is independent of physical cadence. Save individual experiments
through the GUI or run a preset with:

```bash
python scripts/run_experiment.py --config configs/readme_example.toml
```

See [data and outputs](../docs/DATA.md).
