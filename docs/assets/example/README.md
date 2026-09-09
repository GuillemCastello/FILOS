# README example

Generated with the prepared-background workflow using
[`configs/readme_example.toml`](../../../configs/readme_example.toml).

| Setting | Value |
| :--- | :--- |
| Prepared background | `20140101-0341-1128-1195.h5` |
| Source observation | `20140101.h5`, indices 341–460 inclusive |
| Original crop bounds | x: 1128–1576, y: 1195–1643 (exclusive upper bounds) |
| Native field | 448 × 448 pixels |
| Frames / physical cadence | 120 / 60 seconds |
| Physical time span | 119 minutes between first and last frames |
| Filament / background / dynamics seeds | 1235 / 1236 / 1236 |
| Oscillation model | `luna_2022_curvature` |
| Threads | 2188 |
| Median longitudinal period | Approximately 61.2 minutes |
| Oscillation onset | 20 minutes |

The two MP4s are the unmodified exports from the completed run: 120 frames at
30 FPS, lasting four seconds. The GIF is a smaller preview played three times
slower, with reduced frame rate, spatial resolution, and color depth. Its display
processing does not affect the underlying HDF5 simulation data.

`frame_zero_comparison.png` and `geometry_diagnostics.png` are the run's original
static diagnostic exports. Geometry profiles show a sample of threads; histograms
summarize the realized population.

The background's identity and preparation coverage are recorded in the
[library inventory](../../../backgrounds/library.json). Obtain that background
file separately and run:

```bash
python scripts/run_experiment.py --config configs/readme_example.toml
```

Full HDF5 simulation output stays under `simulations/experiments/`; only these
small presentation assets are included in the repository.
