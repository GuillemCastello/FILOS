# README example

Generated with [`configs/readme_example.toml`](../../../configs/readme_example.toml).

| Setting | Value |
| :--- | :--- |
| Prepared background | `20190105-0187-1039-1108.h5` |
| Source observation | `20190105.h5`, indices 187–575 inclusive |
| Original crop bounds | x: 1039–1487, y: 1108–1556 (exclusive upper bounds) |
| Native field | 448 × 448 pixels |
| Frames / physical cadence | 389 / 60 seconds |
| Physical time span | 388 minutes between first and last frames |
| Playback | 30 FPS, approximately 13 seconds |
| Filament / background / dynamics seeds | 1235 / 1236 / 1236 |
| Oscillation model | `luna_2022_curvature` |
| Threads | 2188 |
| Median longitudinal period | Approximately 61.2 minutes |
| Oscillation onset | 20 minutes |

The two MP4s are the original exports from the completed run. The inline animated
WebP contains all 389 frames at the same 30 FPS, with lossy compression for a
smaller download. No frames were repeated or interpolated to lengthen playback.

The preparation target is 400 frames. The supplied sequence stops at 389 because
candidate crops failed quiet-region screening near the end of the longer interval.
Every retained frame passed structural and pixel-validity checks; sampled frames
also passed detector screening. Original pixels and timestamps are unchanged.

The figures are exported from this run's realized static state:

- `frame_zero_comparison.png`: observed background and synthetic filament.
- `geometry_diagnostics.png`: thread geometry and distributions, with redundant
  histogram titles removed.
- `luna_dynamics_diagnostics.png`: expected periods from realized dip curvature
  and height, their distribution, and numerical summary.

Background identity and screening coverage are recorded in the
[library inventory](../../../backgrounds/library.json). Obtain the named background
file separately and run:

```bash
python scripts/run_experiment.py --config configs/readme_example.toml
```

Full HDF5 output stays under `simulations/experiments/`; only presentation assets
are included in the repository.
