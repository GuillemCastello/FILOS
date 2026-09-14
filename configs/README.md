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

The oscillation center is placed along the filament with `center_spine_fraction`
(0 = start, 0.5 = middle, 1 = end) and at `center_height_km` (`"auto"` uses the
selected thread's center height). The longitudinal and transverse displacement
amplitudes set the motion along and across a thread at the center.
`half_strength_distance_km` sets the distance where those amplitudes fall to 50%;
larger values spread the oscillation farther. Strength decreases with each
thread's initial 3D distance, with a fixed decay shape, and becomes zero below 5%.
The default half-strength distance is 11,900 km.

For older configurations, replace `sphere_radius_km` and
`kernel_half_weight_radius_fraction` with `half_strength_distance_km` equal to
their product, and remove `kernel_power`. The decay power is now fixed at 1.1.
