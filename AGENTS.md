# AGENTS.md — working with FILOS

Guidance for coding agents (and humans) using or modifying FILOS, the solar
**Fil**ament **O**scillation **S**imulator. FILOS synthesizes Hα filaments
(measured spine → 3D circular-dip threads → plasma/opacity → GONG-like
instrument degradation), inserts them into real, unchanged GONG quiet-Sun
background sequences, oscillates the threads, and exports HDF5 datasets, MP4s and
opacity-derived segmentation masks.

Human-facing docs: [README.md](README.md) and [docs/](docs/README.md). This file
condenses them and adds code-level facts the docs omit. If the two disagree, the
code is authoritative.

## Ground rules

1. **Run everything from the repository root with `.venv` active.** Some scripts
   take repo-relative defaults from the current directory (`run_experiment.py
   --config`, `plot_luna_height_model.py --output-directory`).
2. **Never commit data.** `backgrounds/**/*.h5`, `FITS_files/*`,
   `FilamentSegmentator/*`, `simulations/`, `scratch/`, `notebooks/local/*`,
   `*.pdf` and `*.tex` are git-ignored on purpose. Do commit the small reference
   assets in `synthetic_filaments/data/`, `processed/`, `tests/data/`.
3. **Do not edit `synthetic_filaments/data/heinzel_calibrated_extension_1_to_100Mm.csv`.**
   The loader enforces its SHA-256 (`HEINZEL_EXTENSION_SHA256` in
   `opacity_table.py`), and the scientific regression compares it byte for byte
   with `processed/heinzel_table1_extension_final/output/`.
4. **Do not run `processed/heinzel_table1_extension_final/build_extension.py`.**
   It is an archived script that uses absolute `/mnt/data/` paths and deletes
   its output directory.
5. **Only one production (video) job runs at a time per experiments root.**
   `start_experiment_worker` refuses to start while another job is queued or
   running, and the worker holds `simulations/experiments/.generation.lock`. The
   GUI and `run_experiment.py` share that root, so do not start a CLI run while
   a GUI video is generating.
6. **Outputs are large.** Existing 448×448 runs take roughly 0.5–1.3 GB each;
   size grows with frame count and thread count. Check free disk space before batch runs.
7. **Never modify prepared backgrounds or source observations in place.** The
   model deliberately does no interpolation, normalization or frame repetition.

## Environment

- Python **3.12 or 3.13** (`requires-python = ">=3.12,<3.14"`). Linux/macOS only:
  the job runner uses `fcntl` and detached `start_new_session` workers.
- **FFmpeg with `libx264`** on `PATH` for MP4 export; FFprobe for the GUI
  regression. Previews, mask derivation and tests that don't make videos work without it.
- No GPU, PyTorch or detector is needed except for optional background
  preparation with `--use-detector`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # pinned deps + streamlit/tomli-w + notebook libs + `-e .`
```

Optional extras (in `pyproject.toml`): `.[prepare]` (torch/timm/transformers for
the detector), `.[quality]` (ruff, **not** installed by `requirements.txt`),
`.[gui]`, `.[notebook]`. Install one with `python -m pip install -e '.[quality]'`.
The Python package is named `synthetic-filaments` and imports as
`synthetic_filaments`.

Saving any TOML (`create_experiment`, `save_experiment_config`) imports
`tomli_w`. It is in `requirements.txt` but is only the `gui` extra in
`pyproject.toml`.

## Data prerequisites

| Path | Needed for | Notes |
| :--- | :--- | :--- |
| `backgrounds/*.h5` | Every simulation | Prepared FILOS backgrounds, supplied separately. Only the **top level** is scanned; `backgrounds/archive/` (older 120-frame files) is never auto-selected. |
| `FITS_files/*.h5` | Only `prepare_backgrounds.py` | Full-disk sources with `time_series` (t, y, x) and `tdeltas` (s). |
| `FilamentSegmentator/models/detector_v1/` | Only `--use-detector` | `config.json` + `model.safetensors`. |
| `tests/data/reference_background.h5` | Scientific regression | Tracked 3-frame historical fixture. Not a library background. |

The active library has two 389-frame, 448×448, 60 s cadence sequences from
`20190105.h5`, listed in `backgrounds/INVENTORY.md` and `backgrounds/library.json`.

## Running simulations

Every run has two phases: a **static preview** (frame zero, held in memory),
then a **production job**. The job is a detached worker process that reuses the
preview's frozen static state, runs the dynamics, and writes HDF5, MP4s, masks
and plots. Production refuses to start unless the preview matches the
current config (`preview_is_current`).

### Command line (preferred for agents)

```bash
python scripts/run_experiment.py --config configs/readme_example.toml
```

This loads the TOML, generates the preview, creates a **new** experiment
directory `simulations/experiments/<UTC-stamp>-<slug>/`, launches the worker and
polls until it finishes. It prints the run directory, or raises with the path to
`<job>/run.log`. CLI runs write straight into `runs/`; there is no save step and
no `preview/` folder. Worker records go to `simulations/experiments/.jobs/job-*/`
(`status.json`, `run.log`, snapshot). Without `--config` it uses
`configs/default_experiment.toml`.

To check a config without running anything:

```python
from synthetic_filaments import load_experiment_config, validate_production_config
resolved = validate_production_config(load_experiment_config("configs/my.toml"))
# Checks the schema, value ranges, background file and frame range, and that ffmpeg is on PATH.
```

### Streamlit GUI (for humans)

```bash
python -m streamlit run scripts/experiment_gui.py   # http://localhost:8501
```

**Create experiment** makes the directory at once, without `experiment.toml`.
An unsaved preview exists only in the Streamlit process memory; its PNGs are written to
`<experiment>/preview/<stamp>/` immediately, and **Save preview** adds the static
data and config. Video jobs use `defer_save=True`: the job and its working data
live in `$TMPDIR/filos-<uid>-<hash>/` until **Save video** (or **Close**). MP4s
and plots are copied into `runs/` when the job finishes; HDF5, configs and NPZ
masks are not. Closing the browser tab or killing the server does **not** autosave. Agents cannot drive this UI meaningfully; use the CLI or API.

### Python API (batch runs, sweeps, tests)

This recipe is verified to work: it takes about 10 s and writes only to a temp dir.

```python
import tempfile, time
from pathlib import Path
import numpy as np
from synthetic_filaments import (create_experiment, generate_experiment_preview,
                                 list_jobs, load_experiment_config, start_experiment_worker)
from synthetic_filaments.background_preparation import write_background

tmp = Path(tempfile.mkdtemp(prefix="filos-"))
bg = write_background(tmp / "bg.h5", np.ones((3, 128, 128), np.float32), cadence_s=20.0,
                      native_pixel_km=2500.0, disk_mu=0.8, limb_direction_deg=45.0)
config = load_experiment_config("configs/default_experiment.toml")   # plain dict of sections
config["inputs"]["h5_background_path"] = str(bg)                     # or a real backgrounds/ file
config["static"]["thread_count_cap"] = 5
config["dynamics"]["n_frames"] = 3
config["experiment"]["name"] = "smoke test"   # the run label comes from the config passed to the worker

preview = generate_experiment_preview(config)
root = tmp / "experiments"                     # separate root = separate lock and job queue
experiment = create_experiment(config["experiment"]["name"], experiments_root=root, source_config=config)
job = start_experiment_worker(experiment, user_config=config, preview=preview, experiments_root=root)
while (status := next(j for j in list_jobs(root) if j["job_id"] == job["job_id"]))["state"] \
        not in {"completed", "failed"}:
    time.sleep(1)
print(status["state"], status.get("output_directory"), status.get("error"))
```

Notes:
- A tiny synthetic background like the one above is the fastest way to exercise the
  pipeline. Its physics is meaningless, so use a real prepared background for science.
- Passing a distinct `experiments_root` gives each run its own lock and job queue,
  so you *can* run jobs in parallel that way, memory permitting (see Performance).
  `run_experiment.py` always uses the default root.
- Lower-level functions (`generate_from_h5_background`, `initialize_dynamics`,
  `step_dynamics`, `simulate_and_save_filament_dynamics`, `save_gong_video`, …)
  are exported from `synthetic_filaments/__init__.py` and are what the reference
  notebook uses. They bypass the experiment bookkeeping.

## Experiment TOML

`configs/default_experiment.toml` is the reference schema;
`configs/readme_example.toml` reproduces the README example (389 frames).
Start new configs by copying one of these.

**Every section and every field is required. Unknown fields are rejected**
(`normalize_experiment_config`). The one exception: a missing
`dynamic_background.seed` is silently filled from `dynamics.seed` (or 0); write
it explicitly. Integer fields must be TOML integers:
`n_frames = 120.0` fails. Float fields accept integers. `"auto"` is accepted
only where noted.

| Field | Meaning / constraint (from code) |
| :--- | :--- |
| `experiment.name` | `^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$`; slugified into the directory name. `description` ≤ 500 chars. |
| `inputs.h5_background_path` | A prepared file **or** a directory. Relative paths resolve against the **project root**, not the CWD. For a directory, the file is `sorted(dir.glob("*.h5"))[default_rng(dynamic_background.seed).integers(n)]`, so adding or removing files changes which file a seed picks. Jobs freeze the resolved absolute path. |
| `static.seed` | Geometry/plasma RNG. It also sets filament orientation (uniform −90°…90°) and chirality. |
| `static.thread_density_per_mm`, `thread_count_cap` | Thread count = `max(1, round(density × spine length in Mm))`, capped. This dominates runtime and memory. |
| `static.spine_library_entry_index` | `"auto"` or an integer ≥ 0 index into `synthetic_filaments/data/spine_library.npz`. |
| `static.height_mean_km` | Must be in [1 000, 100 000]. |
| `static.temp_center_K` | Must lie in the opacity-table temperature domain; `temp_tr_K` > `temp_center_K`. |
| `static.dip_curvature_radius_*` | min ≤ median ≤ max, and min ≥ `thread_length_min_km / π`. |
| `static.source_fraction` | **Has no effect on GUI/CLI runs, but it must still lie in [0, 1]** (it is validated at save and production time). The preview always replaces it with the default 3.22/6.93. The renderer uses an *effective* S/I_bg interpolated at the threads' mean dip-bottom height from Heinzel et al. (2015) (10/20/30 Mm → 3.22/2.98/2.80 over 6.93) and extends it outside that range by a geometric-dilution ratio W(h). Metadata records `source_fraction_effective` (the value actually used). `source_fraction_configured` is always the default, not your TOML value. |
| `static.mask_tau_threshold` | Binary mask cutoff `tau > threshold` for automatic masks (default 0.1 ≈ 9.5 % line-centre absorption). Config validation accepts any finite value, but a value ≤ 0 makes the preview fail when it renders the masks. |
| `static.psf_sigma_px` | Gaussian PSF σ applied to the transmission on the **high-resolution (2× oversampled) grid**, before block-averaging to native pixels. The default 4.0 equals 2 native pixels. |
| `static.gong_bandpass_line_fraction` | Fraction of the GONG passband occupied by the line, in [0, 1]. |
| `dynamic_background.seed` | Library file selection only. |
| `dynamic_background.start_index`, `frame_step` | Frames used: `start + step·k` for k < `n_frames`. The last index must exist; frames are never repeated or interpolated. Simulation cadence = file cadence × `frame_step`. The preview always uses `start_index` alone. |
| `dynamics.seed` | Brownian-walk RNG and `center_spine_fraction = "auto"` site choice. |
| `dynamics.n_frames` | Integer ≥ 1. Spatial size and cadence always come from the background file. |
| `dynamics.oscillation_mode` | `"luna_2022_curvature"`: each thread's longitudinal period is `P = 2π [g(h)(1/R + 1/(R☉+h))]^(-1/2)` from its realized dip radius R and height h; the transverse period is `transverse_period_s`. `period_s` must be > 0 but is unused. `"shared_period"`: both components of every thread use `period_s`. `transverse_period_s` is ignored but must still be present. |
| `dynamics.*_displacement_amplitude_km` | Coherent amplitudes at the oscillation centre (finite; 0 disables a component). |
| `dynamics.center_spine_fraction` | 0–1 along the spine, or `"auto"` (a random material-bearing thread chosen by `dynamics.seed`). |
| `dynamics.center_height_km` | ≥ 0, or `"auto"` (the selected thread's centroid height). |
| `dynamics.half_strength_distance_km` | Weight `w(d) = exp[−ln2 (d/L)^1.1]`, set to 0 where w < 0.05. The run fails if no material-bearing thread reaches 5 %. |
| `dynamics.damping_time_s`, `oscillation_start_time_s`, `phase_rad` | `ξ(t) = A e^{−q/τ}[sin(2πq/P + φ) − sin φ]`, q = t − t₀; there is no coherent motion before t₀. |
| `dynamics.brownian_step_min_km` ≤ `brownian_step_max_km` | Independent cumulative random walk per thread. |
| `export.compression` | `"lzf"` (default), `"gzip"` or `"none"`. `gzip_level` (0–9 integer) is required even when unused. |
| `export.video_lower/upper_percentile` | Fixed contrast from frame zero, in [0, 100] with upper > lower. `thread_mask_dilation_px` is an integer ≥ 0 that only affects `labels/thread_mask_native`. |
| `video.fps` | Integer 1–120; playback only, unrelated to physical cadence. `velocity_limit_km_s` is `"auto"` or > 0. `quiver_stride_px` is ≥ 1. |

Legacy drafts are migrated on load: old dip-depth fields, crop/detector
controls and `dynamics.cadence_s` are dropped. Full-disk input paths must be
replaced by hand. For old `sphere_radius_km` / `kernel_half_weight_radius_fraction`,
set `half_strength_distance_km` to their product and remove `kernel_power` (see
`configs/README.md`).

## Outputs

```text
simulations/experiments/<UTC>-<slug>/
├── experiment.toml                       editable config (relative paths kept)
├── preview/<stamp>/                      GUI only: plots; data after "Save preview"
└── runs/
    ├── index.json                        rebuilt run index
    ├── failed-<UTC>-<hash>/              failure diagnostics (run.log, snapshot, status.json)
    └── sim-<UTC>-s<dyn-seed>-<hash10>-<label>/
        ├── simulation.h5  simulation.json  experiment.toml (frozen, absolute bg path)
        ├── status.json  run.log  static_state/
        ├── gong.mp4  velocity.mp4
        ├── frame_zero_comparison.png  geometry_diagnostics.png  luna_dynamics_diagnostics.png
        └── opacity_masks/masks_frame_NNNN.npz (+ masks_frame_0000.png)
```

Key `simulation.h5` datasets (schema version 6; arrays are `time, y, x` with
rows = y; plot with `origin="lower"`; the high-res grid is 2× native):

- `time/time_s`; `video/raw_gong` (synthetic intensity, float64);
  `video/processed_uint8` (fixed-contrast frames used for MP4 encoding);
  `forward_model/background_frames` (the unchanged background).
- `radiative/tau_highres`, `radiative/tau_native` (sources for opacity masks),
  `radiative/line_absorption_native`, `radiative/observable_absorption_native`.
- `labels/thread_mask_native` (geometry mask **with** safety dilation, not
  opacity-based), `labels/thread_geometry_mask_native`,
  `labels/coherent_velocity_xy_km_s` (t, 2, y, x), `labels/velocity_opacity_weight_native`,
  `labels/opacity_change_{highres,native}`.
- `state/*_displacements_km` (t, thread, 3); `dynamics/thread_*` per-thread
  periods, gravity, weights; `geometry/initial/*`, `geometry/spine/*`;
  `metadata/*_json` (JSON strings). Root attributes include `complete`,
  `generator_git_commit`, and `thread_position_reconstruction`.

`simulation.json` holds the full resolved config, the background provenance
(`initial_forward_model_metadata.real_background`), summaries and file hashes.
Open everything read-only. Load NPZ files with `np.load(..., allow_pickle=False)`.

Opacity masks: `binary_{highres,native}` (uint8, `tau > threshold`),
`absorption_{highres,native}` (float32, `1 − exp(−tau)`), `tau_threshold`,
`time_s`. To re-derive masks with another threshold, **pass `--output`**: the
default output is the run's own `opacity_masks/`, and existing frames there are
overwritten.

```bash
python scripts/derive_segmentation_masks.py <run-dir-or-simulation.h5> --threshold 0.05 --output scratch/masks_t005
```

## Backgrounds

- **Use:** drop prepared `.h5` files into `backgrounds/`. Format v1: dataset
  `time_series` (float, positive, finite, (t, y, x), y and x ≥ 8), dataset
  `time_s` (0, cadence, 2·cadence …), attributes `filos_background_version = 1`,
  `cadence_s`, `native_pixel_km`, `disk_mu` ∈ (0, 1], `limb_direction_deg` ∈ [0, 360),
  and optionally `provenance` (JSON). See `docs/BACKGROUNDS.md`.
- **Wrap your own cropped data:** call
  `synthetic_filaments.background_preparation.write_background(path, frames, cadence_s=…, native_pixel_km=…, disk_mu=…, limb_direction_deg=…, provenance={…})`.
  It validates the format, never overwrites, and does **not** screen for solar
  structures. Note that static config requires `disk_mu` ≥ 0.05.
- **Prepare from full-disk data:** `python scripts/prepare_backgrounds.py --frames 389`
  (default target 400; `--size 448 --count 2 --cadence 60 --seed 0`). It finds
  uninterrupted intervals, screens every frame, never overwrites, and exits 1 if
  nothing was prepared. If an output file already exists, that whole source is
  printed as "Skipped" (the `FileExistsError` is caught). Use `--output <dir>`
  to rebuild into a fresh directory.

## Checks

| Command | Needs | Time (32-core dev box) |
| :--- | :--- | ---: |
| `python -m unittest discover -s tests -v` | Nothing extra; the video-worker test is skipped without FFmpeg | ~30 s |
| `python scripts/run_scientific_regression.py --include-dynamics` | Tracked fixture only; `--dynamics-frames` ≤ 3 | ~70 s |
| `python scripts/run_experiment_regression.py` | Prepared library, FFmpeg, FFprobe; writes to `/tmp` and `scratch/` | longer |
| `python -m ruff check .` | `.[quality]` extra | lint |

Streamlit "missing ScriptRunContext" warnings and `ResourceWarning: subprocess …
is still running` during the unit tests are expected noise.

The scientific regression asserts **exact image hashes** for the reference case.
Any intentional change to the static forward model will break it. Update the expected
values deliberately and explain why in the commit, rather than loosening checks.

## Code map

All code is in the flat package `synthetic_filaments/`. Scripts are thin entry points.

| Concern | Module(s) |
| :--- | :--- |
| Static config, constants, validation | `config.py` (`make_static_config`, `validate_static_config`) |
| Spine library and thread geometry | `geometry.py` |
| Plasma loading, Heinzel opacity | `plasma.py`, `opacity_table.py` |
| Static synthesis, source function, rendering, PSF | `generator.py`, `render.py`, `degradation.py` |
| Background loading / preparation / detector | `dynamic_background.py`, `background_preparation.py`, `detector.py` (optional) |
| Oscillation laws and time stepping | `oscillation.py`, `dynamics.py` |
| HDF5 recorder, run index, videos, masks | `simulation_io.py`, `video.py`, `segmentation.py` |
| Static-state persistence (`static_state/`) | `io.py` (no pickle; manifest-verified) |
| Experiment TOML schema and migration | `experiment_config.py` (`SECTION_FIELDS`) |
| Preview, caching, jobs, worker, save flow | `experiment_runner.py`, `cache.py`, `execution.py` |
| Paths | `paths.py` (`PROJECT_ROOT` = parent of the package) |
| GUI | `scripts/experiment_gui.py`; worker entry point `scripts/run_experiment_worker.py` (launched by repo path) |

### When changing code

- **Adding or renaming a TOML field** means updating the matching tuple in
  `experiment_config.py` (`STATIC_FIELDS`, `DYNAMICS_FIELDS`, …), the underlying
  defaults/validator (`config._CANONICAL_VALUES`, `dynamics._DYNAMICS_DEFAULTS`,
  `simulation_io.make_export_config`), every TOML in `configs/`, the GUI
  editor, the docs and, if needed, a migration in `normalize_experiment_config`.
  Old configs without the field will otherwise fail to load. A new **static**
  field must also go into one of `_GEOMETRY_STATIC_FIELDS`,
  `_PLASMA_STATIC_FIELDS` or `_OBSERVATION_STATIC_FIELDS` in
  `experiment_runner.py`. Otherwise the in-process stage cache can silently
  reuse stale geometry, plasma or render results after the field is edited.
- **Changing static physics:** bump `STATIC_FORWARD_MODEL_REVISION` (and
  `SOURCE_FUNCTION_POLICY` if the source function changes) in
  `experiment_config.py`. Both feed the persisted preview fingerprint, so a bump
  stops old saved previews from counting as current. Only
  `STATIC_FORWARD_MODEL_REVISION` reaches the in-process cache keys; the render
  key in `experiment_runner.py` hard-codes its own source-function string. Bump
  `PREVIEW_RENDER_VERSION` in `experiment_runner.py` for diagnostic-figure changes.
- **Changing the HDF5 layout:** bump `SIMULATION_DATASET_SCHEMA_VERSION`
  (`simulation_io.py`) or `DYNAMICS_SCHEMA_VERSION` (`dynamics.py`). For the
  `static_state/` layout, bump `STATIC_OUTPUT_SCHEMA_VERSION` (`io.py`).
- Keep the existing conventions: plain-dict configs validated by explicit
  functions; seeded `numpy.random.default_rng` (no OS entropy); atomic
  write-then-`os.replace`; no pickle; read-only access to inputs; ruff
  (line length 100, rules E4/E7/E9/F/I).
- The detector and torch must stay out of the runtime import path; a unit test
  checks this.
- Add tests under `tests/` (unittest). Use `write_background` with tiny arrays
  and `tempfile` roots, as `tests/test_experiment_persistence.py` does.

## Performance

Set these environment variables before launching (defaults in parentheses):
`FILAMENT_PLASMA_WORKERS` (8; 1 = serial), `FILAMENT_PLASMA_BATCH_SIZE` (32),
`FILAMENT_RENDER_BATCH_SIZE` (8192), `FILAMENT_DYNAMICS_WORKERS` (4),
`FILAMENT_DYNAMICS_QUEUE_SIZE` (8), `FILAMENT_CACHE_GIB` (16; 0 disables the stage
cache), `FILAMENT_H5_CACHE_MIB` (512). The cache budgets do not cap total
process memory. Workers inherit the launching environment. Notebook tooling also
reads `FILAMENT_NOTEBOOK_FAST=1` and `FILAMENT_NOTEBOOK_OUTPUT_ROOT`. Benchmarks:
`scripts/benchmark_preview.py --output scratch/x.json` and
`scripts/benchmark_dynamics.py --frames 120` (see `docs/PERFORMANCE.md`).

## Other entry points

- `python scripts/plot_luna_height_model.py` exports Luna period / gravity cut-off
  tables and figures to `scratch/luna_height_model/`.
- `python scripts/build_refactored_notebook.py` (over)writes
  `notebooks/07_refactored_forward_model.ipynb`. That file is generated and untracked but
  **not** git-ignored, so don't commit it by accident.
  After building, `python scripts/execute_refactored_notebook.py` runs it in place using the
  `python3` kernel and `configs/default_experiment.toml`. Full runs write
  standalone `simulations/sim-*`; `FILAMENT_NOTEBOOK_FAST=1` writes 3 frames
  under `scratch/` with no videos.
- Put personal notebooks and analysis in `notebooks/local/` (ignored).
