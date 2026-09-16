# Data and outputs

[Documentation index](README.md)

## Inputs

Simulations read self-contained background sequences from `backgrounds/` or a
custom file/directory path. Each contains the observation pixels, regular time
axis, pixel scale, and disk position. See the [background guide](BACKGROUNDS.md)
for the format and one-time preparation process.

Full-disk sources in `FITS_files/` and detector weights in `FilamentSegmentator/`
are only preparation inputs. They are not needed to run a prepared background.
Large input and output HDF5 files stay local and are ignored by Git.

## Configuration

[`configs/default_experiment.toml`](../configs/default_experiment.toml) is the
starting profile for new experiments. Choose the background through
`inputs.h5_background_path`: a prepared file or library directory. Relative paths
resolve against the project root.

`dynamic_background` contains the library selection seed, starting frame, and
frame step. The dimensions and cadence come from the chosen file. The default
run uses 120 frames. Each experiment saves its own settings; changing the shared
default does not rewrite existing experiments.

For command-line use:

```bash
python scripts/run_experiment.py --config configs/default_experiment.toml
```

## Generated results

```text
simulations/
└── experiments/
    ├── .jobs/                     Command-line worker snapshots, status, and logs
    └── <experiment-id>/
        ├── experiment.toml        Saved experiment configuration
        ├── preview/<timestamp>/   Preview plots; static data/config after Save preview
        └── runs/
            ├── index.json        Run index
            └── <simulation-id>/
                ├── experiment.toml
                ├── simulation.h5
                ├── simulation.json
                ├── static_state/
                ├── frame_zero_comparison.png
                ├── geometry_diagnostics.png
                ├── luna_dynamics_diagnostics.png
                ├── opacity_masks/     Per-frame NPZ masks and generated PNGs
                ├── gong.mp4
                └── velocity.mp4
```

The GUI creates the experiment folder immediately. Generation writes preview PNGs,
video MP4s, and generated mask plots into it. **Save preview** commits static data
and its configuration; **Save video** commits that video's data and configuration.
GUI worker snapshots and unsaved simulation data live in a temporary `filos-*`
directory until saved. Successful video saves remove the temporary simulation copy.
**Close** saves pending successful results and requests saving any running job on
completion, using each result's matching configuration.

Production jobs freeze the selected background path into their configuration
snapshot. Input metadata and the static preview are checked before the worker
runs. The saved HDF5 contains the background pixels used in the simulation.
The interface can load standalone simulations and clone them into experiments.

Production requires FFmpeg with `libx264` on `PATH`. GUI workflow regression also
uses FFprobe. HDF5 export defaults to lossless LZF compression; see
[performance controls](PERFORMANCE.md).

## Detailed segmentation masks

Automatic mask generation derives binary and continuous absorption masks from
`radiative/tau_highres` and `radiative/tau_native` for every saved frame. These are
separate from the broad geometry-based `labels/thread_mask_native` training label.
Use **Save preview** or **Save video** to keep the NPZ arrays. Video exports add
`opacity_masks/masks_frame_XXXX.npz` and a frame-zero PNG; previews save
`opacity_masks.npz`. Existing simulations can be processed with
`scripts/derive_segmentation_masks.py`. See the
[segmentation guide](SEGMENTATION.md) for definitions, plotting, and the dataset schema.

## Scientific reference assets

The opacity table and measured-spine library in `synthetic_filaments/data/` ship
with the Python package. The calibration archive lives in
[`processed/`](../processed/README.md).

`tests/data/reference_background.h5` is a small, three-frame extract of the
historical observation crop. It preserves exact scientific-regression checks
without requiring full-disk data or detector weights. It is a test fixture,
separate from the screened background library.
