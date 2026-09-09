# Data and outputs

[Documentation index](README.md)

## Observations

The default configuration reads `FITS_files/20140101.h5`. Despite the directory
name, this workflow consumes an HDF5 sequence, with a `time_series` dataset
arranged as `(frame, height, width)`. Supply the observation file separately; the
repository does not download it.

Use `inputs.h5_background_path` in an experiment to select another compatible
sequence. Relative paths resolve against the project root. The requested crop,
starting frame, frame spacing, and number of frames must fit the source data.
The default profile requests 240 frames at a cadence of 60 seconds.

## Detector assets

Place the local model in this layout:

```text
FilamentSegmentator/
└── models/
    └── detector_v1/
        ├── config.json
        └── model.safetensors
```

The detector uses locally supplied weights to exclude background crops containing
real filaments. It requires the `detector` dependency extra and loads the model
offline. The default experiment enables it; set
`dynamic_background.use_detector = false` to select crops without detection.

The Streamlit launcher sets `CUDA_VISIBLE_DEVICES=1` only if that variable is not
already defined. Set it before launch to select a different GPU, or use an empty
value for CPU execution. The detector uses CPU when CUDA is unavailable.

## Configuration

[`configs/default_experiment.toml`](../configs/default_experiment.toml) is the
template for new experiments. Its sections group experiment metadata, observation
inputs, static geometry/plasma, dynamic background selection, oscillation dynamics,
dataset export, and video settings.

Use the GUI to save each experiment's configuration. Existing experiments and run
snapshots have their own TOML files; changing the shared default does not rewrite
those saved files.

## Generated results

```text
simulations/
└── experiments/
    ├── .jobs/                     Worker snapshots, status, and logs
    └── <experiment-id>/
        ├── experiment.toml        Saved experiment configuration
        └── runs/
            ├── index.json        Run index
            └── <simulation-id>/
                ├── experiment.toml
                ├── simulation.h5
                ├── gong.mp4
                └── velocity.mp4
```

Runs also contain supporting metadata. The interface can load standalone
simulations and clone them into experiments. All simulation output stays local
and is ignored by Git.

Production generation requires FFmpeg with `libx264` available on `PATH` for its
two MP4 exports. The GUI regression additionally uses FFprobe to inspect videos.
HDF5 export defaults to lossless LZF compression; see
[performance controls](PERFORMANCE.md) for alternatives.

## Packaged scientific assets

The opacity table and measured-spine library in `synthetic_filaments/data/` ship
with the Python package. Keep them in version control. The full calibration
archive lives in [`processed/`](../processed/README.md); scientific regression
compares its master table against the runtime copy.
