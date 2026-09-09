# Local detector assets

Place the separately supplied detector in `models/detector_v1/` with its
`config.json` and `model.safetensors` files. The Python detector integration lives
in [`synthetic_filaments/detector.py`](../synthetic_filaments/detector.py).

Model assets are ignored by Git. Keep this directory name: the runtime loads the
model from this location. Install the `detector` extra when using detection, or
disable `dynamic_background.use_detector` in an experiment to run without it.

See [data and outputs](../docs/DATA.md).
