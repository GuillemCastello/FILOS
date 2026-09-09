# Local observations

Place separately supplied GONG observation sequences here. The default experiment
reads `20140101.h5`, with a `time_series` dataset of shape `(frame, height, width)`.

Observation files are ignored by Git. Keep this directory name: it is the
runtime's default input location. To use another file, set
`inputs.h5_background_path` in your experiment configuration.

See [data and outputs](../docs/DATA.md).
