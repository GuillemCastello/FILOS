# Performance controls

[Documentation index](README.md)

Execution settings are environment variables; physical experiment settings live
in TOML. Set environment variables before starting the GUI or a benchmark.

| Variable | Default | Purpose |
| :--- | ---: | :--- |
| `FILAMENT_PLASMA_WORKERS` | `8` | Plasma worker count; `1` selects serial execution |
| `FILAMENT_PLASMA_BATCH_SIZE` | `32` | Thread batch size for plasma work |
| `FILAMENT_RENDER_BATCH_SIZE` | `8192` | Sample batch size for rendering |
| `FILAMENT_CACHE_GIB` | `16` | In-memory stage-cache budget in GiB; `0` disables retention |
| `FILAMENT_H5_CACHE_MIB` | `512` | HDF5 sequence chunk-cache budget in MiB |
| `CUDA_VISIBLE_DEVICES` | `1` in the GUI launcher | Visible GPU(s) for the detector; empty selects CPU |

Cache budgets are not limits on total process memory. Arrays and worker processes
also consume memory. On a smaller machine, reduce cache sizes and worker counts:

```bash
FILAMENT_PLASMA_WORKERS=2 FILAMENT_CACHE_GIB=2 FILAMENT_H5_CACHE_MIB=128 \
  uv run --extra gui --extra detector streamlit run scripts/experiment_gui.py
```

## Measure previews

```bash
uv run --extra detector python scripts/benchmark_preview.py \
  --h5 FITS_files/20140101.h5 \
  --output scratch/preview_benchmark.json
```

The benchmark defaults to three fresh-process repetitions and records stage
timings, work counts, cache behavior, scientific hashes, and process-tree memory.
Use `--workers 4` to select plasma parallelism, `--profile` for an additional
profiling pass, or `--config path/to/experiment.toml --skip-small` for a saved
experiment. `--disable-detector` allows measurement without local detector weights.

## Measure dynamics and exports

```bash
uv run --extra detector python scripts/benchmark_dynamics.py --frames 240
```

This runs the reference dynamics and both MP4 exports in temporary storage that
is removed on completion. It requires reference observations, detector assets,
and FFmpeg.

LZF is the default lossless export compression. Pass `--compression gzip` or
`--compression none` to compare alternatives. Use the measured runtime and file
size for your workload when choosing a format.
