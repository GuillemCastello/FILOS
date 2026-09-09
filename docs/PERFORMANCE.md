# Performance controls

[Documentation index](README.md)

Set execution variables before starting the GUI or a benchmark. Physical
experiment controls belong in TOML.

| Variable | Default | Purpose |
| :--- | ---: | :--- |
| `FILAMENT_PLASMA_WORKERS` | `8` | Plasma worker count; `1` selects serial execution |
| `FILAMENT_PLASMA_BATCH_SIZE` | `32` | Thread batch size for plasma work |
| `FILAMENT_RENDER_BATCH_SIZE` | `8192` | Sample batch size for rendering |
| `FILAMENT_CACHE_GIB` | `16` | Stage-cache budget in GiB; `0` disables retention |
| `FILAMENT_H5_CACHE_MIB` | `512` | HDF5 chunk-cache budget in MiB |

Budgets are not limits on total process memory; arrays and workers also consume
memory. On a smaller machine:

```bash
FILAMENT_PLASMA_WORKERS=2 FILAMENT_CACHE_GIB=2 FILAMENT_H5_CACHE_MIB=128 \
  python -m streamlit run scripts/experiment_gui.py
```

## Benchmarks

```bash
python scripts/benchmark_preview.py --output scratch/preview_benchmark.json
python scripts/benchmark_dynamics.py --frames 120
```

Preview benchmarking records cold/warm stage timings, cache behavior, scientific
hashes, and process-tree memory. It defaults to three fresh-process repetitions.
Use `--h5 backgrounds/my_sequence.h5` or `--config path/to/experiment.toml` to
select inputs, `--workers 4` to select parallelism, and `--profile` for a separate
profiling pass. Use `--skip-small` to omit the smaller comparison workload.

Dynamics benchmarking times simulation and both MP4 exports in temporary storage
removed on completion. It requires a sufficiently long prepared background and
FFmpeg. Select a file with `--h5-background backgrounds/my_sequence.h5`.

LZF is the default lossless export compression. Pass `--compression gzip` or
`--compression none` to compare runtime and file size for your workload.
