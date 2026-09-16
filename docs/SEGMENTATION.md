# Opacity-derived segmentation masks

[Documentation index](README.md)

These masks follow the rendered Hα optical depth instead of the full thread
geometry. They preserve weak absorption as a continuous map and provide a binary
mask at a chosen threshold. No dilation, hole filling, or geometry padding is used.

## In Streamlit

1. Set **Mask tau threshold** under **Thread plasma and image formation** (default 0.1).
2. **Generate preview** automatically plots both mask grids and saves the PNGs.
   **Save preview** saves static arrays, NPZ masks, and the matching configuration.
3. **Generate video** automatically creates masks for every frame. MP4s and the
   frame-zero mask plot are saved immediately; **Save video** commits the HDF5,
   configs, metadata, and NPZ masks from temporary working storage.
4. Use **Mask frame (zero-based)** to inspect other frames. Each displayed plot is
   saved beside the video's plots.

**Close** saves the last generated preview and all pending videos using their own
matching configurations. A running video receives a request to save when it finishes.
Edits made after generation do not change these snapshots. Browser-tab closure or
server termination cannot trigger the GUI Close action.

Changing the threshold applies to the next generation. Existing saved results keep
their original threshold.

## Command line

From the repository root with the virtual environment activated:

```bash
python scripts/derive_segmentation_masks.py simulations/experiments/<experiment>/runs/<run>
```

You can also pass `simulation.h5` directly and choose the threshold and output:

```bash
python scripts/derive_segmentation_masks.py /path/to/simulation.h5 \
    --threshold 0.05 --output scratch/detailed_masks
```

The script exports every frame and saves a frame-zero comparison plot. It reads
the simulation one frame at a time, without rerunning the model or decoding the MP4.
No FFmpeg is needed for mask generation.

## Mask definitions

The default threshold is **0.1**. The binary mask selects `tau > threshold` (strictly
greater); the continuous mask is `1 - exp(-tau)`, between zero and one. It describes
line-center absorption, not a segmentation probability or the final image contrast.
A threshold of 0.1 corresponds to approximately 9.5% line-center absorption.

| Grid | Source | Meaning |
| :--- | :--- | :--- |
| `highres` | `radiative/tau_highres` | Finest stored opacity grid, before instrument blur |
| `native` | `radiative/tau_native` | Effective optical depth after PSF blur and pixel integration; matches the simulation image grid |

The native optical depth is derived from degraded transmission, not an average of
high-resolution optical depth or a resized binary mask. Preview masks use the
equivalent `tau_map` and `soft_mask` arrays already held in memory.

Lower thresholds include weaker absorption and generally larger areas. Higher
thresholds retain stronger absorption. Thresholds must be finite and positive;
selecting every nonzero pixel would include faint Gaussian tails. The continuous
maps retain weak structure regardless of the binary threshold. High-resolution
masks cannot recover detail finer than the stored simulation grid.

The existing `labels/thread_mask_native` is a different product: it covers projected
thread geometry with safety dilation, including regions of negligible opacity.
It is not used or modified here. Changing `export.thread_mask_dilation_px` or the
experiment's `static.mask_tau_threshold` does not update an existing mask export.
## Saved files

Each preview generation has its own `preview/<timestamp>/` directory containing
its plots. **Save preview** adds `opacity_masks.npz`, `static_state/`,
`experiment.toml`, and `preview.json`. Only static simulation data is saved;
the TOML keeps the complete editor settings for reloading.

**Save video** commits one NPZ per frame under the run's `opacity_masks/` directory:

```text
opacity_masks/
├── masks_frame_0000.npz
├── masks_frame_0001.npz
├── ...
└── masks_frame_0000.png
```

Each NPZ contains these two-dimensional arrays:

| Array | Type and shape |
| :--- | :--- |
| `binary_highres` | `uint8`, `(highres_y, highres_x)`, values 0 or 1 |
| `absorption_highres` | `float32`, same high-resolution shape |
| `binary_native` | `uint8`, `(native_y, native_x)`, values 0 or 1 |
| `absorption_native` | `float32`, same native shape |

Scalar `tau_threshold` records the cutoff; video archives also contain `time_s`.
Read archives with `numpy.load(..., allow_pickle=False)`. One archive per frame
keeps export and inspection memory bounded. Frame indices and spatial coordinates
match the simulation HDF5, which may have a wider field of view than the MP4.

The CLI also accepts an explicit `.h5` output for compatibility with earlier mask
exports. Existing simulation data and geometry labels remain unchanged.

The GUI saves frame zero automatically and saves other plots when viewed.
[`save_mask_plot(path, index)`](../synthetic_filaments/segmentation.py) accepts an
NPZ directory or a legacy mask HDF5 file to save any selected frame.
