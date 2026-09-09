# Prepared backgrounds

[Documentation index](README.md)

A background is one self-contained HDF5 file. Copying it to another machine is
sufficient to run simulations: its original observations and detector model are
not runtime dependencies.

## Use a supplied library

Place `.h5` files in `backgrounds/`. The GUI lists those files and offers automatic
selection using the background seed. A custom path can refer to one file or a
directory of `.h5` files. Selection uses the sorted filenames; changing directory
contents can change a seed's selection. Completed jobs record the chosen file
explicitly, so adding library files does not change their recorded background.

The simulation uses the file's full spatial extent. `start_index` and `frame_step`
select the time interval, and `dynamics.n_frames` selects its length. Simulation
cadence equals the recorded cadence multiplied by `frame_step`. Files that are
too short for a request produce an error instead of repeated or interpolated frames.

## Prepare full-disk observations

```bash
python scripts/prepare_backgrounds.py --frames 389
```

The command above uses the tested length for the supplied observations.
The default target, when `--frames` is omitted, is **400 frames** (about 6 hours 40 minutes at one-minute
cadence), from `FITS_files/*.h5`. Each source needs:

- `time_series`: floating-point images shaped `(time, height, width)`.
- `tdeltas`: strictly increasing timestamps in seconds, one per image.
- Aligned, approximately circular solar-disk support: finite positive pixels
  inside the disk, nonpositive or invalid pixels outside it.

Preparation finds non-overlapping uninterrupted intervals at the requested
cadence. It screens candidate crops on every frame, rejecting invalid/nonpositive
pixels and large or elongated dark/bright structures. Surviving crops are copied
unchanged with lossless LZF compression. Overlapping crops from the same interval
are avoided. Output filenames identify the source, starting frame, and crop origin.

```bash
python scripts/prepare_backgrounds.py FITS_files/20190105.h5 \
  --frames 389 --size 448 --count 2 --cadence 60 --seed 0
```

`--count` is a maximum per source, not a guarantee. Longer intervals are more
likely to encounter gaps or evolving structures. With the supplied sources, only
`20190105.h5` has an uninterrupted interval of at least 400 frames at 60 seconds.
Preparation skips shorter intervals; it never fills gaps or repeats frames.
For the supplied library, 389-frame crops pass the full screening; the final
frames of the 400-frame candidates do not. The command above uses that tested
length. The default target remains 400 for future observations. Existing outputs are never
overwritten; use a separate `--output` directory to rebuild a library.

### Optional preparation detector

Install `python -m pip install -e '.[prepare]'` and supply the model under
`FilamentSegmentator/models/detector_v1/`. Add `--use-detector` to preparation.
By default it checks the first frame, every 30th frame, and the last frame using
expanded exclusion boxes. Set `--detector-stride 1` to check every frame.
All-frame structural screening runs in either case. Model errors stop the run.
GPU selection, if needed, can be set before preparation with `CUDA_VISIBLE_DEVICES`.

The detector detects filaments; the structural rules screen dark/bright features.
These are automated screening criteria, not a guarantee of physical quietness.
Inspect the resulting sequences before publishing a library.

## File format, version 1

| Item | Meaning |
| :--- | :--- |
| Dataset `time_series` | Finite positive floating-point pixels, `(frames, height, width)`; spatial dimensions at least 8 |
| Dataset `time_s` | Seconds from the sequence start, beginning at zero with regular spacing |
| Attribute `filos_background_version` | Integer `1` |
| Attribute `cadence_s` | Positive source frame spacing in seconds |
| Attribute `native_pixel_km` | Positive physical pixel scale in km/pixel |
| Attribute `disk_mu` | Cosine of the viewing angle at the crop center, in `(0, 1]` |
| Attribute `limb_direction_deg` | Direction toward the limb in image coordinates, `[0, 360)`; +x is 0°, +y is 90° |
| Attribute `provenance` | JSON object describing the source and preparation; optional for supplied custom files |

Preparation records the source filename, original bounds, source frame times,
original disk geometry, and screening coverage in `provenance`. The source path
is informational; loading never opens it. The pixel scale and viewing geometry
are held fixed over each prepared sequence, as in the model's existing observing
approximation.

## Supply already-cropped observations

Use the writer with your measured geometry and regular frame timing:

```python
from synthetic_filaments.background_preparation import write_background

# frames: your floating-point array shaped (time, height, width).
# Set these values from the observations, not from the example numbers.
write_background(
    "backgrounds/my_sequence.h5",
    frames,
    cadence_s=60.0,
    native_pixel_km=700.0,
    disk_mu=0.9,
    limb_direction_deg=45.0,
    provenance={"source_file": "my_observations", "screening": "manually reviewed"},
)
```

This writer validates the file contract but does not screen solar structures.
Review your input first. Unlike full-disk preparation, it needs neither a disk
mask nor a detector. It preserves floating-point dtype and pixel values.

## Existing experiments

Old TOML drafts can be opened, but their full-disk paths must be replaced with a
prepared sequence or library. Obsolete detector/crop controls and manually set
cadence are discarded during loading. Old results remain on disk; this migration
does not regenerate or overwrite them.
