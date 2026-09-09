# Prepared library inventory

The active library contains two **389-frame** sequences, close to the 400-frame
preparation target. Both come from the uninterrupted interval at source indices
187–575 inclusive in `20190105.h5`. They span 388 minutes at one-minute cadence.

| File | Frames | Shape | Cadence | Size |
| :--- | ---: | :--- | ---: | ---: |
| `20190105-0187-1012-659.h5` | 389 | 448 × 448 | 60 s | 293.5 MiB |
| `20190105-0187-1039-1108.h5` | 389 | 448 × 448 | 60 s | 293.8 MiB |

Every frame passed finite-positive-pixel and quiet-structure screening. Detector
screening covered the first frame, every 30th frame, and the last frame. Pixel
values and timestamps were copied without interpolation, normalization, or repetition.

The tested 400-frame candidates failed near the end of the interval, so the
supplied sequences stop before those failures. The 396-frame October interval
also produced no passing crops in the tested search. Other supplied sources do
not have 400 uninterrupted one-minute frames. These results describe the tested
candidates, not every possible crop.

Prepared with:

```bash
python scripts/prepare_backgrounds.py FITS_files/20190105.h5 --frames 389 --use-detector --count 2
```

See [library.json](library.json) for complete provenance and SHA-256 digests. The
HDF5 files stay local and are distributed separately. The previous 120-frame
library is retained in [archive/](archive/README.md) and is excluded from automatic
selection because the loader only scans the top level of `backgrounds/`.
