# Prepared library inventory

These files were prepared locally from the supplied GONG observations. The HDF5
files are distributed separately; this inventory records their identities.

| File | Frames | Shape | Cadence | Size |
| :--- | ---: | :--- | ---: | ---: |
| `20140101-0341-1128-1195.h5` | 120 | 448 × 448 | 60 s | 90.5 MiB |
| `20140215-0894-1039-1108.h5` | 120 | 448 × 448 | 60 s | 90.8 MiB |
| `20140215-1116-895-1231.h5` | 120 | 448 × 448 | 60 s | 90.8 MiB |
| `20141018-0667-635-1294.h5` | 120 | 448 × 448 | 60 s | 90.9 MiB |
| `20190105-0001-1012-659.h5` | 120 | 448 × 448 | 60 s | 90.5 MiB |
| `20190105-0001-1039-1108.h5` | 120 | 448 × 448 | 60 s | 90.5 MiB |

Each sequence passed structural and finite-positive-pixel screening on all 120
frames, plus detector checks at offsets 0, 30, 60, 90, and 119. No interpolation,
normalization, or resampling was applied. The source timestamps are preserved in
the provenance. See [library.json](library.json) for metadata and SHA-256 digests.

The same preparation settings found no passing sequences in `20140323.h5` or
`20141216.h5`. This describes the tested candidates and intervals, not every
possible crop in those source files.

Prepared with:

```bash
python scripts/prepare_backgrounds.py --use-detector --count 2
```
