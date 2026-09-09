# Background library

Put supplied prepared `.h5` sequences here, or build them from `FITS_files/`:

```bash
python scripts/prepare_backgrounds.py --frames 389
```

Preparation now targets 400 frames per sequence. Each file carries its own
image sequence, timing, scale, and disk position.
Users running simulations need no original observation files or detector model.
The GUI lists this directory automatically. HDF5 payloads are ignored by Git.

See the [format and preparation guide](../docs/BACKGROUNDS.md).

See the [initial library inventory](INVENTORY.md) for locally prepared sequences.

The active library has 389 screened frames per sequence, close to the 400-frame
target. Earlier 120-frame files are kept in [archive/](archive/README.md).
