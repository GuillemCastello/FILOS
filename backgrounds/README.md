# Background library

Put supplied prepared `.h5` sequences here, or build them from `FITS_files/`:

```bash
python scripts/prepare_backgrounds.py
```

Each file carries its own image sequence, timing, scale, and disk position.
Users running simulations need no original observation files or detector model.
The GUI lists this directory automatically. HDF5 payloads are ignored by Git.

See the [format and preparation guide](../docs/BACKGROUNDS.md).

See the [initial library inventory](INVENTORY.md) for locally prepared sequences.
