# Full-disk source observations

Place source GONG `.h5` files here for one-time background preparation. Each needs
`time_series` images shaped `(time, height, width)` and matching `tdeltas` timestamps
in seconds, with aligned solar-disk support.

```bash
python scripts/prepare_backgrounds.py
```

Simulations read the resulting portable sequences in `backgrounds/`; they do not
open these full-disk files. Source files are ignored by Git. See the
[background guide](../docs/BACKGROUNDS.md).
