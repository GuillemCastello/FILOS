# Optional preparation detector

The detector is only used when preparing a background library with
`--use-detector`. Simulations and the GUI do not import it.

Supply `models/detector_v1/config.json` and `models/detector_v1/model.safetensors`,
then install the optional dependencies:

```bash
python -m pip install -e '.[prepare]'
python scripts/prepare_backgrounds.py --use-detector
```

Model assets are ignored by Git. See [background preparation](../docs/BACKGROUNDS.md).
