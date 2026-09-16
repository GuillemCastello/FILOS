#!/usr/bin/env python3
"""Export detailed opacity masks for every frame of an existing FILOS simulation."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_filaments.segmentation import export_video_masks, export_video_masks_npz, save_mask_plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Run directory or simulation.h5")
    parser.add_argument("--threshold", type=float, default=0.1, help="Positive tau cutoff (default: 0.1)")
    parser.add_argument("--output", type=Path, help="NPZ directory (default: run/opacity_masks); .h5 for legacy HDF5")
    args = parser.parse_args()
    source = args.source / "simulation.h5" if args.source.is_dir() else args.source
    output = args.output or source.parent / "opacity_masks"

    def report(done: int, total: int) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"Masks: {done}/{total} frames", flush=True)

    export = export_video_masks if output.suffix == ".h5" else export_video_masks_npz
    path = export(source, output, args.threshold, progress=report)
    plot = save_mask_plot(path)
    print(f"Saved masks: {path}")
    print(f"Saved frame-zero plot: {plot}")


if __name__ == "__main__":
    main()
