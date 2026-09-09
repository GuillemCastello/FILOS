#!/usr/bin/env python3
"""Prepare portable quiet-Sun sequences from local full-disk observations."""

import argparse
from pathlib import Path

from synthetic_filaments.background_preparation import prepare_backgrounds
from synthetic_filaments.paths import DEFAULT_BACKGROUNDS_DIR, DEFAULT_FITS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="*", help="HDF5 files; default: FITS_files/*.h5")
    parser.add_argument("--output", type=Path, default=DEFAULT_BACKGROUNDS_DIR)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--size", type=int, default=448, help="Square crop size in pixels")
    parser.add_argument("--count", type=int, default=2, help="Maximum sequences per source")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cadence", type=float, default=60.0, help="Required source cadence in seconds")
    parser.add_argument("--use-detector", action="store_true", help="Also screen sampled frames with the local detector")
    parser.add_argument("--detector-stride", type=int, default=30)
    args = parser.parse_args()
    sources = args.sources or sorted(DEFAULT_FITS_DIR.glob("*.h5"))
    if not sources:
        parser.error("no source files found in FITS_files/")
    total = 0
    for source in sources:
        print(f"Preparing {source.name}…", flush=True)
        try:
            results = prepare_backgrounds(source, args.output, n_frames=args.frames,
                                          crop_shape=(args.size, args.size), count=args.count,
                                          seed=args.seed, cadence_s=args.cadence,
                                          use_detector=args.use_detector,
                                          detector_stride=args.detector_stride)
        except (ValueError, OSError) as error:
            print(f"  Skipped: {error}", flush=True)
            continue
        total += len(results)
    print(f"Prepared {total} backgrounds in {args.output}.")
    if total == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
