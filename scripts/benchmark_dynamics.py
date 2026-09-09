#!/usr/bin/env python3
"""Benchmark the notebook-authoritative dynamics and MP4 workflow."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

from run_scientific_regression import (  # noqa: E402
    canonical_dynamics_config,
    canonical_static_result,
)

from synthetic_filaments import (
    DEFAULT_H5_DYNAMICS_PATH,  # noqa: E402
    load_h5_background_sequence,
    make_export_config,
    save_gong_video,
    save_velocity_video,
    simulate_and_save_filament_dynamics,
)


def _parse_arguments() -> argparse.Namespace:
    """Return validated command-line controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--h5-background", type=Path, default=DEFAULT_H5_DYNAMICS_PATH)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="lzf")
    arguments = parser.parse_args()
    if arguments.frames < 1:
        parser.error("--frames must be positive")
    return arguments


def main() -> None:
    """Run the canonical workflow in temporary storage and print timings."""
    arguments = _parse_arguments()
    total_started = time.perf_counter()
    dynamics_config = canonical_dynamics_config(arguments.frames)
    backgrounds = load_h5_background_sequence(
        arguments.h5_background,
        n_frames=arguments.frames,
        seed=int(dynamics_config["seed"]),
        start_index=0,
        frame_step=1,
    )
    dynamics_config["cadence_s"] = backgrounds["cadence_s"]
    backgrounds_finished = time.perf_counter()
    initial = canonical_static_result(
        arguments.h5_background,
        backgrounds=backgrounds,
    )
    static_finished = time.perf_counter()

    compression = None if arguments.compression == "none" else arguments.compression
    export_config = make_export_config(compression=compression, gzip_level=4)
    with tempfile.TemporaryDirectory(prefix="filament-dynamics-benchmark-", dir="/tmp") as root:
        output_root = Path(root)
        saved = simulate_and_save_filament_dynamics(
            initial,
            dynamics_config,
            background_frames=backgrounds["frames"],
            simulations_root=output_root / "simulations",
            export_config=export_config,
            label="benchmark",
        )
        simulation_finished = time.perf_counter()
        gong_path = save_gong_video(saved["h5_path"], output_root / "gong.mp4", fps=30)
        gong_finished = time.perf_counter()
        velocity_path = save_velocity_video(
            saved["h5_path"], output_root / "velocity.mp4", fps=30
        )
        finished = time.perf_counter()
        report = {
            "frames": arguments.frames,
            "compression": arguments.compression,
            "background_selection_s": backgrounds_finished - total_started,
            "static_s": static_finished - backgrounds_finished,
            "simulation_s": simulation_finished - static_finished,
            "simulation_s_per_frame": (simulation_finished - static_finished)
            / arguments.frames,
            "gong_video_s": gong_finished - simulation_finished,
            "velocity_video_s": finished - gong_finished,
            "total_s": finished - total_started,
            "h5_size_mib": Path(saved["h5_path"]).stat().st_size / 2**20,
            "gong_size_mib": gong_path.stat().st_size / 2**20,
            "velocity_size_mib": velocity_path.stat().st_size / 2**20,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
