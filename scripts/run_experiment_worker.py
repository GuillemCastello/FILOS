#!/usr/bin/env python3
"""Run one immutable local-experiment snapshot in a background process."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filaments.experiment_runner import run_worker_job  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse explicit worker paths supplied by the GUI launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-directory", type=Path, required=True)
    parser.add_argument("--experiment-directory", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Execute one job, leaving all status and diagnostics on disk."""
    arguments = _parse_arguments()
    print(f"Worker PID: {os.getpid()}", flush=True)
    print(f"Experiment: {arguments.experiment_directory.resolve()}", flush=True)
    print(f"Job: {arguments.job_directory.resolve()}", flush=True)
    try:
        saved = run_worker_job(
            arguments.job_directory,
            arguments.experiment_directory,
            experiments_root=arguments.experiments_root,
        )
    except BaseException:
        traceback.print_exc()
        raise
    print(f"Completed: {saved['directory']}", flush=True)


if __name__ == "__main__":
    main()
