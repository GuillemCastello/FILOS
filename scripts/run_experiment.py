#!/usr/bin/env python3
"""Run an experiment TOML through the same preview and worker workflow as the GUI."""

import argparse
import time
from pathlib import Path

from synthetic_filaments import (
    create_experiment,
    generate_experiment_preview,
    list_jobs,
    load_experiment_config,
    start_experiment_worker,
)
from synthetic_filaments.experiment_config import DEFAULT_EXPERIMENTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default_experiment.toml"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    print("Generating static preview…", flush=True)
    preview = generate_experiment_preview(config)
    experiment = create_experiment(config["experiment"]["name"], source_config=config)
    job = start_experiment_worker(experiment, user_config=config, preview=preview)
    previous = None
    while True:
        status = next(item for item in list_jobs(DEFAULT_EXPERIMENTS_ROOT)
                      if item["job_id"] == job["job_id"])
        progress = (status["state"], status.get("stage"), status.get("completed_frames"))
        if progress != previous:
            print(f"{progress[0]}: {progress[1]} ({progress[2]} frames)", flush=True)
            previous = progress
        if status["state"] == "completed":
            print(f"Saved {status['output_directory']}", flush=True)
            return
        if status["state"] == "failed":
            raise RuntimeError(f"{status['error']} — see {status['job_directory']}/run.log")
        time.sleep(1)


if __name__ == "__main__":
    main()
