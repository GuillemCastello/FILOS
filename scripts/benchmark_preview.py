#!/usr/bin/env python3
"""Benchmark exact cold/cached previews in fresh processes; profile separately."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import resource
import statistics
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default_experiment.toml")
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--background-seed", type=int)
    parser.add_argument("--reference-cap", type=int)
    parser.add_argument("--small-cap", type=int, default=100)
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--disable-detector", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--profile", action="store_true", help="Run a separate cProfile pass")
    parser.add_argument("--dump-static", action="store_true", help="Retain trusted local probe input")
    parser.add_argument("--package-root", type=Path, default=ROOT,
                        help="Optional preserved source snapshot for before/after comparisons")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "scratch"):
        parser.error("--output must be under project scratch/")
    return args


def scientific_digest(state: dict[str, Any]) -> str:
    """Hash all physical arrays, thread fields and configuration in fixed order."""
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    for name in ("config", "spine", "threads", "arrays"):
        visit(state[name])
    visit(state["metadata"]["thread_placement"])
    return digest.hexdigest()


def _tree_rss(pid: int) -> int:
    """Sum Linux RSS for a process and descendants (shared pages count per process)."""
    total = 0
    pending = [pid]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            for task in Path(f"/proc/{current}/task").iterdir():
                try:
                    pending.extend(int(child) for child in (task / "children").read_text().split())
                except OSError:
                    continue
        except (OSError, ValueError):
            continue
    return total


def _worker(args: argparse.Namespace) -> None:
    from synthetic_filaments import detector, generate_experiment_preview

    config = json.loads(args.worker_config.read_text())
    cache: dict[str, Any] = {}
    rows = []
    reference = None
    for mode in ("cold", "warm"):
        stop = threading.Event()
        peak = [0]

        def sample() -> None:
            while not stop.is_set():
                peak[0] = max(peak[0], _tree_rss(os.getpid()))
                stop.wait(0.1)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        last = [0.0]

        def progress(event: dict[str, Any]) -> None:
            now = time.monotonic()
            if now - last[0] >= 25 or event.get("completed") == event.get("total"):
                print(f"{mode}: {event['stage']} {event.get('completed')}/{event.get('total')} "
                      f"{event['overall_elapsed_seconds']:.2f}s", flush=True)
                last[0] = now

        profiler = cProfile.Profile() if args.worker_profile else None
        if profiler is not None:
            profiler.enable()
        started = time.perf_counter()
        try:
            preview = generate_experiment_preview(config, cached_stages=cache,
                                                  progress_callback=progress)
            elapsed = time.perf_counter() - started
        finally:
            if profiler is not None:
                profiler.disable()
            stop.set()
            sampler.join()
        if profiler is not None:
            profiler.dump_stats(str(args.output.with_suffix(f".{mode}.pstats")))
        signature = scientific_digest(preview["static_state"])
        if reference is not None and signature != reference:
            raise AssertionError("cold and cached scientific state differs")
        reference = signature
        rows.append({
            "mode": mode, "elapsed_seconds": elapsed,
            "sampled_peak_process_tree_rss_bytes": peak[0],
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "stage_durations_seconds": preview["stage_durations_seconds"],
            "scientific_sha256": signature, "frame_zero": preview["frame_zero"],
            "cache_hits": sorted({e["stage"] for e in preview["progress_events"] if e.get("cached")}),
        })
        if mode == "cold" and args.dump_static:
            with args.output.with_suffix(".static.pkl").open("wb") as handle:
                pickle.dump(preview["static_state"], handle, protocol=5)
    state = preview["static_state"]
    parallel_support = importlib.util.find_spec("synthetic_filaments.execution") is not None
    configured_workers = int(os.environ.get("FILAMENT_PLASMA_WORKERS", "8"))
    batch_size = int(os.environ.get("FILAMENT_PLASMA_BATCH_SIZE", "32"))
    report = {
        "config": config, "measurements": rows,
        "instrumentation": "cProfile" if args.worker_profile else "wall_clock_with_RSS_sampling",
        "detector_device": None if detector._DETECTOR_STATE is None else str(
            detector._DETECTOR_STATE["device"]),
        "n_threads": len(state["threads"]),
        "sampled_point_count": sum(len(t["s"]) for t in state["threads"]),
        "placement": preview["thread_placement"],
        "column_mass_loading": preview["column_mass_loading"],
        "effective_execution": {
            "plasma_workers": (
                configured_workers if parallel_support and len(state["threads"]) > batch_size else 1
            ),
            "plasma_batch_size": batch_size if parallel_support else None,
            "cache_gib": float(os.environ.get("FILAMENT_CACHE_GIB", "16")) if parallel_support else None,
            "h5_sequence_cache_mib": int(os.environ.get("FILAMENT_H5_CACHE_MIB", "512"))
            if parallel_support else None,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = _arguments()
    sys.path.insert(0, str(args.package_root.resolve()))
    if args.workers is not None:
        os.environ["FILAMENT_PLASMA_WORKERS"] = str(args.workers)
    if args.worker_config is not None:
        _worker(args)
        return
    from synthetic_filaments import load_experiment_config

    config = load_experiment_config(args.config)
    if args.h5 is not None:
        config["inputs"]["h5_background_path"] = str(args.h5.resolve())
    else:
        source = Path(config["inputs"]["h5_background_path"])
        config["inputs"]["h5_background_path"] = str(
            source.resolve() if source.is_absolute() else (ROOT / source).resolve())
    for section, field, value in (("static", "seed", args.seed),
                                   ("dynamic_background", "seed", args.background_seed),
                                   ("static", "thread_count_cap", args.reference_cap)):
        if value is not None:
            config[section][field] = value
    if args.disable_detector:
        config["dynamic_background"]["use_detector"] = False
    artifacts = args.output.parent / (args.output.stem + "_runs")
    artifacts.mkdir(parents=True, exist_ok=True)
    cases = [("reference", config)]
    if not args.skip_small:
        small = deepcopy(config)
        small["static"]["thread_count_cap"] = args.small_cap
        cases.append(("small", small))
    reports = []
    for label, case in cases:
        config_path = artifacts / f"{label}_config.json"
        config_path.write_text(json.dumps(case, indent=2) + "\n")
        runs = []
        for repeat in range(args.repeats + int(args.profile)):
            profiling = repeat == args.repeats
            output = artifacts / f"{label}_{'profile' if profiling else repeat}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-config",
                       str(config_path), "--output", str(output), "--package-root",
                       str(args.package_root.resolve())]
            if profiling:
                command.append("--worker-profile")
            if args.dump_static and repeat == 0:
                command.append("--dump-static")
            subprocess.run(command, check=True, cwd=ROOT)
            if not profiling:
                runs.append(json.loads(output.read_text()))
        signatures = {m["scientific_sha256"] for r in runs for m in r["measurements"]}
        if len(signatures) != 1:
            raise AssertionError("scientific output changed across fresh-process repeats")
        reports.append({"label": label, "runs": runs, "median_seconds": {
            mode: statistics.median(m["elapsed_seconds"] for r in runs
                                    for m in r["measurements"] if m["mode"] == mode)
            for mode in ("cold", "warm")}})
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "package_root": str(args.package_root.resolve()),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "h5py", "matplotlib", "threadpoolctl")},
        "python": sys.version, "cpu_count": os.cpu_count(),
        "execution_environment": {k: v for k, v in os.environ.items()
                                  if k.startswith("FILAMENT_") or k == "CUDA_VISIBLE_DEVICES"},
        "cases": reports,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({c["label"]: c["median_seconds"] for c in reports}), flush=True)


if __name__ == "__main__":
    main()
