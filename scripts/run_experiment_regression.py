#!/usr/bin/env python3
"""Exercise TOML, preview, background-worker, versioning, clone, and failure paths."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filaments import (  # noqa: E402
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    OSCILLATION_MODE_LUNA_2022,
    OSCILLATION_MODE_SHARED_PERIOD,
    clone_simulation_as_experiment,
    create_experiment,
    generate_experiment_preview,
    list_jobs,
    load_experiment_config,
    load_static_result,
    luna_2022_longitudinal_period_s,
    next_background_seed,
    next_experiment_seeds,
    normalize_experiment_config,
    preview_is_current,
    save_experiment_config,
    start_experiment_worker,
    static_preview_fingerprint,
    validate_experiment_config,
    validate_preview_config,
    validate_production_config,
    validate_save_config,
)


def _array_sha256(array: np.ndarray) -> str:
    """Hash array dtype, shape, and contiguous bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


def _check_runtime_cache_contract() -> None:
    """Check bounded reuse and source replacement using disposable derived inputs."""
    from synthetic_filaments import geometry, opacity_table
    from synthetic_filaments.cache import StageCache, cache_memory_info
    from synthetic_filaments.dynamic_background import _read_source_frame

    cache = StageCache()
    original_limit = os.environ.get("FILAMENT_CACHE_GIB")
    original_spine_path = geometry.SPINE_LIBRARY_PATH
    original_spine_cache = geometry._SPINE_LIBRARY_CACHE
    original_opacity_path = opacity_table.HEINZEL_EXTENSION_PATH
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    try:
        os.environ["FILAMENT_CACHE_GIB"] = "16"
        data = np.zeros(1_000_000, dtype=np.uint8)
        cache.put("probe", "whole", data)
        before = cache_memory_info()["payload_bytes"]
        cache.put("probe", "view", data[1:])
        if cache_memory_info()["payload_bytes"] - before >= 1024:
            raise AssertionError("RAM cache double-counts shared NumPy backing storage")
        cache.clear()
        os.environ["FILAMENT_CACHE_GIB"] = str(2_100_000 / 2**30)
        cache.put("probe", "a", data)
        cache.put("probe", "b", np.ones_like(data))
        assert cache.get("probe", "a") is data
        cache.put("probe", "c", np.ones_like(data))
        if cache.get("probe", "b") is not None or cache.get("probe", "a") is not data:
            raise AssertionError("RAM cache eviction did not preserve most recently used entry")
        os.environ["FILAMENT_CACHE_GIB"] = "0"
        if cache.get("probe", "a") is not None or np.any(data):
            raise AssertionError("disabled cache retained an entry or invalidated caller data")
        os.environ["FILAMENT_CACHE_GIB"] = "16"
        with tempfile.TemporaryDirectory(prefix="cache-regression-", dir=scratch) as directory:
            root = Path(directory)
            source = root / "source.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("time_series", data=np.ones((2, 8, 8), dtype=np.float32))
            stages: dict[str, object] = {}
            with h5py.File(source, "r") as handle:
                first, key = _read_source_frame(handle["time_series"], source, 0, stages)
                again, _ = _read_source_frame(handle["time_series"], source, 0, stages)
                if first is not again or first.flags.writeable:
                    raise AssertionError("decoded source frame was reread or remained writable")
            stamp = source.stat()
            with h5py.File(source, "r+") as handle:
                handle["time_series"][0, 0, 0] = 2.0
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with h5py.File(source, "r") as handle:
                changed, changed_key = _read_source_frame(handle["time_series"], source, 0, stages)
                if changed_key == key or changed[0, 0] != 2.0 or first[0, 0] != 1.0:
                    raise AssertionError("source replacement reused stale decoded data")

            copied_spine = root / "spine.npz"
            shutil.copy2(original_spine_path, copied_spine)
            geometry.SPINE_LIBRARY_PATH = copied_spine
            first_library = geometry._load_spine_library()
            stamp = copied_spine.stat()
            copied_spine.write_bytes(copied_spine.read_bytes())
            os.utime(copied_spine, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            if geometry._load_spine_library() is first_library:
                raise AssertionError("spine library ignored a changed runtime source identity")

            copied_table = root / "opacity.csv"
            shutil.copy2(original_opacity_path, copied_table)
            opacity_table.HEINZEL_EXTENSION_PATH = copied_table
            opacity_table.load_heinzel_opacity_table()
            stamp = copied_table.stat()
            contents = copied_table.read_bytes()
            copied_table.write_bytes(b"X" + contents[1:])
            os.utime(copied_table, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            try:
                opacity_table.load_heinzel_opacity_table()
            except ValueError as error:
                if "checksum" not in str(error):
                    raise
            else:
                raise AssertionError("opacity loader reused a replaced calibration without validation")
    finally:
        cache.clear()
        geometry.SPINE_LIBRARY_PATH = original_spine_path
        geometry._SPINE_LIBRARY_CACHE = original_spine_cache
        opacity_table.HEINZEL_EXTENSION_PATH = original_opacity_path
        if original_limit is None:
            os.environ.pop("FILAMENT_CACHE_GIB", None)
        else:
            os.environ["FILAMENT_CACHE_GIB"] = original_limit


def _assert_same_types_and_values(left: object, right: object, path: str = "config") -> None:
    """Require recursively identical TOML values and Python container/scalar types."""
    if type(left) is not type(right):
        raise AssertionError(f"{path} type changed: {type(left).__name__} -> {type(right).__name__}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path} keys changed")
        for key in left:
            _assert_same_types_and_values(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{path} length changed")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_same_types_and_values(left_item, right_item, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path} value changed: {left!r} -> {right!r}")


def _video_shape(path: Path) -> tuple[int, int, int]:
    """Return encoded video width, height, and frame count through FFprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("FFprobe is required by the GUI video regression")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])


def _wait_for_job(
    job_id: str,
    experiments_root: Path,
    experiment: Path,
    timeout_s: float = 600.0,
) -> dict[str, object]:
    """Wait for one detached worker to reach a terminal durable state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        matching = [
            job
            for job in list_jobs(experiments_root, experiment_directory=experiment)
            if job.get("job_id") == job_id
        ]
        if matching and matching[0].get("state") in {"completed", "failed"}:
            return matching[0]
        time.sleep(0.5)
    raise TimeoutError(f"worker {job_id} did not finish within {timeout_s:.0f} s")


def _run_current_experiment(
    experiment: Path,
    experiments_root: Path,
    config: dict[str, dict[str, object]],
    preview: dict[str, object],
) -> dict[str, object]:
    """Launch and wait for the current experiment, requiring complete products."""
    launched = start_experiment_worker(
        experiment,
        experiments_root=experiments_root,
        user_config=config,
        preview=preview,
    )
    status = _wait_for_job(launched["job_id"], experiments_root, experiment)
    if status["state"] != "completed":
        log_path = Path(status["job_directory"]) / "run.log"
        log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        raise AssertionError(f"worker failed: {status.get('error')}\n{log}")
    output = Path(status["output_directory"])
    required = (
        output / "experiment.toml",
        output / "simulation.h5",
        output / "simulation.json",
        output / "gong.mp4",
        output / "velocity.mp4",
        output / "run.log",
        output / "status.json",
        output / "static_state/manifest.json",
        output / "frame_zero_comparison.png",
        output / "geometry_diagnostics.png",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"completed worker is missing products: {missing}")
    return status


def main() -> None:
    """Run the complete local-experiment acceptance workflow in temporary storage."""
    _check_runtime_cache_contract()
    default = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG_PATH)
    resolved = validate_experiment_config(default)
    expected = {
        "static_seed": 1235,
        "background_seed": 1236,
        "dynamics_seed": 1236,
        "n_frames": 240,
        "crop_shape": (448, 448),
        "mode": OSCILLATION_MODE_LUNA_2022,
        "compression": "lzf",
        "fps": 30,
        "quiver_stride_px": 8,
    }
    actual = {
        "static_seed": resolved["static_seed"],
        "background_seed": resolved["background_seed"],
        "dynamics_seed": resolved["dynamics"]["seed"],
        "n_frames": resolved["dynamics"]["n_frames"],
        "crop_shape": resolved["dynamic_background"]["crop_shape"],
        "mode": resolved["dynamics"]["oscillation_mode"],
        "compression": resolved["export"]["compression"],
        "fps": resolved["video"]["fps"],
        "quiver_stride_px": resolved["video"]["quiver_stride_px"],
    }
    if actual != expected:
        raise AssertionError(f"default TOML differs from notebook 07: {actual}")
    expected_dip_fields = {
        "dip_curvature_radius_min_km",
        "dip_curvature_radius_max_km",
        "dip_curvature_radius_median_km",
        "dip_curvature_radius_sigma_ln",
    }
    actual_dip_fields = {
        name for name in default["static"] if name.startswith("dip_")
    }
    if actual_dip_fields != expected_dip_fields:
        raise AssertionError(
            f"default TOML does not use the radius-only dip contract: {actual_dip_fields}"
        )
    if next_experiment_seeds(1235, 1236) != (931150013, 1959674300):
        raise AssertionError("seed randomization changed or is not deterministic")
    if next_background_seed(1236) != 136491877:
        raise AssertionError("background seed randomization changed or is not deterministic")

    legacy = deepcopy(default)
    del legacy["dynamic_background"]["seed"]
    legacy["static"].update(
        {
            "dip_parameterization": "dip_depth",
            "dip_depth_min_km": 500.0,
            "dip_depth_max_km": 12_000.0,
            "dip_depth_median_km": 3_000.0,
            "dip_depth_sigma_ln": 0.55,
        }
    )
    migrated = normalize_experiment_config(legacy)
    if migrated["dynamic_background"]["seed"] != legacy["dynamics"]["seed"]:
        raise AssertionError("legacy background seed migration changed crop selection")
    if "seed" in legacy["dynamic_background"]:
        raise AssertionError("background seed migration mutated its source mapping")
    if {
        name for name in migrated["static"] if name.startswith("dip_")
    } != expected_dip_fields:
        raise AssertionError("legacy dip-depth migration did not produce the radius-only contract")
    if any(name not in legacy["static"] for name in ("dip_parameterization", "dip_depth_min_km")):
        raise AssertionError("legacy dip-depth migration mutated its source mapping")
    for name in expected_dip_fields:
        if migrated["static"][name] != default["static"][name]:
            raise AssertionError(f"legacy migration changed existing radius control {name}")

    static_fingerprint = static_preview_fingerprint(default)
    for section, field, replacement in (
        ("dynamics", "seed", 9),
        ("dynamics", "n_frames", 1),
        ("export", "gzip_level", 1),
        ("video", "fps", 12),
        ("dynamic_background", "frame_step", 2),
        ("experiment", "description", "fingerprint probe"),
    ):
        changed = deepcopy(default)
        changed[section][field] = replacement
        if static_preview_fingerprint(changed) != static_fingerprint:
            raise AssertionError(f"run-only edit invalidated static preview: {section}.{field}")
    changed = deepcopy(default)
    changed["static"]["seed"] += 1
    if static_preview_fingerprint(changed) == static_fingerprint:
        raise AssertionError("static seed did not invalidate the static preview")

    invalid_run = deepcopy(default)
    invalid_run["dynamics"]["n_frames"] = 0
    invalid_run["video"]["fps"] = 0
    validate_preview_config(invalid_run)
    for validator in (validate_save_config, validate_production_config):
        try:
            validator(invalid_run)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{validator.__name__} accepted invalid run-only controls")

    cache_config = deepcopy(default)
    cache_config["static"]["thread_count_cap"] = 1
    cache_config["dynamic_background"]["use_detector"] = False
    stage_cache: dict[str, object] = {}
    cache_base = generate_experiment_preview(cache_config, cached_stages=stage_cache)

    plasma_config = deepcopy(cache_config)
    plasma_config["static"]["temp_center_K"] = 8_000.0
    plasma_cached = generate_experiment_preview(plasma_config, cached_stages=stage_cache)
    plasma_fresh = generate_experiment_preview(plasma_config, cached_stages={})
    if (
        cache_base["cache_keys"]["geometry"] != plasma_cached["cache_keys"]["geometry"]
        or cache_base["cache_keys"]["plasma"] == plasma_cached["cache_keys"]["plasma"]
    ):
        raise AssertionError("plasma-only edit did not reuse exactly the geometry stage")

    observation_config = deepcopy(plasma_config)
    observation_config["static"]["psf_sigma_px"] = 3.5
    observation_cached = generate_experiment_preview(
        observation_config,
        cached_stages=stage_cache,
    )
    observation_fresh = generate_experiment_preview(observation_config, cached_stages={})
    reverted = generate_experiment_preview(cache_config, cached_stages=stage_cache)
    if reverted["static_state"] is not cache_base["static_state"]:
        raise AssertionError("returning to previous settings discarded a retained realization")
    if (
        observation_cached["display"]["geometry_diagnostics_png"]
        is not plasma_cached["display"]["geometry_diagnostics_png"]
        or observation_cached["display"]["luna_dynamics_diagnostics_png"]
        is not cache_base["display"]["luna_dynamics_diagnostics_png"]
    ):
        raise AssertionError("observation edits recomputed unchanged diagnostic figures")

    background_config = deepcopy(cache_config)
    background_config["dynamic_background"]["seed"] = next_background_seed(
        cache_config["dynamic_background"]["seed"]
    )
    background_cached = generate_experiment_preview(background_config, cached_stages=stage_cache)
    background_fresh = generate_experiment_preview(background_config, cached_stages={})
    if (
        background_cached["cache_keys"]["geometry"] != cache_base["cache_keys"]["geometry"]
        or background_cached["cache_keys"]["plasma"] != cache_base["cache_keys"]["plasma"]
    ):
        raise AssertionError("crop-seed edits recomputed unchanged intrinsic physical state")
    if (
        plasma_cached["cache_keys"]["plasma"] != observation_cached["cache_keys"]["plasma"]
        or plasma_cached["cache_keys"]["optical_depth"]
        != observation_cached["cache_keys"]["optical_depth"]
        or plasma_cached["cache_keys"]["render"] == observation_cached["cache_keys"]["render"]
    ):
        raise AssertionError("observation-only edit did not reuse plasma and optical depth")

    for label, cached, fresh in (
        ("plasma", plasma_cached, plasma_fresh),
        ("observation", observation_cached, observation_fresh),
        ("background", background_cached, background_fresh),
    ):
        for array_name in ("tau_map", "degraded_intensity"):
            if not np.array_equal(
                cached["static_state"]["arrays"][array_name],
                fresh["static_state"]["arrays"][array_name],
            ):
                raise AssertionError(
                    f"{label}-edit cached {array_name} differs from a fresh computation"
                )
        if cached["static_state"]["config"] != fresh["static_state"]["config"]:
            raise AssertionError(f"{label}-edit cached state retained a stale configuration")

    with tempfile.TemporaryDirectory(prefix="filament-experiment-regression-", dir="/tmp") as root:
        regression_root = Path(root) / "experiments"
        roundtrip_path = Path(root) / "roundtrip.toml"
        save_experiment_config(default, roundtrip_path)
        roundtrip = load_experiment_config(roundtrip_path)
        _assert_same_types_and_values(default, roundtrip)

        experiment = create_experiment(
            "worker regression",
            experiments_root=regression_root,
            source_config=default,
        )
        config_path = experiment / "experiment.toml"
        config = load_experiment_config(config_path)
        config["experiment"]["description"] = "Three-frame Luna and manual worker validation"
        config["dynamics"]["n_frames"] = 3
        save_experiment_config(config, config_path)
        before_preview = {
            path.relative_to(experiment): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in experiment.rglob("*")
            if path.is_file()
        }
        cache: dict[str, object] = {}
        luna_preview = generate_experiment_preview(config, cached_stages=cache)
        after_preview = {
            path.relative_to(experiment): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in experiment.rglob("*")
            if path.is_file()
        }
        if before_preview != after_preview or (experiment / "preview").exists():
            raise AssertionError("memory-only preview created or modified application files")
        if not preview_is_current(config, luna_preview):
            raise AssertionError("new in-memory preview was not marked current")
        if not all(luna_preview["display"].values()):
            raise AssertionError("in-memory preview display products are missing")
        if set(luna_preview["display"]) != {
            "frame_zero_comparison_png",
            "geometry_diagnostics_png",
            "luna_dynamics_diagnostics_png",
        }:
            raise AssertionError("in-memory preview has the wrong display products")
        preview_threads = luna_preview["static_state"]["threads"]
        expected_periods_s = np.asarray(
            luna_2022_longitudinal_period_s(
                1_000.0
                * np.asarray(
                    [thread["dip_radius_km"] for thread in preview_threads], dtype=float
                ),
                1_000.0
                * np.asarray(
                    [thread["height_km"] for thread in preview_threads], dtype=float
                ),
            ),
            dtype=float,
        )
        expected_summary = luna_preview["luna_dynamics"]["expected_period_s"]
        if not np.allclose(
            [
                expected_summary["minimum"],
                expected_summary["median"],
                expected_summary["maximum"],
            ],
            [
                np.min(expected_periods_s),
                np.median(expected_periods_s),
                np.max(expected_periods_s),
            ],
            rtol=0.0,
            atol=0.0,
        ):
            raise AssertionError("preview Luna summary does not match realized R and h")
        if "effective_thread_length_bounds_km" not in luna_preview["thread_placement"]:
            raise AssertionError("preview is missing effective thread-length diagnostics")
        first_luna = _run_current_experiment(
            experiment, regression_root, config, luna_preview
        )
        second_luna = _run_current_experiment(
            experiment, regression_root, config, luna_preview
        )
        if first_luna["output_directory"] == second_luna["output_directory"]:
            raise AssertionError("two generations reused the same immutable result directory")

        first_output = Path(first_luna["output_directory"])
        with h5py.File(first_output / "simulation.h5", "r") as handle:
            if handle["video/raw_gong"].shape[0] != 3:
                raise AssertionError("Luna worker wrote the wrong frame count")
            frame_zero = np.asarray(handle["video/raw_gong"][0])
            if not np.all(
                np.asarray(handle["dynamics/thread_longitudinal_periods_s"])
                < np.asarray(handle["dynamics/thread_gravity_cutoff_periods_s"])
            ):
                raise AssertionError("a Luna thread period reached its gravity cut-off")
            for path in (
                "geometry/initial/point_temperature_K",
                "geometry/initial/point_pressure_dyn_cm2",
                "geometry/initial/point_mean_molecular_mass",
                "geometry/initial/point_loaded_mask",
                "geometry/initial/thread_column_mass_converged",
            ):
                if path not in handle:
                    raise AssertionError(f"production HDF5 is missing {path}")
        if _array_sha256(frame_zero) != luna_preview["frame_zero"]["frame_sha256"]:
            raise AssertionError("Luna persisted frame zero differs from its preview")
        restored = load_static_result(first_output / "static_state")
        if _array_sha256(restored["arrays"]["degraded_intensity"]) != luna_preview[
            "frame_zero"
        ]["frame_sha256"]:
            raise AssertionError("static-state snapshot roundtrip changed frame zero")
        gong_shape = _video_shape(first_output / "gong.mp4")
        velocity_shape = _video_shape(first_output / "velocity.mp4")
        if gong_shape[0] >= 448 or gong_shape[2] != 3:
            raise AssertionError(f"H-alpha video was not modestly cropped: {gong_shape}")
        if (
            velocity_shape[0] != gong_shape[0]
            or velocity_shape[1] != gong_shape[1] + 8
            or velocity_shape[2] != 3
        ):
            raise AssertionError(
                "velocity and H-alpha videos do not have matched side-by-side dimensions: "
                f"gong={gong_shape}, velocity={velocity_shape}"
            )
        with (first_output / "simulation.json").open(encoding="utf-8") as metadata_file:
            run_metadata = json.load(metadata_file)
        velocity_visualization = run_metadata.get("velocity_video_visualization", {})
        if velocity_visualization.get("render_version") != 5:
            raise AssertionError(
                f"velocity-video visualization metadata is incomplete: {velocity_visualization}"
            )
        if velocity_visualization.get("arrow_length_encodes_speed") is not True:
            raise AssertionError("velocity-video arrow semantics are not explicit")
        if velocity_visualization.get("arrow_color") != "black":
            raise AssertionError("velocity-video arrow color is not explicit")

        clone = clone_simulation_as_experiment(
            first_output,
            "cloned worker run",
            experiments_root=regression_root,
        )
        clone_config = load_experiment_config(clone / "experiment.toml")
        source_snapshot = load_experiment_config(first_output / "experiment.toml")
        clone_without_name = deepcopy(clone_config)
        source_without_name = deepcopy(source_snapshot)
        clone_without_name["experiment"]["name"] = "same"
        source_without_name["experiment"]["name"] = "same"
        _assert_same_types_and_values(source_without_name, clone_without_name)

        config = load_experiment_config(config_path)
        config["dynamics"]["oscillation_mode"] = OSCILLATION_MODE_SHARED_PERIOD
        if not preview_is_current(config, luna_preview):
            raise AssertionError("dynamics mode edit invalidated the static preview")
        cached_luna_preview = generate_experiment_preview(config, cached_stages=cache)
        if (
            cached_luna_preview["cache_keys"]["display"]
            != luna_preview["cache_keys"]["display"]
            or cached_luna_preview["display"]["luna_dynamics_diagnostics_png"]
            != luna_preview["display"]["luna_dynamics_diagnostics_png"]
            or cached_luna_preview["luna_dynamics"] != luna_preview["luna_dynamics"]
        ):
            raise AssertionError("dynamics-only edit changed the static Luna diagnostics")
        manual = _run_current_experiment(
            experiment, regression_root, config, luna_preview
        )
        with h5py.File(Path(manual["output_directory"]) / "simulation.h5", "r") as handle:
            longitudinal = np.asarray(handle["dynamics/thread_longitudinal_periods_s"])
            transverse = np.asarray(handle["dynamics/thread_transverse_periods_s"])
        if not np.all(longitudinal == 3600.0) or not np.array_equal(longitudinal, transverse):
            raise AssertionError("manual worker did not persist shared 3600 s periods")

        completed_before_failure = len(list((experiment / "runs").glob("sim-*")))
        config = load_experiment_config(config_path)
        config["dynamics"]["sphere_radius_km"] = 1.0
        if not preview_is_current(config, luna_preview):
            raise AssertionError("run-only failure input invalidated the static preview")
        failed_launch = start_experiment_worker(
            experiment,
            experiments_root=regression_root,
            user_config=config,
            preview=luna_preview,
        )
        failed = _wait_for_job(failed_launch["job_id"], regression_root, experiment)
        if failed["state"] != "failed" or not failed.get("failure_directory"):
            raise AssertionError(f"deliberate worker failure was not retained: {failed}")
        failure_directory = Path(failed["failure_directory"])
        if not all(
            (failure_directory / name).is_file()
            for name in ("experiment.toml", "preview.json", "run.log", "status.json")
        ) or not (failure_directory / "static_state/manifest.json").is_file():
            raise AssertionError("failure directory is missing configuration, log, or status")
        if len(list((experiment / "runs").glob("sim-*"))) != completed_before_failure:
            raise AssertionError("failed worker left a falsely completed sim-* directory")

        report = {
            "default_profile": actual,
            "roundtrip": "pass",
            "preview_frame_zero_sha256": luna_preview["frame_zero"]["frame_sha256"],
            "preview_application_files_unchanged": before_preview == after_preview,
            "preview_cache_stages": sorted(cache),
            "dependency_cache_identity": "pass",
            "luna_runs": [first_luna["output_directory"], second_luna["output_directory"]],
            "manual_run": manual["output_directory"],
            "clone": str(clone),
            "failure_directory": str(failure_directory),
        }
        print(json.dumps(report, indent=2))
    print("PASS — local experiment GUI workflow regression completed")


if __name__ == "__main__":
    main()
