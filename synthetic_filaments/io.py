"""Atomic persistence for static forward-model results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

STATIC_OUTPUT_SCHEMA_VERSION = 5


def json_ready(value: Any) -> Any:
    """Convert nested NumPy and path values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def generator_git_commit() -> str:
    """Return the repository commit recorded in generated artifacts."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def generated_utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for persisted metadata."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    """Write one indented JSON document with deterministic key ordering."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    """Return the uppercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _split_array_fields(record: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Separate NumPy arrays from JSON-compatible values in one record."""
    arrays = {
        name: np.asarray(value)
        for name, value in record.items()
        if isinstance(value, np.ndarray)
    }
    values = {name: value for name, value in record.items() if name not in arrays}
    return arrays, values


def _pack_thread_arrays(
    threads: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """Pack every one-dimensional thread array without using object storage."""
    array_fields = sorted(
        {
            name
            for thread in threads
            for name, value in thread.items()
            if isinstance(value, np.ndarray)
        }
    )
    packed: dict[str, np.ndarray] = {}
    scalar_threads: list[dict[str, Any]] = []
    for thread_index, thread in enumerate(threads):
        arrays, values = _split_array_fields(thread)
        missing = sorted(set(array_fields) - set(arrays))
        if missing:
            raise ValueError(
                f"thread {thread_index} is missing array fields required for serialization: {missing}"
            )
        scalar_threads.append(values)

    for name in array_fields:
        values = [np.asarray(thread[name]) for thread in threads]
        if any(value.ndim != 1 for value in values):
            raise ValueError(f"thread array field {name!r} must be one-dimensional")
        dtypes = {value.dtype.str for value in values}
        if len(dtypes) > 1:
            raise ValueError(f"thread array field {name!r} has inconsistent dtypes")
        counts = np.asarray([value.size for value in values], dtype=np.int64)
        packed[f"offsets__{name}"] = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        packed[f"values__{name}"] = (
            np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        )
    return packed, scalar_threads, array_fields


def _verified_manifest(directory: Path) -> dict[str, Any]:
    """Load one static-state manifest and verify every recorded file digest."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    legacy_version = json.load(handle).get("static_output_schema_version")
            except (AttributeError, json.JSONDecodeError):
                legacy_version = "unknown"
            raise ValueError(
                "unsupported legacy static-state schema without a checksum manifest; "
                f"expected version {STATIC_OUTPUT_SCHEMA_VERSION}, received {legacy_version!r}"
            )
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    version = manifest.get("static_output_schema_version")
    if version != STATIC_OUTPUT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported static-state schema version; "
            f"expected {STATIC_OUTPUT_SCHEMA_VERSION}, received {version!r}"
        )
    checksums = manifest.get("file_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("static-state manifest is missing file checksums")
    for relative_name, expected in checksums.items():
        path = directory / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"static-state checksum mismatch for {relative_name}: "
                f"expected {expected}, received {actual}"
            )
    return manifest


def _thread_geometry_arrays(threads: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Flatten ragged thread geometry into offsets plus aligned point arrays."""
    point_counts = np.asarray([np.asarray(thread["s"]).size for thread in threads], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(point_counts, dtype=np.int64)])

    def concatenate(name: str, dtype: np.dtype | type = np.float64) -> np.ndarray:
        if not threads:
            return np.empty(0, dtype=dtype)
        return np.concatenate([np.asarray(thread[name], dtype=dtype) for thread in threads])

    scalar_fields = (
        "radius_km",
        "height_km",
        "dip_depth_km",
        "dip_radius_km",
        "length_km",
        "spine_anchor_km",
        "transverse_offset_km",
        "pitch_deg",
        "tau0",
        "fill_depth_km",
        "filled_depth_km",
        "fill_fraction",
        "realized_fill_fraction",
        "requested_column_mass_g_cm2",
        "realized_column_mass_g_cm2",
        "column_mass_capacity_g_cm2",
        "loaded_length_fraction",
        "loaded_length_km",
        "column_mass_residual_g_cm2",
        "column_mass_relative_residual",
        "column_mass_target_tolerance_g_cm2",
        "column_mass_lower_endpoint_g_cm2",
        "column_mass_upper_endpoint_g_cm2",
        "column_mass_last_refinement_change_g_cm2",
        "column_mass_integration_relative_tolerance",
    )
    arrays: dict[str, np.ndarray] = {
        "thread_offsets": offsets,
        "s_km": concatenate("s"),
        "x_km": concatenate("x"),
        "y_km": concatenate("y"),
        "z_km": concatenate("z"),
        "spine_coordinate_km": concatenate("s_spine"),
        "tau_profile": concatenate("tau_along"),
        "radius_profile_km": concatenate("radius_along_km"),
        "temperature_K": concatenate("temperature_K"),
        "pressure_dyn_cm2": concatenate("pressure_dyn_cm2"),
        "mean_molecular_mass": concatenate("mean_molecular_mass"),
        "loaded_mask": concatenate("loaded_mask", dtype=bool),
    }
    for name in scalar_fields:
        arrays[name] = np.asarray([thread[name] for thread in threads], dtype=np.float64)
    arrays["column_mass_capacity_limited"] = np.asarray(
        [thread["column_mass_capacity_limited"] for thread in threads], dtype=bool
    )
    for name in (
        "column_mass_converged",
        "column_mass_lower_resolution_limited",
        "column_mass_numerical_failure",
    ):
        arrays[name] = np.asarray([thread[name] for thread in threads], dtype=bool)
    for name in (
        "column_mass_bisection_iterations",
        "column_mass_integration_intervals",
        "column_mass_integration_points",
        "column_mass_refinement_levels",
    ):
        arrays[name] = np.asarray([thread[name] for thread in threads], dtype=np.int64)
    for name in (
        "height_low",
        "height_high",
        "temperature_low",
        "temperature_high",
        "pressure_low",
        "pressure_high",
    ):
        arrays[f"opacity_saturation_{name}_count"] = np.asarray(
            [thread.get("opacity_table_saturation", {}).get(name, 0) for thread in threads],
            dtype=np.int64,
        )
    return arrays


def stage_static_result(result: dict[str, Any], target: str | Path) -> Path:
    """Write a complete static result into an unpublished staging directory."""
    output = Path(target).resolve()
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    arrays_directory = staging / "arrays"
    arrays_directory.mkdir()
    try:
        array_names = sorted(result["arrays"])
        for name in array_names:
            values = result["arrays"][name]
            np.save(arrays_directory / f"{name}.npy", np.asarray(values), allow_pickle=False)
        np.savez_compressed(
            arrays_directory / "thread_geometry.npz",
            **_thread_geometry_arrays(result["threads"]),
        )
        spine_arrays, spine_values = _split_array_fields(result["spine"])
        np.savez_compressed(arrays_directory / "spine.npz", **spine_arrays)
        packed_threads, thread_values, thread_array_fields = _pack_thread_arrays(
            result["threads"]
        )
        np.savez_compressed(arrays_directory / "threads.npz", **packed_threads)
        _write_json(staging / "config.json", result["config"])
        _write_json(staging / "spine.json", spine_values)
        _write_json(staging / "threads.json", thread_values)
        _write_json(
            staging / "metadata.json",
            {
                **result["metadata"],
                "static_output_schema_version": STATIC_OUTPUT_SCHEMA_VERSION,
                "generated_utc": generated_utc_now(),
                "generator_git_commit": generator_git_commit(),
            },
        )
        files = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        _write_json(
            staging / "manifest.json",
            {
                "static_output_schema_version": STATIC_OUTPUT_SCHEMA_VERSION,
                "array_names": array_names,
                "spine_array_fields": sorted(spine_arrays),
                "thread_array_fields": thread_array_fields,
                "thread_count": len(result["threads"]),
                "file_sha256": {
                    str(path.relative_to(staging)): _file_sha256(path) for path in files
                },
            },
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def publish_staged_result(staging: str | Path, target: str | Path) -> Path:
    """Atomically publish one staged result without replacing existing data."""
    source = Path(staging).resolve()
    output = Path(target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, output)
    return output


def save_static_result(result: dict[str, Any], target: str | Path) -> Path:
    """Stage and atomically publish all arrays, geometry, config, and metadata."""
    staging = stage_static_result(result, target)
    try:
        return publish_staged_result(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_static_result(source: str | Path) -> dict[str, Any]:
    """Load and verify one complete schema-5 static result without pickle."""
    directory = Path(source).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    manifest = _verified_manifest(directory)

    def load_json(name: str) -> Any:
        path = directory / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    arrays = {
        name: np.load(directory / "arrays" / f"{name}.npy", allow_pickle=False)
        for name in manifest["array_names"]
    }
    spine = dict(load_json("spine.json"))
    with np.load(directory / "arrays/spine.npz", allow_pickle=False) as stored_spine:
        for name in manifest["spine_array_fields"]:
            spine[name] = np.array(stored_spine[name], copy=True)

    scalar_threads = load_json("threads.json")
    if not isinstance(scalar_threads, list) or len(scalar_threads) != manifest["thread_count"]:
        raise ValueError("static-state thread scalar records do not match the manifest")
    threads = [dict(values) for values in scalar_threads]
    with np.load(directory / "arrays/threads.npz", allow_pickle=False) as stored_threads:
        for name in manifest["thread_array_fields"]:
            offsets = np.asarray(stored_threads[f"offsets__{name}"], dtype=np.int64)
            values = stored_threads[f"values__{name}"]
            if offsets.shape != (len(threads) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
                raise ValueError(f"invalid ragged offsets for thread field {name!r}")
            if np.any(np.diff(offsets) < 0):
                raise ValueError(f"non-monotonic ragged offsets for thread field {name!r}")
            for index, thread in enumerate(threads):
                thread[name] = np.array(values[offsets[index] : offsets[index + 1]], copy=True)

    config = load_json("config.json")
    metadata = load_json("metadata.json")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("static-state config and metadata must contain JSON objects")
    return {
        "config": config,
        "spine": spine,
        "threads": threads,
        "arrays": arrays,
        "metadata": metadata,
    }
