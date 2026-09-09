"""Load self-contained, unchanged HDF5 background sequences."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from time import monotonic

import h5py
import numpy as np

from .cache import stage_cache
from .paths import DEFAULT_BACKGROUNDS_DIR

BACKGROUND_VERSION = 1
DEFAULT_H5_DYNAMICS_PATH = DEFAULT_BACKGROUNDS_DIR
H5_TIME_SERIES_KEY = "time_series"


def _read_source_frame(
    dataset: h5py.Dataset,
    source_path: Path,
    index: int,
    cached_stages: dict | None = None,
) -> tuple[np.ndarray, str]:
    """Read once per source identity; cached observation arrays are immutable."""
    stat = source_path.stat()
    key = repr((str(source_path.resolve()), stat.st_size, stat.st_mtime_ns,
                stat.st_ctime_ns, dataset.name, dataset.shape, str(dataset.dtype), index))
    cache = None if cached_stages is None else stage_cache(cached_stages)
    frame = None if cache is None else cache.get("source_frame", key)
    if frame is None:
        frame = np.asarray(dataset[index])
        if cache is not None:
            frame.flags.writeable = False
            cache.put("source_frame", key, frame)
    return frame, key


def _sequence_read_options() -> dict[str, int]:
    """Retain decompressed time chunks while reading an aligned sequence."""
    size_mib = int(os.environ.get("FILAMENT_H5_CACHE_MIB", "512"))
    if size_mib < 0:
        raise ValueError("FILAMENT_H5_CACHE_MIB must be non-negative")
    return {"rdcc_nbytes": size_mib * 2**20, "rdcc_nslots": 10007}


def _array_sha256(array: np.ndarray) -> str:
    """Hash array dtype, shape, and contiguous bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


def resolve_background_path(path: str | Path, seed: int = 0) -> Path:
    """Select one library file reproducibly, or use an explicit sequence file."""
    source = Path(path).expanduser().resolve()
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("background seed must be a non-negative integer")
    if source.is_dir():
        files = sorted(source.glob("*.h5"))
        if not files:
            raise FileNotFoundError(
                f"No prepared backgrounds in {source}. Add supplied sequences or run "
                "python scripts/prepare_backgrounds.py."
            )
        return files[int(np.random.default_rng(seed).integers(len(files)))]
    if not source.is_file():
        raise FileNotFoundError(f"Prepared background is missing: {source}")
    return source


def background_info(path: str | Path) -> dict[str, object]:
    """Read the small sequence header and validate its geometry and time axis."""
    source = Path(path).resolve()
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("filos_background_version") != BACKGROUND_VERSION:
            raise ValueError(
                f"{source.name} is not a prepared FILOS background. "
                "Run python scripts/prepare_backgrounds.py on full-disk observations first."
            )
        dataset = handle[H5_TIME_SERIES_KEY]
        if dataset.ndim != 3 or dataset.shape[0] < 1 or min(dataset.shape[1:]) < 8:
            raise ValueError("time_series must have shape (frames, height >= 8, width >= 8)")
        if dataset.dtype.kind != "f":
            raise ValueError("time_series must contain floating-point intensities")
        info = {name: float(handle.attrs[name]) for name in (
            "native_pixel_km", "disk_mu", "limb_direction_deg", "cadence_s",
        )}
        if not all(np.isfinite(value) for value in info.values()):
            raise ValueError("background geometry and cadence must be finite")
        if info["native_pixel_km"] <= 0 or info["cadence_s"] <= 0:
            raise ValueError("background pixel size and cadence must be positive")
        if not 0 < info["disk_mu"] <= 1 or not 0 <= info["limb_direction_deg"] < 360:
            raise ValueError("background disk_mu must be in (0, 1] and limb direction in [0, 360)")
        times = np.asarray(handle["time_s"], dtype=float)
        expected = np.arange(dataset.shape[0]) * info["cadence_s"]
        if times.shape != expected.shape or not np.allclose(times, expected, rtol=0, atol=1e-6):
            raise ValueError("time_s must start at zero and match the recorded regular cadence")
        provenance = json.loads(handle.attrs.get("provenance", "{}"))
        if not isinstance(provenance, dict):
            raise ValueError("background provenance must be a JSON object")
        return {
            **info,
            "shape": tuple(dataset.shape),
            "dtype": str(dataset.dtype),
            "provenance": provenance,
        }


def load_h5_background_sequence(
    path: str | Path = DEFAULT_H5_DYNAMICS_PATH,
    *,
    n_frames: int,
    seed: int = 0,
    start_index: int = 0,
    frame_step: int = 1,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cached_stages: dict | None = None,
) -> dict[str, object]:
    """Read prepared frames and observing geometry without detection or recropping."""
    started = monotonic()
    for name, value, minimum in (
        ("n_frames", n_frames, 1), ("start_index", start_index, 0), ("frame_step", frame_step, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    source = resolve_background_path(path, seed)
    info = background_info(source)
    indices = start_index + frame_step * np.arange(n_frames, dtype=np.int64)
    if indices[-1] >= info["shape"][0]:
        raise ValueError(
            f"{source.name} has {info['shape'][0]} frames; requested frame {indices[-1]}. "
            "Reduce the frame count/start/step or choose a longer background."
        )
    with h5py.File(source, "r", **_sequence_read_options()) as handle:
        dataset = handle[H5_TIME_SERIES_KEY]
        if n_frames == 1:
            first, _ = _read_source_frame(dataset, source, start_index, cached_stages)
            frames = first[None].copy()
        else:
            frames = np.asarray(dataset[start_index:int(indices[-1]) + 1:frame_step])
    if not np.isfinite(frames).all() or np.any(frames <= 0):
        raise ValueError("prepared background frames must contain finite positive intensities")
    height, width = info["shape"][1:]
    bounds = (0, 0, width, height)
    stat = source.stat()
    metadata = {
        "background_source": "prepared_sequence",
        "source_path": str(source),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dataset_key": H5_TIME_SERIES_KEY,
        "dataset_shape": list(info["shape"]),
        "dataset_dtype": info["dtype"],
        "frame_indices": indices.tolist(),
        "crop_xyxy_px": list(bounds),
        "crop_shape": [height, width],
        "native_pixel_km": info["native_pixel_km"],
        "background_disk_mu": info["disk_mu"],
        "background_limb_direction_deg": info["limb_direction_deg"],
        "cadence_s": info["cadence_s"] * frame_step,
        "selection_seed": int(seed),
        "provenance": info["provenance"],
        "processing_applied": "none",
        "frames_sha256": _array_sha256(frames),
    }
    if progress_callback is not None:
        progress_callback({
            "stage": "background", "completed": 1, "total": 1,
            "elapsed_seconds": monotonic() - started,
            "description": f"Prepared background loaded: {source.name}.",
        })
    return {
        "source_path": source,
        "frames": frames,
        "frame_indices": indices,
        "crop_xyxy_px": bounds,
        "native_shape": (height, width),
        "native_pixel_km": info["native_pixel_km"],
        "disk_mu": info["disk_mu"],
        "limb_direction_deg": info["limb_direction_deg"],
        "cadence_s": info["cadence_s"] * frame_step,
        "metadata": metadata,
    }
