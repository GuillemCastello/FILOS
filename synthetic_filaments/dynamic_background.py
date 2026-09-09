"""Load aligned, unchanged HDF5 backgrounds for filament dynamics."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from numbers import Integral
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
from scipy.ndimage import binary_closing, find_objects, label

from . import config as config_module
from .cache import stage_cache
from .detector import detect_filament_boxes, detector_model_identity
from .paths import DEFAULT_FITS_DIR

DEFAULT_H5_DYNAMICS_PATH = DEFAULT_FITS_DIR / "20140101.h5"
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


def _infer_disk_geometry(
    frame: np.ndarray,
) -> tuple[tuple[float, float], float, np.ndarray]:
    """Infer the aligned disk center, radius, and positive support."""
    support = np.isfinite(frame) & (frame > 0.0)
    rows, columns = np.where(support)
    if rows.size == 0:
        raise ValueError("the first HDF5 frame contains no positive solar-disk support")

    x_min, x_max = int(columns.min()), int(columns.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    center_row = int(round(center_y))
    center_column = int(round(center_x))
    row_support = np.flatnonzero(support[center_row])
    column_support = np.flatnonzero(support[:, center_column])
    if row_support.size == 0 or column_support.size == 0:
        raise ValueError("could not measure the HDF5 solar disk through its center")

    radius_x = 0.5 * float(row_support[-1] - row_support[0])
    radius_y = 0.5 * float(column_support[-1] - column_support[0])
    if (
        radius_x <= 0.0
        or radius_y <= 0.0
        or abs(radius_x - radius_y) > 2.0
        or abs(center_x - 0.5 * (row_support[0] + row_support[-1])) > 2.0
        or abs(center_y - 0.5 * (column_support[0] + column_support[-1])) > 2.0
    ):
        raise ValueError("the positive HDF5 support is not an aligned approximately circular disk")
    return (float(center_x), float(center_y)), 0.5 * (radius_x + radius_y), support


def _off_disk_mask(
    frame_shape: tuple[int, int],
    *,
    disk_center: tuple[float, float],
    disk_radius: float,
) -> np.ndarray:
    """Return pixels whose centers lie beyond the inferred disk boundary."""
    rows, columns = np.ogrid[: frame_shape[0], : frame_shape[1]]
    radius_to_outer_pixel_edge = disk_radius + 0.5
    return (columns - disk_center[0]) ** 2 + (
        rows - disk_center[1]
    ) ** 2 > radius_to_outer_pixel_edge**2


def _filament_boxes(
    frame: np.ndarray,
    *,
    off_disk_mask: np.ndarray,
    use_detector: bool,
    detection_threshold: float,
) -> tuple[np.ndarray | None, str | None, dict[str, object] | None]:
    """Run the optional full-disk detector and normalize its error metadata."""
    if not use_detector:
        return None, "disabled", None
    try:
        boxes, preprocessing = detect_filament_boxes(
            frame,
            off_disk_mask=off_disk_mask,
            threshold=detection_threshold,
            return_preprocessing_metadata=True,
        )
    except (RuntimeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}", None
    return np.asarray(boxes, dtype=float).reshape(-1, 4), None, preprocessing


def _crop_overlaps_boxes(
    boxes: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    expand_fraction: float,
    expand_px: float,
) -> bool:
    """Return whether an expanded detector box intersects crop bounds."""
    expanded = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    if expanded.size == 0:
        return False
    widths = expanded[:, 2] - expanded[:, 0]
    heights = expanded[:, 3] - expanded[:, 1]
    pad_x = np.maximum(expand_fraction * widths, expand_px)
    pad_y = np.maximum(expand_fraction * heights, expand_px)
    expanded[:, 0] -= pad_x
    expanded[:, 1] -= pad_y
    expanded[:, 2] += pad_x
    expanded[:, 3] += pad_y
    x0, y0, x1, y1 = bounds
    return bool(
        np.any(
            (expanded[:, 0] < x1)
            & (expanded[:, 2] > x0)
            & (expanded[:, 1] < y1)
            & (expanded[:, 3] > y0)
        )
    )


def _quiet_crop_ok(
    crop: np.ndarray,
    *,
    dark_threshold: float = 0.92,
    bright_threshold: float = 1.12,
    dark_fraction: float = 0.02,
    bright_fraction: float = 0.03,
    maximum_span_fraction: float = 0.35,
) -> bool:
    """Reject crops containing large or elongated dark/bright structures."""
    median = float(np.median(crop))
    if median <= 0.0:
        return False
    normalized = crop / median
    maximum_dimension = max(crop.shape)
    connectivity = np.ones((3, 3), dtype=bool)
    masks_and_limits = (
        (normalized < dark_threshold, dark_fraction),
        (normalized > bright_threshold, bright_fraction),
    )
    for raw_mask, area_fraction in masks_and_limits:
        if not raw_mask.any():
            continue
        mask = raw_mask | binary_closing(raw_mask, structure=connectivity, iterations=2)
        labeled, component_count = label(mask, structure=connectivity)
        if component_count == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]
        for component_index, bounds in enumerate(find_objects(labeled)):
            if bounds is None:
                continue
            span = max(
                bounds[0].stop - bounds[0].start,
                bounds[1].stop - bounds[1].start,
            )
            if (
                sizes[component_index] > area_fraction * crop.size
                or span > maximum_span_fraction * maximum_dimension
            ):
                return False
    return True


def _candidate_bounds(
    *,
    frame_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    rng: np.random.Generator,
    random_attempts: int,
) -> list[tuple[int, int, int, int]]:
    """Return deterministic random candidates followed by a shuffled grid."""
    frame_ny, frame_nx = frame_shape
    crop_ny, crop_nx = crop_shape
    maximum_x0 = frame_nx - crop_nx
    maximum_y0 = frame_ny - crop_ny
    candidates = [
        (int(x0), int(y0), int(x0 + crop_nx), int(y0 + crop_ny))
        for x0, y0 in zip(
            rng.integers(0, maximum_x0 + 1, size=random_attempts),
            rng.integers(0, maximum_y0 + 1, size=random_attempts),
            strict=True,
        )
    ]

    stride = max(min(crop_shape) // 32, 16)
    grid_x = list(range(0, maximum_x0 + 1, stride))
    grid_y = list(range(0, maximum_y0 + 1, stride))
    if grid_x[-1] != maximum_x0:
        grid_x.append(maximum_x0)
    if grid_y[-1] != maximum_y0:
        grid_y.append(maximum_y0)
    grid = [(x0, y0, x0 + crop_nx, y0 + crop_ny) for y0 in grid_y for x0 in grid_x]
    permutation = rng.permutation(len(grid))
    candidates.extend(grid[int(index)] for index in permutation)
    return candidates


def _select_quiet_crop(
    frame: np.ndarray,
    *,
    crop_shape: tuple[int, int],
    support: np.ndarray,
    boxes: np.ndarray | None,
    rng: np.random.Generator,
    random_attempts: int,
    box_expand_fraction: float,
    box_expand_px: float,
) -> tuple[tuple[int, int, int, int], int]:
    """Select one supported, detector-clear, heuristic-quiet crop."""
    tested = 0
    invalid_support = (~support).astype(np.int32)
    invalid_integral = np.pad(
        invalid_support.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )
    for bounds in _candidate_bounds(
        frame_shape=frame.shape,
        crop_shape=crop_shape,
        rng=rng,
        random_attempts=random_attempts,
    ):
        x0, y0, x1, y1 = bounds
        invalid_count = (
            invalid_integral[y1, x1]
            - invalid_integral[y0, x1]
            - invalid_integral[y1, x0]
            + invalid_integral[y0, x0]
        )
        if invalid_count:
            continue
        if boxes is not None and _crop_overlaps_boxes(
            boxes,
            bounds,
            expand_fraction=box_expand_fraction,
            expand_px=box_expand_px,
        ):
            continue
        tested += 1
        if _quiet_crop_ok(frame[y0:y1, x0:x1]):
            return bounds, tested

    raise RuntimeError(
        "no fully supported filament-free HDF5 crop passed detector exclusion "
        f"and quiet-context screening after {tested} eligible candidates"
    )


def load_h5_background_frames_at_crop(
    path: str | Path = DEFAULT_H5_DYNAMICS_PATH,
    *,
    crop_xyxy_px: tuple[int, int, int, int],
    frame_indices: Sequence[int] | np.ndarray,
) -> dict[str, object]:
    """Read explicit HDF5 frames at fixed exclusive crop bounds.

    This loader performs no detector inference, random selection, or image
    processing. It is intended for production reuse of previously reviewed
    crop bounds.
    """
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if len(crop_xyxy_px) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in crop_xyxy_px
    ):
        raise ValueError(
            f"crop_xyxy_px must contain four integer exclusive bounds; received {crop_xyxy_px!r}"
        )
    requested_indices = list(frame_indices)
    if not requested_indices or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in requested_indices
    ):
        raise ValueError("frame_indices must be a nonempty sequence of integers")
    if any(value < 0 for value in requested_indices):
        raise ValueError("frame_indices must be non-negative")

    x0, y0, x1, y1 = (int(value) for value in crop_xyxy_px)
    indices = np.asarray(requested_indices, dtype=np.int64)
    source_stat = source_path.stat()
    with h5py.File(source_path, "r", **_sequence_read_options()) as handle:
        if H5_TIME_SERIES_KEY not in handle:
            raise KeyError(f"{source_path} is missing dataset {H5_TIME_SERIES_KEY!r}")
        dataset = handle[H5_TIME_SERIES_KEY]
        if dataset.ndim != 3:
            raise ValueError(
                f"{H5_TIME_SERIES_KEY!r} must have shape (time, y, x); received {dataset.shape}"
            )
        frame_count, frame_ny, frame_nx = (int(value) for value in dataset.shape)
        if not (0 <= x0 < x1 <= frame_nx and 0 <= y0 < y1 <= frame_ny):
            raise ValueError(
                f"crop bounds {(x0, y0, x1, y1)} lie outside frame shape {(frame_ny, frame_nx)}"
            )
        if np.any(indices >= frame_count):
            raise IndexError(
                "requested HDF5 background frames exceed the available time axis; "
                f"largest requested={int(np.max(indices))}, available={frame_count}"
            )

        if len(indices) == 1:
            index = int(indices[0])
            frames = np.asarray(dataset[index : index + 1, y0:y1, x0:x1], dtype=np.float32)
        else:
            differences = np.diff(indices)
            if np.all(differences == differences[0]) and differences[0] > 0:
                step = int(differences[0])
                stop = int(indices[-1]) + step
                frames = np.asarray(
                    dataset[int(indices[0]) : stop : step, y0:y1, x0:x1],
                    dtype=np.float32,
                )
            else:
                frames = np.stack(
                    [
                        np.asarray(dataset[int(index), y0:y1, x0:x1], dtype=np.float32)
                        for index in indices
                    ]
                )
        dataset_shape = tuple(int(value) for value in dataset.shape)
        dataset_dtype = str(dataset.dtype)

    crop_ny, crop_nx = y1 - y0, x1 - x0
    expected_shape = (len(indices), crop_ny, crop_nx)
    if frames.shape != expected_shape:
        raise ValueError(
            "loaded HDF5 crop sequence has an unexpected shape; "
            f"received {frames.shape}, expected {expected_shape}"
        )
    if not np.isfinite(frames).all() or np.any(frames <= 0.0):
        raise ValueError("selected HDF5 crops must contain finite strictly positive values")

    crop_center = (0.5 * (x0 + x1 - 1), 0.5 * (y0 + y1 - 1))
    metadata: dict[str, object] = {
        "background_source": "aligned_h5_time_series_fixed_crop",
        "source_path": str(source_path),
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "dataset_key": H5_TIME_SERIES_KEY,
        "dataset_shape": list(dataset_shape),
        "dataset_dtype": dataset_dtype,
        "frame_indices": indices.tolist(),
        "crop_xyxy_px": [x0, y0, x1, y1],
        "crop_shape": [crop_ny, crop_nx],
        "crop_center_px": list(crop_center),
        "crop_selection": "fixed_validated_bounds",
        "processing_applied": "none",
        "frames_sha256": _array_sha256(frames),
    }
    return {
        "source_path": source_path,
        "frames": frames,
        "frame_indices": indices,
        "crop_xyxy_px": (x0, y0, x1, y1),
        "crop_center_px": crop_center,
        "native_shape": (crop_ny, crop_nx),
        "metadata": metadata,
    }


def load_h5_background_sequence(
    path: str | Path = DEFAULT_H5_DYNAMICS_PATH,
    *,
    crop_shape: tuple[int, int],
    n_frames: int,
    seed: int,
    start_index: int = 0,
    frame_step: int = 1,
    random_attempts: int = 512,
    use_detector: bool = True,
    detection_threshold: float = 0.5,
    box_expand_fraction: float = 0.25,
    box_expand_px: float = 12.0,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cached_stages: dict | None = None,
) -> dict[str, object]:
    """Load one unchanged quiet crop from consecutive ``time_series`` frames.

    Crop selection uses only the first requested frame. The same integer bounds
    are read from all later frames without normalization, interpolation,
    filtering, registration, or added noise.
    """
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for name, value in (
        ("n_frames", n_frames),
        ("seed", seed),
        ("start_index", start_index),
        ("frame_step", frame_step),
        ("random_attempts", random_attempts),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer; received {value!r}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; received {n_frames!r}")
    if seed < 0 or start_index < 0 or frame_step < 1 or random_attempts < 1:
        raise ValueError("seed/start_index must be non-negative and step/attempts positive")
    if (
        len(crop_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in crop_shape)
        or any(value < 8 for value in crop_shape)
    ):
        raise ValueError(f"crop_shape must contain two integers >= 8; received {crop_shape!r}")
    if not 0.0 < detection_threshold < 1.0:
        raise ValueError("detection_threshold must lie in (0, 1)")
    if box_expand_fraction < 0.0 or box_expand_px < 0.0:
        raise ValueError("box expansion values must be non-negative")

    background_started = monotonic()
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "background",
                "completed": 0,
                "total": 1,
                "elapsed_seconds": 0.0,
                "description": "Reading the first source frame and inferring disk geometry.",
            }
        )
    frame_indices = start_index + frame_step * np.arange(n_frames, dtype=np.int64)
    source_stat = source_path.stat()
    read_options = _sequence_read_options() if n_frames > 1 else {}
    cache = None if cached_stages is None else stage_cache(cached_stages)
    with h5py.File(source_path, "r", **read_options) as handle:
        if H5_TIME_SERIES_KEY not in handle:
            raise KeyError(f"{source_path} is missing dataset {H5_TIME_SERIES_KEY!r}")
        dataset = handle[H5_TIME_SERIES_KEY]
        if dataset.ndim != 3:
            raise ValueError(
                f"{H5_TIME_SERIES_KEY!r} must have shape (time, y, x); received {dataset.shape}"
            )
        if frame_indices[-1] >= dataset.shape[0]:
            raise IndexError(
                "requested HDF5 background frames exceed the available time axis; "
                f"last requested={int(frame_indices[-1])}, available={dataset.shape[0]}"
            )

        frame_ny, frame_nx = int(dataset.shape[1]), int(dataset.shape[2])
        crop_ny, crop_nx = (int(value) for value in crop_shape)
        if crop_ny > frame_ny or crop_nx > frame_nx:
            raise ValueError(
                f"crop_shape {crop_shape} exceeds HDF5 frame shape {(frame_ny, frame_nx)}"
            )

        raw_frame, source_key = _read_source_frame(dataset, source_path, start_index, cached_stages)
        first_frame = np.asarray(raw_frame, dtype=np.float32)
        disk = None if cache is None else cache.get("source_disk", source_key)
        if disk is None:
            disk = _infer_disk_geometry(first_frame)
            if cache is not None:
                disk[2].flags.writeable = False
                cache.put("source_disk", source_key, disk)
        disk_center, disk_radius, support = disk
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "background",
                    "completed": 1,
                    "total": 1,
                    "elapsed_seconds": monotonic() - background_started,
                    "description": "First background frame and disk geometry are ready.",
                }
            )
        detector_started = monotonic()
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "detector",
                    "completed": 0,
                    "total": 1,
                    "elapsed_seconds": 0.0,
                    "description": (
                        "Loading the filament detector and selecting a quiet crop."
                        if use_detector
                        else "Selecting a quiet crop without detector screening."
                    ),
                }
            )
        detector_off_disk_mask = _off_disk_mask(
            first_frame.shape,
            disk_center=disk_center,
            disk_radius=disk_radius,
        )
        detector_key = json.dumps((source_key, bool(use_detector), float(detection_threshold),
                                   detector_model_identity() if use_detector else None))
        detected = None if cache is None else cache.get("source_detector", detector_key)
        if detected is None:
            detected = _filament_boxes(
                first_frame,
                off_disk_mask=detector_off_disk_mask,
                use_detector=use_detector,
                detection_threshold=detection_threshold,
            )
            if cache is not None and (detected[0] is not None or not use_detector):
                cache.put("source_detector", detector_key, detected)
        boxes, detector_error, detector_preprocessing = detected
        rng = np.random.default_rng(seed)
        crop_xyxy, eligible_candidates_tested = _select_quiet_crop(
            first_frame,
            crop_shape=(crop_ny, crop_nx),
            support=support,
            boxes=boxes,
            rng=rng,
            random_attempts=random_attempts,
            box_expand_fraction=box_expand_fraction,
            box_expand_px=box_expand_px,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "detector",
                    "completed": 1,
                    "total": 1,
                    "elapsed_seconds": monotonic() - detector_started,
                    "description": (
                        "Detector and quiet-crop selection complete."
                        if boxes is not None
                        else "Quiet-crop selection complete; detector was disabled or unavailable."
                    ),
                    "detector_used": boxes is not None,
                }
            )
        x0, y0, x1, y1 = crop_xyxy
        if n_frames == 1:
            frames = first_frame[None, y0:y1, x0:x1].copy()
        elif frame_step == 1:
            frames = np.asarray(
                dataset[start_index : start_index + n_frames, y0:y1, x0:x1],
                dtype=np.float32,
            )
        else:
            frames = np.asarray(dataset[frame_indices, y0:y1, x0:x1], dtype=np.float32)
        dataset_shape = tuple(int(value) for value in dataset.shape)
        dataset_dtype = str(dataset.dtype)

    if frames.shape != (n_frames, crop_ny, crop_nx):
        raise ValueError(
            f"loaded HDF5 crop sequence has an unexpected shape; received {frames.shape}"
        )
    if not np.isfinite(frames).all() or np.any(frames <= 0.0):
        raise ValueError("selected HDF5 crops must contain finite strictly positive values")

    crop_center_x = 0.5 * (crop_xyxy[0] + crop_xyxy[2] - 1)
    crop_center_y = 0.5 * (crop_xyxy[1] + crop_xyxy[3] - 1)
    offset_x = crop_center_x - disk_center[0]
    offset_y = crop_center_y - disk_center[1]
    radial_fraction = min(float(np.hypot(offset_x, offset_y) / disk_radius), 1.0)
    disk_mu = float(np.sqrt(max(1.0 - radial_fraction**2, 0.0)))
    limb_direction_deg = (
        float(np.rad2deg(np.arctan2(offset_y, offset_x)) % 360.0) if radial_fraction > 0.0 else 0.0
    )
    native_pixel_km = float(config_module.SOLAR_RADIUS_KM / disk_radius)
    detector_used = boxes is not None
    metadata: dict[str, object] = {
        "background_source": "aligned_h5_time_series",
        "source_path": str(source_path),
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "dataset_key": H5_TIME_SERIES_KEY,
        "dataset_shape": list(dataset_shape),
        "dataset_dtype": dataset_dtype,
        "frame_indices": frame_indices.tolist(),
        "crop_xyxy_px": list(crop_xyxy),
        "crop_shape": [crop_ny, crop_nx],
        "crop_center_px": [crop_center_x, crop_center_y],
        "disk_center_px": list(disk_center),
        "disk_radius_px": float(disk_radius),
        "native_pixel_km": native_pixel_km,
        "background_disk_mu": disk_mu,
        "background_limb_direction_deg": limb_direction_deg,
        "detector_requested": bool(use_detector),
        "detector_used": bool(detector_used),
        "detector_error": detector_error,
        "detector_preprocessing": detector_preprocessing,
        "n_exclusion_boxes": None if boxes is None else int(len(boxes)),
        "eligible_candidates_tested": int(eligible_candidates_tested),
        "acceptance": "quiet+boxfree" if detector_used else "quiet",
        "processing_applied": "none",
        "frames_sha256": _array_sha256(frames),
    }
    return {
        "source_path": source_path,
        "frames": frames,
        "frame_indices": frame_indices,
        "crop_xyxy_px": tuple(int(value) for value in crop_xyxy),
        "crop_center_px": (crop_center_x, crop_center_y),
        "disk_center_px": tuple(float(value) for value in disk_center),
        "disk_radius_px": float(disk_radius),
        "native_pixel_km": native_pixel_km,
        "disk_mu": disk_mu,
        "limb_direction_deg": limb_direction_deg,
        "native_shape": (crop_ny, crop_nx),
        "metadata": metadata,
    }
