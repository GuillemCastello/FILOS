"""One-time preparation of portable quiet-Sun background sequences."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_closing, find_objects, label

from .config import SOLAR_RADIUS_KM
from .dynamic_background import BACKGROUND_VERSION, background_info


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


def regular_windows(times: np.ndarray, n_frames: int, cadence_s: float) -> list[int]:
    """Find non-overlapping windows with no missing or irregular time steps."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    if n_frames < 2 or not np.isfinite(cadence_s) or cadence_s <= 0:
        raise ValueError("preparation needs at least two frames and a positive cadence")
    breaks = np.r_[0, np.flatnonzero(~np.isclose(np.diff(times), cadence_s, rtol=0, atol=1e-6)) + 1,
                   len(times)]
    return [start for lo, hi in zip(breaks[:-1], breaks[1:], strict=True)
            for start in range(int(lo), int(hi) - n_frames + 1, n_frames)]


def write_background(
    path: str | Path,
    frames: np.ndarray,
    *,
    cadence_s: float,
    native_pixel_km: float,
    disk_mu: float,
    limb_direction_deg: float,
    provenance: dict | None = None,
) -> Path:
    """Write a portable sequence without changing its pixel values; never overwrite."""
    path = Path(path)
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.dtype.kind != "f":
        raise ValueError("frames must be a floating-point array shaped (time, height, width)")
    if not np.isfinite(frames).all() or np.any(frames <= 0):
        raise ValueError("background frames must contain finite positive intensities")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A partial file must never appear in the library's *.h5 selection.
    temporary = path.with_suffix(".h5.partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    try:
        with h5py.File(temporary, "x") as handle:
            handle.create_dataset("time_series", data=frames, compression="lzf", chunks=(1, *frames.shape[1:]))
            handle.create_dataset("time_s", data=np.arange(len(frames)) * cadence_s)
            handle.attrs.update({
                "filos_background_version": BACKGROUND_VERSION,
                "cadence_s": cadence_s,
                "native_pixel_km": native_pixel_km,
                "disk_mu": disk_mu,
                "limb_direction_deg": limb_direction_deg,
                "provenance": json.dumps(provenance or {}, sort_keys=True),
            })
        background_info(temporary)
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_backgrounds(
    source: str | Path,
    output: str | Path,
    *,
    n_frames: int = 400,
    crop_shape: tuple[int, int] = (448, 448),
    count: int = 2,
    seed: int = 0,
    cadence_s: float = 60.0,
    use_detector: bool = False,
    detector_stride: int = 30,
) -> list[Path]:
    """Screen every crop frame and save up to count quiet sequences from one source."""
    source, output = Path(source), Path(output)
    if count < 1 or detector_stride < 1 or min(crop_shape) < 8:
        raise ValueError("count/stride must be positive and crop dimensions at least 8")
    rng = np.random.default_rng(seed)
    saved = []
    with h5py.File(source, "r", rdcc_nbytes=512 * 2**20, rdcc_nslots=10007) as handle:
        dataset = handle["time_series"]
        if dataset.ndim != 3 or any(a > b for a, b in zip(crop_shape, dataset.shape[1:], strict=True)):
            raise ValueError("source must be a full-disk time series larger than the requested crop")
        times = np.asarray(handle["tdeltas"], dtype=float)
        if times.shape != (len(dataset),):
            raise ValueError("tdeltas must contain one timestamp per source frame")
        windows = regular_windows(times, n_frames, cadence_s)
        if not windows:
            raise ValueError(f"no uninterrupted {n_frames}-frame intervals at {cadence_s:g} s cadence")
        for start in windows:
            first = np.asarray(dataset[start])
            center, radius, support = _infer_disk_geometry(first)
            candidates = _candidate_bounds(frame_shape=first.shape, crop_shape=crop_shape,
                                           rng=rng, random_attempts=512)
            candidates = list(dict.fromkeys(candidates))
            # Keep a small pool; checking many nearly identical crops adds little variety.
            valid = []
            for x0, y0, x1, y1 in candidates:
                crop = first[y0:y1, x0:x1]
                if support[y0:y1, x0:x1].all() and _quiet_crop_ok(crop):
                    valid.append((x0, y0, x1, y1))
                if len(valid) == 128:
                    break
            checked_detector_frames = []
            # Reject evolving regions early, then check every remaining frame.
            check_order = dict.fromkeys(
                [0, n_frames - 1, *range(0, n_frames, detector_stride), *range(n_frames)]
            )
            for checked, offset in enumerate(check_order, 1):
                if not valid:
                    break
                frame = first if offset == 0 else np.asarray(dataset[start + offset])
                boxes = None
                if use_detector and (offset % detector_stride == 0 or offset == n_frames - 1):
                    from .detector import detect_filament_boxes

                    boxes = detect_filament_boxes(frame, off_disk_mask=_off_disk_mask(
                        frame.shape, disk_center=center, disk_radius=radius), threshold=0.5)
                    checked_detector_frames.append(start + offset)
                accepted = []
                for bounds in valid:
                    x0, y0, x1, y1 = bounds
                    crop = frame[y0:y1, x0:x1]
                    if not np.isfinite(crop).all() or np.any(crop <= 0) or not _quiet_crop_ok(crop):
                        continue
                    if boxes is not None and _crop_overlaps_boxes(
                        boxes, bounds, expand_fraction=0.25, expand_px=12.0,
                    ):
                        continue
                    accepted.append(bounds)
                valid = accepted
                if checked % 100 == 0:
                    print(f"  Checked {checked}/{n_frames} frames; {len(valid)} crops remain", flush=True)
            print(f"  {source.name} frames {start}:{start + n_frames}: {len(valid)} passing crops", flush=True)
            chosen = []
            for bounds in valid:
                x0, y0, x1, y1 = bounds
                # Avoid distributing almost identical overlapping patches from the same interval.
                if any(_crop_overlaps_boxes(np.asarray([previous]), bounds,
                                            expand_fraction=0, expand_px=0) for previous in chosen):
                    continue
                crop_center = np.array([(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2])
                offset = crop_center - np.asarray(center)
                mu = float(np.sqrt(max(1 - np.sum(offset**2) / radius**2, 0)))
                path = output / f"{source.stem}-{start:04d}-{x0}-{y0}.h5"
                frames = np.asarray(dataset[start:start + n_frames, y0:y1, x0:x1])
                write_background(path, frames, cadence_s=cadence_s,
                                 native_pixel_km=SOLAR_RADIUS_KM / radius, disk_mu=mu,
                                 limb_direction_deg=float(np.degrees(np.arctan2(offset[1], offset[0])) % 360),
                                 provenance={
                                     "source_file": source.name,
                                     "source_crop_xyxy_px": list(bounds),
                                     "source_start_index": start,
                                     "source_times_s": times[start:start + n_frames].tolist(),
                                     "source_disk_center_px": list(center),
                                     "source_disk_radius_px": radius,
                                     "quiet_screened_frames": n_frames,
                                     "detector_frame_indices": sorted(checked_detector_frames),
                                     "screening": "quiet-structure heuristics on every frame",
                                 })
                print(f"  Saved {path}", flush=True)
                chosen.append(bounds)
                saved.append(path)
                if len(saved) == count:
                    return saved
    return saved
