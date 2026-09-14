"""Functional HDF5 persistence for filament-dynamics simulations.

The public API in this module deliberately consists only of functions and
plain dictionaries.  A recorder dictionary owns an unpublished HDF5 staging
file until finalize_simulation publishes the complete directory atomically.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import closing
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import line

from .degradation import block_average
from .geometry import effective_thread_radius
from .render import _project_points, _project_vectors, thread_tau_pixel_contributions

SIMULATION_DATASET_SCHEMA_VERSION = 6
DEFAULT_SIMULATIONS_ROOT = Path(__file__).resolve().parents[1] / "simulations"

_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_EXPORT_DEFAULTS: dict[str, object] = {
    "thread_mask_dilation_px": 5,
    "video_lower_percentile": 0.0,
    "video_upper_percentile": 100.0,
    "velocity_opacity_floor": 1.0e-6,
    "compression": "lzf",
    "gzip_level": 4,
}


def make_export_config(**overrides: object) -> dict[str, object]:
    """Return a validated configuration for derived HDF5 products.

    Parameters are intentionally limited to products computed by this module:
    geometry-mask dilation, fixed video contrast, velocity-label opacity
    threshold, and HDF5 compression.
    """
    unknown = sorted(set(overrides) - set(_EXPORT_DEFAULTS))
    if unknown:
        raise TypeError(f"unknown export configuration fields: {', '.join(unknown)}")

    config = dict(_EXPORT_DEFAULTS)
    config.update(overrides)
    dilation = config["thread_mask_dilation_px"]
    if isinstance(dilation, bool) or not isinstance(dilation, Integral) or dilation < 0:
        raise ValueError("thread_mask_dilation_px must be an integer greater than or equal to 0")

    lower = config["video_lower_percentile"]
    upper = config["video_upper_percentile"]
    for name, value in (
        ("video_lower_percentile", lower),
        ("video_upper_percentile", upper),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a real number")
        if not np.isfinite(value) or not 0.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be finite and lie in [0, 100]")
    if float(upper) <= float(lower):
        raise ValueError("video_upper_percentile must exceed video_lower_percentile")

    opacity_floor = config["velocity_opacity_floor"]
    if (
        isinstance(opacity_floor, bool)
        or not isinstance(opacity_floor, (int, float, np.number))
        or not np.isfinite(opacity_floor)
        or float(opacity_floor) < 0.0
    ):
        raise ValueError("velocity_opacity_floor must be finite and non-negative")

    compression = config["compression"]
    if compression not in {None, "gzip", "lzf"}:
        raise ValueError("compression must be None, 'gzip', or 'lzf'")
    gzip_level = config["gzip_level"]
    if (
        isinstance(gzip_level, bool)
        or not isinstance(gzip_level, Integral)
        or not 0 <= int(gzip_level) <= 9
    ):
        raise ValueError("gzip_level must be an integer in [0, 9]")

    config["thread_mask_dilation_px"] = int(dilation)
    config["video_lower_percentile"] = float(lower)
    config["video_upper_percentile"] = float(upper)
    config["velocity_opacity_floor"] = float(opacity_floor)
    config["gzip_level"] = int(gzip_level)
    return config


def _json_ready(value: Any) -> Any:
    """Convert nested scientific values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: object) -> str:
    """Serialize a value deterministically for metadata and identifiers."""
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _generated_utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generator_git_commit() -> str:
    """Return the repository commit associated with generated output."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _file_sha256(path: Path) -> str:
    """Return the uppercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _require_dict(value: object, name: str) -> dict[str, Any]:
    """Return a required plain dictionary or raise a focused error."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    return value


def _required(mapping: dict[str, Any], key: str, owner: str) -> Any:
    """Read one required dictionary field with a useful error message."""
    try:
        return mapping[key]
    except KeyError as error:
        raise KeyError(f"{owner} is missing required field {key!r}") from error


def _simulation_id(
    dynamics_config: dict[str, Any],
    static_config: dict[str, Any],
    export_config: dict[str, object],
    label: str | None,
) -> str:
    """Build a readable unique identifier without advancing random state."""
    if label is not None and not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "label must start with an alphanumeric character and contain at most "
            "40 alphanumeric, underscore, or hyphen characters"
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dynamics": dynamics_config,
                "static": static_config,
                "export": export_config,
            }
        ).encode("utf-8")
    ).hexdigest()[:10]
    seed = dynamics_config.get("seed", "na")
    label_suffix = "" if label is None else f"-{label}"
    return f"sim-{timestamp}-s{seed}-{digest}{label_suffix}"


def _compression_kwargs(export_config: dict[str, object]) -> dict[str, object]:
    """Return valid compression arguments for h5py."""
    compression = export_config["compression"]
    if compression is None:
        return {}
    arguments: dict[str, object] = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        arguments["compression_opts"] = int(export_config["gzip_level"])
    return arguments


def _frame_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose chunks optimized for reading and writing one frame."""
    if len(shape) == 4:
        return 1, shape[1], min(shape[2], 256), min(shape[3], 256)
    if len(shape) == 3:
        return 1, min(shape[1], 256), min(shape[2], 256)
    if len(shape) == 2:
        return 1, min(shape[1], 512)
    return (min(shape[0], 1024),)


def _set_dataset_metadata(
    dataset: h5py.Dataset,
    units: str,
    description: str,
    dimensions: str,
) -> h5py.Dataset:
    """Attach common scientific metadata to one HDF5 dataset."""
    dataset.attrs["units"] = units
    dataset.attrs["description"] = description
    dataset.attrs["dimensions"] = dimensions
    return dataset


def _create_frame_dataset(
    recorder: dict[str, Any],
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type,
    units: str,
    description: str,
    dimensions: str,
) -> h5py.Dataset:
    """Create and register one fixed-size frame dataset."""
    dataset = recorder["handle"].create_dataset(
        path,
        shape=shape,
        dtype=dtype,
        chunks=_frame_chunks(shape),
        **_compression_kwargs(recorder["export_config"]),
    )
    recorder["datasets"][path] = _set_dataset_metadata(
        dataset,
        units,
        description,
        dimensions,
    )
    return dataset


def _create_static_dataset(
    recorder: dict[str, Any],
    path: str,
    values: object,
    units: str,
    description: str,
    dimensions: str,
    *,
    compress: bool = True,
) -> h5py.Dataset:
    """Create one fixed-state dataset."""
    array = np.asarray(values)
    keywords: dict[str, object] = {}
    if compress and array.ndim > 0 and array.size > 1:
        keywords.update(_compression_kwargs(recorder["export_config"]))
    dataset = recorder["handle"].create_dataset(path, data=array, **keywords)
    return _set_dataset_metadata(dataset, units, description, dimensions)


def _create_json_dataset(
    recorder: dict[str, Any],
    path: str,
    value: object,
    description: str,
) -> None:
    """Store one JSON value as a UTF-8 HDF5 scalar."""
    dataset = recorder["handle"].create_dataset(
        path,
        data=_canonical_json(value),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    _set_dataset_metadata(dataset, "JSON", description, "scalar")


def _translated_threads(
    initial_threads: list[dict[str, Any]],
    displacements_km: np.ndarray,
) -> list[dict[str, Any]]:
    """Return thread dictionaries translated by per-thread xyz offsets."""
    if displacements_km.shape != (len(initial_threads), 3):
        raise ValueError(
            "thread_displacements_km must have shape "
            f"({len(initial_threads)}, 3); received {displacements_km.shape}"
        )
    translated: list[dict[str, Any]] = []
    for thread, displacement in zip(initial_threads, displacements_km, strict=True):
        translated_thread = dict(thread)
        for component, axis in enumerate(("x", "y", "z")):
            translated_thread[axis] = (
                np.asarray(_required(thread, axis, "thread"), dtype=float) + displacement[component]
            )
        translated.append(translated_thread)
    return translated


def _disk_structure(radius_px: int) -> np.ndarray:
    """Return a Euclidean disk structuring element."""
    if radius_px <= 0:
        return np.ones((1, 1), dtype=bool)
    coordinates = np.arange(-radius_px, radius_px + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return xx**2 + yy**2 <= radius_px**2


def _prepare_dynamic_geometry(
    state: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Pack invariant centerline coordinates for repeated geometry masks."""
    threads = state["initial_threads"]
    point_counts = np.asarray(
        [np.asarray(_required(thread, "x", "thread")).size for thread in threads],
        dtype=np.int64,
    )
    if np.any(point_counts < 2):
        raise ValueError("each initial thread must contain at least two centerline points")
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(point_counts, dtype=np.int64)]
    )
    point_thread_indices = np.repeat(np.arange(len(threads), dtype=np.int64), point_counts)
    segment_starts = np.concatenate(
        [
            np.arange(offsets[index], offsets[index + 1] - 1, dtype=np.int64)
            for index in range(len(threads))
            if point_counts[index] > 1
        ]
    )
    native_pixel_km = (
        float(_required(state["static_config"], "pixel_size_km", "static_config"))
        * int(_required(state["static_config"], "downsample_factor", "static_config"))
    )
    return {
        "x_km": np.concatenate(
            [np.asarray(_required(thread, "x", "thread"), dtype=float) for thread in threads]
        ),
        "y_km": np.concatenate(
            [np.asarray(_required(thread, "y", "thread"), dtype=float) for thread in threads]
        ),
        "z_km": np.concatenate(
            [np.asarray(_required(thread, "z", "thread"), dtype=float) for thread in threads]
        ),
        "radii_px": np.concatenate(
            [np.asarray(effective_thread_radius(thread), dtype=float) for thread in threads]
        )
        / native_pixel_km,
        "point_thread_indices": point_thread_indices,
        "segment_starts": segment_starts,
    }


def _thread_geometry_masks_native(
    state: dict[str, Any],
    prepared: dict[str, np.ndarray],
    displacements_km: np.ndarray,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize projected physical thread tubes on the native grid."""
    static_config = state["static_config"]
    factor = int(_required(static_config, "downsample_factor", "static_config"))
    native_shape = (
        int(_required(static_config, "ny", "static_config")) // factor,
        int(_required(static_config, "nx", "static_config")) // factor,
    )
    native_pixel_km = float(_required(static_config, "pixel_size_km", "static_config")) * factor
    geometry_mask = np.zeros(native_shape, dtype=bool)
    point_threads = prepared["point_thread_indices"]
    displacements = np.asarray(displacements_km, dtype=float)
    expected_shape = (len(state["initial_threads"]), 3)
    if displacements.shape != expected_shape:
        raise ValueError(
            f"displacements_km must have shape {expected_shape}; received {displacements.shape}"
        )
    projected_x, projected_y = _project_points(
        static_config,
        prepared["x_km"] + displacements[point_threads, 0],
        prepared["y_km"] + displacements[point_threads, 1],
        prepared["z_km"] + displacements[point_threads, 2],
    )
    # A native pixel is centered on the mean of its high-resolution block.
    block_center_offset = (factor - 1) / (2.0 * factor)
    columns = np.rint(projected_x / native_pixel_km - block_center_offset).astype(np.int64)
    rows = np.rint(projected_y / native_pixel_km - block_center_offset).astype(np.int64)
    radii_px = prepared["radii_px"]
    starts = prepared["segment_starts"]
    ends = starts + 1
    row_steps = np.abs(rows[ends] - rows[starts])
    column_steps = np.abs(columns[ends] - columns[starts])
    changed = (row_steps != 0) | (column_steps != 0)
    short_steps = changed & (row_steps <= 1) & (column_steps <= 1)
    line_rows_parts = [
        rows[starts],
        rows[ends[short_steps]],
    ]
    line_columns_parts = [
        columns[starts],
        columns[ends[short_steps]],
    ]
    line_radii_parts = [
        radii_px[starts],
        radii_px[ends[short_steps]],
    ]
    for segment_index in np.flatnonzero(changed & ((row_steps > 1) | (column_steps > 1))):
        start = starts[segment_index]
        end = ends[segment_index]
        segment_rows, segment_columns = line(
            int(rows[start]),
            int(columns[start]),
            int(rows[end]),
            int(columns[end]),
        )
        segment_radii = np.linspace(
            radii_px[start],
            radii_px[end],
            segment_rows.size,
        )
        line_rows_parts.append(segment_rows[1:])
        line_columns_parts.append(segment_columns[1:])
        line_radii_parts.append(segment_radii[1:])

    line_rows = np.concatenate(line_rows_parts)
    line_columns = np.concatenate(line_columns_parts)
    if line_rows.size:
        conservative_radii = np.concatenate(line_radii_parts) + np.sqrt(2.0) / 2.0
        offset_limit = int(np.ceil(np.max(conservative_radii)))
        offsets = np.arange(-offset_limit, offset_limit + 1)
        offset_rows, offset_columns = np.meshgrid(offsets, offsets, indexing="ij")
        for row_offset, column_offset in zip(
            offset_rows.ravel(),
            offset_columns.ravel(),
            strict=True,
        ):
            covered = row_offset**2 + column_offset**2 <= conservative_radii**2
            expanded_rows = line_rows + row_offset
            expanded_columns = line_columns + column_offset
            valid = (
                covered
                & (expanded_rows >= 0)
                & (expanded_rows < native_shape[0])
                & (expanded_columns >= 0)
                & (expanded_columns < native_shape[1])
            )
            geometry_mask[expanded_rows[valid], expanded_columns[valid]] = True

    if dilation_px == 0:
        return geometry_mask, geometry_mask.copy()
    dilated_mask = binary_dilation(
        geometry_mask,
        structure=_disk_structure(dilation_px),
    )
    return geometry_mask, dilated_mask


def _coherent_thread_velocities_xy(
    state: dict[str, Any],
    frame: dict[str, Any],
) -> np.ndarray:
    """Return coherent image-plane velocity per thread, excluding Brownian motion."""
    longitudinal = np.asarray(
        _required(
            frame,
            "unweighted_longitudinal_thread_velocities_km_s",
            "frame",
        ),
        dtype=float,
    )
    transverse = np.asarray(
        _required(
            frame,
            "unweighted_transverse_thread_velocities_km_s",
            "frame",
        ),
        dtype=float,
    )
    longitudinal_directions = np.asarray(
        _required(
            state,
            "thread_longitudinal_directions",
            "dynamics_state",
        ),
        dtype=float,
    )
    transverse_directions = np.asarray(
        _required(
            state,
            "thread_transverse_directions",
            "dynamics_state",
        ),
        dtype=float,
    )
    weights = np.asarray(_required(frame, "thread_weights", "frame"), dtype=float)
    unweighted = (
        longitudinal[:, None] * longitudinal_directions
        + transverse[:, None] * transverse_directions
    )
    return _project_vectors(state["static_config"], weights[:, None] * unweighted)


def _opacity_weighted_velocity_native(
    state: dict[str, Any],
    frame: dict[str, Any],
    threads: list[dict[str, Any]] | None,
    opacity_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute native-grid image-plane coherent velocity and optical-depth weight.

    For component c in {x, y}, the dense target is

    $$
    v_c(x,y,t) =
    \frac{\sum_j \tau_j(x,y,t) v_{j,c}(t)}
         {\sum_j \tau_j(x,y,t)}.
    $$

    Each thread velocity is projected from its full intrinsic XYZ components
    with the derivative of the position transform before opacity weighting.
    Brownian displacement affects the current opacity footprint but Brownian
    increments do not enter the coherent velocity target.
    """
    static_config = state["static_config"]
    ny = int(_required(static_config, "ny", "static_config"))
    nx = int(_required(static_config, "nx", "static_config"))
    prepared_numerator = frame.get("coherent_velocity_numerator_highres")
    if prepared_numerator is None:
        if threads is None:
            raise ValueError("translated threads are required without a prepared velocity field")
        thread_velocities = _coherent_thread_velocities_xy(state, frame)
        numerator_highres = np.zeros((2, ny, nx), dtype=float)
        for thread, velocity_xy in zip(threads, thread_velocities, strict=True):
            rows, columns, tau_values = thread_tau_pixel_contributions(
                static_config,
                thread,
            )
            np.add.at(
                numerator_highres[0],
                (rows, columns),
                tau_values * velocity_xy[0],
            )
            np.add.at(
                numerator_highres[1],
                (rows, columns),
                tau_values * velocity_xy[1],
            )
    else:
        numerator_highres = np.asarray(prepared_numerator, dtype=float)
        if numerator_highres.shape != (2, ny, nx):
            raise ValueError(
                "coherent_velocity_numerator_highres must have shape "
                f"{(2, ny, nx)}; received {numerator_highres.shape}"
            )
        if not np.isfinite(numerator_highres).all():
            raise ValueError("coherent_velocity_numerator_highres must be finite")

    factor = int(_required(static_config, "downsample_factor", "static_config"))
    psf_sigma = float(_required(static_config, "psf_sigma_px", "static_config"))
    if np.any(numerator_highres):
        numerator_blurred = gaussian_filter(
            numerator_highres,
            sigma=(0.0, psf_sigma, psf_sigma),
        )
        numerator_native = block_average(numerator_blurred, factor)
    else:
        numerator_native = np.zeros((2, ny // factor, nx // factor), dtype=float)
    opacity_weight = block_average(
        gaussian_filter(
            np.asarray(_required(frame, "tau_map", "frame"), dtype=float),
            sigma=psf_sigma,
        ),
        factor,
    )
    velocity = np.zeros_like(numerator_native)
    valid = opacity_weight > opacity_floor
    np.divide(
        numerator_native,
        opacity_weight[None, :, :],
        out=velocity,
        where=valid[None, :, :],
    )
    velocity[:, ~valid] = 0.0
    return velocity, opacity_weight


def _write_initial_geometry(recorder: dict[str, Any]) -> None:
    """Persist ragged initial geometry and fixed coherent-motion bases."""
    state = recorder["dynamics_state"]
    threads = state["initial_threads"]
    if not threads:
        raise ValueError("dynamics_state initial_threads must not be empty")
    point_counts = np.asarray(
        [np.asarray(_required(thread, "s", "thread")).size for thread in threads],
        dtype=np.int64,
    )
    if np.any(point_counts < 2):
        raise ValueError("each initial thread must contain at least two points")
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(point_counts, dtype=np.int64)])

    def concatenate(field: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(_required(thread, field, "thread"), dtype=np.float64) for thread in threads]
        )

    positions = np.column_stack([concatenate("x"), concatenate("y"), concatenate("z")])
    _create_static_dataset(
        recorder,
        "geometry/initial/thread_offsets",
        offsets,
        "point index",
        "Ragged offsets; thread j occupies offsets[j]:offsets[j+1].",
        "thread_plus_one",
    )
    _create_static_dataset(
        recorder,
        "geometry/initial/point_positions_km",
        positions,
        "km",
        "Initial intrinsic (unprojected) thread-centerline coordinates in x, y, z order.",
        "point,xyz",
    )
    for dataset_name, field, units, description in (
        ("point_arclength_km", "s", "km", "Realized thread arclength."),
        (
            "point_spine_coordinate_km",
            "s_spine",
            "km",
            "Associated spine coordinate.",
        ),
        (
            "point_tau_weight",
            "tau_along",
            "dimensionless",
            "Along-thread opacity weight.",
        ),
        (
            "point_temperature_K",
            "temperature_K",
            "K",
            "Hydrostatic/PCTR temperature at each refined centerline point.",
        ),
        (
            "point_pressure_dyn_cm2",
            "pressure_dyn_cm2",
            "dyn cm^-2",
            "Hydrostatic gas pressure at each refined centerline point.",
        ),
        (
            "point_mean_molecular_mass",
            "mean_molecular_mass",
            "hydrogen-atom mass",
            "Dimensionless mean molecular mass at each refined centerline point.",
        ),
    ):
        _create_static_dataset(
            recorder,
            f"geometry/initial/{dataset_name}",
            concatenate(field),
            units,
            description,
            "point",
        )
    _create_static_dataset(
        recorder,
        "geometry/initial/point_loaded_mask",
        np.concatenate([np.asarray(thread["loaded_mask"], dtype=bool) for thread in threads]),
        "boolean",
        "True for refined centerline samples inside the loaded cool-plasma region.",
        "point",
    )
    _create_static_dataset(
        recorder,
        "geometry/initial/point_radius_km",
        np.concatenate([effective_thread_radius(thread) for thread in threads]),
        "km",
        "Physical cross-section radius at every centerline point.",
        "point",
    )

    scalar_fields = (
        ("radius_km", "radius_km", "km"),
        ("tau0", "tau0", "dimensionless"),
        ("height_km", "height_km", "km"),
        ("dip_depth_km", "dip_depth_km", "km"),
        ("dip_radius_km", "dip_radius_km", "km"),
        ("length_km", "length_km", "km"),
        ("spine_anchor_km", "spine_anchor_km", "km"),
        ("transverse_offset_km", "transverse_offset_km", "km"),
        ("pitch_deg", "pitch_deg", "degree"),
        ("requested_column_mass_g_cm2", "requested_column_mass_g_cm2", "g cm^-2"),
        ("realized_column_mass_g_cm2", "realized_column_mass_g_cm2", "g cm^-2"),
        ("column_mass_capacity_g_cm2", "column_mass_capacity_g_cm2", "g cm^-2"),
        ("loaded_length_fraction", "loaded_length_fraction", "dimensionless"),
        ("loaded_length_km", "loaded_length_km", "km"),
        ("column_mass_residual_g_cm2", "column_mass_residual_g_cm2", "g cm^-2"),
        ("column_mass_relative_residual", "column_mass_relative_residual", "dimensionless"),
        (
            "column_mass_target_tolerance_g_cm2",
            "column_mass_target_tolerance_g_cm2",
            "g cm^-2",
        ),
        (
            "column_mass_lower_endpoint_g_cm2",
            "column_mass_lower_endpoint_g_cm2",
            "g cm^-2",
        ),
        (
            "column_mass_upper_endpoint_g_cm2",
            "column_mass_upper_endpoint_g_cm2",
            "g cm^-2",
        ),
        (
            "column_mass_last_refinement_change_g_cm2",
            "column_mass_last_refinement_change_g_cm2",
            "g cm^-2",
        ),
    )
    for dataset_name, field, units in scalar_fields:
        _create_static_dataset(
            recorder,
            f"geometry/initial/thread_{dataset_name}",
            np.asarray(
                [_required(thread, field, "thread") for thread in threads],
                dtype=float,
            ),
            units,
            f"Per-thread {dataset_name.replace('_', ' ')}.",
            "thread",
        )

    _create_static_dataset(
        recorder,
        "geometry/initial/thread_column_mass_capacity_limited",
        np.asarray([thread["column_mass_capacity_limited"] for thread in threads], dtype=bool),
        "boolean",
        "True when the requested column mass exceeds the full hydrostatic dip capacity.",
        "thread",
    )
    for field in (
        "column_mass_converged",
        "column_mass_lower_resolution_limited",
        "column_mass_numerical_failure",
    ):
        _create_static_dataset(
            recorder,
            f"geometry/initial/thread_{field}",
            np.asarray([thread[field] for thread in threads], dtype=bool),
            "boolean",
            f"Per-thread {field.replace('_', ' ')}.",
            "thread",
        )
    for field in (
        "column_mass_bisection_iterations",
        "column_mass_integration_intervals",
        "column_mass_integration_points",
        "column_mass_refinement_levels",
    ):
        _create_static_dataset(
            recorder,
            f"geometry/initial/thread_{field}",
            np.asarray([thread[field] for thread in threads], dtype=np.int64),
            "count",
            f"Per-thread {field.replace('_', ' ')}.",
            "thread",
        )
    _create_json_dataset(
        recorder,
        "geometry/initial/thread_plasma_diagnostics_json",
        [
            {
                "column_mass_status": thread["column_mass_status"],
                "column_mass_integration_relative_tolerance": thread[
                    "column_mass_integration_relative_tolerance"
                ],
                "opacity_table_saturation": thread["opacity_table_saturation"],
                "opacity_diagnostics": thread["opacity_diagnostics"],
            }
            for thread in threads
        ],
        "Complete per-thread mass/opacity diagnostics aligned with thread index.",
    )

    spine = _require_dict(_required(state, "spine", "dynamics_state"), "spine")
    for dataset_name, field, units in (
        ("arclength_km", "s", "km"),
        ("x_km", "x", "km"),
        ("y_km", "y", "km"),
        ("tangent_x", "tangent_x", "dimensionless"),
        ("tangent_y", "tangent_y", "dimensionless"),
        ("normal_x", "normal_x", "dimensionless"),
        ("normal_y", "normal_y", "dimensionless"),
    ):
        _create_static_dataset(
            recorder,
            f"geometry/spine/{dataset_name}",
            np.asarray(_required(spine, field, "spine")),
            units,
            f"Fixed spine {dataset_name.replace('_', ' ')}.",
            "spine_point",
        )
    provenance_fields = (
        "model",
        "source",
        "library_index",
        "source_true_length_km",
        "source_epoch",
    )
    _create_json_dataset(
        recorder,
        "geometry/spine/provenance_json",
        {field: spine.get(field) for field in provenance_fields if field in spine},
        "Spine model and measured-library provenance.",
    )

    for path, state_field, units, description, dimensions in (
        (
            "dynamics/thread_longitudinal_directions_xyz",
            "thread_longitudinal_directions",
            "dimensionless",
            "Fixed intrinsic (unprojected) longitudinal unit vectors in x, y, z order.",
            "thread,xyz",
        ),
        (
            "dynamics/thread_transverse_directions_xyz",
            "thread_transverse_directions",
            "dimensionless",
            "Fixed intrinsic (unprojected) transverse unit vectors in x, y, z order.",
            "thread,xyz",
        ),
        (
            "dynamics/thread_spatial_weights",
            "initial_thread_weights",
            "dimensionless",
            "Fixed localized coherent-motion weight for each thread.",
            "thread",
        ),
        (
            "dynamics/thread_longitudinal_periods_s",
            "thread_longitudinal_periods_s",
            "s",
            "Resolved longitudinal oscillation period for every thread.",
            "thread",
        ),
        (
            "dynamics/thread_transverse_periods_s",
            "thread_transverse_periods_s",
            "s",
            "Resolved transverse oscillation period for every thread.",
            "thread",
        ),
        (
            "dynamics/thread_prominence_heights_m",
            "thread_prominence_heights_m",
            "m",
            "Realized dip-bottom height used by the pendulum model for every thread.",
            "thread",
        ),
        (
            "dynamics/thread_gravity_m_s2",
            "thread_gravity_m_s2",
            "m s^-2",
            "Height-dependent solar gravity used by the pendulum model for every thread.",
            "thread",
        ),
        (
            "dynamics/thread_gravity_cutoff_periods_s",
            "thread_gravity_cutoff_periods_s",
            "s",
            "Height-dependent gravity-only cut-off period for every thread.",
            "thread",
        ),
    ):
        _create_static_dataset(
            recorder,
            path,
            np.asarray(_required(state, state_field, "dynamics_state")),
            units,
            description,
            dimensions,
        )


def _write_forward_model_state(recorder: dict[str, Any]) -> None:
    """Persist backgrounds and fixed detector quantities for reconstruction."""
    state = recorder["dynamics_state"]
    config = state["config"]
    n_frames = int(_required(config, "n_frames", "dynamics_config"))
    native_shape = recorder["native_shape"]

    background_frames = state.get("background_frames")
    if background_frames is None:
        background = np.asarray(_required(state, "background", "dynamics_state"))
        if background.shape != native_shape:
            raise ValueError(
                f"background must have native shape {native_shape}; received {background.shape}"
            )
        backgrounds = np.broadcast_to(background, (n_frames, *native_shape))
    else:
        backgrounds = np.asarray(background_frames)
        if backgrounds.shape != (n_frames, *native_shape):
            raise ValueError(
                "background_frames must have shape "
                f"{(n_frames, *native_shape)}; received {backgrounds.shape}"
            )
    _create_static_dataset(
        recorder,
        "forward_model/background_frames",
        backgrounds,
        "normalized GONG intensity",
        "Native aligned background assigned to every saved frame.",
        "time,y,x",
    )

    support = state.get("support")
    if support is not None:
        support_array = np.asarray(support, dtype=np.uint8)
        if support_array.shape != native_shape:
            raise ValueError(
                f"support must have native shape {native_shape}; received {support_array.shape}"
            )
        _create_static_dataset(
            recorder,
            "forward_model/support_native",
            support_array,
            "binary",
            "Valid common-disk support on the native grid.",
            "y,x",
        )
    detector_residual = np.asarray(
        state.get("fixed_detector_residual", np.zeros(native_shape, dtype=float))
    )
    if detector_residual.shape != native_shape:
        raise ValueError(
            "fixed_detector_residual must have native shape "
            f"{native_shape}; received {detector_residual.shape}"
        )
    _create_static_dataset(
        recorder,
        "forward_model/fixed_detector_residual",
        detector_residual,
        "normalized GONG intensity",
        "Fixed detector residual retained for exact forward-model reconstruction.",
        "y,x",
    )

    forward_group = recorder["handle"]["forward_model"]
    for name in ("source_fraction", "source_level"):
        if name in state:
            forward_group.attrs[name] = float(state[name])
    static_config = state["static_config"]
    pixel_size = float(_required(static_config, "pixel_size_km", "static_config"))
    factor = int(_required(static_config, "downsample_factor", "static_config"))
    forward_group.attrs["native_pixel_km"] = pixel_size * factor
    forward_group.attrs["highres_pixel_km"] = pixel_size


def _allocate_frame_datasets(recorder: dict[str, Any]) -> None:
    """Allocate the canonical time-dependent simulation schema."""
    state = recorder["dynamics_state"]
    config = state["config"]
    n_frames = int(_required(config, "n_frames", "dynamics_config"))
    n_threads = len(state["initial_threads"])
    native_y, native_x = recorder["native_shape"]
    highres_y, highres_x = recorder["highres_shape"]
    raw_dtype = np.asarray(state["initial_gong"]).dtype
    tau_dtype = np.asarray(state["initial_tau"]).dtype

    definitions = (
        (
            "time/time_s",
            (n_frames,),
            np.float64,
            "s",
            "Absolute simulation time.",
            "time",
        ),
        (
            "video/raw_gong",
            (n_frames, native_y, native_x),
            raw_dtype,
            "normalized GONG intensity",
            "Raw GONG-like frame before display processing.",
            "time,y,x",
        ),
        (
            "video/processed_uint8",
            (n_frames, native_y, native_x),
            np.uint8,
            "8-bit grayscale",
            "Fixed-contrast frames ready for conventional video encoding.",
            "time,y,x",
        ),
        (
            "radiative/tau_highres",
            (n_frames, highres_y, highres_x),
            tau_dtype,
            "line-center optical depth",
            "High-resolution optical-depth map used by the forward model.",
            "time,y,x",
        ),
        (
            "radiative/tau_native",
            (n_frames, native_y, native_x),
            np.float32,
            "effective line-center optical depth",
            "Native effective optical depth recovered from integrated transmission.",
            "time,y,x",
        ),
        (
            "radiative/line_absorption_native",
            (n_frames, native_y, native_x),
            np.float32,
            "fraction",
            "Native physical line-center absorption.",
            "time,y,x",
        ),
        (
            "radiative/observable_absorption_native",
            (n_frames, native_y, native_x),
            np.float32,
            "fraction",
            "Native passband-diluted observable absorption.",
            "time,y,x",
        ),
        (
            "labels/thread_geometry_mask_native",
            (n_frames, native_y, native_x),
            np.uint8,
            "binary",
            "Projected thread tubes expanded by physical radius only.",
            "time,y,x",
        ),
        (
            "labels/thread_mask_native",
            (n_frames, native_y, native_x),
            np.uint8,
            "binary",
            "Geometry-derived thread mask with configured safety dilation.",
            "time,y,x",
        ),
        (
            "labels/coherent_velocity_xy_km_s",
            (n_frames, 2, native_y, native_x),
            np.float32,
            "km s^-1",
            "Opacity-weighted coherent image-plane velocity in x, y component order.",
            "time,component_xy,y,x",
        ),
        (
            "labels/velocity_opacity_weight_native",
            (n_frames, native_y, native_x),
            np.float32,
            "linear optical-depth weight",
            "Opacity denominator for velocity supervision and loss weighting.",
            "time,y,x",
        ),
        (
            "labels/opacity_change_highres",
            (n_frames, highres_y, highres_x),
            np.float32,
            "line-center optical-depth change",
            "High-resolution optical depth minus frame-0 optical depth.",
            "time,y,x",
        ),
        (
            "labels/opacity_change_native",
            (n_frames, native_y, native_x),
            np.float32,
            "effective line-center optical-depth change",
            "Native effective optical depth minus frame-0 native optical depth.",
            "time,y,x",
        ),
        (
            "state/thread_displacements_km",
            (n_frames, n_threads, 3),
            np.float64,
            "km",
            "Total rigid translation per thread in x, y, z order.",
            "time,thread,xyz",
        ),
        (
            "state/coherent_thread_displacements_km",
            (n_frames, n_threads, 3),
            np.float64,
            "km",
            "Coherent rigid translation per thread.",
            "time,thread,xyz",
        ),
        (
            "state/brownian_thread_displacements_km",
            (n_frames, n_threads, 3),
            np.float64,
            "km",
            "Cumulative Brownian rigid translation per thread.",
            "time,thread,xyz",
        ),
        (
            "state/brownian_step_displacements_km",
            (n_frames, n_threads, 3),
            np.float64,
            "km per cadence",
            "Brownian displacement increment for each frame transition.",
            "time,thread,xyz",
        ),
        (
            "dynamics/unweighted_longitudinal_thread_displacements_km",
            (n_frames, n_threads),
            np.float64,
            "km",
            "Unweighted longitudinal analytic displacement per thread.",
            "time,thread",
        ),
        (
            "dynamics/unweighted_transverse_thread_displacements_km",
            (n_frames, n_threads),
            np.float64,
            "km",
            "Unweighted transverse analytic displacement per thread.",
            "time,thread",
        ),
        (
            "dynamics/unweighted_longitudinal_thread_velocities_km_s",
            (n_frames, n_threads),
            np.float64,
            "km s^-1",
            "Unweighted longitudinal analytic velocity per thread.",
            "time,thread",
        ),
        (
            "dynamics/unweighted_transverse_thread_velocities_km_s",
            (n_frames, n_threads),
            np.float64,
            "km s^-1",
            "Unweighted transverse analytic velocity per thread.",
            "time,thread",
        ),
    )
    for definition in definitions:
        _create_frame_dataset(recorder, *definition)

    velocity_dataset = recorder["datasets"]["labels/coherent_velocity_xy_km_s"]
    velocity_dataset.attrs["component_names"] = "x,y"
    velocity_dataset.attrs["coordinate_frame"] = "projected_image_plane"
    velocity_dataset.attrs["projection"] = "derivative of render._project_points, including height"
    mask_dataset = recorder["datasets"]["labels/thread_mask_native"]
    mask_dataset.attrs["safety_dilation_px"] = int(
        recorder["export_config"]["thread_mask_dilation_px"]
    )


def _initialize_hdf5(recorder: dict[str, Any]) -> None:
    """Write static state and allocate all canonical frame datasets."""
    state = recorder["dynamics_state"]
    handle = recorder["handle"]
    handle.attrs["simulation_dataset_schema_version"] = SIMULATION_DATASET_SCHEMA_VERSION
    handle.attrs["dynamics_schema_version"] = int(recorder["dynamics_schema_version"])
    handle.attrs["simulation_id"] = recorder["simulation_id"]
    handle.attrs["generated_utc"] = recorder["generated_utc"]
    handle.attrs["generator_git_commit"] = recorder["generator_git_commit"]
    handle.attrs["complete"] = False
    handle.attrs["coordinate_order"] = "array rows are y; columns are x"
    handle.attrs["native_pixel_center_highres"] = "index * factor + (factor - 1) / 2"
    handle.attrs["thread_position_reconstruction"] = (
        "position(frame,thread,point)=initial_position(thread,point)"
        "+state/thread_displacements_km[frame,thread]"
    )

    _create_json_dataset(
        recorder,
        "metadata/dynamics_config_json",
        state["config"],
        "Exact coherent and Brownian dynamics configuration.",
    )
    _create_json_dataset(
        recorder,
        "metadata/static_config_json",
        state["static_config"],
        "Exact static forward-model configuration.",
    )
    _create_json_dataset(
        recorder,
        "metadata/export_config_json",
        recorder["export_config"],
        "Derived training-data export configuration.",
    )
    _create_json_dataset(
        recorder,
        "metadata/initial_metadata_json",
        state.get("initial_metadata", {}),
        "Realized metadata from the initial static filament.",
    )
    _create_json_dataset(
        recorder,
        "metadata/oscillation_site_json",
        state.get("site", {}),
        "Realized center and basis of the localized oscillation.",
    )

    _write_initial_geometry(recorder)
    _write_forward_model_state(recorder)
    _allocate_frame_datasets(recorder)


def open_simulation_recorder(
    dynamics_state: dict[str, Any],
    simulations_root: str | Path = DEFAULT_SIMULATIONS_ROOT,
    export_config: dict[str, object] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Open an unpublished functional recorder for one dynamics state.

    The returned dictionary must be passed to write_simulation_frame followed
    by finalize_simulation.  Call abort_simulation after an exception.
    """
    from .dynamics import DYNAMICS_SCHEMA_VERSION

    state = _require_dict(dynamics_state, "dynamics_state")
    config = _require_dict(
        _required(state, "config", "dynamics_state"),
        "dynamics_state['config']",
    )
    static_config = _require_dict(
        _required(state, "static_config", "dynamics_state"),
        "dynamics_state['static_config']",
    )
    threads = _required(state, "initial_threads", "dynamics_state")
    if not isinstance(threads, list) or not all(isinstance(thread, dict) for thread in threads):
        raise TypeError("dynamics_state initial_threads must be a list of dicts")
    n_frames = _required(config, "n_frames", "dynamics_config")
    if isinstance(n_frames, bool) or not isinstance(n_frames, Integral) or n_frames < 1:
        raise ValueError("dynamics_config n_frames must be a positive integer")

    if export_config is None:
        export = make_export_config()
    else:
        export_mapping = _require_dict(export_config, "export_config")
        export = make_export_config(**export_mapping)

    initial_gong = np.asarray(_required(state, "initial_gong", "dynamics_state"))
    initial_tau = np.asarray(_required(state, "initial_tau", "dynamics_state"))
    if initial_gong.ndim != 2 or initial_tau.ndim != 2:
        raise ValueError("initial_gong and initial_tau must be two-dimensional")
    factor = int(_required(static_config, "downsample_factor", "static_config"))
    configured_highres_shape = (
        int(_required(static_config, "ny", "static_config")),
        int(_required(static_config, "nx", "static_config")),
    )
    configured_native_shape = (
        configured_highres_shape[0] // factor,
        configured_highres_shape[1] // factor,
    )
    if initial_gong.shape != configured_native_shape:
        raise ValueError(
            "initial_gong shape does not match static configuration: "
            f"{initial_gong.shape} versus {configured_native_shape}"
        )
    if initial_tau.shape != configured_highres_shape:
        raise ValueError(
            "initial_tau shape does not match static configuration: "
            f"{initial_tau.shape} versus {configured_highres_shape}"
        )

    root = Path(simulations_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    simulation_id = _simulation_id(config, static_config, export, label)
    output_directory = root / simulation_id
    staging_directory = root / f".{simulation_id}.tmp-{os.getpid()}"
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if staging_directory.exists():
        raise FileExistsError(staging_directory)
    staging_directory.mkdir(parents=False, exist_ok=False)

    recorder: dict[str, Any] = {
        "dynamics_state": state,
        "export_config": export,
        "root": root,
        "simulation_id": simulation_id,
        "output_directory": output_directory,
        "staging_directory": staging_directory,
        "h5_path": staging_directory / "simulation.h5",
        "metadata_path": staging_directory / "simulation.json",
        "generated_utc": _generated_utc_now(),
        "generator_git_commit": _generator_git_commit(),
        "dynamics_schema_version": DYNAMICS_SCHEMA_VERSION,
        "native_shape": tuple(int(value) for value in initial_gong.shape),
        "highres_shape": tuple(int(value) for value in initial_tau.shape),
        "prepared_geometry": _prepare_dynamic_geometry(state),
        "datasets": {},
        "frame_count": 0,
        "first_tau_highres": None,
        "first_tau_native": None,
        "video_limits": None,
        "raw_min": np.inf,
        "raw_max": -np.inf,
        "maximum_velocity_km_s": 0.0,
        "mask_fraction_sum": 0.0,
        "maximum_absolute_opacity_change": 0.0,
        "published": False,
        "finalized": False,
        "saved": None,
    }
    try:
        recorder["handle"] = h5py.File(recorder["h5_path"], "w")
        _initialize_hdf5(recorder)
    except BaseException:
        handle = recorder.get("handle")
        if isinstance(handle, h5py.File) and handle.id.valid:
            handle.close()
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise
    state["prepare_export_fields"] = True
    return recorder


def _set_video_limits(recorder: dict[str, Any], frame: dict[str, Any]) -> None:
    """Set one fixed contrast mapping from valid pixels in frame zero."""
    raw = np.asarray(_required(frame, "gong_image", "frame"))
    valid = np.isfinite(raw)
    support = recorder["dynamics_state"].get("support")
    if support is not None:
        valid &= np.asarray(support, dtype=bool)
    values = raw[valid]
    if values.size == 0:
        raise ValueError("cannot process video because frame 0 has no valid pixels")
    lower, upper = np.percentile(
        values,
        [
            recorder["export_config"]["video_lower_percentile"],
            recorder["export_config"]["video_upper_percentile"],
        ],
    )
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("processed-video intensity limits must be finite")
    if upper <= lower:
        upper = lower + max(abs(lower) * 1.0e-6, 1.0e-6)
    recorder["video_limits"] = (lower, upper)

    dataset = recorder["datasets"]["video/processed_uint8"]
    dataset.attrs["source_dataset"] = "/video/raw_gong"
    dataset.attrs["contrast_scope"] = "fixed from valid frame-0 pixels"
    dataset.attrs["lower_percentile"] = recorder["export_config"]["video_lower_percentile"]
    dataset.attrs["upper_percentile"] = recorder["export_config"]["video_upper_percentile"]
    dataset.attrs["input_vmin"] = lower
    dataset.attrs["input_vmax"] = upper
    dataset.attrs["conversion"] = (
        "round(255*clip((raw_gong-input_vmin)/(input_vmax-input_vmin),0,1))"
    )


def _processed_frame(recorder: dict[str, Any], raw_frame: np.ndarray) -> np.ndarray:
    """Convert one scalar frame to fixed-contrast grayscale."""
    limits = recorder["video_limits"]
    if limits is None:
        raise RuntimeError("video limits have not been initialized")
    lower, upper = limits
    normalized = np.clip((raw_frame - lower) / (upper - lower), 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return np.rint(255.0 * normalized).astype(np.uint8)


def write_simulation_frame(
    recorder: dict[str, Any],
    frame: dict[str, Any],
) -> None:
    """Write one frame and its canonical derived supervision arrays."""
    record = _require_dict(recorder, "recorder")
    current_frame = _require_dict(frame, "frame")
    if record.get("finalized") or record.get("published"):
        raise RuntimeError("cannot write frames after finalization")
    handle = record.get("handle")
    if not isinstance(handle, h5py.File) or not handle.id.valid:
        raise RuntimeError("simulation recorder is closed")

    expected_index = int(record["frame_count"])
    frame_index = _required(current_frame, "index", "frame")
    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
        raise TypeError("frame index must be an integer")
    if int(frame_index) != expected_index:
        raise ValueError(
            f"frames must be written in order; expected {expected_index}, received {frame_index}"
        )
    n_frames = int(record["dynamics_state"]["config"]["n_frames"])
    if expected_index >= n_frames:
        raise ValueError("received more frames than configured")

    raw_frame = np.asarray(_required(current_frame, "gong_image", "frame"))
    tau_highres = np.asarray(_required(current_frame, "tau_map", "frame"))
    if raw_frame.shape != record["native_shape"]:
        raise ValueError(
            f"gong_image must have shape {record['native_shape']}; received {raw_frame.shape}"
        )
    if tau_highres.shape != record["highres_shape"]:
        raise ValueError(
            f"tau_map must have shape {record['highres_shape']}; received {tau_highres.shape}"
        )
    if record["video_limits"] is None:
        _set_video_limits(record, current_frame)

    displacements = np.asarray(
        _required(current_frame, "thread_displacements_km", "frame"),
        dtype=float,
    )
    geometry_mask, dilated_mask = _thread_geometry_masks_native(
        record["dynamics_state"],
        record["prepared_geometry"],
        displacements,
        int(record["export_config"]["thread_mask_dilation_px"]),
    )
    threads = None
    if "coherent_velocity_numerator_highres" not in current_frame:
        threads = _translated_threads(
            record["dynamics_state"]["initial_threads"],
            displacements,
        )
    velocity_xy, velocity_weight = _opacity_weighted_velocity_native(
        record["dynamics_state"],
        current_frame,
        threads,
        float(record["export_config"]["velocity_opacity_floor"]),
    )
    line_absorption = np.asarray(
        _required(current_frame, "soft_mask", "frame"),
        dtype=float,
    )
    observable_absorption = np.asarray(
        _required(current_frame, "observable_soft_mask", "frame"),
        dtype=float,
    )
    if (
        line_absorption.shape != record["native_shape"]
        or observable_absorption.shape != record["native_shape"]
    ):
        raise ValueError("frame absorption arrays must match the native image shape")
    tau_native = -np.log(np.maximum(1.0 - line_absorption, 1.0e-12))

    if record["first_tau_highres"] is None:
        record["first_tau_highres"] = np.array(tau_highres, dtype=float, copy=True)
        record["first_tau_native"] = np.array(tau_native, dtype=float, copy=True)
    first_tau_native = record["first_tau_native"]
    if first_tau_native is None:
        raise RuntimeError("frame-0 native opacity was not initialized")
    opacity_change_highres = tau_highres - record["first_tau_highres"]
    opacity_change_native = tau_native - first_tau_native

    values: dict[str, object] = {
        "time/time_s": _required(current_frame, "time_s", "frame"),
        "video/raw_gong": raw_frame,
        "video/processed_uint8": _processed_frame(record, raw_frame),
        "radiative/tau_highres": tau_highres,
        "radiative/tau_native": tau_native.astype(np.float32),
        "radiative/line_absorption_native": line_absorption.astype(np.float32),
        "radiative/observable_absorption_native": observable_absorption.astype(np.float32),
        "labels/thread_geometry_mask_native": geometry_mask.astype(np.uint8),
        "labels/thread_mask_native": dilated_mask.astype(np.uint8),
        "labels/coherent_velocity_xy_km_s": velocity_xy.astype(np.float32),
        "labels/velocity_opacity_weight_native": velocity_weight.astype(np.float32),
        "labels/opacity_change_highres": opacity_change_highres.astype(np.float32),
        "labels/opacity_change_native": opacity_change_native.astype(np.float32),
        "state/thread_displacements_km": displacements,
        "state/coherent_thread_displacements_km": _required(
            current_frame,
            "coherent_thread_displacements_km",
            "frame",
        ),
        "state/brownian_thread_displacements_km": _required(
            current_frame,
            "brownian_thread_displacements_km",
            "frame",
        ),
        "state/brownian_step_displacements_km": _required(
            current_frame,
            "brownian_step_displacements_km",
            "frame",
        ),
        "dynamics/unweighted_longitudinal_thread_displacements_km": _required(
            current_frame,
            "unweighted_longitudinal_thread_displacements_km",
            "frame",
        ),
        "dynamics/unweighted_transverse_thread_displacements_km": _required(
            current_frame,
            "unweighted_transverse_thread_displacements_km",
            "frame",
        ),
        "dynamics/unweighted_longitudinal_thread_velocities_km_s": _required(
            current_frame,
            "unweighted_longitudinal_thread_velocities_km_s",
            "frame",
        ),
        "dynamics/unweighted_transverse_thread_velocities_km_s": _required(
            current_frame,
            "unweighted_transverse_thread_velocities_km_s",
            "frame",
        ),
    }
    for path, value in values.items():
        record["datasets"][path][expected_index] = value

    finite_raw = raw_frame[np.isfinite(raw_frame)]
    if finite_raw.size:
        record["raw_min"] = min(record["raw_min"], float(np.min(finite_raw)))
        record["raw_max"] = max(record["raw_max"], float(np.max(finite_raw)))
    speed = np.sqrt(np.sum(velocity_xy**2, axis=0))
    record["maximum_velocity_km_s"] = max(
        record["maximum_velocity_km_s"],
        float(np.max(speed, initial=0.0)),
    )
    record["mask_fraction_sum"] += float(np.mean(dilated_mask))
    record["maximum_absolute_opacity_change"] = max(
        record["maximum_absolute_opacity_change"],
        float(np.max(np.abs(opacity_change_highres), initial=0.0)),
    )
    record["frame_count"] += 1


def _dataset_catalog(recorder: dict[str, Any]) -> dict[str, dict[str, object]]:
    """Return shape, type, units, dimensions, and description for every dataset."""
    catalog: dict[str, dict[str, object]] = {}

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        catalog[f"/{name}"] = {
            "shape": [int(value) for value in item.shape],
            "dtype": str(item.dtype),
            "units": _json_ready(item.attrs.get("units", "unspecified")),
            "dimensions": _json_ready(item.attrs.get("dimensions", "unspecified")),
            "description": _json_ready(item.attrs.get("description", "")),
        }

    recorder["handle"].visititems(collect)
    return catalog


def _statistics(recorder: dict[str, Any]) -> dict[str, float]:
    """Return compact numerical summaries accumulated during streaming."""
    frame_count = int(recorder["frame_count"])
    return {
        "raw_gong_min": float(recorder["raw_min"]),
        "raw_gong_max": float(recorder["raw_max"]),
        "maximum_coherent_velocity_km_s": float(recorder["maximum_velocity_km_s"]),
        "mean_dilated_thread_mask_fraction": float(recorder["mask_fraction_sum"] / frame_count),
        "maximum_absolute_highres_opacity_change": float(
            recorder["maximum_absolute_opacity_change"]
        ),
    }


def _simulation_summary(
    recorder: dict[str, Any],
    h5_hash: str,
    statistics: dict[str, float],
) -> dict[str, object]:
    """Build the compact record used by the root simulation index."""
    state = recorder["dynamics_state"]
    dynamics_config = state["config"]
    static_config = state["static_config"]
    periods = np.asarray(state["thread_longitudinal_periods_s"], dtype=float)
    n_frames = int(dynamics_config["n_frames"])
    cadence_s = float(dynamics_config["cadence_s"])
    return {
        "simulation_id": recorder["simulation_id"],
        "generated_utc": recorder["generated_utc"],
        "relative_directory": recorder["simulation_id"],
        "h5_file": f"{recorder['simulation_id']}/simulation.h5",
        "h5_sha256": h5_hash,
        "h5_size_bytes": int(recorder["h5_path"].stat().st_size),
        "generator_git_commit": recorder["generator_git_commit"],
        "static_seed": static_config.get("seed"),
        "dynamics_seed": dynamics_config.get("seed"),
        "n_frames": n_frames,
        "cadence_s": cadence_s,
        "duration_s": (n_frames - 1) * cadence_s,
        "native_shape_yx": list(recorder["native_shape"]),
        "n_threads": len(state["initial_threads"]),
        "oscillation_start_time_s": dynamics_config.get("oscillation_start_time_s"),
        "longitudinal_displacement_amplitude_km": dynamics_config.get(
            "longitudinal_displacement_amplitude_km"
        ),
        "transverse_displacement_amplitude_km": dynamics_config.get(
            "transverse_displacement_amplitude_km"
        ),
        "oscillation_mode": dynamics_config.get("oscillation_mode"),
        "period_s": dynamics_config.get("period_s"),
        "transverse_period_s": dynamics_config.get("transverse_period_s"),
        "minimum_longitudinal_period_s": float(np.min(periods)),
        "median_longitudinal_period_s": float(np.median(periods)),
        "maximum_longitudinal_period_s": float(np.max(periods)),
        "damping_time_s": dynamics_config.get("damping_time_s"),
        "brownian_step_min_km": dynamics_config.get("brownian_step_min_km"),
        "brownian_step_max_km": dynamics_config.get("brownian_step_max_km"),
        "half_strength_distance_km": dynamics_config.get("half_strength_distance_km"),
        "background_source": static_config.get("background_source"),
        "disk_mu": static_config.get("disk_mu"),
        **statistics,
    }


def _write_metadata_and_close(recorder: dict[str, Any]) -> dict[str, object]:
    """Complete the staged HDF5 file and write its self-contained JSON record."""
    expected_frames = int(recorder["dynamics_state"]["config"]["n_frames"])
    if recorder["frame_count"] != expected_frames:
        raise RuntimeError(
            "cannot finalize incomplete simulation: wrote "
            f"{recorder['frame_count']} of {expected_frames} frames"
        )

    statistics = _statistics(recorder)
    catalog = _dataset_catalog(recorder)
    handle = recorder["handle"]
    handle.attrs["statistics_json"] = _canonical_json(statistics)
    handle.attrs["complete"] = True
    handle.flush()
    handle.close()

    h5_hash = _file_sha256(recorder["h5_path"])
    summary = _simulation_summary(recorder, h5_hash, statistics)
    state = recorder["dynamics_state"]
    metadata: dict[str, object] = {
        "simulation_dataset_schema_version": SIMULATION_DATASET_SCHEMA_VERSION,
        "dynamics_schema_version": int(recorder["dynamics_schema_version"]),
        "simulation_id": recorder["simulation_id"],
        "generated_utc": recorder["generated_utc"],
        "generator_git_commit": recorder["generator_git_commit"],
        "files": {
            "hdf5": "simulation.h5",
            "hdf5_sha256": h5_hash,
            "metadata": "simulation.json",
        },
        "dynamics_parameters": state["config"],
        "static_forward_model_parameters": state["static_config"],
        "export_parameters": recorder["export_config"],
        "realized_oscillation_site": state.get("site", {}),
        "initial_forward_model_metadata": state.get("initial_metadata", {}),
        "statistics": statistics,
        "video_processing": {
            "input": "/video/raw_gong",
            "output": "/video/processed_uint8",
            "fixed_input_vmin": recorder["video_limits"][0],
            "fixed_input_vmax": recorder["video_limits"][1],
            "temporal_normalization": False,
        },
        "thread_mask_definition": {
            "source": "projected 3D centerlines and local physical radii",
            "uses_opacity": False,
            "native_safety_dilation_px": recorder["export_config"]["thread_mask_dilation_px"],
            "undilated_dataset": "/labels/thread_geometry_mask_native",
            "training_dataset": "/labels/thread_mask_native",
            "native_pixel_center_highres": "index * factor + (factor - 1) / 2",
        },
        "velocity_label_definition": {
            "dataset": "/labels/coherent_velocity_xy_km_s",
            "component_order": ["x", "y"],
            "coordinate_frame": "projected_image_plane",
            "projection": "derivative of position projection, including height motion",
            "units": "km s^-1",
            "mixing": "linear-optical-depth-weighted mean",
            "brownian_velocity_included": False,
            "brownian_position_included": True,
            "loss_weight_dataset": "/labels/velocity_opacity_weight_native",
        },
        "motion_model": {
            "oscillation_mode": state["config"].get("oscillation_mode"),
            "longitudinal_periods": "/dynamics/thread_longitudinal_periods_s",
            "transverse_periods": "/dynamics/thread_transverse_periods_s",
            "prominence_heights": "/dynamics/thread_prominence_heights_m",
            "height_dependent_gravity": "/dynamics/thread_gravity_m_s2",
            "gravity_cutoff_periods": "/dynamics/thread_gravity_cutoff_periods_s",
            "total_position": (
                "initial position plus coherent displacement plus Brownian displacement"
            ),
        },
        "three_dimensional_reconstruction": {
            "initial_positions": "/geometry/initial/point_positions_km",
            "thread_offsets": "/geometry/initial/thread_offsets",
            "frame_displacements": "/state/thread_displacements_km",
            "equation": (
                "position(frame,thread,point) = initial_position(thread,point) "
                "+ thread_displacements_km(frame,thread)"
            ),
        },
        "dataset_catalog": catalog,
        "summary": summary,
    }
    metadata_path = recorder["metadata_path"]
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
        json.dump(_json_ready(metadata), metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
        metadata_file.flush()
        os.fsync(metadata_file.fileno())
    return summary


def _summary_from_metadata(metadata_path: Path) -> dict[str, object]:
    """Read one published simulation summary for index reconstruction."""
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")

    directory = metadata_path.parent
    simulation_id = metadata.get("simulation_id")
    if not isinstance(simulation_id, str) or simulation_id != directory.name:
        raise ValueError(
            f"{metadata_path} simulation_id must equal directory name {directory.name!r}"
        )
    files = metadata.get("files", {})
    h5_name = files.get("hdf5", "simulation.h5") if isinstance(files, dict) else "simulation.h5"
    if not isinstance(h5_name, str) or Path(h5_name).name != h5_name or h5_name in {"", ".", ".."}:
        raise ValueError(f"{metadata_path} contains an unsafe HDF5 filename")
    h5_path = directory / h5_name
    if not h5_path.is_file():
        raise FileNotFoundError(f"published simulation is missing {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"{h5_path} is not marked complete")

    stored_summary = metadata.get("summary")
    if isinstance(stored_summary, dict):
        summary = dict(stored_summary)
    else:
        dynamics = metadata.get("dynamics_parameters", {})
        statistics = metadata.get("statistics", {})
        summary = {
            "simulation_id": simulation_id,
            "generated_utc": metadata.get("generated_utc"),
            "generator_git_commit": metadata.get("generator_git_commit"),
            "dynamics_seed": dynamics.get("seed") if isinstance(dynamics, dict) else None,
            "n_frames": dynamics.get("n_frames") if isinstance(dynamics, dict) else None,
            **(statistics if isinstance(statistics, dict) else {}),
        }
    summary["simulation_id"] = simulation_id
    summary["relative_directory"] = simulation_id
    summary["h5_file"] = f"{simulation_id}/{h5_name}"
    summary["h5_size_bytes"] = int(h5_path.stat().st_size)
    if isinstance(files, dict) and "hdf5_sha256" in files:
        summary["h5_sha256"] = files["hdf5_sha256"]
    return _json_ready(summary)


def rebuild_simulation_index(root: str | Path) -> Path:
    """Atomically rebuild index.json from published simulation.json files.

    The lock serializes concurrent finalizers and manual rebuilds.  Invalid or
    incomplete published directories raise an error and remain untouched.
    """
    simulations_root = Path(root).resolve()
    simulations_root.mkdir(parents=True, exist_ok=True)
    lock_path = simulations_root / ".index.lock"
    index_path = simulations_root / "index.json"

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            summaries = [
                _summary_from_metadata(metadata_path)
                for metadata_path in sorted(simulations_root.glob("sim-*/simulation.json"))
            ]
            summaries.sort(
                key=lambda item: (
                    str(item.get("generated_utc", "")),
                    str(item.get("simulation_id", "")),
                )
            )
            temporary_path = simulations_root / (f".index.json.tmp-{os.getpid()}-{time.time_ns()}")
            try:
                with temporary_path.open(
                    "x",
                    encoding="utf-8",
                    newline="\n",
                ) as index_file:
                    json.dump(summaries, index_file, indent=2, sort_keys=True)
                    index_file.write("\n")
                    index_file.flush()
                    os.fsync(index_file.fileno())
                os.replace(temporary_path, index_path)
                directory_descriptor = os.open(simulations_root, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return index_path


def finalize_simulation(recorder: dict[str, Any]) -> dict[str, Any]:
    """Publish a complete simulation and attempt a recoverable index update."""
    record = _require_dict(recorder, "recorder")
    if record.get("saved") is not None:
        return record["saved"]
    if record.get("published"):
        raise RuntimeError("published recorder has no saved result")
    try:
        _write_metadata_and_close(record)
        if record["output_directory"].exists():
            raise FileExistsError(record["output_directory"])
        os.replace(record["staging_directory"], record["output_directory"])
        record["published"] = True
        record["finalized"] = True
    except BaseException:
        abort_simulation(record)
        raise

    indexed = True
    index_error: str | None = None
    try:
        index_path = rebuild_simulation_index(record["root"])
    except Exception as error:
        indexed = False
        index_error = f"{type(error).__name__}: {error}"
        index_path = record["root"] / "index.json"

    saved: dict[str, Any] = {
        "simulation_id": record["simulation_id"],
        "directory": record["output_directory"],
        "h5_path": record["output_directory"] / "simulation.h5",
        "metadata_path": record["output_directory"] / "simulation.json",
        "index_path": index_path,
        "indexed": indexed,
        "index_error": index_error,
    }
    record["dynamics_state"]["prepare_export_fields"] = False
    record["saved"] = saved
    return saved


def abort_simulation(recorder: dict[str, Any]) -> None:
    """Close and remove only an unpublished simulation staging directory."""
    record = _require_dict(recorder, "recorder")
    handle = record.get("handle")
    if isinstance(handle, h5py.File) and handle.id.valid:
        handle.close()
    staging = record.get("staging_directory")
    if not record.get("published") and isinstance(staging, Path) and staging.exists():
        shutil.rmtree(staging)
    state = record.get("dynamics_state")
    if isinstance(state, dict):
        state["prepare_export_fields"] = False
    record["finalized"] = True


def simulate_and_save_filament_dynamics(
    initial: dict[str, Any],
    dynamics_config: dict[str, Any],
    *,
    background_frames: np.ndarray | None = None,
    simulations_root: str | Path = DEFAULT_SIMULATIONS_ROOT,
    export_config: dict[str, object] | None = None,
    label: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run dynamics and atomically publish the canonical HDF5 dataset.

    ``progress_callback``, when supplied, receives the completed and total
    frame counts after every successfully persisted frame.
    """
    from .dynamics import (
        initialize_dynamics,
        iter_production_frames,
    )

    state = initialize_dynamics(
        initial,
        dynamics_config,
        background_frames=background_frames,
    )
    recorder = open_simulation_recorder(
        state,
        simulations_root=simulations_root,
        export_config=export_config,
        label=label,
    )
    try:
        with closing(iter_production_frames(state)) as frames:
            for frame in frames:
                write_simulation_frame(recorder, frame)
                if progress_callback is not None:
                    progress_callback(
                        int(recorder["frame_count"]),
                        int(state["config"]["n_frames"]),
                    )
        return finalize_simulation(recorder)
    except BaseException:
        abort_simulation(recorder)
        raise
