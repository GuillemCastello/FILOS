"""Top-view H-alpha optical-depth rasterization and transfer functions."""

from collections.abc import Callable
from time import monotonic

import numpy as np
from scipy.special import erf

from .config import StaticConfig, image_height_km, image_width_km
from .execution import integer_setting, thread_pool
from .geometry import Thread, effective_thread_radius

# The projected profile of a uniformly filled circular cross-section has
# variance R^2 / 4 along either image axis. A Gaussian with sigma = R / 2
# therefore preserves that second moment without treating the physical radius
# itself as sigma (which made individual threads wider than their diameters).
THREAD_GAUSSIAN_SIGMA_PER_RADIUS = 0.5


def _accumulate_sparse(
    target: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
) -> None:
    """Accumulate ordered sparse values into one independent image."""
    np.add.at(target, (rows, columns), values)


def _pixel_integrated_gaussian_1d(
    pixel_indices: np.ndarray, center_px: float, sigma_px: float
) -> np.ndarray:
    """Integrate an unnormalised Gaussian over unit-width detector pixels."""
    root_two_sigma = np.sqrt(2.0) * sigma_px
    lower = (pixel_indices - 0.5 - center_px) / root_two_sigma
    upper = (pixel_indices + 0.5 - center_px) / root_two_sigma
    return np.sqrt(np.pi / 2.0) * sigma_px * (erf(upper) - erf(lower))


def _limb_unit_vector(config: StaticConfig) -> tuple[float, float]:
    phi = np.deg2rad(config["limb_direction_deg"])
    return float(np.cos(phi)), float(np.sin(phi))


def _project_points(
    config: StaticConfig,
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points onto the image plane for the configured disk position.

    At disk center (mu = 1) this is the identity. Off-center, structures at
    height z shift toward the limb (parallax) and the limb-direction axis is
    foreshortened by mu (Gunar et al. 2018 projection effects). The thread
    cross-section PSF is kept isotropic — acceptable at GONG resolution.
    """
    mu = config["disk_mu"]
    if mu >= 0.999:
        return x_km, y_km
    ex, ey = _limb_unit_vector(config)
    tan_theta = np.sqrt(max(1.0 - mu**2, 0.0)) / mu

    x = x_km + z_km * tan_theta * ex
    y = y_km + z_km * tan_theta * ey

    cx, cy = image_width_km(config) / 2.0, image_height_km(config) / 2.0
    along = (x - cx) * ex + (y - cy) * ey
    x = x + (mu - 1.0) * along * ex
    y = y + (mu - 1.0) * along * ey
    return x, y


def _project_vectors(config: StaticConfig, vectors_xyz: np.ndarray) -> np.ndarray:
    """Apply the derivative of ``_project_points`` to intrinsic XYZ vectors.

    The last input axis contains x, y, z; the returned axis contains image-plane
    x, y. Height motion contributes to the projected limb-direction velocity.
    """
    vectors = np.asarray(vectors_xyz, dtype=float)
    mu = config["disk_mu"]
    if mu >= 0.999:
        return vectors[..., :2].copy()
    limb = np.asarray(_limb_unit_vector(config))
    along = np.sum(vectors[..., :2] * limb, axis=-1)
    shift = (mu - 1.0) * along + np.sqrt(max(1.0 - mu**2, 0.0)) * vectors[..., 2]
    return vectors[..., :2] + shift[..., None] * limb


def thread_tau_pixel_contributions(
    config: StaticConfig,
    thread: Thread,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse raw optical-depth contributions for one projected thread.

    Returned arrays are ``(row_indices, column_indices, values)``. Pixel
    indices may repeat because contributions from multiple centerline samples
    must be accumulated rather than assigned.
    """
    px = config["pixel_size_km"]
    # Projecting the centerline already changes the line-convolution density
    # by the appropriate orientation-dependent factor. An additional 1/mu
    # multiplier double-counts foreshortening for radial threads.
    slant = 1.0
    sigmas_px = THREAD_GAUSSIAN_SIGMA_PER_RADIUS * effective_thread_radius(thread) / px
    ds_px = np.gradient(thread["s"]) / px
    amplitudes = (
        slant * thread["tau0"] * thread["tau_along"] * ds_px / (np.sqrt(2.0 * np.pi) * sigmas_px)
    )
    x_proj, y_proj = _project_points(
        config,
        thread["x"],
        thread["y"],
        thread["z"],
    )
    x_px = x_proj / px
    y_px = y_proj / px
    maximum_amplitude = float(np.max(amplitudes, initial=0.0))
    active = (amplitudes > 0.0) & (amplitudes >= 1e-7 * maximum_amplitude)
    if not np.any(active):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
    x_px = x_px[active]
    y_px = y_px[active]
    amplitudes = amplitudes[active]
    sigmas_px = sigmas_px[active]
    half_windows = np.maximum(
        np.ceil(4.0 * sigmas_px + 0.5).astype(int),
        1,
    )
    centers_x = np.rint(x_px).astype(int)
    centers_y = np.rint(y_px).astype(int)
    maximum_half_window = int(half_windows.max())
    offsets = np.arange(
        -maximum_half_window,
        maximum_half_window + 1,
    )
    x_indices = centers_x[:, None] + offsets
    y_indices = centers_y[:, None] + offsets
    within_window = np.abs(offsets)[None, :] <= half_windows[:, None]
    valid_x = within_window & (x_indices >= 0) & (x_indices < config["nx"])
    valid_y = within_window & (y_indices >= 0) & (y_indices < config["ny"])
    gx = _pixel_integrated_gaussian_1d(
        x_indices,
        x_px[:, None],
        sigmas_px[:, None],
    )
    gy = _pixel_integrated_gaussian_1d(
        y_indices,
        y_px[:, None],
        sigmas_px[:, None],
    )
    gx = np.where(valid_x, gx, 0.0)
    gy = np.where(valid_y, gy, 0.0)
    blob = gy[:, :, None] * gx[:, None, :]
    values = amplitudes[:, None, None] * blob
    valid = valid_y[:, :, None] & valid_x[:, None, :]
    y_grid = np.broadcast_to(y_indices[:, :, None], valid.shape)
    x_grid = np.broadcast_to(x_indices[:, None, :], valid.shape)
    return (
        np.asarray(y_grid[valid], dtype=np.int64),
        np.asarray(x_grid[valid], dtype=np.int64),
        np.asarray(values[valid], dtype=float),
    )


def rasterize_thread_tau_into(
    tau_map: np.ndarray,
    config: StaticConfig,
    thread: Thread,
) -> None:
    """Add one raw projected thread contribution to ``tau_map`` in place."""
    if tau_map.shape != (config["ny"], config["nx"]):
        raise ValueError(
            f"tau_map shape must be {(config['ny'], config['nx'])}; received {tau_map.shape}"
        )
    row_indices, column_indices, values = thread_tau_pixel_contributions(config, thread)
    np.add.at(tau_map, (row_indices, column_indices), values)


def rasterize_tau_map(
    config: StaticConfig,
    threads: list[Thread],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Project threads and sum their pixel-integrated optical depth."""
    tau_map = np.zeros((config["ny"], config["nx"]))
    last_progress = monotonic()
    for thread_index, thread in enumerate(threads):
        rasterize_thread_tau_into(tau_map, config, thread)
        now = monotonic()
        if progress_callback is not None and (
            (thread_index + 1) % 25 == 0
            or now - last_progress >= 0.5
            or thread_index + 1 == len(threads)
        ):
            progress_callback(thread_index + 1, len(threads))
            last_progress = now
    return tau_map


def prepare_dynamic_tau_rasterization(
    config: StaticConfig,
    threads: list[Thread],
) -> dict[str, np.ndarray | int]:
    """Pack invariant per-thread opacity samples for repeated dynamics frames.

    Dynamics only translate rigid thread centerlines.  The optical-depth
    amplitudes, Gaussian widths, and active centerline samples therefore stay
    fixed for the complete sequence and can be calculated once.  Packed
    samples retain the original thread/sample order so repeated rasterization
    has the same floating-point accumulation order as the simple reference
    implementation.
    """
    px = float(config["pixel_size_km"])
    packed: dict[str, list[np.ndarray]] = {
        "x_km": [],
        "y_km": [],
        "z_km": [],
        "amplitudes": [],
        "sigmas_px": [],
        "half_windows": [],
        "thread_indices": [],
    }
    for thread_index, thread in enumerate(threads):
        slant = 1.0
        sigmas_px = THREAD_GAUSSIAN_SIGMA_PER_RADIUS * effective_thread_radius(thread) / px
        ds_px = np.gradient(thread["s"]) / px
        amplitudes = (
            slant
            * thread["tau0"]
            * thread["tau_along"]
            * ds_px
            / (np.sqrt(2.0 * np.pi) * sigmas_px)
        )
        maximum_amplitude = float(np.max(amplitudes, initial=0.0))
        active = (amplitudes > 0.0) & (amplitudes >= 1.0e-7 * maximum_amplitude)
        if not np.any(active):
            continue
        active_sigmas = np.asarray(sigmas_px[active], dtype=float)
        packed["x_km"].append(np.asarray(thread["x"], dtype=float)[active])
        packed["y_km"].append(np.asarray(thread["y"], dtype=float)[active])
        packed["z_km"].append(np.asarray(thread["z"], dtype=float)[active])
        packed["amplitudes"].append(np.asarray(amplitudes[active], dtype=float))
        packed["sigmas_px"].append(active_sigmas)
        packed["half_windows"].append(
            np.maximum(np.ceil(4.0 * active_sigmas + 0.5).astype(np.int64), 1)
        )
        packed["thread_indices"].append(
            np.full(int(np.count_nonzero(active)), thread_index, dtype=np.int64)
        )

    result: dict[str, np.ndarray | int] = {"n_threads": len(threads)}
    for name, parts in packed.items():
        dtype = np.int64 if name in {"half_windows", "thread_indices"} else float
        result[name] = (
            np.concatenate(parts)
            if parts
            else np.empty(0, dtype=dtype)
        )
    return result


def rasterize_prepared_dynamic_tau(
    config: StaticConfig,
    prepared: dict[str, np.ndarray | int],
    displacements_km: np.ndarray,
    thread_values: np.ndarray | None = None,
    *,
    sample_batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Rasterize translated packed samples and optional weighted fields.

    If ``thread_values`` is provided with shape ``(n_threads, n_fields)``, the
    second returned array contains $\sum_j \tau_j v_{j,k}$ for each field
    $k$.  Otherwise the second result is ``None``.  Batches limit temporary
    Gaussian-kernel memory without changing contribution order.
    """
    n_threads = int(prepared["n_threads"])
    displacements = np.asarray(displacements_km, dtype=float)
    if displacements.shape != (n_threads, 3):
        raise ValueError(
            f"displacements_km must have shape {(n_threads, 3)}; received {displacements.shape}"
        )
    if sample_batch_size is None:
        sample_batch_size = integer_setting("FILAMENT_RENDER_BATCH_SIZE", 8_192)
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be positive")

    values = None if thread_values is None else np.asarray(thread_values, dtype=float)
    if values is not None and (values.ndim != 2 or values.shape[0] != n_threads):
        raise ValueError(
            "thread_values must have shape (n_threads, n_fields); "
            f"received {values.shape} for {n_threads} threads"
        )

    tau_map = np.zeros((config["ny"], config["nx"]))
    weighted_fields = (
        None
        if values is None
        else np.zeros((values.shape[1], config["ny"], config["nx"]))
    )
    thread_indices = np.asarray(prepared["thread_indices"], dtype=np.int64)
    half_windows = np.asarray(prepared["half_windows"], dtype=np.int64)
    x_km = np.asarray(prepared["x_km"], dtype=float)
    y_km = np.asarray(prepared["y_km"], dtype=float)
    z_km = np.asarray(prepared["z_km"], dtype=float)
    amplitudes = np.asarray(prepared["amplitudes"], dtype=float)
    sigmas_px = np.asarray(prepared["sigmas_px"], dtype=float)
    if thread_indices.size == 0:
        return tau_map, weighted_fields

    active_fields = (
        np.zeros(0, dtype=bool)
        if values is None
        else np.any(values != 0.0, axis=0)
    )
    has_active_fields = bool(np.any(active_fields))
    maximum_half_window = int(np.max(half_windows))
    offsets = np.arange(-maximum_half_window, maximum_half_window + 1)
    px = float(config["pixel_size_km"])
    worker_count = 1 + int(np.count_nonzero(active_fields))
    executor = thread_pool("raster", worker_count) if has_active_fields else None
    for start in range(0, thread_indices.size, sample_batch_size):
        stop = min(start + sample_batch_size, thread_indices.size)
        sample_slice = slice(start, stop)
        sample_threads = thread_indices[sample_slice]
        translated_x = x_km[sample_slice] + displacements[sample_threads, 0]
        translated_y = y_km[sample_slice] + displacements[sample_threads, 1]
        translated_z = z_km[sample_slice] + displacements[sample_threads, 2]
        projected_x, projected_y = _project_points(
            config,
            translated_x,
            translated_y,
            translated_z,
        )
        x_px = projected_x / px
        y_px = projected_y / px
        batch_sigmas_px = sigmas_px[sample_slice]
        batch_amplitudes = amplitudes[sample_slice]
        batch_half_windows = half_windows[sample_slice]
        x_indices = np.rint(x_px).astype(np.int64)[:, None] + offsets
        y_indices = np.rint(y_px).astype(np.int64)[:, None] + offsets
        within_window = np.abs(offsets)[None, :] <= batch_half_windows[:, None]
        valid_x = within_window & (x_indices >= 0) & (x_indices < config["nx"])
        valid_y = within_window & (y_indices >= 0) & (y_indices < config["ny"])
        gx = _pixel_integrated_gaussian_1d(
            x_indices,
            x_px[:, None],
            batch_sigmas_px[:, None],
        )
        gy = _pixel_integrated_gaussian_1d(
            y_indices,
            y_px[:, None],
            batch_sigmas_px[:, None],
        )
        gx = np.where(valid_x, gx, 0.0)
        gy = np.where(valid_y, gy, 0.0)
        blob = gy[:, :, None] * gx[:, None, :]
        contribution_cube = batch_amplitudes[:, None, None] * blob
        valid = valid_y[:, :, None] & valid_x[:, None, :]
        row_grid = np.broadcast_to(y_indices[:, :, None], valid.shape)
        column_grid = np.broadcast_to(x_indices[:, None, :], valid.shape)
        rows = np.asarray(row_grid[valid], dtype=np.int64)
        columns = np.asarray(column_grid[valid], dtype=np.int64)
        contributions = np.asarray(contribution_cube[valid], dtype=float)
        if executor is None or weighted_fields is None or values is None:
            _accumulate_sparse(tau_map, rows, columns, contributions)
            continue

        repeated_threads = np.broadcast_to(
            sample_threads[:, None, None],
            valid.shape,
        )[valid]
        futures = [
            executor.submit(
                _accumulate_sparse,
                tau_map,
                rows,
                columns,
                contributions,
            )
        ]
        for field_index in range(values.shape[1]):
            if not active_fields[field_index]:
                continue
            field_contributions = contributions * values[repeated_threads, field_index]
            futures.append(
                executor.submit(
                    _accumulate_sparse,
                    weighted_fields[field_index],
                    rows,
                    columns,
                    field_contributions,
                )
            )
        for future in futures:
            future.result()
    return tau_map, weighted_fields


def render_halpha_absorption(
    background: np.ndarray,
    tau_map: np.ndarray,
    source_fraction: float,
    source_level: float | None = None,
    bandpass_line_fraction: float = 1.0,
) -> np.ndarray:
    """I = I_bg * exp(-tau) + S * (1 - exp(-tau)), with S a *scalar*.

    S is the filament's own source function -- scattered disk radiation plus
    a small thermal part -- so it is spatially smooth: source_fraction times
    the scalar background level (its median unless source_level is given).
    A per-pixel S (the old behavior) turns the filament into a pure
    multiplicative screen through which the background's fine mottle stays
    visible at full relative contrast, even where tau is large.
    """
    if source_level is None:
        source_level = float(np.median(background))
    transmission = dilute_transmission(
        np.exp(-tau_map),
        bandpass_line_fraction,
    )
    return compose_transmission(background, transmission, source_fraction, source_level)


def dilute_transmission(
    line_center_transmission: np.ndarray,
    bandpass_line_fraction: float,
) -> np.ndarray:
    r"""Apply $T_{\rm obs}=1-\eta(1-T_{\rm line})$ for a filtergram."""
    if not 0.0 <= bandpass_line_fraction <= 1.0:
        raise ValueError("bandpass_line_fraction must lie in [0, 1]")
    transmission = np.asarray(line_center_transmission, dtype=float)
    return np.clip(
        1.0 - bandpass_line_fraction * (1.0 - transmission),
        0.0,
        1.0,
    )


def compose_transmission(
    background: np.ndarray,
    transmission: np.ndarray,
    source_fraction: float,
    source_level: float,
) -> np.ndarray:
    """I = I_bg * T + S * (1 - T) with scalar S = source_fraction * source_level.

    One absorption formula for both paths: hi-res rendering passes
    T = exp(-tau) at full resolution; the real-background path passes a
    transmission already PSF-degraded to the native GONG grid
    (degradation.degrade_transmission) together with the native crop.
    """
    source = source_fraction * source_level
    return background * transmission + source * (1.0 - transmission)


def make_masks(tau_map: np.ndarray, tau_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft absorption mask and hard binary filament mask."""
    soft_mask = 1.0 - np.exp(-tau_map)
    hard_mask = tau_map > tau_threshold
    return soft_mask, hard_mask
