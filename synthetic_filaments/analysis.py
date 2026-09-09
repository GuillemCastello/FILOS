"""Diagnostics for static filaments and time-distance oscillation analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)
from scipy.optimize import least_squares
from scipy.signal import lombscargle


def padded_mask_bounds(mask: np.ndarray, padding: int = 18) -> tuple[int, int, int, int]:
    """Return clipped ``(row0, row1, column0, column1)`` bounds around a mask."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"mask must be two-dimensional; received shape {values.shape}")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError(f"padding must be an integer >= 0; received {padding!r}")
    rows, columns = np.where(values)
    if rows.size == 0:
        raise ValueError("mask contains no selected pixels")
    return (
        max(int(rows.min()) - padding, 0),
        min(int(rows.max()) + padding + 1, values.shape[0]),
        max(int(columns.min()) - padding, 0),
        min(int(columns.max()) + padding + 1, values.shape[1]),
    )


def measure_static_cohesion(
    final_image: np.ndarray,
    background: np.ndarray,
    physical_mask: np.ndarray,
    soft_mask: np.ndarray,
    tau_map: np.ndarray,
    *,
    observable_absorption_threshold: float = 0.01,
    mask_dilation_iterations: int = 8,
) -> dict[str, float | int]:
    """Measure the cohesion and numerical summaries used by the reference notebook.

    ``soft_mask`` is native physical line absorption before passband dilution.
    ``physical_mask`` uses the same native grid. ``tau_map`` remains on the
    oversampled physical grid.
    """
    final = np.asarray(final_image, dtype=float)
    reference = np.asarray(background, dtype=float)
    mask = np.asarray(physical_mask, dtype=bool)
    native_soft_mask = np.asarray(soft_mask, dtype=float)
    highres_tau = np.asarray(tau_map, dtype=float)
    if final.ndim != 2 or reference.shape != final.shape:
        raise ValueError("final_image and background must be aligned two-dimensional arrays")
    if mask.shape != final.shape or native_soft_mask.shape != final.shape:
        raise ValueError("native masks must match final_image")
    if not np.isfinite(final).all() or not np.isfinite(reference).all():
        raise ValueError("image arrays must contain only finite values")
    if not np.isfinite(highres_tau).all() or np.any(highres_tau < 0.0):
        raise ValueError("tau_map must contain finite non-negative optical depth")

    absorption = np.clip(1.0 - final / np.maximum(reference, 1.0e-6), 0.0, 1.0)
    observable_mask = absorption > float(observable_absorption_threshold)
    observable_mask &= binary_dilation(mask, iterations=mask_dilation_iterations)
    components, component_count = label(
        observable_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    component_sizes = np.bincount(components[observable_mask])
    largest_component_fraction = (
        float(component_sizes.max() / observable_mask.sum()) if component_sizes.size else 0.0
    )
    filled_mask = binary_fill_holes(observable_mask)
    hole_fraction = float((filled_mask & ~observable_mask).sum() / max(int(filled_mask.sum()), 1))
    core = distance_transform_edt(observable_mask) > 1.5
    if not core.any():
        core = observable_mask
    high_pass = absorption - gaussian_filter(absorption, sigma=1.0)
    interior_high_pass_rms = float(np.sqrt(np.mean(high_pass[core] ** 2))) if core.any() else 0.0
    native_tau = -np.log(np.maximum(1.0 - native_soft_mask, 1.0e-12))
    tau_values = native_tau[mask]
    absorption_values = absorption[observable_mask]
    if tau_values.size == 0 or absorption_values.size == 0:
        raise ValueError("diagnostics require non-empty physical and observable masks")

    return {
        "observable_components": int(component_count),
        "largest_component_fraction": largest_component_fraction,
        "hole_fraction": hole_fraction,
        "interior_high_pass_rms": interior_high_pass_rms,
        "highres_tau_max": float(highres_tau.max()),
        "native_tau_p50": float(np.percentile(tau_values, 50)),
        "native_tau_p90": float(np.percentile(tau_values, 90)),
        "observed_absorption_p50": float(np.percentile(absorption_values, 50)),
        "observed_absorption_p90": float(np.percentile(absorption_values, 90)),
    }


def _distance_to_image_boundary(
    x: float,
    y: float,
    direction_x: float,
    direction_y: float,
    image_width: int,
    image_height: int,
) -> float:
    """Return distance from a point to the first image boundary along a ray."""
    candidates: list[float] = []
    if direction_x > 0.0:
        candidates.append((image_width - 1 - x) / direction_x)
    elif direction_x < 0.0:
        candidates.append(-x / direction_x)
    if direction_y > 0.0:
        candidates.append((image_height - 1 - y) / direction_y)
    elif direction_y < 0.0:
        candidates.append(-y / direction_y)
    positive = [distance for distance in candidates if distance >= 0.0]
    return min(positive) if positive else 0.0


def extract_time_distance(
    video: np.ndarray,
    *,
    x: float,
    y: float,
    angle_deg: float,
    width_px: float = 1.0,
    length_px: float | None = None,
    spatial_step_px: float = 1.0,
    width_step_px: float = 1.0,
) -> dict[str, np.ndarray]:
    """Extract a time-distance diagram through a finite-width linear slit.

    Parameters use image coordinates: ``x`` is column, ``y`` is row, and
    ``angle_deg`` increases counterclockwise from the positive x direction.
    The returned mapping contains ``values`` with shape ``(time, distance)``
    and the signed distance coordinates in pixels.
    """
    frames = np.asarray(video)
    if frames.ndim != 3:
        raise ValueError(f"video must have shape (time, y, x); received {frames.shape}")
    numeric = {
        "x": x,
        "y": y,
        "angle_deg": angle_deg,
        "width_px": width_px,
        "spatial_step_px": spatial_step_px,
        "width_step_px": width_step_px,
    }
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("slit coordinates and sampling controls must be finite")
    if width_px <= 0.0 or spatial_step_px <= 0.0 or width_step_px <= 0.0:
        raise ValueError("slit width and sampling steps must be > 0")
    if length_px is not None and (not np.isfinite(length_px) or length_px <= 0.0):
        raise ValueError("length_px must be None or a finite value > 0")

    n_time, image_height, image_width = frames.shape
    if not (0.0 <= x <= image_width - 1 and 0.0 <= y <= image_height - 1):
        raise ValueError(f"slit center {(x, y)} lies outside shape {frames.shape[1:]}")
    angle_rad = np.deg2rad(angle_deg)
    direction_x, direction_y = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    normal_x, normal_y = -direction_y, direction_x
    if length_px is None:
        positive = _distance_to_image_boundary(
            x, y, direction_x, direction_y, image_width, image_height
        )
        negative = _distance_to_image_boundary(
            x, y, -direction_x, -direction_y, image_width, image_height
        )
        distances = np.arange(-negative, positive + 0.5 * spatial_step_px, spatial_step_px)
    else:
        distances = np.arange(
            -length_px / 2.0,
            length_px / 2.0 + 0.5 * spatial_step_px,
            spatial_step_px,
        )
    across = (
        np.asarray([0.0])
        if width_px <= 1.0
        else np.arange(-width_px / 2.0, width_px / 2.0 + 0.5 * width_step_px, width_step_px)
    )
    slit_x = x + distances[None, :] * direction_x + across[:, None] * normal_x
    slit_y = y + distances[None, :] * direction_y + across[:, None] * normal_y
    valid = (
        (slit_x >= 0.0)
        & (slit_x <= image_width - 1)
        & (slit_y >= 0.0)
        & (slit_y <= image_height - 1)
    )
    coordinates = np.vstack((slit_y.ravel(), slit_x.ravel()))
    values = np.empty((n_time, distances.size), dtype=np.float64)
    for time_index in range(n_time):
        sampled = map_coordinates(
            frames[time_index].astype(float, copy=False),
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).reshape(slit_x.shape)
        sampled[~valid] = np.nan
        counts = np.sum(np.isfinite(sampled), axis=0)
        values[time_index] = np.divide(
            np.nansum(sampled, axis=0),
            counts,
            out=np.full(distances.size, np.nan),
            where=counts > 0,
        )
    return {
        "values": values,
        "distance_px": distances,
        "slit_x_px": slit_x,
        "slit_y_px": slit_y,
    }


def _subpixel_dark_minimum(image: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """Refine an integer dark ridge through local parabolic interpolation."""
    n_distance, n_time = image.shape
    refined = ridge.astype(float, copy=True)
    for time_index in range(n_time):
        position = int(round(ridge[time_index]))
        if position <= 0 or position >= n_distance - 1:
            continue
        left, center, right = image[position - 1 : position + 2, time_index]
        denominator = left - 2.0 * center + right
        if denominator > 1.0e-12:
            offset = 0.5 * (left - right) / denominator
            refined[time_index] = position + np.clip(offset, -1.0, 1.0)
    return refined


def track_dark_band(
    time_distance: np.ndarray,
    *,
    initial_position_px: float | None = None,
    distance_range_px: tuple[float, float] | None = None,
    max_jump_px: int = 6,
    jump_penalty: float = 0.15,
    initial_penalty: float = 0.5,
    spatial_sigma_px: float = 1.5,
    temporal_sigma_frames: float = 0.7,
) -> np.ndarray:
    """Track the minimum-cost dark ridge in a ``(distance, time)`` diagram."""
    values = np.asarray(time_distance, dtype=float)
    if values.ndim != 2:
        raise ValueError("time_distance must be two-dimensional with axes (distance, time)")
    if isinstance(max_jump_px, bool) or not isinstance(max_jump_px, int) or max_jump_px < 1:
        raise ValueError("max_jump_px must be an integer >= 1")
    controls = (jump_penalty, initial_penalty, spatial_sigma_px, temporal_sigma_frames)
    if not all(np.isfinite(value) and value >= 0.0 for value in controls):
        raise ValueError("ridge penalties and smoothing scales must be finite and >= 0")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("time_distance contains no finite samples")
    image = np.where(finite, values, np.nanmedian(values))
    image = gaussian_filter(
        image,
        sigma=(spatial_sigma_px, temporal_sigma_frames),
        mode="nearest",
    )
    p10 = np.percentile(image, 10, axis=0)
    p90 = np.percentile(image, 90, axis=0)
    local_cost = (image - p10[None, :]) / np.maximum(p90 - p10, 1.0e-12)[None, :]
    n_distance, n_time = local_cost.shape
    if distance_range_px is not None:
        lower = max(0, int(np.floor(distance_range_px[0])))
        upper = min(n_distance - 1, int(np.ceil(distance_range_px[1])))
        if lower >= upper:
            raise ValueError("distance_range_px does not contain a valid interval")
        allowed = np.zeros(n_distance, dtype=bool)
        allowed[lower : upper + 1] = True
        local_cost[~allowed] = np.inf

    positions = np.arange(n_distance, dtype=float)
    accumulated = np.full((n_distance, n_time), np.inf)
    predecessor = np.full((n_distance, n_time), -1, dtype=np.int32)
    accumulated[:, 0] = local_cost[:, 0]
    if initial_position_px is not None:
        if not np.isfinite(initial_position_px):
            raise ValueError("initial_position_px must be finite")
        accumulated[:, 0] += initial_penalty * (positions - initial_position_px) ** 2
    for time_index in range(1, n_time):
        for current_position in range(n_distance):
            if not np.isfinite(local_cost[current_position, time_index]):
                continue
            previous_min = max(0, current_position - max_jump_px)
            previous_max = min(n_distance, current_position + max_jump_px + 1)
            previous_positions = np.arange(previous_min, previous_max)
            candidate_costs = (
                accumulated[previous_min:previous_max, time_index - 1]
                + jump_penalty * (previous_positions - current_position) ** 2
            )
            best_local_index = int(np.argmin(candidate_costs))
            accumulated[current_position, time_index] = (
                local_cost[current_position, time_index] + candidate_costs[best_local_index]
            )
            predecessor[current_position, time_index] = previous_positions[best_local_index]
    ridge = np.empty(n_time, dtype=float)
    ridge[-1] = int(np.argmin(accumulated[:, -1]))
    for time_index in range(n_time - 1, 0, -1):
        previous = predecessor[int(ridge[time_index]), time_index]
        if previous < 0:
            raise RuntimeError("dark-band tracking failed; widen the search or max_jump_px")
        ridge[time_index - 1] = previous
    return _subpixel_dark_minimum(image, ridge)


def damped_sine_model(
    time_s: np.ndarray,
    *,
    equilibrium_px: float,
    drift_px_s: float,
    amplitude_px: float,
    period_s: float,
    damping_time_s: float,
    phase_rad: float,
    reference_time_s: float,
) -> np.ndarray:
    """Evaluate a damped sinusoid around a linearly drifting equilibrium."""
    relative_time = np.asarray(time_s, dtype=float) - reference_time_s
    return (
        equilibrium_px
        + drift_px_s * relative_time
        + amplitude_px
        * np.exp(-relative_time / damping_time_s)
        * np.sin(2.0 * np.pi * relative_time / period_s + phase_rad)
    )


def _estimate_period(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_period_s: float,
    maximum_period_s: float,
    frequency_count: int = 5000,
) -> float:
    """Estimate an initial oscillation period with Lomb--Scargle power."""
    frequencies = np.linspace(1.0 / maximum_period_s, 1.0 / minimum_period_s, frequency_count)
    power = lombscargle(
        time_s,
        values - np.mean(values),
        2.0 * np.pi * frequencies,
        precenter=False,
        normalize=True,
    )
    return float(1.0 / frequencies[int(np.argmax(power))])


def _parameter_uncertainties(
    jacobian: np.ndarray,
    cost: float,
    observation_count: int,
    parameter_count: int,
) -> np.ndarray:
    """Local one-sigma errors; unidentifiable parameters receive infinity."""
    degrees_of_freedom = observation_count - parameter_count
    if degrees_of_freedom <= 0:
        return np.full(parameter_count, np.nan)
    residual_variance = 2.0 * cost / degrees_of_freedom
    try:
        # Normalize columns before the SVD so parameter units do not set the rank.
        column_scale = np.linalg.norm(jacobian, axis=0)
        column_scale[column_scale == 0.0] = 1.0
        _, singular_values, right_vectors = np.linalg.svd(
            jacobian / column_scale, full_matrices=False
        )
        relative_tolerance = np.finfo(float).eps * max(jacobian.shape)
        resolved = singular_values > relative_tolerance * singular_values[0]
        inverse = right_vectors[resolved].T / singular_values[resolved]
        uncertainties = (
            np.sqrt(residual_variance * np.sum(inverse**2, axis=1)) / column_scale
        )
        # A pseudoinverse alone would assign spuriously small errors to null modes.
        unresolved = np.linalg.norm(right_vectors[~resolved], axis=0) > relative_tolerance
        uncertainties[unresolved] = np.inf
        return uncertainties
    except np.linalg.LinAlgError:
        return np.full(parameter_count, np.nan)


def fit_damped_sine(
    ridge_px: Sequence[float] | np.ndarray,
    *,
    time_s: Sequence[float] | np.ndarray | None = None,
    period_bounds_s: tuple[float, float] | None = None,
    damping_bounds_s: tuple[float, float] | None = None,
    robust: bool = True,
) -> dict[str, object]:
    """Fit a damped sinusoid and linear drift to a tracked ridge.

    Errors use a local Jacobian covariance. With ``robust=True``, the modified
    Jacobian and robust cost give approximate errors, not calibrated intervals.
    """
    ridge = np.asarray(ridge_px, dtype=float)
    times = (
        np.arange(ridge.size, dtype=float) if time_s is None else np.asarray(time_s, dtype=float)
    )
    if ridge.ndim != 1 or times.shape != ridge.shape:
        raise ValueError("ridge_px and time_s must be aligned one-dimensional arrays")
    valid = np.isfinite(times) & np.isfinite(ridge)
    fit_times = times[valid]
    fit_ridge = ridge[valid]
    if fit_times.size < 10:
        raise ValueError("at least 10 finite ridge samples are required")
    order = np.argsort(fit_times)
    fit_times, fit_ridge = fit_times[order], fit_ridge[order]
    reference_time = float(fit_times[0])
    relative_time = fit_times - reference_time
    duration = float(relative_time[-1])
    time_spacing = float(np.median(np.diff(fit_times)))
    if duration <= 0.0 or time_spacing <= 0.0:
        raise ValueError("time_s must increase over a non-zero interval")
    if period_bounds_s is None:
        minimum_period, maximum_period = max(4.0 * time_spacing, duration / 20.0), duration
    else:
        minimum_period, maximum_period = (float(value) for value in period_bounds_s)
    if not 0.0 < minimum_period < maximum_period:
        raise ValueError("period_bounds_s must contain two positive ordered values")
    linear_coefficients = np.polyfit(relative_time, fit_ridge, deg=1)
    detrended = fit_ridge - np.polyval(linear_coefficients, relative_time)
    initial_period = _estimate_period(
        relative_time,
        detrended,
        minimum_period,
        maximum_period,
    )
    initial_amplitude = max(
        0.5 * (np.percentile(fit_ridge, 95) - np.percentile(fit_ridge, 5)),
        1.0e-3,
    )
    if damping_bounds_s is None:
        minimum_damping = max(time_spacing, 0.1 * initial_period)
        maximum_damping = max(20.0 * duration, minimum_damping * 10.0)
    else:
        minimum_damping, maximum_damping = (float(value) for value in damping_bounds_s)
    if not 0.0 < minimum_damping < maximum_damping:
        raise ValueError("damping_bounds_s must contain two positive ordered values")
    initial_parameters = np.asarray(
        [
            np.median(fit_ridge),
            linear_coefficients[0],
            initial_amplitude,
            initial_period,
            np.clip(2.0 * duration, minimum_damping, maximum_damping),
            np.arcsin(np.clip(detrended[0] / initial_amplitude, -1.0, 1.0)),
        ],
        dtype=float,
    )
    position_span = max(float(np.ptp(fit_ridge)), 1.0)
    lower_bounds = np.asarray(
        [
            np.min(fit_ridge) - position_span,
            -2.0 * position_span / duration,
            0.0,
            minimum_period,
            minimum_damping,
            -2.0 * np.pi,
        ]
    )
    upper_bounds = np.asarray(
        [
            np.max(fit_ridge) + position_span,
            2.0 * position_span / duration,
            max(3.0 * position_span, 1.0),
            maximum_period,
            maximum_damping,
            2.0 * np.pi,
        ]
    )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return (
            damped_sine_model(
                fit_times,
                equilibrium_px=parameters[0],
                drift_px_s=parameters[1],
                amplitude_px=parameters[2],
                period_s=parameters[3],
                damping_time_s=parameters[4],
                phase_rad=parameters[5],
                reference_time_s=reference_time,
            )
            - fit_ridge
        )

    optimization = least_squares(
        residuals,
        x0=initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        loss="soft_l1" if robust else "linear",
        f_scale=1.0,
        max_nfev=50_000,
    )
    fitted = optimization.x
    model = damped_sine_model(
        times,
        equilibrium_px=fitted[0],
        drift_px_s=fitted[1],
        amplitude_px=fitted[2],
        period_s=fitted[3],
        damping_time_s=fitted[4],
        phase_rad=fitted[5],
        reference_time_s=reference_time,
    )
    names = (
        "equilibrium_px",
        "drift_px_s",
        "amplitude_px",
        "period_s",
        "damping_time_s",
        "phase_rad",
    )
    uncertainties = _parameter_uncertainties(
        optimization.jac,
        float(optimization.cost),
        fit_times.size,
        fitted.size,
    )
    parameters = {name: float(value) for name, value in zip(names, fitted, strict=True)}
    parameter_uncertainties = {
        name: float(value) for name, value in zip(names, uncertainties, strict=True)
    }
    try:
        column_scale = np.linalg.norm(optimization.jac, axis=0)
        column_scale[column_scale == 0.0] = 1.0
        singular_values = np.linalg.svd(
            optimization.jac / column_scale,
            compute_uv=False,
        )
        rank_tolerance = np.finfo(float).eps * max(optimization.jac.shape)
        jacobian_rank = int(
            np.count_nonzero(singular_values > rank_tolerance * singular_values[0])
        )
        jacobian_condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0.0
            else float("inf")
        )
    except np.linalg.LinAlgError:
        jacobian_rank = 0
        jacobian_condition = float("inf")
    active_bound_count = int(np.count_nonzero(optimization.active_mask))
    recovery_trustworthy = bool(
        optimization.success
        and jacobian_rank == fitted.size
        and active_bound_count == 0
        and np.isfinite(uncertainties).all()
        and jacobian_condition < 1.0 / np.sqrt(np.finfo(float).eps)
    )
    parameters["reference_time_s"] = reference_time
    parameter_uncertainties["reference_time_s"] = 0.0
    return {
        "parameters": parameters,
        "uncertainties": parameter_uncertainties,
        "model_px": model,
        "ridge_px": ridge,
        "residuals_px": ridge - model,
        "time_s": times,
        "success": bool(optimization.success),
        "message": str(optimization.message),
        "cost": float(optimization.cost),
        "diagnostics": {
            "numerical_termination_success": bool(optimization.success),
            "parameter_recovery_trustworthy": recovery_trustworthy,
            "jacobian_rank": jacobian_rank,
            "parameter_count": int(fitted.size),
            "scaled_jacobian_condition_number": jacobian_condition,
            "active_bound_count": active_bound_count,
            "robust_uncertainties_are_approximate": bool(robust),
        },
    }
