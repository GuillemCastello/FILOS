"""Functional orchestration for one static synthetic filament."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from time import monotonic
from typing import Any, Mapping

import numpy as np

from .config import (
    make_static_config,
    update_static_config,
    validate_static_config,
)
from .degradation import degrade_transmission
from .geometry import (
    Spine,
    Thread,
    interpolate_spine,
    make_spine,
    make_threads,
    spine_total_length,
)
from .opacity_table import heinzel_opacity_table_metadata
from .plasma import assign_plasma
from .render import (
    compose_transmission,
    dilute_transmission,
    make_masks,
    rasterize_tau_map,
    render_halpha_absorption,
)

HEINZEL_SOURCE_HEIGHT_KM = np.asarray([10_000.0, 20_000.0, 30_000.0])
HEINZEL_SOURCE_FRACTION = np.asarray([3.22, 2.98, 2.80]) / 6.93
_OPACITY_TABLE_METADATA = heinzel_opacity_table_metadata()

StaticFilamentResult = dict[str, Any]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    started: float,
    description: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    **details: Any,
) -> None:
    """Emit one lightweight numerical-pipeline progress event."""
    if callback is None:
        return
    callback(
        {
            "stage": stage,
            "completed": completed,
            "total": total,
            "elapsed_seconds": monotonic() - started,
            "description": description,
            **details,
        }
    )


def make_h5_static_config(
    backgrounds: Mapping[str, Any],
    *,
    seed: int,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the static configuration aligned to one selected HDF5 crop."""
    native_shape = tuple(int(value) for value in backgrounds["native_shape"])
    selection_rng = np.random.default_rng(seed)
    config = make_static_config(
        seed=seed,
        native_shape=native_shape,
        disk_mu=float(backgrounds["disk_mu"]),
        orientation_deg=float(selection_rng.uniform(-90.0, 90.0)),
        chirality=int(selection_rng.choice([-1, 1])),
        overrides=config_overrides,
    )
    return update_static_config(
        config,
        pixel_size_km=float(backgrounds["native_pixel_km"]) / config["downsample_factor"],
        limb_direction_deg=float(backgrounds["limb_direction_deg"]),
    )


def generate_from_h5_background(
    backgrounds: Mapping[str, Any],
    *,
    seed: int,
    config_overrides: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StaticFilamentResult:
    """Generate one static filament directly on HDF5 sequence frame zero.

    ``backgrounds`` must be the plain dictionary returned by
    :func:`load_h5_background_sequence`. The observing geometry and detector
    scale are taken from that dictionary, so the returned image is already the
    exact initial condition for a dynamics run. Source HDF5 arrays are read but
    never modified.
    """
    required = {
        "frames",
        "native_shape",
        "native_pixel_km",
        "disk_mu",
        "limb_direction_deg",
        "metadata",
    }
    missing = sorted(required - set(backgrounds))
    if missing:
        raise ValueError(f"HDF5 background selection is missing fields: {missing}")

    frames = np.asarray(backgrounds["frames"])
    native_shape = tuple(int(value) for value in backgrounds["native_shape"])
    if frames.ndim != 3 or frames.shape[0] < 1 or frames.shape[1:] != native_shape:
        raise ValueError(
            "HDF5 background frames must have shape (time, rows, columns) matching "
            f"native_shape; received frames={frames.shape}, native_shape={native_shape}"
        )
    frame_zero = np.asarray(frames[0], dtype=float)
    support = np.isfinite(frame_zero) & (frame_zero > 0.0)
    if not support.all():
        raise ValueError("the selected HDF5 crop must contain finite strictly positive values")

    config = make_h5_static_config(
        backgrounds,
        seed=seed,
        config_overrides=config_overrides,
    )
    result = generate_static_filament(
        config,
        native_background=frame_zero,
        native_background_info=backgrounds["metadata"],
        progress_callback=progress_callback,
    )
    result["arrays"]["support"] = support.astype(np.uint8)
    result["metadata"]["synthesis_operation"] = "aligned_h5_frame_insertion"
    result["metadata"]["h5_frame_index"] = int(
        np.asarray(backgrounds.get("frame_indices", [0]))[0]
    )
    result["metadata"]["h5_crop_xyxy_px"] = list(backgrounds.get("crop_xyxy_px", ()))
    return result


def _mean_realized_abs_pitch_deg(
    spine: Spine,
    threads: list[Thread],
) -> float | None:
    """Return the arclength-weighted chord angle relative to the spine."""
    weighted_angle = 0.0
    total_length = 0.0
    for thread in threads:
        dx = np.diff(thread["x"])
        dy = np.diff(thread["y"])
        segment_length = np.hypot(dx, dy)
        valid = segment_length > 0.0
        if not np.any(valid):
            continue
        anchor_s = float(thread["spine_anchor_km"])
        if not np.isfinite(anchor_s):
            if not thread["s_spine"].size:
                continue
            anchor_s = float(thread["s_spine"][len(thread["s_spine"]) // 2])
        _, _, anchor_tx, anchor_ty, _, _ = interpolate_spine(
            spine,
            np.asarray([anchor_s], dtype=float),
        )
        tx = float(anchor_tx[0])
        ty = float(anchor_ty[0])
        direction_x = np.divide(
            dx,
            segment_length,
            out=np.zeros_like(dx),
            where=valid,
        )
        direction_y = np.divide(
            dy,
            segment_length,
            out=np.zeros_like(dy),
            where=valid,
        )
        dot = direction_x * tx + direction_y * ty
        cross = direction_x * ty - direction_y * tx
        angle = np.rad2deg(np.arctan2(np.abs(cross), dot))
        weighted_angle += float(np.sum(angle[valid] * segment_length[valid]))
        total_length += float(np.sum(segment_length[valid]))
    return weighted_angle / total_length if total_length > 0.0 else None


def _effective_source_fraction(threads: list[Thread], fallback: float) -> float:
    """Interpolate the separately published $S/I_{\rm bgr}$ height values."""
    if not threads:
        return float(fallback)
    mean_dip_bottom_height_km = float(np.mean([thread["height_km"] for thread in threads]))
    return float(
        np.interp(
            mean_dip_bottom_height_km,
            HEINZEL_SOURCE_HEIGHT_KM,
            HEINZEL_SOURCE_FRACTION,
        )
    )


def _aggregate_opacity_saturation(threads: list[Thread]) -> dict[str, Any]:
    """Aggregate opacity-table clipping counts across all thread points."""
    names = (
        "height_low",
        "height_high",
        "temperature_low",
        "temperature_high",
        "pressure_low",
        "pressure_high",
    )
    point_counts = {
        name: int(
            sum(thread.get("opacity_table_saturation", {}).get(name, 0) for thread in threads)
        )
        for name in names
    }
    thread_counts = {
        name: int(
            sum(thread.get("opacity_table_saturation", {}).get(name, 0) > 0 for thread in threads)
        )
        for name in names
    }
    return {
        "table_height_domain_km": _OPACITY_TABLE_METADATA["height_domain_km"],
        "table_temperature_domain_K": _OPACITY_TABLE_METADATA["temperature_domain_K"],
        "table_pressure_domain_dyn_cm2": _OPACITY_TABLE_METADATA[
            "pressure_domain_dyn_cm2"
        ],
        "saturated_point_counts": point_counts,
        "affected_thread_counts": thread_counts,
    }


def _aggregate_opacity_diagnostics(threads: list[Thread]) -> dict[str, Any]:
    """Aggregate temperature exclusions and useful opacity ranges."""
    diagnostics = [dict(thread.get("opacity_diagnostics", {})) for thread in threads]

    def finite_values(name: str) -> list[float]:
        return [
            float(value)
            for diagnostic in diagnostics
            if (value := diagnostic.get(name)) is not None and np.isfinite(value)
        ]

    coefficient_minima = finite_values("absorption_coefficient_min_positive_cm1")
    coefficient_maxima = finite_values("absorption_coefficient_max_cm1")
    local_tau_minima = finite_values("post_taper_local_tau_min_positive")
    local_tau_maxima = finite_values("post_taper_local_tau_max")
    return {
        "table_temperature_domain_K": _OPACITY_TABLE_METADATA["temperature_domain_K"],
        "loaded_sample_count": int(
            sum(diagnostic.get("loaded_sample_count", 0) for diagnostic in diagnostics)
        ),
        "temperature_supported_loaded_sample_count": int(
            sum(
                diagnostic.get("table_temperature_supported_loaded_sample_count", 0)
                for diagnostic in diagnostics
            )
        ),
        "temperature_excluded_loaded_sample_count": int(
            sum(
                diagnostic.get("table_temperature_excluded_loaded_sample_count", 0)
                for diagnostic in diagnostics
            )
        ),
        "entirely_zero_opacity_thread_count": int(
            sum(bool(diagnostic.get("entirely_zero_opacity")) for diagnostic in diagnostics)
        ),
        "entirely_zero_after_foot_taper_thread_count": int(
            sum(
                bool(diagnostic.get("entirely_zero_after_foot_taper"))
                for diagnostic in diagnostics
            )
        ),
        "absorption_coefficient_positive_range_cm1": [
            min(coefficient_minima) if coefficient_minima else None,
            max(coefficient_maxima) if coefficient_maxima else None,
        ],
        "post_taper_local_tau_positive_range": [
            min(local_tau_minima) if local_tau_minima else None,
            max(local_tau_maxima) if local_tau_maxima else None,
        ],
    }
def _render_static_state(
    config: Mapping[str, Any],
    spine: Spine,
    threads: list[Thread],
    placement: Mapping[str, Any],
    *,
    native_background: np.ndarray,
    native_background_info: Mapping[str, Any] | None,
    raster_progress_callback: Callable[[int, int], None] | None = None,
    precomputed_tau_map: np.ndarray | None = None,
) -> StaticFilamentResult:
    """Render one already-resolved physical state on a native GONG background."""
    resolved_config = validate_static_config(config)
    length_km = spine_total_length(spine)
    source_fraction = _effective_source_fraction(
        threads,
        resolved_config["source_fraction"],
    )

    expected_tau_shape = (resolved_config["ny"], resolved_config["nx"])
    if precomputed_tau_map is None:
        tau_map = rasterize_tau_map(
            resolved_config,
            threads,
            progress_callback=raster_progress_callback,
        )
    else:
        tau_map = np.asarray(precomputed_tau_map, dtype=float)
        if tau_map.shape != expected_tau_shape:
            raise ValueError(
                "precomputed_tau_map shape must match the internal grid; "
                f"expected {expected_tau_shape}, received {tau_map.shape}"
            )
        if not np.isfinite(tau_map).all() or np.any(tau_map < 0.0):
            raise ValueError("precomputed_tau_map must contain finite non-negative values")
    expected_native_shape = (
        resolved_config["ny"] // resolved_config["downsample_factor"],
        resolved_config["nx"] // resolved_config["downsample_factor"],
    )
    background_native = np.asarray(native_background, dtype=float)
    if background_native.shape != expected_native_shape:
        raise ValueError(
            "native_background shape must match the final detector grid; "
            f"expected {expected_native_shape}, received {background_native.shape}"
        )
    if not np.isfinite(background_native).all() or np.any(background_native < 0.0):
        raise ValueError("native_background must contain finite non-negative values")

    real_background = dict(native_background_info or {})
    real_background["provided_by_caller"] = True
    line_transmission_native = degrade_transmission(tau_map, resolved_config)
    transmission_native = dilute_transmission(
        line_transmission_native,
        resolved_config["gong_bandpass_line_fraction"],
    )
    source_level = float(np.median(background_native))
    degraded = compose_transmission(
        background_native,
        transmission_native,
        source_fraction,
        source_level,
    )
    highres_intensity = render_halpha_absorption(
        np.ones_like(tau_map),
        tau_map,
        source_fraction,
        source_level=1.0,
        bandpass_line_fraction=resolved_config["gong_bandpass_line_fraction"],
    )

    soft_mask_highres, hard_mask_highres = make_masks(
        tau_map,
        resolved_config["mask_tau_threshold"],
    )
    tau_native = -np.log(np.maximum(line_transmission_native, 1.0e-12))
    soft_mask = 1.0 - line_transmission_native
    observable_soft_mask = 1.0 - transmission_native
    hard_mask = tau_native > resolved_config["mask_tau_threshold"]

    arrays = {
        "highres_intensity": highres_intensity,
        "degraded_intensity": degraded,
        "tau_map": tau_map,
        "filament_mask": hard_mask.astype(np.uint8),
        "soft_mask": soft_mask,
        "observable_soft_mask": observable_soft_mask,
        "filament_mask_highres": hard_mask_highres.astype(np.uint8),
        "soft_mask_highres": soft_mask_highres,
        "background": background_native,
    }
    tau_in_mask = tau_map[hard_mask_highres] if hard_mask_highres.any() else np.zeros(1)
    metadata: dict[str, Any] = {
        "seed": resolved_config["seed"],
        "n_threads_generated": len(threads),
        "n_body_threads": len(threads),
        "thread_pitch_target_mean_abs_deg": resolved_config["thread_pitch_mean_deg"],
        "thread_pitch_sample_mean_abs_deg": (
            float(np.mean([abs(thread["pitch_deg"]) for thread in threads]))
            if threads
            else None
        ),
        "thread_pitch_realized_mean_abs_deg": _mean_realized_abs_pitch_deg(
            spine,
            threads,
        ),
        "chirality": resolved_config["chirality"],
        "spine_model": "measured_library",
        "spine_source": spine["source"],
        "spine_library_index": spine["library_index"],
        "spine_source_true_length_km": spine["source_true_length_km"],
        "spine_source_epoch": spine["source_epoch"],
        "background_source": "real_gong_crop",
        "used_real_background": True,
        "real_background": real_background,
        "disk_mu": resolved_config["disk_mu"],
        "spine_length_mm": float(length_km / 1_000.0),
        "spine_width_mm": float(resolved_config["spine_width_km"] / 1_000.0),
        "fov_mm": float(resolved_config["nx"] * resolved_config["pixel_size_km"] / 1_000.0),
        "degraded_pixel_km": float(
            resolved_config["pixel_size_km"] * resolved_config["downsample_factor"]
        ),
        "final_observation_array": "degraded_intensity",
        "oversampled_diagnostic_array": "highres_intensity",
        "thread_cross_section_model": "pixel_integrated_gaussian_sigma_radius_over_2",
        "thread_plan_view_model": "straight_anchor_local_shear",
        "dip_geometry_model": "luna_constant_curvature_vertical_plane",
        "dip_parameterization": placement["dip_parameterization"],
        "opacity_model": "heinzel2015_promweaver_calibrated_extension_1_to_100Mm",
        "opacity_table": deepcopy(_OPACITY_TABLE_METADATA),
        "source_function_model": "heinzel2015_published_10_20_30Mm_clamped",
        "source_function_height_domain_km": [
            float(HEINZEL_SOURCE_HEIGHT_KM[0]),
            float(HEINZEL_SOURCE_HEIGHT_KM[-1]),
        ],
        "source_height_statistic": "mean_dip_bottom_height_km",
        "source_fraction_configured": resolved_config["source_fraction"],
        "source_fraction_effective": source_fraction,
        "gong_bandpass_line_fraction": resolved_config["gong_bandpass_line_fraction"],
        "filament_psf_sigma_internal_px": resolved_config["psf_sigma_px"],
        "filament_psf_sigma_gong_px": (
            resolved_config["psf_sigma_px"] / resolved_config["downsample_factor"]
        ),
        "aspect_ratio": float(length_km / resolved_config["spine_width_km"]),
        "tau_max": float(tau_map.max()),
        "tau_p95_in_mask": float(np.percentile(tau_in_mask, 95)),
        "pct_mask_tau_gt3": float(100.0 * (tau_in_mask > 3.0).mean()),
        "tau_mean_in_mask": float(tau_in_mask.mean()),
        "mask_area_fraction": float(hard_mask.mean()),
        "source_level": source_level,
        "highres_is_filament_only": True,
        "intensity_min": float(degraded.min()),
        "intensity_max": float(degraded.max()),
        "spine_total_length_km": length_km,
        "mean_dip_depth_km": float(np.mean([thread["dip_depth_km"] for thread in threads])),
        "median_dip_depth_km": float(np.median([thread["dip_depth_km"] for thread in threads])),
        "mean_dip_radius_km": float(np.mean([thread["dip_radius_km"] for thread in threads])),
        "median_dip_radius_km": float(np.median([thread["dip_radius_km"] for thread in threads])),
        "mean_tau0": float(np.mean([thread["tau0"] for thread in threads])),
        "thread_placement": deepcopy(dict(placement)),
        "opacity_table_saturation": _aggregate_opacity_saturation(threads),
        "opacity_diagnostics": _aggregate_opacity_diagnostics(threads),
        "column_mass_loading": {
            "model": "capacity_limited_hydrostatic",
            "requested_g_cm2": float(resolved_config["column_mass_g_cm2"]),
            "realized_min_g_cm2": float(min(t["realized_column_mass_g_cm2"] for t in threads)),
            "realized_median_g_cm2": float(
                np.median([t["realized_column_mass_g_cm2"] for t in threads])
            ),
            "realized_max_g_cm2": float(max(t["realized_column_mass_g_cm2"] for t in threads)),
            "capacity_limited_thread_count": sum(
                t["column_mass_capacity_limited"] for t in threads
            ),
            "converged_thread_count": sum(t["column_mass_converged"] for t in threads),
            "lower_resolution_limited_thread_count": sum(
                t["column_mass_lower_resolution_limited"] for t in threads
            ),
            "numerical_failure_thread_count": sum(
                t["column_mass_numerical_failure"] for t in threads
            ),
            "maximum_absolute_relative_residual": float(
                max(abs(t["column_mass_relative_residual"]) for t in threads)
            ),
            "maximum_converged_absolute_relative_residual": float(
                max(
                    (
                        abs(t["column_mass_relative_residual"])
                        for t in threads
                        if t["column_mass_converged"]
                    ),
                    default=0.0,
                )
            ),
            "refinement_levels_range": [
                int(min(t["column_mass_refinement_levels"] for t in threads)),
                int(max(t["column_mass_refinement_levels"] for t in threads)),
            ],
            "integration_points_range": [
                int(min(t["column_mass_integration_points"] for t in threads)),
                int(max(t["column_mass_integration_points"] for t in threads)),
            ],
        },
    }
    return {
        "config": resolved_config,
        "spine": spine,
        "threads": threads,
        "arrays": arrays,
        "metadata": metadata,
    }


def generate_static_filament(
    config: Mapping[str, Any],
    *,
    native_background: np.ndarray,
    native_background_info: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StaticFilamentResult:
    """Generate one physical filament on an untouched real GONG crop.

    The returned value contains only plain dictionaries, NumPy arrays, and
    scalar metadata. The supplied background is never blurred or noised; the
    synthetic transmission alone receives the GONG observation operator.
    """
    geometry = generate_static_geometry(config, progress_callback=progress_callback)
    plasma = assign_static_plasma(geometry, progress_callback=progress_callback)
    started = monotonic()
    _progress(
        progress_callback,
        "render",
        started,
        "Rasterizing optical depth and applying the observation operator.",
        completed=0,
        total=1,
    )
    result = render_static_state(
        plasma,
        native_background=native_background,
        native_background_info=native_background_info,
        progress_callback=progress_callback,
    )
    _progress(
        progress_callback,
        "render",
        started,
        "Static image composition complete.",
        completed=1,
        total=1,
    )
    return result


def generate_static_geometry(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate an unmodified geometry state suitable for RAM reuse."""
    resolved_config = validate_static_config(config)
    rng = np.random.default_rng(resolved_config["seed"])
    started = monotonic()
    _progress(
        progress_callback,
        "spine",
        started,
        "Selecting and orienting a measured spine.",
        completed=0,
        total=1,
    )
    spine = make_spine(resolved_config, rng)
    _progress(
        progress_callback,
        "spine",
        started,
        "Measured spine selected and oriented.",
        completed=1,
        total=1,
    )
    length_km = spine_total_length(spine)
    realized_count = max(
        int(round(resolved_config["thread_density_per_mm"] * length_km / 1_000.0)),
        1,
    )
    realized_count = min(realized_count, resolved_config["thread_count_cap"])
    resolved_config = update_static_config(resolved_config, n_threads=realized_count)
    started = monotonic()
    _progress(
        progress_callback,
        "geometry",
        started,
        "Placing thread centerlines and checking separation.",
        completed=0,
        total=realized_count,
    )
    threads, placement = make_threads(
        resolved_config,
        spine,
        rng,
        progress_callback=lambda completed, total, attempts, relaxed: _progress(
            progress_callback,
            "geometry",
            started,
            (
                f"Placed {completed}/{total} thread candidates; "
                f"{attempts} attempts, {relaxed} relaxed placements."
            ),
            completed=completed,
            total=total,
            placement_attempts=attempts,
            relaxed_placements=relaxed,
        ),
    )
    _progress(
        progress_callback,
        "geometry",
        started,
        "Thread centerline placement complete.",
        completed=len(threads),
        total=realized_count,
        placement_attempts=placement.get("placement_attempts"),
        relaxed_placements=placement.get("relaxed_placements"),
        sampled_point_count=sum(int(thread["s"].size) for thread in threads),
    )
    return {
        "config": resolved_config,
        "spine": spine,
        "threads": threads,
        "placement": placement,
        "spine_length_km": length_km,
        "rng_state": deepcopy(rng.bit_generator.state),
    }


def assign_static_plasma(
    geometry: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Copy reusable geometry and assign plasma/opacity to the copy."""
    config = deepcopy(geometry["config"])
    spine = deepcopy(geometry["spine"])
    threads = deepcopy(geometry["threads"])
    started = monotonic()
    _progress(
        progress_callback,
        "plasma",
        started,
        "Solving plasma loading and assigning H-alpha opacity.",
        completed=0,
        total=len(threads),
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(geometry["rng_state"])
    assign_plasma(
        config,
        threads,
        rng,
        spine_length=float(geometry["spine_length_km"]),
        progress_callback=lambda completed, total: _progress(
            progress_callback,
            "plasma",
            started,
            f"Solved plasma loading and opacity for {completed}/{total} threads.",
            completed=completed,
            total=total,
        ),
    )
    _progress(
        progress_callback,
        "plasma",
        started,
        "Plasma loading and opacity assignment complete.",
        completed=len(threads),
        total=len(threads),
    )
    return {
        "config": config,
        "spine": spine,
        "threads": threads,
        "placement": deepcopy(geometry["placement"]),
        "spine_length_km": float(geometry["spine_length_km"]),
        "rng_state": deepcopy(geometry["rng_state"]),
    }


def render_static_state(
    plasma: Mapping[str, Any],
    *,
    native_background: np.ndarray,
    native_background_info: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    precomputed_tau_map: np.ndarray | None = None,
) -> StaticFilamentResult:
    """Render one reusable plasma state over a selected native background."""
    started = monotonic()
    return _render_static_state(
        plasma["config"],
        plasma["spine"],
        plasma["threads"],
        plasma["placement"],
        native_background=native_background,
        native_background_info=native_background_info,
        raster_progress_callback=lambda completed, total: _progress(
            progress_callback,
            "render",
            started,
            f"Rasterized optical depth for {completed}/{total} threads.",
            completed=completed,
            total=total,
        ),
        precomputed_tau_map=precomputed_tau_map,
    )
