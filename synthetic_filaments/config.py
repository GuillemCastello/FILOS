"""Validated plain-dictionary configuration for the static forward model.

The static model has one executable scientific profile: an unscaled measured
spine populated with physical circular-dip threads and inserted into a real
GONG quiet-Sun crop. Configuration is ordinary data. Construction,
validation, and updates are explicit functions.
"""

from __future__ import annotations

from math import pi
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from .opacity_table import load_heinzel_opacity_table

SOLAR_RADIUS_KM = 695_508.0
GONG_DISK_RADIUS_PX = 900.0
GONG_PIXEL_KM = SOLAR_RADIUS_KM / GONG_DISK_RADIUS_PX
GONG_OVERSAMPLE = 2
DEFAULT_SOURCE_FRACTION = 3.22 / 6.93

DIP_PARAMETERIZATION_CURVATURE_RADIUS = "curvature_radius"
DEFAULT_DIP_CURVATURE_RADIUS_PARAMETERS = {
    "dip_curvature_radius_min_km": 50_000.0,
    "dip_curvature_radius_max_km": 250_000.0,
    "dip_curvature_radius_median_km": 100_000.0,
    "dip_curvature_radius_sigma_ln": 0.25,
}
StaticConfig = dict[str, Any]


_CANONICAL_VALUES: dict[str, Any] = {
    # Measured-spine population.
    "spine_library_entry_index": None,
    "spine_library_length_bounds_km": (60_000.0, 500_000.0),
    "spine_width_km": 7_500.0,
    "thread_density_per_mm": 50.0,
    "thread_count_cap": 50_000,
    "n_threads": 0,  # resolved from measured arclength by the generator
    # Circular dipped-thread geometry.
    "thread_length_min_km": 20_000.0,
    "thread_length_max_km": 100_000.0,
    "thread_radius_min_km": 75.0,
    "thread_radius_max_km": 300.0,
    "thread_pitch_mean_deg": 25.0,
    "thread_pitch_std_deg": 6.0,
    "height_mean_km": 20_000.0,
    "height_std_km": 2_000.0,
    **DEFAULT_DIP_CURVATURE_RADIUS_PARAMETERS,
    "thread_separation_radii": 2.5,
    "thread_point_spacing_km": 150.0,
    "endpoint_radius_floor": 0.4,
    "width_modulation_amplitude": 0.35,
    "width_modulation_correlation": 0.25,
    # WPFS/PCTR plasma and H-alpha opacity.
    "temp_center_K": 7_000.0,
    "temp_tr_K": 100_000.0,
    "pctr_gamma": 2.0,
    "ionization_center": 0.3,
    "transition_pressure_dyn_cm2": 0.015,
    "column_mass_g_cm2": 1.0e-4,
    "microturbulent_velocity_kms": 5.0,
    "halpha_wavelength_angstrom": 6562.8,
    "foot_taper_fraction": 0.10,
    # Source function, passband, masks, and instrument transfer.
    "source_fraction": DEFAULT_SOURCE_FRACTION,
    "gong_bandpass_line_fraction": 0.65,
    "mask_tau_threshold": 0.10,
    "psf_sigma_px": 4.0,
    "downsample_factor": GONG_OVERSAMPLE,
    "pixel_size_km": GONG_PIXEL_KM / GONG_OVERSAMPLE,
    "limb_direction_deg": 0.0,
}


def image_width_km(config: Mapping[str, Any]) -> float:
    """Return the internal field-of-view width in kilometres."""
    return float(config["nx"] * config["pixel_size_km"])


def image_height_km(config: Mapping[str, Any]) -> float:
    """Return the internal field-of-view height in kilometres."""
    return float(config["ny"] * config["pixel_size_km"])


def opacity_temperature_domain_K() -> tuple[float, float]:
    """Return the installed opacity table's supported temperature domain."""
    temperatures = np.asarray(load_heinzel_opacity_table()["temperature_K"], dtype=float)
    return float(temperatures[0]), float(temperatures[-1])


def make_static_config(
    seed: int,
    native_shape: tuple[int, int],
    disk_mu: float,
    orientation_deg: float,
    chirality: int,
    overrides: Mapping[str, Any] | None = None,
) -> StaticConfig:
    """Build the canonical static configuration as a validated plain dict.

    ``native_shape`` is the final GONG detector shape ``(rows, columns)``.
    The internal grid is exactly twice this shape in each direction.
    """
    try:
        native_ny, native_nx = native_shape
    except (TypeError, ValueError) as error:
        raise ValueError("native_shape must be a two-element (rows, columns) tuple") from error

    config = dict(_CANONICAL_VALUES)
    config.update(
        {
            "seed": seed,
            "nx": native_nx * GONG_OVERSAMPLE,
            "ny": native_ny * GONG_OVERSAMPLE,
            "disk_mu": disk_mu,
            "orientation_deg": orientation_deg,
            "chirality": chirality,
        }
    )
    requested = dict(overrides or {})
    unknown = sorted(set(requested) - set(config))
    if unknown:
        raise ValueError(f"unknown static-config overrides: {unknown}")
    reserved = {
        "seed",
        "nx",
        "ny",
        "pixel_size_km",
        "downsample_factor",
        "disk_mu",
        "orientation_deg",
        "chirality",
        "n_threads",
    }
    conflicts = sorted(set(requested) & reserved)
    if conflicts:
        raise ValueError(
            f"grid, context, seed, and realized-count overrides are reserved: {conflicts}"
        )
    config.update(requested)
    return validate_static_config(config)


def update_static_config(config: Mapping[str, Any], **updates: Any) -> StaticConfig:
    """Return a validated copy of ``config`` with explicit field updates."""
    unknown = sorted(set(updates) - set(config))
    if unknown:
        raise ValueError(f"unknown static-config fields: {unknown}")
    updated = dict(config)
    updated.update(updates)
    return validate_static_config(updated)


def _require_integer(config: Mapping[str, Any], name: str, minimum: int) -> None:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; received {value!r}")


def _require_finite(config: Mapping[str, Any], name: str) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number; received {value!r}")
    return float(value)


def _require_positive(config: Mapping[str, Any], name: str, *, allow_zero: bool = False) -> None:
    value = _require_finite(config, name)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not valid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}; received {value!r}")


def validate_static_config(config: Mapping[str, Any]) -> StaticConfig:
    """Validate every static-model control and return an independent dict."""
    required = set(_CANONICAL_VALUES) | {
        "seed",
        "nx",
        "ny",
        "disk_mu",
        "orientation_deg",
        "chirality",
    }
    missing = sorted(required - set(config))
    unknown = sorted(set(config) - required)
    if missing:
        raise ValueError(f"static config is missing fields: {missing}")
    if unknown:
        raise ValueError(f"static config contains unsupported fields: {unknown}")

    validated = dict(config)
    _require_integer(validated, "seed", 0)
    _require_integer(validated, "nx", 8)
    _require_integer(validated, "ny", 8)
    _require_integer(validated, "n_threads", 0)
    _require_integer(validated, "thread_count_cap", 1)
    _require_integer(validated, "downsample_factor", 1)

    for name in (
        "pixel_size_km",
        "spine_width_km",
        "thread_density_per_mm",
        "thread_length_min_km",
        "thread_length_max_km",
        "thread_radius_min_km",
        "thread_radius_max_km",
        "dip_curvature_radius_min_km",
        "dip_curvature_radius_max_km",
        "dip_curvature_radius_median_km",
        "thread_point_spacing_km",
        "temp_center_K",
        "temp_tr_K",
        "pctr_gamma",
        "transition_pressure_dyn_cm2",
        "column_mass_g_cm2",
        "halpha_wavelength_angstrom",
        "foot_taper_fraction",
        "width_modulation_correlation",
    ):
        _require_positive(validated, name)
    for name in (
        "thread_pitch_std_deg",
        "height_std_km",
        "dip_curvature_radius_sigma_ln",
        "thread_separation_radii",
        "microturbulent_velocity_kms",
        "psf_sigma_px",
        "width_modulation_amplitude",
    ):
        _require_positive(validated, name, allow_zero=True)
    for name in (
        "orientation_deg",
        "thread_pitch_mean_deg",
        "height_mean_km",
        "source_fraction",
        "gong_bandpass_line_fraction",
        "mask_tau_threshold",
        "limb_direction_deg",
        "disk_mu",
        "ionization_center",
        "endpoint_radius_floor",
    ):
        _require_finite(validated, name)

    bounds = validated["spine_library_length_bounds_km"]
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("spine_library_length_bounds_km must contain (minimum, maximum)")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite([lower, upper]).all() or not 0.0 < lower <= upper:
        raise ValueError(
            "spine_library_length_bounds_km must be finite, positive, and ordered; "
            f"received {bounds!r}"
        )
    validated["spine_library_length_bounds_km"] = (lower, upper)

    entry_index = validated["spine_library_entry_index"]
    if entry_index is not None and (
        isinstance(entry_index, bool) or not isinstance(entry_index, Integral) or entry_index < 0
    ):
        raise ValueError(
            f"spine_library_entry_index must be None or an integer >= 0; received {entry_index!r}"
        )
    if validated["chirality"] not in {-1, 1} or isinstance(validated["chirality"], bool):
        raise ValueError(f"chirality must be -1 or 1; received {validated['chirality']!r}")
    if not 0.05 <= validated["disk_mu"] <= 1.0:
        raise ValueError(f"disk_mu must lie in [0.05, 1]; received {validated['disk_mu']!r}")
    if not 0.0 <= validated["source_fraction"] <= 1.0:
        raise ValueError("source_fraction must lie in [0, 1]")
    if not 0.0 <= validated["gong_bandpass_line_fraction"] <= 1.0:
        raise ValueError("gong_bandpass_line_fraction must lie in [0, 1]")
    if not 0.0 <= validated["ionization_center"] <= 1.0:
        raise ValueError("ionization_center must lie in [0, 1]")
    if not 0.0 <= validated["width_modulation_amplitude"] <= 1.0:
        raise ValueError("width_modulation_amplitude must lie in [0, 1]")
    if not 0.0 < validated["endpoint_radius_floor"] <= 1.0:
        raise ValueError("endpoint_radius_floor must lie in (0, 1]")
    if not 1_000.0 <= validated["height_mean_km"] <= 100_000.0:
        raise ValueError("height_mean_km must lie in [1,000, 100,000] km")
    if not 0.0 < validated["foot_taper_fraction"] <= 0.5:
        raise ValueError("foot_taper_fraction must lie in (0, 0.5]")
    if not validated["thread_radius_min_km"] <= validated["thread_radius_max_km"]:
        raise ValueError("thread_radius_min_km must be <= thread_radius_max_km")
    if not validated["thread_length_min_km"] <= validated["thread_length_max_km"]:
        raise ValueError("thread_length_min_km must be <= thread_length_max_km")
    if not (
        validated["dip_curvature_radius_min_km"]
        <= validated["dip_curvature_radius_median_km"]
        <= validated["dip_curvature_radius_max_km"]
    ):
        raise ValueError(
            "dip_curvature_radius_median_km must lie within the configured radius bounds"
        )
    minimum_compatible_radius_km = validated["thread_length_min_km"] / pi
    if validated["dip_curvature_radius_min_km"] < minimum_compatible_radius_km:
        raise ValueError(
            "dip_curvature_radius_min_km must be >= thread_length_min_km / pi "
            "so every sampled radius supports the minimum circular-arc length"
        )
    temperature_min_K, temperature_max_K = opacity_temperature_domain_K()
    if not temperature_min_K <= validated["temp_center_K"] <= temperature_max_K:
        raise ValueError(
            "temp_center_K must lie within the installed opacity-table domain "
            f"[{temperature_min_K:g}, {temperature_max_K:g}] K; "
            f"received {validated['temp_center_K']!r}"
        )
    if not validated["temp_tr_K"] > validated["temp_center_K"]:
        raise ValueError("temp_tr_K must be greater than temp_center_K")
    factor = validated["downsample_factor"]
    if factor >= min(validated["nx"], validated["ny"]):
        raise ValueError("downsample_factor must be smaller than nx and ny")
    if validated["nx"] % factor or validated["ny"] % factor:
        raise ValueError("downsample_factor must divide nx and ny exactly")
    return validated
