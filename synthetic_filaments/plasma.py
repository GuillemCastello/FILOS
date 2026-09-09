"""WPFS hydrostatic loading and fast H-alpha opacity for filament threads.

The physical branch loads each magnetic dip up to its column-mass capacity,
integrates hydrostatic pressure from the PCTR boundary, interpolates the
Heinzel, Gunár & Anzer (2015) ionization and level-population table, and
derives absolute line-center optical depth.
"""

from collections import deque
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from time import monotonic

import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import RegularGridInterpolator

from .config import StaticConfig
from .execution import (
    discard_process_pool,
    integer_setting,
    process_pool,
    process_spawn_available,
)
from .geometry import Thread, _smoothstep, effective_thread_radius
from .opacity_table import load_heinzel_opacity_table

# CGS constants
K_BOLTZ = 1.380649e-16  # erg / K
M_HYDROGEN = 1.6726e-24  # g
G_SUN = 2.74e4  # cm / s^2
HELIUM_ABUNDANCE = 0.1
LIGHT_SPEED = 2.99792458e10  # cm / s

# The production table preserves the 75 published Heinzel et al. (2015)
# values at 10, 20, and 30 Mm exactly, and supplies a calibrated Promweaver
# PRD/cone-boundary height response on a 1--100 Mm grid. Its factor is
# f = n_p^2 / n_2 in units of 1e16 cm^-3, consistent with the hydrogen-only
# definition used to calibrate the published table.
_OPACITY_TABLE = load_heinzel_opacity_table()
_TABLE_HEIGHT_KM = _OPACITY_TABLE["height_km"]
_TABLE_TEMPERATURE_K = _OPACITY_TABLE["temperature_K"]
_TABLE_PRESSURE_DYN_CM2 = _OPACITY_TABLE["pressure_dyn_cm2"]
_TABLE_IONIZATION = _OPACITY_TABLE["ionization"]
_TABLE_F_1E16_CM3 = _OPACITY_TABLE["f_1e16_cm3"]
_IONIZATION_INTERPOLATOR = RegularGridInterpolator(
    (_TABLE_HEIGHT_KM, _TABLE_TEMPERATURE_K, _TABLE_PRESSURE_DYN_CM2),
    _TABLE_IONIZATION,
    bounds_error=False,
)
_F_INTERPOLATOR = RegularGridInterpolator(
    (_TABLE_HEIGHT_KM, _TABLE_TEMPERATURE_K, _TABLE_PRESSURE_DYN_CM2),
    _TABLE_F_1E16_CM3,
    bounds_error=False,
)

MASS_INTEGRATION_INITIAL_INTERVALS = 16
MASS_INTEGRATION_MAX_INTERVALS = 8_192
MASS_INTEGRATION_RELATIVE_TOLERANCE = 2.0e-5
MASS_INTEGRATION_ABSOLUTE_TOLERANCE_G_CM2 = 1.0e-14
MASS_INTEGRATION_CONSECUTIVE_CONFIRMATIONS = 2
COLUMN_MASS_RELATIVE_TOLERANCE = 1.0e-4
COLUMN_MASS_MAX_BISECTION_ITERATIONS = 64


def _temperature_profile(config: StaticConfig, xi: np.ndarray) -> np.ndarray:
    """T(xi) along the field, xi in [0, 1] from dip center to edge."""
    return (
        config["temp_center_K"]
        + (config["temp_tr_K"] - config["temp_center_K"])
        * np.clip(xi, 0.0, 1.0) ** config["pctr_gamma"]
    )


def _ionization_degree(config: StaticConfig, temp: np.ndarray) -> np.ndarray:
    """i(T) after Gunar et al. (2013): i_c at the cool center, -> 1 in the PCTR."""
    frac = (config["temp_tr_K"] - temp) / (config["temp_tr_K"] - config["temp_center_K"])
    return 1.0 - (1.0 - config["ionization_center"]) * np.clip(frac, 0.0, 1.0) ** 2


def _cumulative_integral(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Cumulative Simpson integral with an explicit zero first element."""
    if len(values) < 2:
        return np.zeros_like(values)
    return np.asarray(
        cumulative_simpson(values, x=coordinate, initial=0.0),
        dtype=float,
    )


def _hydrostatic_pressure(
    config: StaticConfig,
    z_km: np.ndarray,
    s_km: np.ndarray,
    absolute_vertical_slope: np.ndarray,
    temperature_K: np.ndarray,
    mean_molecular_mass: np.ndarray,
    loaded: np.ndarray,
) -> np.ndarray:
    r"""Integrate $p=p_{\rm tr}\exp[\int \mu m_Hg|dz/ds|/(kT)\,ds]$."""
    pressure = np.full_like(
        temperature_K,
        config["transition_pressure_dyn_cm2"],
        dtype=float,
    )
    indices = np.flatnonzero(loaded)
    if indices.size == 0:
        return pressure
    center = int(indices[np.argmin(z_km[indices])])
    coefficient_per_vertical_km = (
        mean_molecular_mass * M_HYDROGEN * G_SUN / (K_BOLTZ * temperature_K) * 1.0e5
    )
    coefficient_per_arc_km = coefficient_per_vertical_km * absolute_vertical_slope

    left = np.arange(indices[0], center + 1)
    left_distance = s_km[left] - s_km[left[0]]
    left_integral = _cumulative_integral(
        coefficient_per_arc_km[left],
        left_distance,
    )
    pressure[left] = config["transition_pressure_dyn_cm2"] * np.exp(left_integral)

    right = np.arange(indices[-1], center - 1, -1)
    right_distance = s_km[right[0]] - s_km[right]
    right_integral = _cumulative_integral(
        coefficient_per_arc_km[right],
        right_distance,
    )
    right_pressure = config["transition_pressure_dyn_cm2"] * np.exp(right_integral)
    if center in left and center in right:
        # Gunár et al. use the lower of the two independently integrated
        # central pressures and reverse-integrate the opposite side. The
        # procedural dips are symmetric, so this is normally a no-op; the
        # minimum keeps small discretization asymmetries conservative.
        center_pressure = min(float(pressure[center]), float(right_pressure[-1]))
        if pressure[center] > 0.0:
            pressure[left] *= center_pressure / pressure[center]
        if right_pressure[-1] > 0.0:
            right_pressure *= center_pressure / right_pressure[-1]
    pressure[right] = right_pressure
    return pressure


def _physical_loaded_state(
    config: StaticConfig,
    thread: Thread,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    """Solve hydrostatic loading on a grid adapted to the loaded interval.

    The bisection bracket is the physical interval from an empty dip to the
    full dip. Each non-empty state includes both loading boundaries and the dip
    center, and doubles its local interval count until its integrated column
    mass satisfies the convergence tolerance on two consecutive refinements.
    The original rendering samples are merged only after convergence so they
    cannot disturb the nested integration sequence.
    """
    original_s = np.asarray(thread["s"], dtype=float)
    if (
        original_s.ndim != 1
        or original_s.size < 2
        or not np.isfinite(original_s).all()
        or np.any(np.diff(original_s) <= 0.0)
    ):
        raise ValueError("thread arclength must be finite and strictly increasing")
    center_s = 0.5 * float(original_s[-1])
    half_length = center_s
    if not np.isfinite(half_length) or half_length <= 0.0:
        raise ValueError("thread length must be finite and positive for plasma loading")

    def sampled_state(fraction: float, sample_s: np.ndarray) -> dict[str, object]:
        loaded_half_length = fraction * half_length
        if loaded_half_length <= 8.0 * np.spacing(center_s):
            raise RuntimeError(
                "column-mass loading is below floating-point arclength resolution; "
                f"fraction={fraction:.6e}, half-length={half_length:.6e} km"
            )
        left_boundary = center_s - loaded_half_length
        right_boundary = center_s + loaded_half_length
        xi = np.abs(sample_s - center_s) / loaded_half_length
        loaded = (sample_s >= left_boundary) & (sample_s <= right_boundary)
        temperature = _temperature_profile(config, np.clip(xi, 0.0, 1.0))
        initial_ionization = _ionization_degree(config, temperature)
        mean_molecular_mass = (1.0 + 4.0 * HELIUM_ABUNDANCE) / (
            1.0 + HELIUM_ABUNDANCE + initial_ionization
        )
        angle = sample_s / float(thread["dip_radius_km"]) - (
            float(thread["length_km"]) / (2.0 * float(thread["dip_radius_km"]))
        )
        z_km = (
            float(thread["height_km"])
            + 2.0 * float(thread["dip_radius_km"]) * np.sin(0.5 * angle) ** 2
        )
        pressure = _hydrostatic_pressure(
            config,
            z_km,
            sample_s,
            np.abs(np.sin(angle)),
            temperature,
            mean_molecular_mass,
            loaded,
        )
        density = mean_molecular_mass * M_HYDROGEN * pressure / (K_BOLTZ * temperature)
        return {
            "s": sample_s,
            "z": z_km,
            "temperature": temperature,
            "pressure": pressure,
            "mean_molecular_mass": mean_molecular_mass,
            "loaded": loaded,
            "density": density,
        }

    def integration_state(fraction: float, interval_count: int) -> dict[str, object]:
        loaded_half_length = fraction * half_length
        loaded_grid = np.linspace(
            center_s - loaded_half_length,
            center_s + loaded_half_length,
            interval_count + 1,
        )
        result = sampled_state(fraction, loaded_grid)
        column_mass = float(
            simpson(
                np.asarray(result["density"], dtype=float),
                x=loaded_grid * 1.0e5,
            )
        )
        if not np.isfinite(column_mass) or column_mass <= 0.0:
            raise RuntimeError(
                "hydrostatic column-mass integration produced a non-finite or "
                f"non-positive value at loaded fraction {fraction:.6e}"
            )
        result["column_mass"] = column_mass
        result["interval_count"] = interval_count
        return result

    def converged_state(fraction: float) -> tuple[dict[str, object], int, float]:
        previous: dict[str, object] | None = None
        refinement_levels = 0
        consecutive_confirmations = 0
        interval_count = MASS_INTEGRATION_INITIAL_INTERVALS
        change = float("inf")
        tolerance = float("nan")
        while interval_count <= MASS_INTEGRATION_MAX_INTERVALS:
            current = integration_state(fraction, interval_count)
            refinement_levels += 1
            if previous is not None:
                change = abs(float(current["column_mass"]) - float(previous["column_mass"]))
                tolerance = max(
                    MASS_INTEGRATION_ABSOLUTE_TOLERANCE_G_CM2,
                    MASS_INTEGRATION_RELATIVE_TOLERANCE
                    * abs(float(current["column_mass"])),
                )
                if change <= tolerance:
                    consecutive_confirmations += 1
                else:
                    consecutive_confirmations = 0
                if (
                    consecutive_confirmations
                    >= MASS_INTEGRATION_CONSECUTIVE_CONFIRMATIONS
                ):
                    return current, refinement_levels, change
            previous = current
            interval_count *= 2
        assert previous is not None
        raise RuntimeError(
            "column-mass integration did not converge within the adaptive-grid "
            f"budget: fraction={fraction:.6e}, intervals="
            f"{MASS_INTEGRATION_MAX_INTERVALS}, last mass="
            f"{float(previous['column_mass']):.6e} g cm^-2, last change="
            f"{change:.6e} g cm^-2, tolerance={tolerance:.6e} g cm^-2, "
            f"radius={float(thread['dip_radius_km']):.6e} km, "
            f"length={float(thread['length_km']):.6e} km"
        )

    requested_mass = float(config["column_mass_g_cm2"])
    lower_endpoint_mass = 0.0
    full, full_refinement_levels, full_change = converged_state(1.0)
    capacity = float(full["column_mass"])
    target_tolerance = COLUMN_MASS_RELATIVE_TOLERANCE * requested_mass
    bisection_iterations = 0
    if capacity <= requested_mass:
        fraction = 1.0
        selected = full
        selected_refinement_levels = full_refinement_levels
        selected_change = full_change
        status = "capacity_limited" if capacity < requested_mass - target_tolerance else "converged"
    else:
        lower = 0.0
        upper = 1.0
        selected = full
        selected_refinement_levels = full_refinement_levels
        selected_change = full_change
        status = "numerical_failure"
        for bisection_iterations in range(1, COLUMN_MASS_MAX_BISECTION_ITERATIONS + 1):
            middle = 0.5 * (lower + upper)
            candidate, refinement_levels, integration_change = converged_state(middle)
            candidate_mass = float(candidate["column_mass"])
            if abs(candidate_mass - requested_mass) <= target_tolerance:
                fraction = middle
                selected = candidate
                selected_refinement_levels = refinement_levels
                selected_change = integration_change
                status = "converged"
                break
            if candidate_mass < requested_mass:
                lower = middle
            else:
                upper = middle
            selected = candidate
            selected_refinement_levels = refinement_levels
            selected_change = integration_change
        else:
            raise RuntimeError(
                "column-mass bisection did not converge to the requested tolerance: "
                f"target={requested_mass:.6e} g cm^-2, last="
                f"{float(selected['column_mass']):.6e} g cm^-2, iterations="
                f"{COLUMN_MASS_MAX_BISECTION_ITERATIONS}"
            )

    integration_s = np.asarray(selected["s"], dtype=float)
    refined_s = np.unique(np.concatenate((original_s, integration_s, [center_s])))
    realized = sampled_state(fraction, refined_s)
    angle = refined_s / float(thread["dip_radius_km"]) - (
        float(thread["length_km"]) / (2.0 * float(thread["dip_radius_km"]))
    )
    along_chord_km = float(thread["dip_radius_km"]) * np.sin(angle)
    original_x = np.asarray(thread["x"], dtype=float)
    original_y = np.asarray(thread["y"], dtype=float)
    anchor_x = float(np.interp(center_s, original_s, original_x))
    anchor_y = float(np.interp(center_s, original_s, original_y))
    chord_x = float(original_x[-1] - original_x[0])
    chord_y = float(original_y[-1] - original_y[0])
    chord_length = float(np.hypot(chord_x, chord_y))
    if not np.isfinite(chord_length) or chord_length <= 0.0:
        raise ValueError("thread plan-view chord must have finite positive length")
    direction_x = chord_x / chord_length
    direction_y = chord_y / chord_length
    thread["x"] = anchor_x + along_chord_km * direction_x
    thread["y"] = anchor_y + along_chord_km * direction_y
    thread["s_spine"] = (
        float(thread["spine_anchor_km"])
        + along_chord_km * np.cos(np.deg2rad(float(thread["pitch_deg"])))
    )
    thread_xi = np.abs(refined_s - center_s) / half_length
    thread["radius_along_km"] = float(thread["radius_km"]) * (
        config["endpoint_radius_floor"]
        + (1.0 - config["endpoint_radius_floor"])
        * np.sqrt(np.maximum(1.0 - thread_xi**2, 0.0))
    )
    thread["s"] = refined_s
    thread["z"] = np.asarray(realized["z"], dtype=float)

    temperature = np.asarray(realized["temperature"], dtype=float)
    pressure = np.asarray(realized["pressure"], dtype=float)
    mean_molecular_mass = np.asarray(realized["mean_molecular_mass"], dtype=float)
    loaded = np.asarray(realized["loaded"], dtype=bool)
    column_mass = float(selected["column_mass"])
    residual = column_mass - requested_mass
    diagnostics = {
        "requested_column_mass_g_cm2": requested_mass,
        "realized_column_mass_g_cm2": column_mass,
        "column_mass_capacity_g_cm2": capacity,
        "column_mass_capacity_limited": status == "capacity_limited",
        "loaded_length_fraction": float(fraction),
        "loaded_length_km": float(fraction * thread["length_km"]),
        "column_mass_residual_g_cm2": float(residual),
        "column_mass_relative_residual": float(residual / requested_mass),
        "column_mass_status": status,
        "column_mass_converged": status == "converged",
        "column_mass_lower_resolution_limited": False,
        "column_mass_numerical_failure": False,
        "column_mass_target_tolerance_g_cm2": float(target_tolerance),
        "column_mass_lower_endpoint_g_cm2": lower_endpoint_mass,
        "column_mass_upper_endpoint_g_cm2": capacity,
        "column_mass_bisection_iterations": bisection_iterations,
        "column_mass_integration_intervals": int(selected["interval_count"]),
        "column_mass_integration_points": int(integration_s.size),
        "column_mass_refinement_levels": selected_refinement_levels,
        "column_mass_last_refinement_change_g_cm2": float(selected_change),
        "column_mass_integration_relative_tolerance": (MASS_INTEGRATION_RELATIVE_TOLERANCE),
    }
    return temperature, pressure, mean_molecular_mass, loaded, fraction, diagnostics


def _table_ionization_and_f(
    temperature_K: np.ndarray,
    pressure_dyn_cm2: np.ndarray,
    height_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Interpolate the opacity table and report every clipped query axis."""
    saturation = {
        "height_low": height_km < _TABLE_HEIGHT_KM[0],
        "height_high": height_km > _TABLE_HEIGHT_KM[-1],
        "temperature_low": temperature_K < _TABLE_TEMPERATURE_K[0],
        "temperature_high": temperature_K > _TABLE_TEMPERATURE_K[-1],
        "pressure_low": pressure_dyn_cm2 < _TABLE_PRESSURE_DYN_CM2[0],
        "pressure_high": pressure_dyn_cm2 > _TABLE_PRESSURE_DYN_CM2[-1],
    }
    query = np.column_stack(
        [
            np.clip(height_km, _TABLE_HEIGHT_KM[0], _TABLE_HEIGHT_KM[-1]),
            np.clip(
                temperature_K,
                _TABLE_TEMPERATURE_K[0],
                _TABLE_TEMPERATURE_K[-1],
            ),
            np.clip(
                pressure_dyn_cm2,
                _TABLE_PRESSURE_DYN_CM2[0],
                _TABLE_PRESSURE_DYN_CM2[-1],
            ),
        ]
    )
    ionization = np.asarray(_IONIZATION_INTERPOLATOR(query), dtype=float)
    factor = 1.0e16 * np.asarray(_F_INTERPOLATOR(query), dtype=float)
    return ionization, factor, saturation


def _physical_thread_opacity(
    config: StaticConfig,
    thread: Thread,
) -> tuple[float, np.ndarray, float, dict[str, int], dict[str, object], dict[str, object]]:
    """Return opacity, loaded depth, table clipping, and column-mass diagnostics."""
    temperature, pressure, mean_molecular_mass, loaded, loaded_fraction, loading = (
        _physical_loaded_state(
            config,
            thread,
        )
    )
    ionization, factor, saturation = _table_ionization_and_f(
        temperature,
        pressure,
        thread["z"],
    )
    electron_density = pressure / (
        K_BOLTZ * temperature * (1.0 + 1.1 / np.maximum(ionization, 1.0e-6))
    )
    level_two_density = electron_density**2 / factor
    wavelength_cm = config["halpha_wavelength_angstrom"] * 1.0e-8
    line_frequency = LIGHT_SPEED / wavelength_cm
    turbulent_speed = config["microturbulent_velocity_kms"] * 1.0e5
    doppler_width = (line_frequency / LIGHT_SPEED) * np.sqrt(
        2.0 * K_BOLTZ * temperature / M_HYDROGEN + turbulent_speed**2
    )
    line_profile_center = 1.0 / (np.sqrt(np.pi) * doppler_width)
    absorption_coefficient = 1.7e-2 * level_two_density * line_profile_center
    # Heinzel et al. tabulate i and f only over 6,000--14,000 K. Hotter PCTR
    # material lies outside the validated fast-synthesis domain and is
    # H-alpha-invisible for this approximation; do not fabricate opacity by
    # silently holding the 14,000 K table row constant up to 100,000 K.
    table_temperature_valid = (temperature >= _TABLE_TEMPERATURE_K[0]) & (
        temperature <= _TABLE_TEMPERATURE_K[-1]
    )
    absorption_coefficient = np.where(
        loaded & table_temperature_valid,
        absorption_coefficient,
        0.0,
    )

    center = int(np.argmin(thread["z"]))
    center_coefficient = float(absorption_coefficient[center])
    radius = effective_thread_radius(thread)
    saturation_counts = {name: int(np.count_nonzero(values)) for name, values in saturation.items()}
    positive = absorption_coefficient > 0.0
    opacity_diagnostics: dict[str, object] = {
        "table_temperature_domain_K": [
            float(_TABLE_TEMPERATURE_K[0]),
            float(_TABLE_TEMPERATURE_K[-1]),
        ],
        "table_temperature_excluded_loaded_sample_count": int(
            np.count_nonzero(loaded & ~table_temperature_valid)
        ),
        "table_temperature_supported_loaded_sample_count": int(
            np.count_nonzero(loaded & table_temperature_valid)
        ),
        "loaded_sample_count": int(np.count_nonzero(loaded)),
        "positive_opacity_sample_count": int(np.count_nonzero(positive)),
        "entirely_zero_opacity": not bool(np.any(positive)),
        "absorption_coefficient_min_positive_cm1": (
            float(np.min(absorption_coefficient[positive])) if np.any(positive) else None
        ),
        "absorption_coefficient_max_cm1": float(np.max(absorption_coefficient, initial=0.0)),
        "normalization_used_dip_center": center_coefficient > 0.0,
        "gong_bandpass_line_fraction": float(config["gong_bandpass_line_fraction"]),
    }
    half_angle = thread["length_km"] / (2.0 * thread["dip_radius_km"])
    loaded_depth_km = (
        2.0 * thread["dip_radius_km"] * np.sin(0.5 * loaded_fraction * half_angle) ** 2
    )
    if not np.any(positive):
        thread["temperature_K"] = temperature
        thread["pressure_dyn_cm2"] = pressure
        thread["mean_molecular_mass"] = mean_molecular_mass
        thread["loaded_mask"] = loaded
        return (
            0.0,
            np.zeros_like(thread["s"]),
            float(loaded_depth_km),
            saturation_counts,
            loading,
            opacity_diagnostics,
        )

    reference = center if center_coefficient > 0.0 else int(np.argmax(absorption_coefficient))
    reference_coefficient = float(absorption_coefficient[reference])
    reference_radius_km = float(radius[reference])
    central_tau = reference_coefficient * 2.0 * reference_radius_km * 1.0e5
    radius_ratio = radius / max(reference_radius_km, 1.0e-12)
    profile = absorption_coefficient / reference_coefficient * radius_ratio
    opacity_diagnostics["normalization_reference_index"] = reference
    opacity_diagnostics["tau_profile_min_positive"] = float(np.min(profile[profile > 0.0]))
    opacity_diagnostics["tau_profile_max"] = float(np.max(profile))
    thread["temperature_K"] = temperature
    thread["pressure_dyn_cm2"] = pressure
    thread["mean_molecular_mass"] = mean_molecular_mass
    thread["loaded_mask"] = loaded
    return (
        central_tau,
        profile,
        float(loaded_depth_km),
        saturation_counts,
        loading,
        opacity_diagnostics,
    )


def _foot_envelope(
    config: StaticConfig,
    s_spine: np.ndarray,
    spine_length: float,
) -> np.ndarray:
    """Smoothstep ramp so the filament body fades in/out at its feet."""
    ramp_len = max(config["foot_taper_fraction"] * spine_length, 1.0)
    t_start = np.clip(s_spine / ramp_len, 0.0, 1.0)
    t_end = np.clip((spine_length - s_spine) / ramp_len, 0.0, 1.0)
    return _smoothstep(t_start) * _smoothstep(t_end)


def _assign_thread_plasma(
    config: StaticConfig,
    thread: Thread,
    thread_index: int,
    spine_length: float | None,
) -> Thread:
    """Run the unchanged loading and opacity operations for one thread."""
    try:
        (
            central_tau,
            profile,
            loaded_depth,
            saturation,
            loading,
            opacity_diagnostics,
        ) = _physical_thread_opacity(config, thread)
    except RuntimeError as error:
        raise RuntimeError(
            f"plasma assignment failed for thread {thread_index}: {error}"
        ) from error
    thread["tau0"] = central_tau
    thread["tau_along"] = profile
    thread["fill_depth_km"] = loaded_depth
    thread["filled_depth_km"] = loaded_depth
    thread["fill_fraction"] = 1.0
    thread["realized_fill_fraction"] = float(np.mean(profile > 0.0))
    thread["opacity_table_saturation"] = saturation
    thread["opacity_diagnostics"] = opacity_diagnostics
    thread.update(loading)
    if spine_length is not None and thread["s_spine"].size:
        foot_envelope = _foot_envelope(
            config,
            thread["s_spine"],
            spine_length,
        )
        thread["tau_along"] = thread["tau_along"] * foot_envelope
        opacity_diagnostics["foot_envelope_min"] = float(np.min(foot_envelope))
        opacity_diagnostics["foot_envelope_max"] = float(np.max(foot_envelope))
    positive_profile = thread["tau_along"] > 0.0
    opacity_diagnostics["post_taper_positive_sample_count"] = int(
        np.count_nonzero(positive_profile)
    )
    opacity_diagnostics["post_taper_tau_profile_min_positive"] = (
        float(np.min(thread["tau_along"][positive_profile]))
        if np.any(positive_profile)
        else None
    )
    opacity_diagnostics["post_taper_tau_profile_max"] = float(
        np.max(thread["tau_along"], initial=0.0)
    )
    opacity_diagnostics["entirely_zero_after_foot_taper"] = not bool(
        np.any(positive_profile)
    )
    opacity_diagnostics["post_taper_local_tau_min_positive"] = (
        float(thread["tau0"] * np.min(thread["tau_along"][positive_profile]))
        if np.any(positive_profile)
        else None
    )
    opacity_diagnostics["post_taper_local_tau_max"] = float(
        thread["tau0"] * np.max(thread["tau_along"], initial=0.0)
    )
    return thread


def _assign_plasma_batch(
    config: StaticConfig,
    threads: list[Thread],
    start: int,
    spine_length: float | None,
) -> tuple[list[Thread], Exception | None, Thread | None]:
    """Return successes preceding any failure so parent mutation stays ordered."""
    completed = []
    for index, thread in enumerate(threads, start=start):
        try:
            completed.append(_assign_thread_plasma(config, thread, index, spine_length))
        except Exception as error:
            return completed, error, thread
    return completed, None, None


def assign_plasma(
    config: StaticConfig,
    threads: list[Thread],
    _rng: np.random.Generator,
    spine_length: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    workers: int | None = None,
) -> list[Thread]:
    """Assign opacity in input order, using persistent workers for large lists.

    ``workers`` and FILAMENT_PLASMA_WORKERS affect execution only. The default
    is eight workers; lists of at most one 32-thread batch stay serial. Explicit
    ``workers`` overrides also permit parallel benchmarking of small lists.
    Notebook/REPL namespaces without a main filename support imported workers;
    sessions with placeholder script paths such as ``<stdin>`` stay serial.
    The supplied RNG remains untouched, as in the serial implementation.
    """
    center_temperature = float(config["temp_center_K"])
    minimum_temperature = float(_TABLE_TEMPERATURE_K[0])
    maximum_temperature = float(_TABLE_TEMPERATURE_K[-1])
    if not minimum_temperature <= center_temperature <= maximum_temperature:
        raise ValueError(
            "temp_center_K lies outside the installed H-alpha opacity-table "
            f"domain [{minimum_temperature:.0f}, {maximum_temperature:.0f}] K; "
            f"received {center_temperature:.6g} K"
        )
    worker_count = (
        integer_setting("FILAMENT_PLASMA_WORKERS", 8) if workers is None else workers
    )
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count < 1:
        raise ValueError("workers must be a positive integer")
    batch_size = integer_setting("FILAMENT_PLASMA_BATCH_SIZE", 32)
    parallel = (
        worker_count > 1
        and process_spawn_available()
        and bool(threads)
        and (workers is not None or len(threads) > batch_size)
    )
    last_progress = monotonic()

    def report(thread_index: int) -> None:
        nonlocal last_progress
        now = monotonic()
        if progress_callback is not None and (
            (thread_index + 1) % 25 == 0
            or now - last_progress >= 0.5
            or thread_index + 1 == len(threads)
        ):
            progress_callback(thread_index + 1, len(threads))
            last_progress = now

    if not parallel:
        for thread_index, thread in enumerate(threads):
            _assign_thread_plasma(config, thread, thread_index, spine_length)
            report(thread_index)
        return threads

    pool = process_pool("plasma", worker_count)
    pending = deque()
    starts = iter(range(0, len(threads), batch_size))

    def submit_next() -> None:
        start = next(starts, None)
        if start is not None:
            pending.append((start, pool.submit(
                _assign_plasma_batch,
                config,
                threads[start:start + batch_size],
                start,
                spine_length,
            )))

    try:
        # Bound serialization and queued work while keeping every worker busy.
        for _ in range(2 * worker_count):
            submit_next()
        while pending:
            start, future = pending.popleft()
            completed, error, failed_thread = future.result()
            for offset, result in enumerate(completed):
                thread_index = start + offset
                # Retain caller-visible dictionary identity, as the serial API does.
                threads[thread_index].clear()
                threads[thread_index].update(result)
                report(thread_index)
            if error is not None:
                # Preserve any diagnostic mutations made before the failing operation.
                assert failed_thread is not None
                failed_index = start + len(completed)
                threads[failed_index].clear()
                threads[failed_index].update(failed_thread)
                raise error
            submit_next()
    except BrokenProcessPool:
        # Surface this failure unchanged; the next preview gets a fresh pool.
        discard_process_pool("plasma", worker_count, pool)
        raise
    finally:
        for _, future in pending:
            future.cancel()
    return threads
