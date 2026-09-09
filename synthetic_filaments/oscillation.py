"""Analytic period and displacement functions for filament oscillations."""

from __future__ import annotations

from typing import Final

import numpy as np

OSCILLATION_MODE_SHARED_PERIOD: Final = "shared_period"
OSCILLATION_MODE_LUNA_2022: Final = "luna_2022_curvature"
OSCILLATION_MODES: Final = frozenset({OSCILLATION_MODE_SHARED_PERIOD, OSCILLATION_MODE_LUNA_2022})

# Luna et al. (2022), A&A 660, A54, use g0 = 274 m s^-2.  Keep the
# pendulum-model core in SI units; morphology and plotting code perform their
# explicit km/Mm conversions at the model boundary.
SOLAR_SURFACE_GRAVITY_M_S2: Final = 274.0
SOLAR_RADIUS_M: Final = 696.3e6

# Compatibility value for consumers of the pre-height public constant.  The
# height-dependent implementation below never uses kilometre units internally.
SOLAR_SURFACE_GRAVITY_KM_S2: Final = SOLAR_SURFACE_GRAVITY_M_S2 / 1_000.0


def _finite_array(value: float | np.ndarray, name: str) -> np.ndarray:
    """Return a finite floating-point array without changing its shape."""
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def solar_gravity_m_s2(
    prominence_height_m: float | np.ndarray,
    *,
    solar_radius_m: float = SOLAR_RADIUS_M,
    surface_gravity_m_s2: float = SOLAR_SURFACE_GRAVITY_M_S2,
) -> float | np.ndarray:
    r"""Return solar gravity at height $h$ above the photosphere.

    With photospheric gravity $g_0$ and solar radius $R_\odot$,

    $$
    g(h)=g_0\left(\frac{R_\odot}{R_\odot+h}\right)^2.
    $$

    All lengths are in metres and the returned gravity is in
    $\mathrm{m\,s^{-2}}$.
    """
    heights = _finite_array(prominence_height_m, "prominence_height_m")
    if np.any(heights < 0.0):
        raise ValueError("prominence_height_m must contain only values >= 0")
    if not np.isfinite(solar_radius_m) or solar_radius_m <= 0.0:
        raise ValueError("solar_radius_m must be finite and > 0")
    if not np.isfinite(surface_gravity_m_s2) or surface_gravity_m_s2 <= 0.0:
        raise ValueError("surface_gravity_m_s2 must be finite and > 0")

    gravity = surface_gravity_m_s2 * (solar_radius_m / (solar_radius_m + heights)) ** 2
    return float(gravity) if heights.ndim == 0 else gravity


def gravity_cutoff_period_s(
    prominence_height_m: float | np.ndarray,
    *,
    solar_radius_m: float = SOLAR_RADIUS_M,
    surface_gravity_m_s2: float = SOLAR_SURFACE_GRAVITY_M_S2,
) -> float | np.ndarray:
    r"""Return the height-dependent gravity-only cut-off period.

    $$
    P_{\mathrm{cut,g}}(h)=2\pi\sqrt{\frac{R_\odot+h}{g(h)}}.
    $$
    """
    heights = _finite_array(prominence_height_m, "prominence_height_m")
    gravity = np.asarray(
        solar_gravity_m_s2(
            heights,
            solar_radius_m=solar_radius_m,
            surface_gravity_m_s2=surface_gravity_m_s2,
        ),
        dtype=float,
    )
    periods_s = 2.0 * np.pi * np.sqrt((solar_radius_m + heights) / gravity)
    return float(periods_s) if heights.ndim == 0 else periods_s


def equivalent_curvature_radius_m(
    curvature_radius_m: float | np.ndarray,
    prominence_height_m: float | np.ndarray,
    *,
    solar_radius_m: float = SOLAR_RADIUS_M,
) -> float | np.ndarray:
    r"""Return $R_{\rm eq}(h)$ for a dip at height $h$.

    $$
    \frac{1}{R_{\rm eq}(h)}=\frac{1}{R}+\frac{1}{R_\odot+h}.
    $$
    """
    radii = _finite_array(curvature_radius_m, "curvature_radius_m")
    heights = _finite_array(prominence_height_m, "prominence_height_m")
    if np.any(radii <= 0.0):
        raise ValueError("curvature_radius_m must contain only values > 0")
    if np.any(heights < 0.0):
        raise ValueError("prominence_height_m must contain only values >= 0")
    if not np.isfinite(solar_radius_m) or solar_radius_m <= 0.0:
        raise ValueError("solar_radius_m must be finite and > 0")
    try:
        radii, heights = np.broadcast_arrays(radii, heights)
    except ValueError as error:
        raise ValueError(
            "curvature_radius_m and prominence_height_m must be broadcast-compatible"
        ) from error

    heliocentric_radius_m = solar_radius_m + heights
    equivalent_m = radii * heliocentric_radius_m / (radii + heliocentric_radius_m)
    return float(equivalent_m) if equivalent_m.ndim == 0 else equivalent_m


def luna_2022_longitudinal_period_s(
    curvature_radius_m: float | np.ndarray,
    prominence_height_m: float | np.ndarray,
    *,
    solar_radius_m: float = SOLAR_RADIUS_M,
    surface_gravity_m_s2: float = SOLAR_SURFACE_GRAVITY_M_S2,
) -> float | np.ndarray:
    r"""Return the height-dependent corrected pendulum period.

    With dip curvature radius $R$, height $h$, solar radius $R_\odot$, and
    height-dependent gravity $g(h)$,

    $$
    P(R,h)=2\pi\left[g(h)\left(\frac{1}{R}+\frac{1}{R_\odot+h}\right)\right]^{-1/2}.
    $$

    Lengths must be supplied in metres and the returned period is in seconds.
    """
    equivalent_m = np.asarray(
        equivalent_curvature_radius_m(
            curvature_radius_m,
            prominence_height_m,
            solar_radius_m=solar_radius_m,
        ),
        dtype=float,
    )
    gravity = np.asarray(
        solar_gravity_m_s2(
            prominence_height_m,
            solar_radius_m=solar_radius_m,
            surface_gravity_m_s2=surface_gravity_m_s2,
        ),
        dtype=float,
    )
    equivalent_m, gravity = np.broadcast_arrays(equivalent_m, gravity)
    periods_s = 2.0 * np.pi * np.sqrt(equivalent_m / gravity)
    return float(periods_s) if periods_s.ndim == 0 else periods_s


def luna_2022_inferred_curvature_radius_m(
    period_s: float | np.ndarray,
    prominence_height_m: float | np.ndarray,
    *,
    solar_radius_m: float = SOLAR_RADIUS_M,
    surface_gravity_m_s2: float = SOLAR_SURFACE_GRAVITY_M_S2,
) -> float | np.ndarray:
    r"""Infer dip curvature radius from period and prominence height.

    $$
    R(P,h)=\left[\frac{4\pi^2}{g(h)P^2}-\frac{1}{R_\odot+h}\right]^{-1}.
    $$

    Every period must be strictly below $P_{\mathrm{cut,g}}(h)$; otherwise no
    finite positive gravity-only curvature radius exists.
    """
    periods = _finite_array(period_s, "period_s")
    heights = _finite_array(prominence_height_m, "prominence_height_m")
    if np.any(periods <= 0.0):
        raise ValueError("period_s must contain only values > 0")
    if np.any(heights < 0.0):
        raise ValueError("prominence_height_m must contain only values >= 0")
    try:
        periods, heights = np.broadcast_arrays(periods, heights)
    except ValueError as error:
        raise ValueError("period_s and prominence_height_m must be broadcast-compatible") from error

    cutoff_s = np.asarray(
        gravity_cutoff_period_s(
            heights,
            solar_radius_m=solar_radius_m,
            surface_gravity_m_s2=surface_gravity_m_s2,
        ),
        dtype=float,
    )
    if np.any(periods >= cutoff_s):
        raise ValueError("period_s must be strictly below the gravity cut-off at every height")

    # Algebraically identical to the published inverse, but stable and
    # dimensionally transparent near the cut-off.
    period_fraction_squared = (periods / cutoff_s) ** 2
    heliocentric_radius_m = solar_radius_m + heights
    radii_m = heliocentric_radius_m * period_fraction_squared / (
        1.0 - period_fraction_squared
    )
    return float(radii_m) if radii_m.ndim == 0 else radii_m


def make_thread_oscillation_periods(
    *,
    mode: str,
    curvature_radii_m: np.ndarray,
    prominence_heights_m: np.ndarray,
    shared_period_s: float,
    transverse_period_s: float | None,
) -> dict[str, np.ndarray]:
    """Resolve one period model into aligned per-thread arrays."""
    radii = np.asarray(curvature_radii_m, dtype=float)
    heights = np.asarray(prominence_heights_m, dtype=float)
    if radii.ndim != 1 or heights.shape != radii.shape:
        raise ValueError("thread radii and heights must be aligned one-dimensional arrays")
    gravity = np.asarray(solar_gravity_m_s2(heights), dtype=float)
    cutoff = np.asarray(gravity_cutoff_period_s(heights), dtype=float)

    if mode == OSCILLATION_MODE_SHARED_PERIOD:
        longitudinal = np.full(radii.shape, shared_period_s, dtype=float)
        transverse = np.full(radii.shape, shared_period_s, dtype=float)
    elif mode == OSCILLATION_MODE_LUNA_2022:
        if transverse_period_s is None:
            raise ValueError(
                "transverse_period_s is required when oscillation_mode is "
                f"{OSCILLATION_MODE_LUNA_2022!r}"
            )
        longitudinal = np.asarray(luna_2022_longitudinal_period_s(radii, heights), dtype=float)
        transverse = np.full(radii.shape, transverse_period_s, dtype=float)
    else:
        allowed = ", ".join(sorted(OSCILLATION_MODES))
        raise ValueError(f"unknown oscillation mode {mode!r}; expected one of: {allowed}")

    if (
        not np.isfinite(longitudinal).all()
        or not np.isfinite(transverse).all()
        or np.any(longitudinal <= 0.0)
        or np.any(transverse <= 0.0)
    ):
        raise ValueError("all thread oscillation periods must be finite and > 0")
    return {
        "longitudinal_s": longitudinal,
        "transverse_s": transverse,
        "prominence_heights_m": np.array(heights, copy=True),
        "gravity_m_s2": gravity,
        "gravity_cutoff_s": cutoff,
    }


def evaluate_damped_thread_oscillation(
    time_s: float,
    *,
    oscillation_start_time_s: float,
    damping_time_s: float,
    phase_rad: float,
    longitudinal_amplitude_km: float,
    transverse_amplitude_km: float,
    periods: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    r"""Evaluate unweighted coherent displacement and velocity components.

    For component $c\in\{\parallel,\perp\}$ and thread $j$,

    $$
    \xi_{c,j}(t)=A_c e^{-q/\tau_d}
    \left[\sin\left(2\pi q/P_{c,j}+\phi\right)-\sin\phi\right],
    \qquad q=t-t_0.
    $$

    Displacement and velocity are zero for $t<t_0$.
    """
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError(f"time_s must be finite and >= 0; received {time_s!r}")

    longitudinal_periods = np.asarray(periods["longitudinal_s"], dtype=float)
    transverse_periods = np.asarray(periods["transverse_s"], dtype=float)
    if longitudinal_periods.ndim != 1 or transverse_periods.shape != longitudinal_periods.shape:
        raise ValueError("thread period arrays must be aligned one-dimensional arrays")

    n_threads = longitudinal_periods.size
    zeros = np.zeros(n_threads, dtype=float)
    if time_s < oscillation_start_time_s:
        return {
            "longitudinal_displacement_km": zeros,
            "transverse_displacement_km": zeros.copy(),
            "longitudinal_velocity_km_s": zeros.copy(),
            "transverse_velocity_km_s": zeros.copy(),
        }

    elapsed_time_s = float(time_s - oscillation_start_time_s)
    envelope = float(np.exp(-elapsed_time_s / damping_time_s))

    def component(
        amplitude_km: float,
        component_periods_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Preserve the original scalar operation order for shared periods.
        if component_periods_s.size and np.all(component_periods_s == component_periods_s[0]):
            angular_frequency = 2.0 * np.pi / float(component_periods_s[0])
            phase = angular_frequency * elapsed_time_s + phase_rad
            phase_offset = np.sin(phase) - np.sin(phase_rad)
            displacement_scale = envelope * phase_offset
            velocity_scale = envelope * (
                angular_frequency * np.cos(phase) - phase_offset / damping_time_s
            )
            return (
                np.full(n_threads, amplitude_km * displacement_scale, dtype=float),
                np.full(n_threads, amplitude_km * velocity_scale, dtype=float),
            )

        angular_frequency = 2.0 * np.pi / component_periods_s
        phase = angular_frequency * elapsed_time_s + phase_rad
        phase_offset = np.sin(phase) - np.sin(phase_rad)
        displacement = amplitude_km * envelope * phase_offset
        velocity = (
            amplitude_km
            * envelope
            * (angular_frequency * np.cos(phase) - phase_offset / damping_time_s)
        )
        return displacement, velocity

    longitudinal_displacement, longitudinal_velocity = component(
        longitudinal_amplitude_km,
        longitudinal_periods,
    )
    transverse_displacement, transverse_velocity = component(
        transverse_amplitude_km,
        transverse_periods,
    )
    return {
        "longitudinal_displacement_km": longitudinal_displacement,
        "transverse_displacement_km": transverse_displacement,
        "longitudinal_velocity_km_s": longitudinal_velocity,
        "transverse_velocity_km_s": transverse_velocity,
    }
