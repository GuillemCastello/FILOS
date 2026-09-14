"""Function-oriented 3D filament dynamics and rendering.

The module consumes a completed static-result dictionary. It preserves the
static spine, thread plasma, and frame-zero image while translating whole
thread centerlines through a localized damped oscillation and an independent
cumulative Brownian walk.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterator
from copy import deepcopy
from numbers import Integral

import numpy as np

from .degradation import degrade_transmission
from .execution import integer_setting, thread_pool
from .oscillation import (
    OSCILLATION_MODE_LUNA_2022,
    OSCILLATION_MODE_SHARED_PERIOD,
    OSCILLATION_MODES,
    SOLAR_RADIUS_M,
    SOLAR_SURFACE_GRAVITY_M_S2,
    evaluate_damped_thread_oscillation,
    make_thread_oscillation_periods,
)
from .render import (
    _project_vectors,
    compose_transmission,
    dilute_transmission,
    make_masks,
    prepare_dynamic_tau_rasterization,
    rasterize_prepared_dynamic_tau,
)

DYNAMICS_SCHEMA_VERSION = 9
BROWNIAN_RNG_STREAM_ID = 40_001

_DYNAMICS_DEFAULTS: dict[str, object] = {
    "n_frames": 240,
    "cadence_s": 60.0,
    "longitudinal_displacement_amplitude_km": 18_000.0,
    "transverse_displacement_amplitude_km": 2_000.0,
    "oscillation_mode": OSCILLATION_MODE_LUNA_2022,
    "period_s": 3_600.0,
    "transverse_period_s": 1_500.0,
    "damping_time_s": 14_400.0,
    "phase_rad": 0.0,
    "brownian_step_min_km": 200.0,
    "brownian_step_max_km": 500.0,
    "center_spine_fraction": 0.55,
    "center_height_km": None,
    "half_strength_distance_km": 11_900.0,
    "oscillation_start_time_s": 1_200.0,
}


def make_dynamics_config(seed: int, **overrides: object) -> dict[str, object]:
    """Return the notebook-authoritative dynamics configuration as a plain dict."""
    unknown = sorted(set(overrides) - set(_DYNAMICS_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown dynamics configuration fields: {unknown}")
    config = {"seed": seed, **_DYNAMICS_DEFAULTS, **overrides}
    if (
        config["oscillation_mode"] == OSCILLATION_MODE_SHARED_PERIOD
        and "transverse_period_s" not in overrides
    ):
        config["transverse_period_s"] = None
    _validate_dynamics_config(config)
    return config


def _validate_dynamics_config(config: dict[str, object]) -> None:
    """Reject invalid dynamics parameters before any arrays are allocated."""
    seed = config["seed"]
    n_frames = config["n_frames"]
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0; received {seed!r}")
    if isinstance(n_frames, bool) or not isinstance(n_frames, Integral) or n_frames < 1:
        raise ValueError(f"n_frames must be an integer >= 1; received {n_frames!r}")

    mode = config["oscillation_mode"]
    if mode not in OSCILLATION_MODES:
        allowed = ", ".join(sorted(OSCILLATION_MODES))
        raise ValueError(f"oscillation_mode must be one of {allowed}; received {mode!r}")

    for name in ("cadence_s", "period_s", "damping_time_s", "half_strength_distance_km"):
        value = config[name]
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0; received {value!r}")

    transverse_period = config["transverse_period_s"]
    if transverse_period is not None and (
        not np.isfinite(transverse_period) or transverse_period <= 0.0
    ):
        raise ValueError(
            f"transverse_period_s must be None or finite and > 0; received {transverse_period!r}"
        )
    if mode == OSCILLATION_MODE_LUNA_2022 and transverse_period is None:
        raise ValueError(
            "transverse_period_s is required when oscillation_mode is "
            f"{OSCILLATION_MODE_LUNA_2022!r}"
        )

    for name in (
        "phase_rad",
        "longitudinal_displacement_amplitude_km",
        "transverse_displacement_amplitude_km",
    ):
        value = config[name]
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}")

    for name in (
        "oscillation_start_time_s",
        "brownian_step_min_km",
        "brownian_step_max_km",
    ):
        value = config[name]
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0; received {value!r}")
    if config["brownian_step_max_km"] < config["brownian_step_min_km"]:
        raise ValueError(
            "brownian_step_max_km must be >= brownian_step_min_km; "
            f"received min={config['brownian_step_min_km']!r}, "
            f"max={config['brownian_step_max_km']!r}"
        )

    center_fraction = config["center_spine_fraction"]
    if center_fraction is not None and (
        not np.isfinite(center_fraction) or not 0.0 <= center_fraction <= 1.0
    ):
        raise ValueError(
            f"center_spine_fraction must be None or lie in [0, 1]; received {center_fraction!r}"
        )
    center_height = config["center_height_km"]
    if center_height is not None and (not np.isfinite(center_height) or center_height < 0.0):
        raise ValueError(
            f"center_height_km must be None or finite and >= 0; received {center_height!r}"
        )


def _copy_record(record: dict[str, object]) -> dict[str, object]:
    """Copy one dictionary and duplicate every NumPy array it owns."""
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else deepcopy(value)
        for key, value in record.items()
    }


def _copy_threads(threads: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return independent copies of all thread dictionaries."""
    return [_copy_record(thread) for thread in threads]


def _spine_total_length_km(spine: dict[str, object]) -> float:
    """Return the final arclength coordinate of a spine."""
    arclength = np.asarray(spine["s"], dtype=float)
    if arclength.ndim != 1 or arclength.size == 0:
        raise ValueError("spine['s'] must be a non-empty one-dimensional array")
    return float(arclength[-1])


def _interpolate_spine(
    spine: dict[str, object],
    arclength_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate spine position and orthonormal image-plane basis."""
    query = np.asarray(arclength_km, dtype=float)
    spine_s = np.asarray(spine["s"], dtype=float)
    x = np.interp(query, spine_s, np.asarray(spine["x"], dtype=float))
    y = np.interp(query, spine_s, np.asarray(spine["y"], dtype=float))
    tangent_x = np.interp(query, spine_s, np.asarray(spine["tangent_x"], dtype=float))
    tangent_y = np.interp(query, spine_s, np.asarray(spine["tangent_y"], dtype=float))
    tangent_norm = np.hypot(tangent_x, tangent_y)
    tangent_x = tangent_x / tangent_norm
    tangent_y = tangent_y / tangent_norm
    return x, y, tangent_x, tangent_y, -tangent_y, tangent_x


def _vertex_integration_weights(s_km: np.ndarray) -> np.ndarray:
    """Return trapezoidal line-integration weights at centerline vertices."""
    s = np.asarray(s_km, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("thread arclength must be a non-empty 1D array")
    if s.size == 1:
        return np.ones(1, dtype=float)
    ds = np.diff(s)
    if np.any(ds <= 0.0) or not np.isfinite(ds).all():
        raise ValueError("thread arclength must be finite and strictly increasing")
    weights = np.empty_like(s)
    weights[0] = 0.5 * ds[0]
    weights[-1] = 0.5 * ds[-1]
    weights[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return weights


def _material_centroid(thread: dict[str, object]) -> np.ndarray:
    """Return the optical-depth-weighted 3D centroid of one thread."""
    s = np.asarray(thread["s"], dtype=float)
    line_weights = _vertex_integration_weights(s)
    tau_along = np.asarray(thread["tau_along"], dtype=float)
    if tau_along.shape == s.shape:
        line_weights = line_weights * np.maximum(tau_along, 0.0)
    if not np.any(line_weights > 0.0):
        line_weights = _vertex_integration_weights(s)
    coordinates = np.column_stack(
        [
            np.asarray(thread["x"], dtype=float),
            np.asarray(thread["y"], dtype=float),
            np.asarray(thread["z"], dtype=float),
        ]
    )
    return np.average(coordinates, axis=0, weights=line_weights)


def _thread_motion_bases(
    threads: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed horizontal longitudinal/transverse bases per thread."""
    longitudinal = np.empty((len(threads), 3), dtype=float)
    for index, thread in enumerate(threads):
        x = np.asarray(thread["x"], dtype=float)
        y = np.asarray(thread["y"], dtype=float)
        if (
            x.ndim != 1
            or y.ndim != 1
            or x.size < 2
            or y.shape != x.shape
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError(
                f"thread {index} must have finite aligned x/y centerline arrays "
                "with at least two points"
            )
        chord_xy = np.asarray([x[-1] - x[0], y[-1] - y[0]], dtype=float)
        chord_norm = float(np.linalg.norm(chord_xy))
        if not np.isfinite(chord_norm) or chord_norm <= 0.0:
            raise ValueError(
                f"thread {index} has a degenerate x-y projection and no "
                "longitudinal dynamics direction"
            )
        longitudinal[index] = (
            chord_xy[0] / chord_norm,
            chord_xy[1] / chord_norm,
            0.0,
        )
    transverse = np.column_stack(
        [
            -longitudinal[:, 1],
            longitudinal[:, 0],
            np.zeros(len(threads), dtype=float),
        ]
    )
    return longitudinal, transverse


def _thread_anchor_km(thread: dict[str, object]) -> float:
    """Return one thread's representative spine arclength."""
    anchor = float(thread.get("spine_anchor_km", float("nan")))
    if np.isfinite(anchor):
        return anchor
    spine_coordinates = np.asarray(thread.get("s_spine", []), dtype=float)
    return float(np.mean(spine_coordinates)) if spine_coordinates.size else float("nan")


def _select_site(
    config: dict[str, object],
    spine: dict[str, object],
    threads: list[dict[str, object]],
    initial_centroids_km: np.ndarray,
    longitudinal_directions: np.ndarray,
    transverse_directions: np.ndarray,
) -> dict[str, object]:
    """Choose a deterministic material-bearing oscillation center."""
    material_indices = np.asarray(
        [
            index
            for index, thread in enumerate(threads)
            if float(thread["tau0"]) > 0.0
            and np.asarray(thread["tau_along"]).shape == np.asarray(thread["s"]).shape
            and np.any(np.asarray(thread["tau_along"]) > 0.0)
            and np.isfinite(_thread_anchor_km(thread))
        ],
        dtype=int,
    )
    if material_indices.size == 0:
        raise ValueError("dynamics requires at least one material-bearing thread")

    anchors = np.asarray([_thread_anchor_km(thread) for thread in threads], dtype=float)
    total_length_km = _spine_total_length_km(spine)
    center_fraction = config["center_spine_fraction"]
    if center_fraction is None:
        rng = np.random.default_rng(int(config["seed"]))
        source_index = int(rng.choice(material_indices))
        center_s_km = anchors[source_index]
    else:
        center_s_km = float(center_fraction) * total_length_km
        source_index = int(
            material_indices[np.argmin(np.abs(anchors[material_indices] - center_s_km))]
        )

    spine_values = _interpolate_spine(spine, np.asarray([center_s_km], dtype=float))
    center_x, center_y = (float(values[0]) for values in spine_values[:2])
    configured_height = config["center_height_km"]
    center_height_km = (
        float(initial_centroids_km[source_index, 2])
        if configured_height is None
        else float(configured_height)
    )
    center = np.asarray([center_x, center_y, center_height_km], dtype=float)

    distances = np.linalg.norm(initial_centroids_km[material_indices] - center, axis=1)
    nearest_distance_km = float(distances[int(np.argmin(distances))])
    material_weights = _local_weights(
        initial_centroids_km[material_indices], {"center_xyz_km": center}, config
    )
    if not np.any(material_weights > 0.0):
        raise ValueError(
            "no material-bearing thread reaches 5% oscillation weight at the requested center; "
            f"nearest distance={nearest_distance_km:.3f} km"
        )

    return {
        "spine_arclength_km": float(center_s_km),
        "spine_fraction": float(center_s_km / total_length_km),
        "center_xyz_km": tuple(float(value) for value in center),
        "source_thread_index": source_index,
        "source_longitudinal_xyz": tuple(
            float(value) for value in longitudinal_directions[source_index]
        ),
        "source_transverse_xyz": tuple(
            float(value) for value in transverse_directions[source_index]
        ),
        "nearest_material_distance_km": nearest_distance_km,
    }


def _local_weights(
    centroids_km: np.ndarray,
    site: dict[str, object],
    config: dict[str, object],
) -> np.ndarray:
    r"""Evaluate distance-based oscillation weights at initial thread centroids.

    The weight at centroid distance $d$ is

    $$w(d)=\exp\left[-\ln(2)\left(\frac{d}{L}\right)^{1.1}\right],$$

    where $L$ is the distance at which oscillation amplitude falls to half
    its value at the center. The decay shape is fixed at the original default.
    Weights below 0.05 are set to zero.
    """
    center = np.asarray(site["center_xyz_km"], dtype=float)
    distance = np.linalg.norm(np.asarray(centroids_km, dtype=float) - center, axis=1)
    normalized = distance / float(config["half_strength_distance_km"])
    weights = np.exp(-np.log(2.0) * normalized ** 1.1)
    return np.where(weights >= 0.05, weights, 0.0)


def _sample_brownian_steps(
    rng: np.random.Generator,
    n_threads: int,
    config: dict[str, object],
) -> np.ndarray:
    """Sample one independent isotropic 3D random-walk step per thread."""
    maximum_km = float(config["brownian_step_max_km"])
    if maximum_km == 0.0:
        return np.zeros((n_threads, 3), dtype=float)

    amplitudes_km = rng.uniform(
        float(config["brownian_step_min_km"]),
        maximum_km,
        size=n_threads,
    )
    cos_polar_angle = rng.uniform(-1.0, 1.0, size=n_threads)
    azimuth_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_threads)
    sin_polar_angle = np.sqrt(np.maximum(1.0 - cos_polar_angle**2, 0.0))
    directions = np.column_stack(
        [
            sin_polar_angle * np.cos(azimuth_rad),
            sin_polar_angle * np.sin(azimuth_rad),
            cos_polar_angle,
        ]
    )
    return amplitudes_km[:, None] * directions


def _array_sha256(array: np.ndarray) -> str:
    """Hash array dtype, shape, and contiguous bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


def _geometry_sha256(threads: list[dict[str, object]]) -> str:
    """Hash the initial 3D thread centerlines."""
    digest = hashlib.sha256()
    for thread in threads:
        for name in ("x", "y", "z"):
            array = np.ascontiguousarray(thread[name], dtype=np.float64)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _expected_shapes(static_config: dict[str, object]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return high-resolution and native image shapes."""
    highres_shape = (int(static_config["ny"]), int(static_config["nx"]))
    factor = int(static_config["downsample_factor"])
    native_shape = (highres_shape[0] // factor, highres_shape[1] // factor)
    return highres_shape, native_shape


def _native_line_transmission(state: dict[str, object], tau_map: np.ndarray) -> np.ndarray:
    """Apply the fixed GONG PSF and detector integration."""
    return degrade_transmission(tau_map, state["static_config"])


def _compose_line_transmission(
    state: dict[str, object],
    line_transmission: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Composite native line transmission over one aligned background."""
    transmission = dilute_transmission(
        line_transmission,
        float(state["static_config"]["gong_bandpass_line_fraction"]),
    )
    return compose_transmission(
        background,
        transmission,
        float(state["source_fraction"]),
        float(state["source_level"]),
    )


def _compose_native(
    state: dict[str, object],
    tau_map: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Render one optical-depth map over one native background."""
    return _compose_line_transmission(
        state,
        _native_line_transmission(state, tau_map),
        background,
    )


def initialize_dynamics(
    initial_result: dict[str, object],
    dynamics_config: dict[str, object],
    background_frames: np.ndarray | None = None,
) -> dict[str, object]:
    """Initialize a mutable dynamics state from one static result."""
    config = make_dynamics_config(**dict(dynamics_config))
    threads = initial_result.get("threads")
    if not isinstance(threads, list) or not threads:
        raise ValueError("dynamics requires a static result containing threads")

    static_config = deepcopy(initial_result["config"])
    spine = _copy_record(initial_result["spine"])
    initial_threads = _copy_threads(threads)
    initial_metadata = deepcopy(initial_result["metadata"])
    arrays = initial_result["arrays"]

    periods = make_thread_oscillation_periods(
        mode=str(config["oscillation_mode"]),
        curvature_radii_m=1_000.0
        * np.asarray(
            [thread["dip_radius_km"] for thread in initial_threads],
            dtype=float,
        ),
        prominence_heights_m=1_000.0
        * np.asarray(
            [thread["height_km"] for thread in initial_threads],
            dtype=float,
        ),
        shared_period_s=float(config["period_s"]),
        transverse_period_s=(
            None if config["transverse_period_s"] is None else float(config["transverse_period_s"])
        ),
    )
    longitudinal_periods = np.array(periods["longitudinal_s"], copy=True)
    transverse_periods = np.array(periods["transverse_s"], copy=True)
    initial_centroids = np.vstack([_material_centroid(thread) for thread in initial_threads])
    longitudinal_directions, transverse_directions = _thread_motion_bases(initial_threads)
    site = _select_site(
        config,
        spine,
        initial_threads,
        initial_centroids,
        longitudinal_directions,
        transverse_directions,
    )
    initial_weights = _local_weights(initial_centroids, site, config)

    highres_shape, native_shape = _expected_shapes(static_config)
    for key in ("tau_map", "degraded_intensity", "background"):
        if key not in arrays:
            raise ValueError(f"static result is missing required array {key!r}")

    initial_tau = np.asarray(arrays["tau_map"], dtype=float)
    initial_gong = np.asarray(arrays["degraded_intensity"], dtype=float)
    background = np.array(arrays["background"], dtype=float, copy=True)
    if initial_tau.shape != highres_shape:
        raise ValueError(
            f"initial tau_map must have shape {highres_shape}; received {initial_tau.shape}"
        )
    for name, array in (("degraded_intensity", initial_gong), ("background", background)):
        if array.shape != native_shape:
            raise ValueError(
                f"initial {name} must have shape {native_shape}; received {array.shape}"
            )
    if (
        not np.isfinite(initial_tau).all()
        or not np.isfinite(initial_gong).all()
        or not np.isfinite(background).all()
    ):
        raise ValueError("initial tau, GONG image, and background must be finite")
    if np.any(initial_tau < 0.0) or np.any(initial_gong < 0.0) or np.any(background < 0.0):
        raise ValueError("initial tau, GONG image, and background must be non-negative")

    support = None
    if "support" in arrays:
        requested_support = np.asarray(arrays["support"])
        if requested_support.shape != native_shape:
            raise ValueError(
                f"initial support must have shape {native_shape}; received {requested_support.shape}"
            )
        support = np.array(requested_support, dtype=np.uint8, copy=True)

    aligned_backgrounds = None
    if background_frames is not None:
        requested_backgrounds = np.asarray(background_frames)
        expected_sequence_shape = (int(config["n_frames"]), *native_shape)
        if requested_backgrounds.shape != expected_sequence_shape:
            raise ValueError(
                "background_frames must align one-to-one with dynamics frames; "
                f"expected {expected_sequence_shape}, received {requested_backgrounds.shape}"
            )
        if not np.isfinite(requested_backgrounds).all() or np.any(requested_backgrounds < 0.0):
            raise ValueError("background_frames must be finite and non-negative")
        if not np.array_equal(requested_backgrounds[0], background):
            raise ValueError("background_frames[0] must equal the static result background exactly")
        aligned_backgrounds = np.array(requested_backgrounds, copy=True)

    n_threads = len(initial_threads)
    prepared_tau = prepare_dynamic_tau_rasterization(static_config, initial_threads)
    rerendered_tau, _ = rasterize_prepared_dynamic_tau(
        static_config,
        prepared_tau,
        np.zeros((n_threads, 3), dtype=float),
    )
    if not np.allclose(rerendered_tau, initial_tau, rtol=5e-12, atol=5e-12):
        maximum_error = float(np.max(np.abs(rerendered_tau - initial_tau)))
        raise ValueError(
            "static thread state does not reproduce its saved tau_map; "
            f"maximum absolute error={maximum_error:.6e}"
        )

    state: dict[str, object] = {
        "config": config,
        "static_config": static_config,
        "spine": spine,
        "initial_threads": initial_threads,
        "initial_metadata": initial_metadata,
        "oscillation_periods": periods,
        "thread_longitudinal_periods_s": longitudinal_periods,
        "thread_transverse_periods_s": transverse_periods,
        "thread_prominence_heights_m": np.array(periods["prominence_heights_m"], copy=True),
        "thread_gravity_m_s2": np.array(periods["gravity_m_s2"], copy=True),
        "thread_gravity_cutoff_periods_s": np.array(periods["gravity_cutoff_s"], copy=True),
        "initial_centroids_km": initial_centroids,
        "thread_longitudinal_directions": longitudinal_directions,
        "thread_transverse_directions": transverse_directions,
        "site": site,
        "initial_thread_weights": initial_weights,
        "brownian_rng": np.random.default_rng(
            np.random.SeedSequence([int(config["seed"]), BROWNIAN_RNG_STREAM_ID])
        ),
        "coherent_displacements_km": np.zeros((n_threads, 3), dtype=float),
        "brownian_displacements_km": np.zeros((n_threads, 3), dtype=float),
        "brownian_step_displacements_km": np.zeros((n_threads, 3), dtype=float),
        "displacements_km": np.zeros((n_threads, 3), dtype=float),
        "current_index": 0,
        "support": support,
        "initial_tau": np.array(initial_tau, copy=True),
        "initial_gong": np.array(initial_gong, copy=True),
        "background": background,
        "background_frames": aligned_backgrounds,
        "source_fraction": float(
            initial_metadata.get(
                "source_fraction_effective",
                static_config["source_fraction"],
            )
        ),
        "source_level": float(initial_metadata.get("source_level", np.median(background))),
        "prepared_dynamic_tau": prepared_tau,
        "prepare_export_fields": False,
    }
    if not 0.0 <= state["source_fraction"] <= 1.0:
        raise ValueError("effective source fraction must lie in [0, 1]")
    if not np.isfinite(state["source_level"]) or state["source_level"] < 0.0:
        raise ValueError("source level must be finite and non-negative")

    initial_composite = _compose_native(state, initial_tau, background)
    state["fixed_detector_residual"] = initial_gong - initial_composite
    return state


def dynamics_background_at_frame(
    state: dict[str, object],
    frame_index: int,
) -> np.ndarray:
    """Return the native observed background assigned to one frame."""
    n_frames = int(state["config"]["n_frames"])
    if not 0 <= frame_index < n_frames:
        raise IndexError(f"frame_index must lie in [0, {n_frames - 1}]")
    background_frames = state["background_frames"]
    return state["background"] if background_frames is None else background_frames[frame_index]


def _oscillation_state(
    state: dict[str, object],
    time_s: float,
) -> dict[str, np.ndarray]:
    """Return explicit per-thread coherent driver components."""
    config = state["config"]
    return evaluate_damped_thread_oscillation(
        time_s,
        oscillation_start_time_s=float(config["oscillation_start_time_s"]),
        damping_time_s=float(config["damping_time_s"]),
        phase_rad=float(config["phase_rad"]),
        longitudinal_amplitude_km=float(config["longitudinal_displacement_amplitude_km"]),
        transverse_amplitude_km=float(config["transverse_displacement_amplitude_km"]),
        periods=state["oscillation_periods"],
    )


def _coherent_thread_velocities_xy(
    state: dict[str, object],
    oscillation: dict[str, np.ndarray],
) -> np.ndarray:
    """Return spatially weighted coherent image-plane velocity per thread."""
    longitudinal = oscillation["longitudinal_velocity_km_s"][:, None]
    transverse = oscillation["transverse_velocity_km_s"][:, None]
    unweighted = (
        longitudinal * state["thread_longitudinal_directions"]
        + transverse * state["thread_transverse_directions"]
    )
    return _project_vectors(
        state["static_config"], state["initial_thread_weights"][:, None] * unweighted
    )


def _render_current_state(
    state: dict[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    dict[str, np.ndarray],
]:
    """Return rendered arrays, optionally including export velocity weights."""
    frame_index = int(state["current_index"])
    oscillation = _oscillation_state(
        state,
        frame_index * float(state["config"]["cadence_s"]),
    )
    weighted_velocity_numerator = None
    if frame_index == 0:
        tau_map = np.array(state["initial_tau"], copy=True)
        if state["prepare_export_fields"]:
            velocities = _coherent_thread_velocities_xy(state, oscillation)
            if np.any(velocities):
                _, weighted_velocity_numerator = rasterize_prepared_dynamic_tau(
                    state["static_config"],
                    state["prepared_dynamic_tau"],
                    state["displacements_km"],
                    thread_values=velocities,
                )
            else:
                weighted_velocity_numerator = np.zeros((2, *tau_map.shape), dtype=float)
        return (
            tau_map,
            np.array(state["initial_gong"], copy=True),
            _native_line_transmission(state, tau_map),
            weighted_velocity_numerator,
            oscillation,
        )

    thread_values = (
        _coherent_thread_velocities_xy(state, oscillation)
        if state["prepare_export_fields"]
        else None
    )
    tau_map, weighted_velocity_numerator = rasterize_prepared_dynamic_tau(
        state["static_config"],
        state["prepared_dynamic_tau"],
        state["displacements_km"],
        thread_values=thread_values,
    )
    line_transmission = _native_line_transmission(state, tau_map)
    background = dynamics_background_at_frame(state, frame_index)
    residual = state["fixed_detector_residual"] if state["background_frames"] is None else 0.0
    gong_image = np.maximum(
        _compose_line_transmission(state, line_transmission, background) + residual,
        0.0,
    )
    return tau_map, gong_image, line_transmission, weighted_velocity_numerator, oscillation


def current_dynamics_frame(
    state: dict[str, object], *, export_only: bool = False
) -> dict[str, object]:
    """Render without advancing; export_only omits unused diagnostic masks."""
    config = state["config"]
    static_config = state["static_config"]
    frame_index = int(state["current_index"])
    time_s = frame_index * float(config["cadence_s"])
    tau_map, gong_image, line_transmission, weighted_velocity_numerator, oscillation = (
        _render_current_state(state)
    )
    if not export_only:
        soft_mask_highres, hard_mask_highres = make_masks(
            tau_map,
            float(static_config["mask_tau_threshold"]),
        )
        tau_native = -np.log(np.maximum(line_transmission, 1.0e-12))
        hard_mask = tau_native > float(static_config["mask_tau_threshold"])
    soft_mask = 1.0 - line_transmission
    observable_soft_mask = 1.0 - dilute_transmission(
        line_transmission,
        float(static_config["gong_bandpass_line_fraction"]),
    )
    reference_index = int(state["site"]["source_thread_index"])

    frame = {
        "index": frame_index,
        "time_s": float(time_s),
        "gong_image": gong_image,
        "tau_map": tau_map,
        "soft_mask": soft_mask,
        "observable_soft_mask": observable_soft_mask,
        "thread_displacements_km": np.array(state["displacements_km"], copy=True),
        "coherent_thread_displacements_km": np.array(state["coherent_displacements_km"], copy=True),
        "brownian_thread_displacements_km": np.array(state["brownian_displacements_km"], copy=True),
        "brownian_step_displacements_km": np.array(
            state["brownian_step_displacements_km"], copy=True
        ),
        "thread_weights": np.array(state["initial_thread_weights"], copy=True),
        "unweighted_longitudinal_thread_displacements_km": np.array(
            oscillation["longitudinal_displacement_km"], copy=True
        ),
        "unweighted_transverse_thread_displacements_km": np.array(
            oscillation["transverse_displacement_km"], copy=True
        ),
        "unweighted_longitudinal_thread_velocities_km_s": np.array(
            oscillation["longitudinal_velocity_km_s"], copy=True
        ),
        "unweighted_transverse_thread_velocities_km_s": np.array(
            oscillation["transverse_velocity_km_s"], copy=True
        ),
        "driving_longitudinal_displacement_km": float(
            oscillation["longitudinal_displacement_km"][reference_index]
        ),
        "driving_transverse_displacement_km": float(
            oscillation["transverse_displacement_km"][reference_index]
        ),
        "driving_longitudinal_velocity_km_s": float(
            oscillation["longitudinal_velocity_km_s"][reference_index]
        ),
        "driving_transverse_velocity_km_s": float(
            oscillation["transverse_velocity_km_s"][reference_index]
        ),
    }
    if not export_only:
        frame.update(
            filament_mask=hard_mask.astype(np.uint8),
            filament_mask_highres=hard_mask_highres.astype(np.uint8),
            soft_mask_highres=soft_mask_highres,
        )
    if weighted_velocity_numerator is not None:
        frame["coherent_velocity_numerator_highres"] = weighted_velocity_numerator
    return frame


def _advance_dynamics(state: dict[str, object]) -> None:
    """Advance the seeded displacement state without rendering."""
    config = state["config"]
    next_index = int(state["current_index"]) + 1
    if next_index >= int(config["n_frames"]):
        raise StopIteration("the configured sequence is complete")

    state["current_index"] = next_index
    time_s = next_index * float(config["cadence_s"])
    oscillation = _oscillation_state(state, time_s)
    unweighted_displacements = (
        oscillation["longitudinal_displacement_km"][:, None]
        * state["thread_longitudinal_directions"]
        + oscillation["transverse_displacement_km"][:, None] * state["thread_transverse_directions"]
    )
    state["coherent_displacements_km"] = (
        state["initial_thread_weights"][:, None] * unweighted_displacements
    )
    brownian_step = _sample_brownian_steps(
        state["brownian_rng"],
        len(state["initial_threads"]),
        config,
    )
    state["brownian_step_displacements_km"] = brownian_step
    state["brownian_displacements_km"] += brownian_step
    state["displacements_km"] = (
        state["coherent_displacements_km"] + state["brownian_displacements_km"]
    )


def step_dynamics(state: dict[str, object]) -> dict[str, object]:
    """Advance one cadence, mutate the state, and render the new frame."""
    _advance_dynamics(state)
    return current_dynamics_frame(state)


def iter_production_frames(state: dict[str, object]) -> Iterator[dict[str, object]]:
    """Render a bounded sequence concurrently and yield in deterministic order.

    Only small displacement arrays are copied per queued frame. The producer
    alone advances the RNG with the same per-cadence operations as step_dynamics;
    render workers share invariant geometry and never mutate the live state.
    """
    workers = integer_setting("FILAMENT_DYNAMICS_WORKERS", 4)
    queue_size = integer_setting("FILAMENT_DYNAMICS_QUEUE_SIZE", 8)
    executor = thread_pool("dynamics", workers) if workers > 1 else None
    pending = deque()
    n_frames = int(state["config"]["n_frames"])
    next_index = int(state["current_index"])
    displacement_fields = (
        "coherent_displacements_km", "brownian_displacements_km",
        "brownian_step_displacements_km", "displacements_km",
    )
    try:
        while next_index < n_frames or pending:
            while next_index < n_frames and len(pending) < queue_size:
                if next_index != int(state["current_index"]):
                    _advance_dynamics(state)
                snapshot = dict(state)
                for field in displacement_fields:
                    snapshot[field] = np.array(state[field], copy=True)
                if executor is None:
                    yield current_dynamics_frame(snapshot, export_only=True)
                else:
                    pending.append(executor.submit(
                        current_dynamics_frame, snapshot, export_only=True
                    ))
                next_index += 1
            if pending:
                yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        # Finish outstanding readers before recorder cleanup can release state.
        for future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except BaseException:
                    pass


def _period_summary(periods_s: np.ndarray) -> dict[str, float]:
    """Return stable summary statistics for one period array."""
    return {
        "minimum": float(np.min(periods_s)),
        "median": float(np.median(periods_s)),
        "maximum": float(np.max(periods_s)),
    }


def _sequence_metadata(
    state: dict[str, object],
    frames: list[dict[str, object]],
) -> dict[str, object]:
    """Build reproducibility metadata for a completed in-memory sequence."""
    config = state["config"]
    static_config = state["static_config"]
    luna_mode = config["oscillation_mode"] == OSCILLATION_MODE_LUNA_2022
    highres_shape, native_shape = _expected_shapes(static_config)
    return {
        "dynamics_schema_version": DYNAMICS_SCHEMA_VERSION,
        "seed": int(config["seed"]),
        "n_frames": int(config["n_frames"]),
        "cadence_s": float(config["cadence_s"]),
        "duration_s": (int(config["n_frames"]) - 1) * float(config["cadence_s"]),
        "oscillation_start_time_s": float(config["oscillation_start_time_s"]),
        "oscillation_mode": config["oscillation_mode"],
        "displacement_model": (
            "thread_local_longitudinal_transverse_damped_equilibrium_return"
            "_plus_independent_brownian_random_walk"
        ),
        "displacement_equation": (
            "Delta_r_j(t_n)=w_j*(A_longitudinal*g_longitudinal_j(q)"
            "*e_longitudinal_j+A_transverse*g_transverse_j(q)*e_transverse_j)"
            "+sum_{k=1}^n A_jk*u_jk; q=t_n-t0; g_cj(q)=0 for q<0, "
            "otherwise exp(-q/tau)*(sin(2*pi*q/P_cj+phase)-sin(phase))"
        ),
        "period_model": {
            "longitudinal": (
                "luna_2022_corrected_pendulum_from_thread_dip_radius"
                if luna_mode
                else "configured_shared_period"
            ),
            "transverse": (
                "configured_transverse_period" if luna_mode else "configured_shared_period"
            ),
            "luna_2022_equation": (
                "P_j(h_j)=2*pi/sqrt(g(h_j)*(1/R_j+1/(R_sun+h_j)))"
                if luna_mode
                else None
            ),
            "luna_2022_gravity_equation": (
                "g(h)=g0*(R_sun/(R_sun+h))^2" if luna_mode else None
            ),
            "luna_2022_height_source": (
                "per-thread realized dip-bottom height" if luna_mode else None
            ),
            "luna_2022_gas_pressure_slow_mode_included": False if luna_mode else None,
            "solar_surface_gravity_m_s2": (
                SOLAR_SURFACE_GRAVITY_M_S2 if luna_mode else None
            ),
            "solar_radius_m": SOLAR_RADIUS_M if luna_mode else None,
            "thread_prominence_heights_m": (
                _period_summary(state["thread_prominence_heights_m"])
                if luna_mode
                else None
            ),
            "thread_gravity_m_s2": (
                _period_summary(state["thread_gravity_m_s2"]) if luna_mode else None
            ),
            "thread_gravity_cutoff_periods_s": (
                _period_summary(state["thread_gravity_cutoff_periods_s"])
                if luna_mode
                else None
            ),
            "reference": (
                "Luna et al. 2022, A&A 660 A54, doi:10.1051/0004-6361/202142907"
                if luna_mode
                else None
            ),
        },
        "thread_longitudinal_periods_s": _period_summary(state["thread_longitudinal_periods_s"]),
        "thread_transverse_periods_s": _period_summary(state["thread_transverse_periods_s"]),
        "driver_scalar_reference_thread_index": int(state["site"]["source_thread_index"]),
        "velocity_model": (
            "analytic_derivative_of_coherent_displacement_plus_discrete_brownian_step"
        ),
        "velocity_equation": (
            "v_coherent_j(t)=d(xi_j)/dt; Brownian increments are displacements per cadence"
        ),
        "motion_basis": "fixed_per_thread_initial_xy_endpoint_chord_and_left_perpendicular",
        "spatial_kernel": "fixed_compact_super_gaussian_on_initial_thread_material_centroids",
        "spatial_weights_time_dependent": False,
        "coherent_equilibrium_return_period_s": (None if luna_mode else float(config["period_s"])),
        "brownian_motion": {
            "model": "cumulative_independent_isotropic_3d_random_walk",
            "amplitude_distribution": "uniform_km_per_simulation_step",
            "step_min_km": float(config["brownian_step_min_km"]),
            "step_max_km": float(config["brownian_step_max_km"]),
            "rng_stream_id": BROWNIAN_RNG_STREAM_ID,
            "spatial_kernel_applied": False,
        },
        "advection_scheme": (
            "coherent_absolute_translation_from_frame_0_plus_cumulative_brownian_rigid_translation"
        ),
        "background_evolution": (
            "consecutive_aligned_observed_frames"
            if state["background_frames"] is not None
            else "static_frame_0_background_and_detector_residual"
        ),
        "opacity_evolution": "advected_threads_with_static_plasma_parameters",
        "frame_0_preserved_exactly": True,
        "initial_geometry_sha256": _geometry_sha256(state["initial_threads"]),
        "thread_longitudinal_directions_sha256": _array_sha256(
            state["thread_longitudinal_directions"]
        ),
        "thread_transverse_directions_sha256": _array_sha256(state["thread_transverse_directions"]),
        "thread_longitudinal_periods_sha256": _array_sha256(state["thread_longitudinal_periods_s"]),
        "thread_transverse_periods_sha256": _array_sha256(state["thread_transverse_periods_s"]),
        "thread_prominence_heights_sha256": _array_sha256(state["thread_prominence_heights_m"]),
        "thread_gravity_sha256": _array_sha256(state["thread_gravity_m_s2"]),
        "thread_gravity_cutoff_periods_sha256": _array_sha256(
            state["thread_gravity_cutoff_periods_s"]
        ),
        "initial_thread_weights_sha256": _array_sha256(state["initial_thread_weights"]),
        "brownian_step_displacements_sha256": _array_sha256(
            np.stack([frame["brownian_step_displacements_km"] for frame in frames])
        ),
        "brownian_cumulative_displacements_sha256": _array_sha256(
            np.stack([frame["brownian_thread_displacements_km"] for frame in frames])
        ),
        "background_sha256": _array_sha256(state["background"]),
        "background_frames_sha256": (
            _array_sha256(state["background_frames"])
            if state["background_frames"] is not None
            else None
        ),
        "frame_0_sha256": _array_sha256(frames[0]["gong_image"]),
        "maximum_fixed_detector_residual": float(np.max(np.abs(state["fixed_detector_residual"]))),
        "resolution": {
            "highres_shape": list(highres_shape),
            "native_shape": list(native_shape),
            "highres_pixel_km": float(static_config["pixel_size_km"]),
            "native_pixel_km": float(static_config["pixel_size_km"])
            * int(static_config["downsample_factor"]),
            "gong_frames": "native",
            "filament_masks": "native",
            "soft_masks": "native_line_center",
            "observable_soft_masks": "native_passband_diluted",
            "tau_maps": "highres",
            "filament_masks_highres": "highres",
            "soft_masks_highres": "highres_line_center",
        },
        "oscillation_site": deepcopy(state["site"]),
    }


def simulate_filament_dynamics(
    initial_result: dict[str, object],
    dynamics_config: dict[str, object],
    *,
    background_frames: np.ndarray | None = None,
) -> dict[str, object]:
    """Generate a complete localized-advection sequence from frame zero."""
    state = initialize_dynamics(initial_result, dynamics_config, background_frames)
    frames = [current_dynamics_frame(state)]
    while int(state["current_index"]) + 1 < int(state["config"]["n_frames"]):
        frames.append(step_dynamics(state))

    return {
        "config": deepcopy(state["config"]),
        "static_config": deepcopy(state["static_config"]),
        "site": deepcopy(state["site"]),
        "spine": _copy_record(state["spine"]),
        "initial_threads": _copy_threads(state["initial_threads"]),
        "thread_longitudinal_directions": np.array(
            state["thread_longitudinal_directions"], copy=True
        ),
        "thread_transverse_directions": np.array(state["thread_transverse_directions"], copy=True),
        "thread_longitudinal_periods_s": np.array(
            state["thread_longitudinal_periods_s"], copy=True
        ),
        "thread_transverse_periods_s": np.array(state["thread_transverse_periods_s"], copy=True),
        "thread_prominence_heights_m": np.array(
            state["thread_prominence_heights_m"], copy=True
        ),
        "thread_gravity_m_s2": np.array(state["thread_gravity_m_s2"], copy=True),
        "thread_gravity_cutoff_periods_s": np.array(
            state["thread_gravity_cutoff_periods_s"], copy=True
        ),
        "initial_metadata": deepcopy(state["initial_metadata"]),
        "background": np.array(state["background"], copy=True),
        "background_frames": (
            None
            if state["background_frames"] is None
            else np.array(state["background_frames"], copy=True)
        ),
        "support": (None if state["support"] is None else np.array(state["support"], copy=True)),
        "frames": frames,
        "metadata": _sequence_metadata(state, frames),
    }
