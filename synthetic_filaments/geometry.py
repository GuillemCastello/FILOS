"""Procedural 3D filament skeleton: spine curve and dipped thread centerlines.

All coordinates are in km. Intrinsic ``(x, y)`` coordinates lie in the local
solar-surface plane and ``z`` is height above the photosphere. Each magnetic
dip is a constant-curvature circular segment in a vertical plane, following
the geometry of Luna et al. (2012, 2022).
"""

from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .config import (
    DIP_PARAMETERIZATION_CURVATURE_RADIUS,
    StaticConfig,
    image_height_km,
    image_width_km,
)

SPINE_LIBRARY_PATH = Path(__file__).with_name("data") / "spine_library.npz"
_SPINE_LIBRARY_CACHE: tuple[Path, tuple[int, int, int], dict[str, np.ndarray]] | None = None
Spine = dict[str, Any]
Thread = dict[str, Any]


def spine_total_length(spine: Spine) -> float:
    """Return the realized spine arclength in kilometres."""
    return float(spine["s"][-1])


def interpolate_spine(
    spine: Spine,
    s_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample spine position, tangent, and normal at arclengths ``s_query``."""
    x = np.interp(s_query, spine["s"], spine["x"])
    y = np.interp(s_query, spine["s"], spine["y"])
    tx = np.interp(s_query, spine["s"], spine["tangent_x"])
    ty = np.interp(s_query, spine["s"], spine["tangent_y"])
    norm = np.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    return x, y, tx, ty, -ty, tx


def effective_thread_radius(thread: Thread) -> np.ndarray:
    """Return the per-point thread radius, falling back to a scalar radius."""
    radius_along = thread.get("radius_along_km")
    if radius_along is not None and np.asarray(radius_along).size:
        return np.asarray(radius_along)
    return np.full(np.asarray(thread["s"]).shape, float(thread["radius_km"]))


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep on t assumed already clipped to [0, 1]: soft 0->1 ramp."""
    return t * t * (3.0 - 2.0 * t)


def _thread_xi(s: np.ndarray, length_km: float, monotonic: bool) -> np.ndarray:
    """Normalized position along a thread: 0 at the physically "central" point
    (dip bottom), 1 at the far ends.
    Shared by the radius taper and the opacity profile so both stay in
    lock-step along the same thread.
    """
    if monotonic:
        xi = s / max(length_km, 1.0)
    else:
        s_mid = s[-1] / 2.0
        xi = np.abs(s - s_mid) / max(s[-1] / 2.0, 1.0)
    return np.clip(xi, 0.0, 1.0)


def _tapered_radius(
    radius_km: float, s: np.ndarray, length_km: float, floor: float, monotonic: bool
) -> np.ndarray:
    """Elliptical radius taper: full radius at the center (or start), thinner ends."""
    xi = _thread_xi(s, length_km, monotonic)
    return radius_km * (floor + (1.0 - floor) * np.sqrt(1.0 - xi**2))


def _load_spine_library() -> dict[str, np.ndarray]:
    """Load and validate the committed spine library on first use."""
    global _SPINE_LIBRARY_CACHE
    path = Path(SPINE_LIBRARY_PATH).resolve()
    try:
        stat = path.stat()
    except OSError as error:
        raise FileNotFoundError(f"measured spine library is unavailable at {path}") from error
    signature = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    if (
        _SPINE_LIBRARY_CACHE is not None
        and _SPINE_LIBRARY_CACHE[0] == path
        and _SPINE_LIBRARY_CACHE[1] == signature
    ):
        return _SPINE_LIBRARY_CACHE[2]
    with np.load(path, allow_pickle=False) as data:
        required = {"points", "true_length_km", "source_file_id", "epoch", "format_version"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"measured spine library is missing arrays: {missing}")
        if int(data["format_version"]) != 1:
            raise ValueError(f"unsupported measured spine-library format: {data['format_version']}")
        points = np.asarray(data["points"], dtype=float)
        sources = np.asarray(data["source_file_id"], dtype=str)
        true_length = np.asarray(data["true_length_km"], dtype=float)
        epochs = np.asarray(data["epoch"], dtype=str)
    if points.ndim != 3 or points.shape[0] == 0 or points.shape[2] != 2:
        raise ValueError(f"measured spine points have invalid shape {points.shape}")
    entry_count = len(points)
    if sources.shape != (entry_count,):
        raise ValueError("measured spine source identifiers are not aligned with points")
    if true_length.shape != (entry_count,) or epochs.shape != (entry_count,):
        raise ValueError("measured spine lengths/epochs are not aligned with points")
    if not np.isfinite(points).all() or not np.isfinite(true_length).all():
        raise ValueError("measured spine library contains non-finite geometry")
    library = {
        "points": points,
        "source_file_id": sources,
        "true_length_km": true_length,
        "epoch": epochs,
    }
    _SPINE_LIBRARY_CACHE = (path, signature, library)
    return library


def _library_spine_points(
    library: dict[str, np.ndarray],
    config: StaticConfig,
    rng: np.random.Generator,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, str, int, float, str]:
    requested_index = config["spine_library_entry_index"]
    if requested_index is not None:
        if requested_index >= len(library["points"]):
            raise IndexError(
                "spine_library_entry_index is outside the committed library; "
                f"received {requested_index}, size={len(library['points'])}"
            )
        index = int(requested_index)
    else:
        theta = np.deg2rad(config["orientation_deg"])
        cosine, sine = np.cos(theta), np.sin(theta)
        points_all = library["points"]
        width = np.ptp(
            points_all[:, :, 0] * cosine - points_all[:, :, 1] * sine,
            axis=1,
        )
        height = np.ptp(
            points_all[:, :, 0] * sine + points_all[:, :, 1] * cosine,
            axis=1,
        )
        eligible = np.flatnonzero(
            (width <= 0.9 * image_width_km(config))
            & (height <= 0.9 * image_height_km(config))
            & (library["true_length_km"] >= config["spine_library_length_bounds_km"][0])
            & (library["true_length_km"] <= config["spine_library_length_bounds_km"][1])
        )
        if eligible.size == 0:
            raise ValueError(
                "no measured spine-library entry fits the requested field of view "
                "and configured length bounds"
            )
        index = int(rng.choice(eligible))
    points = np.array(library["points"][index], dtype=float, copy=True)
    points -= points.mean(axis=0)
    if rng.integers(2):
        points[:, 0] *= -1.0
    if rng.integers(2):
        points[:, 1] *= -1.0
    if requested_index is None:
        # A single-axis reflection changes the footprint after rotation.
        # The opposite reflection parity recovers the eligible footprint.
        rotated = points @ np.array([[cosine, sine], [-sine, cosine]])
        if (
            np.ptp(rotated[:, 0]) > 0.9 * image_width_km(config)
            or np.ptp(rotated[:, 1]) > 0.9 * image_height_km(config)
        ):
            points[:, 0] *= -1.0

    ds = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    cumulative = np.concatenate([[0.0], np.cumsum(ds)])
    if cumulative[-1] <= 0:
        raise ValueError("spine library entry has zero arclength")
    query = np.linspace(0.0, cumulative[-1], n_points)
    x = np.interp(query, cumulative, points[:, 0])
    y = np.interp(query, cumulative, points[:, 1])
    return (
        x,
        y,
        str(library["source_file_id"][index]),
        index,
        float(library["true_length_km"][index]),
        str(library["epoch"][index]),
    )


def make_spine(config: StaticConfig, rng: np.random.Generator) -> Spine:
    """Select, orient, and parameterize one unscaled measured spine."""
    n_pts = 400
    library = _load_spine_library()
    (
        x,
        y,
        source,
        library_index,
        source_true_length_km,
        source_epoch,
    ) = _library_spine_points(library, config, rng, n_pts)

    theta = np.deg2rad(config["orientation_deg"])
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = x * np.sin(theta) + y * np.cos(theta)

    xr += image_width_km(config) / 2.0
    yr += image_height_km(config) / 2.0
    if config["spine_library_entry_index"] is None:
        # Preserve mean-centering where it fits; otherwise translate the
        # bounding box just enough to retain the selection's 5% edge margin.
        for coordinates, extent in (
            (xr, image_width_km(config)),
            (yr, image_height_km(config)),
        ):
            shift = np.clip(
                0.0, 0.05 * extent - coordinates.min(), 0.95 * extent - coordinates.max()
            )
            coordinates += shift

    # Reparameterize by true arclength so thread sampling is uniform in s.
    ds = np.hypot(np.diff(xr), np.diff(yr))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    dx = np.gradient(xr, s)
    dy = np.gradient(yr, s)
    norm = np.hypot(dx, dy)
    tx, ty = dx / norm, dy / norm

    return {
        "s": s,
        "x": xr,
        "y": yr,
        "tangent_x": tx,
        "tangent_y": ty,
        "normal_x": -ty,
        "normal_y": tx,
        "model": "measured_library",
        "source": source,
        "library_index": library_index,
        "source_true_length_km": source_true_length_km,
        "source_epoch": source_epoch,
    }


def _smooth_noise(n: int, correlation_pts: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-amplitude smooth signal used by spine and population modulation."""
    if n < 4:
        return np.zeros(n)
    noise = rng.standard_normal(n)
    smooth = gaussian_filter1d(noise, sigma=max(correlation_pts, 1.0), mode="reflect")
    peak = np.max(np.abs(smooth))
    return smooth / peak if peak > 0 else smooth


def _sample_dip_curvature_radius(
    config: StaticConfig,
    rng: np.random.Generator,
) -> float:
    """Sample the explicitly configured bounded lognormal dip curvature."""
    radius_km = rng.lognormal(
        np.log(config["dip_curvature_radius_median_km"]),
        config["dip_curvature_radius_sigma_ln"],
    )
    return float(
        np.clip(
            radius_km,
            config["dip_curvature_radius_min_km"],
            config["dip_curvature_radius_max_km"],
        )
    )


# Lateral confinement of thread anchors to the local channel envelope:
# CHANNEL_CONFINEMENT_FACTOR sets how far (in channel half-widths) an anchor
# can move before the tanh saturates; CHANNEL_CONFINEMENT_FLOOR is the minimum
# envelope factor used even where the channel has tapered to near zero.
CHANNEL_CONFINEMENT_FACTOR = 1.6
CHANNEL_CONFINEMENT_FLOOR = 0.25
THREAD_HEIGHT_MIN_KM = 1_000.0
THREAD_HEIGHT_MAX_KM = 100_000.0
THREAD_LENGTH_SPINE_FRACTION_CAP = 0.95
# Execution controls; they do not change candidates or scientific parameters.
_SEGMENT_INDEX_REBUILD_THREADS = 32
_SEGMENT_DISTANCE_BLOCK_SIZE = 8192


def _circular_dip_parameters_from_radius(
    arc_length_km: float,
    curvature_radius_km: float,
) -> tuple[float, float, float]:
    """Return depth, half-angle, and chord length for a sampled curvature."""
    if not np.isfinite(arc_length_km) or arc_length_km <= 0.0:
        raise ValueError("circular-dip arc length must be finite and positive")
    if not np.isfinite(curvature_radius_km) or curvature_radius_km <= 0.0:
        raise ValueError("circular-dip curvature radius must be finite and positive")
    half_angle = arc_length_km / (2.0 * curvature_radius_km)
    if half_angle > 0.5 * np.pi:
        raise ValueError(
            "curvature radius is too small for a circular segment no deeper than a "
            "semicircle: require arc_length_km <= pi * curvature_radius_km"
        )
    depth_km = 2.0 * curvature_radius_km * np.sin(0.5 * half_angle) ** 2
    chord_length_km = 2.0 * curvature_radius_km * np.sin(half_angle)
    return float(depth_km), float(half_angle), float(chord_length_km)


def _segment_pair_distances(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> np.ndarray:
    """Exact distances between corresponding pairs of nonzero 3D segments."""
    u, v, w = b - a, d - c, a - c
    uu = np.einsum("ij,ij->i", u, u)
    vv = np.einsum("ij,ij->i", v, v)
    minimum = np.full(len(a), np.inf)
    for point, origin, direction, norm2 in (
        (a, c, v, vv), (b, c, v, vv), (c, a, u, uu), (d, a, u, uu),
    ):
        offset = point - origin
        t = np.clip(np.einsum("ij,ij->i", offset, direction) / norm2, 0.0, 1.0)
        delta = offset - t[:, None] * direction
        minimum = np.minimum(minimum, np.einsum("ij,ij->i", delta, delta))
    cross = np.cross(u, v)
    denominator = np.einsum("ij,ij->i", cross, cross)
    nonparallel = denominator > np.finfo(float).eps**2 * uu * vv
    s = np.zeros(len(a))
    t = np.zeros(len(a))
    np.divide(np.einsum("ij,ij->i", np.cross(v, w), cross), denominator,
              out=s, where=nonparallel)
    np.divide(np.einsum("ij,ij->i", np.cross(u, w), cross), denominator,
              out=t, where=nonparallel)
    interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    delta = w[interior] + s[interior, None] * u[interior] - t[interior, None] * v[interior]
    minimum[interior] = np.minimum(
        minimum[interior], np.einsum("ij,ij->i", delta, delta)
    )
    return np.sqrt(minimum)


def _nearby_centerline_distance(
    points: np.ndarray,
    segment_batches: list[tuple[cKDTree, np.ndarray, np.ndarray]],
    search_distance_km: float,
    maximum_spacing_km: float,
    *,
    reject_at_or_below_km: float = -np.inf,
) -> float:
    """Exact segment distance within the search radius; infinity beyond it.

    If a partial minimum is at most ``reject_at_or_below_km``, return that
    upper bound early. Callers may only use this for candidates that cannot
    meet separation or improve their already retained fallback.

    Segment midpoints are at most half a segment length from any point on
    their segment, so adding the maximum spacing makes this search complete.
    """
    a, b = points[:-1], points[1:]
    midpoints = 0.5 * (a + b)
    minimum = float("inf")
    for tree, starts, ends in segment_batches:
        # Midpoints lie on their segments, so their nearest distance also
        # bounds the true minimum. This greatly narrows searches in dense dips.
        midpoint_distances, nearest = tree.query(midpoints)
        midpoint_minimum = float(midpoint_distances.min())
        if midpoint_minimum <= reject_at_or_below_km:
            return midpoint_minimum
        # Exact distances to one nearby segment per candidate segment provide
        # a cheap upper bound. This frequently rejects a losing fallback
        # before constructing its much larger all-neighbor pair list.
        nearby_distances = _segment_pair_distances(a, b, starts[nearest], ends[nearest])
        nearby_minimum = float(nearby_distances.min())
        if nearby_minimum <= reject_at_or_below_km:
            return nearby_minimum
        upper = min(search_distance_km, minimum, nearby_minimum)
        neighbors = tree.query_ball_point(
            midpoints, upper + maximum_spacing_km, return_sorted=False,
        )
        counts = np.fromiter((len(row) for row in neighbors), dtype=int, count=len(a))
        if counts.sum() == 0:
            continue
        candidate_indices = np.repeat(np.arange(len(a)), counts)
        existing_indices = np.concatenate(neighbors).astype(int)
        for offset in range(0, len(existing_indices), _SEGMENT_DISTANCE_BLOCK_SIZE):
            block = slice(offset, offset + _SEGMENT_DISTANCE_BLOCK_SIZE)
            candidate_block = candidate_indices[block]
            existing_block = existing_indices[block]
            distances = _segment_pair_distances(
                a[candidate_block], b[candidate_block],
                starts[existing_block], ends[existing_block],
            )
            minimum = min(minimum, float(distances.min()))
            if minimum <= reject_at_or_below_km:
                return minimum
    return minimum


def _minimum_interthread_probe_distance(placed_points: list[np.ndarray]) -> float | None:
    """Measure the exact realized minimum between probes from distinct threads."""
    if len(placed_points) < 2:
        return None
    points = np.vstack(placed_points)
    thread_ids = np.repeat(np.arange(len(placed_points)), 3)
    distances, neighbors = cKDTree(points).query(points, k=4)
    belongs_to_other_thread = thread_ids[neighbors] != thread_ids[:, None]
    return float(np.where(belongs_to_other_thread, distances, np.inf).min())


def _circular_dip_centerline(
    *,
    anchor_x_km: float,
    anchor_y_km: float,
    direction_x: float,
    direction_y: float,
    arc_length_km: float,
    bottom_height_km: float,
    curvature_radius_km: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct a straight-plan-view, constant-curvature vertical dip."""
    _depth_km, half_angle, _chord_length_km = _circular_dip_parameters_from_radius(
        arc_length_km,
        curvature_radius_km,
    )
    angle = np.linspace(-half_angle, half_angle, n_points)
    along_chord_km = curvature_radius_km * np.sin(angle)
    x = anchor_x_km + along_chord_km * direction_x
    y = anchor_y_km + along_chord_km * direction_y
    # Stable form of R (1 - cos(angle)) for shallow arcs.
    z = bottom_height_km + 2.0 * curvature_radius_km * np.sin(0.5 * angle) ** 2
    s = curvature_radius_km * (angle + half_angle)
    return x, y, z, s, along_chord_km, curvature_radius_km


def make_threads(
    config: StaticConfig,
    spine: Spine,
    rng: np.random.Generator,
    *,
    center_positions_km: np.ndarray | None = None,
    transverse_offsets_km: np.ndarray | None = None,
    pitch_means_deg: np.ndarray | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> tuple[list[Thread], dict[str, float | int | None]]:
    """Generate straight-plan-view threads with circular vertical dips.

    Each thread is anchored to the measured spine. Its fixed image-plane
    direction is obtained by rotating the local spine tangent at that anchor
    through the sampled shear angle. The vertical coordinate is a
    constant-curvature circular segment following Luna et al. (2012, 2022).

    Candidate threads retain the WPFS-style minimum 3D centerline separation.
    If no placement clears that distance in eight attempts, the
    most-separated attempt is retained and counted in the returned placement
    diagnostics.
    """
    threads: list[Thread] = []
    total = spine_total_length(spine)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"thread placement requires a finite positive spine length; received {total!r} km"
        )
    if config["n_threads"] < 1:
        raise ValueError("thread placement requires n_threads >= 1")

    requested_minimum_length = float(config["thread_length_min_km"])
    requested_maximum_length = float(config["thread_length_max_km"])
    spine_length_cap = THREAD_LENGTH_SPINE_FRACTION_CAP * total
    maximum_thread_length = min(requested_maximum_length, spine_length_cap)
    minimum_thread_length = min(requested_minimum_length, maximum_thread_length)
    if maximum_thread_length < 2.0 * config["thread_point_spacing_km"]:
        raise ValueError(
            "thread_point_spacing_km is too coarse for every effective thread "
            "length: require 2 * thread_point_spacing_km <= effective maximum; "
            f"received spacing={config['thread_point_spacing_km']:.6g} km and "
            f"effective maximum={maximum_thread_length:.6g} km"
        )
    supported_maximum_length = np.pi * config["dip_curvature_radius_min_km"]
    if minimum_thread_length > supported_maximum_length:
        raise ValueError(
            "the effective thread-length lower bound is incompatible with the "
            "curvature-radius range: require effective minimum length <= pi * "
            "dip_curvature_radius_min_km; "
            f"received effective minimum={minimum_thread_length:.6g} km, "
            "dip_curvature_radius_min_km="
            f"{config['dip_curvature_radius_min_km']:.6g} km, and supported "
            f"maximum={supported_maximum_length:.6g} km. Increase the minimum "
            "curvature radius or reduce the requested thread-length minimum."
        )
    half_width = config["spine_width_km"] / 2.0
    if center_positions_km is not None:
        center_positions_km = np.asarray(center_positions_km, dtype=float)
        if center_positions_km.shape != (config["n_threads"],):
            raise ValueError(
                "center_positions_km must contain one value per configured "
                f"thread; expected {(config['n_threads'],)}, received "
                f"{center_positions_km.shape}"
            )
        if (
            not np.isfinite(center_positions_km).all()
            or np.any(center_positions_km < 0.0)
            or np.any(center_positions_km > total)
        ):
            raise ValueError("center_positions_km must be finite and lie on the spine")
    for name, values in (
        ("transverse_offsets_km", transverse_offsets_km),
        ("pitch_means_deg", pitch_means_deg),
    ):
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != (config["n_threads"],):
            raise ValueError(
                f"{name} must contain one value per configured thread; "
                f"expected {(config['n_threads'],)}, received {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        if name == "transverse_offsets_km":
            transverse_offsets_km = values
        else:
            pitch_means_deg = values

    # Shared low-frequency width modulation along the spine.
    n_mod = 256
    mod_noise = _smooth_noise(
        n_mod,
        config["width_modulation_correlation"] * n_mod,
        rng,
    )
    mod_s = np.linspace(0.0, total, n_mod)
    width_mod = 1.0 + config["width_modulation_amplitude"] * mod_noise

    def channel_envelope(
        s_pos: float | np.ndarray,
    ) -> float | np.ndarray:
        positions = np.asarray(s_pos, dtype=float)
        foot = np.sqrt(np.maximum(1.0 - (2.0 * positions / total - 1.0) ** 2, 0.0))
        envelope = np.maximum(foot, 0.15) * np.interp(
            positions,
            mod_s,
            width_mod,
        )
        return float(envelope) if envelope.ndim == 0 else envelope

    def confinement_limit(
        s_pos: float | np.ndarray,
    ) -> float | np.ndarray:
        limit = (
            CHANNEL_CONFINEMENT_FACTOR
            * np.maximum(channel_envelope(s_pos), CHANNEL_CONFINEMENT_FLOOR)
            * half_width
        )
        return float(limit) if np.ndim(limit) == 0 else limit

    def sample_thread_center(_thread_index: int) -> float:
        return float(rng.uniform(0.05 * total, 0.95 * total))

    if center_positions_km is not None:

        def sample_thread_center(thread_index: int) -> float:
            return float(center_positions_km[thread_index])

    mean_radius = 0.5 * (config["thread_radius_min_km"] + config["thread_radius_max_km"])
    min_sep = config["thread_separation_radii"] * mean_radius
    placed_points: list[np.ndarray] = []
    segment_batches: list[tuple[cKDTree, np.ndarray, np.ndarray]] = []
    # Accepted centerlines only append to these buffers. Existing tree views
    # therefore remain valid until the next complete rebuild.
    maximum_segments = max(
        int(np.ceil(maximum_thread_length / config["thread_point_spacing_km"])) + 1,
        4,
    )
    starts_buffer = np.empty((config["n_threads"] * maximum_segments, 3))
    ends_buffer = np.empty_like(starts_buffer)
    segment_count = 0
    indexed_segment_count = 0
    pending_thread_count = 0
    achieved_minimum = float("inf")
    placement_attempts = 0
    relaxed_placements = 0
    last_progress = monotonic()

    def report_progress(completed: int, *, force: bool = False) -> None:
        nonlocal last_progress
        now = monotonic()
        if progress_callback is not None and (
            force or completed % 25 == 0 or now - last_progress >= 0.5
        ):
            progress_callback(
                completed,
                int(config["n_threads"]),
                placement_attempts,
                relaxed_placements,
            )
            last_progress = now

    for thread_index in range(config["n_threads"]):
        candidate = None
        fallback = None
        best_min_dist = -np.inf
        sampled_dip_radius = _sample_dip_curvature_radius(config, rng)
        thread_maximum_length = min(maximum_thread_length, np.pi * sampled_dip_radius)
        for _attempt in range(8):
            placement_attempts += 1
            requested_arc_length = rng.uniform(
                minimum_thread_length,
                thread_maximum_length,
            )
            pitch_mean = (
                float(pitch_means_deg[thread_index])
                if pitch_means_deg is not None
                else config["chirality"] * config["thread_pitch_mean_deg"]
            )
            pitch_deg = rng.normal(pitch_mean, config["thread_pitch_std_deg"])
            pitch_rad = np.deg2rad(pitch_deg)
            cos_pitch = float(np.cos(pitch_rad))
            sin_pitch = float(np.sin(pitch_rad))
            dip_radius = sampled_dip_radius
            dip_depth, half_angle, chord_length = _circular_dip_parameters_from_radius(
                requested_arc_length,
                dip_radius,
            )

            # Keep the chord's tangent-direction projection within the measured
            # spine interval. The dip itself remains one globally straight line
            # in x-y; the spine supplies only its anchor and local orientation.
            spine_span = min(
                chord_length * abs(cos_pitch),
                total,
            )
            s_center = sample_thread_center(thread_index)
            s_start = s_center - spine_span / 2.0
            s_end = s_center + spine_span / 2.0
            # Shift an endpoint-crossing interval back onto the spine instead
            # of clipping it shorter.  The sampled thread-length bounds refer
            # to realized threads, including anchors near either foot.
            if s_start < 0.0:
                s_start = 0.0
                s_end = spine_span
            elif s_end > total:
                s_start = total - spine_span
                s_end = total
            # The physical anchor is the centre of the realized interval.
            # Endpoint shifting preserves length but can move that centre away
            # from the originally sampled position.
            s_center = 0.5 * (s_start + s_end)
            if requested_arc_length < 2.0 * config["thread_point_spacing_km"]:
                continue

            envelope = channel_envelope(s_center)
            u_offset = (
                float(transverse_offsets_km[thread_index])
                if transverse_offsets_km is not None
                else rng.normal(0.0, envelope * config["spine_width_km"] / 3.0)
            )
            height = float(
                np.clip(
                    rng.normal(
                        config["height_mean_km"],
                        config["height_std_km"],
                    ),
                    THREAD_HEIGHT_MIN_KM,
                    THREAD_HEIGHT_MAX_KM,
                )
            )

            sx, sy, tx, ty, nx, ny = (
                float(value[0])
                for value in interpolate_spine(
                    spine,
                    np.asarray([s_center], dtype=float),
                )
            )
            u_limit = float(confinement_limit(s_center))
            realized_offset = float(u_limit * np.tanh(u_offset / u_limit))
            anchor_x = sx + realized_offset * nx
            anchor_y = sy + realized_offset * ny
            direction_x = cos_pitch * tx + sin_pitch * nx
            direction_y = cos_pitch * ty + sin_pitch * ny

            # Three equally spaced arc probes represent the actual circular
            # centerline used below, including its vertical displacement.
            probe_angle = half_angle * np.asarray([-0.5, 0.0, 0.5])
            probe_along = dip_radius * np.sin(probe_angle)
            probe_height = height + 2.0 * dip_radius * np.sin(0.5 * probe_angle) ** 2
            probes = np.column_stack(
                [
                    anchor_x + probe_along * direction_x,
                    anchor_y + probe_along * direction_y,
                    probe_height,
                ]
            )
            n_pts = max(
                int(np.ceil(requested_arc_length / config["thread_point_spacing_km"])) + 1,
                5,
            )
            if n_pts % 2 == 0:
                n_pts += 1
            centerline = _circular_dip_centerline(
                anchor_x_km=anchor_x,
                anchor_y_km=anchor_y,
                direction_x=direction_x,
                direction_y=direction_y,
                arc_length_km=requested_arc_length,
                bottom_height_km=height,
                curvature_radius_km=dip_radius,
                n_points=n_pts,
            )
            points = np.column_stack(centerline[:3])
            candidate_min_dist = _nearby_centerline_distance(
                points, segment_batches, max(min_sep, achieved_minimum),
                config["thread_point_spacing_km"],
                reject_at_or_below_km=best_min_dist,
            )

            candidate_data = (
                requested_arc_length,
                s_center,
                realized_offset,
                height,
                pitch_deg,
                cos_pitch,
                anchor_x,
                anchor_y,
                direction_x,
                direction_y,
                dip_depth,
                dip_radius,
                probes,
                centerline,
                candidate_min_dist,
            )
            if candidate_min_dist >= min_sep:
                candidate = candidate_data
                break
            if candidate_min_dist > best_min_dist:
                fallback = candidate_data
                best_min_dist = candidate_min_dist

        if candidate is None:
            candidate = fallback
            if fallback is not None:
                relaxed_placements += 1
        if candidate is None:
            report_progress(thread_index + 1)
            continue  # every attempt had a degenerate (too-short) length
        (
            requested_arc_length,
            s_center,
            realized_offset,
            height,
            pitch_deg,
            cos_pitch,
            anchor_x,
            anchor_y,
            direction_x,
            direction_y,
            dip_depth,
            dip_radius,
            probes,
            centerline,
            candidate_min_dist,
        ) = candidate
        placed_points.append(probes)
        achieved_minimum = min(achieved_minimum, candidate_min_dist)
        x, y, z, s_local, along_chord, dip_radius = centerline
        points = np.column_stack((x, y, z))
        next_segment_count = segment_count + len(points) - 1
        starts_buffer[segment_count:next_segment_count] = points[:-1]
        ends_buffer[segment_count:next_segment_count] = points[1:]
        segment_count = next_segment_count
        pending_thread_count += 1
        # Rebuild the complete index in batches; index recent placements
        # separately so every accepted thread participates immediately.
        if pending_thread_count == _SEGMENT_INDEX_REBUILD_THREADS:
            starts = starts_buffer[:segment_count]
            ends = ends_buffer[:segment_count]
            segment_batches = [(cKDTree(0.5 * (starts + ends)), starts, ends)]
            indexed_segment_count = segment_count
            pending_thread_count = 0
        else:
            starts = starts_buffer[indexed_segment_count:segment_count]
            ends = ends_buffer[indexed_segment_count:segment_count]
            batch = (cKDTree(0.5 * (starts + ends)), starts, ends)
            if indexed_segment_count == 0:
                segment_batches = [batch]
            else:
                segment_batches = [segment_batches[0], batch]
        s_spine = np.clip(
            s_center + along_chord * cos_pitch,
            0.0,
            total,
        )
        thread_length = float(s_local[-1])

        radius = rng.uniform(
            config["thread_radius_min_km"],
            config["thread_radius_max_km"],
        )

        threads.append(
            {
                "s": s_local,
                "x": x,
                "y": y,
                "z": z,
                "radius_km": radius,
                "height_km": height,
                "dip_depth_km": dip_depth,
                "dip_radius_km": dip_radius,
                "length_km": thread_length,
                "s_spine": s_spine,
                "spine_anchor_km": s_center,
                "transverse_offset_km": realized_offset,
                "pitch_deg": pitch_deg,
                "tau0": 0.0,
                "fill_depth_km": 0.0,
                "filled_depth_km": 0.0,
                "fill_fraction": 1.0,
                "realized_fill_fraction": 1.0,
                "tau_along": np.array([]),
                "radius_along_km": _tapered_radius(
                    radius,
                    s_local,
                    thread_length,
                    config["endpoint_radius_floor"],
                    monotonic=False,
                ),
            }
        )
        report_progress(thread_index + 1)

    report_progress(int(config["n_threads"]), force=True)

    placement = {
        "dip_parameterization": DIP_PARAMETERIZATION_CURVATURE_RADIUS,
        "requested_thread_length_bounds_km": [
            requested_minimum_length,
            requested_maximum_length,
        ],
        "effective_thread_length_bounds_km": [
            minimum_thread_length,
            maximum_thread_length,
        ],
        "realized_thread_length_range_km": [
            float(min(thread["length_km"] for thread in threads)),
            float(max(thread["length_km"] for thread in threads)),
        ],
        "thread_length_spine_fraction_cap": THREAD_LENGTH_SPINE_FRACTION_CAP,
        "thread_length_spine_cap_km": float(spine_length_cap),
        "thread_length_definition": "three_dimensional_circular_arc_length",
        "thread_length_cap_explanation": (
            "The effective maximum arc length is the smaller of the requested "
            "maximum and 95% of the realized spine arclength."
        ),
        "thread_length_minimum_capped": bool(minimum_thread_length < requested_minimum_length),
        "thread_length_maximum_capped": bool(maximum_thread_length < requested_maximum_length),
        "requested_separation_km": float(min_sep),
        "achieved_minimum_centerline_separation_km": (
            float(achieved_minimum) if np.isfinite(achieved_minimum) else None
        ),
        "achieved_minimum_probe_separation_km": _minimum_interthread_probe_distance(
            placed_points
        ),
        "placement_attempts": placement_attempts,
        "relaxed_placements": relaxed_placements,
    }
    return threads, placement
