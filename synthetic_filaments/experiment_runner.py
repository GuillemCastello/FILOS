"""Functional preview and background-job orchestration for local experiments."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np

from .cache import stage_cache
from .dynamic_background import (
    load_h5_background_sequence,
)
from .experiment_config import (
    DEFAULT_EXPERIMENTS_ROOT,
    HEINZEL_EXTENSION_PATH,
    SPINE_LIBRARY_PATH,
    STATIC_FORWARD_MODEL_REVISION,
    experiment_config_sha256,
    file_sha256,
    load_experiment_config,
    save_experiment_config,
    static_preview_fingerprint,
    validate_preview_config,
    validate_production_config,
)
from .generator import (
    assign_static_plasma,
    generate_static_geometry,
    make_h5_static_config,
    render_static_state,
)
from .io import load_static_result, save_static_result
from .oscillation import (
    gravity_cutoff_period_s,
    luna_2022_longitudinal_period_s,
    solar_gravity_m_s2,
)
from .paths import PROJECT_ROOT
from .simulation_io import rebuild_simulation_index, simulate_and_save_filament_dynamics
from .video import (
    VELOCITY_VIDEO_RENDER_VERSION,
    save_gong_video,
    save_velocity_video,
)

StatusCallback = Callable[[str, Mapping[str, Any]], None]
PREVIEW_RENDER_VERSION = 7
STATIC_STATE_SNAPSHOT_VERSION = 1

_GEOMETRY_STATIC_FIELDS = {
    "spine_library_entry_index",
    "spine_library_length_bounds_km",
    "spine_width_km",
    "thread_density_per_mm",
    "thread_count_cap",
    "thread_length_min_km",
    "thread_length_max_km",
    "thread_radius_min_km",
    "thread_radius_max_km",
    "thread_pitch_mean_deg",
    "thread_pitch_std_deg",
    "height_mean_km",
    "height_std_km",
    "dip_curvature_radius_min_km",
    "dip_curvature_radius_max_km",
    "dip_curvature_radius_median_km",
    "dip_curvature_radius_sigma_ln",
    "thread_separation_radii",
    "thread_point_spacing_km",
    "endpoint_radius_floor",
    "width_modulation_amplitude",
    "width_modulation_correlation",
}
_PLASMA_STATIC_FIELDS = {
    "temp_center_K",
    "temp_tr_K",
    "pctr_gamma",
    "ionization_center",
    "transition_pressure_dyn_cm2",
    "column_mass_g_cm2",
    "microturbulent_velocity_kms",
    "halpha_wavelength_angstrom",
    "foot_taper_fraction",
}
_OBSERVATION_STATIC_FIELDS = {
    "gong_bandpass_line_fraction",
    "mask_tau_threshold",
    "psf_sigma_px",
}


def _json_ready(value: Any) -> Any:
    """Convert nested scientific values into JSON-compatible values."""
    if isinstance(value, Mapping):
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Write one JSON object atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _array_sha256(array: np.ndarray) -> str:
    """Hash array dtype, shape, and contiguous bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


def _frame_zero_comparison_figure(initial_result: Mapping[str, Any]) -> Any:
    """Build the raw/background-composite comparison figure in memory."""
    import matplotlib.pyplot as plt

    arrays = initial_result["arrays"]
    background = np.asarray(arrays["background"], dtype=float)
    final_image = np.asarray(arrays["degraded_intensity"], dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.25), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (background, final_image),
        ("Raw HDF5 background", "Background + synthetic filament"),
        strict=True,
    ):
        axis.imshow(image, origin="lower", cmap="gray")
        axis.set_title(title)
        axis.set_axis_off()
    return figure


def _geometry_diagnostics_figure(initial_result: Mapping[str, Any]) -> Any:
    """Build spine, thread-profile, and realized-distribution diagnostics."""
    import matplotlib.pyplot as plt

    threads = initial_result["threads"]
    if not threads:
        raise ValueError("geometry preview requires at least one realized thread")
    sample_indices = np.unique(np.linspace(0, len(threads) - 1, min(70, len(threads)), dtype=int))
    figure, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), constrained_layout=True)

    for index in sample_indices:
        thread = threads[int(index)]
        axes[0, 0].plot(
            np.asarray(thread["x"]) / 1_000.0,
            np.asarray(thread["y"]) / 1_000.0,
            color="#31688E",
            alpha=0.22,
            linewidth=0.8,
        )
    spine = initial_result["spine"]
    axes[0, 0].plot(
        np.asarray(spine["x"]) / 1_000.0,
        np.asarray(spine["y"]) / 1_000.0,
        color="#FDE725",
        linewidth=2.2,
        label="Measured spine",
    )
    axes[0, 0].set(
        title=f"Spine + {len(sample_indices)} sampled threads",
        xlabel="x [Mm]",
        ylabel="y [Mm]",
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(loc="best")

    for index in sample_indices[: min(40, len(sample_indices))]:
        thread = threads[int(index)]
        s = np.asarray(thread["s"], dtype=float)
        axes[0, 1].plot(
            s / max(float(s[-1]), 1.0),
            np.asarray(thread["z"]) / 1_000.0,
            color="#35B779",
            alpha=0.25,
            linewidth=0.8,
        )
    axes[0, 1].set(
        title="Magnetic-dip profiles",
        xlabel="Normalized thread arclength",
        ylabel="Height [Mm]",
    )

    distributions = (
        ("height_km", 1.0 / 1_000.0, "Dip-bottom height [Mm]"),
        ("length_km", 1.0 / 1_000.0, "Thread length [Mm]"),
        ("radius_km", 1.0, "Thread radius [km]"),
        ("dip_depth_km", 1.0, "Dip depth [km]"),
    )
    histogram_axes = (axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    color = plt.colormaps["viridis"](0.58)
    for axis, (name, scale, label) in zip(histogram_axes, distributions, strict=True):
        values = np.asarray([thread[name] for thread in threads], dtype=float) * scale
        axis.hist(values, bins=35, color=color, edgecolor="white", linewidth=0.25)
        axis.axvline(np.median(values), color="#D55E00", linewidth=1.5, label="Median")
        axis.set(title=label, xlabel=label, ylabel="Threads")
        axis.legend(loc="best")
    return figure


def _luna_preview_data(initial_result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Evaluate Luna periods from every realized thread radius and height."""
    threads = initial_result["threads"]
    if not threads:
        raise ValueError("Luna preview requires at least one realized thread")
    radii_m = 1_000.0 * np.asarray(
        [thread["dip_radius_km"] for thread in threads], dtype=float
    )
    heights_m = 1_000.0 * np.asarray(
        [thread["height_km"] for thread in threads], dtype=float
    )
    return {
        "curvature_radius_m": radii_m,
        "prominence_height_m": heights_m,
        "gravity_m_s2": np.asarray(solar_gravity_m_s2(heights_m), dtype=float),
        "gravity_cutoff_period_s": np.asarray(
            gravity_cutoff_period_s(heights_m), dtype=float
        ),
        "expected_period_s": np.asarray(
            luna_2022_longitudinal_period_s(radii_m, heights_m), dtype=float
        ),
    }


def _summary_triplet(values: np.ndarray) -> dict[str, float]:
    """Return minimum, median, and maximum for one finite array."""
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _luna_preview_summary(
    data: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Summarize the realized inputs and outputs shown in the Luna figure."""
    return {
        "prominence_height_m": _summary_triplet(data["prominence_height_m"]),
        "curvature_radius_m": _summary_triplet(data["curvature_radius_m"]),
        "gravity_m_s2": _summary_triplet(data["gravity_m_s2"]),
        "gravity_cutoff_period_s": _summary_triplet(
            data["gravity_cutoff_period_s"]
        ),
        "expected_period_s": _summary_triplet(data["expected_period_s"]),
    }


def _luna_diagnostics_figure(
    initial_result: Mapping[str, Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    """Build realized radius-height Luna diagnostics in memory."""
    import matplotlib.pyplot as plt

    data = _luna_preview_data(initial_result)
    radii_mm = data["curvature_radius_m"] / 1.0e6
    heights_mm = data["prominence_height_m"] / 1.0e6
    gravity = data["gravity_m_s2"]
    cutoff_min = data["gravity_cutoff_period_s"] / 60.0
    periods_min = data["expected_period_s"] / 60.0
    sample_indices = np.unique(
        np.linspace(0, periods_min.size - 1, min(5_000, periods_min.size), dtype=int)
    )

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
    scatter_radius = axes[0, 0].scatter(
        radii_mm[sample_indices],
        periods_min[sample_indices],
        c=heights_mm[sample_indices],
        cmap="viridis",
        s=8,
        alpha=0.55,
        linewidths=0.0,
        rasterized=True,
    )
    axes[0, 0].set(
        title="Expected period by dip curvature (log scale)",
        xlabel="Dip curvature radius [Mm]",
        ylabel="Luna period [min]",
    )
    axes[0, 0].set_xscale("log")
    figure.colorbar(scatter_radius, ax=axes[0, 0], label="Dip-bottom height [Mm]")

    scatter_height = axes[0, 1].scatter(
        heights_mm[sample_indices],
        periods_min[sample_indices],
        c=np.log10(radii_mm[sample_indices]),
        cmap="viridis",
        s=8,
        alpha=0.55,
        linewidths=0.0,
        rasterized=True,
    )
    axes[0, 1].set(
        title="Expected period by thread height",
        xlabel="Dip-bottom height [Mm]",
        ylabel="Luna period [min]",
    )
    figure.colorbar(scatter_height, ax=axes[0, 1], label=r"$\log_{10}(R/\mathrm{Mm})$")

    color = plt.colormaps["viridis"](0.58)
    axes[1, 0].hist(
        periods_min,
        bins=40,
        color=color,
        edgecolor="white",
        linewidth=0.25,
    )
    axes[1, 0].axvline(
        np.median(periods_min), color="#D55E00", linewidth=1.5, label="Median"
    )
    axes[1, 0].set(
        title="Expected longitudinal periods",
        xlabel="Luna period [min]",
        ylabel="Threads",
    )
    axes[1, 0].legend(loc="best")

    summary_values = (
        ("Height [Mm]", heights_mm),
        (r"$g(h)$ [m s$^{-2}$]", gravity),
        (r"$P_{\rm cut,g}$ [min]", cutoff_min),
        ("Curvature R [Mm]", radii_mm),
        ("Expected P [min]", periods_min),
    )
    cell_text = [
        [f"{np.min(values):.3g}", f"{np.median(values):.3g}", f"{np.max(values):.3g}"]
        for _label, values in summary_values
    ]
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Realized-thread marginal summary")
    table = axes[1, 1].table(
        cellText=cell_text,
        rowLabels=[label for label, _values in summary_values],
        colLabels=["Minimum", "Median", "Maximum"],
        cellLoc="right",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    for axis in axes.flat[:3]:
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.6)
    figure.suptitle("Height-dependent Luna corrected-pendulum diagnostics")

    return figure, _luna_preview_summary(data)


def _figure_png_bytes(figure: Any) -> bytes:
    """Render and close one publication-quality figure without filesystem I/O."""
    import matplotlib.pyplot as plt

    output = io.BytesIO()
    try:
        figure.savefig(output, format="png", dpi=300, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(figure)


def _load_h5_backgrounds(
    resolved: Mapping[str, Any],
    *,
    n_frames: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    cached_stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the deterministic aligned crop requested by one experiment."""
    background_arguments = dict(resolved["dynamic_background"])
    return load_h5_background_sequence(
        resolved["inputs"]["h5_background_path"],
        n_frames=n_frames,
        seed=int(resolved["background_seed"]),
        progress_callback=progress_callback,
        cached_stages=cached_stages,
        **background_arguments,
    )


def _cache_key(value: object) -> str:
    """Return one deterministic in-process stage cache key."""
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _cache_input_stamps(source: Path) -> dict[str, Any]:
    """Extra runtime invalidation, independent of persisted scientific fingerprints."""
    return {
        "files": {
            str(path.resolve()): path.stat().st_ctime_ns
            for path in (source, SPINE_LIBRARY_PATH, HEINZEL_EXTENSION_PATH)
        },
    }


def _cached_stage(
    cache: dict[str, Any],
    name: str,
    key: str,
) -> Any | None:
    """Look up any retained realization with matching stage dependencies."""
    return stage_cache(cache).get(name, key)


def _store_stage(cache: dict[str, Any], name: str, key: str, value: Any) -> None:
    """Retain recently used realizations within the process RAM budget."""
    stage_cache(cache).put(name, key, value)


def _load_preview_source(config_or_directory: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Resolve either an in-memory draft or a legacy experiment directory."""
    if isinstance(config_or_directory, Mapping):
        return deepcopy(dict(config_or_directory))
    source = Path(config_or_directory).resolve()
    if source.is_dir():
        source = source / "experiment.toml"
    return load_experiment_config(source, check_inputs=False)


def generate_experiment_preview(
    config_or_directory: Mapping[str, Any] | str | Path,
    *,
    cached_stages: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Calculate a static preview entirely in memory from the current draft."""
    started = time.monotonic()
    stage_durations: dict[str, float] = {}
    events: list[dict[str, Any]] = []

    def report(event: Mapping[str, Any]) -> None:
        value = dict(event)
        value["overall_elapsed_seconds"] = time.monotonic() - started
        events.append(value)
        completed = value.get("completed")
        total = value.get("total")
        if completed is not None and total is not None and completed >= total:
            stage_durations[str(value["stage"])] = float(value["elapsed_seconds"])
        if progress_callback is not None:
            progress_callback(value)

    def complete_stage(
        stage: str,
        stage_started: float,
        description: str,
        *,
        cached: bool = False,
        **details: Any,
    ) -> None:
        elapsed = 0.0 if cached else time.monotonic() - stage_started
        report(
            {
                "stage": stage,
                "completed": 1,
                "total": 1,
                "elapsed_seconds": elapsed,
                "description": description,
                "cached": cached,
                **details,
            }
        )

    report(
        {
            "stage": "validation",
            "completed": 0,
            "total": 1,
            "elapsed_seconds": 0.0,
            "description": "Validating static preview inputs and source assets.",
        }
    )
    user_config = _load_preview_source(config_or_directory)
    cache = {} if cached_stages is None else cached_stages
    resolved = validate_preview_config(user_config, cached_stages=cache)
    fingerprint = static_preview_fingerprint(user_config)
    complete_stage(
        "validation",
        started,
        "Static preview inputs and source assets are valid.",
    )

    source_path = Path(resolved["inputs"]["h5_background_path"])
    source_stat = source_path.stat()
    input_stamps = _cache_input_stamps(source_path)
    background_key = _cache_key(
        {
            "path": source_path,
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "ctime_ns": source_stat.st_ctime_ns,
            "background_seed": resolved["background_seed"],
            "loader": resolved["dynamic_background"],
        }
    )
    backgrounds = _cached_stage(cache, "background", background_key)
    if backgrounds is None:
        backgrounds = _load_h5_backgrounds(
            resolved, n_frames=1, progress_callback=report, cached_stages=cache,
        )
        _store_stage(cache, "background", background_key, backgrounds)
    else:
        complete_stage("background", time.monotonic(), "Background selection reused.", cached=True)

    static_config = make_h5_static_config(
        backgrounds,
        seed=int(resolved["static_seed"]),
        config_overrides=resolved["static_overrides"],
    )
    geometry_key = _cache_key(
        {
            "model_revision": STATIC_FORWARD_MODEL_REVISION,
            "intrinsic_frame": {
                name: static_config[name]
                for name in ("nx", "ny", "pixel_size_km", "orientation_deg", "chirality")
            },
            "spine_library": resolved["input_identities"]["spine_library"],
            "spine_ctime_ns": input_stamps["files"][str(SPINE_LIBRARY_PATH.resolve())],
            "static_seed": resolved["static_seed"],
            "static": {
                name: static_config[name] for name in sorted(_GEOMETRY_STATIC_FIELDS)
            },
        }
    )
    geometry = _cached_stage(cache, "geometry", geometry_key)
    if geometry is None:
        geometry = generate_static_geometry(static_config, progress_callback=report)
        _store_stage(cache, "geometry", geometry_key, geometry)
    else:
        complete_stage("spine", time.monotonic(), "Measured spine reused.", cached=True)
        complete_stage("geometry", time.monotonic(), "Thread geometry reused.", cached=True)

    plasma_key = _cache_key(
        {
            "geometry_key": geometry_key,
            "opacity_table": resolved["input_identities"]["opacity_table"],
            "opacity_ctime_ns": input_stamps["files"][str(HEINZEL_EXTENSION_PATH.resolve())],
            "static": {name: static_config[name] for name in sorted(_PLASMA_STATIC_FIELDS)},
        }
    )
    plasma = _cached_stage(cache, "plasma", plasma_key)
    if plasma is None:
        geometry_for_plasma = dict(geometry)
        geometry_for_plasma["config"] = static_config
        plasma = assign_static_plasma(geometry_for_plasma, progress_callback=report)
        _store_stage(cache, "plasma", plasma_key, plasma)
    else:
        complete_stage("plasma", time.monotonic(), "Plasma and opacity state reused.", cached=True)

    background_hash = _array_sha256(np.asarray(backgrounds["frames"])[0])
    optical_depth_key = _cache_key(
        {
            "plasma_key": plasma_key,
            "projection": {
                name: static_config[name]
                for name in ("nx", "ny", "pixel_size_km", "disk_mu", "limb_direction_deg")
            },
            "rasterization": "pixel_integrated_gaussian_sigma_radius_over_2",
        }
    )
    render_key = _cache_key(
        {
            "optical_depth_key": optical_depth_key,
            "background_key": background_key,
            "background_sha256": background_hash,
            "static": {
                name: static_config[name] for name in sorted(_OBSERVATION_STATIC_FIELDS)
            },
            "source_function_policy": "heinzel2015_published_height_interpolation",
        }
    )
    initial_result = _cached_stage(cache, "render", render_key)
    if initial_result is None:
        tau_map = _cached_stage(cache, "optical_depth", optical_depth_key)
        render_started = time.monotonic()
        report(
            {
                "stage": "render",
                "completed": 0,
                "total": 1,
                "elapsed_seconds": 0.0,
                "description": (
                    "Reusing optical depth and composing the static image."
                    if tau_map is not None
                    else "Rasterizing optical depth and composing the static image."
                ),
                "optical_depth_cached": tau_map is not None,
            }
        )
        plasma_for_render = dict(plasma)
        plasma_for_render["config"] = static_config
        initial_result = render_static_state(
            plasma_for_render,
            native_background=np.asarray(backgrounds["frames"])[0],
            native_background_info=backgrounds["metadata"],
            progress_callback=report,
            precomputed_tau_map=tau_map,
        )
        if tau_map is None:
            _store_stage(
                cache,
                "optical_depth",
                optical_depth_key,
                initial_result["arrays"]["tau_map"],
            )
        initial_result["arrays"]["support"] = np.ones(
            backgrounds["native_shape"], dtype=np.uint8
        )
        initial_result["metadata"]["synthesis_operation"] = "aligned_h5_frame_insertion"
        initial_result["metadata"]["h5_frame_index"] = int(backgrounds["frame_indices"][0])
        initial_result["metadata"]["h5_crop_xyxy_px"] = list(backgrounds["crop_xyxy_px"])
        _store_stage(cache, "render", render_key, initial_result)
        complete_stage("render", render_started, "Static image composition complete.")
    else:
        complete_stage("render", time.monotonic(), "Rendered static state reused.", cached=True)

    display_key = _cache_key({"render_key": render_key, "display": PREVIEW_RENDER_VERSION})
    display = _cached_stage(cache, "display", display_key)
    if display is None:
        display_started = time.monotonic()
        report(
            {
                "stage": "diagnostics",
                "completed": 0,
                "total": 3,
                "elapsed_seconds": 0.0,
                "description": "Building in-memory display diagnostics.",
            }
        )
        morphology_key = _cache_key({"plasma": plasma_key, "display": PREVIEW_RENDER_VERSION})
        geometry_png = _cached_stage(cache, "geometry_display", morphology_key)
        if geometry_png is None:
            geometry_png = _figure_png_bytes(_geometry_diagnostics_figure(initial_result))
            _store_stage(cache, "geometry_display", morphology_key, geometry_png)
        luna_key = _cache_key({"geometry": geometry_key, "display": PREVIEW_RENDER_VERSION})
        luna_display = _cached_stage(cache, "luna_display", luna_key)
        if luna_display is None:
            luna_figure, luna_summary = _luna_diagnostics_figure(initial_result)
            luna_display = (_figure_png_bytes(luna_figure), luna_summary)
            _store_stage(cache, "luna_display", luna_key, luna_display)
        luna_png, luna_summary = luna_display
        display = {
            "frame_zero_comparison_png": _figure_png_bytes(
                _frame_zero_comparison_figure(initial_result)
            ),
            "geometry_diagnostics_png": geometry_png,
            "luna_dynamics_diagnostics_png": luna_png,
        }
        _store_stage(cache, "display", display_key, display)
        report(
            {
                "stage": "diagnostics",
                "completed": 3,
                "total": 3,
                "elapsed_seconds": time.monotonic() - display_started,
                "description": "In-memory display diagnostics complete.",
            }
        )
    else:
        complete_stage("diagnostics", time.monotonic(), "Display diagnostics reused.", cached=True)
        luna_summary = _luna_preview_summary(_luna_preview_data(initial_result))

    metadata = initial_result["metadata"]
    return {
        "render_version": PREVIEW_RENDER_VERSION,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_sha256": experiment_config_sha256(user_config),
        "static_fingerprint": fingerprint,
        "cache_input_stamps": input_stamps,
        "static_state": initial_result,
        "background_provenance": deepcopy(backgrounds["metadata"]),
        "display": display,
        "luna_dynamics": deepcopy(luna_summary),
        "column_mass_loading": deepcopy(metadata["column_mass_loading"]),
        "opacity_table_saturation": deepcopy(metadata["opacity_table_saturation"]),
        "opacity_diagnostics": deepcopy(metadata["opacity_diagnostics"]),
        "thread_placement": deepcopy(metadata["thread_placement"]),
        "stage_durations_seconds": stage_durations,
        "progress_events": events,
        "cache_keys": {
            "background": background_key,
            "geometry": geometry_key,
            "plasma": plasma_key,
            "optical_depth": optical_depth_key,
            "render": render_key,
            "display": display_key,
        },
        "frame_zero": {
            "native_shape_yx": list(initial_result["arrays"]["degraded_intensity"].shape),
            "crop_xyxy_px": list(backgrounds["crop_xyxy_px"]),
            "disk_mu": float(backgrounds["disk_mu"]),
            "limb_direction_deg": float(backgrounds["limb_direction_deg"]),
            "native_pixel_km": float(backgrounds["native_pixel_km"]),
            "background_file": str(backgrounds["source_path"]),
            "orientation_deg": float(initial_result["config"]["orientation_deg"]),
            "chirality": int(initial_result["config"]["chirality"]),
            "spine_library_index": int(metadata["spine_library_index"]),
            "spine_source": metadata["spine_source"],
            "spine_length_mm": float(metadata["spine_length_mm"]),
            "n_threads": int(metadata["n_threads_generated"]),
            "tau_max": float(metadata["tau_max"]),
            "frame_sha256": _array_sha256(initial_result["arrays"]["degraded_intensity"]),
            "background_sha256": background_hash,
        },
    }


def load_preview(experiment_directory: str | Path) -> dict[str, Any] | None:
    """Load historical disk preview metadata for read-only compatibility."""
    path = Path(experiment_directory).resolve() / "preview/preview.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        preview = json.load(handle)
    if not isinstance(preview, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return preview


def preview_is_current(
    config_or_directory: Mapping[str, Any] | str | Path,
    preview: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether an in-memory preview matches current effective static inputs."""
    if preview is None or preview.get("render_version") != PREVIEW_RENDER_VERSION:
        return False
    config = _load_preview_source(config_or_directory)
    if preview.get("static_fingerprint") != static_preview_fingerprint(config):
        return False
    stamps = preview.get("cache_input_stamps")
    if stamps is not None:
        try:
            if any(Path(path).stat().st_ctime_ns != value for path, value in stamps["files"].items()):
                return False
        except OSError:
            return False
    return True


def _load_status(status_path: Path) -> dict[str, Any]:
    """Load one worker status file."""
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not isinstance(status, dict):
        raise ValueError(f"{status_path} must contain a JSON object")
    return status


def update_job_status(
    job_directory: str | Path,
    *,
    status_directory: str | Path | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Merge status atomically, writing any final run copy before publishing the job."""
    job = Path(job_directory).resolve()
    status_path = job / "status.json"
    status = _load_status(status_path) if status_path.is_file() else {}
    status.update(updates)
    status["updated_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if status_directory is not None:
        _atomic_json(Path(status_directory) / "status.json", status)
    _atomic_json(status_path, status)
    return status


def list_jobs(
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    *,
    experiment_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return durable worker records, newest first."""
    jobs_root = Path(experiments_root).resolve() / ".jobs"
    if not jobs_root.is_dir():
        return []
    selected_experiment = (
        None if experiment_directory is None else str(Path(experiment_directory).resolve())
    )
    jobs = []
    for directory in sorted(jobs_root.glob("job-*"), reverse=True):
        status_path = directory / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = _load_status(status_path)
            status["job_directory"] = str(directory)
        except Exception as error:
            status = {
                "job_id": directory.name,
                "state": "invalid",
                "error": f"{type(error).__name__}: {error}",
                "job_directory": str(directory),
            }
        if selected_experiment is None or status.get("experiment_directory") == selected_experiment:
            jobs.append(status)
    return jobs


def _pid_is_running(pid: object) -> bool:
    """Return whether a recorded worker process still exists."""
    if isinstance(pid, bool) or not isinstance(pid, Integral) or pid < 1:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_job(experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT) -> dict[str, Any] | None:
    """Return the first live queued/running job, if any."""
    for job in list_jobs(experiments_root):
        if job.get("state") not in {"queued", "running"}:
            continue
        if _pid_is_running(job.get("pid")):
            return job
        if job.get("state") == "queued" and job.get("pid") is None:
            return job
    return None


def start_experiment_worker(
    experiment_directory: str | Path,
    *,
    user_config: Mapping[str, Any] | None = None,
    preview: Mapping[str, Any] | None = None,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> dict[str, Any]:
    """Persist current draft/static state and launch a detached production worker."""
    experiment = Path(experiment_directory).resolve()
    config_path = experiment / "experiment.toml"
    current_config = (
        load_experiment_config(config_path, check_inputs=False)
        if user_config is None
        else deepcopy(dict(user_config))
    )
    resolved = validate_production_config(current_config)
    if preview is None or not preview_is_current(current_config, preview):
        raise ValueError("generate a current in-memory static preview before starting production")
    static_result = preview.get("static_state")
    if not isinstance(static_result, Mapping):
        raise ValueError("the current preview does not contain a complete static state")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required to generate the two canonical MP4 files")
    existing = active_job(experiments_root)
    if existing is not None:
        raise RuntimeError(f"generation job {existing['job_id']} is already active")

    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "runs").mkdir(exist_ok=True)
    save_experiment_config(current_config, config_path)
    current_config["inputs"]["h5_background_path"] = str(resolved["inputs"]["h5_background_path"])
    config_hash = experiment_config_sha256(current_config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    job_id = f"job-{stamp}-{config_hash[:10].lower()}"
    jobs_root = Path(experiments_root).resolve() / ".jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job = jobs_root / job_id
    staging = jobs_root / f".{job_id}.tmp-{os.getpid()}"
    if job.exists() or staging.exists():
        raise FileExistsError(job)
    staging.mkdir()
    try:
        snapshot_path = save_experiment_config(current_config, staging / "experiment.toml")
        save_static_result(dict(static_result), staging / "static_state")
        preview_snapshot = {
            "snapshot_version": STATIC_STATE_SNAPSHOT_VERSION,
            "render_version": preview["render_version"],
            "generated_utc": preview["generated_utc"],
            "config_sha256": config_hash,
            "static_fingerprint": preview["static_fingerprint"],
            "frame_zero": preview["frame_zero"],
            "background_provenance": preview["background_provenance"],
            "column_mass_loading": preview["column_mass_loading"],
            "opacity_table_saturation": preview["opacity_table_saturation"],
            "opacity_diagnostics": preview["opacity_diagnostics"],
            "thread_placement": preview["thread_placement"],
            "stage_durations_seconds": preview["stage_durations_seconds"],
        }
        _atomic_json(staging / "preview.json", preview_snapshot)
        _atomic_json(
            staging / "status.json",
            {
                "job_id": job_id,
                "state": "queued",
                "stage": "queued",
                "experiment_directory": str(experiment),
                "config_sha256": config_hash,
                "config_file_sha256": file_sha256(snapshot_path),
                "static_fingerprint": preview["static_fingerprint"],
                "completed_frames": 0,
                "total_frames": int(resolved["dynamics"]["n_frames"]),
                "output_directory": None,
                "error": None,
                "pid": None,
                "updated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        )
        os.replace(staging, job)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    status = _load_status(job / "status.json")

    log_path = job / "run.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_experiment_worker.py"),
        "--job-directory",
        str(job),
        "--experiment-directory",
        str(experiment),
        "--experiments-root",
        str(Path(experiments_root).resolve()),
    ]
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
    except BaseException as error:
        update_job_status(
            job,
            state="running",
            stage="launch",
            error=f"{type(error).__name__}: {error}",
        )
        move_failed_run(experiment, job, None)
        raise
    status = update_job_status(job, pid=process.pid)
    status["job_directory"] = str(job)
    return status


def _notify(callback: StatusCallback | None, stage: str, **values: Any) -> None:
    """Send one stage update when a caller supplied a callback."""
    if callback is not None:
        callback(stage, values)


def _augment_simulation_metadata(
    metadata_path: Path,
    *,
    user_config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    initial_result: Mapping[str, Any],
    config_hash: str,
    config_file_hash: str,
    preview: Mapping[str, Any],
    gong_path: Path,
    velocity_path: Path,
) -> None:
    """Add experiment lineage and video products to the canonical JSON sidecar."""
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["experiment_configuration"] = {
        "config_sha256": config_hash,
        "config_file_sha256": config_file_hash,
        "snapshot": "experiment.toml",
        "user_configuration": user_config,
        "resolved_inputs": resolved["inputs"],
        "resolved_static": initial_result["config"],
        "resolved_dynamics": resolved["dynamics"],
        "resolved_background_seed": resolved["background_seed"],
        "resolved_export": resolved["export"],
        "resolved_dynamic_background": resolved["dynamic_background"],
        "resolved_video": resolved["video"],
        "preview_frame_zero_sha256": preview["frame_zero"]["frame_sha256"],
    }
    metadata.setdefault("files", {}).update(
        {
            "experiment_config": "experiment.toml",
            "static_state": "static_state",
            "frame_zero_comparison": "frame_zero_comparison.png",
            "geometry_diagnostics": "geometry_diagnostics.png",
            "gong_video": gong_path.name,
            "gong_video_sha256": file_sha256(gong_path),
            "velocity_video": velocity_path.name,
            "velocity_video_sha256": file_sha256(velocity_path),
            "run_log": "run.log",
        }
    )
    metadata["velocity_video_visualization"] = {
        "render_version": VELOCITY_VIDEO_RENDER_VERSION,
        "spatial_view": "same fixed filament crop as the H-alpha video",
        "background": "speed colormap only",
        "speed_color_source": "/labels/coherent_velocity_xy_km_s",
        "direction_arrow_source": "/labels/coherent_velocity_xy_km_s",
        "arrow_sampling": "maximum opacity-weighted speed per sampling cell",
        "arrow_color": "black",
        "arrow_length_encodes_speed": True,
        "arrow_length_scaling": "linear local speed / fixed run-wide speed limit",
        "speed_scale": "fixed linear scale across all frames",
    }
    _atomic_json(metadata_path, metadata)


def run_experiment(
    experiment_directory: str | Path,
    snapshot_path: str | Path,
    *,
    status_callback: StatusCallback | None = None,
) -> dict[str, Any]:
    """Run the complete HDF5 and two-MP4 workflow from an immutable snapshot."""
    experiment = Path(experiment_directory).resolve()
    snapshot = Path(snapshot_path).resolve()
    job = snapshot.parent
    user_config = load_experiment_config(snapshot, check_inputs=False)
    resolved = validate_production_config(user_config)
    config_hash = experiment_config_sha256(user_config)
    with (job / "preview.json").open("r", encoding="utf-8") as handle:
        preview = json.load(handle)
    if (
        preview.get("snapshot_version") != STATIC_STATE_SNAPSHOT_VERSION
        or preview.get("config_sha256") != config_hash
        or preview.get("static_fingerprint") != static_preview_fingerprint(user_config)
    ):
        raise ValueError("the immutable static-state snapshot does not match the run configuration")
    initial_result = load_static_result(job / "static_state")
    expected_frame_hash = preview["frame_zero"]["frame_sha256"]
    actual_frame_hash = _array_sha256(initial_result["arrays"]["degraded_intensity"])
    if actual_frame_hash != expected_frame_hash:
        raise AssertionError(
            "loaded immutable frame zero differs from its snapshot metadata; "
            f"expected {expected_frame_hash}, received {actual_frame_hash}"
        )

    _notify(status_callback, "background_loading")
    n_frames = int(resolved["dynamics"]["n_frames"])
    start_index = int(user_config["dynamic_background"]["start_index"])
    frame_step = int(user_config["dynamic_background"]["frame_step"])
    backgrounds = load_h5_background_sequence(
        resolved["inputs"]["h5_background_path"],
        n_frames=n_frames, start_index=start_index, frame_step=frame_step,
        seed=resolved["background_seed"],
    )
    frozen_background_hash = preview["frame_zero"]["background_sha256"]
    loaded_background_hash = _array_sha256(np.asarray(backgrounds["frames"])[0])
    if loaded_background_hash != frozen_background_hash:
        raise AssertionError(
            "source frame zero changed after preview; "
            f"expected {frozen_background_hash}, received {loaded_background_hash}"
        )
    if not np.array_equal(backgrounds["frames"][0], initial_result["arrays"]["background"]):
        raise AssertionError("loaded production background frame zero differs from static state")

    _notify(status_callback, "simulation", completed_frames=0, total_frames=n_frames)

    def frame_progress(completed_frames: int, total_frames: int) -> None:
        _notify(
            status_callback,
            "simulation",
            completed_frames=completed_frames,
            total_frames=total_frames,
        )

    label = _run_label(str(user_config["experiment"]["name"]))
    saved = simulate_and_save_filament_dynamics(
        initial_result,
        resolved["dynamics"],
        background_frames=backgrounds["frames"],
        simulations_root=experiment / "runs",
        export_config=resolved["export"],
        label=label,
        progress_callback=frame_progress,
    )
    run_directory = Path(saved["directory"])
    _notify(
        status_callback,
        "simulation_complete",
        completed_frames=n_frames,
        total_frames=n_frames,
        output_directory=str(run_directory),
    )
    try:
        with h5py.File(saved["h5_path"], "r") as handle:
            persisted_frame_zero = np.asarray(handle["video/raw_gong"][0])
        persisted_hash = _array_sha256(persisted_frame_zero)
        if persisted_hash != expected_frame_hash:
            raise AssertionError(
                "persisted dynamics frame zero differs from the preview; "
                f"expected {expected_frame_hash}, received {persisted_hash}"
            )

        _notify(status_callback, "gong_video", completed_frames=n_frames, total_frames=n_frames)
        video = resolved["video"]
        gong_path = save_gong_video(
            saved["h5_path"],
            run_directory / "gong.mp4",
            fps=int(video["fps"]),
        )
        _notify(status_callback, "velocity_video", completed_frames=n_frames, total_frames=n_frames)
        velocity_path = save_velocity_video(
            saved["h5_path"],
            run_directory / "velocity.mp4",
            fps=int(video["fps"]),
            velocity_limit_km_s=video["velocity_limit_km_s"],
            quiver_stride_px=int(video["quiver_stride_px"]),
        )
        shutil.copy2(snapshot, run_directory / "experiment.toml")
        shutil.copytree(job / "static_state", run_directory / "static_state")
        (run_directory / "frame_zero_comparison.png").write_bytes(
            _figure_png_bytes(_frame_zero_comparison_figure(initial_result))
        )
        (run_directory / "geometry_diagnostics.png").write_bytes(
            _figure_png_bytes(_geometry_diagnostics_figure(initial_result))
        )
        _augment_simulation_metadata(
            Path(saved["metadata_path"]),
            user_config=user_config,
            resolved=resolved,
            initial_result=initial_result,
            config_hash=config_hash,
            config_file_hash=file_sha256(snapshot),
            preview=preview,
            gong_path=gong_path,
            velocity_path=velocity_path,
        )
        _notify(status_callback, "complete", completed_frames=n_frames, total_frames=n_frames)
        return {
            **saved,
            "gong_path": gong_path,
            "velocity_path": velocity_path,
            "config_sha256": config_hash,
            "preview_frame_zero_sha256": expected_frame_hash,
            "persisted_frame_zero_sha256": persisted_hash,
        }
    except BaseException:
        raise


def _run_label(name: str) -> str:
    """Return a valid compact label for the canonical simulation identifier."""
    label = "-".join(part for part in _slug_words(name) if part)
    return (label or "experiment")[:24].rstrip("-")


def _slug_words(name: str) -> list[str]:
    """Return lowercase alphanumeric words for a run label."""
    normalized = "".join(
        character.lower() if character.isalnum() else " " for character in name
    )
    return normalized.split()


def move_failed_run(
    experiment_directory: str | Path,
    job_directory: str | Path,
    published_run: str | Path | None,
) -> Path:
    """Retain failure diagnostics and move any partial published run out of ``sim-*``."""
    experiment = Path(experiment_directory).resolve()
    job = Path(job_directory).resolve()
    status = _load_status(job / "status.json")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    failure = experiment / "runs" / f"failed-{stamp}-{str(status.get('config_sha256', 'unknown'))[:10].lower()}"
    failure.parent.mkdir(parents=True, exist_ok=True)
    source_run = None if published_run is None else Path(published_run).resolve()
    if source_run is not None and source_run.is_dir():
        os.replace(source_run, failure)
        rebuild_simulation_index(experiment / "runs")
    else:
        failure.mkdir()
    sys.stdout.flush()
    sys.stderr.flush()
    for name in ("experiment.toml", "preview.json", "run.log"):
        source = job / name
        if source.is_file():
            shutil.copy2(source, failure / name)
    static_state = job / "static_state"
    if static_state.is_dir() and not (failure / "static_state").exists():
        shutil.copytree(static_state, failure / "static_state")
    update_job_status(
        job,
        status_directory=failure,
        state="failed",
        failure_directory=str(failure),
    )
    return failure


def run_worker_job(
    job_directory: str | Path,
    experiment_directory: str | Path,
    *,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> dict[str, Any]:
    """Execute one durable job under the global single-generation lock."""
    job = Path(job_directory).resolve()
    experiment = Path(experiment_directory).resolve()
    lock_path = Path(experiments_root).resolve() / ".generation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    published_run: Path | None = None

    def status_callback(stage: str, values: Mapping[str, Any]) -> None:
        update_job_status(
            job,
            state="running",
            stage="finalizing" if stage == "complete" else stage,
            elapsed_seconds=time.monotonic() - started,
            **dict(values),
        )

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = "another generation worker owns the global experiment lock"
            update_job_status(job, state="running", stage="lock", error=message)
            move_failed_run(experiment, job, None)
            raise RuntimeError(message) from error

        update_job_status(
            job,
            state="running",
            stage="validation",
            pid=os.getpid(),
            started_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            elapsed_seconds=0.0,
        )
        try:
            saved = run_experiment(
                experiment,
                job / "experiment.toml",
                status_callback=status_callback,
            )
            published_run = Path(saved["directory"])
            sys.stdout.flush()
            sys.stderr.flush()
            log_path = job / "run.log"
            if log_path.is_file():
                shutil.copy2(log_path, published_run / "run.log")
            update_job_status(
                job,
                status_directory=published_run,
                state="completed",
                stage="complete",
                elapsed_seconds=time.monotonic() - started,
                finished_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                output_directory=str(published_run),
                h5_path=str(saved["h5_path"]),
                gong_path=str(saved["gong_path"]),
                velocity_path=str(saved["velocity_path"]),
                error=None,
            )
            return saved
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            failed_status = update_job_status(
                job,
                state="running",
                stage="failed",
                elapsed_seconds=time.monotonic() - started,
                finished_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                error=message,
            )
            if published_run is None and failed_status.get("output_directory"):
                published_run = Path(failed_status["output_directory"])
            move_failed_run(experiment, job, published_run)
            raise
