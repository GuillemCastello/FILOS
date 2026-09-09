"""Validated TOML configuration and filesystem layout for local experiments.

The public API is function-oriented. User-facing TOML values are kept separate
from physical constants and derived observing quantities. Validation resolves
every section into the existing forward-model configuration dictionaries before
any synthesis is started.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .config import (
    DEFAULT_DIP_CURVATURE_RADIUS_PARAMETERS,
    DEFAULT_SOURCE_FRACTION,
    make_static_config,
)
from .detector import DETECTOR_MODEL_PATH
from .dynamic_background import H5_TIME_SERIES_KEY, _read_source_frame
from .dynamics import make_dynamics_config
from .geometry import SPINE_LIBRARY_PATH
from .opacity_table import HEINZEL_EXTENSION_PATH, heinzel_opacity_table_metadata
from .paths import PROJECT_ROOT
from .simulation_io import make_export_config

DEFAULT_EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs/default_experiment.toml"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "simulations/experiments"
STATIC_PREVIEW_FINGERPRINT_VERSION = 1
STATIC_FORWARD_MODEL_REVISION = 3
SOURCE_FUNCTION_POLICY = "heinzel2015_published_10_20_30Mm_clamped"

EXPERIMENT_SECTIONS = (
    "experiment",
    "inputs",
    "static",
    "dynamic_background",
    "dynamics",
    "export",
    "video",
)

EXPERIMENT_FIELDS = ("name", "description")
INPUT_FIELDS = ("h5_background_path",)
STATIC_FIELDS = (
    "seed",
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
    "temp_center_K",
    "temp_tr_K",
    "pctr_gamma",
    "ionization_center",
    "transition_pressure_dyn_cm2",
    "column_mass_g_cm2",
    "microturbulent_velocity_kms",
    "halpha_wavelength_angstrom",
    "foot_taper_fraction",
    "source_fraction",
    "gong_bandpass_line_fraction",
    "mask_tau_threshold",
    "psf_sigma_px",
)
DYNAMIC_BACKGROUND_FIELDS = (
    "seed",
    "crop_height_px",
    "crop_width_px",
    "start_index",
    "frame_step",
    "random_attempts",
    "use_detector",
    "detection_threshold",
    "box_expand_fraction",
    "box_expand_px",
)
DYNAMICS_FIELDS = (
    "seed",
    "n_frames",
    "cadence_s",
    "longitudinal_displacement_amplitude_km",
    "transverse_displacement_amplitude_km",
    "oscillation_mode",
    "period_s",
    "transverse_period_s",
    "damping_time_s",
    "phase_rad",
    "brownian_step_min_km",
    "brownian_step_max_km",
    "center_spine_fraction",
    "center_height_km",
    "sphere_radius_km",
    "kernel_half_weight_radius_fraction",
    "kernel_power",
    "oscillation_start_time_s",
)
EXPORT_FIELDS = (
    "thread_mask_dilation_px",
    "video_lower_percentile",
    "video_upper_percentile",
    "velocity_opacity_floor",
    "compression",
    "gzip_level",
)
VIDEO_FIELDS = ("fps", "velocity_limit_km_s", "quiver_stride_px")

SECTION_FIELDS = {
    "experiment": EXPERIMENT_FIELDS,
    "inputs": INPUT_FIELDS,
    "static": STATIC_FIELDS,
    "dynamic_background": DYNAMIC_BACKGROUND_FIELDS,
    "dynamics": DYNAMICS_FIELDS,
    "export": EXPORT_FIELDS,
    "video": VIDEO_FIELDS,
}

_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_LEGACY_DIP_DEPTH_FIELDS = (
    "dip_depth_min_km",
    "dip_depth_max_km",
    "dip_depth_median_km",
    "dip_depth_sigma_ln",
)


def _plain_mapping(value: object, name: str) -> dict[str, Any]:
    """Return one mapping as an independent plain dictionary."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table")
    return deepcopy(dict(value))


def _require_exact_keys(mapping: Mapping[str, Any], expected: tuple[str, ...], owner: str) -> None:
    """Reject missing and unknown TOML keys in one table."""
    missing = sorted(set(expected) - set(mapping))
    unknown = sorted(set(mapping) - set(expected))
    if missing:
        raise ValueError(f"{owner} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{owner} contains unsupported fields: {unknown}")


def _resolve_project_path(value: object, field_name: str) -> Path:
    """Resolve one user path relative to the project root."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _auto_or_number(value: object, field_name: str) -> float | None:
    """Resolve the TOML ``auto`` sentinel to ``None`` or return a finite float."""
    if value == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise TypeError(f"{field_name} must be 'auto' or a finite real number")
    return float(value)


def normalize_experiment_config(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize supported legacy fields and return independent TOML sections.

    Older configurations used ``dynamics.seed`` for both motion and quiet-crop
    selection. Copying that value into the new background section preserves the
    previously selected crop without mutating the source mapping. Legacy dip
    selectors and depth distributions are removed; existing curvature-radius
    controls are retained, otherwise the maintained radius defaults are added.
    """
    root = _plain_mapping(config, "experiment configuration")
    static_section = root.get("static")
    if isinstance(static_section, Mapping) and (
        "dip_parameterization" in static_section
        or any(name in static_section for name in _LEGACY_DIP_DEPTH_FIELDS)
    ):
        upgraded_static = deepcopy(dict(static_section))
        legacy_parameterization = upgraded_static.pop("dip_parameterization", "dip_depth")
        if legacy_parameterization not in {"dip_depth", "curvature_radius"}:
            raise ValueError(
                "legacy static.dip_parameterization must be 'dip_depth' or "
                f"'curvature_radius'; received {legacy_parameterization!r}"
            )
        for name in _LEGACY_DIP_DEPTH_FIELDS:
            upgraded_static.pop(name, None)
        for name, value in DEFAULT_DIP_CURVATURE_RADIUS_PARAMETERS.items():
            upgraded_static.setdefault(name, value)
        root["static"] = upgraded_static
    background_section = root.get("dynamic_background")
    dynamics_section = root.get("dynamics")
    if isinstance(background_section, Mapping) and "seed" not in background_section:
        if not isinstance(dynamics_section, Mapping) or "seed" not in dynamics_section:
            raise ValueError(
                "legacy [dynamic_background] without seed requires dynamics.seed for migration"
            )
        upgraded_background = deepcopy(dict(background_section))
        upgraded_background["seed"] = deepcopy(dynamics_section["seed"])
        root["dynamic_background"] = upgraded_background
    _require_exact_keys(root, EXPERIMENT_SECTIONS, "experiment configuration")
    sections: dict[str, dict[str, Any]] = {}
    for section_name in EXPERIMENT_SECTIONS:
        section = _plain_mapping(root[section_name], f"[{section_name}]")
        _require_exact_keys(section, SECTION_FIELDS[section_name], f"[{section_name}]")
        sections[section_name] = section
    return sections


def _validate_user_shape(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper for schema normalization and migration."""
    return normalize_experiment_config(config)


def _validate_experiment_section(section: Mapping[str, Any]) -> None:
    """Validate identifying experiment text."""
    name = section["name"]
    description = section["description"]
    if not isinstance(name, str) or not _EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment.name must start with an alphanumeric character and contain only "
            "letters, numbers, spaces, underscores, or hyphens (maximum 80 characters)"
        )
    if not isinstance(description, str) or len(description) > 500:
        raise ValueError("experiment.description must be a string of at most 500 characters")


def _validate_h5_request(
    path: Path,
    dynamic_background: Mapping[str, Any],
    dynamics: Mapping[str, Any],
) -> tuple[tuple[int, int, int], str]:
    """Validate requested HDF5 indices and crop dimensions from dataset metadata."""
    start = dynamic_background["start_index"]
    step = dynamic_background["frame_step"]
    n_frames = dynamics["n_frames"]
    shape, dtype = _inspect_h5_source(
        path,
        dynamic_background,
        start_index=int(start),
        load_first_frame=False,
    )
    available_frames = shape[0]
    last_index = int(start) + int(step) * (int(n_frames) - 1)
    if last_index >= available_frames:
        raise IndexError(
            "requested HDF5 background frames exceed the available time axis; "
            f"last requested={last_index}, available={available_frames}"
        )
    return shape, dtype


def _inspect_h5_source(
    path: Path,
    dynamic_background: Mapping[str, Any],
    *,
    start_index: int,
    load_first_frame: bool,
    cached_stages: dict[str, Any] | None = None,
) -> tuple[tuple[int, int, int], str]:
    """Inspect one background source and optionally read its selected first frame."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if H5_TIME_SERIES_KEY not in handle:
            raise KeyError(f"{path} is missing dataset {H5_TIME_SERIES_KEY!r}")
        dataset = handle[H5_TIME_SERIES_KEY]
        if dataset.ndim != 3:
            raise ValueError(
                f"{H5_TIME_SERIES_KEY!r} must have shape (time, y, x); received {dataset.shape}"
            )
        shape = tuple(int(value) for value in dataset.shape)
        available_frames, frame_height, frame_width = shape
        if start_index >= available_frames:
            raise IndexError(
                "dynamic_background.start_index exceeds the available time axis; "
                f"received={start_index}, available={available_frames}"
            )
        crop_height = dynamic_background["crop_height_px"]
        crop_width = dynamic_background["crop_width_px"]
        for name, value, maximum in (
            ("dynamic_background.crop_height_px", crop_height, frame_height),
            ("dynamic_background.crop_width_px", crop_width, frame_width),
        ):
            if int(value) > maximum:
                raise ValueError(
                    f"{name} must be an integer in [8, {maximum}]; received {value!r}"
                )
        if load_first_frame:
            first_frame, _ = _read_source_frame(dataset, path, start_index, cached_stages)
            if first_frame.shape != (frame_height, frame_width):
                raise ValueError(
                    "the selected HDF5 first frame has an unexpected shape; "
                    f"received {first_frame.shape}"
                )
        dtype = str(dataset.dtype)
    return shape, dtype


def _validate_dynamic_background(
    section: Mapping[str, Any],
    *,
    preview: bool,
) -> tuple[int, dict[str, Any]]:
    """Validate active crop-selection controls and return loader arguments."""
    arguments = dict(section)
    for name, minimum in (
        ("seed", 0),
        ("start_index", 0),
        ("random_attempts", 1),
        ("crop_height_px", 8),
        ("crop_width_px", 8),
    ):
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(
                f"dynamic_background.{name} must be an integer >= {minimum}; "
                f"received {value!r}"
            )
    if preview:
        arguments["frame_step"] = 1
    else:
        frame_step = arguments["frame_step"]
        if (
            isinstance(frame_step, bool)
            or not isinstance(frame_step, Integral)
            or frame_step < 1
        ):
            raise ValueError(
                "dynamic_background.frame_step must be an integer >= 1; "
                f"received {frame_step!r}"
            )
    if not isinstance(arguments["use_detector"], bool):
        raise TypeError("dynamic_background.use_detector must be true or false")
    if arguments["use_detector"] or not preview:
        threshold = arguments["detection_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or not 0.0 < float(threshold) < 1.0
        ):
            raise ValueError("dynamic_background.detection_threshold must lie in (0, 1)")
        for name in ("box_expand_fraction", "box_expand_px"):
            value = arguments[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"dynamic_background.{name} must be finite and >= 0")
    else:
        arguments["detection_threshold"] = 0.5
        arguments["box_expand_fraction"] = 0.0
        arguments["box_expand_px"] = 0.0
    background_seed = int(arguments.pop("seed"))
    loader_arguments = {
        "crop_shape": (int(arguments.pop("crop_height_px")), int(arguments.pop("crop_width_px"))),
        **arguments,
    }
    return background_seed, loader_arguments


def _validate_video(section: Mapping[str, Any]) -> dict[str, Any]:
    """Validate MP4 display controls and resolve the automatic velocity limit."""
    fps = section["fps"]
    stride = section["quiver_stride_px"]
    if isinstance(fps, bool) or not isinstance(fps, Integral) or not 1 <= fps <= 120:
        raise ValueError("video.fps must be an integer in [1, 120]")
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("video.quiver_stride_px must be a positive integer")
    velocity_limit = _auto_or_number(section["velocity_limit_km_s"], "video.velocity_limit_km_s")
    if velocity_limit is not None and velocity_limit <= 0.0:
        raise ValueError("video.velocity_limit_km_s must be 'auto' or > 0")
    return {
        "fps": int(fps),
        "velocity_limit_km_s": velocity_limit,
        "quiver_stride_px": int(stride),
    }


def _resolve_static_section(
    section: Mapping[str, Any],
    *,
    native_shape: tuple[int, int],
    active_controls_only: bool,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Resolve one user static section into the canonical core configuration."""
    static_values = dict(section)
    if active_controls_only:
        static_values["source_fraction"] = DEFAULT_SOURCE_FRACTION
    static_seed = static_values.pop("seed")
    spine_index = static_values["spine_library_entry_index"]
    if spine_index == "auto":
        static_values["spine_library_entry_index"] = None
    elif isinstance(spine_index, bool) or not isinstance(spine_index, Integral) or spine_index < 0:
        raise ValueError("static.spine_library_entry_index must be 'auto' or an integer >= 0")
    bounds = static_values["spine_library_length_bounds_km"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("static.spine_library_length_bounds_km must contain two values")
    static_values["spine_library_length_bounds_km"] = tuple(bounds)

    static_config = make_static_config(
        seed=static_seed,
        native_shape=native_shape,
        disk_mu=1.0,
        orientation_deg=0.0,
        chirality=1,
        overrides=static_values,
    )
    static_overrides = {name: static_config[name] for name in STATIC_FIELDS if name != "seed"}
    return int(static_seed), static_overrides, static_config


def _resolve_run_sections(
    sections: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve dynamics, export, and video sections for saving or production."""

    dynamics_values = dict(sections["dynamics"])
    dynamics_values["center_height_km"] = _auto_or_number(
        dynamics_values["center_height_km"], "dynamics.center_height_km"
    )
    dynamics_values["center_spine_fraction"] = _auto_or_number(
        dynamics_values["center_spine_fraction"], "dynamics.center_spine_fraction"
    )
    if dynamics_values["oscillation_mode"] == "shared_period":
        dynamics_values["transverse_period_s"] = None
    dynamics_config = make_dynamics_config(**dynamics_values)

    export_values = dict(sections["export"])
    if export_values["compression"] == "none":
        export_values["compression"] = None
    export_config = make_export_config(**export_values)
    video_config = _validate_video(sections["video"])
    return dynamics_config, export_config, video_config


def _file_identity(path: Path) -> dict[str, Any]:
    """Return an inexpensive identity for one required source or model file."""
    try:
        stat = path.stat()
    except OSError as error:
        raise FileNotFoundError(f"required preview input is unavailable: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"required preview input is not a file: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _directory_identity(path: Path) -> dict[str, Any]:
    """Return stat identities for files directly owned by one model directory."""
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"required detector model directory is unavailable: {root}")
    files = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = candidate.stat()
        files.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if not files:
        raise FileNotFoundError(f"required detector model directory contains no files: {root}")
    return {"path": str(root), "files": files}


def _preview_asset_identities(*, use_detector: bool) -> dict[str, Any]:
    """Return identities for scientific assets active in a static preview."""
    opacity_metadata = heinzel_opacity_table_metadata()
    identities = {
        "opacity_table": {
            **_file_identity(HEINZEL_EXTENSION_PATH),
            "sha256": opacity_metadata["sha256"],
        },
        "spine_library": _file_identity(SPINE_LIBRARY_PATH),
    }
    if use_detector:
        identities["detector_model"] = _directory_identity(DETECTOR_MODEL_PATH)
    return identities


def _resolve_preview_config(
    config: Mapping[str, Any],
    *,
    load_first_frame: bool,
    cached_stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve only inputs that participate in the static preview."""
    sections = normalize_experiment_config(config)
    h5_path = _resolve_project_path(
        sections["inputs"]["h5_background_path"],
        "inputs.h5_background_path",
    )
    background_seed, background_arguments = _validate_dynamic_background(
        sections["dynamic_background"],
        preview=True,
    )
    native_shape = tuple(int(value) for value in background_arguments["crop_shape"])
    static_seed, static_overrides, static_config = _resolve_static_section(
        sections["static"],
        native_shape=native_shape,
        active_controls_only=True,
    )
    h5_shape, h5_dtype = _inspect_h5_source(
        h5_path,
        {
            "crop_height_px": native_shape[0],
            "crop_width_px": native_shape[1],
        },
        start_index=int(background_arguments["start_index"]),
        load_first_frame=load_first_frame,
        cached_stages=cached_stages,
    )
    h5_identity = {
        **_file_identity(h5_path),
        "dataset_key": H5_TIME_SERIES_KEY,
        "dataset_shape": list(h5_shape),
        "dataset_dtype": h5_dtype,
    }
    assets = _preview_asset_identities(use_detector=bool(background_arguments["use_detector"]))
    return {
        "inputs": {"h5_background_path": h5_path},
        "static_seed": static_seed,
        "static_overrides": static_overrides,
        "validated_static_config": static_config,
        "background_seed": background_seed,
        "dynamic_background": background_arguments,
        "h5_dataset_shape": h5_shape,
        "h5_dataset_dtype": h5_dtype,
        "input_identities": {"h5_background": h5_identity, **assets},
    }


def validate_preview_config(
    config: Mapping[str, Any], *, cached_stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and resolve only the inputs used by one static preview.

    The selected first background frame is read once. Dynamics, export, video,
    experiment text, and future background-frame spacing are deliberately not
    interpreted by this action.
    """
    return _resolve_preview_config(config, load_first_frame=True, cached_stages=cached_stages)


def validate_save_config(
    config: Mapping[str, Any],
    *,
    check_inputs: bool = True,
) -> dict[str, Any]:
    """Validate the complete configuration draft before an explicit save."""
    sections = normalize_experiment_config(config)
    _validate_experiment_section(sections["experiment"])
    h5_path = _resolve_project_path(
        sections["inputs"]["h5_background_path"],
        "inputs.h5_background_path",
    )
    background_seed, background_arguments = _validate_dynamic_background(
        sections["dynamic_background"],
        preview=False,
    )
    dynamics_config, export_config, video_config = _resolve_run_sections(sections)
    native_shape = tuple(int(value) for value in background_arguments["crop_shape"])
    static_seed, static_overrides, static_config = _resolve_static_section(
        sections["static"],
        native_shape=native_shape,
        active_controls_only=False,
    )

    if check_inputs:
        h5_shape, h5_dtype = _validate_h5_request(
            h5_path,
            sections["dynamic_background"],
            dynamics_config,
        )
    else:
        h5_shape = None
        h5_dtype = None

    return {
        "experiment": deepcopy(sections["experiment"]),
        "inputs": {
            "h5_background_path": h5_path,
        },
        "static_seed": int(static_seed),
        "static_overrides": static_overrides,
        "validated_static_config": static_config,
        "background_seed": background_seed,
        "dynamic_background": background_arguments,
        "dynamics": dynamics_config,
        "export": export_config,
        "video": video_config,
        "h5_dataset_shape": h5_shape,
        "h5_dataset_dtype": h5_dtype,
    }


def validate_production_config(
    config: Mapping[str, Any],
    *,
    check_inputs: bool = True,
    check_ffmpeg: bool = True,
) -> dict[str, Any]:
    """Validate the complete production request and required video encoder."""
    resolved = validate_save_config(config, check_inputs=check_inputs)
    if check_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is unavailable; production MP4 generation cannot start")
    return resolved


def validate_experiment_config(
    config: Mapping[str, Any],
    *,
    check_inputs: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for complete save-time configuration validation."""
    return validate_save_config(config, check_inputs=check_inputs)


def load_experiment_config(
    path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    *,
    check_inputs: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse and normalize one TOML configuration without changing the file.

    By default execution validation is deferred so an unavailable input path can
    be repaired by the editor. ``check_inputs`` remains as a compatibility
    option for callers that explicitly request complete save-time validation.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with source.open("rb") as handle:
            config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {source}: {error}") from error
    sections = normalize_experiment_config(config)
    if check_inputs is not None:
        validate_save_config(sections, check_inputs=check_inputs)
    return sections


def experiment_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash one configuration independent of TOML whitespace and key order."""
    sections = _validate_user_shape(config)
    payload = json.dumps(sections, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def static_preview_fingerprint_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical effective-input payload for a static preview hash."""
    resolved = _resolve_preview_config(config, load_first_frame=False)
    static_values = dict(resolved["static_overrides"])
    static_values.pop("source_fraction", None)
    static_values["seed"] = resolved["static_seed"]
    return {
        "fingerprint_version": STATIC_PREVIEW_FINGERPRINT_VERSION,
        "static_forward_model_revision": STATIC_FORWARD_MODEL_REVISION,
        "source_function_policy": SOURCE_FUNCTION_POLICY,
        "static": static_values,
        "background_seed": resolved["background_seed"],
        "dynamic_background": resolved["dynamic_background"],
        "input_identities": resolved["input_identities"],
    }


def static_preview_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash only effective scientific and source inputs of a static preview."""
    payload = static_preview_fingerprint_payload(config)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def next_experiment_seeds(static_seed: int, dynamics_seed: int) -> tuple[int, int]:
    """Derive a new explicit reproducible seed pair from the current pair.

    The function deliberately uses no operating-system entropy. Every click in
    the GUI stores concrete seeds in TOML, while the deterministic derivation
    keeps the complete experiment reproducible.
    """
    for name, value in (("static_seed", static_seed), ("dynamics_seed", dynamics_seed)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = f"synthetic-filament-seeds-v1:{int(static_seed)}:{int(dynamics_seed)}"
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    maximum_seed = 2**31 - 1
    next_static = int.from_bytes(digest[:8], "big") % maximum_seed
    next_dynamics = int.from_bytes(digest[8:16], "big") % maximum_seed
    if next_static == next_dynamics:
        next_dynamics = (next_dynamics + 1) % maximum_seed
    return next_static, next_dynamics


def next_background_seed(background_seed: int) -> int:
    """Derive the next explicit crop-selection seed without system entropy."""
    if (
        isinstance(background_seed, bool)
        or not isinstance(background_seed, Integral)
        or background_seed < 0
    ):
        raise ValueError("background_seed must be a non-negative integer")
    payload = f"synthetic-filament-background-seed-v1:{int(background_seed)}"
    maximum_seed = 2**31 - 1
    next_seed = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")
    next_seed %= maximum_seed
    if next_seed == int(background_seed):
        next_seed = (next_seed + 1) % maximum_seed
    return next_seed


def file_sha256(path: str | Path) -> str:
    """Return the uppercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def save_experiment_config(
    config: Mapping[str, Any],
    path: str | Path,
    *,
    check_inputs: bool = True,
) -> Path:
    """Validate and atomically save one complete TOML configuration."""
    sections = normalize_experiment_config(config)
    validate_save_config(sections, check_inputs=check_inputs)
    try:
        import tomli_w
    except ImportError as error:
        raise RuntimeError("saving TOML requires the project 'gui' dependency extra") from error

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{datetime.now(UTC).timestamp()}")
    try:
        text = tomli_w.dumps(sections)
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _slugify(name: str) -> str:
    """Return a short filesystem-safe experiment slug."""
    slug = _SLUG_PATTERN.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("experiment name must contain at least one letter or number")
    return slug[:48].rstrip("-")


def create_experiment(
    name: str,
    *,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    source_config: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically create one editable experiment from a complete configuration."""
    root = Path(experiments_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = (
        load_experiment_config(DEFAULT_EXPERIMENT_CONFIG_PATH)
        if source_config is None
        else _validate_user_shape(source_config)
    )
    config["experiment"]["name"] = name
    _validate_experiment_section(config["experiment"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"{stamp}-{_slugify(name)}"
    staging = root / f".{directory.name}.tmp-{os.getpid()}"
    if directory.exists() or staging.exists():
        raise FileExistsError(directory)
    staging.mkdir()
    try:
        (staging / "runs").mkdir()
        save_experiment_config(config, staging / "experiment.toml")
        os.replace(staging, directory)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return directory


def list_experiments(
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> list[dict[str, Any]]:
    """Return editable experiment directories with lightweight display metadata."""
    root = Path(experiments_root).resolve()
    if not root.is_dir():
        return []
    records = []
    for path in sorted(root.iterdir(), reverse=True):
        config_path = path / "experiment.toml"
        if not path.is_dir() or not config_path.is_file():
            continue
        try:
            config = load_experiment_config(config_path, check_inputs=False)
            name = config["experiment"]["name"]
            error = None
        except Exception as exception:
            name = path.name
            error = f"{type(exception).__name__}: {exception}"
        records.append({"name": name, "directory": path, "config_path": config_path, "error": error})
    return records


def list_standalone_simulations(
    simulations_root: str | Path = PROJECT_ROOT / "simulations",
) -> list[dict[str, Any]]:
    """Return canonical root-level simulations for read-only inspection."""
    root = Path(simulations_root).resolve()
    if not root.is_dir():
        return []
    records = []
    for directory in sorted(root.glob("sim-*"), reverse=True):
        metadata_path = directory / "simulation.json"
        h5_path = directory / "simulation.h5"
        if not directory.is_dir() or not metadata_path.is_file() or not h5_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
            error = None
        except Exception as exception:
            metadata = {}
            summary = {}
            error = f"{type(exception).__name__}: {exception}"
        records.append(
            {
                "simulation_id": directory.name,
                "directory": directory,
                "metadata_path": metadata_path,
                "h5_path": h5_path,
                "metadata": metadata,
                "summary": summary,
                "error": error,
            }
        )
    return records


def _supported_config_from_mapping(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy supported controls from an older immutable snapshot into today's schema."""
    config = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG_PATH, check_inputs=False)
    for section_name, field_names in SECTION_FIELDS.items():
        source_section = source.get(section_name)
        if not isinstance(source_section, Mapping):
            continue
        for field_name in field_names:
            if field_name in source_section:
                config[section_name][field_name] = deepcopy(source_section[field_name])
    source_background = source.get("dynamic_background")
    source_dynamics = source.get("dynamics")
    if (
        isinstance(source_background, Mapping)
        and "seed" not in source_background
        and isinstance(source_dynamics, Mapping)
        and "seed" in source_dynamics
    ):
        config["dynamic_background"]["seed"] = deepcopy(source_dynamics["seed"])
    return _validate_user_shape(config)


def _clone_config_from_metadata(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a canonical simulation record back to supported user controls."""
    config = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG_PATH, check_inputs=False)
    experiment_metadata = metadata.get("experiment_configuration")
    if isinstance(experiment_metadata, Mapping):
        embedded = experiment_metadata.get("user_configuration")
        if isinstance(embedded, Mapping):
            return _supported_config_from_mapping(embedded)

    initial_metadata = metadata.get("initial_forward_model_metadata", {})
    static_parameters = metadata.get("static_forward_model_parameters", {})
    dynamics_parameters = metadata.get("dynamics_parameters", {})
    export_parameters = metadata.get("export_parameters", {})
    recovered_background_seed: int | None = None
    if isinstance(initial_metadata, Mapping):
        static_seed = initial_metadata.get("source_static_seed", initial_metadata.get("seed"))
        if isinstance(static_seed, Integral) and not isinstance(static_seed, bool):
            config["static"]["seed"] = int(static_seed)
        real_background = initial_metadata.get("real_background", {})
        if isinstance(real_background, Mapping):
            for seed_name in ("selection_seed", "background_seed", "seed"):
                candidate_seed = real_background.get(seed_name)
                if (
                    isinstance(candidate_seed, Integral)
                    and not isinstance(candidate_seed, bool)
                    and candidate_seed >= 0
                ):
                    recovered_background_seed = int(candidate_seed)
                    break
            source_path = real_background.get("source_path")
            if isinstance(source_path, str):
                resolved_source = Path(source_path).resolve()
                try:
                    config["inputs"]["h5_background_path"] = str(
                        resolved_source.relative_to(PROJECT_ROOT)
                    )
                except ValueError:
                    config["inputs"]["h5_background_path"] = str(resolved_source)
            crop_shape = real_background.get("crop_shape")
            if isinstance(crop_shape, list) and len(crop_shape) == 2:
                config["dynamic_background"]["crop_height_px"] = crop_shape[0]
                config["dynamic_background"]["crop_width_px"] = crop_shape[1]
            frame_indices = real_background.get("frame_indices")
            if isinstance(frame_indices, list) and frame_indices:
                config["dynamic_background"]["start_index"] = int(frame_indices[0])
                if len(frame_indices) > 1:
                    config["dynamic_background"]["frame_step"] = int(
                        frame_indices[1] - frame_indices[0]
                    )
            detector_requested = real_background.get("detector_requested")
            if isinstance(detector_requested, bool):
                config["dynamic_background"]["use_detector"] = detector_requested

    if isinstance(static_parameters, Mapping):
        for name in STATIC_FIELDS:
            if name == "seed" or name not in static_parameters:
                continue
            value = static_parameters[name]
            config["static"][name] = "auto" if value is None else value
        # A generated result stores the realized index. Auto preserves the original
        # seed-driven random-stream sequence when the source metadata is available.
        if isinstance(initial_metadata, Mapping) and "source_static_seed" in initial_metadata:
            config["static"]["spine_library_entry_index"] = "auto"
    if isinstance(dynamics_parameters, Mapping):
        for name in DYNAMICS_FIELDS:
            if name in dynamics_parameters:
                value = dynamics_parameters[name]
                if name in {"center_height_km", "center_spine_fraction"} and value is None:
                    value = "auto"
                elif name == "transverse_period_s" and value is None:
                    value = dynamics_parameters.get("period_s", config["dynamics"]["period_s"])
                config["dynamics"][name] = value
    if recovered_background_seed is not None:
        config["dynamic_background"]["seed"] = recovered_background_seed
    else:
        config["dynamic_background"]["seed"] = config["dynamics"]["seed"]
    if isinstance(export_parameters, Mapping):
        for name in EXPORT_FIELDS:
            if name in export_parameters:
                value = export_parameters[name]
                config["export"][name] = "none" if name == "compression" and value is None else value
    return config


def clone_simulation_as_experiment(
    simulation_directory: str | Path,
    name: str,
    *,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> Path:
    """Clone a simulation snapshot, or recover supported controls from legacy metadata."""
    source = Path(simulation_directory).resolve()
    metadata_path = source / "simulation.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    snapshot_path = source / "experiment.toml"
    if snapshot_path.is_file():
        with snapshot_path.open("rb") as handle:
            snapshot = tomllib.load(handle)
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"{snapshot_path} must contain TOML tables")
        config = _supported_config_from_mapping(snapshot)
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        config = _clone_config_from_metadata(metadata)
    config["experiment"]["name"] = name
    return create_experiment(name, experiments_root=experiments_root, source_config=config)
