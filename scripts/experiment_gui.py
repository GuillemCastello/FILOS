#!/usr/bin/env python3
"""Terminal-launched Streamlit interface for local filament experiments."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from synthetic_filaments.experiment_config import (  # noqa: E402
    clone_simulation_as_experiment,
    create_experiment,
    experiment_config_sha256,
    list_experiments,
    list_standalone_simulations,
    load_experiment_config,
    next_background_seed,
    next_experiment_seeds,
    save_experiment_config,
    validate_save_config,
)
from synthetic_filaments.experiment_runner import (  # noqa: E402
    generate_experiment_preview,
    list_jobs,
    preview_is_current,
    start_experiment_worker,
)
from synthetic_filaments.oscillation import luna_2022_longitudinal_period_s  # noqa: E402

st.set_page_config(page_title="Synthetic filament experiments", page_icon="☀️", layout="wide")

STATIC_MORPHOLOGY = {
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
}

FIELD_HELP = {
    "spine_library_entry_index": "Automatic selection preserves seed-driven morphology.",
    "spine_library_length_bounds_km": "Eligible measured-spine true-length interval.",
    "thread_density_per_mm": "Mean thread population per megametre of spine arclength.",
    "height_mean_km": "Mean dip-bottom height; each realized thread height enters the Luna model.",
    "height_std_km": "Dip-bottom height spread; realized values are bounded to 1–100 Mm.",
    "dip_curvature_radius_min_km": "Minimum circular-dip curvature radius.",
    "dip_curvature_radius_max_km": "Maximum circular-dip curvature radius.",
    "dip_curvature_radius_median_km": (
        "Median curvature radius R. Together with height, this sets the predominant Luna period."
    ),
    "dip_curvature_radius_sigma_ln": (
        "Natural-log standard deviation of the curvature-radius distribution."
    ),
    "column_mass_g_cm2": (
        "Requested column mass along each thread. Loading is limited by the dip's "
        "hydrostatic capacity; actual mass and saturation are saved with the result."
    ),
    "source_fraction": (
        "Recorded for compatibility. Static synthesis uses the published "
        "height-dependent source-function interpolation."
    ),
    "use_detector": "Exclude HDF5 crop candidates that overlap detected real filaments.",
    "oscillation_mode": (
        "Manual uses one shared period; Luna derives each longitudinal period from dip curvature "
        "and realized dip-bottom height."
    ),
    "period_s": "Shared longitudinal period in manual mode; retained as metadata in Luna mode.",
    "transverse_period_s": "Used in Luna mode. Manual mode uses period_s for both components.",
    "center_spine_fraction": "Automatic selects a deterministic material-bearing thread from the seed.",
    "center_height_km": "Automatic uses the selected material-bearing thread centroid height.",
    "compression": "LZF is the maintained fast lossless default.",
    "video_lower_percentile": "Fixed frame-zero lower contrast percentile.",
    "video_upper_percentile": "Fixed frame-zero upper contrast percentile.",
    "velocity_limit_km_s": "Automatic uses the maximum coherent velocity in the completed run.",
    "quiver_stride_px": (
        "Arrow sampling-cell size. Smaller values show more arrows. Direction follows the "
        "velocity vector and arrow length increases linearly with local speed."
    ),
}


def _label(name: str) -> str:
    """Return a compact human-readable field label."""
    replacements = {"km": "km", "px": "px", "mm": "Mm", "s": "s", "fps": "FPS"}
    words = [replacements.get(word, word.capitalize()) for word in name.split("_")]
    return " ".join(words)


def _number_input(
    label: str,
    value: int | float,
    *,
    key: str,
    help_text: str | None,
    disabled: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    """Render a number input while preserving integer and floating-point types."""
    if isinstance(value, int) and not isinstance(value, bool):
        return int(
            st.number_input(
                label,
                value=value,
                step=1,
                key=key,
                help=help_text,
                disabled=disabled,
            )
        )
    numeric_value = float(value)
    step = max(abs(numeric_value) * 0.05, 1.0e-8)
    bounds = {}
    if (minimum is None or numeric_value >= minimum) and (
        maximum is None or numeric_value <= maximum
    ):
        if minimum is not None:
            bounds["min_value"] = float(minimum)
        if maximum is not None:
            bounds["max_value"] = float(maximum)
    return float(
        st.number_input(
            label,
            value=numeric_value,
            step=step,
            format="%.10g",
            key=key,
            help=help_text,
            disabled=disabled,
            **bounds,
        )
    )


def _edit_field(
    section: str,
    name: str,
    value: Any,
    prefix: str,
    *,
    disabled: bool = False,
) -> Any:
    """Render one typed configuration widget and return its current value."""
    key = f"{prefix}:{section}:{name}"
    label = _label(name)
    help_text = FIELD_HELP.get(name)
    if name == "oscillation_mode":
        options = ("luna_2022_curvature", "shared_period")
        return st.selectbox(
            label,
            options,
            index=options.index(value),
            key=key,
            help=help_text,
            disabled=disabled,
        )
    if name == "compression":
        options = ("lzf", "gzip", "none")
        return st.selectbox(
            label,
            options,
            index=options.index(value),
            key=key,
            help=help_text,
            disabled=disabled,
        )
    if name in {
        "spine_library_entry_index",
        "center_height_km",
        "center_spine_fraction",
        "velocity_limit_km_s",
    }:
        automatic = value == "auto"
        mode_key = f"{key}:mode"
        mode = st.selectbox(
            label,
            ("auto", "fixed"),
            index=0 if automatic else 1,
            key=mode_key,
            help=help_text,
            disabled=disabled,
        )
        if mode == "auto":
            return "auto"
        fallback = {
            "spine_library_entry_index": 0,
            "center_spine_fraction": 0.5,
        }.get(name, 1.0)
        return _number_input(
            f"{label} value",
            fallback if automatic else value,
            key=key,
            help_text=help_text,
            disabled=disabled,
        )
    if name == "spine_library_length_bounds_km":
        columns = st.columns(2)
        with columns[0]:
            lower = _number_input(
                "Minimum spine length km",
                value[0],
                key=f"{key}:minimum",
                help_text=help_text,
                disabled=disabled,
            )
        with columns[1]:
            upper = _number_input(
                "Maximum spine length km",
                value[1],
                key=f"{key}:maximum",
                help_text=help_text,
                disabled=disabled,
            )
        return [lower, upper]
    if isinstance(value, bool):
        return st.checkbox(
            label,
            value=value,
            key=key,
            help=help_text,
            disabled=disabled,
        )
    if isinstance(value, (int, float)):
        limits = {
            "width_modulation_amplitude": (0.0, 1.0),
            "temp_center_K": (6_000.0, 14_000.0),
        }.get(name, (None, None))
        return _number_input(
            label,
            value,
            key=key,
            help_text=help_text,
            disabled=disabled,
            minimum=limits[0],
            maximum=limits[1],
        )
    return st.text_input(
        label,
        value=str(value),
        key=key,
        help=help_text,
        disabled=disabled,
    )


def _edit_fields(
    config: dict[str, dict[str, Any]],
    section: str,
    names: list[str],
    prefix: str,
    *,
    disabled_names: set[str] | None = None,
) -> None:
    """Update one configuration section from its rendered widgets."""
    for name in names:
        config[section][name] = _edit_field(
            section,
            name,
            config[section][name],
            prefix,
            disabled=name in (disabled_names or set()),
        )


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object for read-only display."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _tail(path: Path, maximum_lines: int = 35) -> str:
    """Return the end of one UTF-8 worker log."""
    if not path.is_file():
        return "Waiting for worker output."
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-maximum_lines:]).rstrip() or "Waiting for worker output."


def _set_active_experiment(directory: Path) -> None:
    """Open one editable experiment in this browser session."""
    st.session_state["active_kind"] = "experiment"
    st.session_state["active_path"] = str(directory.resolve())


def _set_active_new(config: Mapping[str, Any]) -> None:
    """Open one unsaved experiment draft without creating application files."""
    st.session_state["active_kind"] = "new"
    st.session_state.pop("active_path", None)
    st.session_state["draft:new"] = deepcopy(dict(config))


def _set_active_simulation(directory: Path) -> None:
    """Open one standalone simulation read-only in this browser session."""
    st.session_state["active_kind"] = "simulation"
    st.session_state["active_path"] = str(directory.resolve())


def _close_active() -> None:
    """Unload the current item without affecting background workers."""
    st.session_state.pop("active_kind", None)
    st.session_state.pop("active_path", None)


def _clear_editor_widgets(prefix: str) -> None:
    """Discard cached editor values after changing the TOML outside its form."""
    for key in list(st.session_state):
        if str(key).startswith(f"{prefix}:"):
            del st.session_state[key]


def _advance_editor_revision(prefix: str) -> int:
    """Force Streamlit to remount the form with every expander closed."""
    key = f"editor-revision:{prefix}"
    revision = int(st.session_state.get(key, 0)) + 1
    st.session_state[key] = revision
    return revision


def _home() -> None:
    """Render New and Load controls."""
    st.title("Synthetic filament experiments")
    st.caption("Local configuration, preview, HDF5 generation, and video rendering.")
    new_tab, load_tab = st.tabs(("New", "Load"))
    with new_tab:
        with st.form("new-experiment"):
            name = st.text_input("Experiment name", placeholder="damping-time-scan")
            submitted = st.form_submit_button("Create experiment", type="primary")
        if submitted:
            try:
                config = load_experiment_config(check_inputs=False)
                config["experiment"]["name"] = name
                validate_save_config(config, check_inputs=False)
            except Exception as error:
                st.error(f"{type(error).__name__}: {error}")
            else:
                _set_active_new(config)
                st.rerun()

    with load_tab:
        experiments = list_experiments()
        if experiments:
            labels = [f"{item['name']} — {item['directory'].name}" for item in experiments]
            selected = st.selectbox("Editable experiments", range(len(labels)), format_func=labels.__getitem__)
            if st.button("Load experiment", type="primary"):
                _set_active_experiment(experiments[selected]["directory"])
                st.rerun()
        else:
            st.info("No GUI experiments yet.")

        standalone = list_standalone_simulations()
        if standalone:
            labels = [item["simulation_id"] for item in standalone]
            selected = st.selectbox(
                "Standalone simulations (read-only)",
                range(len(labels)),
                format_func=labels.__getitem__,
            )
            if st.button("Load standalone simulation"):
                _set_active_simulation(standalone[selected]["directory"])
                st.rerun()


def _derived_preview(preview: Mapping[str, Any]) -> None:
    """Display realized, read-only preview metadata."""
    frame_zero = preview["frame_zero"]
    luna = preview.get("luna_dynamics", {})
    st.subheader("Realized preview")
    columns = st.columns(6)
    columns[0].metric("Threads", f"{frame_zero['n_threads']:,}")
    columns[1].metric("Spine index", frame_zero["spine_library_index"])
    columns[2].metric("Spine length", f"{frame_zero['spine_length_mm']:.2f} Mm")
    columns[3].metric("Disk μ", f"{frame_zero['disk_mu']:.3f}")
    columns[4].metric("Tau max", f"{frame_zero['tau_max']:.3f}")
    expected_period_s = luna.get("expected_period_s", {}).get("median")
    columns[5].metric(
        "Median Luna period",
        f"{expected_period_s / 60.0:.1f} min" if expected_period_s is not None else "—",
    )
    loading = preview.get("column_mass_loading")
    if loading:
        st.caption(
            f"Column mass: requested {loading['requested_g_cm2']:.3g}, "
            f"realized median {loading['realized_median_g_cm2']:.3g} g/cm² · "
            f"capacity-limited {loading['capacity_limited_thread_count']}/{frame_zero['n_threads']} · "
            "maximum converged |relative residual| "
            f"{loading.get('maximum_converged_absolute_relative_residual', 0.0):.3g}"
        )
    placement = preview.get("thread_placement", {})
    requested = placement.get("requested_thread_length_bounds_km")
    effective = placement.get("effective_thread_length_bounds_km")
    realized = placement.get("realized_thread_length_range_km")
    if requested and effective and realized:
        st.caption(
            f"Thread arc lengths — requested {requested[0] / 1_000:.2f}–"
            f"{requested[1] / 1_000:.2f} Mm; effective {effective[0] / 1_000:.2f}–"
            f"{effective[1] / 1_000:.2f} Mm; realized {realized[0] / 1_000:.2f}–"
            f"{realized[1] / 1_000:.2f} Mm. "
            f"{placement.get('thread_length_cap_explanation', '')}"
        )
    opacity = preview.get("opacity_diagnostics", {})
    if opacity:
        st.caption(
            "Opacity diagnostics — temperature-excluded loaded samples "
            f"{opacity.get('temperature_excluded_loaded_sample_count', 0):,}; "
            f"zero-opacity threads {opacity.get('entirely_zero_opacity_thread_count', 0):,}; "
            f"zero after foot taper {opacity.get('entirely_zero_after_foot_taper_thread_count', 0):,}."
        )
    st.caption(
        f"Shape {frame_zero['native_shape_yx']} · orientation {frame_zero['orientation_deg']:.2f}° · "
        f"chirality {frame_zero['chirality']:+d} · HDF5 crop {frame_zero['crop_xyxy_px']} · "
        f"pixel {frame_zero['native_pixel_km']:.2f} km · "
        f"limb direction {frame_zero['limb_direction_deg']:.2f}°"
    )


def _completed_run(directory: Path) -> None:
    """Display one completed run and its playable videos."""
    metadata_path = directory / "simulation.json"
    if not metadata_path.is_file():
        return
    metadata = _load_json(metadata_path)
    summary = metadata.get("summary", {})
    st.markdown(f"**{directory.name}**")
    st.caption(
        f"{summary.get('n_frames', '?')} frames · {summary.get('n_threads', '?')} threads · "
        f"{summary.get('oscillation_mode', '?')}"
    )
    gong_path = directory / "gong.mp4"
    velocity_path = directory / "velocity.mp4"
    gong_column, velocity_column = st.columns(2)
    if gong_path.is_file():
        with gong_column:
            st.caption("Hα intensity")
            st.video(str(gong_path))
    if velocity_path.is_file():
        with velocity_column:
            st.caption("Coherent velocity — color and arrow length: speed")
            st.video(str(velocity_path))


@st.fragment(run_every=2.0)
def _job_and_run_status(experiment: Path) -> None:
    """Refresh active worker progress, logs, and completed products."""
    jobs = list_jobs(experiment_directory=experiment)
    if jobs:
        latest = jobs[0]
        state = latest.get("state", "unknown")
        stage = latest.get("stage", "unknown").replace("_", " ")
        completed = int(latest.get("completed_frames", 0))
        total = max(int(latest.get("total_frames", 1)), 1)
        st.subheader("Latest generation job")
        st.write(f"{state.capitalize()} — {stage}")
        st.progress(min(completed / total, 1.0), text=f"{completed} / {total} frames")
        st.caption(f"Elapsed: {float(latest.get('elapsed_seconds', 0.0)):.1f} s")
        if latest.get("error"):
            st.error(latest["error"])
        with st.expander("Worker log", expanded=state == "failed"):
            st.code(_tail(Path(latest["job_directory"]) / "run.log"), language="text")

    run_directories = sorted((experiment / "runs").glob("sim-*"), reverse=True)
    if run_directories:
        st.subheader("Completed runs")
        _completed_run(run_directories[0])
        if len(run_directories) > 1:
            with st.expander(f"Older runs ({len(run_directories) - 1})"):
                for directory in run_directories[1:]:
                    _completed_run(directory)
    failed = sorted((experiment / "runs").glob("failed-*"), reverse=True)
    if failed:
        st.caption(f"Retained failed runs: {len(failed)}")


def _promote_new_session(experiment: Path) -> None:
    """Associate an unsaved draft and preview with its newly saved directory."""
    identifier = str(experiment.resolve())
    for name in ("draft", "preview", "preview-cache", "preview-timings"):
        old_key = f"{name}:new"
        if old_key in st.session_state:
            st.session_state[f"{name}:{identifier}"] = st.session_state.pop(old_key)
    _set_active_experiment(experiment)


def _preview_progress_callback(
    identifier: str,
    config: Mapping[str, Any],
) -> tuple[Any, Callable[[Mapping[str, Any]], None]]:
    """Create Streamlit placeholders and a callback for preview-stage progress."""
    stage_order = {
        name: index
        for index, name in enumerate(
            ("validation", "background", "detector", "spine", "geometry", "plasma", "render", "diagnostics")
        )
    }
    status = st.status("Preparing static preview…", expanded=True)
    detail = status.empty()
    progress = st.progress(0.0, text="Estimating remaining work…")
    stage_lines: dict[str, str] = {}
    timing_history = st.session_state.get(f"preview-timings:{identifier}", [])
    target_pixels = int(config["dynamic_background"]["crop_height_px"]) * int(
        config["dynamic_background"]["crop_width_px"]
    )
    detector_requested = bool(config["dynamic_background"]["use_detector"])
    current_thread_count: int | None = None
    current_sampled_point_count: int | None = None
    projected_placement_attempts: float | None = None
    background_cached: bool | None = None

    def callback(event: Mapping[str, Any]) -> None:
        nonlocal background_cached, current_thread_count
        nonlocal current_sampled_point_count, projected_placement_attempts
        stage = str(event["stage"])
        completed = event.get("completed")
        total = event.get("total")
        stage_fraction = 0.0
        if completed is not None and total not in (None, 0):
            stage_fraction = min(max(float(completed) / float(total), 0.0), 1.0)
        if stage == "geometry" and total not in (None, 0):
            current_thread_count = int(total)
            if event.get("sampled_point_count") is not None:
                current_sampled_point_count = int(event["sampled_point_count"])
            attempts = event.get("placement_attempts")
            if attempts is not None and completed not in (None, 0):
                projected_placement_attempts = float(attempts) * float(total) / float(completed)
        if stage == "background" and event.get("completed") == event.get("total"):
            background_cached = bool(event.get("cached"))
        index = stage_order.get(stage, 0)
        overall = min((index + stage_fraction) / len(stage_order), 1.0)
        elapsed = float(event.get("overall_elapsed_seconds", 0.0))
        cached = " · cached" if event.get("cached") else ""
        stage_lines[stage] = (
            f"{stage.replace('_', ' ').title()}: {event.get('description', '')} "
            f"({float(event.get('elapsed_seconds', 0.0)):.1f} s{cached})"
        )
        status.update(
            label=f"{stage.replace('_', ' ').title()} — {elapsed:.1f} s elapsed",
            state="running",
        )
        detail.markdown("  \n".join(stage_lines[name] for name in stage_order if name in stage_lines))
        comparable = [
            item
            for item in timing_history[-5:]
            if bool(item.get("detector_requested")) == detector_requested
            and (
                background_cached is None
                or bool(item.get("background_cached")) == background_cached
            )
        ]
        if comparable and current_thread_count is not None:
            estimates = []
            mean_length_km = 0.5 * (
                float(config["static"]["thread_length_min_km"])
                + float(config["static"]["thread_length_max_km"])
            )
            point_spacing_km = float(config["static"]["thread_point_spacing_km"])
            estimated_points = max(
                int(current_thread_count * (mean_length_km / point_spacing_km + 1.0)),
                current_thread_count,
            )
            target_points = current_sampled_point_count or estimated_points
            target_attempts = projected_placement_attempts or float(current_thread_count)
            for item in comparable:
                historical_threads = max(int(item["n_threads"]), 1)
                historical_points = max(
                    int(item.get("sampled_point_count", historical_threads)),
                    1,
                )
                historical_attempts = max(
                    int(item.get("placement_attempts", historical_threads)),
                    1,
                )
                historical_shape = item["native_shape_yx"]
                historical_pixels = max(int(historical_shape[0]) * int(historical_shape[1]), 1)
                work_scale = 0.25 * current_thread_count / historical_threads
                work_scale += 0.30 * target_points / historical_points
                work_scale += 0.25 * target_pixels / historical_pixels
                work_scale += 0.20 * target_attempts / historical_attempts
                estimates.append(float(item["total_seconds"]) * work_scale)
            estimate = max(float(sum(estimates) / len(estimates)) - elapsed, 0.0)
            text = f"Estimated remaining: {estimate:.1f} s (session work-scaled)"
        else:
            text = "Estimating remaining work from this session…"
        progress.progress(overall, text=text)

    return (status, callback)


def _experiment_page(experiment: Path | None) -> None:
    """Render one saved experiment or an unsaved in-memory draft."""
    identifier = "new" if experiment is None else str(experiment.resolve())
    draft_key = f"draft:{identifier}"
    preview_key = f"preview:{identifier}"
    cache_key = f"preview-cache:{identifier}"
    timing_key = f"preview-timings:{identifier}"
    prefix = f"editor:{identifier}"

    top = st.columns((5, 1))
    top[0].title("Unsaved experiment" if experiment is None else experiment.name)
    if top[1].button("Close", use_container_width=True):
        _close_active()
        st.rerun()

    config_path = None if experiment is None else experiment / "experiment.toml"
    if draft_key not in st.session_state:
        try:
            if config_path is None:
                raise RuntimeError("the unsaved draft is missing from this browser session")
            st.session_state[draft_key] = load_experiment_config(config_path, check_inputs=False)
        except Exception as error:
            st.error(f"Cannot load configuration: {type(error).__name__}: {error}")
            return

    action_notice_key = f"action-notice:{identifier}"
    action_notice = st.session_state.pop(action_notice_key, None)
    if action_notice:
        st.success(action_notice)

    preview = st.session_state.get(preview_key)

    def state_flags(current_config: Mapping[str, Any]) -> tuple[bool, bool]:
        """Return unsaved-draft and current-preview flags for the visible editor state."""
        if config_path is None:
            current_unsaved = True
        else:
            try:
                saved_config = load_experiment_config(config_path, check_inputs=False)
                current_unsaved = experiment_config_sha256(
                    current_config
                ) != experiment_config_sha256(saved_config)
            except Exception:
                current_unsaved = True
        try:
            current_preview = preview_is_current(current_config, preview)
        except Exception:
            current_preview = False
        return current_unsaved, current_preview

    config = deepcopy(st.session_state[draft_key])
    unsaved, preview_current = state_flags(config)
    state_columns = st.columns(2)
    saved_state_indicator = state_columns[0].empty()
    preview_state_indicator = state_columns[1].empty()
    saved_state_indicator.info(
        "Unsaved configuration changes" if unsaved else "Configuration matches saved TOML"
    )
    preview_state_indicator.info(
        "Static preview is current" if preview_current else "Static preview is absent or stale"
    )

    editor_revision = int(st.session_state.get(f"editor-revision:{prefix}", 0))

    with st.form(f"experiment-form:{prefix}:{editor_revision}"):
        seed_columns = st.columns(4)
        new_static_seed = seed_columns[0].form_submit_button(
            "🎲 New filament seed", use_container_width=True
        )
        new_background_seed = seed_columns[1].form_submit_button(
            "🎲 New background seed", use_container_width=True
        )
        new_dynamics_seed = seed_columns[2].form_submit_button(
            "🎲 New dynamics seed", use_container_width=True
        )
        new_all_seeds = seed_columns[3].form_submit_button(
            "🎲 New all seeds", use_container_width=True
        )
        st.caption(
            f"Explicit seeds — filament: {config['static']['seed']} · "
            f"background: {config['dynamic_background']['seed']} · "
            f"dynamics: {config['dynamics']['seed']}"
        )

        with st.expander("Experiment details"):
            config["experiment"]["name"] = st.text_input(
                "Name", value=config["experiment"]["name"], key=f"{prefix}:name"
            )
            config["experiment"]["description"] = st.text_area(
                "Description",
                value=config["experiment"]["description"],
                key=f"{prefix}:description",
            )

        with st.expander("Filament morphology"):
            _edit_fields(
                config,
                "static",
                [name for name in config["static"] if name in STATIC_MORPHOLOGY],
                prefix,
            )
            try:
                radius_km = float(config["static"]["dip_curvature_radius_median_km"])
                height_km = float(config["static"]["height_mean_km"])
                if radius_km <= 0.0:
                    raise ValueError("median curvature radius must be greater than zero")
                median_period_min = luna_2022_longitudinal_period_s(
                    1_000.0 * radius_km,
                    1_000.0 * height_km,
                ) / 60.0
            except (TypeError, ValueError, FloatingPointError) as error:
                st.warning(f"Period estimate unavailable: {error}")
            else:
                st.caption(
                    "Configured median-radius estimate at the mean height: "
                    f"{median_period_min:.1f} min."
                )

        with st.expander("Thread plasma and image formation"):
            _edit_fields(
                config,
                "static",
                [name for name in config["static"] if name not in STATIC_MORPHOLOGY],
                prefix,
                disabled_names={"source_fraction"},
            )
            st.caption(
                "Source fraction is retained for file compatibility; the active model uses "
                "the published height interpolation."
            )

        with st.expander("HDF5 background selection"):
            _edit_fields(config, "inputs", list(config["inputs"]), prefix)
            background_names = list(config["dynamic_background"])
            inactive_detector = (
                set()
                if config["dynamic_background"]["use_detector"]
                else {"detection_threshold", "box_expand_fraction", "box_expand_px"}
            )
            _edit_fields(
                config,
                "dynamic_background",
                background_names,
                prefix,
                disabled_names=inactive_detector,
            )

        with st.expander("Dynamics"):
            _edit_fields(config, "dynamics", list(config["dynamics"]), prefix)

        with st.expander("Export and video"):
            _edit_fields(config, "export", list(config["export"]), prefix)
            _edit_fields(config, "video", list(config["video"]), prefix)

        buttons = st.columns(3)
        save_requested = buttons[0].form_submit_button("Save configuration")
        preview_requested = buttons[1].form_submit_button("Generate preview", type="primary")
        run_requested = buttons[2].form_submit_button("Generate video")

    st.session_state[draft_key] = deepcopy(config)
    unsaved, preview_current = state_flags(config)
    saved_state_indicator.info(
        "Unsaved configuration changes" if unsaved else "Configuration matches saved TOML"
    )
    preview_state_indicator.info(
        "Static preview is current" if preview_current else "Static preview is absent or stale"
    )
    seed_requested = new_static_seed or new_background_seed or new_dynamics_seed or new_all_seeds
    if seed_requested:
        next_static, next_dynamics = next_experiment_seeds(
            int(config["static"]["seed"]),
            int(config["dynamics"]["seed"]),
        )
        new_background = next_background_seed(int(config["dynamic_background"]["seed"]))
        if new_static_seed or new_all_seeds:
            config["static"]["seed"] = next_static
        if new_background_seed or new_all_seeds:
            config["dynamic_background"]["seed"] = new_background
        if new_dynamics_seed or new_all_seeds:
            config["dynamics"]["seed"] = next_dynamics
        st.session_state[draft_key] = deepcopy(config)
        _clear_editor_widgets(prefix)
        _advance_editor_revision(prefix)
        st.session_state[action_notice_key] = (
            f"Draft seeds updated: filament={config['static']['seed']}, "
            f"background={config['dynamic_background']['seed']}, "
            f"dynamics={config['dynamics']['seed']}."
        )
        st.rerun()
    elif save_requested:
        try:
            validate_save_config(config)
            if experiment is None:
                saved_experiment = create_experiment(
                    str(config["experiment"]["name"]),
                    source_config=config,
                )
                _promote_new_session(saved_experiment)
            else:
                save_experiment_config(config, config_path)
                st.session_state[action_notice_key] = "Configuration saved."
            _clear_editor_widgets(prefix)
            _advance_editor_revision(prefix)
            st.rerun()
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
    elif preview_requested:
        status, progress_callback = _preview_progress_callback(identifier, config)
        preview_started = time.monotonic()
        try:
            new_preview = generate_experiment_preview(
                config,
                cached_stages=st.session_state.setdefault(cache_key, {}),
                progress_callback=progress_callback,
            )
        except Exception as error:
            status.update(label="Preview failed", state="error", expanded=True)
            st.error(f"{type(error).__name__}: {error}")
        else:
            elapsed = time.monotonic() - preview_started
            st.session_state[preview_key] = new_preview
            history = st.session_state.setdefault(timing_key, [])
            history.append(
                {
                    "total_seconds": elapsed,
                    "n_threads": new_preview["frame_zero"]["n_threads"],
                    "sampled_point_count": sum(
                        len(thread["s"]) for thread in new_preview["static_state"]["threads"]
                    ),
                    "placement_attempts": new_preview["thread_placement"][
                        "placement_attempts"
                    ],
                    "native_shape_yx": new_preview["frame_zero"]["native_shape_yx"],
                    "detector_requested": config["dynamic_background"]["use_detector"],
                    "background_cached": any(
                        event.get("stage") == "background" and event.get("cached") is True
                        for event in new_preview["progress_events"]
                    ),
                    "stage_durations_seconds": new_preview["stage_durations_seconds"],
                }
            )
            del history[:-5]
            preview = new_preview
            preview_current = True
            preview_state_indicator.info("Static preview is current")
            status.update(
                label=f"Static preview complete in {elapsed:.1f} s",
                state="complete",
                expanded=False,
            )
    elif run_requested:
        try:
            if preview is None or not preview_is_current(config, preview):
                raise ValueError("generate a current static preview before starting production")
            if experiment is None:
                experiment = create_experiment(
                    str(config["experiment"]["name"]),
                    source_config=config,
                )
                _promote_new_session(experiment)
            job = start_experiment_worker(
                experiment,
                user_config=config,
                preview=preview,
            )
            st.session_state[f"action-notice:{experiment.resolve()}"] = (
                f"Background job started: {job['job_id']}"
            )
            st.rerun()
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")

    if preview is None:
        st.info("No preview is retained in this browser session. Generate one before production.")
    else:
        if not preview_current:
            st.warning("Showing the last successful preview; its static inputs are stale.")
        _derived_preview(preview)
        _, comparison_column, _ = st.columns((1, 4, 1))
        comparison_column.image(
            preview["display"]["frame_zero_comparison_png"],
            caption="Exact frame zero: raw HDF5 background and synthetic filament",
        )
        _, geometry_column, _ = st.columns((1, 5, 1))
        geometry_column.image(
            preview["display"]["geometry_diagnostics_png"],
            caption="Realized spine, sampled threads, and morphology distributions",
        )
        _, luna_column, _ = st.columns((1, 5, 1))
        luna_column.image(
            preview["display"]["luna_dynamics_diagnostics_png"],
            caption=(
                "Expected Luna periods from each realized thread's curvature radius "
                "and dip-bottom height"
            ),
        )

    if experiment is not None:
        _job_and_run_status(experiment)


def _simulation_page(directory: Path) -> None:
    """Render one canonical standalone simulation read-only with clone support."""
    top = st.columns((5, 1))
    top[0].title(directory.name)
    top[0].caption("Read-only standalone simulation")
    if top[1].button("Close", use_container_width=True):
        _close_active()
        st.rerun()

    metadata_path = directory / "simulation.json"
    if not metadata_path.is_file():
        st.error(f"Missing {metadata_path}")
        return
    metadata = _load_json(metadata_path)
    summary = metadata.get("summary", {})
    columns = st.columns(4)
    columns[0].metric("Frames", summary.get("n_frames", "?"))
    columns[1].metric("Threads", summary.get("n_threads", "?"))
    columns[2].metric("Cadence", f"{summary.get('cadence_s', '?')} s")
    columns[3].metric("Mode", summary.get("oscillation_mode", "?"))
    st.json(summary)

    gong_path = directory / "gong.mp4"
    velocity_path = directory / "velocity.mp4"
    gong_column, velocity_column = st.columns(2)
    if gong_path.is_file():
        with gong_column:
            st.caption("Hα intensity")
            st.video(str(gong_path))
    if velocity_path.is_file():
        with velocity_column:
            st.caption("Coherent velocity — color and arrow length: speed")
            st.video(str(velocity_path))

    with st.form(f"clone:{directory.name}"):
        clone_name = st.text_input("New experiment name", value=f"clone-{directory.name[-18:]}")
        clone_requested = st.form_submit_button("Clone as experiment", type="primary")
    if clone_requested:
        try:
            experiment = clone_simulation_as_experiment(directory, clone_name)
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
        else:
            _set_active_experiment(experiment)
            st.rerun()


def main() -> None:
    """Route the current Streamlit session to its selected local item."""
    kind = st.session_state.get("active_kind")
    active_path = st.session_state.get("active_path")
    if kind is None:
        _home()
        return
    if kind == "new":
        _experiment_page(None)
    elif kind == "experiment" and active_path is not None:
        _experiment_page(Path(active_path))
    elif kind == "simulation":
        if active_path is None:
            _close_active()
            st.rerun()
        _simulation_page(Path(active_path))
    else:
        _close_active()
        st.rerun()


if __name__ == "__main__":
    main()
