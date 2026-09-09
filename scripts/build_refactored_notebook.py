#!/usr/bin/env python3
"""Build the function-oriented reference notebook from readable source cells."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/07_refactored_forward_model.ipynb"


def markdown(source: str):
    """Create one Markdown cell."""
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    """Create one code cell."""
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells = [
    markdown(
        r"""
# Refactored physical forward model

This is the maintained executable reference for the function-oriented model.
It reproduces the accepted static result from `06_repaired_forward_model.ipynb`
and retains the two supported dynamics modes:

- `shared_period`: one manually selected longitudinal period;
- `luna_2022_curvature`: one longitudinal period per thread from dip curvature.

All model state is plain data: dictionaries, scalars, and NumPy arrays. Raw
observations are read-only. Generated files go to `simulations/` or `scratch/`.
"""
    ),
    markdown(
        """
## Setup and controls

Edit the controls cell for normal experiments. `FILAMENT_NOTEBOOK_FAST=1`
selects three frames, suppresses videos, and writes under `scratch/`; it is
used only for automated top-to-bottom validation.

Scientific controls come from `configs/default_experiment.toml`, the same
validated profile used when the GUI creates a new experiment.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filaments import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    OSCILLATION_MODE_LUNA_2022,
    OSCILLATION_MODE_SHARED_PERIOD,
    extract_time_distance,
    fit_damped_sine,
    generate_from_h5_background,
    load_h5_background_sequence,
    load_experiment_config,
    make_dynamics_config,
    make_export_config,
    measure_static_cohesion,
    padded_mask_bounds,
    save_gong_video,
    save_static_result,
    save_velocity_video,
    simulate_and_save_filament_dynamics,
    track_dark_band,
    validate_experiment_config,
)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.grid": False,
    "image.cmap": "gray",
})
print(f"Project root: {ROOT}")
"""
    ),
    code(
        r"""
FAST_VALIDATION = os.getenv("FILAMENT_NOTEBOOK_FAST", "0") == "1"

EXPERIMENT_CONFIG = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG_PATH)
RESOLVED_EXPERIMENT = validate_experiment_config(EXPERIMENT_CONFIG)
SEED = RESOLVED_EXPERIMENT["static_seed"]
CONFIG_OVERRIDES = RESOLVED_EXPERIMENT["static_overrides"]

SAVE_STATIC_RESULT = False
RUN_DYNAMICS = True
SAVE_DYNAMICS = True
MAKE_GONG_VIDEO = not FAST_VALIDATION
MAKE_VELOCITY_VIDEO = not FAST_VALIDATION
RUN_TIME_DISTANCE_FIT = False

DYNAMICS_MODE = RESOLVED_EXPERIMENT["dynamics"]["oscillation_mode"]
DYNAMICS_FRAME_COUNT = (
    3 if FAST_VALIDATION else int(RESOLVED_EXPERIMENT["dynamics"]["n_frames"])
)

default_output = ROOT / "scratch/refactored_forward_model"
OUTPUT_ROOT = Path(os.getenv("FILAMENT_NOTEBOOK_OUTPUT_ROOT", default_output))
SIMULATIONS_ROOT = OUTPUT_ROOT / "simulations" if FAST_VALIDATION else ROOT / "simulations"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

H5_BACKGROUND_PATH = RESOLVED_EXPERIMENT["inputs"]["h5_background_path"]

assert isinstance(SEED, int) and SEED >= 0
print(json.dumps({
    "fast_validation": FAST_VALIDATION,
    "seed": SEED,
    "h5_background": str(H5_BACKGROUND_PATH),
    "dynamics_mode": DYNAMICS_MODE,
    "dynamics_frames": DYNAMICS_FRAME_COUNT,
}, indent=2))
"""
    ),
    markdown(
        r"""
## Real GONG context

The loader deterministically selects one quiet crop from HDF5 frame zero, then
uses those exact pixel bounds for every video frame. The raw background is
retained pixel-for-pixel; only the synthetic transmission is blurred and
detector-integrated.
"""
    ),
    code(
        r"""
background_started = time.perf_counter()
backgrounds = load_h5_background_sequence(
    H5_BACKGROUND_PATH,
    n_frames=DYNAMICS_FRAME_COUNT if RUN_DYNAMICS else 1,
    seed=RESOLVED_EXPERIMENT["dynamics"]["seed"],
    **RESOLVED_EXPERIMENT["dynamic_background"],
)
BACKGROUND_SELECTION_SECONDS = time.perf_counter() - background_started
context_image = backgrounds["frames"][0]
context_support = np.isfinite(context_image) & (context_image > 0.0)

print("HDF5 crop:", backgrounds["crop_xyxy_px"])
print("Disk mu:", backgrounds["disk_mu"])
print("Native pixel:", f"{backgrounds['native_pixel_km']:.3f} km")
print("Shape:", context_image.shape)
print("Selection:", f"{BACKGROUND_SELECTION_SECONDS:.2f} s")
print("Supported percentiles:", np.percentile(context_image[context_support], [1, 50, 99]))

figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
axes[0].imshow(context_image, origin="lower", cmap="gray")
axes[0].set_title("HDF5 frame-zero background")
axes[0].set_axis_off()
axes[1].hist(context_image[context_support], bins=80, color=plt.colormaps["viridis"](0.55))
axes[1].set_xlabel("Normalized intensity")
axes[1].set_ylabel("Pixels")
axes[1].set_title("Supported GONG intensity")
plt.show()
"""
    ),
    markdown(
        r"""
## Static filament

The canonical sequence is measured spine $\rightarrow$ dipped threads
$\rightarrow$ hydrostatic plasma/PCTR $\rightarrow$ H$\alpha$ optical depth
$\rightarrow$ independently degraded transmission $\rightarrow$ GONG composite.
"""
    ),
    code(
        r"""
started = time.perf_counter()
result = generate_from_h5_background(
    backgrounds,
    seed=SEED,
    config_overrides=CONFIG_OVERRIDES,
)
elapsed_seconds = time.perf_counter() - started
config = result["config"]
arrays = result["arrays"]
metadata = result["metadata"]

background = arrays["background"].astype(float)
final_image = arrays["degraded_intensity"].astype(float)
physical_mask = arrays["filament_mask"].astype(bool)
support = arrays["support"].astype(bool)
absorption = np.clip(1.0 - final_image / np.maximum(background, 1.0e-6), 0.0, 1.0)

print(f"Generation: {elapsed_seconds:.2f} s")
print("Spine:", metadata["spine_source"])
print("Spine index:", metadata["spine_library_index"])
print("Spine length:", f"{metadata['spine_length_mm']:.3f} Mm")
print("Threads:", metadata["n_threads_generated"])
print("Tau max:", metadata["tau_max"])
print("Effective source fraction:", metadata["source_fraction_effective"])
print("Thread placement:", json.dumps(metadata["thread_placement"], indent=2))
print("Opacity-table saturation:", json.dumps(metadata["opacity_table_saturation"], indent=2))
"""
    ),
    code(
        r"""
figure, axes = plt.subplots(1, 2, figsize=(7, 3.3), constrained_layout=True)
for axis, image, title in zip(
    axes,
    (background, final_image),
    ("Raw HDF5 background", "Background + synthetic filament"),
    strict=True,
):
    axis.imshow(image, origin="lower", cmap="gray")
    axis.set_title(title)
    axis.set_axis_off()
if SAVE_STATIC_RESULT:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_ROOT / "static_overview.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_ROOT / "static_overview.png", dpi=300, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## Geometry, plasma, and image-formation diagnostics"),
    code(
        r"""
thread_indices = np.linspace(0, len(result["threads"]) - 1, 70, dtype=int)
figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for index in thread_indices:
    thread = result["threads"][index]
    axes[0].plot(thread["x"] / 1_000.0, thread["y"] / 1_000.0, color="#31688E", alpha=0.18)
axes[0].plot(
    result["spine"]["x"] / 1_000.0,
    result["spine"]["y"] / 1_000.0,
    color="#FDE725",
    linewidth=2.4,
    label="measured spine",
)
axes[0].set_aspect("equal")
axes[0].set_xlabel("x [Mm]")
axes[0].set_ylabel("y [Mm]")
axes[0].legend()
for index in thread_indices[:40]:
    thread = result["threads"][index]
    axes[1].plot(thread["s"] / thread["s"][-1], thread["z"] / 1_000.0, color="#35B779", alpha=0.25)
axes[1].set_xlabel("Normalized thread arclength")
axes[1].set_ylabel("Height [Mm]")
axes[1].set_title("Magnetic-dip profiles")
plt.show()

tau0 = np.asarray([thread["tau0"] for thread in result["threads"]])
dip_depth = np.asarray([thread["dip_depth_km"] for thread in result["threads"]])
filled_depth = np.asarray([thread["filled_depth_km"] for thread in result["threads"]])
print("Tau0 percentiles:", np.percentile(tau0, [1, 50, 90, 99]))
print("Dip-depth percentiles [km]:", np.percentile(dip_depth, [10, 50, 90]))
print("Exact filled-depth percentiles [km]:", np.percentile(filled_depth, [10, 50, 90]))
"""
    ),
    code(
        r"""
row0, row1, column0, column1 = padded_mask_bounds(physical_mask)
native_slice = np.s_[row0:row1, column0:column1]
factor = config["downsample_factor"]
highres_slice = np.s_[row0 * factor:row1 * factor, column0 * factor:column1 * factor]
tau_native = -np.log(np.maximum(1.0 - arrays["soft_mask"], 1.0e-12))

figure, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
panels = (
    (arrays["tau_map"][highres_slice], r"Internal line-center $\tau$"),
    (arrays["soft_mask_highres"][highres_slice], "Internal physical absorption"),
    (arrays["soft_mask"][native_slice], "Native physical absorption"),
    (arrays["observable_soft_mask"][native_slice], "GONG-passband absorption"),
)
for axis, (image, title) in zip(axes, panels, strict=True):
    artist = axis.imshow(image, origin="lower", cmap="viridis", vmin=0.0)
    axis.set_title(title)
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, fraction=0.045)
plt.show()

diagnostics = measure_static_cohesion(
    final_image,
    background,
    physical_mask,
    arrays["soft_mask"],
    arrays["tau_map"],
)
print(json.dumps(diagnostics, indent=2))
assert np.array_equal(background, context_image.astype(float))
assert diagnostics["observable_components"] == 1
assert np.isclose(
    config["pixel_size_km"] * factor,
    backgrounds["native_pixel_km"],
)
print("PASS — static physical and background-integrity checks")
"""
    ),
    code(
        r"""
STATIC_OUTPUT_PATH = None
if SAVE_STATIC_RESULT:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    STATIC_OUTPUT_PATH = save_static_result(result, OUTPUT_ROOT / f"static-{stamp}")
    print("Saved static result:", STATIC_OUTPUT_PATH)
"""
    ),
    markdown(
        r"""
## Dynamics on the same aligned real HDF5 backgrounds

The static preview is already frame zero: no second background selection,
translation, or re-rendering occurs. Each later frame uses the aligned,
unchanged real background at that timestamp. Luna mode derives the longitudinal
period of thread $i$ from its dip curvature radius $R_i$ and realized
dip-bottom height $h_i$:

$$
g(h_i)=g_0\left(\frac{R_\odot}{R_\odot+h_i}\right)^2,
\qquad
P_i = 2\pi\left[g(h_i)\left(\frac{1}{R_i}+\frac{1}{R_\odot+h_i}\right)\right]^{-1/2},
$$

where $g_0=274\ \mathrm{m\,s^{-2}}$ is photospheric gravity and
$R_\odot=696.3\ \mathrm{Mm}$ is the solar radius. Manual mode uses `period_s`
for every thread. Both modes retain the same localized damping and Brownian
walk.
"""
    ),
    code(
        r"""
dynamics_values = dict(RESOLVED_EXPERIMENT["dynamics"])
dynamics_values["n_frames"] = DYNAMICS_FRAME_COUNT
DYNAMICS_CONFIG = make_dynamics_config(**dynamics_values)
EXPORT_CONFIG = make_export_config(**RESOLVED_EXPERIMENT["export"])

SAVED_DYNAMICS = None
DYNAMICS_TIMING = {}
if RUN_DYNAMICS:
    DYNAMICS_TIMING["background_selection_s"] = BACKGROUND_SELECTION_SECONDS
    dynamics_initial = result
    assert np.array_equal(
        dynamics_initial["arrays"]["background"],
        backgrounds["frames"][0],
    )
    if SAVE_DYNAMICS:
        simulation_started = time.perf_counter()
        SAVED_DYNAMICS = simulate_and_save_filament_dynamics(
            dynamics_initial,
            DYNAMICS_CONFIG,
            background_frames=backgrounds["frames"],
            simulations_root=SIMULATIONS_ROOT,
            export_config=EXPORT_CONFIG,
            label="refactored-notebook",
        )
        DYNAMICS_TIMING["simulation_s"] = time.perf_counter() - simulation_started
        DYNAMICS_TIMING["simulation_s_per_frame"] = (
            DYNAMICS_TIMING["simulation_s"] / DYNAMICS_CONFIG["n_frames"]
        )
    print("HDF5 crop:", backgrounds["metadata"]["crop_xyxy_px"])
    print("Detector boxes:", backgrounds["metadata"]["n_exclusion_boxes"])
    print("Dynamics mode:", DYNAMICS_CONFIG["oscillation_mode"])
    print("Saved dynamics:", None if SAVED_DYNAMICS is None else SAVED_DYNAMICS["directory"])
    print("Dynamics timing [s]:", json.dumps(DYNAMICS_TIMING, indent=2))
"""
    ),
    markdown("## Saved dynamics inspection"),
    code(
        r"""
TIME_DISTANCE = None
FIT = None
if SAVED_DYNAMICS is not None:
    with h5py.File(SAVED_DYNAMICS["h5_path"], "r") as handle:
        required = (
            "video/raw_gong",
            "state/thread_displacements_km",
            "labels/thread_mask_native",
            "labels/coherent_velocity_xy_km_s",
            "labels/opacity_change_native",
        )
        assert all(path in handle for path in required)
        print("Schema:", handle.attrs["simulation_dataset_schema_version"])
        print("Raw GONG:", handle["video/raw_gong"].shape)
        print("Thread displacement:", handle["state/thread_displacements_km"].shape)
        print("Velocity labels:", handle["labels/coherent_velocity_xy_km_s"].shape)
        print("Statistics:", json.loads(handle.attrs["statistics_json"]))

        first_frame = np.asarray(handle["video/raw_gong"][0])
        first_mask = np.asarray(handle["labels/thread_mask_native"][0], dtype=bool)
        rows, columns = np.where(first_mask)
        slit_x = float(np.median(columns))
        slit_y = float(np.median(rows))
        raw_video = np.asarray(handle["video/raw_gong"])
        TIME_DISTANCE = extract_time_distance(
            raw_video,
            x=slit_x,
            y=slit_y,
            angle_deg=0.0,
            width_px=5.0,
            length_px=120.0,
        )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(first_frame, origin="lower", cmap="gray")
    axes[0].plot(TIME_DISTANCE["slit_x_px"][0], TIME_DISTANCE["slit_y_px"][0], color="#D55E00")
    axes[0].set_title("Frame zero and analysis slit")
    axes[1].imshow(TIME_DISTANCE["values"].T, origin="lower", aspect="auto", cmap="gray")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Distance sample")
    axes[1].set_title("Time-distance diagram")
    plt.show()

    if RUN_TIME_DISTANCE_FIT:
        ridge = track_dark_band(TIME_DISTANCE["values"].T)
        times_s = np.arange(ridge.size) * DYNAMICS_CONFIG["cadence_s"]
        onset_s = float(DYNAMICS_CONFIG["oscillation_start_time_s"])
        post_onset = times_s >= onset_s
        usable_samples = int(np.count_nonzero(post_onset & np.isfinite(ridge)))
        if not np.isclose(DYNAMICS_CONFIG["phase_rad"], 0.0):
            print(
                "Time-distance fit skipped: this optional notebook fit supports "
                "phase_rad = 0 because the dynamics driver includes a decaying "
                "phase-offset term for nonzero phase."
            )
        elif usable_samples < 10:
            print(
                "Time-distance fit skipped: fewer than 10 finite post-onset "
                f"samples are available ({usable_samples}); the run may end "
                "before or too soon after oscillation onset."
            )
        else:
            FIT = fit_damped_sine(
                ridge[post_onset],
                time_s=times_s[post_onset] - onset_s,
                period_bounds_s=(20.0 * 60.0, 180.0 * 60.0),
            )
            print(json.dumps(FIT["parameters"], indent=2))
"""
    ),
    code(
        r"""
GONG_VIDEO_PATH = None
VELOCITY_VIDEO_PATH = None
if SAVED_DYNAMICS is not None and MAKE_GONG_VIDEO:
    video_started = time.perf_counter()
    GONG_VIDEO_PATH = save_gong_video(
        SAVED_DYNAMICS["h5_path"],
        OUTPUT_ROOT / "dynamics_gong.mp4",
        fps=RESOLVED_EXPERIMENT["video"]["fps"],
    )
    print(f"GONG video ({time.perf_counter() - video_started:.2f} s):", GONG_VIDEO_PATH)
if SAVED_DYNAMICS is not None and MAKE_VELOCITY_VIDEO:
    video_started = time.perf_counter()
    VELOCITY_VIDEO_PATH = save_velocity_video(
        SAVED_DYNAMICS["h5_path"],
        OUTPUT_ROOT / "dynamics_velocity.mp4",
        fps=RESOLVED_EXPERIMENT["video"]["fps"],
        velocity_limit_km_s=RESOLVED_EXPERIMENT["video"]["velocity_limit_km_s"],
        quiver_stride_px=RESOLVED_EXPERIMENT["video"]["quiver_stride_px"],
    )
    print(f"Velocity video ({time.perf_counter() - video_started:.2f} s):", VELOCITY_VIDEO_PATH)
"""
    ),
    markdown("## Final checks"),
    code(
        r"""
status = {
    "operation": metadata["synthesis_operation"],
    "h5_crop": list(backgrounds["crop_xyxy_px"]),
    "seed": SEED,
    "spine_source": metadata["spine_source"],
    "threads": metadata["n_threads_generated"],
    "tau_max": metadata["tau_max"],
    "observable_components": diagnostics["observable_components"],
    "dynamics_mode": DYNAMICS_CONFIG["oscillation_mode"] if RUN_DYNAMICS else None,
    "dynamics_frames": DYNAMICS_CONFIG["n_frames"] if RUN_DYNAMICS else 0,
    "simulation_h5": None if SAVED_DYNAMICS is None else str(SAVED_DYNAMICS["h5_path"]),
}
assert status["observable_components"] == 1
print(json.dumps(status, indent=2))
print("PASS — refactored forward-model notebook completed top-to-bottom")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (synthetic-filaments)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
