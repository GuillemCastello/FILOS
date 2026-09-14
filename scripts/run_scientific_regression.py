#!/usr/bin/env python3
"""Run deterministic scientific checks against the notebook reference case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

import h5py
import numpy as np

from synthetic_filaments import (
    HEINZEL_EXTENSION_SHA256,
    OSCILLATION_MODE_LUNA_2022,
    OSCILLATION_MODE_SHARED_PERIOD,
    SIMULATION_DATASET_SCHEMA_VERSION,
    SOLAR_RADIUS_M,
    SOLAR_SURFACE_GRAVITY_M_S2,
    current_dynamics_frame,
    dynamics_background_at_frame,
    fit_damped_sine,
    generate_from_h5_background,
    gravity_cutoff_period_s,
    initialize_dynamics,
    load_h5_background_sequence,
    load_heinzel_opacity_table,
    load_static_result,
    luna_2022_inferred_curvature_radius_m,
    luna_2022_longitudinal_period_s,
    make_dynamics_config,
    make_export_config,
    measure_static_cohesion,
    save_static_result,
    simulate_and_save_filament_dynamics,
    solar_gravity_m_s2,
    step_dynamics,
    update_static_config,
)
from synthetic_filaments.geometry import make_spine, make_threads
from synthetic_filaments.plasma import MASS_INTEGRATION_MAX_INTERVALS, assign_plasma

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5 = ROOT / "tests/data/reference_background.h5"
HEINZEL_PROVENANCE_ROOT = ROOT / "processed/heinzel_table1_extension_final"

# September 2026 radius-only reference with nested Simpson mass integration.
REFERENCE_HASHES = {
    "background": "96AA45AFEF8091425DAFA74B524B708A38FAF1EF55419B93361E5785A9B276A9",
    "degraded_intensity": "4E5E1CE8D56E5FC539DA15759C5C7FEA09A53960C554AF955625057E038C81CD",
    "filament_mask": "23C0828B884AFE3C80117E4DD5BFB5C54D3DB76DC18EBF5604E91F2AE93D07BA",
    "filament_mask_highres": "65625AFC8ED2767C901DA4A1C194F72453AFA4FCE3B2DA5F76BE881C59B8AEEF",
    "highres_intensity": "26DEB25952D21931DD5F8A03D88420575ABA13D2F8C9F815B21AFBA0AE9DBFFE",
    "observable_soft_mask": "677B8E3613C2AEF6603CDD825B29A5F97524729745936B16F98BFB4759F98FF5",
    "soft_mask": "C6F398C45C35D855974C27772E2BD24A985BB3F13C82537DF97858950F90528C",
    "soft_mask_highres": "CDE5AA94DF845CD5DCFB9DEECB7C9F5D82825D6DEB4C4A3E4DA09C9C624765C9",
    "support": "99C793A704977A1353D20E4673DB06C696D753D4FBB2A289152923F71E078E06",
    "tau_map": "B8F48AFFFDEEFD6938BD59E2BFCD96CB48E2DA3FAD7B0C831A14CEF85A7E1EC5",
}


def check_heinzel_opacity_table() -> dict[str, object]:
    """Check the installed 1--100 Mm table and all published Table-1 anchors."""
    table = load_heinzel_opacity_table()
    archived_master = (
        HEINZEL_PROVENANCE_ROOT
        / "output/heinzel_calibrated_extension_1_to_100Mm.csv"
    )
    packaged_master = ROOT / "synthetic_filaments/data" / archived_master.name
    if archived_master.read_bytes() != packaged_master.read_bytes():
        raise AssertionError("packaged opacity table differs from the archived master table")
    if table["sha256"] != HEINZEL_EXTENSION_SHA256:
        raise AssertionError("loaded opacity-table checksum differs from its approved checksum")

    reference = np.genfromtxt(
        HEINZEL_PROVENANCE_ROOT / "data/heinzel2015_table1_reference.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    max_i_error = 0.0
    max_f_error = 0.0
    for row in reference:
        height_index = int(np.searchsorted(table["height_km"], row["height_Mm"] * 1_000.0))
        temperature_index = int(
            np.searchsorted(table["temperature_K"], row["temperature_K"])
        )
        pressure_index = int(
            np.searchsorted(table["pressure_dyn_cm2"], row["pressure_dyn_cm2"])
        )
        max_i_error = max(
            max_i_error,
            abs(
                float(table["ionization"][height_index, temperature_index, pressure_index])
                - float(row["i_heinzel2015"])
            ),
        )
        max_f_error = max(
            max_f_error,
            abs(
                float(table["f_1e16_cm3"][height_index, temperature_index, pressure_index])
                - float(row["f_heinzel2015_1e16_cm3"])
            ),
        )
    if max_i_error != 0.0 or max_f_error != 0.0:
        raise AssertionError(
            "calibrated extension does not exactly preserve all published anchors: "
            f"max_i_error={max_i_error}, max_f_error={max_f_error}"
        )
    return {
        "sha256": table["sha256"],
        "rows": table["row_count"],
        "height_domain_Mm": [
            float(table["height_km"][0] / 1_000.0),
            float(table["height_km"][-1] / 1_000.0),
        ],
        "published_anchor_rows": int(reference.size),
        "maximum_anchor_i_error": max_i_error,
        "maximum_anchor_f_error": max_f_error,
    }


def check_luna_height_model() -> dict[str, float]:
    """Check the h=0 limit, 100 Mm cut-off, and inversion round trip."""
    radii_m = np.asarray([25.0e6, 100.0e6, 500.0e6])
    legacy_h0_s = 2.0 * np.pi / np.sqrt(
        SOLAR_SURFACE_GRAVITY_M_S2 * (1.0 / radii_m + 1.0 / SOLAR_RADIUS_M)
    )
    generalized_h0_s = np.asarray(luna_2022_longitudinal_period_s(radii_m, 0.0))
    if not np.allclose(generalized_h0_s, legacy_h0_s, rtol=5.0e-15, atol=0.0):
        raise AssertionError("height-dependent Luna model does not recover the h=0 equation")

    surface_cutoff_s = float(gravity_cutoff_period_s(0.0))
    cutoff_100_s = float(gravity_cutoff_period_s(100.0e6))
    expected_cutoff_100_s = surface_cutoff_s * (1.0 + 100.0e6 / SOLAR_RADIUS_M) ** 1.5
    if not np.isclose(cutoff_100_s, expected_cutoff_100_s, rtol=5.0e-15, atol=0.0):
        raise AssertionError("100 Mm gravity cut-off violates the analytic height scaling")

    heights_m = np.asarray([0.0, 1.0e6, 20.0e6, 100.0e6])
    trial_radii_m = np.asarray([20.0e6, 80.0e6, 300.0e6])[:, None]
    periods_s = luna_2022_longitudinal_period_s(trial_radii_m, heights_m[None, :])
    inferred_m = luna_2022_inferred_curvature_radius_m(periods_s, heights_m[None, :])
    if not np.allclose(inferred_m, trial_radii_m, rtol=2.0e-14, atol=1.0e-6):
        raise AssertionError("height-dependent Luna inversion does not recover curvature radius")
    if not np.all(np.diff(np.asarray(solar_gravity_m_s2(heights_m))) < 0.0):
        raise AssertionError("solar gravity must decrease strictly over increasing positive heights")

    return {
        "surface_cutoff_min": surface_cutoff_s / 60.0,
        "cutoff_100_Mm_min": cutoff_100_s / 60.0,
        "maximum_h0_period_error_s": float(np.max(np.abs(generalized_h0_s - legacy_h0_s))),
        "maximum_inversion_relative_error": float(
            np.max(np.abs(inferred_m / trial_radii_m - 1.0))
        ),
    }


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous array bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest().upper()


def canonical_static_result(
    h5_path: Path,
    *,
    backgrounds: dict[str, object] | None = None,
) -> dict[str, object]:
    """Generate the exact HDF5-aligned static case selected in notebook 07."""
    if backgrounds is None:
        backgrounds = load_h5_background_sequence(
            h5_path,
            n_frames=1,
            seed=1236,
            start_index=0,
            frame_step=1,
        )
    return generate_from_h5_background(
        backgrounds,
        seed=1235,
        config_overrides={
            "spine_library_length_bounds_km": (150_000.0, 300_000.0),
            "thread_density_per_mm": 15.0,
            "column_mass_g_cm2": 1.0e-4,
            "gong_bandpass_line_fraction": 0.65,
            "psf_sigma_px": 4.0,
        },
    )


def check_static_result(result: dict[str, object], output_root: Path) -> dict[str, object]:
    """Check every stored reference array plus key physical diagnostics."""
    config = result["config"]
    expected_dip_fields = {
        "dip_curvature_radius_min_km",
        "dip_curvature_radius_max_km",
        "dip_curvature_radius_median_km",
        "dip_curvature_radius_sigma_ln",
    }
    actual_dip_fields = {name for name in config if name.startswith("dip_")}
    if actual_dip_fields != expected_dip_fields:
        raise AssertionError(
            f"static model does not use the radius-only dip contract: {actual_dip_fields}"
        )
    radii_km = np.asarray([thread["dip_radius_km"] for thread in result["threads"]])
    lengths_km = np.asarray([thread["length_km"] for thread in result["threads"]])
    depths_km = np.asarray([thread["dip_depth_km"] for thread in result["threads"]])
    expected_depths_km = radii_km * (1.0 - np.cos(lengths_km / (2.0 * radii_km)))
    if not np.allclose(depths_km, expected_depths_km, rtol=2.0e-14, atol=1.0e-10):
        raise AssertionError("derived dip depth violates its circular-arc equation")
    if not np.all(lengths_km <= np.pi * radii_km):
        raise AssertionError("a thread arc exceeds its radius-supported semicircle")
    if np.min(radii_km) < config["dip_curvature_radius_min_km"] or np.max(
        radii_km
    ) > config["dip_curvature_radius_max_km"]:
        raise AssertionError("sampled thread curvature radius exceeds its configured bounds")

    threads = result["threads"]
    converged_count = sum(bool(thread["column_mass_converged"]) for thread in threads)
    capacity_limited_count = sum(
        bool(thread["column_mass_capacity_limited"]) for thread in threads
    )
    if (converged_count, capacity_limited_count) != (1379, 809):
        raise AssertionError(
            "reference column-mass status counts changed: "
            f"converged={converged_count}, capacity_limited={capacity_limited_count}"
        )
    maximum_integration_intervals = max(
        int(thread["column_mass_integration_intervals"]) for thread in threads
    )
    if maximum_integration_intervals > MASS_INTEGRATION_MAX_INTERVALS:
        raise AssertionError("reference column-mass integration exceeded its budget")
    maximum_converged_residual = max(
        (
            abs(float(thread["column_mass_relative_residual"]))
            for thread in threads
            if thread["column_mass_converged"]
        ),
        default=0.0,
    )
    if maximum_converged_residual > 1.0e-4:
        raise AssertionError("reference column-mass solution exceeded its target tolerance")

    arrays = result["arrays"]
    actual_hashes = {name: array_sha256(arrays[name]) for name in REFERENCE_HASHES}
    mismatches = {
        name: {"expected": REFERENCE_HASHES[name], "actual": actual}
        for name, actual in actual_hashes.items()
        if actual != REFERENCE_HASHES[name]
    }
    if mismatches:
        raise AssertionError(f"static reference hashes changed: {mismatches}")

    metadata = result["metadata"]
    expected_scalars = {
        "spine_library_index": 1413,
        "n_threads_generated": 2188,
        "tau_max": 1.7897586201694728,
        "source_fraction_effective": 0.4301808940480987,
    }
    for name, expected in expected_scalars.items():
        actual = metadata[name]
        if isinstance(expected, float):
            if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-14):
                raise AssertionError(f"{name}: expected {expected!r}, received {actual!r}")
        elif actual != expected:
            raise AssertionError(f"{name}: expected {expected!r}, received {actual!r}")

    cohesion = measure_static_cohesion(
        arrays["degraded_intensity"],
        arrays["background"],
        arrays["filament_mask"],
        arrays["soft_mask"],
        arrays["tau_map"],
    )
    if cohesion["observable_components"] != 1:
        raise AssertionError(f"reference filament fragmented: {cohesion}")
    if not np.isclose(cohesion["largest_component_fraction"], 1.0):
        raise AssertionError(f"reference filament lost its dominant component: {cohesion}")

    saved_static = save_static_result(result, output_root / "static-result")
    with np.load(saved_static / "arrays/thread_geometry.npz", allow_pickle=False) as geometry:
        if geometry["thread_offsets"].size != 2189:
            raise AssertionError("persisted thread geometry has the wrong offset count")
        if geometry["filled_depth_km"].shape != (2188,):
            raise AssertionError("persisted filled-depth diagnostic is incomplete")
        for field in (
            "requested_column_mass_g_cm2", "realized_column_mass_g_cm2",
            "column_mass_capacity_g_cm2", "column_mass_capacity_limited",
            "loaded_length_fraction",
        ):
            if not np.array_equal(geometry[field], [t[field] for t in result["threads"]]):
                raise AssertionError(f"persisted {field} differs from the realized loading")
    restored = load_static_result(saved_static)
    for name, values in result["arrays"].items():
        if not np.array_equal(values, restored["arrays"][name]):
            raise AssertionError(f"static-state roundtrip changed array {name}")
    for name, values in result["spine"].items():
        if isinstance(values, np.ndarray) and not np.array_equal(values, restored["spine"][name]):
            raise AssertionError(f"static-state roundtrip changed spine field {name}")
    if not all(
        field in restored["threads"][0]
        for field in ("temperature_K", "pressure_dyn_cm2", "mean_molecular_mass", "loaded_mask")
    ):
        raise AssertionError("static-state roundtrip omitted thermodynamic thread arrays")
    return {
        "hashes": actual_hashes,
        "cohesion": cohesion,
        "mass_integration": {
            "converged_threads": converged_count,
            "capacity_limited_threads": capacity_limited_count,
            "maximum_intervals": maximum_integration_intervals,
            "maximum_converged_relative_residual": maximum_converged_residual,
        },
    }


def check_numerical_repairs() -> dict[str, object]:
    """Exercise optimizer scaling and adaptive low-mass loading deterministically."""
    time_s = np.arange(240, dtype=float) * 60.0
    ridge_px = 50.0 + 10.0 * np.exp(-time_s / 14_400.0) * np.sin(
        2.0 * np.pi * time_s / 3_600.0
    )
    fit = fit_damped_sine(ridge_px, time_s=time_s)
    fitted_parameters = fit["parameters"]
    for name, expected, tolerance in (
        ("amplitude_px", 10.0, 1.0e-5),
        ("period_s", 3_600.0, 1.0e-3),
        ("damping_time_s", 14_400.0, 1.0e-2),
    ):
        if not np.isclose(fitted_parameters[name], expected, rtol=0.0, atol=tolerance):
            raise AssertionError(
                f"scaled robust fit did not recover {name}: {fitted_parameters[name]}"
            )

    from synthetic_filaments import make_static_config

    config = make_static_config(
        seed=1235,
        native_shape=(512, 512),
        disk_mu=1.0,
        orientation_deg=0.0,
        chirality=1,
    )
    config = update_static_config(config, n_threads=1)
    rng = np.random.default_rng(1235)
    spine = make_spine(config, rng)
    threads, _placement = make_threads(config, spine, rng)
    realized = []
    tau_maxima = []
    for requested_mass in (1.0e-5, 1.0e-7, 1.0e-8):
        mass_config = update_static_config(config, column_mass_g_cm2=requested_mass)
        selected_threads = deepcopy(threads)
        assign_plasma(
            mass_config,
            selected_threads,
            np.random.default_rng(1235),
            spine_length=float(spine["s"][-1]),
        )
        thread = selected_threads[0]
        if not thread["column_mass_converged"]:
            raise AssertionError(f"low mass {requested_mass} did not report convergence")
        if abs(thread["column_mass_relative_residual"]) > 1.0e-4:
            raise AssertionError(f"low mass {requested_mass} exceeded target tolerance")
        realized.append(float(thread["realized_column_mass_g_cm2"]))
        tau_maxima.append(float(thread["tau0"] * np.max(thread["tau_along"])))
    if len(set(realized)) != len(realized) or len(set(tau_maxima)) != len(tau_maxima):
        raise AssertionError("low requested masses collapsed to one sampled plasma state")

    tight_radius_km = 21_000.0
    tight_length_km = np.pi * tight_radius_km * (1.0 - 1.0e-12)
    tight_config = update_static_config(
        config,
        thread_length_min_km=tight_length_km,
        thread_length_max_km=tight_length_km,
        dip_curvature_radius_min_km=tight_radius_km,
        dip_curvature_radius_max_km=tight_radius_km,
        dip_curvature_radius_median_km=tight_radius_km,
        dip_curvature_radius_sigma_ln=0.0,
    )
    tight_threads, _placement = make_threads(
        tight_config,
        spine,
        np.random.default_rng(20260907),
    )
    assign_plasma(
        tight_config,
        tight_threads,
        np.random.default_rng(20260907),
        spine_length=float(spine["s"][-1]),
    )
    tight_thread = tight_threads[0]
    if not tight_thread["column_mass_converged"]:
        raise AssertionError("tight near-semicircular dip did not converge")
    if abs(tight_thread["column_mass_relative_residual"]) > 1.0e-4:
        raise AssertionError("tight dip exceeded the requested mass tolerance")
    if tight_thread["column_mass_integration_intervals"] > MASS_INTEGRATION_MAX_INTERVALS:
        raise AssertionError("tight dip exceeded the adaptive integration budget")
    return {
        "robust_fit": {
            "amplitude_px": fitted_parameters["amplitude_px"],
            "period_s": fitted_parameters["period_s"],
            "damping_time_s": fitted_parameters["damping_time_s"],
            "recovery_trustworthy": fit["diagnostics"]["parameter_recovery_trustworthy"],
        },
        "requested_mass_g_cm2": [1.0e-5, 1.0e-7, 1.0e-8],
        "realized_mass_g_cm2": realized,
        "local_tau_maxima": tau_maxima,
        "tight_dip": {
            "curvature_radius_km": tight_radius_km,
            "length_km": tight_length_km,
            "depth_km": tight_thread["dip_depth_km"],
            "integration_intervals": tight_thread["column_mass_integration_intervals"],
            "relative_mass_residual": tight_thread["column_mass_relative_residual"],
        },
    }


def canonical_dynamics_config(n_frames: int) -> dict[str, object]:
    """Return the notebook-selected Luna dynamics controls."""
    return make_dynamics_config(
        seed=1236,
        n_frames=n_frames,
        cadence_s=60.0,
        oscillation_start_time_s=1_200.0,
        longitudinal_displacement_amplitude_km=18_000.0,
        transverse_displacement_amplitude_km=2_000.0,
        oscillation_mode=OSCILLATION_MODE_LUNA_2022,
        period_s=3_600.0,
        transverse_period_s=1_500.0,
        damping_time_s=14_400.0,
        phase_rad=0.0,
        brownian_step_min_km=200.0,
        brownian_step_max_km=500.0,
        center_spine_fraction=0.55,
        center_height_km=None,
        half_strength_distance_km=11_900.0,
    )


def check_dynamics(
    static_result: dict[str, object],
    h5_path: Path,
    output_root: Path,
    n_frames: int,
) -> dict[str, object]:
    """Check real-background alignment, frame zero, dynamics, and HDF5 output."""
    backgrounds = load_h5_background_sequence(
        h5_path,
        n_frames=n_frames,
        seed=1236,
        start_index=0,
        frame_step=1,
    )
    expected_bounds = (0, 0, 448, 448)
    if tuple(backgrounds["metadata"]["crop_xyxy_px"]) != expected_bounds:
        raise AssertionError(f"HDF5 crop changed: {backgrounds['metadata']['crop_xyxy_px']}")

    initial = static_result
    if not np.array_equal(initial["arrays"]["background"], backgrounds["frames"][0]):
        raise AssertionError("static preview and dynamics frame-zero background differ")
    dynamics_config = canonical_dynamics_config(n_frames)
    state = initialize_dynamics(
        initial,
        dynamics_config,
        background_frames=backgrounds["frames"],
    )
    frame_zero = current_dynamics_frame(state)
    if not np.array_equal(frame_zero["gong_image"], initial["arrays"]["degraded_intensity"]):
        raise AssertionError("dynamics frame zero differs from its static initial condition")
    if not np.array_equal(dynamics_background_at_frame(state, 0), backgrounds["frames"][0]):
        raise AssertionError("dynamics frame-zero background changed")

    export_config = make_export_config(
        thread_mask_dilation_px=5,
        video_lower_percentile=0.0,
        video_upper_percentile=100.0,
        velocity_opacity_floor=1.0e-6,
        compression="gzip",
        gzip_level=4,
    )
    saved = simulate_and_save_filament_dynamics(
        initial,
        dynamics_config,
        background_frames=backgrounds["frames"],
        simulations_root=output_root / "simulations",
        export_config=export_config,
        label="regression",
    )
    required_paths = (
        "video/raw_gong",
        "state/thread_displacements_km",
        "labels/thread_mask_native",
        "labels/coherent_velocity_xy_km_s",
        "labels/opacity_change_native",
        "dynamics/thread_prominence_heights_m",
        "dynamics/thread_gravity_m_s2",
        "dynamics/thread_gravity_cutoff_periods_s",
    )
    with h5py.File(saved["h5_path"], "r") as handle:
        if handle.attrs["simulation_dataset_schema_version"] != SIMULATION_DATASET_SCHEMA_VERSION:
            raise AssertionError("saved dynamics schema differs from the current writer")
        for field in (
            "requested_column_mass_g_cm2", "realized_column_mass_g_cm2",
            "column_mass_capacity_g_cm2", "column_mass_capacity_limited",
            "loaded_length_fraction",
        ):
            if not np.array_equal(
                handle[f"geometry/initial/thread_{field}"], [t[field] for t in initial["threads"]]
            ):
                raise AssertionError(f"HDF5 {field} differs from the realized loading")
        missing = [path for path in required_paths if path not in handle]
        if missing:
            raise AssertionError(f"saved dynamics data are missing {missing}")
        raw_gong = np.asarray(handle["video/raw_gong"])
        processed = np.asarray(handle["video/processed_uint8"])
        background_frames = np.asarray(handle["forward_model/background_frames"])
        line_absorption = np.asarray(handle["radiative/line_absorption_native"])
        observable_absorption = np.asarray(
            handle["radiative/observable_absorption_native"]
        )
        tau_highres = np.asarray(handle["radiative/tau_highres"])
        if raw_gong.shape != (n_frames, 448, 448):
            raise AssertionError("saved dynamics video has the wrong shape")
        if not np.array_equal(raw_gong[0], frame_zero["gong_image"]):
            raise AssertionError("saved frame zero changed during persistence")
        if not np.array_equal(tau_highres[0], initial["arrays"]["tau_map"]):
            raise AssertionError("saved frame-zero opacity changed during persistence")
        if not np.array_equal(background_frames, backgrounds["frames"]):
            raise AssertionError("saved HDF5 backgrounds differ from the selected source crop")

        bandpass_fraction = float(initial["config"]["gong_bandpass_line_fraction"])
        if not np.allclose(
            observable_absorption,
            bandpass_fraction * line_absorption,
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise AssertionError("saved passband absorption violates its transfer equation")
        source_intensity = float(initial["metadata"]["source_fraction_effective"]) * float(
            initial["metadata"]["source_level"]
        )
        reconstructed_raw = (
            background_frames * (1.0 - observable_absorption)
            + source_intensity * observable_absorption
        )
        if not np.allclose(raw_gong, reconstructed_raw, rtol=1.0e-6, atol=1.0e-7):
            raise AssertionError("saved raw GONG frames violate the transfer equation")

        video_dataset = handle["video/processed_uint8"]
        lower = float(video_dataset.attrs["input_vmin"])
        upper = float(video_dataset.attrs["input_vmax"])
        reconstructed_processed = np.rint(
            255.0 * np.clip((raw_gong - lower) / (upper - lower), 0.0, 1.0)
        ).astype(np.uint8)
        if not np.array_equal(processed, reconstructed_processed):
            raise AssertionError("processed video is not the documented fixed mapping of raw GONG")
        if not np.array_equal(
            np.asarray(handle["time/time_s"]),
            np.arange(n_frames, dtype=float) * dynamics_config["cadence_s"],
        ):
            raise AssertionError("saved frame times do not match the configured cadence")
        if np.any(handle["state/thread_displacements_km"][0]):
            raise AssertionError("frame-zero thread displacement is not zero")
        expected_heights_m = 1_000.0 * np.asarray(
            [thread["height_km"] for thread in initial["threads"]], dtype=float
        )
        if not np.array_equal(
            np.asarray(handle["dynamics/thread_prominence_heights_m"]), expected_heights_m
        ):
            raise AssertionError("persisted Luna heights differ from realized dip-bottom heights")
        if np.any(handle["labels/opacity_change_highres"][0]):
            raise AssertionError("frame-zero opacity-change label is not zero")
        if n_frames > 1:
            if not np.any(handle["state/brownian_thread_displacements_km"][1]):
                raise AssertionError("Brownian dynamics did not advance after frame zero")
            if np.array_equal(tau_highres[1], tau_highres[0]):
                raise AssertionError("thread motion did not alter the rendered opacity")

    manual_overrides = canonical_dynamics_config(2)
    manual_seed = int(manual_overrides.pop("seed"))
    manual_overrides.update(
        {
            "oscillation_mode": OSCILLATION_MODE_SHARED_PERIOD,
            "transverse_period_s": None,
            "oscillation_start_time_s": 0.0,
        }
    )
    manual_state = initialize_dynamics(
        initial,
        make_dynamics_config(seed=manual_seed, **manual_overrides),
        background_frames=np.broadcast_to(backgrounds["frames"][0], (2, 448, 448)).copy(),
    )
    expected_manual_periods = np.full(len(initial["threads"]), 3_600.0)
    if not np.array_equal(
        manual_state["thread_longitudinal_periods_s"], expected_manual_periods
    ):
        raise AssertionError("manual mode did not apply the shared longitudinal period")
    if not np.array_equal(
        manual_state["thread_transverse_periods_s"], expected_manual_periods
    ):
        raise AssertionError("manual mode did not apply the shared transverse period")
    manual_frame = step_dynamics(manual_state)
    if not np.isfinite(manual_frame["gong_image"]).all():
        raise AssertionError("manual-mode rendered intensity contains non-finite values")
    if not np.any(np.abs(manual_frame["coherent_thread_displacements_km"]) > 0.0):
        raise AssertionError("manual mode did not produce coherent displacement")
    return {
        "background_crop_xyxy_px": expected_bounds,
        "background_hash": backgrounds["metadata"]["frames_sha256"],
        "period_min_s": float(np.min(state["thread_longitudinal_periods_s"])),
        "period_median_s": float(np.median(state["thread_longitudinal_periods_s"])),
        "period_max_s": float(np.max(state["thread_longitudinal_periods_s"])),
        "manual_period_s": float(manual_state["thread_longitudinal_periods_s"][0]),
        "h5_schema": SIMULATION_DATASET_SCHEMA_VERSION,
    }


def main() -> None:
    """Parse paths and run the selected scientific checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-background", type=Path, default=DEFAULT_H5)
    parser.add_argument("--include-dynamics", action="store_true")
    parser.add_argument("--dynamics-frames", type=int, default=3)
    arguments = parser.parse_args()

    if arguments.dynamics_frames < 1:
        parser.error("--dynamics-frames must be positive")
    with tempfile.TemporaryDirectory(prefix="filament-regression-", dir="/tmp") as directory:
        output_root = Path(directory)
        static_result = canonical_static_result(arguments.h5_background)
        report = {
            "heinzel_opacity_table": check_heinzel_opacity_table(),
            "luna_height_model": check_luna_height_model(),
            "numerical_repairs": check_numerical_repairs(),
            "static": check_static_result(static_result, output_root),
        }
        if arguments.include_dynamics:
            report["dynamics"] = check_dynamics(
                static_result,
                arguments.h5_background,
                output_root,
                arguments.dynamics_frames,
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS — scientific regression completed")


if __name__ == "__main__":
    main()
