#!/usr/bin/env python3
"""Export height-dependent Luna-model plots and a numerical table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/filament-modelling-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from synthetic_filaments import (  # noqa: E402
    SOLAR_RADIUS_M,
    SOLAR_SURFACE_GRAVITY_M_S2,
    gravity_cutoff_period_s,
    luna_2022_inferred_curvature_radius_m,
    luna_2022_longitudinal_period_s,
    solar_gravity_m_s2,
)

DEFAULT_CURVATURE_RADII_MM = (25.0, 50.0, 100.0, 200.0)
DEFAULT_PERIODS_MIN = (30.0, 60.0, 90.0, 120.0, 150.0)


def _parse_arguments() -> argparse.Namespace:
    """Return validated output controls in display-boundary units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("scratch/luna_height_model"),
    )
    parser.add_argument(
        "--curvature-radii-mm",
        nargs="+",
        type=float,
        default=list(DEFAULT_CURVATURE_RADII_MM),
    )
    parser.add_argument(
        "--periods-min",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERIODS_MIN),
    )
    arguments = parser.parse_args()
    if not np.isfinite(arguments.curvature_radii_mm).all() or np.any(
        np.asarray(arguments.curvature_radii_mm) <= 0.0
    ):
        parser.error("--curvature-radii-mm must contain finite positive values")
    if not np.isfinite(arguments.periods_min).all() or np.any(
        np.asarray(arguments.periods_min) <= 0.0
    ):
        parser.error("--periods-min must contain finite positive values")
    return arguments


def _column_label(prefix: str, value: float, suffix: str) -> str:
    """Return a stable CSV column label for one user-facing value."""
    value_text = f"{value:g}".replace(".", "p")
    return f"{prefix}_{value_text}{suffix}"


def main() -> None:
    """Write the 1–100 Mm comparison table and publication-format figure."""
    arguments = _parse_arguments()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    heights_mm = np.arange(1.0, 101.0, 1.0)
    heights_m = heights_mm * 1.0e6
    gravity = np.asarray(solar_gravity_m_s2(heights_m), dtype=float)
    cutoff_min = np.asarray(gravity_cutoff_period_s(heights_m), dtype=float) / 60.0

    curvature_radii_mm = np.asarray(arguments.curvature_radii_mm, dtype=float)
    period_curves_min = {
        float(radius_mm): np.asarray(
            luna_2022_longitudinal_period_s(radius_mm * 1.0e6, heights_m), dtype=float
        )
        / 60.0
        for radius_mm in curvature_radii_mm
    }
    periods_min = np.asarray(arguments.periods_min, dtype=float)
    minimum_cutoff_min = float(np.min(cutoff_min))
    if np.any(periods_min >= minimum_cutoff_min):
        raise ValueError(
            "all inversion periods must remain below the gravity cut-off across 1–100 Mm; "
            f"minimum cut-off is {minimum_cutoff_min:.6f} min"
        )
    inferred_curves_mm = {
        float(period_min): np.asarray(
            luna_2022_inferred_curvature_radius_m(period_min * 60.0, heights_m),
            dtype=float,
        )
        / 1.0e6
        for period_min in periods_min
    }

    columns = [heights_mm, gravity, cutoff_min]
    names = ["height_Mm", "gravity_m_s2", "gravity_cutoff_period_min"]
    for radius_mm, curve in period_curves_min.items():
        columns.append(curve)
        names.append(_column_label("period_R", radius_mm, "Mm_min"))
    for period_min, curve in inferred_curves_mm.items():
        columns.append(curve)
        names.append(_column_label("inferred_R_P", period_min, "min_Mm"))
    table_path = output_directory / "luna_height_model.csv"
    np.savetxt(
        table_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.10g",
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    axes[0, 0].plot(heights_mm, gravity, color="#31688E", linewidth=2.0)
    axes[0, 0].set(
        title="Solar gravity",
        xlabel="Prominence-thread height [Mm]",
        ylabel=r"$g(h)$ [m s$^{-2}$]",
    )

    axes[0, 1].plot(heights_mm, cutoff_min, color="#B57614", linewidth=2.0)
    axes[0, 1].set(
        title="Gravity-only cut-off period",
        xlabel="Prominence-thread height [Mm]",
        ylabel=r"$P_{\rm cut,g}(h)$ [min]",
    )

    period_colors = plt.colormaps["viridis"](
        np.linspace(0.12, 0.88, len(period_curves_min))
    )
    for (radius_mm, curve), color in zip(
        period_curves_min.items(), period_colors, strict=True
    ):
        axes[1, 0].plot(
            heights_mm,
            curve,
            color=color,
            linewidth=1.8,
            label=fr"$R={radius_mm:g}$ Mm",
        )
    axes[1, 0].set(
        title="Height-dependent pendulum period",
        xlabel="Prominence-thread height [Mm]",
        ylabel=r"$P(R,h)$ [min]",
    )
    axes[1, 0].legend(loc="best", ncols=2, fontsize=8)

    inferred_colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(inferred_curves_mm))
    )
    for (period_min, curve), color in zip(
        inferred_curves_mm.items(), inferred_colors, strict=True
    ):
        axes[1, 1].plot(
            heights_mm,
            curve,
            color=color,
            linewidth=1.8,
            label=fr"$P={period_min:g}$ min",
        )
    axes[1, 1].set(
        title="Seismological curvature inference (log scale)",
        xlabel="Prominence-thread height [Mm]",
        ylabel=r"Inferred $R(P,h)$ [Mm]",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(loc="best", ncols=2, fontsize=8)

    for axis in axes.flat:
        axis.set_xlim(1.0, 100.0)
        axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.7)
    figure.suptitle("Luna corrected-pendulum model over 1–100 Mm")

    output_paths = {}
    for extension in ("png", "pdf", "svg"):
        figure_path = output_directory / f"luna_height_model.{extension}"
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        output_paths[extension] = str(figure_path)
    plt.close(figure)

    parameters = {
        "solar_radius_m": SOLAR_RADIUS_M,
        "surface_gravity_m_s2": SOLAR_SURFACE_GRAVITY_M_S2,
        "height_range_Mm": [1.0, 100.0],
        "height_step_Mm": 1.0,
        "curvature_radii_Mm": curvature_radii_mm.tolist(),
        "inversion_periods_min": periods_min.tolist(),
        "outputs": {"table": str(table_path), **output_paths},
    }
    parameters_path = output_directory / "model_parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
