"""Publication-ready quicklooks for one static forward-model result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_static_summary(result: dict[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    """Save a 300-DPI PNG and vector PDF summary for one static result."""
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    quicklooks = output / "quicklooks"
    quicklooks.mkdir(parents=True, exist_ok=True)
    arrays = result["arrays"]
    panels = (
        ("background", "Observed quiet background", "gray"),
        ("degraded_intensity", "Synthetic filament insertion", "gray"),
        ("tau_map", r"Internal line-center $\tau$", "viridis"),
        ("observable_soft_mask", "GONG-passband absorption", "viridis"),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 9.0),
        constrained_layout=True,
    )
    for axis, (key, title, colormap) in zip(axes.flat, panels, strict=True):
        image = np.asarray(arrays[key])
        artist = axis.imshow(image, origin="lower", cmap=colormap)
        axis.set_title(title)
        axis.set_axis_off()
        if colormap == "viridis":
            figure.colorbar(artist, ax=axis, fraction=0.045)
    png_path = quicklooks / "summary.png"
    pdf_path = quicklooks / "summary.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path
