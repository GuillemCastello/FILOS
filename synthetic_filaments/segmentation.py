"""Detailed masks from optical depth, without geometry padding or dilation."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import h5py
import numpy as np
from matplotlib.figure import Figure


def opacity_masks(tau: np.ndarray, threshold: float = 0.1) -> dict[str, np.ndarray]:
    """Return binary foreground and continuous line-center absorption for one frame."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The optical-depth threshold must be finite and greater than zero.")
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 2 or not np.all(np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError("Optical depth must be a finite, non-negative two-dimensional array.")
    return {
        "binary": (tau > threshold).astype(np.uint8),
        "absorption": (-np.expm1(-tau)).astype(np.float32),
    }


def preview_masks(arrays: dict, threshold: float = 0.1) -> dict[str, np.ndarray]:
    """Use the preview's high-resolution tau and instrument-degraded absorption."""
    tau_native = -np.log(np.maximum(1.0 - arrays["soft_mask"], 1.0e-12))
    result = {}
    for grid, tau in (("highres", arrays["tau_map"]), ("native", tau_native)):
        for name, mask in opacity_masks(tau, threshold).items():
            result[f"{name}_{grid}"] = mask
    return result


def mask_figure(masks: dict[str, np.ndarray], threshold: float, title: str = "") -> Figure:
    """Plot both grids with fixed absorption limits and no display smoothing."""
    figure = Figure(figsize=(10, 9), layout="constrained")
    axes = figure.subplots(2, 2)
    for row, grid in enumerate(("highres", "native")):
        label = "High resolution (before PSF)" if grid == "highres" else "Native (after PSF)"
        absorption = axes[row, 0].imshow(
            masks[f"absorption_{grid}"], origin="lower", cmap="viridis",
            vmin=0, vmax=1, interpolation="nearest",
        )
        axes[row, 0].set_title(f"{label}: absorption")
        figure.colorbar(absorption, ax=axes[row, 0], label="Line-center absorption fraction")
        axes[row, 1].imshow(
            masks[f"binary_{grid}"], origin="lower", cmap="gray",
            vmin=0, vmax=1, interpolation="nearest",
        )
        axes[row, 1].set_title(f"{label}: tau > {threshold:g}")
        for axis in axes[row]:
            axis.set_xlabel("Column (pixels)")
            axis.set_ylabel("Row (pixels)")
    if title:
        figure.suptitle(title)
    return figure


def export_video_masks(
    source_path: Path,
    output_path: Path,
    threshold: float = 0.1,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Write both mask grids for every simulation frame, keeping one frame in memory.

    The simulation is opened read-only. A temporary output is replaced only after
    all frames have been written successfully.
    """
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("The mask output must be different from the simulation file.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The optical-depth threshold must be finite and greater than zero.")
    with h5py.File(source_path, "r") as source:
        highres = source["radiative/tau_highres"]
        native = source["radiative/tau_native"]
        times = source["time/time_s"]
        if highres.ndim != 3 or native.ndim != 3:
            raise ValueError("Opacity grids must have shape (time, y, x).")
        n_frames = highres.shape[0]
        if (
            n_frames == 0 or native.shape[0] != n_frames or times.shape != (n_frames,)
        ):
            raise ValueError("Opacity grids and timestamps must contain the same nonzero frames.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output_path.parent, suffix=".h5", delete=False) as temp:
            temporary_path = Path(temp.name)
        try:
            with h5py.File(temporary_path, "w") as output:
                output.attrs["source_simulation"] = str(source_path)
                output.attrs["tau_threshold"] = float(threshold)
                output.attrs["binary_definition"] = "tau > tau_threshold; no dilation"
                output.attrs["absorption_definition"] = "1 - exp(-tau); line center"
                output.attrs["axis_order"] = "time,y,x"
                output.create_dataset("time_s", data=times[:])
                for grid, tau in (("highres", highres), ("native", native)):
                    for name, dtype in (("binary", np.uint8), ("absorption", np.float32)):
                        output.create_dataset(
                            f"{name}_{grid}", shape=tau.shape, dtype=dtype,
                            chunks=(1, *tau.shape[1:]), compression="lzf",
                        )
                for index in range(n_frames):
                    for grid, tau in (("highres", highres), ("native", native)):
                        for name, mask in opacity_masks(tau[index], threshold).items():
                            output[f"{name}_{grid}"][index] = mask
                    if progress is not None:
                        progress(index + 1, n_frames)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return output_path


def read_mask_frame(path: Path, index: int = 0) -> tuple[dict, float, float]:
    """Read a single exported frame, its threshold, and simulation time in seconds."""
    path = Path(path)
    if path.is_dir():
        with np.load(path / f"masks_frame_{index:04d}.npz", allow_pickle=False) as handle:
            masks = {
                f"{name}_{grid}": handle[f"{name}_{grid}"]
                for grid in ("highres", "native") for name in ("binary", "absorption")
            }
            return masks, float(handle["tau_threshold"]), float(handle["time_s"])
    with h5py.File(path, "r") as handle:
        masks = {
            f"{name}_{grid}": handle[f"{name}_{grid}"][index]
            for grid in ("highres", "native") for name in ("binary", "absorption")
        }
        return masks, float(handle.attrs["tau_threshold"]), float(handle["time_s"][index])


def save_mask_plot(path: Path, index: int = 0) -> Path:
    """Save a selected frame beside masks.h5, for command-line inspection."""
    masks, threshold, time_s = read_mask_frame(path, index)
    figure = mask_figure(masks, threshold, f"Frame {index} · t = {time_s:g} s")
    directory = Path(path) if Path(path).is_dir() else Path(path).parent
    plot_path = directory / f"masks_frame_{index:04d}.png"
    figure.savefig(plot_path, dpi=150)
    figure.clear()
    return plot_path


def export_video_masks_npz(
    source_path: Path,
    output_directory: Path,
    threshold: float = 0.1,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Export one compressed NPZ per frame, without holding the video in RAM."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        highres = source["radiative/tau_highres"]
        native = source["radiative/tau_native"]
        times = source["time/time_s"]
        count = len(times)
        if count == 0 or highres.ndim != 3 or native.ndim != 3 or (
            highres.shape[0] != count or native.shape[0] != count
        ):
            raise ValueError("Opacity grids and timestamps must contain the same nonzero frames.")
        for index in range(count):
            masks = {}
            for grid, tau in (("highres", highres), ("native", native)):
                for name, mask in opacity_masks(tau[index], threshold).items():
                    masks[f"{name}_{grid}"] = mask
            target = output_directory / f"masks_frame_{index:04d}.npz"
            with NamedTemporaryFile(dir=output_directory, suffix=".npz", delete=False) as temp:
                temporary = Path(temp.name)
            try:
                np.savez_compressed(
                    temporary, **masks, tau_threshold=threshold, time_s=times[index],
                )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            if progress is not None:
                progress(index + 1, count)
    return output_directory
