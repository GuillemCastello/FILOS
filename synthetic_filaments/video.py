"""Create inspection videos from completed simulation HDF5 files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

VIDEO_HEADER_PX = 32
GONG_VIDEO_MINIMUM_FOV_FRACTION = 0.72
GONG_VIDEO_MASK_PADDING_FRACTION = 0.20
VELOCITY_VIDEO_HEADER_PX = 40
VELOCITY_VIDEO_RENDER_VERSION = 5
VELOCITY_ARROW_MAXIMUM_LENGTH_STRIDE_FRACTION = 2.0
VELOCITY_ARROW_MINIMUM_LENGTH_PX = 3.0
VELOCITY_ARROW_MINIMUM_MAXIMUM_LENGTH_PX = 18.0
VELOCITY_DISPLAY_FLOOR_FRACTION = 0.002


@lru_cache(maxsize=16)
def _video_font(size_px: int) -> object:
    """Return a bundled Pillow font, with a portable built-in fallback."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size_px)
    except OSError:
        return ImageFont.load_default()


def _gif_writer(output_path: Path, fps: int):
    """Return the Pillow writer for an explicitly requested GIF."""
    from matplotlib.animation import PillowWriter

    if fps < 1:
        raise ValueError(f"fps must be positive; received {fps!r}")
    if output_path.suffix.lower() != ".gif":
        raise ValueError("video output must end in .mp4 or .gif")
    return PillowWriter(fps=fps)


def _write_rgb_mp4(
    frames: Iterable[np.ndarray],
    *,
    frame_shape: tuple[int, int, int],
    output_path: Path,
    fps: int,
    title: str,
) -> Path:
    """Stream RGB arrays directly to FFmpeg without redrawing a plot."""
    if fps < 1:
        raise ValueError(f"fps must be positive; received {fps!r}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is unavailable; use a .gif output path")
    height, width, channels = frame_shape
    if channels != 3 or height < 2 or width < 2:
        raise ValueError(f"frame_shape must be (height, width, 3); received {frame_shape}")
    if height % 2 or width % 2:
        raise ValueError(
            f"MP4 frame dimensions must be even for yuv420p; received {(height, width)}"
        )

    staging = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={title}",
        str(staging),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("FFmpeg input pipe was not created")
        for index, frame in enumerate(frames):
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != frame_shape:
                raise ValueError(
                    f"video frame {index} has shape {array.shape}; expected {frame_shape}"
                )
            process.stdin.write(np.ascontiguousarray(array).tobytes())
        process.stdin.close()
        error_output = b"" if process.stderr is None else process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            message = error_output.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg exited with status {return_code}: {message}")
        os.replace(staging, output_path)
    except BaseException:
        process.kill()
        process.wait()
        if staging.exists():
            staging.unlink()
        raise
    return output_path


def _add_gong_header(frame: np.ndarray, time_s: float) -> np.ndarray:
    """Return one display-oriented RGB GONG frame with a compact time label."""
    from PIL import Image, ImageDraw

    display = np.flipud(np.asarray(frame, dtype=np.uint8))
    rgb = np.repeat(display[:, :, None], 3, axis=2)
    canvas = np.full((rgb.shape[0] + VIDEO_HEADER_PX, rgb.shape[1], 3), 255, dtype=np.uint8)
    canvas[VIDEO_HEADER_PX:] = rgb
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).text(
        (8, 9),
        f"Synthetic filament | t = {time_s / 60.0:.1f} min",
        fill=(0, 0, 0),
        font=_video_font(14),
    )
    return _pad_rgb_to_even(np.asarray(image))


def _expanded_bounds(
    low: int,
    high: int,
    limit: int,
    *,
    minimum_fraction: float,
    padding_fraction: float,
) -> tuple[int, int]:
    """Expand one inclusive feature interval without dropping feature pixels."""
    feature_size = high - low + 1
    padded_size = int(np.ceil(feature_size * (1.0 + 2.0 * padding_fraction)))
    minimum_size = int(np.ceil(limit * minimum_fraction))
    target_size = min(max(feature_size, padded_size, minimum_size), limit)

    center = 0.5 * (low + high + 1)
    start = int(round(center - 0.5 * target_size))
    start = min(max(start, 0), limit - target_size)
    stop = start + target_size
    if start > low or stop <= high:
        raise AssertionError("expanded video crop does not contain the filament bounds")
    return start, stop


def _pad_rgb_to_even(frame: np.ndarray) -> np.ndarray:
    """Pad the bottom/right display edges for YUV 4:2:0-compatible dimensions."""
    values = np.asarray(frame, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (height, width, 3); received {values.shape}")
    height, width = values.shape[:2]
    padded_height = height + height % 2
    padded_width = width + width % 2
    if (padded_height, padded_width) == (height, width):
        return values
    padded = np.full((padded_height, padded_width, 3), 255, dtype=np.uint8)
    padded[:height, :width] = values
    return padded


def _even_dimension(value: int) -> int:
    """Return ``value`` rounded up to the next encoder-compatible dimension."""
    return value + value % 2


def _gong_video_crop(handle: h5py.File) -> tuple[slice, slice]:
    """Return a modest fixed zoom containing the filament throughout the run."""
    frames = handle["video/processed_uint8"]
    native_height, native_width = (int(value) for value in frames.shape[1:])
    if "labels/thread_mask_native" not in handle:
        return slice(0, native_height), slice(0, native_width)

    masks = handle["labels/thread_mask_native"]
    union = np.zeros((native_height, native_width), dtype=bool)
    for start in range(0, masks.shape[0], 32):
        union |= np.any(np.asarray(masks[start : start + 32], dtype=bool), axis=0)
    rows, columns = np.where(union)
    if rows.size == 0:
        return slice(0, native_height), slice(0, native_width)

    row_start, row_stop = _expanded_bounds(
        int(rows.min()),
        int(rows.max()),
        native_height,
        minimum_fraction=GONG_VIDEO_MINIMUM_FOV_FRACTION,
        padding_fraction=GONG_VIDEO_MASK_PADDING_FRACTION,
    )
    column_start, column_stop = _expanded_bounds(
        int(columns.min()),
        int(columns.max()),
        native_width,
        minimum_fraction=GONG_VIDEO_MINIMUM_FOV_FRACTION,
        padding_fraction=GONG_VIDEO_MASK_PADDING_FRACTION,
    )
    return slice(row_start, row_stop), slice(column_start, column_stop)


@lru_cache(maxsize=16)
def _colormap_lut(name: str) -> np.ndarray:
    """Return one 256-entry RGB lookup table from a Matplotlib colormap."""
    from matplotlib import colormaps

    colors = colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3]
    return np.rint(255.0 * colors).astype(np.uint8)


def _colorize(values: np.ndarray, lower: float, upper: float, lut: np.ndarray) -> np.ndarray:
    """Map one scalar array to RGB through a fixed lookup table."""
    normalized = np.clip((np.asarray(values, dtype=float) - lower) / (upper - lower), 0.0, 1.0)
    indices = np.rint(255.0 * normalized).astype(np.uint8)
    return lut[indices]


def _velocity_arrow_samples(
    velocity_xy: np.ndarray,
    opacity_weight: np.ndarray,
    *,
    velocity_limit: float,
    stride_px: int,
) -> np.ndarray:
    """Select one representative unit vector and speed per occupied cell."""
    height, width = velocity_xy.shape[1:]
    speed = np.hypot(velocity_xy[0], velocity_xy[1])
    minimum_speed = VELOCITY_DISPLAY_FLOOR_FRACTION * velocity_limit
    samples: list[tuple[float, float, float, float, float]] = []
    for row_start in range(0, height, stride_px):
        row_stop = min(row_start + stride_px, height)
        for column_start in range(0, width, stride_px):
            column_stop = min(column_start + stride_px, width)
            cell_weight = opacity_weight[row_start:row_stop, column_start:column_stop]
            cell_speed = speed[row_start:row_stop, column_start:column_stop]
            score = cell_weight * cell_speed
            if not np.any(score > 0.0):
                continue
            local_row, local_column = np.unravel_index(int(np.argmax(score)), score.shape)
            row = row_start + int(local_row)
            column = column_start + int(local_column)
            velocity_x = float(velocity_xy[0, row, column])
            velocity_y = float(velocity_xy[1, row, column])
            local_speed = float(speed[row, column])
            if local_speed <= minimum_speed:
                continue
            unit_x = velocity_x / local_speed
            unit_y = velocity_y / local_speed
            samples.append((float(column), float(row), unit_x, unit_y, local_speed))
    if not samples:
        return np.empty((0, 5), dtype=float)
    return np.asarray(samples, dtype=float)


def _velocity_arrow_lengths_px(
    speed_km_s: np.ndarray | float,
    *,
    velocity_limit: float,
    stride_px: int,
) -> np.ndarray:
    """Map speed linearly to visible arrow length on the video canvas."""
    maximum_length = max(
        VELOCITY_ARROW_MAXIMUM_LENGTH_STRIDE_FRACTION * stride_px,
        VELOCITY_ARROW_MINIMUM_MAXIMUM_LENGTH_PX,
    )
    normalized_speed = np.clip(np.asarray(speed_km_s, dtype=float) / velocity_limit, 0.0, 1.0)
    return (
        VELOCITY_ARROW_MINIMUM_LENGTH_PX
        + (maximum_length - VELOCITY_ARROW_MINIMUM_LENGTH_PX) * normalized_speed
    )


def _add_velocity_arrows(
    image: object,
    velocity_xy: np.ndarray,
    opacity_weight: np.ndarray,
    *,
    velocity_limit: float,
    stride_px: int,
) -> None:
    """Draw thin black arrows whose lengths encode local coherent speed."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    height = velocity_xy.shape[1]
    samples = _velocity_arrow_samples(
        velocity_xy,
        opacity_weight,
        velocity_limit=velocity_limit,
        stride_px=stride_px,
    )
    arrow_lengths = _velocity_arrow_lengths_px(
        samples[:, 4],
        velocity_limit=velocity_limit,
        stride_px=stride_px,
    )
    for (column, row, unit_x, unit_y, _speed), arrow_length in zip(
        samples,
        arrow_lengths,
        strict=True,
    ):
        head_length = max(0.28 * arrow_length, 2.0)
        center_x = column
        center_y = float(VELOCITY_VIDEO_HEADER_PX + height - 1 - row)
        unit_y = -unit_y
        start_x = center_x - 0.5 * arrow_length * unit_x
        start_y = center_y - 0.5 * arrow_length * unit_y
        end_x = center_x + 0.5 * arrow_length * unit_x
        end_y = center_y + 0.5 * arrow_length * unit_y
        angle = float(np.arctan2(end_y - start_y, end_x - start_x))
        left = (
            end_x - head_length * np.cos(angle - np.pi / 6.0),
            end_y - head_length * np.sin(angle - np.pi / 6.0),
        )
        right = (
            end_x - head_length * np.cos(angle + np.pi / 6.0),
            end_y - head_length * np.sin(angle + np.pi / 6.0),
        )
        segments = (
            (start_x, start_y, end_x, end_y),
            (*left, end_x, end_y),
            (*right, end_x, end_y),
        )
        for segment in segments:
            draw.line(segment, fill=(0, 0, 0), width=1)


def _velocity_rgb_frame(
    velocity_xy: np.ndarray,
    opacity_weight: np.ndarray,
    time_s: float,
    *,
    velocity_limit: float,
    quiver_stride_px: int,
    speed_lut: np.ndarray,
) -> np.ndarray:
    """Build one speed field with high-contrast coherent-direction arrows."""
    from PIL import Image, ImageDraw

    values = np.asarray(velocity_xy, dtype=float)
    weight = np.asarray(opacity_weight, dtype=float)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError(f"velocity_xy must have shape (2, y, x); received {values.shape}")
    if weight.shape != values.shape[1:]:
        raise ValueError("velocity and opacity-weight frames must share one spatial shape")
    speed = np.hypot(values[0], values[1])
    height, width = values.shape[1:]
    speed_rgb = _colorize(np.flipud(speed), 0.0, velocity_limit, speed_lut)
    canvas = np.full((height + VELOCITY_VIDEO_HEADER_PX, width, 3), 255, dtype=np.uint8)
    canvas[VELOCITY_VIDEO_HEADER_PX:] = speed_rgb

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = _video_font(11)
    draw.text(
        (8, 3),
        f"Speed: 0–{velocity_limit:.2f} km/s",
        fill=(0, 0, 0),
        font=font,
    )
    time_label = f"t={time_s / 60.0:.1f} min"
    time_width = draw.textbbox((0, 0), time_label, font=font)[2]
    draw.text(
        (width - time_width - 8, 3),
        time_label,
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((8, 21), "Arrows: direction + speed", fill=(0, 0, 0), font=font)
    _add_velocity_arrows(
        image,
        values,
        weight,
        velocity_limit=velocity_limit,
        stride_px=quiver_stride_px,
    )
    return _pad_rgb_to_even(np.asarray(image))


def save_gong_video(
    h5_path: str | Path,
    output_path: str | Path,
    *,
    fps: int = 30,
    dpi: int = 140,
) -> Path:
    """Render the fixed-contrast GONG sequence stored in one simulation file."""
    source = Path(h5_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4":
        with h5py.File(source, "r") as handle:
            frames = handle["video/processed_uint8"]
            times = handle["time/time_s"]
            row_slice, column_slice = _gong_video_crop(handle)
            crop_height = row_slice.stop - row_slice.start
            crop_width = column_slice.stop - column_slice.start
            rgb_frames = (
                _add_gong_header(
                    frames[index, row_slice, column_slice],
                    float(times[index]),
                )
                for index in range(frames.shape[0])
            )
            return _write_rgb_mp4(
                rgb_frames,
                frame_shape=(
                    _even_dimension(crop_height + VIDEO_HEADER_PX),
                    _even_dimension(crop_width),
                    3,
                ),
                output_path=output,
                fps=fps,
                title="Synthetic filament dynamics",
            )

    import matplotlib.pyplot as plt

    writer = _gif_writer(output, fps)
    with h5py.File(source, "r") as handle:
        frames = handle["video/processed_uint8"]
        times = handle["time/time_s"]
        row_slice, column_slice = _gong_video_crop(handle)
        figure, axis = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        artist = axis.imshow(
            frames[0, row_slice, column_slice],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        title = axis.set_title("")
        axis.set_axis_off()
        with writer.saving(figure, str(output), dpi=dpi):
            for index in range(frames.shape[0]):
                artist.set_data(frames[index, row_slice, column_slice])
                title.set_text(f"Synthetic filament — t = {float(times[index]) / 60.0:.1f} min")
                writer.grab_frame()
        plt.close(figure)
    return output


def save_velocity_video(
    h5_path: str | Path,
    output_path: str | Path,
    *,
    fps: int = 30,
    dpi: int = 140,
    velocity_limit_km_s: float | None = None,
    quiver_stride_px: int = 8,
) -> Path:
    """Render coherent speed color and speed-scaled direction arrows."""
    source = Path(h5_path).resolve()
    output = Path(output_path).resolve()
    if quiver_stride_px < 1:
        raise ValueError("quiver_stride_px must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        velocity = handle["labels/coherent_velocity_xy_km_s"]
        velocity_weight = handle["labels/velocity_opacity_weight_native"]
        times = handle["time/time_s"]
        if velocity_limit_km_s is None:
            statistics = json.loads(handle.attrs["statistics_json"])
            velocity_limit = float(statistics["maximum_coherent_velocity_km_s"])
        else:
            velocity_limit = float(velocity_limit_km_s)
        if not np.isfinite(velocity_limit) or velocity_limit <= 0.0:
            velocity_limit = 1.0

        row_slice, column_slice = _gong_video_crop(handle)
        crop_height = row_slice.stop - row_slice.start
        crop_width = column_slice.stop - column_slice.start

        if output.suffix.lower() == ".mp4":
            speed_lut = _colormap_lut("viridis")
            rgb_frames = (
                _velocity_rgb_frame(
                    velocity[index, :, row_slice, column_slice],
                    velocity_weight[index, row_slice, column_slice],
                    float(times[index]),
                    velocity_limit=velocity_limit,
                    quiver_stride_px=quiver_stride_px,
                    speed_lut=speed_lut,
                )
                for index in range(velocity.shape[0])
            )
            return _write_rgb_mp4(
                rgb_frames,
                frame_shape=(
                    _even_dimension(crop_height + VELOCITY_VIDEO_HEADER_PX),
                    _even_dimension(crop_width),
                    3,
                ),
                output_path=output,
                fps=fps,
                title="Coherent filament velocity",
            )

        import matplotlib.pyplot as plt

        writer = _gif_writer(output, fps)

        first = np.asarray(velocity[0, :, row_slice, column_slice], dtype=float)
        first_speed = np.hypot(first[0], first[1])
        aspect = crop_width / crop_height
        figure, axis = plt.subplots(
            figsize=(max(5.2 * aspect, 4.5), 5.2),
            constrained_layout=True,
        )
        speed_artist = axis.imshow(
            first_speed,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=velocity_limit,
        )
        axis.set_axis_off()
        figure.colorbar(speed_artist, ax=axis, label=r"Speed [km s$^{-1}$]")
        frame_title = axis.set_title("")
        quiver = None

        with writer.saving(figure, str(output), dpi=dpi):
            for index in range(velocity.shape[0]):
                values = np.asarray(velocity[index, :, row_slice, column_slice], dtype=float)
                weight = np.asarray(velocity_weight[index, row_slice, column_slice], dtype=float)
                speed = np.hypot(values[0], values[1])
                speed_artist.set_data(speed)
                if quiver is not None:
                    quiver.remove()
                samples = _velocity_arrow_samples(
                    values,
                    weight,
                    velocity_limit=velocity_limit,
                    stride_px=quiver_stride_px,
                )
                arrow_lengths = _velocity_arrow_lengths_px(
                    samples[:, 4],
                    velocity_limit=velocity_limit,
                    stride_px=quiver_stride_px,
                )
                quiver = axis.quiver(
                    samples[:, 0],
                    samples[:, 1],
                    samples[:, 2] * arrow_lengths,
                    samples[:, 3] * arrow_lengths,
                    color="black",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.0025,
                )
                frame_title.set_text(
                    "Coherent velocity — color and arrow length: speed — "
                    f"t = {float(times[index]) / 60.0:.1f} min"
                )
                writer.grab_frame()
        plt.close(figure)
    return output
