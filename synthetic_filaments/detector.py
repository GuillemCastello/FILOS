"""Functional bridge to the local DETR filament detector."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

DETECTOR_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "FilamentSegmentator" / "models" / "detector_v1"
)
GONG_IMAGE_MEAN = 0.4530
GONG_IMAGE_STD = 0.3382
SUNSPOT_LABEL = 3

_DETECTOR_STATE: dict[str, Any] | None = None


def detector_model_identity(model_path: str | Path = DETECTOR_MODEL_PATH) -> tuple:
    """Identify local model contents so replaced weights invalidate RAM reuse."""
    path = Path(model_path).resolve()
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            stat = candidate.stat()
            files.append((str(candidate.relative_to(path)), stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns))
    return str(path), tuple(files)


def _prepare_detector_image(
    image: np.ndarray,
    *,
    off_disk_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a finite normalized detector copy and preprocessing metadata."""
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"image must be two-dimensional; received shape {values.shape}")
    mask = np.asarray(off_disk_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError(
            "off_disk_mask must have the same shape as image; "
            f"received {mask.shape} and {values.shape}"
        )

    detector_values = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(detector_values)
    finite_off_disk = detector_values[mask & finite]
    used_zero_fallback = finite_off_disk.size == 0
    fill_value = 0.0 if used_zero_fallback else float(np.mean(finite_off_disk, dtype=np.float64))
    invalid = ~finite
    detector_values[invalid] = fill_value
    np.maximum(detector_values, 0.0, out=detector_values)
    maximum = float(np.max(detector_values))
    if maximum > 0.0:
        detector_values /= maximum

    metadata: dict[str, object] = {
        "nonfinite_pixel_count": int(np.count_nonzero(invalid)),
        "finite_off_disk_sample_count": int(finite_off_disk.size),
        "off_disk_fill_value": fill_value,
        "off_disk_zero_fallback": bool(used_zero_fallback),
        "normalization_maximum": maximum,
    }
    return detector_values, metadata


def _load_detector(model_path: str | Path = DETECTOR_MODEL_PATH) -> dict[str, Any]:
    """Load the local detector once and return its model, processor, and device."""
    global _DETECTOR_STATE
    path = Path(model_path).resolve()
    identity = detector_model_identity(path)
    if _DETECTOR_STATE is not None and _DETECTOR_STATE.get("identity") == identity:
        return _DETECTOR_STATE
    if not path.is_dir():
        raise RuntimeError(f"detector model directory is missing: {path}")
    try:
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError as error:
        raise RuntimeError(
            "detector extras are unavailable; run `uv sync --extra detector`"
        ) from error

    previous_offline_environment = os.environ.get("HF_HUB_OFFLINE")
    try:
        from huggingface_hub import constants as hub_constants
    except ImportError:
        hub_constants = None
    previous_offline_runtime = None if hub_constants is None else bool(hub_constants.HF_HUB_OFFLINE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if hub_constants is not None:
        hub_constants.HF_HUB_OFFLINE = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"for .*copying from a non-meta parameter in the checkpoint",
                category=UserWarning,
            )
            model = DetrForObjectDetection.from_pretrained(path, local_files_only=True)
    except Exception as error:
        raise RuntimeError(f"could not load local detector at {path}: {error}") from error
    finally:
        if previous_offline_environment is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline_environment
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = previous_offline_runtime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    processor = DetrImageProcessor(image_mean=GONG_IMAGE_MEAN, image_std=GONG_IMAGE_STD)
    _DETECTOR_STATE = {
        "identity": identity,
        "model_path": path,
        "model": model,
        "processor": processor,
        "device": device,
    }
    return _DETECTOR_STATE


def clear_detector_cache() -> None:
    """Drop the process-local detector references so their memory can be reclaimed."""
    global _DETECTOR_STATE
    _DETECTOR_STATE = None


def detect_filament_boxes(
    image: np.ndarray,
    *,
    off_disk_mask: np.ndarray,
    threshold: float = 0.5,
    model_path: str | Path = DETECTOR_MODEL_PATH,
    return_preprocessing_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Return non-sunspot filament boxes in ``(x0, y0, x1, y1)`` pixels.

    Nonfinite pixels are filled with the finite off-disk mean on a copy. If no
    finite off-disk sample exists, zero is used. The input is then clipped and
    normalized exactly once by its finite non-negative maximum, matching the
    detector training interface. Physical GPU selection remains controlled by
    ``CUDA_VISIBLE_DEVICES`` before this function is imported.
    """
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be finite and lie in (0, 1); received {threshold!r}")
    values, preprocessing_metadata = _prepare_detector_image(
        image,
        off_disk_mask=off_disk_mask,
    )
    try:
        import torch
        from torchvision.transforms.functional import to_tensor
    except ImportError as error:
        raise RuntimeError(
            "detector extras are unavailable; run `uv sync --extra detector`"
        ) from error
    state = _load_detector(model_path)
    tensor = to_tensor(values).to(torch.float32)
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    inputs = state["processor"]([tensor], do_rescale=False, return_tensors="pt")
    target_sizes = [(int(tensor.shape[1]), int(tensor.shape[2]))]
    with torch.no_grad():
        predictions = state["model"](
            pixel_values=inputs["pixel_values"].to(state["device"]),
            pixel_mask=inputs["pixel_mask"].to(state["device"]),
        )
    result = state["processor"].post_process_object_detection(
        predictions,
        target_sizes=target_sizes,
        threshold=float(threshold),
    )[0]
    keep = result["labels"] != SUNSPOT_LABEL
    boxes = result["boxes"][keep].detach().cpu().numpy()
    box_array = np.asarray(boxes, dtype=float)
    if return_preprocessing_metadata:
        return box_array, preprocessing_metadata
    return box_array
