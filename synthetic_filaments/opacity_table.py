"""Validated access to the calibrated Heinzel Table-1 height extension."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

HEINZEL_EXTENSION_FILENAME = "heinzel_calibrated_extension_1_to_100Mm.csv"
HEINZEL_EXTENSION_SHA256 = (
    "eb53e508e39774ceee93ebcc9ffc7faf83cd40c6c07631c1a88494c7d4d8d684"
)
HEINZEL_EXTENSION_PATH = Path(__file__).with_name("data") / HEINZEL_EXTENSION_FILENAME

_REQUIRED_COLUMNS = {
    "height_Mm",
    "temperature_K",
    "pressure_dyn_cm2",
    "i_extension",
    "f_extension_1e16_cm3",
}
_EXPECTED_HEIGHT_MM = np.arange(1.0, 101.0, dtype=float)
_EXPECTED_TEMPERATURE_K = np.asarray(
    [6_000.0, 8_000.0, 10_000.0, 12_000.0, 14_000.0]
)
_EXPECTED_PRESSURE_DYN_CM2 = np.asarray([0.01, 0.02, 0.05, 0.10, 0.20])


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return an array that cannot accidentally mutate the calibrated grid."""
    values.setflags(write=False)
    return values


def load_heinzel_opacity_table() -> dict[str, Any]:
    """Load the approved table, rechecking replaced assets before RAM reuse."""
    if not HEINZEL_EXTENSION_PATH.is_file():
        raise FileNotFoundError(
            f"required opacity table is missing: {HEINZEL_EXTENSION_PATH}"
        )

    path = HEINZEL_EXTENSION_PATH.resolve()
    stat = path.stat()
    return _load_checked_table(
        path, (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )


@lru_cache(maxsize=1)
def _load_checked_table(path: Path, signature: tuple[int, int, int]) -> dict[str, Any]:
    """Cache one checksum-verified asset identity; its arrays are immutable."""
    table_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(table_bytes).hexdigest()
    if actual_sha256 != HEINZEL_EXTENSION_SHA256:
        raise ValueError(
            "calibrated opacity table checksum changed; validate the replacement and "
            "update HEINZEL_EXTENSION_SHA256 explicitly"
        )

    raw = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if raw.ndim != 1:
        raise ValueError("calibrated opacity table must be a one-dimensional row table")
    columns = set(raw.dtype.names or ())
    missing_columns = sorted(_REQUIRED_COLUMNS - columns)
    if missing_columns:
        raise ValueError(f"calibrated opacity table is missing columns: {missing_columns}")

    height_mm = np.asarray(raw["height_Mm"], dtype=float)
    temperature_K = np.asarray(raw["temperature_K"], dtype=float)
    pressure_dyn_cm2 = np.asarray(raw["pressure_dyn_cm2"], dtype=float)
    ionization = np.asarray(raw["i_extension"], dtype=float)
    factor_1e16_cm3 = np.asarray(raw["f_extension_1e16_cm3"], dtype=float)
    values = np.column_stack(
        [height_mm, temperature_K, pressure_dyn_cm2, ionization, factor_1e16_cm3]
    )
    if not np.isfinite(values).all():
        raise ValueError("calibrated opacity table contains non-finite values")

    order = np.lexsort((pressure_dyn_cm2, temperature_K, height_mm))
    height_mm = height_mm[order]
    temperature_K = temperature_K[order]
    pressure_dyn_cm2 = pressure_dyn_cm2[order]
    ionization = ionization[order]
    factor_1e16_cm3 = factor_1e16_cm3[order]

    shape = (
        _EXPECTED_HEIGHT_MM.size,
        _EXPECTED_TEMPERATURE_K.size,
        _EXPECTED_PRESSURE_DYN_CM2.size,
    )
    expected_rows = int(np.prod(shape))
    if height_mm.size != expected_rows:
        raise ValueError(
            f"calibrated opacity table has {height_mm.size} rows; expected {expected_rows}"
        )
    expected_height, expected_temperature, expected_pressure = np.meshgrid(
        _EXPECTED_HEIGHT_MM,
        _EXPECTED_TEMPERATURE_K,
        _EXPECTED_PRESSURE_DYN_CM2,
        indexing="ij",
    )
    if not (
        np.array_equal(height_mm, expected_height.ravel())
        and np.array_equal(temperature_K, expected_temperature.ravel())
        and np.array_equal(pressure_dyn_cm2, expected_pressure.ravel())
    ):
        raise ValueError(
            "calibrated opacity table must contain the complete unique 1--100 Mm, "
            "five-temperature, five-pressure Cartesian grid"
        )
    if np.any((ionization <= 0.0) | (ionization > 1.0)):
        raise ValueError("calibrated ionization degrees must lie in (0, 1]")
    if np.any(factor_1e16_cm3 <= 0.0):
        raise ValueError("calibrated level-population factors must be positive")

    return {
        "height_km": _readonly(_EXPECTED_HEIGHT_MM.copy() * 1_000.0),
        "temperature_K": _readonly(_EXPECTED_TEMPERATURE_K.copy()),
        "pressure_dyn_cm2": _readonly(_EXPECTED_PRESSURE_DYN_CM2.copy()),
        "ionization": _readonly(ionization.reshape(shape)),
        "f_1e16_cm3": _readonly(factor_1e16_cm3.reshape(shape)),
        "row_count": expected_rows,
        "sha256": actual_sha256,
    }


def heinzel_opacity_table_metadata() -> dict[str, Any]:
    """Return JSON-safe provenance and supported-domain metadata."""
    table = load_heinzel_opacity_table()
    return {
        "filename": HEINZEL_EXTENSION_FILENAME,
        "sha256": table["sha256"],
        "row_count": table["row_count"],
        "height_domain_km": [float(table["height_km"][0]), float(table["height_km"][-1])],
        "temperature_domain_K": [
            float(table["temperature_K"][0]),
            float(table["temperature_K"][-1]),
        ],
        "pressure_domain_dyn_cm2": [
            float(table["pressure_dyn_cm2"][0]),
            float(table["pressure_dyn_cm2"][-1]),
        ],
        "model": "Heinzel-2015 anchors plus calibrated Promweaver PRD/cone height response",
        "published_anchor_heights_km": [10_000.0, 20_000.0, 30_000.0],
        "scientific_caution": "model-based extension; apply extra caution at 1--5 Mm",
    }
