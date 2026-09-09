"""Shared project paths.

Commands accept explicit path overrides. These defaults match the checked-out
project layout and keep raw observations outside the Python package.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS_DIR = PROJECT_ROOT / "FITS_files"
DEFAULT_BACKGROUNDS_DIR = PROJECT_ROOT / "backgrounds"
