"""Focused checks for the effective H-alpha source function."""

import math
import unittest

import numpy as np

from synthetic_filaments.config import SOLAR_RADIUS_KM
from synthetic_filaments.generator import _effective_source_fraction


class SourceFunctionTests(unittest.TestCase):
    def test_published_anchors_are_exact(self):
        for height_km, source in ((10_000.0, 3.22), (20_000.0, 2.98), (30_000.0, 2.80)):
            with self.subTest(height_km=height_km):
                self.assertEqual(
                    _effective_source_fraction([{"height_km": height_km}], 0.0),
                    source / 6.93,
                )

    def test_interpolation_is_unchanged(self):
        for height_km in (15_000.0, 25_000.0):
            with self.subTest(height_km=height_km):
                expected = np.interp(
                    height_km, [10_000.0, 20_000.0, 30_000.0],
                    np.asarray([3.22, 2.98, 2.80]) / 6.93,
                )
                self.assertEqual(
                    _effective_source_fraction([{"height_km": height_km}], 0.0), expected,
                )

    def test_dilution_extension_below_and_above_anchors(self):
        cases = ((5_000.0, 10_000.0, 3.22), (50_000.0, 30_000.0, 2.80))
        for height_km, anchor_km, source in cases:
            with self.subTest(height_km=height_km):
                dilution = 0.5 * (
                    1.0 - math.sqrt(1.0 - SOLAR_RADIUS_KM**2 / (SOLAR_RADIUS_KM + height_km)**2)
                )
                anchor_dilution = 0.5 * (
                    1.0 - math.sqrt(1.0 - SOLAR_RADIUS_KM**2 / (SOLAR_RADIUS_KM + anchor_km)**2)
                )
                actual = _effective_source_fraction([{"height_km": height_km}], 0.0)
                self.assertAlmostEqual(actual, source / 6.93 * dilution / anchor_dilution, places=14)
                if height_km < anchor_km:
                    self.assertGreater(actual, source / 6.93)
                else:
                    self.assertLess(actual, source / 6.93)

    def test_continuity_at_anchor_boundaries(self):
        for height_km, source in ((10_000.0, 3.22), (30_000.0, 2.80)):
            for offset_km in (-0.001, 0.001):
                with self.subTest(height_km=height_km, offset_km=offset_km):
                    self.assertAlmostEqual(
                        _effective_source_fraction([{"height_km": height_km + offset_km}], 0.0),
                        source / 6.93, delta=1.0e-8,
                    )

    def test_one_source_fraction_uses_mean_dip_bottom_height(self):
        for heights in ((5_000.0, 45_000.0), (40_000.0, 60_000.0)):
            with self.subTest(heights=heights):
                self.assertEqual(
                    _effective_source_fraction([{"height_km": h} for h in heights], 0.0),
                    _effective_source_fraction([{"height_km": np.mean(heights)}], 0.0),
                )

    def test_no_threads_preserves_fallback(self):
        self.assertEqual(_effective_source_fraction([], 0.42), 0.42)


if __name__ == "__main__":
    unittest.main()
