"""Check opacity masks, per-frame exports, and preservation of existing results."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from synthetic_filaments.segmentation import (
    export_video_masks,
    export_video_masks_npz,
    opacity_masks,
    preview_masks,
    read_mask_frame,
    save_mask_plot,
)


class SegmentationTests(unittest.TestCase):
    def test_threshold_and_absorption_without_padding(self):
        tau = np.array([[0, 0.1, 0.2], [0, np.log(2), 100]])
        masks = opacity_masks(tau)
        np.testing.assert_array_equal(masks["binary"], [[0, 0, 1], [0, 1, 1]])
        self.assertAlmostEqual(float(masks["absorption"][1, 1]), 0.5)
        self.assertEqual(masks["absorption"][0, 0], 0)
        self.assertEqual(masks["absorption"][1, 2], 1)
        for threshold in (0, -1, np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                opacity_masks(tau, threshold)

    def test_preview_uses_native_absorption_not_resized_highres_mask(self):
        arrays = {"tau_map": np.ones((4, 6)), "soft_mask": np.zeros((2, 3))}
        arrays["soft_mask"][0, 1] = 0.5
        masks = preview_masks(arrays)
        self.assertEqual(masks["binary_highres"].sum(), 24)
        np.testing.assert_array_equal(masks["binary_native"], [[0, 1, 0], [0, 0, 0]])
        np.testing.assert_allclose(masks["absorption_native"], arrays["soft_mask"])

    def test_video_export_all_frames_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "simulation.h5"
            destination = root / "opacity_masks" / "masks.h5"
            highres = np.zeros((3, 4, 6))
            native = np.zeros((3, 2, 3))
            for index in range(3):
                highres[index, 1, index] = 0.3
                native[index, 0, index] = 0.4
            with h5py.File(source, "w") as handle:
                handle["radiative/tau_highres"] = highres
                handle["radiative/tau_native"] = native
                handle["time/time_s"] = [0, 60, 120]
                handle["labels/thread_mask_native"] = np.ones_like(native, dtype=np.uint8)
            original = source.read_bytes()
            progress = []
            export_video_masks(source, destination, progress=lambda *args: progress.append(args))
            self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
            self.assertEqual(original, source.read_bytes())
            with h5py.File(destination, "r") as handle:
                self.assertEqual(handle["binary_highres"].dtype, np.dtype("uint8"))
                self.assertEqual(handle["absorption_native"].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(handle["time_s"][:], [0, 60, 120])
                np.testing.assert_array_equal(handle["binary_highres"][:], highres > 0.1)
                np.testing.assert_array_equal(handle["binary_native"][:], native > 0.1)
            masks, threshold, time_s = read_mask_frame(destination, 2)
            self.assertEqual(time_s, 120)
            self.assertEqual(threshold, 0.1)
            self.assertEqual(masks["binary_native"][0, 2], 1)
            plot = save_mask_plot(destination, 2)
            self.assertTrue(plot.read_bytes().startswith(b"\x89PNG"))

            npz_directory = root / "npz_masks"
            export_video_masks_npz(source, npz_directory, threshold=0.2)
            self.assertEqual(len(list(npz_directory.glob("*.npz"))), 3)
            for index in range(3):
                masks, threshold, time_s = read_mask_frame(npz_directory, index)
                self.assertEqual(threshold, 0.2)
                self.assertEqual(time_s, index * 60)
                np.testing.assert_array_equal(masks["binary_highres"], highres[index] > 0.2)
                np.testing.assert_array_equal(masks["binary_native"], native[index] > 0.2)
                np.testing.assert_allclose(masks["absorption_native"], -np.expm1(-native[index]), rtol=1e-6)
            self.assertTrue(save_mask_plot(npz_directory, 2).is_file())
            self.assertEqual(original, source.read_bytes())

            previous_output = destination.read_bytes()
            with h5py.File(source, "r+") as handle:
                handle["radiative/tau_highres"][1, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                export_video_masks(source, destination)
            self.assertEqual(destination.read_bytes(), previous_output)
            self.assertEqual(list(destination.parent.glob("*.h5")), [destination])
            with self.assertRaises(ValueError):
                export_video_masks(source, source)


if __name__ == "__main__":
    unittest.main()
