"""Portable backgrounds, timestamps, and preparation checks with small local fixtures."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from synthetic_filaments.background_preparation import (
    prepare_backgrounds,
    regular_windows,
    write_background,
)
from synthetic_filaments.dynamic_background import (
    background_info,
    load_h5_background_sequence,
    resolve_background_path,
)
from synthetic_filaments.experiment_config import (
    load_experiment_config,
    static_preview_fingerprint,
    validate_preview_config,
    validate_save_config,
)


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frames = np.arange(8 * 16 * 24, dtype=np.float32).reshape(8, 16, 24) + 1
        self.path = self.root / "source" / "quiet.h5"
        write_background(self.path, self.frames, cadence_s=20, native_pixel_km=700,
                         disk_mu=0.8, limb_direction_deg=45,
                         provenance={"source_file": "unavailable-original.h5"})

    def config(self):
        config = load_experiment_config()
        config["inputs"]["h5_background_path"] = str(self.path)
        config["dynamics"]["n_frames"] = 3
        return config

    def test_portable_pixels_geometry_and_timing(self):
        relocated = self.root / "relocated.h5"
        shutil.move(self.path, relocated)
        result = load_h5_background_sequence(relocated, n_frames=3, start_index=1, frame_step=2)
        np.testing.assert_array_equal(result["frames"], self.frames[[1, 3, 5]])
        self.assertEqual(result["native_shape"], (16, 24))
        self.assertEqual(result["native_pixel_km"], 700)
        self.assertEqual(result["disk_mu"], 0.8)
        self.assertEqual(result["limb_direction_deg"], 45)
        self.assertEqual(result["cadence_s"], 40)
        self.assertEqual(result["metadata"]["provenance"]["source_file"], "unavailable-original.h5")

    def test_runtime_does_not_import_detector_or_torch(self):
        code = ("import sys; from synthetic_filaments.dynamic_background import load_h5_background_sequence; "
                "load_h5_background_sequence(sys.argv[1], n_frames=1); "
                "assert 'synthetic_filaments.detector' not in sys.modules; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", code, str(self.path)], check=True)

    def test_library_selection_and_preview_identity(self):
        library = self.path.parent
        self.assertEqual(resolve_background_path(library, 123), self.path.resolve())
        config = self.config()
        config["inputs"]["h5_background_path"] = str(library)
        fingerprint = static_preview_fingerprint(config)
        explicit = deepcopy(config)
        explicit["inputs"]["h5_background_path"] = str(resolve_background_path(library, 1236))
        self.assertEqual(fingerprint, static_preview_fingerprint(explicit))
        explicit["dynamic_background"]["frame_step"] = 2
        self.assertEqual(fingerprint, static_preview_fingerprint(explicit))
        explicit["dynamic_background"]["start_index"] = 1
        self.assertNotEqual(fingerprint, static_preview_fingerprint(explicit))
        shutil.copy2(self.path, library / "second.h5")
        self.assertEqual(resolve_background_path(library, 42), resolve_background_path(library, 42))

    def test_configuration_uses_recorded_shape_and_cadence(self):
        config = self.config()
        config["dynamic_background"]["frame_step"] = 2
        resolved = validate_save_config(config)
        self.assertEqual(resolved["h5_dataset_shape"][1:], (16, 24))
        self.assertEqual(resolved["dynamics"]["cadence_s"], 40)
        config["dynamics"]["n_frames"] = 9
        validate_preview_config(config)
        with self.assertRaises((ValueError, IndexError)):
            validate_save_config(config)

    def test_invalid_sequences_and_requests(self):
        for arguments in ({"n_frames": 0}, {"n_frames": 9}, {"n_frames": 1, "frame_step": 0}):
            with self.assertRaises(ValueError):
                load_h5_background_sequence(self.path, **arguments)
        with h5py.File(self.path, "r+") as handle:
            handle["time_s"][3] += 1
        with self.assertRaisesRegex(ValueError, "time_s"):
            background_info(self.path)
        with h5py.File(self.path, "r+") as handle:
            handle["time_s"][3] -= 1
            handle["time_series"][2, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite positive"):
            load_h5_background_sequence(self.path, n_frames=3)
        with h5py.File(self.path, "r+") as handle:
            del handle.attrs["filos_background_version"]
        with self.assertRaisesRegex(ValueError, "not a prepared"):
            background_info(self.path)

    def test_time_gaps_are_not_silently_filled(self):
        self.assertEqual(regular_windows(np.array([0, 60, 120, 240, 300, 360]), 3, 60), [0, 3])
        self.assertEqual(regular_windows(np.array([0, 20, 80, 140]), 3, 60), [1])
        with self.assertRaises(ValueError):
            regular_windows(np.array([0, 60, 60]), 2, 60)

    def test_preparation_checks_later_frames_and_preserves_pixels(self):
        y, x = np.indices((64, 64))
        disk = (x - 31.5)**2 + (y - 31.5)**2 < 30**2
        frames = np.broadcast_to(disk.astype(np.float32) * 100, (3, 64, 64)).copy()
        raw = self.root / "raw.h5"
        with h5py.File(raw, "w") as handle:
            handle["time_series"] = frames
            handle["tdeltas"] = [0, 60, 120]
        results = prepare_backgrounds(raw, self.root / "good", n_frames=3, crop_shape=(16, 16), count=1)
        self.assertEqual(len(results), 1)
        info = background_info(results[0])
        x0, y0, x1, y1 = info["provenance"]["source_crop_xyxy_px"]
        np.testing.assert_array_equal(load_h5_background_sequence(results[0], n_frames=3)["frames"],
                                      frames[:, y0:y1, x0:x1])
        with h5py.File(raw, "r+") as handle:
            handle["time_series"][2] = 0
        self.assertEqual(prepare_backgrounds(raw, self.root / "bad", n_frames=3,
                                             crop_shape=(16, 16), count=1), [])

    def test_writer_never_overwrites_and_cleans_invalid_output(self):
        with self.assertRaises(FileExistsError):
            write_background(self.path, self.frames, cadence_s=20, native_pixel_km=700,
                             disk_mu=0.8, limb_direction_deg=45)
        invalid = self.root / "invalid.h5"
        with self.assertRaises(ValueError):
            write_background(invalid, self.frames, cadence_s=-1, native_pixel_km=700,
                             disk_mu=0.8, limb_direction_deg=45)
        self.assertFalse(invalid.exists())
        self.assertFalse(invalid.with_suffix(".h5.partial").exists())


if __name__ == "__main__":
    unittest.main()
