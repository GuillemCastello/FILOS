"""Exercise GUI save boundaries with real preview data and short video workers."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from synthetic_filaments.background_preparation import write_background
from synthetic_filaments.experiment_config import create_experiment, list_experiments, load_experiment_config
from synthetic_filaments.experiment_runner import (
    _temporary_experiments_root,
    generate_experiment_preview,
    list_jobs,
    load_preview,
    preview_is_saved,
    save_experiment_preview,
    save_preview_plots,
    save_video_job,
    start_experiment_worker,
)
from synthetic_filaments.segmentation import read_mask_frame


class ExperimentPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="filos-save-test-")
        cls.root = Path(cls.temporary.name)
        cls.experiments = cls.root / "experiments"
        cls.config = load_experiment_config(check_inputs=False)
        background = cls.root / "background.h5"
        write_background(background, np.ones((3, 128, 128), dtype=np.float32),
                         cadence_s=20, native_pixel_km=2500, disk_mu=0.8, limb_direction_deg=45)
        cls.config["inputs"]["h5_background_path"] = str(background)
        cls.config["static"]["thread_count_cap"] = 1
        cls.config["dynamics"]["n_frames"] = 2
        cls.preview = generate_experiment_preview(cls.config)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_temporary_experiments_root(cls.experiments), ignore_errors=True)
        cls.temporary.cleanup()

    def experiment(self):
        return create_experiment("save test", experiments_root=self.experiments,
                                 source_config=self.config, save_config=False)

    def test_preview_save_reload_and_matching_configuration(self):
        experiment = self.experiment()
        self.assertEqual(list(experiment.iterdir()), [experiment / "runs"])
        self.assertIn(experiment, [item["directory"] for item in list_experiments(self.experiments)])
        plots = save_preview_plots(experiment, self.preview)
        self.assertTrue(list(plots.glob("*.png")))
        self.assertTrue(all(path.suffix == ".png" for path in plots.iterdir()))
        self.assertFalse(preview_is_saved(experiment, self.preview))
        save_experiment_preview(experiment, self.preview)
        self.assertTrue(preview_is_saved(experiment, self.preview))
        self.assertFalse(list(experiment.rglob("simulation.h5")))
        restored = load_preview(experiment)
        self.assertEqual(restored["user_configuration"], self.config)
        np.testing.assert_array_equal(restored["static_state"]["arrays"]["tau_map"],
                                      self.preview["static_state"]["arrays"]["tau_map"])
        (plots / "opacity_masks.png").unlink()
        self.assertFalse(preview_is_saved(experiment, self.preview))
        save_experiment_preview(experiment, self.preview)
        self.assertTrue(preview_is_saved(experiment, self.preview))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_video_explicit_save_and_close_during_worker(self):
        for save_during_worker in (False, True):
            with self.subTest(save_during_worker=save_during_worker):
                experiment = self.experiment()
                save_preview_plots(experiment, self.preview)
                with self.assertRaisesRegex(ValueError, "Save preview"):
                    start_experiment_worker(experiment, user_config=self.config, preview=self.preview,
                                            experiments_root=self.experiments, defer_save=True)
                save_experiment_preview(experiment, self.preview)
                saved_config = (experiment / "experiment.toml").read_bytes()
                video_config = deepcopy(self.config)
                video_config["video"]["fps"] = 12
                job = start_experiment_worker(experiment, user_config=video_config, preview=self.preview,
                                              experiments_root=self.experiments, defer_save=True)
                if save_during_worker:
                    save_video_job(job["job_directory"])
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    status = next(item for item in list_jobs(self.experiments)
                                  if item["job_id"] == job["job_id"])
                    if status["state"] == "failed":
                        self.fail(Path(job["job_directory"], "run.log").read_text())
                    if status["state"] == "completed" and (
                        not save_during_worker or status.get("data_saved")
                    ):
                        break
                    time.sleep(0.2)
                else:
                    self.fail(f"Worker timed out: {status}")
                output = Path(status["output_directory"])
                self.assertTrue((output / "gong.mp4").is_file())
                self.assertTrue((output / "opacity_masks/masks_frame_0000.png").is_file())
                self.assertEqual((experiment / "experiment.toml").read_bytes(), saved_config)
                if not save_during_worker:
                    self.assertTrue(all(path.suffix in {".png", ".mp4"}
                                        for path in output.rglob("*") if path.is_file()))
                    with patch("synthetic_filaments.experiment_runner.shutil.copytree",
                               side_effect=OSError("test disk failure")):
                        with self.assertRaisesRegex(OSError, "test disk failure"):
                            save_video_job(job["job_directory"])
                    self.assertFalse(json.loads(Path(job["job_directory"], "status.json").read_text())["data_saved"])
                    save_video_job(job["job_directory"])
                self.assertTrue((output / "simulation.h5").is_file())
                self.assertEqual(load_experiment_config(output / "experiment.toml")["video"]["fps"], 12)
                self.assertEqual(len(list((output / "opacity_masks").glob("*.npz"))), 2)
                masks, threshold, time_s = read_mask_frame(output / "opacity_masks", 1)
                self.assertEqual(threshold, self.config["static"]["mask_tau_threshold"])
                self.assertEqual(time_s, 20)
                self.assertEqual(masks["binary_native"].dtype, np.uint8)
                save_video_job(job["job_directory"])

    def test_gui_import_in_spawned_worker_does_not_call_streamlit(self):
        result = subprocess.run(
            [sys.executable, "-c", "import runpy; runpy.run_path('scripts/experiment_gui.py', run_name='__mp_main__')"],
            capture_output=True, text=True, check=True, env=dict(os.environ),
        )
        self.assertNotIn("ScriptRunContext", result.stderr)
        self.assertNotIn("No runtime found", result.stderr)

    def test_gui_create_and_close_save_last_generated_config(self):
        from streamlit.testing.v1 import AppTest

        with patch("synthetic_filaments.experiment_config.create_experiment",
                   side_effect=lambda name, **kwargs: create_experiment(
                       name, experiments_root=self.experiments, **kwargs)), patch(
                       "synthetic_filaments.experiment_config.list_experiments", return_value=[]):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "scripts/experiment_gui.py", default_timeout=30).run()
            app.text_input[0].set_value("created in GUI")
            next(button for button in app.button if button.label == "Create experiment").click().run()
            self.assertFalse(app.exception)
            experiment = Path(app.session_state["active_path"])
            self.assertTrue(experiment.is_dir())
            self.assertFalse((experiment / "experiment.toml").exists())
            app.session_state[f"preview:{experiment}"] = self.preview
            next(button for button in app.button if button.label == "Close").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(preview_is_saved(experiment, self.preview))
            self.assertEqual(load_experiment_config(experiment / "experiment.toml"), self.config)

    def test_gui_close_stays_open_if_saving_fails(self):
        from streamlit.testing.v1 import AppTest

        experiment = self.experiment()
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "scripts/experiment_gui.py", default_timeout=30)
        app.session_state["active_kind"] = "experiment"
        app.session_state["active_path"] = str(experiment)
        app.session_state[f"draft:{experiment}"] = deepcopy(self.config)
        app.session_state[f"preview:{experiment}"] = self.preview
        app.run()
        self.assertFalse(app.exception)
        with patch("synthetic_filaments.experiment_runner.save_experiment_preview",
                   side_effect=OSError("test disk failure")):
            next(button for button in app.button if button.label == "Close").click().run()
        self.assertEqual(app.session_state["active_kind"], "experiment")
        self.assertIn("test disk failure", app.error[0].value)

    def test_gui_can_reopen_an_empty_experiment(self):
        from streamlit.testing.v1 import AppTest

        experiment = self.experiment()
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "scripts/experiment_gui.py")
        app.session_state["active_kind"] = "experiment"
        app.session_state["active_path"] = str(experiment)
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(button.label == "Generate preview" for button in app.button))

    def test_gui_generate_then_save_preview(self):
        from streamlit.testing.v1 import AppTest

        experiment = self.experiment()
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "scripts/experiment_gui.py",
                                default_timeout=60)
        app.session_state["active_kind"] = "experiment"
        app.session_state["active_path"] = str(experiment)
        app.session_state[f"draft:{experiment}"] = deepcopy(self.config)
        app.run()
        app.number_input(key=f"editor:{experiment}:static:mask_tau_threshold").set_value(0.2)
        next(button for button in app.button if button.label == "Generate preview").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(list(experiment.rglob("*.png"))), 4)
        self.assertFalse(list(experiment.rglob("*.npz")))
        next(button for button in app.button if button.label == "Save preview").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        preview = app.session_state[f"preview:{experiment}"]
        self.assertTrue(preview_is_saved(experiment, preview))
        with np.load(next(experiment.rglob("opacity_masks.npz"))) as masks:
            self.assertEqual(float(masks["tau_threshold"]), 0.2)


if __name__ == "__main__":
    unittest.main()
