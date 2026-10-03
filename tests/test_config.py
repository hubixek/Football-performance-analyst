import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401  (the repository on the path)
import config


class Home(unittest.TestCase):
    def test_default_is_football_in_the_home_folder(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOOTBALL_HOME", None)
            self.assertEqual(config.football_home(), Path.home() / "football")

    def test_the_environment_moves_everything(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"FOOTBALL_HOME": tmp}):
            try:
                importlib.reload(config)
                self.assertEqual(config.HOME, Path(tmp))
                self.assertEqual(config.CALIB_DIR, Path(tmp) / "calib")
                self.assertEqual(config.ANALYSIS_DIR, Path(tmp) / "analysis")
                self.assertEqual(config.VIDEOS_DIR, Path(tmp) / "videos")
                self.assertEqual(config.model_path("v8_dual"), Path(tmp) / "runs" / "detect" / "runs" / "v8_dual" / "weights" / "best.pt")
            finally:
                os.environ.pop("FOOTBALL_HOME", None)
                importlib.reload(config)


class Reference(unittest.TestCase):
    def test_the_shipped_reference_has_both_lenses(self):
        for half in config.HALVES:
            self.assertTrue((config.REFERENCE_DIR / f"mecz2_{half}.json").exists())

    def test_it_is_copied_when_missing_and_left_alone_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = Path(tmp) / "calib"
            copied = config.ensure_reference_calibration("mecz2", calib)
            self.assertEqual(sorted(p.name for p in copied), ["mecz2_bottom.json", "mecz2_top.json"])
            (calib / "mecz2_top.json").write_text("own")                       # a calibration the user made by hand is never overwritten
            self.assertEqual(config.ensure_reference_calibration("mecz2", calib), [])
            self.assertEqual((calib / "mecz2_top.json").read_text(), "own")

    def test_a_missing_reference_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                config.ensure_reference_calibration("nope", Path(tmp) / "calib", Path(tmp) / "none")
            self.assertIn("nope_top.json", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
