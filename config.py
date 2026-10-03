"""Where the project keeps its data: videos, calibrations, models and the analysis folders of the matches.

All of it lives under one folder, FOOTBALL_HOME (default ~/football):
  videos/     the recordings (<match>_dual.mp4)
  calib/      the calibration of every match (<match>_top.json, <match>_bottom.json)
  runs/detect/runs/<model>/weights/best.pt   the detection models
  analysis/   one folder per match (the output of process_match.py)

The reference calibration that the automatic calibration of a new match starts from (mecz2, the same camera model) is shipped in
calib_ref/ of the repository; ensure_reference_calibration() copies it into calib/ when it is not there yet.

Usage (in a script):
  from config import HOME, CALIB_DIR, ANALYSIS_DIR, model_path
"""
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
HALVES = ("top", "bottom")


def football_home():
    """FOOTBALL_HOME from the environment, else ~/football."""
    return Path(os.environ.get("FOOTBALL_HOME") or Path.home() / "football").expanduser()


HOME = football_home()
VIDEOS_DIR = HOME / "videos"
CALIB_DIR = HOME / "calib"
ANALYSIS_DIR = HOME / "analysis"
REFERENCE_DIR = REPO / "calib_ref"


def model_path(name="v8_dual"):
    """The weights of a detection model of this project (best.pt of the run `name`)."""
    return HOME / "runs" / "detect" / "runs" / name / "weights" / "best.pt"


def ensure_reference_calibration(ref="mecz2", calib_dir=None, reference_dir=None):
    """Make sure the reference calibration of both lenses is in calib_dir; copy it from the repository when it is missing.

    Returns the list of files that were copied. Raises FileNotFoundError when a file is neither in calib_dir nor in the repository."""
    calib_dir = Path(calib_dir) if calib_dir is not None else CALIB_DIR
    reference_dir = Path(reference_dir) if reference_dir is not None else REFERENCE_DIR
    copied = []
    for half in HALVES:
        target = calib_dir / f"{ref}_{half}.json"
        if target.exists():
            continue
        source = reference_dir / f"{ref}_{half}.json"
        if not source.exists():
            raise FileNotFoundError(f"the reference calibration {target.name} is neither in {calib_dir} nor in {reference_dir}")
        calib_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(target)
    return copied
