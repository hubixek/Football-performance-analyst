#!/usr/bin/env python3
"""Analyze a whole match with one command.
 
Runs all steps in order and saves every result to one folder:
  1. detect.py      -> detections.csv   (players, referees, ball candidates)
  2. ball_track.py  -> ball.csv         (one ball position per frame)
  3. teams.py       -> teams.csv        (team of every player)
  4. possession.py  -> possession.csv, possession.png
  + summary.json with the main statistics
 
Steps whose result already exists are skipped (the detection of a full match takes
a while), so after changing e.g. possession parameters only the later steps run.
Use --force to run everything again, or --from-step to rerun from a given step.
 
Usage:
  python analyze.py ~/football/videos/mecz1_pp.mp4 \
      --model ~/football/runs/detect/runs/v6/weights/best.pt
  # results in ~/football/analysis/mecz1_pp/
 
  python analyze.py ~/football/videos/mecz1_pp.mp4 --model ... --preview   # + preview video
  python analyze.py ~/football/videos/mecz1_pp.mp4 --model ... --from-step teams
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
 
HERE = Path(__file__).resolve().parent
STEPS = ["detect", "ball", "teams", "possession"]
 
 
def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="match video")
    p.add_argument("--model", required=True, help="detection model (best.pt)")
    p.add_argument("--out-dir", help="results folder (default: ~/football/analysis/<video name>)")
    p.add_argument("--preview", action="store_true", help="also create a preview video with possession")
    p.add_argument("--force", action="store_true", help="run all steps again")
    p.add_argument("--from-step", choices=STEPS, help="rerun from this step on")
    p.add_argument("--imgsz", type=int, default=1920)
    a = p.parse_args()
 
    video = Path(a.video).expanduser().resolve()
    model = Path(a.model).expanduser().resolve()
    out = Path(a.out_dir).expanduser() if a.out_dir else Path.home() / "football" / "analysis" / video.stem
    out.mkdir(parents=True, exist_ok=True)
 
    files = {
        "detect": out / "detections.csv",
        "ball": out / "ball.csv",
        "teams": out / "teams.csv",
        "possession": out / "possession.csv",
    }
    first = STEPS.index(a.from_step) if a.from_step else (0 if a.force else len(STEPS))
 
    def needed(step):
        return STEPS.index(step) >= first or not files[step].exists()
 
    py = sys.executable
    start = time.time()
 
    if needed("detect"):
        run([py, HERE / "detect.py", video, "--model", model, "--out", files["detect"], "--imgsz", a.imgsz])
        first = min(first, STEPS.index("ball"))  # later steps depend on new detections
    else:
        print(f"skip detect (exists: {files['detect']})")
 
    if needed("ball"):
        run([py, HERE / "ball_track.py", files["detect"], "--out", files["ball"]])
        first = min(first, STEPS.index("teams"))
    else:
        print(f"skip ball tracking (exists: {files['ball']})")
 
    if needed("teams"):
        run([py, HERE / "teams.py", files["detect"], "--video-in", video, "--out", files["teams"]])
        first = min(first, STEPS.index("possession"))
    else:
        print(f"skip teams (exists: {files['teams']})")
 
    if needed("possession") or a.preview:
        cmd = [py, HERE / "possession.py", files["teams"], "--ball", files["ball"],
               "--out", files["possession"], "--plot", out / "possession.png"]
        if a.preview:
            cmd += ["--video-in", video, "--video-out", out / "preview.mp4"]
        run(cmd)
    else:
        print(f"skip possession (exists: {files['possession']})")
 
    # summary
    counts, frames = {"0": 0, "1": 0}, 0
    with open(files["possession"]) as f:
        for r in csv.DictReader(f):
            frames += 1
            if r["possession_team"] in counts:
                counts[r["possession_team"]] += 1
    total = counts["0"] + counts["1"]
    summary = {
        "video": str(video),
        "model": str(model),
        "created": datetime.now().isoformat(timespec="seconds"),
        "frames": frames,
        "frames_with_possession": total,
        "possession_percent": {f"team_{t}": round(100 * c / total, 1) if total else None
                               for t, c in counts.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
 
    print("\n=== Summary ===")
    for t, c in counts.items():
        pct = 100 * c / total if total else 0
        print(f"Team {t}: {pct:.1f}% possession")
    print(f"Results in {out}  ({(time.time() - start) / 60:.1f} min)")
 
 
if __name__ == "__main__":
    main()
