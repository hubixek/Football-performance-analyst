#!/usr/bin/env python3
"""Analyse a whole match recorded with the dual-lens camera, with one command.

Steps (results in one folder, finished steps are skipped):
  0. calibration      calibrate_match.py   (only if calib/<name>_top.json / _bottom.json are missing
                                             and --ref is given)
  1. detect           dual_detect.py     -> detections.csv
  2. ball             dual_ball.py       -> ball.csv, ball.png
  3. teams            dual_teams.py      -> teams.csv, teams_colors.json
  4. possession       dual_possession.py -> possession.csv, summary.json, possession.png

Usage:
  python analyze_dual.py --match matches/mecz1.json \
      --model ~/football/runs/detect/runs/v7_dual/weights/best.pt --ref mecz2
  python analyze_dual.py --match matches/mecz1.json --model ... --limit 300     # quick test
  python analyze_dual.py --match matches/mecz1.json --model ... --from-step teams
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["detect", "ball", "teams", "possession"]


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="match config (match_config.py)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", default=str(Path.home() / "football" / "videos"))
    ap.add_argument("--calib-dir", default=str(Path.home() / "football" / "calib"))
    ap.add_argument("--pitch", default=str(HERE / "pitch" / "pitch_6v6.json"))
    ap.add_argument("--out-dir", help="default: ~/football/analysis/<name>_dual")
    ap.add_argument("--ref", help="reference match for automatic calibration, e.g. mecz2")
    ap.add_argument("--limit", type=float, default=0, help="only the first N seconds of play (test)")
    ap.add_argument("--from-step", choices=STEPS, help="rerun from this step on")
    ap.add_argument("--force", action="store_true", help="rerun all steps")
    a = ap.parse_args()

    match = Path(a.match).expanduser().resolve()
    cfg = json.loads(match.read_text())
    name = cfg["name"]
    video = Path(a.videos).expanduser() / cfg["video"]
    if not video.exists():
        video = Path(a.videos).expanduser() / f"{name}_dual.mp4"
    cdir = Path(a.calib_dir).expanduser()
    out = Path(a.out_dir).expanduser() if a.out_dir else Path.home() / "football" / "analysis" / f"{name}_dual"
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    start = time.time()

    # 0) calibration
    if not all((cdir / f"{name}_{h}.json").exists() for h in ("top", "bottom")):
        if not a.ref:
            raise SystemExit(f"No calibration for {name} in {cdir} - give --ref <reference match> "
                             "to calibrate automatically, or calibrate manually")
        run([py, HERE / "calibrate_match.py", video, "--name", name, "--ref", a.ref,
             "--calib-dir", cdir, "--pitch", a.pitch, "--match", match])

    files = {"detect": out / "detections.csv", "ball": out / "ball.csv",
             "teams": out / "teams.csv", "possession": out / "summary.json"}
    first = 0 if a.force else STEPS.index(a.from_step) if a.from_step else len(STEPS)

    def needed(step):
        return STEPS.index(step) >= first or not files[step].exists()

    if needed("detect"):
        cmd = [py, HERE / "dual_detect.py", video, "--match", match, "--model", Path(a.model).expanduser(),
               "--calib-dir", cdir, "--pitch", a.pitch, "--out", files["detect"]]
        if a.limit:
            cmd += ["--limit", a.limit]
        run(cmd)
        first = min(first, 1)
    if needed("ball"):
        run([py, HERE / "dual_ball.py", files["detect"], "--out", files["ball"], "--plot", out / "ball.png"])
        first = min(first, 2)
    if needed("teams"):
        run([py, HERE / "dual_teams.py", files["detect"], "--video", video, "--out", files["teams"]])
        first = min(first, 3)
    if needed("possession"):
        run([py, HERE / "dual_possession.py", files["teams"], "--ball", files["ball"],
             "--out-dir", out, "--pitch", a.pitch])

    summary = json.loads(files["possession"].read_text())
    p = summary["possession_percent"]
    print(f"\n=== {name}: team 0 {p['team_0']}% | team 1 {p['team_1']}% possession ===")
    print(f"Results in {out}  ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
