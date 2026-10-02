#!/usr/bin/env python3
"""A match from the video to the report with ONE command and no manual input.

  python process_match.py ~/football/videos/mecz5_dual.mp4

The name of the match is the name of the video without "_dual" (--name changes it). Steps (a finished step is not repeated, --force repeats it):
  1. half times     matches/<name>.json from the number of players on the pitch (auto_halves.py); needs the calibration, so the
                    calibration is made first with a provisional config of the whole recording (calibrate_match.py, from the match --ref)
  2. analysis       detection, ball, teams (classifier trained on the kick-offs), goals (detect_goals.py), possession, tracks (analyze_dual.py)
  3. statistics     match_stats.py, then the goals again (now with the direction of attack, so the scorers come from the side of the goal), the
                    steps after the goals once more, passes and shots (match_actions.py)
  4. report         report.html (match_report.py): the score and the goals found by the program, the names of the teams from the colour of
                    the kits (--names changes them), the date from the video file (--date changes it)

Nothing from the match report of the league is needed. The result: <analysis folder>/report.html and a short summary on the screen.

Usage:
  python process_match.py ~/football/videos/mecz5_dual.mp4
  python process_match.py VIDEO --names "Czarni,Pomarańczowi" --date 2026-10-02 --model PATH --force
  python process_match.py VIDEO --dry-run        (only prints the commands)
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
HOME = Path.home() / "football"


def kit_name(bgr):
    """A colour name of a kit from its BGR colour (Polish, for the names of the teams), or None when the colour is not clear
    (a washed-out kit in the daylight looks grey): then the teams are called Ciemni and Jasni."""
    b, g, r = (int(x) for x in bgr)
    hsv = cv2.cvtColor(__import__("numpy").uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 60:
        return "Czarni"
    if s < 35 and v > 190:
        return "Biali"
    if s < 130 or v < 90:
        return None
    for hi, name in ((8, "Czerwoni"), (22, "Pomarańczowi"), (35, "Żółci"), (85, "Zieloni"), (105, "Błękitni"), (130, "Niebiescy"), (165, "Fioletowi"), (180, "Czerwoni")):
        if h < hi:
            return name
    return "Czerwoni"


def video_date(video):
    """The date of the match: the date the video file was written (the recording ends with the match)."""
    return datetime.datetime.fromtimestamp(os.path.getmtime(video)).strftime("%Y-%m-%d")


class Runner:
    def __init__(self, dry):
        self.dry = dry
        self.py = sys.executable

    def run(self, title, cmd):
        cmd = [str(c) for c in cmd]
        print(f"\n=== {title}\n$ " + " ".join(cmd), flush=True)
        if self.dry:
            return
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"STOPPED: the step '{title}' failed (exit code {r.returncode}). The steps already done are kept; run the command again after fixing it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", help="name of the match (default: the name of the video without _dual)")
    ap.add_argument("--model", default=str(HOME / "runs" / "detect" / "runs" / "v8_dual" / "weights" / "best.pt"))
    ap.add_argument("--pitch", default=str(HERE / "pitch" / "pitch_6v6.json"))
    ap.add_argument("--ref", default="mecz2", help="the match whose calibration is the reference for the automatic calibration of this one")
    ap.add_argument("--calib-dir", default=str(HOME / "calib"))
    ap.add_argument("--analysis-dir", default=str(HOME / "analysis"))
    ap.add_argument("--out-dir", help="the analysis folder (default: <analysis-dir>/<name>_dual)")
    ap.add_argument("--names", help='names of the teams, e.g. "Czarni,Pomarańczowi" (team 0 first; default: from the colour of the kits)')
    ap.add_argument("--date", help="date of the match (default: the date of the video file)")
    ap.add_argument("--force", action="store_true", help="repeat the steps that are done")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    video = Path(a.video).expanduser()
    if not a.dry_run and not video.exists():
        raise SystemExit(f"video not found: {video}")
    name = a.name or video.stem.replace("_dual", "")
    cfg = HERE / "matches" / f"{name}.json"
    cdir = Path(a.calib_dir).expanduser()
    out = Path(a.out_dir).expanduser() if a.out_dir else Path(a.analysis_dir).expanduser() / (name + "_dual")
    model, pitch = Path(a.model).expanduser(), Path(a.pitch).expanduser()
    R = Runner(a.dry_run)
    T = lambda f: str(HERE / f)
    print(f"match {name}: video {video}, analysis in {out}")

    # 1. half times
    have_calib = all((cdir / f"{name}_{c}.json").exists() for c in ("top", "bottom"))
    if a.force or not cfg.exists():
        if a.force or not have_calib:
            cap = cv2.VideoCapture(str(video)) if not a.dry_run else None
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 29.97)) if cap is not None else 3300.0
            prov = {"name": name, "video": video.name, "halves": [{"start": "00:00:00", "end": "", "start_s": 0.0, "end_s": float(int(dur))}],
                    "calibration": {c: f"calib/{name}_{c}.json" for c in ("top", "bottom")}}
            if not a.dry_run:
                cfg.parent.mkdir(parents=True, exist_ok=True)
                cfg.write_text(json.dumps(prov, indent=2))
            print(f"\n(provisional config of the whole recording written to {cfg})")
        if a.force or not have_calib:
            R.run("calibration", [R.py, T("calibrate_match.py"), video, "--name", name, "--ref", a.ref, "--calib-dir", cdir,
                                  "--pitch", pitch, "--match", cfg, "--model", model])
        R.run("half times", [R.py, T("auto_halves.py"), video, "--name", name, "--model", model, "--pitch", pitch, "--calib-dir", cdir, "--out", cfg])
    else:
        print(f"\nhalf times: {cfg} exists (--force makes them again)")

    # 2. analysis
    common = [R.py, T("analyze_dual.py"), "--match", cfg, "--model", model, "--ref", a.ref, "--out-dir", out, "--auto-goals", "--pitch", pitch,
              "--calib-dir", cdir]
    R.run("analysis (detection, ball, teams, goals, possession, tracks)", common + (["--force"] if a.force else []))

    # 3. statistics, the goals with the direction of attack, the steps after the goals again, passes and shots
    R.run("statistics", [R.py, T("match_stats.py"), out, "--pitch", pitch])
    R.run("goals again (with the direction of attack)", common + ["--from-step", "events"])
    R.run("statistics again", [R.py, T("match_stats.py"), out, "--pitch", pitch])
    R.run("passes and shots", [R.py, T("match_actions.py"), out, "--pitch", pitch])

    # 4. report
    names = a.names
    if not names and not a.dry_run:
        try:
            col = json.loads((out / "teams_colors.json").read_text())
            n0, n1 = kit_name(col["team_0_bgr"]), kit_name(col["team_1_bgr"])
            names = f"{n0},{n1}" if (n0 and n1 and n0 != n1) else "Ciemni,Jasni"
        except Exception:
            names = "Drużyna A,Drużyna B"
    date = a.date or (video_date(video) if video.exists() else "")
    R.run("report", [R.py, T("match_report.py"), out, "--pitch", pitch, "--match", cfg, "--names", names or "A,B", "--date", date, "--lang", "pl"])

    if not a.dry_run:
        try:
            d = json.loads(__import__("re").search(r'id="match-data"[^>]*>(.*?)</script>', (out / "report.html").read_text(), __import__("re").S).group(1))
            print(f"\nDONE: {names}  {d['score'][0]} - {d['score'][1]} (the score from the goals found by the program: {d.get('score_automatic')}), "
                  f"{len(d.get('goals', []))} goals found")
        except Exception:
            print("\nDONE")
        print(f"report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
