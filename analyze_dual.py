#!/usr/bin/env python3
"""Analyse a whole match recorded with the dual-lens camera, with one command.

Steps (results in one folder, finished steps are skipped):
  0. calibration      calibrate_match.py   (only if calib/<name>_top.json / _bottom.json are missing
                                             and --ref is given)
  1. detect           dual_detect.py     -> detections.csv
  2. ball             dual_ball.py       -> ball.csv, ball.png
  3. teams            dual_teams.py      -> teams.csv, teams_colors.json
  4. goals (optional) match_events.py --goals -> restarts.json (when --goals is given, or --auto-goals finds them with detect_goals.py)
  5. possession       dual_possession.py -> possession.csv, summary.json, possession.png
  6. tracks           dual_tracks.py     -> tracks.csv, players.csv, tracks_sample.png, players.png

Usage:
  python analyze_dual.py --match matches/mecz1.json \
      --model ~/football/runs/detect/runs/v7_dual/weights/best.pt --ref mecz2
  python analyze_dual.py --match matches/mecz1.json --model ... --limit 300     # quick test
  python analyze_dual.py --match matches/mecz1.json --model ... --from-step teams
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import config

HERE = Path(__file__).resolve().parent
STEPS = ["detect", "ball", "teams", "events", "possession", "tracks"]


def goals_from_auto(path):
    """The goal times (mm:ss) that detect_goals.py found: kick-offs recognised by the ball (goals_auto.csv, category goal)."""
    goals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("category") == "goal" and r.get("goal_s"):
                goals.append(float(r["goal_s"]))
    return ",".join(f"{int(g) // 60}:{int(g) % 60:02d}" for g in sorted(goals))


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="match config (match_config.py)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", default=str(config.VIDEOS_DIR))
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    ap.add_argument("--pitch", default=str(HERE / "pitch" / "pitch_6v6.json"))
    ap.add_argument("--out-dir", help="default: ~/football/analysis/<name>_dual")
    ap.add_argument("--ref", help="reference match for automatic calibration, e.g. mecz2")
    ap.add_argument("--limit", type=float, default=0, help="only the first N seconds of play (test)")
    ap.add_argument("--from-step", choices=STEPS, help="rerun from this step on")
    ap.add_argument("--force", action="store_true", help="rerun all steps")
    ap.add_argument("--goals", help="times of the goals, e.g. 14:21,21:29,38:04 (as in a video player): no possession "
                    "while the ball is fetched and the teams take their places after a goal")
    ap.add_argument("--auto-goals", action="store_true",
                    help="find the goals automatically (detect_goals.py: kick-offs recognised by the ball) when --goals is not given; "
                         "to apply them to a finished analysis add --from-step events")
    ap.add_argument("--teams-method", choices=["auto", "kickoff", "colour", "clf"], default="auto",
                    help="how the teams are found (dual_teams.py --method): kickoff = from the kick-offs, colour = shirt colour over the whole "
                         "match, clf = a classifier of the appearance trained on the kick-offs, auto = kick-offs when enough are usable (default)")
    ap.add_argument("--teams-self-train", type=int, default=0, help="--teams-method clf: rounds of learning again with the sure detections of the whole match")
    ap.add_argument("--reuse-colors", action="store_true",
                    help="teams step: reuse the jersey colours of an earlier run (no video reading)")
    a = ap.parse_args()

    match = Path(a.match).expanduser().resolve()
    try:
        cfg = json.loads(match.read_text())
    except json.JSONDecodeError as e:
        line = match.read_text().splitlines()[e.lineno - 1] if e.lineno else ""
        raise SystemExit(f"the match config {match} is not valid JSON: line {e.lineno}, column {e.colno}: {line.strip()!r}\n"
                         f"(a text value needs quotes, e.g. \"video\": \"mecz4_2dual.mp4\", with a comma at the end of the line)")
    name = cfg["name"]
    video = Path(a.videos).expanduser() / cfg["video"]
    if not video.exists():
        video = Path(a.videos).expanduser() / f"{name}_dual.mp4"
    if not video.exists():
        videos = sorted(p.name for p in Path(a.videos).expanduser().glob("*.mp4"))
        raise SystemExit(f"video not found: tried {Path(a.videos).expanduser() / cfg['video']} and {video}\n"
                         f"the field \"video\" of {match} is '{cfg['video']}'; the videos in {Path(a.videos).expanduser()}: "
                         f"{', '.join(videos) or 'none'}\nfix the field (or the --videos folder) and run again")
    cdir = Path(a.calib_dir).expanduser()
    out = Path(a.out_dir).expanduser() if a.out_dir else config.ANALYSIS_DIR / f"{name}_dual"
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
             "teams": out / "teams.csv", "events": out / "restarts.json", "possession": out / "summary.json",
             "tracks": out / "players.csv"}
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
        cmd = [py, HERE / "dual_teams.py", files["detect"], "--video", video, "--out", files["teams"], "--ball", files["ball"]]
        cmd += ["--method", a.teams_method]
        if a.teams_self_train:
            cmd += ["--self-train", str(a.teams_self_train)]
        if a.reuse_colors:
            cmd.append("--reuse-colors")
        run(cmd)
        first = min(first, 3)
    if a.auto_goals and not a.goals and files["teams"].exists():
        run([py, HERE / "detect_goals.py", out, "--pitch", a.pitch, "--match", match])
        a.goals = goals_from_auto(out / "goals_auto.csv")
        print(f"goals found automatically: {a.goals}" if a.goals else "no goals found automatically")
    if a.goals and (needed("events") or a.from_step in ("events", "possession") or a.force):
        run([py, HERE / "match_events.py", out, "--pitch", a.pitch, "--goals", a.goals])
        first = min(first, 4)
    if needed("possession"):
        cmd = [py, HERE / "dual_possession.py", files["teams"], "--ball", files["ball"],
               "--out-dir", out, "--pitch", a.pitch]
        if a.goals:
            cmd += ["--restarts", files["events"]]
        run(cmd)

    if needed("tracks"):
        run([py, HERE / "dual_tracks.py", files["teams"], "--pitch", a.pitch])

    summary = json.loads(files["possession"].read_text())
    p = summary["possession_percent"]
    print(f"\n=== {name}: team 0 {p['team_0']}% | team 1 {p['team_1']}% possession ===")
    print(f"Results in {out}  ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
