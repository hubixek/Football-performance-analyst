#!/usr/bin/env python3
"""Short video (and a few still frames) of what the model detected, drawn on the recording.

For a chosen piece of the match, both lenses one above the other:
  - players: box in the colour of the team from teams.csv (team 0 blue, team 1 red, unknown grey), goalkeepers "GK",
  - referees: yellow box,
  - ball candidates: small green box with the confidence; the candidate the ball tracker used is marked with a
    thick circle, candidates dropped because they lie on the upper body of a person (head, cap, shirt) are magenta,
  - the header: time, how many players of each team are in the frame, the team in possession.

Usage:
  python draw_detections.py ~/football/analysis/mecz1_dual_v8 --video ~/football/videos/mecz1_dual.mp4 \\
      --start 38:39 --duration 20
"""
import argparse
import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TEAM_COL = {"0": (230, 120, 30), "1": (40, 40, 220), "-1": (160, 160, 160), "": (160, 160, 160)}
REF_COL, BALL_COL, USED_COL, DROP_COL = (0, 230, 255), (60, 220, 60), (0, 255, 255), (220, 0, 220)


def parse_time(text):
    t = 0.0
    for part in str(text).split(":"):
        t = t * 60 + float(part)
    return t


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def on_body(ball_box, boxes, share=0.6, inner=0.7):
    cx, cy = (ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2
    for x1, y1, x2, y2 in boxes:
        m = (1 - inner) / 2 * (x2 - x1)
        if x1 + m <= cx <= x2 - m and y1 <= cy <= y1 + share * (y2 - y1):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder (teams.csv, ball.csv, possession.csv)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", default="5:00", help="start in the recording, e.g. 12:30")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds")
    ap.add_argument("--scale", type=float, default=0.6, help="size of the output video (1.0 = 2048 px wide)")
    ap.add_argument("--stills", type=int, default=4, help="how many single frames to save as JPG as well")
    ap.add_argument("--out")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    t0 = parse_time(a.start)
    t1 = t0 + a.duration
    dets = defaultdict(list)
    with open(folder / "teams.csv") as f:
        for r in csv.DictReader(f):
            t = float(r["time_s"])
            if t < t0:
                continue
            if t > t1:
                break
            dets[int(r["frame"])].append(r)
    if not dets:
        raise SystemExit(f"no detections between {mmss(t0)} and {mmss(t1)}")
    ball = {}
    with open(folder / "ball.csv") as f:
        for r in csv.DictReader(f):
            if r["x_m"] and t0 <= float(r["time_s"]) <= t1:
                ball[int(r["frame"])] = (float(r["x_m"]), float(r["y_m"]), r["source"])
    poss = {}
    if (folder / "possession.csv").exists():
        with open(folder / "possession.csv") as f:
            for r in csv.DictReader(f):
                if t0 <= float(r["time_s"]) <= t1:
                    poss[int(r["frame"])] = r["possession_team"]

    out = Path(a.out).expanduser() if a.out else folder / f"detections_{mmss(t0).replace(':', 'm')}_{int(a.duration)}s.mp4"
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    frames = sorted(dets)
    step = np.median(np.diff(frames)) if len(frames) > 1 else 1
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30) / step
    vw = None
    tmp = out.with_suffix(".tmp.mp4")
    still_at = set(np.linspace(0, len(frames) - 1, a.stills).astype(int).tolist()) if a.stills else set()
    counts = defaultdict(int)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    pos = frames[0]
    for k, fr in enumerate(frames):
        while pos < fr:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            break
        h = img.shape[0] // 2
        rows = dets[fr]
        people = defaultdict(list)
        for r in rows:
            if r["class"] in ("player", "referee"):
                people[r["cam"]].append(tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2")))
        n_team = defaultdict(int)
        used = None
        if fr in ball and ball[fr][2] == "detected":
            bx, by, _ = ball[fr]
            cands = [r for r in rows if r["class"] == "ball"]
            if cands:
                used = min(cands, key=lambda r: np.hypot(float(r["x_m"]) - bx, float(r["y_m"]) - by))
                if np.hypot(float(used["x_m"]) - bx, float(used["y_m"]) - by) > 0.5:
                    used = None
        for r in rows:
            x1, y1, x2, y2 = (int(float(r[c])) for c in ("x1", "y1", "x2", "y2"))
            dy = 0 if r["cam"] == "top" else h
            y1, y2 = y1 + dy, y2 + dy
            if r["class"] == "player":
                col = TEAM_COL.get(r.get("team", ""), TEAM_COL[""])
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                label = ("GK " if r.get("gk") == "1" else "") + (f"T{r['team']}" if r.get("team") in ("0", "1") else "?")
                cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                n_team[r.get("team", "")] += 1
            elif r["class"] == "referee":
                cv2.rectangle(img, (x1, y1), (x2, y2), REF_COL, 3)
                cv2.putText(img, "REF", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, REF_COL, 2, cv2.LINE_AA)
                counts["referee"] += 1
            elif r["class"] == "ball":
                box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
                dropped = on_body(box, people[r["cam"]])
                col = DROP_COL if dropped else BALL_COL
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), col, 2)
                cv2.putText(img, f"{float(r['conf']):.2f}", (x2 + 5, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                counts["ball_dropped" if dropped else "ball"] += 1
                if r is used:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 26, USED_COL, 4, cv2.LINE_AA)
        for t in ("0", "1"):
            counts[f"team_{t}"] += n_team[t]
        t = float(rows[0]["time_s"])
        header = (f"{mmss(t)}   team 0 (blue): {n_team['0']}   team 1 (red): {n_team['1']}   unknown: {n_team['-1']}   "
                  f"possession: {poss.get(fr, '') or '-'}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(img, header, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "green = ball candidate, circle = used by the tracker, magenta = dropped (on a body)",
                    (20, h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        small = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (small.shape[1], small.shape[0]))
        vw.write(small)
        if k in still_at:
            cv2.imwrite(str(out.with_name(f"{out.stem}_{mmss(t).replace(':', 'm')}_{k:03d}.jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    if vw is not None:
        vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    else:
        tmp.rename(out)
    n = len(frames)
    print(f"{n} frames {mmss(t0)}-{mmss(t1)}: per frame team 0 {counts['team_0'] / n:.1f}, team 1 {counts['team_1'] / n:.1f}, "
          f"referee {counts['referee'] / n:.1f}; ball candidates {counts['ball'] / n:.2f} (+ {counts['ball_dropped'] / n:.2f} dropped on a body)")
    print(f"Saved {out} and {len(still_at)} still frames next to it")


if __name__ == "__main__":
    main()
