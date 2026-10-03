#!/usr/bin/env python3
"""Extract frames for annotation from a dual-lens recording.

Splits every chosen moment into the top and bottom half (one image per lens) and saves them as
  <out>/<name>/<name>_top_<tenths of a second>.jpg, <name>_bottom_<...>.jpg

Which moments (--mode):
  uniform   a moment every --every seconds, during the halves of the match config (default; both lenses)
  hard      moments where the model probably misses the ball - the frames that teach it most. Needs --analysis, the
            analysis folder of the same match (analyze_dual.py): 
              - gaps in the ball track: the detector did not find the ball for --min-gap to --max-gap s while the ball was in play
                (the longer the gap and the faster the ball moved across it, the higher the priority),
              - the moments right after a shot (actions.csv from match_actions.py): the ball is fast and small then.
            Only the lens in which the ball should be seen is saved (from the ball position and the calibration of the match).
  both      uniform and hard together

Usage:
  python extract_dual_frames.py ~/football/videos/mecz2_dual.mp4 --match matches/mecz2.json --every 20
  python extract_dual_frames.py ~/football/videos/mecz4_dual.mp4 --match matches/mecz4.json --mode both --every 60 \\
      --analysis ~/football/analysis/mecz4_dual --max-hard 60
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import config


def read_ball(folder):
    rows = []
    with open(folder / "ball.csv") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["time_s"]), int(r["half"]), float(r["x_m"]), float(r["y_m"]), r["source"]))
    return rows


def read_states(folder):
    p = folder / "possession.csv"
    if not p.exists():
        return None
    t, st = [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            t.append(float(r["time_s"]))
            st.append(r.get("ball_state", ""))
    return np.array(t), np.array(st)


def hard_targets(folder, halves, min_gap, max_gap, max_n, spacing, pitch_lw):
    """[(time, reason, x, y)] sorted by priority, at least `spacing` seconds apart."""
    L, W = pitch_lw
    ball = read_ball(folder)
    states = read_states(folder)
    cands = []
    detected = [b for b in ball if b[4] == "detected"]
    for a, b in zip(detected, detected[1:]):
        if a[1] != b[1]:
            continue
        gap = b[0] - a[0]
        if not (min_gap <= gap <= max_gap):
            continue
        mid = (a[0] + b[0]) / 2
        x, y = (a[2] + b[2]) / 2, (a[3] + b[3]) / 2
        if not (0 <= x <= L and 0 <= y <= W):
            continue
        if states is not None:
            i = int(np.clip(np.searchsorted(states[0], mid), 0, len(states[0]) - 1))
            if states[1][i] in ("out", "still", "restart"):
                continue                                              # dead ball: nothing to learn from a lost ball
        speed = float(np.hypot(b[2] - a[2], b[3] - a[3]) / gap)
        cands.append((gap * (1 + speed / 5.0), mid, "ball_gap", x, y))
    shots = folder / "actions.csv"
    if shots.exists():
        pos = {round(b[0], 2): (b[2], b[3]) for b in ball}
        times = np.array([b[0] for b in ball])
        with open(shots) as f:
            for r in csv.DictReader(f):
                if r["type"] != "shot":
                    continue
                for dt, bonus in ((0.3, 1000.0), (0.8, 900.0)):
                    t = float(r["time_s"]) + dt
                    i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1)) if len(times) else None
                    x, y = (ball[i][2], ball[i][3]) if i is not None and abs(times[i] - t) < 1.0 else (None, None)
                    cands.append((bonus, t, "shot", x, y))
    cands.sort(key=lambda c: -c[0])
    chosen = []
    for score, t, reason, x, y in cands:
        if any(h["start_s"] <= t < h["end_s"] for h in halves) and all(abs(t - c[0]) >= spacing for c in chosen):
            chosen.append((t, reason, x, y))
        if len(chosen) >= max_n:
            break
    return sorted(chosen)


def choose_lens(calibs, x, y, w, h):
    """The lens (top / bottom) in which the pitch point (x, y) is seen, nearest to the image centre; None if unknown."""
    if x is None or not calibs:
        return None
    from calibrate import pitch_to_pixels
    best = None
    for part, calib in calibs.items():
        uv = pitch_to_pixels(calib, np.array([[x, y]]))[0]
        if not np.isfinite(uv).all() or not (10 <= uv[0] < w - 10 and 10 <= uv[1] < h - 10):
            continue
        d = np.hypot(uv[0] - calib.get("cx", w / 2), uv[1] - calib.get("cy", h / 2)) / calib.get("f", w)
        if best is None or d < best[0]:
            best = (d, part)
    return None if best is None else best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--match", required=True, help="match config with half times (match_config.py)")
    ap.add_argument("--out", default=str(config.HOME / "frames_dual"))
    ap.add_argument("--mode", choices=["uniform", "hard", "both"], default="uniform")
    ap.add_argument("--every", type=float, default=20.0, help="uniform: seconds between moments")
    ap.add_argument("--analysis", help="hard: analysis folder of this match (ball.csv, possession.csv, optionally actions.csv)")
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR), help="hard: calibration of the match, to pick the lens")
    ap.add_argument("--pitch", help="hard: pitch json (default 56 x 32.4 m)")
    ap.add_argument("--max-hard", type=int, default=60, help="hard: how many moments")
    ap.add_argument("--min-gap", type=float, default=0.5, help="hard: shortest gap in the ball detections, s")
    ap.add_argument("--max-gap", type=float, default=6.0, help="hard: longest gap (longer ones are usually a dead ball), s")
    ap.add_argument("--spacing", type=float, default=4.0, help="hard: moments at least this far apart, s")
    a = ap.parse_args()

    cfg = json.loads(Path(a.match).read_text())
    name = cfg["name"]
    out = Path(a.out).expanduser() / name
    out.mkdir(parents=True, exist_ok=True)
    halves = cfg["halves"]

    moments = []                                                       # (time, reason, x, y)
    if a.mode in ("uniform", "both"):
        for half in halves:
            t = half["start_s"]
            while t < half["end_s"]:
                moments.append((t, "uniform", None, None))
                t += a.every
    if a.mode in ("hard", "both"):
        if not a.analysis:
            raise SystemExit("--mode hard needs --analysis (the analysis folder of this match)")
        pitch = json.loads(Path(a.pitch).read_text()) if a.pitch else {"length_m": 56.0, "width_m": 32.4}
        hard = hard_targets(Path(a.analysis).expanduser(), halves, a.min_gap, a.max_gap, a.max_hard, a.spacing, (pitch["length_m"], pitch["width_m"]))
        print(f"{len(hard)} hard moments: " + ", ".join(f"{r} {sum(1 for h in hard if h[1] == r)}" for r in sorted({h[1] for h in hard})))
        moments += hard
    calibs = {}
    cdir = Path(a.calib_dir).expanduser()
    if a.mode in ("hard", "both") and all((cdir / f"{name}_{p}.json").exists() for p in ("top", "bottom")):
        calibs = {p: json.loads((cdir / f"{name}_{p}.json").read_text()) for p in ("top", "bottom")}
    elif a.mode in ("hard", "both"):
        print(f"no calibration {name}_top/bottom.json in {cdir}: both lenses are saved for the hard moments")

    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    manifest, n_img = [], 0
    for t, reason, x, y in sorted(moments):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[0] // 2, img.shape[1]
        parts = {"top": img[:h], "bottom": img[h:2 * h]}
        lens = choose_lens(calibs, x, y, w, h) if reason != "uniform" else None
        for part, crop in parts.items():
            if lens is not None and part != lens:
                continue
            fname = f"{name}_{part}_{int(round(t * 10)):06d}.jpg"
            cv2.imwrite(str(out / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append((fname, round(t, 1), reason, "" if x is None else round(x, 1), "" if y is None else round(y, 1)))
            n_img += 1
    cap.release()
    with open(out / "frames.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "time_s", "reason", "ball_x_m", "ball_y_m"])
        wr.writerows(manifest)
    print(f"{len(moments)} moments -> {n_img} frames in {out} (list with the reason for every frame: frames.csv)")


if __name__ == "__main__":
    main()
