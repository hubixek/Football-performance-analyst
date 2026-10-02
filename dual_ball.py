#!/usr/bin/env python3
"""Ball tracking in pitch coordinates (metres) for dual-lens detections.

Input: CSV from dual_detect.py (up to 3 ball candidates per frame + players/referees).
Output: one ball position per processed frame: frame, time_s, half, x_m, y_m, conf, source
(source = detected | interpolated).

Steps:
  1. Spare balls: a spot where ball candidates keep appearing for a large part of the match
     while no player is near is a ball lying still (spare ball, cone, bottle). Candidates
     there are ignored unless a player is right next to them.
  2. Tracking: the ball is acquired from a confident candidate near a player; after that
     only candidates reachable from the last position are accepted (max speed --max-speed m/s).
     A clearly better candidate elsewhere (more confident, near players) that persists for
     --switch-s seconds takes over (the tracker was on a wrong object).
  3. Gaps up to --interp-s seconds are filled by linear interpolation (never across halves).

Usage:
  python dual_ball.py ~/football/analysis/mecz1_dual/detections.csv \
      --out ~/football/analysis/mecz1_dual/ball.csv --plot ~/football/analysis/mecz1_dual/ball.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path, on_body=0.6, inner=0.7):
    """Frames, ball candidates and people. A ball candidate whose centre lies on the upper body of a detected person
    (inside the middle `inner` of the box width and the upper `on_body` of its height: head, cap, shirt) is dropped:
    the model takes white caps, heads, socks and shirts for the ball there, while the real ball at a player's feet lies
    in the lower part of the box or below it. on_body = 0 switches this off."""
    frames = {}
    balls, people = defaultdict(list), defaultdict(list)
    cand, boxes = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            frames[fr] = (float(r["time_s"]), int(r["half"]))
            xy = (float(r["x_m"]), float(r["y_m"]))
            box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")) if r.get("x1") not in (None, "") else None
            if r["class"] == "ball":
                cand[fr].append((xy, float(r["conf"]), r.get("cam", ""), box))
            else:
                people[fr].append(xy)
                if box is not None:
                    boxes[(fr, r.get("cam", ""))].append(box)
    dropped = 0
    for fr, lst in cand.items():
        for xy, conf, cam, box in lst:
            if on_body > 0 and box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hit = False
                for x1, y1, x2, y2 in boxes.get((fr, cam), ()):
                    m = (1 - inner) / 2 * (x2 - x1)
                    if x1 + m <= cx <= x2 - m and y1 <= cy <= y1 + on_body * (y2 - y1):
                        hit = True
                        break
                if hit:
                    dropped += 1
                    continue
            balls[fr].append((*xy, conf))
    n = sum(len(v) for v in cand.values())
    if on_body > 0:
        print(f"ball candidates: {n}, {dropped} ({100 * dropped / max(n, 1):.1f}%) dropped - on the upper body of a person (head, cap, shirt)")
    return frames, balls, people


def nearest_person(pt, people):
    if not people:
        return np.inf
    p = np.asarray(people)
    return float(np.min(np.hypot(p[:, 0] - pt[0], p[:, 1] - pt[1])))


def static_spots(frames, balls, people, cell=0.5, min_share=0.15, far=3.0):
    """Grid cells where candidates lie still, away from players, for a large part of the match."""
    count = defaultdict(int)
    for fr, cands in balls.items():
        seen = set()
        for x, y, _ in cands:
            if nearest_person((x, y), people.get(fr, [])) > far:
                seen.add((int(np.floor(x / cell)), int(np.floor(y / cell))))
        for c in seen:
            count[c] += 1
    n = max(len(frames), 1)
    spots = {c for c, k in count.items() if k / n >= min_share}
    # also block the neighbouring cells (a lying ball is detected a bit differently each time)
    return {(cx + dx, cy + dy) for cx, cy in spots for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def track(frames, balls, people, a, spots):
    order = sorted(frames)
    in_spot = lambda x, y: (int(np.floor(x / a.cell)), int(np.floor(y / a.cell))) in spots
    track, last, last_t, last_half = {}, None, None, None
    challenger, challenger_since = None, None

    def score(c, fr):
        d = nearest_person(c[:2], people.get(fr, []))
        return c[2] + (0.3 if d < a.near else 0.0)

    for fr in order:
        t, half = frames[fr]
        cands = [c for c in balls.get(fr, [])
                 if not in_spot(c[0], c[1]) or nearest_person(c[:2], people.get(fr, [])) < 1.5]
        if half != last_half:
            last, challenger = None, None           # new half: acquire again
            last_half = half
        if last is not None and t - last_t > a.lost_s:
            last = None
        chosen = None
        if last is not None:
            reach = a.gate + a.max_speed * (t - last_t)
            options = [c for c in cands if np.hypot(c[0] - last[0], c[1] - last[1]) <= reach]
            if options:
                chosen = max(options, key=lambda c: score(c, fr) - 0.5 * np.hypot(c[0] - last[0], c[1] - last[1]) / reach)
            others = [c for c in cands if c not in options and c[2] >= a.acquire_conf
                      and nearest_person(c[:2], people.get(fr, [])) < a.near]
            best = max(others, key=lambda c: score(c, fr), default=None)
            current = score(chosen, fr) if chosen else 0.0
            if best is not None and score(best, fr) > current + a.switch_margin:
                if challenger is None or np.hypot(best[0] - challenger[0], best[1] - challenger[1]) > 3.0:
                    challenger_since = t
                challenger = best
                if t - challenger_since >= a.switch_s:
                    # remove the frames spent on the wrong object
                    for f2 in [f2 for f2 in track if frames[f2][0] >= challenger_since]:
                        del track[f2]
                    chosen, challenger = best, None
            else:
                challenger = None
        else:
            options = [c for c in cands if c[2] >= a.acquire_conf
                       and nearest_person(c[:2], people.get(fr, [])) < a.near]
            if options:
                chosen = max(options, key=lambda c: score(c, fr))
        if chosen is not None:
            track[fr] = chosen
            last, last_t = chosen, t
    return track


def interpolate(frames, track, max_gap_s):
    out = {fr: (c[0], c[1], c[2], "detected") for fr, c in track.items()}
    known = sorted(track)
    for f0, f1 in zip(known, known[1:]):
        (t0, h0), (t1, h1) = frames[f0], frames[f1]
        if h0 != h1 or t1 - t0 > max_gap_s:
            continue
        for fr in (f for f in frames if f0 < f < f1):
            s = (frames[fr][0] - t0) / (t1 - t0)
            x = track[f0][0] + s * (track[f1][0] - track[f0][0])
            y = track[f0][1] + s * (track[f1][1] - track[f0][1])
            out[fr] = (x, y, 0.0, "interpolated")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="CSV from dual_detect.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", help="optional PNG with the ball path of the first minute")
    ap.add_argument("--max-speed", type=float, default=30.0, help="fastest ball, m/s")
    ap.add_argument("--gate", type=float, default=1.5, help="extra search radius, m")
    ap.add_argument("--near", type=float, default=3.0, help="'near a player' distance, m")
    ap.add_argument("--acquire-conf", type=float, default=0.3)
    ap.add_argument("--switch-margin", type=float, default=0.25)
    ap.add_argument("--switch-s", type=float, default=0.4, help="seconds a better candidate must persist")
    ap.add_argument("--lost-s", type=float, default=1.5, help="seconds without the ball before re-acquiring")
    ap.add_argument("--interp-s", type=float, default=1.0, help="longest gap filled by interpolation, s")
    ap.add_argument("--cell", type=float, default=0.5, help="grid size for spare-ball spots, m")
    ap.add_argument("--on-body", type=float, default=0.6,
                    help="drop ball candidates on the upper part of a person's box (this share of its height; 0 = off)")
    a = ap.parse_args()

    frames, balls, people = load(Path(a.detections).expanduser(), a.on_body)
    spots = static_spots(frames, balls, people, a.cell)
    trk = track(frames, balls, people, a, spots)
    ball = interpolate(frames, trk, a.interp_s)

    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "x_m", "y_m", "conf", "source"])
        for fr in sorted(ball):
            x, y, c, src = ball[fr]
            t, h = frames[fr]
            w.writerow([fr, t, h, round(x, 2), round(y, 2), round(c, 3), src])
    n = len(frames)
    if n == 0:
        raise SystemExit("no detections of people in the file: the detection step found nothing (wrong or missing video, half times "
                         "outside the video, or a calibration that does not fit)")
    det = sum(1 for v in ball.values() if v[3] == "detected")
    print(f"Frames: {n} | spare-ball spots: {len(spots) // 9} | ball detected: {100 * det / n:.1f}% "
          f"| with interpolation: {100 * len(ball) / n:.1f}%")
    print(f"Saved {out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = sorted(ball)
        t0 = frames[order[0]][0]
        pts = [(ball[f][0], ball[f][1], ball[f][3]) for f in order if frames[f][0] - t0 < 60]
        fig, ax = plt.subplots(figsize=(10, 6))
        pitch_x, pitch_y = [0, 56, 56, 0, 0], [0, 0, 32.4, 32.4, 0]
        ax.plot(pitch_x, pitch_y, "k-")
        ax.plot([28, 28], [0, 32.4], "k-")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="tab:gray", lw=0.8)
        for src, col in (("detected", "tab:red"), ("interpolated", "tab:orange")):
            sel = [p for p in pts if p[2] == src]
            if sel:
                ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=6, c=col, label=src)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend()
        ax.set_title("Ball path, first minute")
        fig.savefig(Path(a.plot).expanduser(), dpi=120)
        print(f"Saved plot {a.plot}")


if __name__ == "__main__":
    main()
