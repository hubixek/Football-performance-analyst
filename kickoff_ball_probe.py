#!/usr/bin/env python3
"""Is there ANY ball candidate on the centre spot at the kick-offs that were recognised only by the formation of the players?

On two matches the only two goals that the program missed were kick-offs of this kind: the players stood in the two groups, but the ball was
not tracked on the centre spot. Before anything is built to find such goals, this checks the cheap hypothesis: the detector saw the ball there
(raw candidates of detections.csv), only the ball tracker did not keep it.

For every kick-off of goals_auto.csv (starts too) the raw ball candidates of detections.csv within --radius metres of the centre spot, in the
window from --before seconds before to --after seconds after the kick-off: in how many frames, and the highest confidence. The same numbers
for --controls random moments of the match (away from every kick-off): a candidate on the centre spot also appears by chance (the painted spot,
a stray ball), and only the difference to the controls is information.

Usage:
  python kickoff_ball_probe.py ~/football/analysis/mecz4_2dual --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with detections.csv and goals_auto.csv")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--radius", type=float, default=1.5, help="distance from the centre spot, m")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    cx, cy = pitch["length_m"] / 2, pitch["width_m"] / 2
    ev = []
    with open(folder / "goals_auto.csv") as f:
        for r in csv.DictReader(f):
            ev.append({"t": float(r["kickoff_s"]), "half": int(r["half"]), "how": r["how"], "cat": r["category"]})
    t_all, near_t, near_c = [], [], []
    with open(folder / "detections.csv") as f:
        for r in csv.DictReader(f):
            t = float(r["time_s"])
            t_all.append(t)
            if r["class"] == "ball" and (float(r["x_m"]) - cx) ** 2 + (float(r["y_m"]) - cy) ** 2 <= a.radius ** 2:
                near_t.append(t)
                near_c.append(float(r["conf"]))
    t_all = np.array(t_all)
    near_t, near_c = np.array(near_t), np.array(near_c)

    def stat(t0, before=None, after=None):
        b_, a_ = (a.before if before is None else before), (a.after if after is None else after)
        m = (near_t >= t0 - b_) & (near_t <= t0 + a_)
        return (len(np.unique(np.round(near_t[m], 3))), float(near_c[m].max()) if m.any() else 0.0)

    print(f"raw ball candidates within {a.radius} m of the centre spot, from {a.before:.0f} s before to {a.after:.0f} s after the kick-off\n")
    print(" kick-off  half  how        category            frames  highest confidence   frames in the wide window (8 s before to 2 s after)")
    for e in sorted(ev, key=lambda e: e["t"]):
        n, c = stat(e["t"])
        nw, _ = stat(e["t"], 8.0, 2.0)
        print(f"  {mmss(e['t'])}   {e['half']:4d}  {e['how']:9s}  {e['cat']:18s}  {n:6d}  {c:.2f}                {nw:6d}")

    # where the ball lies at the kick-offs recognised by the ball (tracked ball.csv): the centre spot in the coordinates of this calibration
    bt, bx, by = [], [], []
    if (folder / "ball.csv").exists():
        with open(folder / "ball.csv") as f:
            for r in csv.DictReader(f):
                if r["x_m"]:
                    bt.append(float(r["time_s"])); bx.append(float(r["x_m"])); by.append(float(r["y_m"]))
    bt, bx, by = np.array(bt), np.array(bx), np.array(by)
    offs = []
    for e in ev:
        if e["how"] == "ball" and len(bt):
            m = (bt >= e["t"] - 1.5) & (bt <= e["t"] + 0.5)
            if m.sum() >= 3:
                offs.append((float(np.median(bx[m])) - cx, float(np.median(by[m])) - cy))
    if offs:
        o = np.array(offs)
        print(f"\nthe ball at the {len(o)} kick-offs recognised by the ball, relative to the centre spot of the pitch model ({cx:.1f}, {cy:.1f}): "
              f"median offset dx = {np.median(o[:, 0]):+.2f} m, dy = {np.median(o[:, 1]):+.2f} m; the largest of the {len(o)}: "
              f"|dx| = {np.abs(o[:, 0]).max():.2f} m, |dy| = {np.abs(o[:, 1]).max():.2f} m "
              f"(the spot is the same everywhere: a median offset of more than 0.5 m means that the calibration is shifted there)")
    rng = np.random.default_rng(a.seed)
    lo, hi = float(t_all.min()), float(t_all.max())
    ctrl, ctrl_w = [], []
    while len(ctrl) < a.controls:
        t0 = rng.uniform(lo + a.before, hi - a.after)
        if any(abs(t0 - e["t"]) < 40 for e in ev) or not ((t_all >= t0 - 1) & (t_all <= t0 + 1)).any():
            continue                                                 # near a kick-off, or between the halves (no detections)
        ctrl.append(stat(t0))
        ctrl_w.append(stat(t0, 8.0, 2.0))
    ns = np.array([n for n, _ in ctrl])
    print(f"\ncontrols: {len(ns)} random moments of the match (away from the kick-offs): at least 1 frame with a candidate on the spot in "
          f"{100 * np.mean(ns >= 1):.0f}% of them, at least 3 frames in {100 * np.mean(ns >= 3):.0f}%, at least 6 frames in {100 * np.mean(ns >= 6):.0f}%; "
          f"the most frames in a control: {int(ns.max())}")
    nw_ = np.array([n for n, _ in ctrl_w])
    print(f"controls, the wide window: at least 3 frames in {100 * np.mean(nw_ >= 3):.0f}%, at least 6 in {100 * np.mean(nw_ >= 6):.0f}%, at least 10 in {100 * np.mean(nw_ >= 10):.0f}%; "
          f"the most frames in a control: {int(nw_.max())}")
    kb = [stat(e["t"])[0] for e in ev if e["how"] == "ball"]
    kf = [stat(e["t"])[0] for e in ev if e["how"] != "ball"]
    print(f"kick-offs recognised by the ball: frames with a candidate on the spot, median {np.median(kb) if kb else float('nan'):.0f}; "
          f"by the formation only: median {np.median(kf) if kf else float('nan'):.0f}")


if __name__ == "__main__":
    main()
