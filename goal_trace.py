#!/usr/bin/env python3
"""What do the ball track and the possession look like just before every goal? One line per goal.

The goals are the 12 (or 13) known shots of a match: a place to see why match_actions.py does not recognise them as shots.
Reads possession.csv (the ball, the controller, the state) and the goals of goals_auto.csv (detect_goals.py; category goal).

For every goal:
  known   share of the frames in the last 3 s before the goal with a known ball position
  dist    distance of the ball to the goal line (m) 4, 3, 2, 1 and 0 s before the goal ('.' = ball not seen within 0.4 s)
  vmax    the highest speed of the ball in the last 6 s (m/s, from positions 0.4 s apart)
  ctrl    the last time before the goal when a player of the scoring team controlled the ball: seconds before the goal and
          the distance of the ball to the goal line then (m)
  state   the ball state of the possession file 1 s before the goal

Summary at the end: how many goals have the ball known, near the goal, a control within the shooting range ...

Usage:
  python goal_trace.py ~/football/analysis/mecz1_dual_v8 --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with possession.csv and goals_auto.csv")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--shot-range", type=float, default=22.0, help="the shooting range used by match_actions.py, m")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    L = json.loads(Path(a.pitch).read_text())["length_m"]
    T, H, X, Y, C, S = [], [], [], [], [], []
    with open(folder / "possession.csv") as f:
        for r in csv.DictReader(f):
            T.append(float(r["time_s"]))
            H.append(int(r["half"]))
            X.append(float(r["ball_x"]) if r["ball_x"] else np.nan)
            Y.append(float(r["ball_y"]) if r["ball_y"] else np.nan)
            C.append(int(r["controller_team"]) if r["controller_team"] != "" else -1)
            S.append(r.get("ball_state", ""))
    T, H, X, Y, C, S = map(np.array, (T, H, X, Y, C, S))
    goals = []
    with open(folder / "goals_auto.csv") as f:
        for r in csv.DictReader(f):
            if r["category"] == "goal" and r["goal_s"]:
                goals.append({"half": int(r["half"]), "t": float(r["goal_s"]), "side": r["side"], "scorer": r["scorer"], "rough": r["goal_time_is_rough"] == "1"})
    if not goals:
        raise SystemExit("no goals in goals_auto.csv (run detect_goals.py)")

    def dline(x, side):
        return x if side == "left" else L - x

    print(" #  goal   side   known  dist to the goal line 4/3/2/1/0 s before          vmax   last control of the scorer's team          state")
    rows = []
    for k, g in enumerate(goals, 1):
        side = g["side"]
        if not side:
            print(f"{k:2d}  {mmss(g['t'])}  (no side: the ball was not seen at a goal; goal time rough)")
            rows.append(None)
            continue
        m = (H == g["half"])
        w3 = m & (T >= g["t"] - 3) & (T <= g["t"])
        known = float(np.mean(~np.isnan(X[w3]))) if w3.any() else 0.0
        d = []
        for s in (4, 3, 2, 1, 0):
            j = np.flatnonzero(m & (np.abs(T - (g["t"] - s)) <= 0.4) & ~np.isnan(X))
            d.append(f"{dline(X[j[np.argmin(np.abs(T[j] - (g['t'] - s)))]], side):4.1f}" if len(j) else "   .")
        dnum = [float(x) if x.strip() != "." else np.nan for x in d]
        w6 = np.flatnonzero(m & (T >= g["t"] - 6) & (T <= g["t"]) & ~np.isnan(X))
        vmax = 0.0
        for i in w6:
            jj = w6[(T[w6] > T[i] + 0.3) & (T[w6] <= T[i] + 0.6)]
            if len(jj):
                vmax = max(vmax, float(np.hypot(X[jj[0]] - X[i], Y[jj[0]] - Y[i]) / (T[jj[0]] - T[i])))
        sc = int(g["scorer"]) if g["scorer"] != "" else None
        cj = np.flatnonzero(m & (T <= g["t"] + 0.5) & (T >= g["t"] - 15) & (C == sc) & ~np.isnan(X)) if sc is not None else []
        if len(cj):
            i = cj[-1]
            ctrl = f"{g['t'] - T[i]:4.1f} s before, {dline(X[i], side):4.1f} m from the goal line"
            cd, cb = float(dline(X[i], side)), float(g["t"] - T[i])
        else:
            ctrl, cd, cb = "none in the last 15 s", np.nan, np.nan
        jst = np.flatnonzero(m & (np.abs(T - (g["t"] - 1)) <= 0.2))
        state = S[jst[0]] if len(jst) else "?"
        print(f"{k:2d}  {mmss(g['t'])}  {side:5s}  {known:4.0%}   {'/'.join(d)}   {vmax:6.1f}   {ctrl:42s} {state}")
        rows.append({"known": known, "d": dnum, "vmax": vmax, "cd": cd, "cb": cb})

    ok = [r for r in rows if r is not None]
    n = len(ok)
    print(f"\n{len(goals)} goals, {n} with a side:")
    print(f"  ball known in at least half of the last 3 s: {sum(r['known'] >= 0.5 for r in ok)}")
    print(f"  ball seen within 12 m of the goal line in the last 3 s: {sum(np.nanmin(r['d']) <= 12 if not np.all(np.isnan(r['d'])) else False for r in ok)}")
    print(f"  ball seen to move faster than 10 m/s in the last 6 s (the shot itself): {sum(r['vmax'] >= 10 for r in ok)}")
    print(f"  a control by the scorer's team in the last 15 s: {sum(not np.isnan(r['cd']) for r in ok)}; "
          f"of these at most {a.shot_range:.0f} m from the goal line: {sum((not np.isnan(r['cd'])) and r['cd'] <= a.shot_range for r in ok)}; "
          f"and within 3 s of the goal: {sum((not np.isnan(r['cd'])) and r['cd'] <= a.shot_range and r['cb'] <= 3 for r in ok)}")


if __name__ == "__main__":
    main()
