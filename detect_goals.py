#!/usr/bin/env python3
"""Goals of a match found automatically, without goal times: from the kick-offs and the ball track.

After a goal the game restarts from the centre spot, so every kick-off that is not the start of a half is a goal candidate.
What decides (measured on a match with 13 known goals: 12 of 12 kick-offs recognised by the BALL on the centre spot were
goals, 0 of 7 kick-offs recognised by the formation of the players alone were):
  goal     a kick-off recognised by the ball on the centre spot
  check    a kick-off recognised only by the formation, but with the ball seen at a goal before it (at least --check-frames
           frames): maybe a goal whose kick-off the ball was not seen at; look at it in the video
  no       any other kick-off recognised only by the formation (a free kick, a corner, a setup)
The first kick-off of a half is its start (not a goal). Kick-offs found twice (less than --merge-s apart) are merged.

Who scored: the goal SIDE (the goal at which the ball was last seen) and the direction of attack in that half (stats.json) give the
scorer; this matched the real score, while "the team that kicks off conceded" (nearest player to the centre spot) was wrong in 3 of 11
goals. The side is used when the ball was seen at a goal, the kicker otherwise.

Goal time: the first sighting of the ball BEHIND the goal line (in the net, more than --net-margin m beyond the line) in the last
such stay; the ball sits in the net and is picked up later, so the last sighting would be late (on a real match the ball was behind
the line 3 s before the time of "the start of the last stay near the goal"). Without a sighting in the net: the start of the last stay
near the goal. The time from the net is not accepted when it is earlier than the start of the last stay near the goal by more than
--net-max-early seconds: then the ball was lying behind the goal for a long time before (a spare ball), not a goal (real match: two
estimates would have been 34 s and 18 s early). When the ball was not seen at the goal the time is the kick-off minus --rough-s
(marked ~). The search window: at most --look-back seconds and not before the previous kick-off. The times are estimates.

Output: a table, goals_auto.csv in the analysis folder, the score and the --goals line for analyze_dual.py.

Usage:
  python detect_goals.py ~/football/analysis/mecz1_dual_v8 --pitch pitch/pitch_6v6.json --match matches/mecz1.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dual_teams import find_kickoffs


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder: teams.csv, ball.csv, stats.json")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--match", help="match config (half times); without it the first kick-off of every half is the start")
    ap.add_argument("--merge-s", type=float, default=15.0, help="kick-offs closer than this are one event")
    ap.add_argument("--start-s", type=float, default=90.0, help="the first kick-off of a half this soon after its start is the start of the half")
    ap.add_argument("--look-back", type=float, default=90.0, help="how far before the kick-off to look for the ball at the goal, s")
    ap.add_argument("--goal-zone", type=float, default=3.0, help="the ball this close to a goal line (m) ...")
    ap.add_argument("--goal-width", type=float, default=5.0, help="... and within the goal width plus 1.5 m on each side is 'at the goal'")
    ap.add_argument("--stay-gap", type=float, default=3.0, help="sightings at the goal closer than this (s) are one stay")
    ap.add_argument("--net-margin", type=float, default=0.5, help="the ball this far behind the goal line (m) is in the net")
    ap.add_argument("--net-max-early", type=float, default=8.0, help="the time from the net may be at most this much earlier than the start of the last stay near the goal, s")
    ap.add_argument("--net-gap", type=float, default=6.0, help="sightings in the net closer than this (s) are one stay (the ball is often not seen in the net)")
    ap.add_argument("--check-frames", type=int, default=10, help="a formation-only kick-off needs this many ball frames at a goal to be worth a check")
    ap.add_argument("--rough-s", type=float, default=15.0, help="goal time without a ball at the goal: kick-off minus this")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    with open(folder / "teams.csv") as f:
        rows = list(csv.DictReader(f))
    events = sorted(find_kickoffs(rows, folder / "ball.csv", L, W), key=lambda e: e[1])
    if not events:
        raise SystemExit("no kick-offs found")

    merged = []
    for half, t, frames, how in events:
        if merged and merged[-1]["half"] == half and t - merged[-1]["t"] < a.merge_s:
            m = merged[-1]
            m["frames"] += frames
            if how == "ball":
                m["how"] = "ball"
            continue
        merged.append({"half": half, "t": t, "frames": list(frames), "how": how})

    half_start = {}
    if a.match:
        cfg = json.loads(Path(a.match).expanduser().read_text())
        half_start = {i + 1: h["start_s"] for i, h in enumerate(cfg["halves"])}
    seen_half = set()
    for e in merged:
        first = e["half"] not in seen_half
        seen_half.add(e["half"])
        e["start"] = first and ((e["t"] - half_start[e["half"]] < a.start_s) if e["half"] in half_start else True)

    attacks_right = {}
    if (folder / "stats.json").exists():
        attacks_right = json.loads((folder / "stats.json").read_text()).get("attacks_right", {})

    bt, bx, by, bh = [], [], [], []
    with open(folder / "ball.csv") as f:
        for r in csv.DictReader(f):
            if r["x_m"]:
                bt.append(float(r["time_s"]))
                bx.append(float(r["x_m"]))
                by.append(float(r["y_m"]))
                bh.append(int(r["half"]))
    bt, bx, by, bh = map(np.array, (bt, bx, by, bh))

    by_frame = defaultdict(list)
    for r in rows:
        if r["class"] == "player" and r.get("team") in ("0", "1") and r.get("gk") != "1":
            by_frame[int(r["frame"])].append((float(r["x_m"]), float(r["y_m"]), int(r["team"])))
    cx, cy = L / 2, W / 2
    half_goal_y = a.goal_width / 2 + 1.5

    out, prev_t = [], {}
    for e in merged:
        votes = Counter()
        for fr in e["frames"][::3]:
            pl = by_frame.get(fr, [])
            if pl:
                d = [np.hypot(x - cx, y - cy) for x, y, _ in pl]
                k = int(np.argmin(d))
                if d[k] < 3.0:
                    votes[pl[k][2]] += 1
        kicker = votes.most_common(1)[0][0] if votes else None
        lo = max(e["t"] - a.look_back, prev_t.get(e["half"], -1e9) + 1.0)      # the ball cannot have scored before the previous restart
        sel = (bh == e["half"]) & (bt < e["t"]) & (bt >= lo)
        near = sel & (np.abs(by - cy) <= half_goal_y) & ((bx <= a.goal_zone) | (bx >= L - a.goal_zone))
        n_near, g_t, g_side, last_seen = int(near.sum()), None, None, None
        if near.any():
            idx = np.flatnonzero(near)
            tt = bt[idx]
            k = len(tt) - 1
            while k > 0 and tt[k] - tt[k - 1] <= a.stay_gap:                    # the last stay near the goal: where does it start?
                k -= 1
            g_t, last_seen = float(tt[k]), float(tt[-1])
            g_side = "left" if bx[idx[-1]] < cx else "right"
            behind = sel & (np.abs(by - cy) <= half_goal_y) & ((bx < -a.net_margin) | (bx > L + a.net_margin))
            if behind.any():                                           # the ball seen in the net: the goal is about then
                tb = bt[np.flatnonzero(behind)]
                kb = len(tb) - 1
                while kb > 0 and tb[kb] - tb[kb - 1] <= a.net_gap:
                    kb -= 1
                if float(tb[kb]) >= g_t - a.net_max_early:
                    g_t = float(tb[kb])
        prev_t[e["half"]] = e["t"]
        ar = attacks_right.get(f"half_{e['half']}")
        by_side = (int(ar) if g_side == "right" else 1 - int(ar)) if (g_side is not None and ar is not None) else None
        by_kicker = None if kicker is None else 1 - kicker
        if e["start"]:
            cat, by_side, by_kicker = "start of the half", None, None
        elif e["how"] == "ball":
            cat = "goal"
        elif g_t is not None and n_near >= a.check_frames:
            cat = "check"
        else:
            cat = "no"
        rough = cat == "goal" and g_t is None
        if rough:
            g_t = e["t"] - a.rough_s
        scorer = by_side if by_side is not None else by_kicker
        out.append({**e, "kicker": kicker, "goal_t": g_t, "side": g_side, "n_near": n_near, "by_side": by_side, "by_kicker": by_kicker,
                    "scorer": scorer, "cat": cat, "rough": rough, "gap": None})
    for p, e in zip(out, out[1:]):
        e["gap"] = e["t"] - p["t"] if p["half"] == e["half"] else None

    print(f"{len(events)} kick-offs found, {len(merged)} distinct (after merging those less than {a.merge_s:.0f} s apart)\n")
    print(" #  half  kick-off  gap     how        goal time   side   frames  scored by (side/kicker)  category")
    for k, e in enumerate(out, 1):
        gt = ("~" if e["rough"] else " ") + mmss(e["goal_t"]) if e["goal_t"] is not None else "     -"
        who = "-" if e["start"] else f"{'-' if e['by_side'] is None else e['by_side']}/{'-' if e['by_kicker'] is None else e['by_kicker']}"
        gap = f"{e['gap']:5.0f}s" if e["gap"] is not None else "     -"
        print(f"{k:2d}  {e['half']:4d}  {mmss(e['t'])}    {gap}  {e['how']:9s}  {gt:9s}   {e['side'] or '-':5s}  {e['n_near']:6d}  {who:23s}  {e['cat']}")

    goals = [e for e in out if e["cat"] == "goal"]
    checks = [e for e in out if e["cat"] == "check"]
    sc = Counter(e["scorer"] for e in goals)
    both = [e for e in goals if e["by_side"] is not None and e["by_kicker"] is not None]
    print(f"\ngoals: {len(goals)} (kick-offs recognised by the ball); to check in the video: {len(checks)}; the rest of the kick-offs are not goals")
    print(f"score (team 0 : team 1): {sc[0]} : {sc[1]}   (the side of the goal; the kicker where the ball was not seen at a goal)")
    print(f"side and kicker agree on the scorer in {sum(1 for e in both if e['by_side'] == e['by_kicker'])} of {len(both)} goals "
          f"(the side is the better of the two)")
    for e in checks:
        s = e["side"]
        print(f"CHECK {mmss(e['t'])}: ball at the {s} goal from {mmss(e['goal_t'])} ({e['n_near']} frames) - was there a goal? If yes: "
              f"{mmss(e['goal_t'])} would be the goal time, scored by team {e['by_side'] if e['by_side'] is not None else '?'}")
    with open(folder / "goals_auto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["half", "kickoff_s", "kickoff", "gap_s", "how", "category", "goal_s", "goal", "goal_time_is_rough", "side", "frames_near_goal",
                    "scored_by_side", "scored_by_kicker", "scorer"])
        for e in out:
            w.writerow([e["half"], round(e["t"], 1), mmss(e["t"]), "" if e["gap"] is None else round(e["gap"], 1), e["how"], e["cat"],
                        "" if e["goal_t"] is None else round(e["goal_t"], 1), "" if e["goal_t"] is None else mmss(e["goal_t"]),
                        int(e["rough"]), e["side"] or "", e["n_near"], "" if e["by_side"] is None else e["by_side"],
                        "" if e["by_kicker"] is None else e["by_kicker"], "" if e["scorer"] is None or e["start"] else e["scorer"]])
    if goals:
        print(f"\nfor analyze_dual.py (estimates; ~ = rough): --goals {','.join(mmss(e['goal_t']) for e in goals)}")
    print(f"Saved {folder / 'goals_auto.csv'}")


if __name__ == "__main__":
    main()
