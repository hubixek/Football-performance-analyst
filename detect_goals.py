#!/usr/bin/env python3
"""Goals of a match found automatically, without goal times: from the kick-offs and the ball track.

After a goal the game restarts from the centre spot, so every kick-off that is not the start of a half is a goal candidate.
What decides (measured on a match with 13 known goals: 12 of 12 kick-offs recognised by the BALL on the centre spot were
goals, 0 of 7 kick-offs recognised by the formation of the players alone were):
  goal     a kick-off recognised by the ball on the centre spot
  check    only with --list-checks: a kick-off recognised only by the formation, but with the ball seen at a goal before it (at least
           --check-frames frames). On two matches 0 of 13 such kick-offs judged in the video were goals, so they are not listed by default
  no       any other kick-off recognised only by the formation (a free kick, a corner, a setup)
  goal     also a kick-off recognised only by the formation, when the detector saw the ball on the centre spot in at least --raw-frames frames
           AND the ball was seen at a goal before it (valid evidence: at least --min-frames frames, at most --max-evidence-gap s before; without it
           a second formation 30 s after a kick-off, e.g. the players waiting, would be a goal)
           (the raw candidates of detections.csv within --raw-radius m of the centre spot, from 3 s before the first to 1 s after the last frame of the event; or in at
           least --raw-wide-frames frames from 8 s before to 2 s after the time of the event): the ball
           tracker needs an unbroken still ball and can lose it. Measured on two matches: the two formation-only kick-offs that were goals had 9 and
           0 frames, the other 13 had 0-3, random moments of the matches at least 6 frames in 1-2% of the cases
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

A goal just before the final whistle of a half has no kick-off after it: it is looked for separately, in the ball track after the last kick-off of
every half (needs --match for the end of the half): the ball seen in the net (behind a goal line) is listed as a possible goal to check in the video.

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


def same_restart(prev, e, g_t, window):
    """A kick-off recognised by the ball that comes less than `window` seconds after the previous kick-off (or start) of the same half, with no ball
    seen at a goal in between, is the SAME restart seen twice (the teams wait, the formation breaks and forms again), not a new goal. Two real goals
    43 s apart (37:21 and 38:04 of a match) each had the ball at a goal before them."""
    return (g_t is None and prev is not None and prev["half"] == e["half"] and prev["cat"] in ("goal", "start of the half")
            and 0 <= e["t"] - prev["t"] < window)


def mark_starts(merged, half_start, start_s, start_window):
    """The first kick-off of a half is its start (when it is within start_s of the start of the half; without a start of the half every first one
    is). Players can line up for the kick-off a few dozen seconds before the whistle, which gives two events at the beginning: every event less
    than start_window seconds after the first one is the start of the half too (the half times found by auto_halves.py begin when the players
    are on the pitch, up to a minute before the whistle)."""
    first_t = {}
    for e in merged:
        first_t.setdefault(e["half"], e["t"])
        h = e["half"]
        within = (first_t[h] - half_start[h] < start_s) if h in half_start else True
        e["start"] = within and (e["t"] - first_t[h] < start_window or e["t"] == first_t[h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder: teams.csv, ball.csv, stats.json")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--match", help="match config (half times); without it the first kick-off of every half is the start")
    ap.add_argument("--merge-s", type=float, default=15.0, help="kick-offs closer than this are one event")
    ap.add_argument("--same-restart-s", type=float, default=45.0, help="a ball kick-off this soon after the previous one, with no ball at a goal before it, is the same restart")
    ap.add_argument("--start-window", type=float, default=60.0, help="a kick-off this soon after the first one of a half is its start too (the players line up first)")
    ap.add_argument("--start-s", type=float, default=90.0, help="the first kick-off of a half this soon after its start is the start of the half")
    ap.add_argument("--look-back", type=float, default=90.0, help="how far before the kick-off to look for the ball at the goal, s")
    ap.add_argument("--goal-zone", type=float, default=3.0, help="the ball this close to a goal line (m) ...")
    ap.add_argument("--goal-width", type=float, default=5.0, help="... and within the goal width plus 1.5 m on each side is 'at the goal'")
    ap.add_argument("--stay-gap", type=float, default=3.0, help="sightings at the goal closer than this (s) are one stay")
    ap.add_argument("--net-margin", type=float, default=0.5, help="the ball this far behind the goal line (m) is in the net")
    ap.add_argument("--net-max-early", type=float, default=8.0, help="the time from the net may be at most this much earlier than the start of the last stay near the goal, s")
    ap.add_argument("--net-gap", type=float, default=6.0, help="sightings in the net closer than this (s) are one stay (the ball is often not seen in the net)")
    ap.add_argument("--raw-frames", type=int, default=6, help="a formation-only kick-off with this many frames of raw ball candidates on the centre spot is a goal (0 = off)")
    ap.add_argument("--raw-wide-frames", type=int, default=10, help="the same with the wide window (8 s before to 2 s after the time of the event): the ball is put on the spot "
                    "before the formation is complete, so the time of a formation event can be 5 s after the ball (0 = off)")
    ap.add_argument("--raw-radius", type=float, default=1.5, help="distance from the centre spot of those raw ball candidates, m")
    ap.add_argument("--list-checks", action="store_true", help="also list the kick-offs recognised only by the formation that have a ball at a goal before them (0 of 13 were goals)")
    ap.add_argument("--check-frames", type=int, default=10, help="a formation-only kick-off needs this many ball frames at a goal to be worth a check")
    ap.add_argument("--add-goals", default="", help="goals the program missed, known from the video: MM:SS=TEAM, comma separated, e.g. 33:10=1,41:00=0 "
                    "(TEAM 0 = the darker kit, 1 = the lighter); they are added as goals of the match")
    ap.add_argument("--late-frames", type=int, default=4, help="a stay in the net after the last kick-off of a half needs this many ball frames to be listed")
    ap.add_argument("--rough-s", type=float, default=15.0, help="goal time without a ball at the goal: kick-off minus this")
    ap.add_argument("--min-frames", type=int, default=3, help="a ball at the goal seen in fewer frames is no evidence (a chance sighting)")
    ap.add_argument("--max-evidence-gap", type=float, default=40.0, help="a ball at the goal more than this many seconds before the kick-off is no evidence (goals of a real match: 4-20 s)")
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

    half_start, half_end = {}, {}
    if a.match:
        cfg = json.loads(Path(a.match).expanduser().read_text())
        half_start = {i + 1: h["start_s"] for i, h in enumerate(cfg["halves"])}
        half_end = {i + 1: h["end_s"] for i, h in enumerate(cfg["halves"])}
    mark_starts(merged, half_start, a.start_s, a.start_window)

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
    ftime = {}
    for r in rows:
        ftime.setdefault(int(r["frame"]), float(r["time_s"]))
    raw_t = np.array([])
    if a.raw_frames > 0 and (folder / "detections.csv").exists():
        rt = []
        with open(folder / "detections.csv") as f:
            for r in csv.DictReader(f):
                if r["class"] == "ball" and (float(r["x_m"]) - cx) ** 2 + (float(r["y_m"]) - cy) ** 2 <= a.raw_radius ** 2:
                    rt.append(float(r["time_s"]))
        raw_t = np.unique(np.round(np.array(rt), 3))

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
        if g_t is not None and (n_near < a.min_frames or e["t"] - g_t > a.max_evidence_gap):
            g_t = g_side = None                                        # a chance sighting of the ball near a goal, not the goal
        prev_t[e["half"]] = e["t"]
        ar = attacks_right.get(f"half_{e['half']}")
        by_side = (int(ar) if g_side == "right" else 1 - int(ar)) if (g_side is not None and ar is not None) else None
        by_kicker = None if kicker is None else 1 - kicker
        # the window of the raw ball candidates: from 3 s before the first to 1 s after the last frame of the event (the formation lasts longer than the
        # moment the ball is put on the spot, so a window around the mean time of the event can miss the ball)
        ts = [ftime[f] for f in e["frames"] if f in ftime]
        t_lo, t_hi = (min(ts), max(ts)) if ts else (e["t"], e["t"])
        n_raw = int(((raw_t >= t_lo - 3.0) & (raw_t <= t_hi + 1.0)).sum()) if len(raw_t) else 0
        n_raw_wide = int(((raw_t >= e["t"] - 8.0) & (raw_t <= e["t"] + 2.0)).sum()) if len(raw_t) else 0
        if e["start"]:
            cat, by_side, by_kicker = "start of the half", None, None
        elif e["how"] == "ball":
            cat = "goal"
        elif (a.raw_frames > 0 and g_t is not None
              and (n_raw >= a.raw_frames or (a.raw_wide_frames > 0 and n_raw_wide >= a.raw_wide_frames))):       # the ball at the centre AND the ball seen at a goal before it
            cat = "goal"
            e["how"] = "ball (raw)"
        elif a.list_checks and g_t is not None and n_near >= a.check_frames:
            cat = "check"
        else:
            cat = "no"
        if cat == "goal" and e["how"] == "ball" and same_restart(out[-1] if out else None, {"half": e["half"], "t": e["t"]}, g_t, a.same_restart_s):
            cat = "no"                                                  # the same restart as the previous kick-off, seen twice
        rough = cat == "goal" and g_t is None
        if rough:
            g_t = e["t"] - a.rough_s
        scorer = by_side if by_side is not None else by_kicker
        out.append({**e, "kicker": kicker, "goal_t": g_t, "side": g_side, "n_near": n_near, "by_side": by_side, "by_kicker": by_kicker,
                    "scorer": scorer, "cat": cat, "rough": rough, "gap": None})
    for p, e in zip(out, out[1:]):
        e["gap"] = e["t"] - p["t"] if p["half"] == e["half"] else None

    for item in [x for x in a.add_goals.split(",") if x.strip()]:
        tt, team_ = item.split("=")
        m_, s_ = tt.strip().split(":")
        g = int(m_) * 60 + int(s_)
        hh = next((h for h in sorted(half_start) if half_start[h] <= g <= half_end.get(h, 1e9)), 1)
        out.append({"half": hh, "t": float(g) + a.rough_s, "frames": [], "how": "manual", "start": False, "kicker": None, "goal_t": float(g), "side": None,
                    "n_near": 0, "by_side": None, "by_kicker": None, "scorer": int(team_), "cat": "goal", "rough": False, "gap": None})
    out.sort(key=lambda e: (e["half"], e["t"]))
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
    # goals at the end of a half: no kick-off follows them
    if half_end:
        for h in sorted(half_end):
            ks = [e["t"] for e in out if e["half"] == h and e["cat"] in ("goal", "start of the half")]
            if not ks:
                continue
            sel = (bh == h) & (bt > max(ks) + 5.0) & (bt <= half_end[h] + 5.0) & (np.abs(by - cy) <= half_goal_y) & ((bx < -a.net_margin) | (bx > L + a.net_margin))
            if not sel.any():
                print(f"half {h}: no ball in the net after the last kick-off at {mmss(max(ks))} (until {mmss(half_end[h])})")
                continue
            tb = bt[np.flatnonzero(sel)]
            xs = bx[np.flatnonzero(sel)]
            start = 0
            for k in range(1, len(tb) + 1):
                if k == len(tb) or tb[k] - tb[k - 1] > a.net_gap:
                    n = k - start
                    if n >= a.late_frames:
                        side = "left" if xs[start] < cx else "right"
                        ar = attacks_right.get(f"half_{h}")
                        who = "?" if ar is None else (int(ar) if side == "right" else 1 - int(ar))
                        print(f"POSSIBLE GOAL without a kick-off after it (the end of half {h}): the ball in the net of the {side} goal from {mmss(tb[start])} "
                              f"({n} frames, until {mmss(tb[k - 1])}); if it was a goal, it was scored by team {who}")
                    start = k
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
