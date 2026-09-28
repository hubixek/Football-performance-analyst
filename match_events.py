#!/usr/bin/env python3
"""Kick-offs, goals and restart windows of a match (from teams.csv and ball.csv, no video).

A kick-off from the centre spot is recognised by its formation:
  - the ball lies still within --centre-r m of the centre spot for at least --min-s seconds,
  - at least one player stands next to it (the kicker), but at most --max-crowd players are
    within --crowd-r m (in ordinary play at the centre there is a crowd around the ball),
  - at least --min-players outfield players with a team are visible.
The team of the kicker kicks off. After a goal the team that conceded kicks off, so every
kick-off that is not the start of a half is a goal for the other team.

Restart window (used by dual_possession.py --restarts): after a goal the ball is fetched from the
net and carried to the centre, then the teams take their places. Nobody has possession in that
time. The window starts when the ball reappeared after the last gap (out of the pitch or not
detected) before the kick-off - at most --carry-s seconds before it - and ends when the ball
starts moving.

Output (in the analysis folder): events.csv, restarts.json; the summary is printed.

Usage:
  python match_events.py ~/football/analysis/mecz1_dual --pitch pitch/pitch_6v6.json --expected-score 11-2
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dual_possession import out_of_play_frames


def load(folder):
    frames, players, ball = {}, defaultdict(list), {}
    with open(folder / "teams.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            frames[fr] = (float(r["time_s"]), int(r["half"]))
            if r["class"] == "player" and r["team"] in ("0", "1") and r.get("gk") != "1":
                players[fr].append((float(r["x_m"]), float(r["y_m"]), int(r["team"])))
    with open(folder / "ball.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            ball[fr] = (float(r["x_m"]), float(r["y_m"]))
            frames.setdefault(fr, (float(r["time_s"]), int(r["half"])))
    return frames, players, ball


def formation_frames(frames, players, ball, L, W, centre_r, kicker_r, crowd_r, max_crowd, min_players):
    """Frames that look like a kick-off: frame -> team of the player nearest to the ball."""
    cx, cy = L / 2, W / 2
    found = {}
    for fr, (bx, by) in ball.items():
        if fr not in frames or np.hypot(bx - cx, by - cy) > centre_r:
            continue
        pl = players.get(fr, [])
        if len(pl) < min_players:
            continue
        d = np.array([np.hypot(x - bx, y - by) for x, y, _ in pl])
        if not (d <= kicker_r).any() or int((d <= crowd_r).sum()) > max_crowd:
            continue
        found[fr] = pl[int(np.argmin(d))][2]
    return found


def episodes(frames, ball, found, min_s, gap_s=0.6, still_r=1.2, merge_s=8.0):
    """Groups of formation frames that follow each other closely and where the ball lies still."""
    order = sorted(found, key=lambda f: frames[f][0])
    groups, cur = [], []
    for fr in order:
        if cur and (frames[fr][1] != frames[cur[-1]][1] or frames[fr][0] - frames[cur[-1]][0] > gap_s):
            groups.append(cur)
            cur = []
        cur.append(fr)
    if cur:
        groups.append(cur)
    eps = []
    for g in groups:
        pts = np.array([ball[f] for f in g])
        if frames[g[-1]][0] - frames[g[0]][0] < min_s or np.max(np.hypot(*(pts - np.median(pts, axis=0)).T)) > still_r:
            continue
        team = Counter(found[f] for f in g).most_common(1)[0][0]
        eps.append({"t0": frames[g[0]][0], "t1": frames[g[-1]][0], "half": frames[g[0]][1], "team": team})
    merged = []
    for e in eps:
        if merged and merged[-1]["half"] == e["half"] and e["t0"] - merged[-1]["t1"] < merge_s:
            merged[-1]["t1"] = e["t1"]
        else:
            merged.append(dict(e))
    return merged


def reappearances(frames, ball, out_frames, min_gap_s=1.5):
    """Times at which the ball came back after a gap (not detected or out of the pitch)."""
    times = []
    known = sorted(f for f in ball if f in frames and f not in out_frames)
    for a, b in zip(known, known[1:]):
        if frames[b][0] - frames[a][0] >= min_gap_s and frames[a][1] == frames[b][1]:
            times.append(frames[b][0])
    return times


def diagnose(frames, players, ball, L, W, radius=3.0, min_s=1.0):
    """For every moment the ball lies near the centre spot: how many players are where.
    Use it to choose the thresholds of the kick-off detection."""
    cx, cy = L / 2, W / 2
    near = sorted((fr for fr, (bx, by) in ball.items() if fr in frames and np.hypot(bx - cx, by - cy) <= radius),
                  key=lambda f: frames[f][0])
    groups, cur = [], []
    for fr in near:
        if cur and (frames[fr][1] != frames[cur[-1]][1] or frames[fr][0] - frames[cur[-1]][0] > 1.0):
            groups.append(cur)
            cur = []
        cur.append(fr)
    if cur:
        groups.append(cur)
    print(f"ball within {radius} m of the centre spot for at least {min_s} s: episodes below (medians over the frames)")
    print("time   half  length  ball-still  players seen  nearest  within 2.5 m  within 5 m  within 8 m")
    n = 0
    for g in groups:
        t0, t1 = frames[g[0]][0], frames[g[-1]][0]
        if t1 - t0 < min_s:
            continue
        n += 1
        pts = np.array([ball[f] for f in g])
        spread = float(np.max(np.hypot(*(pts - np.median(pts, axis=0)).T)))
        st = []
        for f in g:
            pl = players.get(f, [])
            d = np.sort([np.hypot(x - ball[f][0], y - ball[f][1]) for x, y, _ in pl]) if pl else np.array([99.0])
            st.append((len(pl), d[0], (d <= 2.5).sum(), (d <= 5).sum(), (d <= 8).sum()))
        m = np.median(np.array(st), axis=0)
        print(f"{mmss(t0)}  {frames[g[0]][1]:>4}  {t1 - t0:5.1f}s  {spread:6.1f} m    {m[0]:8.0f}    {m[1]:6.1f} m  {m[2]:9.0f}  {m[3]:9.0f}  {m[4]:9.0f}")
    print(f"{n} episodes. A kick-off: the ball still (spread about 1 m or less), a nearest player within 2.5 m, few players within 5 m.")


def parse_time(text):
    """'14:21' -> 861 s, '1:02:03' -> 3723 s, '861' -> 861 s."""
    parts = [float(x) for x in text.strip().split(":")]
    t = 0.0
    for x in parts:
        t = t * 60 + x
    return t


def centre_groups(frames, ball, L, W, radius=3.0, min_s=1.0):
    """Moments the ball lies near the centre spot: list of (t0, t1)."""
    cx, cy = L / 2, W / 2
    near = sorted((fr for fr, (bx, by) in ball.items() if fr in frames and np.hypot(bx - cx, by - cy) <= radius),
                  key=lambda f: frames[f][0])
    groups, cur = [], []
    for fr in near:
        if cur and (frames[fr][1] != frames[cur[-1]][1] or frames[fr][0] - frames[cur[-1]][0] > 1.0):
            groups.append(cur)
            cur = []
        cur.append(fr)
    if cur:
        groups.append(cur)
    return [(frames[g[0]][0], frames[g[-1]][0]) for g in groups if frames[g[-1]][0] - frames[g[0]][0] >= min_s]


def windows_from_goals(goal_times, centre, before_s=2.0, min_delay_s=3.0, max_delay_s=90.0, default_s=40.0):
    """Restart window of every goal: from just before the goal until the ball is kicked off from
    the centre (the first moment the ball lies at the centre spot after the goal), else default_s."""
    windows = []
    for g in goal_times:
        nxt = [(t0, t1) for t0, t1 in centre if g + min_delay_s <= t0 <= g + max_delay_s]
        if nxt:
            windows.append({"start": g - before_s, "end": nxt[0][1], "type": "goal", "kickoff_found": True})
        else:
            windows.append({"start": g - before_s, "end": g + default_s, "type": "goal", "kickoff_found": False})
    return windows


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder of a match (teams.csv, ball.csv)")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--centre-r", type=float, default=2.0, help="ball this close to the centre spot, m")
    ap.add_argument("--kicker-r", type=float, default=2.5, help="the kicker stands this close to the ball, m")
    ap.add_argument("--crowd-r", type=float, default=5.0)
    ap.add_argument("--max-crowd", type=int, default=3, help="most players within --crowd-r of the ball")
    ap.add_argument("--min-players", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=1.5, help="the ball must lie still this long, s")
    ap.add_argument("--carry-s", type=float, default=25.0, help="longest carry of the ball to the centre, s")
    ap.add_argument("--half-start-s", type=float, default=60.0,
                    help="a kick-off this soon after the start of a half is the start of the half")
    ap.add_argument("--goals", help="times of the goals in the recording, e.g. 14:21,21:29,38:04 (as in a video "
                    "player): restart windows are made from them, the automatic kick-off detection is not used")
    ap.add_argument("--diagnose", action="store_true",
                    help="print the player distances at every moment the ball lies near the centre, then stop")
    ap.add_argument("--expected-goals", type=int, help="number of goals in the match (to check)")
    ap.add_argument("--expected-score", help="final score, e.g. 11-2 (to check)")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    frames, players, ball = load(folder)
    if a.diagnose:
        diagnose(frames, players, ball, L, W)
        return
    if a.goals:
        goal_times = sorted(parse_time(x) for x in a.goals.split(",") if x.strip())
        windows = windows_from_goals(goal_times, centre_groups(frames, ball, L, W))
        (folder / "restarts.json").write_text(json.dumps(windows, indent=1))
        print(f"{len(goal_times)} goals given")
        print("goal     kick-off found (ball back at the centre)    window without possession")
        for g, w in zip(goal_times, windows):
            found_txt = f"{mmss(w['end'])} ({w['end'] - g:3.0f} s after the goal)" if w["kickoff_found"] else "not found - fixed 40 s window"
            print(f"{mmss(g)}    {found_txt:<42} {w['end'] - w['start']:4.0f} s")
        tot = sum(w["end"] - w["start"] for w in windows)
        print(f"kick-offs found for {sum(w['kickoff_found'] for w in windows)} of {len(windows)} goals; "
              f"{tot / 60:.1f} min in total without possession")
        print(f"Saved {folder / 'restarts.json'}")
        return
    found = formation_frames(frames, players, ball, L, W, a.centre_r, a.kicker_r, a.crowd_r, a.max_crowd, a.min_players)
    eps = episodes(frames, ball, found, a.min_s)

    half_first = {}
    for fr, (t, h) in frames.items():
        half_first[h] = min(half_first.get(h, t), t)
    out_frames = out_of_play_frames(frames, ball, L, W, 1.0, 1.0)
    back = reappearances(frames, ball, out_frames)

    events, windows = [], []
    seen_start = set()
    for e in eps:
        is_start = e["half"] not in seen_start and e["t0"] - half_first[e["half"]] < a.half_start_s
        if is_start:
            seen_start.add(e["half"])
            events.append({**e, "type": "half_start"})
            windows.append({"start": e["t0"], "end": e["t1"], "type": "half_start"})
        else:
            events.append({**e, "type": "goal"})
            prior = [t for t in back if e["t0"] - a.carry_s <= t <= e["t0"]]
            windows.append({"start": min(prior) if prior else e["t0"], "end": e["t1"], "type": "goal"})

    goals = [e for e in events if e["type"] == "goal"]
    scored = {0: sum(1 for e in goals if e["team"] == 1), 1: sum(1 for e in goals if e["team"] == 0)}
    with open(folder / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "time", "half", "type", "kicking_team", "scored_by"])
        for e in events:
            w.writerow([round(e["t0"], 1), mmss(e["t0"]), e["half"], e["type"], e["team"],
                        "" if e["type"] != "goal" else 1 - e["team"]])
    (folder / "restarts.json").write_text(json.dumps(windows, indent=1))

    print(f"formation frames: {len(found)} | kick-off episodes: {len(events)} "
          f"({sum(e['type'] == 'half_start' for e in events)} half starts, {len(goals)} after a goal)")
    print("time   half  type        kicks off  window before the kick-off")
    for e, wdw in zip(events, windows):
        print(f"{mmss(e['t0'])}  {e['half']:>4}  {e['type']:<10}  team {e['team']}     {e['t0'] - wdw['start']:4.0f} s")
    print(f"\nestimated goals: team 0 scored {scored[0]}, team 1 scored {scored[1]} (total {len(goals)})")
    if a.expected_goals is not None:
        print(f"expected total {a.expected_goals}: {'OK' if len(goals) == a.expected_goals else 'DIFFERENT'} "
              f"(found {len(goals)}, kick-offs incl. half starts {len(events)} of {a.expected_goals + 2})")
    if a.expected_score:
        x, y = (int(v) for v in a.expected_score.split("-"))
        same = sorted([scored[0], scored[1]]) == sorted([x, y])
        print(f"expected score {x}-{y}: {'OK' if same else 'DIFFERENT'} (estimated {scored[0]}-{scored[1]})")
    print(f"Saved {folder / 'events.csv'}, restarts.json")


if __name__ == "__main__":
    main()
