#!/usr/bin/env python3
"""Ball possession per team from positions on the pitch (dual-lens analysis).

For every frame with a ball position the player closest to the ball controls it if he is
within --control m. Possession changes only after the other team has controlled the ball
for --switch-s seconds (duels, deflections); while the ball travels (passes) it stays with
the last team. Frames without a ball for more than --reset-s seconds, or with a player of
unknown team, count for nobody.

Input:  teams.csv (dual_teams.py) + ball.csv (dual_ball.py)
Output: possession.csv (per frame), summary.json, possession.png (timeline and where on the
        pitch each team had the ball)

Usage:
  python dual_possession.py ~/football/analysis/mecz1_dual/teams.csv \
      --ball ~/football/analysis/mecz1_dual/ball.csv --out-dir ~/football/analysis/mecz1_dual \
      --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(teams_path, ball_path):
    players = defaultdict(list)            # frame -> [(x, y, team)]
    frames = {}
    with open(teams_path) as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            frames[fr] = (float(r["time_s"]), int(r["half"]))
            if r["class"] == "player" and r["team"] != "":
                players[fr].append((float(r["x_m"]), float(r["y_m"]), int(r["team"])))
    ball = {}
    with open(ball_path) as f:
        for r in csv.DictReader(f):
            ball[int(r["frame"])] = (float(r["x_m"]), float(r["y_m"]))
    return frames, players, ball


def out_of_play_frames(frames, ball, L, W, margin, min_s):
    """Frames in which the ball is out of play: it left the pitch (more than `margin` m outside
    the lines) and did not come back for at least min_s seconds. Frames without a ball position
    between leaving and returning count as out of play too. Short excursions over the line are
    position noise (calibration error near the edges) and are ignored."""
    order = sorted(frames)

    def outside(fr):
        if fr not in ball:
            return False
        x, y = ball[fr]
        return not (-margin <= x <= L + margin and -margin <= y <= W + margin)

    out, i = set(), 0
    while i < len(order):
        if not outside(order[i]):
            i += 1
            continue
        half, j = frames[order[i]][1], i
        while j < len(order) and frames[order[j]][1] == half and (order[j] not in ball or outside(order[j])):
            j += 1
        t_end = frames[order[j]][0] if j < len(order) and frames[order[j]][1] == half else frames[order[j - 1]][0]
        if t_end - frames[order[i]][0] >= min_s:
            out.update(order[i:j])
        i = j
    return out


def still_ball_frames(frames, ball, min_s, radius, max_gap_s=1.0):
    """Frames in which the ball lies still: it stays within `radius` metres of where it lay for
    at least min_s seconds (restart after a goal, free kick, corner, kick-in, ball waiting).
    Nobody has possession of a ball like that; the next possession starts when it moves."""
    known = sorted(f for f in ball if f in frames)
    still, k = set(), 0
    while k < len(known):
        anchor = np.median([ball[f] for f in known[k:k + 5]], axis=0)
        j = k
        while j + 1 < len(known):
            a, b = known[j], known[j + 1]
            if (frames[b][1] != frames[a][1] or frames[b][0] - frames[a][0] > max_gap_s
                    or np.hypot(ball[b][0] - anchor[0], ball[b][1] - anchor[1]) > radius):
                break
            j += 1
        if frames[known[j]][0] - frames[known[k]][0] >= min_s:
            still.update(known[k:j + 1])
            k = j + 1
        else:
            k += 1
    return still


def possession(frames, players, ball, control, switch_s, reset_s, L=None, W=None, out_margin=1.0, out_min_s=1.0,
               still_s=2.0, still_radius=1.0, restarts=()):
    """Rows: (frame, time, half, ball_x, ball_y, controller, possession, in_play, state).
    state: "play", "out" (ball left the pitch), "still" (ball lies still) or None (no ball position).
    in_play: True only for "play". While the ball is out or still nobody has possession; it
    restarts with the first player who gets the ball afterwards."""
    out_frames = out_of_play_frames(frames, ball, L, W, out_margin, out_min_s) if L is not None else set()
    still_frames = still_ball_frames(frames, ball, still_s, still_radius) if still_s else set()
    # restart windows after a goal (match_events.py): fetching the ball, carrying it, taking places
    restart_frames = {fr for fr, (t, _) in frames.items() if any(w["start"] <= t <= w["end"] for w in restarts)}
    rows = []
    current, candidate, cand_since, last_ball_t = None, None, None, None
    last_half = None
    for fr in sorted(frames):
        t, half = frames[fr]
        if half != last_half:
            current, candidate, last_half = None, None, half
        if fr in out_frames or fr in still_frames or fr in restart_frames:
            current, candidate = None, None
            if fr in ball:
                last_ball_t = t
                rows.append((fr, t, half, ball[fr][0], ball[fr][1], None, None, False,
                             "out" if fr in out_frames else "still" if fr in still_frames else "restart"))
            else:
                rows.append((fr, t, half, None, None, None, None, None, None))
            continue
        if fr not in ball:
            if last_ball_t is not None and t - last_ball_t > reset_s:
                current, candidate = None, None
            rows.append((fr, t, half, None, None, None, current, None, None))
            continue
        last_ball_t = t
        bx, by = ball[fr]
        controller = None
        pl = players.get(fr, [])
        if pl:
            d = [np.hypot(x - bx, y - by) for x, y, _ in pl]
            i = int(np.argmin(d))
            if d[i] <= control:
                controller = pl[i][2]
        if controller in (0, 1) and controller != current:
            if controller != candidate:
                candidate, cand_since = controller, t
            if current is None or t - cand_since >= switch_s:
                current, candidate = controller, None
        elif controller == current:
            candidate = None
        rows.append((fr, t, half, bx, by, controller, current, True, "play"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", help="CSV from dual_teams.py")
    ap.add_argument("--ball", required=True, help="CSV from dual_ball.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--control", type=float, default=1.5, help="max distance player-ball to control it, m")
    ap.add_argument("--switch-s", type=float, default=0.3)
    ap.add_argument("--reset-s", type=float, default=2.0)
    ap.add_argument("--out-margin", type=float, default=1.0,
                    help="ball this many metres outside the lines is out of play (calibration error is about 1 m)")
    ap.add_argument("--out-min-s", type=float, default=1.0,
                    help="the ball must stay outside at least this long to count as out of play (shorter = noise)")
    ap.add_argument("--still-s", type=float, default=2.0,
                    help="a ball that lies still this long is dead (restart, set piece); 0 = off")
    ap.add_argument("--still-radius", type=float, default=1.0, help="'lies still' = moves less than this, m")
    ap.add_argument("--restarts", help="restarts.json from match_events.py: no possession while the ball is fetched "
                    "and the teams take their places after a goal")
    ap.add_argument("--window-s", type=float, default=60.0, help="smoothing window of the timeline plot")
    a = ap.parse_args()

    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    frames, players, ball = load(Path(a.teams).expanduser(), Path(a.ball).expanduser())
    restarts = json.loads(Path(a.restarts).expanduser().read_text()) if a.restarts else []
    rows = possession(frames, players, ball, a.control, a.switch_s, a.reset_s, L, W, a.out_margin, a.out_min_s, a.still_s, a.still_radius, restarts)

    with open(out / "possession.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "ball_x", "ball_y", "controller_team", "possession_team", "ball_in_play", "ball_state"])
        for fr, t, h, bx, by, c, cur, ip, st in rows:
            w.writerow([fr, t, h, "" if bx is None else round(bx, 2), "" if by is None else round(by, 2),
                        "" if c is None else c, "" if cur is None else cur, "" if ip is None else int(ip), st or ""])

    def share(sel):
        n0 = sum(1 for r in sel if r[6] == 0)
        n1 = sum(1 for r in sel if r[6] == 1)
        tot = n0 + n1
        return (round(100 * n0 / tot, 1), round(100 * n1 / tot, 1)) if tot else (None, None)

    known = [r for r in rows if r[7] is not None]
    summary = {"frames": len(rows), "assigned_share": round(100 * sum(r[6] in (0, 1) for r in rows) / len(rows), 1),
               "ball_out_of_pitch_percent": round(100 * sum(1 for r in known if r[8] == "out") / max(len(known), 1), 1),
               "ball_still_percent": round(100 * sum(1 for r in known if r[8] == "still") / max(len(known), 1), 1),
               "ball_restart_percent": round(100 * sum(1 for r in known if r[8] == "restart") / max(len(known), 1), 1)}
    summary["possession_percent"] = dict(zip(("team_0", "team_1"), share(rows)))
    for h in sorted({r[2] for r in rows}):
        summary[f"half_{h}"] = dict(zip(("team_0", "team_1"), share([r for r in rows if r[2] == h])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors_file = Path(a.teams).expanduser().with_name(Path(a.teams).stem + "_colors.json")
    cols = ["tab:blue", "tab:red"]
    if colors_file.exists():
        info = json.loads(colors_file.read_text())
        cols = [tuple(v / 255 for v in info[f"team_{t}_bgr"][::-1]) for t in (0, 1)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1.6]})
    # timeline: share of team 0 in a moving window
    ts = np.array([r[1] for r in rows])
    own = np.array([1.0 if r[6] == 0 else 0.0 if r[6] == 1 else np.nan for r in rows])
    t_grid = np.arange(ts.min(), ts.max(), 5.0)
    val = [np.nanmean(own[(ts > g - a.window_s) & (ts <= g)]) * 100
           if np.any(np.isfinite(own[(ts > g - a.window_s) & (ts <= g)])) else np.nan for g in t_grid]
    ax1.plot(t_grid / 60, val, color="k", lw=1)
    ax1.fill_between(t_grid / 60, val, 50, where=np.array(val) >= 50, color=cols[0], alpha=0.6, label="team 0")
    ax1.fill_between(t_grid / 60, val, 50, where=np.array(val) < 50, color=cols[1], alpha=0.6, label="team 1")
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("time in the recording [min]")
    ax1.set_ylabel(f"team 0 possession [%]\n(last {a.window_s:.0f} s)")
    ax1.legend(loc="upper right")
    # where on the pitch the ball was, coloured by the team in possession
    for t in (0, 1):
        pts = np.array([(r[3], r[4]) for r in rows if r[6] == t and r[3] is not None])
        if len(pts):
            ax2.scatter(pts[::3, 0], pts[::3, 1], s=3, color=cols[t], alpha=0.5, label=f"team {t}")
    ax2.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k-")
    ax2.plot([L / 2, L / 2], [0, W], "k-")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.set_title("Ball position by team in possession")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "possession.png", dpi=120)
    print(f"Saved {out / 'possession.csv'}, summary.json, possession.png")


if __name__ == "__main__":
    main()
