"""Helpers of the tests: the repository on the path and a small synthetic match (players, ball, goals, kick-offs) written in the formats of the analysis."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

L, W = 56.0, 32.4
FPS = 5.0
HALF_S = 400.0                      # a half of the synthetic match
GAP = 15.0                          # the goal is this many seconds before its kick-off
WAIT = 5.0


def _formation(rng, kicker_team, attack_right_team):
    """Static positions of 12 players for a kick-off: the team that attacks to the right stands on the left half. One player of the team that kicks off is on the spot."""
    pos = {}
    ids_left = [i for i in range(12) if i % 2 == attack_right_team]
    ids_right = [i for i in range(12) if i % 2 != attack_right_team]
    kicker = next(i for i in (ids_left if kicker_team == attack_right_team else ids_right))
    for ids, x0, x1 in ((ids_left, 8.0, 25.0), (ids_right, 31.0, 48.0)):
        for i in ids:
            pos[i] = (float(rng.uniform(x0, x1)), float(rng.uniform(4.0, W - 4.0)))
    pos[kicker] = (L / 2 + 0.3, W / 2)
    return pos


def make_match(folder, goals, ball_at_centre=None, extra_formations=(), seed=0, halves=2):
    """Writes teams.csv, ball.csv, stats.json, pitch.json and match.json of a synthetic match into folder and returns the truth.

    goals: [(half, seconds from the start of the half, team that scores)]. Team 0 attacks to the right in the first half, team 1 in the second.
    ball_at_centre: {goal index: False} - the ball is NOT on the centre spot at that goal's kick-off (the formation alone), default True.
    extra_formations: [(half, seconds from the start of the half[, ball on the spot])] - a formation of the players with the ball far from the goals (waiting players).
    A goal: the ball flies into the net (3 s), stays there 2 s, the players walk back, the ball is carried to the centre, the kick-off 15 s after the goal.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ball_at_centre = ball_at_centre or {}
    attack_right = {1: 0, 2: 1}
    teams, ball = [], []
    truth = {"goals": [], "kickoffs": []}
    frame = 0
    for half in range(1, halves + 1):
        t0 = (half - 1) * (HALF_S + 200.0) + 20.0
        ar = attack_right[half]
        events = [("start", 5.0, None, True)]                         # (kind, kick-off time in the half, team that scores, ball on the spot)
        for k, (h, tg, team) in enumerate(goals):
            if h == half:
                events.append(("goal", tg + GAP, team, ball_at_centre.get(k, True)))
        for item in extra_formations:
            if item[0] == half:
                events.append(("waiting", item[1], None, bool(item[2]) if len(item) > 2 else False))
        events.sort(key=lambda e: e[1])
        pos = {i: (float(rng.uniform(5, L - 5)), float(rng.uniform(4, W - 4))) for i in range(12)}
        bpos = (L / 2, W / 2)
        n = int(HALF_S * FPS)
        for k in range(n):
            t = k / FPS
            frame_t = t0 + t
            state, form, ball_at = "play", None, None
            for kind, te, team, on_spot in events:
                if te - 1.0 <= t < te + 4.0:
                    kicker_team = (1 - team) if team is not None else 0
                    form = _formation(np.random.default_rng(int(te * 10) + half), kicker_team, ar)
                    state, ball_at = "form", ((L / 2, W / 2) if on_spot else (float(rng.uniform(10, 18)), 8.0))
                    break
                if kind == "goal" and te - GAP <= t < te - GAP + 5.0:        # the ball flies to the net and stays there
                    goes_right = (team == ar)
                    nx = L + 1.2 if goes_right else -1.2
                    ball_at = (nx, W / 2 + float(rng.normal(0, 0.5)))
                    state = "net"
                    break
                if kind == "goal" and te - GAP + 5.0 <= t < te - 1.0:        # the ball is carried to the centre
                    frac = (t - (te - GAP + 5.0)) / (GAP - 6.0)
                    goes_right = (team == ar)
                    nx = L + 1.2 if goes_right else -1.2
                    ball_at = (nx + (L / 2 - nx) * frac, W / 2)
                    state = "walk"
                    break
            if state == "form":
                for i in range(12):
                    pos[i] = (form[i][0] + float(rng.normal(0, 0.04)), form[i][1] + float(rng.normal(0, 0.04)))
            else:
                for i in range(12):
                    x = float(np.clip(pos[i][0] + rng.normal(0, 0.9), 2, L - 2))
                    y = float(np.clip(pos[i][1] + rng.normal(0, 0.9), 1, W - 1))
                    pos[i] = (x, y)
            if ball_at is None:
                bpos = (float(np.clip(bpos[0] + rng.normal(0, 1.5), 6, L - 6)), float(np.clip(bpos[1] + rng.normal(0, 1.5), 2, W - 2)))
                ball_at = bpos
            for i in range(12):
                teams.append({"frame": frame, "time_s": round(frame_t, 3), "half": half, "cam": "top", "class": "player", "conf": 0.9,
                              "x_m": round(pos[i][0], 2), "y_m": round(pos[i][1], 2), "team": i % 2, "gk": 0})
            if state in ("form", "net", "walk") or rng.random() < 0.85:
                ball.append({"frame": frame, "time_s": round(frame_t, 3), "half": half, "x_m": round(ball_at[0], 2), "y_m": round(ball_at[1], 2), "conf": 0.7, "source": "detected"})
            frame += 1
        for kind, te, team, on_spot in events:
            truth["kickoffs"].append((half, t0 + te, kind))
            if kind == "goal":
                truth["goals"].append((half, t0 + te - GAP, team))
    for name, rows in (("teams.csv", teams), ("ball.csv", ball)):
        with open(folder / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (folder / "stats.json").write_text(json.dumps({"attacks_right": {"half_1": 0, "half_2": 1}}))
    (folder / "pitch.json").write_text(json.dumps({"length_m": L, "width_m": W}))
    cfg = {"halves": [{"start_s": (h - 1) * (HALF_S + 200.0) + 15.0, "end_s": (h - 1) * (HALF_S + 200.0) + 20.0 + HALF_S + 5.0} for h in range(1, halves + 1)]}
    (folder / "match.json").write_text(json.dumps(cfg))
    return truth


def read_goals(folder):
    with open(Path(folder) / "goals_auto.csv") as f:
        return list(csv.DictReader(f))
