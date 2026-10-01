#!/usr/bin/env python3
"""Passes, shots, expected goals (xG), corners and entries into the penalty area, estimated from the ball track.

All of it comes from possession.csv (dual_possession.py): where the ball was, which team controlled it (a player of that
team within 1.5 m) and when it was out of play. Players are not identified, so an action is described by what the ball did.

  pass      the ball leaves a team's control, travels at least --pass-min-d m at --pass-min-speed m/s and is next
            controlled within 3 s: by the same team = completed, by the other team (or out of play) = not completed
  shot      the ball leaves a team's control within --shot-range m of the opponent's goal at --shot-speed m/s or more,
            heads for the goal (--goal-width m plus a margin) and is not received by a team mate
            outcome: goal (a goal time within a few seconds; times from --goals or restarts.json), saved (the other team gets
            the ball within 3.5 m of the goal line), blocked (further away), on target / off target (from the aim)
  xG        the chance that a shot becomes a goal from the position it was taken from, a logistic model of the distance
            and of the angle under which the goal is seen (anchored at about 40% at 6 m, 12% at 12 m, 4% at 20 m for a
            5 m wide goal; these anchors are an assumption for a 6v6 pitch, not a fit to data)
  corner    the ball leaves behind the goal line and the defending team touched it last
  box       the team has the ball and it enters the opponent's penalty area

Passes and shots are ESTIMATES: the ball is visible only in part of the frames and is lost most easily when it flies fast.
The script therefore reports how large a share of the ball frames is known and, when goal times are given, how many of the
goals were found as shots.

Output (in the analysis folder): actions.csv, actions.json (per team, match and halves; read by match_report.py)

Usage:
  python match_actions.py ~/football/analysis/mecz1_dual --pitch pitch/pitch_6v6.json --goals 14:21,21:29
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_time(text):
    t = 0.0
    for part in str(text).split(":"):
        t = t * 60 + float(part)
    return t


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def load_possession(path):
    t, half, bx, by, ctrl, poss, inplay, state = [], [], [], [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["time_s"])); half.append(int(r["half"]))
            bx.append(float(r["ball_x"]) if r["ball_x"] != "" else np.nan)
            by.append(float(r["ball_y"]) if r["ball_y"] != "" else np.nan)
            ctrl.append(int(r["controller_team"]) if r["controller_team"] != "" else -1)
            poss.append(int(r["possession_team"]) if r["possession_team"] != "" else -1)
            inplay.append(int(r["ball_in_play"]) if r["ball_in_play"] != "" else -1)
            state.append(r.get("ball_state", ""))
    return (np.array(t), np.array(half), np.array(bx), np.array(by), np.array(ctrl), np.array(poss), np.array(inplay), np.array(state))


def xg_model(xn, yn, L, W, gw):
    """Chance of a goal from (xn, yn) in coordinates in which the team attacks the goal at x = L."""
    d = np.hypot(L - xn, yn - W / 2)
    a1 = np.arctan2(yn - (W / 2 - gw / 2), max(L - xn, 0.3))
    a2 = np.arctan2(yn - (W / 2 + gw / 2), max(L - xn, 0.3))
    theta = abs(a1 - a2)
    z = -1.87 - 0.0975 * d + 2.565 * theta
    return float(1 / (1 + np.exp(-z)))


def ball_speed(t, bx, by, half_window=4):
    """Speed of the ball (m/s): the slope of a straight line fitted to the known positions in a window of about 0.6 s.
    The positions shake by about half a metre, so the difference of two neighbouring frames would give errors of several m/s."""
    n = len(t)
    v = np.full(n, np.nan)
    ok = ~np.isnan(bx)
    for i in range(n):
        lo, hi = max(i - half_window, 0), min(i + half_window + 1, n)
        idx = np.flatnonzero(ok[lo:hi]) + lo
        if len(idx) < 5:
            continue
        tt = t[idx] - t[idx].mean()
        den = float((tt ** 2).sum())
        if den <= 0:
            continue
        v[i] = float(np.hypot((tt * (bx[idx] - bx[idx].mean())).sum() / den, (tt * (by[idx] - by[idx].mean())).sum() / den))
    return v


def control_segments(t, half, ctrl, merge_gap=0.4, min_s=0.2):
    """Runs of frames in which the same team controls the ball (short gaps are bridged, a touch shorter than min_s is dropped)."""
    segs, i, n = [], 0, len(t)
    while i < n:
        if ctrl[i] in (0, 1):
            team, last, k = ctrl[i], i, i + 1
            while k < n and half[k] == half[i]:
                if ctrl[k] == team:
                    last = k
                elif ctrl[k] in (0, 1):
                    break
                elif t[k] - t[last] > merge_gap:
                    break
                k += 1
            if t[last] - t[i] >= min_s:
                segs.append({"team": int(team), "i0": i, "i1": last, "half": int(half[i])})
            i = last + 1
        else:
            i += 1
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder (possession.csv, stats.json)")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--goals", help="times of the goals in the recording, e.g. 14:21,21:29 (else goals_auto.csv of detect_goals.py, else restarts.json)")
    ap.add_argument("--goal-width", type=float, default=5.0, help="width of the goal in metres (the xG and 'on target' depend on it)")
    ap.add_argument("--shot-speed", type=float, default=6.0, help="slowest ball speed of a shot, m/s (goals of a real match: 5-30 m/s)")
    ap.add_argument("--shot-range", type=float, default=32.0, help="farthest a shot is taken from the goal line, m (goals of a real match were scored from up to 34 m)")
    ap.add_argument("--pass-min-d", type=float, default=4.0, help="shortest pass, m")
    ap.add_argument("--pass-min-speed", type=float, default=5.0, help="slowest pass, m/s")
    ap.add_argument("--debug", action="store_true", help="print why the possible passes were not counted (for finding what limits the numbers)")
    ap.add_argument("--fly-speed", type=float, default=7.0, help="a ball faster than this is not controlled by a player next to it, m/s")
    ap.add_argument("--big-chance", type=float, default=0.25, help="xG from which a shot is a big chance")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    pen_d, pen_w, gw = pitch.get("penalty_area_depth_m", 9.0), pitch.get("penalty_area_width_m", 20.0), a.goal_width
    t, half, bx, by, ctrl, poss, inplay, state = load_possession(folder / "possession.csv")
    right = json.loads((folder / "stats.json").read_text()).get("attacks_right", {}) if (folder / "stats.json").exists() else {}
    if not right:
        raise SystemExit("stats.json with 'attacks_right' is needed (run match_stats.py first)")
    dt = float(np.median(np.diff(t[:2000])))

    def attacks_right(team, h):
        return right.get(f"half_{h}") == team

    def norm(team, h, x, y):
        return (x, y) if attacks_right(team, h) else (L - x, W - y)

    def half_at(g):
        return int(half[min(int(np.searchsorted(t, g)), len(t) - 1)])

    goal_info, goals_source = [], "none"
    if a.goals:
        goal_info = [{"t": g, "half": half_at(g), "scorer": None} for g in sorted(parse_time(x) for x in a.goals.split(",") if x.strip())]
        goals_source = "--goals"
    elif (folder / "goals_auto.csv").exists():
        with open(folder / "goals_auto.csv") as f:
            for r in csv.DictReader(f):
                if r.get("category") == "goal" and r.get("goal_s"):
                    goal_info.append({"t": float(r["goal_s"]), "half": int(r["half"]), "scorer": int(r["scorer"]) if r.get("scorer", "") != "" else None})
        goal_info.sort(key=lambda x: x["t"])
        goals_source = "goals_auto.csv (detect_goals.py)"
    elif (folder / "restarts.json").exists():
        goal_info = [{"t": w["start"] + 2.0, "half": half_at(w["start"] + 2.0), "scorer": None}
                     for w in sorted(json.loads((folder / "restarts.json").read_text()), key=lambda w: w["start"]) if w.get("type") == "goal"]
        goals_source = "restarts.json"
    goal_times = [g["t"] for g in goal_info]

    known = ~np.isnan(bx)
    speed_now = ball_speed(t, bx, by)
    ctrl_eff = np.where(speed_now > a.fly_speed, -1, ctrl)             # a ball that flies past a player is not controlled by him
    segs = control_segments(t, half, ctrl_eff)

    # ------------------------------------------------------------------ shots
    shots, shot_segments = [], set()
    for si, s in enumerate(segs):
        i1, team, h = s["i1"], s["team"], s["half"]
        p0 = np.array([bx[i1], by[i1]])
        x0n, y0n = norm(team, h, *p0)
        if L - x0n > a.shot_range or L - x0n < 0:
            continue
        tr = t[i1]
        nxt = segs[si + 1] if si + 1 < len(segs) and segs[si + 1]["half"] == h else None
        j = next((j for j in range(i1 + 1, min(i1 + int(1.0 / dt) + 2, len(t)))
                  if half[j] == h and known[j] and t[j] - tr >= 0.1 and np.hypot(bx[j] - p0[0], by[j] - p0[1]) >= 2.5), None)
        if j is not None:                                              # the flight was seen: its speed and direction
            p1 = np.array([bx[j], by[j]])
            speed = float(np.hypot(*(p1 - p0)) / (t[j] - tr))
            min_speed = a.shot_speed
        else:                                                          # the ball was lost in the air: from where the action ends
            end, at_goal = None, False
            if nxt is not None and 0 < t[nxt["i0"]] - tr <= 5.0:
                end = (nxt["i0"], np.array([bx[nxt["i0"]], by[nxt["i0"]]]))
                xe, _ = norm(team, h, bx[nxt["i0"]], by[nxt["i0"]])
                at_goal = nxt["team"] != team and L - xe <= 3.5           # the other team (the goalkeeper) has it at the goal line
            if end is None or not at_goal:
                k = next((k for k in range(i1 + 1, min(i1 + int(5.0 / dt) + 1, len(t)))
                          if half[k] == h and state[k] == "out" and known[k]), None)
                if k is not None and (nxt is None or nxt["i0"] > k):
                    xe, _ = norm(team, h, bx[k], by[k])
                    if xe > L - 1.0:                                       # it left behind the opponent's goal line
                        end, at_goal = (k, np.array([bx[k], by[k]])), True
                    elif end is None and t[k] - tr <= 3.5:
                        end = (k, np.array([bx[k], by[k]]))
            if end is None:
                continue
            p1 = end[1]
            speed = float(np.hypot(*(p1 - p0)) / max(t[end[0]] - tr, dt))
            min_speed = 0.0 if at_goal else a.shot_speed * 0.8            # ending at the goal line: no speed needed (the ball was not seen)
        dist = float(np.hypot(*(p1 - p0)))
        if dist < 2.5:
            continue
        x1n, y1n = norm(team, h, *p1)
        dxn, dyn = (x1n - x0n) / dist, (y1n - y0n) / dist
        if speed < min_speed or dxn < 0.3:
            continue
        run = (L - x0n) / dxn                                          # distance along the ball's path to the goal line
        if min_speed > 0 and run / speed > 3.0:
            continue
        y_int = y0n + run * dyn
        if abs(y_int - W / 2) > gw / 2 + 4.0:
            continue
        nxt = segs[si + 1] if si + 1 < len(segs) and segs[si + 1]["half"] == h else None
        after = (t[nxt["i0"]] - tr) if nxt is not None else None
        if nxt is not None and nxt["team"] == team and after <= 2.5:
            continue                                                   # a team mate received it: a pass, not a shot
        aim_on_target = abs(y_int - W / 2) <= gw / 2
        outcome = "on_target" if aim_on_target else "off_target"
        if nxt is not None and nxt["team"] != team and after <= 3.0:
            xr, yr = norm(team, h, bx[nxt["i0"]], by[nxt["i0"]])
            outcome = "saved" if (L - xr <= 3.5 and aim_on_target) else "blocked"
        goal = aim_on_target and any(-2.0 <= g - tr <= 9.0 for g in goal_times)
        if goal:
            outcome = "goal"
        shots.append({"si": si, "team": team, "half": h, "t": tr, "x": float(p0[0]), "y": float(p0[1]), "xn": float(x0n), "yn": float(y0n),
                      "speed": float(speed), "outcome": outcome, "inferred": False, "xg": xg_model(x0n, y0n, L, W, gw)})
    no_origin = []
    for gi in goal_info:                                               # a goal is always a shot: add the goals no shot was found for
        g, sc, h = gi["t"], gi["scorer"], gi["half"]
        cands = [sh for sh in shots if not sh.get("inferred") and -2.0 <= g - sh["t"] <= 9.0 and (sc is None or sh["team"] == sc)]
        if cands:                                                      # the ball track found this shot already: it is the goal
            best = max(cands, key=lambda sh: sh["t"])
            best["outcome"] = "goal"
            continue
        origin = None
        if sc is not None:
            # (a) the last control by the scoring team in the 15 s before the goal, in its attacking half
            for i in np.flatnonzero((half == h) & (t >= g - 15) & (t <= g - 0.3) & (ctrl == sc) & known)[::-1]:
                xn_, _ = norm(sc, h, bx[i], by[i])
                if xn_ >= L / 2:
                    origin = (int(i), sc, "control")
                    break
            # (b) else the last ball position 1.5-8 s before the goal that is in the attacking half and not at the goal
            if origin is None:
                for i in np.flatnonzero((half == h) & (t >= g - 8) & (t <= g - 1.5) & known)[::-1]:
                    xn_, _ = norm(sc, h, bx[i], by[i])
                    if xn_ >= L / 2 and L - xn_ >= 3.0:
                        origin = (int(i), sc, "ball")
                        break
        else:                                                          # no scorer known: the last control of any team in its attacking half
            for sg in reversed([sg for sg in segs if 0 < g - t[sg["i1"]] <= 10.0]):
                xn_, _ = norm(sg["team"], sg["half"], bx[sg["i1"]], by[sg["i1"]])
                if xn_ >= L / 2 and not np.isnan(xn_):
                    origin = (sg["i1"], sg["team"], "control")
                    break
        if origin is None:
            no_origin.append(g)
            continue
        i, team, src = origin
        xg0, yg0 = norm(team, h, bx[i], by[i])
        shots.append({"si": -1, "team": int(team), "half": h, "t": float(t[i]), "x": float(bx[i]), "y": float(by[i]),
                      "xn": float(xg0), "yn": float(yg0), "speed": float("nan"), "outcome": "goal", "inferred": True, "origin": src,
                      "xg": xg_model(xg0, yg0, L, W, gw)})
    dedup = []
    for sh in sorted(shots, key=lambda x: x["t"]):
        if not any(o["team"] == sh["team"] and sh["t"] - o["t"] < 1.5 for o in dedup):
            dedup.append(sh)
    shots = dedup
    shot_segments = {sh["si"] for sh in shots}

    # ------------------------------------------------------------------ passes
    passes = []
    why = {"no next control segment": 0, "next control later than 3 s": 0, "shorter than --pass-min-d": 0, "slower than --pass-min-speed": 0,
           "counted as a pass": 0, "left the pitch (out) without a pass": 0}
    for si, s in enumerate(segs):
        if si in shot_segments:
            continue
        i1, team, h = s["i1"], s["team"], s["half"]
        p0 = np.array([bx[i1], by[i1]])
        nxt = segs[si + 1] if si + 1 < len(segs) and segs[si + 1]["half"] == h else None
        g = (t[nxt["i0"]] - t[i1]) if nxt is not None else None
        p1, completed = None, None
        if nxt is not None and 0 < g <= 3.0:
            p1 = np.array([bx[nxt["i0"]], by[nxt["i0"]]]); completed = nxt["team"] == team
        else:                                                          # the ball left the pitch after the pass?
            end = min(i1 + int(3.0 / dt) + 1, len(t))
            j = next((j for j in range(i1 + 1, end) if half[j] == h and state[j] == "out" and known[j]), None)
            if j is not None and (nxt is None or nxt["i0"] > j):
                p1, g, completed = np.array([bx[j], by[j]]), t[j] - t[i1], False
        if p1 is None:
            why["no next control segment" if nxt is None else "next control later than 3 s"] += 1
            continue
        d = float(np.hypot(*(p1 - p0)))
        if d < a.pass_min_d:
            why["shorter than --pass-min-d"] += 1
            continue
        if d / max(g, dt) < a.pass_min_speed:
            why["slower than --pass-min-speed"] += 1
            continue
        why["counted as a pass"] += 1
        x0n, _ = norm(team, h, *p0)
        x1n, _ = norm(team, h, *p1)
        passes.append({"team": team, "half": h, "t": t[i1], "x": float(p0[0]), "y": float(p0[1]), "x2": float(p1[0]), "y2": float(p1[1]),
                       "completed": bool(completed), "forward": bool(x1n - x0n >= 3.0), "opp_half": bool(x0n >= L / 2)})

    # ------------------------------------------------------------------ corners
    corners = []
    i = 0
    while i < len(t):
        if state[i] == "out" and (i == 0 or state[i - 1] != "out"):
            j = i
            while j < len(t) and state[j] == "out" and half[j] == half[i]:
                j += 1
            xs, ys = bx[i:j], by[i:j]
            m = ~np.isnan(xs)
            if m.sum() >= 1:
                viol = np.stack([-xs[m], xs[m] - L, -ys[m], ys[m] - W])
                side = int(np.argmax(viol.max(axis=1)))                 # 0 left goal line, 1 right goal line, 2/3 side lines
                if side in (0, 1):
                    h = int(half[i])
                    last = next((int(poss[k]) for k in range(i - 1, max(i - int(4.0 / dt), 0), -1) if poss[k] in (0, 1) or ctrl[k] in (0, 1)), None)
                    last = None if last is None else (last if last in (0, 1) else int(ctrl[max(i - 1, 0)]))
                    if last in (0, 1):
                        own_goal_x = 0 if attacks_right(last, h) else L
                        if (side == 0 and own_goal_x == 0) or (side == 1 and own_goal_x == L):
                            corners.append({"team": 1 - last, "half": h, "t": float(t[i])})
            i = j
        else:
            i += 1

    # ------------------------------------------------------------------ entries into the penalty area
    boxes = []
    for team in (0, 1):
        last_in = -1e9
        for i in range(len(t)):
            if poss[i] != team or not known[i]:
                continue
            xn, yn = norm(team, int(half[i]), bx[i], by[i])
            if xn >= L - pen_d and abs(yn - W / 2) <= pen_w / 2:
                if t[i] - last_in > 2.0:
                    boxes.append({"team": team, "half": int(half[i]), "t": float(t[i])})
                last_in = t[i]

    # ------------------------------------------------------------------ totals
    def totals(sel_half):
        out = {}
        for team in (0, 1):
            ps = [p for p in passes if p["team"] == team and (sel_half is None or p["half"] == sel_half)]
            sh = [s for s in shots if s["team"] == team and (sel_half is None or s["half"] == sel_half)]
            done = [p for p in ps if p["completed"]]
            out[f"team_{team}"] = {
                "passes_attempted": len(ps), "passes_completed": len(done),
                "pass_accuracy_percent": round(100 * len(done) / len(ps), 1) if ps else None,
                "passes_forward": sum(p["forward"] for p in done), "passes_in_opponent_half": sum(p["opp_half"] for p in ps),
                "shots": len(sh), "shots_on_target": sum(s["outcome"] in ("goal", "saved", "on_target") for s in sh),
                "shots_off_target": sum(s["outcome"] == "off_target" for s in sh), "shots_blocked": sum(s["outcome"] == "blocked" for s in sh),
                "goals_from_shots": sum(s["outcome"] == "goal" for s in sh),
                "xg": round(sum(s["xg"] for s in sh), 2), "big_chances": sum(s["xg"] >= a.big_chance for s in sh),
                "corners": sum(1 for c in corners if c["team"] == team and (sel_half is None or c["half"] == sel_half)),
                "box_entries": sum(1 for b in boxes if b["team"] == team and (sel_half is None or b["half"] == sel_half)),
            }
        return out
    halves = sorted({int(h) for h in half})
    result = {"match": totals(None)}
    for h in halves:
        result[f"half_{h}"] = totals(h)
    quality = {"ball_visible_percent": round(100 * float(known.mean()), 1), "goal_width_m": gw, "goals_given": len(goal_times)}
    matched = None
    if goal_times:
        matched = [any(-2.0 <= g - s["t"] <= 9.0 and s["outcome"] in ("goal", "on_target", "saved") and not s.get("inferred") for s in shots) for g in goal_times]
        quality["goals_found_as_shots"] = int(sum(matched))
        quality["goals_found_percent"] = round(100 * sum(matched) / len(goal_times), 1)
    (folder / "actions.json").write_text(json.dumps({"scopes": result, "quality": quality, "params": vars(a)}, indent=1))
    with open(folder / "actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "time_s", "time", "half", "team", "x_from", "y_from", "x_to", "y_to", "outcome", "xg", "speed_ms"])
        for p in passes:
            w.writerow(["pass", round(p["t"], 2), mmss(p["t"]), p["half"], p["team"], round(p["x"], 1), round(p["y"], 1), round(p["x2"], 1), round(p["y2"], 1),
                        "completed" if p["completed"] else "lost", "", ""])
        for s in shots:
            w.writerow(["shot", round(s["t"], 2), mmss(s["t"]), s["half"], s["team"], round(s["x"], 1), round(s["y"], 1), "", "",
                        s["outcome"] + (" (inferred from the goal)" if s.get("inferred") else ""), round(s["xg"], 3),
                        "" if s["speed"] != s["speed"] else round(s["speed"], 1)])
        for c in corners:
            w.writerow(["corner", round(c["t"], 2), mmss(c["t"]), c["half"], c["team"], "", "", "", "", "", "", ""])
        for b in boxes:
            w.writerow(["box_entry", round(b["t"], 2), mmss(b["t"]), b["half"], b["team"], "", "", "", "", "", "", ""])

    m = result["match"]
    print(f"{'':32s} {'team 0':>8s} {'team 1':>8s}")
    for label, key in (("passes attempted", "passes_attempted"), ("passes completed", "passes_completed"), ("pass accuracy [%]", "pass_accuracy_percent"),
                       ("shots", "shots"), ("shots on target", "shots_on_target"), ("shots blocked", "shots_blocked"), ("xG", "xg"),
                       ("big chances", "big_chances"), ("corners", "corners"), ("entries into the penalty area", "box_entries")):
        print(f"{label:32s} {str(m['team_0'][key]):>8s} {str(m['team_1'][key]):>8s}")
    print(f"\nball position known in {quality['ball_visible_percent']}% of the frames of the match (the numbers of passes and shots are lower bounds)")
    if a.debug:
        durs = np.array([t[sg["i1"]] - t[sg["i0"]] for sg in segs])
        gaps = np.array([t[b["i0"]] - t[c["i1"]] for c, b in zip(segs, segs[1:]) if c["half"] == b["half"]])
        ctrl_known = float(np.mean(ctrl[known] >= 0)) if known.any() else 0.0
        print(f"\n--- debug ---\nframes {len(t)} at {1 / dt:.0f} fps; ball known in {100 * known.mean():.1f}%; among those a team controls the ball in {100 * ctrl_known:.1f}%")
        print(f"frames where the ball is faster than {a.fly_speed} m/s (control dropped): {100 * float(np.nanmean(speed_now > a.fly_speed)):.1f}% of the known-speed frames")
        print(f"control segments: {len(segs)}, duration median {np.median(durs):.1f} s (10th pct {np.percentile(durs, 10):.1f} s, 90th pct {np.percentile(durs, 90):.1f} s)")
        print(f"gap between two control segments: median {np.median(gaps):.1f} s, share under 0.5 s {100 * float(np.mean(gaps < 0.5)):.0f}%, over 3 s {100 * float(np.mean(gaps > 3)):.0f}%")
        for k, v in why.items():
            print(f"  {v:6d}  {k}")
    if goal_times:
        inferred = sum(1 for sh in shots if sh.get("inferred"))
        src = Counter(sh.get("origin") for sh in shots if sh.get("inferred"))
        print(f"goals given {len(goal_times)} (from {goals_source}); found from the ball track as a shot on target: {sum(matched)} ({quality['goals_found_percent']}%); "
              f"the other {inferred} were added as shots (origin: {src['control']} from a control of the scoring team, {src['ball']} from the last ball position)"
              + (f"; {len(no_origin)} goals without any position of the shot: {', '.join(mmss(g) for g in no_origin)}" if no_origin else ""))
        miss = [mmss(g) for g, ok in zip(goal_times, matched) if not ok]
        if miss:
            print("goals without a shot found (the ball was probably not visible at the shot): " + ", ".join(miss))
        xg_sum = m["team_0"]["xg"] + m["team_1"]["xg"]
        print(f"xG of all the shots found {xg_sum:.1f} vs {len(goal_times)} goals: "
              + ("far fewer shots are found than were taken (the ball is lost when it flies fast)" if xg_sum < 0.6 * len(goal_times) else "of the same order"))
    print(f"Saved {folder / 'actions.json'} and actions.csv")


if __name__ == "__main__":
    main()
