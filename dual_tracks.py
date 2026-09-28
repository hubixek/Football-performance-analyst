#!/usr/bin/env python3
"""Player tracking in pitch coordinates (metres) for the dual-lens analysis.

Input: teams.csv (dual_teams.py): every player detection with position in metres, team and the
goalkeeper flag. The video is not needed.

Steps, per half of the match:
  1. Tracker: a constant-velocity Kalman filter per player, detections assigned to tracks frame
     by frame (Hungarian algorithm, gate grows while a track is lost, team labels and the goalkeeper
     flag are soft hints, because they are wrong now and then).
  2. Stitching: pieces of tracks (a player was hidden or missed for a few seconds) are joined when
     the gap is short and the distance can be covered by a running player.
  3. Player slots: a team has at most --team-size players on the pitch at a time, so pieces are
     put into at most that many slots per team; extra pieces overlapping in time are ghosts
     (double detections, spectators) and are dropped.
  4. Statistics from smoothed tracks: distance, speed, sprints. The positions are noisy (about
     0.5 m), so raw steps would give several times the real distance; the tracks are smoothed
     (Savitzky-Golay, --smooth-s seconds) and gaps longer than --interp-s are not counted.

People outside the pitch (bench, coach, spectators; more than --pitch-margin outside the lines) are
not tracked. A player who leaves the pitch and one who comes on later are different stints (slots),
so there can be more than --team-size stints per team and half, but never more than --team-size at a
time; stints that begin or end at a line are listed as substitutions.

Substitutions are rolling (a player can go off and come on again later). A player who comes on
again is a new stint: without recognising people by their looks he cannot be told from another
substitute, so the numbers (T0-H1-P3...) are stints, not people, and there is no total per person for
the match. Compare stints with each other (distance per minute), the team totals and the number
of players who came on (about 3 of 4 changes are recognised).

Player numbers are only stable while a player is followed. After a longer gap or when two
players of one team cross paths at close range, two players can be swapped. Numbers are per half
(sides change, substitutions); use them to compare the players of one half, and the team totals
for the match.

Output (--out-dir): tracks.csv, players.csv, tracks_sample.png, players.png; a summary is printed.

distance_m counts only the time the track is reliable (counted_pct of the time on the pitch);
distance_est_m scales it to the whole time of the stint. For a TEAM use team_distance.csv: the rate
of distance per reliable minute times 5 field players times the length of the half - the sum of
distance_est_m over the stints is too big whenever there are more stints than players.

Accuracy (simulated match with known truth, 12 players, noise 0.5 m, occlusions, false and double
detections, 8 % wrong team labels): team distance (distance_est_m) between -1 % and +8 %; the
distance of a single player is off by a median of about 15 % and by more than 40 % for the worst
tenth, because a player's track is put together from about 6 pieces per half and pieces of team
mates get mixed (70 % of the detections of a slot belong to one player). Trust team totals,
positions and speed profiles; treat the totals of single players as rough.

Usage:
  python dual_tracks.py ~/football/analysis/mecz1_dual/teams.csv --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

BIG = 1e3


# ----------------------------------------------------------------------------- substitution zone
def at_line(p, L, W, d=1.5, zone="near"):
    """p = (t, x, y): in the zone where players go off and come on: within d metres of the
    near touchline (the bench side, y = W), or of any line with zone="any"."""
    if zone == "any":
        return min(p[1], L - p[1], p[2], W - p[2]) < d
    return p[2] > W - d


def is_change(last, first, gap, bounds):
    """A track that ends in the substitution zone and one that starts there more than sub_gap
    seconds later are two players (one went off, one came on), not one player who was hidden."""
    if bounds is None:
        return False
    L, W, sub_gap, zone = bounds
    return gap > sub_gap and at_line(last, L, W, zone=zone) and at_line(first, L, W, zone=zone)


# ----------------------------------------------------------------------------- tracker
class Track:
    counter = 0

    def __init__(self, t, x, y, team, gk, sm, tid=-1):
        Track.counter += 1
        self.ids = [tid]
        self.id = Track.counter
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([sm ** 2, sm ** 2, 25.0, 25.0])
        self.t = t
        self.pts = [(t, x, y)]
        self.teams = Counter([team] if team in (0, 1) else [])
        self.gk = int(gk)

    def predict(self, t, sa):
        dt = max(t - self.t, 1e-3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = sa ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        return F @ self.s, F @ self.P @ F.T + Q

    def update(self, t, x, y, team, gk, sa, sm, vmax, tid=-1):
        self.ids.append(tid)
        s, P = self.predict(t, sa)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ P @ H.T + sm ** 2 * np.eye(2)
        K = P @ H.T @ np.linalg.inv(S)
        s = s + K @ (np.array([x, y]) - H @ s)
        self.P = (np.eye(4) - K @ H) @ P
        v = np.hypot(s[2], s[3])
        if v > vmax:
            s[2:] *= vmax / v
        self.s, self.t = s, t
        self.pts.append((t, x, y))
        if team in (0, 1):
            self.teams[team] += 1
        self.gk += int(gk)

    @property
    def team(self):
        return self.teams.most_common(1)[0][0] if self.teams else -1

    @property
    def gk_ratio(self):
        return self.gk / max(len(self.pts), 1)


def track_half(frames, sa=6.0, sm=0.6, vmax=9.0, max_lost=2.0, base_gate=2.5, gate_speed=6.0, min_hits=3,
               tentative_s=0.2, dup_dist=2.0, bounds=None):
    """frames: list of (t, [(x, y, team, gk), ...]) sorted by time. Returns finished tracks.
    A new track is tentative until it has min_hits detections; a tentative track that is not
    continued within tentative_s seconds was a false or double detection and is thrown away
    (otherwise the many false detections would steal the detections of real players).
    A detection that is left over next to a track which has just been updated (within dup_dist
    metres, same team or unknown) is a double detection of that player and starts no track."""
    active, done = [], []
    for t, dets in frames:
        still = []
        for tr in active:
            if len(tr.pts) < min_hits:
                if t - tr.t <= tentative_s:
                    still.append(tr)
            elif t - tr.t > max_lost:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        used_tr, used_det = set(), set()
        if active and dets:
            C = np.full((len(active), len(dets)), BIG)
            for i, tr in enumerate(active):
                s, _ = tr.predict(t, sa)
                gate = min(base_gate + gate_speed * (t - tr.t), 9.0)
                for j, det in enumerate(dets):
                    x, y, team, gk = det[:4]
                    if is_change(tr.pts[-1], (t, x, y), t - tr.t, bounds):
                        continue
                    d = np.hypot(s[0] - x, s[1] - y)
                    if d > gate:
                        continue
                    c = d + 0.3 * (t - tr.t) - (0.3 if len(tr.pts) >= min_hits else 0.0)
                    if team in (0, 1) and tr.team in (0, 1) and team != tr.team:
                        c += 2.0
                    if gk and tr.gk_ratio < 0.3:
                        c += 2.0
                    elif not gk and tr.gk_ratio > 0.7 and len(tr.pts) > 15:
                        c += 1.0
                    C[i, j] = c
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < BIG:
                    x, y, team, gk = dets[j][:4]
                    active[i].update(t, x, y, team, gk, sa, sm, vmax, dets[j][4] if len(dets[j]) > 4 else -1)
                    used_tr.add(i)
                    used_det.add(j)
        updated = [active[i] for i in used_tr]
        for j, det in enumerate(dets):
            if j not in used_det:
                x, y, team, gk = det[:4]
                if any(np.hypot(tr.pts[-1][1] - x, tr.pts[-1][2] - y) < dup_dist
                       and not (team in (0, 1) and tr.team in (0, 1) and team != tr.team) for tr in updated):
                    continue
                active.append(Track(t, x, y, team, gk, sm, det[4] if len(det) > 4 else -1))
    done.extend(active)
    return [tr for tr in done if len(tr.pts) >= min_hits]


# ----------------------------------------------------------------------------- stitching
def stitch(tracks, max_gap=4.0, run_speed=6.0, base=1.6, bounds=None):
    """Join tracks that follow each other. Returns chains (lists of tracks in time order)."""
    order = sorted(tracks, key=lambda tr: tr.pts[0][0])
    n = len(order)
    C = np.full((n, n), BIG)
    for i, a in enumerate(order):
        ta, xa, ya = a.pts[-1]
        for j, b in enumerate(order):
            if i == j:
                continue
            tb, xb, yb = b.pts[0]
            gap = tb - ta
            if gap < -0.05 or gap > max_gap:
                continue
            if is_change(a.pts[-1], b.pts[0], gap, bounds):
                continue
            d = np.hypot(xb - xa, yb - ya)
            if d > min(base + run_speed * max(gap, 0.0), 14.0):
                continue
            if a.team in (0, 1) and b.team in (0, 1) and a.team != b.team:
                continue
            if (a.gk_ratio > 0.5) != (b.gk_ratio > 0.5) and min(len(a.pts), len(b.pts)) > 15:
                continue
            C[i, j] = d + 0.4 * max(gap, 0.0)
    nxt = {}
    for i, j in zip(*linear_sum_assignment(C)):
        if C[i, j] < BIG:
            nxt[i] = j
    has_prev = set(nxt.values())
    chains = []
    for i in range(n):
        if i in has_prev:
            continue
        chain, k = [order[i]], i
        while k in nxt:
            k = nxt[k]
            chain.append(order[k])
        chains.append(chain)
    return chains


def chain_info(chain, L=None, gk_zone=8.0):
    """L: pitch length; a goalkeeper track must stay near a goal (median within gk_zone m of a goal line)."""
    votes = Counter()
    for tr in chain:
        votes.update(tr.teams)
    pts = sorted(p for tr in chain for p in tr.pts)
    n = sum(len(tr.pts) for tr in chain)
    med_x = float(np.median([p[1] for p in pts]))
    near_goal = L is None or med_x < gk_zone or med_x > L - gk_zone
    return {"chain": chain, "pts": pts, "ids": [i for tr in chain for i in tr.ids], "start": pts[0][0], "end": pts[-1][0],
            "team": votes.most_common(1)[0][0] if votes else -1,
            "gk": sum(tr.gk for tr in chain) / max(n, 1) > 0.5 and near_goal, "n": n}




def assign_slots(chains, team_size=6, min_s=1.0, bounds=None, overlap_s=2.0):
    """Stints of players: at most team_size on the pitch at a time and team. Chains that overlap
    busy slots are ghosts; a substitute often comes on a moment before the player who leaves has
    vanished, so up to overlap_s seconds of overlap are allowed. A player who goes off and one who
    comes on (see is_change) are different stints, never joined into one slot."""
    infos = sorted((chain_info(c, bounds[0] if bounds else None) for c in chains), key=lambda c: c["start"])
    infos = [c for c in infos if c["end"] - c["start"] >= min_s]
    slots = {0: [], 1: []}
    ghosts = 0
    for c in infos:
        teams = [c["team"]] if c["team"] in (0, 1) else [0, 1]
        best = None
        for t in teams:
            free = [s for s in slots[t] if s["end"] <= c["start"] + 0.05 and s["gk"] == c["gk"]
                    and not is_change(s["pts"][-1], c["pts"][0], c["start"] - s["end"], bounds)]
            for s in free:
                cost = np.hypot(s["pts"][-1][1] - c["pts"][0][1], s["pts"][-1][2] - c["pts"][0][2]) + 0.3 * (c["start"] - s["end"])
                if best is None or cost < best[0]:
                    best = (cost, t, s)
        if best is not None:
            _, t, s = best
            s["pts"] += c["pts"]
            s["ids"] += c["ids"]
            s["end"] = c["end"]
            s["n"] += c["n"]
            continue
        t_new = next((t for t in teams if sum(1 for s in slots[t] if s["end"] > c["start"] + overlap_s) < team_size
                      and not (c["gk"] and any(s["gk"] and s["end"] > c["start"] + overlap_s for s in slots[t]))), None)
        if t_new is None:
            ghosts += 1
            continue
        slots[t_new].append({"pts": list(c["pts"]), "ids": list(c["ids"]), "start": c["start"], "end": c["end"],
                             "gk": c["gk"], "n": c["n"]})
    return slots, ghosts


# ----------------------------------------------------------------------------- statistics
def smooth_track(pts, dt, interp_s=2.0, smooth_s=1.0, vmax=9.0):
    """pts: [(t, x, y)] of one player. Returns arrays (t, x, y, speed, valid, detected) on a uniform grid."""
    ts = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    o = np.argsort(ts)
    ts, xs, ys = ts[o], xs[o], ys[o]
    grid = np.arange(ts[0], ts[-1] + dt / 2, dt)
    gx, gy = np.interp(grid, ts, xs), np.interp(grid, ts, ys)
    idx = np.searchsorted(ts, grid)
    left, right = ts[np.clip(idx - 1, 0, len(ts) - 1)], ts[np.clip(idx, 0, len(ts) - 1)]
    valid = (right - left) <= interp_s
    detected = np.isin(np.round(grid / dt).astype(int), np.round(ts / dt).astype(int))
    speed = np.zeros(len(grid))
    win = int(round(smooth_s / dt)) | 1
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        w = min(win, b - a)
        if w % 2 == 0:
            w -= 1
        if w < 5:
            speed[a:b] = 0.0
            valid[a:b] = False
            continue
        sx = savgol_filter(gx[a:b], w, 2)
        sy = savgol_filter(gy[a:b], w, 2)
        vx = savgol_filter(gx[a:b], w, 2, deriv=1, delta=dt)
        vy = savgol_filter(gy[a:b], w, 2, deriv=1, delta=dt)
        gx[a:b], gy[a:b] = sx, sy
        speed[a:b] = np.hypot(vx, vy)
    # speeds no amateur reaches are jumps of the track to another player: not counted (+-3 frames)
    bad = speed > vmax
    bad = np.convolve(bad.astype(float), np.ones(7), mode="same") > 0
    valid &= ~bad
    speed[~valid] = 0.0
    return grid, gx, gy, speed, valid, detected


def stats_of(grid, speed, valid, detected, dt, sprint_speed=5.5, sprint_s=1.0, hi_speed=4.0):
    n_valid = int(valid.sum())
    v = speed[valid]
    sprints, run = 0, 0
    for s, ok in zip(speed, valid):
        if ok and s >= sprint_speed:
            run += 1
        else:
            if run * dt >= sprint_s:
                sprints += 1
            run = 0
    if run * dt >= sprint_s:
        sprints += 1
    return {"first_s": float(grid[0]), "last_s": float(grid[-1]), "on_pitch_min": float((grid[-1] - grid[0]) / 60),
            "coverage_pct": round(100 * float(detected.sum()) / max(len(grid), 1), 1),
            "counted_pct": round(100 * n_valid / max(len(grid), 1), 1),
            "distance_m": round(float(v.sum() * dt), 0),
            "distance_est_m": round(float(v.sum() * dt) / max(n_valid / max(len(grid), 1), 0.3), 0),
            "dist_per_min_m": round(float(v.sum() * dt) / max(n_valid / max(len(grid), 1), 0.3) / max((grid[-1] - grid[0]) / 60, 1 / 60), 0),
            "avg_speed_ms": round(float(v.mean()) if n_valid else 0.0, 2),
            "top_speed_ms": round(float(np.percentile(v, 99)) if n_valid else 0.0, 1), "sprints": sprints,
            "hi_run_m": round(float(v[v >= hi_speed].sum() * dt), 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", help="teams.csv from dual_teams.py")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--out-dir", help="default: the folder of teams.csv")
    ap.add_argument("--team-size", type=int, default=6, help="players of a team on the pitch (with the goalkeeper)")
    ap.add_argument("--pitch-margin", type=float, default=0.3,
                    help="detections further than this (m) outside the lines are people at the bench, not players")
    ap.add_argument("--sub-gap", type=float, default=0.5,
                    help="a track ending at the bench touchline and another starting there this many seconds later "
                         "or more are two different players (rolling substitutions)")
    ap.add_argument("--sub-zone", choices=["near", "any"], default="near",
                    help="where players go off and come on: the near touchline (bench side) or any line")
    ap.add_argument("--smooth-s", type=float, default=1.0)
    ap.add_argument("--interp-s", type=float, default=2.0, help="gaps longer than this are not counted")
    ap.add_argument("--max-gap", type=float, default=4.0, help="longest gap that is bridged when stitching, s")
    ap.add_argument("--sprint-speed", type=float, default=5.5, help="m/s (5.5 m/s = 20 km/h)")
    ap.add_argument("--sample-start", type=float, default=120.0, help="start of the sample plot, s after the start of a half")
    a = ap.parse_args()

    teams_path = Path(a.teams).expanduser()
    out = Path(a.out_dir).expanduser() if a.out_dir else teams_path.parent
    out.mkdir(parents=True, exist_ok=True)
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]

    per = defaultdict(lambda: defaultdict(list))         # half -> time -> detections
    outside = total = 0
    m = a.pitch_margin
    with open(teams_path) as f:
        for r in csv.DictReader(f):
            if r["class"] != "player":
                continue
            x, y = float(r["x_m"]), float(r["y_m"])
            total += 1
            if not (-m <= x <= L + m and -m <= y <= W + m):
                outside += 1
                continue
            team = int(r["team"]) if r["team"] in ("0", "1") else -1
            per[int(r["half"])][float(r["time_s"])].append((x, y, team, r.get("gk") == "1"))
    print(f"player detections: {total}, {outside} ({100 * outside / max(total, 1):.1f}%) outside the pitch by more than {m} m "
          "(bench, coach, spectators) are not tracked")

    rows_tracks, rows_players, samples, changes = [], [], {}, {}
    for half in sorted(per):
        frames = sorted(per[half].items())
        dt = float(np.median(np.diff([t for t, _ in frames[:2000]])))
        bounds = (L, W, a.sub_gap, a.sub_zone)
        tracks = track_half(frames, bounds=bounds)
        chains = stitch(tracks, max_gap=a.max_gap, bounds=bounds)
        slots, ghosts = assign_slots(chains, a.team_size, bounds=bounds)
        # players who came on: a track of at least 1 s that starts at the bench line later than 30 s after the start of the half
        infos = [chain_info(c) for c in chains]
        came_on = {0: 0, 1: 0}
        for ci in infos:
            if ci["end"] - ci["start"] >= 1.0 and ci["team"] in (0, 1) and ci["start"] - frames[0][0] > 30 \
                    and at_line(ci["pts"][0], L, W, zone=a.sub_zone):
                came_on[ci["team"]] += 1
        changes[half] = came_on
        print(f"half {half}: {len(frames)} frames, {len(tracks)} track pieces -> {len(chains)} chains -> "
              f"slots {len(slots[0])} + {len(slots[1])}, ghosts dropped {ghosts}")
        for team in (0, 1):
            ordered = sorted(slots[team], key=lambda s: (not s["gk"], s["start"]))
            for k, s in enumerate(ordered, 1):
                pid = f"T{team}-H{half}-{'GK' if s['gk'] else 'P'}{k}"
                grid, gx, gy, speed, valid, detected = smooth_track(s["pts"], dt, a.interp_s, a.smooth_s)
                st = stats_of(grid, speed, valid, detected, dt, a.sprint_speed)
                pts_sorted = sorted(s["pts"])
                first_gap = st["first_s"] - frames[0][0]
                last_gap = frames[-1][0] - st["last_s"]
                st["enters_at_line"] = int(first_gap > 30 and at_line(pts_sorted[0], L, W, zone=a.sub_zone))
                st["leaves_at_line"] = int(last_gap > 30 and at_line(pts_sorted[-1], L, W, zone=a.sub_zone))
                rows_players.append({"half": half, "team": team, "player": pid, "role": "GK" if s["gk"] else "field", **st})
                for t, x, y, v, ok, det in zip(grid, gx, gy, speed, valid, detected):
                    if ok:
                        rows_tracks.append((half, round(t, 3), pid, team, round(x, 2), round(y, 2), round(v, 2),
                                            "detected" if det else "interpolated"))
                samples[pid] = (half, team, grid, gx, gy, valid)

    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["half", "time_s", "player", "team", "x_m", "y_m", "speed_ms", "source"])
        w.writerows(rows_tracks)
    with open(out / "players.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_players[0].keys()))
        w.writeheader()
        w.writerows(rows_players)

    print("\nNOTE: the rows are pieces of tracks (stints), not people: pieces of several players get mixed, and a "
          "player who comes on again is a new piece. Use the team totals below.")
    print(f"\n{'player':10s} {'min':>5s} {'seen%':>6s} {'counted%':>9s} {'dist m':>7s} {'est m':>6s} {'avg m/s':>8s} {'top m/s':>8s} {'sprints':>8s}")
    for r in rows_players:
        print(f"{r['player']:10s} {r['on_pitch_min']:5.1f} {r['coverage_pct']:6.1f} {r['counted_pct']:9.1f} {r['distance_m']:7.0f} {r['distance_est_m']:6.0f} "
              f"{r['avg_speed_ms']:8.2f} {r['top_speed_ms']:8.1f} {r['sprints']:8d}")
    team_rows = []
    for half in sorted(per):
        dur_min = (max(per[half]) - min(per[half])) / 60
        for team in (0, 1):
            fld = [r for r in rows_players if r["half"] == half and r["team"] == team and r["role"] == "field"]
            gks = [r for r in rows_players if r["half"] == half and r["team"] == team and r["role"] != "field"]
            cnt_min = sum(r["on_pitch_min"] * r["counted_pct"] / 100 for r in fld)
            if cnt_min > 0:
                rate = sum(r["distance_m"] for r in fld) / cnt_min                     # m per minute of a field player
                team_rows.append({"half": half, "team": team, "field_stints": len(fld), "gk_stints": len(gks),
                                  "m_per_player_min": round(rate), "team_field_km": round(rate * (a.team_size - 1) * dur_min / 1000, 1),
                                  "per_field_player_m": round(rate * dur_min)})
    with open(out / "team_distance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(team_rows[0].keys()))
        w.writeheader()
        w.writerows(team_rows)
    subs = [r for r in rows_players if r["enters_at_line"] or r["leaves_at_line"]]
    print(f"\nstints starting or ending at a line more than 30 s after the start / before the end of the half "
          f"(substitutions): {len(subs)}")
    for r in subs:
        what = ("comes on at %02d:%02d" % (int(r["first_s"]) // 60, int(r["first_s"]) % 60) if r["enters_at_line"] else "") + \
               (" " if r["enters_at_line"] and r["leaves_at_line"] else "") + \
               ("goes off at %02d:%02d" % (int(r["last_s"]) // 60, int(r["last_s"]) % 60) if r["leaves_at_line"] else "")
        print(f"  {r['player']}: {what}")
    for half in sorted(per):
        print(f"half {half}: players who came on (substitutions, about 3 of 4 recognised): "
              f"team 0 {changes[half][0]}, team 1 {changes[half][1]}")
    print("\nteam distance (field players): rate of the reliable track time x 5 field players x length of the half")
    for r in team_rows:
        print(f"half {r['half']} team {r['team']}: {r['team_field_km']} km in total, {r['per_field_player_m']} m per field player "
              f"({r['m_per_player_min']} m per minute); tracks: {r['field_stints']} field + {r['gk_stints']} goalkeeper "
              f"(a team has 5 + 1: more means fragments)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["tab:blue", "tab:red"]
    cf = teams_path.with_name(teams_path.stem + "_colors.json")
    if cf.exists():
        c = json.loads(cf.read_text())
        cols = [tuple(v / 255 for v in c[f"team_{t}_bgr"][::-1]) for t in (0, 1)]
    halves = sorted(per)
    fig, axes = plt.subplots(1, len(halves), figsize=(7 * len(halves), 5))
    for ax, half in zip(np.atleast_1d(axes), halves):
        t0 = min(t for t in per[half]) + a.sample_start
        for pid, (h, team, grid, gx, gy, valid) in samples.items():
            if h != half:
                continue
            m = (grid >= t0) & (grid < t0 + 30) & valid
            if m.sum() > 2:
                idx = np.flatnonzero(m)
                for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):     # no lines across gaps
                    ax.plot(gx[seg], gy[seg], "-", color=cols[team], lw=1.4, alpha=0.8)
                ax.plot(gx[idx[-1]], gy[idx[-1]], "o", color=cols[team], ms=6, mec="k")
                ax.text(gx[idx[-1]] + 0.4, gy[idx[-1]] - 0.4, pid.split("-")[-1], fontsize=7)
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k-", lw=1)
        ax.plot([L / 2, L / 2], [0, W], "k-", lw=1)
        ax.set_xlim(-2, L + 2)
        ax.set_ylim(W + 2, -2)
        ax.set_aspect("equal")
        ax.set_title(f"Half {half}: tracks over 30 s ({int(t0) // 60:02d}:{int(t0) % 60:02d} in the recording)")
    fig.tight_layout()
    fig.savefig(out / "tracks_sample.png", dpi=110)

    fig, axes = plt.subplots(1, len(halves), figsize=(7 * len(halves), 4.5), sharey=True)
    for ax, half in zip(np.atleast_1d(axes), halves):
        for team in (0, 1):
            rs = [r for r in rows_players if r["half"] == half and r["team"] == team]
            xs = np.arange(len(rs)) + team * (len(rs) + 1)
            ax.bar(xs, [r["distance_est_m"] for r in rs], color=cols[team], edgecolor="k")
            for x, r in zip(xs, rs):
                ax.text(x, r["distance_est_m"] + 15, r["player"].split("-")[-1], ha="center", fontsize=7)
        ax.set_title(f"Half {half}: distance per track piece [m], NOT per player (left team 0, right team 1)")
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out / "players.png", dpi=110)
    print(f"\nSaved {out / 'tracks.csv'}, players.csv, tracks_sample.png, players.png")


if __name__ == "__main__":
    main()
