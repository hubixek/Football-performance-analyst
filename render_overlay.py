#!/usr/bin/env python3
"""Top-down animation of a match, made from the results of the analysis (the video is not needed).

Drawn: the pitch with stripes and markings, the players as circles in the colour of their team,
the goalkeepers marked with "1", the ball, and a bar at the top with the possession of the last
minute and the state of the ball (possession / out of play / lying still / restart).

Players are not identified (see the README), so the circles are "puppets": every team has a fixed
set of circles (--max-per-team, the goalkeeper among them) that follow the detections.
  - a circle never moves faster than --vmax m/s, so it cannot teleport: when a detection is far
    away it runs there,
  - a circle never appears or disappears: without a detection it goes on for a moment and then stands
    still, and when a player shows up elsewhere the nearest idle circle runs there,
  - one circle of every team is the goalkeeper: it stays near the own goal of the team and prefers
    the detections flagged as goalkeeper,
  - the animation is drawn at --fps frames per second with the positions in between interpolated.

Usage:
  python render_overlay.py ~/football/analysis/mecz1_dual --pitch pitch/pitch_6v6.json \
      --start 38:35 --duration 40 --gif
"""
import argparse
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.optimize import linear_sum_assignment

PALETTES = {"reference": {0: (200, 90, 30), 1: (40, 30, 175)}}          # BGR: blue and red
GRASS = ((44, 112, 48), (50, 124, 54))                                    # two stripe colours, BGR
WHITE, BLACK = (245, 245, 245), (25, 25, 25)
BIG = 1e3


def parse_time(text):
    """'12:30' -> 750 s, '1:02:03' -> 3723 s, '750' -> 750 s."""
    t = 0.0
    for part in str(text).split(":"):
        t = t * 60 + float(part)
    return t


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


# ------------------------------------------------------------------ data
def load_possession(folder):
    rows = []
    with open(folder / "possession.csv") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["time_s"]), int(r["half"]),
                         float(r["ball_x"]) if r["ball_x"] else np.nan, float(r["ball_y"]) if r["ball_y"] else np.nan,
                         int(r["possession_team"]) if r["possession_team"] != "" else -1,
                         r.get("ball_state", "") or ""))
    return rows


def load_players(folder, t0, t1, half, L, W, margin=0.3):
    """time -> [(x, y, team, gk flag)] for the players inside the pitch (team -1 = the colour was not clear)."""
    frames = defaultdict(list)
    with open(folder / "teams.csv") as f:
        for r in csv.DictReader(f):
            t = float(r["time_s"])
            if t > t1 + 1:
                break
            if t < t0 or int(r["half"]) != half or r["class"] != "player":
                continue
            x, y = float(r["x_m"]), float(r["y_m"])
            if -margin <= x <= L + margin and -margin <= y <= W + margin:
                team = int(r["team"]) if r["team"] in ("0", "1") else -1         # unknown team: -1
                frames[t].append((x, y, team, r.get("gk") == "1"))
    return frames


def own_goals(folder, half, frames, L):
    """team -> x of its own goal line (0 or L). Best source: stats.json (from the whole half);
    else the goalkeeper detections of the window; else who stands further to the left."""
    sj = folder / "stats.json"
    if sj.exists():
        right = json.loads(sj.read_text()).get("attacks_right", {}).get(f"half_{half}")
        if right in (0, 1):
            return {right: 0.0, 1 - right: L}, "stats.json"
    gk_x = defaultdict(list)
    all_x = defaultdict(list)
    for dets in frames.values():
        for x, y, team, gk in dets:
            if team in (0, 1):
                (gk_x if gk else all_x)[team].append(x)
    for team in (0, 1):
        if len(gk_x[team]) >= 10:
            left = float(np.median(gk_x[team])) < L / 2
            return {team: 0.0 if left else L, 1 - team: L if left else 0.0}, "goalkeeper detections"
    m0, m1 = np.mean(all_x[0] or [L / 2]), np.mean(all_x[1] or [L / 2])
    return ({0: 0.0, 1: L} if m0 < m1 else {0: L, 1: 0.0}), "positions of the teams"


# ------------------------------------------------------------------ puppets
class Puppet:
    def __init__(self, team, gk=False):
        self.team = team
        self.active, self.gk = False, gk
        self.gk_score = 0.0                                   # how often it was matched to a detection in the goalkeeper kit
        self.x = self.y = self.vx = self.vy = 0.0
        self.tx = self.ty = 0.0
        self.chasing = False
        self.last_seen = self.born = 0.0

    def start(self, t, x, y):
        self.active, self.x, self.y, self.vx, self.vy = True, x, y, 0.0, 0.0
        self.tx, self.ty, self.chasing = x, y, False
        self.last_seen = self.born = t

    def alpha(self, t):
        return min(1.0, (t - self.born) / 0.3 + 0.15)        # only a new circle fades in

    def predicted(self, dt):
        return (self.tx, self.ty) if self.chasing else (self.x + self.vx * dt, self.y + self.vy * dt)


class PuppetSystem:
    """Fixed sets of circles for both teams. A circle never appears or disappears while the animation
    runs and never moves faster than vmax:
      - it follows a detection near it; the team label of a detection is only a hint (a wrong or unknown
        label costs a little), because the colour of a player is misjudged now and then,
      - without detections it stands still (it runs on for a moment first),
      - when a player shows up elsewhere and is seen several times in a row, the nearest circle that has
        been without a detection for a while RUNS there; the farther away it is, the longer the player
        must be seen first,
      - the goalkeeper of a team is the circle that follows the goalkeeper kit (he may leave the goal)."""

    def __init__(self, goals, L, W, n=6, vmax=8.5, beta=0.30, persist=4, lost_s=0.5, vel_mix=0.30, grow=0.6):
        self.goals, self.L, self.W, self.vmax, self.beta, self.vel_mix = goals, L, W, vmax, beta, vel_mix
        self.persist, self.lost_s, self.grow = persist, lost_s, grow
        self.slots = [Puppet(team) for team in (0, 1) for _ in range(n)]
        self.runs = {0: [], 1: []}                         # distances of the runs of idle circles to new players
        self.warm_until = -1e9                             # before this time the circles settle instantly (warm-up)
        self.t = None
        self.cand = []

    def _first_of(self, team):
        return next(p for p in self.slots if p.team == team)

    def _init(self, t, dets):
        for team in (0, 1):
            gx = self.goals[team]
            mine = [d for d in dets if d[2] in (team, -1)]
            near = sorted([d for d in mine if abs(d[0] - gx) < 12], key=lambda d: (not (d[3] and d[2] == team), abs(d[0] - gx)))
            gk = self._first_of(team)
            if near:
                gk.start(t, near[0][0], near[0][1])
            else:                                                    # goalkeeper not seen yet: waits in the goal
                gk.start(t, gx + (2.5 if gx == 0 else -2.5), self.W / 2)
            gk.gk, gk.gk_score = True, 5.0

    @staticmethod
    def _dedupe(dets, d_min=1.2):
        out = []
        for d in sorted(dets, key=lambda d: (not d[3], d[2] == -1)):          # goalkeeper kit and known teams first
            if all(np.hypot(d[0] - o[0], d[1] - o[1]) >= d_min for o in out):
                out.append(d)
        return out

    def _match(self, puppets, dets, allowed, gate_fn, dt):
        """Hungarian assignment of puppets to the detections with the indices in `allowed`."""
        idx = sorted(allowed)
        if not puppets or not idx:
            return {}
        C = np.full((len(puppets), len(idx)), BIG)
        for i, p in enumerate(puppets):
            px, py = p.predicted(dt)
            for k, j in enumerate(idx):
                x, y, tm, gk = dets[j]
                d = np.hypot(px - x, py - y)
                if d > gate_fn(p, j):
                    continue
                pen = 2.5 if tm not in (-1, p.team) else (0.3 if tm == -1 else 0.0)      # the label is only a hint
                if gk:                                               # a goalkeeper-kit detection belongs to the goalkeeper circle
                    pen += -1.0 if (p.gk and tm in (-1, p.team)) else 0.8
                C[i, k] = max(d + pen, 0.0)
        out = {}
        for i, k in zip(*linear_sum_assignment(C)):
            if C[i, k] < BIG:
                out[i] = idx[k]
        return out

    def _move(self, p, x, y, t, dt, chase):
        bx, by = (p.x, p.y) if p.chasing or chase else (p.x + p.vx * dt, p.y + p.vy * dt)
        beta = 1.0 if (p.chasing or chase) else self.beta
        nx, ny = bx + beta * (x - bx), by + beta * (y - by)
        step, lim = np.hypot(nx - p.x, ny - p.y), (self.vmax * dt if t >= self.warm_until else 1e9)
        if step > lim:                                               # never faster than a human
            nx, ny = p.x + (nx - p.x) * lim / step, p.y + (ny - p.y) * lim / step
        m = self.vel_mix
        p.vx, p.vy = (1 - m) * p.vx + m * (nx - p.x) / dt, (1 - m) * p.vy + m * (ny - p.y) / dt
        p.x, p.y, p.tx, p.ty, p.last_seen = nx, ny, x, y, t
        p.chasing = bool(np.hypot(x - nx, y - ny) > 2.0)

    def step(self, t, dets):
        dets = self._dedupe(dets)
        if self.t is None:
            self._init(t, dets)
        dt = max(t - self.t, 1e-3) if self.t is not None else 1 / 15
        self.t = t
        act = [p for p in self.slots if p.active]
        fresh = [p for p in act if t - p.last_seen <= self.lost_s]
        lost = [p for p in act if p not in fresh]
        hit = set()
        for p in self.slots:
            p.gk_score *= 0.98
        # 1) circles that were seen a moment ago follow a detection near them
        m = self._match(fresh, dets, set(range(len(dets))), lambda p, j: min(2.2 + 4.0 * (t - p.last_seen), 9.0), dt)
        for i, j in m.items():
            self._move(fresh[i], dets[j][0], dets[j][1], t, dt, chase=False)
            hit.add(id(fresh[i]))
            if dets[j][3] and dets[j][2] in (-1, fresh[i].team):
                fresh[i].gk_score += 1.0
        matched_d = set(m.values())
        # 2) detections nobody follows: a candidate must be seen several times in a row
        #    (the detections shake, so the same player may be up to 2 m from the previous one; two missed frames are forgiven)
        cand_next, cnt_of, taken = [], {}, set()
        for j, (x, y, tm, gk) in enumerate(dets):
            if j in matched_d:
                continue
            best = None
            for ci, c in enumerate(self.cand):
                if ci not in taken:
                    dd = np.hypot(c["x"] - x, c["y"] - y)
                    if dd < 2.0 and (best is None or dd < best[0]):
                        best = (dd, ci)
            if best is None:
                c = {"x": x, "y": y, "cnt": 1, "miss": 0}
            else:
                taken.add(best[1])
                c = dict(self.cand[best[1]])
                c["cnt"] += 1
                c["miss"] = 0
                c["x"], c["y"] = c["x"] + 0.5 * (x - c["x"]), c["y"] + 0.5 * (y - c["y"])
            c["j"] = j
            cand_next.append(c)
            cnt_of[j] = c["cnt"]
        for ci, c in enumerate(self.cand):
            if ci not in taken and c["miss"] < 2:                    # not seen in this frame: kept for two more frames
                cand_next.append({**c, "miss": c["miss"] + 1, "j": None})
        persistent = {j for j, c in cnt_of.items() if c >= self.persist}
        # 3) the nearest idle circle runs to a persistent candidate (the farther, the longer it must have been seen)
        m2 = self._match(lost, dets, persistent,
                         lambda p, j: 1e9 if t < self.warm_until else 4.0 + max(cnt_of[j] - self.persist, 0) * self.grow, dt)
        consumed = set(m2.values())
        for i, j in m2.items():
            if t >= self.warm_until:
                self.runs[lost[i].team].append(float(np.hypot(lost[i].x - dets[j][0], lost[i].y - dets[j][1])))
            self._move(lost[i], dets[j][0], dets[j][1], t, dt, chase=True)
            hit.add(id(lost[i]))
            if dets[j][3] and dets[j][2] in (-1, lost[i].team):
                lost[i].gk_score += 1.0
        persistent -= consumed
        # 4) a circle that has never been used (only at the start): starts at the candidate
        for j in sorted(persistent):
            x, y, tm, gk = dets[j]
            n_act = {tt: sum(1 for p in self.slots if p.team == tt and p.active) for tt in (0, 1)}
            for team in ([tm] if tm in (0, 1) else sorted((0, 1), key=lambda tt: n_act[tt])):
                free = [p for p in self.slots if p.team == team and not p.active]
                if free:
                    free[0].start(t, x, y)
                    hit.add(id(free[0]))
                    consumed.add(j)
                    break
        self.cand = [c for c in cand_next if c["j"] not in consumed and c["cnt"] < self.persist + 120]
        # the goalkeeper label of a team moves to the circle that follows the goalkeeper kit
        for team in (0, 1):
            mine = [p for p in self.slots if p.team == team and p.active]
            holder = next((p for p in mine if p.gk), None)
            best = max(mine, key=lambda p: p.gk_score, default=None)
            if best is not None and best is not holder and best.gk_score >= 3.0 and (holder is None or best.gk_score > holder.gk_score + 2.0):
                for p in mine:
                    p.gk = p is best
        # 5) the rest go on for a moment / stand still
        for p in self.slots:
            if not p.active or id(p) in hit:
                continue
            if p.chasing:                                            # still running to the last place it was seen at
                self._move(p, p.tx, p.ty, t, dt, chase=True)
                p.last_seen = min(p.last_seen, t - dt)               # a target that is not seen does not count as seen
                continue
            p.vx, p.vy = p.vx * 0.85 ** (dt * 15), p.vy * 0.85 ** (dt * 15)
            p.x = float(np.clip(p.x + p.vx * dt, 0, self.L))
            p.y = float(np.clip(p.y + p.vy * dt, 0, self.W))

    def snapshot(self, t):
        return {(p.team, i): (p.x, p.y, p.alpha(t), p.gk) for i, p in enumerate(self.slots) if p.active}


def smooth_segments(arr, sigma_frames):
    """Gaussian smoothing (both directions in time, so no lag) of every run of known values of an (n, 2) array."""
    out = arr.copy()
    if sigma_frames <= 0:
        return out
    known = ~np.isnan(arr[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, known.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 3:
            out[a:b] = gaussian_filter1d(arr[a:b], sigma_frames, axis=0, mode="nearest")
    return out


def backlash(path, R):
    """A point that stays where it is until the path leaves the circle of radius R around it, then is pushed along."""
    out = np.empty_like(path)
    a = path[0].copy()
    for i, q in enumerate(path):
        d = q - a
        n = np.hypot(d[0], d[1])
        if n > R:
            a = q - d / n * R
        out[i] = a
    return out


def deadband(path, R):
    """Backlash forwards and backwards in time, averaged: a shaking circle stands still, a moving one has no lag."""
    return 0.5 * (backlash(path, R) + backlash(path[::-1], R)[::-1])


def smooth_adaptive(arr, dt, base_s, dead_m=0.5):
    """Smoothing without lag. Slow and standing circles get a dead zone (the circle does not move while the detection
    only shakes within dead_m metres), sprinting ones only a light smoothing. dead_m = 0: no dead zone, the smoothing
    strength depends on the speed (heavy for slow, light for fast)."""
    out = arr.copy()
    known = ~np.isnan(arr[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, known.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 3:
            continue
        seg = arr[a:b]
        pf, pm, ps = (gaussian_filter1d(seg, base_s * k / dt, axis=0, mode="nearest") for k in (0.5, 1.0, 2.4))
        v = gaussian_filter1d(np.hypot(*np.gradient(pm, dt, axis=0).T), 0.3 / dt, mode="nearest")
        w_fast = np.clip((v - 3.0) / 2.0, 0.0, 1.0)
        if dead_m > 0:
            slow = gaussian_filter1d(deadband(pm, dead_m), 0.2 / dt, axis=0, mode="nearest")
            out[a:b] = (1.0 - w_fast)[:, None] * slow + w_fast[:, None] * pf
        else:
            w_slow = np.clip((1.6 - v) / 1.0, 0.0, 1.0)
            out[a:b] = w_slow[:, None] * ps + (1.0 - w_slow - w_fast)[:, None] * pm + w_fast[:, None] * pf
    return out


def smooth_states(states, dt, base_s, adaptive=True, dead_m=0.5):
    """The circles of all frames: every circle's path is smoothed, so the movement is as smooth as a player's."""
    keys = {k for st in states for k in st}
    for key in keys:
        arr = np.array([[st[key][0], st[key][1]] if key in st else [np.nan, np.nan] for st in states])
        sm = smooth_adaptive(arr, dt, base_s, dead_m) if adaptive else smooth_segments(arr, base_s / dt)
        for i, st in enumerate(states):
            if key in st:
                x, y, al, gk = st[key]
                st[key] = (float(sm[i, 0]), float(sm[i, 1]), al, gk)
    return states


def shake_cm(states, dt, still_s=3.0, v_max=0.5):
    """How much the circles shake while they stand still: the path minus the same path smoothed over 3 s,
    only where the circle does not move faster than v_max for still_s seconds (so slowing down is not counted)."""
    res, n_still, n_all = [], 0, 0
    for key in {k for st in states for k in st}:
        pts = np.array([st[key][:2] for st in states if key in st])
        if len(pts) < int(still_s / dt) + 5:
            continue
        v = np.hypot(*np.gradient(gaussian_filter1d(pts, 0.3 / dt, axis=0, mode="nearest"), dt, axis=0).T)
        calm = maximum_filter1d(v, size=int(still_s / dt) | 1, mode="nearest") < v_max
        n_all += len(pts)
        n_still += int(calm.sum())
        if calm.sum() > 10:
            resid = pts - gaussian_filter1d(pts, 1.5 / dt, axis=0, mode="nearest")
            res.append(np.sqrt((resid[calm] ** 2).sum(-1).mean()))
    return (float(np.mean(res)) * 100 if res else None), (n_still / n_all if n_all else 0.0)


def lerp_states(a, b, f):
    """Circles between two data frames: positions and transparency interpolated."""
    out = {}
    for key in set(a) | set(b):
        if key in a and key in b:
            (xa, ya, aa, ga), (xb, yb, ab, gb) = a[key], b[key]
            out[key] = (xa + f * (xb - xa), ya + f * (yb - ya), aa + f * (ab - aa), gb)
        elif key in a:
            x, y, al, g = a[key]
            out[key] = (x, y, al * (1 - f), g)
        else:
            x, y, al, g = b[key]
            out[key] = (x, y, al * f, g)
    return out


# ------------------------------------------------------------------ drawing
class Canvas:
    def __init__(self, L, W, pitch, scale=18, margin=2.0, header=92):
        self.L, self.W, self.s, self.m, self.h = L, W, scale, margin, header
        self.width = int((L + 2 * margin) * scale)
        self.height = int((W + 2 * margin) * scale) + header
        self.background = self._pitch(pitch)

    def px(self, x, y):
        return int(round((x + self.m) * self.s)), int(round((y + self.m) * self.s)) + self.h

    def pxf(self, x, y):
        """Pixel position with fractions: circles are drawn with sub-pixel accuracy, so slow movement is not stair-stepped."""
        return (x + self.m) * self.s, (y + self.m) * self.s + self.h

    def _pitch(self, pitch):
        img = np.zeros((self.height, self.width, 3), np.uint8)
        n = 12
        edges = np.linspace(-self.m, self.L + self.m, n + 1)
        for i in range(n):
            (x0, y0), (x1, y1) = self.px(edges[i], -self.m), self.px(edges[i + 1], self.W + self.m)
            cv2.rectangle(img, (x0, y0), (x1, y1), GRASS[i % 2], -1)
        img[:self.h] = (35, 35, 35)
        th = max(2, self.s // 6)
        line = lambda a, b: cv2.line(img, self.px(*a), self.px(*b), WHITE, th, cv2.LINE_AA)
        L, W = self.L, self.W
        for a, b in (((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))):
            line(a, b)
        r = pitch.get("centre_circle_radius_m", 4.75)
        cv2.circle(img, self.px(L / 2, W / 2), int(r * self.s), WHITE, th, cv2.LINE_AA)
        cv2.circle(img, self.px(L / 2, W / 2), max(th, 3), WHITE, -1, cv2.LINE_AA)
        d, w = pitch.get("penalty_area_depth_m", 9.0), pitch.get("penalty_area_width_m", 20.0)
        y0, y1 = (W - w) / 2, (W + w) / 2
        for x_line, x_front in ((0, d), (L, L - d)):
            line((x_front, y0), (x_front, y1))
            line((x_line, y0), (x_front, y0))
            line((x_line, y1), (x_front, y1))
        gw = pitch.get("goal_width_m", 5.0)                      # goals: nets behind the goal lines
        for x_line, sign in ((0, -1), (L, 1)):
            (xa, ya), (xb, yb) = self.px(x_line, (W - gw) / 2), self.px(x_line + sign * 1.6, (W + gw) / 2)
            cv2.rectangle(img, (min(xa, xb), ya), (max(xa, xb), yb), (215, 215, 215), 1, cv2.LINE_AA)
            for k in range(1, 6):
                yy = ya + (yb - ya) * k // 6
                cv2.line(img, (min(xa, xb), yy), (max(xa, xb), yy), (190, 190, 190), 1, cv2.LINE_AA)
        return img


def text_center(img, text, center, scale, color, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(img, text, (center[0] - tw // 2, center[1] + th // 2), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


SHIFT = 4                                                     # fixed-point bits of the sub-pixel drawing


def circle_f(img, cx, cy, r, color, thickness):
    k = 1 << SHIFT
    cv2.circle(img, (int(round(cx * k)), int(round(cy * k))), int(round(r * k)), color, thickness, cv2.LINE_AA, SHIFT)


# the digit "1" of the goalkeeper as a shape (a stem and a slanted flag), so that it moves with sub-pixel accuracy
# together with its circle; cv2.putText can only place text at whole pixels and the digit would shake inside the circle
ONE_STEM = np.array([[-1.8, -7.6], [1.8, -7.6], [1.8, 7.6], [-1.8, 7.6]])
ONE_FLAG = np.array([[-1.8, -7.6], [1.8, -7.6], [-1.8, -3.4], [-5.4, -1.4], [-5.4, -4.6]])
ONE_FOOT = np.array([[-4.4, 5.6], [4.4, 5.6], [4.4, 7.6], [-4.4, 7.6]])


def draw_one(img, cx, cy, k=1.0):
    for shape in (ONE_STEM, ONE_FLAG, ONE_FOOT):
        pts = np.round((shape * k + np.array([cx, cy])) * (1 << SHIFT)).astype(np.int32)
        cv2.fillPoly(img, [pts], WHITE, cv2.LINE_AA, SHIFT)


def draw_player(img, canvas, x, y, team, gk, alpha, colors):
    cx, cy = canvas.pxf(x, y)
    r = 0.9 * canvas.s
    layer = img if alpha >= 0.98 else img.copy()
    circle_f(layer, cx + 2, cy + 3, r, (20, 60, 25), -1)
    circle_f(layer, cx, cy, r, colors[team], -1)
    circle_f(layer, cx, cy, r, WHITE, 2)
    if gk:
        draw_one(layer, cx, cy, canvas.s / 18.0)
    if layer is not img:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def draw_ball(img, canvas, x, y):
    """A small football (always the same, the possession is written in the header)."""
    cx, cy = canvas.pxf(x, y)
    circle_f(img, cx, cy, 10, WHITE, -1)
    circle_f(img, cx, cy, 10, BLACK, 3)
    pent = np.array([[cx + 4.8 * np.cos(a), cy + 4.8 * np.sin(a)] for a in np.linspace(-np.pi / 2, 3 * np.pi / 2, 6)[:-1]])
    cv2.fillConvexPoly(img, np.round(pent * (1 << SHIFT)).astype(np.int32), BLACK, cv2.LINE_AA, SHIFT)


def draw_header(img, canvas, t, half, shares, state_text, colors, names, state_color):
    w = canvas.width
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, names[0], (18, 27), font, 0.75, WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, (18 + cv2.getTextSize(names[0], font, 0.75, 1)[0][0] + 16, 20), 7, colors[0], -1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(names[1], font, 0.75, 1)
    cv2.putText(img, names[1], (w - 18 - tw, 27), font, 0.75, WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, (w - 18 - tw - 16, 20), 7, colors[1], -1, cv2.LINE_AA)
    text_center(img, f"{mmss(t)}   half {half}", (w // 2, 22), 0.6, (200, 200, 200), 1)
    text_center(img, state_text, (w // 2, 46), 0.7, state_color, 1)
    x0, x1, y0, y1 = 18, w - 18, 66, 82
    split = int(x0 + (x1 - x0) * shares[0])
    cv2.rectangle(img, (x0, y0), (split, y1), colors[0], -1)
    cv2.rectangle(img, (split, y0), (x1, y1), colors[1], -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 1)
    cv2.putText(img, f"{shares[0] * 100:.0f}%", (x0 + 6, y1 - 4), font, 0.5, WHITE, 1, cv2.LINE_AA)
    label = f"{shares[1] * 100:.0f}%"
    (tw, _), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.putText(img, label, (x1 - 6 - tw, y1 - 4), font, 0.5, WHITE, 1, cv2.LINE_AA)
    label_txt = "possession, last 60 s"
    (lw, lh), _ = cv2.getTextSize(label_txt, font, 0.42, 1)
    xa, ya = w // 2 - lw // 2 - 6, y0 + 1
    roi = img[ya:y1, xa:xa + lw + 12]
    cv2.addWeighted(np.full_like(roi, 30), 0.6, roi, 0.4, 0, roi)
    cv2.putText(img, label_txt, (w // 2 - lw // 2, y1 - 4), font, 0.42, (235, 235, 235), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder of a match (teams.csv, possession.csv, ideally stats.json)")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--start", default=None, help="start in the recording, e.g. 12:30 (default: 2 min into the first half)")
    ap.add_argument("--duration", type=float, default=30.0, help="length in seconds")
    ap.add_argument("--fps", type=int, default=30, help="frames per second of the animation (positions in between are interpolated)")
    ap.add_argument("--scale", type=int, default=18, help="pixels per metre")
    ap.add_argument("--max-per-team", type=int, default=6, help="circles per team, the goalkeeper included")
    ap.add_argument("--vmax", type=float, default=8.5, help="fastest a circle may move, m/s")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="smoothing of the paths of the circles in seconds (0 = none); standing players get about twice as much, "
                         "sprinting ones half: the detections shake by about 0.5 m")
    ap.add_argument("--deadzone", type=float, default=0.5,
                    help="a circle stays where it is while the detection of the player only shakes within this many metres "
                         "(0 = off); raise it if the log says the standing circles shake by more than about 11 cm")
    ap.add_argument("--colors", choices=["reference", "kit"], default="reference",
                    help="reference = blue and red, kit = the colours of the kits found by dual_teams.py")
    ap.add_argument("--names", default="Team 0,Team 1")
    ap.add_argument("--out")
    ap.add_argument("--gif", action="store_true", help="also make a small GIF (for the README)")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    names = a.names.split(",")
    poss = load_possession(folder)
    times = np.array([r[0] for r in poss])
    t0 = parse_time(a.start) if a.start else float(times.min()) + 120
    half = poss[min(int(np.searchsorted(times, t0)), len(poss) - 1)][1]
    t1 = t0 + a.duration
    warm = 4.0
    idx = [i for i, r in enumerate(poss) if t0 <= r[0] <= t1 and r[1] == half]
    warm_idx = [i for i, r in enumerate(poss) if t0 - warm <= r[0] < t0 and r[1] == half]
    if not idx:
        raise SystemExit(f"No frames between {mmss(t0)} and {mmss(t1)} (the recording has {mmss(times.min())} - {mmss(times.max())})")
    dt = float(np.median(np.diff(times[:2000])))
    data_fps = 1 / dt
    sub = max(int(round(a.fps / data_fps)), 1)

    colors = dict(PALETTES["reference"])
    cf = folder / "teams_colors.json"
    if a.colors == "kit" and cf.exists():
        c = json.loads(cf.read_text())
        colors = {0: tuple(c["team_0_bgr"]), 1: tuple(c["team_1_bgr"])}

    print(f"window {mmss(t0)} - {mmss(t1)} (half {half}), {len(idx)} data frames at {data_fps:.0f} fps -> {a.fps} fps; reading players ...")
    frames = load_players(folder, t0 - warm, t1, half, L, W)
    goals, source = own_goals(folder, half, frames, L)
    print(f"own goals: team 0 defends x={goals[0]:.0f}, team 1 defends x={goals[1]:.0f} (from {source})")
    puppets = PuppetSystem(goals, L, W, a.max_per_team, a.vmax)
    puppets.warm_until = t0                                     # the warm-up before the window: the circles settle instantly

    states, balls = [], []
    ball_s, last_known_t = None, -9.0
    for i in warm_idx + idx:
        t = poss[i][0]
        puppets.step(t, frames.get(t, []))
        if i in idx:
            states.append(puppets.snapshot(t))
            bx, by = poss[i][2], poss[i][3]
            if not np.isnan(bx):
                ball_s = (bx, by) if ball_s is None or np.hypot(bx - ball_s[0], by - ball_s[1]) > 8 else \
                    (ball_s[0] + 0.6 * (bx - ball_s[0]), ball_s[1] + 0.6 * (by - ball_s[1]))
                last_known_t = t
            balls.append(ball_s if ball_s is not None and t - last_known_t <= 0.5 else None)

    for team in (0, 1):
        gk_x = [next((v[0] for k, v in st.items() if k[0] == team and v[3]), np.nan) for st in states]
        flagged = sum(1 for dets in frames.values() if any(g and tm == team for _, _, tm, g in dets))
        runs = np.array(puppets.runs[team])
        print(f"team {team}: goalkeeper kit detected in {flagged} of {len(frames)} frames, the '1' circle is on average "
              f"{np.nanmean(np.abs(np.array(gk_x) - goals[team])):.1f} m from its own goal; "
              + (f"idle circles ran to a new player {len(runs)}x (median {np.median(runs):.1f} m, longest {runs.max():.1f} m)"
                 if len(runs) else "no circle had to run to a new player"))
    states = smooth_states(states, dt, a.smooth, dead_m=a.deadzone)
    raw_ball = np.array([[b[0], b[1]] if b is not None else [np.nan, np.nan] for b in balls])
    hidden = int(np.sum(np.isnan(raw_ball[1:, 0]) & ~np.isnan(raw_ball[:-1, 0])))
    kn = ~np.isnan(raw_ball[:, 0])
    jumps = int(np.sum(np.hypot(*np.diff(raw_ball, axis=0).T)[kn[1:] & kn[:-1]] > 5.0)) if kn.sum() > 2 else 0
    print(f"ball: hidden {hidden}x (no position for more than 0.5 s), jumps of more than 5 m between two frames {jumps}x")
    ball_arr = smooth_segments(np.array([[b[0], b[1]] if b is not None else [np.nan, np.nan] for b in balls]), 0.4 * a.smooth / dt)
    balls = [None if np.isnan(r[0]) else (float(r[0]), float(r[1])) for r in ball_arr]

    speeds = []
    for key in {k for st in states for k in st}:
        pts = np.array([st[key][:2] for st in states if key in st])
        if len(pts) > 3:
            speeds += list(np.hypot(*np.diff(pts, axis=0).T) / dt)
    print(f"circles after smoothing: median speed {np.median(speeds):.1f} m/s, 99th percentile {np.percentile(speeds, 99):.1f} m/s, "
          f"fastest {max(speeds):.1f} m/s (players run up to about 6-8 m/s)")
    shake, share = shake_cm(states, dt)
    if shake is not None:
        print(f"circles that stand still for at least 3 s ({share:.0%} of the time): they shake by {shake:.1f} cm "
              f"(one pixel = {100 / a.scale:.1f} cm; below about {200 / a.scale:.0f} cm the shake is not visible)")

    team_arr = np.array([r[4] for r in poss])
    c0, c1 = np.cumsum(team_arr == 0), np.cumsum(team_arr == 1)
    win = int(round(60 / dt))

    canvas = Canvas(L, W, pitch, a.scale)
    out = Path(a.out).expanduser() if a.out else folder / f"overlay_{mmss(t0).replace(':', 'm')}_{int(a.duration)}s.mp4"
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (canvas.width, canvas.height))
    preview_at = len(idx) // 3
    n_frames = 0
    for k, i in enumerate(idx):
        t, _, _, _, pteam, state = poss[i]
        nxt = k + 1 if k + 1 < len(idx) else k
        for s in range(sub):
            f = s / sub if nxt != k else 0.0
            ts = t + f * dt
            img = canvas.background.copy()
            circles = lerp_states(states[k], states[nxt], f)
            for (team, _), (x, y, al, gk) in sorted(circles.items(), key=lambda kv: kv[1][1]):
                draw_player(img, canvas, x, y, team, gk, min(max(al, 0.0), 1.0), colors)
            b0, b1 = balls[k], balls[nxt]
            ball = None
            if b0 is not None and b1 is not None and np.hypot(b1[0] - b0[0], b1[1] - b0[1]) < 8:
                ball = (b0[0] + f * (b1[0] - b0[0]), b0[1] + f * (b1[1] - b0[1]))
            elif b0 is not None:
                ball = b0
            if ball is not None and -2.0 <= ball[0] <= L + 2.0 and -2.0 <= ball[1] <= W + 2.0:
                draw_ball(img, canvas, ball[0], ball[1])
            j = max(i - win, 0)
            n0, n1 = c0[i] - c0[j], c1[i] - c1[j]
            shares = (n0 / (n0 + n1), n1 / (n0 + n1)) if n0 + n1 > 0 else (0.5, 0.5)
            state_text = (f"{names[pteam]} has the ball" if pteam in (0, 1) else
                          {"out": "ball out of play", "still": "ball lying still", "restart": "restart after a goal"}.get(state, "no possession"))
            draw_header(img, canvas, ts, half, shares, state_text, colors, names,
                        tuple(int(0.45 * c + 0.55 * 255) for c in colors[pteam]) if pteam in (0, 1) else (170, 220, 255))
            vw.write(img)
            n_frames += 1
            if k == preview_at and s == 0:
                cv2.imwrite(str(out.with_name("overlay_preview.png")), img)
    vw.release()

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
        if a.gif:
            gif = out.with_suffix(".gif")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf",
                            "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", str(gif)], check=True)
            print(f"Saved {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    else:
        tmp.rename(out)
        print("ffmpeg not found: the video is in mp4v format (may not play in a browser)")
    print(f"Saved {out} ({n_frames} frames) and {out.with_name('overlay_preview.png')}")


if __name__ == "__main__":
    main()
