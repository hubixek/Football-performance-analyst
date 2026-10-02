#!/usr/bin/env python3
"""Team assignment for dual-lens detections (jersey colour, K-means) with goalkeepers.

Same method as teams.py (shirt colour without turf pixels, two clusters, outliers = -1,
players in the referee's colour relabeled as referee), but every detection is cut from the
lens image it was detected in (top or bottom half of the frame).

Goalkeepers wear their own kits, which belong to neither cluster. They are found at the
kick-off of each half: the players standing deepest on the left and on the right are the
goalkeepers, their kit colours are taken from the first --gk-kickoff-s seconds. The team of a
goalkeeper is the team standing in that half of the pitch at the kick-off (sides swap in the
second half). Everybody wearing a goalkeeper kit in the match gets the team of that goalkeeper.

Input:  CSV from dual_detect.py + the dual-lens video.
Output: the same CSV with columns `team` (0, 1, -1; empty for referee/ball) and `gk` (1 for a
        goalkeeper), <out>_colors.json with the team colours, <out>_rgb.npz with the jersey
        colours (--reuse-colors repeats the assignment without reading the video again).

Usage:
  python dual_teams.py ~/football/analysis/mecz1_dual/detections.csv \
      --video ~/football/videos/mecz1_dual.mp4 --out ~/football/analysis/mecz1_dual/teams.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np



def grass_color(img):
    """Median Lab colour of the frame - most of a football frame is the pitch surface."""
    small = cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA)[18:]          # lower 2/3
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


SHIRT = (0.15, 0.5)           # part of the box height used for the shirt colour (a narrower band 0.15-0.38 was tried: worse)


def jersey_color(img, box, grass=None, grass_dist=20.0):
    """Median Lab colour of the shirt area, without pixels similar to the pitch surface."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    ya, yb = int(y1 + SHIRT[0] * h), int(y1 + SHIRT[1] * h)
    crop = img[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if grass is not None:
        keep = np.linalg.norm(lab - grass, axis=1) > grass_dist
        if keep.sum() >= 5:
            lab = lab[keep]
    return np.median(lab, axis=0)


def lab_to_bgr(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def cdist(a, b):
    """Colour distance in Lab; lightness counts less (sun and shade change it most)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.sqrt(0.5 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


APP_SHIRT = (0.15, 0.40)                                   # the shirt band of the appearance descriptor
APP_BANDS = ((0.15, 0.32), (0.32, 0.50), (0.52, 0.72))     # upper shirt, lower shirt, shorts


def appearance(img, box):
    """Appearance of a player, 26 numbers: median Lab of the shirt (3), median Lab of three body bands (9), hue histogram weighted
    by saturation plus mean saturation and brightness (14). On the crops of a night match judged by eye the hue histogram told the
    teams apart in 94% of the cases, the median colour used so far in 79-87% (team_features_probe.py)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    def band(b):
        c = img[max(int(y1 + b[0] * h), 0):max(int(y1 + b[1] * h), 0), max(int(x1 + 0.25 * w), 0):max(int(x2 - 0.25 * w), 0)]
        return c if c.shape[0] >= 2 and c.shape[1] >= 2 else None

    top = band(APP_SHIRT)
    if top is None:
        return None
    med = np.median(cv2.cvtColor(top, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hist, _ = np.histogram(hsv[:, 0], bins=12, range=(0, 180), weights=hsv[:, 1].astype(np.float64) + 1.0)
    hv = np.concatenate([hist / max(hist.sum(), 1e-6), [hsv[:, 1].mean() / 255, hsv[:, 2].mean() / 255]])
    bands = []
    for b in APP_BANDS:
        c = band(b)
        bands.append(np.median(cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0) if c is not None else med)
    return np.concatenate([med, np.concatenate(bands), hv]).astype(np.float32)


def fit_lda(X, y, shrink=0.3):
    """Linear discriminant on standardised numbers with a shrunk covariance; the score is positive for class 1."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    c0, c1 = Z[y == 0].mean(0), Z[y == 1].mean(0)
    S = np.cov(np.vstack([Z[y == 0] - c0, Z[y == 1] - c1]).T)
    d = S.shape[0]
    S = (1 - shrink) * S + shrink * np.eye(d) * np.trace(S) / d
    w = np.linalg.solve(S, c1 - c0)
    return mu, sd, w, float(-0.5 * w @ (c0 + c1))


def lda_score(model, X):
    mu, sd, w, b = model
    return ((X - mu) / sd) @ w + b


def read_colors(video, rows, by_frame, grass_dist, with_desc=False):
    """Jersey colour (and, with with_desc, the appearance descriptor) of every player/referee detection, reading the video in order."""
    colors, descs = {}, {}
    cap = cv2.VideoCapture(str(video))
    pos = -1
    frames = sorted(by_frame)
    for k, fr in enumerate(frames):
        if fr - pos > 60 or fr < pos:          # big jump (e.g. half time): seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            pos = fr
        while pos < fr:                        # small step: grab frames without decoding
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            break
        h = img.shape[0] // 2
        halves = {"top": img[:h], "bottom": img[h:2 * h]}
        grass = {c: grass_color(im) for c, im in halves.items()}
        for i in by_frame[fr]:
            r = rows[i]
            box = [float(r[k2]) for k2 in ("x1", "y1", "x2", "y2")]
            c = jersey_color(halves[r["cam"]], box, grass[r["cam"]], grass_dist)
            if c is not None:
                colors[i] = c
                if with_desc:
                    d = appearance(halves[r["cam"]], box)
                    if d is not None:
                        descs[i] = d
        if (k + 1) % 1000 == 0:
            print(f"colours: {k + 1}/{len(frames)} frames", flush=True)
    cap.release()
    return (colors, descs) if with_desc else colors


def kickoff_window(rows, kickoff_s):
    """Per half: the detections of the first kickoff_s seconds, grouped by frame."""
    t0, frames = {}, defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        if r["class"] != "player":
            continue
        h, t = int(r["half"]), float(r["time_s"])
        t0[h] = min(t0.get(h, t), t)
    for i, r in enumerate(rows):
        if r["class"] == "player" and float(r["time_s"]) <= t0[int(r["half"])] + kickoff_s:
            frames[int(r["half"])][int(r["frame"])].append(i)
    return frames


def goalkeeper_kits(rows, colors, window, L, zone):
    """Kit colour of the goalkeeper on each side of each half: (half, 'left'/'right') -> Lab."""
    refs = {}
    for h, frames in window.items():
        left, right = [], []
        for idxs in frames.values():
            xs = [(float(rows[i]["x_m"]), i) for i in idxs if i in colors]
            if not xs:
                continue
            x, i = min(xs)
            if x < zone:
                left.append(colors[i])
            x, i = max(xs)
            if x > L - zone:
                right.append(colors[i])
        if len(left) >= 5:
            refs[(h, "left")] = np.median(left, axis=0)
        if len(right) >= 5:
            refs[(h, "right")] = np.median(right, axis=0)
    return refs


def assign(rows, colors, L=56.0, outlier=2.5, min_limit=15.0, ref_fix=True, ref_max=20.0, ref_max_share=0.10,
           use_gk=True, gk_kickoff_s=15.0, gk_zone=7.0, gk_radius=22.0, gk_max_share=0.30):
    ref_idx = [i for i in colors if rows[i]["class"] == "referee"]
    player_idx = [i for i in colors if rows[i]["class"] == "player"]

    # 1) goalkeeper kits from the kick-off of each half
    window = kickoff_window(rows, gk_kickoff_s) if use_gk else {}
    kits = goalkeeper_kits(rows, colors, window, L, gk_zone) if use_gk else {}
    provisional = set()
    if kits:
        kit_colors = np.float32(list(kits.values()))
        for i in player_idx:
            if float(np.min(cdist(kit_colors, colors[i]))) <= gk_radius:
                provisional.add(i)
    elif use_gk:
        print("WARNING: no goalkeepers found at the kick-off - goalkeepers are not treated separately")

    # 2) referee colour (hard cap on its range)
    ref_center, ref_limit = None, 0.0
    if ref_fix and len(ref_idx) >= 20:
        ref_data = np.float32([colors[i] for i in ref_idx])
        ref_center = np.median(ref_data, axis=0)
        ref_limit = min(max(outlier * np.median(np.linalg.norm(ref_data - ref_center, axis=1)), min_limit), ref_max)

    def ref_distance(i):
        return float(np.linalg.norm(colors[i] - ref_center)) if ref_center is not None else np.inf

    # early safety: a goalkeeper kit that many players wear is not a goalkeeper kit
    if provisional and len(provisional) > gk_max_share * max(len(player_idx), 1):
        print(f"WARNING: {len(provisional)} of {len(player_idx)} players ({100 * len(provisional) / len(player_idx):.0f}%) "
              "have a goalkeeper-like colour - the goalkeeper kit is not distinct from the teams, "
              "goalkeeper handling switched off")
        provisional, kits = set(), {}

    def fit_teams(excluded):
        idx = [i for i in player_idx if i not in excluded and ref_distance(i) > ref_limit]
        if len(idx) < 10:
            raise SystemExit("Too few player detections to split into teams")
        data = np.float32([colors[i] for i in idx])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
        cv2.setRNGSeed(0)                      # the same result on every run
        _, labels, cen = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        if cen[0][0] > cen[1][0]:              # team 0 = the darker kit, so the numbers stay stable
            cen, labels = cen[::-1].copy(), 1 - labels
        dist = cdist(data, cen[labels])
        return cen, [max(outlier * np.median(dist[labels == t]), min_limit) for t in (0, 1)]

    def resolve(cen):
        """Provisional goalkeepers that are clearly closer to a goalkeeper kit than to their team."""
        out = set()
        for i in provisional:
            d_kit = float(np.min(cdist(kit_colors, colors[i])))
            d_team = float(np.min(cdist(cen, colors[i])))
            if d_kit < 0.6 * d_team:
                out.add(i)
        return out

    # 3) two teams from the outfield players; iterate, because removing goalkeepers moves the
    #    team colours slightly and that changes who is clearly a goalkeeper
    gk = set(provisional)
    for _ in range(3):
        centers, limits = fit_teams(gk)
        new = resolve(centers) if provisional else set()
        if new == gk:
            break
        gk = new
    else:
        centers, limits = fit_teams(gk)
    if gk and len(gk) > gk_max_share * max(len(player_idx), 1):
        print(f"WARNING: {len(gk)} of {len(player_idx)} players look like goalkeepers - "
              "goalkeeper handling switched off")
        gk = set()
        centers, limits = fit_teams(gk)

    team, relabeled = {}, set()
    for i in player_idx:
        if i in gk:
            continue
        d = cdist(centers, colors[i])
        t = int(np.argmin(d))
        if ref_distance(i) <= ref_limit and ref_distance(i) < 0.5 * d[t]:
            relabeled.add(i)
        team[i] = t if d[t] <= limits[t] else -1
    if len(relabeled) > ref_max_share * max(len(player_idx), 1):
        print(f"WARNING: referee correction would relabel {len(relabeled)} of {len(player_idx)} players "
              f"({100 * len(relabeled) / len(player_idx):.0f}%) - the referee colour is not distinct, "
              "correction switched off")
        relabeled = set()
    for i in relabeled:
        team.pop(i, None)

    # 5) teams of the goalkeepers: the team standing in that half of the pitch at the kick-off
    gk_teams = {}
    if gk:
        defender = {}
        for h, frames in window.items():
            votes = {"left": Counter(), "right": Counter()}
            for idxs in frames.values():
                for i in idxs:
                    x = float(rows[i]["x_m"])
                    if team.get(i) in (0, 1):
                        if gk_zone + 1 < x < L / 2 - 1:
                            votes["left"][team[i]] += 1
                        elif L / 2 + 1 < x < L - gk_zone - 1:
                            votes["right"][team[i]] += 1
            for side in ("left", "right"):
                if votes[side]:
                    defender[(h, side)] = votes[side].most_common(1)[0][0]
        by_team = defaultdict(list)
        for key, col in kits.items():
            if key in defender:
                by_team[defender[key]].append(col)
        kit_of = {t: np.median(v, axis=0) for t, v in by_team.items()}
        distinct = len(kit_of) == 2 and float(cdist(kit_of[0], kit_of[1])) > 15.0
        for i in gk:
            h, x = int(rows[i]["half"]), float(rows[i]["x_m"])
            if distinct:
                t = int(np.argmin([float(cdist(kit_of[k], colors[i])) for k in (0, 1)]))
            else:
                t = defender.get((h, "left" if x < L / 2 else "right"), -1)
            gk_teams[i] = t
        team.update(gk_teams)
    return team, relabeled, gk, centers, ref_center, limits


def find_kickoffs(rows, ball_path, L=56.0, W=32.4, radius=1.5, still_m=0.8, still_s=0.8, min_players=8, max_gap_s=1.0,
                  circle_r=4.75, line_m=0.5, calm_m=0.8, min_s=1.0):
    """Kick-offs (start of a half and after every goal). Two ways to recognise one:
      - by the ball: it lies still on the centre spot and the players stand in two groups, one on each half,
      - by the players alone (the ball is often not seen on the centre spot): at most two players in the centre
        circle (the ones taking the kick-off), everybody else clearly on one half, both halves about equally full,
        and the players hardly move for at least min_s seconds.
    Returns [(half, time, [frames], how)]."""
    ball = {}
    if ball_path is not None:
        with open(ball_path) as f:
            for r in csv.DictReader(f):
                if r["x_m"]:
                    ball[int(r["frame"])] = (float(r["time_s"]), int(r["half"]), float(r["x_m"]), float(r["y_m"]))
    people, info = defaultdict(list), {}
    for i, r in enumerate(rows):
        if r["class"] == "player":
            x, y = float(r["x_m"]), float(r["y_m"])
            f = int(r["frame"])
            info[f] = (float(r["time_s"]), int(r["half"]))
            if 0 <= x <= L and 0 <= y <= W:
                people[f].append((x, y))
    frames = sorted(info)
    ftimes = np.array([info[f][0] for f in frames])

    def split_ok(pts):
        if len(pts) < min_players:
            return False
        p = np.array(pts)
        in_circle = np.hypot(p[:, 0] - L / 2, p[:, 1] - W / 2) < circle_r
        rest = p[~in_circle]
        if in_circle.sum() > 2 or np.any(np.abs(rest[:, 0] - L / 2) < line_m):
            return False
        left, right = int(np.sum(rest[:, 0] < L / 2)), int(np.sum(rest[:, 0] > L / 2))
        return min(left, right) >= 3 and abs(left - right) <= 3

    # by the ball
    good = {}
    btimes = np.array([ball[f][0] for f in sorted(ball)]) if ball else np.array([])
    bxy = np.array([ball[f][2:] for f in sorted(ball)]) if ball else np.zeros((0, 2))
    for k, f in enumerate(sorted(ball)):
        t, h, bx, by = ball[f]
        if np.hypot(bx - L / 2, by - W / 2) > radius:
            continue
        j = int(np.searchsorted(btimes, t - still_s))
        if np.max(np.hypot(*(bxy[j:k + 1] - bxy[k]).T)) > still_m:
            continue
        pts = people.get(f, [])
        xs = np.array([p[0] for p in pts])
        left, right = int(np.sum(xs < L / 2 - 0.5)), int(np.sum(xs > L / 2 + 0.5))
        if left + right >= min_players and min(left, right) >= 3 and abs(left - right) <= 3:
            good[f] = "ball"
    # by the players alone: the formation, and nobody moves (positions 1 s apart match within calm_m)
    for k, f in enumerate(frames):
        if f in good or not split_ok(people.get(f, [])):
            continue
        t = info[f][0]
        j = int(np.searchsorted(ftimes, t - 1.0))
        g = frames[j]
        if g == f or info[g][1] != info[f][1] or not split_ok(people.get(g, [])):
            continue
        a, b = np.array(people[f]), np.array(people[g])
        d = np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]), axis=1)
        if np.median(d) <= calm_m:
            good[f] = "formation"
    events, cur = [], []
    for f in sorted(good):
        t, h = info[f]
        if cur and (t - cur[-1][1] > max_gap_s or h != cur[-1][2]):
            events.append(cur)
            cur = []
        cur.append((f, t, h, good[f]))
    if cur:
        events.append(cur)
    out = []
    for ev in events:
        if ev[-1][1] - ev[0][1] < (0.5 if any(e[3] == "ball" for e in ev) else min_s):
            continue
        how = "ball" if any(e[3] == "ball" for e in ev) else "formation"
        out.append((ev[0][2], float(np.mean([e[1] for e in ev])), [e[0] for e in ev], how))
    return out


LAST_DECISIONS = []     # what kickoff_teams decided about every kick-off: (half, time_s, how, reason or "kept")


def kickoff_teams(rows, colors, events, player_idx, gk, L=56.0, tau_s=600.0, gk_zone=8.0, centre_m=1.5,
                  min_side=8, per_event=12, outlier=2.5, min_limit=15.0):
    """Teams learnt from the kick-offs: at a kick-off every team stands on its own half, so the players on each side
    are labelled examples of the two kits. Sides are fixed within a half and swap at half time. Every player detection
    gets the team whose kit it is closest to, compared with the examples from the kick-offs NEAREST IN TIME (weight
    exp(-|dt| / tau_s)), so a kit that looks different as the light changes (sunset) is still recognised.
    Returns (team, centers, limits, info) or None if the kick-offs are not enough."""
    by_frame = defaultdict(list)
    for i in player_idx:
        by_frame[int(rows[i]["frame"])].append(i)
    per_ev = []                                                        # per kick-off: (half, time, how, [(side, colour)])
    for ev in events:
        h, t, frs = ev[0], ev[1], ev[2]
        how = ev[3] if len(ev) > 3 else "ball"
        pick = [frs[int(k)] for k in np.linspace(0, len(frs) - 1, min(per_event, len(frs)))]
        smp = []
        for f in pick:
            for i in by_frame.get(f, ()):
                x = float(rows[i]["x_m"])
                if i in gk or abs(x - L / 2) < centre_m or x < gk_zone or x > L - gk_zone:
                    continue
                smp.append((0 if x < L / 2 else 1, colors[i]))
        if sum(1 for sd, _ in smp if sd == 0) >= 3 and sum(1 for sd, _ in smp if sd == 1) >= 3:
            per_ev.append((h, t, how, smp))
    # a kick-off is kept only if its two sides differ clearly in colour (a real kick-off: two kits apart) and the side
    # of each kit agrees with the other kick-offs of that half (checked against the kick-offs found by the ball first)
    def side_medians(smp):
        return [np.median(np.float32([c for sd, c in smp if sd == k]), axis=0) for k in (0, 1)]
    def spread(smp, med):
        return float(np.median([cdist(med[sd], c) for sd, c in smp]))
    kept, dropped = [], 0
    LAST_DECISIONS.clear()
    for h in sorted({e[0] for e in per_ev}):
        evs = sorted([e for e in per_ev if e[0] == h], key=lambda e: (e[2] != "ball", e[1]))
        ref = None
        for e in evs:
            med = side_medians(e[3])
            sep = float(cdist(med[0], med[1]))
            sp = spread(e[3], med)
            if sep < 1.5 * sp:
                dropped += 1
                LAST_DECISIONS.append((h, e[1], e[2], f"the two sides look alike (colour distance {sep:.1f} is below 1.5 x the spread {sp:.1f})"))
                continue                                               # the two sides do not look like two kits
            if ref is not None:
                keep = float(cdist(ref[0], med[0]) + cdist(ref[1], med[1]))
                swap = float(cdist(ref[0], med[1]) + cdist(ref[1], med[0]))
                if swap < keep:
                    dropped += 1
                    LAST_DECISIONS.append((h, e[1], e[2], "the kits stand on the other sides than at the first kick-off of the half"))
                    continue                                           # the kits stand on the wrong sides: not a kick-off
            else:
                ref = med
            kept.append(e)
            LAST_DECISIONS.append((h, e[1], e[2], "kept"))
    samples = [(h, t, sd, c) for h, t, how, smp in kept for sd, c in smp]  # (half, time, side, colour)
    halves = sorted({s[0] for s in samples})
    if not halves:
        return None
    cent = {}
    for h in halves:
        for side in (0, 1):
            c = [s[3] for s in samples if s[0] == h and s[2] == side]
            if len(c) < min_side:
                return None
            cent[(h, side)] = np.median(np.float32(c), axis=0)
    # side of team A in every half: A = left side of the first half; in the other halves by colour (normally swapped)
    h0 = halves[0]
    side_of_a = {h0: 0}
    for h in halves[1:]:
        keep = float(cdist(cent[(h0, 0)], cent[(h, 0)]) + cdist(cent[(h0, 1)], cent[(h, 1)]))
        swap = float(cdist(cent[(h0, 0)], cent[(h, 1)]) + cdist(cent[(h0, 1)], cent[(h, 0)]))
        side_of_a[h] = 0 if keep < swap else 1
    lab = np.array([0 if s[2] == side_of_a[s[0]] else 1 for s in samples])  # 0 = team A, 1 = team B
    col = np.float32([s[3] for s in samples])
    ts = np.array([s[1] for s in samples])
    flipped = bool(np.median(col[lab == 0][:, 0]) > np.median(col[lab == 1][:, 0]))
    if flipped:
        lab = 1 - lab                                                  # team 0 = the darker kit, as before
    # which team defends the left goal in every half (the team standing on the left half at the kick-offs)
    left_team = {h: (0 if side_of_a[h] == 0 else 1) ^ int(flipped) for h in halves}
    centers = np.float32([np.median(col[lab == t], axis=0) for t in (0, 1)])
    limits = [max(outlier * float(np.median(cdist(col[lab == t], centers[t]))), min_limit) for t in (0, 1)]
    # time-local kit colours: weighted medians approximated by weighted means in 30 s steps
    team = {}
    times = np.array([float(rows[i]["time_s"]) for i in player_idx])
    order = np.argsort(times)
    step = 30.0
    for b0 in np.arange(times.min(), times.max() + step, step):
        sel = order[(times[order] >= b0) & (times[order] < b0 + step)]
        if not len(sel):
            continue
        w = np.exp(-np.abs(ts - (b0 + step / 2)) / tau_s)
        loc = [(col[lab == t] * w[lab == t, None]).sum(0) / w[lab == t].sum() for t in (0, 1)]
        for j in sel:
            i = player_idx[j]
            if i in gk:
                continue
            d = [float(cdist(loc[t], colors[i])) for t in (0, 1)]
            t = int(np.argmin(d))
            team[i] = t if d[t] <= 1.3 * limits[t] else -1
    info = {"events": len(kept), "by_ball": sum(1 for e in kept if e[2] == "ball"), "dropped": dropped, "samples": len(samples), "per_half": {h: sum(1 for s in samples if s[0] == h) for h in halves},
            "swapped_at_half_time": {h: side_of_a[h] != side_of_a[h0] for h in halves[1:]}, "left_team": left_team}
    return team, centers, limits, info


def gk_by_side(rows, team, gk, left_team, L):
    """A goalkeeper belongs to the team that defends the goal he stands at (left_team: the team on the left half in every half)."""
    n = 0
    for i in gk:
        h = int(rows[i]["half"])
        if h in left_team:
            lt = left_team[h]
            new_t = lt if float(rows[i]["x_m"]) < L / 2 else 1 - lt
            n += team.get(i) != new_t
            team[i] = new_t
    return n


def clf_teams(rows, colors, descs, events, player_idx, gk, L=56.0, gk_zone=8.0, centre_m=1.5, per_event=12, min_side=8, outlier=2.5, min_limit=15.0,
              self_train=0, pseudo_margin=1.5, pseudo_cap=3000):
    """Teams from a classifier trained on the kick-offs. At a kick-off every team stands on its own half, so the players on the two
    sides are labelled examples (the labels are right whatever the colours look like). A linear discriminant learns the appearance
    descriptor of the two teams from them and classifies every detection of the match.
    Kick-offs used: those with the ball on the centre spot and the start of every half (the formation of the players alone can be a
    free kick set-up with wrong sides); if there are fewer than 2, all of them. After half time the teams change ends: when the
    kick-offs of both halves exist, the classifier trained on one half decides how the sides of the other half belong to the teams.
    Returns (team, score, centers, limits, info) or None; score is positive towards team 1, team 0 is the team with the darker shirts."""
    by_frame = defaultdict(list)
    for i in player_idx:
        if i in descs:
            by_frame[int(rows[i]["frame"])].append(i)
    half_t0 = {}
    for r in rows:
        h, t = int(r["half"]), float(r["time_s"])
        half_t0[h] = min(half_t0.get(h, 1e18), t)
    good = [e for e in events if e[3] == "ball" or e[1] - half_t0.get(e[0], 0.0) < 90.0]
    if len(good) < 2:
        good = list(events)
    samples = []                                                    # (half, side, index)
    used = Counter()
    for ev in good:
        h, frs = ev[0], ev[2]
        n_before = len(samples)
        for f in [frs[int(k)] for k in np.linspace(0, len(frs) - 1, min(per_event, len(frs)))]:
            for i in by_frame.get(f, ()):
                x = float(rows[i]["x_m"])
                if i in gk or abs(x - L / 2) < centre_m or x < gk_zone or x > L - gk_zone:
                    continue
                samples.append((h, 0 if x < L / 2 else 1, i))
        if len(samples) > n_before:
            used[h] += 1
    halves = sorted({s[0] for s in samples})
    if not halves:
        return None
    X_all = {s[2]: descs[s[2]] for s in samples}
    h0 = halves[0]
    swapped, cross = True, {}
    if len(halves) >= 2:
        h1 = halves[1]
        A = [(s[1], s[2]) for s in samples if s[0] == h0]
        B = [(s[1], s[2]) for s in samples if s[0] == h1]
        if min(sum(1 for a in A if a[0] == k) for k in (0, 1)) >= min_side and min(sum(1 for b in B if b[0] == k) for k in (0, 1)) >= min_side:
            model = fit_lda(np.array([X_all[i] for _, i in A]), np.array([a for a, _ in A]))
            sc = lda_score(model, np.array([X_all[i] for _, i in B]))
            side = np.array([b for b, _ in B])
            keep = float(np.mean((sc > 0) == (side == 1)))             # the sides of the second half as in the first
            cross = {"keep": keep, "swap": 1 - keep}
            swapped = keep < 0.5
    y = np.array([s[1] if (s[0] == h0 or not swapped) else 1 - s[1] for s in samples])
    X = np.array([X_all[s[2]] for s in samples])
    if min(int((y == 0).sum()), int((y == 1).sum())) < min_side:
        return None
    model = fit_lda(X, y)
    idx = [i for i in player_idx if i in descs]
    Xall = np.array([descs[i] for i in idx])
    pseudo = []
    rng = np.random.default_rng(0)
    for _ in range(self_train):
        # the kick-offs are around the centre line only; the appearance of a kit depends on the place (floodlights): the detections
        # the classifier is sure about, from the whole pitch, are added as examples and the classifier is learnt again
        raw = lda_score(model, Xall)
        pos, neg = np.flatnonzero(raw >= pseudo_margin), np.flatnonzero(raw <= -pseudo_margin)
        m = min(len(pos), len(neg), pseudo_cap)
        if m < 50:
            break
        pos, neg = rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)
        model = fit_lda(np.vstack([X, Xall[pos], Xall[neg]]), np.concatenate([y, np.ones(m, int), np.zeros(m, int)]))
        pseudo.append(2 * m)
    # the training accuracy (on the kick-off players)
    train_acc = float(np.mean((lda_score(model, X) > 0) == (y == 1)))
    # team 0 is the one with the darker shirts
    Lval = lambda cls: float(np.median([colors[s[2]][0] for s, yy in zip(samples, y) if yy == cls and s[2] in colors]))
    flip = Lval(0) > Lval(1)
    sign = -1.0 if flip else 1.0
    sc_all = lda_score(model, Xall) * sign
    score = {i: float(v) for i, v in zip(idx, sc_all)}
    team = {i: int(v > 0) for i, v in score.items()}
    ex_team = np.array([(yy if not flip else 1 - yy) for yy in y])
    centers = np.float32([np.median([colors[s[2]] for s, t in zip(samples, ex_team) if t == k and s[2] in colors], axis=0) for k in (0, 1)])
    limits = [max(outlier * float(np.median([float(cdist(centers[k], colors[s[2]])) for s, t in zip(samples, ex_team) if t == k and s[2] in colors])), min_limit) for k in (0, 1)]
    left_team = {}
    for h in halves:
        votes = Counter(int(ex_team[k]) for k, smp in enumerate(samples) if smp[0] == h and smp[1] == 0)
        if votes:
            left_team[h] = votes.most_common(1)[0][0]
    info = {"left_team": left_team, "pseudo": pseudo, "events": sum(used.values()), "examples": len(samples), "per_half": {h: sum(1 for s in samples if s[0] == h) for h in halves},
            "swapped": {h: (swapped and h != h0) for h in halves[1:]}, "cross": cross, "train_acc": train_acc}
    return team, score, centers, limits, info


def consistent_teams(rows, team, colors, centers, limits, gk, L=56.0, W=32.4, field=5, link_m=1.2, window_s=1.0, score=None):
    """Two corrections of the colour-based teams, both using that the same player keeps his team:
    1) smoothing over time: detections are linked to the nearest detection in the previous frame (within link_m);
       along a chain every detection gets the team that most detections of the chain around it (+-window_s) have.
       A player whose shirt looks different for a moment (sun, shadow, blur, turning) keeps his team.
    2) team size: a team has at most `field` outfield players on the pitch. If in a frame one team has too many and
       the other too few, the players of the crowded team whose colour is closest to the other team (and not far
       from it) move to it.
    Returns the corrected teams and the numbers of detections changed by 1) and by 2)."""
    team = dict(team)
    idx = [i for i, t in team.items() if t in (0, 1) and i not in gk and rows[i]["class"] == "player"]
    by_frame = defaultdict(list)
    for i in idx:
        by_frame[int(rows[i]["frame"])].append(i)
    frames = sorted(by_frame)
    pos = {i: (float(rows[i]["x_m"]), float(rows[i]["y_m"])) for i in idx}
    tim = {i: float(rows[i]["time_s"]) for i in idx}
    # 1) chains
    chain, prev, next_id = {}, [], 0
    for f in frames:
        cur = by_frame[f]
        used = set()
        pairs = sorted((np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]), i, j) for i in cur for j in prev)
        for d, i, j in pairs:
            if d > link_m or i in chain or j in used:
                continue
            chain[i] = chain[j]
            used.add(j)
        for i in cur:
            if i not in chain:
                chain[i], next_id = next_id, next_id + 1
        prev = cur
    members = defaultdict(list)
    for i in idx:
        members[chain[i]].append(i)
    smoothed = 0
    for c, ms in members.items():
        if len(ms) < 3:
            continue
        ms.sort(key=lambda i: tim[i])
        tt = np.array([tim[i] for i in ms])
        lab = np.array([team[i] for i in ms])
        lo = np.searchsorted(tt, tt - window_s)
        hi = np.searchsorted(tt, tt + window_s, side="right")
        cs = np.concatenate([[0], np.cumsum(lab == 1)])
        for k, i in enumerate(ms):
            n = hi[k] - lo[k]
            ones = cs[hi[k]] - cs[lo[k]]
            new = 1 if ones * 2 > n else 0 if ones * 2 < n else team[i]
            if new != team[i]:
                team[i] = new
                smoothed += 1
    # 2) team size per frame
    balanced = 0
    for f in frames:
        on = [i for i in by_frame[f] if -0.5 <= pos[i][0] <= L + 0.5 and -0.5 <= pos[i][1] <= W + 0.5]
        n = {t: sum(1 for i in on if team[i] == t) for t in (0, 1)}
        for big, small in ((1, 0), (0, 1)):
            k = min(n[big] - field, field - n[small])
            if k <= 0:
                continue
            if score is not None:                                      # the ones the classifier is least sure about
                cand = [i for i in on if team[i] == big and i in score and abs(score[i]) <= 2.0]
                cand.sort(key=lambda i: score[i] if big == 1 else -score[i])
            else:
                cand = [i for i in on if team[i] == big                # not hopelessly far from the other team's colour
                        and float(cdist(centers[small], colors[i])) <= max(limits[small], 3.0 * float(cdist(centers[big], colors[i])))]
                cand.sort(key=lambda i: float(cdist(centers[small], colors[i])) - float(cdist(centers[big], colors[i])))
            for i in cand[:k]:
                team[i] = small
                balanced += 1
            n[big] -= len(cand[:k])
            n[small] += len(cand[:k])
    return team, smoothed, balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="CSV from dual_detect.py")
    ap.add_argument("--video", required=True, help="dual-lens video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlier", type=float, default=2.5)
    ap.add_argument("--min-limit", type=float, default=15.0)
    ap.add_argument("--grass-dist", type=float, default=20.0)
    ap.add_argument("--no-ref-fix", action="store_true")
    ap.add_argument("--no-gk", action="store_true", help="do not treat goalkeepers separately")
    ap.add_argument("--gk-kickoff-s", type=float, default=15.0, help="seconds after the start of a half used to find the goalkeepers")
    ap.add_argument("--gk-radius", type=float, default=22.0, help="colour distance to a goalkeeper kit")
    ap.add_argument("--pitch-length", type=float, default=56.0)
    ap.add_argument("--pitch-width", type=float, default=32.4)
    ap.add_argument("--field-players", type=int, default=5, help="outfield players per team on the pitch")
    ap.add_argument("--no-consistency", action="store_true", help="no smoothing over time and no team-size correction")
    ap.add_argument("--ball", help="ball.csv from dual_ball.py: needed to find the kick-offs (--method kickoff)")
    ap.add_argument("--method", choices=["auto", "kickoff", "colour", "clf"], default="auto",
                    help="kickoff = teams learnt from the kick-offs (each team on its own half, light changes over the match "
                         "followed); colour = two clusters of the shirt colour over the whole match; auto = kickoff when enough "
                         "kick-offs are found, else colour; clf = a classifier of the appearance (hue histogram, body bands) trained on the "
                         "players of the kick-offs (needs a video pass for the descriptors); auto (default) = clf, else kickoff, else colour")
    ap.add_argument("--self-train", type=int, default=0, help="--method clf: this many rounds of learning again with the sure detections of the whole match added")
    ap.add_argument("--reuse-colors", action="store_true", help="reuse the colours saved by an earlier run (no video)")
    a = ap.parse_args()

    with open(Path(a.detections).expanduser()) as f:
        rows = list(csv.DictReader(f))
    by_frame = defaultdict(list)
    for i, r in enumerate(rows):
        if r["class"] in ("player", "referee"):
            by_frame[int(r["frame"])].append(i)

    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_name(out.stem + "_rgb.npz")
    descs = {}
    z = np.load(cache) if (a.reuse_colors and cache.exists()) else None
    if z is not None and (a.method not in ("clf", "auto") or "didx" in z.files):
        colors = {int(i): np.float32(c) for i, c in zip(z["idx"], z["lab"])}
        if "didx" in z.files:
            descs = {int(i): np.float32(d) for i, d in zip(z["didx"], z["desc"])}
        print(f"colours of {len(colors)} detections loaded from {cache}" + (f", appearance of {len(descs)}" if descs else ""))
    else:
        if z is not None:
            print("the saved colours have no appearance descriptors: the video is read again")
        colors, descs = read_colors(Path(a.video).expanduser(), rows, by_frame, a.grass_dist, with_desc=True)
        np.savez(cache, idx=np.array(list(colors.keys())), lab=np.float32(list(colors.values())),
                 didx=np.array(list(descs.keys())), desc=np.float32(list(descs.values())))

    team, relabeled, gk, centers, ref_center, limits = assign(
        rows, colors, a.pitch_length, a.outlier, a.min_limit, not a.no_ref_fix, use_gk=not a.no_gk,
        gk_kickoff_s=a.gk_kickoff_s, gk_radius=a.gk_radius)
    method, score = "colour", None
    if a.method == "clf" and not a.ball:
        raise SystemExit("--method clf needs --ball")
    res = None
    if a.method in ("auto", "clf") and a.ball:
        events = find_kickoffs(rows, Path(a.ball).expanduser(), a.pitch_length, a.pitch_width)
        player_idx = [i for i in colors if rows[i]["class"] == "player" and i not in relabeled]
        res = clf_teams(rows, colors, descs, events, player_idx, gk, a.pitch_length, self_train=a.self_train) if events and descs else None
        if res is None and a.method == "clf":
            raise SystemExit(f"--method clf: not enough kick-offs with players on both sides ({len(events)} found) - use --method colour")
        if res is None:
            print("no classifier from the kick-offs (too few kick-offs with players on both sides): the kick-off colours are tried next")
    if res is not None:
        c_team, score, centers, limits, info = res
        team.update(c_team)
        method = "clf"
        print(f"goalkeepers by the side they defend: {gk_by_side(rows, team, gk, info['left_team'], a.pitch_length)} goalkeeper detections changed team")
        if info["events"] < 3:
            print(f"WARNING: the classifier is trained on only {info['events']} kick-off(s): the teams may be unreliable (check with team_check.py)")
        if info["pseudo"]:
            print(f"self-training: {len(info['pseudo'])} rounds, {info['pseudo']} detections of the whole match added as examples")
        print(f"teams from a classifier trained on {info['examples']} players of {info['events']} kick-offs (per half {info['per_half']}); "
              f"sides swapped at half time: {info['swapped']}" + (f" (the classifier of the first half puts {100 * info['cross']['swap']:.0f}% of the second half "
              f"examples with swapped sides, {100 * info['cross']['keep']:.0f}% with the same sides)" if info["cross"] else "") +
              f"; accuracy on its own examples {100 * info['train_acc']:.0f}%")
    elif a.method in ("auto", "kickoff") and a.ball:
        events = find_kickoffs(rows, Path(a.ball).expanduser(), a.pitch_length, a.pitch_width)
        player_idx = [i for i in colors if rows[i]["class"] == "player" and i not in relabeled]
        res = kickoff_teams(rows, colors, events, player_idx, gk, a.pitch_length) if events else None
        if res is not None:
            k_team, centers, limits, info = res
            team.update(k_team)
            method = "kickoff"
            # goalkeepers by the side of the pitch: the one in the left half belongs to the team defending the left goal
            n_gk = gk_by_side(rows, team, gk, info["left_team"], a.pitch_length)
            print(f"goalkeepers by the side they defend (from the kick-offs): {n_gk} goalkeeper detections changed team")
            if info["dropped"] > info["events"]:
                print(f"WARNING: {info['dropped']} of {info['dropped'] + info['events']} kick-offs rejected:")
                for h, t, how, why in sorted(LAST_DECISIONS, key=lambda d: d[1]):
                    if why != "kept":
                        print(f"   half {h} {int(t) // 60:02d}:{int(t) % 60:02d} {how:9s} {why}")
            print(f"teams from the kick-offs: {info['events']} kick-offs used ({info['by_ball']} with the ball on the centre spot, "
                  f"{info['events'] - info['by_ball']} from the players' positions alone; {info['dropped']} rejected), {info['samples']} examples "
                  f"(per half {info['per_half']}), sides swapped at half time: {info['swapped_at_half_time']}")
        elif a.method == "kickoff":
            raise SystemExit(f"--method kickoff: not enough kick-offs found ({len(events)}) - use --method colour")
        else:
            print(f"only {len(events)} kick-offs found, none usable - teams from the shirt colour (clusters)")
            for h, t, how, why in sorted(LAST_DECISIONS, key=lambda d: d[1]):
                if why != "kept":
                    print(f"   half {h} {int(t) // 60:02d}:{int(t) % 60:02d} {how:9s} {why}")
    elif a.method == "kickoff":
        raise SystemExit("--method kickoff needs --ball")
    if not a.no_consistency:
        team, n_smooth, n_bal = consistent_teams(rows, team, colors, centers, limits, gk, a.pitch_length, a.pitch_width, a.field_players, score=score)
        print(f"team corrections: {n_smooth} detections by smoothing over time, {n_bal} by the team size ({a.field_players} outfield players)")
    for i in relabeled:
        rows[i] = dict(rows[i], **{"class": "referee"})

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["team", "gk"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "team": team.get(i, ""), "gk": 1 if i in gk else ""})
    info = {"team_0_bgr": lab_to_bgr(centers[0]), "team_1_bgr": lab_to_bgr(centers[1]),
            "referee_bgr": lab_to_bgr(ref_center) if ref_center is not None else None}
    out.with_name(out.stem + "_colors.json").write_text(json.dumps(info, indent=2))

    counts = {t: sum(1 for i, v in team.items() if v == t and i not in gk) for t in (0, 1, -1)}
    print(f"team 0: {counts[0]} detections, colour BGR {info['team_0_bgr']}")
    print(f"team 1: {counts[1]} detections, colour BGR {info['team_1_bgr']}")
    print(f"unknown (-1): {counts[-1]} | relabeled as referee: {len(relabeled)}")
    gk_count = Counter((rows[i]["half"], team[i]) for i in gk)
    print(f"goalkeepers: {len(gk)} detections " + ", ".join(
        f"half {h} team {t}: {n}" for (h, t), n in sorted(gk_count.items())))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
