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


def jersey_color(img, box, grass=None, grass_dist=20.0):
    """Median Lab colour of the shirt area, without pixels similar to the pitch surface."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    ya, yb = int(y1 + 0.15 * h), int(y1 + 0.5 * h)
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


def read_colors(video, rows, by_frame, grass_dist):
    """Jersey colour of every player/referee detection, reading the video in order."""
    colors = {}
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
        if (k + 1) % 1000 == 0:
            print(f"colours: {k + 1}/{len(frames)} frames", flush=True)
    cap.release()
    return colors


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
    return team, relabeled, gk, centers, ref_center


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
    if a.reuse_colors and cache.exists():
        z = np.load(cache)
        colors = {int(i): np.float32(c) for i, c in zip(z["idx"], z["lab"])}
        print(f"colours of {len(colors)} detections loaded from {cache}")
    else:
        colors = read_colors(Path(a.video).expanduser(), rows, by_frame, a.grass_dist)
        np.savez(cache, idx=np.array(list(colors.keys())), lab=np.float32(list(colors.values())))

    team, relabeled, gk, centers, ref_center = assign(
        rows, colors, a.pitch_length, a.outlier, a.min_limit, not a.no_ref_fix, use_gk=not a.no_gk,
        gk_kickoff_s=a.gk_kickoff_s, gk_radius=a.gk_radius)
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
