#!/usr/bin/env python3
"""Team assignment for dual-lens detections (jersey colour, K-means).

Same method as teams.py (shirt colour without turf pixels, two clusters, outliers = -1,
players in the referee's colour relabeled as referee), but every detection is cut from the
lens image it was detected in (top or bottom half of the frame).

Input:  CSV from dual_detect.py + the dual-lens video.
Output: the same CSV with a `team` column (0, 1, -1; empty for referee/ball) and
        <out>_colors.json with the team colours (used for plots).

Usage:
  python dual_teams.py ~/football/analysis/mecz1_dual/detections.csv \
      --video ~/football/videos/mecz1_dual.mp4 --out ~/football/analysis/mecz1_dual/teams.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from teams import grass_color, jersey_color, lab_to_bgr


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


def assign(rows, colors, outlier=2.5, min_limit=15.0, ref_fix=True, ref_max=20.0, ref_max_share=0.10):
    ref_idx = [i for i in colors if rows[i]["class"] == "referee"]
    player_idx = [i for i in colors if rows[i]["class"] == "player"]
    ref_center, ref_limit = None, 0.0
    if ref_fix and len(ref_idx) >= 20:
        ref_data = np.float32([colors[i] for i in ref_idx])
        ref_center = np.median(ref_data, axis=0)
        # hard cap: referee detections over a whole match are noisy (lighting, model mistakes),
        # a wide range would swallow the players' shirts
        ref_limit = min(max(outlier * np.median(np.linalg.norm(ref_data - ref_center, axis=1)), min_limit), ref_max)

    def ref_distance(i):
        return float(np.linalg.norm(colors[i] - ref_center)) if ref_center is not None else np.inf

    fit_idx = [i for i in player_idx if ref_distance(i) > ref_limit]
    if len(fit_idx) < 10:
        raise SystemExit("Too few player detections to split into teams")
    data = np.float32([colors[i] for i in fit_idx])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    dist = np.linalg.norm(data - centers[labels], axis=1)
    limits = [max(outlier * np.median(dist[labels == t]), min_limit) for t in (0, 1)]

    team, relabeled = {}, set()
    for i in player_idx:
        d = np.linalg.norm(centers - colors[i], axis=1)
        t = int(np.argmin(d))
        # referee only if the shirt is clearly closer to the referee than to the own team
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
    return team, relabeled, centers, ref_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="CSV from dual_detect.py")
    ap.add_argument("--video", required=True, help="dual-lens video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlier", type=float, default=2.5)
    ap.add_argument("--min-limit", type=float, default=15.0)
    ap.add_argument("--grass-dist", type=float, default=20.0)
    ap.add_argument("--no-ref-fix", action="store_true")
    a = ap.parse_args()

    with open(Path(a.detections).expanduser()) as f:
        rows = list(csv.DictReader(f))
    by_frame = defaultdict(list)
    for i, r in enumerate(rows):
        if r["class"] in ("player", "referee"):
            by_frame[int(r["frame"])].append(i)

    colors = read_colors(Path(a.video).expanduser(), rows, by_frame, a.grass_dist)
    team, relabeled, centers, ref_center = assign(rows, colors, a.outlier, a.min_limit, not a.no_ref_fix)
    for i in relabeled:
        rows[i] = dict(rows[i], **{"class": "referee"})

    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["team"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "team": team.get(i, "")})
    info = {"team_0_bgr": lab_to_bgr(centers[0]), "team_1_bgr": lab_to_bgr(centers[1]),
            "referee_bgr": lab_to_bgr(ref_center) if ref_center is not None else None}
    out.with_name(out.stem + "_colors.json").write_text(json.dumps(info, indent=2))

    counts = {t: sum(1 for v in team.values() if v == t) for t in (0, 1, -1)}
    print(f"team 0: {counts[0]} detections, colour BGR {info['team_0_bgr']}")
    print(f"team 1: {counts[1]} detections, colour BGR {info['team_1_bgr']}")
    print(f"unknown (-1): {counts[-1]} | relabeled as referee: {len(relabeled)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
