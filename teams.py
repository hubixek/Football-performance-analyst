#!/usr/bin/env python3
"""Assign players to two teams by jersey color.
 
For every player detection the script takes the shirt area (upper part of the box),
removes pixels similar to the pitch color of that frame (so green shirts in a
different shade still work) and computes the median color (Lab). The colors are
clustered into two teams with K-means. Players whose color is far from both teams
(goalkeepers in a different kit, substitutes in bibs) get team -1.
The referee's shirt color is learned from detections classified as referee; players
with that color are relabeled as referee (the model sometimes confuses them).
 
Input:  CSV from detect.py + the original video.
Output: the same CSV with an extra column `team` (0, 1, -1; empty for referee/ball).
 
Usage:
  python teams.py ~/football/analysis/mecz1_test_detections.csv \
      --video-in ~/football/videos/mecz1_test.mp4 \
      --out ~/football/analysis/mecz1_test_teams.csv \
      --ball ~/football/analysis/mecz1_test_ball.csv \
      --video-out ~/football/analysis/mecz1_test_teams.mp4
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
 
import cv2
import numpy as np
 
from ball_track import to_h264
 
 
def grass_color(img):
    """Median Lab color of the frame - most of a football frame is the pitch surface."""
    small = cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA)[18:]  # lower 2/3
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)
 
 
def jersey_color(img, box, grass=None, grass_dist=20.0):
    """Median Lab color of the shirt area, without pixels similar to the pitch surface.
 
    Only pixels close to the actual grass color of this frame are removed, so a green
    shirt of a different shade than the pitch is kept.
    """
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
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("detections", help="CSV from detect.py")
    p.add_argument("--video-in", required=True, help="original video")
    p.add_argument("--out", required=True, help="output CSV with the team column")
    p.add_argument("--ball", help="optional CSV from ball_track.py (drawn in the preview)")
    p.add_argument("--video-out", help="optional preview video")
    p.add_argument("--outlier", type=float, default=2.5,
                   help="team -1 if color distance > outlier x median distance of the team")
    p.add_argument("--grass-dist", type=float, default=20.0,
                   help="pixels closer than this to the pitch color are ignored (Lab units)")
    p.add_argument("--no-grass-mask", action="store_true",
                   help="do not remove pitch-colored pixels (if a shirt is almost the pitch color)")
    p.add_argument("--no-ref-fix", action="store_true",
                   help="do not relabel players whose shirt color matches the referee")
    p.add_argument("--min-limit", type=float, default=15.0,
                   help="color distance always accepted as the team (Lab units)")
    a = p.parse_args()
 
    with open(Path(a.detections).expanduser()) as f:
        rows = list(csv.DictReader(f))
    by_frame = defaultdict(list)
    for i, r in enumerate(rows):
        if r["class"] in ("player", "referee"):
            by_frame[int(r["frame"])].append(i)
 
    # 1. jersey color of every player detection
    colors = {}
    cap = cv2.VideoCapture(str(Path(a.video_in).expanduser()))
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        grass = None if a.no_grass_mask else grass_color(img)
        for i in by_frame.get(frame, []):
            r = rows[i]
            c = jersey_color(img, [float(r[k]) for k in ("x1", "y1", "x2", "y2")], grass, a.grass_dist)
            if c is not None:
                colors[i] = c
        frame += 1
        if frame % 500 == 0:
            print(f"colors: frame {frame}")
    cap.release()
    # 2. referee color: learned from detections the model classified as referee
    ref_idx = [i for i in colors if rows[i]["class"] == "referee"]
    player_idx = [i for i in colors if rows[i]["class"] == "player"]
    ref_center, ref_limit = None, 0.0
    if len(ref_idx) >= 20 and not a.no_ref_fix:
        ref_data = np.float32([colors[i] for i in ref_idx])
        ref_center = np.median(ref_data, axis=0)
        ref_limit = max(a.outlier * np.median(np.linalg.norm(ref_data - ref_center, axis=1)), a.min_limit)
 
    def ref_distance(i):
        return float(np.linalg.norm(colors[i] - ref_center)) if ref_center is not None else float("inf")
 
    # 3. two teams with K-means (players that look like the referee are left out of fitting)
    fit_idx = [i for i in player_idx if ref_distance(i) > ref_limit]
    if len(fit_idx) < 10:
        raise SystemExit("Too few player detections to split into teams")
    data = np.float32([colors[i] for i in fit_idx])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    fit_dist = np.linalg.norm(data - centers[labels], axis=1)
    limits = [max(a.outlier * np.median(fit_dist[labels == t]), a.min_limit) for t in (0, 1)]
 
    # 4. assign every player detection
    team, relabeled = {}, set()
    for i in player_idx:
        d = np.linalg.norm(centers - colors[i], axis=1)
        t = int(np.argmin(d))
        dr = ref_distance(i)
        if dr <= ref_limit and dr < d[t]:
            relabeled.add(i)  # looks like the referee, not a player
            continue
        team[i] = t if d[t] <= limits[t] else -1
    for i in relabeled:
        rows[i] = {**rows[i], "class": "referee"}
 
    counts = {t: sum(1 for v in team.values() if v == t) for t in (0, 1, -1)}
    team_bgr = [lab_to_bgr(c) for c in centers]
    print(f"team 0: {counts[0]} detections, color BGR {team_bgr[0]}")
    print(f"team 1: {counts[1]} detections, color BGR {team_bgr[1]}")
    print(f"unknown (-1): {counts[-1]} detections")
    if ref_center is not None:
        print(f"relabeled player -> referee: {len(relabeled)} detections "
              f"(referee color BGR {lab_to_bgr(ref_center)})")
 
    # 3. CSV with the team column
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["team"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "team": team.get(i, "")})
    print(f"Saved {out}")
 
    # 4. optional preview video
    if not a.video_out:
        return
    ball = {}
    if a.ball:
        with open(Path(a.ball).expanduser()) as f:
            for r in csv.DictReader(f):
                ball[int(r["frame"])] = (float(r["x"]), float(r["y"]))
    all_by_frame = defaultdict(list)
    for i, r in enumerate(rows):
        if r["class"] != "ball":
            all_by_frame[int(r["frame"])].append(i)
 
    cap = cv2.VideoCapture(str(Path(a.video_in).expanduser()))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wd, ht = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final = Path(a.video_out).expanduser()
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, ht))
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for i in all_by_frame.get(frame, []):
            r = rows[i]
            x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            if r["class"] == "referee":
                color, label = (0, 255, 255), "REF"
            else:
                t = team.get(i, -1)
                color = team_bgr[t] if t >= 0 else (160, 160, 160)
                label = f"T{t}" if t >= 0 else "?"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if frame in ball:
            bx, by = ball[frame]
            cv2.circle(img, (int(bx), int(by)), 14, (0, 255, 0), 3)
        writer.write(img)
        frame += 1
    cap.release()
    writer.release()
    print("Re-encoding video to H.264...")
    to_h264(tmp, final)
    print(f"Saved preview {final}")
 
 
if __name__ == "__main__":
    main()
