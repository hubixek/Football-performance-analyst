#!/usr/bin/env python3
"""Detection on a dual-lens recording, with positions on the pitch in metres.

For every processed frame (only during the halves of the match):
  1. the frame is split into the top and bottom half (one image per lens),
  2. the model detects players, referees and ball candidates in both halves,
  3. every detection is mapped to pitch coordinates with the calibration of its lens
     (feet for people, centre for the ball),
  4. people more than --margin m outside the pitch are dropped (substitutes, coaches,
     spectators), ball candidates more than --ball-margin m outside the pitch too,
  5. a person seen by both lenses (around the halfway line, detections closer than
     --merge-dist m) is kept once: from the lens
     where he is closer to the image centre, i.e. where the calibration is most accurate,
  6. up to --max-balls ball candidates per frame are kept (the real ball is chosen later).

Output CSV: frame, time_s, half (1/2 of the match), cam (top/bottom), class, conf,
x1, y1, x2, y2 (pixels in that lens image), x_m, y_m (pitch position in metres).

Usage:
  python dual_detect.py ~/football/videos/mecz1_dual.mp4 --match matches/mecz1.json \
      --model ~/football/runs/detect/runs/v7_dual/weights/best.pt --calib-dir ~/football/calib \
      --pitch pitch/pitch_6v6.json --out ~/football/analysis/mecz1_dual/detections.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import pixels_to_pitch

CAMS = ("top", "bottom")


def inside(xy, pitch, margin):
    L, W = pitch["length_m"], pitch["width_m"]
    return bool(np.isfinite(xy).all() and -margin < xy[0] < L + margin and -margin < xy[1] < W + margin)


def centre_distance(calib, u, v):
    """Distance from the optical centre relative to the focal length (lower = more accurate)."""
    return float(np.hypot(u - calib["cx"], v - calib["cy"]) / calib["f"])


def process_frame(dets, calibs, pitch, margin=1.0, ball_margin=2.0, merge_dist=2.5, max_balls=3):
    """dets: list of dicts {cam, cls, conf, x1, y1, x2, y2} from both lenses of one frame.
    Returns the kept detections with x_m, y_m added."""
    people, balls = [], []
    for d in dets:
        is_ball = d["cls"] == "ball"
        u = (d["x1"] + d["x2"]) / 2
        v = (d["y1"] + d["y2"]) / 2 if is_ball else d["y2"]
        xy = pixels_to_pitch(calibs[d["cam"]], np.array([[u, v]]))[0]
        if not inside(xy, pitch, ball_margin if is_ball else margin):
            continue
        d = dict(d, x_m=float(xy[0]), y_m=float(xy[1]), quality=centre_distance(calibs[d["cam"]], u, v))
        (balls if is_ball else people).append(d)

    def dedupe(items, same_class):
        # a detection seen by both lenses: keep the one closer to its image centre
        items = sorted(items, key=lambda d: d["quality"])
        kept = []
        for d in items:
            dup = any(k["cam"] != d["cam"] and (not same_class or k["cls"] == d["cls"])
                      and np.hypot(k["x_m"] - d["x_m"], k["y_m"] - d["y_m"]) < merge_dist for k in kept)
            if not dup:
                kept.append(d)
        return kept

    people = dedupe(people, same_class=False)
    balls = sorted(dedupe(balls, same_class=True), key=lambda d: -d["conf"])[:max_balls]
    return people + balls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--match", required=True, help="match config with half times")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", default=str(Path.home() / "football" / "calib"))
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2, help="process every n-th frame (2 = 15 fps)")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--ball-margin", type=float, default=2.0)
    ap.add_argument("--merge-dist", type=float, default=2.5,
                    help="detections from the two lenses closer than this (m) are one person")
    ap.add_argument("--limit", type=float, default=0,
                    help="process only this many seconds of playing time (quick test), 0 = whole match")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = json.loads(Path(a.match).read_text())
    cdir = Path(a.calib_dir).expanduser()
    calibs = {c: json.loads((cdir / f"{cfg['name']}_{c}.json").read_text()) for c in CAMS}
    pitch = json.loads(Path(a.pitch).read_text())
    model = YOLO(str(Path(a.model).expanduser()))
    names = model.names

    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
        raise SystemExit(f"cannot open the video {a.video} (does the file exist, and is it a video?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    halves = cfg["halves"]
    if a.limit:
        # quick test: only the first --limit seconds of the first half
        halves = [dict(halves[0], end_s=min(halves[0]["end_s"], halves[0]["start_s"] + a.limit))]
    total = sum(int((h["end_s"] - h["start_s"]) * fps / a.stride) for h in halves)
    done = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "cam", "class", "conf", "x1", "y1", "x2", "y2", "x_m", "y_m"])
        for half_no, half in enumerate(halves, 1):
            first, last = int(half["start_s"] * fps), int(half["end_s"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            for frame in range(first, last):
                ok = cap.grab()
                if not ok:
                    break
                if (frame - first) % a.stride:
                    continue
                ok, img = cap.retrieve()
                if not ok:
                    break
                h = img.shape[0] // 2
                crops = {"top": img[:h], "bottom": img[h:2 * h]}
                dets = []
                for cam, r in zip(CAMS, model.predict([crops[c] for c in CAMS], imgsz=a.imgsz, conf=a.conf,
                                                      verbose=False)):
                    for c, conf, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                                         r.boxes.xyxy.tolist()):
                        dets.append({"cam": cam, "cls": names[int(c)], "conf": conf,
                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                for d in process_frame(dets, calibs, pitch, a.margin, a.ball_margin, a.merge_dist):
                    w.writerow([frame, round(frame / fps, 3), half_no, d["cam"], d["cls"], round(d["conf"], 3),
                                round(d["x1"], 1), round(d["y1"], 1), round(d["x2"], 1), round(d["y2"], 1),
                                round(d["x_m"], 2), round(d["y_m"], 2)])
                done += 1
                if done % 500 == 0:
                    print(f"{done}/{total} frames", flush=True)
    cap.release()
    if done == 0:
        raise SystemExit("no frame was processed: the half times of the match config lie outside the video, or the video cannot be read")
    print(f"Saved {out} ({done} frames)")


if __name__ == "__main__":
    main()
