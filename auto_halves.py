#!/usr/bin/env python3
"""The half times of a match found automatically: no need to watch the video and write down the whistles.

Every --step seconds one frame is taken, the detection model counts the people on the pitch (inside the pitch lines by the calibration of the
match; the two lenses are merged) and a half is a stretch of the recording in which at least --min-players people are on the pitch (the
count is smoothed over about 10 s). The break between the halves is the stretch without them (the players leave the pitch, the pitch is
empty). Short gaps (timeouts, the count dropping when players leave the pitch for a moment) are bridged (--gap), short stretches (warm-up on a
corner of the pitch) are dropped (--min-segment). The two longest stretches are the halves.

Writes matches/<name>.json in the format of match_config.py (the half times are the first and the last moment with the players on the pitch,
so the start of a half can be up to a few seconds early; the kick-offs are found later from the positions of the players and the ball).
With --compare an existing (manual) config is compared with the result and the differences of the four boundaries are printed.

The calibration of the match (calib/<name>_top.json and _bottom.json) is needed: calibrate_match.py can be run with a provisional config
of the whole recording (process_match.py does it).

Usage:
  python auto_halves.py ~/football/videos/mecz1_dual.mp4 --name mecz1 --model ~/football/runs/detect/runs/v8_dual/weights/best.pt --pitch pitch/pitch_6v6.json
  ... --compare matches/mecz1.json --out /tmp/mecz1_auto.json     (a check against the half times written by hand)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import config

CAMS = ("top", "bottom")
HERE = Path(__file__).resolve().parent


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def count_people(boxes_by_cam, calibs, pitch, margin=1.0, merge=2.5):
    """People on the pitch in one frame: foot points of the boxes of both lenses in metres, inside the pitch, the same person seen by both
    lenses counted once. boxes_by_cam: {cam: [(x1, y1, x2, y2), ...]}."""
    from calibrate import pixels_to_pitch
    L, W = pitch["length_m"], pitch["width_m"]
    pts = []
    for cam, boxes in boxes_by_cam.items():
        if not boxes:
            continue
        feet = np.array([[(b[0] + b[2]) / 2, b[3]] for b in boxes], float)
        xy = pixels_to_pitch(calibs[cam], feet)
        for p in xy:
            if np.isfinite(p).all() and -margin < p[0] < L + margin and -margin < p[1] < W + margin:
                pts.append(p)
    kept = []
    for p in pts:
        if all(np.hypot(*(p - q)) > merge for q in kept):
            kept.append(p)
    return len(kept)


def find_halves(times, counts, min_players=8, smooth=5, gap=30.0, min_segment=300.0, step=2.0, halves=2):
    """The halves from the counts of the people on the pitch. Returns ([(start_s, end_s), ...], notes)."""
    c = np.asarray(counts, float)
    k = smooth // 2
    med = np.array([np.median(c[max(0, i - k):i + k + 1]) for i in range(len(c))])
    play = med >= min_players
    segs, i = [], 0
    while i < len(play):
        if play[i]:
            j = i
            while j + 1 < len(play) and play[j + 1]:
                j += 1
            segs.append([times[i], times[j]])
            i = j + 1
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    notes = []
    long_ = [s for s in merged if s[1] - s[0] + step >= min_segment]
    if len(long_) > halves:
        long_ = sorted(sorted(long_, key=lambda s: s[1] - s[0], reverse=True)[:halves])
        notes.append(f"more than {halves} stretches with the players on the pitch: the {halves} longest are taken")
    if len(long_) == 1 and halves == 2 and long_[0][1] - long_[0][0] > 8 * min_segment:      # one stretch long enough for two halves
        # one continuous stretch: look for the deepest dip of the count in the middle part
        s0, s1 = long_[0]
        mid = [(t, m) for t, m in zip(times, med) if s0 + 0.3 * (s1 - s0) <= t <= s0 + 0.7 * (s1 - s0)]
        if mid:
            t_dip = min(mid, key=lambda x: x[1])
            lo, hi = t_dip[0], t_dip[0]
            low = [t for t, m in zip(times, med) if s0 <= t <= s1 and m <= t_dip[1] + 1]
            near = [t for t in low if abs(t - t_dip[0]) < 240]
            lo, hi = min(near), max(near)
            long_ = [[s0, lo], [hi, s1]]
            notes.append(f"no empty pitch between the halves: split at the lowest count ({int(lo)}-{int(hi)} s, {t_dip[1]:.0f} people)")
    if len(long_) != halves:
        notes.append(f"found {len(long_)} stretches with at least {min_players} people on the pitch instead of {halves}")
    return [(max(0.0, s[0] - step), s[1] + step) for s in long_], notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", required=True, help="name of the match (the calibration calib/<name>_top.json is used)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    ap.add_argument("--out", help="the config to write (default: matches/<name>.json next to the scripts)")
    ap.add_argument("--step", type=float, default=2.0, help="seconds between the frames that are counted")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-players", type=int, default=8, help="people on the pitch for the play (6v6: 12 play, some are not seen)")
    ap.add_argument("--gap", type=float, default=30.0, help="a gap this short does not end a half, s")
    ap.add_argument("--min-segment", type=float, default=300.0, help="a stretch shorter than this is not a half, s")
    ap.add_argument("--halves", type=int, default=2)
    ap.add_argument("--compare", help="an existing config (written by hand): the differences of the half times are printed")
    a = ap.parse_args()

    from ultralytics import YOLO
    pitch = json.loads(Path(a.pitch).read_text())
    cdir = Path(a.calib_dir).expanduser()
    calibs = {c: json.loads((cdir / f"{a.name}_{c}.json").read_text()) for c in CAMS}
    model = YOLO(str(Path(a.model).expanduser()))
    people = [i for i, n in model.names.items() if n in ("player", "referee")]
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
        raise SystemExit(f"cannot open the video {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    times, counts = [], []
    t = 0.0
    while t < duration - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            break
        h = img.shape[0] // 2
        crops = {"top": img[:h], "bottom": img[h:2 * h]}
        boxes = {}
        for cam, r in zip(CAMS, model.predict([crops[c] for c in CAMS], imgsz=a.imgsz, conf=a.conf, verbose=False)):
            boxes[cam] = [tuple(b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if int(c) in people]
        times.append(t)
        counts.append(count_people(boxes, calibs, pitch))
        if len(times) % 200 == 0:
            print(f"{len(times)} frames counted ({fmt(t)} of {fmt(duration)})", flush=True)
        t += a.step
    cap.release()
    found, notes = find_halves(times, counts, a.min_players, gap=a.gap, min_segment=a.min_segment, step=a.step, halves=a.halves)
    per_min = [int(np.median(counts[i:i + int(60 / a.step)])) for i in range(0, len(counts), int(60 / a.step))]
    print("people on the pitch per minute of the recording (median): " + " ".join(str(x) for x in per_min))
    for n in notes:
        print("NOTE:", n)
    if len(found) != a.halves:
        raise SystemExit(f"the halves were not found automatically ({len(found)} of {a.halves}); use match_config.py and write the half times by hand")
    halves = [{"start": fmt(s), "end": fmt(e), "start_s": round(s, 1), "end_s": round(e, 1)} for s, e in found]
    for i, h in enumerate(halves, 1):
        print(f"half {i}: {h['start']} - {h['end']} ({(h['end_s'] - h['start_s']) / 60:.1f} min)")
    print(f"break: {(halves[1]['start_s'] - halves[0]['end_s']) / 60:.1f} min" if len(halves) > 1 else "")
    if a.compare:
        ref = json.loads(Path(a.compare).expanduser().read_text())["halves"]
        for i, (h, r) in enumerate(zip(halves, ref), 1):
            print(f"compared with {Path(a.compare).name}: half {i}: start {h['start_s'] - r['start_s']:+.0f} s, end {h['end_s'] - r['end_s']:+.0f} s")
    out = Path(a.out).expanduser() if a.out else HERE / "matches" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"name": a.name, "video": Path(a.video).name, "halves": halves,
           "calibration": {c: f"calib/{a.name}_{c}.json" for c in CAMS}}
    out.write_text(json.dumps(cfg, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
