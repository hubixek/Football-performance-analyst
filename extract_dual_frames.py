#!/usr/bin/env python3
"""Extract frames for annotation from a dual-lens recording.

Takes a frame every --every seconds, only during the halves given in the match config,
splits it into the top and bottom half and saves both:
  <out>/<name>/<name>_top_<tenths of a second>.jpg, <name>_bottom_<...>.jpg

Usage:
  python extract_dual_frames.py ~/football/videos/mecz2_dual.mp4 --match matches/mecz2.json \
      --out ~/football/frames_dual --every 20
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--match", required=True, help="match config with half times (match_config.py)")
    ap.add_argument("--out", default=str(Path.home() / "football" / "frames_dual"))
    ap.add_argument("--every", type=float, default=20.0, help="seconds between frames")
    a = ap.parse_args()

    cfg = json.loads(Path(a.match).read_text())
    name = cfg["name"]
    out = Path(a.out).expanduser() / name
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    n = 0
    for half in cfg["halves"]:
        t = half["start_s"]
        while t < half["end_s"]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                break
            h = img.shape[0] // 2
            for part, crop in (("top", img[:h]), ("bottom", img[h:2 * h])):
                cv2.imwrite(str(out / f"{name}_{part}_{int(round(t * 10)):06d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            t += a.every
    cap.release()
    print(f"{n} moments -> {2 * n} frames in {out}")


if __name__ == "__main__":
    main()
