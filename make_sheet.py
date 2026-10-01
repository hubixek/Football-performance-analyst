#!/usr/bin/env python3
"""Tile several images into ONE sheet (with a small caption on each), e.g. to send a single picture to the chat instead of ten.

Usage:
  python make_sheet.py img1.jpg img2.jpg img3.jpg --out sheet.jpg
  python make_sheet.py ~/football/analysis/probe/v8/detections_*.jpg --out ~/sheet.jpg --cols 2 --width 2000
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=2000, help="width of the whole sheet in pixels")
    ap.add_argument("--crop", help="optional crop of every image as x1,y1,x2,y2 in pixels of the original (e.g. around the ball)")
    a = ap.parse_args()

    cell_w = a.width // a.cols
    tiles = []
    for path in a.images:
        img = cv2.imread(str(Path(path).expanduser()))
        if img is None:
            print(f"cannot read {path}, skipped")
            continue
        if a.crop:
            x1, y1, x2, y2 = (int(v) for v in a.crop.split(","))
            img = img[y1:y2, x1:x2]
        h = int(img.shape[0] * cell_w / img.shape[1])
        t = cv2.resize(img, (cell_w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (min(cell_w, 12 + 14 * len(Path(path).stem[-26:])), 26), (0, 0, 0), -1)
        cv2.putText(t, Path(path).stem[-26:], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        raise SystemExit("no images")
    rows = []
    for r0 in range(0, len(tiles), a.cols):
        row = tiles[r0:r0 + a.cols]
        hmax = max(t.shape[0] for t in row)
        row = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in row]
        while len(row) < a.cols:
            row.append(np.full((hmax, cell_w, 3), 30, np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = Path(a.out).expanduser()
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(tiles)} images -> one sheet {sheet.shape[1]}x{sheet.shape[0]} px: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
