#!/usr/bin/env python3
"""Duplicates of people in the detections of an analysis (detections.csv).

The detector can give one person two boxes in the same lens: a `player` box and a `referee` box. dual_detect.py merges
only detections seen by BOTH lenses, so such pairs stay in the data. This script measures them, per analysis folder:
  - how many referee boxes lie on a player box (IoU >= --iou) = duplicates, and their median confidence,
  - the other referee boxes (the real referee and the false ones), their median confidence,
  - how many of the other referee boxes there are per lens and frame (one referee exists: much more than 1 means false referees),
  - the same for player boxes lying on a referee box.

Usage:
  python person_duplicates.py ~/football/analysis/mecz1_dual ~/football/analysis/mecz1_dual_v8
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def med(v):
    return f"{np.median(v):.2f}" if len(v) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="analysis folders with detections.csv")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    for folder in a.folders:
        by = defaultdict(lambda: {"player": [], "referee": []})
        with open(Path(folder).expanduser() / "detections.csv") as f:
            for r in csv.DictReader(f):
                if r["class"] in ("player", "referee"):
                    by[(r["frame"], r["cam"])][r["class"]].append((tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")), float(r["conf"])))
        dup_r, own_r, dup_p, own_p, n_frames = [], [], [], [], 0
        extra_per_key = []
        for d in by.values():
            n_frames += 1
            others = 0
            for b, cf in d["referee"]:
                if any(iou(b, pb) >= a.iou for pb, _ in d["player"]):
                    dup_r.append(cf)
                else:
                    own_r.append(cf)
                    others += 1
            for b, cf in d["player"]:
                (dup_p if any(iou(b, rb) >= a.iou for rb, _ in d["referee"]) else own_p).append(cf)
            extra_per_key.append(others)
        refs = len(dup_r) + len(own_r)
        ex = np.array(extra_per_key)
        print(f"===== {Path(folder).name}: {n_frames} lens-frames with people, {refs} referee boxes, {len(dup_r) + len(dup_p)} overlaps")
        print(f"  referee boxes on a player box (duplicates): {len(dup_r)} ({100 * len(dup_r) / max(refs, 1):.0f}%), median confidence {med(dup_r)}")
        print(f"  other referee boxes: {len(own_r)} ({100 * len(own_r) / max(refs, 1):.0f}%), median confidence {med(own_r)}; "
              f"per lens-frame: mean {ex.mean():.2f}, more than one in {100 * np.mean(ex > 1):.1f}% of them (one referee exists)")
        print(f"  player boxes on a referee box: {len(dup_p)}, median confidence {med(dup_p)} (other player boxes {med(own_p)})")


if __name__ == "__main__":
    main()
