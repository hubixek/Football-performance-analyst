#!/usr/bin/env python3
"""Copy detections.csv of a finished analysis into a new folder, trimmed to exactly the piece that
`analyze_dual.py --limit N` processes: the first N seconds of the first half.

Why: the detector is the slow step (about an hour for a match). If two models were already run on the whole match, a short
comparison of the later steps (ball, teams, possession) needs no detector at all: trim the existing detections and run
`analyze_dual.py --from-step ball` on the trimmed folder. The model that has no detections yet is run with --limit N.

Usage:
  python cut_detections.py ~/football/analysis/mecz1_dual ~/football/analysis/probe/v7 --match matches/mecz1.json --seconds 480
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="analysis folder with detections.csv")
    ap.add_argument("dst", help="new folder")
    ap.add_argument("--match", required=True, help="match config (half times)")
    ap.add_argument("--seconds", type=float, default=480.0, help="how many seconds of the first half to keep")
    a = ap.parse_args()

    cfg = json.loads(Path(a.match).expanduser().read_text())
    h = cfg["halves"][0]
    t0, t1 = h["start_s"], min(h["end_s"], h["start_s"] + a.seconds)
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with open(src / "detections.csv") as f, open(dst / "detections.csv", "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            total += 1
            t = float(row["time_s"])
            if int(row["half"]) == 1 and t0 <= t < t1:
                w.writerow(row)
                kept += 1
    print(f"{kept} of {total} detections kept ({t0:.0f}-{t1:.0f} s of the recording, {t1 - t0:.0f} s) -> {dst / 'detections.csv'}")


if __name__ == "__main__":
    main()
