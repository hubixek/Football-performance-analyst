#!/usr/bin/env python3
"""List the kick-offs the analysis found (start of a half and restarts after goals), with the time in the recording as in a
video player. Use it to write down the times of the goals without scanning the whole match:

  - open the video a minute before every listed kick-off,
  - if the ball went into the net shortly before it, that was a goal: write down the exact second of the goal,
  - a kick-off with no goal before it (apart from the start of a half) is a false one: the program found a formation
    that looked like a kick-off.

The goal times go to `analyze_dual.py --goals 14:21,21:29,...` (and `match_actions.py --goals`); they should be accurate to
about 2 seconds.

Usage:
  python list_kickoffs.py ~/football/analysis/mecz1_dual_v8 --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from pathlib import Path

from dual_teams import find_kickoffs


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with detections.csv and ball.csv")
    ap.add_argument("--pitch", required=True)
    a = ap.parse_args()
    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    with open(folder / "detections.csv") as f:
        rows = list(csv.DictReader(f))
    events = find_kickoffs(rows, folder / "ball.csv", pitch["length_m"], pitch["width_m"])
    print(f"{len(events)} kick-offs found in {folder.name} (the start of a half is not a goal)\n")
    print(" #  half  kick-off  how        watch from   gap to the previous")
    prev = None
    for k, (half, t, frames, how) in enumerate(sorted(events, key=lambda e: e[1]), 1):
        gap = f"{t - prev:5.0f} s" if prev is not None else "      -"
        print(f"{k:2d}  {half:4d}  {mmss(t)}     {how:9s}  {mmss(max(t - 90, 0))}      {gap}")
        prev = t
    print("\nA goal is followed by a kick-off after 20-60 s. Two kick-offs less than about 15 s apart are usually one event found twice.")


if __name__ == "__main__":
    main()
