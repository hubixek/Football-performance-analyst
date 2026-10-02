#!/usr/bin/env python3
"""Would the rule "at most one referee per lens and frame" fix the players that the detector took for the referee?

There is one referee in a match. If in one frame of one lens there are several boxes of the class `referee`, at most one of them can be
the referee: the one with the highest confidence is kept, the others are players. This script tests the rule on the crops you judged
(team_check.py):
  - players (your truth 0 or 1) that the analysis shows as a referee: was there another referee box in the same lens and frame, and with a
    higher confidence? Then the rule would turn them into players.
  - real referees (your truth r): would the rule keep them? (they lose when a player box of the class referee has a higher confidence)

It also checks the question that decides how much the mistakes matter for the statistics: does the person that the analysis shows as a
referee have a SECOND row of the class player (a duplicate: the same box seen as a player and as a referee)? Then he is counted as a player
by the statistics (the team of the player row) and the referee row does no harm.

Usage:
  python referee_rule_probe.py ~/football/analysis/mecz4_2dual
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with detections.csv and team_check/")
    a = ap.parse_args()
    folder = Path(a.folder).expanduser()
    truth = {r["id"]: r["truth"].strip().lower() for r in csv.DictReader(open(folder / "team_check" / "team_check.csv"))}
    key = list(csv.DictReader(open(folder / "team_check" / ".answers_of_the_analysis.csv")))
    refs = defaultdict(list)                                       # (frame, cam) -> [(conf, x1, y1)]
    players = defaultdict(list)                                    # (frame, cam) -> [box of a player row]
    with open(folder / "detections.csv") as f:
        for r in csv.DictReader(f):
            if r["class"] == "referee":
                refs[(r["frame"], r["cam"])].append((float(r["conf"]), round(float(r["x1"]), 1), round(float(r["y1"]), 1)))
            elif r["class"] == "player":
                players[(r["frame"], r["cam"])].append(tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2")))
    out = {"player": [], "referee": []}
    for k in key:
        t = truth.get(k["id"], "")
        if t not in ("0", "1", "r") or k["class"] != "referee":
            continue
        boxes = sorted(refs.get((k["frame"], k["cam"]), []), reverse=True)
        me = (round(float(k["x1"]), 1), round(float(k["y1"]), 1))
        rank = next((j for j, b in enumerate(boxes) if (b[1], b[2]) == me), None)
        box = tuple(float(k[c]) for c in ("x1", "y1", "x2", "y2"))
        dup = any(iou(box, pb) >= 0.5 for pb in players.get((k["frame"], k["cam"]), []))
        out["referee" if t == "r" else "player"].append((len(boxes), rank, boxes[rank][0] if rank is not None else None, dup))
    p, rr = out["player"], out["referee"]
    print(f"players (truth 0/1) that the analysis shows as a referee: {len(p)}")
    for n, rank, conf, dup in p:
        print(f"   {n} referee box(es) in that lens and frame; this one is number {'?' if rank is None else rank + 1} by confidence" + (f" ({conf:.2f})" if conf is not None else "")
              + ("; the same person also has a row of the class PLAYER" if dup else "; NO row of the class player: lost for the statistics"))
    fixed = sum(1 for n, rank, _, d in p if n > 1 and rank not in (None, 0))
    print(f"-> of {len(p)} such crops {sum(1 for *_, d in p if d)} have a second row of the class player (harmless), {sum(1 for *_, d in p if not d)} are really lost as a referee")
    print(f"-> the rule would turn {fixed} of {len(p)} into players\n")
    print(f"real referees (truth r) that the analysis shows as a referee: {len(rr)}")
    for n, rank, conf, dup in rr:
        print(f"   {n} referee box(es) in that lens and frame; this one is number {'?' if rank is None else rank + 1} by confidence" + (f" ({conf:.2f})" if conf is not None else "")
              + ("; also a PLAYER row on him" if dup else ""))
    lost = sum(1 for n, rank, _, d in rr if n > 1 and rank not in (None, 0))
    print(f"-> the rule would wrongly turn {lost} of {len(rr)} real referees into players")


if __name__ == "__main__":
    main()
