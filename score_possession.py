#!/usr/bin/env python3
"""Score the possession of one or more analyses of the same match on the moments you judged by eye
(the folder made by compare_possession.py: disagreements.csv with your column `truth`, and the hidden file
.answers_of_the_analyses.csv with the times).

For every analysis folder: how many of the judged moments it gets right, which mistakes it makes, and the possession of
the whole match. With --table also one line per moment (truth and what every analysis says).

Usage:
  python score_possession.py ~/football/analysis/mecz1_dual ~/football/analysis/mecz1_dual_v8 ~/football/analysis/mecz1_dual_v9
"""
import argparse
import bisect
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--truth-dir", default=str(Path.home() / "football" / "analysis" / "mecz1_dual_v8" / "disagreements"),
                    help="folder with disagreements.csv (your truth) and .answers_of_the_analyses.csv (times)")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    d = Path(a.truth_dir).expanduser()
    truth = {r["id"]: r["truth"].strip() for r in csv.DictReader(open(d / "disagreements.csv"))}
    times = {r["id"]: float(r["time_s"]) for r in csv.DictReader(open(d / ".answers_of_the_analyses.csv"))}
    ids = [i for i in times if truth.get(i, "") != ""]
    tr = {i: (truth[i] if truth[i] in ("0", "1") else "-") for i in ids}
    print(f"{len(ids)} judged moments (truth: team 0 {sum(v == '0' for v in tr.values())}, team 1 {sum(v == '1' for v in tr.values())}, "
          f"nobody {sum(v == '-' for v in tr.values())})\n")
    preds = {}
    for folder in a.folders:
        f = Path(folder).expanduser()
        rows = [(float(r["time_s"]), r["possession_team"] or "-") for r in csv.DictReader(open(f / "possession.csv"))]
        ts = [r[0] for r in rows]
        p = {i: rows[min(bisect.bisect_left(ts, times[i]), len(rows) - 1)][1] for i in ids}
        preds[f.name] = p
        ok = sum(p[i] == tr[i] for i in ids)
        errs = Counter((p[i], tr[i]) for i in ids if p[i] != tr[i])
        s = json.load(open(f / "summary.json"))
        pp = s["possession_percent"]
        wrong = ", ".join(f"{k[0]} instead of {k[1]}: {v}x" for k, v in errs.most_common(4))
        print(f"{f.name:16s} right {ok:2d} of {len(ids)} | possession {pp['team_0']}% / {pp['team_1']}% (assigned {s['assigned_share']}%)")
        print(f"{'':16s} mistakes: {wrong or 'none'}")
    if a.table:
        names = list(preds)
        print("\n id  time    truth  " + "  ".join(f"{n[-8:]:>8s}" for n in names))
        for i in ids:
            t = times[i]
            print(f"{i:>3s}  {int(t) // 60:02d}:{int(t) % 60:02d}  {tr[i]:>5s}  " +
                  "  ".join(f"{preds[n][i]:>8s}{'' if preds[n][i] == tr[i] else '*'}"[:9].rjust(9) for n in names))


if __name__ == "__main__":
    main()
