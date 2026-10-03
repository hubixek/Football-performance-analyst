#!/usr/bin/env python3
"""Compare the possession of two or more analyses of the same match (e.g. made with different detection models) and pick
the moments where they disagree, so that a person can settle them by eye in a few minutes.

Prints how often the analyses agree and where they differ. Then chooses --n moments where the analyses do not all name
the same team (or only some of them name a team), at least --gap seconds apart, and for every moment saves an image of the
lens in which the ball is: the ball marked with a circle, the time and a number. Fill in the column `truth` of
disagreements.csv (0 = team 0 had the ball, 1 = team 1, - = nobody / cannot tell; a ball in flight belongs to the team
of the last player who played it; the answers of the analyses are not in that file, so that they do not steer you) and run the
script again with --score: it tells which analysis was right more often.

Usage:
  python compare_possession.py ~/football/analysis/probe/v7 ~/football/analysis/probe/v8 ~/football/analysis/probe/v9 \\
      --video ~/football/videos/mecz1_dual.mp4 --name mecz1 --labels v7,v8,v9 --out ~/football/analysis/probe/disagreements
  python compare_possession.py ... --score
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import config


def load(folder):
    out = {}
    with open(Path(folder).expanduser() / "possession.csv") as f:
        for r in csv.DictReader(f):
            out[int(r["frame"])] = (float(r["time_s"]), int(r["half"]),
                                    float(r["ball_x"]) if r["ball_x"] else np.nan, float(r["ball_y"]) if r["ball_y"] else np.nan,
                                    int(r["possession_team"]) if r["possession_team"] != "" else -1)
    return out


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="two or more analysis folders of the same match")
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True, help="name of the match (for the calibration files)")
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    ap.add_argument("--labels", help="names of the analyses, comma separated, in the order of the folders (default A,B,C...)")
    ap.add_argument("--n", type=int, default=30, help="how many moments to check")
    ap.add_argument("--gap", type=float, default=20.0, help="moments at least this far apart, s")
    ap.add_argument("--min-run", type=float, default=1.0, help="a disagreement must last at least this long, s")
    ap.add_argument("--out", help="output folder (default: <last folder>/disagreements)")
    ap.add_argument("--score", action="store_true", help="score the filled-in disagreements.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if len(a.folders) < 2:
        raise SystemExit("give at least two analysis folders")
    labels = a.labels.split(",") if a.labels else [chr(65 + i) for i in range(len(a.folders))]
    if len(labels) != len(a.folders):
        raise SystemExit("--labels must have one name per folder")

    out = Path(a.out).expanduser() if a.out else Path(a.folders[-1]).expanduser() / "disagreements"
    sheet = out / "disagreements.csv"
    key = out / ".answers_of_the_analyses.csv"                         # kept apart, so the answers do not steer the eye

    if a.score:
        truth = {r["id"]: r["truth"].strip() for r in csv.DictReader(open(sheet))}
        rows = [dict(r, truth=truth.get(r["id"], "")) for r in csv.DictReader(open(key))]
        done = [r for r in rows if r["truth"] in ("0", "1", "-")]
        if not done:
            raise SystemExit(f"fill in the column 'truth' in {sheet} first (0, 1 or -)")
        def team(v):
            return v.strip() if v.strip() in ("0", "1") else "-"
        print(f"{len(done)} moments settled")
        for lab in labels:
            ok = sum(team(r[lab]) == team(r["truth"]) for r in done)
            errs = Counter((team(r[lab]), team(r["truth"])) for r in done if team(r[lab]) != team(r["truth"]))
            wrong = ", ".join(f"{k[0]} instead of {k[1]}: {v}x" for k, v in errs.most_common(4))
            print(f"  {lab:8s} right in {ok} ({100 * ok / len(done):.0f}%); mistakes: {wrong or 'none'}")
        return

    data = [load(f) for f in a.folders]
    common = sorted(set.intersection(*[set(d) for d in data]))
    if not common:
        raise SystemExit("the analyses have no frames in common")
    T = np.array([[d[f][4] for f in common] for d in data])           # (analyses, frames)
    times = np.array([data[0][f][0] for f in common])
    n = len(common)
    print(f"{n} frames in all {len(data)} analyses")
    for k, lab in enumerate(labels):
        print(f"  {lab}: team 0 {100 * (T[k] == 0).mean():.1f}% of the frames, team 1 {100 * (T[k] == 1).mean():.1f}%, nobody {100 * (T[k] < 0).mean():.1f}%")
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            both = (T[i] >= 0) & (T[j] >= 0)
            print(f"  {labels[i]} vs {labels[j]}: both name a team in {100 * both.mean():.1f}% of the frames, the same team in {100 * (T[i][both] == T[j][both]).mean():.1f}% of those; "
                  f"only one names a team in {100 * ((T[i] >= 0) ^ (T[j] >= 0)).mean():.1f}%")

    dis = ~np.all(T == T[0], axis=0) & np.any(T >= 0, axis=0)
    runs, i = [], 0
    while i < n:
        if dis[i]:
            j = i
            while j + 1 < n and dis[j + 1] and np.all(T[:, j + 1] == T[:, i]) and times[j + 1] - times[j] < 0.5:
                j += 1
            if times[j] - times[i] >= a.min_run:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    rng = np.random.default_rng(a.seed)
    rng.shuffle(runs)
    chosen = []
    for i, j in runs:
        m = (i + j) // 2
        f = common[m]
        pts = np.array([[d[f][2], d[f][3]] for d in data])
        pts = pts[~np.isnan(pts[:, 0])]
        if not len(pts) or any(abs(times[m] - times[k]) < a.gap for k, _ in chosen):
            continue
        chosen.append((m, tuple(np.median(pts, axis=0))))
        if len(chosen) >= a.n:
            break
    chosen.sort()
    print(f"{len(runs)} disagreements lasting at least {a.min_run} s; {len(chosen)} moments chosen")

    from calibrate import pitch_to_pixels
    cdir = Path(a.calib_dir).expanduser()
    calibs = {p: json.loads((cdir / f"{a.name}_{p}.json").read_text()) for p in ("top", "bottom")}
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    rows = []
    for k, (m, (bx, by)) in enumerate(chosen, 1):
        f = common[m]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        h = img.shape[0] // 2
        best = None
        for part, crop in (("top", img[:h]), ("bottom", img[h:2 * h])):
            uv = pitch_to_pixels(calibs[part], np.array([[bx, by]]))[0]
            if np.isfinite(uv).all() and 0 <= uv[0] < crop.shape[1] and 0 <= uv[1] < crop.shape[0]:
                d = abs(uv[0] - crop.shape[1] / 2)
                if best is None or d < best[0]:
                    best = (d, part, crop.copy(), uv)
        if best is None:
            continue
        _, part, crop, uv = best
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(crop, (u, v), 38, (0, 255, 255), 3, cv2.LINE_AA)
        x0, y0 = max(0, u - 520), max(0, v - 300)
        view = crop[y0:y0 + 600, x0:x0 + 1040]
        cv2.rectangle(view, (0, 0), (260, 44), (0, 0, 0), -1)
        cv2.putText(view, f"#{k}  {mmss(times[m])}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        name = f"{k:02d}_{mmss(times[m]).replace(':', 'm')}.jpg"
        cv2.imwrite(str(out / name), view)
        row = {"id": k, "time": mmss(times[m]), "time_s": round(float(times[m]), 1), "image": name, "truth": ""}
        for q, lab in enumerate(labels):
            row[lab] = "-" if T[q][m] < 0 else int(T[q][m])
        rows.append(row)
    cap.release()
    with open(sheet, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "time", "image", "truth"])
        w.writeheader()
        w.writerows({c: r[c] for c in ("id", "time", "image", "truth")} for r in rows)
    with open(key, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "time_s"] + labels)
        w.writeheader()
        w.writerows({c: r[c] for c in ["id", "time_s"] + labels} for r in rows)
    print(f"Saved {len(rows)} images and {sheet}")
    print("Open the images; for every one write in 'truth' who had the ball (0 = team 0, 1 = team 1, - = nobody / cannot tell), "
          "then run the same command with --score. The answers of the analyses are kept in a separate file, so that they do not influence you.")


if __name__ == "__main__":
    main()
