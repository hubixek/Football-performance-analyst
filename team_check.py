#!/usr/bin/env python3
"""How well are players assigned to teams? A blind check on a sample of player crops.

Takes --n player detections spread over the whole match (inside the pitch, goalkeepers included, both lenses),
cuts them out of the video and puts them on sheets of numbered tiles (team_check_1.jpg, ...). You write for every
number who it is in team_check.csv:
  0 = team 0 (the darker kit), 1 = team 1 (the lighter kit), r = referee, x = not a player / cannot tell.
The answers of the analysis are kept in a separate file so that they do not steer you.
Then run the script again with --score: it prints how often the analysis was right, the mistakes per team, and saves
team_check_errors.jpg with the wrongly assigned crops, to see what the mistakes have in common.

Usage:
  python team_check.py ~/football/analysis/mecz1_dual --video ~/football/videos/mecz1_dual.mp4
  python team_check.py ~/football/analysis/mecz1_dual --video ~/football/videos/mecz1_dual.mp4 --score
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H, PER_ROW, ROWS = 120, 220, 10, 5


def crop_of(img, r, pad=0.15):
    h = img.shape[0] // 2
    lens = img[:h] if r["cam"] == "top" else img[h:2 * h]
    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    w, hh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * w)), int(min(lens.shape[1], x2 + pad * w))
    y1, y2 = int(max(0, y1 - pad * hh)), int(min(lens.shape[0], y2 + pad * hh))
    c = lens[y1:y2, x1:x2]
    if c.size == 0:
        return np.zeros((TILE_H, TILE_W, 3), np.uint8)
    s = min(TILE_W / c.shape[1], (TILE_H - 24) / c.shape[0])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    tile = np.full((TILE_H, TILE_W, 3), 30, np.uint8)
    oy, ox = 24 + (TILE_H - 24 - c.shape[0]) // 2, (TILE_W - c.shape[1]) // 2
    tile[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return tile


def sheets(tiles, labels, prefix):
    out = []
    per = PER_ROW * ROWS
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        rows = []
        for r0 in range(0, len(chunk), PER_ROW):
            row = []
            for k, t in enumerate(chunk[r0:r0 + PER_ROW]):
                t = t.copy()
                cv2.putText(t, labels[s + r0 + k], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                row.append(t)
            while len(row) < PER_ROW:
                row.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
            rows.append(np.hstack(row))
        path = f"{prefix}_{s // per + 1}.jpg"
        cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder (teams.csv)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output folder (default: <folder>/team_check)")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--answers-from", help="--score: take the answers from this teams.csv (e.g. after a change of the method) "
                    "instead of the ones saved when the crops were made")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    out = Path(a.out).expanduser() if a.out else folder / "team_check"
    sheet_csv, key_csv = out / "team_check.csv", out / ".answers_of_the_analysis.csv"

    if a.score:
        truth = {r["id"]: r["truth"].strip().lower() for r in csv.DictReader(open(sheet_csv))}
        key = list(csv.DictReader(open(key_csv)))
        if a.answers_from:
            now = {}
            with open(Path(a.answers_from).expanduser()) as f:
                for r in csv.DictReader(f):
                    now[(r["frame"], r["cam"], round(float(r["x1"]), 1), round(float(r["y1"]), 1))] = r
            missing = 0
            for k in key:
                r = now.get((k["frame"], k["cam"], round(float(k["x1"]), 1), round(float(k["y1"]), 1)))
                if r is None:
                    missing += 1
                    continue
                k["class"], k["team"], k["gk"] = r["class"], r.get("team", ""), r.get("gk", "")
            print(f"answers taken from {a.answers_from}" + (f" ({missing} crops not found there, old answers kept)" if missing else ""))
        done = [k for k in key if truth.get(k["id"], "") in ("0", "1", "r")]
        if not done:
            raise SystemExit(f"fill in the column 'truth' in {sheet_csv} first (0, 1, r or x)")
        def ana(k):
            return "r" if k["class"] == "referee" else (k["team"] if k["team"] in ("0", "1") else "?")
        ok = sum(ana(k) == truth[k["id"]] for k in done)
        print(f"{len(done)} crops judged (x = {sum(1 for v in truth.values() if v == 'x')} left out): "
              f"the analysis is right in {ok} ({100 * ok / len(done):.0f}%)")
        conf = Counter((truth[k["id"]], ana(k)) for k in done)
        print("really \\ analysis:   team 0   team 1   referee   unknown")
        for t, name in (("0", "team 0"), ("1", "team 1"), ("r", "referee")):
            print(f"{name:18s} {conf[(t, '0')]:7d} {conf[(t, '1')]:8d} {conf[(t, 'r')]:9d} {conf[(t, '?')]:9d}")
        gk = [k for k in done if k.get("gk") == "1"]
        if gk:
            print(f"goalkeepers among them: {len(gk)}, right {sum(ana(k) == truth[k['id']] for k in gk)}")
        wrong = [k for k in done if ana(k) != truth[k["id"]]]
        if wrong:
            cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
            tiles, labels = [], []
            for k in sorted(wrong, key=lambda k: int(k["frame"])):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(k["frame"]))
                ok_, img = cap.read()
                if ok_:
                    tiles.append(crop_of(img, k))
                    labels.append(f"#{k['id']} {ana(k)}>{truth[k['id']]}")
            cap.release()
            paths = sheets(tiles, labels, str(out / "team_check_errors"))
            print(f"Saved {', '.join(paths)}: the wrong ones, label 'analysis>really'")
        return

    rows = []
    with open(folder / "teams.csv") as f:
        for r in csv.DictReader(f):
            if r["class"] in ("player", "referee") and 0 <= float(r["x_m"]) <= 56 and 0 <= float(r["y_m"]) <= 32.4:
                rows.append(r)
    if len(rows) < a.n:
        raise SystemExit("too few player detections")
    rng = np.random.default_rng(a.seed)
    by_minute = defaultdict(list)
    for r in rows:
        by_minute[int(float(r["time_s"]) // 60)].append(r)
    minutes = sorted(by_minute)
    pick = []
    while len(pick) < a.n:                                        # spread over the whole match: one per minute in turn
        for m in rng.permutation(minutes):
            if len(pick) >= a.n:
                break
            lst = by_minute[m]
            pick.append(lst[int(rng.integers(len(lst)))])
    pick.sort(key=lambda r: int(r["frame"]))
    order = rng.permutation(len(pick))                            # tile numbers in random order: no pattern over time
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    tiles = {}
    for r in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, img = cap.read()
        if ok:
            tiles[id(r)] = crop_of(img, r)
    cap.release()
    numbered = [(i + 1, pick[j]) for i, j in enumerate(order) if id(pick[j]) in tiles]
    numbered.sort()
    paths = sheets([tiles[id(r)] for _, r in numbered], [f"#{i}" for i, _ in numbered], str(out / "team_check"))
    with open(sheet_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "truth"])
        for i, _ in numbered:
            w.writerow([i, ""])
    with open(key_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "frame", "cam", "x1", "y1", "x2", "y2", "class", "team", "gk"])
        for i, r in numbered:
            w.writerow([i, r["frame"], r["cam"], r["x1"], r["y1"], r["x2"], r["y2"], r["class"], r.get("team", ""), r.get("gk", "")])
    print(f"Saved {len(numbered)} crops on {', '.join(paths)} and {sheet_csv}")
    print("For every number write in 'truth': 0 = darker kit, 1 = lighter kit, r = referee, x = not a player / cannot tell")


if __name__ == "__main__":
    main()
