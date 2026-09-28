#!/usr/bin/env python3
"""Feasibility test: does the appearance of players tell teammates apart?

Nothing is assigned or changed. The script takes long, clean pieces of tracks (about 90 % of the
detections of a piece are one player), cuts the player out of the video at several moments and
describes his look by colour: head/hair, shirt, shorts, socks (median Lab colour of the pixels
that are not grass). Then it asks the question that decides whether individual IDs are possible:

  Two pieces of tracks of the same team that exist at the same time are two different players.
  Is the first half of a piece more like the second half of the SAME piece than like the second
  half of a teammate who was on the pitch at the same time?

Reported: AUC (1.0 = always, 0.5 = chance) and the rank-1 accuracy (the own second half is the
closest one among the teammates on the pitch; chance is about 1 / number of teammates).
Rule of thumb: AUC above 0.9 and rank-1 above 60 % - looks are enough to build on; AUC below 0.75 -
teammates look too alike at this resolution and individual IDs need something else.
A picture of the crops (reid_probe_crops.png) shows what the program sees.

Usage:
  python reid_probe.py ~/football/analysis/mecz1_dual/teams.csv --video ~/football/videos/mecz1_dual.mp4 \
      --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from dual_tracks import track_half

BANDS = [("head", 0.0, 0.17), ("shirt", 0.17, 0.50), ("shorts", 0.50, 0.72), ("socks", 0.72, 1.0)]


def load(teams_path, L, W, half, t0, t1, margin=0.3):
    meta, frames = [], defaultdict(list)
    with open(teams_path) as f:
        for r in csv.DictReader(f):
            if r["class"] != "player" or r["team"] not in ("0", "1") or r.get("gk") == "1" or int(r["half"]) != half:
                continue
            t = float(r["time_s"])
            x, y = float(r["x_m"]), float(r["y_m"])
            if not (t0 <= t < t1) or not (-margin <= x <= L + margin and -margin <= y <= W + margin):
                continue
            meta.append((int(r["frame"]), r["cam"], float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
            frames[t].append((x, y, int(r["team"]), False, len(meta) - 1))
    return meta, sorted(frames.items())


def appearance(crop, grass_lab):
    """Median Lab colour of head, shirt, shorts and socks without grass pixels (12 numbers)."""
    h, w = crop.shape[:2]
    if h < 24 or w < 8:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)[:, int(w * 0.2):int(w * 0.8)]
    feat = []
    for _, a, b in BANDS:
        reg = lab[int(a * h):max(int(b * h), int(a * h) + 1)].reshape(-1, 3)
        keep = np.linalg.norm(reg - grass_lab, axis=1) > 22
        feat += list(np.median(reg[keep] if keep.sum() >= 5 else reg, axis=0))
    return np.array(feat, np.float32)


def read_features(video, wanted):
    """wanted: frame -> [(key, cam, x1, y1, x2, y2)]. Returns key -> (features, small crop)."""
    cap = cv2.VideoCapture(str(video))
    out = {}
    for n, frame in enumerate(sorted(wanted)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            continue
        h = img.shape[0] // 2
        halves = {"top": img[:h], "bottom": img[h:2 * h]}
        grass = {}
        for key, cam, x1, y1, x2, y2 in wanted[frame]:
            if cam not in grass:
                low = cv2.cvtColor(cv2.resize(halves[cam][int(h * 0.65):], None, fx=0.25, fy=0.25), cv2.COLOR_BGR2LAB)
                grass[cam] = np.median(low.reshape(-1, 3), axis=0).astype(np.float32)
            crop = halves[cam][max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
            if crop.size == 0:
                continue
            f = appearance(crop, grass[cam])
            if f is not None:
                out[key] = (f, cv2.resize(crop, (32, 64)))
        if (n + 1) % 200 == 0:
            print(f"  read {n + 1}/{len(wanted)} frames", flush=True)
    cap.release()
    return out


def evaluate(pieces, min_overlap=5.0):
    allf = np.array([p["A"] for p in pieces] + [p["B"] for p in pieces])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    z = lambda v: (v - mu) / sd
    dist = lambda a, b: float(np.linalg.norm(z(a) - z(b)))
    pos, neg, hits, chances = [], [], [], []
    for i, p in enumerate(pieces):
        d_own = dist(p["A"], p["B"])
        pos.append(d_own)
        rivals = [q for j, q in enumerate(pieces) if j != i and q["team"] == p["team"]
                  and min(p["t1"], q["t1"]) - max(p["t0"], q["t0"]) >= min_overlap]
        neg += [dist(p["A"], q["B"]) for q in rivals]
        if len(rivals) >= 2:
            hits.append(d_own < min(dist(p["A"], q["B"]) for q in rivals))
            chances.append(1 / (len(rivals) + 1))
    pos, neg = np.array(pos), np.array(neg)
    auc = float(np.mean([(a < neg).mean() + 0.5 * (a == neg).mean() for a in pos])) if len(neg) else float("nan")
    return auc, (float(np.mean(hits)) if hits else float("nan")), (float(np.mean(chances)) if chances else float("nan")), len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", help="teams.csv from dual_teams.py")
    ap.add_argument("--video", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start-s", type=float, help="start in the recording (default: 60 s after the start of the half)")
    ap.add_argument("--minutes", type=float, default=8.0, help="length of the part of the match used")
    ap.add_argument("--min-piece-s", type=float, default=14.0)
    ap.add_argument("--samples", type=int, default=16, help="crops per piece")
    ap.add_argument("--min-height", type=float, default=28.0, help="smallest player height in pixels")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    teams_path = Path(a.teams).expanduser()
    out = Path(a.out_dir).expanduser() if a.out_dir else teams_path.parent
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    times = []
    with open(teams_path) as f:
        for r in csv.DictReader(f):
            if int(r["half"]) == a.half:
                times.append(float(r["time_s"]))
    t0 = a.start_s if a.start_s is not None else min(times) + 60
    meta, frames = load(teams_path, L, W, a.half, t0, t0 + a.minutes * 60)
    print(f"{len(meta)} detections in {a.minutes:.0f} min of half {a.half}")

    tracks = [tr for tr in track_half(frames) if tr.pts[-1][0] - tr.pts[0][0] >= a.min_piece_s and tr.team in (0, 1)]
    print(f"{len(tracks)} clean pieces of tracks of at least {a.min_piece_s:.0f} s")
    wanted, keys = defaultdict(list), {}
    for pi, tr in enumerate(tracks):
        good = [k for k in range(len(tr.ids)) if (meta[tr.ids[k]][5] - meta[tr.ids[k]][3]) >= a.min_height]
        if len(good) < a.samples:
            continue
        for si, k in enumerate(np.linspace(0, len(good) - 1, a.samples).astype(int)):
            row = tr.ids[good[k]]
            fr, cam, x1, y1, x2, y2 = meta[row]
            keys[(pi, si)] = tr.pts[good[k]][0]
            wanted[fr].append(((pi, si), cam, x1, y1, x2, y2))
    print(f"reading {sum(len(v) for v in wanted.values())} crops from {len(wanted)} video frames ...")
    feats = read_features(a.video, wanted)

    pieces = []
    for pi, tr in enumerate(tracks):
        got = sorted((keys[(pi, k)], feats[(pi, k)]) for k in range(a.samples) if (pi, k) in feats)
        if len(got) < 10:
            continue
        h = len(got) // 2
        pieces.append({"id": pi, "team": tr.team, "t0": tr.pts[0][0], "t1": tr.pts[-1][0],
                       "A": np.mean([g[1][0] for g in got[:h]], axis=0), "B": np.mean([g[1][0] for g in got[h:]], axis=0),
                       "crops": [g[1][1] for g in got]})
    print(f"{len(pieces)} pieces with enough crops")
    if len(pieces) < 6:
        raise SystemExit("Too few long pieces - try more minutes (--minutes) or a smaller --min-piece-s")

    print("\n=== can the looks tell teammates apart? ===")
    aucs, accs = [], []
    for team in (0, 1):
        ps = [p for p in pieces if p["team"] == team]
        if len(ps) >= 4:
            auc, acc, chance, n = evaluate(ps)
            aucs.append(auc)
            accs.append(acc)
            print(f"team {team}: {len(ps)} pieces | AUC {auc:.2f} | rank-1 accuracy {acc:.0%} over {n} pieces (chance about {chance:.0%})")
    auc, acc = float(np.nanmean(aucs)), float(np.nanmean(accs))
    print(f"average over the teams: AUC {auc:.2f}, rank-1 accuracy {acc:.0%}")
    verdict = ("promising: looks are enough to build on" if auc >= 0.9 and acc >= 0.6 else
               "unclear: some information in the looks, not enough alone" if auc >= 0.75 else
               "poor: teammates look too alike at this resolution")
    print(f"verdict: {verdict}")
    print("(a necessary test, not a sufficient one: the two halves of a piece are close in time, which is a little optimistic)")

    # picture: 6 teammates on the pitch at the same time, 8 crops each
    best = max(pieces, key=lambda p: sum(1 for q in pieces if q["team"] == p["team"] and min(p["t1"], q["t1"]) - max(p["t0"], q["t0"]) >= 5))
    group = [best] + [q for q in pieces if q is not best and q["team"] == best["team"] and min(best["t1"], q["t1"]) - max(best["t0"], q["t0"]) >= 5]
    group = group[:6]
    rows = []
    for p in group:
        idx = np.linspace(0, len(p["crops"]) - 1, 8).astype(int)
        rows.append(np.hstack([p["crops"][i] for i in idx]))
    cv2.imwrite(str(out / "reid_probe_crops.png"), cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(f"Saved {out / 'reid_probe_crops.png'}: every row is one player of team {best['team']} (8 moments), the players were on the pitch together")


if __name__ == "__main__":
    main()
