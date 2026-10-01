#!/usr/bin/env python3
"""What does the model do with referees and players, when the validation labels hold only the people ON the pitch?

The mAP of the class `referee` is then misleading: a referee standing behind the line is detected correctly but counts
as a false positive. This script compares predictions and labels box by box (confidence as in the analysis, 0.1 by
default) and answers the questions that matter for the analysis, which drops everything outside the pitch anyway:

  1. labelled referees (on the pitch): found as referee / found, but only as a player / not found at all,
  2. labelled players: found as player (of these: with an extra referee box on the same person = a duplicate, harmless
     for the teams) / found ONLY as a referee (the player is lost) / not found,
  3. predicted referees by what they sit on: a labelled referee / a labelled player / nothing labelled (an unlabelled
     person, e.g. the referee or the bench outside the pitch).
Sheets of crops to judge by eye:
  referee_check_unlabelled.jpg   predicted referees that match no label
  referee_check_lost_players.jpg labelled players found only as a referee
  referee_check_missed.jpg       labelled referees that were not found as a referee (from the labels, label: miss / player)

Usage:
  python val_referee_check.py ~/football/dataset_dual --split val_night --model ~/football/runs/detect/runs/v8_dual/weights/best.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H, PER_ROW = 110, 200, 12


def read_labels(path, W, H):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, xc, yc, w, h = int(p[0]), *(float(v) for v in p[1:])
                out.append((c, np.array([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tile(img, box, pad=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * w)), int(min(img.shape[1], x2 + pad * w))
    y1, y2 = int(max(0, y1 - pad * h)), int(min(img.shape[0], y2 + pad * h))
    c = img[y1:y2, x1:x2]
    t = np.full((TILE_H, TILE_W, 3), 30, np.uint8)
    if c.size == 0:
        return t
    s = min(TILE_W / c.shape[1], TILE_H / c.shape[0])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    oy, ox = (TILE_H - c.shape[0]) // 2, (TILE_W - c.shape[1]) // 2
    t[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return t


def sheet(tiles, labels, path):
    if not tiles:
        return False
    tiles = tiles[:96]
    rows = []
    for r0 in range(0, len(tiles), PER_ROW):
        row = []
        for k, t in enumerate(tiles[r0:r0 + PER_ROW]):
            t = t.copy()
            cv2.putText(t, labels[r0 + k], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(t)
        while len(row) < PER_ROW:
            row.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def med(v):
    return f"{np.median(v):.2f}" if v else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset folder with images/<split> and labels/<split>")
    ap.add_argument("--split", default="val_night")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.1, help="the analysis (dual_detect.py) uses 0.1")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--out", help="folder for the sheets (default: the dataset folder)")
    a = ap.parse_args()

    from ultralytics import YOLO
    root = Path(a.dataset).expanduser()
    out = Path(a.out).expanduser() if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(a.model).expanduser()))
    inv = {v: k for k, v in model.names.items()}
    P, R = inv["player"], inv["referee"]
    imgs = sorted(p for p in (root / "images" / a.split).iterdir() if p.suffix.lower() in (".jpg", ".png"))

    gt_ref, gt_pl, pr_ref = Counter(), Counter(), Counter()
    conf_of = {"on a labelled referee": [], "on a labelled player": [], "on nothing labelled": []}
    n_pred_player_unl = 0
    t_unl, l_unl, t_lost, l_lost, t_miss, l_miss = [], [], [], [], [], []
    for path in imgs:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        gt = read_labels(root / "labels" / a.split / (path.stem + ".txt"), W, H)
        res = model.predict(str(path), imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        cls = res.boxes.cls.tolist()
        cfs = res.boxes.conf.tolist() if getattr(res.boxes, "conf", None) is not None else [1.0] * len(cls)
        pred = [(int(c), np.array(b), float(cf)) for c, b, cf in zip(cls, res.boxes.xyxy.tolist(), cfs) if int(c) in (P, R)]
        gt_people = [(c, b) for c, b in gt if c in (P, R)]
        tag = path.stem[-9:]
        for gc, gb in gt_people:
            has_p = any(pc == P and iou(gb, pb) >= a.iou for pc, pb, _ in pred)
            has_r = any(pc == R and iou(gb, pb) >= a.iou for pc, pb, _ in pred)
            if gc == R:
                if has_r:
                    gt_ref["found as referee"] += 1
                else:
                    gt_ref["found, but only as a player" if has_p else "not found at all"] += 1
                    t_miss.append(tile(img, gb))
                    l_miss.append(("player " if has_p else "miss ") + tag)
            else:
                if has_p:
                    gt_pl["found as player"] += 1
                    gt_pl["  of these with an extra referee box (duplicate)"] += int(has_r)
                elif has_r:
                    gt_pl["found ONLY as a referee (lost)"] += 1
                    t_lost.append(tile(img, gb))
                    l_lost.append(tag)
                else:
                    gt_pl["not found"] += 1
        for c, b, cf in pred:
            best = max(((iou(b, gb), gc) for gc, gb in gt_people), default=(0.0, None))
            matched = best[0] >= a.iou
            if c == R:
                if matched and best[1] == R:
                    key = "on a labelled referee"
                elif matched:
                    key = "on a labelled player"
                else:
                    key = "on nothing labelled"
                    t_unl.append(tile(img, b))
                    l_unl.append(f"{cf:.2f} {tag[-6:]}")
                pr_ref[key] += 1
                conf_of[key].append(cf)
            elif not matched:
                n_pred_player_unl += 1

    print(f"{len(imgs)} images of {a.split}, model {Path(a.model).parent.parent.name}, confidence >= {a.conf}")
    tot = sum(gt_ref.values())
    print(f"\n1. labelled referees (on the pitch): {tot}")
    for k in ("found as referee", "found, but only as a player", "not found at all"):
        print(f"   {k:48s} {gt_ref[k]:4d}  ({100 * gt_ref[k] / max(tot, 1):.0f}%)")
    tot = sum(gt_pl.values()) - gt_pl["  of these with an extra referee box (duplicate)"]
    print(f"\n2. labelled players: {tot}")
    for k in ("found as player", "  of these with an extra referee box (duplicate)", "found ONLY as a referee (lost)", "not found"):
        print(f"   {k:48s} {gt_pl[k]:4d}  ({100 * gt_pl[k] / max(tot, 1):.0f}%)")
    tot = sum(pr_ref.values())
    print(f"\n3. predicted referees: {tot}   (median confidence in brackets)")
    for k in ("on a labelled referee", "on a labelled player", "on nothing labelled"):
        print(f"   {k:48s} {pr_ref[k]:4d}  ({100 * pr_ref[k] / max(tot, 1):.0f}%)  [{med(conf_of[k])}]")
    print(f"\nfor comparison: predicted players on nothing labelled: {n_pred_player_unl}")
    for t, l, name, what in ((t_unl, l_unl, "referee_check_unlabelled.jpg", "predicted referees on nothing labelled (confidence, frame)"),
                             (t_lost, l_lost, "referee_check_lost_players.jpg", "labelled players found only as a referee"),
                             (t_miss, l_miss, "referee_check_missed.jpg", "labelled referees not found as a referee")):
        if sheet(t, l, out / name):
            print(f"Saved {out / name}: {what} ({min(len(t), 96)} of {len(t)})")


if __name__ == "__main__":
    main()
