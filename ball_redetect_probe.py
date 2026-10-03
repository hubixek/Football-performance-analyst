#!/usr/bin/env python3
"""Can the gaps of the ball track be filled by detecting again with a low confidence? Measured by hiding pieces of a complete track.

The ball track (dual_ball.py) is known in 12-80% of the frames in the 8 s before a goal, which is why shots cannot be read from it. This probe
detects the ball again with --conf far below the 0.1 of dual_detect.py, on selected windows only, keeps up to --max-cands candidates per frame
and fills a gap with the cheapest chain of candidates that joins the anchor before it to the anchor after it (confident candidates, no jump faster than --vmax m/s). The frames in which the
ball was DETECTED (not interpolated) are anchors: the path must start and end on them, so it cannot wander off after another object.

Hold-out test (the only honest one, because the truth is known): in windows where the track is detected in almost every frame a piece of
--gaps seconds is hidden, the gap is filled and compared with the hidden detections. Reported for the low-confidence detector and for the
linear interpolation that dual_ball.py uses now: how often a frame is filled, and how often the filled position is within 1 m of the truth.
Goal windows: the same filling on the 6 s before every goal, with a clip of each (--clips DIR, the filled ball circled) to look at: a count
of frames does not tell a real ball from a plausible chain of false candidates.

Needs the video, the model, the calibration, goals_auto.csv (detect_goals.py), ball.csv and possession.csv of the analysis folder.
The detections of the windows are kept in --cache, so that the filling can be tried again without the GPU.

Usage:
  python ball_redetect_probe.py ~/football/analysis/mecz1_dual ~/football/videos/mecz1_dual.mp4 --match matches/mecz1.json \\
      --model ~/football/runs/detect/runs/v8_dual/weights/best.pt --pitch pitch/pitch_6v6.json --clips ~/football/analysis/mecz1_dual/redetect
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import config

CAMS = ("top", "bottom")


def fill_gaps(cands, anchors, times, vmax=35.0, miss=1.0, conf_weight=1.0):
    """Positions of every frame: an anchor where one exists, and in a gap between two anchors the best chain of candidates joining them.

    cands: per frame a list of (x, y, conf); anchors: (n, 2) array with NaN where the ball was not detected. Every gap is solved alone as the
    cheapest path from the anchor before it to the anchor after it (dynamic programming). A step from a frame to a later one (frames without
    a ball in between cost `miss` each) is impossible if it needs more than vmax m/s; a candidate costs conf_weight x (1 - conf) and a step
    0.005 x speed. So the filled ball always continues the track on both sides; a gap that cannot be joined stays empty, and so do the frames
    before the first and after the last anchor. Returns an (n, 2) array with NaN where there is no ball."""
    n = len(cands)
    out = anchors.copy()
    known = np.flatnonzero(~np.isnan(anchors[:, 0]))
    for a, b in zip(known[:-1], known[1:]):
        if b - a < 2:
            continue
        nodes = [(a, anchors[a][0], anchors[a][1], 0.0)]
        for i in range(a + 1, b):
            nodes += [(i, c[0], c[1], conf_weight * (1 - c[2])) for c in cands[i]]
        nodes.append((b, anchors[b][0], anchors[b][1], 0.0))
        best = [np.inf] * len(nodes)
        prev = [-1] * len(nodes)
        best[0] = 0.0
        for j in range(1, len(nodes)):
            fj, xj, yj, cj = nodes[j]
            for k in range(j):
                fk, xk, yk, _ = nodes[k]
                if fk >= fj or best[k] == np.inf:
                    continue
                dt = times[fj] - times[fk]
                d = float(np.hypot(xj - xk, yj - yk))
                if dt <= 0 or d / dt > vmax:
                    continue
                v = best[k] + (fj - fk - 1) * miss + 0.005 * d / dt + cj
                if v < best[j]:
                    best[j], prev[j] = v, k
        if best[-1] == np.inf:
            continue
        j = prev[-1]
        while j > 0:
            out[nodes[j][0]] = (nodes[j][1], nodes[j][2])
            j = prev[j]
    return out


def linear_fill(anchors, times):
    """The interpolation of dual_ball.py: straight lines between the anchors, nothing before the first or after the last one."""
    out = np.full_like(anchors, np.nan)
    ok = ~np.isnan(anchors[:, 0])
    if ok.sum() < 2:
        return anchors.copy()
    inside = (times >= times[ok][0]) & (times <= times[ok][-1])
    for c in (0, 1):
        out[inside, c] = np.interp(times[inside], times[ok], anchors[ok, c])
    return out


def holdout_scores(truth, mask, filled, tol=1.0):
    """Over the hidden frames that have a true position: how many there are, how many were filled, how many within tol m of the truth."""
    eval_ = mask & ~np.isnan(truth[:, 0])
    got = eval_ & ~np.isnan(filled[:, 0])
    err = np.hypot(truth[got, 0] - filled[got, 0], truth[got, 1] - filled[got, 1])
    return {"hidden": int(eval_.sum()), "filled": int(got.sum()), "right": int((err < tol).sum())}


def on_the_pitch(c, L, W, margin=0.5, goal_half=3.5, behind=2.0):
    """A ball candidate is plausible inside the lines (with a margin) or just behind a goal line within the goal mouth (the ball in the net).
    A ball beside the pitch (a spare ball by the bench, a ball on the road) is not the ball of the game."""
    x, y = c[0], c[1]
    if -margin <= x <= L + margin and -margin <= y <= W + margin:
        return True
    return (-behind <= x < -margin or L + margin < x <= L + behind) and abs(y - W / 2) <= goal_half


def detect_window(cap, cv2, model, ball_ids, calibs, pixels_to_pitch, frames, L, W, conf, imgsz, max_cands):
    """Ball candidates of the listed frames (low confidence), as per frame lists of (x_m, y_m, conf)."""
    want = {int(f): k for k, f in enumerate(frames)}
    cands = [[] for _ in frames]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    for fr in range(int(frames[0]), int(frames[-1]) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if fr not in want:
            continue
        h = img.shape[0] // 2
        res = model.predict([img[:h], img[h:2 * h]], imgsz=imgsz, conf=conf, verbose=False)
        found = []
        for cam, r in zip(CAMS, res):
            for box, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cl) not in ball_ids:
                    continue
                u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                xy = pixels_to_pitch(calibs[cam], np.array([[u, v]]))[0]
                if np.isfinite(xy).all() and -2.0 < xy[0] < L + 2.0 and -2.0 < xy[1] < W + 2.0:
                    found.append((float(xy[0]), float(xy[1]), float(c)))
        found.sort(key=lambda c: -c[2])
        kept = []
        for c in found:                                                   # the same ball seen by both lenses
            if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 1.0 for k in kept):
                kept.append(c)
        cands[want[fr]] = kept[:max_cands]
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("video")
    ap.add_argument("--match", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    ap.add_argument("--conf", type=float, default=0.02, help="detection confidence (dual_detect.py uses 0.1)")
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--pre", type=float, default=6.0, help="goal windows: seconds before the goal")
    ap.add_argument("--post", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=35.0, help="fastest ball, m/s")
    ap.add_argument("--miss", type=float, default=1.0, help="cost of a frame without a ball")
    ap.add_argument("--conf-weight", type=float, default=1.0, help="cost of a candidate = this x (1 - confidence)")
    ap.add_argument("--controls", type=int, default=8, help="hold-out windows: 10 s in which the ball is detected in almost every frame")
    ap.add_argument("--gaps", default="0.5,1,2", help="hidden pieces of the hold-out test, s")
    ap.add_argument("--cache", help="json file with the candidates of the windows (read if it exists, else written)")
    ap.add_argument("--clips")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import cv2
    from calibrate import pixels_to_pitch

    folder = Path(a.folder).expanduser()
    with open(folder / "possession.csv") as f:
        poss = list(csv.DictReader(f))
    fr_all = np.array([int(r["frame"]) for r in poss])
    t_all = np.array([float(r["time_s"]) for r in poss])
    half_all = np.array([int(r["half"]) for r in poss])
    det = {int(r["frame"]): (float(r["x_m"]), float(r["y_m"])) for r in csv.DictReader(open(folder / "ball.csv")) if r["source"] == "detected"}
    detected_all = np.array([det.get(int(f), (np.nan, np.nan)) for f in fr_all])
    goals = [(float(r["goal_s"]), r["goal"]) for r in csv.DictReader(open(folder / "goals_auto.csv")) if r.get("category") == "goal" and r.get("goal_s")]
    windows = [(g - a.pre, g + a.post, f"goal {lab}") for g, lab in goals]
    rng = np.random.default_rng(a.seed)
    n_win = 150
    full = [i for i in rng.permutation(len(t_all) - n_win)
            if half_all[i] == half_all[i + n_win - 1] and np.mean(~np.isnan(detected_all[i:i + n_win, 0])) >= 0.95
            and all(abs(t_all[i] - w[0]) > 30 for w in windows)]
    chosen = []
    for i in full:
        if all(abs(t_all[i] - t_all[j]) >= 20 for j in chosen):
            chosen.append(i)
        if len(chosen) >= a.controls:
            break
    controls = [(t_all[i], t_all[i] + 10.0, f"control {int(t_all[i]) // 60:02d}:{int(t_all[i]) % 60:02d}") for i in chosen]

    cfg = json.loads(Path(a.match).read_text())
    cdir = Path(a.calib_dir).expanduser()
    calibs = {c: json.loads((cdir / f"{cfg['name']}_{c}.json").read_text()) for c in CAMS}
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    cache_path = Path(a.cache).expanduser() if a.cache else None
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    model, ball_ids = None, []
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))

    def window(t0, t1, label):
        nonlocal model, ball_ids
        idx = np.flatnonzero((t_all >= t0) & (t_all <= t1))
        key = f"{label}|{a.conf}"
        if key not in cache:
            if model is None:
                from ultralytics import YOLO
                model = YOLO(str(Path(a.model).expanduser()))
                ball_ids = [k for k, v in model.names.items() if v == "ball"]
            cache[key] = detect_window(cap, cv2, model, ball_ids, calibs, pixels_to_pitch, fr_all[idx], L, W, a.conf, a.imgsz, a.max_cands)
            if cache_path:
                cache_path.write_text(json.dumps(cache))
        return idx, [[tuple(c) for c in cs if on_the_pitch(c, L, W)] for cs in cache[key]]

    # ------------------------------------------------------------ hold-out test
    gaps = [float(g) for g in a.gaps.split(",")]
    methods = ("detector, ball among the candidates", "detector, ball NOT among the candidates", "linear interpolation")
    tot = {g: {m: [0, 0, 0] for m in methods} for g in gaps}
    for t0, t1, label in controls:
        idx, cands = window(t0, t1, label)
        times, truth = t_all[idx], detected_all[idx]
        for g in gaps:
            n_gap = int(round(g * 15))
            start = len(idx) // 2 - n_gap // 2
            mask = np.zeros(len(idx), bool)
            mask[start:start + n_gap] = True
            anchors = truth.copy()
            anchors[mask] = np.nan
            without = [[c for c in cs if np.isnan(truth[i, 0]) or not mask[i] or np.hypot(c[0] - truth[i, 0], c[1] - truth[i, 1]) > 2.0]
                       for i, cs in enumerate(cands)]                                    # the ball itself removed from the hidden frames
            for name, filled in zip(methods, (fill_gaps(cands, anchors, times, a.vmax, a.miss, a.conf_weight),
                                              fill_gaps(without, anchors, times, a.vmax, a.miss, a.conf_weight), linear_fill(anchors, times))):
                s = holdout_scores(truth, mask, filled)
                for k, v in enumerate((s["hidden"], s["filled"], s["right"])):
                    tot[g][name][k] += v
    print(f"hold-out test: {len(controls)} windows in which the ball is detected in almost every frame, a piece of the track hidden, then filled")
    print("'ball among the candidates' is circular (the hidden ball was found at confidence >= 0.1, so the low-confidence run finds it again): it only")
    print("shows that the filling follows a ball that is there. 'ball NOT among the candidates' removes the ball from the hidden frames: everything filled")
    print("is then a FALSE ball (a chain of other objects), so 'filled' is the false-fill rate of the method.")
    print(f"{'hidden':>8s}  {'method':42s} {'frames':>7s} {'filled':>7s} {'within 1 m':>11s}")
    for g in gaps:
        for name, (h, fl, ok) in tot[g].items():
            print(f"{g:6.1f} s  {name:42s} {h:7d} {fl:7d} {ok:11d}   filled {100 * fl / max(h, 1):4.0f}% of the hidden, right {100 * ok / max(h, 1):4.0f}%")

    # ------------------------------------------------------------ goal windows
    print("\nthe windows before the goals (anchors: frames with the ball detected; nothing is known about the truth in the gaps, look at the clips):")
    rows = []
    for t0, t1, label in windows:
        idx, cands = window(t0, t1, label)
        times, anchors = t_all[idx], detected_all[idx]
        filled = fill_gaps(cands, anchors, times, a.vmax, a.miss, a.conf_weight)
        lin = linear_fill(anchors, times)
        n = len(idx)
        det_n = int((~np.isnan(anchors[:, 0])).sum())
        fill_n = int((~np.isnan(filled[:, 0])).sum())
        lin_n = int((~np.isnan(lin[:, 0])).sum())
        print(f"{label:14s} frames {n:3d} | detected {det_n:3d} ({100 * det_n / n:3.0f}%) | + low-conf detector {fill_n:3d} ({100 * fill_n / n:3.0f}%) "
              f"| + linear interpolation {lin_n:3d} ({100 * lin_n / n:3.0f}%)")
        for k, i in enumerate(idx):
            rows.append({"window": label, "frame": int(fr_all[i]), "time_s": t_all[i], "detected_x": anchors[k][0], "detected_y": anchors[k][1],
                         "filled_x": filled[k][0], "filled_y": filled[k][1]})
        if a.clips:
            from compare_actions import render_clip
            out_dir = Path(a.clips).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            m = ~np.isnan(filled[:, 0])
            centre = [float(np.median(filled[m, 0])), float(np.median(filled[m, 1]))] if m.any() else [L / 2, W / 2]
            balls = [tuple(p) if not np.isnan(p[0]) else None for p in filled]
            render_clip(cap, cv2, list(fr_all[idx]), balls, calibs, centre, [], f"{label} detected {det_n} filled {fill_n}/{n}",
                        out_dir / (label.replace(" ", "_").replace(":", "m") + ".mp4"), cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if rows:
        with open(folder / "redetect_probe.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {folder / 'redetect_probe.csv'}")


if __name__ == "__main__":
    main()
