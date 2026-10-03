#!/usr/bin/env python3
"""Measure how well match_actions.py finds passes and shots: precision from clips of the detections, recall from clips of random windows.

Two kinds of clips are cut from the match video (short, the lens where the ball is, cropped around the action):
  - detections: --n random passes and --n random shots of actions.csv (the start and the end of the action are circled). For every one
    you write in detections.csv: real = y/n (was it a pass / a shot at all), completed = y/n for a pass (the ball reached a team mate),
    on_target = y/n for a shot (blank = cannot tell). The answers of the program are in a hidden file, so that they do not steer you.
  - windows: --n random windows of --win seconds with the ball in play. For every one you write in windows.csv how many passes and how many
    shots (every attempt, also a lost pass and a blocked shot) you see. The counts of the program are hidden.
Run again with --score: precision of passes and shots (with a 95% interval), how often completed / on target agree with you, and the recall
estimated from the windows (program count x precision / your count).

After a change of match_actions.py: --windows-from DIR keeps the windows and your counts of an earlier run and only refreshes the counts of the
program from the current actions.csv (the clips are cut for the new detections only), so the recall of the new version needs no recounting.

Usage:
  python compare_actions.py ~/football/analysis/mecz1_dual --video ~/football/videos/mecz1_dual.mp4 --name mecz1 --n 30
  python compare_actions.py ~/football/analysis/mecz1_dual --score
  python compare_actions.py ~/football/analysis/mecz1_dual --video ~/football/videos/mecz1_dual.mp4 --name mecz1 \\
      --out ~/football/analysis/mecz1_dual/action_check_players --windows-from ~/football/analysis/mecz1_dual/action_check
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

KEY_DET, KEY_WIN = ".answers_detections.csv", ".answers_windows.csv"
ON_TARGET = ("goal", "saved", "on_target")


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def load_actions(path):
    """Passes and shots of actions.csv as dicts (outcome without the note '(inferred from the goal)')."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["type"] not in ("pass", "shot"):
                continue
            num = lambda k: float(r[k]) if r[k] != "" else np.nan
            out.append({"type": r["type"], "t": float(r["time_s"]), "half": int(r["half"]), "team": int(r["team"]),
                        "x": num("x_from"), "y": num("y_from"), "x2": num("x_to"), "y2": num("y_to"),
                        "outcome": r["outcome"].replace(" (inferred from the goal)", ""), "inferred": "inferred" in r["outcome"]})
    return out


def load_possession(path):
    """time_s, half, ball_x, ball_y, ball_in_play of possession.csv as arrays (frame numbers are the row labels)."""
    fr, t, h, bx, by, ip = [], [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            fr.append(int(r["frame"])); t.append(float(r["time_s"])); h.append(int(r["half"]))
            bx.append(float(r["ball_x"]) if r["ball_x"] != "" else np.nan)
            by.append(float(r["ball_y"]) if r["ball_y"] != "" else np.nan)
            ip.append(int(r["ball_in_play"]) if r["ball_in_play"] != "" else -1)
    return np.array(fr), np.array(t), np.array(h), np.array(bx), np.array(by), np.array(ip)


def pick_detections(actions, kind, n, gap, rng):
    """n random actions of one kind at least gap seconds apart, in time order."""
    cand = [a for a in actions if a["type"] == kind]
    order = rng.permutation(len(cand))
    chosen = []
    for i in order:
        a = cand[i]
        if all(abs(a["t"] - b["t"]) >= gap for b in chosen):
            chosen.append(a)
        if len(chosen) >= n:
            break
    return sorted(chosen, key=lambda a: a["t"])


def pick_windows(t, half, known, in_play, n, win, rng, min_known=0.5, min_play=0.7):
    """n non-overlapping windows [t0, t0 + win) inside one half, with the ball in play (min_play of the frames) and known (min_known)."""
    if len(t) < 2:
        return []
    cand = []
    for i in rng.permutation(len(t)):
        t0 = t[i]
        j = int(np.searchsorted(t, t0 + win))
        if j >= len(t) or half[j] != half[i] or half[j - 1] != half[i]:
            continue
        sl = slice(i, j)
        if np.mean(in_play[sl] == 1) >= min_play and np.mean(known[sl]) >= min_known:
            cand.append(t0)
    chosen = []
    for t0 in cand:
        if all(abs(t0 - c) >= win for c in chosen):
            chosen.append(t0)
        if len(chosen) >= n:
            break
    return sorted(chosen)


def refresh_window_key(win_key, actions):
    """The windows of an earlier run with the counts of the program recomputed from the current actions."""
    out = []
    for k in win_key:
        t0, t1 = float(k["t0"]), float(k["t1"])
        out.append(dict(k, program_passes=sum(1 for x in actions if x["type"] == "pass" and t0 <= x["t"] < t1),
                        program_shots=sum(1 for x in actions if x["type"] == "shot" and t0 <= x["t"] < t1)))
    return out


def wilson(k, n, z=1.96):
    """95% interval of a share k of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def yn(v):
    v = (v or "").strip().lower()
    return True if v == "y" else False if v == "n" else None


def score(det_rows, det_key, win_rows, win_key):
    """Returns a dict with precision, agreement and recall estimates from the filled-in sheets and the hidden answers of the program."""
    truth = {r["id"]: r for r in det_rows}
    res = {}
    for kind in ("pass", "shot"):
        judged = [(k, truth[k["id"]]) for k in det_key if k["type"] == kind and yn(truth.get(k["id"], {}).get("real")) is not None]
        real = [(k, r) for k, r in judged if yn(r["real"])]
        s = {"judged": len(judged), "real": len(real)}
        if judged:
            s["precision"] = len(real) / len(judged)
            s["precision_ci"] = wilson(len(real), len(judged))
        if kind == "pass":
            rated = [(k, yn(r.get("completed"))) for k, r in real if yn(r.get("completed")) is not None]
            s["completed_agree"] = (sum((k["program_outcome"] == "completed") == v for k, v in rated), len(rated))
        else:
            rated = [(k, yn(r.get("on_target"))) for k, r in real if yn(r.get("on_target")) is not None]
            s["on_target_agree"] = (sum((k["program_outcome"] in ON_TARGET) == v for k, v in rated), len(rated))
        res[kind] = s
    wtruth = {r["id"]: r for r in win_rows}
    for kind, col in (("pass", "passes"), ("shot", "shots")):
        pairs = []
        for k in win_key:
            u = wtruth.get(k["id"], {}).get(col, "").strip()
            if u.isdigit():
                pairs.append((int(u), int(k[f"program_{col}"])))
        s = res[kind]
        s["windows"] = len(pairs)
        s["user_count"] = sum(u for u, _ in pairs)
        s["program_count"] = sum(p for _, p in pairs)
        if s["user_count"] and "precision" in s:
            s["recall_estimate"] = min(1.0, s["program_count"] * s["precision"] / s["user_count"])
    return res


def render_clip(cap, cv2, frames, balls, calibs, centre_xy, marks, label, out_path, fps, size=(1280, 720)):
    """Writes a clip of the frames: the lens that holds centre_xy, cropped around it, ball and marks circled. Returns False if the point is on no lens."""
    from calibrate import pitch_to_pixels
    first = frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ok, img = cap.read()
    if not ok:
        return False
    h = img.shape[0] // 2
    best = None
    for part, off in (("top", 0), ("bottom", h)):
        uv = pitch_to_pixels(calibs[part], np.array([centre_xy]))[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < h:
            d = abs(uv[0] - img.shape[1] / 2)
            if best is None or d < best[0]:
                best = (d, part, off, uv)
    if best is None:
        return False
    _, part, off, uv = best
    cw, ch = min(size[0], img.shape[1]), min(size[1], h)
    x0 = int(np.clip(uv[0] - cw / 2, 0, img.shape[1] - cw))
    y0 = int(np.clip(uv[1] - ch / 2, 0, h - ch))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    if not writer.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames_arr = np.asarray(frames)
    for fr in range(first, int(frames[-1]) + 1):                       # every frame of the video; the analysis may use only every n-th
        ok, img = cap.read()
        if not ok:
            break
        ball = balls[max(int(np.searchsorted(frames_arr, fr, side="right")) - 1, 0)]   # the ball of the last analysed frame up to this one
        crop = img[off + y0:off + y0 + ch, x0:x0 + cw].copy()
        for (mx, my), colour in marks:
            p = pitch_to_pixels(calibs[part], np.array([[mx, my]]))[0]
            if np.isfinite(p).all():                                   # a point the lens model cannot project is not drawn
                cv2.circle(crop, (int(p[0]) - x0, int(p[1]) - y0), 30, colour, 2, cv2.LINE_AA)
        if ball is not None and np.isfinite(ball).all():
            p = pitch_to_pixels(calibs[part], np.array([ball]))[0]
            if np.isfinite(p).all():
                cv2.circle(crop, (int(p[0]) - x0, int(p[1]) - y0), 14, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(crop, (0, 0), (300, 44), (0, 0, 0), -1)
        cv2.putText(crop, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(crop)
    writer.release()
    return True


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_score(res):
    for kind, label in (("pass", "passes"), ("shot", "shots")):
        s = res[kind]
        line = f"{label}: {s['judged']} judged"
        if "precision" in s:
            lo, hi = s["precision_ci"]
            line += f", real {s['real']} -> precision {100 * s['precision']:.0f}% (95%: {100 * lo:.0f}-{100 * hi:.0f}%)"
        print(line)
        key, what = ("completed_agree", "completed") if kind == "pass" else ("on_target_agree", "on target")
        ok, n = s[key]
        if n:
            print(f"  {what} agrees with you in {ok} of {n}")
        if s["windows"]:
            line = f"  windows: you counted {s['user_count']}, the program {s['program_count']} in {s['windows']} windows"
            if "recall_estimate" in s:
                line += f" -> estimated recall {100 * s['recall_estimate']:.0f}% (program count x precision / your count)"
            print(line)
    print("small samples: read the intervals, not only the percentages")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="analysis folder with actions.csv and possession.csv (match_actions.py output)")
    ap.add_argument("--video")
    ap.add_argument("--name", help="name of the match (calibration files)")
    ap.add_argument("--calib-dir", default=str(Path.home() / "football" / "calib"))
    ap.add_argument("--n", type=int, default=30, help="passes, shots and windows to judge")
    ap.add_argument("--win", type=float, default=10.0, help="length of a window, s")
    ap.add_argument("--gap", type=float, default=15.0, help="detections at least this far apart, s")
    ap.add_argument("--pre", type=float, default=2.5, help="seconds of a clip before the action")
    ap.add_argument("--post", type=float, default=3.5, help="seconds of a clip after the action")
    ap.add_argument("--out", help="output folder (default: <folder>/action_check)")
    ap.add_argument("--windows-from", help="folder of an earlier run: reuse its windows and your counts, refresh the counts of the program")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    out = Path(a.out).expanduser() if a.out else folder / "action_check"

    if a.score:
        res = score(read_csv(out / "detections.csv"), read_csv(out / KEY_DET), read_csv(out / "windows.csv"), read_csv(out / KEY_WIN))
        print_score(res)
        return
    if not a.video or not a.name:
        raise SystemExit("--video and --name are needed to cut the clips")

    import cv2
    actions = load_actions(folder / "actions.csv")
    fr, t, half, bx, by, ip = load_possession(folder / "possession.csv")
    rng = np.random.default_rng(a.seed)
    dets = pick_detections(actions, "pass", a.n, a.gap, rng) + pick_detections(actions, "shot", a.n, a.gap, rng)
    wins = [] if a.windows_from else pick_windows(t, half, ~np.isnan(bx), ip, a.n, a.win, rng)
    cdir = Path(a.calib_dir).expanduser()
    calibs = {p: json.loads((cdir / f"{a.name}_{p}.json").read_text()) for p in ("top", "bottom")}
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out.mkdir(parents=True, exist_ok=True)

    def clip(t0, t1, centre, marks, label, name):
        i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
        sl = slice(i0, max(i1, i0 + 1))
        balls = [(bx[i], by[i]) if not np.isnan(bx[i]) else None for i in range(sl.start, sl.stop)]
        return render_clip(cap, cv2, list(fr[sl]), balls, calibs, centre, marks, label, out / name, fps)

    det_rows, det_key = [], []
    for k, d in enumerate(sorted(dets, key=lambda d: (d["type"], d["t"])), 1):
        end = (d["x2"], d["y2"]) if not np.isnan(d["x2"]) else None
        marks = [((d["x"], d["y"]), (0, 255, 255))] + ([(end, (255, 200, 0))] if end else [])
        centre = [(d["x"] + end[0]) / 2, (d["y"] + end[1]) / 2] if end else [d["x"], d["y"]]
        name = f"{k:02d}_{d['type']}_{mmss(d['t']).replace(':', 'm')}.mp4"
        if clip(d["t"] - a.pre, d["t"] + a.post, centre, marks, f"#{k} {d['type']} {mmss(d['t'])}", name):
            det_rows.append({"id": k, "type": d["type"], "time": mmss(d["t"]), "clip": name, "real": "", "completed": "", "on_target": ""})
            det_key.append({"id": k, "type": d["type"], "time_s": round(d["t"], 2), "program_outcome": d["outcome"], "inferred": int(d["inferred"])})
    win_rows, win_key = [], []
    for k, t0 in enumerate(wins, 1):
        m = (t >= t0) & (t < t0 + a.win) & ~np.isnan(bx)
        if not m.any():
            continue
        centre = [float(np.median(bx[m])), float(np.median(by[m]))]
        name = f"w{k:02d}_{mmss(t0).replace(':', 'm')}.mp4"
        if clip(t0, t0 + a.win, centre, [], f"window {k} {mmss(t0)}", name):
            win_rows.append({"id": k, "time": mmss(t0), "clip": name, "passes": "", "shots": ""})
            win_key.append({"id": k, "t0": round(t0, 2), "t1": round(t0 + a.win, 2),
                            "program_passes": sum(1 for x in actions if x["type"] == "pass" and t0 <= x["t"] < t0 + a.win),
                            "program_shots": sum(1 for x in actions if x["type"] == "shot" and t0 <= x["t"] < t0 + a.win)})
    if a.windows_from:
        old = Path(a.windows_from).expanduser()
        win_rows = read_csv(old / "windows.csv")
        win_key = refresh_window_key(read_csv(old / KEY_WIN), actions)
    cap.release()
    write_csv(out / "detections.csv", ["id", "type", "time", "clip", "real", "completed", "on_target"], det_rows)
    write_csv(out / KEY_DET, ["id", "type", "time_s", "program_outcome", "inferred"], det_key)
    write_csv(out / "windows.csv", ["id", "time", "clip", "passes", "shots"], win_rows)
    write_csv(out / KEY_WIN, ["id", "t0", "t1", "program_passes", "program_shots"], win_key)
    if a.windows_from:
        print(f"Saved {len(det_rows)} detection clips to {out}; {len(win_rows)} windows and your counts reused from {a.windows_from}")
        print("Fill in detections.csv (real, completed, on_target: y/n), then run with --score.")
    else:
        print(f"Saved {len(det_rows)} detection clips and {len(win_rows)} window clips to {out}")
        print("Fill in detections.csv (real, completed, on_target: y/n) and windows.csv (passes, shots: counts), then run with --score.")


if __name__ == "__main__":
    main()
