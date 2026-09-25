#!/usr/bin/env python3
"""Ball possession per team.
 
For every frame with a ball position, the player closest to the ball (measured in
player heights, so it does not depend on camera zoom) is the one in control, if he is
close enough. The team in possession changes only after the other team has controlled
the ball for several frames in a row (avoids flicker in duels). While the ball travels
between players (passes, clearances), possession stays with the last team.
Frames without a ball position or with an unknown team (-1, e.g. a goalkeeper in a
different kit) do not count for either team.
 
Input:  CSV from teams.py + CSV from ball_track.py
Output: summary in the terminal, per-frame CSV, optional plot and preview video.
 
Usage:
  python possession.py ~/football/analysis/mecz1_test_teams.csv \
      --ball ~/football/analysis/mecz1_test_ball.csv \
      --out ~/football/analysis/mecz1_test_possession.csv \
      --plot ~/football/analysis/mecz1_test_possession.png \
      --video-in ~/football/videos/mecz1_test.mp4 \
      --video-out ~/football/analysis/mecz1_test_possession.mp4
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
 
import cv2
 
from ball_track import to_h264
 
TEAM_COLORS = {0: (255, 140, 0), 1: (0, 0, 255)}  # BGR, only for drawing
 
 
def load_players(path):
    players = defaultdict(list)  # frame -> [(foot_x, foot_y, height, team, box)]
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["class"] != "player" or r["team"] == "":
                continue
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            players[int(r["frame"])].append(
                ((x1 + x2) / 2, y2 - 0.1 * (y2 - y1), max(y2 - y1, 1.0), int(r["team"]), (x1, y1, x2, y2)))
    return players
 
 
def load_ball(path):
    ball = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ball[int(r["frame"])] = (float(r["x"]), float(r["y"]))
    return ball
 
 
def compute_possession(players, ball, n_frames, a):
    rows = []  # frame, controller team (raw), controller box, team in possession
    current, candidate, run, last_ball = None, None, 0, None
    for f in range(n_frames):
        if f not in ball:
            if last_ball is not None and f - last_ball > a.reset_frames:
                current, candidate, run = None, None, 0
            rows.append((f, None, None, None))
            continue
        last_ball = f
        bx, by = ball[f]
        best = min(players.get(f, []), key=lambda p: math.hypot(bx - p[0], by - p[1]) / p[2], default=None)
        controller, box = None, None
        if best is not None and math.hypot(bx - best[0], by - best[1]) / best[2] <= a.control:
            controller, box = best[3], best[4]
 
        if controller in (0, 1) and controller != current:
            run = run + 1 if controller == candidate else 1
            candidate = controller
            if current is None or run >= a.switch_frames:
                current, candidate, run = controller, None, 0
        elif controller == current:
            candidate, run = None, 0
        rows.append((f, controller, box, current))
    return rows
 
 
def summary(rows, fps):
    counts = {0: 0, 1: 0}
    for _, _, _, team in rows:
        if team in counts:
            counts[team] += 1
    total = counts[0] + counts[1]
    return counts, total
 
 
def save_plot(rows, fps, path, window_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    window = max(int(window_s * fps), 1)
    times, share = [], []
    for i in range(0, len(rows), max(window // 4, 1)):
        chunk = [r[3] for r in rows[max(0, i - window):i + 1] if r[3] in (0, 1)]
        if chunk:
            times.append(i / fps / 60)
            share.append(100 * chunk.count(0) / len(chunk))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(times, share, 50, where=[s >= 50 for s in share], color="tab:orange", alpha=0.6,
                    interpolate=True, label="team 0")
    ax.fill_between(times, share, 50, where=[s < 50 for s in share], color="tab:red", alpha=0.6,
                    interpolate=True, label="team 1")
    ax.axhline(50, color="grey", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("time [min]")
    ax.set_ylabel(f"team 0 possession [%]\n(last {window_s:.0f} s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
 
 
def draw_video(video_in, video_out, rows, ball):
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final = Path(video_out)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    counts = {0: 0, 1: 0}
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= len(rows):
            break
        _, controller, box, team = rows[f]
        if team in counts:
            counts[team] += 1
        if box is not None and controller in TEAM_COLORS:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), TEAM_COLORS[controller], 3)
        if f in ball:
            cv2.circle(img, (int(ball[f][0]), int(ball[f][1])), 12, (0, 255, 0), 2)
        total = counts[0] + counts[1]
        if total:
            p0 = 100 * counts[0] / total
            text = f"Possession  T0 {p0:.0f}%  |  T1 {100 - p0:.0f}%"
            now = f"now: T{team}" if team in (0, 1) else "now: -"
            cv2.rectangle(img, (20, 20), (560, 100), (0, 0, 0), -1)
            cv2.putText(img, text, (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, now, (35, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        TEAM_COLORS.get(team, (200, 200, 200)), 2)
        writer.write(img)
        f += 1
    cap.release()
    writer.release()
    print("Re-encoding video to H.264...")
    to_h264(tmp, final)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("teams", help="CSV from teams.py")
    p.add_argument("--ball", required=True, help="CSV from ball_track.py")
    p.add_argument("--out", required=True, help="per-frame possession CSV")
    p.add_argument("--plot", help="optional PNG with possession over time")
    p.add_argument("--video-in", help="original video (for the preview)")
    p.add_argument("--video-out", help="optional preview video")
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--control", type=float, default=1.0,
                   help="max ball distance to the player's feet, in player heights")
    p.add_argument("--switch-frames", type=int, default=5,
                   help="frames the other team must control the ball to take possession")
    p.add_argument("--reset-frames", type=int, default=60,
                   help="frames without a ball after which possession is unknown")
    p.add_argument("--plot-window", type=float, default=30, help="plot smoothing window in seconds")
    a = p.parse_args()
 
    players = load_players(Path(a.teams).expanduser())
    ball = load_ball(Path(a.ball).expanduser())
    n_frames = max(list(players) + list(ball)) + 1
    rows = compute_possession(players, ball, n_frames, a)
 
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "controller_team", "possession_team"])
        for fr, controller, _, team in rows:
            w.writerow([fr, round(fr / a.fps, 3), "" if controller is None else controller,
                        "" if team is None else team])
 
    counts, total = summary(rows, a.fps)
    print(f"Frames: {n_frames} | with possession assigned: {total} ({100 * total / n_frames:.1f}%)")
    if total:
        for t in (0, 1):
            print(f"Team {t}: {100 * counts[t] / total:.1f}%  ({counts[t] / a.fps:.0f} s)")
    print(f"Saved {out}")
 
    if a.plot:
        save_plot(rows, a.fps, Path(a.plot).expanduser(), a.plot_window)
        print(f"Saved plot {a.plot}")
    if a.video_in and a.video_out:
        draw_video(Path(a.video_in).expanduser(), Path(a.video_out).expanduser(), rows, ball)
        print(f"Saved preview {a.video_out}")
 
 
if __name__ == "__main__":
    main()
