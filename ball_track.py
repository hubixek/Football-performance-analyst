#!/usr/bin/env python3
"""Choose the real ball from ball candidates and fill short gaps.
 
Input:  CSV from detect.py (up to N ball candidates per frame + players/referees).
Output: CSV with at most one ball position per frame:
        frame, time_s, x, y, conf, source   (source = detected | interpolated)
 
How the ball is chosen:
  1. Acquire: the most confident candidate that is close to players.
  2. Follow:  in the next frames only candidates within a distance gate from the
              last position are accepted (the gate grows with the number of missed frames).
  3. Switch:  if another candidate is clearly more likely to be the match ball
              (closer to the frame centre - the follow camera keeps the ball there -
              more confident, near players) for several frames, the tracker switches to it.
              This fixes getting stuck on a spare ball after the ball went out of play.
  4. Release: if the tracked "ball" stays far from all players for too long,
              those frames are dropped and the ball is acquired again.
  5. Fill:    short gaps between detections are filled by linear interpolation.
 
Usage:
  python ball_track.py ~/football/analysis/mecz1_test_detections.csv \
      --out ~/football/analysis/mecz1_test_ball.csv \
      --video-in ~/football/videos/mecz1_test.mp4 \
      --video-out ~/football/analysis/mecz1_test_ball.mp4
"""
import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
 
import cv2
 
 
def to_h264(tmp_path, final_path):
    """Re-encode the OpenCV (mp4v) video to H.264 - much smaller and plays everywhere."""
    tmp_path, final_path = Path(tmp_path), Path(final_path)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_path),
           "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an", str(final_path)]
    try:
        subprocess.run(cmd, check=True)
        tmp_path.unlink()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg failed - keeping the mp4v video")
        tmp_path.replace(final_path)
 
 
def load(path):
    balls, players, times = defaultdict(list), defaultdict(list), {}
    with open(path) as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            times[fr] = float(r["time_s"])
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            if r["class"] == "ball":
                balls[fr].append(((x1 + x2) / 2, (y1 + y2) / 2, float(r["conf"])))
            else:
                players[fr].append(((x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0)))
    n_frames = max(times) + 1 if times else 0
    fps = next((fr / t for fr, t in sorted(times.items(), reverse=True) if t > 0), 30.0)
    return balls, players, n_frames, fps
 
 
def distance_to_players(ball, players):
    """Distance to the nearest player, measured in player heights (0 if no players visible)."""
    if not players:
        return 0.0
    return min(math.hypot(ball[0] - px, ball[1] - py) / h for px, py, h in players)
 
 
def global_score(c, pl, a):
    """How likely a candidate is the match ball, regardless of the current track.
 
    Uses confidence, distance to the frame centre (the follow camera keeps the ball
    near the centre) and whether there are players nearby.
    """
    cx, cy = a.width / 2, a.height / 2
    centre = 1 - math.hypot(c[0] - cx, c[1] - cy) / math.hypot(cx, cy)
    near = 0.2 if distance_to_players(c, pl) <= a.near else 0.0
    return c[2] + a.centre_weight * centre + near
 
 
def track_ball(balls, players, n_frames, a):
    track, last, last_f, far_frames = {}, None, None, []
    challenger, challenger_frames = None, []
    for f in range(n_frames):
        cands, pl = balls.get(f, []), players.get(f, [])
        chosen = None
 
        if last is not None and f - last_f > a.max_gap:
            last = None  # lost for too long -> acquire again
 
        if last is not None:
            gate = a.gate_base + a.gate_speed * (f - last_f)
            options = [(c, math.hypot(c[0] - last[0], c[1] - last[1])) for c in cands]
            options = [(c, d) for c, d in options if d <= gate]
            if options:
                chosen = max(options, key=lambda cd: cd[0][2] - 0.5 * cd[1] / gate)[0]
 
            # Challenger: a clearly better candidate outside the gate for several frames
            # (e.g. the tracker stuck on a spare ball after a throw-in) -> switch to it.
            others = [c for c in cands if c is not chosen and c[2] >= a.acquire_conf
                      and math.hypot(c[0] - last[0], c[1] - last[1]) > gate]
            best = max(others, key=lambda c: global_score(c, pl, a), default=None)
            current = global_score(chosen, pl, a) if chosen else float("-inf")
            if best is not None and global_score(best, pl, a) > current + a.switch_margin:
                if challenger is None or math.hypot(best[0] - challenger[0], best[1] - challenger[1]) \
                        > a.gate_base + a.gate_speed:
                    challenger_frames = []  # a different challenger than before
                challenger = best
                challenger_frames.append(f)
                if len(challenger_frames) >= a.switch_frames:
                    for ff in challenger_frames:  # frames spent on the wrong object
                        track.pop(ff, None)
                    chosen, far_frames = best, []
                    challenger, challenger_frames = None, []
            else:
                challenger, challenger_frames = None, []
        else:
            options = [c for c in cands
                       if c[2] >= a.acquire_conf and distance_to_players(c, pl) <= a.near]
            if options:
                chosen = max(options, key=lambda c: global_score(c, pl, a))
 
        if chosen is None:
            continue
 
        if distance_to_players(chosen, pl) > a.near:
            far_frames.append(f)
        else:
            far_frames = []
 
        if len(far_frames) > a.far_frames:
            # tracked object stayed away from play for too long: probably not the match ball
            for ff in far_frames:
                track.pop(ff, None)
            last, far_frames = None, []
            continue
 
        track[f] = chosen
        last, last_f = chosen, f
    return track
 
 
def interpolate(track, max_gap, max_jump):
    filled = {f: (x, y, c, "detected") for f, (x, y, c) in track.items()}
    known = sorted(track)
    for f0, f1 in zip(known, known[1:]):
        gap = f1 - f0
        (x0, y0, _), (x1, y1, _) = track[f0], track[f1]
        # fill only short gaps, and not across a switch to a different object
        if 1 < gap <= max_gap and math.hypot(x1 - x0, y1 - y0) <= max_jump(gap):
            for f in range(f0 + 1, f1):
                t = (f - f0) / gap
                filled[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.0, "interpolated")
    return filled
 
 
def draw_video(video_in, video_out, ball):
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(video_out).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = Path(video_out).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        trail = [ball[i] for i in range(max(0, f - 15), f + 1) if i in ball]
        for x, y, _, _ in trail:
            cv2.circle(img, (int(x), int(y)), 3, (0, 200, 255), -1)
        if f in ball:
            x, y, _, src = ball[f]
            color = (0, 255, 0) if src == "detected" else (0, 255, 255)
            cv2.circle(img, (int(x), int(y)), 14, color, 3)
        writer.write(img)
        f += 1
    cap.release()
    writer.release()
    print("Re-encoding video to H.264...")
    to_h264(tmp_out, video_out)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("detections", help="CSV from detect.py")
    p.add_argument("--out", required=True, help="output CSV with the ball track")
    p.add_argument("--video-in", help="original video (for the preview video)")
    p.add_argument("--video-out", help="preview video with the tracked ball")
    p.add_argument("--acquire-conf", type=float, default=0.3, help="min confidence to start tracking")
    p.add_argument("--near", type=float, default=4.0, help="max distance to players (in player heights)")
    p.add_argument("--far-frames", type=int, default=45, help="frames far from players before release")
    p.add_argument("--gate-base", type=float, default=40.0, help="base search radius in pixels")
    p.add_argument("--gate-speed", type=float, default=45.0, help="extra radius per missed frame")
    p.add_argument("--max-gap", type=int, default=30, help="frames without ball before re-acquiring")
    p.add_argument("--interp", type=int, default=20, help="max gap (frames) filled by interpolation")
    p.add_argument("--centre-weight", type=float, default=1.0, help="weight of distance to frame centre")
    p.add_argument("--switch-margin", type=float, default=0.2, help="how much better a challenger must be")
    p.add_argument("--switch-frames", type=int, default=10, help="frames a challenger must win to switch")
    p.add_argument("--width", type=int, default=1920, help="video width in pixels")
    p.add_argument("--height", type=int, default=1080, help="video height in pixels")
    a = p.parse_args()
 
    balls, players, n_frames, fps = load(Path(a.detections).expanduser())
    track = track_ball(balls, players, n_frames, a)
    ball = interpolate(track, a.interp, lambda gap: a.gate_base + a.gate_speed * gap)
 
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x", "y", "conf", "source"])
        for fr in sorted(ball):
            x, y, c, src = ball[fr]
            w.writerow([fr, round(fr / fps, 3), round(x, 1), round(y, 1), round(c, 3), src])
 
    detected = sum(1 for v in ball.values() if v[3] == "detected")
    print(f"Frames: {n_frames} | ball detected: {detected} ({100 * detected / max(n_frames, 1):.1f}%)"
          f" | with interpolation: {len(ball)} ({100 * len(ball) / max(n_frames, 1):.1f}%)")
    print(f"Saved {out}")
 
    if a.video_in and a.video_out:
        draw_video(Path(a.video_in).expanduser(), Path(a.video_out).expanduser(), ball)
        print(f"Saved preview {a.video_out}")
 
 
if __name__ == "__main__":
    main()
