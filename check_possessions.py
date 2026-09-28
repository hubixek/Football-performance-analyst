#!/usr/bin/env python3
"""List the longest possessions of a match and cut a short clip around each of them.

Use it to check by eye whether long possessions are real play or stoppages (ball out of play,
free kick set-up, spare ball). Times are seconds from the start of the recording.

The clips show the whole dual-lens frame scaled down: the top half is the left half of the
pitch, the bottom half the right half. Each clip starts --before seconds before the possession
and ends --after seconds after it (at most --max-clip seconds long).

Usage:
  python check_possessions.py ~/football/analysis/mecz1_dual --video ~/football/videos/mecz1_dual.mp4 --top 5
"""
import argparse
import subprocess
from pathlib import Path

from match_stats import load, sequences


def mmss(t):
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder of a match")
    ap.add_argument("--video", required=True)
    ap.add_argument("--top", type=int, default=5, help="how many longest possessions")
    ap.add_argument("--before", type=float, default=5.0)
    ap.add_argument("--after", type=float, default=5.0)
    ap.add_argument("--max-clip", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1024, help="width of the clips in pixels")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    _, _, poss = load(folder)
    longest = sorted(sequences(poss, 1.0), key=lambda q: q["t0"] - q["t1"])[:a.top]
    out = folder / "clips"
    out.mkdir(exist_ok=True)

    print(f"{'#':>2} {'start in recording':>19} {'end':>7} {'length':>8}  team  half  clip")
    for n, q in enumerate(longest, 1):
        start = max(q["t0"] - a.before, 0.0)
        length = min(q["t1"] + a.after - start, a.max_clip)
        clip = out / f"longest_{n}_{mmss(q['t0']).replace(':', 'm')}s_team{q['team']}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(Path(a.video).expanduser()),
                        "-t", f"{length:.2f}", "-vf", f"scale={a.width}:-2", "-c:v", "libx264", "-crf", "24",
                        "-preset", "veryfast", "-an", str(clip)], check=True)
        print(f"{n:>2} {mmss(q['t0']):>19} {mmss(q['t1']):>7} {q['t1'] - q['t0']:>7.1f}s  {q['team']:>4}  {q['half']:>4}  {clip.name}")
    print(f"\nClips in {out}")
    print(f"In each clip the possession starts {a.before:.0f} s after the beginning.")


if __name__ == "__main__":
    main()
