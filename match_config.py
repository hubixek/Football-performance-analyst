#!/usr/bin/env python3
"""Create a match config: asks for the start and end of each half.

Times can be entered as HH:MM:SS, MM:SS or seconds (e.g. 1:05:30, 8:20, 500).
They are checked (format, order, within the video length) and optionally a preview
frame is saved for every time, so you can verify you picked the right moment.

Usage:
  python match_config.py ~/football/videos/mecz1_dual.mp4
  python match_config.py ~/football/videos/mecz1_dual.mp4 --name mecz1 --preview

Output: matches/<name>.json in the repository, e.g.
  {
    "name": "mecz1",
    "video": "mecz1_dual.mp4",
    "halves": [{"start": "00:08:20", "end": "00:25:10", "start_s": 500.0, "end_s": 1510.0}, ...],
    "calibration": {"top": "calib/mecz1_top.json", "bottom": "calib/mecz1_bottom.json"}
  }
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def video_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def parse_time(text):
    """'1:05:30' / '8:20' / '500' -> seconds (float)."""
    parts = text.strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError
    seconds = 0.0
    for p in parts:
        if p == "":
            raise ValueError
        seconds = seconds * 60 + float(p)
    if seconds < 0:
        raise ValueError
    return seconds


def fmt(seconds):
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def ask_time(prompt, minimum, duration):
    while True:
        text = input(f"{prompt}: ").strip()
        try:
            t = parse_time(text)
        except ValueError:
            print("  Nieprawidłowy format. Wpisz np. 8:20, 00:08:20 albo 500.")
            continue
        if t <= minimum:
            print(f"  Czas musi być późniejszy niż {fmt(minimum)}.")
            continue
        if duration and t > duration:
            print(f"  Nagranie trwa tylko {fmt(duration)}.")
            continue
        return t


def save_preview(video, seconds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(seconds), "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=1024:-1", "-q:v", "3", str(path)], check=False)


def yes(prompt, default=True):
    ans = input(f"{prompt} [{'T/n' if default else 't/N'}]: ").strip().lower()
    return default if ans == "" else ans in ("t", "tak", "y", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="match video (dual-lens recording)")
    ap.add_argument("--name", help="match name (default: video name without _dual)")
    ap.add_argument("--halves", type=int, default=2, help="number of halves (default 2)")
    ap.add_argument("--preview", action="store_true", help="save a frame for every entered time")
    ap.add_argument("--out-dir", default=str(HERE / "matches"))
    a = ap.parse_args()

    video = Path(a.video).expanduser()
    if not video.exists():
        sys.exit(f"Nie ma pliku {video}")
    name = a.name or video.stem.replace("_dual", "")
    out = Path(a.out_dir) / f"{name}.json"
    if out.exists() and not yes(f"{out} już istnieje. Nadpisać?", default=False):
        sys.exit("Przerwano.")

    duration = video_duration(video)
    print(f"Mecz: {name}")
    print(f"Nagranie: {video.name}" + (f" ({fmt(duration)})" if duration else ""))
    print("Podawaj czasy z nagrania, np. 8:20 albo 00:08:20.\n")

    while True:
        halves, last = [], 0.0
        for i in range(1, a.halves + 1):
            start = ask_time(f"Połowa {i} – początek", last - 1e-9 if i == 1 else last, duration)
            end = ask_time(f"Połowa {i} – koniec", start, duration)
            halves.append({"start": fmt(start), "end": fmt(end), "start_s": start, "end_s": end})
            last = end

        print("\nPodsumowanie:")
        for i, h in enumerate(halves, 1):
            print(f"  Połowa {i}: {h['start']} – {h['end']}  ({fmt(h['end_s'] - h['start_s'])})")
        if len(halves) > 1:
            print(f"  Przerwa: {fmt(halves[1]['start_s'] - halves[0]['end_s'])}")

        if a.preview:
            prev_dir = Path(a.out_dir) / "previews"
            for i, h in enumerate(halves, 1):
                for key in ("start", "end"):
                    save_preview(video, h[f"{key}_s"], prev_dir / f"{name}_half{i}_{key}.jpg")
            print(f"\nKlatki podglądowe zapisane w {prev_dir}")

        if yes("\nZapisać?"):
            break
        print("Wpisz czasy jeszcze raz.\n")

    config = {
        "name": name,
        "video": video.name,
        "halves": halves,
        "calibration": {"top": f"calib/{name}_top.json", "bottom": f"calib/{name}_bottom.json"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()
