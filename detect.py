
#!/usr/bin/env python3
"""Run the detection model on a match video, filter detections and save them to CSV.
 
Filtering:
  - ball: only the N most confident detections per frame are kept (--max-balls, default 3);
    the real ball is chosen later from these candidates using its trajectory
  - players/referees: optionally kept only if their feet are inside a pitch polygon
    (--pitch, only for a static camera; with a moving camera leave it out)
 
Usage:
  python detect.py ~/football/videos/mecz1_test.mp4 \
      --model ~/football/runs/detect/runs/v6/weights/best.pt \
      --out ~/football/analysis/mecz1_test_detections.csv \
      --video ~/football/analysis/mecz1_test_filtered.mp4
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
 
import cv2
import numpy as np
from ultralytics import YOLO
 
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
 
 
COLORS = {"player": (255, 120, 0), "referee": (0, 200, 255), "ball": (0, 0, 255)}
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="input match video")
    p.add_argument("--model", required=True)
    p.add_argument("--pitch", help="JSON with the pitch polygon (static camera only)")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--video", help="optional output video with filtered boxes")
    p.add_argument("--imgsz", type=int, default=1920)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=10, help="pixels allowed outside the pitch")
    p.add_argument("--max-balls", type=int, default=3, help="ball candidates kept per frame")
    a = p.parse_args()
 
    model = YOLO(str(Path(a.model).expanduser()))
    names = model.names
    polygon = None
    if a.pitch:
        polygon = np.array(json.loads(Path(a.pitch).read_text())["polygon"], np.int32)
 
    video = str(Path(a.source).expanduser())
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
 
    out_csv = Path(a.out).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = None
 
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "class", "conf", "x1", "y1", "x2", "y2", "foot_x", "foot_y"])
 
        results = model.predict(video, imgsz=a.imgsz, conf=a.conf, stream=True, verbose=False)
        for frame_idx, r in enumerate(results):
            kept, balls = [], []
            for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                name = names[int(cls)]
                x1, y1, x2, y2 = box
                foot = ((x1 + x2) / 2, y2)
                if name == "ball":
                    balls.append((name, conf, box, ((x1 + x2) / 2, (y1 + y2) / 2)))
                elif polygon is None or cv2.pointPolygonTest(polygon, foot, True) > -a.margin:
                    kept.append((name, conf, box, foot))
            balls.sort(key=lambda b: b[1], reverse=True)
            kept.extend(balls[:a.max_balls])
 
            for name, conf, (x1, y1, x2, y2), (fx, fy) in kept:
                w.writerow([frame_idx, round(frame_idx / fps, 3), name, round(conf, 3),
                            round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                            round(fx, 1), round(fy, 1)])
 
            if a.video:
                img = r.orig_img.copy()
                if polygon is not None:
                    cv2.polylines(img, [polygon], True, (0, 255, 255), 2)
                for name, conf, (x1, y1, x2, y2), _ in kept:
                    c = COLORS.get(name, (255, 255, 255))
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                    cv2.putText(img, f"{name} {conf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                if writer is None:
                    out_video = Path(a.video).expanduser()
                    out_video.parent.mkdir(parents=True, exist_ok=True)
                    h, wd = img.shape[:2]
                    tmp_video = out_video.with_suffix(".tmp.mp4")
                    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
                writer.write(img)
 
            if frame_idx % 250 == 0:
                print(f"frame {frame_idx}/{total}")
 
    if writer:
        writer.release()
        print("Re-encoding video to H.264...")
        to_h264(tmp_video, out_video)
        print(f"Saved video to {out_video}")
    print(f"Saved detections to {out_csv}")
 
 
if __name__ == "__main__":
    main()
