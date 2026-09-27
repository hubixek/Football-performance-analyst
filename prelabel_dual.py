#!/usr/bin/env python3
"""Pre-annotate dual-lens frames for CVAT, keeping only what is on the pitch.

For every frame the half (top/bottom) is read from the file name, the detections of the
model are mapped to pitch coordinates with the calibration of that half, and:
  - players/referees whose feet are more than --margin m outside the pitch are dropped
    (substitutes, coaches, spectators),
  - balls more than --ball-margin m outside the pitch are dropped (spare balls).
Output: ZIP in YOLO 1.1 format for "Upload annotations" in CVAT.

Usage:
  python prelabel_dual.py ~/football/frames_dual/mecz2 --name mecz2 \
      --model ~/football/runs/detect/runs/v6/weights/best.pt --calib-dir ~/football/calib \
      --pitch pitch/pitch_6v6.json
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from calibrate import pixels_to_pitch

NAMES = ["player", "referee", "ball"]


def inside(xy, pitch, margin):
    L, W = pitch["length_m"], pitch["width_m"]
    return np.isfinite(xy).all(axis=1) & (xy[:, 0] > -margin) & (xy[:, 0] < L + margin) \
        & (xy[:, 1] > -margin) & (xy[:, 1] < W + margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="folder from extract_dual_frames.py")
    ap.add_argument("--name", required=True, help="match name (for the calibration files)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", default=str(Path.home() / "football" / "calib"))
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=1.0, help="metres outside the pitch still kept (people)")
    ap.add_argument("--ball-margin", type=float, default=2.0, help="metres outside the pitch still kept (ball)")
    a = ap.parse_args()

    frames = Path(a.frames).expanduser()
    imgs = sorted(frames.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"No frames in {frames}")
    cdir = Path(a.calib_dir).expanduser()
    calibs = {h: json.loads((cdir / f"{a.name}_{h}.json").read_text()) for h in ("top", "bottom")}
    pitch = json.loads(Path(a.pitch).read_text())
    model = YOLO(str(Path(a.model).expanduser()))
    names = model.names

    out = frames.parent / f"{frames.name}_prelabel.zip"
    kept = dropped = 0
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("obj.names", "\n".join(NAMES))
        z.writestr("obj.data", f"classes = {len(NAMES)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{p.name}" for p in imgs))
        for path in imgs:
            # one image at a time: a list passed at once becomes one huge batch (out of GPU memory)
            r = model.predict(str(path), imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            half = "top" if "_top_" in Path(r.path).name else "bottom"
            lines = []
            if len(r.boxes):
                cls = [names[int(c)] for c in r.boxes.cls.tolist()]
                xyxy = r.boxes.xyxy.cpu().numpy()
                xywhn = r.boxes.xywhn.cpu().numpy()
                # feet for people, centre for the ball
                pts = np.where(np.array(cls)[:, None] == "ball",
                               np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], 1),
                               np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1))
                xy = pixels_to_pitch(calibs[half], pts)
                ok_people = inside(xy, pitch, a.margin)
                ok_ball = inside(xy, pitch, a.ball_margin)
                for c, (x, y, w, h), okp, okb in zip(cls, xywhn, ok_people, ok_ball):
                    if c not in NAMES or not (okb if c == "ball" else okp):
                        dropped += 1
                        continue
                    kept += 1
                    lines.append(f"{NAMES.index(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            z.writestr(f"obj_train_data/{Path(r.path).stem}.txt", "\n".join(lines))
    print(f"{len(imgs)} frames: kept {kept} detections, dropped {dropped} outside the pitch")
    print(f"Saved {out}  -> CVAT: Upload annotations, format YOLO 1.1")


if __name__ == "__main__":
    main()
