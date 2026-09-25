#!/usr/bin/env python3
"""Click the pitch boundary on a frame and save it as a polygon (JSON).
 
Usage:
  python pick_points.py pitch_mecz1.jpg pitches/mecz1_pp.json
 
Controls:
  left click  - add a point (go around the pitch in order)
  u           - undo last point
  s           - save and quit
  q / Esc     - quit without saving
"""
import json
import sys
from pathlib import Path
 
import cv2
 
img_path, out_path = sys.argv[1], Path(sys.argv[2])
img = cv2.imread(img_path)
if img is None:
    sys.exit(f"Cannot read image {img_path}")
points = []
 
 
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"point {len(points)}: ({x}, {y})")
 
 
def draw():
    view = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if len(points) > 1:
        for a, b in zip(points, points[1:] + points[:1]):
            cv2.line(view, tuple(a), tuple(b), (0, 255, 255), 2)
    return view
 
 
cv2.namedWindow("pitch", cv2.WINDOW_NORMAL)
cv2.resizeWindow("pitch", 1600, 900)
cv2.setMouseCallback("pitch", on_click)
 
while True:
    cv2.imshow("pitch", draw())
    key = cv2.waitKey(30) & 0xFF
    if key == ord("u") and points:
        points.pop()
    elif key == ord("s"):
        if len(points) < 3:
            print("Need at least 3 points")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"polygon": points}, indent=2))
        print(f"Saved {len(points)} points to {out_path}")
        break
    elif key in (ord("q"), 27):
        print("Quit without saving")
        break
 
cv2.destroyAllWindows()
