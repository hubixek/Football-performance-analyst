#!/usr/bin/env python3
"""Re-calibrate a lens automatically, starting from a calibration of another match.

Works when the camera was set up in a similar (not identical) position:
  1. median of many frames -> background without players
  2. white-line mask + distance map (distance of every pixel to the nearest line)
  3. the reference calibration is corrected so that the pitch lines drawn from the
     pitch model lie on the white lines (least squares on the distance map)
  4. quality check: share of drawn line points that lie on a white line

White objects that are not pitch lines (posts, goals, cars, buildings) are ignored:
white pixels must have grass nearby and lie inside the pitch area (+ margin) given by
the reference calibration; a robust (Huber) loss limits the effect of what remains.

The lens distortion is kept from the reference; only the camera position changes.
For a big change of the camera position, calibrate manually (calibrate.py).

Usage:
  python auto_calibrate.py ~/football/videos/mecz2_dual.mp4 --half top \
      --ref calib/mecz1_top.json --pitch pitch/pitch_6v6.json --out calib/mecz2_top.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from calibrate import densify, pitch_lines, pitch_to_pixels


def median_background(video, half, n_frames, start_s, end_s):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_s * fps)
    last = min(total - 1, int(end_s * fps)) if end_s else total - 1
    frames = []
    for idx in np.linspace(first, last, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        if not ok:
            continue
        h = img.shape[0] // 2
        frames.append(img[:h] if half == "top" else img[h:2 * h])
    cap.release()
    if len(frames) < 5:
        raise SystemExit("Could not read enough frames")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def line_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 170) & (hsv[..., 1] < 60)
    # keep only white pixels that have grass nearby (removes sky, buildings, cars)
    grass = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50)).astype(np.uint8)
    grass_near = cv2.dilate(grass, np.ones((25, 25), np.uint8)) > 0
    mask = (white & grass_near).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def pitch_area(calib, pitch, size, margin, step=8):
    """Mask of image pixels that lie on the pitch (+ margin) according to the calibration."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(0, w, step) + step / 2, np.arange(0, h, step) + step / 2)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    if calib["model"] == "distorted":
        K, d = np.array(calib["K"]), np.array(calib["dist"])
        pts = cv2.undistortPoints(pts, K, d, P=K)
    pts = pts.reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ np.array(calib["H"]).T
    # pixels above the horizon map "behind" the camera: the homogeneous w changes sign
    centre = np.array([pitch["length_m"] / 2, pitch["width_m"] / 2])
    ref_px = cv2.perspectiveTransform(centre.reshape(1, 1, 2).astype(np.float64),
                                      np.linalg.inv(np.array(calib["H"]))).reshape(2)
    ref_w = (np.array([*ref_px, 1.0]) @ np.array(calib["H"]).T)[2]
    ok = np.sign(hom[:, 2]) == np.sign(ref_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = hom[:, :2] / hom[:, 2:3]
    L, W = pitch["length_m"], pitch["width_m"]
    ok &= (xy[:, 0] > -margin) & (xy[:, 0] < L + margin) & (xy[:, 1] > -margin) & (xy[:, 1] < W + margin)
    small = ok.reshape(xs.shape).astype(np.uint8) * 255
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def with_correction(ref, params, scale):
    """Reference calibration with a small homography correction C (in undistorted pixels)."""
    C = np.eye(3) + np.append(params, 0).reshape(3, 3)
    T = np.diag([scale, scale, 1.0])
    Cpx = np.linalg.inv(T) @ C @ T   # correction defined in normalised coordinates
    out = dict(ref)
    out["H"] = (np.array(ref["H"]) @ Cpx).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--half", choices=["top", "bottom"], required=True)
    ap.add_argument("--ref", required=True, help="calibration of another match (same lens)")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=60, help="frames for the median background")
    ap.add_argument("--start", type=float, default=0, help="use frames from this second")
    ap.add_argument("--end", type=float, default=0, help="... until this second (0 = end)")
    ap.add_argument("--margin-m", type=float, default=3.0,
                    help="only white pixels inside the pitch + this margin (metres) are used")
    ap.add_argument("--min-inliers", type=float, default=0.6, help="required share of lines on white")
    a = ap.parse_args()

    ref = json.loads(Path(a.ref).read_text())
    pitch = json.loads(Path(a.pitch).read_text())

    bg = median_background(Path(a.video).expanduser(), a.half, a.frames, a.start, a.end)
    h, w = bg.shape[:2]
    mask = line_mask(bg)
    # keep only white pixels inside the pitch (+ margin) as seen with the reference calibration:
    # removes goals, posts, fences, cars and buildings behind the pitch
    mask = cv2.bitwise_and(mask, pitch_area(ref, pitch, (w, h), a.margin_m))
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

    # pitch-line points that are visible in the reference view
    world = np.vstack([densify(l) if len(l) < 10 else l for l in pitch_lines(pitch)])
    px = pitch_to_pixels(ref, world)
    inside = (px[:, 0] > 5) & (px[:, 0] < w - 5) & (px[:, 1] > 5) & (px[:, 1] < h - 5)
    world = world[inside]
    if len(world) < 50:
        raise SystemExit("Too few pitch lines visible with the reference calibration")

    def sample(calib, cap):
        p = pitch_to_pixels(calib, world)
        # bilinear sampling keeps the cost smooth, so the optimiser gets useful gradients
        d = map_coordinates(dist, [np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)],
                            order=1, mode="nearest")
        outside = (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
        d[outside] = cap
        return np.minimum(d, cap)

    scale = 1.0 / w
    params = np.zeros(8)
    before = sample(ref, 1e9)
    # coarse-to-fine: wide basin first, then precise
    for cap in (80, 40, 15, 6):
        blurred = cv2.GaussianBlur(dist, (0, 0), cap / 6)
        dist_backup, dist[:] = dist.copy(), blurred
        res = least_squares(lambda q: sample(with_correction(ref, q, scale), cap), params,
                            loss="huber", f_scale=cap / 4, diff_step=1e-4, max_nfev=400)
        dist[:] = dist_backup
        params = res.x

    calib = with_correction(ref, params, scale)
    after = sample(calib, 1e9)
    inliers_before = float((before < 4).mean())
    inliers_after = float((after < 4).mean())
    print(f"Line points on white lines: before {inliers_before:.0%}, after {inliers_after:.0%}")
    print(f"Median distance to lines: before {np.median(before):.1f} px, after {np.median(after):.1f} px")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    calib["auto_from"] = str(a.ref)
    calib["inliers"] = inliers_after
    check = bg.copy()
    check[mask > 0] = (0.5 * check[mask > 0] + [0, 0, 127]).astype(np.uint8)
    for line in pitch_lines(pitch):
        p = pitch_to_pixels(calib, densify(line) if len(line) < 10 else line)
        cv2.polylines(check, [p.round().astype(np.int32)], False, (0, 255, 255), 2)
    cv2.imwrite(str(out.with_suffix(".check.jpg")), check)
    print(f"Check image: {out.with_suffix('.check.jpg')}  (yellow = calibration, red = detected lines)")

    if inliers_after < a.min_inliers:
        print(f"Only {inliers_after:.0%} of the lines match - calibrate this match manually (calibrate.py)")
        raise SystemExit(1)
    out.write_text(json.dumps(calib, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
