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

from scipy.spatial import cKDTree

from calibrate import _look_at, densify, fisheye_project, pitch_lines, pitch_to_pixels, pixels_to_pitch


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


def turf_region(img):
    """Mask of the pitch surface.

    Colour of the turf is taken from the lower part of the image (mostly pitch); similar
    pixels are selected by hue/chroma (ignoring brightness, so light and dark mowing stripes
    both count), thin connections to hedges and trees are cut, lines are closed over and only
    the largest connected area is kept.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab[int(h * 0.65):].reshape(-1, 3), axis=0)
    chroma = np.linalg.norm(lab[..., 1:] - ref[1:], axis=2)
    light = lab[..., 0]
    turf = ((chroma < 14) & (light > ref[0] - 50) & (light < ref[0] + 45)).astype(np.uint8)
    turf = cv2.morphologyEx(turf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(turf, 8)
    if n <= 1:
        return turf * 255
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == biggest).astype(np.uint8)
    # convex hull: fills holes (players, shadows, lines) even when they touch the border
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.fillPoly(filled, [cv2.convexHull(np.vstack(contours))], 1)
    return filled * 255


def line_mask(img):
    """White pitch lines: thin bright, low-saturation structures inside the turf region."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    white = (tophat > 18) & (hsv[..., 1] < 80) & (v > 110)
    # inside the pitch (convex hull) AND next to turf-coloured pixels: cuts the connection of
    # far lines to walls, cars and fences behind them
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h = img.shape[0]
    ref = np.median(lab[int(h * 0.65):].reshape(-1, 3), axis=0)
    near_turf = (np.linalg.norm(lab[..., 1:] - ref[1:], axis=2) < 18).astype(np.uint8)
    near_turf = cv2.dilate(near_turf, np.ones((11, 11), np.uint8)) > 0
    region = cv2.dilate(turf_region(img), np.ones((9, 9), np.uint8)) > 0
    mask = (white & region & near_turf).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # keep only line-like pieces: long, and nowhere thicker than a line close to the camera
    # (drops balls, sky gaps in foliage and other blobs)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    half_width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    max_hw = np.zeros(n)
    np.maximum.at(max_hw, labels.ravel(), half_width.ravel())
    length = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (length >= 40) & (max_hw <= 14)
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def pitch_area(calib, pitch, size, margin, step=8):
    """Mask of image pixels that lie on the pitch (+ margin) according to the calibration."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(0, w, step) + step / 2, np.arange(0, h, step) + step / 2)
    with np.errstate(all="ignore"):
        xy = pixels_to_pitch(calib, np.stack([xs.ravel(), ys.ravel()], axis=1))
    L, W = pitch["length_m"], pitch["width_m"]
    if calib["model"] != "fisheye":
        # homography models: pixels above the horizon map "behind" the camera -> keep only
        # pixels whose mapped point projects back onto themselves
        back = pitch_to_pixels(calib, np.nan_to_num(xy, nan=1e6))
        ok = np.linalg.norm(back - np.stack([xs.ravel(), ys.ravel()], axis=1), axis=1) < 2 * step
    else:
        ok = np.isfinite(xy).all(axis=1)
    ok &= (xy[:, 0] > -margin) & (xy[:, 0] < L + margin) & (xy[:, 1] > -margin) & (xy[:, 1] < W + margin)
    small = ok.reshape(xs.shape).astype(np.uint8) * 255
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def n_params(ref):
    return 6 if ref["model"] == "fisheye" else 8


def with_correction(ref, params, scale):
    """Reference calibration with a small correction of the camera position.

    fisheye: correction of the pose (rotation, translation); homography models: a small
    homography correction C in undistorted pixels.
    """
    out = dict(ref)
    if ref["model"] == "fisheye":
        out["rvec"] = (np.array(ref["rvec"]) + params[:3] * 0.1).tolist()
        out["tvec"] = (np.array(ref["tvec"]) + params[3:6]).tolist()
        return out
    C = np.eye(3) + np.append(params, 0).reshape(3, 3)
    T = np.diag([scale, scale, 1.0])
    Cpx = np.linalg.inv(T) @ C @ T   # correction defined in normalised coordinates
    out["H"] = (np.array(ref["H"]) @ Cpx).tolist()
    return out


def global_search(mask, pitch, ref, verbose=True, anchor=None):
    """Find a fisheye calibration from scratch: search camera positions and orientations so
    that the pitch lines of the model cover the white lines detected in the image.

    Needed when the camera stood somewhere else or was turned differently than in the
    reference match. The lens parameters of the reference are only a starting point.

    anchor: calibration of the other lens of the same camera. Both lenses sit in one device,
    so the camera position is taken from it and only the viewing direction is searched.
    """
    h, w = mask.shape
    sc = 0.25                                            # search on a 4x smaller image
    small = cv2.resize(mask, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) > 0
    hs, ws = small.shape
    dist = cv2.distanceTransform((~small).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(small)
    if len(xs) < 50:
        raise SystemExit("Too few white line pixels found - is the pitch visible in this half?")
    pick = np.random.default_rng(0).choice(len(xs), size=min(1500, len(xs)), replace=False)
    detected = np.stack([xs[pick], ys[pick]], axis=1).astype(float)
    world = np.vstack([densify(l, 25) if len(l) < 10 else l for l in pitch_lines(pitch)])
    L, W = pitch["length_m"], pitch["width_m"]
    cap = 25.0

    def calib_of(x):
        return {"model": "fisheye", "f": x[0], "cx": x[1], "cy": x[2], "k": [x[3], x[4], x[5]],
                "rvec": x[6:9].tolist(), "tvec": x[9:12].tolist()}

    def residuals(x):
        px = fisheye_project(calib_of(x), world) * sc
        ok = np.isfinite(px).all(axis=1) & (px[:, 0] >= 0) & (px[:, 0] < ws - 1) & (px[:, 1] >= 0) \
            & (px[:, 1] < hs - 1)
        fwd = np.full(len(world), cap * 0.5)             # model line outside the image: small cost
        if ok.any():
            fwd[ok] = np.minimum(map_coordinates(dist, [px[ok, 1], px[ok, 0]], order=1), cap)
        if ok.sum() < 20:
            rev = np.full(len(detected), cap)
        else:
            d, _ = cKDTree(px[ok]).query(detected)
            rev = np.minimum(d, cap)
        # the camera stands behind the near touchline, 1-20 m above the pitch: rules out the
        # mirror solution on the other side of the (symmetric) pitch
        R, _ = cv2.Rodrigues(x[6:9])
        cam = -R.T @ x[9:12]
        height = -cam[2]
        pen = 50.0 * np.array([max(0.0, W - 1.0 - cam[1]), max(0.0, 1.0 - height), max(0.0, height - 20.0)])
        return np.concatenate([fwd, 2.0 * rev, pen])     # detected lines must be explained by the model

    k0 = list(ref["k"]) if ref.get("model") == "fisheye" else [0.0, 0.0, 0.0]
    f0 = ref.get("f", 0.3 * w)
    anchor_centre = None
    starts = []
    if anchor is not None:
        R_a, _ = cv2.Rodrigues(np.array(anchor["rvec"], float))
        anchor_centre = -R_a.T @ np.array(anchor["tvec"], float)
        k0, f0 = list(anchor["k"]), anchor["f"]
        for tx in np.linspace(0, L, 9):
            for ty in (W * 0.2, W * 0.45, W * 0.7):
                r, t = _look_at(anchor_centre, [tx, ty, 0])
                for fs in (0.85, 1.0, 1.15):
                    starts.append(np.concatenate([[f0 * fs, w / 2, h / 2, *k0], r, t]))
    for tx in ((L / 4, 3 * L / 4, L / 2) if anchor is None else ()):
        for cx_off in (-8.0, 0.0, 8.0):
            for dy in (1.0, 4.0, 8.0):
                for height in (3.0, 6.0, 10.0):
                    cam = np.array([L / 2 + cx_off, W + dy, -height])
                    r, t = _look_at(cam, [tx, W * 0.45, 0])
                    for f in sorted({f0, 500.0, 800.0, 1100.0, 1400.0, 1800.0}):
                        starts.append(np.concatenate([[f, w / 2, h / 2, *k0], r, t]))
    costs = [float(np.sum(residuals(x) ** 2)) for x in starts]
    order = np.argsort(costs)[:12]
    lo = np.array([0.1 * w, 0.2 * w, 0.0, -0.6, -0.6, -0.6] + [-np.inf] * 6)
    hi = np.array([2.0 * w, 0.8 * w, 1.0 * h, 0.6, 0.6, 0.6] + [np.inf] * 6)
    best = None
    for i in order:
        x0 = np.clip(starts[i], lo + 1e-6, hi - 1e-6)
        try:
            r = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=5.0,
                              x_scale="jac", max_nfev=150)
        except (ValueError, cv2.error):
            continue
        if best is None or r.cost < best.cost:
            best = r
    calib = calib_of(best.x)
    if verbose:
        cam = -cv2.Rodrigues(np.array(calib["rvec"]))[0].T @ np.array(calib["tvec"])
        print(f"Search: camera at x={cam[0]:.1f} m, y={cam[1]:.1f} m, height={-cam[2]:.1f} m, "
              f"f={calib['f']:.0f}px")
    return calib


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
    ap.add_argument("--search", action="store_true",
                    help="find the camera position from scratch (camera placed or turned differently "
                         "than in the reference match); the reference only gives starting lens values")
    ap.add_argument("--same-camera", help="calibration of the other lens of this match (same device): "
                    "its camera position is reused and only the viewing direction is searched")
    ap.add_argument("--min-inliers", type=float, default=0.6, help="required share of lines on white")
    a = ap.parse_args()

    ref = json.loads(Path(a.ref).read_text())
    pitch = json.loads(Path(a.pitch).read_text())

    bg = median_background(Path(a.video).expanduser(), a.half, a.frames, a.start, a.end)
    h, w = bg.shape[:2]
    mask = line_mask(bg)
    if a.search or a.same_camera:
        anchor = json.loads(Path(a.same_camera).read_text()) if a.same_camera else None
        ref = global_search(mask, pitch, ref, anchor=anchor)
    # keep only white pixels inside the pitch (+ margin) as seen with the reference calibration:
    # removes goals, posts, fences, cars and buildings behind the pitch
    mask = cv2.bitwise_and(mask, pitch_area(ref, pitch, (w, h), a.margin_m))
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

    # pitch-line points that are visible in the reference view
    world = np.vstack([densify(l) if len(l) < 10 else l for l in pitch_lines(pitch)])
    px = pitch_to_pixels(ref, world)
    px = np.nan_to_num(px, nan=-1e6)
    inside = (px[:, 0] > 5) & (px[:, 0] < w - 5) & (px[:, 1] > 5) & (px[:, 1] < h - 5)
    world = world[inside]
    if len(world) < 50:
        raise SystemExit("Too few pitch lines visible with the reference calibration")

    def sample(calib, cap):
        p = np.nan_to_num(pitch_to_pixels(calib, world), nan=-1e6)
        # bilinear sampling keeps the cost smooth, so the optimiser gets useful gradients
        d = map_coordinates(dist, [np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)],
                            order=1, mode="nearest")
        outside = (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
        d[outside] = cap
        return np.minimum(d, cap)

    scale = 1.0 / w
    params = np.zeros(n_params(ref))
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
        for seg in np.split(p, np.where(~np.isfinite(p).all(axis=1))[0]):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) > 1:
                cv2.polylines(check, [seg.round().astype(np.int32)], False, (0, 255, 255), 2)
    cv2.imwrite(str(out.with_suffix(".check.jpg")), check)
    print(f"Check image: {out.with_suffix('.check.jpg')}  (yellow = calibration, red = detected lines)")

    out.write_text(json.dumps(calib, indent=2))
    print(f"Saved {out}")
    if inliers_after < a.min_inliers:
        print(f"WARNING: only {inliers_after:.0%} of the lines match - check the image; "
              "if the yellow lines do not lie on the pitch lines, calibrate this match manually")


if __name__ == "__main__":
    main()
