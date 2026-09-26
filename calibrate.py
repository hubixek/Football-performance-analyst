#!/usr/bin/env python3
"""Calibrate one camera lens: map image pixels to pitch coordinates in metres.

Input: points clicked with pick_pitch_points.html on one half of the dual-lens frame
(at least 6 points, spread over the visible part of the pitch, not all on one line).

Two models are fitted and the better one is kept:
  - homography: straight lines stay straight (enough for a normal lens)
  - distorted:  lens distortion (k1, k2) is estimated first, then a homography on
                undistorted points (needed for wide-angle lenses with curved lines)

Output JSON is used by other scripts via pixels_to_pitch() below.
A check image with the pitch lines drawn back onto the frame is saved as well.

Usage:
  python calibrate.py --points top_points.json --image top.jpg \
      --pitch pitch/pitch_6v6.json --out calib/mecz1_top.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def pixels_to_pitch(calib, pts):
    """Nx2 pixel coordinates -> Nx2 pitch coordinates in metres."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    if calib["model"] == "distorted":
        K, dist = np.array(calib["K"]), np.array(calib["dist"])
        pts = cv2.undistortPoints(pts, K, dist, P=K)
    return cv2.perspectiveTransform(pts, np.array(calib["H"])).reshape(-1, 2)


def pitch_to_pixels(calib, pts):
    """Nx2 pitch coordinates in metres -> Nx2 pixel coordinates (for drawing)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    und = cv2.perspectiveTransform(pts, np.linalg.inv(np.array(calib["H"]))).reshape(-1, 2)
    if calib["model"] != "distorted":
        return und
    K, dist = np.array(calib["K"]), np.array(calib["dist"])
    norm = (und - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    obj = np.hstack([norm, np.ones((len(norm), 1))]).astype(np.float32)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)
    return img.reshape(-1, 2)


def rmse(calib, img_pts, world_pts):
    err = pixels_to_pitch(calib, img_pts) - world_pts
    return float(np.sqrt((err ** 2).sum(axis=1).mean())), np.sqrt((err ** 2).sum(axis=1))


def fit_homography(img_pts, world_pts):
    H, _ = cv2.findHomography(img_pts, world_pts, 0)
    return {"model": "homography", "H": H.tolist()}


def fit_distorted(img_pts, world_pts, size):
    w, h = size
    K0 = np.array([[w * 0.5, 0, w / 2], [0, w * 0.5, h / 2], [0, 0, 1]], np.float64)
    obj = np.hstack([world_pts, np.zeros((len(world_pts), 1))]).astype(np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    _, K, dist, _, _ = cv2.calibrateCamera([obj], [img_pts.astype(np.float32)], (w, h), K0, None, flags=flags)
    und = cv2.undistortPoints(img_pts.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(-1, 2)
    H, _ = cv2.findHomography(und, world_pts, 0)
    return {"model": "distorted", "K": K.tolist(), "dist": dist.ravel().tolist(), "H": H.tolist()}


def pitch_lines(p):
    L, W = p["length_m"], p["width_m"]
    d, bw, r = p["penalty_area_depth_m"], p["penalty_area_width_m"], p["centre_circle_radius_m"]
    y0, y1 = (W - bw) / 2, (W + bw) / 2
    seg = lambda a, b, n=40: np.linspace(a, b, n)
    lines = [seg((0, 0), (L, 0)), seg((L, 0), (L, W)), seg((L, W), (0, W)), seg((0, W), (0, 0)),
             seg((L / 2, 0), (L / 2, W)),
             np.array([(0, y0), (d, y0), (d, y1), (0, y1)]), np.array([(L, y0), (L - d, y0), (L - d, y1), (L, y1)])]
    t = np.linspace(0, 2 * np.pi, 120)
    lines.append(np.stack([L / 2 + r * np.cos(t), W / 2 + r * np.sin(t)], axis=1))
    return [np.asarray(l, np.float32) for l in lines]


def densify(poly, n=20):
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        out.append(np.linspace(a, b, n, endpoint=False))
    out.append(poly[-1:])
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="JSON from pick_pitch_points.html")
    ap.add_argument("--image", required=True, help="the same frame (for the check image)")
    ap.add_argument("--pitch", required=True, help="pitch definition, e.g. pitch/pitch_6v6.json")
    ap.add_argument("--out", required=True, help="output calibration JSON")
    ap.add_argument("--model", choices=["auto", "homography", "distorted"], default="auto")
    a = ap.parse_args()

    pitch = json.loads(Path(a.pitch).read_text())
    kp = {k["id"]: (k["x"], k["y"]) for k in pitch["keypoints"]}
    data = json.loads(Path(a.points).read_text())
    ids = [p["id"] for p in data["points"]]
    img_pts = np.array([(p["u"], p["v"]) for p in data["points"]], np.float64)
    world_pts = np.array([kp[i] for i in ids], np.float64)
    if len(ids) < 6:
        raise SystemExit(f"Only {len(ids)} points - click at least 6 (better 8+)")

    candidates = []
    if a.model in ("auto", "homography"):
        candidates.append(fit_homography(img_pts, world_pts))
    if a.model in ("auto", "distorted"):
        try:
            candidates.append(fit_distorted(img_pts, world_pts, (data["width"], data["height"])))
        except cv2.error as e:
            print("Distortion model failed:", e)
    for c in candidates:
        c["rmse_m"], _ = rmse(c, img_pts, world_pts)
        print(f"{c['model']:10s} mean error: {c['rmse_m']:.2f} m")
    best = min(candidates, key=lambda c: c["rmse_m"])
    best["image_size"] = [data["width"], data["height"]]
    best["point_ids"] = ids

    _, per_point = rmse(best, img_pts, world_pts)
    print(f"\nUsing '{best['model']}'. Error per point (m):")
    for i, e in sorted(zip(ids, per_point), key=lambda x: -x[1]):
        flag = "  <- check this point" if e > max(3 * best["rmse_m"], 1.0) else ""
        print(f"  {i:2d} {pitch['keypoints'][i]['name']:26s} {e:5.2f}{flag}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    print(f"\nSaved {out}")

    img = cv2.imread(a.image)
    for line in pitch_lines(pitch):
        px = pitch_to_pixels(best, densify(line) if len(line) < 10 else line)
        cv2.polylines(img, [px.round().astype(np.int32)], False, (0, 255, 255), 2)
    for (u, v), i in zip(img_pts, ids):
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, str(i), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    check = out.with_suffix(".check.jpg")
    cv2.imwrite(str(check), img)
    print(f"Check image: {check}  (yellow lines should lie on the pitch lines)")


if __name__ == "__main__":
    main()
