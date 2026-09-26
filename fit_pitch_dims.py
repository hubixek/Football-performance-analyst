#!/usr/bin/env python3
"""Estimate pitch dimensions (width, penalty area, centre circle) from clicked points.

If the pitch model does not match the real pitch, no calibration can fit all points
and the error stays high on some points (e.g. penalty area corners). This script
searches for the dimensions for which the lens calibrations fit the clicked points best,
using the points of all given lenses together.

The pitch length stays fixed (one real distance must be known: a calibration cannot
tell a bigger pitch from a camera standing further away).

Usage:
  python fit_pitch_dims.py --points calib/mecz1_top_points.json calib/mecz1_bottom_points.json \
      --pitch pitch/pitch_6v6.json --out pitch/pitch_6v6_fitted.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from calibrate import fit_distorted, rmse

PARAMS = ["width_m", "penalty_area_depth_m", "penalty_area_width_m", "centre_circle_radius_m"]


def keypoints(p):
    """Keypoint coordinates (metres) for the given dimensions - same layout as pitch_6v6.json."""
    L, W = p["length_m"], p["width_m"]
    d, bw, r = p["penalty_area_depth_m"], p["penalty_area_width_m"], p["centre_circle_radius_m"]
    y0, y1 = (W - bw) / 2, (W + bw) / 2
    return {0: (0, 0), 1: (L / 2, 0), 2: (L, 0), 3: (L, W), 4: (L / 2, W), 5: (0, W),
            6: (0, y0), 7: (d, y0), 8: (d, y1), 9: (0, y1),
            10: (L, y0), 11: (L - d, y0), 12: (L - d, y1), 13: (L, y1),
            14: (L / 2, W / 2 - r), 15: (L / 2, W / 2 + r), 16: (L / 2 - r, W / 2),
            17: (L / 2 + r, W / 2), 18: (L / 2, W / 2)}


def total_error(dims, sets):
    kp = keypoints(dims)
    errs = []
    for data in sets:
        ids = [p["id"] for p in data["points"]]
        img = np.array([(p["u"], p["v"]) for p in data["points"]], np.float64)
        world = np.array([kp[i] for i in ids], np.float64)
        try:
            calib = fit_distorted(img, world, (data["width"], data["height"]))
            e, _ = rmse(calib, img, world)
        except Exception:
            e = 1e3
        errs.append(e if np.isfinite(e) else 1e3)
    return float(np.mean(errs)), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True, help="points JSON of one or more lenses")
    ap.add_argument("--pitch", required=True, help="current pitch definition")
    ap.add_argument("--out", required=True, help="pitch definition with fitted dimensions")
    a = ap.parse_args()

    pitch = json.loads(Path(a.pitch).read_text())
    sets = [json.loads(Path(p).read_text()) for p in a.points]

    start = np.array([pitch[k] for k in PARAMS], float)
    before, errs_before = total_error(pitch, sets)
    print("Before: " + ", ".join(f"{k}={pitch[k]}" for k in PARAMS))
    print(f"  mean error {before:.2f} m  (per lens: {', '.join(f'{e:.2f}' for e in errs_before)})")

    def cost(x):
        dims = dict(pitch, **dict(zip(PARAMS, x)))
        W, d, bw, r = x
        if not (15 < W < 50 and 3 < d < 20 and 5 < bw < W - 1 and 2 < r < 12):
            return 1e3
        return total_error(dims, sets)[0]

    res = minimize(cost, start, method="Nelder-Mead",
                   options={"xatol": 0.02, "fatol": 1e-3, "maxiter": 600, "initial_simplex": None})
    fitted = dict(pitch, **{k: round(float(v), 2) for k, v in zip(PARAMS, res.x)})
    after, errs_after = total_error(fitted, sets)
    print("After:  " + ", ".join(f"{k}={fitted[k]}" for k in PARAMS))
    print(f"  mean error {after:.2f} m  (per lens: {', '.join(f'{e:.2f}' for e in errs_after)})")

    kp = keypoints(fitted)
    fitted["keypoints"] = [dict(k, x=round(kp[k["id"]][0], 3), y=round(kp[k["id"]][1], 3))
                           for k in pitch["keypoints"]]
    Path(a.out).write_text(json.dumps(fitted, indent=2))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
