#!/usr/bin/env python3
"""Calibrate both lenses and estimate the pitch dimensions from the clicked lines.

  1. separate fisheye calibration of both lenses with the current pitch dimensions
  2. in turns: measure the dimensions from the clicked line points mapped onto the pitch
     (width, penalty area depth and width, centre circle radius) and recalibrate both
     lenses with them, until the dimensions stop changing
The pitch length stays fixed (one real distance must be known).

Usage:
  python refine_pitch.py --name mecz2 --calib-dir ~/football/calib \
      --pitch pitch/pitch_6v6.json --out-pitch pitch/pitch_6v6.json
Uses calib/<name>_top_points.json, <name>_bottom_points.json and the matching .jpg frames;
writes calib/<name>_top.json, <name>_bottom.json and check images.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scipy.optimize import least_squares

from calibrate import (fisheye_project, fit_fisheye, line_geometry, pixels_to_pitch, rmse, sample_line,
                       save_check_image, world_distance)
from fit_pitch_dims import keypoints
import config


def load(points_path, pitch):
    data = json.loads(Path(points_path).read_text())
    ids = [p["id"] for p in data["points"]]
    img = np.array([(p["u"], p["v"]) for p in data["points"]], float)
    lines = [(l["line"], (l["u"], l["v"])) for l in data.get("lines", [])]
    return data, ids, img, lines


def estimate_dims(pitch, halves):
    """Measure dimensions from line points mapped with the current calibrations."""
    L, W = pitch["length_m"], pitch["width_m"]
    pts = {}
    for h in halves:
        for name, uv in h["lines"]:
            pts.setdefault(name, []).append(uv)
    mapped = {}
    for h in halves:
        for name in {n for n, _ in h["lines"]}:
            uv = np.array([uv for n, uv in h["lines"] if n == name])
            xy = pixels_to_pitch(h["calib"], uv)
            mapped.setdefault(name, []).append(xy[np.isfinite(xy).all(axis=1)])
    mapped = {k: np.vstack(v) for k, v in mapped.items() if sum(len(a) for a in v) >= 3}
    med = lambda name, axis: float(np.median(mapped[name][:, axis]))
    new = {}
    if "touch_near" in mapped and "touch_far" in mapped:
        new["width_m"] = med("touch_near", 1) - med("touch_far", 1)
    depths = []
    if "box_left_front" in mapped and "goal_left" in mapped:
        depths.append(med("box_left_front", 0) - med("goal_left", 0))
    if "box_right_front" in mapped and "goal_right" in mapped:
        depths.append(med("goal_right", 0) - med("box_right_front", 0))
    if depths:
        new["penalty_area_depth_m"] = float(np.mean(depths))
    widths = []
    for side in ("left", "right"):
        a, b = f"box_{side}_bottom", f"box_{side}_top"
        if a in mapped and b in mapped:
            widths.append(med(a, 1) - med(b, 1))
    if widths:
        new["penalty_area_width_m"] = float(np.mean(widths))
    if "circle" in mapped:
        centre = np.array([L / 2, (med("touch_near", 1) + med("touch_far", 1)) / 2
                           if "touch_near" in mapped and "touch_far" in mapped else W / 2])
        new["centre_circle_radius_m"] = float(np.median(np.linalg.norm(mapped["circle"] - centre, axis=1)))
    return new


def with_dims(pitch, dims):
    p = dict(pitch, **{k: round(v, 2) for k, v in dims.items()})
    kp = keypoints(p)
    p["keypoints"] = [dict(k, x=round(kp[k["id"]][0], 3), y=round(kp[k["id"]][1], 3)) for k in pitch["keypoints"]]
    return p


def joint_fit(pitch, halves, keys):
    """One least-squares fit of the pitch dimensions and both lenses together."""
    dims0 = np.array([pitch[k] for k in keys], float)
    lens0 = [np.concatenate([[h["calib"]["f"], h["calib"]["cx"], h["calib"]["cy"], *h["calib"]["k"]],
                             h["calib"]["rvec"], h["calib"]["tvec"]]) for h in halves]
    x0 = np.concatenate([dims0, *lens0])
    penalty = 4000.0

    def calib_of(v):
        return {"model": "fisheye", "f": v[0], "cx": v[1], "cy": v[2], "k": [v[3], v[4], v[5]],
                "rvec": v[6:9].tolist(), "tvec": v[9:12].tolist()}

    def residuals(x):
        p = dict(pitch, **dict(zip(keys, x[:4])))
        kp = keypoints(p)
        geoms = line_geometry(p)
        res = []
        for j, h in enumerate(halves):
            c = calib_of(x[4 + 12 * j: 16 + 12 * j])
            world = np.array([kp[i] for i in h["ids"]], float)
            proj = fisheye_project(c, world)
            res.append(np.where(np.isfinite(proj), proj - h["img"], penalty).ravel())
            for name in {n for n, _ in h["lines"]}:
                uv = np.array([uv for n, uv in h["lines"] if n == name])
                curve = fisheye_project(c, sample_line(geoms[name], n=300, extend=0.03))
                curve = curve[np.isfinite(curve).all(axis=1)]
                if len(curve) == 0:
                    res.append(np.full(len(uv), penalty))
                else:
                    res.append(np.min(np.linalg.norm(uv[:, None] - curve[None], axis=2), axis=1))
        return np.concatenate(res)

    lo_dims, hi_dims = np.array([15, 3, 5, 2], float), np.array([50, 20, 40, 12], float)
    lo_l = np.array([0.1 * 2048, 0.2 * 2048, 0.0, -0.6, -0.6, -0.6] + [-np.inf] * 6)
    hi_l = np.array([1.5 * 2048, 0.8 * 2048, 1024.0, 0.6, 0.6, 0.6] + [np.inf] * 6)
    lo = np.concatenate([lo_dims] + [lo_l] * len(halves))
    hi = np.concatenate([hi_dims] + [hi_l] * len(halves))
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    scale_l = [50, 20, 20, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
    x_scale = np.array([0.5, 0.5, 0.5, 0.3] + scale_l * len(halves))
    res = least_squares(residuals, x0, bounds=(lo, hi), loss="huber", f_scale=5.0, x_scale=x_scale,
                        diff_step=1e-4, max_nfev=5000)
    dims = dict(zip(keys, res.x[:4].tolist()))
    for j, h in enumerate(halves):
        h["calib"] = calib_of(res.x[4 + 12 * j: 16 + 12 * j])
        r = residuals(res.x)
    r = r[np.abs(r) < 0.99 * penalty]
    return dims, float(np.sqrt(np.mean(r ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="match name, e.g. mecz2")
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--out-pitch", required=True, help="pitch definition with the estimated dimensions")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--relax", type=float, default=1.0, help="over-relaxation of dimension updates")
    a = ap.parse_args()

    cdir = Path(a.calib_dir).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    halves = []
    for half in ("top", "bottom"):
        data, ids, img, lines = load(cdir / f"{a.name}_{half}_points.json", pitch)
        halves.append({"half": half, "data": data, "ids": ids, "img": img, "lines": lines, "calib": None})

    keys = ["width_m", "penalty_area_depth_m", "penalty_area_width_m", "centre_circle_radius_m"]
    print("Start: " + ", ".join(f"{k}={pitch[k]}" for k in keys))
    kp = {k["id"]: (k["x"], k["y"]) for k in pitch["keypoints"]}
    print("\n1) Separate calibration of both lenses (about a minute each) ...")
    for h in halves:
        world = np.array([kp[i] for i in h["ids"]], float)
        size = (h["data"]["width"], h["data"]["height"])
        h["calib"] = fit_fisheye(h["img"], world, h["lines"], pitch, size, verbose=True)
        print(f"  {h['half']:6s} reprojection {h['calib']['pixel_rmse']:.1f} px")

    print("\n2) Estimating pitch dimensions (calibration and measurement in turns) ...")
    for it in range(1, a.iters + 1):
        new = estimate_dims(pitch, halves)
        # over-relaxation: the lens fit absorbs part of a wrong dimension, so a measured
        # change is only part of the real one
        dims = {k: pitch[k] + a.relax * (new[k] - pitch[k]) for k in new}
        change = max(abs(dims[k] - pitch[k]) for k in dims) if dims else 0
        pitch = with_dims(pitch, dims)
        kp = {k["id"]: (k["x"], k["y"]) for k in pitch["keypoints"]}
        errs = []
        for h in halves:
            world = np.array([kp[i] for i in h["ids"]], float)
            size = (h["data"]["width"], h["data"]["height"])
            h["calib"] = fit_fisheye(h["img"], world, h["lines"], pitch, size, verbose=False, init=h["calib"])
            errs.append(h["calib"]["pixel_rmse"])
        print(f"  {it}: " + ", ".join(f"{k}={pitch[k]}" for k in keys)
              + f"  (reprojection {errs[0]:.1f} / {errs[1]:.1f} px)")
        if change < 0.05:
            print("  dimensions stable")
            break

    # final calibration with the final dimensions
    kp = {k["id"]: (k["x"], k["y"]) for k in pitch["keypoints"]}
    print("\n=== Final: " + ", ".join(f"{k}={pitch[k]}" for k in keys))
    for h in halves:
        world = np.array([kp[i] for i in h["ids"]], float)
        size = (h["data"]["width"], h["data"]["height"])
        calib = fit_fisheye(h["img"], world, h["lines"], pitch, size, verbose=False, init=h["calib"])
        calib["rmse_m"], per_point = rmse(calib, h["img"], world)
        calib["image_size"] = list(size)
        calib["point_ids"] = h["ids"]
        out = cdir / f"{a.name}_{h['half']}.json"
        out.write_text(json.dumps(calib, indent=2))
        save_check_image(calib, pitch, cdir / f"{a.name}_{h['half']}.jpg", h["img"], h["ids"], h["lines"],
                         out.with_suffix(".check.jpg"))
        print(f"  {h['half']:6s} keypoints {calib['rmse_m']:.2f} m, reprojection {calib['pixel_rmse']:.1f} px"
              f" -> {out}")
    Path(a.out_pitch).write_text(json.dumps(pitch, indent=2))
    print(f"Saved pitch dimensions to {a.out_pitch}")


if __name__ == "__main__":
    main()
