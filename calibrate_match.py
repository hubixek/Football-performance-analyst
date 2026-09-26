#!/usr/bin/env python3
"""Automatic calibration of a new match, using the camera as one rigid device.

The two lenses of the Veo camera are fixed in one housing: their lens parameters and their
relative rotation do not change between matches. Only the position and orientation of the
whole camera change (6 numbers). This script:
  1. takes the device model from a well calibrated reference match (both halves),
  2. detects the white pitch lines in a median background of the new match (both halves),
  3. searches the camera position and orientation for which the pitch lines of the model
     cover the detected lines in BOTH halves at once (many starting positions),
  4. refines the result (small corrections of the focal lengths allowed) and saves the
     calibration of both halves with check images.

Usage:
  python calibrate_match.py ~/football/videos/mecz1_dual.mp4 --name mecz1 --ref mecz2 \
      --calib-dir ~/football/calib --pitch pitch/pitch_6v6.json
Optional: --match matches/mecz1.json  (median background only from the playing time)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from auto_calibrate import line_mask, median_background, pitch_area
from calibrate import _look_at, densify, fisheye_project, pitch_lines

HALVES = ("top", "bottom")


def rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float))[0]


def device_model(top, bottom):
    """Lens parameters of both lenses + their relative pose, from a reference calibration."""
    R_t, R_b = rot(top["rvec"]), rot(bottom["rvec"])
    C_t = -R_t.T @ np.asarray(top["tvec"], float)
    C_b = -R_b.T @ np.asarray(bottom["tvec"], float)
    return {
        "intr": {h: {k: c[k] for k in ("f", "cx", "cy", "k")} for h, c in zip(HALVES, (top, bottom))},
        "R_rel": R_b @ R_t.T,                  # rotation from top-lens frame to bottom-lens frame
        "offset": R_t @ (C_b - C_t),           # bottom lens centre in the top-lens frame
    }


def lens_calibs(dev, rvec_top, centre, f_scale=(1.0, 1.0), corr=None):
    """Both lens calibrations for a given camera pose (pose of the top lens).

    corr: optional per-lens corrections {"top": (dk1, dcx, dcy), "bottom": (...)}."""
    R_t = rot(rvec_top)
    C_t = np.asarray(centre, float)
    R_b = dev["R_rel"] @ R_t
    C_b = C_t + R_t.T @ dev["offset"]
    out = {}
    for half, R, C, fs in (("top", R_t, C_t, f_scale[0]), ("bottom", R_b, C_b, f_scale[1])):
        intr = dev["intr"][half]
        dk1, dcx, dcy = corr[half] if corr else (0.0, 0.0, 0.0)
        k = list(intr["k"])
        k[0] = k[0] + dk1
        out[half] = {"model": "fisheye", "f": intr["f"] * fs, "cx": intr["cx"] + dcx, "cy": intr["cy"] + dcy,
                     "k": k, "rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "tvec": (-R @ C).tolist()}
    return out


def skeleton(mask):
    """Thin white lines to one pixel, so every line counts by its length, not its thickness."""
    img = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while img.any():
        eroded = cv2.erode(img, kernel)
        skel |= img - cv2.dilate(eroded, kernel)
        img = eroded
    return skel * 255


class LineEvidence:
    """Distance map and a sample of detected line pixels of one half, at a given scale."""

    def __init__(self, mask, scale, n_detected=1500):
        h, w = mask.shape
        self.scale = scale
        small = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) > 0
        small = skeleton(small.astype(np.uint8)) > 0
        self.h, self.w = small.shape
        self.dist = cv2.distanceTransform((~small).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        ys, xs = np.nonzero(small)
        pick = np.random.default_rng(0).choice(len(xs), size=min(n_detected, len(xs)), replace=False)
        self.detected = np.stack([xs[pick], ys[pick]], axis=1).astype(float) if len(xs) else np.zeros((0, 2))


def half_residuals(calib, ev, world, cap):
    px = fisheye_project(calib, world) * ev.scale
    ok = np.isfinite(px).all(axis=1) & (px[:, 0] >= 0) & (px[:, 0] < ev.w - 1) & (px[:, 1] >= 0) \
        & (px[:, 1] < ev.h - 1)
    fwd = np.full(len(world), cap * 0.5)
    if ok.any():
        fwd[ok] = np.minimum(map_coordinates(ev.dist, [px[ok, 1], px[ok, 0]], order=1), cap)
    if ok.sum() < 30 or len(ev.detected) == 0:
        rev = np.full(len(ev.detected), cap)
    else:
        rev = np.minimum(cKDTree(px[ok]).query(ev.detected)[0], cap)
    return fwd, rev, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="dual-lens match video")
    ap.add_argument("--name", required=True, help="name of the new match, e.g. mecz1")
    ap.add_argument("--ref", required=True, help="reference match with a good calibration, e.g. mecz2")
    ap.add_argument("--calib-dir", default=str(Path.home() / "football" / "calib"))
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--match", help="match config (half times) - background from the playing time only")
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()

    cdir = Path(a.calib_dir).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    dev = device_model(*(json.loads((cdir / f"{a.ref}_{h}.json").read_text()) for h in HALVES))

    start_s, end_s = 0, 0
    if a.match:
        cfg = json.loads(Path(a.match).read_text())
        start_s, end_s = cfg["halves"][0]["start_s"], cfg["halves"][-1]["end_s"]

    bgs, masks = {}, {}
    for half in HALVES:
        bgs[half] = median_background(Path(a.video).expanduser(), half, a.frames, start_s, end_s)
        masks[half] = line_mask(bgs[half])
        print(f"{half:6s}: {int((masks[half] > 0).sum())} line pixels detected")

    world = np.vstack([densify(l, 25) if len(l) < 10 else l for l in pitch_lines(pitch)])

    def make_residuals(scale, cap, with_f):
        evs = {h: LineEvidence(masks[h], scale) for h in HALVES}

        def residuals(x):
            fs = (x[6], x[7]) if with_f else (1.0, 1.0)
            corr = {"top": tuple(x[8:11]), "bottom": tuple(x[11:14])} if with_f else None
            calibs = lens_calibs(dev, x[:3], x[3:6], fs, corr)
            res = []
            for h in HALVES:
                fwd, rev, n_vis = half_residuals(calibs[h], evs[h], world, cap)
                res += [fwd, 2.0 * rev, [0.0 if n_vis >= 100 else cap * (100 - n_vis) / 10]]
            c = np.asarray(x[3:6])
            # plausible camera: behind the near touchline, 1.5 - 15 m above the pitch
            res.append([max(0.0, (W - 2) - c[1]) * 20, max(0.0, 1.5 + c[2]) * 20, max(0.0, -c[2] - 15) * 20])
            return np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in res])
        return residuals

    # 1) coarse search over camera positions and viewing directions of the top lens
    coarse = make_residuals(0.25, 25.0, False)
    starts = []
    for cx in (L / 2 - 8, L / 2, L / 2 + 8):
        for dy in (1.0, 3.0, 6.0, 10.0):
            for height in (2.5, 4.0, 6.0, 9.0):
                C = np.array([cx, W + dy, -height])
                for tx in (2.0, 8.0, 14.0, 20.0, 36.0, 42.0, 48.0, 54.0):
                    for ty in (W * 0.35, W * 0.6):
                        r, _ = _look_at(C, [tx, ty, 0])
                        starts.append(np.concatenate([r, C]))
    costs = np.array([np.sum(coarse(x) ** 2) for x in starts])
    best = None
    for i in np.argsort(costs)[:12]:
        r = least_squares(coarse, starts[i], loss="soft_l1", f_scale=5.0, x_scale="jac", max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    C = best.x[3:6]
    print(f"Camera: x={C[0]:.1f} m, y={C[1]:.1f} m, height={-C[2]:.1f} m (coarse)")

    # 2) refinement, first with a wide capture range, then precise; each lens may correct its
    #    focal length (+-12 %), distortion k1 (+-0.15) and image centre (+-80 px)
    x = np.concatenate([best.x, [1.0, 1.0], [0, 0, 0], [0, 0, 0]])
    lo = np.array([-np.inf] * 6 + [0.88, 0.88] + [-0.15, -80, -80] * 2)
    hi = np.array([np.inf] * 6 + [1.12, 1.12] + [0.15, 80, 80] * 2)
    scale = np.array([0.01] * 3 + [0.2] * 3 + [0.01, 0.01] + [0.01, 5, 5] * 2)
    for sc, cap in ((0.25, 30.0), (0.5, 15.0), (0.5, 6.0)):
        fine = make_residuals(sc, cap, True)
        x = least_squares(fine, x, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, x_scale=scale,
                          diff_step=1e-3, max_nfev=300).x
    corr = {"top": tuple(x[8:11]), "bottom": tuple(x[11:14])}
    calibs = lens_calibs(dev, x[:3], x[3:6], (x[6], x[7]), corr)
    C = x[3:6]
    print(f"Camera: x={C[0]:.1f} m, y={C[1]:.1f} m, height={-C[2]:.1f} m")
    for i, half in enumerate(HALVES):
        dk1, dcx, dcy = corr[half]
        print(f"  {half:6s} lens corrections: focal x{x[6 + i]:.3f}, k1 {dk1:+.3f}, centre {dcx:+.0f},{dcy:+.0f} px")

    # 3) quality: share of detected line pixels explained by the model, per half
    ok_all = True
    for half in HALVES:
        # draw the model lines, then check how many detected line pixels lie close to them
        h_img, w_img = masks[half].shape
        drawn = np.zeros((h_img, w_img), np.uint8)
        n_vis = 0
        for line in pitch_lines(pitch):
            pts = fisheye_project(calibs[half], densify(line, 200) if len(line) < 10 else line)
            for seg in np.split(pts, np.where(~np.isfinite(pts).all(axis=1))[0]):
                seg = seg[np.isfinite(seg).all(axis=1)]
                if len(seg) > 1:
                    cv2.polylines(drawn, [seg.round().astype(np.int32)], False, 255, 1)
        n_vis = int((drawn > 0).sum())
        model_dist = cv2.distanceTransform(255 - drawn, cv2.DIST_L2, 5)
        # only line pixels on the pitch (+2 m): walls, goal posts and trees behind it do not count
        on_pitch = cv2.bitwise_and(masks[half], pitch_area(calibs[half], pitch, (w_img, h_img), 2.0))
        ys, xs = np.nonzero(on_pitch)
        explained = float((model_dist[ys, xs] < 8).mean()) if len(xs) else 0.0
        ok_all &= explained >= 0.6 and n_vis >= 500
        calibs[half].update({"auto_from": a.ref, "explained_lines": explained, "image_size": [2048, 1024]})
        out = cdir / f"{a.name}_{half}.json"
        out.write_text(json.dumps(calibs[half], indent=2))
        check = bgs[half].copy()
        check[masks[half] > 0] = (0.5 * check[masks[half] > 0] + [0, 0, 127]).astype(np.uint8)
        for line in pitch_lines(pitch):
            p = fisheye_project(calibs[half], densify(line) if len(line) < 10 else line)
            for seg in np.split(p, np.where(~np.isfinite(p).all(axis=1))[0]):
                seg = seg[np.isfinite(seg).all(axis=1)]
                if len(seg) > 1:
                    cv2.polylines(check, [seg.round().astype(np.int32)], False, (0, 255, 255), 2)
        cv2.imwrite(str(out.with_suffix(".check.jpg")), check)
        print(f"{half:6s}: {explained:.0%} of detected line pixels explained -> {out}")
    if not ok_all:
        print("WARNING: weak match - check the images; if the yellow lines do not lie on the pitch "
              "lines, calibrate this match manually (calibrate.py) and use it as a new reference")


if __name__ == "__main__":
    main()
