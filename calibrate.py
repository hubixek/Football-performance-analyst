#!/usr/bin/env python3
"""Calibrate one camera lens: map image pixels to pitch coordinates in metres.

Input: points clicked with pick_pitch_points.html on one half of the dual-lens frame
(at least 6 points, spread over the visible part of the pitch, not all on one line).

Optionally also points clicked along pitch lines ("lines" in the JSON): they constrain the
lens distortion where there are no line intersections (e.g. the lower part of the image),
and a fisheye lens model is fitted (focal length, principal point, k1..k3, camera pose) -
needed for the very wide-angle Veo lenses.

Without line points two models are fitted and the better one is kept:
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


def undistort_radial(pts, K, dist):
    """Undistort pixels for a radial model (k1, k2, k3; no tangential terms).

    Solves r_d = r (1 + k1 r^2 + k2 r^4 + k3 r^6) for r by bisection on the range where
    the model is monotonic - robust also near the image corners of wide-angle lenses,
    where the iterative OpenCV method can fail.
    """
    k1, k2, k3 = dist[0], dist[1], dist[4] if len(dist) > 4 else 0.0
    xd = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yd = (pts[:, 1] - K[1, 2]) / K[1, 1]
    rd = np.hypot(xd, yd)
    f = lambda r: r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    grid = np.linspace(0, 5, 5001)
    deriv = 1 + 3 * k1 * grid ** 2 + 5 * k2 * grid ** 4 + 7 * k3 * grid ** 6
    r_max = grid[np.argmax(deriv <= 0)] if (deriv <= 0).any() else grid[-1]
    lo, hi = np.zeros_like(rd), np.full_like(rd, r_max)
    for _ in range(60):
        mid = (lo + hi) / 2
        below = f(mid) < rd
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    r = (lo + hi) / 2
    scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
    return np.stack([xd * scale * K[0, 0] + K[0, 2], yd * scale * K[1, 1] + K[1, 2]], axis=1)


# --- fisheye (Kannala-Brandt) model: for very wide-angle lenses ---------------------------
# theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6),  pixel = f * theta_d * dir + c

def _fe_R(calib):
    R, _ = cv2.Rodrigues(np.asarray(calib["rvec"], float))
    return R, np.asarray(calib["tvec"], float)


_TH = np.linspace(0, 1.6, 321)


def _fe_theta_max(k):
    th = _TH
    deriv = 1 + 3 * k[0] * th ** 2 + 5 * k[1] * th ** 4 + 7 * k[2] * th ** 6
    bad = np.where(deriv <= 0.05)[0]
    return min(th[bad[0]] if len(bad) else 1.55, 1.55)


def fisheye_project(calib, world_xy):
    """Nx2 pitch coordinates (m) -> Nx2 pixels; NaN where the point cannot be seen."""
    R, t = _fe_R(calib)
    k = calib["k"]
    obj = np.hstack([np.asarray(world_xy, float).reshape(-1, 2), np.zeros((len(world_xy), 1))])
    cam = obj @ R.T + t
    r = np.hypot(cam[:, 0], cam[:, 1])
    theta = np.arctan2(r, cam[:, 2])
    td = theta * (1 + k[0] * theta ** 2 + k[1] * theta ** 4 + k[2] * theta ** 6)
    scale = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 0.0)
    uv = np.stack([calib["f"] * cam[:, 0] * scale + calib["cx"], calib["f"] * cam[:, 1] * scale + calib["cy"]], 1)
    uv[theta > _fe_theta_max(k)] = np.nan
    return uv


def fisheye_unproject(calib, uv):
    """Nx2 pixels -> Nx2 pitch coordinates (m): ray from the camera intersected with the pitch."""
    R, t = _fe_R(calib)
    k = calib["k"]
    uv = np.asarray(uv, float).reshape(-1, 2)
    xd, yd = (uv[:, 0] - calib["cx"]) / calib["f"], (uv[:, 1] - calib["cy"]) / calib["f"]
    td = np.hypot(xd, yd)
    tmax = _fe_theta_max(k)
    f = lambda th: th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6)
    lo, hi = np.zeros_like(td), np.full_like(td, tmax)
    for _ in range(60):
        mid = (lo + hi) / 2
        below = f(mid) < td
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    theta = (lo + hi) / 2
    s = np.where(td > 1e-12, np.sin(theta) / np.maximum(td, 1e-12), 0.0)
    ray = np.stack([xd * s, yd * s, np.cos(theta)], axis=1) @ R        # camera -> world direction
    centre = -R.T @ t
    lam = -centre[2] / ray[:, 2]
    pts = centre[None, :2] + lam[:, None] * ray[:, :2]
    pts[(lam <= 0) | (td > f(tmax))] = np.nan
    return pts


def _look_at(cam, target):
    z = np.asarray(target, float) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, -1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel(), -R @ cam


def fit_fisheye(img_pts, world_pts, line_obs, pitch, size, verbose=True, init=None):
    """Fit f, principal point, k1..k3 and camera pose to keypoints and points on lines.

    Many starting camera positions (behind the near touchline, around the halfway line,
    different heights) are tried with a fast fit on subsampled data; the best ones are then
    refined on all data. Points that the model cannot see get a large penalty, so solutions
    where the pitch "disappears" from the image are rejected.
    """
    from scipy.optimize import least_squares
    w, h = size
    L, W = pitch["length_m"], pitch["width_m"]
    geoms = line_geometry(pitch)
    names = sorted({o[0] for o in line_obs})
    line_uv = {n: np.array([o[1] for o in line_obs if o[0] == n], float) for n in names}
    penalty = 2.0 * max(w, h)

    def calib_of(x):
        return {"model": "fisheye", "f": x[0], "cx": x[1], "cy": x[2], "k": [x[3], x[4], x[5]],
                "rvec": x[6:9].tolist(), "tvec": x[9:12].tolist()}

    def make_residuals(n_samples, line_step):
        samples = {n: sample_line(geoms[n], n=n_samples, extend=0.03) for n in names}
        uvs = {n: line_uv[n][::line_step] for n in names}

        def residuals(x):
            c = calib_of(x)
            proj = fisheye_project(c, world_pts)
            res = [np.where(np.isfinite(proj), proj - img_pts, penalty).ravel()]
            for n in names:
                curve = fisheye_project(c, samples[n])
                curve = curve[np.isfinite(curve).all(axis=1)]
                if len(curve) == 0:
                    res.append(np.full(len(uvs[n]), penalty))
                else:
                    res.append(np.min(np.linalg.norm(uvs[n][:, None] - curve[None], axis=2), axis=1))
            return np.concatenate(res)
        return residuals

    fast, full = make_residuals(120, 2), make_residuals(400, 1)

    # which half of the pitch does this lens look at?
    target_x = L / 4 if np.mean(world_pts[:, 0]) < L / 2 else 3 * L / 4
    starts = []
    for dy in (0.5, 3.0, 7.0):
        for height in (3.0, 6.0, 10.0):
            cam = np.array([L / 2, W + dy, -height])
            for tgt_y in (W * 0.35, W * 0.6):
                r, t = _look_at(cam, [target_x, tgt_y, 0])
                for f in (500.0, 750.0):
                    starts.append(np.concatenate([[f, w / 2, h / 2, 0, 0, 0], r, t]))
    lo = np.array([0.1 * w, 0.2 * w, 0.0 * h, -0.6, -0.6, -0.6] + [-np.inf] * 6)
    hi = np.array([1.5 * w, 0.8 * w, 1.0 * h, 0.6, 0.6, 0.6] + [np.inf] * 6)

    if init is not None:
        # continue from a previous calibration: skip the multi-start search
        x0 = np.concatenate([[init["f"], init["cx"], init["cy"], *init["k"]], init["rvec"], init["tvec"]])
        starts = [np.clip(x0, lo + 1e-6, hi - 1e-6)]
    quick = []
    for x0 in starts:
        try:
            quick.append(least_squares(fast, x0, bounds=(lo, hi), loss="huber", f_scale=10.0,
                                       x_scale="jac", max_nfev=60))
        except (ValueError, cv2.error):
            pass
    quick.sort(key=lambda r: r.cost)
    best = None
    for r0 in quick[:2]:
        r = least_squares(full, r0.x, bounds=(lo, hi), loss="huber", f_scale=5.0, x_scale="jac",
                          max_nfev=400, ftol=1e-7)
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("fisheye fit failed")
    calib = calib_of(best.x)
    res = full(best.x)
    hidden = int((np.abs(res) >= 0.99 * penalty).sum())
    res = res[np.abs(res) < 0.99 * penalty]
    calib["pixel_rmse"] = float(np.sqrt(np.mean(res ** 2)))
    if verbose:
        cam = -cv2.Rodrigues(np.array(calib["rvec"]))[0].T @ np.array(calib["tvec"])
        print(f"  camera at x={cam[0]:.1f} m, y={cam[1]:.1f} m, height={-cam[2]:.1f} m; "
              f"f={calib['f']:.0f}px, centre=({calib['cx']:.0f},{calib['cy']:.0f})"
              + (f"; {hidden} points not visible in the model!" if hidden else ""))
    return calib


def pixels_to_pitch(calib, pts):
    """Nx2 pixel coordinates -> Nx2 pitch coordinates in metres."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if calib["model"] == "fisheye":
        return fisheye_unproject(calib, pts)
    if calib["model"] == "distorted":
        pts = undistort_radial(pts, np.array(calib["K"]), np.array(calib["dist"]))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.array(calib["H"])).reshape(-1, 2)


def pitch_to_pixels(calib, pts):
    """Nx2 pitch coordinates in metres -> Nx2 pixel coordinates (for drawing)."""
    if calib["model"] == "fisheye":
        return fisheye_project(calib, pts)
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
    err = np.sqrt(((pixels_to_pitch(calib, img_pts) - world_pts) ** 2).sum(axis=1))
    err = np.where(np.isfinite(err), err, 99.0)
    return float(np.sqrt((err ** 2).mean())), err


def fit_homography(img_pts, world_pts):
    H, _ = cv2.findHomography(img_pts, world_pts, 0)
    return {"model": "homography", "H": H.tolist()}


def fit_distorted(img_pts, world_pts, size):
    w, h = size
    K0 = np.array([[w * 0.5, 0, w / 2], [0, w * 0.5, h / 2], [0, 0, 1]], np.float64)
    obj = np.hstack([world_pts, np.zeros((len(world_pts), 1))]).astype(np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj], [img_pts.astype(np.float32)], (w, h), K0, None,
                                                   flags=flags)
    und = undistort_radial(img_pts.astype(np.float64), K, dist.ravel())
    H, _ = cv2.findHomography(und, world_pts, 0)
    return {"model": "distorted", "K": K.tolist(), "dist": dist.ravel().tolist(), "H": H.tolist(),
            "rvec": rvecs[0].ravel().tolist(), "tvec": tvecs[0].ravel().tolist()}


# --- refinement with points clicked along pitch lines -------------------------------------

def line_geometry(p):
    """World geometry of named pitch lines: ('seg', a, b) or ('circle', centre, r)."""
    L, W = p["length_m"], p["width_m"]
    d, bw, r = p["penalty_area_depth_m"], p["penalty_area_width_m"], p["centre_circle_radius_m"]
    y0, y1 = (W - bw) / 2, (W + bw) / 2
    return {
        "touch_far": ("seg", (0, 0), (L, 0)), "touch_near": ("seg", (0, W), (L, W)),
        "goal_left": ("seg", (0, 0), (0, W)), "goal_right": ("seg", (L, 0), (L, W)),
        "halfway": ("seg", (L / 2, 0), (L / 2, W)),
        "box_left_front": ("seg", (d, y0), (d, y1)), "box_left_top": ("seg", (0, y0), (d, y0)),
        "box_left_bottom": ("seg", (0, y1), (d, y1)),
        "box_right_front": ("seg", (L - d, y0), (L - d, y1)), "box_right_top": ("seg", (L, y0), (L - d, y0)),
        "box_right_bottom": ("seg", (L, y1), (L - d, y1)),
        "circle": ("circle", (L / 2, W / 2), r),
    }


def sample_line(geom, n=600, extend=0.0):
    kind = geom[0]
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, n)
        (cx, cy), r = geom[1], geom[2]
        return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
    a, b = np.array(geom[1], float), np.array(geom[2], float)
    t = np.linspace(-extend, 1 + extend, n)[:, None]
    return a + t * (b - a)


def world_distance(geom, xy):
    """Distance (m) of world points to a named line."""
    if geom[0] == "circle":
        return np.abs(np.linalg.norm(xy - np.array(geom[1]), axis=1) - geom[2])
    a, b = np.array(geom[1], float), np.array(geom[2], float)
    t = np.clip(((xy - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(xy - (a + t[:, None] * (b - a)), axis=1)


def pose_to_calib(K, dist, rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    G = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float)])
    H = np.linalg.inv(G)
    return {"model": "distorted", "K": K.tolist(), "dist": list(map(float, dist)), "H": (H / H[2, 2]).tolist(),
            "rvec": list(map(float, rvec)), "tvec": list(map(float, tvec))}


def _project_valid(obj, r, t, K, dist, size, margin=0.3):
    """Project 3D points, dropping points behind the camera or where the distortion
    polynomial is no longer monotonic (there the projection folds back and is meaningless)."""
    R, _ = cv2.Rodrigues(np.asarray(r, float))
    cam = obj @ R.T + np.asarray(t, float)
    front = cam[:, 2] > 1e-3
    xn = cam[:, :2] / np.where(front, cam[:, 2], 1)[:, None]
    rr = np.linalg.norm(xn, axis=1)
    k1, k2, k3 = dist[0], dist[1], dist[4]
    # d/dr [r (1 + k1 r^2 + k2 r^4 + k3 r^6)] > 0  -> distortion still monotonic
    deriv = 1 + 3 * k1 * rr ** 2 + 5 * k2 * rr ** 4 + 7 * k3 * rr ** 6
    ok = front & (deriv > 0.05)
    if not ok.any():
        return np.zeros((0, 2)), ok
    proj, _ = cv2.projectPoints(obj[ok], r, t, K, dist)
    proj = proj.reshape(-1, 2)
    w, h = size
    inside = (proj[:, 0] > -margin * w) & (proj[:, 0] < (1 + margin) * w) & \
             (proj[:, 1] > -margin * h) & (proj[:, 1] < (1 + margin) * h)
    idx = np.where(ok)[0]
    ok[:] = False
    ok[idx[inside]] = True
    return proj[inside], ok


def refine_with_lines(start, img_pts, world_pts, line_obs, pitch, size):
    """Fit focal length, principal point, k1..k3 and pose using keypoints and line points.

    Several starting points (focal length, k1) are tried, each with an initial pose from
    solvePnP on the keypoints; the solution with the lowest cost is kept.
    """
    from scipy.optimize import least_squares
    w, h = size
    geoms = line_geometry(pitch)
    names = sorted({o[0] for o in line_obs})
    samples = {n: np.hstack([sample_line(geoms[n], n=500, extend=0.03), np.zeros((500, 1))]) for n in names}
    line_uv = {n: np.array([o[1] for o in line_obs if o[0] == n], float) for n in names}
    obj_kp = np.hstack([world_pts, np.zeros((len(world_pts), 1))]).astype(np.float64)
    penalty = 0.25 * max(w, h)   # residual for a line point when its line is not visible at all

    def unpack(x):
        K = np.array([[x[0], 0, x[1]], [0, x[0], x[2]], [0, 0, 1]])
        return K, np.array([x[3], x[4], 0, 0, x[5]]), x[6:9], x[9:12]

    def residuals(x):
        K, dist, r, t = unpack(x)
        res = []
        proj, ok = _project_valid(obj_kp, r, t, K, dist, size, margin=2.0)
        kp_res = np.full((len(obj_kp), 2), penalty)
        kp_res[ok] = proj - img_pts[ok]
        res.append(kp_res.ravel())
        for n in names:
            curve, _ = _project_valid(samples[n], r, t, K, dist, size)
            if len(curve) == 0:
                res.append(np.full(len(line_uv[n]), penalty))
                continue
            d = np.min(np.linalg.norm(line_uv[n][:, None, :] - curve[None, :, :], axis=2), axis=1)
            res.append(d)
        return np.concatenate(res)

    lo = np.array([0.15 * w, 0.3 * w, 0.2 * h, -1.0, -1.0, -1.0] + [-np.inf] * 6)
    hi = np.array([2.5 * w, 0.7 * w, 0.8 * h, 1.0, 1.0, 1.0] + [np.inf] * 6)
    best = None
    starts = [(f, k1) for f in (0.22 * w, 0.3 * w, 0.4 * w, 0.55 * w) for k1 in (-0.35, -0.2, -0.05)]
    if start is not None and "rvec" in start:
        K0 = np.array(start["K"])
        starts.insert(0, (K0[0, 0], float(np.array(start["dist"])[0])))
    for f, k1 in starts:
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        dist = np.array([k1, 0.0, 0, 0, 0])
        ok, r, t = cv2.solvePnP(obj_kp, img_pts.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        x0 = np.concatenate([[f, w / 2, h / 2, k1, 0.0, 0.0], r.ravel(), t.ravel()])
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0,
                                x_scale="jac", max_nfev=600)
        except (cv2.error, ValueError):
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("line refinement failed")
    K, dist, r, t = unpack(best.x)
    calib = pose_to_calib(K, dist, r, t)
    r_final = residuals(best.x)
    r_final = r_final[np.abs(r_final) < 0.99 * penalty]
    calib["pixel_rmse"] = float(np.sqrt(np.mean(r_final ** 2))) if len(r_final) else float("nan")
    return calib


def line_errors(calib, line_obs, pitch):
    geoms = line_geometry(pitch)
    out = {}
    for name in sorted({o[0] for o in line_obs}):
        uv = np.array([o[1] for o in line_obs if o[0] == name])
        out[name] = world_distance(geoms[name], pixels_to_pitch(calib, uv))
    return out


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


def save_check_image(calib, pitch, image_path, img_pts, ids, line_obs, out_path):
    img = cv2.imread(str(image_path))
    h_img, w_img = img.shape[:2]
    for line in pitch_lines(pitch):
        px = pitch_to_pixels(calib, densify(line) if len(line) < 10 else line)
        ok = np.isfinite(px).all(axis=1) & (np.abs(np.nan_to_num(px)) < 4 * max(w_img, h_img)).all(axis=1)
        for seg in np.split(px, np.where(~ok)[0]):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) > 1:
                cv2.polylines(img, [seg.round().astype(np.int32)], False, (0, 255, 255), 2)
    for _, (u, v) in line_obs:
        cv2.circle(img, (int(u), int(v)), 4, (255, 0, 255), -1)
    for (u, v), i in zip(img_pts, ids):
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, str(i), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)


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
    line_obs = [(l["line"], (l["u"], l["v"])) for l in data.get("lines", [])]
    if line_obs:
        per_line = {}
        for name, _ in line_obs:
            per_line[name] = per_line.get(name, 0) + 1
        print("Points on lines: " + ", ".join(f"{k}: {v}" for k, v in sorted(per_line.items())))
        if len(per_line) == 1 and len(line_obs) > 15:
            raise SystemExit("All line points are assigned to ONE line - in pick_pitch_points.html choose the "
                             "right line before clicking (or Shift+click points to reassign them)")

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

    if line_obs:
        print(f"\nFitting fisheye model with {len(line_obs)} points on lines (may take a minute) ...")
        base = best
        fe = fit_fisheye(img_pts, world_pts, line_obs, pitch, (data["width"], data["height"]))
        fe["rmse_m"], _ = rmse(fe, img_pts, world_pts)
        before, after = line_errors(base, line_obs, pitch), line_errors(fe, line_obs, pitch)
        print(f"fisheye    mean error: {fe['rmse_m']:.2f} m  (reprojection {fe['pixel_rmse']:.1f} px)")
        print(f"Distance of line points to their pitch line (mean m, {base['model']} -> fisheye):")
        for name in after:
            print(f"  {name:18s} {np.nanmean(before[name]):6.2f} -> {np.nanmean(after[name]):6.2f}  "
                  f"({len(after[name])} pts)")
        best = fe
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
    h_img, w_img = img.shape[:2]
    for line in pitch_lines(pitch):
        px = pitch_to_pixels(best, densify(line) if len(line) < 10 else line)
        ok = np.isfinite(px).all(axis=1) & (np.abs(px) < 4 * max(w_img, h_img)).all(axis=1)
        # draw only the continuous visible parts (points far outside the image are unreliable)
        for seg in np.split(px, np.where(~ok)[0]):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) > 1:
                cv2.polylines(img, [seg.round().astype(np.int32)], False, (0, 255, 255), 2)
    for _, (u, v) in line_obs:
        cv2.circle(img, (int(u), int(v)), 4, (255, 0, 255), -1)
    for (u, v), i in zip(img_pts, ids):
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, str(i), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    check = out.with_suffix(".check.jpg")
    cv2.imwrite(str(check), img)
    print(f"Check image: {check}  (yellow lines should lie on the pitch lines)")


if __name__ == "__main__":
    main()
