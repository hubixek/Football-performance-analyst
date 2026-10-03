import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401  (the repository on the path)
import compare_actions as ca


def actions(n_pass=40, n_shot=10):
    rows = []
    for i in range(n_pass):
        rows.append({"type": "pass", "t": 10.0 + 5.0 * i, "half": 1, "team": i % 2, "x": 20.0, "y": 10.0, "x2": 30.0, "y2": 12.0,
                     "outcome": "completed" if i % 3 else "lost", "inferred": False})
    for i in range(n_shot):
        rows.append({"type": "shot", "t": 12.0 + 20.0 * i, "half": 1, "team": i % 2, "x": 45.0, "y": 16.0, "x2": np.nan, "y2": np.nan,
                     "outcome": ["goal", "saved", "off_target", "blocked"][i % 4], "inferred": i == 0})
    return rows


class Picking(unittest.TestCase):
    def test_detections_are_spaced_sorted_and_reproducible(self):
        a = actions()
        d1 = ca.pick_detections(a, "pass", 10, 15.0, np.random.default_rng(1))
        d2 = ca.pick_detections(a, "pass", 10, 15.0, np.random.default_rng(1))
        self.assertEqual([x["t"] for x in d1], [x["t"] for x in d2])
        ts = [x["t"] for x in d1]
        self.assertEqual(ts, sorted(ts))
        self.assertTrue(all(b - a_ >= 15.0 for a_, b in zip(ts, ts[1:])))
        self.assertTrue(all(x["type"] == "pass" for x in d1))

    def test_only_passes_of_the_wanted_length(self):
        a = actions()
        for i, p in enumerate(a):
            if p["type"] == "pass":
                p["x2"] = p["x"] + (2.5 if i % 2 else 10.0)                  # alternately a short and a long pass
        short = ca.pick_detections(a, "pass", 50, 1.0, np.random.default_rng(0), dist=(2.0, 4.0))
        self.assertTrue(short and all(2.0 <= ca.pass_length(p) < 4.0 for p in short))
        self.assertEqual(len(ca.pick_detections(a, "pass", 50, 1.0, np.random.default_rng(0), dist=(4.0, 99.0))), len([p for p in a if p["type"] == "pass" and ca.pass_length(p) >= 4.0]))
        self.assertTrue(all(p["type"] == "shot" for p in ca.pick_detections(a, "shot", 5, 1.0, np.random.default_rng(0), dist=(2.0, 4.0))))   # shots are not filtered by length

    def test_fewer_candidates_than_asked(self):
        self.assertEqual(len(ca.pick_detections(actions(n_shot=3), "shot", 30, 5.0, np.random.default_rng(0))), 3)

    def test_windows_do_not_overlap_and_stay_in_play(self):
        t = np.arange(0, 300, 0.2)
        half = np.where(t < 150, 1, 2)
        in_play = np.ones(len(t), int)
        in_play[(t >= 40) & (t < 70)] = 0                          # a stoppage: no window may sit in it
        known = np.ones(len(t), bool)
        w = ca.pick_windows(t, half, known, in_play, 8, 10.0, np.random.default_rng(0))
        self.assertGreaterEqual(len(w), 5)
        self.assertTrue(all(b - a >= 10.0 for a, b in zip(w, w[1:])))
        for t0 in w:
            self.assertFalse(t0 + 10.0 > 40 and t0 < 70)
            self.assertEqual(half[np.searchsorted(t, t0)], half[np.searchsorted(t, t0 + 9.8)])   # inside one half

    def test_windows_need_a_known_ball(self):
        t = np.arange(0, 100, 0.2)
        self.assertEqual(ca.pick_windows(t, np.ones(len(t), int), np.zeros(len(t), bool), np.ones(len(t), int), 5, 10.0, np.random.default_rng(0)), [])


class Scoring(unittest.TestCase):
    def test_wilson(self):
        lo, hi = ca.wilson(8, 10)
        self.assertTrue(lo < 0.8 < hi)
        self.assertAlmostEqual(ca.wilson(10, 10)[1], 1.0, places=6)
        self.assertTrue(np.isnan(ca.wilson(0, 0)[0]))

    def test_precision_agreement_and_recall(self):
        key = [{"id": str(i), "type": "pass", "time_s": i, "program_outcome": "completed" if i < 6 else "lost", "inferred": 0} for i in range(1, 11)]
        key += [{"id": str(i), "type": "shot", "time_s": i, "program_outcome": "saved" if i < 14 else "off_target", "inferred": 0} for i in range(11, 15)]
        det = []
        for k in key:
            i = int(k["id"])
            if k["type"] == "pass":      # 8 of 10 real; of the real ones the user says completed for i < 5
                det.append({"id": k["id"], "real": "n" if i in (9, 10) else "y", "completed": "y" if i < 5 else "n", "on_target": ""})
            else:                        # 3 of 4 real; on target: only the first two
                det.append({"id": k["id"], "real": "n" if i == 14 else "y", "completed": "", "on_target": "y" if i < 13 else "n"})
        det.append({"id": "99", "real": "y", "completed": "", "on_target": ""})      # an id the program key does not have: ignored
        win_key = [{"id": "1", "t0": 0, "t1": 10, "program_passes": 4, "program_shots": 1},
                   {"id": "2", "t0": 20, "t1": 30, "program_passes": 2, "program_shots": 0},
                   {"id": "3", "t0": 40, "t1": 50, "program_passes": 9, "program_shots": 9}]
        wins = [{"id": "1", "passes": "6", "shots": "1"}, {"id": "2", "passes": "4", "shots": "2"}, {"id": "3", "passes": "", "shots": ""}]
        r = ca.score(det, key, wins, win_key)
        p, s = r["pass"], r["shot"]
        self.assertEqual((p["judged"], p["real"]), (10, 8))
        self.assertAlmostEqual(p["precision"], 0.8)
        # real passes 1..8: the program says completed for 1..5, the user for 1..4 -> agree in 7 of 8
        self.assertEqual(p["completed_agree"], (7, 8))
        self.assertEqual((s["judged"], s["real"]), (4, 3))
        # real shots 11, 12, 13: program on target for 11, 12, 13 (saved), user for 11, 12 -> agree in 2 of 3
        self.assertEqual(s["on_target_agree"], (2, 3))
        # windows 1 and 2 counted: user 10 passes, program 6 -> recall 6 * 0.8 / 10
        self.assertEqual((p["windows"], p["user_count"], p["program_count"]), (2, 10, 6))
        self.assertAlmostEqual(p["recall_estimate"], 0.48)
        self.assertAlmostEqual(s["recall_estimate"], min(1.0, 1 * 0.75 / 3))

    def test_no_recall_from_a_sample_limited_to_a_range_of_lengths(self):
        key = [{"id": str(i), "type": "pass", "time_s": i, "program_outcome": "completed", "inferred": 0} for i in range(1, 5)]
        det = [{"id": k["id"], "real": "y", "completed": "y", "on_target": ""} for k in key]
        wins = [{"id": "1", "passes": "5", "shots": "0"}]
        wkey = [{"id": "1", "t0": 0, "t1": 10, "program_passes": 4, "program_shots": 0}]
        self.assertIn("recall_estimate", ca.score(det, key, wins, wkey)["pass"])
        r = ca.score(det, key, wins, wkey, {"pass_dist": [2.0, 4.0], "only": "passes"})
        self.assertNotIn("recall_estimate", r["pass"])
        self.assertTrue(r["pass"]["recall_not_estimated"])
        self.assertEqual(r["pass"]["precision"], 1.0)                              # the precision of the range is still reported

    def test_nothing_judged_yet(self):
        key = [{"id": "1", "type": "pass", "time_s": 1, "program_outcome": "lost", "inferred": 0}]
        r = ca.score([{"id": "1", "real": "", "completed": "", "on_target": ""}], key, [], [])
        self.assertEqual(r["pass"]["judged"], 0)
        self.assertNotIn("precision", r["pass"])
        self.assertNotIn("recall_estimate", r["pass"])

    def test_no_recall_without_a_user_count(self):
        key = [{"id": "1", "type": "pass", "time_s": 1, "program_outcome": "lost", "inferred": 0}]
        det = [{"id": "1", "real": "y", "completed": "", "on_target": ""}]
        r = ca.score(det, key, [{"id": "1", "passes": "0", "shots": "0"}], [{"id": "1", "t0": 0, "t1": 10, "program_passes": 2, "program_shots": 0}])
        self.assertNotIn("recall_estimate", r["pass"])


class Refresh(unittest.TestCase):
    def test_counts_of_the_program_are_recomputed_for_the_same_windows(self):
        a = actions()                                                          # passes every 5 s from 10 s, shots every 20 s from 12 s
        key = [{"id": "1", "t0": "10.0", "t1": "20.0", "program_passes": "99", "program_shots": "99"},
               {"id": "2", "t0": "100.0", "t1": "110.0", "program_passes": "99", "program_shots": "99"}]
        new = ca.refresh_window_key(key, a)
        self.assertEqual([(k["program_passes"], k["program_shots"]) for k in new], [(2, 1), (2, 0)])
        self.assertEqual([k["id"] for k in new], ["1", "2"])
        self.assertEqual(key[0]["program_passes"], "99")                       # the input is not changed


class Files(unittest.TestCase):
    def test_load_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "actions.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["type", "time_s", "time", "half", "team", "x_from", "y_from", "x_to", "y_to", "outcome", "xg", "speed_ms"])
                w.writerow(["pass", 10.0, "00:10", 1, 0, 20, 10, 30, 12, "completed", "", ""])
                w.writerow(["shot", 20.0, "00:20", 1, 1, 45, 16, "", "", "goal (inferred from the goal)", 0.2, ""])
                w.writerow(["corner", 30.0, "00:30", 1, 1, "", "", "", "", "", "", ""])
            a = ca.load_actions(p)
        self.assertEqual([x["type"] for x in a], ["pass", "shot"])
        self.assertEqual((a[1]["outcome"], a[1]["inferred"]), ("goal", True))
        self.assertTrue(np.isnan(a[1]["x2"]))

    def test_render_clip_on_a_synthetic_video(self):
        import cv2
        calib = {"model": "homography", "H": [[56 / 400, 0, 0], [0, 32.4 / 200, 0], [0, 0, 1]]}      # pixels -> metres of one lens of 400 x 200
        with tempfile.TemporaryDirectory() as tmp:
            video, clip = Path(tmp) / "v.mp4", Path(tmp) / "c.mp4"
            wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (400, 400))
            for _ in range(20):
                wr.write(np.zeros((400, 400, 3), np.uint8))
            wr.release()
            cap = cv2.VideoCapture(str(video))
            frames = list(range(2, 12))
            balls = [(20.0 + i, 10.0) if i % 3 else None for i in range(10)]
            ok = ca.render_clip(cap, cv2, frames, balls, {"top": calib, "bottom": calib}, [25.0, 12.0],
                                [((20.0, 10.0), (0, 255, 255))], "#1 pass", clip, 5.0)
            off = ca.render_clip(cap, cv2, frames, balls, {"top": calib, "bottom": calib}, [500.0, 12.0], [], "x", Path(tmp) / "o.mp4", 5.0)
            cap.release()
            n = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertTrue(ok)
        self.assertEqual(n, 10)
        self.assertFalse(off)                                                  # a point outside both lenses: no clip

    def test_render_clip_survives_points_the_lens_model_cannot_project(self):
        import cv2
        import calibrate
        real = calibrate.pitch_to_pixels

        def nan_for_far_points(calib, pts):                                    # like a fisheye model for a point outside its range
            out = real(calib, pts)
            out[np.asarray(pts)[:, 0] > 40] = np.nan
            return out
        calib = {"model": "homography", "H": [[56 / 400, 0, 0], [0, 32.4 / 200, 0], [0, 0, 1]]}
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.mp4"
            wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (400, 400))
            for _ in range(10):
                wr.write(np.zeros((400, 400, 3), np.uint8))
            wr.release()
            cap = cv2.VideoCapture(str(video))
            calibrate.pitch_to_pixels = nan_for_far_points
            try:
                ok = ca.render_clip(cap, cv2, list(range(5)), [(50.0, 10.0)] * 5, {"top": calib, "bottom": calib}, [25.0, 12.0],
                                    [((50.0, 10.0), (0, 255, 255))], "x", Path(tmp) / "c.mp4", 5.0)
            finally:
                calibrate.pitch_to_pixels = real
            cap.release()
        self.assertTrue(ok)

    def test_render_clip_follows_the_video_when_the_analysis_uses_every_second_frame(self):
        import cv2
        calib = {"model": "homography", "H": [[56 / 400, 0, 0], [0, 32.4 / 200, 0], [0, 0, 1]]}
        with tempfile.TemporaryDirectory() as tmp:
            video, clip = Path(tmp) / "v.mp4", Path(tmp) / "c.mp4"
            wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (400, 400))
            for k in range(40):                                              # a white square moves 6 px per video frame
                img = np.zeros((400, 400, 3), np.uint8)
                cv2.rectangle(img, (20 + 6 * k, 90), (40 + 6 * k, 110), (255, 255, 255), -1)
                wr.write(img)
            wr.release()
            cap = cv2.VideoCapture(str(video))
            frames = list(range(4, 30, 2))                                   # the analysis: every second frame
            balls = [((30 + 6 * f) * 56 / 400, 100 * 32.4 / 200) for f in frames]
            ok = ca.render_clip(cap, cv2, frames, balls, {"top": calib, "bottom": calib}, [20.0, 16.0], [], "x", clip, 10.0)
            cap.release()
            out = cv2.VideoCapture(str(clip))
            n = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
            gaps = []
            for _ in range(n):
                good, img = out.read()
                white = np.argwhere((img > 200).all(axis=2)[60:])            # below the label
                yellow = np.argwhere((img[:, :, 0] < 90) & (img[:, :, 1] > 180) & (img[:, :, 2] > 180))
                if len(white) and len(yellow):
                    gaps.append(abs(white[:, 1].mean() - yellow[:, 1].mean()))
        self.assertTrue(ok)
        self.assertEqual(n, 28 - 4 + 1)                                      # all the video frames of the span, not one per row
        self.assertGreater(len(gaps), n // 2)
        self.assertLess(float(np.median(gaps)), 15.0)                        # the circle sits on the moving square


if __name__ == "__main__":
    unittest.main()
