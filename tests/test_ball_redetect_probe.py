import unittest

import numpy as np

import helpers  # noqa: F401  (the repository on the path)
import ball_redetect_probe as br

T = np.arange(0, 3.0, 1 / 15)


def path_points(n):
    return [(10.0 + 0.5 * i, 16.0) for i in range(n)]


class Plausible(unittest.TestCase):
    def test_a_spare_ball_beside_the_pitch_is_not_a_candidate(self):
        self.assertTrue(br.on_the_pitch((28.0, 16.0, 0.5), 56.0, 32.4))
        self.assertTrue(br.on_the_pitch((56.0 + 1.2, 16.2, 0.5), 56.0, 32.4))     # the ball in the net
        self.assertFalse(br.on_the_pitch((36.0, 34.0, 0.9), 56.0, 32.4))          # a spare ball by the bench
        self.assertFalse(br.on_the_pitch((-1.5, 4.0, 0.9), 56.0, 32.4))           # behind the goal line but not in the goal mouth


class Gaps(unittest.TestCase):
    def setUp(self):
        self.real = np.array(path_points(len(T)))
        self.mask = np.zeros(len(T), bool)
        self.mask[20:30] = True
        self.anchors = self.real.copy()
        self.anchors[self.mask] = np.nan

    def test_the_gap_is_filled_with_the_detections_not_with_a_straight_line(self):
        curve = self.real.copy()
        i = np.arange(len(T))
        curve[:, 0] = np.where(i <= 25, 10.0 + 0.5 * i, 22.5 - 0.5 * (i - 25))      # a ball that turns back in the middle of the gap
        anchors = curve.copy()
        anchors[self.mask] = np.nan
        cands = [[(curve[k, 0], curve[k, 1], 0.2), (50.0, 5.0, 0.15)] for k in range(len(T))]     # the real ball and a static decoy
        filled = br.fill_gaps(cands, anchors, T)
        s = br.holdout_scores(curve, self.mask, filled)
        lin = br.holdout_scores(curve, self.mask, br.linear_fill(anchors, T))
        self.assertEqual((s["hidden"], s["filled"], s["right"]), (10, 10, 10))
        self.assertLess(lin["right"], 10)

    def test_a_gap_without_candidates_stays_empty_in_the_detector_but_not_in_the_interpolation(self):
        filled = br.fill_gaps([[] for _ in T], self.anchors, T)
        self.assertEqual(br.holdout_scores(self.real, self.mask, filled)["filled"], 0)
        self.assertEqual(br.holdout_scores(self.real, self.mask, br.linear_fill(self.anchors, T))["filled"], 10)

    def test_anchors_are_kept_and_a_far_decoy_cannot_pull_the_path_away(self):
        cands = [[(40.0, 30.0, 0.9)] if self.mask[i] else [] for i in range(len(T))]   # confident decoy far from both anchors
        filled = br.fill_gaps(cands, self.anchors, T)
        self.assertTrue(np.allclose(filled[~self.mask], self.real[~self.mask]))
        self.assertTrue(np.isnan(filled[self.mask]).all())

    def test_a_chain_that_cannot_reach_the_next_anchor_is_not_taken(self):
        cands = [[(self.real[i, 0] + 3.0, 16.0, 0.9)] if self.mask[i] else [] for i in range(len(T))]
        cands[29] = [(self.real[29, 0] + 40.0, 16.0, 0.9)]                          # the chain ends where the next anchor cannot be reached
        filled = br.fill_gaps(cands, self.anchors, T, vmax=20.0)
        self.assertTrue(np.isnan(filled[29, 0]))

    def test_the_edges_of_a_window_are_not_filled(self):
        anchors = self.real.copy()
        anchors[:5] = np.nan
        anchors[-5:] = np.nan
        cands = [[(self.real[i, 0], 16.0, 0.9)] for i in range(len(T))]
        filled = br.fill_gaps(cands, anchors, T)
        self.assertTrue(np.isnan(filled[:5, 0]).all() and np.isnan(filled[-5:, 0]).all())

    def test_linear_fill_does_not_extrapolate(self):
        anchors = np.full((6, 2), np.nan)
        anchors[2], anchors[4] = (2.0, 0.0), (4.0, 0.0)
        out = br.linear_fill(anchors, np.arange(6.0))
        self.assertTrue(np.isnan(out[[0, 1, 5]]).all())
        self.assertEqual(out[3, 0], 3.0)

    def test_scores_count_only_frames_with_a_true_position(self):
        truth = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [4.0, 4.0]])
        filled = np.array([[1.2, 1.0], [2.0, 2.0], [9.0, 9.0], [np.nan, np.nan]])
        s = br.holdout_scores(truth, np.ones(4, bool), filled)
        self.assertEqual((s["hidden"], s["filled"], s["right"]), (3, 2, 1))


if __name__ == "__main__":
    unittest.main()
