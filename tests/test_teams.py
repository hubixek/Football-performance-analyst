import csv
import tempfile
import unittest

import cv2
import numpy as np

import helpers
from dual_teams import appearance, clf_teams, find_kickoffs, fit_lda, gk_by_side, lda_score


class Appearance(unittest.TestCase):
    def test_descriptor_has_26_numbers_and_tells_a_black_from_an_orange_shirt(self):
        def player(bgr):
            img = np.full((120, 80, 3), (30, 80, 30), np.uint8)
            img[10:110, 20:60] = (40, 40, 40)
            img[20:45, 22:58] = bgr
            return appearance(img, (20.0, 10.0, 60.0, 110.0))
        black, orange = player((38, 31, 39)), player((23, 56, 105))
        self.assertEqual(black.shape, (26,))
        self.assertEqual(black.dtype, np.float32)
        self.assertNotEqual(int(np.argmax(black[12:24])), int(np.argmax(orange[12:24])))     # the hue histogram has a different peak
        self.assertGreater(orange[24], black[24])                                              # orange is more saturated

    def test_an_empty_box_gives_nothing(self):
        self.assertIsNone(appearance(np.zeros((50, 50, 3), np.uint8), (10.0, 10.0, 11.0, 11.0)))


class Discriminant(unittest.TestCase):
    def test_two_gaussian_classes_are_separated_on_unseen_data(self):
        rng = np.random.default_rng(0)
        mu = rng.normal(0, 1, 26)
        X0, X1 = rng.normal(mu, 1.5, (300, 26)), rng.normal(-mu, 1.5, (300, 26))
        model = fit_lda(np.vstack([X0, X1]), np.array([0] * 300 + [1] * 300))
        T0, T1 = rng.normal(mu, 1.5, (200, 26)), rng.normal(-mu, 1.5, (200, 26))
        acc = (np.mean(lda_score(model, T0) < 0) + np.mean(lda_score(model, T1) > 0)) / 2
        self.assertGreater(acc, 0.95)


class GoalkeepersBySide(unittest.TestCase):
    def test_the_goalkeeper_takes_the_team_that_defends_the_goal_he_stands_at(self):
        rows = [{"half": "1", "x_m": "3.0"}, {"half": "1", "x_m": "52.0"}, {"half": "2", "x_m": "3.0"}, {"half": "2", "x_m": "52.0"}]
        team = {0: 1, 1: 1, 2: 0, 3: 0}                       # wrong on purpose
        changed = gk_by_side(rows, team, {0, 1, 2, 3}, {1: 0, 2: 1}, 56.0)
        self.assertEqual(changed, 2)
        self.assertEqual(team, {0: 0, 1: 1, 2: 1, 3: 0})


class ClassifierFromKickoffs(unittest.TestCase):
    def test_teams_are_learnt_from_the_kickoffs_and_the_sides_change_after_half_time(self):
        d = tempfile.mkdtemp()
        helpers.make_match(d, [(1, 100.0, 0), (1, 250.0, 1), (2, 150.0, 0)])
        with open(d + "/teams.csv") as f:
            rows = list(csv.DictReader(f))
        events = find_kickoffs(rows, d + "/ball.csv", helpers.L, helpers.W)
        self.assertEqual(len(events), 5)
        self.assertTrue(all(e[3] == "ball" for e in events))
        rng = np.random.default_rng(1)
        mu = rng.normal(0, 1, 26)
        descs, colors = {}, {}
        for i, r in enumerate(rows):
            sign = 1.0 if int(r["team"]) == 1 else -1.0
            descs[i] = np.float32(rng.normal(sign * mu, 1.2))
            colors[i] = np.float32([40.0 if int(r["team"]) == 0 else 90.0, 128, 128])        # team 0 has the darker kit
        res = clf_teams(rows, colors, descs, events, list(range(len(rows))), set(), helpers.L)
        self.assertIsNotNone(res)
        team, score, centers, limits, info = res
        acc = np.mean([team[i] == int(rows[i]["team"]) for i in team])
        self.assertGreater(acc, 0.95)
        self.assertEqual(info["swapped"], {2: True})


if __name__ == "__main__":
    unittest.main()
