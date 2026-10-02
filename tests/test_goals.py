import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers
from detect_goals import mark_starts, same_restart


def run_detect(folder, *extra):
    r = subprocess.run([sys.executable, str(helpers.ROOT / "detect_goals.py"), str(folder), "--pitch", str(Path(folder) / "pitch.json"),
                        "--match", str(Path(folder) / "match.json"), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return helpers.read_goals(folder)


def ev(half, t):
    return {"half": half, "t": float(t), "start": None}


class StartOfHalf(unittest.TestCase):
    def test_a_line_up_before_the_whistle_is_part_of_the_start(self):
        m = [ev(1, 20), ev(1, 55), ev(1, 280), ev(2, 1700), ev(2, 1740), ev(2, 1900)]
        mark_starts(m, {1: 0.0, 2: 1690.0}, 90.0, 60.0)
        self.assertEqual([e["start"] for e in m], [True, True, False, True, True, False])

    def test_the_first_event_far_from_the_start_is_a_goal_candidate(self):
        m = [ev(1, 400)]
        mark_starts(m, {1: 0.0}, 90.0, 60.0)
        self.assertFalse(m[0]["start"])


class SameRestart(unittest.TestCase):
    P = {"half": 1, "t": 1449.0, "cat": "goal"}

    def test_the_teams_waited_and_the_formation_formed_twice(self):
        self.assertTrue(same_restart(self.P, {"half": 1, "t": 1480.0}, None, 45.0))

    def test_a_ball_at_a_goal_in_between_makes_it_a_new_goal(self):
        self.assertFalse(same_restart(self.P, {"half": 1, "t": 1480.0}, 1470.0, 45.0))

    def test_later_or_in_another_half_is_not_the_same_restart(self):
        self.assertFalse(same_restart(self.P, {"half": 1, "t": 1524.0}, None, 45.0))
        self.assertFalse(same_restart(self.P, {"half": 2, "t": 1480.0}, None, 45.0))


class DetectGoalsOnASyntheticMatch(unittest.TestCase):
    GOALS = [(1, 100.0, 0), (1, 250.0, 1), (2, 150.0, 0)]

    def make(self, goals=None, **kw):
        d = tempfile.mkdtemp()
        truth = helpers.make_match(d, goals if goals is not None else self.GOALS, **kw)
        return d, truth

    def test_goals_scorers_and_times(self):
        d, truth = self.make()
        rows = run_detect(d)
        goals = [r for r in rows if r["category"] == "goal"]
        self.assertEqual(len([r for r in rows if r["category"] == "start of the half"]), 2)
        self.assertEqual(len(goals), 3)
        for r, (half, t, team) in zip(goals, truth["goals"]):
            self.assertEqual(int(r["half"]), half)
            self.assertEqual(int(r["scorer"]), team)
            self.assertLess(abs(float(r["goal_s"]) - t), 5.0)

    def test_a_waiting_formation_without_a_ball_at_a_goal_is_not_a_goal(self):
        d, _ = self.make(extra_formations=[(1, 330.0)])
        self.assertEqual(len([r for r in run_detect(d) if r["category"] == "goal"]), 3)

    def test_the_same_restart_seen_twice_is_one_goal(self):
        d, _ = self.make(extra_formations=[(1, 145.0, True)])        # 30 s after the kick-off at 115 s: the ball is on the spot again, no goal before it
        self.assertEqual(len([r for r in run_detect(d) if r["category"] == "goal"]), 3)

    def test_two_goals_30_seconds_apart_stay_two_goals(self):
        d, truth = self.make([(1, 100.0, 0), (1, 130.0, 1)])
        goals = [r for r in run_detect(d) if r["category"] == "goal"]
        self.assertEqual(len(goals), 2)
        self.assertEqual([int(r["scorer"]) for r in goals], [0, 1])

    def test_a_kickoff_without_the_ball_is_a_goal_only_with_raw_ball_candidates(self):
        d, truth = self.make(ball_at_centre={1: False})
        self.assertEqual(len([r for r in run_detect(d) if r["category"] == "goal"]), 2)        # the formation alone is not enough
        kick = next(t for h, t, k in truth["kickoffs"] if k == "goal" and abs(t - (truth["goals"][1][1] + 15)) < 1)
        with open(Path(d) / "detections.csv", "w") as f:
            f.write("frame,time_s,half,cam,class,conf,x1,y1,x2,y2,x_m,y_m\n")
            for k in range(12):                                                                # the detector saw the ball on the spot in 12 frames
                f.write(f"{k},{kick - 1.0 + k * 0.2:.3f},1,top,ball,0.8,0,0,1,1,28.1,16.3\n")
        rows = run_detect(d)
        goals = [r for r in rows if r["category"] == "goal"]
        self.assertEqual(len(goals), 3)
        self.assertIn("raw", goals[1]["how"])
        self.assertEqual(int(goals[1]["scorer"]), 1)

    def test_a_goal_before_the_whistle_of_the_half_is_listed_as_a_possible_goal(self):
        d, _ = self.make([(1, 100.0, 0)])
        # the ball sits in the left net near the end of the first half, no kick-off follows
        import csv
        with open(Path(d) / "ball.csv") as f:
            rows = list(csv.DictReader(f))
        t_end = max(float(r["time_s"]) for r in rows if r["half"] == "1")
        extra = [{"frame": 90000 + k, "time_s": round(t_end - 8 + k * 0.2, 3), "half": 1, "x_m": -1.3, "y_m": 16.2, "conf": 0.6, "source": "detected"} for k in range(20)]
        with open(Path(d) / "ball.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows + extra, key=lambda r: float(r["time_s"])))
        r = subprocess.run([sys.executable, str(helpers.ROOT / "detect_goals.py"), d, "--pitch", d + "/pitch.json", "--match", d + "/match.json"], capture_output=True, text=True)
        self.assertIn("POSSIBLE GOAL", r.stdout)


if __name__ == "__main__":
    unittest.main()
