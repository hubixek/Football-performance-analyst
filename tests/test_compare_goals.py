import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers
from compare_goals import max_matching

HEAD = ["half", "kickoff_s", "kickoff", "gap_s", "how", "category", "goal_s", "goal", "goal_time_is_rough", "side", "frames_near_goal", "scored_by_side", "scored_by_kicker", "scorer"]


def write_goals(folder, events):
    """events: (half, kick-off s, how, category, goal s or '', rough)."""
    with open(Path(folder) / "goals_auto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for h, k, how, cat, g, rough in events:
            w.writerow([h, k, "", "", how, cat, g, "", rough, "", 0, "", "", ""])


class Matching(unittest.TestCase):
    def test_a_maximum_matching_beats_the_greedy_one(self):
        # events at the minutes 18, 21, 22, 26; official goals 18, 22, 23, 25: only a one-to-one assignment matches all four within a minute
        m = max_matching([0, 1, 2, 3], [18, 22, 23, 25], [18, 21, 22, 26], 1)
        self.assertEqual(len(m), 4)

    def test_no_event_is_used_twice(self):
        m = max_matching([0, 1], [10], [10, 10], 1)
        self.assertEqual(len(m), 1)


class CompareGoalsCli(unittest.TestCase):
    def run_cli(self, events, minutes, *extra):
        d = tempfile.mkdtemp()
        write_goals(d, events)
        r = subprocess.run([sys.executable, str(helpers.ROOT / "compare_goals.py"), d, "--minutes", minutes, *extra], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        return r.stdout

    NIGHT = [(1, 63, "formation", "start of the half", "", 0), (1, 935, "ball", "goal", 924, 0), (1, 1110, "ball", "goal", 1091, 0), (1, 1176, "ball", "goal", 1156, 0),
             (1, 1449, "ball", "goal", 1434, 1), (2, 1698, "ball", "start of the half", "", 0), (2, 2268, "ball (raw)", "goal", 2251, 0), (2, 2553, "ball", "goal", 2531, 0)]

    def test_the_clock_of_the_first_half_runs_three_minutes_ahead_in_the_report_of_the_league(self):
        out = self.run_cli(self.NIGHT, "18,22,23,25,35,39")
        self.assertIn("found by the ball: 6 (100%)", out)
        self.assertIn("precision of the goals found by the ball: 6 of 6", out)
        line = next(l for l in out.splitlines() if l.startswith("clock shift"))
        shift = int(line.split("{1: ")[1].split(",")[0])
        self.assertTrue(150 <= shift <= 200, shift)

    def test_a_goal_found_only_by_the_formation_and_a_goal_without_any_event(self):
        events = [(1, 63, "ball", "start of the half", "", 0), (1, 300, "ball", "goal", 285, 0), (1, 600, "formation", "no", 585, 0)]
        out = self.run_cli(events, "5,10,24", "--no-align")
        self.assertIn("explained only by a formation-only kick-off: 1 [10]", out)
        self.assertIn("no event: 1 [24]", out)

    def test_a_false_goal_is_reported(self):
        events = [(1, 63, "ball", "start of the half", "", 0), (1, 300, "ball", "goal", 285, 0), (1, 900, "ball", "goal", 885, 0)]
        out = self.run_cli(events, "5", "--no-align")
        self.assertIn("goals of the program with no official goal", out)
        self.assertIn("precision of the goals found by the ball: 1 of 2", out)


if __name__ == "__main__":
    unittest.main()
