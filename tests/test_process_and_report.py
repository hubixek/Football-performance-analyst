import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers
import process_match as pm
from analyze_dual import goals_from_auto
from match_report import load_goals
from test_compare_goals import write_goals


class KitNames(unittest.TestCase):
    def test_clear_colours_are_named_and_unclear_ones_are_not(self):
        self.assertEqual(pm.kit_name((38, 31, 39)), "Czarni")
        self.assertEqual(pm.kit_name((23, 56, 105)), "Pomarańczowi")
        self.assertEqual(pm.kit_name((230, 230, 230)), "Biali")
        self.assertEqual(pm.kit_name((40, 40, 210)), "Czerwoni")
        self.assertIsNone(pm.kit_name((122, 120, 120)))       # a washed-out kit in daylight


class Runner(unittest.TestCase):
    def run_dry(self, name, calib):
        r = subprocess.run([sys.executable, str(helpers.ROOT / "process_match.py"), "nonexistent.mp4", "--name", name, "--calib-dir", calib, "--dry-run", "--out-dir", tempfile.mkdtemp()],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        return r.stdout

    def test_a_new_match_runs_all_steps_in_order(self):
        out = self.run_dry("t_new_match", tempfile.mkdtemp())
        steps = [l[4:] for l in out.splitlines() if l.startswith("=== ")]
        self.assertEqual([s.split(" ")[0] for s in steps], ["calibration", "half", "analysis", "statistics", "goals", "statistics", "passes", "report"])

    def test_an_interrupted_run_continues_with_the_half_times(self):
        """A provisional config of the whole recording (left by an interrupted run) is not a result: the half times are made, the finished calibration is kept."""
        cfg = helpers.ROOT / "matches" / "t_interrupted.json"
        calib = tempfile.mkdtemp()
        for c in ("top", "bottom"):
            (Path(calib) / f"t_interrupted_{c}.json").write_text("{}")
        cfg.parent.mkdir(exist_ok=True)
        try:
            cfg.write_text(json.dumps({"name": "t_interrupted", "provisional": True, "halves": [{"start_s": 0.0, "end_s": 600.0}]}))
            out = self.run_dry("t_interrupted", calib)
            steps = [l[4:].split(" ")[0] for l in out.splitlines() if l.startswith("=== ")]
            self.assertEqual(steps[:2], ["half", "analysis"])                       # no calibration, the half times first
            cfg.write_text(json.dumps({"name": "t_interrupted", "halves": [{"start_s": 30.0, "end_s": 230.0}, {"start_s": 330.0, "end_s": 560.0}]}))
            out = self.run_dry("t_interrupted", calib)
            self.assertIn("exists", out)                                            # a finished config is kept
            self.assertNotIn("=== half times", out)
        finally:
            cfg.unlink(missing_ok=True)


class ReportGoals(unittest.TestCase):
    def test_minutes_are_counted_from_the_first_kickoff_of_every_half(self):
        d = tempfile.mkdtemp()
        # half times found automatically begin before the whistles (0:00 and 29:10; the whistles at 0:52 and 29:38)
        write_goals(d, [(1, 52, "ball", "start of the half", "", 0), (1, 282, "ball", "goal", 266, 0), (2, 1778, "ball", "start of the half", "", 0), (2, 2241, "ball", "goal", 2226, 0)])
        with open(Path(d) / "goals_auto.csv") as f:
            rows = list(csv.DictReader(f))
        rows[1]["side"], rows[1]["scorer"] = "left", "0"
        rows[3]["side"], rows[3]["scorer"] = "left", "1"
        with open(Path(d) / "goals_auto.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (Path(d) / "match.json").write_text(json.dumps({"halves": [{"start_s": 0.0, "end_s": 1558.0}, {"start_s": 1750.0, "end_s": 3278.0}]}))
        goals = load_goals(Path(d), str(Path(d) / "match.json"))
        self.assertEqual([(g["minute"], g["team"]) for g in goals], [(4, 0), (33, 1)])

    def test_goals_for_the_restarts_are_the_goal_rows_only(self):
        d = tempfile.mkdtemp()
        write_goals(d, [(1, 63, "ball", "start of the half", "", 0), (1, 935, "ball", "goal", 924, 0), (1, 1033, "formation", "no", "", 0), (1, 1110, "ball", "goal", 1091, 0)])
        self.assertEqual(goals_from_auto(Path(d) / "goals_auto.csv"), "15:24,18:11")


if __name__ == "__main__":
    unittest.main()
