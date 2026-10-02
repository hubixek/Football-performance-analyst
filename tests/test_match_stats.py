import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401  (the repository on the path)
import match_stats
from helpers import L, W

DT = 0.2


def run(team, n, start, x=20.0, half=1):
    """n possession samples of one team (team None = nobody has the ball) every DT seconds from start."""
    return [{"frame": 0, "t": round(start + i * DT, 3), "half": half, "x": x if team is not None else None,
             "y": W / 2 if team is not None else None, "team": team} for i in range(n)]


def write_match(folder, poss):
    """teams.csv with a static 6v6 (team 0 on the left in half 1, on the right in half 2, a goalkeeper each and a referee) and possession.csv."""
    xs0 = [10, 12, 14, 16, 18, 20]
    xs1 = [36, 38, 40, 42, 44, 46]
    ys = [4, 8, 12, 16, 20, 24]
    frames = sorted({0} | {int(round(p["t"] / DT)) for p in poss})
    rows = []
    for fr in frames:
        for half in (1, 2):
            f = fr + (1000 if half == 2 else 0)
            for team, xs in ((0, xs0), (1, xs1)):
                for x, y in zip(xs, ys):
                    xx = x if half == 1 else L - x
                    rows.append((f, fr * DT, half, "player", xx, y, team, 0))
                rows.append((f, fr * DT, half, "player", 3.0 if (half == 1) == (team == 0) else L - 3.0, W / 2, team, 1))
            rows.append((f, fr * DT, half, "player", 28.0, 1.0, "", 0))
            rows.append((f, fr * DT, half, "ball", 28.0, 1.0, 0, 0))
    with open(Path(folder) / "teams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "class", "x_m", "y_m", "team", "gk"])
        w.writerows(rows)
    with open(Path(folder) / "possession.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "ball_x", "ball_y", "possession_team"])
        for p in poss:
            w.writerow([int(round(p["t"] / DT)), p["t"], p["half"], "" if p["x"] is None else p["x"],
                        "" if p["y"] is None else p["y"], "" if p["team"] is None else p["team"]])


def scenario():
    """Half 1: team 0 holds the ball 5 s in its own third, team 1 wins it at x=20 (its opponent's half), a gap of 1 s without possession, team 0 wins it back at x=50 and keeps it 11.8 s."""
    return run(0, 26, 0.0, x=10.0) + run(1, 20, 5.2, x=20.0) + run(None, 5, 9.2) + run(0, 60, 10.2, x=50.0)


class Sequences(unittest.TestCase):
    def test_runs_and_durations(self):
        s = match_stats.sequences(scenario(), 1.0)
        self.assertEqual([q["team"] for q in s], [0, 1, 0])
        self.assertEqual([round(q["t1"] - q["t0"], 1) for q in s], [5.0, 3.8, 11.8])

    def test_short_run_is_dropped(self):
        poss = run(0, 26, 0.0) + run(1, 2, 5.2) + run(None, 5, 5.6) + run(0, 26, 6.6)
        s = match_stats.sequences(poss, 1.0)
        self.assertEqual([q["team"] for q in s], [0, 0])

    def test_runs_of_one_team_around_a_dropped_short_run_are_merged(self):
        poss = run(0, 26, 0.0) + run(1, 2, 5.2) + run(0, 26, 5.6)
        s = match_stats.sequences(poss, 1.0)
        self.assertEqual(len(s), 1)
        self.assertEqual((s[0]["t0"], s[0]["t1"]), (0.0, 10.6))

    def test_no_possession_gap_separates_runs_of_the_same_team(self):
        poss = run(0, 26, 0.0) + run(None, 5, 5.2) + run(0, 26, 6.2)
        self.assertEqual(len(match_stats.sequences(poss, 1.0)), 2)

    def test_half_change_separates_runs(self):
        poss = run(0, 26, 0.0, half=1) + run(0, 26, 5.2, half=2)
        s = match_stats.sequences(poss, 1.0)
        self.assertEqual([q["half"] for q in s], [1, 2])

    def test_empty(self):
        self.assertEqual(match_stats.sequences([], 1.0), [])


class Direction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        write_match(self.folder, scenario())

    def test_load_keeps_only_players_with_a_team(self):
        players, info, poss = match_stats.load(self.folder)
        pl = players[0]
        self.assertEqual(len(pl), 14)                      # 12 field players + 2 goalkeepers, no referee, no ball
        self.assertEqual(sum(1 for p in pl if p[3]), 2)
        self.assertEqual(info[0], (0.0, 1))
        self.assertEqual(len(poss), len(scenario()))
        self.assertIsNone(next(p for p in poss if p["team"] is None)["x"])

    def test_attack_direction_per_half_ignores_goalkeepers(self):
        players, info, _ = match_stats.load(self.folder)
        d = match_stats.attack_right(players, info)
        self.assertEqual(d, {(1, 0): True, (1, 1): False, (2, 0): False, (2, 1): True})

    def test_normalisation(self):
        d = {(1, 0): True, (1, 1): False}
        self.assertEqual(match_stats.norm_x(10, 1, 0, d, L), 10)
        self.assertEqual(match_stats.norm_x(10, 1, 1, d, L), L - 10)
        self.assertEqual(match_stats.norm_xy(10, 5, 1, 1, d, L, W), (L - 10, W - 5))


class MainOutput(unittest.TestCase):
    def test_stats_json_on_the_synthetic_match(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_match(folder, scenario())
            pitch = folder / "pitch.json"
            pitch.write_text(json.dumps({"length_m": L, "width_m": W}))
            with mock.patch.object(sys, "argv", ["match_stats.py", str(folder), "--pitch", str(pitch)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                match_stats.main()
            out = json.loads((folder / "stats.json").read_text())
            for name in ("stats.png", "possession_halves.png"):
                self.assertTrue((folder / name).exists())
        self.assertEqual(out["attacks_right"], {"half_1": 0})
        t0, t1 = out["match"]["team_0"], out["match"]["team_1"]
        self.assertEqual(out["half_1"], out["match"])
        # team 0: 26 samples at x=10 (own third) and 60 at x=50 (final third)
        self.assertEqual(t0["possession_by_third_percent"], {"own": 30.2, "middle": 0.0, "final": 69.8})
        self.assertEqual(t0["possession_in_opponent_half_percent"], 69.8)
        # team 1 attacks left: x=20 is 36 m from its own goal = middle third, opponent's half
        self.assertEqual(t1["possession_by_third_percent"], {"own": 0.0, "middle": 100.0, "final": 0.0})
        self.assertEqual(t1["possession_in_opponent_half_percent"], 100.0)
        self.assertEqual((t0["possessions"], t0["possession_avg_s"], t0["possession_longest_s"], t0["possessions_over_10s"]),
                         (2, 8.4, 11.8, 1))
        self.assertEqual((t1["possessions"], t1["possession_avg_s"], t1["possessions_over_10s"]), (1, 3.8, 0))
        # shape without the goalkeeper: team 0 on x 10..20, team 1 on x 36..46 attacking left
        for t in (t0, t1):
            self.assertEqual((t["average_position_m"], t["team_length_m"], t["team_width_m"]), (15.0, 10.0, 20.0))
        # each team won the ball once, both in the opponent's half
        self.assertEqual((t0["recoveries"], t1["recoveries"]), (1, 1))
        self.assertEqual((t0["recoveries_in_opponent_half"], t1["recoveries_in_opponent_half"]), (1, 1))
        self.assertEqual((t0["ball_losses"], t1["ball_losses"]), (1, 1))

    def test_recovery_after_a_long_pause_is_not_counted(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        poss = run(0, 26, 0.0, x=10.0) + run(None, 25, 5.2) + run(1, 26, 10.4, x=20.0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_match(folder, poss)
            pitch = folder / "pitch.json"
            pitch.write_text(json.dumps({"length_m": L, "width_m": W}))
            with mock.patch.object(sys, "argv", ["match_stats.py", str(folder), "--pitch", str(pitch)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                match_stats.main()
            out = json.loads((folder / "stats.json").read_text())
        self.assertEqual(out["match"]["team_1"]["recoveries"], 0)
        self.assertEqual(out["match"]["team_0"]["ball_losses"], 0)


if __name__ == "__main__":
    unittest.main()
