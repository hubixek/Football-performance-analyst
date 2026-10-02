import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import helpers  # noqa: F401  (the repository on the path)
import match_actions
from helpers import L, W

FPS = 15.0
DT = 1 / FPS


def write_analysis(folder, ball, players):
    """possession.csv, tracks.csv, stats.json and pitch.json of a half.

    ball: per frame (x, y) or None (the ball is not seen); players: {name: (team, x, y)} standing still or {name: (team, [(x, y) per frame])}.
    The tracks keep their own times (shifted by 12 ms and jittered) like the real ones; team 0 attacks to the right."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(folder / "possession.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "ball_x", "ball_y", "controller_team", "possession_team", "ball_in_play", "ball_state"])
        for i, b in enumerate(ball):
            w.writerow([i, round(100 + i * DT, 3), 1] + ([b[0], b[1], "", "", 1, "play"] if b else ["", "", "", "", "", ""]))
    with open(folder / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["half", "time_s", "player", "team", "x_m", "y_m", "speed_ms", "source"])
        for i in range(len(ball)):
            for name, (team, *pos) in players.items():
                x, y = pos[0][i] if isinstance(pos[0], list) else pos
                w.writerow([1, round(100 + i * DT + 0.012 + float(rng.uniform(-0.01, 0.01)), 3), name, team, x, y, 0.0, "detected"])
    (folder / "stats.json").write_text(json.dumps({"attacks_right": {"half_1": 0}}))
    (folder / "pitch.json").write_text(json.dumps({"length_m": L, "width_m": W}))


def line(p, q, n):
    return [(p[0] + (q[0] - p[0]) * k / (n + 1), p[1] + (q[1] - p[1]) * k / (n + 1)) for k in range(1, n + 1)]


def run_actions(folder, *extra):
    argv = ["match_actions.py", str(folder), "--pitch", str(Path(folder) / "pitch.json"), *extra]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        match_actions.main()
    with open(Path(folder) / "actions.csv") as f:
        return [r for r in csv.DictReader(f) if r["type"] == "pass"]


A, B, D = (10.0, 16.0), (20.0, 16.0), (40.0, 16.0)
PLAYERS = {"T0-H1-P1": (0, *A), "T0-H1-P2": (0, *B), "T1-H1-P1": (1, *D)}


def long_passes_ball():
    """A has the ball, it flies unseen to B (a completed pass), B touches it, loses sight of it for a moment and touches it again (a dribble),
    then it flies unseen to D of the other team (a lost pass)."""
    ball = [A] * 10 + [None] * 10 + [B] * 5 + [None] * 3 + [B] * 7 + [None] * 10 + [D] * 10
    return ball


class PlayerPasses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_completed_lost_and_no_pass_for_a_dribble(self):
        write_analysis(self.folder, long_passes_ball(), PLAYERS)
        p = run_actions(self.folder, "--passes", "players")
        self.assertEqual([(r["team"], r["outcome"]) for r in p], [("0", "completed"), ("0", "lost")])
        self.assertAlmostEqual(float(p[0]["x_from"]), 10.0, places=1)
        self.assertAlmostEqual(float(p[0]["x_to"]), 20.0, places=1)
        self.assertAlmostEqual(float(p[0]["time_s"]), 100 + 9 * DT, places=1)

    def test_a_short_pass_seen_all_the_way(self):
        near = (16.0, 16.0)                                                    # 6 m, the ball is seen on the way
        ball = [A] * 10 + line(A, near, 4) + [near] * 10
        write_analysis(self.folder, ball, {"T0-H1-P1": (0, *A), "T0-H1-P2": (0, *near)})
        p = run_actions(self.folder, "--passes", "players")
        self.assertEqual([(r["team"], r["outcome"]) for r in p], [("0", "completed")])

    def test_a_player_carrying_the_ball_is_not_passing_to_himself(self):
        """One player sprints 6 m/s with the ball, the ball is out of sight for 1.5 s: two touches 9 m apart by the SAME player, which the
        distance and speed limits alone would take for a pass."""
        path = [(20.0 + 0.4 * i, 16.0) for i in range(40)]
        ball = [None if 10 <= i <= 31 else path[i] for i in range(40)]
        write_analysis(self.folder, ball, {"T0-H1-P2": (0, path)})
        self.assertEqual(run_actions(self.folder, "--passes", "players"), [])

    def test_the_other_team_receiving_a_pass_that_is_too_short_is_not_a_pass(self):
        near = (12.0, 16.0)                                                    # 2 m: under --pass-min-d
        ball = [A] * 10 + [near] * 10
        write_analysis(self.folder, ball, {"T0-H1-P1": (0, *A), "T1-H1-P1": (1, *near)})
        self.assertEqual(run_actions(self.folder, "--passes", "players"), [])

    def test_ball_leaving_the_pitch_is_a_lost_pass(self):
        out = (L + 3.0, 16.0)
        ball = [A] * 10 + line(A, out, 4) + [out] * 40
        write_analysis(self.folder, ball, {"T0-H1-P1": (0, *A)})
        with open(self.folder / "possession.csv") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            if i >= 14:
                r["ball_state"], r["ball_in_play"] = "out", "0"
        with open(self.folder / "possession.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        p = run_actions(self.folder, "--passes", "players")
        self.assertEqual([(r["team"], r["outcome"]) for r in p], [("0", "lost")])

    def test_players_method_needs_the_tracks(self):
        write_analysis(self.folder, long_passes_ball(), PLAYERS)
        (self.folder / "tracks.csv").unlink()
        with self.assertRaises(SystemExit) as cm:
            run_actions(self.folder, "--passes", "players")
        self.assertIn("tracks.csv", str(cm.exception))

    def test_default_method_does_not_read_the_tracks(self):
        write_analysis(self.folder, long_passes_ball(), PLAYERS)
        (self.folder / "tracks.csv").unlink()
        run_actions(self.folder)                                               # the default 'ball' method is unchanged and needs no tracks


if __name__ == "__main__":
    unittest.main()
