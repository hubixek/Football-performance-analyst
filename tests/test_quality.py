import csv
import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  (the repository on the path)
import quality


def write_calib(folder, name, explained=0.7, covered=0.9, bottom=None):
    for half in ("top", "bottom"):
        d = {"model": "fisheye"}
        e, c = (explained, covered) if half == "top" or bottom is None else bottom
        if e is not None:
            d.update({"explained_lines": e, "covered_lines": c})
        (Path(folder) / f"{name}_{half}.json").write_text(json.dumps(d))


def write_possession(folder, halves=(1, 2), known=1.0):
    with open(Path(folder) / "possession.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "half", "ball_x", "ball_y", "controller_team", "possession_team", "ball_in_play", "ball_state"])
        i = 0
        for h in halves:
            for k in range(20):
                has = k < int(20 * known)
                w.writerow([i, i / 15, h, 10 if has else "", 10 if has else "", "", "", 1, "play"])
                i += 1


def write_goals(folder, sides):
    """sides: for every goal the value of scored_by_side ('' = the ball was not seen at a goal)."""
    with open(Path(folder) / "goals_auto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["half", "kickoff_s", "category", "goal_s", "scored_by_side", "scorer"])
        w.writerow([1, 10, "start of the half", "", "", ""])
        for k, s in enumerate(sides):
            w.writerow([1, 100 + 100 * k, "goal", 90 + 100 * k, s, 0])


def by_id(result):
    return {c["id"]: c for c in result["checks"]}


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        write_possession(self.folder)
        write_goals(self.folder, ["1", "0"])

    def test_a_good_analysis_is_ok(self):
        write_calib(self.folder, "m")
        r = quality.assess(self.folder, "m", self.folder)
        self.assertEqual(r["level"], "ok")
        self.assertEqual(by_id(r)["calibration"]["status"], "ok")

    def test_a_poor_calibration_is_bad_even_if_one_lens_is_fine(self):
        write_calib(self.folder, "m", 0.7, 0.9, bottom=(0.45, 0.9))               # explained < 50% on one lens
        r = quality.assess(self.folder, "m", self.folder)
        self.assertEqual((r["level"], by_id(r)["calibration"]["status"]), ("bad", "bad"))
        write_calib(self.folder, "m", 0.7, 0.9, bottom=(0.7, 0.55))               # covered < 60%
        self.assertEqual(quality.assess(self.folder, "m", self.folder)["level"], "bad")

    def test_the_limits_are_the_ones_of_the_project(self):
        write_calib(self.folder, "m", 0.50, 0.60)                                  # exactly at the limits: still good
        self.assertEqual(quality.assess(self.folder, "m", self.folder)["level"], "ok")

    def test_a_calibration_made_by_hand_has_no_numbers_and_is_not_judged(self):
        write_calib(self.folder, "m", None, None)
        r = quality.assess(self.folder, "m", self.folder)
        self.assertEqual(by_id(r)["calibration"]["status"], "info")
        self.assertEqual(r["level"], "ok")

    def test_one_half_is_a_warning(self):
        write_possession(self.folder, halves=(1,))
        r = quality.assess(self.folder)
        self.assertEqual((r["level"], by_id(r)["halves"]["status"]), ("warn", "warn"))

    def test_a_ball_seen_rarely_is_a_warning(self):
        write_possession(self.folder, known=0.5)
        self.assertEqual(by_id(quality.assess(self.folder))["ball"]["status"], "warn")

    def test_a_guessed_scorer_makes_the_score_approximate(self):
        write_goals(self.folder, ["1", "", "0", ""])
        r = quality.assess(self.folder)
        self.assertEqual(r["level"], "warn")
        self.assertIn("goals_guess", by_id(r))
        self.assertIn("2 z 4", by_id(r)["goals_guess"]["pl"])

    def test_no_goal_found_is_a_warning(self):
        write_goals(self.folder, [])
        self.assertEqual(by_id(quality.assess(self.folder))["goals"]["status"], "warn")

    def test_the_worst_level_wins(self):
        write_calib(self.folder, "m", 0.4, 0.9)
        write_goals(self.folder, [""])
        self.assertEqual(quality.assess(self.folder, "m", self.folder)["level"], "bad")

    def test_every_check_has_both_languages(self):
        write_calib(self.folder, "m")
        for c in quality.assess(self.folder, "m", self.folder)["checks"]:
            self.assertTrue(c["pl"] and c["en"])


if __name__ == "__main__":
    unittest.main()


class Report(unittest.TestCase):
    def setUp(self):
        import match_report
        self.mr = match_report
        self.qual = {"level": "warn", "measured": quality.MEASURED, "checks": [
            {"id": "calibration", "status": "ok", "pl": "ok-pl", "en": "ok-en"},
            {"id": "goals_guess", "status": "warn", "pl": "zgadnięty <b>strzelec</b>", "en": "guessed"},
            {"id": "info", "status": "info", "pl": "uwaga-pl", "en": "note-en"}]}

    def test_the_banner_lists_what_is_not_ok_and_escapes_it(self):
        html = self.mr.quality_banner(self.qual, self.mr.TEXT["pl"], "pl")
        self.assertIn('class="quality warn"', html)
        self.assertIn("Analiza z zastrzeżeniami", html)
        self.assertIn("zgadnięty &lt;b&gt;strzelec&lt;/b&gt;", html)
        self.assertIn("uwaga-pl", html)                                         # the standing note is always there
        self.assertNotIn("ok-pl", html)                                         # what is fine is not listed

    def test_every_level_has_a_title_in_both_languages(self):
        for lang in ("pl", "en"):
            for level in ("ok", "warn", "bad"):
                self.assertTrue(self.mr.TEXT[lang]["quality_title"][level])

    def test_the_panel_states_the_measured_numbers_and_what_is_not_available(self):
        m = quality.MEASURED
        for lang in ("pl", "en"):
            html = self.mr.shows_panel(self.qual, self.mr.TEXT[lang])
            self.assertEqual(html.count("<li>"), 6)
            self.assertIn(str(m["passes_precision"]) + "%", html)
            self.assertIn(str(m["shots_precision"]) + "%", html)
            self.assertIn(f"{m['goals_found']}", html)
