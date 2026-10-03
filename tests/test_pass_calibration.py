import unittest

import helpers  # noqa: F401  (the repository on the path)
import match_report
import pass_calibration as pc
import quality


class Estimate(unittest.TestCase):
    def test_the_table_is_the_judged_sample(self):
        self.assertEqual(sum(pc.CALIBRATION["completed"].values()), 54)
        self.assertEqual(sum(pc.CALIBRATION["lost"].values()), 35)
        self.assertEqual(pc.judged(), 89)

    def test_the_estimate_is_far_above_the_raw_share_when_lost_is_unreliable(self):
        """mecz4_3dual: the program says 175 completed and 161 lost (52%); the verdict 'lost' is mostly wrong, so the real accuracy is much higher."""
        point, lo, hi = pc.estimate(175, 161)
        self.assertAlmostEqual(point, 86.5, delta=0.6)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        self.assertGreater(point - 100 * 175 / 336, 30)

    def test_the_formula_by_hand(self):
        # 100 completed, 0 lost: true completed 100 x 38/54, true lost 100 x 2/54 -> 38/40
        self.assertAlmostEqual(pc.estimate(100, 0)[0], 95.0, places=6)
        # 0 completed, 100 lost: 15 / (15 + 5)
        self.assertAlmostEqual(pc.estimate(0, 100)[0], 75.0, places=6)

    def test_more_lost_verdicts_never_raise_the_estimate_above_what_they_imply(self):
        a = pc.estimate(100, 0)[0]
        b = pc.estimate(100, 100)[0]
        c = pc.estimate(0, 100)[0]
        self.assertGreater(a, b)
        self.assertGreater(b, c)

    def test_no_pass_no_estimate(self):
        self.assertIsNone(pc.estimate(0, 0))

    def test_the_interval_is_reproducible_and_narrower_with_a_better_table(self):
        self.assertEqual(pc.estimate(175, 161), pc.estimate(175, 161))
        big = {"completed": {k: 10 * v for k, v in pc.CALIBRATION["completed"].items()}, "lost": {k: 10 * v for k, v in pc.CALIBRATION["lost"].items()}, "matches": []}
        s, b = pc.estimate(175, 161), pc.estimate(175, 161, big)
        self.assertLess(b[2] - b[1], s[2] - s[1])


class Report(unittest.TestCase):
    def data(self, ok0=110, att0=184, ok1=65, att1=152):
        actions = {"scopes": {"match": {"team_0": {"passes_attempted": att0, "passes_completed": ok0, "pass_accuracy_percent": round(100 * ok0 / att0, 1) if att0 else None},
                                       "team_1": {"passes_attempted": att1, "passes_completed": ok1, "pass_accuracy_percent": round(100 * ok1 / att1, 1) if att1 else None}}}}
        return match_report.collect({}, {}, {}, actions, "match")

    def test_collect_adds_the_estimate_and_keeps_the_raw_counts(self):
        d = self.data()
        self.assertEqual((d[0]["pass_ok"], d[0]["pass_att"]), (110, 184))
        self.assertAlmostEqual(d[0]["pass_acc"], 59.8, places=1)
        self.assertGreater(d[0]["pass_acc_est"], 80)
        lo, hi = d[0]["pass_acc_ci"]
        self.assertTrue(lo < d[0]["pass_acc_est"] < hi)

    def test_the_pane_shows_the_estimate_with_the_raw_counts_and_a_note(self):
        d = self.data()
        for lang in ("pl", "en"):
            html = match_report.pane("match", d, match_report.TEXT[lang], ["#00f", "#f00"], hidden=False, vis=84.0)
            self.assertIn("≈", html)
            self.assertIn("(110/184)", html)
            self.assertIn(f"{d[0]['pass_acc_ci'][0]:.0f}-{d[0]['pass_acc_ci'][1]:.0f}%", html)
            self.assertNotIn(f">{d[0]['pass_acc']:.0f}% (110/184)", html)           # the raw share is no longer the headline number

    def test_a_scope_without_passes_has_no_pass_row(self):
        d = self.data(0, 0, 0, 0)
        self.assertIsNone(d[0]["pass_acc_est"])
        html = match_report.pane("half_1", d, match_report.TEXT["pl"], ["#00f", "#f00"], hidden=True, vis=84.0)
        self.assertNotIn("≈", html)

    def test_the_panel_of_what_the_report_shows_mentions_the_correction(self):
        html = match_report.shows_panel({"measured": quality.MEASURED}, match_report.TEXT["pl"])
        self.assertIn("skorygowanym", html)


if __name__ == "__main__":
    unittest.main()
