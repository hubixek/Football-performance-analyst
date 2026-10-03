#!/usr/bin/env python3
"""Pass accuracy with the known error of the verdict "completed / lost" corrected.

match_actions.py calls a pass completed when the next touch is of the same team and lost otherwise. Judged by eye (89 passes of the current method on two
matches, see CALIBRATION) the verdict "completed" is right for 70% of the passes it names (4% were real passes that were lost, 26% were no pass at all), but the
verdict "lost" is right for only 14% of the passes it names (43% were real passes that were completed, 43% were no pass). The raw share of "completed"
(52-57% in the matches judged) therefore understates the accuracy of the passes the program finds a great deal.

This module turns the counts of the two verdicts into an estimate of the share of real passes that were completed:
  true completed = completed x P(real and completed | said completed) + lost x P(real and completed | said lost)
  true lost      = completed x P(real and lost | said completed)      + lost x P(real and lost | said lost)
  accuracy       = true completed / (true completed + true lost)
The interval comes from drawing the four rates again from the judged counts (a Dirichlet distribution with a prior of 0.5 per cell).

Limits: the estimate describes the passes the program FINDS (roughly half of the real ones; the passes it misses may be lost more often than the ones it finds,
which would make the true accuracy lower); the rates come from two matches and 89 judged passes, so the interval is wide (about +-9 points); the same
correction is applied to both teams, which keeps their order but cannot show a real difference between them.
To update the table: judge a new sample with compare_actions.py (passes of the current method), count the cells and edit CALIBRATION.

Usage:
  python pass_calibration.py 175 161        # 175 passes called completed, 161 lost
"""
import argparse
import sys

import numpy as np

# What the author said about the passes the program called completed / lost (touches of players, --passes players): mecz1 60 passes (30 of the default run and 30 of 2-4 m from the
# --pass-min-d 2 run), mecz4_3dual 29 of 30 (one real pass without a verdict is left out): 89 in all.
CALIBRATION = {
    "completed": {"real_completed": 38, "real_lost": 2, "not_a_pass": 14},
    "lost": {"real_completed": 15, "real_lost": 5, "not_a_pass": 15},
    "matches": ["mecz1", "mecz4_3dual"],
}


def judged(calib=CALIBRATION):
    return sum(sum(calib[v].values()) for v in ("completed", "lost"))


def _rates(counts):
    n = sum(counts.values())
    return counts["real_completed"] / n, counts["real_lost"] / n


def _accuracy(completed, lost, rc, rl):
    """Estimated share of the real passes that were completed, from the rates (real completed, real lost) of the two verdicts."""
    tp_c = completed * rc[0] + lost * rl[0]
    tp_l = completed * rc[1] + lost * rl[1]
    return tp_c / (tp_c + tp_l) if tp_c + tp_l > 0 else None


def estimate(completed, lost, calib=CALIBRATION, draws=4000, seed=0):
    """(point, low, high) of the pass accuracy in percent (95% interval), or None when there is no pass."""
    if completed + lost <= 0:
        return None
    rc, rl = _rates(calib["completed"]), _rates(calib["lost"])
    point = _accuracy(completed, lost, rc, rl)
    rng = np.random.default_rng(seed)
    keys = ("real_completed", "real_lost", "not_a_pass")
    dc = rng.dirichlet([calib["completed"][k] + 0.5 for k in keys], draws)
    dl = rng.dirichlet([calib["lost"][k] + 0.5 for k in keys], draws)
    vals = np.array([_accuracy(completed, lost, (a[0], a[1]), (b[0], b[1])) for a, b in zip(dc, dl)])
    return 100 * point, 100 * float(np.percentile(vals, 2.5)), 100 * float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("completed", type=int, help="passes the program called completed")
    ap.add_argument("lost", type=int, help="passes the program called lost")
    a = ap.parse_args()
    e = estimate(a.completed, a.lost)
    if e is None:
        sys.exit("no passes")
    raw = 100 * a.completed / (a.completed + a.lost)
    print(f"the program says {raw:.0f}% completed; the estimate of the share of real passes that were completed: {e[0]:.0f}% (95%: {e[1]:.0f}-{e[2]:.0f}%)")


if __name__ == "__main__":
    main()
