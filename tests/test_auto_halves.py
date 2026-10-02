import unittest

import numpy as np

import helpers  # noqa: F401  (the repository on the path)
from auto_halves import find_halves

STEP = 2.0


def profile(duration, plays, spikes=(), seed=0):
    """People on the pitch every STEP seconds: 0-2 on an empty pitch, about 10-11 seen during the play."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration, STEP)
    counts = rng.integers(0, 3, len(times)).astype(float)
    for a, b in plays:
        m = (times >= a) & (times <= b)
        counts[m] = np.clip(rng.normal(10.5, 1.5, m.sum()).round(), 4, 13)
    for a, b, v in spikes:
        counts[(times >= a) & (times <= b)] = v
    return times, counts


class FindHalves(unittest.TestCase):
    def test_two_halves_with_an_empty_pitch_between(self):
        t, c = profile(3300, [(0, 1549), (1697, 3198)])
        halves, notes = find_halves(t, c, step=STEP)
        self.assertEqual(len(halves), 2)
        for (s, e), (rs, re_) in zip(halves, [(0, 1549), (1697, 3198)]):
            self.assertLess(abs(s - rs), 6)
            self.assertLess(abs(e - re_), 6)

    def test_warm_up_is_dropped_and_a_timeout_is_bridged(self):
        t, c = profile(3600, [(150, 280), (400, 1900), (2100, 2530), (2550, 3600)], [(2530, 2550, 3)])
        halves, _ = find_halves(t, c, step=STEP)
        self.assertEqual([(round(a / 10), round(b / 10)) for a, b in halves], [(40, 190), (210, 360)])

    def test_break_without_an_empty_pitch_is_found_by_the_lowest_count(self):
        t, c = profile(3300, [(0, 1549), (1550, 1700), (1700, 3198)], [(1549, 1700, 6)])
        halves, _ = find_halves(t, c, step=STEP)
        self.assertEqual(len(halves), 2)
        self.assertLess(abs(halves[0][1] - 1550), 10)
        self.assertLess(abs(halves[1][0] - 1700), 10)

    def test_a_single_half_is_not_split(self):
        t, c = profile(1700, [(30, 1600)])
        halves, notes = find_halves(t, c, step=STEP)
        self.assertEqual(len(halves), 1)
        self.assertTrue(notes)                      # the caller is told that two halves were not found


if __name__ == "__main__":
    unittest.main()
