#!/usr/bin/env python3
"""Compare the goals the program found with the OFFICIAL minutes of the goals of the match (from the match report of the league).

The match clock: a half lasts --half-minutes (25 by default; measured on a match: the clock of the second half starts at 25:00 at its
kick-off). The clock of a half is counted from the start of the half found by detect_goals.py (the first kick-off of the half); the start
can be found a few seconds late, and the official minutes of one half can run ahead of the video (the league's clock), so a shift of the clock
of every half (at most --max-shift seconds) is looked for that lets most goals found by the BALL on the centre spot be matched one to one with an
official goal within --tol minutes (without --no-align). The matches without any shift are printed as well. A goal found at the clock time c is in the
minute floor(c / 60) + 1, like the official minutes.

For every official goal the program event that explains it is shown: a goal found by the ball, or a kick-off found only by the formation of
the players (the program does not call it a goal), or none (the goal at the end of a half has no kick-off after it, or the kick-off was not
found). Also the goals of the program that have no official goal in their minute (+-1 minute).

Needs goals_auto.csv of detect_goals.py in the analysis folder (run detect_goals.py first).

Usage:
  python compare_goals.py ~/football/analysis/mecz4_2dual --minutes 18,22,23,25,35,39
"""
import argparse
import csv
from pathlib import Path


def max_matching(nodes_e, nodes_o, minute_e, tol):
    """Maximum one-to-one matching (Kuhn) between the events and the official goals whose minutes differ by at most tol.
    Returns {index in nodes_e: index in nodes_o}."""
    adj = {i: [j for j, mo in enumerate(nodes_o) if abs(minute_e[i] - mo) <= tol] for i in range(len(nodes_e))}
    match_o = {}

    def try_(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_o or try_(match_o[j], seen):
                match_o[j] = i
                return True
        return False

    for i in range(len(nodes_e)):
        try_(i, set())
    return {i: j for j, i in match_o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with goals_auto.csv")
    ap.add_argument("--minutes", required=True, help="the official minutes of ALL the goals of the match, comma separated, e.g. 18,22,23,25,35,39")
    ap.add_argument("--half-minutes", type=float, default=25.0, help="length of a half by the match clock")
    ap.add_argument("--tol", type=int, default=1, help="a program event explains an official goal in a minute this close, minutes")
    ap.add_argument("--max-shift", type=float, default=300.0, help="the largest shift of the clock of a half that is tried, s")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--rough-s", type=float, default=15.0, help="a goal is this many seconds before its kick-off when the program has no better time")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    rows = list(csv.DictReader(open(folder / "goals_auto.csv")))
    official = sorted(int(x) for x in a.minutes.split(",") if x.strip())
    hl = a.half_minutes * 60
    starts = {}
    events = []
    for r in rows:
        h, kt = int(r["half"]), float(r["kickoff_s"])
        if r["category"] == "start of the half":
            starts[h] = kt
            continue
        gt = float(r["goal_s"]) if r.get("goal_s") and r.get("goal_time_is_rough", "0") != "1" else kt - a.rough_s
        events.append({"half": h, "kickoff": kt, "t": gt, "how": r["how"], "cat": r["category"], "scorer": r.get("scorer", "")})
    if not starts:
        raise SystemExit("goals_auto.csv has no start of a half: run detect_goals.py with --match")
    halves = sorted(starts)
    off_in_half = {h: [m for m in official if (m - 1) * 60 // hl == h - 1] for h in halves}

    def minute(e, shift):
        h = e["half"]
        clock = (e["t"] - starts[h] + shift[h]) + (h - 1) * hl
        return int(clock // 60) + 1

    def matches(h, s_):
        """One-to-one matches between the goals of the program in half h (the clock shifted by s_) and the official goals, within tol minutes."""
        sh = dict(shift)
        sh[h] = s_
        ev = [e for e in events if e["half"] == h and e["cat"] == "goal"]
        return len(max_matching(ev, off_in_half[h], [minute(e, sh) for e in ev], a.tol))

    shift = {h: 0.0 for h in halves}
    base = {h: matches(h, 0.0) for h in halves}
    if not a.no_align:
        for h in halves:
            best, best_s = -1, 0.0
            s_ = -a.max_shift
            while s_ <= a.max_shift:
                score = matches(h, s_)
                if score > best or (score == best and abs(s_) < abs(best_s)):
                    best, best_s = score, s_
                s_ += 5.0
            shift[h] = best_s
    n_ball = {h: sum(1 for e in events if e["half"] == h and e["cat"] == "goal") for h in halves}
    print("goals found by the ball matched one to one with an official goal (within %d min), per half: without a shift %s, with the best shift %s" %
          (a.tol, {h: f"{base[h]} of {n_ball[h]}" for h in halves}, {h: f"{matches(h, shift[h])} of {n_ball[h]}" for h in halves}))
    print("clock shift of the halves (s):", {h: int(v) for h, v in shift.items()}, "(0 = the clock starts at the first kick-off found)\n")

    used, printed = set(), set()
    explained = {}                                                   # (half, index in off_in_half[h]) -> (event index, minute)
    for h in halves:
        ev_idx = [k for k, e in enumerate(events) if e["half"] == h and e["cat"] == "goal"]
        ms = [minute(events[k], shift) for k in ev_idx]
        for i, j in max_matching(ev_idx, off_in_half[h], ms, a.tol).items():
            explained[(h, j)] = (ev_idx[i], ms[i])
            used.add(ev_idx[i])
        for j, mo in enumerate(off_in_half[h]):                      # the official goals left: a kick-off found only by the formation
            if (h, j) in explained:
                continue
            c = [(abs(minute(e, shift) - mo), k) for k, e in enumerate(events) if e["half"] == h and k not in used and abs(minute(e, shift) - mo) <= a.tol]
            if c:
                _, k = min(c)
                explained[(h, j)] = (k, minute(events[k], shift))
                used.add(k)
    print("official goal   event that explains it (kick-off)                 how        category      minute of the event")
    result = []
    for m in official:
        h = (m - 1) * 60 // hl + 1
        off_in_half_idx = [jj for jj, mo in enumerate(off_in_half[h]) if mo == m]
        # find the first unprinted explained slot of this minute
        slot = next((jj for jj in off_in_half_idx if (h, jj) in explained and (h, jj) not in printed), None)
        if slot is not None:
            printed.add((h, slot))
            k, em = explained[(h, slot)]
            e = events[k]
            kind = "BALL" if e["cat"] == "goal" else "formation only"
            print(f"{m:6d}'          {int(e['kickoff']) // 60:02d}:{int(e['kickoff']) % 60:02d} (goal about {int(e['t']) // 60:02d}:{int(e['t']) % 60:02d})"
                  f"{'':20s}{e['how']:9s}  {e['cat']:12s}  {em}'  -> {kind}")
            result.append((m, kind))
        else:
            print(f"{m:6d}'          no event near (a goal at the end of a half has no kick-off after it)")
            result.append((m, "none"))
    unexplained = [e for k, e in enumerate(events) if e["cat"] == "goal" and k not in used]
    print()
    n = len(official)
    by_ball = sum(1 for _, k in result if k == "BALL")
    form = [m for m, k in result if k == "formation only"]
    none = [m for m, k in result if k == "none"]
    print(f"official goals: {n}; found by the ball: {by_ball} ({100 * by_ball / n:.0f}%); explained only by a formation-only kick-off: {len(form)} {form}; "
          f"no event: {len(none)} {none}")
    if unexplained:
        print("goals of the program with no official goal in that minute (+-%d): " % a.tol +
              ", ".join(f"{int(e['t']) // 60:02d}:{int(e['t']) % 60:02d} (minute {minute(e, shift)})" for e in unexplained))
    else:
        print("every goal of the program has an official goal in its minute")
    n_goal_ev = sum(1 for e in events if e["cat"] == "goal")
    print(f"precision of the goals found by the ball: {n_goal_ev - len(unexplained)} of {n_goal_ev}")


if __name__ == "__main__":
    main()
