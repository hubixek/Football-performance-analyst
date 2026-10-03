#!/usr/bin/env python3
"""Is the analysis of a match trustworthy? Checks the finished analysis folder and writes quality.json, which match_report.py shows as a banner.

Checks (level: ok < warn < bad):
  calibration  both lenses explain >= 50% of the detected pitch-line pixels and cover >= 60% of the model lines (the limit from PROJECT_STATUS.md,
               measured on three matches); below that the positions in metres are wrong (bad). Calibrations made by hand have no such numbers (not checked).
  halves       two halves were found (warn otherwise)
  ball         the ball position is known in at least 70% of the frames (warn below; typical values on three matches: 83-95%; the limit is a guess, not measured)
  goals        a goal found without the ball seen at a goal has its scorer guessed from the team that kicks off, which was right in 1 of 4 such goals on one
               match: the score is then approximate (warn); no goal found at all is a warn too
  info         always: how many goals the program finds (a note, not a check)

The measured accuracy of what the report shows is kept in MEASURED below (one place, with the matches it comes from); update it when a measurement changes.

Usage:
  python quality.py ~/football/analysis/mecz5_dual --name mecz5 --calib-dir ~/football/calib
"""
import argparse
import csv
import json
from pathlib import Path
import config

ORDER = {"ok": 0, "info": 0, "warn": 1, "bad": 2}
EXPLAINED_MIN, COVERED_MIN, BALL_MIN = 0.50, 0.60, 70.0

# What was measured, on which matches (mecz1 by day, mecz4_2dual at night, mecz4_3dual by day), judged by the author; update with every new measurement.
MEASURED = {
    "matches": 3,
    "goals_found": 25, "goals_total": 29, "goals_false": 0,                      # mecz1 12/13, mecz4_2dual 6/6, mecz4_3dual 7/10
    "passes_precision": 72, "passes_judged": 60, "passes_recall": 55, "passes_completed_agree": 69,
    "shots_precision": 37, "shots_judged": 35,
}


def _load_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _check(cid, status, pl, en):
    return {"id": cid, "status": status, "pl": pl, "en": en}


def check_calibration(name, calib_dir):
    if not name or not calib_dir:
        return None
    vals = {}
    for half in ("top", "bottom"):
        p = Path(calib_dir) / f"{name}_{half}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        if d.get("explained_lines") is None or d.get("covered_lines") is None:
            return _check("calibration", "info", "Kalibracja zrobiona ręcznie: brak miar jakości.", "Calibration made by hand: no quality measures.")
        vals[half] = (float(d["explained_lines"]), float(d["covered_lines"]))
    ex, co = min(v[0] for v in vals.values()), min(v[1] for v in vals.values())
    if ex < EXPLAINED_MIN or co < COVERED_MIN:
        return _check("calibration", "bad",
                      f"Kalibracja kamery jest słaba (linie wyjaśnione {100 * ex:.0f}%, pokryte {100 * co:.0f}%; wymagane {100 * EXPLAINED_MIN:.0f}% i "
                      f"{100 * COVERED_MIN:.0f}%): pozycje w metrach mogą być błędne i nie należy ufać statystykom.",
                      f"The camera calibration is poor (lines explained {100 * ex:.0f}%, covered {100 * co:.0f}%; needed {100 * EXPLAINED_MIN:.0f}% and "
                      f"{100 * COVERED_MIN:.0f}%): positions in metres may be wrong, do not trust the statistics.")
    return _check("calibration", "ok", f"Kalibracja kamery dobra (linie wyjaśnione {100 * ex:.0f}%, pokryte {100 * co:.0f}%).",
                  f"Camera calibration good (lines explained {100 * ex:.0f}%, covered {100 * co:.0f}%).")


def check_halves(folder):
    halves = {r["half"] for r in _load_csv(Path(folder) / "possession.csv") if r.get("half")}
    if len(halves) >= 2:
        return _check("halves", "ok", "Znaleziono obie połowy.", "Both halves were found.")
    return _check("halves", "warn", f"Znaleziono {len(halves)} z 2 połów: statystyki dotyczą tylko analizowanego fragmentu.",
                  f"{len(halves)} of 2 halves were found: the statistics cover only the analysed part.")


def check_ball(folder):
    actions = Path(folder) / "actions.json"
    vis = None
    if actions.exists():
        vis = json.loads(actions.read_text()).get("quality", {}).get("ball_visible_percent")
    if vis is None:
        rows = _load_csv(Path(folder) / "possession.csv")
        if not rows:
            return None
        vis = 100.0 * sum(1 for r in rows if r.get("ball_x")) / len(rows)
    if vis < BALL_MIN:
        return _check("ball", "warn", f"Piłkę widać tylko w {vis:.0f}% klatek (zwykle 83-95%): posiadanie i podania mogą być mocno zaniżone.",
                      f"The ball is seen in only {vis:.0f}% of the frames (usually 83-95%): possession and passes may be far too low.")
    return _check("ball", "ok", f"Piłkę widać w {vis:.0f}% klatek.", f"The ball is seen in {vis:.0f}% of the frames.")


def check_goals(folder):
    rows = [r for r in _load_csv(Path(folder) / "goals_auto.csv") if r.get("category") == "goal" and r.get("goal_s")]
    if not rows:
        return _check("goals", "warn", "Nie znaleziono żadnego gola: wynik jest nieznany.", "No goal was found: the score is unknown.")
    guessed = [r for r in rows if r.get("scored_by_side", "") == ""]
    if guessed:
        n, g = len(rows), len(guessed)
        return _check("goals_guess", "warn",
                      f"Wynik jest orientacyjny: w {g} z {n} goli nie widziano piłki przy bramce, więc strzelca ustalono z tego, kto wznawia grę, co się często myli.",
                      f"The score is approximate: in {g} of {n} goals the ball was not seen at a goal, so the scorer was guessed from who kicks off, which is often wrong.")
    return _check("goals", "ok", f"Strzelca każdego z {len(rows)} goli ustalono z miejsca, w którym piłka weszła do bramki.",
                  f"The scorer of each of the {len(rows)} goals comes from the goal the ball went into.")


def assess(folder, name=None, calib_dir=None):
    """The checks of an analysis folder and the worst level: {'level': 'ok'|'warn'|'bad', 'checks': [...], 'measured': MEASURED}."""
    folder = Path(folder)
    checks = [c for c in (check_calibration(name, calib_dir), check_halves(folder), check_ball(folder), check_goals(folder)) if c]
    m = MEASURED
    checks.append(_check(
        "info", "info",
        f"Gole: na {m['matches']} meczach program znalazł {m['goals_found']} z {m['goals_total']} goli, żadnego fałszywego; gol na końcu połowy albo wznowienie bez piłki na środku "
        "może zostać pominięty.",
        f"Goals: on {m['matches']} matches the program found {m['goals_found']} of {m['goals_total']} goals, none false; a goal at the end of a half or a restart "
        "without the ball on the centre spot can be missed."))
    level = max((c["status"] for c in checks), key=lambda s: ORDER[s])
    level = "ok" if level == "info" else level
    return {"level": level, "checks": checks, "measured": m}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="analysis folder of a finished match")
    ap.add_argument("--name", help="name of the match (to read its calibration)")
    ap.add_argument("--calib-dir", default=str(config.CALIB_DIR))
    a = ap.parse_args()
    folder = Path(a.folder).expanduser()
    result = assess(folder, a.name, Path(a.calib_dir).expanduser())
    (folder / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"quality: {result['level']}")
    for c in result["checks"]:
        print(f"  [{c['status']}] {c['en']}")
    print(f"Saved {folder / 'quality.json'}")


if __name__ == "__main__":
    main()
