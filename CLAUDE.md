# Football Performance Analyst

Analysis of 6v6 football matches from a dual-lens recording (two 2048x1024 views stacked in one 2048x2048 video): detection (YOLOv8), ball, teams, possession,
goals, statistics and an HTML report. Python, OpenCV, numpy. One command per match: `process_match.py VIDEO`.

Read first: `PROJECT_STATUS.md` (the full state, every decision with its evidence, the measured results, the traps) and `README.md`.

## Tests (always run them before and after a change)
```bash
python -m unittest discover -s tests -v      # about 10 s, needs only numpy and opencv-python-headless
```
- Always with `-s tests`. Without it `tests` can resolve to the test package of the Ultralytics library and fail on `import pytest`.
- The tests use a small synthetic match (`tests/helpers.py`), so they need no GPU, no video and no model.
- A change of a rule of the goal detection, the half times or the teams needs a test that fails without it.

## What a cloud session CANNOT do
The cloud session has only this repository. There are NO match videos, NO model weights (`*.pt`), NO analysis folders and NO GPU, so the pipeline cannot be
run on a real match here. Do not try to download models or videos. Work that fits here: code, unit tests on synthetic data, refactoring, documentation, CI,
tools that are tested with stand-ins for the model. Anything that changes a result on real data must be checked on the author's PC (RTX 3070) with
`compare_goals.py` and `team_check.py --score`; say so in the pull request instead of claiming it works.

## Rules of the project
- Measure before building: a threshold or a rule is changed only with a measurement on the two matches (`mecz1` by day, `mecz4_2dual` at night), the
  numbers are in `PROJECT_STATUS.md`. Do not tune a rule to make a test pass.
- The goal rules were set on two matches only; do not describe them as validated on more.
- Never commit videos, weights, `*.npz`, analysis folders (see `.gitignore`).
- Scripts are flat in the repository root, each one an `argparse` command with a docstring that says what it does and gives a usage example.
- Code, comments and the README are in English. The author writes to you in Polish: answer in Polish.
- Tell the truth about what was and was not run: an untested script is called untested.

## Where things are
`process_match.py` (one command) -> `auto_halves.py`, `calibrate_match.py`, `analyze_dual.py` (detection, ball, teams, goals, possession) -> `match_stats.py`,
`match_actions.py`, `match_report.py`. Goals: `detect_goals.py`, checked against the league's minutes with `compare_goals.py`. Teams: `dual_teams.py`.
Measurement tools: `team_check.py`, `team_features_probe.py`, `kickoff_ball_probe.py`, `referee_rule_probe.py`. Cleanup: `tidy_project.py`.
