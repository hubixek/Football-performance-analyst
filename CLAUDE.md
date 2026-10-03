# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
Analysis of 6v6 football matches from a dual-lens Veo recording (two 2048x1024 views stacked in one 2048x2048 video, fixed camera, so a pixel maps to a position on the
56 x 32.4 m pitch): detection (YOLOv8), ball, teams, possession, goals, passes, statistics and an HTML report. Python, OpenCV, numpy, Ultralytics.
One command per match: `process_match.py VIDEO`. Read `PROJECT_STATUS.md` (every decision with its evidence, measured results, traps; in Polish) and `README.md` first.

## Commands
```bash
python -m unittest discover -s tests -v                                  # all tests, ~12 s, no GPU/video/model needed
python -m unittest discover -s tests -p "test_match_actions.py"          # one file
cd tests && python -m unittest test_match_actions.PlayerPasses.test_ball_leaving_the_pitch_is_a_lost_pass   # one test
python process_match.py ~/football/videos/X_dual.mp4 --name X            # whole match (hours; --dry-run prints the steps; a finished step is skipped, --force repeats it)
python analyze_dual.py --match matches/X.json --model ... --from-step teams   # rerun from a step: detect, ball, teams, events, possession, tracks
```
- Always use `-s tests`; without it `tests` can resolve to the test package of Ultralytics and fail on `import pytest`. Two tests need matplotlib (skipped without it).
- On the author's PC the interpreter is `~/yolo-env/bin/python` (Python 3.14, Ultralytics 8.4); plain `python` is not on the PATH of the WSL shell.
- Training is the `yolo detect train` CLI run from `~/football` (the v8 recipe is in `PROJECT_STATUS.md`); from a script run through stdin pass `workers=0` to `model.val`, or the data loader dies.
- A test for every change of a rule of the goals, half times, teams or passes that fails without the change.

## Architecture: files are the interfaces
The pipeline is a chain of scripts that communicate only through CSV/JSON in one analysis folder (`~/football/analysis/<match>_dual/`); each step can be rerun alone.
`process_match.py` runs: `calibrate_match.py` (automatic calibration from the reference match `mecz2`, two rigid lenses) -> `auto_halves.py` (half times from the number of
people on the pitch) -> `analyze_dual.py` (calls `dual_detect.py` -> `detections.csv`, `dual_ball.py` -> `ball.csv`, `dual_teams.py` -> `teams.csv`, `detect_goals.py` -> `goals_auto.csv`,
`dual_possession.py` -> `possession.csv`, `dual_tracks.py` -> `tracks.csv`) -> `match_stats.py` (-> `stats.json`, whose `attacks_right` per half is needed by later steps) ->
goals again with the direction of attack -> `match_actions.py` (-> `actions.csv/json`) -> `match_report.py` (-> `report.html`).
- Time base: analysis runs on every 2nd frame (15 Hz of a 29.97 fps video). `frame` in `possession.csv` is the video frame number, `time_s` is the video time. `tracks.csv` rows keep the
  time of each detection, which differs by a few hundredths of a second between players, so match players to a frame by the nearest row within half a frame, never by exact time.
- `ball.csv` `source` is `detected` or `interpolated` (gaps up to 1 s are filled linearly); only `detected` frames are observations.
- Goals are found from kick-offs (a ball on the centre spot) and the ball in the net, not from a goal event. The scorer comes from the goal side seen plus the attack direction; where the ball
  was not seen at a goal it falls back to the kicking-off team, which is unreliable (1 of 4 right on `mecz4_3dual`).
- Passes (`match_actions.py --passes auto`) are read from touches of tracked players (`player_control`, `player_passes`) when `tracks.csv` exists, else from the ball flight. The report shows the pass accuracy as an estimate corrected by `pass_calibration.py` (its table is from judged passes: update it after every new judgement of passes; the raw share of completed passes
  understates the accuracy). Shots and xG come from the ball track and are hidden in the report (measured precision 37% on 35 judged shots).
- `config.py` (`FOOTBALL_HOME`, reference calibration in `calib_ref/`) exists but the scripts still use `~/football` directly.

## Measuring: the rules of the project
- Measure before building. A threshold or rule changes only with a measurement on real matches; a rule set on some matches is tested on another. Do not tune to make a test pass,
  and do not tune on the match that was the independent test without saying that it no longer is one.
- Matches: `mecz1` (day) and `mecz4_2dual` (night) are the validation sets and are never trained on; `mecz2`, `mecz4`, `mecz5` are training; `mecz4_3dual` (day, 10 goals, 6-4) is the third match.
  The goal rules were set on `mecz1` and `mecz4_2dual`; on `mecz4_3dual` 7 of 10 goals were found, 0 false.
- The author's judgements in `eval/` cannot be redone: do not overwrite them. Tools that produce the samples for them: `compare_goals.py` (official minutes), `team_check.py --score`,
  `compare_possession.py`, `compare_actions.py` (clips of passes/shots/windows; `--windows-from` reuses counts), `kickoff_ball_probe.py`, `ball_redetect_probe.py`.
- A model is accepted only if the ball AP50 rises on both validation sets (`mecz1`, `mecz4_2dual`); v9 did not and v8 stays.
- State precisely what was run and what was not; an untested script is called untested; do not describe results as validated on more matches than they were.

## Traps
- Do not edit scripts that `process_match.py` calls while it runs: its later steps start fresh interpreters and would run the edited code.
- `match_actions.py` and friends write into the analysis folder. For experiments make a scratch folder with symlinks to `possession.csv`, `tracks.csv`, `stats.json` and run there.
- `argparse` help strings must write `%` as `%%`. `*:Zone.Identifier` files (Windows downloads) are ignored by git; never commit videos, weights, `*.npz`, analysis folders (see `.gitignore`).
- Calibration is accepted at explained >= 50% and covered >= 60% (printed by `calibrate_match.py`).
- Scripts are flat in the repository root, each an `argparse` command with a docstring that says what it does and gives a usage example.

## Conventions
Code, comments, README in English; the author writes in Polish, answer in Polish. The project is CC BY-NC 4.0 (`LICENSE`); the detector (Ultralytics YOLOv8) is AGPL-3.0.
The cloud session has only the repository (no videos, weights, GPU): there only code, synthetic tests and documentation are possible, and anything that changes a result on real data must be
measured on the author's PC and said so in the pull request.
