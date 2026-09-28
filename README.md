# Football Performance Analyst

Analysis of 6v6 football matches from video: ball possession, where on the pitch each team had the
ball, ball recoveries, team shape and team running distance, all in metres on the pitch.

> Status: the dual-lens analysis works end-to-end on full matches (calibration, detection, ball
> tracking, teams, possession, statistics, player tracks); detection model v7_dual.

## Recording
Matches are recorded with a Veo camera at the halfway line. Two recordings exist:
- **follow-cam**: a moving view that follows the ball (first version of this project, tag `followcam-v1`),
- **dual-lens**: the raw image of both wide-angle lenses, one above the other (2048×2048, each half
  shows one half of the pitch). The camera does not move, so every pixel maps to a position on the
  pitch in metres. **The current pipeline uses this recording.**

## How it works
1. **Lens calibration** – a fisheye model maps pixels to pitch coordinates. The two lenses are one rigid
   device, so a new match is calibrated automatically from a reference match: only the position and
   orientation of the camera are searched, by fitting the pitch lines of the model to the white lines
   detected in both halves.
2. **Detection** – players, referees and ball candidates in both halves (YOLOv8m, 2048 px), mapped to
   pitch coordinates. People outside the pitch (bench, coach, spectators) are dropped, a player seen by
   both lenses is kept once.
3. **Ball tracking** – the ball is followed in metres with a physical speed limit; spare balls lying next
   to the pitch are recognised and ignored; short gaps are interpolated.
4. **Teams and goalkeepers** – jersey colour clustering (K-means). Goalkeepers wear their own kit; they
   are found at the kick-off of each half and get the team standing in that half of the pitch.
5. **Ball possession** – the player closest to the ball (within 1.5 m) controls it; possession changes
   after 0.3 s of control by the other team and stays with the passing team during a pass. Nobody has
   possession while the ball is out of the pitch (more than 1 m behind a line for at least 1 s) or lies
   still (2 s within 1 m: restart, free kick, corner).
6. **Statistics** – possession by thirds, recoveries and where they happen, possession sequences,
   team shape, heatmaps, all as if each team always attacked to the right (sides change after half time).
7. **Player tracks** – tracks of the players in metres; used for the team running distance and for the
   number of substitutions. Individual players are **not** identified (see limitations).

Only the playing time is analysed (half times are set per match, the half-time break is skipped).

## Example result
Full match (54 min recording), won 11:2 by team 0 (the darker kits; team numbers are fixed by the
lightness of the kit):

| | Team 0 (winner) | Team 1 |
|---|---|---|
| Ball possession, whole match | 45.4% | 54.6% |
| Share of own possession in the opponent's half | 44.3% | 32.7% |
| Ball recoveries in the opponent's half | 36 | 27 |
| Possessions / average length | 136 / 7.0 s | 195 / 5.9 s |
| Running distance per field player and half | 2.6–3.3 km (both teams) | |

<!-- update these numbers from analysis/<match>_dual/summary.json, stats.json and team_distance.csv -->

The winner had *less* of the ball but played higher up the pitch and won it back there more often, so
possession alone is misleading; the analysis also reports zones, recoveries, sequences and team shape.

![Ball possession](docs/possession_mecz1_dual.png)
![Match statistics](docs/stats_mecz1_dual.png)

## Usage
```bash
# 1. half times of the match (asks for the start and the end of each half)
python match_config.py ~/football/videos/mecz3_dual.mp4 --preview

# 2. the whole analysis: calibration (from a reference match), detection, ball, teams, possession, tracks
python analyze_dual.py --match matches/mecz3.json \
    --model runs/detect/runs/v7_dual/weights/best.pt --ref mecz2

# a quick test on the first 5 minutes of play / rerun the later steps only
python analyze_dual.py --match matches/mecz3.json --model ... --limit 300
python analyze_dual.py --match matches/mecz3.json --model ... --from-step teams --reuse-colors

# 3. statistics and a check of the longest possessions (clips)
python match_stats.py ~/football/analysis/mecz3_dual --pitch pitch/pitch_6v6.json
python check_possessions.py ~/football/analysis/mecz3_dual --video ~/football/videos/mecz3_dual.mp4
```
Results in `~/football/analysis/<match>_dual/`: `detections.csv`, `ball.csv`, `teams.csv`, `possession.csv`,
`summary.json`, `stats.json`, `tracks.csv`, `players.csv`, `team_distance.csv` and the plots.
Optional: `--goals 14:21,21:29,...` gives the times of the goals, so that no possession is counted while
the ball is fetched and the teams take their places after a goal.

## Pitch
6v6 artificial pitch, 56 × 32.4 m. Penalty area 11.1 × 19.6 m, centre circle radius 4.75 m – estimated
from clicked line points together with the lens calibration (`refine_pitch.py`); the length was measured
on a satellite image. Definition with 19 keypoints: [`pitch/pitch_6v6.json`](pitch/pitch_6v6.json).

## Scripts
### Analysis of a match
| Script | Description |
|---|---|
| `analyze_dual.py` | The whole analysis of one match |
| `match_config.py` | Asks for the start and the end of each half, saves `matches/<match>.json` |
| `calibrate_match.py` | Automatic calibration of a match from a reference match (rigid two-lens camera model) |
| `dual_detect.py` | Detection in both halves, positions in metres, pitch filter, merging of both lenses |
| `dual_ball.py` | Ball tracking in metres (speed limit, spare balls, interpolation) |
| `dual_teams.py` | Teams by jersey colour, goalkeepers found at the kick-off |
| `dual_possession.py` | Ball possession: out of play, ball lying still, per-frame CSV, summary, plots |
| `match_stats.py` | Zones, recoveries, possession sequences, team shape, heatmaps |
| `dual_tracks.py` | Player tracks in metres, team running distance, substitutions |
| `check_possessions.py` | Cuts video clips of the longest possessions to check them by eye |
| `match_events.py` | Restart windows after goals from given goal times (`--goals`) |

### Calibration of a new camera (once per camera)
| Script | Description |
|---|---|
| `pick_pitch_points.html` | Browser tool: click pitch keypoints and points along the pitch lines on a frame |
| `calibrate.py` | Manual calibration of one lens (fisheye model fitted to keypoints and line points) |
| `refine_pitch.py` | Calibrates both lenses and estimates the pitch dimensions from the clicked lines |
| `fit_pitch_dims.py`, `auto_calibrate.py` | Helpers of the two scripts above and of `calibrate_match.py` |

### Dataset and experiments
| Script | Description |
|---|---|
| `extract_dual_frames.py` | Frames of both lenses, only during playing time |
| `prelabel_dual.py` | Pre-annotation for CVAT, detections outside the pitch are dropped using the calibration |
| `reid_probe.py` | Experiment: can the looks of players tell teammates apart? (result below) |

The follow-cam version (moving view, positions in pixels) is kept in the tag `followcam-v1`.

## Dataset
- Frames extracted from match recordings, pre-annotated with the current model, corrected in CVAT
- Annotation guidelines: [`ANNOTATION.md`](ANNOTATION.md)
- Classes: `player`, `referee`, `ball` (goalkeepers are annotated as `player`)
- Train/validation split by match (not by frame)
- Videos, frames and datasets are not stored in this repository

## Installation
```bash
git clone https://github.com/hubixek/Football-performance-analyst.git
cd Football-performance-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
FFmpeg is required. A CUDA-capable GPU is recommended (trained and run on an RTX 3070, 8 GB).

## Detection results

### v7_dual (YOLOv8m, imgsz=2048) – dual-lens recordings, current model
Fine-tuned from v6. Data: 96 training frames (1 match), validation: 134 frames of another match recorded
with a different camera position.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.809 | 0.687 |
| player | 0.962 | 0.884 |
| referee | 0.904 | 0.770 |
| ball | 0.561 | 0.408 |

Players are detected more precisely than on the follow-cam view (no motion blur, no cut-off players).
The ball is small in the wide view and is the weakest class.

### Follow-cam models (v1–v6)
Evaluated on a different validation set (one follow-cam match) – not comparable with v7_dual.

| Model | Setup | player mAP50 | referee mAP50 | ball mAP50 |
|---|---|---|---|---|
| v6 | YOLOv8m, 1920 px, 4 matches | 0.940 | 0.881 | 0.752 |
| v5 | YOLOv8s, 1920 px, 4 matches | 0.966 | 0.911 | 0.692 |
| v4 | YOLOv8s, 1920 px, 2 matches | 0.920 | 0.883 | 0.562 |
| v3 | YOLOv8s, 1280 px, goalkeeper merged into player | 0.954 | 0.903 | 0.493 |
| v2 | YOLOv8s, 1280 px, 4 classes (goalkeeper mAP50 0.431) | 0.914 | 0.904 | 0.499 |
| v1 | YOLOv8s, 1280 px, 1 match, 4 classes | 0.916 | 0.838 | 0.427 |

Lessons: full frame resolution mattered most for the ball (v3 → v4), more matches improved every class
(v4 → v5), goalkeepers cannot be told apart from players by colour on a wide view (v2 → v3).

## Known limitations
- **Players are tracked, not identified.** Teammates wear the same kit, players crowd around the ball and
  substitutions are rolling, so a track holds pieces of several players. The looks of teammates (colour of
  head, shirt, shorts and socks, `reid_probe.py`) tell them apart only partly (AUC 0.75; the differences
  are mostly the shorts) and the tracks used for the test were not clean, so there are no per-player
  totals. Reliable: team distance, team shape, heatmaps, number of substitutions (about 3 of 4 changes).
- Possession accuracy is not yet measured against manual labels; a visual check of the longest possessions
  found 4 of 5 correct.
- The ball is small and lost in about 24% of the frames, and hard to see at the centre spot; goals and
  kick-offs are therefore not detected automatically (`--goals` takes the goal times, optional).
- Goalkeepers are recognised at the kick-off of each half (own kit); a goalkeeper kit similar to a team's
  kit switches this off with a warning.

## TODO
- [x] Detection models (v1–v7), annotation workflow with pre-annotation and CVAT
- [x] Fisheye calibration, pitch dimensions estimated from the image, automatic per-match calibration
- [x] Dual-lens analysis in metres: detection, ball, teams and goalkeepers, possession, statistics, tracks
- [ ] More annotated dual-lens matches, especially with the ball near the centre (model v8), a second validation match
- [ ] Goal and kick-off detection (needs the better ball detection)
- [ ] Possession accuracy measured on a manually labelled segment
- [ ] Match report as a single HTML page
- [ ] Automated tests and CI

## Author
Hubert – [GitHub](https://github.com/hubixek)
