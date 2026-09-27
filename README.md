# Football Performance Analyst

The software analyzes 6v6 football matches and generates performance statistics for each
team: ball possession, where on the pitch each team had the ball, and more to come.

> Status: dual-lens analysis working end-to-end (calibration, detection, ball tracking,
> teams, possession); detection model v7

## Recording
Matches are recorded with a Veo camera placed at the halfway line. Two recordings are available:
- **follow-cam**: a moving view that follows the ball (first version of this project),
- **dual-lens**: the raw image of both wide-angle lenses, one above the other (2048×2048,
  each half shows one half of the pitch). The camera does not move, so every pixel can be
  mapped to a position on the pitch in metres. **The current pipeline uses this recording.**

## How it works
1. **Lens calibration** – a fisheye lens model maps pixels to pitch coordinates (metres).
   The two lenses are one rigid device, so a new match is calibrated automatically: only the
   position and orientation of the camera are searched, by fitting the pitch lines of the
   model to the white lines detected in both halves of the image.
2. **Detection** – players, referees and ball candidates in both halves (YOLOv8), mapped to
   pitch coordinates; people outside the pitch (substitutes, coaches, spectators) are dropped,
   a player seen by both lenses is kept once.
3. **Ball tracking** – the ball is followed in metres with a physical speed limit; spare balls
   lying next to the pitch are recognised as static spots and ignored; short gaps are interpolated.
4. **Team assignment** – jersey colour clustering (K-means); goalkeepers or substitutes in a
   different kit are marked as unknown, players in the referee's colour are relabeled.
5. **Ball possession** – the player closest to the ball (within 1.5 m) controls it; possession
   changes only after the other team controls the ball for 0.3 s and stays with the passing team
   during a pass.

Only the playing time is analysed (half times are set per match, the half-time break is skipped).

## Example result
Ball possession in the first 5 minutes of a test match (timeline and where on the pitch each
team had the ball):

![Ball possession](docs/possession_mecz1_dual.png)

## Usage
```bash
# 1. half times of the match (asks for start and end of each half)
python match_config.py ~/football/videos/mecz3_dual.mp4 --preview

# 2. whole analysis: calibration (from a reference match), detection, ball, teams, possession
python analyze_dual.py --match matches/mecz3.json \
    --model runs/detect/runs/v7_dual/weights/best.pt --ref mecz2

# quick test on the first 5 minutes of play / rerun later steps only
python analyze_dual.py --match matches/mecz3.json --model ... --limit 300
python analyze_dual.py --match matches/mecz3.json --model ... --from-step teams
```
Results in `~/football/analysis/<match>_dual/`: `detections.csv`, `ball.csv`, `teams.csv`,
`possession.csv`, `summary.json` (possession for the match and for each half), `ball.png`,
`possession.png`.

## Pitch
6v6 artificial pitch, 56 × 32.4 m. Penalty area 11.1 × 19.6 m, centre circle radius 4.75 m –
estimated from clicked line points together with the lens calibration (`refine_pitch.py`),
length measured on the satellite image. Definition with 19 keypoints: [`pitch/pitch_6v6.json`](pitch/pitch_6v6.json).

## Scripts
### Dual-lens analysis
| Script | Description |
|---|---|
| `analyze_dual.py` | Whole analysis of one match |
| `match_config.py` | Asks for the start and end of each half, saves `matches/<match>.json` |
| `calibrate_match.py` | Automatic calibration of a match from a reference match (rigid two-lens camera model) |
| `dual_detect.py` | Detection in both halves, positions in metres, pitch filter, merging of both lenses |
| `dual_ball.py` | Ball tracking in metres (speed limit, spare balls, interpolation) |
| `dual_teams.py` | Team assignment by jersey colour |
| `dual_possession.py` | Ball possession: per-frame CSV, summary, plots |

### Calibration tools
| Script | Description |
|---|---|
| `pick_pitch_points.html` | Browser tool: click pitch keypoints and points along pitch lines on a frame |
| `calibrate.py` | Manual calibration of one lens (fisheye model fitted to keypoints and line points) |
| `refine_pitch.py` | Calibrates both lenses and estimates the pitch dimensions from the clicked lines |
| `auto_calibrate.py` | Line-based calibration of a single lens (older approach, used by `calibrate_match.py`) |

### Dataset and annotation
| Script | Description |
|---|---|
| `extract_dual_frames.py` | Frames from both lenses, only during playing time |
| `prelabel_dual.py` | Pre-annotation for CVAT, detections outside the pitch are dropped using the calibration |
| `prelabel.py` | Pre-annotation of follow-cam frames |

### Follow-cam pipeline (first version, tag `followcam-v1`)
`analyze.py`, `detect.py`, `ball_track.py`, `teams.py`, `possession.py` – the same analysis on the
moving follow-cam view, in pixels instead of metres.

## Dataset
- Frames extracted from match recordings, pre-annotated with the current model, corrected in CVAT
- Annotation guidelines: [`ANNOTATION.md`](ANNOTATION.md)
- Classes: `player`, `referee`, `ball` (goalkeepers are annotated as `player`)
- Train/validation split by match (not by frame)
- Videos, frames and the dataset are not stored in this repository

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
Fine-tuned from v6. Data: 96 training frames (1 match), validation: 134 frames of another match
recorded with a different camera position.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.809 | 0.687 |
| player | 0.962 | 0.884 |
| referee | 0.904 | 0.770 |
| ball | 0.561 | 0.408 |

Players are detected more precisely than on the follow-cam view (no motion blur, no cut-off
players). The ball is smaller in the wide view and is the weakest class; ball tracking in metres
compensates for part of the missed detections.

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

Lessons: full frame resolution mattered most for the ball (v3 → v4), more matches improved every
class (v4 → v5), goalkeepers cannot be told apart from players by colour on a wide view (v2 → v3).

## TODO
- [x] Detection models (v1–v7), annotation workflow with pre-annotation and CVAT
- [x] Ball possession on follow-cam recordings (first version)
- [x] Fisheye lens calibration with line points, pitch dimensions estimated from the image
- [x] Automatic per-match calibration (rigid two-lens camera model)
- [x] Dual-lens analysis in metres: detection, ball tracking, teams, possession
- [ ] More annotated dual-lens matches (especially ball) and a second validation match
- [ ] Player tracking (stable IDs), goalkeeper identification by position
- [ ] Heatmaps and average positions per team, possession by pitch zone
- [ ] Turnovers and possession sequences
- [ ] Possession accuracy checked against manual labels
- [ ] Out-of-play detection (ball leaving the pitch)

## Author
Hubert – [GitHub](https://github.com/hubixek)