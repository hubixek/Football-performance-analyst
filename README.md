# Football Performance Analyst

The software analyzes 6v6 football matches recorded with a single camera and generates
performance statistics for each team, such as ball possession, positioning and
overall performance.

> Status: Early stage of development – detection model v6 trained, analysis pipeline in progress

## Technologies
- Python
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) – object detection
- [CVAT](https://github.com/cvat-ai/cvat) (self-hosted) – annotation
- FFmpeg – frame extraction
- Training on a local GPU (NVIDIA RTX 3070) via WSL2

## How it works
1. Input: match video recording (e.g. `.mp4`)
2. Detection of players, referees and the ball on each frame (YOLOv8)
3. Filtering detections (one ball per frame, only people inside the pitch)
4. Tracking players across frames
5. Assigning players to teams (jersey color)
6. Mapping positions from the image to pitch coordinates
7. Identifying goalkeepers by position (player closest to own goal)
8. Calculating statistics
9. Output: team statistics (and optionally an annotated video)

## Dataset
- Frames extracted from match recordings with FFmpeg
- Pre-annotated automatically with a YOLO model, then corrected manually in CVAT
- Annotation guidelines: [`ANNOTATION.md`](ANNOTATION.md)
- Classes: `player`, `referee`, `ball`
- Goalkeepers are annotated as `player` – on a single wide camera they are too small
  to be reliably distinguished visually, so they are identified later by position
- Train/validation split by match (not by frame) to avoid data leakage
- Videos, frames and the dataset are not stored in this repository

## Annotation workflow
1. Extract frames from a match:
```bash
   ffmpeg -i videos/mecz4.mp4 -vf fps=0.05 -q:v 2 frames/mecz4/mecz4_%04d.jpg
```
2. Delete frames without play
3. Pre-annotate the frames with the current model:
```bash
   python prelabel.py frames/mecz4 runs/detect/runs/v6/weights/best.pt
```
4. Create a task in CVAT, upload the frames and import the generated ZIP as **YOLO 1.1**
5. Correct the annotations according to `ANNOTATION.md`
6. Export the task as **Ultralytics YOLO Detection** (with images) and add it to the dataset

## Scripts
| Script | Description |
|---|---|
| `prelabel.py` | Runs a YOLO model on extracted frames and creates a ZIP ready to import into CVAT |
| `remap_classes.py` | One-time conversion of labels from 4 classes (with goalkeeper) to 3 classes |

## Installation
```bash
git clone https://github.com/hubixek/Football-performance-analyst.git
cd Football-performance-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
FFmpeg is required for frame extraction. A CUDA-capable GPU is recommended for training.

## Training
```bash
# YOLOv8s
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=100 imgsz=1920 batch=4 patience=20
# YOLOv8m (8 GB GPU)
yolo train model=yolov8m.pt data=dataset/data.yaml epochs=300 imgsz=1920 batch=2 patience=100
```
Frames are 1920×1080, so training at `imgsz=1920` keeps the full resolution,
which matters for the small ball.

## Results
All models are evaluated on the same validation match (91 frames).

### v6 (YOLOv8m, imgsz=1920) – current model
Data: 4 training matches. Best epoch 165 of 265.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.858 | 0.661 |
| player | 0.940 | 0.819 |
| referee | 0.881 | 0.806 |
| ball | 0.752 | 0.357 |

The larger model raised ball recall from 0.62 to 0.80 at the cost of lower ball
precision (0.66) and ~2x slower inference.
### v5 (YOLOv8s, imgsz=1920) – 2 more matches
Data: 4 training matches.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.856 | 0.668 |
| player | 0.966 | 0.829 |
| referee | 0.911 | 0.832 |
| ball | 0.692 | 0.342 |

Two additional training matches improved every class; ball precision
recovered from 0.55 to 0.75 while recall increased to 0.62.

### v4 (YOLOv8s, imgsz=1920) – full frame resolution
Data: 2 training matches.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.788 | 0.643 |
| player | 0.920 | 0.803 |
| referee | 0.883 | 0.830 |
| ball | 0.562 | 0.295 |

Training at the native 1920 px resolution raised ball recall from 0.41 to 0.59.

### v3 (YOLOv8s, imgsz=1280) – goalkeeper merged into player
Data: 2 training matches.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.783 | 0.604 |
| player | 0.954 | 0.805 |
| referee | 0.903 | 0.770 |
| ball | 0.493 | 0.237 |

In the 4-class setup, most goalkeepers were classified as `player`, so the classes were merged.

### v2 (YOLOv8s, imgsz=1280) – 4 classes
Data: 2 training matches.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.687 | 0.555 |
| player | 0.914 | 0.803 |
| goalkeeper | 0.431 | 0.332 |
| referee | 0.904 | 0.823 |
| ball | 0.499 | 0.260 |

Adding a second match improved referee and ball detection, but goalkeeper recall
dropped (0.53 → 0.30): 68% of goalkeepers were classified as `player`.

### v1 (YOLOv8s, imgsz=1280) – 4 classes
Data: 1 training match.

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.684 | 0.533 |
| player | 0.916 | 0.784 |
| goalkeeper | 0.557 | 0.428 |
| referee | 0.838 | 0.656 |
| ball | 0.427 | 0.262 |

## TODO
- [x] Annotation workflow (pre-annotation + CVAT)
- [x] Detection models (v1–v6)
- [x] Merge goalkeeper into player
- [x] Training at full frame resolution
- [ ] Filtering detections (one ball per frame, pitch boundary)
- [ ] Player tracking (ByteTrack)
- [ ] Team assignment (jersey color)
- [ ] Mapping positions to pitch coordinates (homography)
- [ ] Goalkeeper identification by position
- [ ] Ball possession stats
- [ ] Player positioning and heatmaps
- [ ] Overall team performance score
- [ ] Export results to CSV
- [ ] One-command pipeline for dataset building and training

## Author
Hubert – [GitHub](https://github.com/hubixek)