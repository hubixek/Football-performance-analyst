# Football Performance Analyst

The software analyzes 6v6 football matches recorded with a single camera and generates
performance statistics for each team, such as ball possession, positioning and
overall performance.

> Status: Early stage of development – detection model v3 trained

## Technologies
- Python
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) – object detection
- [CVAT](https://github.com/cvat-ai/cvat) (self-hosted) – annotation
- FFmpeg – frame extraction
- Training on a local GPU (NVIDIA RTX 3070) via WSL2

## How it works
1. Input: match video recording (e.g. `.mp4`)
2. Detection of players, referees and the ball on each frame (YOLOv8)
3. Tracking players across frames
4. Assigning players to teams (jersey color)
5. Mapping positions from the image to pitch coordinates
6. Identifying goalkeepers by position (player closest to own goal)
7. Calculating statistics
8. Output: team statistics (and optionally an annotated video)

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
   ffmpeg -i videos/mecz.mp4 -vf fps=0.05 -q:v 2 frames/mecz/mecz%04d.jpg
```
2. Delete frames without play
3. Pre-annotate the frames (pretrained COCO model or own trained model):
```bash
   python prelabel.py frames/mecz
   python prelabel.py frames/mecz runs/detect/runs/v3/weights/best.pt
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
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=100 imgsz=1280 batch=8 patience=20
```

## Results
All models are evaluated on the same validation match (91 frames).

### v3 (YOLOv8s, imgsz=1280) – goalkeeper merged into player
Data: 149 training frames (3 matches).

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.783 | 0.604 |
| player | 0.954 | 0.805 |
| referee | 0.903 | 0.770 |
| ball | 0.493 | 0.237 |

In the 4-class setup, most goalkeepers were classified as `player`, so the classes were merged.
Ball detection (recall 0.41) is now the main bottleneck.

### v1 (YOLOv8s, imgsz=1280) – 4 classes
Data: 89 training frames (2 matches).

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.684 | 0.533 |
| player | 0.916 | 0.784 |
| goalkeeper | 0.557 | 0.428 |
| referee | 0.838 | 0.656 |
| ball | 0.427 | 0.262 |

## TODO
- [x] Annotation workflow (pre-annotation + CVAT)
- [x] First detection models (v1, v3)
- [x] Merge goalkeeper into player
- [ ] Improve ball detection (more matches, higher resolution, SAHI)
- [ ] Player tracking (ByteTrack)
- [ ] Team assignment (jersey color)
- [ ] Mapping positions to pitch coordinates (homography)
- [ ] Goalkeeper identification by position
- [ ] Ball possession stats
- [ ] Player positioning and heatmaps
- [ ] Overall team performance score
- [ ] Export results to CSV

## Author
Hubert – [GitHub](https://github.com/hubixek)