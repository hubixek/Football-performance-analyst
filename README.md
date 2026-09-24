# Football Performance Analyst

The software analyzes 6v6 football matches recorded with a single camera and generates
performance statistics for each team, such as ball possession, positioning and
overall performance.

> Status: Early stage of development – first detection model (v1) trained

## Technologies
- Python
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) – object detection
- [CVAT](https://github.com/cvat-ai/cvat) (self-hosted) – annotation
- FFmpeg – frame extraction
- Training on a local GPU (NVIDIA RTX 3070) via WSL2

## How it works
1. Input: match video recording (e.g. `.mp4`)
2. Detection of players, goalkeepers, referees and the ball on each frame (YOLOv8)
3. Tracking players across frames
4. Assigning players to teams (jersey color)
5. Mapping positions from the image to pitch coordinates
6. Calculating statistics
7. Output: team statistics 

## Dataset
- Frames extracted from match recordings with FFmpeg
- Pre-annotated automatically with a YOLO model, then corrected manually in CVAT
- Annotation guidelines: [`ANNOTATION.md`](ANNOTATION.md)
- Classes: `player`, `goalkeeper`, `referee`, `ball`
- Train/validation split by match (not by frame) to avoid data leakage
- Videos, frames and the dataset are not stored in this repository

## Annotation workflow
1. Extract frames from a match:
```bash
   ffmpeg -i videos/mecz.mp4 -vf fps=0.5 -q:v 2 frames/mecz/mecz_%04d.jpg
```
2. Pre-annotate the frames (pretrained COCO model or own trained model):
```bash
   python prelabel.py frames/mecz
   python prelabel.py frames/mecz runs/detect/runs/v1/weights/best.pt
```
3. Create a task in CVAT, upload the frames and import the generated ZIP (YOLO 1.1)
4. Correct the annotations according to `ANNOTATION.md`
5. Export the project from CVAT (Ultralytics YOLO Detection, with images) and split it into train/val by match

## Scripts
| Script | Description |
|---|---|
| `prelabel.py` | Runs a YOLO model on extracted frames and creates a ZIP ready to import into CVAT |
| `build_dataset.py` | Builds a dataset from frame folders and YOLO 1.1 label exports |

## Installation
```bash
git clone https://github.com/hubixek/Football-performance-analyst.git
cd Football-performance-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
A CUDA-capable GPU is recommended for training.

## Training
```bash
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=100 imgsz=1280 batch=8 patience=20
```

## Results

### v1 (YOLOv8s, imgsz=1280)
Data: 89 training frames (1 match), 91 validation frames (another match).

| Class | mAP50 | mAP50-95 |
|---|---|---|
| all | 0.684 | 0.533 |
| player | 0.916 | 0.784 |
| goalkeeper | 0.557 | 0.428 |
| referee | 0.838 | 0.656 |
| ball | 0.427 | 0.262 |

Next step: more matches in the training set, with focus on goalkeeper and ball examples.

## TODO
- [x] Annotation workflow (pre-annotation + CVAT)
- [x] First detection model (v1)
- [ ] Improve goalkeeper and ball detection (more training data)
- [ ] Player tracking (ByteTrack)
- [ ] Team assignment (jersey color)
- [ ] Mapping positions to pitch coordinates (homography)
- [ ] Ball possession stats
- [ ] Player positioning and heatmaps
- [ ] Overall team performance score
- [ ] Export results to CSV
- [ ] Usage instructions for the full pipeline

## Author
Hubert – [GitHub](https://github.com/hubixek)