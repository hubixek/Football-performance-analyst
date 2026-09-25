import sys, zipfile
from pathlib import Path
from ultralytics import YOLO

frames = Path(sys.argv[1])  
model_path = sys.argv[2] if len(sys.argv) > 2 else "yolov8s.pt"
names = ["player", "referee", "ball"]

model = YOLO(model_path)
pretrained = model_path == "yolov8s.pt"
remap = {0: 0, 32: 2}     

out = frames.with_suffix(".zip")
imgs = sorted(frames.glob("*.jpg"))
with zipfile.ZipFile(out, "w") as z:
    z.writestr("obj.names", "\n".join(names))
    z.writestr("obj.data", f"classes = {len(names)}\nnames = data/obj.names\ntrain = data/train.txt\n")
    z.writestr("train.txt", "\n".join(f"data/obj_train_data/{p.name}" for p in imgs))
    kwargs = dict(conf=0.25, imgsz=1920, stream=True, verbose=False)
    if pretrained:
        kwargs["classes"] = [0, 32]
    for r in model.predict(imgs, **kwargs):
        lines = []
        for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
            cls = remap[int(c)] if pretrained else int(c)
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        z.writestr(f"obj_train_data/{Path(r.path).stem}.txt", "\n".join(lines))
print(f"Gotowe: {out} ({len(imgs)} klatek)")
