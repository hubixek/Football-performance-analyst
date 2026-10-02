#!/usr/bin/env python3
"""Which description of a player's appearance separates the two teams best? A measurement on the crops you judged by eye.

Uses the crops of team_check.py that you judged (team_check/team_check.csv, column `truth` 0 or 1) and the hidden key of that
script (where every crop is in the video). For every crop several descriptions are computed and the teams are classified with
leave-one-out cross validation (every crop is classified by a rule learnt from all the OTHER crops), so the accuracy is honest.

Descriptions:
  current        median Lab colour of the shirt area (what dual_teams.py uses now)
  chroma         only the colour (a*, b*) of the shirt, without the brightness (floodlights change the brightness most)
  relative       the shirt colour minus the colour of the turf next to the player (the same light falls on both)
  bands3         median Lab colour of three bands: upper shirt, lower shirt, shorts (9 numbers)
  hsv            hue histogram weighted by saturation, plus mean saturation and brightness
  labhist        histogram of the colours (a*, b*, L) of the shirt area
  cnn            an embedding of a neural network for images (resnet18; needs torchvision and the weights download; --no-cnn skips it)
  siglip         with --siglip: the SigLIP image embedding of the crop (google/siglip-base-patch16-224; pip install transformers umap-learn),
                 classified like the others AND clustered the way the open-source football analytics do it: UMAP to 3 numbers, then K-means
                 with two clusters, without any labels

The column "K-means" is the accuracy WITHOUT labels (two clusters of the standardised numbers, the better of the two ways to name them):
the pipeline has no labels from you, only the kick-offs.

Classifiers: nearest centroid (on standardised numbers) and a shrinkage linear discriminant. The last line is the accuracy of
the analysis as it is now on the same crops.

Usage:
  python team_features_probe.py ~/football/analysis/mecz4_2dual --video ~/football/videos/mecz4_2dual.mp4
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

SHIRT = (0.15, 0.40)
BANDS = ((0.15, 0.32), (0.32, 0.50), (0.52, 0.72))


def band_lab(img, box, band, wfrac=0.25):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = img[max(int(y1 + band[0] * h), 0):max(int(y1 + band[1] * h), 0), max(int(x1 + wfrac * w), 0):max(int(x2 - wfrac * w), 0)]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def turf_lab(img, box):
    """Median Lab colour left and right of the player at the height of the shirt (other players are rare there, the median ignores them)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ya, yb = max(int(y1 + 0.3 * h), 0), max(int(y1 + 0.9 * h), 0)
    parts = [img[ya:yb, max(int(x1 - 0.8 * w), 0):max(int(x1 - 0.2 * w), 0)], img[ya:yb, min(int(x2 + 0.2 * w), img.shape[1]):min(int(x2 + 0.8 * w), img.shape[1])]]
    px = [cv2.cvtColor(p, cv2.COLOR_BGR2LAB).reshape(-1, 3) for p in parts if p.shape[0] > 1 and p.shape[1] > 1]
    return np.median(np.vstack(px), axis=0).astype(np.float32) if px else None


def descriptors(img, box):
    sh = band_lab(img, box, SHIRT)
    if sh is None:
        return None
    med = np.median(sh, axis=0)
    out = {"current": med}
    out["chroma"] = med[1:]
    tl = turf_lab(img, box)
    out["relative"] = med - tl if tl is not None else med - np.array([100.0, 128, 128], np.float32)
    bands = [band_lab(img, box, b) for b in BANDS]
    out["bands3"] = np.concatenate([np.median(b, axis=0) if b is not None else med for b in bands])
    hsv = cv2.cvtColor(img[max(int(box[1] + SHIRT[0] * (box[3] - box[1])), 0):max(int(box[1] + SHIRT[1] * (box[3] - box[1])), 0),
                           max(int(box[0] + 0.25 * (box[2] - box[0])), 0):max(int(box[2] - 0.25 * (box[2] - box[0])), 0)], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hist, _ = np.histogram(hsv[:, 0], bins=12, range=(0, 180), weights=hsv[:, 1].astype(np.float64) + 1.0)
    out["hsv"] = np.concatenate([hist / max(hist.sum(), 1e-6), [hsv[:, 1].mean() / 255, hsv[:, 2].mean() / 255]]).astype(np.float32)
    h2, _, _ = np.histogram2d(sh[:, 1], sh[:, 2], bins=5, range=((0, 255), (0, 255)))
    hl, _ = np.histogram(sh[:, 0], bins=4, range=(0, 255))
    v = np.concatenate([h2.ravel(), hl]).astype(np.float32)
    out["labhist"] = v / max(v.sum(), 1.0)
    return out


def cnn_embed(crops):
    import torch
    import torchvision
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net.eval()
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    with torch.no_grad():
        for c in crops:
            x = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (64, 128)).astype(np.float32) / 255.0
            x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1))[None]
            out.append(net(x)[0].numpy())
    return np.array(out)


def loo(X, y, kind):
    n, d = X.shape
    ok = 0
    for i in range(n):
        tr = np.arange(n) != i
        Xt, yt = X[tr], y[tr]
        if kind == "cosine":
            c0, c1 = Xt[yt == 0].mean(0), Xt[yt == 1].mean(0)
            z = X[i]
            ok += int((z @ c1 / (np.linalg.norm(z) * np.linalg.norm(c1) + 1e-9)) > (z @ c0 / (np.linalg.norm(z) * np.linalg.norm(c0) + 1e-9))) == y[i]
            continue
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Z, z = (Xt - mu) / sd, (X[i] - mu) / sd
        c0, c1 = Z[yt == 0].mean(0), Z[yt == 1].mean(0)
        if kind == "centroid":
            pred = int(np.linalg.norm(z - c1) < np.linalg.norm(z - c0))
        else:
            S = np.cov(Z.T) if d > 1 else np.array([[Z.var()]])
            S = 0.7 * S + 0.3 * np.eye(d) * np.trace(S) / d
            w = np.linalg.solve(S, c1 - c0)
            pred = int(z @ w - 0.5 * w @ (c0 + c1) > 0)
        ok += pred == y[i]
    return ok / n


def kmeans_acc(X, y):
    """Accuracy of two K-means clusters without labels (the better of the two ways to name the clusters)."""
    Z = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    cv2.setRNGSeed(0)
    _, lab, _ = cv2.kmeans(Z, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    acc = float(np.mean(lab == y))
    return max(acc, 1 - acc)


def siglip_embed(crops, model_name="google/siglip-base-patch16-224", batch=32):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + batch]]
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            out.append(model(**inp).last_hidden_state.mean(dim=1).cpu().numpy())      # as in the football analytics of Roboflow
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder with team_check/team_check.csv (your judgements)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--siglip", action="store_true", help="also the SigLIP embedding, with UMAP and K-means (needs transformers and umap-learn)")
    a = ap.parse_args()

    d = Path(a.folder).expanduser() / "team_check"
    truth = {r["id"]: r["truth"].strip().lower() for r in csv.DictReader(open(d / "team_check.csv"))}
    key = [k for k in csv.DictReader(open(d / ".answers_of_the_analysis.csv")) if truth.get(k["id"]) in ("0", "1")]
    if len(key) < 20:
        raise SystemExit(f"only {len(key)} crops judged as 0 or 1 - judge more crops with team_check.py first")
    cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
    feats, ys, crops, base_ok = {}, [], [], 0
    for k in sorted(key, key=lambda k: int(k["frame"])):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k["frame"]))
        ok, img = cap.read()
        if not ok:
            continue
        lens = img[:img.shape[0] // 2] if k["cam"] == "top" else img[img.shape[0] // 2:]
        box = tuple(float(k[c]) for c in ("x1", "y1", "x2", "y2"))
        ds = descriptors(lens, box)
        if ds is None:
            continue
        for name, v in ds.items():
            feats.setdefault(name, []).append(v)
        ys.append(int(truth[k["id"]]))
        base_ok += (k["team"] == truth[k["id"]])
        x1, y1, x2, y2 = (int(v) for v in box)
        crops.append(lens[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)])
    cap.release()
    y = np.array(ys)
    print(f"{len(y)} crops judged as team 0 ({int((y == 0).sum())}) or team 1 ({int((y == 1).sum())}); chance level {100 * max((y == 0).mean(), (y == 1).mean()):.0f}%\n")
    print(f"{'description':12s} {'numbers':>7s}  {'nearest centroid':>17s}  {'linear discriminant':>20s}  {'K-means (no labels)':>20s}")
    for name in ("current", "chroma", "relative", "bands3", "hsv", "labhist"):
        X = np.array(feats[name], np.float64)
        print(f"{name:12s} {X.shape[1]:7d}  {100 * loo(X, y, 'centroid'):16.0f}%  {100 * loo(X, y, 'lda'):19.0f}%  {100 * kmeans_acc(X, y):19.0f}%")
    if not a.no_cnn:
        try:
            E = cnn_embed(crops)
            print(f"{'cnn':12s} {E.shape[1]:7d}  {100 * loo(E, y, 'cosine'):16.0f}%  {'-':>20s}  {100 * kmeans_acc(E, y):19.0f}%")
        except Exception as e:                                          # no torchvision, no internet for the weights ...
            print(f"cnn          skipped ({type(e).__name__}: {str(e)[:70]})")
    if a.siglip:
        try:
            E = siglip_embed(crops)
            line = f"{'siglip':12s} {E.shape[1]:7d}  {100 * loo(E, y, 'cosine'):16.0f}%  {'-':>20s}  {100 * kmeans_acc(E, y):19.0f}%"
            print(line)
            try:
                import umap
                from sklearn.cluster import KMeans
                R = umap.UMAP(n_components=3, random_state=0).fit_transform(E)
                lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(R)
                acc = float(np.mean(lab == y))
                print(f"{'siglip+UMAP':12s} {3:7d}  {'-':>17s}  {'-':>20s}  {100 * max(acc, 1 - acc):19.0f}%   (UMAP to 3 numbers, K-means with 2 clusters: the open-source recipe)")
                print(f"{'siglip+UMAP':12s} {3:7d}  {100 * loo(R, y, 'centroid'):16.0f}%  {100 * loo(R, y, 'lda'):19.0f}%  (with labels, on the 3 UMAP numbers; UMAP itself saw all crops)")
            except Exception as e:
                print(f"siglip+UMAP  skipped ({type(e).__name__}: {str(e)[:70]}); pip install umap-learn scikit-learn")
        except Exception as e:
            print(f"siglip       skipped ({type(e).__name__}: {str(e)[:90]}); pip install transformers")
    print(f"\nthe analysis as it is now, on the same crops: {100 * base_ok / len(y):.0f}% (its own rule, learnt without these crops)")


if __name__ == "__main__":
    main()
