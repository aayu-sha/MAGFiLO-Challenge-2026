"""
Do the default Mask R-CNN anchors fit our filaments?  CPU only, reads the annotation JSON.

For every filament box (scaled to the network input size) it finds the best-fitting
anchor shape and reports how well it fits, for the default anchors and for a proposed
small + elongated set.  IoU >= 0.7 is a 'good' match for the region-proposal network.

  python check_anchors.py --root <MAGFiLO_1.0_Kaggle_2026 path> --img-size 1280
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--img-size", type=int, default=1280)
a = p.parse_args()

coco = json.load(open(Path(a.root) / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
imgs = {im["id"]: im for im in coco["images"]}

W, H = [], []
for ann in coco["annotations"]:
    xs, ys = [], []
    for poly in ann["segmentation"]:
        xs += poly[0::2]
        ys += poly[1::2]
    if not xs:
        continue
    im = imgs[ann["image_id"]]
    scale = a.img_size / max(im.get("width", 2048), im.get("height", 2048))
    W.append((max(xs) - min(xs)) * scale)
    H.append((max(ys) - min(ys)) * scale)
W, H = np.maximum(np.array(W), 1.0), np.maximum(np.array(H), 1.0)
side = np.maximum(W, H)
print(f"boxes: {len(W)}   input size {a.img_size}px")
print("longest box side (px): " + "  ".join(f"p{q}={np.percentile(side, q):.0f}" for q in (10, 25, 50, 75, 90)))
print(f"longest side < 32 px: {(side < 32).mean():.0%}   < 64 px: {(side < 64).mean():.0%}")


def shapes(sizes, ratios):
    out = []
    for s in sizes:
        for r in ratios:                      # r = height / width
            out.append((s / np.sqrt(r), s * np.sqrt(r)))
    return np.array(out)


def best_iou(anchor_shapes):
    aw, ah = anchor_shapes[:, 0][None], anchor_shapes[:, 1][None]
    inter = np.minimum(W[:, None], aw) * np.minimum(H[:, None], ah)
    union = (W * H)[:, None] + aw * ah - inter
    return (inter / union).max(1)


default = best_iou(shapes((32, 64, 128, 256, 512), (0.5, 1.0, 2.0)))
proposed = best_iou(shapes((16, 32, 64, 128, 256), (0.25, 0.5, 1.0, 2.0, 4.0)))   # = --anchors small
small = side <= np.percentile(side, 25)

print("\nbest anchor IoU (box centred on anchor):")
print(f"{'':28s}{'default':>10s}{'proposed':>10s}")
for name, sel in (("all filaments", np.ones_like(small)), ("smallest quarter", small)):
    for thr in (0.3, 0.5, 0.7):
        print(f"{name + f'  IoU >= {thr}':28s}{(default[sel] >= thr).mean():10.0%}{(proposed[sel] >= thr).mean():10.0%}")
print("\nIf the two columns are close, new anchors will not help.")
