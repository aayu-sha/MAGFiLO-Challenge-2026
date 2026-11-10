"""
Sanity-check submission.csv locally before (re)uploading.

  python check_submission.py --root <MAGFiLO_1.0_Kaggle_2026 path> --csv submission.csv
"""
import argparse, csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--csv", default="submission.csv")
a = p.parse_args()

img_dir = Path(a.root) / "test" / "test_images"
files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
stems = {f.stem for f in files}
sizes = Counter(Image.open(f).size for f in files)       # (W, H)
print(f"test images: {len(files)}   sizes (W,H): {dict(sizes)}")

raw = Path(a.csv).read_text().splitlines()
print("header:", repr(raw[0]))
rows = list(csv.reader(raw[1:]))
ids = [r[0] for r in rows]
print(f"rows: {len(rows)}   unique ids: {len(set(ids))}   bad column counts: {sum(len(r) != 2 for r in rows)}")
print("rows containing a quote or space in the RLE:", sum(('"' in r[1] or ' ' in r[1]) for r in rows if len(r) == 2))

pred_stems = {i.rsplit("_", 1)[0] for i in ids}
print(f"images with >=1 prediction: {len(pred_stems & stems)}/{len(stems)}")
print("stems not matching any test image:", sorted(pred_stems - stems)[:5])
print("test images with NO prediction:", len(stems - pred_stems))

w, h = sizes.most_common(1)[0][0]
bad = 0
areas = []
for fid, rle in rows:
    try:
        m = mask_util.decode({"size": [h, w], "counts": rle.encode()})
        areas.append(int(m.sum()))
        if m.sum() == 0:
            bad += 1
    except Exception as e:
        bad += 1
        print("decode failed:", fid, e)
        break
print(f"decoded with size [{h},{w}]: {len(areas)} ok, {bad} empty/failed; mean area {np.mean(areas):.0f}px, min {min(areas)}, max {max(areas)}")

# overlap check: Kaggle rejects any image whose predicted masks share a pixel
by_img = {}
for fid, rle in rows:
    by_img.setdefault(fid.rsplit("_", 1)[0], []).append(rle)
n_bad_imgs = 0
for stem, rles in by_img.items():
    acc = np.zeros((h, w), dtype=np.uint8)
    for r in rles:
        acc += mask_util.decode({"size": [h, w], "counts": r.encode()})
    if (acc > 1).any():
        n_bad_imgs += 1
        print(f"OVERLAP in {stem}: {(acc > 1).sum()} px")
print("images with overlapping masks:", n_bad_imgs, "(must be 0)")
