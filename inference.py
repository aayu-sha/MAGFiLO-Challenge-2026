"""
Generate the Kaggle submission CSV (columns: filament_id, segmentation_rle).

  python inference.py --root /path/to/MAGFiLO_1.0_Kaggle_2026 \
      --ckpt checkpoints/best_val_loss.pth --out submission.csv

Format follows the organizers' self-evaluation notebook:
  * filament_id       = "<image_id>_<k>", k = 1..K per image
  * segmentation_rle  = pycocotools compressed RLE counts string of a
                        Fortran-ordered uint8 mask at the ORIGINAL image size
Images with no kept prediction produce no rows.
Check the id convention against the Kaggle Data tab's sample submission
(if one is provided) before the first real upload.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from model import build_model, predict

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def encode_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=50, help="drop predicted masks smaller than this (pixels)")
    p.add_argument("--limit", type=int, default=0, help="only first N images (testing)")
    p.add_argument("--anchors", choices=["default", "small"], default="default")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False, img_size=args.img_size, anchors=args.anchors)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()
    # Only detections above the threshold get pasted to full 2048x2048 resolution;
    # pasting the default up-to-100 low-score masks needs ~1.7 GB and OOMs a 6 GB GPU.
    model.roi_heads.score_thresh = args.score_thresh

    img_dir = Path(args.root) / "test" / "test_images"
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in EXTS)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} test images, device {device}")

    rows, t0 = [], time.time()
    for n, f in enumerate(files, 1):
        img = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            res = predict(model, [x], args.score_thresh, args.mask_thresh)[0]

        masks = res["masks"].numpy()
        order = np.argsort(-res["scores"].numpy())
        k = 0
        for j in order:
            m = masks[j]
            if m.sum() < args.min_area:
                continue
            k += 1
            rows.append((f"{f.stem}_{k}", encode_rle(m)))
        if n % 25 == 0 or n == len(files):
            print(f"{n}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_NONE)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)

    total = time.time() - t0
    print(f"wrote {len(rows)} instances to {args.out}")
    print(f"inference time: {total:.1f}s total, {total / max(len(files), 1):.2f}s/image  (report this - efficiency is scored)")


if __name__ == "__main__":
    main()
