"""
Where do the mistakes come from?  Run on the validation split with the chosen settings.

  python diagnose.py --root <root> --ckpt checkpoints_aug/best_ep17.pth --score-thresh 0.9 --min-area 50
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from dataset import FilamentDataset
from model import build_model, make_disjoint
from train import split_by_filename

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--img-size", type=int, default=1024)
p.add_argument("--score-thresh", type=float, default=0.9)
p.add_argument("--min-area", type=int, default=50)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(a.root) / "train"
base = FilamentDataset(root / "train_images", root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
_, val_idx = split_by_filename(base, a.val_frac, a.seed)

model = build_model(pretrained=False, img_size=a.img_size)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.to(device).eval()
model.roi_heads.score_thresh = a.score_thresh


def rles(m):
    return [mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in m]


gt_best, gt_area, pr_best, tp_iou = [], [], [], []
for n, idx in enumerate(val_idx, 1):
    s = base[idx]
    gt = s["masks"]
    gt = gt[gt.flatten(1).any(1).bool()].numpy().astype(bool)
    x = s["image"].repeat(3, 1, 1).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = model([x])[0]
    if len(out["scores"]):
        pm = make_disjoint(out["masks"][:, 0] >= 0.5, out["scores"]).cpu().numpy()
        pm = pm[pm.sum((1, 2)) >= a.min_area]
    else:
        pm = np.zeros((0,) + gt.shape[1:], bool)
    g_r, p_r = rles(gt), rles(pm)
    iou = (np.asarray(mask_util.iou(p_r, g_r, [0] * len(g_r))) if len(p_r) and len(g_r)
           else np.zeros((len(p_r), len(g_r))))
    gt_area += [float(mask_util.area(r)) for r in g_r]
    gt_best += (iou.max(0).tolist() if len(p_r) else [0.0] * len(g_r))
    pr_best += (iou.max(1).tolist() if len(g_r) else [0.0] * len(p_r))
    hit = iou > 0.5
    tp_iou += iou[hit].tolist()
    if n % 20 == 0:
        print(f"{n}/{len(val_idx)}", flush=True)

gt_best, gt_area, pr_best = map(np.array, (gt_best, gt_area, pr_best))
N = len(gt_best)
print(f"\nGround-truth filaments: {N}")
print(f"  found (best IoU > 0.5):          {(gt_best > .5).sum():4d}  ({(gt_best > .5).mean():.0%})")
print(f"  near miss (0.3 - 0.5):           {((gt_best > .3) & (gt_best <= .5)).sum():4d}  ({((gt_best > .3) & (gt_best <= .5)).mean():.0%})")
print(f"  weak overlap (0.1 - 0.3):        {((gt_best > .1) & (gt_best <= .3)).sum():4d}  ({((gt_best > .1) & (gt_best <= .3)).mean():.0%})")
print(f"  not found at all (< 0.1):        {(gt_best <= .1).sum():4d}  ({(gt_best <= .1).mean():.0%})")
P = len(pr_best)
print(f"\nPredictions: {P}")
print(f"  hit (IoU > 0.5):                 {(pr_best > .5).sum():4d}")
print(f"  near-miss false alarm (0.1-0.5): {((pr_best > .1) & (pr_best <= .5)).sum():4d}")
print(f"  pure false alarm (< 0.1):        {(pr_best <= .1).sum():4d}")
print(f"\nMean IoU of hits: {np.mean(tp_iou):.3f}")

q = np.quantile(gt_area, [0.25, 0.5, 0.75])
edges = [0, *q, np.inf]
print("\nHit rate by filament size (area in pixels):")
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (gt_area >= lo) & (gt_area < hi)
    print(f"  {lo:8.0f} - {hi:8.0f}:  {(gt_best[sel] > .5).mean():.0%}  of {sel.sum()}")
