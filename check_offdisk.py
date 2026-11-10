"""
Are any predictions outside the solar disk?  Filaments only exist on the disk, so
off-disk predictions are certain false alarms.  Run when the GPU is free.

  python check_offdisk.py --root <root> --ckpt checkpoints_1280/epoch_16.pth \
      --img-size 1280 --score-thresh 0.8 --min-area 200

It also checks the disk detector itself: ground-truth filaments should all lie on the disk.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import FilamentDataset
from model import build_model, make_disjoint
from train import split_by_filename


def find_disk(gray):
    """Return (cx, cy, r) in full-resolution pixels. numpy only.
    Step 1: rough centre/size from a 3-class Otsu threshold (disk / halo / background).
    Step 2: refine the radius at the steepest drop of the radial brightness profile (the limb)."""
    small = gray[::4, ::4]
    hist, edges = np.histogram(small, bins=64, range=(0, 1))
    pr = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    mt = (pr * c).sum()
    best, thr = -1, 0.5
    for i in range(1, 63):
        for j in range(i + 1, 64):
            w = [pr[:i].sum(), pr[i:j].sum(), pr[j:].sum()]
            if min(w) < 1e-3:
                continue
            m = [(pr[:i] * c[:i]).sum() / w[0], (pr[i:j] * c[i:j]).sum() / w[1], (pr[j:] * c[j:]).sum() / w[2]]
            var = sum(wk * (mk - mt) ** 2 for wk, mk in zip(w, m))
            if var > best:
                best, thr = var, edges[j]
    mask = small >= thr
    cols = np.where(mask.sum(0) >= 20)[0]
    rows = np.where(mask.sum(1) >= 20)[0]
    cx, cy = (cols[0] + cols[-1]) / 2 * 4, (rows[0] + rows[-1]) / 2 * 4
    r0 = ((cols[-1] - cols[0]) + (rows[-1] - rows[0])) / 4 * 4

    yy, xx = np.indices(small.shape)
    rad = np.hypot(xx * 4 - cx, yy * 4 - cy) / 4          # radius in small-image pixels
    ri = rad.astype(int)
    prof = np.bincount(ri.ravel(), small.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    grad = np.gradient(prof)
    lo, hi = int(0.6 * r0 / 4), min(int(1.1 * r0 / 4), len(grad) - 1)
    r = (lo + int(np.argmin(grad[lo:hi]))) * 4
    return cx, cy, float(r)


def inside(masks, disk, tol=1.02):
    cx, cy, r = disk
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        out.append(np.hypot(xs.mean() - cx, ys.mean() - cy) <= r * tol if len(xs) else True)
    return np.array(out, bool)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img-size", type=int, default=1280)
    p.add_argument("--score-thresh", type=float, default=0.8)
    p.add_argument("--min-area", type=int, default=200)
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

    n_gt = n_gt_off = n_pred = n_pred_off = 0
    radii = []
    for n, idx in enumerate(val_idx, 1):
        s = base[idx]
        gray = s["image"][0].numpy()
        disk = find_disk(gray)
        radii.append(disk[2])
        gt = s["masks"]
        gt = gt[gt.flatten(1).any(1).bool()].numpy().astype(bool)
        n_gt += len(gt)
        n_gt_off += int((~inside(gt, disk)).sum())
        x = s["image"].repeat(3, 1, 1).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model([x])[0]
        if len(out["scores"]):
            pm = make_disjoint(out["masks"][:, 0] >= 0.5, out["scores"]).cpu().numpy()
            pm = pm[pm.sum((1, 2)) >= a.min_area]
            n_pred += len(pm)
            n_pred_off += int((~inside(pm, disk)).sum())
        if n % 20 == 0:
            print(f"{n}/{len(val_idx)}", flush=True)

    print(f"\ndetected disk radius: median {np.median(radii):.0f}px (min {min(radii):.0f}, max {max(radii):.0f})")
    print(f"ground-truth filaments with centre off the detected disk: {n_gt_off}/{n_gt}   (should be ~0 if the detector works)")
    print(f"predictions with centre off the detected disk:            {n_pred_off}/{n_pred}")
