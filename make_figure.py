"""
Make a qualitative figure: ground-truth outlines (left) vs. predicted outlines (right)
on a few validation images. Run when the GPU is free.

  python make_figure.py --root <root> --ckpt checkpoints_aug/best_ep17.pth \
      --score-thresh 0.9 --min-area 50 --n 2 --out fig_predictions.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FilamentDataset
from model import build_model, make_disjoint
from train import split_by_filename

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--img-size", type=int, default=1024)
p.add_argument("--score-thresh", type=float, default=0.9)
p.add_argument("--min-area", type=int, default=50)
p.add_argument("--n", type=int, default=2, help="number of validation images (rows)")
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="fig_predictions.png")
a = p.parse_args()


def outline(m):
    pad = np.pad(m, 1)
    inner = (pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1]
             & pad[1:-1, :-2] & pad[1:-1, 2:])
    b = m & ~inner
    t = b.copy()                      # thicken by one pixel so it survives downscaling
    t[1:, :] |= b[:-1, :]; t[:-1, :] |= b[1:, :]
    t[:, 1:] |= b[:, :-1]; t[:, :-1] |= b[:, 1:]
    return t


def paint(gray, masks, colors):
    rgb = np.stack([gray] * 3, -1)
    for m, c in zip(masks, colors):
        rgb[outline(m)] = c
    return rgb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(a.root) / "train"
base = FilamentDataset(root / "train_images", root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
_, val_idx = split_by_filename(base, a.val_frac, a.seed)
chosen = [val_idx[i] for i in np.linspace(0, len(val_idx) - 1, a.n).astype(int)]

model = build_model(pretrained=False, img_size=a.img_size)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.to(device).eval()
model.roi_heads.score_thresh = a.score_thresh

palette = [tuple(c[:3]) for c in plt.get_cmap("tab10").colors]
fig, axes = plt.subplots(a.n, 2, figsize=(7.0, 3.5 * a.n), squeeze=False)
for r, idx in enumerate(chosen):
    s = base[idx]
    img = s["image"][0].numpy()
    lo, hi = np.percentile(img, [1, 99.5])           # display-only contrast stretch
    gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

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

    axes[r, 0].imshow(paint(gray, gt, [(0.1, 1.0, 0.2)] * len(gt)))
    axes[r, 1].imshow(paint(gray, pm, [palette[i % 10] for i in range(len(pm))]))
    axes[r, 0].set_title(f"Annotation ({len(gt)} filaments)", fontsize=9)
    axes[r, 1].set_title(f"Prediction ({len(pm)} filaments)", fontsize=9)
    for ax in axes[r]:
        ax.axis("off")

plt.tight_layout()
fig.savefig(a.out, dpi=200, bbox_inches="tight")
print("saved", a.out)
