"""
Error bars for PQ: bootstrap over validation FILES (annotator passes of one image are correlated,
so the file, not the sample, is the resampling unit).  Also gives a paired comparison of two models.

  python bootstrap_pq.py --root <root> --ckpt weights/maskrcnn_1280_final_fp16.pth --img-size 1280 \
      --score-thresh 0.8 --min-area 200 --save final.npz
  python bootstrap_pq.py --root <root> --ckpt <other.pth> --img-size 1024 \
      --score-thresh 0.9 --min-area 50 --save base.npz --compare final.npz      # PQ(this) - PQ(final)
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
p.add_argument("--img-size", type=int, default=1280)
p.add_argument("--anchors", choices=["default", "small"], default="default")
p.add_argument("--score-thresh", type=float, default=0.8)
p.add_argument("--min-area", type=int, default=200)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-boot", type=int, default=5000)
p.add_argument("--save", default="")
p.add_argument("--compare", default="")
a = p.parse_args()


def rles(m):
    return [mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in m]


def pq(arr):                       # arr columns: sum of IoU of hits, TP, FP, FN
    s = arr.sum(0)
    return s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3])


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(a.root) / "train"
base = FilamentDataset(root / "train_images", root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
_, val_idx = split_by_filename(base, a.val_frac, a.seed)

model = build_model(pretrained=False, img_size=a.img_size, anchors=a.anchors)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.to(device).eval()
model.roi_heads.score_thresh = a.score_thresh

per_file = {}
for n, idx in enumerate(val_idx, 1):
    s = base[idx]
    name = base.images[base.image_ids[idx]]["file_name"]
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
    pr, gr = rles(pm), rles(gt)
    iou = (np.asarray(mask_util.iou(pr, gr, [0] * len(gr))) if len(pr) and len(gr)
           else np.zeros((len(pr), len(gr))))
    hit = iou > 0.5
    row = np.array([iou[hit].sum(), hit.sum(), (hit.sum(1) == 0).sum(), (hit.sum(0) == 0).sum()], float)
    per_file[name] = per_file.get(name, 0) + row
    if n % 20 == 0:
        print(f"{n}/{len(val_idx)}", flush=True)

names = np.array(sorted(per_file))
arr = np.array([per_file[k] for k in names])
rng = np.random.default_rng(a.seed)
idxs = rng.integers(0, len(arr), size=(a.n_boot, len(arr)))
boots = np.array([pq(arr[i]) for i in idxs])
lo, hi = ci(boots)
print(f"\n{len(arr)} validation files | PQ = {pq(arr):.4f}   95% bootstrap CI [{lo:.4f}, {hi:.4f}]"
      f"   (TP {int(arr[:, 1].sum())}, FP {int(arr[:, 2].sum())}, FN {int(arr[:, 3].sum())})")

if a.save:
    np.savez(a.save, names=names, arr=arr)
if a.compare:
    other = np.load(a.compare, allow_pickle=True)
    common = sorted(set(names) & set(other["names"]))
    A = np.array([per_file[k] for k in common])
    B = np.array([other["arr"][list(other["names"]).index(k)] for k in common])
    idx2 = np.random.default_rng(a.seed + 1).integers(0, len(common), size=(a.n_boot, len(common)))
    diff = np.array([pq(A[i]) - pq(B[i]) for i in idx2])
    dlo, dhi = ci(diff)
    print(f"paired difference PQ(this) - PQ({a.compare}) = {pq(A) - pq(B):+.4f}   "
          f"95% CI [{dlo:+.4f}, {dhi:+.4f}]   share of resamples where this model is better: {(diff > 0).mean():.0%}")