"""
Score a checkpoint on the held-out validation split with the organizers' PQ
(pooled over images; TP = pairs with IoU > 0.5, FP = predictions with no hit,
FN = GT filaments with no hit; PQ = sum(IoU of TP) / (TP + 0.5 FP + 0.5 FN)),
and sweep score threshold x minimum area to pick post-processing settings.

Run ONLY when training is not using the GPU (6 GB is tight).

  python validate.py --root <root> --ckpt checkpoints/last.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from dataset import FilamentDataset
from model import build_model, make_disjoint
from train import split_by_filename


def pq_from_cache(cache, score_thresh, min_area):
    """cache: list of dicts with iou [P,G], scores [P], areas [P], n_gt."""
    tp_ious, fp, fn = [], 0, 0
    for c in cache:
        keep = (c["scores"] >= score_thresh) & (c["areas"] >= min_area)
        iou = c["iou"][keep]                      # [P', G]
        n_pred, n_gt = iou.shape
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        hit = iou > 0.5
        tp_ious.extend(iou[hit].tolist())
        fp += int((hit.sum(axis=1) == 0).sum())   # predictions with no GT hit
        fn += int((hit.sum(axis=0) == 0).sum())   # GT with no prediction hit
    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / denom if denom > 0 else 0.0
    return pq, tp, fp, fn


def to_rles(masks_bool):
    return [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks_bool]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-score", type=float, default=0.2,
                   help="lowest score kept when caching predictions (sweep floor)")
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root) / "train"
    base = FilamentDataset(
        root / "train_images",
        root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    _, val_idx = split_by_filename(base, args.val_frac, args.seed)
    if args.limit:
        val_idx = val_idx[: args.limit]
    print(f"{len(val_idx)} validation samples")

    model = build_model(pretrained=False, img_size=args.img_size)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()
    model.roi_heads.score_thresh = args.min_score   # limits full-res mask pasting

    cache, t_inf = [], 0.0
    for n, idx in enumerate(val_idx, 1):
        s = base[idx]
        gt = s["masks"]
        gt = gt[gt.flatten(1).any(1).bool()].numpy().astype(bool)
        x = s["image"].repeat(3, 1, 1).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model([x])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0

        scores = out["scores"].cpu().numpy()
        if len(scores):
            pmt = make_disjoint(out["masks"][:, 0] >= args.mask_thresh, out["scores"])
            pm = pmt.cpu().numpy()
        else:
            pm = np.zeros((0,) + gt.shape[1:], bool)
        pred_rles, gt_rles = to_rles(pm), to_rles(gt)
        areas = np.array([mask_util.area(r) for r in pred_rles], dtype=np.float64)
        if len(pred_rles) and len(gt_rles):
            iou = np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)))
        else:
            iou = np.zeros((len(pred_rles), len(gt_rles)))
        cache.append({"iou": iou, "scores": scores, "areas": areas, "n_gt": len(gt_rles)})
        if n % 20 == 0 or n == len(val_idx):
            print(f"{n}/{len(val_idx)}", flush=True)

    print(f"\ninference (model forward only): {t_inf / len(val_idx):.2f}s/image")

    thresholds = [t for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if t >= args.min_score]
    areas = [0, 50, 200, 500, 1000]
    print("\nPQ  (rows: score threshold, cols: min area px)")
    print("      " + "".join(f"{a:>9}" for a in areas))
    best = (-1, None)
    for t in thresholds:
        row = []
        for a in areas:
            pq, tp, fp, fn = pq_from_cache(cache, t, a)
            row.append(pq)
            if pq > best[0]:
                best = (pq, (t, a, tp, fp, fn))
        print(f"{t:>5} " + "".join(f"{v:>9.4f}" for v in row))

    pq, (t, a, tp, fp, fn) = best
    print(f"\nbest PQ {pq:.4f} at score>={t}, min_area>={a}   (TP {tp}, FP {fp}, FN {fn})")
    print(f"use:  python inference.py ... --score-thresh {t} --min-area {a}")


if __name__ == "__main__":
    main()