"""
Sep 19-22 checklist items: training loop (sanity run + full run).

Usage (from the repo root, next to dataset.py):

  # 1) tiny sanity run - proves the pipeline works end to end
  python train.py --root /path/to/MAGFiLO_1.0_Kaggle_2026 --subset 16 --epochs 2 --img-size 512

  # 2) full run
  python train.py --root /path/to/MAGFiLO_1.0_Kaggle_2026 --epochs 12 --img-size 1024

Validation split is by FILE NAME, so repeated annotator passes of the same
image never straddle train/val. Val loss is only a proxy; model selection
by PQ comes with the metrics step.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import FilamentDataset
from model import build_model


class MaskRCNNView(Dataset):
    """Adapts FilamentDataset samples to torchvision's detection target format."""

    def __init__(self, base, augment=False, flags=None):
        self.base = base
        self.augment = augment
        self.flags = flags                          # consensus keep-flags per base index, or None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        img = s["image"].repeat(3, 1, 1)            # [1,H,W] -> [3,H,W]
        masks = s["masks"]                          # [N,H,W] uint8
        if self.flags is not None and i in self.flags:
            masks = masks[torch.from_numpy(self.flags[i])]   # keep only filaments another annotator also drew
        if self.augment:                            # flips are valid for class-agnostic masks
            if random.random() < 0.5:
                img, masks = img.flip(-1), masks.flip(-1)
            if random.random() < 0.5:
                img, masks = img.flip(-2), masks.flip(-2)
        keep = masks.flatten(1).any(1).bool()       # drop empty masks
        masks = masks[keep]

        boxes = []
        for m in masks:
            ys = torch.where(m.any(1))[0]
            xs = torch.where(m.any(0))[0]
            boxes.append([xs[0].item(), ys[0].item(),
                          xs[-1].item() + 1, ys[-1].item() + 1])
        boxes = (torch.tensor(boxes, dtype=torch.float32)
                 if boxes else torch.zeros((0, 4), dtype=torch.float32))

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),  # class-agnostic
            "masks": masks,
        }
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def split_by_filename(base, val_frac, seed):
    groups = {}
    for idx, img_id in enumerate(base.image_ids):
        groups.setdefault(base.images[img_id]["file_name"], []).append(idx)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    n_val = max(1, int(len(names) * val_frac))
    val_idx = [i for n in names[:n_val] for i in groups[n]]
    train_idx = [i for n in names[n_val:] for i in groups[n]]
    return train_idx, val_idx


def compute_consensus(base, indices, iou_thr, cache_path):
    """
    For each training sample, flag which filaments another annotator pass of the SAME image
    also drew (IoU > iou_thr with any filament in another pass). Files with a single pass
    have no one to agree with, so all their filaments are kept.
    Returns {base_index: bool array, one entry per raw annotation}. Cached as JSON.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        raw = json.load(open(cache_path))
        return {int(k): np.array(v, dtype=bool) for k, v in raw.items()}

    groups = {}
    for idx in indices:
        groups.setdefault(base.images[base.image_ids[idx]]["file_name"], []).append(idx)

    flags = {}
    for n, idxs in enumerate(groups.values(), 1):
        per = []
        for i in idxs:
            m = base[i]["masks"]
            m = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
            per.append([mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in m.astype(bool)])
        for a, i in enumerate(idxs):
            agreed = np.zeros(len(per[a]), dtype=bool)
            if len(idxs) == 1:
                agreed[:] = True                      # nobody to compare with: keep everything
            else:
                for b in range(len(idxs)):
                    if b == a or not per[b] or not per[a]:
                        continue
                    iou = np.asarray(mask_util.iou(per[a], per[b], [0] * len(per[b])))
                    agreed |= (iou > iou_thr).any(axis=1)
            flags[i] = agreed
        if n % 100 == 0:
            print(f"  consensus: {n}/{len(groups)} files", flush=True)

    json.dump({str(k): v.tolist() for k, v in flags.items()}, open(cache_path, "w"))
    return flags


def safe_save(obj, path):
    """Save a checkpoint; if the disk is full, warn and carry on instead of killing the run."""
    try:
        torch.save(obj, path)
    except (RuntimeError, OSError) as e:
        print(f"WARNING: could not save {path} (disk full?): {e}", flush=True)
        try:
            Path(path).unlink()
        except OSError:
            pass


def to_device(images, targets, device):
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


@torch.no_grad()
def val_loss(model, loader, device):
    # torchvision detection returns losses only in train mode; its BatchNorm
    # layers are frozen, so this does not disturb any statistics.
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = to_device(images, targets, device)
        total += sum(model(images, targets).values()).item()
        n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--subset", type=int, default=0, help="use only N train images (sanity run)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--augment", action="store_true", help="random h/v flips on training images")
    p.add_argument("--consensus", action="store_true",
                   help="train only on filaments that another annotator of the same image also drew")
    p.add_argument("--consensus-iou", type=float, default=0.5)
    p.add_argument("--anchors", choices=["default", "small"], default="default")
    p.add_argument("--save-every", type=int, default=1, help="keep a per-epoch checkpoint every N epochs (saves disk)")
    p.add_argument("--resume", default="", help="path to out/last_full.pth to continue a run")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.root) / "train"
    base = FilamentDataset(
        root / "train_images",
        root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    train_idx, val_idx = split_by_filename(base, args.val_frac, args.seed)
    if args.subset:
        train_idx, val_idx = train_idx[: args.subset], val_idx[: max(2, args.subset // 4)]
    flags = None
    if args.consensus:
        flags = compute_consensus(base, train_idx, args.consensus_iou,
                                  f"consensus_iou{args.consensus_iou}_n{len(train_idx)}.json")
        kept = sum(int(f.sum()) for f in flags.values())
        total = sum(len(f) for f in flags.values())
        dropped = {i for i in train_idx if len(flags[i]) and not flags[i].any()}
        train_idx = [i for i in train_idx if i not in dropped]
        print(f"consensus: kept {kept}/{total} filament labels; "
              f"dropped {len(dropped)} samples with no agreed filament")
    print(f"train samples: {len(train_idx)}  val samples: {len(val_idx)}")

    view_train = MaskRCNNView(base, augment=args.augment, flags=flags)
    view_val = MaskRCNNView(base, augment=False)
    kw = dict(collate_fn=collate, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(view_train, train_idx), batch_size=args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(Subset(view_val, val_idx), batch_size=args.batch_size, shuffle=False, **kw)

    model = build_model(pretrained=not args.no_pretrained, img_size=args.img_size,
                        anchors=args.anchors).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_iters = args.epochs * len(train_loader)
    warmup = min(100, max(1, total_iters // 10))

    def lr_lambda(it):
        if it < warmup:
            return (it + 1) / warmup
        prog = (it - warmup) / max(1, total_iters - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, start_epoch = float("inf"), 1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}, best val loss {best:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for it, (images, targets) in enumerate(train_loader, 1):
            images, targets = to_device(images, targets, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            n += 1
            if it % 50 == 0:
                el = time.time() - t0
                eta = el / it * (len(train_loader) - it)
                print(f"  epoch {epoch} it {it}/{len(train_loader)}  loss {run / n:.4f}  "
                      f"{el / it:.2f}s/it  epoch ETA {eta / 60:.1f} min", flush=True)
        vl = val_loss(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}  train_loss {run / max(n, 1):.4f}  "
              f"val_loss {vl:.4f}  time {time.time() - t0:.0f}s", flush=True)

        safe_save(model.state_dict(), out / "last.pth")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            safe_save(model.state_dict(), out / f"epoch_{epoch:02d}.pth")   # pick by PQ, not loss
        if vl < best:
            best = vl
            safe_save(model.state_dict(), out / "best_val_loss.pth")
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best}, out / "last_full.pth")
    print("done. best val loss:", round(best, 4))


if __name__ == "__main__":
    main()
