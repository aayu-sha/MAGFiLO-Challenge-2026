"""
How well do two human annotators agree with each other on the same image?
Scores annotator A's masks as 'predictions' against annotator B's as 'ground truth'
with the same PQ as the contest. CPU only, no model needed.

  python annotator_ceiling.py --root <MAGFiLO_1.0_Kaggle_2026 path>
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from dataset import FilamentDataset

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
a = p.parse_args()

root = Path(a.root) / "train"
base = FilamentDataset(root / "train_images", root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

groups = {}
for idx, img_id in enumerate(base.image_ids):
    groups.setdefault(base.images[img_id]["file_name"], []).append(idx)
multi = {k: v for k, v in groups.items() if len(v) > 1}
print(f"files: {len(groups)}   with 2+ annotation passes: {len(multi)}   "
      f"samples in those files: {sum(len(v) for v in multi.values())}")


def rles(m):
    return [mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in m]


tp_ious, fp, fn, count_gap = [], 0, 0, []
for n, (name, idxs) in enumerate(multi.items(), 1):
    per = []
    for i in idxs:
        m = base[i]["masks"]
        m = m[m.flatten(1).any(1).bool()].numpy().astype(bool)
        per.append(rles(m))
    for x, y in itertools.permutations(range(len(per)), 2):
        pa, gb = per[x], per[y]
        iou = (np.asarray(mask_util.iou(pa, gb, [0] * len(gb))) if len(pa) and len(gb)
               else np.zeros((len(pa), len(gb))))
        hit = iou > 0.5
        tp_ious += iou[hit].tolist()
        fp += int((hit.sum(axis=1) == 0).sum())
        fn += int((hit.sum(axis=0) == 0).sum())
    count_gap.append(max(len(r) for r in per) - min(len(r) for r in per))
    if n % 50 == 0:
        print(f"{n}/{len(multi)}", flush=True)

tp = len(tp_ious)
pq = sum(tp_ious) / (tp + 0.5 * fp + 0.5 * fn)
print(f"\nHuman-vs-human PQ (the realistic ceiling): {pq:.4f}")
print(f"  TP {tp}  FP {fp}  FN {fn}   mean IoU of matched filaments {np.mean(tp_ious):.3f}")
print(f"  filaments one annotator drew that another did not (per pair): "
      f"mean count gap between annotators of the same image = {np.mean(count_gap):.1f}")
