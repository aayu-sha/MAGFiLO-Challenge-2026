"""
Sep 13-14 checklist: data analysis feeding into architecture choice.

Covers:
  1. compute_annotator_disagreement() - for images annotated by 2+ people,
     match filament instances between annotators (PQ-style: IoU>0.5, greedy
     one-to-one) and report SQ/RQ/PQ as a disagreement measure.
  2. compute_touch_overlap()          - within a single annotator's own
     annotation set for one image, check how often two filaments overlap
     (share pixels) or merely touch (adjacent after 1px dilation, no overlap).

Both results matter for architecture choice: heavy disagreement argues for
treating the label as "probability a typical annotator marks this pixel"
rather than a single ground truth; frequent touching/overlap argues for an
instance-aware head instead of semantic-segmentation + connected components.

Self-contained: duplicates the small polygon_to_mask helper from
decode_filament_masks.py so this script has no fragile cross-file import.
"""

import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils


def polygon_to_mask(segmentation, height, width):
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)


def get_instance_rles(annotations, height, width):
    """RLE (not decoded) per annotation -- cheaper for IoU matrices than full masks."""
    rles = []
    for ann in annotations:
        r = mask_utils.frPyObjects(ann["segmentation"], height, width)
        rles.append(mask_utils.merge(r))
    return rles


def dilate(mask, px=1):
    """4-neighborhood dilation by `px` pixels, no scipy dependency."""
    d = mask.astype(bool)
    for _ in range(px):
        shifted = np.zeros_like(d)
        shifted[1:, :] |= d[:-1, :]
        shifted[:-1, :] |= d[1:, :]
        shifted[:, 1:] |= d[:, :-1]
        shifted[:, :-1] |= d[:, 1:]
        d = d | shifted
    return d


def find_repeated_images(coco):
    """Group image entries by file_name; keep groups with 2+ annotator versions."""
    groups = defaultdict(list)
    for img in coco["images"]:
        groups[img["file_name"]].append(img)
    return {fn: imgs for fn, imgs in groups.items() if len(imgs) > 1}


def compute_annotator_disagreement(coco, iou_thresh=0.5):
    """
    For every pair of annotators who both annotated the same physical image,
    match their filament instances (IoU > iou_thresh, greedy one-to-one,
    highest IoU first) and compute PQ-style SQ/RQ/PQ for that pair.

    Returns a list of per-pair result dicts.
    """
    anns_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    repeated = find_repeated_images(coco)
    results = []

    for file_name, imgs in repeated.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                img_a, img_b = imgs[i], imgs[j]
                anns_a = anns_by_image_id.get(img_a["id"], [])
                anns_b = anns_by_image_id.get(img_b["id"], [])
                h, w = img_a["height"], img_a["width"]

                if anns_a and anns_b:
                    rles_a = get_instance_rles(anns_a, h, w)
                    rles_b = get_instance_rles(anns_b, h, w)
                    iou_mat = np.array(mask_utils.iou(rles_a, rles_b, [0] * len(rles_b)))
                else:
                    iou_mat = np.zeros((len(anns_a), len(anns_b)))

                # greedy one-to-one matching, highest IoU first (unique because
                # IoU > 0.5 rules out any instance matching two others at once
                # for non-degenerate, non-overlapping ground truth)
                pairs = [
                    (iou_mat[a, b], a, b)
                    for a in range(iou_mat.shape[0])
                    for b in range(iou_mat.shape[1])
                    if iou_mat[a, b] > iou_thresh
                ]
                pairs.sort(reverse=True)
                used_a, used_b, matched_ious = set(), set(), []
                for iou_val, a_idx, b_idx in pairs:
                    if a_idx in used_a or b_idx in used_b:
                        continue
                    used_a.add(a_idx)
                    used_b.add(b_idx)
                    matched_ious.append(iou_val)

                tp = len(matched_ious)
                fp = len(anns_b) - tp
                fn = len(anns_a) - tp
                sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
                rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 1.0

                results.append({
                    "file_name": file_name,
                    "annotator_a": img_a["id"],
                    "annotator_b": img_b["id"],
                    "n_a": len(anns_a), "n_b": len(anns_b),
                    "tp": tp, "fp": fp, "fn": fn,
                    "sq": sq, "rq": rq, "pq": sq * rq,
                })

    return results


def summarize_disagreement(results):
    if not results:
        return {"n_pairs": 0}
    pqs = [r["pq"] for r in results]
    sqs = [r["sq"] for r in results if r["tp"] > 0]
    rqs = [r["rq"] for r in results]
    return {
        "n_pairs": len(results),
        "mean_pq": float(np.mean(pqs)),
        "mean_sq": float(np.mean(sqs)) if sqs else 0.0,
        "mean_rq": float(np.mean(rqs)),
        "min_pq": float(np.min(pqs)),
        "max_pq": float(np.max(pqs)),
    }


def compute_touch_overlap(coco, dilation_px=1):
    """
    Within each single annotator's own filament set for one image, check every
    pair of instances for (a) true pixel overlap, (b) touching without overlap
    (adjacent after `dilation_px` dilation).
    """
    anns_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)
    images_by_id = {img["id"]: img for img in coco["images"]}

    n_images_checked = 0
    n_images_with_overlap = 0
    n_images_with_touch_only = 0
    total_pairs = overlap_pairs = touch_pairs = 0

    for image_id, anns in anns_by_image_id.items():
        if len(anns) < 2:
            continue
        img = images_by_id[image_id]
        h, w = img["height"], img["width"]
        masks = [polygon_to_mask(a["segmentation"], h, w) for a in anns]
        n_images_checked += 1

        has_overlap = has_touch = False
        dilated = [dilate(m, dilation_px) for m in masks]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                total_pairs += 1
                if np.logical_and(masks[i], masks[j]).any():
                    overlap_pairs += 1
                    has_overlap = True
                elif np.logical_and(dilated[i], masks[j]).any():
                    touch_pairs += 1
                    has_touch = True

        if has_overlap:
            n_images_with_overlap += 1
        elif has_touch:
            n_images_with_touch_only += 1

    return {
        "n_images_checked": n_images_checked,
        "n_images_with_overlap": n_images_with_overlap,
        "n_images_with_touch_only": n_images_with_touch_only,
        "total_pairs_checked": total_pairs,
        "overlap_pairs": overlap_pairs,
        "touch_pairs": touch_pairs,
    }


def _self_test():
    """Synthetic sanity check, no real data needed."""
    h, w = 20, 20
    sq_a = [2, 2, 6, 2, 6, 6, 2, 6, 2, 2]
    sq_b_close = [3, 2, 7, 2, 7, 6, 3, 6, 3, 2]     # shifted +1px, decent overlap
    sq_c_far = [15, 15, 18, 15, 18, 18, 15, 18, 15, 15]  # no overlap with A
    sq_touch = [6, 2, 8, 2, 8, 6, 6, 6, 6, 2]        # touches A's right edge, no overlap

    coco = {
        "images": [
            {"id": "010401-img1", "file_name": "img1.jpeg", "height": h, "width": w},
            {"id": "010402-img1", "file_name": "img1.jpeg", "height": h, "width": w},  # repeat
            {"id": "010401-img2", "file_name": "img2.jpeg", "height": h, "width": w},  # not repeated
        ],
        "annotations": [
            {"id": "a1", "image_id": "010401-img1", "segmentation": [sq_a], "iscrowd": 0},
            {"id": "a2", "image_id": "010402-img1", "segmentation": [sq_b_close], "iscrowd": 0},
            {"id": "a3", "image_id": "010402-img1", "segmentation": [sq_c_far], "iscrowd": 0},
            {"id": "b1", "image_id": "010401-img2", "segmentation": [sq_a], "iscrowd": 0},
            {"id": "b2", "image_id": "010401-img2", "segmentation": [sq_touch], "iscrowd": 0},
        ],
    }

    repeated = find_repeated_images(coco)
    assert list(repeated.keys()) == ["img1.jpeg"], repeated

    disagreement = compute_annotator_disagreement(coco)
    assert len(disagreement) == 1
    r = disagreement[0]
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0, r  # A's square matches B's shifted one; B's far square is a false positive

    overlap = compute_touch_overlap(coco)
    # img1 (annotator 010402: sq_b_close + sq_c_far, far apart) and img2 (sq_a + sq_touch)
    # both have >=2 instances from a single annotator, so both get checked.
    assert overlap["n_images_checked"] == 2, overlap
    assert overlap["n_images_with_touch_only"] == 1, overlap  # img2 only
    assert overlap["n_images_with_overlap"] == 0, overlap

    print("Self-test passed: disagreement matching and touch/overlap detection behave as expected.")


if __name__ == "__main__":
    _self_test()

    # --- Fill these in and run against the real data ---
    JSON_PATH = "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    
    with open(JSON_PATH) as f:
        coco = json.load(f)
    
    results = compute_annotator_disagreement(coco)
    summary = summarize_disagreement(results)
    print("Inter-annotator disagreement:", summary)
    
    overlap = compute_touch_overlap(coco)
    print("Touch/overlap within single annotation sets:", overlap)