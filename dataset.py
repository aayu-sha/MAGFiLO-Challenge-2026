"""
Sep 15-16 checklist item: PyTorch Dataset class for MAGFiLO filament
instance masks.

Representation decision: a STACK OF BINARY MASKS, shape [N, H, W]
(one (H, W) uint8 mask per filament instance in the image), rather
than a single labeled array [H, W].

Why:
  - The Sep 13-14 analysis found filaments touch/overlap within the
    same image. A labeled array can hold only one instance id per
    pixel, so an overlap silently clobbers one instance's mask. A
    mask stack keeps every instance's full extent intact, including
    overlapping pixels.
  - Instance-segmentation architectures (torchvision Mask R-CNN and
    similar) expect exactly this [N, H, W] format as the target.
  - The PQ implementation matches predicted/ground-truth instances
    pairwise by IoU, which is computed mask-by-mask - a stack maps
    onto that directly with no unpacking step.
  - Cost is small: N is on the order of single digits to low tens of
    filaments per image, so N * H * W uint8 masks are a minor
    overhead next to the image itself.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as mask_utils

# Four chirality categories per the annotation schema (Sep 5-6 notes).
# 0 is reserved as the "no chirality info" / background sentinel so real
# categories start at 1 -- convenient if a model ever needs an unlabeled
# class alongside these.
CHIRALITY_TO_ID = {
    "Left": 1,
    "Right": 2,
    "Unidentifiable": 3,
    "Ambiguous": 4,
}


def polygon_to_mask(segmentation, height, width):
    """
    Decode one annotation's polygon(s) into a single binary mask.

    Same function as the Sep 11-12 step: `segmentation` is
    annotation["segmentation"] as-is (a list containing one polygon,
    per the MAGFiLO schema -- iscrowd is always 0, single-piece).
    Uses pycocotools rather than a hand-rolled rasterizer so mask
    decoding stays identical to what assert_single_instance() and
    visualize_samples() already validated.

    Returns np.ndarray, shape (height, width), dtype uint8, values in {0, 1}.
    """
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)


class FilamentDataset(Dataset):
    """
    One sample = one image, with all of its filament instances.

    Returns a dict:
        image    : FloatTensor [1, H, W], pixel values scaled to [0, 1]
        masks    : UInt8Tensor [N, H, W] - one binary mask per instance
        labels   : LongTensor  [N]       - chirality id per instance
        image_id : str (or whatever id type the JSON uses)
        ann_ids  : list        - annotation ids, same order as masks/labels
    """

    def __init__(self, images_dir, annotation_json_path):
        self.images_dir = Path(images_dir)

        with open(annotation_json_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}

        # Map COCO category ids -> our label ids via the category names
        # (the annotations have no 'chirality' field; chirality IS the category).
        self.cat_id_to_label = {
            c["id"]: CHIRALITY_TO_ID[c["name"]] for c in coco["categories"]
        }

        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images that have at least one annotation -- an image
        # with zero filaments would produce an empty [0, H, W] mask stack,
        # which is valid but not useful for training right now.
        self.image_ids = [
            img_id for img_id in self.images if img_id in self.anns_by_image
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_meta = self.images[image_id]
        anns = self.anns_by_image[image_id]

        img_path = self.images_dir / img_meta["file_name"]
        # Force grayscale ('L'), matching the Sep 5-6 finding that these
        # load as single-channel and must NOT be treated as RGB.
        image = Image.open(img_path).convert("L")
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
        h, w = image_np.shape

        masks, labels, ann_ids = [], [], []
        for ann in anns:
            masks.append(polygon_to_mask(ann["segmentation"], h, w))
            labels.append(self.cat_id_to_label[ann["category_id"]])
            ann_ids.append(ann["id"])

        masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "image": image_tensor,
            "masks": masks_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "ann_ids": ann_ids,
        }


def filament_collate_fn(batch):
    """
    N (instance count) varies per image, so masks/labels can't be
    stacked into one tensor across a batch the way torch's default
    collate expects. Images are all the same fixed size (2048x2048),
    so those still stack normally.
    """
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),  # [B,1,H,W]
        "masks": [b["masks"] for b in batch],      # list of [N_i, H, W]
        "labels": [b["labels"] for b in batch],    # list of [N_i]
        "image_id": [b["image_id"] for b in batch],
        "ann_ids": [b["ann_ids"] for b in batch],
    }