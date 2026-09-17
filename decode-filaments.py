"""
Sep 11-12 checklist: polygon-to-mask decoding for MAGFiLO annotations.

Covers:
  1. polygon_to_mask()      - decode a single annotation's polygon into a binary mask
  2. assert_single_instance() - confirm each annotation is one filament (single-piece polygon)
  3. visualize_samples()    - overlay decoded masks on a handful of images for a visual check

Schema reminder (from the competition Data tab):
  - annotation["segmentation"] is a list containing exactly ONE polygon:
        [[x_0, y_0, x_1, y_1, ..., x_(n-1), y_(n-1)]]   with (x_0,y_0) == (x_(n-1),y_(n-1))
  - annotation["iscrowd"] is always 0 (no grouped/crowd annotations)
  - images are 2048x2048 grayscale JPEGs
  - image_id in the JSON is a string; annotations reference it via annotation["image_id"]
"""

import json
import numpy as np
from pycocotools import mask as mask_utils


def polygon_to_mask(segmentation, height, width):
    """
    Decode one annotation's polygon(s) into a single binary mask.

    Parameters
    ----------
    segmentation : list[list[float]]
        annotation["segmentation"] as given in the MAGFiLO JSON.
    height, width : int
        image dimensions (2048, 2048 for this dataset).

    Returns
    -------
    np.ndarray of shape (height, width), dtype uint8, values in {0, 1}
    """
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)  # merges to one RLE even if >1 polygon piece
    binary_mask = mask_utils.decode(rle)  # (H, W) uint8
    return binary_mask


def assert_single_instance(annotations):
    """
    Confirm every annotation maps to exactly one filament instance:
      - segmentation contains exactly one polygon (single-piece, per the schema)
      - iscrowd is 0 for all of them

    Raises AssertionError with the offending annotation id on the first violation.
    Returns the count checked on success.
    """
    for ann in annotations:
        seg = ann["segmentation"]
        assert isinstance(seg, list) and len(seg) == 1, (
            f"annotation {ann['id']} has {len(seg)} polygon pieces, expected 1"
        )
        assert ann.get("iscrowd", 0) == 0, (
            f"annotation {ann['id']} has iscrowd={ann.get('iscrowd')}, expected 0"
        )
        poly = seg[0]
        assert len(poly) % 2 == 0 and len(poly) >= 6, (
            f"annotation {ann['id']} polygon has {len(poly)} coords, "
            f"expected an even count >= 6 (>=3 points)"
        )
        assert (poly[0], poly[1]) == (poly[-2], poly[-1]), (
            f"annotation {ann['id']} polygon is not closed "
            f"(start {poly[0:2]} != end {poly[-2:]})"
        )
    return len(annotations)


def visualize_samples(json_path, images_dir, n_samples=8, out_path="mask_overlay_check.png"):
    """
    Decode masks for n_samples random images and save a grid of image+mask overlays
    for a quick visual sanity check.
    """
    import random
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path

    with open(json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    sample_ids = random.sample(list(anns_by_image.keys()), min(n_samples, len(anns_by_image)))

    cols = 4
    rows = (len(sample_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, image_id in zip(axes, sample_ids):
        img_meta = images_by_id[image_id]
        img_path = Path(images_dir) / img_meta["file_name"]
        img = np.array(Image.open(img_path).convert("L"))
        h, w = img_meta["height"], img_meta["width"]

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_image[image_id]:
            m = polygon_to_mask(ann["segmentation"], h, w)
            combined_mask = np.maximum(combined_mask, m)

        ax.imshow(img, cmap="gray")
        ax.imshow(combined_mask, cmap="Reds", alpha=0.4)
        ax.set_title(f"{image_id}\n{len(anns_by_image[image_id])} filament(s)", fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample_ids):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved visual check to {out_path}")


def _self_test():
    """Quick synthetic check that polygon_to_mask is correct, no real data needed."""
    # A 10x10 image with a 4x4 square filament, top-left corner at (2,2).
    h, w = 10, 10
    square = [2, 2, 6, 2, 6, 6, 2, 6, 2, 2]  # closed square polygon
    seg = [square]
    m = polygon_to_mask(seg, h, w)
    assert m.shape == (h, w)
    assert m.sum() == 16, f"expected 16 filled pixels, got {m.sum()}"
    assert m[3, 3] == 1 and m[0, 0] == 0
    assert_single_instance([{"id": "test-1", "segmentation": seg, "iscrowd": 0}])
    print("Self-test passed: polygon_to_mask and assert_single_instance behave as expected.")


if __name__ == "__main__":
    _self_test()

    # --- Fill these in and run against the real data ---
    JSON_PATH = "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    IMAGES_DIR = "MAGFiLO_1.0_Kaggle_2026/train/train_images"
    
    with open(JSON_PATH) as f:
        coco = json.load(f)
    n_checked = assert_single_instance(coco["annotations"])
    print(f"Checked {n_checked} annotations: all single-piece, iscrowd=0.")
    
    visualize_samples(JSON_PATH, IMAGES_DIR, n_samples=8)