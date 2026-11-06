"""
End-to-end smoke test for FilamentDataset, run against a small synthetic
COCO-style annotation set + generated grayscale images -- stands in for
the real MAGFiLO_1.0_Kaggle_2026/train folder so the Dataset/DataLoader
wiring can be verified without the real (large) dataset present.

Swap `images_dir` / `annotation_json_path` for the real train folder and
JSON to run this against actual data.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from check_labels import JSON_PATH
from dataset import FilamentDataset, filament_collate_fn

H = W = 256  # small size for a fast synthetic test; real data is 2048x2048


def make_synthetic_dataset(root: Path, n_images=4):
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    images_meta = []
    annotations = []
    ann_id = 0
    chiralities = ["Left", "Right", "Unidentifiable", "Ambiguous"]

    for img_idx in range(n_images):
        file_name = f"img_{img_idx}.jpg"
        arr = (np.random.rand(H, W) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(images_dir / file_name)
        images_meta.append({"id": f"image_{img_idx}", "file_name": file_name})

        # a varying number of filament instances per image (0..3), to
        # exercise the "N varies per image" collate logic, including an
        # image with zero annotations (which should just be skipped).
        n_instances = img_idx % 4
        for _ in range(n_instances):
            x0, y0 = np.random.randint(0, W - 40), np.random.randint(0, H - 40)
            polygon = [x0, y0, x0 + 30, y0, x0 + 30, y0 + 20, x0, y0 + 20]
            annotations.append(
                {
                    "id": f"ann_{ann_id}",
                    "image_id": f"image_{img_idx}",
                    "segmentation": [polygon],
                    "chirality": chiralities[ann_id % len(chiralities)],
                }
            )
            ann_id += 1

    coco = {"images": images_meta, "annotations": annotations}
    ann_path = root / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    return images_dir, ann_path, n_images, len(annotations)


def main():
    tmp_root = Path(tempfile.mkdtemp())
    try:
        images_dir, ann_path, n_images, n_anns = make_synthetic_dataset(tmp_root)
        print(f"Synthetic set: {n_images} images, {n_anns} annotations "
              f"(one image has 0 annotations and should be excluded).")

        dataset = FilamentDataset(images_dir, ann_path)
        print(f"Dataset length (images with >=1 annotation): {len(dataset)}")
        assert len(dataset) == n_images - 1, "the zero-annotation image should be excluded"

        # single-sample check
        sample = dataset[0]
        print(f"Sample 0 -> image: {tuple(sample['image'].shape)}, "
              f"dtype {sample['image'].dtype}, "
              f"masks: {tuple(sample['masks'].shape)}, dtype {sample['masks'].dtype}, "
              f"labels: {sample['labels'].tolist()}")
        assert sample["image"].shape == (1, H, W)
        assert sample["masks"].shape[1:] == (H, W)
        assert sample["masks"].shape[0] == sample["labels"].shape[0]
        assert sample["image"].max() <= 1.0 and sample["image"].min() >= 0.0

        # end-to-end DataLoader check, with the batch containing images
        # that have different instance counts N
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=filament_collate_fn
        )
        batch = next(iter(loader))
        print(f"Batch image tensor: {tuple(batch['image'].shape)}")
        print(f"Batch masks (list of per-image tensors): "
              f"{[tuple(m.shape) for m in batch['masks']]}")
        print(f"Batch labels (list of per-image tensors): "
              f"{[l.tolist() for l in batch['labels']]}")
        assert batch["image"].shape == (2, 1, H, W)
        assert len(batch["masks"]) == 2
        assert len(batch["labels"]) == 2
        for m, l in zip(batch["masks"], batch["labels"]):
            assert m.shape[0] == l.shape[0]

        print("\nAll checks passed: Dataset + DataLoader run end-to-end, "
              "variable instance counts per image are handled correctly, "
              "zero-annotation images are excluded, grayscale/normalization "
              "and mask/label shapes are all as expected.")
    finally:
        shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()

from collections import Counter
from dataset import FilamentDataset

ds = FilamentDataset("MAGFiLO_1.0_Kaggle_2026/train", JSON_PATH)  # your real paths
counts = Counter()
for i in range(len(ds)):
    counts.update(ds[i]["labels"].tolist())
print(counts)  # expect roughly {1: ~2535, 2: ~2590, 3: ~3074}