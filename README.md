# Solar Filament Segmentation Challenge 2026 Submission Repository

**Team:** Aayusha Singh & Amrit Roy
**Advisor:** Prof. Azim Ahmadzadeh, University of Missouri–St. Louis
**Competition:** [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (IEEE BigData Cup 2026)

This repository contains our code for the competition: data loading, training, local
evaluation, inference, and submission-file generation for an instance segmentation model
(Mask R-CNN) that outlines each solar filament in a 2048×2048 H-Alpha image.

## 1. Task

Predict a separate instance mask for every filament in a grayscale H-Alpha image, keeping
each filament as one object, not merging distinct filaments, and separating filament
material from background noise. See the competition's Overview tab for the full description.

## 2. Evaluation

Submissions are scored mainly by **Panoptic Quality (PQ)**:

```
PQ = SQ × RQ
SQ = mean IoU over matched pairs
RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
```

A prediction matches a ground-truth filament when IoU > 0.5. Predicted masks within one
image **must not overlap**; the platform rejects the submission otherwise. Dice was the
original metric but was replaced by PQ, so we do not report it.

`validate.py` reimplements this formula (TP, FP and FN pooled over images, as in the
organizers' self-evaluation notebook, and checked on that notebook's toy example) and scores
a checkpoint on our held-out split.

We use only the official `train/` data for training and validation. All local results come
from our own held-out split of that data.

## 3. Repository contents

```
.
├── README.md
├── requirements.txt              # package versions used
├── dataset.py                    # loads MAGFiLO annotations into per-instance masks
├── model.py                      # Mask R-CNN builder, predict(), make_disjoint()
├── train.py                      # training loop, checkpoints, flip augmentation, resume
├── validate.py                   # local PQ + score-threshold / min-area sweep
├── inference.py                  # predicts on test/ and writes the submission CSV
├── check_submission.py           # checks the CSV (ids, RLE decoding, overlaps) before upload
├── diagnose.py                   # error breakdown by type and filament size
├── annotator_ceiling.py          # PQ between different annotators of the same image
├── self-evaluation-notebook.ipynb  # provided by the organizers
├── pipeline.ipynb                # [TODO] end-to-end notebook
├── checkpoints/                  # [TODO] final weights or a download link
└── report.pdf                    # 4-page report
```

## 4. Data

Download from the competition's Data tab and place as:

```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

Data-handling decisions:
- Images are 2048×2048 grayscale, scaled to [0, 1] and **replicated to three channels** to
  match the pretrained backbone. The detector applies ImageNet normalization and resizes
  the image internally to the network input size.
- **No deduplication by file name.** Repeated annotator passes of one image are kept as
  separate samples, as the organizers instruct.
- Each annotation is one polygon for one filament. `category_id` (chirality) is not used.
  Boxes are computed from the masks and every filament gets one class label.
- Validation split: 10% of the **files** (seed 0), so no image appears in both training and
  validation.

## 5. Pipeline

1. **Preprocessing:** as in section 4.
2. **Architecture:** torchvision Mask R-CNN, ResNet50-FPN backbone, COCO-pretrained, with
   new two-class box and mask predictors (background, filament).
3. **Loss:** the standard Mask R-CNN losses (RPN objectness and box, ROI classification and
   box, mask), summed. We did not implement Tversky or clDice losses.
4. **Training:** SGD (momentum 0.9, lr 0.005, weight decay 1e-4), linear warmup then cosine
   decay, batch size 1, mixed precision, gradient clipping at 10, random horizontal and
   vertical flips, [final: 20 epochs at 1024 px / 16 epochs at 1280 px].
5. **Post-processing:** keep predictions above a score threshold, binarize masks at 0.5,
   drop masks below a minimum area, and make masks disjoint by giving contested pixels to
   the higher-scoring instance. We have no separate step targeting fragmentation or
   over-merging.
6. **Inference:** `inference.py` writes one CSV row per predicted filament, with the mask
   encoded as compressed RLE (`pycocotools`).

## 6. Reproducing results

```bash
pip install -r requirements.txt

# train (about 90 min on a 6 GB laptop GPU at 1024 px)
python train.py --root <data path> --epochs 20 --img-size 1024 --batch-size 1 \
    --augment --out checkpoints_aug

# score a checkpoint and pick the score threshold / minimum area
python validate.py --root <data path> --ckpt checkpoints_aug/last.pth

# write and check the submission file
python inference.py --root <data path> --ckpt <checkpoint> \
    --score-thresh 0.9 --min-area 50 --out submission.csv
python check_submission.py --root <data path> --csv submission.csv
```

[TODO: `pipeline.ipynb` runs the same sequence end to end.]

## 7. Submission format

| filament_id | segmentation_rle |
|---|---|
| `<image_stem>_1` | RLE counts string |
| `<image_stem>_2` | RLE counts string |

`segmentation_rle` is the RLE counts string only (masks are 2048×2048), with no quotation
marks. Images with no kept prediction have no rows.

## 8. Results

| Model | Local PQ | Public PQ |
|---|---|---|
| Mask R-CNN, 12 epochs, no augmentation | 0.357 | 0.34 |
| + flip augmentation, 20 epochs (epoch 17) | 0.389 | 0.34 |
| [1280 px input, flips, 16 epochs] | [TBD] | [TBD] |

Inference takes 0.19 s per image (35 s for the 180 test images, including image loading and
RLE encoding) on a laptop RTX 4050 GPU with 6 GB of memory. Between two human annotators of
the same image we measure a PQ of 0.34 (`annotator_ceiling.py`). Details and caveats are in
`report.pdf`.

## 9. Open access and license

This repository is public per the competition's open-access requirement and will remain
public at least until winners are announced (Nov 30, 2026). License: [TBD].

## 10. Citation

```
Azim Ahmadzadeh, Dustin J. Kempton, Qin Li, and Alexei A. Pevtsov.
Solar Filament Segmentation Challenge 2026.
https://kaggle.com/competitions/filament-segmentation-2026, 2026. Kaggle.
```