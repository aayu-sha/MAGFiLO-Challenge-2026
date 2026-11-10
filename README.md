# Solar Filament Segmentation Challenge 2026: Submission Repository

**Team:** Aayusha Singh & Amrit Roy
**Advisor:** Prof. Azim Ahmadzadeh, University of Missouri–St. Louis
**Competition:** [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (IEEE BigData Cup 2026)

Code for our entry: data loading, training, local evaluation, inference and
submission-file generation for a Mask R-CNN instance segmentation model that outlines each
solar filament in a 2048×2048 H-Alpha image. The 4-page report is `report.pdf`.

**Final model:** Mask R-CNN (ResNet50-FPN, COCO-pretrained), flip augmentation, 1280 px input,
16 epochs, score threshold 0.8, minimum mask area 200 px, overlaps removed.
**Public leaderboard PQ: 0.36.** Inference: 0.29 s per image on a laptop RTX 4050 (6 GB).

## 1. Evaluation

Submissions are scored mainly by **Panoptic Quality (PQ)**:

```
PQ = SQ × RQ
SQ = mean IoU over matched pairs
RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
```

A prediction matches a ground-truth filament when IoU > 0.5. Predicted masks within one image
**must not overlap**; the platform rejects the submission otherwise. `validate.py` reimplements
this formula (TP, FP and FN pooled over images, as in the organizers' self-evaluation notebook)
and scores a checkpoint on our held-out split. Dice was replaced by PQ, so we do not report it.

We use only the official `train/` data for training and validation.

## 2. Repository contents

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── report.pdf                      # 4-page report
├── pipeline.ipynb                  # end-to-end notebook (validate, predict, check, plot)
├── weights/
│   └── maskrcnn_1280_final_fp16.pth  # final model weights (fp16, 85 MB), loaded by the notebook
├── make_release_weights.py         # converts a trained checkpoint to the stored fp16 weights
├── dataset.py                      # loads MAGFiLO annotations into per-instance masks
├── model.py                        # Mask R-CNN builder, predict(), make_disjoint()
├── train.py                        # training loop (flips, resume, optional consensus / anchors)
├── validate.py                     # local PQ + score-threshold / min-area sweep
├── inference.py                    # predicts on test/ and writes the submission CSV
├── check_submission.py             # checks the CSV (ids, RLE decoding, overlaps) before upload
├── submission_1280.csv             # final submission file (public PQ 0.36)
├── self-evaluation-notebook.ipynb  # provided by the organizers
│
├── diagnose.py                     # error breakdown by type and filament size
├── annotator_ceiling.py            # PQ between different annotators of the same image
├── check_anchors.py                # how well default anchors fit our filament boxes
├── check_offdisk.py                # are any predictions outside the solar disk?
├── make_figure.py                  # annotation vs prediction figure
├── dataanalysis.py                 # early analysis: annotator disagreement, touching/overlap
├── decode-filaments.py             # polygon-to-mask decoding and visual check
├── test_dataset.py                 # synthetic smoke test for dataset.py
├── check_labels.py                 # checks which fields annotations have (is chirality stored as chirality or category_id?)
├── mask_overlay_check.png          # output of the visual check
└── logs/                           # training logs of every run in the report
```

## 3. Data

Download from the competition's Data tab (the data are **not** included here) and place as:

```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

- Images are 2048×2048 grayscale, scaled to [0, 1] and **replicated to three channels** to match
  the pretrained backbone. The detector applies ImageNet normalization and resizes the image
  internally to the network input size.
- **No deduplication by file name.** Repeated annotator passes of one image are kept as separate
  samples, as the organizers instruct.
- Each annotation is one polygon for one filament. `category_id` (chirality) is not used.
  Boxes are computed from the masks and every filament gets one class label.
- Validation split: 10% of the **files** (seed 0), so no image is in both training and validation.

## 4. Method

1. **Architecture:** torchvision Mask R-CNN, ResNet50-FPN, COCO-pretrained, new two-class
   box and mask predictors (background, filament).
2. **Loss:** the standard Mask R-CNN losses, summed. No custom loss.
3. **Training:** SGD (momentum 0.9, lr 0.005, weight decay 1e-4), linear warmup then cosine
   decay, batch size 1, mixed precision, gradient clipping at 10, random horizontal and vertical
   flips, 16 epochs at 1280 px.
4. **Post-processing:** keep predictions with score >= 0.8, binarize masks at 0.5, drop masks
   smaller than 200 px, and make masks disjoint by giving contested pixels to the
   higher-scoring instance. No separate step targets fragmentation or over-merging.
5. **Inference:** one CSV row per predicted filament, mask encoded as compressed RLE (`pycocotools`).

## 5. Reproducing the results

```bash
pip install -r requirements.txt

# train the final model (about 2 hours on a 6 GB laptop GPU)
python train.py --root <data path> --epochs 16 --img-size 1280 --batch-size 1 \
    --augment --out checkpoints_1280

# score it (also prints the best score threshold / minimum area)
python validate.py --root <data path> --ckpt weights/maskrcnn_1280_final_fp16.pth --img-size 1280

# write and check the submission file
python inference.py --root <data path> --ckpt weights/maskrcnn_1280_final_fp16.pth --img-size 1280 \
    --score-thresh 0.8 --min-area 200 --out submission.csv
python check_submission.py --root <data path> --csv submission.csv
```

`pipeline.ipynb` runs the validation, prediction and checking steps end to end.

**Weights:** the final model is stored in this repository at `weights/maskrcnn_1280_final_fp16.pth` (fp16 to stay
under GitHub's 100 MB file limit; cast to fp32 on loading). No training is needed to reproduce the predictions:
`pipeline.ipynb` loads these weights, scores the validation split, writes `submission.csv` and compares it with
`submission_1280.csv`, the file we uploaded to Kaggle. Only the competition data (not included) must be downloaded.

## 6. Submission format

| filament_id      | segmentation_rle  |
|------------------|-------------------|
| `<image_stem>_1` | RLE counts string |
| `<image_stem>_2` | RLE counts string |

`segmentation_rle` is the RLE counts string only (masks are 2048×2048), with no quotation marks.
Images with no kept prediction have no rows.

## 7. Results

Local PQ is on our validation split (116 samples) with disjoint masks; each row uses its best
score threshold. Public PQ is the Kaggle leaderboard value.

| Model | Local PQ | Public PQ |
|---|---|---|
| Mask R-CNN, 1024 px, 12 epochs, no augmentation | 0.357 | 0.34 |
| + flip augmentation, 20 epochs (epoch 17) | 0.389 | 0.34 |
| **1280 px, flips, 16 epochs (final)** | **0.393** | **0.36** |
| 1280 px + consensus labels (not submitted) | 0.388 | not submitted |
| 1280 px + small anchors (not submitted) | 0.387 | not submitted |

Between two human annotators of the same image we measure a PQ of 0.34
(`annotator_ceiling.py`). Differences of about 0.01 in local PQ are within the noise of our small
validation split. Details and caveats are in `report.pdf`.

## 8. Open access and license

This repository is public per the competition's open-access requirement and will stay public at
least until winners are announced (Nov 30, 2026). License: MIT (see `LICENSE`).

## 9. Citation

```
Azim Ahmadzadeh, Dustin J. Kempton, Qin Li, and Alexei A. Pevtsov.
Solar Filament Segmentation Challenge 2026.
https://kaggle.com/competitions/filament-segmentation-2026, 2026. Kaggle.
```