# Solar Filament Segmentation Challenge 2026 — Submission Repository

**Team:** Aayusha Singh & Amrit Roy
**Advisor:** Prof. Azim Ahmadzadeh, University of Missouri–St. Louis
**Competition:** [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (IEEE BigData Cup 2026)

This repository is the code submission for the competition. It is structured to satisfy the
organizers' stated requirements:
- a `requirements.txt` with exact package versions,
- a single Jupyter notebook illustrating the **entire** pipeline end-to-end,
- everything needed to reproduce the submitted predictions, with no additional files required.

## 1. Task

Predict a separate instance segmentation mask for each solar filament visible in a
2048×2048 grayscale H-Alpha image (GONG network), including fine barb structures, while:
- keeping each filament as one coherent object (not fragmented across multiple predictions),
- not merging distinct filaments into a single prediction,
- separating true filament material from background sensor noise.

Full task description: see the competition's [Overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview) tab.

## 2. Evaluation

The organizers score submissions primarily on **Panoptic Quality (PQ)**, defined as:

```
PQ = SQ × RQ
SQ = mean IoU over matched true-positive pairs
RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
```

A predicted instance is matched to a ground-truth instance only when IoU > 0.5. Mean Dice
score, and the distributions of Dice/IoU and one-to-many/many-to-one match errors, are also
part of the rubric. See `src/metrics.py` for our local implementation, validated against the
organizers' [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).

We do not train or validate against any leaked test data referenced in the competition's
Discussion tab. All results reported here come from our own held-out split of the official
`train/` data.

## 3. Repository Contents

```
.
├── README.md                        # This file
├── requirements.txt                 # Exact package versions
├── pipeline.ipynb                   # Full end-to-end pipeline: data → train → eval → submission
├── src/
│   ├── dataset.py                     # Loads MAGFiLO COCO-style annotations into instance masks
│   ├── model.py                       # Instance segmentation architecture (pretrained backbone)
│   ├── losses.py                      # Fine-structure-aware loss functions
│   ├── metrics.py                     # Local PQ, Dice, IoU implementation
│   ├── postprocess.py                 # Fragmentation/over-merging mitigation
│   └── train.py                       # Training loop, checkpointing
├── inference.py                     # Generates predictions on test/ and writes the submission CSV
├── configs/
│   └── final_config.yaml              # Hyperparameters used for the submitted result
├── checkpoints/
│   └── best_model.pth                 # Final trained weights (or download link if too large for Git)
└── report.pdf                       # 4-page technical report (Overleaf template)
```

> This structure is the target layout for the final submission. See `CHECKLIST.md` (working
> document, not part of the submission) for what is built vs. still in progress.

## 4. Data

Download from the competition's Data tab ("Download All" → "Download as zip") and place as:
```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

Key data-handling decisions (see `src/dataset.py` for implementation):
- Images are 2048×2048 grayscale — loaded and processed as single-channel, never as RGB.
- **No deduplication by filename.** The same image can carry multiple independent
  annotator passes (e.g. `010101-...` vs `010102-...` for the same file); per the
  organizers' instructions, these are kept as separate training samples.
- Each annotation is a single-polygon mask for exactly one filament instance
  (`iscrowd` is always 0); `category_id` encodes chirality (Left/Right/Unidentifiable/
  Ambiguous), which is not used as a training signal for the mask itself.
- Only the images in `test/test_images/` are used at inference time; no external images
  or additional ground-truth metadata are used for training, per the competition rules.

## 5. Pipeline Description

*(Required by the rubric's qualitative criteria — kept in sync with `pipeline.ipynb`.)*

1. **Preprocessing** — grayscale normalization (percentile-based, robust to sensor
   artifacts), resizing/tiling strategy for memory constraints, instance mask construction
   from polygon annotations.
2. **Architecture** — [instance segmentation model + pretrained backbone; final choice and
   justification to be filled in here once locked, per the checklist].
3. **Loss function** — combination targeting both overall mask overlap and fine barb
   recovery [Tversky / clDice — final choice to be filled in once benchmarked].
4. **Training** — [optimizer, schedule, augmentation, cross-validation setup].
5. **Post-processing** — steps specifically added to reduce fragmentation and over-merging,
   since PQ penalizes both explicitly.
6. **Inference & submission generation** — `inference.py` runs the trained model on
   `test/test_images/`, applies the same post-processing as validation, and encodes each
   predicted instance as RLE (via `pycocotools`) into the required submission CSV.

## 6. Reproducing Results

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute pipeline.ipynb
```

`pipeline.ipynb` runs the full sequence — data loading, training (or loading
`checkpoints/best_model.pth` to skip retraining), local evaluation (PQ/Dice), and
submission-file generation — end-to-end, matching the organizers' requirement that no
additional file be needed to reproduce results.

## 7. Submission Format

One CSV row per predicted filament instance:

| filament_id | segmentation_rle |
|---|---|
| `20150125172714Mh_1` | RLE counts string |
| `20150125172714Mh_2` | RLE counts string |

- `filament_id`: image stem + a uniqueness suffix (`_1`, `_2`, ...).
- `segmentation_rle`: RLE **counts only** (mask size is fixed at 2048×2048, so it is not
  stored per row); no quotation marks. Generated via `pycocotools.mask.encode`.
- Predicted instance count per image does not need to match ground truth — matching is by
  overlap, not row order.

## 8. Results

| Config | Dice | IoU | PQ |
|---|---|---|---|
| _final submitted model_ | _TBD_ | _TBD_ | _TBD_ |

Full ablations and comparisons are in `report.pdf`.

## 9. Open-Access & License

This repository is made public per the competition's open-access requirement and will
remain public at least until winners are announced (Nov 30, 2026). License: _TBD_.

## 10. Citation

```
Azim Ahmadzadeh, Dustin J. Kempton, Qin Li, and Alexei A. Pevtsov.
Solar Filament Segmentation Challenge 2026.
https://kaggle.com/competitions/filament-segmentation-2026, 2026. Kaggle.
```