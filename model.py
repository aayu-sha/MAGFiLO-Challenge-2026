"""
Sep 17-18 checklist item: instance-segmentation model with a pretrained backbone.

Architecture: torchvision Mask R-CNN, ResNet50-FPN, COCO-pretrained weights.
Single foreground class ("filament", label 1; 0 is background). Chirality is
not predicted - PQ only scores instance masks.

Notes:
  * The model's internal transform resizes every image to `img_size` for the
    network and maps predicted masks/boxes back to the ORIGINAL resolution
    (2048x2048), so predictions are directly comparable to full-size GT.
  * Input must be 3-channel; grayscale is replicated by the caller.
  * Known weakness (put in the report's Limitations): the mask head outputs
    28x28 per RoI, which limits thin-barb detail.
"""

import torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + filament


def build_model(pretrained=True, img_size=1024, max_detections=100):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        min_size=img_size,
        max_size=img_size,
        box_detections_per_img=max_detections,
        box_score_thresh=0.05,  # keep low here; tune the real threshold on val PQ
    )

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)

    in_ch = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, NUM_CLASSES)
    return model


def make_disjoint(masks, scores):
    """
    Kaggle rejects submissions where predicted masks of one image overlap.
    masks: BoolTensor [K,H,W]; scores: FloatTensor [K].
    Higher-score instances keep contested pixels. Because priority follows score,
    dropping low-score instances later never changes the pixels of higher-score ones,
    so threshold sweeps on cached predictions stay valid.
    """
    if len(masks) == 0:
        return masks
    out = torch.zeros_like(masks)
    occupied = torch.zeros_like(masks[0])
    for j in torch.argsort(scores, descending=True).tolist():
        m = masks[j] & ~occupied
        out[j] = m
        occupied |= m
    return out


@torch.no_grad()
def predict(model, images, score_thresh=0.5, mask_thresh=0.5):
    """
    images: list of FloatTensor [3, H, W] in [0, 1].
    Returns a list (one per image) of dicts:
        masks  : BoolTensor  [K, H, W]  (original resolution, mutually disjoint)
        scores : FloatTensor [K]
    """
    model.eval()
    outs = model(images)
    results = []
    for o in outs:
        keep = o["scores"] >= score_thresh
        scores = o["scores"][keep]
        masks = make_disjoint(o["masks"][keep, 0] >= mask_thresh, scores)
        results.append({"masks": masks.cpu(), "scores": scores.cpu()})
    return results