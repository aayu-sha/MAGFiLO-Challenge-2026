"""
Shrink the final checkpoint to fp16 so it can live inside the repository
(GitHub rejects files over 100 MB; the fp32 file is about 170 MB, fp16 about 85 MB).
The weights are cast back to fp32 when loaded, so nothing else in the code changes.

  python make_release_weights.py --src New/checkpoints_1280/epoch_16.pth \
      --dst weights/maskrcnn_1280_final_fp16.pth
"""
import argparse
from pathlib import Path

import torch

from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", default="weights/maskrcnn_1280_final_fp16.pth")
p.add_argument("--img-size", type=int, default=1280)
a = p.parse_args()

sd = torch.load(a.src, map_location="cpu")
half = {}
for k, v in sd.items():
    if v.is_floating_point():
        h = v.half()
        assert torch.isfinite(h).all(), f"{k} overflows in fp16"
        half[k] = h
    else:
        half[k] = v

dst = Path(a.dst)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(half, dst)
mb = dst.stat().st_size / 1e6
print(f"saved {dst}  ({mb:.1f} MB)")
assert mb < 95, "still too large for GitHub's 100 MB limit"

# the file must load into the model, and the weights must be close to the originals
model = build_model(pretrained=False, img_size=a.img_size)
model.load_state_dict(torch.load(dst, map_location="cpu"))
new = model.state_dict()
worst = max((new[k].float() - sd[k].float()).abs().max().item() for k in sd if sd[k].is_floating_point())
print(f"loads into the model; largest absolute weight difference vs fp32: {worst:.2e}")