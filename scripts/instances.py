"""Person instance masks. Returns a list of full-size boolean masks, one per person.

Backends:
  vision       macOS only; runs the compiled Apple Vision helper (scripts/bin/inst), caches PNG masks
  torchvision  any OS; Mask R-CNN (maskrcnn_resnet50_fpn_v2, BSD), person class, score > 0.5
"""
import pathlib
import platform
import subprocess

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
INST_BIN = HERE / "bin" / "inst"
_tv_model = None


def resolve_backend(name):
    if name != "auto":
        return name
    return "vision" if platform.system() == "Darwin" and INST_BIN.exists() else "torchvision"


def vision_masks(src, cache_dir):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = cache_dir / f"{src.stem}.done"  # marker: also caches "no people found"
    if not done.exists():
        subprocess.run([str(INST_BIN), str(src), str(cache_dir)], check=True)
        done.touch()
    files = sorted(cache_dir.glob(f"{src.stem}-m*.png"), key=lambda p: int(p.stem.rsplit("-m", 1)[1]))
    return [np.array(Image.open(p)) > 127 for p in files]


def torchvision_masks(img, score=0.5):
    global _tv_model
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.transforms.functional import pil_to_tensor
    if _tv_model is None:
        _tv_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    with torch.inference_mode():
        out = _tv_model([pil_to_tensor(img).float() / 255])[0]
    keep = (out["labels"] == 1) & (out["scores"] > score)  # COCO label 1 = person
    return [m[0].numpy() > 0.5 for m in out["masks"][keep]]


def person_masks(src, img, backend="auto", cache_dir=None):
    """src: image path, img: the EXIF-transposed PIL RGB image (same orientation Vision uses)."""
    backend = resolve_backend(backend)
    if backend == "vision":
        masks = vision_masks(pathlib.Path(src), cache_dir)
    elif backend == "torchvision":
        masks = torchvision_masks(img)
    else:
        raise ValueError(f"unknown backend: {backend}")
    for m in masks:
        if m.shape != (img.height, img.width):
            raise RuntimeError(f"{src}: mask size {m.shape[::-1]} != image size {img.size}")
    return [m for m in masks if m.any()]
