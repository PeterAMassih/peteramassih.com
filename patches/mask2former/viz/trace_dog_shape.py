# patches/mask2former/viz/trace_dog_shape.py
"""Reproduce the DOG silhouette in shapes.py by tracing a Mask2Former mask.

The hand-drawn dog read as a blob, so it is traced from a real mask instead:
run Mask2Former on a side-profile dog photo, take the predicted dog mask, and
simplify its contour to a few dozen points in the shapes.py coordinate frame
(centered, y up, facing left). Only the resulting point list lives in the repo
(shapes.py); the photo is not stored.

Source image: "Young Ibizan Hound posing in profile.jpg" from Wikimedia
Commons, CC0 1.0 (public domain). Model: facebook/mask2former-swin-large-coco-panoptic.

Extra deps beyond the render setup: torch, torchvision, transformers, scipy,
opencv-python-headless. Run from this directory:
    .venv/bin/python trace_dog_shape.py
It prints the DOG = [...] block to paste into shapes.py.
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

HERE = Path(__file__).resolve().parent
SRC_URL = ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/"
           "Young_Ibizan_Hound_posing_in_profile.jpg/"
           "960px-Young_Ibizan_Hound_posing_in_profile.jpg")
SRC = HERE / "sources" / "ibizan-profile.jpg"  # cached, not committed
CKPT = "facebook/mask2former-swin-large-coco-panoptic"


def dog_mask(img):
    proc = AutoImageProcessor.from_pretrained(CKPT)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT).eval()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_panoptic_segmentation(out, target_sizes=[img.size[::-1]])[0]
    seg, id2label = res["segmentation"].cpu().numpy(), model.config.id2label
    dogs = [s for s in res["segments_info"] if id2label[s["label_id"]] == "dog"]
    dog = max(dogs, key=lambda s: int((seg == s["id"]).sum()))
    return binary_fill_holes(seg == dog["id"]).astype(np.uint8)


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SRC_URL, headers={
            "User-Agent": "peteramassih.com-figure/1.0 (https://peteramassih.com)"})
        with urllib.request.urlopen(req) as r:
            SRC.write_bytes(r.read())

    mask = dog_mask(Image.open(SRC).convert("RGB"))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    pts = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)

    # Into the shapes.py frame: center, flip y (image y is down), scale to span ~2.1.
    pts[:, 0] -= (pts[:, 0].min() + pts[:, 0].max()) / 2
    pts[:, 1] -= (pts[:, 1].min() + pts[:, 1].max()) / 2
    pts[:, 1] *= -1
    pts *= 2.1 / max(pts[:, 0].ptp(), pts[:, 1].ptp())
    pts = np.round(pts, 3)

    print("DOG = [")
    for i in range(0, len(pts), 5):
        print("    " + ", ".join(f"({x}, {y})" for x, y in pts[i:i + 5]) + ",")
    print("]")


if __name__ == "__main__":
    main()
