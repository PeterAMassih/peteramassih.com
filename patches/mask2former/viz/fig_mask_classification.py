# patches/mask2former/viz/fig_mask_classification.py
"""Build Fig. 2 (mask_classification) for the article: the model's own output.

Unlike the other five files here, this is not a Manim scene. It runs the actual
Mask2Former model on one Creative-Commons image and composes a static figure
that grounds the (class, mask) pair of section 2.2: left, the predicted dog mask
as a translucent gold overlay with a contour; right, the same mask coarsened to
a blocky 0/1 grid, which is all a binary mask ever is. Gold is the model's
prediction, matching the color language of the Manim scenes (see tokens.py).

Source image: "Dog-2617516_1920.jpg" from Wikimedia Commons, dedicated to the
public domain under CC0 1.0 (https://commons.wikimedia.org/wiki/File:Dog-2617516_1920.jpg).
No attribution is required; the figure caption credits it anyway.

Model: facebook/mask2former-swin-large-coco-panoptic (Apache-2.0 weights). On
this image it returns exactly two segments, dog (a thing) and blanket (stuff),
which is why the caption can point at the thing/stuff split of section 1.

Run (from this directory, after the extra deps below are installed):
    .venv/bin/python fig_mask_classification.py

Outputs public/assets/m2f/mask_classification.{webp,png}. Deterministic: the
model is in eval mode with no sampling, so re-running reproduces the committed
figure bit-for-bit given the same checkpoint.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from tokens import BACKGROUND, GOLD  # site palette: #fafafa, #b8860b

HERE = Path(__file__).resolve().parent
SRC_URL = "https://upload.wikimedia.org/wikipedia/commons/4/48/Dog-2617516_1920.jpg"
SRC = HERE / "sources" / "dog-2617516.jpg"          # cached, not committed
OUT = HERE.parents[2] / "public" / "assets" / "m2f"  # repo public/assets/m2f
CKPT = "facebook/mask2former-swin-large-coco-panoptic"

GOLD_RGB = np.array([184, 134, 11], np.float32)   # b8860b
BG_RGB = np.array([250, 250, 250], np.float32)    # fafafa
GRID_RGB = np.array([228, 224, 214], np.float32)  # faint gridline
CELL0_RGB = np.array([245, 243, 237], np.float32)  # empty cell


def load_image():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        # Wikimedia 403s the default urllib agent; identify the tool per its policy.
        req = urllib.request.Request(SRC_URL, headers={
            "User-Agent": "peteramassih.com-figure-build/1.0 (https://peteramassih.com)"})
        with urllib.request.urlopen(req) as r:
            SRC.write_bytes(r.read())
    img = Image.open(SRC).convert("RGB")
    scale = 1333 / max(img.size)  # sane figure resolution; the model resizes internally
    return img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)


def predict_dog_mask(img):
    proc = AutoImageProcessor.from_pretrained(CKPT)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT).eval()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
    seg = res["segmentation"].cpu().numpy()
    id2label = model.config.id2label
    dogs = [s for s in res["segments_info"] if id2label[s["label_id"]] == "dog"]
    if not dogs:
        raise SystemExit("no dog segment found; the source image or checkpoint changed")
    dog = max(dogs, key=lambda s: int((seg == s["id"]).sum()))
    return seg == dog["id"]


def left_panel(img, mask):
    """Desaturated photo, gold fill on the segment, gold contour."""
    base = np.asarray(img).astype(np.float32)
    gray = base @ np.array([0.299, 0.587, 0.114], np.float32)
    photo = 0.62 * gray[..., None] + 0.38 * base   # pull most of the color out
    photo = 0.92 * photo + 0.08 * BG_RGB           # settle into the site off-white
    out = photo.copy()
    out[mask] = out[mask] * 0.58 + GOLD_RGB * 0.42
    contour = mask & ~binary_erosion(mask, iterations=3)
    out[contour] = GOLD_RGB
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((720, 480), Image.LANCZOS)


def right_panel(mask):
    """The same mask coarsened to a blocky 0/1 grid over the pixels."""
    h, w = mask.shape
    cols = 48
    rows = round(cols * h / w)
    ys = (np.arange(rows + 1) * h / rows).astype(int)
    xs = (np.arange(cols + 1) * w / cols).astype(int)
    coarse = np.array([[mask[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].mean() > 0.5
                        for c in range(cols)] for r in range(rows)])
    cp = 15  # cell pixels -> 720x(rows*15)
    grid = np.empty((rows * cp, cols * cp, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            grid[r * cp:(r + 1) * cp, c * cp:(c + 1) * cp] = GOLD_RGB if coarse[r, c] else CELL0_RGB
    grid[::cp] = GRID_RGB
    grid[:, ::cp] = GRID_RGB
    return Image.fromarray(grid.clip(0, 255).astype(np.uint8))


def main():
    img = load_image()
    mask = predict_dog_mask(img)
    left, right = left_panel(img, mask), right_panel(mask)
    margin = gap = 44
    canvas = Image.new("RGB", (margin + 720 + gap + 720 + margin, margin + 480 + margin),
                       tuple(BG_RGB.astype(int)))
    canvas.paste(left, (margin, margin))
    canvas.paste(right, (margin + 720 + gap, margin))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "mask_classification.png")
    canvas.save(OUT / "mask_classification.webp", quality=82, method=6)
    print(f"wrote {OUT/'mask_classification.webp'} ({canvas.size}, dog covers {mask.mean():.1%})")


if __name__ == "__main__":
    main()
