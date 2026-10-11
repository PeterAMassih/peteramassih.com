# patches/mask2former/viz/fig_mask_classification.py
"""Build Fig. 2 (mask_classification) for the article: the model's own output.

Not a Manim scene. It runs the actual Mask2Former model on one Creative-Commons
image and composes a static figure that grounds the (class, mask) pair of
section 2.2. Right, the dog query's soft mask, the sigmoid of its logits at
stride 4, with the 0.5 level drawn as a dark line. Left, the same mask
thresholded at 0.5, as a translucent gold overlay with a contour. Gold is the
model's prediction, matching the color language of the other figures.

Source image: "Dog-2617516_1920.jpg" from Wikimedia Commons, dedicated to the
public domain under CC0 1.0 (https://commons.wikimedia.org/wiki/File:Dog-2617516_1920.jpg).
No attribution is required. The caption notes that the photo is public domain.

Model: facebook/mask2former-swin-large-coco-panoptic (Apache-2.0 weights), run
at the paper's COCO inference size (shorter side 800, longer side at most 1333)
rather than the Hugging Face processor's default square resize, so the mask
keeps the photo's aspect ratio.

Run (from this directory, after the extra deps in README.md are installed):
    .venv/bin/python fig_mask_classification.py

Outputs public/assets/m2f/mask_classification.webp and prints the dog query's
class probability for the caption. Deterministic: the model is in eval mode
with no sampling. The Manim scenes reuse load_image(), photo_panel() and
contour_of() below, so every figure shows the same photo the same way.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


HERE = Path(__file__).resolve().parent
SRC_URL = "https://upload.wikimedia.org/wikipedia/commons/4/48/Dog-2617516_1920.jpg"
SRC = HERE / "sources" / "dog-2617516.jpg"          # cached, not committed
OUT = HERE.parents[2] / "public" / "assets" / "m2f"  # repo public/assets/m2f
CKPT = "facebook/mask2former-swin-large-coco-panoptic"
PANEL = (720, 480)

GOLD_RGB = np.array([184, 134, 11], np.float32)   # b8860b
INK_RGB = np.array([23, 23, 23], np.float32)      # 171717
BG_RGB = np.array([255, 255, 255], np.float32)    # ffffff, the page background


def load_image():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        # Wikimedia 403s the default urllib agent, so identify the tool per its policy.
        req = urllib.request.Request(SRC_URL, headers={
            "User-Agent": "peteramassih.com-figure-build/1.0 (https://peteramassih.com)"})
        with urllib.request.urlopen(req) as r:
            SRC.write_bytes(r.read())
    img = Image.open(SRC).convert("RGB")
    scale = 1333 / max(img.size)  # a sane figure resolution, which the processor resizes again
    return img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)


def dog_query(img):
    """The query that predicts the dog: its class probability and its mask
    logits, cropped to the image (stride 4 of the model input, padding removed).

    It is the query with the highest dog probability. Other queries can carry a
    near-identical mask but predict no-object, which is what one-to-one
    matching trains duplicates to do.
    """
    proc = AutoImageProcessor.from_pretrained(
        CKPT, size={"shortest_edge": 800, "longest_edge": 1333})
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT).eval()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    # Mask logits live on the padded input at stride 4. Keep the image part.
    valid = inputs["pixel_mask"][0].numpy().astype(bool)
    h, w = valid.any(1).sum() // 4, valid.any(0).sum() // 4
    dog = next(k for k, v in model.config.id2label.items() if v == "dog")
    probs = torch.softmax(out.class_queries_logits[0], -1)[:, dog].numpy()
    q = int(np.argmax(probs))
    return {"p_dog": float(probs[q]), "logits": out.masks_queries_logits[0, q, :h, :w].numpy()}


def photo_panel(img):
    """Desaturated photo settled into the site off-white."""
    base = np.asarray(img.resize(PANEL, Image.LANCZOS)).astype(np.float32)
    gray = base @ np.array([0.299, 0.587, 0.114], np.float32)
    photo = 0.62 * gray[..., None] + 0.38 * base
    return 0.92 * photo + 0.08 * BG_RGB


def contour_of(mask, width):
    return mask & ~binary_erosion(mask, iterations=width)


def soft_mask(logits):
    """sigmoid(logits) at panel size. Bilinear upsampling shows the values, not the grid."""
    prob = 1 / (1 + np.exp(-logits))
    return np.asarray(Image.fromarray(prob.astype(np.float32)).resize(PANEL, Image.BILINEAR))


def left_panel(img, soft):
    """The soft mask thresholded at 0.5, over the photo: gold fill, gold contour."""
    out = photo_panel(img)
    mask = soft >= 0.5
    out[mask] = out[mask] * 0.58 + GOLD_RGB * 0.42
    out[contour_of(mask, 3)] = GOLD_RGB
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def right_panel(soft):
    """The soft mask: off-white at 0, gold at 1, with the 0.5 level drawn in ink."""
    out = BG_RGB * (1 - soft[..., None]) + GOLD_RGB * soft[..., None]
    out[contour_of(soft >= 0.5, 2)] = INK_RGB
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def compose(left, right, name):
    margin = gap = 44
    canvas = Image.new("RGB", (margin + PANEL[0] + gap + PANEL[0] + margin, margin + PANEL[1] + margin),
                       tuple(BG_RGB.astype(int)))
    canvas.paste(left, (margin, margin))
    canvas.paste(right, (margin + PANEL[0] + gap, margin))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / name, quality=82, method=6)
    print(f"wrote {OUT / name} {canvas.size}")


def main():
    img = load_image()
    dog = dog_query(img)
    soft = soft_mask(dog["logits"])
    compose(left_panel(img, soft), right_panel(soft), "mask_classification.webp")
    print(f"dog query: p_hat(dog) = {dog['p_dog']:.3f}, "
          f"mask logits {dog['logits'].shape[1]}x{dog['logits'].shape[0]} at stride 4")


if __name__ == "__main__":
    main()
