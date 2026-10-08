# patches/mask2former/viz/fig_point_sampling.py
"""Build Fig. 6 (point_sampling) for the article: the two point rules of
section 8.2 applied to the model's real dog prediction.

Left, matching: K uniform points, one shared set for every mask in the image.
Right, training: 3K uniform candidates, of which the 0.75K whose mask logits
are closest to zero are kept (gold), plus 0.25K fresh uniform points (ink).
The gold line is the prediction's 0.5 level. Selection reads only the
prediction's logits, never a ground truth, which is the point of the figure.

The paper uses K = 12,544 points per mask; the figure draws K = 400 so single
points stay visible. Logits are sampled bilinearly at the point coordinates,
as the released code's point_sample does. Fixed seed, so re-running
reproduces the committed figure.

Run (from this directory): .venv/bin/python fig_point_sampling.py
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from fig_mask_classification import (
    BG_RGB, GOLD_RGB, INK_RGB, PANEL, compose, contour_of, dog_query, load_image, photo_panel)

K = 400
SS = 2  # supersampling for smooth dots


def logit_at(logits, pts):
    """Bilinear logits at normalized (x, y) points, pixel centers at (i + 0.5) / n."""
    h, w = logits.shape
    return map_coordinates(logits, [pts[:, 1] * h - 0.5, pts[:, 0] * w - 0.5], order=1, mode="nearest")


def panel(img, logits, groups):
    """Photo, the prediction's 0.5 level in gold, then each group of points."""
    base = photo_panel(img)
    prob = 1 / (1 + np.exp(-logits))
    up = np.asarray(Image.fromarray(prob.astype(np.float32)).resize(PANEL, Image.BILINEAR))
    base[contour_of(up >= 0.5, 2)] = GOLD_RGB
    big = Image.fromarray(base.clip(0, 255).astype(np.uint8)).resize(
        (PANEL[0] * SS, PANEL[1] * SS), Image.LANCZOS)
    draw = ImageDraw.Draw(big)
    for pts, rgb in groups:
        for x, y in pts * np.array(PANEL) * SS:
            r = 3.4 * SS
            draw.ellipse((x - r - SS, y - r - SS, x + r + SS, y + r + SS), fill=tuple(BG_RGB.astype(int)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rgb.astype(int)))
    return big.resize(PANEL, Image.LANCZOS)


def main():
    rng = np.random.default_rng(0)
    img = load_image()
    logits = dog_query(img)["logits"]

    shared = rng.random((K, 2))

    candidates = rng.random((3 * K, 2))
    n_uncertain = int(0.75 * K)
    uncertain = candidates[np.argsort(np.abs(logit_at(logits, candidates)))[:n_uncertain]]
    fresh = rng.random((K - n_uncertain, 2))

    left = panel(img, logits, [(shared, INK_RGB)])
    right = panel(img, logits, [(fresh, INK_RGB), (uncertain, GOLD_RGB)])
    compose(left, right, "point_sampling.webp")
    inside = (logit_at(logits, uncertain) > 0).mean()
    print(f"K={K}: {n_uncertain} uncertain ({inside:.0%} on the positive side), {K - n_uncertain} fresh")


if __name__ == "__main__":
    main()
