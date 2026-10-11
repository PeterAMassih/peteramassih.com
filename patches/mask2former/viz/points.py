# patches/mask2former/viz/points.py
"""The two point rules of section 8, on the model's real dog prediction.

Left, matching: K uniform points, one set shared by every mask in the image.
Right, training: 3K uniform candidates, of which the 0.75K whose mask logits
are closest to zero are kept (gold), plus 0.25K fresh uniform points (ink).
The gold line is the prediction's 0.5 level. Selection reads only the
prediction, never a ground truth, so the kept points follow its boundary.

The paper uses K = 12,544 points; the figure draws K = 400 so single points
stay visible. Logits are read bilinearly at the point coordinates, as the
released code's point_sample does. A fixed seed makes every render identical.

Needs sources/dog.npz from data.py. Render with render.sh.
"""

import numpy as np
from manim import DOWN, UP, Dot, FadeIn, FadeOut, ImageMobject, LaggedStart, MarkupText, Scene, Text, VGroup
from PIL import Image, ImageFilter

from scipy.ndimage import map_coordinates

from fig_mask_classification import HERE, contour_of, load_image, photo_panel
from theme import FONT, MONO, P

K = 400
PANEL_W, PANEL_Y, PANEL_X = 6.0, -0.1, 3.15


def logit_at(logits, pts):
    """Bilinear logits at normalized (x, y) points, pixel centers at (i + 0.5) / n."""
    h, w = logits.shape
    return map_coordinates(logits, [pts[:, 1] * h - 0.5, pts[:, 0] * w - 0.5], order=1, mode="nearest")


def rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def panel(logits, x):
    """The photo with the prediction's 0.5 level in gold."""
    # A slight blur removes the blanket's fine texture, which the encoder would
    # otherwise spend bytes on in every frame. The reader sees the same photo.
    px = np.asarray(Image.fromarray(photo_panel(load_image()).clip(0, 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)
    size = (px.shape[1], px.shape[0])
    prob = np.asarray(Image.fromarray((1 / (1 + np.exp(-logits))).astype(np.float32)).resize(size, Image.BILINEAR))
    px[contour_of(prob >= 0.5, 2)] = rgb(P["gold"])
    return ImageMobject(px.clip(0, 255).astype(np.uint8)).scale_to_fit_width(PANEL_W).move_to([x, PANEL_Y, 0])


def dots(pts, photo, color, opacity=1.0):
    """Points in normalized image coordinates, (0, 0) at the photo's top left."""
    left, top = photo.get_left()[0], photo.get_top()[1]
    return VGroup(*(Dot([left + u * photo.width, top - v * photo.height, 0], radius=0.032, color=color,
                        fill_opacity=opacity, stroke_width=1, stroke_color=P["bg"]) for u, v in pts))


def label(*parts):
    """Counts in mono, words in Geist. One markup line keeps both on a single
    baseline, which separate Text objects would not."""
    span = lambda t: f'<span font_family="{MONO}" foreground="{P["text"]}">{t}</span>'  # noqa: E731
    return MarkupText(" ".join(span(t) if t[0].isdigit() else t for t in parts), font=FONT, font_size=30,
                      color=P["muted"])


class PointSampling(Scene):
    def construct(self):
        logits = np.load(HERE / "sources" / "dog.npz")["logits"]
        rng = np.random.default_rng(0)
        shared = rng.random((K, 2))
        candidates = rng.random((3 * K, 2))
        n_kept = int(0.75 * K)
        order = np.argsort(np.abs(logit_at(logits, candidates)))
        fresh = rng.random((K - n_kept, 2))

        left, right = panel(logits, -PANEL_X), panel(logits, PANEL_X)
        self.add(left, right)  # a still photo costs few bytes, a fading one costs many
        titles = VGroup(*(Text(t, font=FONT, font_size=32, color=P["text"]).next_to(p, UP, buff=0.3)
                          for t, p in (("matching", left), ("training", right))))
        self.play(FadeIn(titles), run_time=0.6)

        # Each point pops in or out once instead of fading: a fade repaints
        # every point on every frame, which costs several times the bytes.
        pop = lambda anims, t: LaggedStart(*anims, lag_ratio=1, run_time=t)  # noqa: E731
        uniform = dots(shared, left, P["text"])
        left_label = label(f"{K}", "uniform points").next_to(left, DOWN, buff=0.35)
        self.play(pop([FadeIn(d) for d in uniform], 1.0), FadeIn(left_label))
        self.wait(1.3)

        pool = dots(candidates, right, P["muted"], 0.8)
        right_label = label(f"{3 * K:,}", "candidates").next_to(right, DOWN, buff=0.35)
        self.play(pop([FadeIn(d) for d in pool], 1.0), FadeIn(right_label))
        self.wait(1.3)

        kept = [pool[i] for i in order[:n_kept]]
        dropped = [pool[i] for i in order[n_kept:]]
        kept_label = label(f"{n_kept}", "most uncertain").move_to(right_label)
        self.play(pop([FadeOut(d) for d in dropped], 1.0), pop([d.animate.set_fill(P["gold"], 1) for d in kept], 1.0),
                  FadeOut(right_label), FadeIn(kept_label))
        self.wait(1.3)

        extra = dots(fresh, right, P["text"])
        final_label = label(f"{n_kept}", "uncertain +", f"{K - n_kept}", "uniform").next_to(right, DOWN, buff=0.35)
        self.play(pop([FadeIn(d) for d in extra], 0.7), FadeOut(kept_label), FadeIn(final_label))
        self.wait(3.8)
