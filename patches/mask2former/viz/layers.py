# patches/mask2former/viz/layers.py
"""Where the dog query reads, layer by layer (sections 5 and 6).

The model's real cross-attention weights for the dog query in each of the
nine decoder layers, one cell per layer: columns are the feature map the
layer reads (strides 32, 16 and 8), rows are the three rounds. Dark gold
marks the fewest locations that hold half of the query's attention weight,
light gold those that hold 90 percent. Mass levels, unlike a per-map color
scale, compare fairly across maps with 950 or 15,200 locations. The outline is
the layer's gate, which data.py rebuilds exactly as the model does. All
weight falls inside it.

Needs sources/dog.npz from data.py. Render with render.sh.
"""

import numpy as np
from manim import LEFT, FadeIn, ImageMobject, Rectangle, Scene, Text, VGroup
from PIL import Image
from scipy.ndimage import binary_erosion

from fig_mask_classification import HERE, load_image, photo_panel
from theme import FONT, MONO, P

CELL_W, GAP, X0, Y_HEAD = 2.95, 0.18, -2.0, 3.05
PX = (480, 320)  # thumbnail pixels: enough for crisp 1/8 cells at 1600px


def rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def top_mass(attn, share):
    """The fewest locations whose weights add up to `share` of the total."""
    flat = attn.ravel()
    order = np.argsort(flat)[::-1]
    before = np.empty_like(flat)
    before[order] = np.cumsum(flat[order]) - flat[order]
    return (before < share * flat.sum()).reshape(attn.shape)


def thumbnail(base, attn, gate):
    """Photo, attention mass in gold, and the gate the layer applied, as pixels."""
    big = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8)).resize(PX, Image.NEAREST)).astype(bool)  # noqa: E731
    out = base.copy()
    for share, alpha in ((0.9, 0.35), (0.5, 0.85)):
        m = big(top_mass(attn, share))
        out[m] = out[m] * (1 - alpha) + rgb(P["gold"]) * alpha
    g = big(gate)
    out[g & ~binary_erosion(g, iterations=2)] = rgb(P["text"])
    return out.clip(0, 255).astype(np.uint8)


class LayerByLayer(Scene):
    def construct(self):
        d = np.load(HERE / "sources" / "dog.npz")
        base = np.asarray(Image.fromarray(photo_panel(load_image()).clip(0, 255).astype(np.uint8)).resize(PX))
        base = base.astype(np.float32)
        cell_h = CELL_W * PX[1] / PX[0]

        def center(l):
            r, c = divmod(l - 1, 3)
            return np.array([X0 + c * (CELL_W + GAP), Y_HEAD - 0.4 - cell_h / 2 - r * (cell_h + GAP), 0])

        frames = VGroup(*(Rectangle(width=CELL_W, height=cell_h, stroke_color=P["grid"], stroke_width=2)
                          .move_to(center(l)) for l in range(1, 10)))
        frames_left = lambda r: frames[3 * r]  # noqa: E731
        heads = VGroup(*(Text(t, font=MONO, font_size=30, color=P["muted"]).move_to([center(c + 1)[0], Y_HEAD, 0])
                         for c, t in enumerate(("1/32", "1/16", "1/8"))))
        rows = VGroup(*(Text(f"layers {3 * r + 1}–{3 * r + 3}", font=FONT, font_size=30, color=P["muted"])
                        .next_to(frames_left(r), LEFT, buff=0.3) for r in range(3)))

        # Drawn, not faded in: the first frame is what the page shows before playback.
        self.add(heads, rows, frames)
        self.wait(0.6)

        for l in range(1, 10):
            img = ImageMobject(thumbnail(base, d[f"attn_{l}"], d[f"gate_{l}"]))
            img.scale_to_fit_width(CELL_W).move_to(center(l))
            mark = frames[l - 1].copy().set_stroke(P["gold"], width=5)
            self.play(FadeIn(img), FadeIn(mark), run_time=0.6)
            self.play(mark.animate.set_stroke(P["grid"], width=2), run_time=0.25)
        self.wait(3.5)
