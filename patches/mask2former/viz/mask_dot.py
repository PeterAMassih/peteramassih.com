# patches/mask2former/viz/mask_dot.py
"""A mask is a dot product (the architecture section).

The mask head gives each pixel the logit query . pixel, with no bias, where
query is the dog query's mask embedding and pixel is that pixel's embedding.
Both have 256 numbers, so the scene draws each pixel as a dot in a plane
whose horizontal axis is the query's direction. A dot's horizontal position
times the query's length is then exactly its logit (data.py asserts this),
and the vertical axis only spreads the dots out. Logit 0 is probability 0.5,
so the mask is every dot right of one vertical line.

The dots are a 13 by 20 grid of sample cells, each the embedding at its
cell's center. The scene ends on the full-resolution mask from the same rule.

Needs sources/dog.npz from data.py. Render with render.sh.
"""

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, DashedLine, Dot, FadeIn, FadeOut, ImageMobject, LaggedStart,
                   MarkupText, Rectangle, Scene, Text, TransformFromCopy, VGroup, Write)
from PIL import Image, ImageFilter

from fig_mask_classification import HERE, contour_of, load_image, photo_panel
from theme import FONT, MONO, P

ROWS, COLS = 13, 20
PHOTO_W, PHOTO_C = 5.4, np.array([-3.45, 0.45, 0.0])
X0, X1, Y0, Y1 = 0.45, 5.75, -1.75, 2.65  # the plane's drawing box


def rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def photo():
    # A slight blur removes the blanket's fine texture, which the encoder would
    # otherwise spend bytes on in every frame. The reader sees the same photo.
    px = Image.fromarray(photo_panel(load_image()).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    return ImageMobject(np.asarray(px)).scale_to_fit_width(PHOTO_W).move_to(PHOTO_C)


def soft_mask(logits, size):
    """sigmoid(logits) as a gold layer for the photo, with its 0.5 level drawn.
    A separate transparent layer, so fading it in changes only the masked
    pixels and the video stays small."""
    prob = np.asarray(Image.fromarray((1 / (1 + np.exp(-logits))).astype(np.float32)).resize(size, Image.BILINEAR))
    layer = np.zeros((size[1], size[0], 4), np.uint8)
    layer[..., :3] = rgb(P["gold"])
    layer[..., 3] = 255 * np.where(contour_of(prob >= 0.5, 2), 1.0, 0.5 * prob)
    return ImageMobject(layer).scale_to_fit_width(PHOTO_W).move_to(PHOTO_C).set_z_index(1)


def caption(count, words):
    """The count in mono, the words in Geist, on one baseline."""
    return MarkupText(f'<span font_family="{MONO}" foreground="{P["text"]}">{count}</span> {words}', font=FONT,
                      font_size=30, color=P["muted"])


class MaskIsDotProduct(Scene):
    def construct(self):
        d = np.load(HERE / "sources" / "dog.npz")
        h, w = d["x_map"].shape
        rr = ((np.arange(ROWS) + 0.5) * h / ROWS).astype(int)
        cc = ((np.arange(COLS) + 0.5) * w / COLS).astype(int)
        xs, ys = d["x_map"][np.ix_(rr, cc)].ravel(), d["y_map"][np.ix_(rr, cc)].ravel()
        inside = xs > 0

        base = photo()
        cw, ch = base.width / COLS, base.height / ROWS
        corner = base.get_corner(UP + LEFT)
        cells = [corner + [(j + 0.5) * cw, -(i + 0.5) * ch, 0] for i in range(ROWS) for j in range(COLS)]
        on_photo = VGroup(*(Dot(c, radius=0.045, color=P["slate"], stroke_width=1.5, stroke_color=P["bg"])
                            for c in cells)).set_z_index(2)

        lo_x, hi_x, lo_y, hi_y = xs.min() - 0.3, xs.max() + 0.3, ys.min() - 0.3, ys.max() + 0.3
        to_plane = lambda x, y: [X0 + (x - lo_x) / (hi_x - lo_x) * (X1 - X0),  # noqa: E731
                                 Y0 + (y - lo_y) / (hi_y - lo_y) * (Y1 - Y0), 0]
        in_plane = VGroup(*(Dot(to_plane(x, y), radius=0.06, color=P["slate"]) for x, y in zip(xs, ys)))
        x_zero = to_plane(0, 0)[0]
        box = Rectangle(width=X1 - X0 + 0.3, height=Y1 - Y0 + 0.3, stroke_color=P["grid"], stroke_width=2)
        box.move_to([(X0 + X1) / 2, (Y0 + Y1) / 2, 0])
        sampled = caption(f"{ROWS * COLS}", "sample cells").next_to(base, DOWN, buff=0.3)

        # The photo is there from the first frame: fading a textured photo costs most of a video's bytes.
        self.add(base)
        # Dots pop in and out one at a time. A fade would repaint all 260 every frame.
        self.play(LaggedStart(*(FadeIn(p) for p in on_photo), lag_ratio=1), FadeIn(sampled), run_time=0.7)
        self.play(FadeIn(box), LaggedStart(*(TransformFromCopy(a, b) for a, b in zip(on_photo, in_plane)),
                                           lag_ratio=0.004), run_time=1.8)
        self.wait(1.1)

        query = Arrow([X0 - 0.15, Y0 - 0.55, 0], [X1 + 0.15, Y0 - 0.55, 0], buff=0, color=P["gold"],
                      stroke_width=5, max_tip_length_to_length_ratio=0.05)
        query_label = Text("along the query", font=FONT, font_size=30, color=P["gold_text"])
        query_label.next_to(query, DOWN, buff=0.15).align_to(query, RIGHT)
        self.play(FadeIn(query), FadeIn(query_label), run_time=1.0)
        line = DashedLine([x_zero, Y0 - 0.15, 0], [x_zero, Y1 + 0.15, 0], dash_length=0.12, color=P["text"],
                          stroke_width=3)
        half = Text("sigmoid = 0.5", font=MONO, font_size=30, color=P["muted"]).next_to(line, UP, buff=0.15)
        self.play(FadeIn(line), FadeIn(half), run_time=1.0)
        self.wait(0.6)

        squares = VGroup(*(Rectangle(width=cw, height=ch, fill_color=P["gold"], fill_opacity=0.55, stroke_width=0)
                           .move_to(c) for c, k in zip(cells, inside) if k)).set_z_index(1)
        self.play(*(in_plane[i].animate.set_color(P["gold"]) for i in np.flatnonzero(inside)),
                  *(on_photo[i].animate.set_color(P["gold"]) for i in np.flatnonzero(inside)),
                  FadeIn(squares), run_time=1.6)
        formula = Text("mask = sigmoid(query · pixel) > 0.5", font=MONO, font_size=30, color=P["text"])
        self.play(Write(formula.move_to([0, -3.35, 0])), run_time=1.2)
        self.wait(0.8)

        full = soft_mask(d["logits"], (720, 480))
        everything = caption(f"{h * w:,}", "cells, the full mask").move_to(sampled)
        self.play(LaggedStart(*(FadeOut(p) for p in on_photo), lag_ratio=1), run_time=0.6)
        self.play(FadeIn(full), FadeOut(squares), FadeOut(sampled), FadeIn(everything), run_time=0.9)
        self.wait(3.6)
