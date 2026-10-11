# patches/mask2former/viz/attention.py
"""Attention as a soft lookup (section 0), then masked attention (section 5).

Both scenes use the same four locations on the dog photo and the same toy
scores, 2.0, -0.6, -2.2 and 1.1, so the reader meets one example twice. The
scores are illustrative. The mask in the second scene is the model's real
dog mask, which keeps locations 1 and 4 and blocks 2 and 3.

Every number on screen is computed here, not typed. Each scene ends on a
frame that holds the whole result, which render.sh also saves as the poster.
"""

import numpy as np
from manim import (DOWN, LEFT, RIGHT, Circle, Create, FadeIn, FadeOut, GrowFromEdge, ImageMobject,
                   LaggedStart, Line, Rectangle, RoundedRectangle, Scene, Text, Transform, TransformFromCopy,
                   VGroup, Write)
from PIL import Image
from scipy.ndimage import binary_erosion

from fig_mask_classification import HERE, load_image, photo_panel
from theme import FONT, MONO, P

SCORES = np.array([2.0, -0.6, -2.2, 1.1])
KEPT = np.array([True, False, False, True])  # inside the dog mask
SPOTS = [(0.53, 0.57), (0.795, 0.37), (0.12, 0.84), (0.33, 0.29)]  # face, near blanket, far blanket, back

PHOTO_W, PHOTO_C = 4.6, np.array([-3.85, 0.45, 0.0])
ROW_Y = [1.75, 0.80, -0.15, -1.10]
X_ID, X_SCORE, X_EXP, X_BAR, BAR_LEN = -1.0, 0.4, 2.3, 3.45, 2.1
Y_HEAD, Y_SUM, Y_OUT = 2.75, -2.05, -3.2


def label(s, size=32, color=None, font=FONT):
    return Text(s, font=font, font_size=size, color=color or P["text"])


def num(x, digits=2):
    return label("0" if x == 0 else f"{x:.{digits}f}".replace("-", "−"), 32, None, MONO)


def photo(masked):
    """The photo, and optionally the model's dog mask over it in gold."""
    px = photo_panel(load_image())
    if masked:
        logits = np.load(HERE / "sources" / "dog.npz")["logits"]
        size = (px.shape[1], px.shape[0])
        m = np.asarray(Image.fromarray((logits > 0).astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
        gold = np.array([int(P["gold"][i:i + 2], 16) for i in (1, 3, 5)], np.float32)
        px[m] = px[m] * 0.55 + gold * 0.45
        px[m & ~binary_erosion(m, iterations=3)] = gold
    img = ImageMobject(px.clip(0, 255).astype(np.uint8)).scale_to_fit_width(PHOTO_W).move_to(PHOTO_C)
    return img.set_z_index(0)


def marker(i, at):
    dot = Circle(radius=0.24, fill_color=P["slate"], fill_opacity=1, stroke_width=0).move_to(at)
    return VGroup(dot, label(str(i + 1), 28, P["bg"]).move_to(at)).set_z_index(2)


def bar(w):
    return Rectangle(width=max(BAR_LEN * w, 0.001), height=0.34, fill_color=P["gold"],
                     fill_opacity=0.85 if w > 0 else 0, stroke_width=0)


class Board:
    """The layout both scenes share: photo, query, and one row per location."""

    def __init__(self, masked=False):
        self.photo = photo(masked)
        h = self.photo.height
        self.spots = VGroup(*(marker(i, PHOTO_C + [(u - 0.5) * PHOTO_W, (0.5 - v) * h, 0])
                              for i, (u, v) in enumerate(SPOTS)))
        self.ids = VGroup(*(marker(i, [X_ID, y, 0]) for i, y in enumerate(ROW_Y)))
        self.query = VGroup(RoundedRectangle(corner_radius=0.1, width=1.9, height=0.62, stroke_color=P["gold"],
                                             fill_color=P["gold"], fill_opacity=0.14, stroke_width=3),
                            label("query", 32, P["gold_text"])).move_to(PHOTO_C + [0, 2.35, 0])
        self.heads = VGroup(*(label(t, 30, P["muted"]).move_to([x, Y_HEAD, 0])
                              for t, x in (("score", X_SCORE), ("exp(score)", X_EXP), ("weight", X_BAR + 0.85))))
        self.rule = Line([X_SCORE - 0.6, Y_SUM + 0.45, 0], [X_EXP + 0.75, Y_SUM + 0.45, 0], stroke_width=2,
                         color=P["line"])
        self.rule_note = label("weight = exp(score) \u00f7 sum", 30, P["muted"]).move_to([0, Y_OUT, 0])

    def column(self, values, x, digits=2):
        return VGroup(*(num(v, digits).move_to([x, y, 0]) for v, y in zip(values, ROW_Y)))

    def bars(self, weights):
        return VGroup(*(bar(w).move_to([X_BAR, y, 0], aligned_edge=LEFT) for w, y in zip(weights, ROW_Y)))

    def weight_labels(self, weights, bars):
        return VGroup(*(num(w).next_to(b, RIGHT, buff=0.18) for w, b in zip(weights, bars)))

    def total(self, s):
        return VGroup(label("sum", 30, P["muted"]).move_to([X_SCORE, Y_SUM, 0]),
                      num(s).move_to([X_EXP, Y_SUM, 0]))

    def output(self, weights):
        # Full-size indices: subscripts shrink to a few pixels on a phone.
        terms = " + ".join(f"{w:.2f} v{i + 1}" for i, w in enumerate(weights) if w > 0)
        return label(f"output = {terms}", 30, P["text"], MONO).move_to([0, Y_OUT, 0])


class AttentionLookup(Scene):
    def construct(self):
        b = Board()
        e = np.exp(SCORES)
        w = e / e.sum()

        # Drawn, not faded in: the first frame is what the page shows before playback.
        self.add(b.photo, b.spots, b.query)
        self.wait(0.6)
        lines = VGroup(*(Line(b.query.get_bottom(), s.get_center(), buff=0.26, stroke_width=3, color=P["gold"])
                         for s in b.spots))
        hint = label("score = query · key", 32, P["muted"]).move_to([0, Y_OUT, 0])
        self.play(LaggedStart(*(Create(l) for l in lines), lag_ratio=0.15), FadeIn(hint), run_time=1.2)
        self.play(LaggedStart(*(TransformFromCopy(s, i) for s, i in zip(b.spots, b.ids)), lag_ratio=0.12),
                  FadeIn(b.heads[0]), run_time=1.2)
        scores = b.column(SCORES, X_SCORE, digits=1)
        self.play(LaggedStart(*(FadeIn(s, shift=0.1 * LEFT) for s in scores), lag_ratio=0.15), run_time=1.0)
        self.wait(0.8)

        exps = b.column(e, X_EXP)
        self.play(FadeOut(lines), FadeOut(hint), FadeIn(b.heads[1]),
                  LaggedStart(*(TransformFromCopy(s, x) for s, x in zip(scores, exps)), lag_ratio=0.12),
                  run_time=1.4)
        total = b.total(e.sum())
        self.play(Create(b.rule), TransformFromCopy(exps, total), run_time=1.0)
        self.wait(0.5)

        bars = b.bars(w)
        weights = b.weight_labels(w, bars)
        self.play(FadeIn(b.heads[2]), FadeIn(b.rule_note),
                  LaggedStart(*(GrowFromEdge(x, LEFT) for x in bars), lag_ratio=0.12),
                  LaggedStart(*(TransformFromCopy(x, y) for x, y in zip(exps, weights)), lag_ratio=0.12),
                  run_time=1.8)
        self.wait(1.2)
        self.play(FadeOut(b.rule_note), run_time=0.4)
        self.play(Write(b.output(w)), run_time=1.4)
        self.wait(3.5)


class MaskedAttention(Scene):
    def construct(self):
        b = Board()
        masked = photo(True)
        e = np.exp(SCORES)
        w = e / e.sum()
        e_kept = np.where(KEPT, e, 0.0)
        w_kept = e_kept / e_kept.sum()

        scores, exps = b.column(SCORES, X_SCORE, digits=1), b.column(e, X_EXP)
        bars = b.bars(w)
        weights, total, out = b.weight_labels(w, bars), b.total(e.sum()), b.output(w)
        # Drawn, not faded in: the first frame is Fig. 0's result, the question this figure answers.
        self.add(b.photo, b.spots, b.query, b.ids, b.heads, scores, exps, b.rule, total, bars, weights, out)
        self.wait(1.2)

        note = label("previous mask", 30, P["gold_text"]).next_to(b.photo, DOWN, buff=0.22)
        self.play(FadeIn(masked), FadeIn(note), run_time=1.2)
        self.remove(b.photo)
        self.wait(0.6)

        blocked = [i for i in range(4) if not KEPT[i]]
        kept = [i for i in range(4) if KEPT[i]]
        self.play(*(Transform(scores[i], label("−∞", 32, P["ember"]).move_to(scores[i])) for i in blocked),
                  run_time=1.0)
        self.play(*(Transform(exps[i], num(0).move_to(exps[i])) for i in blocked),
                  *(Transform(bars[i], bar(0).move_to([X_BAR, ROW_Y[i], 0], aligned_edge=LEFT)) for i in blocked),
                  *(Transform(weights[i], num(0).move_to([X_BAR + 0.3, ROW_Y[i], 0])) for i in blocked),
                  *(m[i].animate.set_opacity(0.35) for m in (b.ids, b.spots) for i in blocked), run_time=1.2)
        self.play(Transform(total, b.total(e_kept.sum())), run_time=0.9)

        new_bars = b.bars(w_kept)
        new_weights = b.weight_labels(w_kept, new_bars)
        self.play(*(Transform(bars[i], new_bars[i]) for i in kept),
                  *(Transform(weights[i], new_weights[i]) for i in kept), run_time=1.4)
        self.play(Transform(out, b.output(w_kept)), run_time=1.2)
        self.wait(3.5)
