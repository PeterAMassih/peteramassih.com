# patches/mask2former/viz/conv.py
"""One convolution, sliding (section 0).

A 5 by 5 input with a vertical edge, every row 0, 0, 0.5, 1, 1, and a 3 by 3
filter whose every row is -1, 0, +1. The filter visits the nine positions
where it fits and writes one number per position into the feature map. The
map peaks where the window straddles the jump from dark to bright, which is
the point of the figure. Every output is computed here, not typed.

Input cells are shaded as pixel intensities on the same grey ramp in both
themes, because 0 means dark and 1 means bright whatever the page color.
"""

import numpy as np
from manim import (RIGHT, FadeIn, FadeOut, ManimColor, Rectangle, Scene, Text, TransformFromCopy,
                   VGroup, Write, interpolate_color)

from theme import FONT, MONO, P

X = np.tile([0.0, 0.0, 0.5, 1.0, 1.0], (5, 1))
W = np.tile([-1.0, 0.0, 1.0], (3, 1))
Y = np.array([[np.sum(X[r:r + 3, c:c + 3] * W) for c in range(3)] for r in range(3)])
assert np.allclose(Y, [[1.5, 3.0, 1.5]] * 3)
# Identical rows let one row's arithmetic stand for all three.
assert (X == X[0]).all() and (W == W[0]).all()

CELL, TOP = 0.95, 2.95
IN_X, W_X, OUT_X = -5.95, -0.45, 3.2  # left edges of the three grids
Y_EDGE, Y_ROW, Y_SUM = -2.3, -2.95, -3.55


def label(s, size=32, color=None, font=FONT):
    return Text(s, font=font, font_size=size, color=color or P["text"])


def fmt(v):
    return f"{v:g}".replace("-", "−")


def weight(w):
    return "0" if w == 0 else f"({w:+g})".replace("-", "−")


def at(left, r, c):
    return np.array([left + (c + 0.5) * CELL, TOP - (r + 0.5) * CELL, 0])


def cell(pos, fill, opacity):
    return Rectangle(width=CELL, height=CELL, fill_color=fill, fill_opacity=opacity, stroke_color=P["grid"],
                     stroke_width=2).move_to(pos)


def out_cell(r, c):
    share = Y[r, c] / Y.max()
    box = cell(at(OUT_X, r, c), P["slate"], 0.9 * share)
    text = label(f"{Y[r, c]:.1f}", 30, P["bg"] if share > 0.6 else P["text"], MONO).move_to(box)
    return VGroup(box, text)


def arithmetic(r, c):
    """The window's sum, written as one row times three."""
    row = X[r, c:c + 3]
    terms = " + ".join(f"{fmt(x)}×{weight(w)}" for x, w in zip(row, W[0]))
    s = float(row @ W[0])
    first = label(f"row: {terms} = {s:.1f}", 30, font=MONO).move_to([0, Y_ROW, 0])
    result = label(f"{3 * s:.1f}", 30, font=MONO)
    second = VGroup(label(f"3 rows: 3 × {s:.1f} =", 30, font=MONO), result).arrange(RIGHT, buff=0.32)
    return VGroup(first, second.move_to([0, Y_SUM, 0])), result


class ConvolutionSlide(Scene):
    def construct(self):
        grey = lambda v: interpolate_color(ManimColor("#3a3a3a"), ManimColor("#e5e5e5"), v)  # noqa: E731
        inputs = VGroup(*(VGroup(cell(at(IN_X, r, c), grey(X[r, c]), 1),
                                 label(fmt(X[r, c]), 30, "#fafafa" if X[r, c] < 0.5 else "#171717", MONO)
                                 .move_to(at(IN_X, r, c))) for r in range(5) for c in range(5)))
        filt = VGroup(*(VGroup(cell(at(W_X, r, c), P["gold"], 0.14).set_stroke(P["gold"], 2),
                               label(fmt(W[r, c]) if W[r, c] <= 0 else f"+{W[r, c]:g}", 30, font=MONO)
                               .move_to(at(W_X, r, c))) for r in range(3) for c in range(3)))
        empty = VGroup(*(cell(at(OUT_X, r, c), P["slate"], 0) for r in range(3) for c in range(3)))
        heads = VGroup(label("input", 32, P["muted"]).move_to([IN_X + 2.5 * CELL, TOP + 0.4, 0]),
                       label("3×3 filter", 32, P["muted"]).move_to([W_X + 1.5 * CELL, TOP + 0.4, 0]),
                       label("feature map", 32, P["muted"]).move_to([OUT_X + 1.5 * CELL, TOP + 0.4, 0]))
        # Drawn, not faded in: the first frame is what the page shows before playback.
        self.add(heads, inputs, filt, empty)
        self.wait(0.6)

        window = Rectangle(width=3 * CELL, height=3 * CELL, stroke_color=P["gold"], stroke_width=7)
        window.move_to(at(IN_X, 1, 1))
        self.play(FadeIn(window), run_time=0.6)
        lines, result = arithmetic(0, 0)
        self.play(Write(lines), run_time=1.6)
        self.wait(0.6)
        outs = {(r, c): out_cell(r, c) for r in range(3) for c in range(3)}
        # Manim adds a FadeIn's mobject after a copy made in the same play, so the box
        # gets its own play. Otherwise it would cover the number flying into it.
        self.play(FadeIn(outs[0, 0][0]), run_time=0.3)
        self.play(TransformFromCopy(result, outs[0, 0][1]), run_time=0.8)
        self.wait(0.4)
        self.play(FadeOut(lines), run_time=0.4)

        for r, c in [(r, c) for r in range(3) for c in range(3)][1:]:
            self.play(window.animate.move_to(at(IN_X, r + 1, c + 1)), run_time=0.35)
            self.play(FadeIn(outs[r, c]), run_time=0.25)

        # End with the window over the edge, where the map peaks.
        lines, _ = arithmetic(1, 1)
        peak = outs[1, 1][0].copy().set_fill(opacity=0).set_stroke(P["gold"], 7)
        edges = VGroup(label("edge", 30, P["muted"]).move_to([IN_X + 2.5 * CELL, Y_EDGE, 0]),
                       label("edge", 30, P["muted"]).move_to([OUT_X + 1.5 * CELL, TOP - 3 * CELL - 0.4, 0]))
        self.play(window.animate.move_to(at(IN_X, 2, 2)), FadeIn(peak), run_time=0.6)
        self.play(Write(lines), FadeIn(edges), run_time=1.2)
        self.wait(3.5)
