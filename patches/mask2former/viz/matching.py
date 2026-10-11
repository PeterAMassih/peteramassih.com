# patches/mask2former/viz/matching.py
"""Cheapest first versus the best total (the matching section).

The worked 3 by 3 cost matrix from the article's Hungarian fold, rows for
predictions and columns for targets. Taking the cheapest pair first locks in
the 0, then the 2, and is left with the 4, a total of 6. The best assignment
accepts one slightly worse pair and pays 5. Both are computed here, the best
one by trying all six assignments, so the figure cannot drift from the fold.
"""

import itertools

import numpy as np
from manim import Circle, Create, FadeIn, LaggedStart, Rectangle, Scene, Text, VGroup, Write

from theme import FONT, MONO, P

C = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])


def greedy(cost):
    """Repeatedly take the cheapest pair among unused rows and columns."""
    rows, cols, picks = set(range(3)), set(range(3)), []
    while rows:
        r, c = min(((r, c) for r in rows for c in cols), key=lambda rc: cost[rc])
        picks.append((r, c))
        rows.remove(r)
        cols.remove(c)
    return picks


GREEDY = greedy(C)
BEST = list(enumerate(min(itertools.permutations(range(3)), key=lambda p: sum(C[r, c] for r, c in enumerate(p)))))
assert [C[p] for p in GREEDY] == [0, 2, 4] and sum(C[p] for p in BEST) == 5

CELL, Y_MID, Y_TITLE = 1.0, 0.5, 3.1
LEFT_X, RIGHT_X = -3.3, 3.3  # matrix centers
Y_TOTAL, Y_KEY = -1.8, -3.05


def label(s, size=32, color=None, font=FONT):
    return Text(s, font=font, font_size=size, color=color or P["text"])


def at(x0, r, c):
    return np.array([x0 + (c - 1) * CELL, Y_MID - (r - 1) * CELL, 0])


def matrix(x0):
    """Cells, costs, gold prediction numbers on the left, green target numbers on top."""
    cells = VGroup(*(Rectangle(width=CELL, height=CELL, stroke_color=P["grid"], stroke_width=2).move_to(at(x0, r, c))
                     for r in range(3) for c in range(3)))
    costs = VGroup(*(label(str(C[r, c]), 40, font=MONO).move_to(at(x0, r, c)) for r in range(3) for c in range(3)))
    rows = VGroup(*(label(str(r + 1), 32, P["gold_text"], MONO).move_to(at(x0, r, 0) - [0.95, 0, 0]) for r in range(3)))
    cols = VGroup(*(label(str(c + 1), 32, P["green"], MONO).move_to(at(x0, 0, c) + [0, 0.85, 0]) for c in range(3)))
    return VGroup(cells, costs, rows, cols)


def ring(x0, rc, color):
    return Circle(radius=0.38, stroke_color=color, stroke_width=6).move_to(at(x0, *rc))


class GreedyVsBest(Scene):
    def construct(self):
        left, right = matrix(LEFT_X), matrix(RIGHT_X)
        titles = VGroup(label("cheapest first", 34, P["ember"]).move_to([LEFT_X, Y_TITLE, 0]),
                        label("best total", 34).move_to([RIGHT_X, Y_TITLE, 0]))
        key = VGroup(label("rows: predictions", 30, P["gold_text"]),
                     label("columns: targets", 30, P["green"])).arrange(buff=1.0).move_to([0, Y_KEY, 0])
        # Drawn, not faded in: the first frame is what the page shows before playback.
        self.add(left, titles[0], key)
        self.wait(0.6)

        # The running total is laid out whole and revealed piece by piece, so it never
        # shifts or morphs while it grows.
        parts = [str(C[GREEDY[0]])] + [f"+ {C[p]}" for p in GREEDY[1:]] + [f"= {sum(C[p] for p in GREEDY)}"]
        total = VGroup(*(label(t, 36, P["ember"], MONO) for t in parts)).arrange(buff=0.3)
        total.move_to([LEFT_X, Y_TOTAL, 0])
        for i, (r, c) in enumerate(GREEDY):
            shown = [total[i]] + ([total[3]] if i == 2 else [])
            self.play(Create(ring(LEFT_X, (r, c), P["ember"])), *(FadeIn(t) for t in shown), run_time=0.7)
            # A used row and column are out of the running for later picks.
            used = [3 * rr + cc for rr in range(3) for cc in range(3) if (rr == r or cc == c) and (rr, cc) != (r, c)]
            self.play(*(left[1][j].animate.set_opacity(0.3) for j in used), run_time=0.4)
            self.wait(0.3)
        self.wait(0.8)

        self.play(FadeIn(right), FadeIn(titles[1]), run_time=0.9)
        others = [3 * r + c for r in range(3) for c in range(3) if (r, c) not in BEST]
        self.play(LaggedStart(*(Create(ring(RIGHT_X, rc, P["text"])) for rc in BEST), lag_ratio=0.3),
                  *(right[1][j].animate.set_opacity(0.3) for j in others), run_time=1.3)
        best = " + ".join(str(C[p]) for p in BEST) + f" = {sum(C[p] for p in BEST)}"
        self.play(Write(label(best, 36, font=MONO).move_to([RIGHT_X, Y_TOTAL, 0])), run_time=0.9)
        self.wait(3.5)
