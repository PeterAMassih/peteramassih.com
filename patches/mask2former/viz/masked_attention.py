# patches/mask2former/viz/masked_attention.py
"""Storyboard B: masked_attention (~42 s + 2 s hold).

Cross-attention forces a query to drink the whole image; the mask is a
stencil that cuts the background strands, and the survivors thicken until the
beam's total width is exactly what it was: renormalization as conserved
thickness. The stencil and the understanding sharpen each other, breath by
breath, with one failsafe breath where the hole collapses and the stencil
dissolves for a layer (the empty-mask guard, never named).

Strands are static beziers re-targeted by Transform between precomputed
phases; the throat span never changes, so conservation is carried by the
geometry, not by an updater.
"""

import numpy as np
from manim import *  # noqa: F403 -- manim scenes conventionally star-import

from shapes import CAT, silhouette
from tokens import BACKGROUND, FONT_BODY, GOLD, INK, MUTED, SLATE

config.background_color = BACKGROUND

FIELD_C = np.array([0.0, -1.5, 0.0])
FIELD_W, FIELD_H = 11.6, 3.8
CAT_C = np.array([-2.3, -1.55, 0.0])
CAT_SCALE = 0.65
ORB_C = np.array([0.0, 2.35, 0.0])
THROAT_Y = 1.55
THROAT_W = 1.1

# The cat is the one warm thing in a cool world; everything else in the image
# is raw slate. Derived from tokens, never an invented hex.
WARM = interpolate_color(ManimColor(GOLD), ManimColor(MUTED), 0.22)
MUD = interpolate_color(ManimColor(SLATE), ManimColor(WARM), 0.3)

TOTAL_WIDTH = 92.4  # sum of strand stroke widths, conserved across phases


def in_ellipse(p, scale, off=(0.0, 0.0)):
    dx = (p[0] - CAT_C[0] - off[0]) / (0.85 * scale)
    dy = (p[1] - CAT_C[1] - off[1]) / (0.68 * scale)
    return dx * dx + dy * dy <= 1.0


def in_polygon(p, poly):
    x, y = p[0], p[1]
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


def make_origins():
    # Warm strands must be rooted strictly inside the cat, or the tightened
    # stencil would appear to let the query drink through the mask.
    inset = silhouette(CAT, CAT_SCALE * 0.86, "#000000").move_to(CAT_C)
    cat_poly = np.array([inset.point_from_proportion(t)
                         for t in np.linspace(0, 1, 160, endpoint=False)])
    rng = np.random.default_rng(7)
    cat_pts = []
    while len(cat_pts) < 14:
        p = CAT_C + np.array([rng.uniform(-0.62, 0.62),
                              rng.uniform(-0.62, 0.62), 0.0])
        if in_polygon(p, cat_poly):
            cat_pts.append(p)
    bg_pts = []
    while len(bg_pts) < 70:
        p = np.array([rng.uniform(-FIELD_W / 2 + 0.3, FIELD_W / 2 - 0.3),
                      rng.uniform(-3.25, 0.25), 0.0])
        if not in_ellipse(p, 1.25):
            bg_pts.append(p)
    return cat_pts, bg_pts


def strand_curve(origin, slot_x, width, color, opacity, cut=1.0):
    # Rises from its pixel through its throat slot and on to the orb's
    # underside, so the drink is causally closed: no wedge, no severed tips.
    q = ORB_C + DOWN * 0.44 + RIGHT * slot_x * 0.22
    c1 = origin + UP * 1.3
    c2 = np.array([slot_x, THROAT_Y, 0.0])

    def path(t):
        s = t * cut
        return ((1 - s) ** 3 * origin + 3 * (1 - s) ** 2 * s * c1
                + 3 * (1 - s) * s * s * c2 + s ** 3 * q)

    m = ParametricFunction(path, t_range=[0, 1, 1 / 36])
    m.set_stroke(color, width=width, opacity=opacity)
    return m


class MaskedAttention(MovingCameraScene):
    def beat(self, *anims, rt=1.0, **kw):
        self.play(*anims, run_time=rt, **kw)
        self.clock += rt

    def hold(self, t):
        self.wait(t)
        self.clock += t

    def phase_targets(self, alive):
        # Alive strands share the conserved total width and evenly split the
        # throat; dead strands retract toward their origin and dim out. Slot
        # order is a fixed shuffle so warm and cool strands interleave: the
        # bundle mouth reads as a braid, not two sorted blocks.
        idxs = [i for i in self.slot_order if i in set(alive)]
        n = len(idxs)
        # No cap: the survivors must sum to exactly TOTAL_WIDTH so the braid holds
        # its width through the masking beat. That constancy is the renormalization
        # the scene exists to show. At the fewest survivors (n=14) each strand is
        # 6.6 wide, still a clean stroke.
        w = TOTAL_WIDTH / max(n, 1)
        targets = {}
        for rank, i in enumerate(idxs):
            slot = -THROAT_W / 2 + THROAT_W * (rank + 0.5) / n
            warm = i < 14
            color = WARM if warm else SLATE
            # Warm strands read brighter than the cool crowd, so the few warm
            # among many cool is visible during the full drink (the small but
            # nonzero foreground share: softmax never gives it zero). Widths
            # stay equal, so the total mass is still conserved.
            base = 0.62 if n < 30 else 0.5
            op = base if warm else base * 0.6
            targets[i] = strand_curve(self.origins[i], slot, w, color, op)
        for i in range(len(self.origins)):
            if i not in targets:
                color = WARM if i < 14 else SLATE
                targets[i] = strand_curve(self.origins[i],
                                          self.origins[i][0], 0.4,
                                          color, 0.0, cut=0.25)
        return targets

    def retarget(self, alive, rt, **kw):
        targets = self.phase_targets(alive)
        self.beat(*[Transform(self.strands[i], targets[i])
                    for i in range(len(self.strands))], rt=rt, **kw)

    def make_plate(self, hole):
        rect = Rectangle(width=FIELD_W + 0.3, height=FIELD_H + 0.25)
        rect.move_to(FIELD_C)
        if hole is None:
            plate = rect
        else:
            plate = Difference(rect, hole)
        plate.set_fill(INK, opacity=0.09)
        plate.set_stroke(SLATE, width=1.0, opacity=0.35)
        return plate

    def construct(self):
        self.clock = 0.0
        self.camera.background_color = BACKGROUND

        cat_pts, bg_pts = make_origins()
        self.origins = cat_pts + bg_pts
        self.slot_order = list(np.random.default_rng(5).permutation(84))

        field = Rectangle(width=FIELD_W, height=FIELD_H).move_to(FIELD_C)
        field.set_stroke(opacity=0).set_fill(SLATE, opacity=0.1)
        cat = silhouette(CAT, CAT_SCALE, WARM, fill_opacity=0.55)
        cat.set_stroke(WARM, width=1.2, opacity=0.5).move_to(CAT_C)

        orb = Circle(radius=0.42).move_to(ORB_C)
        orb.set_stroke(GOLD, width=2.5).set_fill(GOLD, opacity=0.12)
        # The halo hugs the rim: the series allows nothing beyond a subtle
        # halo on gold.
        halo = Circle(radius=0.455).move_to(ORB_C)
        halo.set_stroke(GOLD, width=2, opacity=0.18)

        # ---- shot 1 (0:00-0:06): the drink --------------------------------
        self.beat(FadeIn(field), FadeIn(cat), rt=0.7)
        self.beat(Create(orb), FadeIn(halo), rt=0.8)
        # The orb glows and the whole field answers: it has no choice.
        self.beat(field.animate.set_fill(opacity=0.16),
                  cat.animate.set_fill(opacity=0.65),
                  halo.animate.set_stroke(opacity=0.35), rt=0.6)
        first = self.phase_targets(range(84))
        self.strands = [first[i] for i in range(84)]
        self.beat(LaggedStart(*[Create(s) for s in self.strands],
                              lag_ratio=0.015), rt=2.1)
        self.beat(orb.animate.set_fill(MUD, opacity=0.8), rt=0.9)
        self.hold(0.9)

        # ---- shot 2 (0:06-0:12): why the background wins -------------------
        frame = self.camera.frame
        frame.save_state()
        self.beat(frame.animate.set(width=7.0).move_to([-3.6, -1.4, 0]),
                  rt=2.0)
        # The deliberate scan: the one linear move in the scene.
        self.beat(frame.animate.move_to([3.0, -1.4, 0]), rt=2.2,
                  rate_func=linear)
        self.beat(Restore(frame), rt=1.8)

        # ---- shot 3 (0:12-0:20): the stencil -------------------------------
        # Imperfect and slightly offset, but still visibly centered on the
        # cat: too much offset reads as a bug, not a rough first mask.
        hole1 = Ellipse(width=2 * 0.85 * 1.45, height=2 * 0.68 * 1.45)
        hole1.move_to(CAT_C + np.array([0.25, 0.14, 0.0]))
        plate = self.make_plate(hole1)
        engraving = Text("−∞", font=FONT_BODY, font_size=64, color=MUTED)
        engraving.scale(0.5).move_to([3.6, -1.5, 0]).set_opacity(0.0)
        self.add(engraving)
        plate.shift(LEFT * 13)
        self.add(plate)
        self.beat(plate.animate.shift(RIGHT * 13),
                  engraving.animate.set_opacity(0.5), rt=2.2)

        alive1 = [i for i in range(84)
                  if i < 14 or in_ellipse(self.origins[i], 1.45,
                                          (0.25, 0.14))]
        targets1 = self.phase_targets(alive1)
        self.hold(0.3)
        # The conservation beat, the most important seconds of the scene: the
        # dying strands' width drains INTO the survivors in the same motion,
        # so the bundle's total width never dips. One slow simultaneous
        # retarget, not a cut followed by a thicken.
        self.beat(*[Transform(self.strands[i], targets1[i])
                    for i in range(84)], rt=2.8)
        self.beat(orb.animate.set_fill(
            interpolate_color(MUD, ManimColor(WARM), 0.6), opacity=0.82),
            rt=1.2)
        self.hold(1.5)

        # ---- shot 4 (0:20-0:26): the loop ----------------------------------
        def breathe(hole, alive, warm_mix, rt_morph, rt_cut):
            self.beat(halo.animate(rate_func=there_and_back)
                      .set_stroke(opacity=0.4),
                      Transform(plate, self.make_plate(hole)), rt=rt_morph)
            targets = self.phase_targets(alive)
            self.beat(*[Transform(self.strands[i], targets[i])
                        for i in range(84)],
                      orb.animate.set_fill(
                          interpolate_color(MUD, ManimColor(WARM), warm_mix),
                          opacity=0.85), rt=rt_cut)

        hole2 = Ellipse(width=2 * 0.85 * 1.25, height=2 * 0.68 * 1.25)
        hole2.move_to(CAT_C + np.array([0.2, 0.1, 0.0]))
        alive2 = [i for i in range(84)
                  if i < 14 or in_ellipse(self.origins[i], 1.25, (0.2, 0.1))]
        breathe(hole2, alive2, 0.75, 0.9, 1.0)

        hole3 = silhouette(CAT, CAT_SCALE * 1.06, INK, fill_opacity=1)
        hole3.move_to(CAT_C)
        alive3 = list(range(14))
        breathe(hole3, alive3, 0.9, 0.9, 1.0)
        # Third breath: the stencil is true now; the light only settles.
        self.beat(halo.animate(rate_func=there_and_back)
                  .set_stroke(opacity=0.4),
                  orb.animate.set_fill(WARM, opacity=0.88), rt=1.4)
        self.hold(0.6)

        # ---- shot 5 (0:26-0:32): the failsafe ------------------------------
        # One breath goes wrong: the hole closes entirely.
        dead = self.phase_targets([])
        self.beat(Transform(plate, self.make_plate(None)),
                  *[Transform(self.strands[i], dead[i]) for i in range(84)],
                  orb.animate.set_fill(GOLD, opacity=0.12), rt=1.4)
        # For that layer the stencil simply dissolves: one open drink.
        open_targets = self.phase_targets(range(84))
        self.beat(plate.animate.set_fill(opacity=0.0)
                  .set_stroke(opacity=0.0),
                  engraving.animate.set_opacity(0.0), rt=0.8)
        self.beat(*[Transform(self.strands[i], open_targets[i])
                    for i in range(84)],
                  orb.animate.set_fill(MUD, opacity=0.8), rt=1.2)
        self.hold(0.4)
        # Recover: redraw the stencil and cut again. Transform interpolates
        # style too, so the dissolved plate fades back in as it reshapes.
        self.beat(Transform(plate, self.make_plate(hole3)),
                  engraving.animate.set_opacity(0.5), rt=1.0)
        targets3 = self.phase_targets(alive3)
        self.beat(*[Transform(self.strands[i], targets3[i])
                    for i in range(84)],
                  orb.animate.set_fill(WARM, opacity=0.88), rt=1.2)

        # ---- shot 6 (0:32-0:42): residue ------------------------------------
        big_open = Text("(", font=FONT_BODY, font_size=120, color=MUTED)
        big_open.scale_to_fit_height(3.2).move_to([-4.35, -1.55, 0])
        big_open.set_opacity(0.6)
        big_close = Text(")", font=FONT_BODY, font_size=120, color=MUTED)
        big_close.scale_to_fit_height(3.2).move_to([-0.25, -1.55, 0])
        big_close.set_opacity(0.6)
        self.beat(Create(big_open), Create(big_close), rt=1.2)
        self.hold(0.3)

        def small(s, size=56):
            t = Text(s, font=FONT_BODY, font_size=size, color=INK)
            return t.scale(0.5).set_opacity(0.75)

        eq_softmax = small("softmax")
        eq_open = small("(")
        eq_qk = small("QK")
        eq_t = small("T", size=36)
        eq_plus_m = small("+ M")
        eq_close = small(")")
        eq_v = small("V")
        eq_softmax.move_to([-1.9, -3.68, 0])
        eq_open.next_to(eq_softmax, RIGHT, buff=0.06)
        eq_qk.next_to(eq_open, RIGHT, buff=0.08)
        eq_t.next_to(eq_qk, UR, buff=0.01).shift(DOWN * 0.12 + LEFT * 0.02)
        eq_plus_m.next_to(eq_t, RIGHT, buff=0.1).align_to(eq_qk, DOWN)
        eq_close.next_to(eq_plus_m, RIGHT, buff=0.08)
        eq_v.next_to(eq_close, RIGHT, buff=0.08)

        # The stencil itself becomes the +M; the parentheses it flew into are
        # the ones that were just drawn around the strands' origin.
        self.beat(Transform(big_open, eq_open), Transform(big_close, eq_close),
                  ReplacementTransform(plate, eq_plus_m),
                  FadeOut(engraving), rt=2.0)
        self.beat(FadeIn(eq_softmax), FadeIn(eq_qk), FadeIn(eq_t),
                  FadeIn(eq_v), rt=1.0)
        self.hold(3.1)
        # Retarget strands to rest exactly as they are (no-op settle keeps the
        # tableau alive to the eye).
        self.beat(halo.animate.set_stroke(opacity=0.25), rt=0.8)
        self.hold(1.6)
        self.hold(2.0)
        print(f"scene clock: {self.clock:.2f} s")
