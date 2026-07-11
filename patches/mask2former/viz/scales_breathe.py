# patches/mask2former/viz/scales_breathe.py
"""Storyboard D: scales_breathe (~30 s + 2 s hold).

At 1/32 resolution a small object does not exist; at 1/8 everything exists
but there is far too much of it. The decoder breathes: one scale per layer,
coarse to fine, three times. The rejected alternative, all scales fused into
one slab, visibly sags with weight and slows the orb to a crawl.

Depth is staged with sheared panes and a moving camera; frost is an overlay
whose opacity is the resolution. The orb's own field is a miniature scene
that loads progressively, one layer per visited pane.
"""

import numpy as np
from manim import *  # noqa: F403 -- manim scenes conventionally star-import

from shapes import DUCK, silhouette
from tokens import BACKGROUND, GOLD, INK, SLATE

config.background_color = BACKGROUND

PANE_W, PANE_H, SHEAR = 7.4, 2.9, 1.15
P32_C = np.array([0.9, 2.0, 0.0])
P16_C = np.array([0.0, 0.1, 0.0])
P8_C = np.array([-0.9, -1.8, 0.0])

# Level embeddings as a lightness ramp within the slate family: dark for
# coarse, light for fine. Teal and gold are reserved nouns elsewhere in the
# series and must not label levels.
TINT_32 = interpolate_color(ManimColor(SLATE), ManimColor(INK), 0.5)
TINT_16 = ManimColor(SLATE)
TINT_8 = interpolate_color(ManimColor(SLATE), ManimColor(BACKGROUND), 0.45)

DUCKLING_LOCAL = np.array([-1.7, -0.6, 0.0])


def parallelogram(center, tint):
    p = Polygon([-PANE_W / 2, -PANE_H / 2, 0], [PANE_W / 2, -PANE_H / 2, 0],
                [PANE_W / 2 + SHEAR, PANE_H / 2, 0],
                [-PANE_W / 2 + SHEAR, PANE_H / 2, 0])
    # The level embedding is worn as an edge tint, never a label.
    p.set_stroke(tint, width=1.8, opacity=0.55)
    p.set_fill(SLATE, opacity=0.05)
    return p.move_to(center)


def water_blob(scale=1.0):
    pts = [(-2.1, -0.1), (-1.2, 0.25), (0.2, 0.15), (1.5, 0.3), (2.2, -0.1),
           (1.6, -0.5), (0.1, -0.65), (-1.4, -0.5)]
    vm = VMobject()
    arr = [np.array([x * scale, y * scale, 0]) for x, y in pts]
    vm.set_points_smoothly([*arr, arr[0]])
    vm.set_stroke(opacity=0).set_fill(SLATE, opacity=0.3)
    return vm


def duck_mass():
    # At 1/32 the duck is only a mass; features are not representable.
    pts = [(-0.7, 0.25), (0.0, 0.45), (0.7, 0.3), (0.9, -0.1), (0.3, -0.4),
           (-0.5, -0.35)]
    vm = VMobject()
    arr = [np.array([x, y, 0]) for x, y in pts]
    vm.set_points_smoothly([*arr, arr[0]])
    vm.set_stroke(opacity=0).set_fill(SLATE, opacity=0.5)
    return vm


def frost(center, opacity):
    f = Polygon([-PANE_W / 2, -PANE_H / 2, 0], [PANE_W / 2, -PANE_H / 2, 0],
                [PANE_W / 2 + SHEAR, PANE_H / 2, 0],
                [-PANE_W / 2 + SHEAR, PANE_H / 2, 0])
    f.set_stroke(opacity=0).set_fill(BACKGROUND, opacity=opacity)
    return f.move_to(center)


def lattice(center, cols, rows):
    # Token grid for one scale, following the pane's shear. Dot count doubles
    # per axis from coarse to fine, so panes 32/16/8 carry 1:4:16 dots, the real
    # stride 32/16/8 token ratio (1024:4096:16384).
    g = VGroup()
    for j in range(rows):
        fy = (j + 0.5) / rows
        for i in range(cols):
            fx = (i + 0.5) / cols
            g.add(Dot([center[0] - PANE_W / 2 - SHEAR / 2 + fx * PANE_W + SHEAR * fy,
                       center[1] - PANE_H / 2 + fy * PANE_H, 0],
                      radius=0.028).set_fill(SLATE, opacity=0.5))
    return g


class ScalesBreathe(MovingCameraScene):
    def beat(self, *anims, rt=1.0, **kw):
        self.play(*anims, run_time=rt, **kw)
        self.clock += rt

    def hold(self, t):
        self.wait(t)
        self.clock += t

    def construct(self):
        self.clock = 0.0
        self.camera.background_color = BACKGROUND

        # ---- build the three panes ----------------------------------------
        pane32 = parallelogram(P32_C, TINT_32)
        c32 = VGroup(water_blob(1.2).move_to(P32_C + np.array([-0.5, -0.3, 0])),
                     duck_mass().move_to(P32_C + np.array([1.3, 0.15, 0])))
        frost32 = frost(P32_C, 0.88)   # near-opaque: the duckling truly vanishes at stride 32

        pane16 = parallelogram(P16_C, TINT_16)
        smudge = water_blob(0.22).move_to(P16_C + DUCKLING_LOCAL)
        smudge.set_fill(SLATE, opacity=0.28)
        c16 = VGroup(water_blob(1.1).move_to(P16_C + np.array([-0.5, -0.35, 0])),
                     silhouette(DUCK, 0.55, SLATE, fill_opacity=0.45)
                     .move_to(P16_C + np.array([1.3, 0.15, 0])),
                     smudge)
        frost16 = frost(P16_C, 0.3)

        pane8 = parallelogram(P8_C, TINT_8)
        duckling = silhouette(DUCK, 0.16, SLATE, fill_opacity=0.75)
        duckling.move_to(P8_C + DUCKLING_LOCAL)
        c8 = VGroup(water_blob(1.05).move_to(P8_C + np.array([-0.5, -0.4, 0])),
                    silhouette(DUCK, 0.58, SLATE, fill_opacity=0.6)
                    .move_to(P8_C + np.array([1.3, 0.15, 0])),
                    duckling)
        frost8 = frost(P8_C, 0.06)

        # Token lattices, one per pane: 6x2, 12x4, 24x8 = 12:48:192 dots = the
        # 1:4:16 stride 32/16/8 ratio. Density, not mere presence, is the cost.
        lat32 = lattice(P32_C, 6, 2)
        lat16 = lattice(P16_C, 12, 4)
        lat8 = lattice(P8_C, 24, 8)
        lattices = VGroup(lat32, lat16, lat8)

        stack = [pane32, c32, frost32, lat32, pane16, c16, frost16, lat16,
                 pane8, c8, frost8, lat8]

        # ---- shot 1 (0:00-0:07): the vanishing ------------------------------
        frame = self.camera.frame
        frame.set(width=9.2).move_to(P8_C)
        for i, m in enumerate(stack):
            m.set_z_index(i)
        self.beat(LaggedStart(*[FadeIn(m) for m in reversed(stack)],
                              lag_ratio=0.06), rt=1.4)
        self.hold(0.3)

        # The dive: a traveling duckling dissolves on the way down, because at
        # 1/32 it is literally not representable.
        traveler = silhouette(DUCK, 0.24, SLATE, fill_opacity=0.85)
        traveler.move_to(P8_C + DUCKLING_LOCAL)
        self.add(traveler)
        mid_smudge = water_blob(0.3).move_to(P16_C + DUCKLING_LOCAL)
        mid_smudge.set_fill(SLATE, opacity=0.35)
        self.beat(frame.animate.move_to(P16_C),
                  Transform(traveler, mid_smudge), rt=2.6)
        gone = water_blob(0.12).move_to(P32_C + DUCKLING_LOCAL)
        gone.set_fill(SLATE, opacity=0.0)
        self.beat(frame.animate.move_to(P32_C).set(width=8.6),
                  Transform(traveler, gone), rt=1.6)
        self.remove(traveler)
        self.hold(1.1)

        # ---- shot 2 (0:07-0:12): the temptation -----------------------------
        self.beat(frame.animate.move_to(P8_C).set(width=9.2), rt=0.9)
        # Density as shimmer: the lattice breathes twice; staying here is
        # heavy.
        for _ in range(2):
            self.beat(lattices.animate(rate_func=there_and_back)
                      .set_opacity(0.8), rt=1.3)
        self.hold(1.5)

        # ---- shot 3 (0:12-0:22): the breath ---------------------------------
        self.beat(frame.animate.move_to([0.3, 0.1, 0]).set(width=14.2),
                  rt=1.2)

        orb_c = np.array([4.6, -3.0, 0.0])
        orb = Circle(radius=0.5).move_to(orb_c)
        orb.set_stroke(GOLD, width=2.5).set_fill(GOLD, opacity=0.05)
        halo = Circle(radius=0.54).move_to(orb_c)
        halo.set_stroke(GOLD, width=2, opacity=0.18)
        # The orb's field: a miniature scene that loads coarse-to-fine. No
        # strokes: outline-only is the series' void signature, and the field
        # must read as filling up, not as empty rings.
        f_water = water_blob(0.22).set_fill(SLATE, opacity=0.0)
        f_duck = silhouette(DUCK, 0.13, SLATE, fill_opacity=0.0)
        f_duck.set_stroke(opacity=0)
        f_ducklet = silhouette(DUCK, 0.055, SLATE, fill_opacity=0.0)
        f_ducklet.set_stroke(opacity=0)
        f_water.move_to(orb_c + np.array([0, -0.05, 0]))
        f_duck.move_to(orb_c + np.array([0.12, 0.1, 0]))
        f_ducklet.move_to(orb_c + np.array([-0.16, -0.12, 0]))
        # No VGroup parent: every part is top-level and moves by the same
        # delta in one play. Group parentage plus standalone member animation
        # desyncs the family; this cannot.
        orb_parts = [orb, halo, f_water, f_duck, f_ducklet]
        # The query rides above every frost overlay: frost encodes each pane's
        # resolution and must never wash out the reader doing the reading.
        for m in orb_parts:
            m.set_z_index(len(stack) + 1)
        self.beat(*[FadeIn(m, scale=0.85) for m in orb_parts], rt=0.6)

        def orb_move(target, extra=(), rt=0.6):
            delta = target - orb.get_center()
            self.beat(*[m.animate.shift(delta) for m in orb_parts],
                      *extra, rt=rt)

        ticks = VGroup(*[Line([5.4 + 0.17 * k, -3.78, 0],
                              [5.4 + 0.17 * k, -3.54, 0])
                         .set_stroke(SLATE, width=1.6, opacity=0.15)
                         for k in range(9)])
        self.add(ticks)

        visit = {32: P32_C + np.array([3.0, -0.2, 0]),
                 16: P16_C + np.array([3.0, -0.2, 0]),
                 8: P8_C + np.array([3.0, -0.2, 0])}

        # Each visit is a move beat then a drink beat. Fill animations are
        # built inside the drink beat: an .animate builder snapshots its
        # target at creation, so one built before the move would drag the
        # layer back to a stale position.
        def sip(target, tick_idx, mob, op, rt_move, rt_drink):
            orb_move(target,
                     extra=[ticks[tick_idx].animate
                            .set_stroke(INK, width=2.2, opacity=0.7)],
                     rt=rt_move)
            self.beat(halo.animate(rate_func=there_and_back)
                      .set_stroke(opacity=0.4),
                      mob.animate.set_fill(opacity=op), rt=rt_drink)

        def breath(k, water_op, duck_op, ducklet_op, rt_move, rt_drink):
            sip(visit[32], 3 * k, f_water, water_op, rt_move, rt_drink)
            sip(visit[16], 3 * k + 1, f_duck, duck_op, rt_move, rt_drink)
            sip(visit[8], 3 * k + 2, f_ducklet, ducklet_op,
                rt_move, rt_drink)

        breath(0, 0.5, 0.5, 0.0, 0.6, 0.4)
        breath(1, 0.55, 0.62, 0.6, 0.55, 0.35)
        breath(2, 0.6, 0.72, 0.9, 0.5, 0.3)
        self.hold(0.1)

        # ---- shot 4 (0:22-0:30): the alternative, rejected -------------------
        for m in stack:
            m.save_state()
        slab_c = np.array([0.0, 0.1, 0.0])
        self.beat(*[m.animate.shift(slab_c - P32_C)
                    for m in [pane32, c32, frost32, lat32]],
                  *[m.animate.shift(slab_c - P8_C)
                    for m in [pane8, c8, frost8, lat8]],
                  rt=2.0)
        slab = VGroup(*stack)
        # The slab sags with weight: it drops, squashes, and bows.
        slab.generate_target()
        slab.target.shift(DOWN * 0.45).stretch(0.93, dim=1)

        def bow(p):
            sag = 0.3 * max(0.0, 1 - ((p[0] - slab_c[0]) / 4.5) ** 2)
            return p + DOWN * sag

        slab.target.apply_function(bow)
        delta_sag = (visit[16] + DOWN * 0.55) - orb.get_center()
        # Under the slab the ring dims to half: burdened, but never buried.
        self.beat(MoveToTarget(slab),
                  orb.animate.shift(delta_sag).set_stroke(opacity=0.5),
                  *[m.animate.shift(delta_sag) for m in orb_parts[1:]],
                  rt=1.1)
        self.beat(*[m.animate.shift(LEFT * 1.6) for m in orb_parts], rt=2.2,
                  rate_func=rate_functions.ease_in_out_sine)
        # The panes separate; the breath resumes, light.
        delta_res = visit[8] - orb.get_center()
        self.beat(*[Restore(m) for m in stack],
                  orb.animate.shift(delta_res).set_stroke(opacity=1.0),
                  *[m.animate.shift(delta_res) for m in orb_parts[1:]],
                  rt=1.5)
        orb_move(visit[32], rt=0.8)
        self.beat(halo.animate(rate_func=there_and_back)
                  .set_stroke(opacity=0.45), rt=0.3)
        self.hold(0.1)
        print(f"scene clock: {self.clock:.2f} s")
        self.hold(2.0)
