# patches/mask2former/viz/query_becomes_segment.py
"""Storyboard C: query_becomes_segment (~35 s + 2 s hold).

A query is a slot that becomes one segment. The orbs cast rough affinity
fields before ever touching the decoder (learnable, supervised X0), the nine
gates sharpen those fields until they claim whole objects, a prism splits
where from what, and at the end the same fields regroup three ways in place:
semantic, instance, panoptic. Nothing re-runs; only the grouping changes.

The image is the same two-ducks-and-a-dog scene as hungarian_matching, for
continuity across the series.
"""

import numpy as np
from manim import *  # noqa: F403 -- manim scenes conventionally star-import

from shapes import DOG, DUCK, silhouette
from tokens import (ACCENT, BACKGROUND, FONT_BODY, GOLD, HOLLOW, INK, MUTED,
                    SLATE)

config.background_color = BACKGROUND

FIELD_C = np.array([0.2, -1.2, 0.0])
DUCK_A = np.array([-3.2, -1.0, 0.0])
DUCK_B = np.array([0.6, -2.2, 0.0])
DOG_P = np.array([3.6, -1.0, 0.0])
FILE_Y = 3.0
GATE_XS = [-1.2 + 0.6 * k for k in range(9)]

GOLD_LIGHT = interpolate_color(ManimColor(GOLD), ManimColor(BACKGROUND), 0.4)
GOLD_DEEP = interpolate_color(ManimColor(GOLD), ManimColor(INK), 0.25)
ACCENT_LIGHT = interpolate_color(ManimColor(ACCENT), ManimColor(BACKGROUND),
                                 0.6)
# A second instance of the same class needs a clearly different tint: dark teal
# against teal reads at panel scale where a pale tint does not.
ACCENT_DARK = interpolate_color(ManimColor(ACCENT), ManimColor(INK), 0.5)


def soft_blob(seed, scale, center):
    # A rough affinity field: lumpy, unsure of its edges.
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.08, 0.2, 3)
    phases = rng.uniform(0, 2 * np.pi, 3)
    pts = []
    for k in range(12):
        th = 2 * np.pi * k / 12
        r = 1.0 + sum(a * np.sin((i + 2) * th + p)
                      for i, (a, p) in enumerate(zip(amps, phases)))
        pts.append(np.array([r * np.cos(th) * scale,
                             r * np.sin(th) * scale * 0.75, 0.0]))
    vm = VMobject()
    vm.set_points_smoothly([*pts, pts[0]])
    return vm.move_to(center)


def soft_field(shape, color, base_op=0.16):
    # Soft luminance: three nested layers standing in for a falloff, since
    # Cairo has no blur. Fill only; outlines arrive with crystallization.
    # Every layer is clipped to the image: an affinity field is a dot
    # product with pixel embeddings and cannot exist off the image.
    clip = Rectangle(width=11.2, height=4.4).move_to(FIELD_C)
    layers = VGroup()
    for s, f in [(1.0, 1.0), (1.2, 0.55), (1.45, 0.3)]:
        m = Intersection(shape.copy().scale(s), clip)
        m.set_stroke(opacity=0).set_fill(color, opacity=base_op * f)
        layers.add(m)
    return layers


class QueryBecomesSegment(Scene):
    def beat(self, *anims, rt=1.0, **kw):
        self.play(*anims, run_time=rt, **kw)
        self.clock += rt

    def hold(self, t):
        self.wait(t)
        self.clock += t

    def construct(self):
        self.clock = 0.0
        self.camera.background_color = BACKGROUND

        field = Rectangle(width=11.2, height=4.4).move_to(FIELD_C)
        field.set_stroke(opacity=0).set_fill(SLATE, opacity=0.1)
        duck_a = silhouette(DUCK, 0.62, SLATE, fill_opacity=0.5)
        duck_a.move_to(DUCK_A)
        duck_b = silhouette(DUCK, 0.55, SLATE, fill_opacity=0.5, mirror=True)
        duck_b.move_to(DUCK_B)
        dog = silhouette(DOG, 0.68, SLATE, fill_opacity=0.62).move_to(DOG_P)

        # ---- shot 1 (0:00-0:06): born with opinions -------------------------
        self.beat(FadeIn(field), FadeIn(duck_a), FadeIn(duck_b), FadeIn(dog),
                  rt=0.9)

        orbs = VGroup()
        for i in range(10):
            o = Circle(radius=0.16).move_to([-6.5 + 0.52 * i, FILE_Y, 0])
            fade = 1.0 if i < 7 else (0.7, 0.45, 0.25)[i - 7]
            o.set_stroke(GOLD, width=2.0, opacity=fade)
            o.set_fill(GOLD, opacity=0.35 * fade)
            # The tail of the file shrinks away: the rest of the hundred.
            o.scale(1.0 if i < 7 else (0.85, 0.7, 0.55)[i - 7])
            orbs.add(o)
        self.beat(LaggedStart(*[Create(o) for o in orbs], lag_ratio=0.08),
                  rt=1.4)

        # Rough fields before any decoding: proposals from birth.
        glow_a = soft_field(soft_blob(3, 1.3, DUCK_A + np.array([0.9, 0.5, 0])),
                            GOLD)
        glow_b = soft_field(soft_blob(5, 1.5, (DUCK_A + DUCK_B) / 2), GOLD)
        glow_d = soft_field(soft_blob(8, 1.4, DOG_P + np.array([-0.6, -0.7, 0])),
                            GOLD)
        glow_x = soft_field(soft_blob(13, 1.1, np.array([0.7, -1.8, 0.0])),
                            GOLD, base_op=0.11)
        # Each lead orb pulses as its field lands: the claim is bound to the
        # claimant by timing, not by a labeled arrow.
        self.beat(LaggedStart(
            *[AnimationGroup(orbs[i].animate(rate_func=there_and_back)
                             .scale(1.35), FadeIn(g))
              for i, g in enumerate([glow_a, glow_b, glow_d, glow_x])],
            lag_ratio=0.32), rt=2.2)
        self.hold(1.5)

        # ---- shot 2 (0:06-0:16): crystallization ----------------------------
        gates = VGroup(*[Line([x, FILE_Y - 0.35, 0], [x, FILE_Y + 0.35, 0])
                         .set_stroke(SLATE, width=1.6, opacity=0.25)
                         for x in GATE_XS])
        self.beat(Create(gates, lag_ratio=0.06), rt=0.8)

        def flashes(idxs):
            return LaggedStart(*[gates[k].animate(rate_func=there_and_back)
                                 .set_stroke(opacity=0.55) for k in idxs],
                               lag_ratio=0.5)

        # Gates 1-3: fields tighten toward what they almost are.
        self.beat(orbs.animate.shift(RIGHT * 1.8), flashes([0, 1, 2]),
                  Transform(glow_a, soft_field(
                      soft_blob(4, 1.0, DUCK_A + np.array([0.35, 0.2, 0])),
                      GOLD, 0.2)),
                  Transform(glow_b, soft_field(
                      soft_blob(6, 1.1, DUCK_A + np.array([0.8, -0.4, 0])),
                      GOLD, 0.2)),
                  Transform(glow_d, soft_field(
                      soft_blob(9, 1.05, DOG_P + np.array([-0.2, -0.3, 0])),
                      GOLD, 0.2)),
                  Transform(glow_x, soft_field(
                      soft_blob(14, 0.9, np.array([0.7, -1.7, 0.0])),
                      GOLD, 0.09)),
                  rt=3.2)
        self.hold(0.9)

        # Two fields overlap. The arc shows query communication, while the text
        # names the one-to-one training constraint that discourages duplicates.
        # The later separation is schematic, not a traced causal mechanism.
        arc = ArcBetweenPoints(orbs[0].get_top() + UP * 0.04,
                               orbs[1].get_top() + UP * 0.04, angle=-PI / 2)
        arc.set_stroke(GOLD, width=2.2, opacity=0.85)
        match_note = Text("one target per query", font=FONT_BODY, font_size=22,
                          color=MUTED).move_to([0.1, 2.25, 0])
        self.beat(Create(arc),
                  FadeIn(match_note),
                  orbs[0].animate(rate_func=there_and_back).scale(1.35),
                  orbs[1].animate(rate_func=there_and_back).scale(1.35),
                  glow_a.animate(rate_func=there_and_back)
                  .set_fill(opacity=0.3),
                  glow_b.animate(rate_func=there_and_back)
                  .set_fill(opacity=0.3),
                  rt=0.9)
        self.hold(0.4)
        self.beat(FadeOut(arc), FadeOut(match_note),
                  Transform(glow_b, soft_field(
                      soft_blob(7, 1.0, DUCK_B + np.array([0.2, 0.3, 0])),
                      GOLD, 0.2)), rt=0.95)

        # Gates 4-9: the yielded orb claims the unclaimed duck; every field
        # snaps to a crisp silhouette.
        crisp_a = soft_field(silhouette(DUCK, 0.62 * 1.05, GOLD)
                             .move_to(DUCK_A), GOLD, 0.28)
        crisp_b = soft_field(silhouette(DUCK, 0.55 * 1.05, GOLD, mirror=True)
                             .move_to(DUCK_B), GOLD, 0.28)
        crisp_d = soft_field(silhouette(DOG, 0.62 * 1.05, GOLD)
                             .move_to(DOG_P), GOLD, 0.28)
        self.beat(orbs.animate.shift(RIGHT * 3.4), flashes([3, 4, 5, 6, 7, 8]),
                  Transform(glow_a, crisp_a), Transform(glow_b, crisp_b),
                  Transform(glow_d, crisp_d),
                  glow_x.animate.set_fill(opacity=0.0), rt=3.15)

        # ---- shot 3 (0:16-0:22): the prism ----------------------------------
        # The class head as refraction: an orb glides into a prism and its
        # light fans into candidate tints; one lobe survives and the orb rides
        # out wearing it, while its field takes the same tint. Where stays in
        # the field, what arrives as color. Gates leave first: they are done.
        self.beat(FadeOut(gates), rt=0.6)

        prism_c = np.array([0.0, 1.9, 0.0])
        prism = Polygon([-0.46, -0.38, 0], [0.46, -0.38, 0], [0.0, 0.5, 0])
        prism.set_stroke(SLATE, width=2.2, opacity=0.85)
        prism.set_fill(INK, opacity=0.05).move_to(prism_c)
        self.beat(Create(prism), rt=0.8)

        def wedge(color, ang, op, hollow=False):
            # Outer wedges shorten with angle so the fan clears the image
            # panel below and the orb file above.
            L = 1.45 - 0.011 * abs(ang)
            w = Polygon([0, 0, 0], [L, 0.18 * L, 0], [L, -0.18 * L, 0])
            if hollow:  # the void class: outline only, the series' empty-set mark
                w.set_stroke(SLATE, width=1.4, opacity=0.5).set_fill(opacity=0)
            else:
                w.set_stroke(width=0).set_fill(color, opacity=op)
            return w.rotate(ang * DEGREES, about_point=ORIGIN).shift(
                prism_c + np.array([0.28, -0.05, 0]))

        def refract(orb, glow, tint, wide):
            slot = orb.get_center()
            self.beat(orb.animate.move_to(prism_c + LEFT * 0.85), rt=0.55)
            self.beat(orb.animate.scale(0.7).move_to(prism_c), rt=0.4)
            # Class selection, not a physics rainbow: the candidate classes fan
            # out, then all but the true one are pulled back into the prism. The
            # wide first pass shows the void class as a hollow candidate, so the
            # K+1 choice reads.
            if wide:
                specs = [(ACCENT, 38, 0.16, False), (SLATE, 19, 0.12, False),
                         (GOLD_DEEP, 0, 0.16, False), (SLATE, -19, 0.10, False),
                         (SLATE, -38, 0.0, True)]
            else:
                specs = [(ACCENT, 26, 0.15, False), (GOLD_DEEP, 3, 0.16, False),
                         (SLATE, -22, 0.11, False)]
            fan = VGroup(*[wedge(c, a, o, hollow=h) for c, a, o, h in specs])
            tint_hex = ManimColor(tint).to_hex()
            keep = next(k for k, s in enumerate(specs)
                        if not s[3] and ManimColor(s[0]).to_hex() == tint_hex)
            self.beat(LaggedStart(*[GrowFromPoint(f, prism_c) for f in fan],
                                  lag_ratio=0.1), rt=0.6 if wide else 0.4)
            losers = [f for k, f in enumerate(fan) if k != keep]
            # Losers retract into the prism; the winner brightens and is worn out.
            self.beat(*[l.animate.scale(0.12).move_to(prism_c).set_opacity(0)
                        for l in losers],
                      fan[keep].animate.set_fill(tint, opacity=0.5),
                      rt=0.55 if wide else 0.4)
            exit_p = fan[keep].get_center_of_mass()
            self.beat(orb.animate.scale(1 / 0.7).move_to(exit_p)
                      .set_stroke(tint).set_fill(tint),
                      glow.animate.set_fill(tint),
                      rt=0.7 if wide else 0.5)
            self.beat(orb.animate.move_to(slot), FadeOut(fan[keep]),
                      rt=0.55 if wide else 0.45)

        refract(orbs[0], glow_a, ACCENT, wide=True)
        refract(orbs[1], glow_b, ACCENT, wide=False)
        refract(orbs[2], glow_d, GOLD_DEEP, wide=False)

        # Classification is over for the matched slots; the rest drop to
        # neutral gray so the class tints are the only colors on the file.
        tail = (0.7, 0.45, 0.25)
        fades = [1.0 if i < 7 else tail[i - 7] for i in range(10)]
        self.beat(*[orbs[i].animate
                    .set_stroke(MUTED, opacity=0.6 * fades[i])
                    .set_fill(MUTED, opacity=0.15 * fades[i])
                    for i in range(3, 10)], rt=0.7)
        self.hold(0.3)

        # One leftover slot rides through the head so the fifth wedge gets
        # its owner: no object. The orb keeps the hollow outline; an empty
        # slot is a prediction too.
        o_void = orbs[3]
        void_slot = o_void.get_center()
        self.beat(o_void.animate.move_to(prism_c + LEFT * 0.85), rt=0.4)
        self.beat(o_void.animate.scale(0.7).move_to(prism_c), rt=0.3)
        void_specs = [(ACCENT, 24, 0.13, False), (GOLD_DEEP, 2, 0.13, False),
                      (SLATE, -22, 0.0, True)]
        void_fan = VGroup(*[wedge(c, a, o, hollow=h)
                            for c, a, o, h in void_specs])
        self.beat(LaggedStart(*[GrowFromPoint(f, prism_c) for f in void_fan],
                              lag_ratio=0.1), rt=0.4)
        self.beat(*[f.animate.scale(0.12).move_to(prism_c).set_opacity(0)
                    for f in void_fan[:2]],
                  void_fan[2].animate.set_stroke(HOLLOW, opacity=0.9),
                  rt=0.35)
        self.beat(o_void.animate.scale(1 / 0.7)
                  .move_to(void_fan[2].get_center_of_mass())
                  .set_stroke(HOLLOW, opacity=0.9).set_fill(opacity=0),
                  rt=0.4)
        self.beat(o_void.animate.move_to(void_slot), FadeOut(void_fan[2]),
                  rt=0.4)
        self.hold(0.5)

        # ---- shot 4 (0:22-0:35): one machine, three tasks -------------------
        # The soft fields settle into the objects as flat fills before the
        # panels multiply: the final panels hold only fills and outlines.
        self.beat(FadeOut(orbs), FadeOut(prism),
                  FadeOut(glow_a), FadeOut(glow_b), FadeOut(glow_d),
                  FadeOut(glow_x),
                  duck_a.animate.set_fill(ACCENT, 0.8),
                  duck_b.animate.set_fill(ACCENT, 0.8),
                  dog.animate.set_fill(GOLD_DEEP, 0.8), rt=0.9)

        panel = VGroup(field, duck_a, duck_b, dog)
        panel_mid = panel.copy()
        panel_right = panel.copy()
        self.add(panel_mid, panel_right)
        labels = VGroup(*[Text(s, font=FONT_BODY, font_size=28, color=MUTED)
                          .move_to([x, -1.55, 0])
                          for s, x in [("semantic", -4.55), ("instance", 0.0),
                                       ("panoptic", 4.55)]])
        self.beat(panel.animate.scale(0.38).move_to([-4.55, -0.4, 0]),
                  panel_mid.animate.scale(0.38).move_to([0, -0.4, 0]),
                  panel_right.animate.scale(0.38).move_to([4.55, -0.4, 0]),
                  FadeIn(labels),
                  rt=2.2)
        self.hold(0.5)

        # Same fields, regrouped in place. Left: every pixel labeled by
        # category, ducks share one hue (semantic). Middle: identities, stuff
        # dropped (instance). Right: both (panoptic).
        sem_field, sem_a, sem_b, sem_dog = (panel[0], panel[1], panel[2],
                                            panel[3])
        ins_field, ins_a, ins_b, ins_dog = (panel_mid[0], panel_mid[1],
                                            panel_mid[2], panel_mid[3])
        pan_field, pan_a, pan_b, pan_dog = (panel_right[0], panel_right[1],
                                            panel_right[2], panel_right[3])
        self.beat(
            # semantic: the two ducks merge into one hue, no identities
            sem_a.animate.set_fill(ACCENT), sem_b.animate.set_fill(ACCENT),
            sem_dog.animate.set_fill(GOLD_DEEP),
            # semantic labels stuff too: same background fill as panoptic
            sem_field.animate.set_fill(SLATE, opacity=0.24).set_stroke(SLATE, width=1.2, opacity=0.45),
            # instance: distinct things (thin ink identity outlines), dimmed
            # stuff
            ins_a.animate.set_fill(ACCENT_DARK)
            .set_stroke(INK, width=1.1, opacity=0.5),
            ins_b.animate.set_fill(ACCENT_LIGHT)
            .set_stroke(INK, width=1.1, opacity=0.5),
            ins_dog.animate.set_fill(GOLD_DEEP)
            .set_stroke(INK, width=1.1, opacity=0.5),
            ins_field.animate.set_fill(opacity=0.04).set_stroke(SLATE, width=1.2, opacity=0.45),
            # panoptic: identities AND labeled stuff
            pan_a.animate.set_fill(ACCENT_DARK)
            .set_stroke(INK, width=1.1, opacity=0.5),
            pan_b.animate.set_fill(ACCENT_LIGHT)
            .set_stroke(INK, width=1.1, opacity=0.5),
            pan_dog.animate.set_fill(GOLD_DEEP)
            .set_stroke(INK, width=1.1, opacity=0.5),
            pan_field.animate.set_fill(SLATE, opacity=0.24).set_stroke(SLATE, width=1.2, opacity=0.45),
            rt=3.4)
        self.hold(2.5)
        print(f"scene clock: {self.clock:.2f} s")
        self.hold(2.0)
