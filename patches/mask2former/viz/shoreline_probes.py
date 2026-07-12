# patches/mask2former/viz/shoreline_probes.py
"""Visual explanation of point-sampled mask losses.

The balance represents uniform sampling for an additive point loss. Matching
reuses one uniform point set across all mask pairs. Training selects points
whose prediction logits are uncertain, which the final shot represents with
points near the predicted contour. The sampler does not inspect ground truth.
"""

import numpy as np
from manim import *  # noqa: F403 -- manim scenes conventionally star-import

from shapes import DUCK, silhouette, warped
from tokens import BACKGROUND, EMBER, FONT_BODY, GOLD, GREEN, INK, MUTED, SLATE

config.background_color = BACKGROUND

SHEET_C = np.array([-2.6, 0.6, 0.0])
PIVOT = np.array([4.1, -1.0, 0.0])
ARM = 1.8


def polygon_of(vm, n=140):
    return np.array([vm.point_from_proportion(t)
                     for t in np.linspace(0, 1, n, endpoint=False)])


def inside(p, poly):
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


def sheet_pair(center, scale, seed, offset):
    # A hard-mask analogy. The symmetric difference shows where binary masks
    # disagree. BCE and soft Dice also receive signal away from this band.
    truth = silhouette(DUCK, scale, GREEN, fill_opacity=0.35)
    truth.set_stroke(GREEN, width=1.4, opacity=0.7).move_to(center)
    pred = silhouette(warped(DUCK, 0.04, seed), scale * 0.99, GOLD,
                      fill_opacity=0.35)
    pred.set_stroke(GOLD, width=1.4, opacity=0.7)
    pred.move_to(center + np.array([*offset, 0.0]))
    shore = Union(Difference(truth.copy(), pred.copy()),
                  Difference(pred.copy(), truth.copy()))
    shore.set_fill(EMBER, opacity=0.8)
    shore.set_stroke(opacity=0)
    return truth, pred, shore


class ShorelineProbes(MovingCameraScene):
    def beat(self, *anims, rt=1.0, **kw):
        self.play(*anims, run_time=rt, **kw)
        self.clock += rt

    def hold(self, t):
        self.wait(t)
        self.clock += t

    def construct(self):
        self.clock = 0.0
        self.camera.background_color = BACKGROUND

        # ---- shot 1 (0:00-0:06): a hard-mask shoreline ----------------------
        truth, pred, shore = sheet_pair(SHEET_C, 1.5, 21, (0.1, 0.07))
        truth.shift(LEFT * 7)
        pred.shift(RIGHT * 7)
        self.add(truth, pred)
        self.beat(truth.animate.shift(RIGHT * 7),
                  pred.animate.shift(LEFT * 7), rt=1.8)
        # Let the overlap sit: green truth, gold prediction, agreement where
        # they stack. The viewer needs this state before the band names it.
        self.hold(2.0)
        shore.set_fill(opacity=0).set_stroke(opacity=0)
        self.add(shore)
        # In the binary analogy, agreement is zero error and disagreement glows.
        self.beat(shore.animate.set_fill(EMBER, opacity=0.8), rt=1.2)

        frame = self.camera.frame
        frame.save_state()
        # Close-up on the shoreline: center on the sheet so the whole band
        # stays in frame, and rest at the apex long enough to read it.
        self.beat(frame.animate.set(width=5.2).move_to(SHEET_C), rt=1.6)
        self.hold(2.0)
        self.beat(Restore(frame), rt=1.2)

        # ---- shot 2 (0:06-0:12): weighing everything ------------------------
        theta = ValueTracker(0.0)

        def arm_end(sign):
            a = theta.get_value()
            return PIVOT + sign * ARM * np.array([np.cos(a), np.sin(a), 0.0])

        def pan_center(sign):
            return arm_end(sign) + DOWN * 0.62

        def beam_draw():
            left, right = arm_end(-1), arm_end(1)
            parts = [Line(left, right).set_stroke(INK, width=3.0,
                                                  opacity=0.85)]
            for end in (left, right):
                pc = end + DOWN * 0.62
                parts.append(Line(end, pc + LEFT * 0.34)
                             .set_stroke(MUTED, width=1.1, opacity=0.7))
                parts.append(Line(end, pc + RIGHT * 0.34)
                             .set_stroke(MUTED, width=1.1, opacity=0.7))
                plate = ArcBetweenPoints(pc + LEFT * 0.42, pc + RIGHT * 0.42,
                                         angle=0.9)
                parts.append(plate.set_stroke(INK, width=2.2, opacity=0.85))
            return VGroup(*parts)

        beam = always_redraw(beam_draw)
        fulcrum = Triangle().scale(0.3).move_to(PIVOT + DOWN * 0.16)
        fulcrum.set_stroke(INK, width=2.0, opacity=0.85)
        fulcrum.set_fill(SLATE, opacity=0.15)

        # Keep the probe geometry before the map leaves.
        poly_truth = polygon_of(truth, n=260)
        poly_pred = polygon_of(pred, n=260)

        def on_shore(p):
            # A needle sparks if any part of its tip touches disagreement;
            # a cold needle standing in the ember band would break the
            # mechanism at freeze-frame.
            for q in (p, p + RIGHT * 0.05, p + LEFT * 0.05,
                      p + UP * 0.05, p + DOWN * 0.05):
                if inside(q, poly_truth) != inside(q, poly_pred):
                    return True
            return False

        the_map = VGroup(truth, pred, shore)
        # The beam appears as the map is already settling onto the left pan, so
        # there is no empty, level balance to freeze on: the first beam a viewer
        # sees is one that is being loaded.
        self.beat(FadeIn(fulcrum), FadeIn(beam),
                  the_map.animate.scale(0.2)
                  .move_to(pan_center(-1) + UP * 0.24), rt=2.0)
        the_map.add_updater(lambda m: m.move_to(pan_center(-1) + UP * 0.24))
        # The dense hard-mask error is exact and heavy. Positive angle drops
        # the left arm, so the loaded pan sinks.
        self.beat(theta.animate.set_value(0.14), rt=1.8)

        dup_truth, dup_pred, dup_shore = sheet_pair(SHEET_C, 1.5, 21,
                                                    (0.1, 0.07))
        dup = VGroup(dup_truth, dup_pred, dup_shore)
        self.beat(FadeIn(dup), rt=1.2)
        self.hold(0.4)

        # ---- shot 3 (0:12-0:20): the probes ---------------------------------
        rng = np.random.default_rng(4)
        needles = []
        hot_flags = []
        for i in range(15):
            for j in range(9):
                c = np.array([SHEET_C[0] - 2.15 + 0.307 * i,
                              SHEET_C[1] - 1.4 + 0.35 * j, 0.0])
                p = c + np.array([rng.uniform(-0.11, 0.11),
                                  rng.uniform(-0.11, 0.11), 0.0])
                hot = on_shore(p)
                line = Line(p + UP * 0.09, p + DOWN * 0.09)
                line.set_stroke(INK, width=1.4, opacity=0.65 if hot else 0.35)
                needle = VGroup(line)
                if hot:
                    tip = Dot(p, radius=0.045).set_fill(EMBER, opacity=0.95)
                    tip.set_stroke(opacity=0)
                    needle.add(tip)
                needles.append(needle)
                hot_flags.append(hot)
        rain = VGroup(*needles)
        rain.shift(UP * 3.2).set_opacity(0.0)
        self.add(rain)
        self.beat(LaggedStart(*[AnimationGroup(
            n.animate.shift(DOWN * 3.2).set_opacity(1.0))
            for n in needles], lag_ratio=0.02), rt=2.2)
        self.hold(0.6)

        # The sparks alone are swept into the right pan as a compact heap,
        # seated low so it rests in the cup instead of floating above the rim.
        sparks = VGroup(*[n[1].copy() for n, h in zip(needles, hot_flags)
                          if h])
        self.add(sparks)
        pile_rng = np.random.default_rng(11)
        heap = [np.array([0.15 * (k % 5 - 2) + pile_rng.uniform(-0.03, 0.03),
                          0.06 + 0.11 * (k // 5) + pile_rng.uniform(-0.02, 0.02),
                          0.0])
                for k in range(len(sparks))]
        self.beat(LaggedStart(*[s.animate.move_to(pan_center(1) + off)
                                for s, off in zip(sparks, heap)],
                              lag_ratio=0.03), rt=1.2)
        # Each spark now rides the pan at its own offset, so the heap stays in
        # the cup and travels down with it.
        for s, off in zip(sparks, heap):
            s.add_updater(
                lambda m, off=off, pc=pan_center: m.move_to(pc(1) + off))
        expectation_note = Text("equal in expectation", font=FONT_BODY,
                                font_size=24, color=MUTED)
        expectation_note.move_to(PIVOT + DOWN * 1.55)
        # The level beam represents equality in expectation, not equality for
        # the single finite sample drawn on screen.
        self.beat(theta.animate.set_value(0.0), FadeIn(expectation_note), rt=1.6,
                  rate_func=rate_functions.ease_in_out_sine)
        self.hold(1.0)
        the_map.suspend_updating()
        for s in sparks:
            s.suspend_updating()
        beam.suspend_updating()

        # ---- shot 4 (0:20-0:26): two rules for two jobs ---------------------
        self.beat(dup.animate.set_opacity(0.15), FadeOut(expectation_note),
                  rain.animate.set_opacity(0.15),
                  rt=1.0)

        pairs = VGroup()
        for k, (seed, off) in enumerate([(31, (0.08, 0.06)),
                                         (41, (-0.07, 0.06)),
                                         (51, (0.07, -0.07))]):
            t, p, s = sheet_pair([-5.4, 2.1 - 1.9 * k, 0], 0.62, seed, off)
            pairs.add(VGroup(t, p, s))
        train_t, train_p, train_s = sheet_pair([2.6, 1.7, 0], 0.95, 61,
                                               (0.1, 0.07))
        train = VGroup(train_t, train_p, train_s)
        self.beat(FadeIn(pairs, lag_ratio=0.15), FadeIn(train), rt=1.0)

        # Matching: one die, stamped identically on every pair.
        stamp_rng = np.random.default_rng(99)
        offs = [np.array([stamp_rng.uniform(-0.85, 0.85),
                          stamp_rng.uniform(-0.55, 0.55), 0.0])
                for _ in range(14)]
        stamps = []
        for k in range(3):
            c = np.array([-5.4, 2.1 - 1.9 * k, 0.0])
            dots = VGroup(*[Dot(c + o, radius=0.035)
                            .set_fill(INK, opacity=0.75).set_stroke(opacity=0)
                            for o in offs])
            stamps.append(dots)

        # Training draws 3K uniform candidates over the full image, keeps .75K
        # near the prediction's zero-logit contour, and adds .25K fresh uniform
        # points. Candidate selection never reads the target.
        train_rng = np.random.default_rng(17)
        candidate_dots = VGroup()
        candidate_positions = []
        for i in range(12):
            for j in range(5):
                p = np.array([1.15 + 0.265 * i,
                              0.85 + 0.42 * j, 0.0])
                p += np.array([train_rng.uniform(-0.08, 0.08),
                               train_rng.uniform(-0.08, 0.08), 0.0])
                d = Dot(p, radius=0.04).set_fill(INK, opacity=0.7)
                d.set_stroke(opacity=0)
                candidate_positions.append(p)
                candidate_dots.add(d)
        candidate_dots.shift(UP * 2.2).set_opacity(0)

        self.add(candidate_dots)
        self.beat(LaggedStart(*[FadeIn(s, scale=1.25) for s in stamps],
                              lag_ratio=0.45),
                  LaggedStart(*[AnimationGroup(
                      d.animate.shift(DOWN * 2.2).set_opacity(0.7))
                      for d in candidate_dots], lag_ratio=0.01), rt=1.5)

        uncertain_curve = polygon_of(train_p, n=180)
        distances = [min(np.linalg.norm(p - q) for q in uncertain_curve)
                     for p in candidate_positions]
        selected_ids = set(np.argsort(distances)[:15])  # .75K of a 20-point final set
        selected = VGroup(*[candidate_dots[i] for i in sorted(selected_ids)])
        rejected = VGroup(*[candidate_dots[i] for i in range(60)
                            if i not in selected_ids])
        self.beat(rejected.animate.set_opacity(0.03),
                  *[d.animate.set_fill(EMBER, opacity=0.9)
                    .set_stroke(INK, width=1.2, opacity=0.9)
                    for d in selected], rt=1.0)

        fresh_rng = np.random.default_rng(23)
        fresh = VGroup(*[
            Dot([fresh_rng.uniform(1.15, 4.07),
                 fresh_rng.uniform(0.85, 2.53), 0.0], radius=0.04)
            .set_fill(INK, opacity=0.9)
            .set_stroke(INK, width=1.2, opacity=0.9)
            for _ in range(5)  # .25K fresh uniform points
        ])
        self.beat(FadeIn(fresh, lag_ratio=0.12), rt=0.6)
        self.hold(1.4)

        # ---- shot 5 (0:26-0:32): residue ------------------------------------
        # The rained needles arrange themselves into a fine lattice, half a
        # beat, then relax back. The only number-shaped thing in the scene.
        rest_pos = [n.get_center() for n in needles]
        lattice = []
        for i in range(15):
            for j in range(9):
                c = np.array([SHEET_C[0] - 2.15 + 0.307 * i,
                              SHEET_C[1] - 1.4 + 0.35 * j, 0.0])
                lattice.append(np.array([SHEET_C[0] - 2.15 + 0.307 * i,
                                         SHEET_C[1] - 1.32 + 0.33 * j, 0.0]))
        # The lattice is the subject of this beat: shot 4's leftovers recede.
        self.beat(*[n.animate.move_to(q).set_opacity(0.6)
                    for n, q in zip(needles, lattice)],
                  pairs.animate.set_opacity(0.15),
                  train.animate.set_opacity(0.15),
                  selected.animate.set_opacity(0.12),
                  fresh.animate.set_opacity(0.12),
                  *[s.animate(rate_func=smooth).set_opacity(0.15)
                    for s in stamps], rt=1.6)
        self.hold(0.7)
        self.beat(*[n.animate.move_to(q) for n, q in zip(needles, rest_pos)],
                  rain.animate.set_opacity(0.45), rt=1.4)
        # Final frame: leveled beam, glowing shoreline, resting needles.
        # Restore each sheet's own translucency; a blanket set_opacity would
        # crush the layered fills into solid paint.
        self.beat(dup_truth.animate.set_fill(opacity=0.4)
                  .set_stroke(opacity=0.7),
                  dup_pred.animate.set_fill(opacity=0.4)
                  .set_stroke(opacity=0.7),
                  dup_shore.animate.set_fill(opacity=0.5)
                  .set_stroke(opacity=0.7),
                  pairs.animate.set_opacity(0.35),
                  train.animate.set_opacity(0.35),
                  selected.animate.set_opacity(0.3),
                  fresh.animate.set_opacity(0.3), rt=1.0)
        self.hold(1.3)
        print(f"scene clock: {self.clock:.2f} s")
        self.hold(2.0)
