# patches/mask2former/viz/hungarian_matching.py
"""Storyboard A: hungarian_matching (~42 s).

The model outputs a set, so the loss must pair predictions with ground truths
by content, not order. Shot map and timings follow storyboards.md section 3.
Every cord survives the whole scene as one object: a straight line in shot 1,
a three-strand braid from shot 3 (the three cost terms: class, BCE, Dice).
Length is cost, ember tint is strain, and the side coil is the total cost as
one physical rope.
"""

import numpy as np
from manim import *  # noqa: F403 -- manim scenes conventionally star-import

from shapes import DOG, DUCK, blob, silhouette, warped
from tokens import (ACCENT, BACKGROUND, EMBER, FONT_BODY, GOLD, GREEN, HOLLOW,
                    MUTED, SLATE)

config.background_color = BACKGROUND

DISC_C = np.array([-1.2, -0.3, 0.0])
COIL_C = np.array([3.6, 1.6, 0.0])  # near the disc, so a swap visibly shortens it
SHELF_Y = -3.15
SHELF_X = (3.2, 6.4)

# Offsets in segment space from the disc center. Lookalikes are deliberately
# close: distance on the disc must read as dissimilarity. The 0.88 factor pulls
# the farthest garbage predictions in so their shapes clear the dashed rim
# (radius 2.75) instead of poking past it.
_OFFSET = {
    "gt1": [-1.55, 0.95],   # duck A
    "gt2": [1.65, 0.85],    # duck B
    "gt3": [0.15, -1.75],   # dog
    "p1": [0.75, 1.65],     # duck-B-like
    "p2": [1.15, -1.05],    # dog-like
    "p3": [-0.75, 1.75],    # duck-A-like, good
    "p4": [-2.55, -0.05],   # duck-A-like, worse
    "p5": [-2.30, -1.20],   # garbage
    "p6": [2.50, -0.90],    # garbage
}
POS = {k: DISC_C + 0.88 * np.array([x, y, 0.0]) for k, (x, y) in _OFFSET.items()}


def _bezier(a, c, b, t):
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t * t * b


class Cord:
    """A connector whose geometry is the argument. Its two ends track mobjects,
    so it follows every shuffle, condensation, and swap."""

    # Three distinguishable threads: the composite cost is class + BCE + Dice,
    # so the strands must not collapse into one rope. Muted teal keeps the third
    # legible without a garish new hue.
    STRAND_COLORS = (GOLD, SLATE,
                     interpolate_color(ManimColor(ACCENT), ManimColor(MUTED), 0.35))

    def __init__(self, start_mob, end_mob, calm=5.3, taut=6.3):
        self.start_mob = start_mob
        self.end_mob = end_mob
        self.calm = ValueTracker(calm)
        self.taut = ValueTracker(taut)
        self.braid = ValueTracker(0.0)
        self.sag = ValueTracker(0.05)
        self.dim = ValueTracker(1.0)
        self.mob = always_redraw(self._draw)

    def length(self):
        return float(np.linalg.norm(
            self.end_mob.get_center() - self.start_mob.get_center()))

    def _draw(self):
        a = self.start_mob.get_center()
        b = self.end_mob.get_center()
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            return VGroup()
        # sqrt bias keeps partial strain unambiguously warm: a strained cord
        # must never read as a correct pairing.
        strain = float(np.clip(
            (length - self.calm.get_value())
            / max(self.taut.get_value() - self.calm.get_value(), 1e-4),
            0, 1)) ** 0.5
        # Slack cords hang; taut cords straighten.
        mid = (a + b) / 2 + DOWN * self.sag.get_value() * (1 - 0.7 * strain)
        n = np.array([-d[1], d[0], 0.0]) / length
        braid = self.braid.get_value()
        amp = 0.085 * braid
        turns = max(2.0, length * 1.2)
        dim = self.dim.get_value()
        strands = VGroup()
        for i, base in enumerate(self.STRAND_COLORS):
            col = interpolate_color(
                ManimColor(MUTED), ManimColor(base), braid)
            # Strain is a tint, not a repaint: once the cord is a braid, the
            # strand identities (class, mask, dice) must stay readable.
            ember_mix = strain * (0.85 - 0.6 * braid)
            col = interpolate_color(col, ManimColor(EMBER), ember_mix)
            phase = i * TAU / 3

            def path(t, ph=phase):
                p = _bezier(a, mid, b, t)
                # The envelope pinches the strands together at the endpoints.
                env = np.sin(np.pi * t)
                off = amp * np.sin(TAU * turns * t + ph) * env
                return p + n * off

            visible = 1.0 if i == 0 else braid
            if visible < 0.02:
                continue
            width = 2.2 - 0.8 * braid if i == 0 else 1.5
            s = ParametricFunction(path, t_range=[0, 1, 1 / 72])
            s.set_stroke(col, width=width, opacity=0.9 * visible * dim)
            strands.add(s)
        # An end-nub marks where the cord grips its partner. It hides under
        # the mask fill at rest and surfaces exactly when an end is in
        # transit during a swap.
        nub = Dot(b, radius=0.05)
        nub.set_fill(interpolate_color(ManimColor(MUTED), ManimColor(EMBER),
                                       strain), opacity=0.9 * dim)
        strands.add(nub)
        return strands


class Coil:
    """The total assignment cost as one rope, coiled at the side. Arc length
    tracks the live sum of cord lengths, so every swap shortens it on screen."""

    def __init__(self, cords):
        self.cords = cords
        self.reveal = ValueTracker(0.0)
        self.mob = always_redraw(self._draw)

    def _draw(self):
        rv = self.reveal.get_value()
        if rv <= 0.001:
            return VGroup()
        L = sum(c.length() for c in self.cords) * 0.85 * rv
        # An Archimedean coil whose arc length equals the summed cord length:
        # turns appear and disappear as cost changes, so it reads as wound
        # rope, never as a gauge. Tight winding (small r0, b) keeps it reading
        # as a coil even when the cost is small, instead of a stray letter.
        r0, b = 0.16, 0.03
        theta_max = (-r0 + np.sqrt(r0 * r0 + 2 * b * L)) / b

        def spiral(t):
            th = theta_max * t
            r = r0 + b * th
            a = np.pi / 2 - th
            return COIL_C + np.array([r * np.cos(a), r * np.sin(a), 0.0])

        # Width 4: three 1.5px strands wound together, and the one deliberate
        # exception to the 2-3px stroke rule; the rope must read heavier than
        # any single cord because it is all of them poured together.
        rope = ParametricFunction(spiral, t_range=[0, 1, 1 / 160])
        rope.set_stroke(EMBER, width=4.0, opacity=0.85 * rv)
        # A short free tail at the outer end: a rope has an end, a ring
        # does not.
        end, near = spiral(1.0), spiral(0.985)
        d = end - near
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return rope
        tail = Line(end, end + d / n * 0.25)
        tail.set_stroke(EMBER, width=4.0, opacity=0.85 * rv)
        return VGroup(rope, tail)


def hollow_ring(r=0.26):
    # Fill color must be set even at opacity 0: Circle defaults to red, and
    # any later opacity animation would expose it.
    ring = Circle(radius=r).set_stroke(HOLLOW, width=2.0)
    ring.set_fill(HOLLOW, opacity=0)
    slash = Line(r * 0.85 * (DL * 0.9), r * 0.85 * (UR * 0.9))
    slash.set_stroke(HOLLOW, width=1.6)
    return VGroup(ring, slash)


class HungarianMatching(MovingCameraScene):
    def beat(self, *anims, rt=1.0, **kw):
        self.play(*anims, run_time=rt, **kw)
        self.clock += rt

    def hold(self, t):
        self.wait(t)
        self.clock += t

    def caption(self, text):
        # One quiet caption at a time, bottom-center, clear of the disc and the
        # shelf. Two in the whole scene: the problem, then its answer.
        return (Text(text, font=FONT_BODY, font_size=30, color=MUTED)
                .move_to([0, -3.5, 0]).set_z_index(5))

    def construct(self):
        self.clock = 0.0
        self.camera.background_color = BACKGROUND

        # ---- cast -------------------------------------------------------
        gts = {
            "gt1": silhouette(DUCK, 0.62, GREEN, fill_opacity=0.92),
            "gt2": silhouette(DUCK, 0.60, GREEN, fill_opacity=0.92,
                              mirror=True),
            "gt3": silhouette(DOG, 0.58, GREEN, fill_opacity=0.92),
        }
        preds = {
            "p1": silhouette(warped(DUCK, 0.12, 5), 0.55, GOLD,
                             fill_opacity=0.8, mirror=True),
            "p2": silhouette(warped(DOG, 0.12, 11), 0.52, GOLD,
                             fill_opacity=0.8),
            "p3": silhouette(warped(DUCK, 0.08, 3), 0.56, GOLD,
                             fill_opacity=0.8),
            "p4": silhouette(warped(DUCK, 0.18, 8), 0.54, GOLD,
                             fill_opacity=0.8),
            "p5": blob(21, 0.5, GOLD, fill_opacity=0.8),
            "p6": blob(47, 0.5, GOLD, fill_opacity=0.8),
        }
        # Corded predictions start directly above their ground truths, so the
        # naive index pairing reads calm and vertical until the shuffle.
        pred_order = ["p1", "p2", "p3", "p4", "p5", "p6"]
        slot_x = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        start_slot = {"p1": 1, "p2": 2, "p3": 3, "p4": 0, "p5": 4, "p6": 5}
        for name, s in start_slot.items():
            preds[name].move_to([slot_x[s], 2.6, 0]).set_z_index(2)
        for name, x in zip(["gt1", "gt2", "gt3"], [-3.0, -1.0, 1.0]):
            gts[name].move_to([x, -2.6, 0]).set_z_index(2)

        # A quiet image strip: where all of this comes from. Borderless pale
        # rectangle so it reads as a photo, never as a legend chip.
        strip_box = Rectangle(width=2.3, height=0.85)
        strip_box.set_stroke(opacity=0)
        strip_box.set_fill(SLATE, opacity=0.1)
        strip = VGroup(
            strip_box,
            silhouette(DUCK, 0.20, SLATE, fill_opacity=0.5).shift(LEFT * 0.72),
            silhouette(DUCK, 0.18, SLATE, fill_opacity=0.5, mirror=True),
            silhouette(DOG, 0.20, SLATE, fill_opacity=0.5).shift(RIGHT * 0.72),
        ).move_to([-5.85, 3.5, 0])

        cords = {k: Cord(preds[f"p{i}"], gts[f"gt{i}"])
                 for i, k in enumerate(["c1", "c2", "c3"], start=1)}

        # ---- shot 1 (0:00-0:07): the naive loss breaks -------------------
        self.beat(FadeIn(strip), rt=0.4)
        self.beat(LaggedStart(*[DrawBorderThenFill(preds[n])
                                for n in pred_order], lag_ratio=0.12), rt=1.1)
        self.beat(LaggedStart(*[DrawBorderThenFill(gts[n])
                                for n in ["gt1", "gt2", "gt3"]],
                              lag_ratio=0.2), rt=0.9)
        for c in cords.values():
            self.add(c.mob)
        self.hold(0.8)

        # A set has no order: storage positions shuffle, pairings do not.
        # This permutation strains all three cords past taut and crosses them.
        shuffle_to = {"p1": 5, "p2": 0, "p3": 1, "p4": 2, "p5": 3, "p6": 4}
        self.beat(*[preds[n].animate(path_arc=0.45)
                    .move_to([slot_x[s], 2.6, 0])
                    for n, s in shuffle_to.items()], rt=1.4)
        label_a = self.caption("order is arbitrary")
        self.beat(FadeIn(label_a), rt=0.5)
        self.hold(1.0)
        self.beat(FadeOut(label_a), FadeOut(strip), rt=0.8)

        # ---- shot 2 (0:07-0:16): masks become points ----------------------
        disc = DashedVMobject(Circle(radius=2.75).move_to(DISC_C),
                              num_dashes=72, dashed_ratio=0.45)
        disc.set_stroke(MUTED, width=1.4, opacity=0.5)
        self.beat(Create(disc),
                  *[c.calm.animate.set_value(1.3) for c in cords.values()],
                  *[c.taut.animate.set_value(3.2) for c in cords.values()],
                  rt=1.2)

        def condense(mask, target, height, rt):
            # The ghost keeps its fill and drains away: outline-only is the
            # void class's look and must not appear here.
            ghost = mask.copy()
            ghost.set_fill(mask.get_fill_color(), opacity=0.4)
            ghost.set_stroke(mask.get_stroke_color(), width=1.2, opacity=0.35)
            self.add(ghost)
            return AnimationGroup(
                mask.animate(path_arc=0.35).move_to(target)
                    .scale_to_fit_height(height),
                FadeOut(ghost, run_time=rt),
                run_time=rt)

        self.beat(condense(preds["p3"], POS["p3"], 0.30, 2.0), rt=2.0)
        rest = ["gt1", "p1", "gt2", "p2", "gt3", "p4", "p5", "p6"]
        anims = []
        for n in rest:
            mob = gts[n] if n.startswith("gt") else preds[n]
            h = 0.34 if n.startswith("gt") else 0.30
            anims.append(condense(mob, POS[n], h, 1.6))
        self.beat(LaggedStart(*anims, lag_ratio=0.24), rt=4.6)
        self.hold(1.2)

        # ---- shot 3 (0:16-0:22): cost is a braid --------------------------
        frame = self.camera.frame
        frame.save_state()
        focus_mid = (POS["p3"] + POS["gt3"]) / 2
        dim_static = [disc, *[preds[n] for n in pred_order if n != "p3"],
                      *[gts[n] for n in ["gt1", "gt2"]]]
        for m in dim_static:
            m.save_state()
        self.beat(frame.animate.set(width=6.8).move_to(focus_mid),
                  *[m.animate.set_opacity(0.2) for m in dim_static],
                  cords["c1"].dim.animate.set_value(0.2),
                  cords["c2"].dim.animate.set_value(0.2),
                  rt=1.4)
        self.beat(cords["c3"].braid.animate.set_value(1.0), rt=1.4)
        self.hold(1.4)
        self.beat(Restore(frame),
                  *[Restore(m) for m in dim_static],
                  cords["c1"].dim.animate.set_value(1.0),
                  cords["c2"].dim.animate.set_value(1.0),
                  cords["c1"].braid.animate.set_value(1.0),
                  cords["c2"].braid.animate.set_value(1.0),
                  rt=1.8)

        # ---- shot 4 (0:22-0:30): untangling -------------------------------
        coil = Coil(list(cords.values()))
        self.add(coil.mob)
        pours = VGroup(*[c.mob.copy().set_stroke(opacity=0.35)
                         for c in cords.values()])
        self.beat(coil.reveal.animate.set_value(1.0),
                  FadeOut(pours, target_position=COIL_C, scale=0.2),
                  rt=1.3)
        self.hold(0.2)

        def swap(cord_a, cord_b, rt, ang_a=-1.6, ang_b=-1.6):
            # Endpoints slide along the disc, never jumping: each traces an
            # arc to the other's ground truth.
            end_a = cord_a.end_mob
            end_b = cord_b.end_mob
            vp_a = VectorizedPoint(end_a.get_center())
            vp_b = VectorizedPoint(end_b.get_center())
            cord_a.end_mob = vp_a
            cord_b.end_mob = vp_b
            arc_a = ArcBetweenPoints(end_a.get_center(), end_b.get_center(),
                                     angle=ang_a)
            arc_b = ArcBetweenPoints(end_b.get_center(), end_a.get_center(),
                                     angle=ang_b)
            self.beat(MoveAlongPath(vp_a, arc_a), MoveAlongPath(vp_b, arc_b),
                      rt=rt)
            cord_a.end_mob = end_b
            cord_b.end_mob = end_a

        swap(cords["c1"], cords["c3"], rt=2.5)   # p1->gt3, p3->gt1
        self.hold(0.6)
        # The second slide bulges away from p2 so its cord stays visible.
        swap(cords["c1"], cords["c2"], rt=2.4,   # p1->gt2, p2->gt3
             ang_b=0.9)
        self.beat(*[c.sag.animate(rate_func=rate_functions.ease_out_back)
                    .set_value(0.16) for c in cords.values()], rt=1.0)

        # ---- shot 5 (0:30-0:36): the leftovers ----------------------------
        shelf = Line([SHELF_X[0], SHELF_Y, 0], [SHELF_X[1], SHELF_Y, 0])
        shelf.set_stroke(MUTED, width=1.6, opacity=0.35)
        shelf_anchor_x = [4.0, 4.9, 5.8]
        rings = [hollow_ring().move_to([x, SHELF_Y + 0.3, 0])
                 .set_stroke(opacity=0.5).set_z_index(3)
                 for x in shelf_anchor_x]
        self.beat(FadeIn(shelf), *[FadeIn(r) for r in rings], rt=0.8)

        leftovers = ["p4", "p5", "p6"]
        self.beat(*[rings[i].animate(path_arc=0.4)
                    .move_to(preds[n].get_center())
                    .scale(1.2).set_stroke(opacity=0.9)
                    for i, n in enumerate(leftovers)], rt=1.5)

        thread_dim = ValueTracker(0.0)
        threads = []
        for i, n in enumerate(leftovers):
            anchor = np.array([shelf_anchor_x[i], SHELF_Y, 0.0])

            def thread(n=n, anchor=anchor):
                # A slack whisper-thin thread, gently bowed, never a post.
                p = preds[n].get_center()
                mid = (anchor + p) / 2 + RIGHT * 0.07 + DOWN * 0.04
                t = VMobject()
                t.set_points_smoothly([anchor, mid, p])
                t.set_stroke(HOLLOW, width=0.9,
                             opacity=0.65 * thread_dim.get_value())
                return t

            threads.append(always_redraw(thread))
        for th in threads:
            self.add(th)
        # Uneven resting heights and offsets: hung things, not chart posts.
        rest = [(shelf_anchor_x[0] + 0.12, SHELF_Y + 0.52),
                (shelf_anchor_x[1] - 0.10, SHELF_Y + 0.68),
                (shelf_anchor_x[2] + 0.06, SHELF_Y + 0.58)]
        self.beat(*[AnimationGroup(
                        preds[n].animate(path_arc=0.45)
                        .move_to([rest[i][0], rest[i][1], 0]),
                        rings[i].animate(path_arc=0.45)
                        .move_to([rest[i][0], rest[i][1], 0]))
                    for i, n in enumerate(leftovers)],
                  thread_dim.animate.set_value(1.0), rt=2.2)
        # A faint tug, not a pull.
        self.beat(*[AnimationGroup(
                        preds[n].animate(rate_func=there_and_back)
                        .shift(DOWN * 0.06),
                        rings[i].animate(rate_func=there_and_back)
                        .shift(DOWN * 0.06))
                    for i, n in enumerate(leftovers)], rt=0.7)
        self.hold(0.8)

        # ---- shot 6 (0:36-0:42): the payoff -------------------------------
        # Everything in segment space is frozen; only the storage list moves.
        for c in cords.values():
            c.mob.suspend_updating()
        coil.mob.suspend_updating()
        for th in threads:
            th.suspend_updating()

        thumb_x = [-4.0, -2.4, -0.8, 0.8, 2.4, 4.0]
        # A faint holder behind the row names it as the storage list, so the
        # shuffle reads as reordering positions, not a second set of masks.
        row = RoundedRectangle(width=9.4, height=1.05, corner_radius=0.14)
        row.set_stroke(MUTED, width=1.2, opacity=0.35).set_fill(opacity=0)
        row.move_to([0, 3.3, 0])
        thumbs = {n: preds[n].copy().scale_to_fit_height(0.5)
                  .move_to([thumb_x[i], 3.3, 0]).set_z_index(3)
                  for i, n in enumerate(pred_order)}
        for t in thumbs.values():
            t.set_fill(GOLD, opacity=0.8).set_stroke(GOLD, width=1.4,
                                                     opacity=0.95)
        self.beat(FadeIn(row),
                  LaggedStart(*[DrawBorderThenFill(thumbs[n])
                                for n in pred_order], lag_ratio=0.1), rt=0.9)

        # One firm reordering, not a frantic sort: positions change, and the
        # matched pairs in segment space do not move at all.
        reorder = {"p1": 5, "p2": 2, "p3": 3, "p4": 0, "p5": 4, "p6": 1}
        self.beat(*[thumbs[n].animate(path_arc=0.3 * (-1) ** i)
                    .move_to([thumb_x[s], 3.3, 0])
                    for i, (n, s) in enumerate(reorder.items())], rt=1.1)
        label_b = self.caption("matching ignores order")
        self.beat(FadeIn(label_b), rt=0.5)
        # The held frame the scene exists for: order just changed, nothing in
        # segment space did.
        self.hold(2.4)
        self.beat(FadeOut(label_b), FadeOut(row),
                  FadeOut(VGroup(*thumbs.values())), rt=0.7)

        # ---- close (0:42-0:44): rest on the solved matching ---------------
        # A shot 7 once rode gradients down the cords here. Gradients are the
        # training story of the loss terms, not the matching, so the scene ends
        # on the resolved assignment and lets that single idea land.
        self.hold(1.5)
        print(f"scene clock: {self.clock:.2f} s")
