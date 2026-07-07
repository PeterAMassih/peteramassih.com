# patches/mask2former/viz/shapes.py
# Flat silhouettes shared by the scenes. Every variant of a shape keeps the
# same anchor count so a distorted prediction can Transform into its clean
# ground-truth counterpart without retessellation artifacts. Features are
# deliberately exaggerated (beak, ear, tail) because smoothing softens them.

import numpy as np
from manim import VMobject

# Floating duck facing left, traced clockwise from the beak tip.
DUCK = [
    (-1.05, 0.32), (-0.85, 0.40), (-0.72, 0.58), (-0.50, 0.62), (-0.36, 0.50),
    (-0.26, 0.32), (0.10, 0.38), (0.55, 0.46), (0.92, 0.66), (1.00, 0.52),
    (0.72, 0.14), (0.45, -0.18), (0.10, -0.34), (-0.30, -0.28), (-0.55, -0.08),
    (-0.70, 0.16), (-1.00, 0.24),
]

# Standing dog facing left, traced from a Mask2Former mask of a side-profile
# Ibizan Hound (Wikimedia, CC0) and simplified: long snout, pointed ears, four
# legs, raised tail. A real outline reads as a dog where the hand-drawn one did
# not. Smoothed through by silhouette(); the straight anchors are only control.
DOG = [
    (-1.05, 0.354), (-1.007, 0.293), (-0.871, 0.296), (-0.904, 0.193), (-0.728, 0.242),
    (-0.594, 0.093), (-0.466, -0.263), (-0.479, -0.688), (-0.223, -0.6), (-0.242, -0.111),
    (0.099, 0.041), (0.196, 0.008), (0.327, -0.387), (0.336, -0.533), (0.242, -0.685),
    (0.36, -0.67), (0.442, -0.482), (0.43, -0.175), (0.652, -0.384), (0.691, -0.615),
    (0.767, -0.609), (0.743, -0.348), (0.631, -0.193), (0.515, 0.196), (0.621, 0.242),
    (0.922, 0.16), (1.05, 0.299), (1.014, 0.43), (0.865, 0.263), (0.239, 0.339),
    (-0.394, 0.33), (-0.591, 0.473), (-0.567, 0.621), (-0.658, 0.688),
]


# Sitting cat facing left, clockwise from the nose: twin pointed ears with a
# dip between them (short and wide-set, or it reads rabbit), straight back,
# haunch, tail curled along the ground.
CAT = [
    (-1.00, 0.35), (-0.88, 0.52), (-0.82, 0.62), (-0.76, 0.90), (-0.58, 0.70),
    (-0.40, 0.92), (-0.30, 0.62), (-0.10, 0.46), (0.30, 0.32), (0.66, 0.20),
    (0.94, -0.02), (0.92, -0.48), (0.78, -0.80), (0.40, -0.94), (0.05, -0.92),
    (-0.34, -0.94), (-0.56, -0.88), (-0.62, -0.38), (-0.78, -0.02),
    (-0.94, 0.16),
]


def _pts(anchors, scale=1.0, mirror=False):
    sx = -scale if mirror else scale
    return [np.array([x * sx, y * scale, 0.0]) for x, y in anchors]


def warped(anchors, amount, seed):
    # A plausible imperfect prediction: a smooth low-frequency deformation of
    # the clean shape. Per-anchor iid jitter tears the outline into cusps;
    # sinusoidal displacement along the perimeter keeps it mask-like.
    rng = np.random.default_rng(seed)
    freqs = rng.integers(1, 4, 4)
    phases = rng.uniform(0.0, 2 * np.pi, 4)
    n = len(anchors)
    out = []
    for i, (x, y) in enumerate(anchors):
        t = 2 * np.pi * i / n
        dx = amount * (np.sin(freqs[0] * t + phases[0])
                       + 0.5 * np.sin(freqs[1] * t + phases[1]))
        dy = amount * (np.sin(freqs[2] * t + phases[2])
                       + 0.5 * np.sin(freqs[3] * t + phases[3]))
        out.append((x + dx, y + dy))
    return out


def silhouette(anchors, scale=1.0, color="#000000", fill_opacity=0.85,
               mirror=False):
    pts = _pts(anchors, scale, mirror)
    vm = VMobject()
    vm.set_points_smoothly([*pts, pts[0]])
    vm.set_stroke(color, width=1.6, opacity=0.95)
    vm.set_fill(color, opacity=fill_opacity)
    return vm


def blob(seed, scale=1.0, color="#000000", fill_opacity=0.85, n=14):
    # A garbage prediction: a lumpy radial closed curve, deterministic per
    # seed. Amplitudes stay small enough that the curve cannot self-cross.
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.04, 0.13, 3)
    phases = rng.uniform(0.0, 2 * np.pi, 3)
    pts = []
    for k in range(n):
        th = 2 * np.pi * k / n
        r = 1.0 + sum(a * np.sin((i + 2) * th + p)
                      for i, (a, p) in enumerate(zip(amps, phases)))
        pts.append(np.array([r * np.cos(th) * scale,
                             r * np.sin(th) * scale * 0.8, 0.0]))
    vm = VMobject()
    vm.set_points_smoothly([*pts, pts[0]])
    vm.set_stroke(color, width=1.6, opacity=0.95)
    vm.set_fill(color, opacity=fill_opacity)
    return vm
