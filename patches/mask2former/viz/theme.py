# patches/mask2former/viz/theme.py
"""Shared look for the article's Manim scenes.

Each scene renders twice, once per site theme, so a video sits on the page
like the inline SVG figures do. render.sh picks the theme through the
M2F_THEME environment variable. The colors are the --color-* and --viz-*
values in src/styles/tokens.css, one meaning per hue: gold is what the model
predicts, green the ground truth, slate image features, ember cost or error.

The frame is 16:10 and designed for an 800px figure that shrinks to about
360px on a phone. Text below font size 30 is unreadable at that width, so no
scene uses smaller type.
"""

import os
from pathlib import Path

import manimpango
from manim import config

PALETTES = {
    "light": dict(bg="#fafafa", text="#171717", muted="#6b6b6b", gold="#b8860b", gold_text="#8a6508",
                  green="#2e7d32", slate="#64748b", ember="#c1440e", line="#9a9a9a", grid="#d4d4d4",
                  box="#ffffff"),
    "dark": dict(bg="#0a0a0a", text="#fafafa", muted="#a3a3a3", gold="#d4a53a", gold_text="#d4a53a",
                 green="#5fae63", slate="#94a3b8", ember="#e8794a", line="#737373", grid="#333333",
                 box="#171717"),
}
THEME = os.environ.get("M2F_THEME", "light")
P = PALETTES[THEME]

# Words in Geist, numbers and formulas in JetBrains Mono, as on the site. Mono
# digits keep columns aligned and stop "1.1" from reading as "11" on a phone.
# Pango mis-tracks variable fonts, so the static instances are registered.
FONT, MONO = "Geist", "JetBrains Mono"
for name in ("Geist-Regular.ttf", "JetBrainsMono-Regular.ttf"):
    manimpango.register_font(str(Path(__file__).parent / "fonts" / name))

config.pixel_width, config.pixel_height = 1600, 1000
config.frame_height, config.frame_width = 8.0, 12.8
config.frame_rate = 30
config.background_color = P["bg"]
