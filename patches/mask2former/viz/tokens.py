# patches/mask2former/viz/tokens.py
# Design tokens for the Peter's Patches Manim scenes.
# Site values come from src/styles/tokens.css (light theme). The scenes must be
# indistinguishable from the page, so these are the page's own colors, not
# invented ones. Never inline a hex in a scene file; import from here.

BACKGROUND = "#fafafa"  # --color-bg
INK = "#171717"         # --color-text
MUTED = "#6b6b6b"       # --color-text-muted
ACCENT = "#0d9488"      # --color-accent

# Semantic cast (storyboards.md section 2). Meanings are fixed for the whole
# series; hues chosen to sit on the light background above.
GOLD = "#b8860b"    # predictions / queries: everything the model asserts
GREEN = "#2e7d32"   # ground truth: everything the data asserts
SLATE = "#64748b"   # image features, pixels, keys: the raw world
EMBER = "#c1440e"   # cost, strain, error; ember on screen means something is wrong
HOLLOW = "#a3a3a3"  # the void class, outline-only, never filled

FONT_BODY = "Geist"
BRAND = "peteramassih.com · Peter's Patches"

# The site's body face, registered with Pango so Text(font=FONT_BODY) resolves.
# The TTF is fetched by the setup steps in README.md, not committed.
import os as _os

import manimpango as _manimpango

_font_path = _os.path.join(_os.path.dirname(__file__), "fonts", "Geist-var.ttf")
_manimpango.register_font(_font_path)

# Thin strokes match the site's restraint: 2-3 px at 1080p.
STROKE = 2.5
STROKE_THIN = 1.4
