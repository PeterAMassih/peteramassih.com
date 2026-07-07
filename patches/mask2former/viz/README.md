# Manim scenes for the Mask2Former article

Five Manim CE scenes implementing the storyboards in `../storyboards.md`.
Colors and typography come from the site's own tokens (`tokens.py`, extracted
from `src/styles/tokens.css`).

## Setup

Requires Homebrew cairo, pango, and ffmpeg (with libvpx for VP9 webm).

```sh
brew install cairo pango ffmpeg
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python manim fonttools
mkdir -p fonts
curl -sL -o "fonts/Geist-var.ttf" \
  "https://github.com/google/fonts/raw/main/ofl/geist/Geist%5Bwght%5D.ttf"
.venv/bin/python -m fontTools.varLib.instancer fonts/Geist-var.ttf wght=400 \
  -o fonts/Geist-Regular.ttf
```

Geist (OFL) is registered with Pango at runtime by `tokens.py`; the fonts are
not committed. The static instance matters: Pango mis-tracks the variable TTF.

## Render

```sh
.venv/bin/manim -ql hungarian_matching.py HungarianMatching   # draft
.venv/bin/manim -qh hungarian_matching.py HungarianMatching   # final, 1080p60
```

Final assets are encoded to webm/mp4/gif into `public/assets/m2f/` by
`export.sh <scene_module> <SceneClass> <asset_name>`.
