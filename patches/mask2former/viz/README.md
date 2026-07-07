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

## Static figure: the model's own prediction (Fig. 2)

`fig_mask_classification.py` is not a Manim scene. It runs the real Mask2Former
model on one public-domain image and composes the section 2.2 figure
(`public/assets/m2f/mask_classification.{webp,png}`): the predicted dog mask in
gold, beside the same mask as a blocky 0/1 grid.

Extra dependencies, on top of the render setup above:

```sh
uv pip install --python .venv/bin/python torch torchvision transformers scipy
.venv/bin/python fig_mask_classification.py
```

The first run downloads the checkpoint (`facebook/mask2former-swin-large-coco-panoptic`,
~850 MB, cached by Hugging Face) and the source image into `sources/` (not
committed). Provenance and licensing are documented in the script header: the
image is CC0 1.0 from Wikimedia Commons, so the figure is redistributable. The
model runs in eval mode with no sampling, so the output is deterministic.
