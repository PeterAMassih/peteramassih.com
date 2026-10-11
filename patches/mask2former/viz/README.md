# Figures for the Mask2Former article

The animations and the one model-output image in
`src/content/writing/mask2former.md` are built here. The two remaining
diagrams (the overview, wide and stacked for phones, and Fig. 1) are inline SVG in the post and take their
colors from the `--viz-*` palette in `src/styles/tokens.css`.

| figure | script | scene | what it shows |
|---|---|---|---|
| 0 | `attention.py` | `AttentionLookup` | attention as a soft lookup on four locations of the dog photo |
| 2 | `fig_mask_classification.py` | (image) | the dog query's mask, thresholded and soft |
| 3 | `mask_dot.py` | `MaskIsDotProduct` | the real pixel embeddings, split by the dog query's mask embedding |
| 4 | `matching.py` | `GreedyVsBest` | cheapest-first versus the best assignment on a 3 by 3 cost matrix |
| 5 | `attention.py` | `MaskedAttention` | the same four locations under the real dog mask |
| 6 | `layers.py` | `LayerByLayer` | where the dog query reads in each of the nine decoder layers |
| 7 | `points.py` | `PointSampling` | the matching and training point rules on the real prediction |

Shared pieces: `theme.py` holds both site palettes and the frame size,
`data.py` runs the model once and caches what the scenes draw in
`sources/dog.npz`, and `encode.py` re-encodes each render for the web.
`data.py` asserts the facts the figures rely on: no attention falls outside
a layer's gate, and the projection in Fig. 3 reproduces the mask logits.

## Setup

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python manim fonttools torch torchvision transformers scipy pillow
mkdir -p fonts
curl -sL -o fonts/Geist-var.ttf "https://github.com/google/fonts/raw/main/ofl/geist/Geist%5Bwght%5D.ttf"
curl -sL -o fonts/JetBrainsMono-var.ttf "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf"
.venv/bin/python -m fontTools.varLib.instancer fonts/Geist-var.ttf wght=400 -o fonts/Geist-Regular.ttf
.venv/bin/python -m fontTools.varLib.instancer fonts/JetBrainsMono-var.ttf wght=400 -o fonts/JetBrainsMono-Regular.ttf
```

Pango mis-tracks variable fonts, so the scenes register the static instances.
Fonts, renders and the cache are not committed.

## Build

```sh
.venv/bin/python data.py                     # once: runs the model, writes sources/dog.npz
./render.sh                                  # every scene, light and dark, MP4 + poster
.venv/bin/python fig_mask_classification.py
```

The first model run downloads the checkpoint
(`facebook/mask2former-swin-large-coco-panoptic`, about 850 MB, cached by
Hugging Face) and the source photo into `sources/`. The photo is CC0 1.0 from
Wikimedia Commons, and its provenance is in `fig_mask_classification.py`. The model
runs in eval mode and the point sampler uses a fixed seed, so every figure
rebuilds identically.

`render.sh` renders each scene once per theme at 1600 by 1000 and writes
`<name>-light.mp4` and `<name>-dark.mp4` into `public/assets/m2f`, each with
two posters. `-start.webp` is the first frame, shown until the clip plays,
so every scene draws its starting state at frame 0 instead of fading it in.
`-end.webp` is the last frame, which holds the whole result and stands in for
the clip under reduced motion. `src/scripts/videos.js` picks the files that
match the reader's theme, plays each animation once when it reaches the middle
of the screen, and stops on its last frame. The frame is drawn for an 800px
figure that shrinks to about 360px on a phone, so no text is set below font
size 30.

Encoding goes through the FFmpeg that ships with PyAV, so no system ffmpeg is
needed. Each clip keeps a single keyframe: they are mostly still frames, and
every extra keyframe repaints the photo for about 60 KB.
