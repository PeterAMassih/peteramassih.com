# Figures for the Mask2Former article

Two figures in `src/content/writing/mask2former.md` show the real model's
output, so they are built here rather than drawn by hand. The other figures
are inline SVG in the post and take their colors from the `--viz-*` palette in
`src/styles/tokens.css`.

- `fig_mask_classification.py` builds `public/assets/m2f/mask_classification.webp`
  (Fig. 2): the dog query's thresholded mask over the photo, beside its soft
  mask with the 0.5 level drawn in black.
- `fig_point_sampling.py` builds `public/assets/m2f/point_sampling.webp`
  (Fig. 6): uniform matching points beside the uncertainty-selected training
  points, on the same query.

## Setup and run

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch torchvision transformers scipy pillow
.venv/bin/python fig_mask_classification.py
.venv/bin/python fig_point_sampling.py
```

The first run downloads the checkpoint (`facebook/mask2former-swin-large-coco-panoptic`,
about 850 MB, cached by Hugging Face) and the source photo into `sources/`
(not committed). The photo is CC0 1.0 from Wikimedia Commons; provenance is in
the script header. The model runs in eval mode and the point sampler uses a
fixed seed, so both figures rebuild identically.
