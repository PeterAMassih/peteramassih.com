#!/bin/sh
# patches/mask2former/viz/render.sh
# Renders every scene once per site theme and writes, for each, an MP4 and two
# posters (its first and last frames) to public/assets/m2f. The post picks the
# files that match the reader's theme.
set -e
cd "$(dirname "$0")"
OUT=../../../public/assets/m2f

render() { # module, scene, published name
  for theme in light dark; do
    M2F_THEME=$theme .venv/bin/manim --disable_caching -r 1600,1000 --fps 30 --media_dir media \
      -o "$3-$theme" "$1" "$2"
    src="media/videos/${1%.py}/1000p30/$3-$theme.mp4"
    .venv/bin/python encode.py "$src" "$OUT/$3-$theme"
  done
}

render attention.py AttentionLookup attention
render attention.py MaskedAttention masked-attention
render layers.py LayerByLayer layers
render matching.py GreedyVsBest matching
render mask_dot.py MaskIsDotProduct mask-dot
render points.py PointSampling point-sampling
