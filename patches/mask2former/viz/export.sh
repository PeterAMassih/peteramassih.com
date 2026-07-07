#!/bin/sh
# export.sh <scene_module.py> <SceneClass> <asset_name>
# Final render at 1080p60, then encode mp4 (h264), webm (vp9, target < 2 MB),
# and a palette-optimized gif fallback into public/assets/m2f/.
set -e
MODULE=$1
CLASS=$2
NAME=$3
.venv/bin/manim -qh --disable_caching "$MODULE" "$CLASS"
SRC="media/videos/${MODULE%.py}/1080p60/${CLASS}.mp4"
OUT="../../../public/assets/m2f"
mkdir -p "$OUT"
ffmpeg -y -i "$SRC" -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -an \
  "$OUT/$NAME.mp4"
ffmpeg -y -i "$SRC" -c:v libvpx-vp9 -crf 32 -b:v 0 -row-mt 1 -an \
  "$OUT/$NAME.webm"
ffmpeg -y -i "$SRC" -vf "fps=10,scale=720:-1:flags=lanczos,palettegen" \
  /tmp/m2f_palette.png
ffmpeg -y -i "$SRC" -i /tmp/m2f_palette.png \
  -filter_complex "fps=10,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "$OUT/$NAME.gif"
rm /tmp/m2f_palette.png
ls -la "$OUT"
