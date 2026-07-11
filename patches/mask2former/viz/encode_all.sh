#!/bin/sh
# encode_all.sh — encode the four Phase 2 scenes from their 1080p60 masters
# into public/assets/m2f (mp4 h264, webm vp9 under 2 MB, palette gif) and
# refresh the contract freeze-frames used by frames/review.html.
set -e
OUT=../../../public/assets/m2f

enc() {
  SRC=media/videos/$1/1080p60/$2.mp4
  CRF=$3
  ffmpeg -y -i "$SRC" -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -an \
    "$OUT/$1.mp4" -loglevel error
  ffmpeg -y -i "$SRC" -c:v libvpx-vp9 -crf "$CRF" -b:v 0 -row-mt 1 -an \
    "$OUT/$1.webm" -loglevel error
  ffmpeg -y -i "$SRC" -vf "fps=10,scale=720:-1:flags=lanczos,palettegen" \
    /tmp/pal_$1.png -loglevel error
  ffmpeg -y -i "$SRC" -i /tmp/pal_$1.png \
    -filter_complex "fps=10,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    "$OUT/$1.gif" -loglevel error
  rm /tmp/pal_$1.png
}

enc masked_attention MaskedAttention 42
enc query_becomes_segment QueryBecomesSegment 36
enc scales_breathe ScalesBreathe 36
enc shoreline_probes ShorelineProbes 36

./frames.sh media/videos/masked_attention/1080p60/MaskedAttention.mp4 \
  ship_ma 5 17 24 29 40 > /dev/null
./frames.sh media/videos/query_becomes_segment/1080p60/QueryBecomesSegment.mp4 \
  ship_qs 4 12 19.6 30 > /dev/null
./frames.sh media/videos/scales_breathe/1080p60/ScalesBreathe.mp4 \
  ship_sb 3 15 25 29.5 > /dev/null
./frames.sh media/videos/shoreline_probes/1080p60/ShorelineProbes.mp4 \
  ship_sp 5 11 18 23.4 28 31 > /dev/null
ls -la "$OUT"
