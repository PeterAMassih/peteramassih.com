#!/bin/sh
# frames.sh <video> <name> <t1> <t2> ...
# Export one PNG per storyboard timestamp for freeze-frame review.
set -e
VIDEO=$1
NAME=$2
shift 2
mkdir -p frames
for T in "$@"; do
  ffmpeg -y -ss "$T" -i "$VIDEO" -frames:v 1 "frames/${NAME}_${T}s.png" \
    -loglevel error
done
ls frames/
