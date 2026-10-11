# patches/mask2former/viz/encode.py
"""Re-encode a Manim render for the web and save two posters: its first frame,
shown until the clip plays, and its last frame, which holds the whole result
and stands in for the clip under prefers-reduced-motion.

Uses the FFmpeg build bundled with PyAV (installed with Manim), so the
pipeline does not depend on a system ffmpeg.

Usage: encode.py <manim render.mp4> <output path without extension>
"""

import sys
from fractions import Fraction

import av


def main(src, dst):
    with av.open(src) as inp, av.open(f"{dst}.mp4", "w", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        rate = vin.average_rate or Fraction(30)
        vout = out.add_stream("libx264", rate=rate)
        vout.width, vout.height, vout.pix_fmt = vin.width, vin.height, "yuv420p"
        # One keyframe for the whole clip: these clips are mostly still frames,
        # and each extra keyframe repaints the photo for about 60 KB.
        vout.options = {"crf": "28", "preset": "veryslow", "tune": "animation", "sc_threshold": "0", "g": "9999"}
        for i, frame in enumerate(inp.decode(vin)):
            if i == 0:
                first = frame
            last = frame
            f = frame.reformat(format="yuv420p")
            f.pts, f.time_base = i, Fraction(1) / rate
            # Manim stitches one clip per animation, each opening on a keyframe.
            # Left set, those frame types would force the keyframes back in.
            f.pict_type = av.video.frame.PictureType.NONE
            out.mux(vout.encode(f))
        out.mux(vout.encode())
    first.to_image().save(f"{dst}-start.webp", quality=85, method=6)
    last.to_image().save(f"{dst}-end.webp", quality=85, method=6)


if __name__ == "__main__":
    main(*sys.argv[1:])
