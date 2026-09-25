"""ffmpeg helpers: clip cutting with the exact encode settings used in training."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Anchored sub-windows (training data and inference): CRF 23, closed GOP of 25 frames.
ENCODE_ANCHOR = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "25",
                 "-keyint_min", "25", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
                 "-an", "-movflags", "+faststart"]
# Zoom windows (inference only): CRF 18.
ENCODE_ZOOM = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "25",
               "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]


def find_ffmpeg() -> str:
    """``$SURGZOOM_FFMPEG`` > imageio-ffmpeg's bundled binary > ``ffmpeg`` on PATH."""
    env = os.environ.get("SURGZOOM_FFMPEG")
    if env:
        return env
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found: `pip install imageio-ffmpeg` or set SURGZOOM_FFMPEG")


def cut_window(src: str | Path, dst: str | Path, offset_s: float, duration_s: float,
               encode: list[str] = ENCODE_ZOOM, ffmpeg: str | None = None) -> bool:
    """Cut ``[offset_s, offset_s + duration_s]`` (seconds from the start of ``src``)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{offset_s:.3f}",
           "-i", str(src), "-t", f"{duration_s:.3f}", *encode, str(dst)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def find_font() -> str:
    """A TrueType font for the burned-in clock (``$SURGZOOM_FONT`` overrides)."""
    env = os.environ.get("SURGZOOM_FONT")
    if env:
        return env
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out and os.path.exists(out):
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if p.exists():
            return str(p)
    except ImportError:
        pass
    raise RuntimeError("no TrueType font found: install fonts-dejavu or set SURGZOOM_FONT")


def _clock_filter(start_s: float, font: str) -> str:
    # Absolute time of frame n at 5 fps = start + n / 5, floored to whole seconds.
    t = f"(n/5+{start_s})"
    return (f"drawtext=fontfile={font}:fontcolor=white:fontsize=40:x=20:y=20:"
            f"text='%{{eif\\:trunc({t}/3600)\\:d\\:2}}\\:"
            f"%{{eif\\:trunc(mod({t}/60,60))\\:d\\:2}}\\:"
            f"%{{eif\\:trunc(mod({t},60))\\:d\\:2}}'")


def make_overlay_clip(src_video: str | Path, dst: str | Path, start_s: float, end_s: float,
                      font: str | None = None, ffmpeg: str | None = None) -> bool:
    """Cut one question clip from a full procedure video in the challenge format.

    5 fps, height <= 576 px, H.264, no audio, keyframe every 5 s, and the absolute
    HH:MM:SS of the source video burned into the top-left corner of every frame.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    font = font or find_font()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    vf = f"fps=5,scale=-2:'min(ih,576)',{_clock_filter(start_s, font)}"
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start_s), "-t", str(end_s - start_s), "-i", str(src_video),
           "-vf", vf, *ENCODE_ANCHOR, str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dst)
    return True
