"""Self-routed temporal zoom and answer normalisation."""
from __future__ import annotations

import re

ZOOM_HALF_1 = 15.0      # pass 2: +-15 s around the first answer
ZOOM_HALF_2 = 7.5       # pass 3: +-7.5 s around the second answer
ZOOM_MIN_CLIP_S = 60.0  # shorter clips are already sampled densely
ANCHOR_SLACK_S = 30.0   # answers this far outside the clip are durations, not times

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_HMS = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")


def clean(text: str | None) -> str:
    """Strip residual reasoning tags."""
    return _THINK.sub("", text or "").strip()


def _seconds(hms: str) -> int | None:
    h, m, s = hms.split(":")
    m, s = int(m), int(s)
    if not (0 <= m < 60 and 0 <= s < 60):
        return None
    return int(h) * 3600 + m * 60 + s


def first_time(text: str | None) -> int | None:
    """First valid HH:MM:SS in ``text``, in seconds."""
    m = _HMS.search(clean(text))
    return _seconds(m.group(1)) if m else None


def zoom_anchor(answer: str, clip_start: float, clip_end: float) -> float | None:
    """Where to zoom, decided from the first answer alone; ``None`` = do not zoom.

    A question is refined iff its answer contains a valid timestamp inside the clip
    (or within 30 s of it) and the clip is at least 60 s long. Duration-style answers
    such as "00:00:22" fall far outside the window and are excluded automatically.
    """
    if clip_end - clip_start < ZOOM_MIN_CLIP_S:
        return None
    t = first_time(answer)
    if t is None or t < clip_start - ANCHOR_SLACK_S or t > clip_end + ANCHOR_SLACK_S:
        return None
    return min(max(float(t), clip_start), clip_end)


def window_around(t: float, half: float, clip_start: float, clip_end: float) -> tuple[float, float]:
    """``[t - half, t + half]`` clamped to the clip, shifted inwards at an edge."""
    dur = clip_end - clip_start
    z0, z1 = max(clip_start, t - half), min(clip_end, t + half)
    want = min(2 * half, dur)
    if z1 - z0 < want:
        if z0 <= clip_start:
            z1 = clip_start + want
        else:
            z0 = clip_end - want
    return z0, z1


def normalize_answer(answer: str) -> str:
    """A timestamp-only answer (e.g. "00:31:33, 00:32:17") becomes its first HH:MM:SS.

    Free-text answers that merely mention a time are left unchanged.
    """
    a = clean(answer)
    if not _HMS.search(a) or re.fullmatch(r"\d{2}:\d{2}:\d{2}", a):
        return a
    rest = re.sub(r"(?i)\band\b", "", _HMS.sub("", a)).strip(" \t\n,;.&-")
    if rest:
        return a
    return _HMS.search(a).group(1)
