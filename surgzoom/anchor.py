"""Anchored sub-windows.

A question that names an absolute time t* in the source-video timeline is answered on
a family-specific sub-window of the clip instead of the whole clip. The sub-window is
cut from the time-overlay clip, so the burned-in clock travels with the pixels and the
model still reads absolute time. Training used exactly the same windows.

Family (matched in this order)     window (absolute seconds)
    retrieval     "When is it retrieved"            [t* - 10, clip end]
    reappear      "last visible just before ... re-appear"   [t* - 30, clip end]
    range         "seen between t1 and t2"          [t1 - 5, t2 + 5]
    cooccur       "also appear at"                  [min t - 10, max t + 10]
    moment        "at that moment"                  [t* - 10, t* + 10]
    insert        "What foreign object was inserted or created"   [t* - 15, t* + 15]
    positions     "At timepoint ... relative central positions"   [t* - 10, t* + 10]
    quadrant      "In which quadrant"               [clip start, t* + 10]
    frame_local   any other question naming a time  [t* - 10, t* + 10]

Windows shorter than 20 s are grown to 20 s; a window covering >= 95 % of the clip, or
a named time outside the clip, leaves the clip uncut (``None``).
"""
from __future__ import annotations

import re

MIN_WINDOW_S = 20.0
SKIP_COVER = 0.95

_TIME = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")

_FAMILIES = [
    ("retrieval", re.compile(r"\bWhen is it retrieved\b", re.I)),
    ("reappear", re.compile(r"last visible just before .{0,20}re-?appear", re.I)),
    ("range", re.compile(r"seen between \d{1,2}:\d{2}:\d{2} and \d{1,2}:\d{2}:\d{2}", re.I)),
    ("cooccur", re.compile(r"\balso appear at\b", re.I)),
    ("moment", re.compile(r"\bat that moment\b", re.I)),
    ("insert", re.compile(r"What foreign object was inserted or created", re.I)),
    ("positions", re.compile(r"At timepoint .{0,40}relative central positions", re.I)),
    ("quadrant", re.compile(r"\bIn which quadrant\b", re.I)),
    ("frame_local", None),
]

#: family -> (seconds kept before the anchor, seconds kept after); None = to the clip edge.
PRE_POST = {
    "retrieval": (10.0, None),
    "reappear": (30.0, None),
    "range": (5.0, 5.0),
    "cooccur": (10.0, 10.0),
    "moment": (10.0, 10.0),
    "insert": (15.0, 15.0),
    "positions": (10.0, 10.0),
    "quadrant": (None, 10.0),
    "frame_local": (10.0, 10.0),
}


def hms_to_seconds(text: str) -> int | None:
    h, m, s = text.split(":")
    m, s = int(m), int(s)
    if not (0 <= m < 60 and 0 <= s < 60):
        return None
    return int(h) * 3600 + m * 60 + s


def question_times(question: str) -> list[float]:
    """Absolute times (seconds) named in the question, in order of appearance."""
    out = []
    for x in _TIME.findall(question or ""):
        t = hms_to_seconds(x)
        if t is not None:
            out.append(float(t))
    return out


def anchor_family(question: str) -> str | None:
    """Family name if the question names an absolute time, else ``None``."""
    if not question_times(question):
        return None
    for name, rx in _FAMILIES:
        if rx is None or rx.search(question):
            return name
    return "frame_local"


def anchor_window(question: str, clip_start: float, clip_end: float) -> tuple[float, float] | None:
    """Sub-window ``(w0, w1)`` in absolute seconds, or ``None`` to keep the whole clip."""
    ts = question_times(question)
    if not ts:
        return None
    if any(t < clip_start - 0.5 or t > clip_end + 0.5 for t in ts):
        return None
    pre, post = PRE_POST[anchor_family(question)]
    lo = clip_start if pre is None else min(ts) - pre
    hi = clip_end if post is None else max(ts) + post
    w0, w1 = max(clip_start, lo), min(clip_end, hi)
    if w1 - w0 < MIN_WINDOW_S:                     # grow symmetrically, one-sided at an edge
        need = MIN_WINDOW_S - (w1 - w0)
        w0 = max(clip_start, w0 - need / 2)
        w1 = min(clip_end, w0 + MIN_WINDOW_S)
        w0 = max(clip_start, w1 - MIN_WINDOW_S)
    if w1 - w0 >= SKIP_COVER * (clip_end - clip_start):
        return None
    if w1 - w0 < 1.0:
        return None
    return (round(w0, 3), round(w1, 3))
