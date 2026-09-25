"""Question-conditioned frame budgeting.

Every question receives the same visual-token budget. A rule-based router reads the
question text alone and decides how that budget is spent:

    time      the wording asks for a timestamp ("hh:mm:ss")        -> 256 frames
    count     counting / aggregation wording, no absolute time      -> 256 frames
    other     everything else                                       -> 128 frames

The image processor then divides the budget over the sampled frames, so dense inputs
are encoded at a lower per-frame resolution (512x288 on a 119 s clip) and sparse inputs
at a higher one (736x416). The same router generated the training data, drives the
offline evaluation and runs in the challenge container.
"""
from __future__ import annotations

import re

#: Visual tokens per question (all frames together).
BUDGET_TOKENS = 20480

#: Frames sampled per question, by router branch.
FRAMES = {"time": 256, "count": 256, "other": 128}

# Timestamp questions always carry the format instruction "in the format hh:mm:ss".
# On all 20,000 train+test questions this rule agrees with answer_format == "time"
# in both directions. The loose alternatives only matter for differently worded,
# out-of-distribution questions; a mis-route just picks the other trained regime.
_TIME_STRICT = re.compile(r"hh:mm:ss", re.I)
_TIME_LOOSE = re.compile(r"hh:mm:ss|timestamp|at what time|at which time|what time", re.I)

_COUNT = re.compile(r"how many|maximum number of|total count", re.I)
# Aggregation over object classes: "After the <X> was inserted/created, which other
# foreign object classes ...". ("With which other classes does X co-occur" is not.)
_COUNT_AGG_CLASSES = re.compile(
    r"^after the .{0,80}(?:inserted|created).*which (?:other|different) foreign object classes",
    re.I | re.S,
)
_PERCENTAGE = re.compile(r"\bin %", re.I)             # "In %, how many of the frames ..."
_ABS_TIME = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")   # "how many ... in frame <t>" is single-frame


def is_time_question(question: str, loose: bool = True) -> bool:
    return bool((_TIME_LOOSE if loose else _TIME_STRICT).search(question or ""))


def is_count_question(question: str) -> bool:
    q = question or ""
    if _ABS_TIME.search(q) or _PERCENTAGE.search(q):
        return False
    return bool(_COUNT.search(q)) or bool(_COUNT_AGG_CLASSES.search(q))


def route(question: str, loose: bool = True) -> str:
    """Return the router branch: ``"time"``, ``"count"`` or ``"other"``.

    ``loose=False`` reproduces the training-data generator exactly; ``loose=True`` is
    what the deployed container uses (a superset on unseen wordings).
    """
    if is_time_question(question, loose):
        return "time"
    if is_count_question(question):
        return "count"
    return "other"


def frames_for(question: str, loose: bool = True) -> int:
    return FRAMES[route(question, loose)]


def sampling_kwargs(question: str, loose: bool = True) -> dict:
    """Per-question ``chat_template_kwargs`` understood by qwen-vl-utils / ms-swift.

    ``min_frames == max_frames`` fixes the number of uniformly sampled frames; it is
    capped by the clip's own frame count, so short clips are seen at their native rate.
    """
    n = frames_for(question, loose)
    return {"min_frames": n, "max_frames": n}
