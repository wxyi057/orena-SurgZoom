"""Video-token budget. Must be applied before ms-swift / qwen-vl-utils are imported,
because qwen-vl-utils reads these variables at import time."""
from __future__ import annotations

import os
import sys

from surgzoom.policy import BUDGET_TOKENS


def configure(budget_tokens: int = BUDGET_TOKENS) -> None:
    """Set the per-question visual-token budget used in training and inference.

    qwen-vl-utils caps the total pixels of a video at MODEL_SEQ_LEN * 32^2 * 0.9, so a
    budget of B visual tokens corresponds to MODEL_SEQ_LEN = B / 0.9. The per-frame cap
    is set high enough never to bind, so the budget alone decides the resolution.
    """
    if "qwen_vl_utils" in sys.modules:
        raise RuntimeError("surgzoom.env.configure() must run before qwen_vl_utils is imported")
    for k in ("FPS_MIN_FRAMES", "FPS_MAX_FRAMES"):   # frame counts are set per question
        os.environ.pop(k, None)
    os.environ["MODEL_SEQ_LEN"] = str(round(budget_tokens / 0.9))
    os.environ["VIDEO_MAX_TOKEN_NUM"] = "1024"
    os.environ["VIDEO_MIN_TOKEN_NUM"] = "64"
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    os.environ.setdefault("USE_HF", "1")               # ms-swift: resolve ids on the HF Hub
