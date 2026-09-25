"""SurgZoom: question-guided temporal focusing for surgical video question answering."""

from surgzoom.anchor import anchor_family, anchor_window
from surgzoom.policy import BUDGET_TOKENS, frames_for, route
from surgzoom.prompt import PREAMBLE, build_system_prompt, user_content

__version__ = "1.0.0"

__all__ = [
    "BUDGET_TOKENS",
    "PREAMBLE",
    "anchor_family",
    "anchor_window",
    "build_system_prompt",
    "frames_for",
    "route",
    "user_content",
    "__version__",
]
