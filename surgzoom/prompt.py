"""Prompt construction (identical in training data, evaluation and the container)."""
from __future__ import annotations

PREAMBLE = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
)


def build_system_prompt(fo_definitions: str) -> str:
    """Fixed preamble followed by the verbatim object-class definitions.

    The definitions are read at run time (``FO_definitions.json`` in the challenge
    input), so newly introduced object classes enter the prompt without retraining.
    """
    return PREAMBLE + fo_definitions


def hms(seconds: float) -> str:
    t = int(seconds)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def user_content(question: str, start_s: float, end_s: float,
                 procedure_type: str | None = None) -> str:
    """User turn. ``start_s``/``end_s`` are the bounds of the pixels actually shown,
    on the source-video timeline (the sub-window for anchored or zoomed questions)."""
    pt = f"Procedure type: {procedure_type}.\n" if procedure_type else ""
    window = f"Clip window: {hms(start_s)} - {hms(end_s)} (source-video timeline).\n"
    return "<video>" + pt + window + question
