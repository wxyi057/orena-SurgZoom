"""SurgZoom inference pipeline.

    anchor   a question naming an absolute time is answered on its family-specific
             sub-window (cut from the time-overlay clip), otherwise on the whole clip
    pass 1   every question, frame count set by the text router, fixed token budget
    route    zoom iff the pass-1 answer is a timestamp inside a clip of >= 60 s
    pass 2   +-15 s window around answer 1, cut from the whole clip, same model
    pass 3   +-7.5 s window around answer 2 (all routed questions)
    output   timestamp-only answers are reduced to a single HH:MM:SS

This mirrors ``challenge/inference.py`` (the submitted container) step for step.
"""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from surgzoom import env as _env
from surgzoom.anchor import anchor_family, anchor_window
from surgzoom.policy import route, sampling_kwargs
from surgzoom.prompt import build_system_prompt, user_content
from surgzoom.video import ENCODE_ANCHOR, ENCODE_ZOOM, cut_window, find_ffmpeg
from surgzoom.zoom import (ZOOM_HALF_1, ZOOM_HALF_2, clean, first_time, normalize_answer,
                           window_around, zoom_anchor)

log = logging.getLogger("surgzoom")

DEFAULT_BASE = "Qwen/Qwen3.5-9B"
DEFAULT_ADAPTER = "wxyi088/orena-SurgZoom"


@dataclass
class Question:
    """One request in the challenge format (times on the source-video timeline, seconds)."""
    qID: str
    question: str
    start_time: float
    end_time: float
    procedure_type: str | None = None
    videoID: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(qID=str(d["qID"]), question=d["question"], start_time=float(d["start_time"]),
                   end_time=float(d["end_time"]), procedure_type=d.get("procedure_type"),
                   videoID=d.get("videoID"))


@dataclass
class Trace:
    """What the pipeline did for one question (useful for debugging and demos)."""
    branch: str = ""
    frames: int = 0
    family: str | None = None
    window: tuple[float, float] | None = None
    passes: list[str] = field(default_factory=list)
    answer: str = ""


def resolve_adapter(adapter: str) -> str:
    """Local directory, or a Hugging Face model repo id to download."""
    if Path(adapter).is_dir():
        return adapter
    from huggingface_hub import snapshot_download
    return snapshot_download(adapter, allow_patterns=["adapter_model.safetensors", "*.json"])


class SurgZoom:
    """Qwen3.5-9B + SurgZoom LoRA adapter with question-guided temporal focusing."""

    def __init__(self, base_model: str = DEFAULT_BASE, adapter: str = DEFAULT_ADAPTER,
                 max_batch_size: int = 4, attn_impl: str = "sdpa", max_new_tokens: int = 64):
        _env.configure()
        import torch
        from swift import RequestConfig, TransformersEngine

        adapter_dir = resolve_adapter(adapter)
        log.info("loading %s + adapter %s", base_model, adapter_dir)
        self.engine = TransformersEngine(base_model, adapters=[adapter_dir], model_type="qwen3_5",
                                         torch_dtype=torch.bfloat16, attn_impl=attn_impl,
                                         max_batch_size=max_batch_size)
        self.request_config = RequestConfig(max_tokens=max_new_tokens, temperature=0.0)
        self.ffmpeg = find_ffmpeg()

    # ------------------------------------------------------------------ helpers
    def _request(self, system: str, q: Question, clip: Path, w0: float, w1: float):
        from swift import InferRequest
        return InferRequest(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_content(q.question, w0, w1, q.procedure_type)}],
            videos=[str(clip)],
            # the frame regime is a property of the question, also for zoom windows
            chat_template_kwargs={"enable_thinking": False, **sampling_kwargs(q.question)},
        )

    def _infer(self, requests: list, tag: str) -> list[str]:
        t0 = time.monotonic()
        try:
            out = [r.choices[0].message.content
                   for r in self.engine.infer(requests, self.request_config, use_tqdm=False)]
        except Exception:                                  # noqa: BLE001
            log.exception("[%s] batched inference failed, falling back to one by one", tag)
            out = []
            for r in requests:
                try:
                    out.append(self.engine.infer([r], self.request_config, use_tqdm=False)[0]
                               .choices[0].message.content)
                except Exception:                          # noqa: BLE001
                    log.exception("[%s] question failed, empty answer", tag)
                    out.append("")
        log.info("[%s] %d question(s) in %.1f s", tag, len(requests), time.monotonic() - t0)
        return out

    # ------------------------------------------------------------------ main entry
    def answer(self, questions: list[Question | dict], clip_dir: str | Path, fo_definitions: str,
               time_budget_s: float | None = None, workdir: str | Path | None = None
               ) -> tuple[list[str], list[Trace]]:
        """Answer a batch of questions.

        ``clip_dir`` holds ``<qID>.mp4`` time-overlay clips (the challenge's
        ``/input/overlayed``). ``time_budget_s`` enables the latency guard: passes 3 and 2
        are skipped if the projected batch time would exceed ``time_budget_s - 30``.
        """
        t_start = time.monotonic()
        qs = [q if isinstance(q, Question) else Question.from_dict(q) for q in questions]
        system = build_system_prompt(fo_definitions)
        clip_dir = Path(clip_dir)
        work = Path(workdir or tempfile.mkdtemp(prefix="surgzoom_"))
        deadline = None if time_budget_s is None else time_budget_s - 30.0
        traces = [Trace(branch=route(q.question), frames=sampling_kwargs(q.question)["max_frames"],
                        family=anchor_family(q.question)) for q in qs]

        # anchored sub-windows, then pass 1
        reqs = []
        for q, tr in zip(qs, traces):
            full = clip_dir / f"{q.qID}.mp4"
            clip, w0, w1 = full, q.start_time, q.end_time
            aw = anchor_window(q.question, q.start_time, q.end_time)
            if aw is not None:
                dst = work / f"{q.qID}_anchor.mp4"
                if cut_window(full, dst, aw[0] - q.start_time, aw[1] - aw[0], ENCODE_ANCHOR, self.ffmpeg):
                    clip, (w0, w1), tr.window = dst, aw, aw
                else:
                    log.warning("anchor cut failed for %s, using the whole clip", q.qID)
            reqs.append(self._request(system, q, clip, w0, w1))
        t1 = time.monotonic()
        answers = [clean(a) for a in self._infer(reqs, "pass1")]
        per_q = (time.monotonic() - t1) / max(len(qs), 1)
        for tr, a in zip(traces, answers):
            tr.passes.append(a)

        # self-routed zoom
        zoom = [(i, t) for i, (q, a) in enumerate(zip(qs, answers))
                if (t := zoom_anchor(a, q.start_time, q.end_time)) is not None]
        log.info("zoom: %d/%d question(s)", len(zoom), len(qs))

        def zoom_pass(cands, half, tag):
            reqs, keep = [], []
            for i, t in cands:
                q = qs[i]
                z0, z1 = window_around(t, half, q.start_time, q.end_time)
                dst = work / f"{q.qID}_{tag}.mp4"
                if cut_window(clip_dir / f"{q.qID}.mp4", dst, z0 - q.start_time, z1 - z0,
                              ENCODE_ZOOM, self.ffmpeg):
                    reqs.append(self._request(system, q, dst, z0, z1))
                    keep.append(i)
            return dict(zip(keep, self._infer(reqs, tag))) if reqs else {}

        def within_budget(n, cost):
            return deadline is None or (time.monotonic() - t_start) + n * cost < deadline

        if zoom and within_budget(len(zoom), per_q + 1.5):
            for i, raw in zoom_pass(zoom, ZOOM_HALF_1, "pass2").items():
                a = clean(raw)
                if a:
                    answers[i] = a
                traces[i].passes.append(a)
            cands3 = []
            for i, _ in zoom:
                q, t = qs[i], first_time(answers[i])
                if t is None or not (q.start_time <= t <= q.end_time):
                    t = (q.start_time + q.end_time) / 2
                cands3.append((i, float(t)))
            if within_budget(len(cands3), per_q + 1.0):
                for i, raw in zoom_pass(cands3, ZOOM_HALF_2, "pass3").items():
                    a = clean(raw)
                    if first_time(a) is not None:        # pass 3 must yield a timestamp
                        answers[i] = a
                    traces[i].passes.append(a)
            else:
                log.warning("latency guard: pass 3 skipped")
        elif zoom:
            log.warning("latency guard: passes 2 and 3 skipped")

        answers = [normalize_answer(a) for a in answers]
        for tr, a in zip(traces, answers):
            tr.answer = a
        log.info("done: %d question(s) in %.1f s", len(qs), time.monotonic() - t_start)
        return answers, traces
