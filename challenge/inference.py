"""SurgZoom — ORena SAVE FOCUS challenge container, SEGMENT track.

Inference script of the submitted image (official pre-evaluation 0.6346). Only
comments, docstrings and log messages differ from the submitted file; the code is
unchanged, and tests/test_parity.py checks its decisions against the `surgzoom`
package.

Model: Qwen3.5-9B + the SurgZoom LoRA adapter (rank 64, all linear layers), the
equal-weight average of three checkpoints trained under three input regimes.

Question-conditioned frame budgeting (identical to training; one budget of 20,480
visual tokens per question; the branch is chosen from the question text alone):
  time-answer questions (the wording asks for "hh:mm:ss")  -> up to 256 frames
  counting / aggregation questions                          -> up to 256 frames
  every other question                                      -> up to 128 frames
Counting uses the dense branch because counting error grew from 23 % to 78 % with
clip length and the errors were under-counts: missed events, not bad arithmetic.
Tokens per frame pair follow qwen-vl-utils' budget rule (MODEL_SEQ_LEN), so long
clips get ~144 (time/count) or ~299 (other) tokens per frame pair and short clips
approach native resolution.

Pipeline (identical to the offline evaluation used for model selection):

  anchor   a question that names an absolute HH:MM:SS is answered on the
           family-specific sub-window it was trained on, cut from the overlay
           clip: "when is it retrieved" -> [t - 10 s, clip end], "at that
           moment" -> t +- 10 s, and so on (nine families).  The burned-in clock
           travels with the pixels, so absolute time survives the cut.  Windows
           covering >= 95 % of the clip are skipped.  Question text only.
  pass 1   every question, on its anchored window (or the whole clip)
  route    a question is zoomed iff its pass-1 answer contains a valid HH:MM:SS
           that lies inside (or within 30 s of) the clip window AND the clip is
           >= 60 s.  Duration-style answers (00:00:22) fall far outside the
           window and are excluded automatically.
  pass 2   +-15 s window around the pass-1 answer, cut from the WHOLE clip (not
           the anchored one), same model
  pass 3   +-7.5 s around the pass-2 answer (all zoomed questions)
  guard    strip <think> residue; a timestamp-only answer that is not exactly one
           HH:MM:SS is replaced by its first valid timestamp

There is no separate re-query for "at that moment" questions: the anchored window
serves that family at pass 1, and re-asking afterwards lowered the local score
(-0.11 points: 5 questions gained, 21 lost).

The overlay variant is used throughout: the burned-in clock is the model's only
source of absolute time and is what it was trained to read.

Latency guard: the pooled budget is 120 s + B x 15 s per batch (a 20 % overrun
forfeits the whole batch).  Wall-clock time is tracked and the script degrades
gracefully — dropping pass 3, then pass 2 — rather than ever overrunning.
"""

import os

# Sampling / visual-token config — identical to training. Must precede any
# swift/transformers import so qwen-vl-utils picks them up.
# Budget regime: total visual tokens per question = 20,480 (MODEL_SEQ_LEN * 0.9);
# frame counts are set PER QUESTION via chat_template_kwargs (see policy_kwargs),
# so no FPS_MIN/MAX_FRAMES here.  VIDEO_MAX_TOKEN_NUM is deliberately non-binding.
os.environ.pop("FPS_MIN_FRAMES", None)
os.environ.pop("FPS_MAX_FRAMES", None)
os.environ["MODEL_SEQ_LEN"] = "22756"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "1024"
os.environ["VIDEO_MIN_TOKEN_NUM"] = "64"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

# Own handler, no propagation: ms-swift reconfigures the root logger on import
# and would otherwise swallow every line after `from swift import ...`.
log = logging.getLogger("segalgo")
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_h)
log.setLevel(logging.INFO)
log.propagate = False

RESOURCES_PATH = Path(__file__).parent / "resources"
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

MODEL_PATH = RESOURCES_PATH / "Qwen3.5-9B"
ADAPTER_PATH = RESOURCES_PATH / "adapter"
VIDEO_DIR = INPUT_PATH / "overlayed"   # trained on the overlay variant

MAX_NEW_TOKENS = 64
MAX_BATCH_SIZE = 4                     # matches the offline evaluation batching

# Question-conditioned frame budgeting (identical to the training-data generator):
# a question whose wording asks for "hh:mm:ss" is a
# time question -> 256 frames; a counting/aggregation question WITHOUT an absolute
# timestamp -> 256 frames; everything else -> 128 frames.  Verified on all 20,000
# train+test rows.  The loose time alternatives only catch differently-worded OOD
# questions; a mis-route just picks the other in-distribution regime.
POLICY_NFRAMES = {"time": 256, "count": 256, "other": 128}
_TIME_Q = re.compile(r"hh:mm:ss|timestamp|at what time|at which time|what time", re.I)
_COUNT_Q = re.compile(r"how many|maximum number of|total count", re.I)
# The fo_class aggregation template needs its "After the ... inserted/created"
# prefix ("with which other FO classes does X co-occur" is not aggregation).
_COUNT_AGGCLS = re.compile(r"^after the .{0,80}(?:inserted|created)"
                           r".*which (?:other|different) foreign object classes",
                           re.I | re.S)
_COUNT_EXCL = re.compile(r"\bin %", re.I)          # "In %, how many frames ..." is a percentage
_TS_ANY = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")  # "how many ... in frame <TS>" is a single-frame question

# Anchored sub-windows (the same tables generated the training data).
# Family -> (seconds kept before the anchor, seconds kept after);
# None means "run to the clip edge".
_ANCHOR_FAMILIES = [
    ("retrieval",   re.compile(r"\bWhen is it retrieved\b", re.I)),
    ("reappear",    re.compile(r"last visible just before .{0,20}re-?appear", re.I)),
    ("range",       re.compile(r"seen between \d{1,2}:\d{2}:\d{2} and \d{1,2}:\d{2}:\d{2}", re.I)),
    ("cooccur",     re.compile(r"\balso appear at\b", re.I)),
    ("moment",      re.compile(r"\bat that moment\b", re.I)),
    ("insert",      re.compile(r"What foreign object was inserted or created", re.I)),
    ("positions",   re.compile(r"At timepoint .{0,40}relative central positions", re.I)),
    ("quadrant",    re.compile(r"\bIn which quadrant\b", re.I)),
    ("frame_local", None),                          # fallback: has a TS, matches nothing above
]
_ANCHOR_PRE_POST = {
    "retrieval": (10.0, None), "reappear": (30.0, None), "range": (5.0, 5.0),
    "cooccur": (10.0, 10.0), "moment": (10.0, 10.0), "insert": (15.0, 15.0),
    "positions": (10.0, 10.0), "quadrant": (None, 10.0), "frame_local": (10.0, 10.0),
}
ANCHOR_MIN_WIN = 20.0                  # shortest sub-window (grow symmetrically)
ANCHOR_SKIP_COVER = 0.95               # a window this close to the whole clip is pointless

# Zoom geometry (identical to the offline evaluation)
ZOOM_HALF_1 = 15.0                     # pass-2 half-window (s)
ZOOM_HALF_2 = 7.5                      # pass-3 half-window (s)
ZOOM_MIN_DUR = 60.0                    # clips shorter than this are dense enough
ANCHOR_SLACK = 30.0                    # clamp out-of-window anchors within this

# Budget fuse. The platform allows 120 + B*15 seconds and forfeits the whole
# batch at 20% overrun; we self-cap well below the overrun cliff.
SETUP_ALLOWANCE = 120.0
PER_QUESTION_BUDGET = 15.0
SELF_MARGIN = 30.0                     # leave this much of the allowance unused

# The system prompt is the trained preamble + the CURRENT FO definitions from
# /input (verbatim). Training used exactly this construction, so swapping in
# the runtime definitions — including any new OOD classes — stays on
# distribution while keeping the class list authoritative.
PREAMBLE = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TS = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")


def clean(text):
    return _THINK.sub("", text or "").strip()


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def ts_seconds(s):
    h, m, sec = s.split(":")
    m, sec = int(m), int(sec)
    if not (0 <= m < 60 and 0 <= sec < 60):
        return None
    return int(h) * 3600 + m * 60 + sec


def first_timestamp(text):
    m = _TS.search(clean(text))
    return ts_seconds(m.group(1)) if m else None


def is_count_question(question):
    q = question or ""
    if _TS_ANY.search(q) or _COUNT_EXCL.search(q):
        return False               # pinned to one frame, or a percentage question
    return bool(_COUNT_Q.search(q)) or bool(_COUNT_AGGCLS.search(q))


def policy_branch(question):
    if _TIME_Q.search(question or ""):
        return "time"
    if is_count_question(question):
        return "count"
    return "other"


def policy_kwargs(question):
    n = POLICY_NFRAMES[policy_branch(question)]
    return {"min_frames": n, "max_frames": n}


def user_content(question, start_s, end_s, procedure_type):
    """Byte-identical to the prompt builder that rendered the training data."""
    pt = f"Procedure type: {procedure_type}.\n" if procedure_type else ""
    hint = f"Clip window: {hms(start_s)} - {hms(end_s)} (source-video timeline).\n"
    return "<video>" + pt + hint + question


def normalize_time_answer(ans):
    """Format guard: a timestamp-only answer must be exactly one HH:MM:SS.

    Only fires when the answer is nothing but timestamps and separators (e.g.
    the model listing "00:31:33, 00:32:17") — free-text answers that merely
    mention a time are left untouched.
    """
    a = clean(ans)
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", a):
        return a
    rest = re.sub(r"(?i)\band\b", "", _TS.sub("", a)).strip(" \t\n,;.&-")
    if rest:
        return a
    m = _TS.search(a)
    return m.group(1) if m else a


# Two encode profiles, each matching the script that produced the clips this model
# was trained or evaluated on.  They differ, so keep them apart:
#   anchor -> crf 23, closed GOP  = how the training and evaluation anchored sub-clips
#                                   were made
#   zoom   -> crf 18              = how the zoom clips of the offline evaluation were made
ENC_ANCHOR = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "25",
              "-keyint_min", "25", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
              "-an", "-movflags", "+faststart"]
ENC_ZOOM = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "25",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]


def cut_window(ffmpeg, src, dst, offset, duration, enc=ENC_ZOOM):
    cmd = ([ffmpeg, "-nostdin", "-y", "-ss", f"{offset:.3f}", "-i", str(src),
            "-t", f"{duration:.3f}"] + enc + [str(dst)])
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def zoom_anchor(req, answer):
    """Self-routing: where (and whether) to zoom, from the pass-1 answer alone."""
    dur = req.end_time - req.start_time
    if dur < ZOOM_MIN_DUR:
        return None
    t = first_timestamp(answer)
    if t is None:
        return None
    if t < req.start_time - ANCHOR_SLACK or t > req.end_time + ANCHOR_SLACK:
        return None            # far outside the window: a duration-style answer
    return min(max(float(t), req.start_time), req.end_time)


def window_around(req, t, half):
    dur = req.end_time - req.start_time
    z0, z1 = max(req.start_time, t - half), min(req.end_time, t + half)
    want = min(2 * half, dur)
    if z1 - z0 < want:         # clamped at a clip edge: extend the other side
        if z0 <= req.start_time:
            z1 = req.start_time + want
        else:
            z0 = req.end_time - want
    return z0, z1


def anchor_family(question):
    """Family name for a question that pins an absolute timestamp; None if it pins none."""
    if not question_timestamps(question):
        return None
    for name, rx in _ANCHOR_FAMILIES:
        if rx is None or rx.search(question):
            return name
    return "frame_local"


def question_timestamps(question):
    out = []
    for x in _TS_ANY.findall(question or ""):
        t = ts_seconds(x)
        if t is not None:
            out.append(float(t))
    return out


def anchor_window(question, clip_start, clip_end):
    """(w0, w1) in absolute seconds, or None to keep the whole clip.

    Identical to the function that generated the training data and drove the
    offline evaluation.
    """
    ts = question_timestamps(question)
    if not ts:
        return None
    if any(t < clip_start - 0.5 or t > clip_end + 0.5 for t in ts):
        return None                          # anchor outside the window: keep the clip
    pre, post = _ANCHOR_PRE_POST[anchor_family(question)]
    lo = clip_start if pre is None else min(ts) - pre
    hi = clip_end if post is None else max(ts) + post
    w0, w1 = max(clip_start, lo), min(clip_end, hi)
    if w1 - w0 < ANCHOR_MIN_WIN:             # grow symmetrically, one-sided at an edge
        need = ANCHOR_MIN_WIN - (w1 - w0)
        w0 = max(clip_start, w0 - need / 2)
        w1 = min(clip_end, w0 + ANCHOR_MIN_WIN)
        w0 = max(clip_start, w1 - ANCHOR_MIN_WIN)
    if w1 - w0 >= ANCHOR_SKIP_COVER * (clip_end - clip_start):
        return None                          # covers ~the whole clip: nothing to gain
    if w1 - w0 < 1.0:
        return None                          # degenerate window on a very short clip
    return (round(w0, 3), round(w1, 3))


def run() -> int:
    t_start = time.monotonic()
    log.info("=== SurgZoom — ORena SAVE FOCUS SEGMENT: Qwen3.5-9B + anchored windows "
             "+ three-branch frame budget + temporal zoom ===")
    try:
        log.info("Adapter: %s", (ADAPTER_PATH / "SOUP_MEMBERS.txt").read_text().strip()
                 if (ADAPTER_PATH / "SOUP_MEMBERS.txt").exists() else "SurgZoom adapter")
    except Exception:
        pass

    from focus import Response, load_requests, save_items
    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:            # local aarch64 dry-run outside the image
        get_ffmpeg_exe = lambda: os.environ.get("FFMPEG_BIN", "ffmpeg")

    log.info("Device available: cuda=%s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU: %s (%.0f GiB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1024**3)

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    B = len(requests)
    allowed = SETUP_ALLOWANCE + B * PER_QUESTION_BUDGET
    deadline = allowed - SELF_MARGIN
    log.info("Batch of %d question(s); allowed %.0f s, self-deadline %.0f s",
             B, allowed, deadline)

    system = PREAMBLE + json.loads((INPUT_PATH / "FO_definitions.json").read_text())
    log.info("System prompt: %d chars (FO definitions taken from /input)", len(system))

    ffmpeg = get_ffmpeg_exe()
    zoom_dir = Path("/tmp/zoom")
    zoom_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model: loaded once per run --------------------------------------
    log.info("Loading engine (model=%s, adapter=%s)", MODEL_PATH, ADAPTER_PATH)
    engine = TransformersEngine(
        str(MODEL_PATH),
        adapters=[str(ADAPTER_PATH)],
        model_type="qwen3_5",
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        max_batch_size=MAX_BATCH_SIZE,
    )
    req_cfg = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    log.info("Engine ready (setup %.1f s)", time.monotonic() - t_start)

    # Warmup inside the setup allowance: a full video-path generation absorbs
    # CUDA context + kernel JIT so the first real question does not pay it.
    try:
        wpath = "/tmp/_warmup.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-y", "-f", "lavfi",
                        "-i", "color=black:size=640x360:rate=5", "-t", "13",
                        "-pix_fmt", "yuv420p", wpath],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_w = time.monotonic()
        engine.infer(
            [InferRequest(
                messages=[{"role": "system", "content": system},
                          {"role": "user",
                           "content": user_content("Is a foreign object visible in the scene?", 0, 13, None)}],
                videos=[wpath],
                chat_template_kwargs={"enable_thinking": False, **policy_kwargs("warmup")})],
            RequestConfig(max_tokens=8, temperature=0.0),
            use_tqdm=False)
        log.info("Warmup generation done (%.1f s)", time.monotonic() - t_w)
    except Exception:
        log.exception("Warmup failed (continuing)")

    def build_ir(req, clip, w0, w1):
        return InferRequest(
            messages=[{"role": "system", "content": system},
                      {"role": "user",
                       "content": user_content(req.question, w0, w1, req.procedure_type)}],
            videos=[str(clip)],
            # frame regime is a property of the QUESTION, so zoom windows of a
            # time question also get the 256-frame regime (as in training/eval)
            chat_template_kwargs={"enable_thinking": False, **policy_kwargs(req.question)},
        )

    def infer_batch(irs, tag):
        """Batched inference with per-question fallback; returns list of raw texts."""
        t0 = time.monotonic()
        try:
            resp = engine.infer(irs, req_cfg, use_tqdm=False)
            outs = [r.choices[0].message.content for r in resp]
        except Exception:
            log.exception("[%s] batched inference failed; per-question fallback", tag)
            outs = []
            for ir in irs:
                try:
                    r = engine.infer([ir], req_cfg, use_tqdm=False)
                    outs.append(r[0].choices[0].message.content)
                except Exception:
                    log.exception("[%s] single question failed; empty answer", tag)
                    outs.append("")
        dt = time.monotonic() - t0
        log.info("[%s] %d question(s) in %.1f s (%.2f s/q)", tag, len(irs), dt,
                 dt / max(len(irs), 1))
        return outs

    def elapsed():
        return time.monotonic() - t_start

    # ---- Anchored sub-windows, then pass 1 -------------------------------
    # A question that pins an absolute HH:MM:SS is answered on the family-specific
    # sub-window it was trained on.  Falling back to the whole clip on a failed cut
    # is safe: that is exactly what an un-anchored question gets.
    clips, hints, n_anchor = [], [], 0
    for r in requests:
        full = VIDEO_DIR / f"{r.qID}.mp4"
        aw = anchor_window(r.question, r.start_time, r.end_time)
        if aw is not None:
            w0, w1 = aw
            dst = zoom_dir / f"{r.qID}_anchor.mp4"
            if cut_window(ffmpeg, full, dst, w0 - r.start_time, w1 - w0, ENC_ANCHOR):
                clips.append(dst); hints.append((w0, w1)); n_anchor += 1
                continue
            log.warning("[anchor] cut failed for %s; falling back to the whole clip", r.qID)
        clips.append(full); hints.append((r.start_time, r.end_time))
    log.info("Anchored sub-windows: %d/%d (elapsed %.0f s)", n_anchor, B,
             time.monotonic() - t_start)

    irs = [build_ir(r, c, w0, w1) for r, c, (w0, w1) in zip(requests, clips, hints)]
    n_branch = {"time": 0, "count": 0, "other": 0}
    for r in requests:
        n_branch[policy_branch(r.question)] += 1
    log.info("Policy routing: %d time / %d count (256f) / %d other (128f)",
             n_branch["time"], n_branch["count"], n_branch["other"])
    t_p1 = time.monotonic()
    answers = [clean(a) for a in infer_batch(irs, "pass1")]
    per_q = (time.monotonic() - t_p1) / max(B, 1)

    # ---- Zoom routing (answer-format self-routing) -----------------------
    zoom = []   # (index, anchor)
    for i, (req, ans) in enumerate(zip(requests, answers)):
        t = zoom_anchor(req, ans)
        if t is not None:
            zoom.append((i, t))
    log.info("Zoom candidates: %d/%d (elapsed %.0f s)", len(zoom), B, elapsed())

    def zoom_pass(cands, half, tag):
        """Cut windows around anchors, re-infer; returns {index: raw answer}."""
        irs, keep = [], []
        for i, t in cands:
            req = requests[i]
            z0, z1 = window_around(req, t, half)
            dst = zoom_dir / f"{req.qID}_{tag}.mp4"
            if cut_window(ffmpeg, VIDEO_DIR / f"{req.qID}.mp4", dst,
                          z0 - req.start_time, z1 - z0):
                irs.append(build_ir(req, dst, z0, z1))
                keep.append(i)
            else:
                log.warning("[%s] cut failed for %s; keeping previous answer", tag, req.qID)
        if not irs:
            return {}
        outs = infer_batch(irs, tag)
        return dict(zip(keep, outs))

    # ---- Pass 2 (+-15 s), budget-gated -----------------------------------
    projected = elapsed() + len(zoom) * (per_q + 1.5)
    if zoom and projected < deadline:
        for i, raw in zoom_pass(zoom, ZOOM_HALF_1, "pass2").items():
            a = clean(raw)
            if a:
                answers[i] = a
    elif zoom:
        log.warning("Skipping pass 2: projected %.0f s > deadline %.0f s",
                    projected, deadline)
        zoom = []

    # ---- Pass 3 (+-7.5 s around the pass-2 answer), budget-gated ---------
    cands3 = []
    for i, _ in zoom:
        req = requests[i]
        t = first_timestamp(answers[i])
        if t is None or not (req.start_time <= t <= req.end_time):
            t = (req.start_time + req.end_time) / 2
        cands3.append((i, float(t)))
    projected = elapsed() + len(cands3) * (per_q + 1.0)
    if cands3 and projected < deadline:
        for i, raw in zoom_pass(cands3, ZOOM_HALF_2, "pass3").items():
            a = clean(raw)
            if first_timestamp(a) is not None:   # pass 3 must yield a timestamp
                answers[i] = a
    elif cands3:
        log.warning("Skipping pass 3: projected %.0f s > deadline %.0f s",
                    projected, deadline)

    # ---- Format guard + output -------------------------------------------
    contents = []
    for req, ans in zip(requests, answers):
        a = clean(ans)
        if _TS.search(a):
            a = normalize_time_answer(a)
        contents.append(a)

    total = elapsed()
    responses = [Response(qID=r.qID, content=c, latency=total / max(B, 1))
                 for r, c in zip(requests, contents)]
    n_empty = sum(1 for r in responses if not r.content.strip())
    if n_empty:
        log.error("ALERT: %d/%d answers are EMPTY — check exception logs above",
                  n_empty, len(responses))
    for r in responses[:5]:
        log.info("  %s -> %r", r.qID, r.content)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d response(s); total %.1f s (allowed %.0f s)",
             len(responses), time.monotonic() - t_start, allowed)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
