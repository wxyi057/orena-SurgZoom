#!/usr/bin/env python
"""Write ms-swift training data (JSONL) for one of the three SurgZoom input regimes.

Each line holds the chat (system / user with ``<video>`` / assistant answer) and the clip
path. Regimes II and III also carry per-row ``chat_template_kwargs`` with the router's
frame count; regime III additionally points anchored questions at their sub-window clip
and states the sub-window bounds in the prompt.

    regime 1   64 frames for every question (frame count fixed by env, see configs/)
    regime 2   router with two branches: 256 frames for time questions, 128 otherwise
    regime 3   router with three branches (+ counting 256) and anchored sub-windows  [final]

    python tools/build_sft_data.py --regime 3 \
        --data-root data/heico-focus-vqa --clips data/clips/heico \
        --data-root data/lapchole-focus-vqa --clips data/clips/lapchole \
        --out data/sft/regime3.jsonl

HeiCo-FOCUS-VQA and LapChole-FOCUS-VQA are combined by passing one ``--data-root``/``--clips``
pair per dataset.
Rows are shuffled with seed 42.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surgzoom.anchor import anchor_window  # noqa: E402
from surgzoom.policy import FRAMES, is_time_question, route  # noqa: E402
from surgzoom.prompt import build_system_prompt, user_content  # noqa: E402


def hms(x) -> int:
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def frames(question: str, regime: int) -> int | None:
    if regime == 1:
        return None
    branch = route(question, loose=False)             # training uses the strict rule
    if regime == 2:
        return 256 if branch == "time" else 128
    return FRAMES[branch]


def build_rows(root: Path, clips: Path, regime: int, split: str, system: str,
               ids=None, relative_to: Path | None = None):
    """ms-swift rows for one dataset; returns (rows, missing clip paths)."""
    ids = {str(i) for i in ids} if ids else None
    rows, missing = [], []
    for r in pq.read_table(root / "data" / "segment" / f"{split}.parquet").to_pylist():
        if ids is not None and str(r["id"]) not in ids:
            continue
        s, e = hms(r["timestamp_start"]), hms(r["timestamp_end"])
        clip, w0, w1 = clips / split / "overlayed" / f"{r['id']}.mp4", s, e
        if regime == 3 and (w := anchor_window(r["question"], s, e)) is not None:
            clip, (w0, w1) = clips / split / "anchored" / f"{r['id']}.mp4", w
        if not clip.exists():
            missing.append(str(clip))
            continue
        # the time rule must agree with the answer format on every training row
        assert is_time_question(r["question"], loose=False) == (r["answer_format"] == "time"), r["id"]
        path = clip.resolve()
        if relative_to:
            path = path.relative_to(relative_to.resolve())
        row = {"messages": [
                   {"role": "system", "content": system},
                   {"role": "user", "content": user_content(r["question"], w0, w1, r["procedure_type"])},
                   {"role": "assistant", "content": r["answer"]}],
               "videos": [str(path)]}
        n = frames(r["question"], regime)
        if n is not None:
            row["chat_template_kwargs"] = {"min_frames": n, "max_frames": n}
        rows.append(row)
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, action="append", required=True)
    ap.add_argument("--clips", type=Path, action="append", required=True)
    ap.add_argument("--regime", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fo-definitions", type=Path, default=ROOT / "configs" / "FO_definitions.txt")
    ap.add_argument("--relative-to", type=Path, default=None,
                    help="write clip paths relative to this directory (portable datasets)")
    a = ap.parse_args()
    if len(a.data_root) != len(a.clips):
        sys.exit("--data-root and --clips must be given the same number of times")

    system = build_system_prompt(a.fo_definitions.read_text())
    rows, missing = [], []
    for root, clips in zip(a.data_root, a.clips):
        r, m = build_rows(root, clips, a.regime, a.split, system, relative_to=a.relative_to)
        rows += r
        missing += m
    if missing:
        sys.exit(f"{len(missing)} clip(s) missing, e.g. {missing[0]} -- run tools/prepare_data.py first")

    random.Random(42).shuffle(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    nf = collections.Counter((row.get("chat_template_kwargs") or {}).get("max_frames") for row in rows)
    print(f"{a.out}: {len(rows)} rows, regime {a.regime}, frames per row {dict(nf)}")


if __name__ == "__main__":
    main()
