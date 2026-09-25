"""Command-line inference on a directory in the challenge input layout.

    <input>/request.json          [{"qID", "videoID", "start_time", "end_time",
                                    "procedure_type", "question"}, ...]
    <input>/FO_definitions.json   object-class definitions (a JSON string)
    <input>/overlayed/<qID>.mp4   time-overlay clips

Example::

    python -m surgzoom.infer --input examples/demo --output answers.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="directory in the challenge input layout")
    ap.add_argument("--output", default=Path("answers.json"), type=Path)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default="wxyi088/orena-SurgZoom", help="HF repo id or local directory")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attn", "eager"])
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds for the whole batch; enables the latency guard "
                         "(the challenge allows 120 + 15 x #questions)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from surgzoom.pipeline import SurgZoom        # heavy imports after argument parsing

    requests = json.loads((args.input / "request.json").read_text())
    fo_definitions = json.loads((args.input / "FO_definitions.json").read_text())
    t0 = time.monotonic()
    model = SurgZoom(args.base_model, args.adapter, max_batch_size=args.batch_size,
                     attn_impl=args.attn_impl)
    answers, traces = model.answer(requests, args.input / "overlayed", fo_definitions,
                                   time_budget_s=args.time_budget)
    total = time.monotonic() - t0

    out = [{"qID": r["qID"], "content": a, "latency": total / len(requests)}
           for r, a in zip(requests, answers)]
    args.output.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'qID':<10} {'branch':<6} {'frames':>6}  {'window':<16} passes -> answer")
    for r, tr in zip(requests, traces):
        win = "whole clip" if tr.window is None else f"{tr.window[0]:.0f}-{tr.window[1]:.0f} s"
        print(f"{r['qID']:<10} {tr.branch:<6} {tr.frames:>6}  {win:<16} "
              f"{' | '.join(tr.passes)} -> {tr.answer}")
    print(f"\nwrote {args.output} ({len(out)} answers, {total:.1f} s incl. model loading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
