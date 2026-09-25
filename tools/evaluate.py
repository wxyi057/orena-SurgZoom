#!/usr/bin/env python
"""Score ``answers.json`` with the official ORena FOCUS evaluator (``orena-focus``).

The reference is one or more official parquets (``data/segment/test.parquet``) or JSON
lists of rows with the same columns (``reference.json`` from ``tools/make_examples.py``).

    python tools/evaluate.py --answers answers.json \
        --reference data/heico-focus-vqa/data/segment/test.parquet \
                    data/lapchole-focus-vqa/data/segment/test.parquet

``--judge none`` scores only rule-based answer formats (time, number, binary, class,
percentage) and skips open-ended / multiple-choice / matching questions, which need the
LLM judge (by default ``Qwen/Qwen3.5-4B``, downloaded on first use).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, nargs="+")
    ap.add_argument("--judge", choices=["default", "none"], default="default")
    ap.add_argument("--output-dir", type=Path, default=None, help="write results.csv / summary.csv")
    a = ap.parse_args()

    from focus import FocusDataset, Response
    from focus.data.formats import JUDGE_FORMATS
    from focus.evaluation import Evaluator

    rows = [r for path in a.reference for r in load_rows(path)]
    if a.judge == "none":
        skipped = sum(r["answer_format"] in JUDGE_FORMATS for r in rows)
        rows = [r for r in rows if r["answer_format"] not in JUDGE_FORMATS]
        print(f"--judge none: skipping {skipped} judge-scored question(s)")
    pairs = [FocusDataset._parse_row(r) for r in rows]
    requests, references = [p[0] for p in pairs], [p[1] for p in pairs]
    answers = {str(x["qID"]): x for x in json.loads(a.answers.read_text())}
    responses = [Response(qID=q, content=x["content"], latency=float(x.get("latency", 0.0)))
                 for q, x in answers.items() if q in {r.qID for r in requests}]

    evaluator = Evaluator()
    results, _ = evaluator.run(requests, references, responses, output_dir=a.output_dir)
    score, buckets = evaluator.pre_evaluation_score(results)
    print(results[["qID", "answer_format", "correctness"]].to_string(index=False))
    print()
    print(buckets.to_string(index=False))
    print(f"\npre-evaluation score (unweighted mean over populated buckets): {score:.4f}")


if __name__ == "__main__":
    main()
