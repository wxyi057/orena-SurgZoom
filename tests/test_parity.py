"""The library must make exactly the decisions of the submitted container script.

Synthetic cases always run. Set SURGZOOM_DATA_ROOTS to one or more (":"-separated)
official dataset copies (directories with data/segment/{train,test}.parquet) to also
check every train and test question.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from surgzoom import anchor, policy, prompt, zoom

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submitted", ROOT / "challenge" / "inference.py")
submitted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submitted)

CASES = [
    ("There is one Sponge in the frame at 00:51:42. When is it retrieved from the surgical site? "
     "Please provide the answer in hh:mm:ss.", 3031.0, 3150.0),
    ("What is the maximum number of Clips appearing at once in a single frame? "
     "Please provide a number.", 5240.0, 5359.0),
    ("Is a Clip seen between 00:10:05 and 00:10:09?", 590.0, 709.0),
    ("Which foreign object classes are visible?", 0.0, 29.0),
]


def _check(question, start, end):
    assert policy.route(question, loose=True) == submitted.policy_branch(question)
    assert policy.sampling_kwargs(question) == submitted.policy_kwargs(question)
    assert anchor.anchor_window(question, start, end) == submitted.anchor_window(question, start, end)
    assert prompt.user_content(question, start, end, "x") == submitted.user_content(question, start, end, "x")


@pytest.mark.parametrize("question,start,end", CASES)
def test_synthetic_parity(question, start, end):
    _check(question, start, end)


def test_prompt_and_postprocessing_parity():
    assert prompt.PREAMBLE == submitted.PREAMBLE
    for a in ["00:31:33, 00:32:17", "00:31:33", "Sponge", "at 00:31:33 the sponge"]:
        expect = submitted.normalize_time_answer(a) if submitted._TS.search(a) else submitted.clean(a)
        assert zoom.normalize_answer(a) == expect


def _hms(x):
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


@pytest.mark.skipif(not os.environ.get("SURGZOOM_DATA_ROOTS"), reason="SURGZOOM_DATA_ROOTS not set")
def test_parity_on_all_questions():
    import pyarrow.parquet as pq
    n = 0
    for root in os.environ["SURGZOOM_DATA_ROOTS"].split(":"):
        for split in ("train", "test"):
            for r in pq.read_table(Path(root) / "data" / "segment" / f"{split}.parquet").to_pylist():
                s, e = _hms(r["timestamp_start"]), _hms(r["timestamp_end"])
                _check(r["question"], s, e)
                # the strict training rule equals the ground-truth answer format
                assert policy.is_time_question(r["question"], loose=False) == (r["answer_format"] == "time")
                resp = zoom_answer = "00:00:22"
                assert zoom.zoom_anchor(resp, s, e) == submitted.zoom_anchor(
                    type("R", (), {"start_time": s, "end_time": e})(), zoom_answer)
                n += 1
    assert n > 0
    print(f"parity checked on {n} questions")
