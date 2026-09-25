#!/usr/bin/env python
"""Build quickstart examples from the official HeiCo-FOCUS-VQA and LapChole-FOCUS-VQA releases.

Both datasets are gated: request access at
https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa and
https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa, then ``hf auth login``.
The script downloads the SEGMENT test questions and only the source videos the chosen
questions come from (one 3 GB HeiCo video, one 0.4 GB LapChole video for the default
cases), cuts their clips and writes one folder in the challenge input layout::

    examples/demo/request.json          questions in the challenge request format
    examples/demo/FO_definitions.json   object-class definitions
    examples/demo/overlayed/<qID>.mp4   clips with the burned-in source clock
    examples/demo/reference.json        ground truth, for tools/evaluate.py
    examples/demo/train_smoke.jsonl     the same questions as ms-swift training rows

    python tools/make_examples.py                                         # default cases
    python tools/make_examples.py --ids heico:2286161 lapchole:1711531    # any test questions

Downloads go to ``data/<name>-focus-vqa`` and clips to ``data/clips/<name>``, the same
places the training pipeline uses, so nothing is fetched or cut twice.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from build_sft_data import build_rows, hms  # noqa: E402
from prepare_data import prepare  # noqa: E402
from surgzoom.prompt import build_system_prompt  # noqa: E402

# revisions are pinned so the question ids stay valid
DATASETS = {
    "heico": ("orena-dkfz/heico-focus-vqa", "4ee0e4b39ee59006b773beec501bb47e251827eb"),
    "lapchole": ("orena-dkfz/lapchole-focus-vqa", "5b3510cde3ba1135c56c4b4b25b50c4f948e235b"),
}
DEFAULT_IDS = [
    "heico:2295666",     # "There is one Needle in the frame at 01:45:48. When is it retrieved ...?"
    "heico:2362897",     # "How many Clip(s) are inserted in the abdomen in this video?"
    "lapchole:1725068",  # "At what time was a Clip first visible in the video?"
    "lapchole:1713930",  # "What types of foreign objects are seen between 00:23:02 and 00:24:03?"
]
SPLIT = "test"


class NoAccess(Exception):
    pass


def fetch(data_root: Path, name: str, rel: str) -> Path:
    """Download one file of an official repo into ``data_root`` unless it is already there."""
    dst = data_root / rel
    if not dst.exists():
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        repo, revision = DATASETS[name]
        try:
            hf_hub_download(repo, rel, repo_type="dataset", revision=revision, local_dir=data_root)
        except GatedRepoError as e:
            raise NoAccess(f"No access to {repo}: request it at https://huggingface.co/datasets/{repo} "
                           "and log in with `hf auth login`.") from e
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="SEGMENT test questions as <dataset>:<id>")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("examples/demo"))
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()

    wanted = {}
    for x in a.ids:
        name, _, qid = x.partition(":")
        if name not in DATASETS or not qid:
            sys.exit(f"expected <dataset>:<id> with dataset in {sorted(DATASETS)}, got {x!r}")
        wanted.setdefault(name, []).append(qid)

    fo = (ROOT / "configs" / "FO_definitions.txt").read_text()
    system = build_system_prompt(fo)
    out = a.out
    (out / "overlayed").mkdir(parents=True, exist_ok=True)
    rows, train = [], []
    for name, ids in wanted.items():
        data_root, clips = a.data_dir / DATASETS[name][0].split("/")[1], a.data_dir / "clips" / name
        try:
            table = fetch(data_root, name, f"data/segment/{SPLIT}.parquet")
            by_id = {str(r["id"]): r for r in pq.read_table(table).to_pylist()}
            if unknown := [i for i in ids if i not in by_id]:
                sys.exit(f"not in the {name} SEGMENT {SPLIT} split: {unknown}")
            for video in sorted({by_id[i]["video"] for i in ids}):
                print(f"{name}: {video}")
                fetch(data_root, name, f"videos/{video}")
        except NoAccess as e:
            print(f"{e} Skipping {name}.")
            continue
        prepare(data_root, clips, [SPLIT], ids, anchored=True, procs=a.procs)
        for i in ids:
            shutil.copyfile(clips / SPLIT / "overlayed" / f"{i}.mp4", out / "overlayed" / f"{i}.mp4")
            rows.append(by_id[i])
        train += build_rows(data_root, clips, 3, SPLIT, system, ids=ids)[0]
    if not rows:
        sys.exit("no examples built")

    request = [{"qID": str(r["id"]), "videoID": r["video"],
                "start_time": float(hms(r["timestamp_start"])), "end_time": float(hms(r["timestamp_end"])),
                "procedure_type": r["procedure_type"], "question": r["question"]} for r in rows]
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (out / "FO_definitions.json").write_text(json.dumps(fo) + "\n")
    (out / "reference.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "train_smoke.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in train))
    print(f"{len(rows)} example(s) ready in {out}")


if __name__ == "__main__":
    main()
