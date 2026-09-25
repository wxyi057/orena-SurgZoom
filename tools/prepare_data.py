#!/usr/bin/env python
"""Build SurgZoom training/evaluation clips from the official FOCUS release.

Input: a local copy of an official dataset repo (HeiCo-FOCUS-VQA or LapChole-FOCUS-VQA), e.g.

    hf download orena-dkfz/heico-focus-vqa --repo-type dataset --local-dir data/heico-focus-vqa
    hf download orena-dkfz/lapchole-focus-vqa --repo-type dataset --local-dir data/lapchole-focus-vqa

which contains ``data/segment/{train,test}.parquet`` and ``videos/*``.

Output (``--out``, default ``data/clips/<name>``)::

    _mezzanine/<video>.mp4           5 fps, <= 576 px copy of each source video (seekable)
    <split>/overlayed/<id>.mp4        one clip per question, burned-in HH:MM:SS clock
    <split>/anchored/<id>.mp4         sub-window clip for questions naming a time (training)

Step 1 transcodes every HeiCo and LapChole source video once: the HeiCo AVIs have no seek
index, so cutting clips directly from them re-decodes from the start of the video for every
question, and one recipe for both datasets keeps their clips identical in format.
All steps are idempotent (existing outputs are skipped).

    python tools/prepare_data.py --data-root data/heico-focus-vqa --name heico --procs 32
    python tools/prepare_data.py --data-root data/lapchole-focus-vqa --name lapchole --procs 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgzoom.anchor import anchor_window  # noqa: E402
from surgzoom.video import ENCODE_ANCHOR, cut_window, find_ffmpeg, find_font, make_overlay_clip  # noqa: E402

MEZZ_ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "25",
               "-keyint_min", "25", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
               "-an", "-movflags", "+faststart"]


def hms(x) -> int:
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _mezzanine(job):
    src, dst, ffmpeg = job
    if dst.exists() and dst.stat().st_size > 0:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4",
           "-i", str(src), "-vf", "fps=5,scale=-2:'min(ih,576)'", *MEZZ_ENCODE, str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"{src}: {r.stderr[-300:]}"
    os.replace(tmp, dst)
    print(f"  mezzanine {dst.name}", flush=True)
    return None


def _clip(job):
    src, dst, start, end, font, ffmpeg = job
    if dst.exists() and dst.stat().st_size > 0:
        return None
    return None if make_overlay_clip(src, dst, start, end, font, ffmpeg) else f"clip failed: {dst}"


def _anchored(job):
    src, dst, offset, dur, ffmpeg = job
    if dst.exists() and dst.stat().st_size > 0:
        return None
    return None if cut_window(src, dst, offset, dur, ENCODE_ANCHOR, ffmpeg) else f"anchor failed: {dst}"


def run(pool, fn, jobs, what):
    print(f"{what}: {len(jobs)} job(s)", flush=True)
    errors = [e for e in pool.imap_unordered(fn, jobs, chunksize=1) if e]
    for e in errors[:10]:
        print("  ERROR", e, file=sys.stderr)
    return errors


def prepare(data_root: Path, out: Path, splits, ids=None, anchored: bool = True, procs: int = 4) -> None:
    """Build mezzanine copies, question clips and anchored sub-windows under ``out``."""
    ffmpeg, font = find_ffmpeg(), find_font()
    ids = {str(i) for i in ids} if ids else None
    rows = []
    for split in splits:
        for r in pq.read_table(data_root / "data" / "segment" / f"{split}.parquet").to_pylist():
            if ids is None or str(r["id"]) in ids:
                rows.append((split, r))
    videos = sorted({r["video"] for _, r in rows})
    mezz = {v: out / "_mezzanine" / (Path(v).stem + ".mp4") for v in videos}
    print(f"{len(rows)} question(s) from {len(videos)} video(s) -> {out}")

    errors = []
    with mp.Pool(procs) as pool:
        errors += run(pool, _mezzanine, [(data_root / "videos" / v, mezz[v], ffmpeg) for v in videos],
                      "1/3 mezzanine")
        clip_jobs, anchor_jobs = [], []
        for split, r in rows:
            s, e = hms(r["timestamp_start"]), hms(r["timestamp_end"])
            ov = out / split / "overlayed" / f"{r['id']}.mp4"
            clip_jobs.append((mezz[r["video"]], ov, s, e, font, ffmpeg))
            w = anchor_window(r["question"], s, e)
            if w is not None and anchored:
                anchor_jobs.append((ov, out / split / "anchored" / f"{r['id']}.mp4", w[0] - s, w[1] - w[0], ffmpeg))
        errors += run(pool, _clip, clip_jobs, "2/3 question clips")
        errors += run(pool, _anchored, anchor_jobs, "3/3 anchored sub-windows")
    if errors:
        sys.exit(f"{len(errors)} error(s); re-run to retry (finished outputs are kept)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--name", default=None,
                    help="name used in the output path (default: from --data-root, e.g. heico or lapchole)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--ids", nargs="*", default=None, help="only these question ids")
    ap.add_argument("--no-anchored", action="store_true", help="skip anchored sub-window clips")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    name = a.name or a.data_root.resolve().name.split("-")[0].lower()
    prepare(a.data_root, a.out or Path("data/clips") / name, a.splits, a.ids, not a.no_anchored, a.procs)
    print("done")


if __name__ == "__main__":
    main()
