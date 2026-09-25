#!/usr/bin/env python
"""Equal-weight average of LoRA adapters ("model soup").

SurgZoom's released adapter is the fp32 mean of three checkpoints trained under the three
input regimes. Averaging is only meaningful when the adapters share the base model, the
LoRA shape and the random seed (hence the same initialisation of the LoRA factors); the
training data and input regime may differ.

    python tools/make_soup.py --adapters ckpt_regime3 ckpt_regime2 ckpt_regime1 --out soup/

Streams tensors with safetensors; peak memory is about one fp32 copy of the adapter.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    files = [d / "adapter_model.safetensors" for d in a.adapters]
    for f in files:
        if not f.exists():
            sys.exit(f"missing {f}")
    acc, dtypes, keys0 = {}, {}, None
    for i, f in enumerate(files):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            if keys0 is None:
                keys0 = keys
            elif keys != keys0:
                sys.exit(f"tensor names differ: {f}")
            for k in keys:
                t = sf.get_tensor(k)
                if i == 0:
                    dtypes[k], acc[k] = t.dtype, t.to(torch.float32)
                else:
                    acc[k] += t.to(torch.float32)
    soup = {k: (v / len(files)).to(dtypes[k]) for k, v in acc.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    save_file(soup, str(a.out / "adapter_model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(a.adapters[0] / "adapter_config.json", a.out / "adapter_config.json")
    (a.out / "SOUP_MEMBERS.txt").write_text("\n".join(str(d) for d in a.adapters) + "\n")
    print(f"averaged {len(files)} adapters ({len(soup)} tensors) -> {a.out}")


if __name__ == "__main__":
    main()
