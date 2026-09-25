#!/usr/bin/env python
"""Download the base model and the SurgZoom adapter into challenge/resources/ so the image
can run offline (the evaluation platform has no network)."""
from pathlib import Path

from huggingface_hub import snapshot_download

here = Path(__file__).resolve().parent / "resources"
snapshot_download("Qwen/Qwen3.5-9B", local_dir=here / "Qwen3.5-9B")
snapshot_download("wxyi088/orena-SurgZoom", local_dir=here / "adapter",
                  allow_patterns=["adapter_model.safetensors", "adapter_config.json"])
print("resources ready:", sorted(p.name for p in here.iterdir()))
