# Training

## Data

The official SEGMENT splits of [HeiCo-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa)
(8,000 training questions) and [LapChole-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)
(5,746), available through the [challenge](https://orena-focus-challenge.org/). No external data.

`tools/prepare_data.py` builds challenge-format clips from the source videos:

1. a 5 fps, ≤ 576 px copy of each HeiCo and LapChole video (this also makes the HeiCo AVIs
   seekable);
2. one clip per question with the source time burned in: `<split>/overlayed/<id>.mp4`;
3. the anchored sub-window of each question that names a time: `<split>/anchored/<id>.mp4`.

`tools/build_sft_data.py` writes one ms-swift row per question: system prompt, `<video>` +
procedure type + window + question, the answer, and the router's frame count.

## Recipe

| | |
|---|---|
| Adapter | LoRA r 64, α 128, dropout 0.05, all linear layers (vision encoder, merger, LLM) |
| Optimiser | AdamW, lr 1e-4, cosine, 3 % warm-up, 15 epochs |
| Batch | 32 (1 per GPU with gradient accumulation), bf16, FlashAttention-2 |
| Seed | 42, identical across regimes so the adapters can be averaged |
| Stack | ms-swift 4.3.2, transformers 5.12.1, peft 0.19.1, qwen-vl-utils 0.0.14 |

## Input regimes

| Regime | Frames per question | Visual tokens | Anchored windows | Checkpoint |
|:-:|---|:-:|:-:|:-:|
| 1 | 64 | 256 per frame pair | – | epoch 11 |
| 2 | 256 timestamp / 128 other | 16,384 | – | epoch 8 |
| 3 | 256 timestamp / 256 counting / 128 other | 20,480 | ✓ | epoch 7 |

```bash
for r in 1 2 3; do
  python tools/build_sft_data.py --regime $r \
      --data-root data/heico-focus-vqa --clips data/clips/heico \
      --data-root data/lapchole-focus-vqa --clips data/clips/lapchole \
      --out data/sft/regime$r.jsonl
  REGIME=$r bash scripts/train.sh data/sft/regime$r.jsonl runs/regime$r
done
python tools/make_soup.py --adapters <regime3-ckpt> <regime2-ckpt> <regime1-ckpt> --out runs/soup
```

Peak memory is about 43 GiB per GPU (regime 3, batch size 1). Multi-node runs take the usual
`NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.

Two-step smoke test on the examples from `tools/make_examples.py`:

```bash
NPROC_PER_NODE=1 GLOBAL_BATCH=1 SAVE_STRATEGY=steps bash scripts/train.sh \
    examples/demo/train_smoke.jsonl runs/smoke --max_steps 2 --save_steps 2
```
